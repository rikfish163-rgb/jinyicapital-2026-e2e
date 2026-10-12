"""Regression tests for bounded ingestion, frozen N1 and experiment fairness."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest
import torch

from ptop.data import concat_partitions
from ptop.experiments import (ExperimentBudget, ExperimentBudgetExceeded, ExperimentPlan,
                              benchmark_windows, requests_for_plan, run_experiments)
from ptop.local_training import (DIAGNOSTIC_SCOPE, LocalBarStore, ReadPolicy,
                                diagnostic_training_factory, local_training_factory,
                                prepare_diagnostic_fold, prepare_local_fold, split_local_targets)
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import BOOK_ORDER_FIELDS, FEATURES, ContractError
from ptop.targets import LabelSpec, TimeFold
from ptop.windows import WindowDataset, fit_row_union


@pytest.fixture
def local_fixture(tmp_path):
    torch.set_num_threads(1)
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    grid = pd.DatetimeIndex([])
    for day in days:
        grid = grid.append(pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=4, freq="min"))
    bars = pd.MultiIndex.from_product([grid, [1, 2]], names=["date", "instrument_id"]).to_frame(index=False)
    for column, name in enumerate(FEATURES):
        bars[name] = np.arange(len(bars), dtype=np.int64) * (column + 1) + 1000 + column
    bars["deal_number"] = 0  # Qlib ignores constant fields instead of dividing by zero.
    bars.loc[0, "bid_price1"] = 0  # Known missing book price; one training cell is absent.
    bars[BOOK_ORDER_FIELDS[0]] += 50000  # Valid positive int32 counts must not wrap.
    bars["available_at"] = bars.date
    bars.loc[2, "available_at"] += pd.Timedelta(minutes=2)  # One bar delayed past its first decision.
    bars.loc[bars.date.dt.normalize().eq(days[-1]), "close"] += 10_000_000
    source = tmp_path / "source"
    source.mkdir()
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    mapping = {1: "SYNTHETIC_A", 2: "SYNTHETIC_B", 3: "SYNTHETIC_EMPTY"}
    samples = []
    for point in grid[grid.minute.isin([32, 34])]:
        decision = point.strftime("%Y%m%dT%H%M")
        for instrument in mapping:
            samples.append({"sample_id": f"{decision}:{instrument}", "decision_id": decision,
                            "decision_time": point, "instrument_id": instrument,
                            "instrument": mapping[instrument]})
    expected = pd.DataFrame(samples)
    spec = LabelSpec("synthetic-contract.v1", "explicit-test-table", "synthetic exact values", "none-test-only", "one-minute explicit support")
    targets = expected[["sample_id"]].assign(target=np.linspace(-0.01, 0.01, len(expected)),
        target_valid=True, target_start=expected.decision_time,
        target_end=expected.decision_time + pd.Timedelta(minutes=1), label_spec_version=spec.version)
    fold = TimeFold("synthetic-local", days[0], days[-1], days[-1] + pd.Timedelta(days=1))
    store = LocalBarStore(source, cache_dir=tmp_path / "cache")
    return source, bars, mapping, expected, targets, spec, grid, fold, store


def _prepared(fixture, **kwargs):
    _, _, mapping, expected, targets, spec, grid, fold, store = fixture
    return prepare_local_fold(store, expected, targets, instrument_map=mapping, label_spec=spec,
        trading_grid=grid, fold=fold, window_length=4, scale_review="synthetic numerical regression; no manual-review claim", **kwargs)


def test_partitioned_n1_equals_pinned_a_full_fit_and_validation_is_frozen(local_fixture):
    prepared = _prepared(local_fixture)
    _, _, mapping, _, _, _, grid, fold, store = local_fixture
    partitions = [store.read_day(day, instrument_map=mapping) for day in pd.to_datetime(["2024-01-02", "2024-01-03"])]
    whole_train = concat_partitions(partitions)
    windows = WindowDataset(whole_train, prepared.train_expected, grid, length=4)
    references = fit_row_union(windows, fit_start=fold.train_start, fit_end="2024-01-03")
    direct = FrozenPreprocessor("N1").fit(whole_train.values, whole_train.observed,
        referenced_rows=references, fit_start=fold.train_start, fit_end="2024-01-03")
    direct.freeze(scale_review="test direct pinned A fit")
    a, b = direct.state_dict(), prepared.processor.state_dict()
    for name in ("fields", "fit_row_count", "fit_rows_fingerprint", "valid_counts", "training_constants"):
        assert a[name] == b[name]
    assert a["statistics"]["ignore"] == b["statistics"]["ignore"]
    np.testing.assert_allclose(a["statistics"]["center"], b["statistics"]["center"], rtol=2e-15, atol=1e-10)
    np.testing.assert_allclose(a["statistics"]["scale"], b["statistics"]["scale"], rtol=2e-14, atol=1e-10)
    np.testing.assert_allclose(direct.transform(whole_train.values, whole_train.observed),
                               prepared.processor.transform(whole_train.values, whole_train.observed), rtol=1e-6, atol=1e-6)
    assert b["fit_row_count"] == 16
    assert b["valid_counts"][FEATURES.index("bid_price1")] == 15
    assert b["statistics"]["ignore"][FEATURES.index("deal_number")]
    before = prepared.processor.fingerprint
    list(prepared.validation_loader())
    assert prepared.processor.fingerprint == before
    assert prepared.normalization_diagnostics["automatic_float32_finite_check"]
    assert prepared.normalization_diagnostics["implementation"].startswith("pooled_")


def test_day_loader_visible_slots_empty_official_symbols_and_raw_only_cache(local_fixture):
    prepared = _prepared(local_fixture)
    batches = list(prepared.validation_loader())
    assert len(batches) == 2
    for batch in batches:
        assert len(batch.metadata) == 3 and batch.metadata.decision_id.nunique() == 1
        assert batch.metadata.loc[~batch.predictable, "instrument_id"].tolist() == [3]
        for row, metadata in batch.metadata.iterrows():
            times = batch.bar_times[row][batch.step_valid[row]]
            assert (times <= metadata.decision_time.to_datetime64()).all()
            assert (pd.DatetimeIndex(times).normalize() == metadata.decision_time.normalize()).all()
    delayed = next(batch for batch in prepared.train_loader() if batch.metadata.decision_time.iloc[0] == pd.Timestamp("2024-01-02 09:32"))
    row = int(np.flatnonzero(delayed.metadata.instrument_id.eq(1))[0])
    assert delayed.step_valid[row, -1]
    assert not delayed.observed[row, -1].any()
    assert delayed.row_positions[row, -1] == -1
    later = next(batch for batch in prepared.train_loader() if batch.metadata.decision_time.iloc[0] == pd.Timestamp("2024-01-02 09:34"))
    row = int(np.flatnonzero(later.metadata.instrument_id.eq(1))[0])
    assert later.observed[row, -3].any()
    assert prepared.store.profile["max_day_rows_read"] == 8
    assert prepared.store.manifest["cache_kind"] == "raw_only_no_learned_tokens"
    assert all(entry["rows"] == 8 for entry in prepared.store.manifest["partitions"].values())
    cache_files = list(prepared.store._scope_dir.iterdir())
    assert {path.suffix for path in cache_files} == {".arrow", ".json"}


def test_train_seed_is_epoch_explicit_and_shorter_windows_reuse_same_n1(local_fixture):
    prepared = _prepared(local_fixture)
    first = [batch.metadata.sample_id.tolist() for batch in prepared.train_loader(2)]
    repeated = [batch.metadata.sample_id.tolist() for batch in prepared.train_loader(2)]
    assert first == repeated
    shorter = prepared.with_window(2, seed=29)
    assert shorter.processor is prepared.processor
    assert shorter.manifest["sampling_keys_fingerprint"] == prepared.manifest["sampling_keys_fingerprint"]
    assert next(iter(shorter.validation_loader())).x.shape == (3, 2, 25)
    with pytest.raises(ContractError, match="fit_window_length"):
        prepared.with_window(5)


def test_negative_order_mask_is_opt_in_and_other_negatives_remain_errors(local_fixture, tmp_path):
    source, bars, *_ = local_fixture
    bars.loc[0, BOOK_ORDER_FIELDS[0]] = -3
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    rejecting = LocalBarStore(source, cache_dir=tmp_path / "rejecting")
    rejecting.materialize(["2024-01-02"], [1, 2])
    with pytest.raises(ContractError, match="未知负值"):
        rejecting.read_day("2024-01-02")
    masking = LocalBarStore(source, cache_dir=tmp_path / "masking", policy=ReadPolicy(negative_order_policy="mask_known"))
    masking.materialize(["2024-01-02"], [1, 2])
    raw = masking.read_day("2024-01-02")
    assert not raw.observed.iloc[0][BOOK_ORDER_FIELDS[0]]
    assert np.isnan(raw.values.iloc[0][BOOK_ORDER_FIELDS[0]])
    assert raw.values.iloc[1][BOOK_ORDER_FIELDS[0]] > 32767
    assert masking.profile["masked_negative_orders"][BOOK_ORDER_FIELDS[0]] == 1
    bars.loc[0, "volume"] = -1
    feather.write_feather(bars, source / "202401.0.feather", chunksize=4)
    masking.materialize(["2024-01-02"], [1, 2])
    with pytest.raises(ContractError, match="volume"):
        masking.read_day("2024-01-02")


def test_source_and_daily_budgets_and_cache_integrity_are_enforced(local_fixture, tmp_path):
    source, *_ = local_fixture
    for policy, message in ((ReadPolicy(max_source_bytes=1), "max_source_bytes"),
                            (ReadPolicy(max_record_batch_rows=1), "max_record_batch_rows"),
                            (ReadPolicy(max_day_rows=1), "max_day_rows")):
        store = LocalBarStore(source, cache_dir=tmp_path / message, policy=policy)
        with pytest.raises(ContractError, match=message):
            store.materialize(["2024-01-02"], [1, 2])
    store = LocalBarStore(source, cache_dir=tmp_path / "integrity")
    store.materialize(["2024-01-02"], [1, 2])
    store.read_day("2024-01-02")
    file = store._scope_dir / "2024-01-02.arrow"
    with file.open("ab") as stream:
        stream.write(b"changed-after-first-read")
    with pytest.raises(ContractError, match="校验失败"):
        store.read_day("2024-01-02")


def test_missing_formal_contract_fails_before_source_read_and_no_mapping_is_guessed(local_fixture):
    *_, store = local_fixture
    with pytest.raises(ContractError, match="official expected keys.*instrument_map.*LabelSpec.*explicit support"):
        prepare_local_fold(store)
    assert store.manifest is None
    diagnostic = prepare_diagnostic_fold(store, days=["2024-01-02", "2024-01-03", "2024-01-04"],
        instrument_ids=[1, 2], validation_start="2024-01-04", endpoint_minutes=["09:32", "09:34"], window_length=4)
    assert diagnostic.scope == DIAGNOSTIC_SCOPE and diagnostic.instrument_map is None
    assert "not official" in diagnostic.manifest["diagnostic_notice"]
    batch = next(iter(diagnostic.validation_loader()))
    assert "instrument" not in batch.metadata
    assert not batch.metadata.production_visibility_verified.any()
    assert batch.metadata.audit_scope.eq("local_observed_grid").all()
    assert diagnostic.label_spec.version.startswith("diagnostic-")


def test_target_support_at_split_boundary_is_purged_without_deleting_keys():
    expected = pd.DataFrame({"sample_id": ["a", "b", "c"], "decision_id": ["d1", "d2", "d3"],
        "instrument_id": [1, 1, 1], "decision_time": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:02", "2024-01-02 10:03"])})
    spec = LabelSpec("test.v1", "test", "test", "test", "test")
    targets = expected[["sample_id"]].assign(target=1.0, target_valid=True, target_start=expected.decision_time,
        target_end=pd.to_datetime(["2024-01-02 10:02", "2024-01-02 10:03", "2024-01-02 10:05"]), label_spec_version=spec.version)
    fold = TimeFold("boundary", "2024-01-02 09:30", "2024-01-02 10:02", "2024-01-02 10:05")
    train, validation, audit = split_local_targets(expected, targets, spec, fold)
    assert train.sample_id.tolist() == ["a"] and not train.target_valid.any()
    assert validation.sample_id.tolist() == ["b", "c"]
    assert validation.target_valid.tolist() == [True, False]
    assert audit.supervision_exclusion.eq("target_reaches_split_boundary").sum() == 2
    assert pd.isna(train.target.iloc[0]) and pd.isna(validation.target.iloc[1])


def test_experiment_scheduler_locks_keys_labels_scaler_and_independent_budget(local_fixture, tmp_path):
    prepared = _prepared(local_fixture)
    plan = ExperimentPlan(({"name": "model-a"}, {"name": "model-b"}), lookbacks=(2, 4), seeds=(7, 17),
                          budget=ExperimentBudget(epochs=2, max_wall_seconds=30.0))
    requests = requests_for_plan(plan)
    assert len(requests) == len({request.run_id for request in requests}) == 8
    calls = []

    def runner(request, current, output_dir):
        calls.append((request.run_id, current.window_length, current.seed, current.processor.fingerprint))
        flags = {key: bool(flag) for batch in current.validation_loader() for key, flag in zip(batch.metadata.sample_id, batch.predictable)}
        frame = current.validation_targets[["sample_id", "target_valid"]].assign(score=0.1)
        frame["predictable"] = frame.sample_id.map(flags)
        return {"validation_predictions": frame, "metrics": {"loss": 0.01}, "best_metric": 0.01,
                "selection_metric": "loss"}

    summary = run_experiments(plan, prepared, output_dir=tmp_path / "grid", runner=runner)
    assert summary.status.eq("completed").all()
    assert len({item[-1] for item in calls}) == 1
    assert summary.eligible_prediction_keys_fingerprint.nunique() == 1
    resumed = run_experiments(plan, prepared, output_dir=tmp_path / "grid", runner=runner, resume_completed=True)
    assert len(calls) == 8 and len(resumed) == 8
    with pytest.raises(ContractError, match="固定计划"):
        run_experiments(replace(plan, seeds=(29,)), prepared, output_dir=tmp_path / "grid", runner=runner)

    def altered_keys(request, current, output_dir):
        flags = {key: bool(flag) for batch in current.validation_loader() for key, flag in zip(batch.metadata.sample_id, batch.predictable)}
        frame = current.validation_targets[["sample_id", "target_valid"]].assign(score=0.1)
        frame["predictable"] = frame.sample_id.map(flags)
        if request.model_spec["name"] == "model-b":
            frame = frame.iloc[1:]
        return {"validation_predictions": frame, "metrics": {}, "best_metric": 0.1, "selection_metric": "loss"}

    mismatch = run_experiments(replace(plan, lookbacks=(4,), seeds=(7,)), prepared,
        output_dir=tmp_path / "mismatch", runner=altered_keys)
    assert mismatch.status.tolist() == ["completed", "failed"]
    assert "采样键" in mismatch.error.iloc[-1]

    def over_budget(request, current, output_dir):
        raise ExperimentBudgetExceeded("declared test budget")

    exceeded = run_experiments(replace(plan, lookbacks=(4,), seeds=(7,)), prepared,
        output_dir=tmp_path / "exceeded", runner=over_budget)
    assert exceeded.status.eq("budget_exceeded").all()


def test_real_window_benchmark_reports_waits_and_memory_without_targets(local_fixture):
    prepared = _prepared(local_fixture)

    class TinyScore(torch.nn.Module):
        def forward(self, x, observed, step_valid):
            return x[:, -1, 0]

    metrics = benchmark_windows(prepared.validation_loader(), model=TinyScore(), device="cpu")
    assert metrics["batches"] == 2 and metrics["samples"] == 6 and metrics["predictable_samples"] == 4
    assert metrics["data_wait_seconds"] > 0 and metrics["forward_seconds"] > 0
    assert metrics["peak_rss_bytes_process"] > 0 and metrics["samples_per_second"] > 0
    assert metrics["peak_cuda_allocated_bytes"] is None


def test_cli_file_factories_require_official_contracts_and_preserve_explicit_fold(local_fixture, tmp_path, monkeypatch):
    source, _, mapping, expected, targets, spec, grid, fold, _ = local_fixture
    model_spec = SimpleNamespace(lookback=4)
    with pytest.raises(ContractError, match="正式 data_factory 缺少"):
        local_training_factory(config={"data_dir": str(source)}, model_spec=model_spec, fold=fold)
    monkeypatch.setattr("ptop.local_training._factory_fields", lambda spec: FEATURES)
    expected_path, target_path = tmp_path / "expected.parquet", tmp_path / "targets.parquet"
    expected.to_parquet(expected_path, index=False)
    targets.to_parquet(target_path, index=False)
    mapping_path = tmp_path / "instrument_map.json"
    mapping_path.write_text(json.dumps(mapping))
    grid_path = tmp_path / "grid.csv"
    pd.DataFrame({"datetime": grid}).to_csv(grid_path, index=False)
    config = {"data_dir": str(source), "cache_dir": str(tmp_path / "factory_cache"),
              "expected_path": str(expected_path), "targets_path": str(target_path),
              "instrument_map_path": str(mapping_path), "grid_path": str(grid_path), "label_spec": vars(spec)}
    prepared, returned_mapping = local_training_factory(config=config, model_spec=model_spec, fold=fold)
    assert returned_mapping == mapping and prepared.scope == "official"
    assert next(iter(prepared.validation_loader())).x.shape == (3, 4, 25)
    diagnostic_config = {"scope": "diagnostic", "data_dir": str(source), "cache_dir": str(tmp_path / "diagnostic_factory_cache"),
                         "days": ["2024-01-02", "2024-01-03", "2024-01-04"], "instrument_ids": [1, 2],
                         "endpoint_minutes": ["09:32", "09:34"]}
    diagnostic, absent_mapping = diagnostic_training_factory(config=diagnostic_config, model_spec=model_spec, fold=fold)
    assert absent_mapping is None and diagnostic.scope == DIAGNOSTIC_SCOPE
    assert diagnostic.fold.name == fold.name and diagnostic.manifest["fold"]["name"] == fold.name


def test_experiment_seed_range_matches_training_config():
    with pytest.raises(ContractError, match="2\*\*32"):
        ExperimentPlan(({"name": "test"},), seeds=(2**32,))
