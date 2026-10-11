import numpy as np
import pandas as pd
import pytest
import torch

from examples.cloud_bridge import fetch_cloud_training, predict_cloud, prepare_fold, supervised_step
from ptop.bundle import REQUIRED_CONTRACTS, create_model, load_bundle, save_bundle
from ptop.official import LABEL_TABLE, SECTION_TABLE, expected_from_sections
from ptop.schema import BOOK_ORDER_FIELDS, FEATURES, ContractError
from ptop.targets import LabelSpec, TimeFold


class QueryRecorder:
    # 仅隔离示例的查询路由，合成数据不代表真实 DAI 或云端训练验证。
    def __init__(self, bars, sections, labels):
        self.frames = {"synthetic_cloud_bar": bars, SECTION_TABLE: sections, LABEL_TABLE: labels}
        self.calls = []

    def __call__(self, sql, *, filters):
        self.calls.append((sql, filters))
        table = next(name for name in self.frames if f"FROM {name}" in sql)
        frame = self.frames[table].copy()
        frame = frame.loc[frame.date.between(*map(pd.Timestamp, filters["date"]))]
        if "instrument" in filters:
            frame = frame.loc[frame.instrument.isin(filters["instrument"])]

        class Result:
            def df(self):
                return frame

        return Result()


@pytest.fixture
def cloud_fixture():
    torch.set_num_threads(1)
    grid = pd.date_range("2024-01-02 09:31", periods=4, freq="min").append(
        pd.date_range("2024-01-03 09:31", periods=4, freq="min"))
    mapping = {1: "SYNTHETIC_A", 2: "SYNTHETIC_B", 3: "SYNTHETIC_EMPTY"}
    bars = pd.MultiIndex.from_product([grid, [1, 2]], names=["date", "instrument_id"]).to_frame(index=False)
    bars["instrument"] = bars.instrument_id.map(mapping)
    bars["adjust_factor"] = 1.0
    for index, field in enumerate(FEATURES):
        bars[field] = 10.0 + index + np.arange(len(bars))
    for field in BOOK_ORDER_FIELDS:
        bars[field] += 50000
    sections = pd.MultiIndex.from_product([grid[[1, 3, 5, 7]], mapping.values()],
                                          names=["date", "instrument"]).to_frame(index=False)
    expected = expected_from_sections(sections, mapping)
    labels = sections.assign(vwap_return=np.linspace(-0.02, 0.02, len(sections)))
    labels.loc[labels.instrument.eq("SYNTHETIC_B"), "vwap_return"] = np.nan
    spec = LabelSpec("synthetic-v1", LABEL_TABLE, "synthetic explicit target", "synthetic none",
                     "synthetic explicit intraday support")
    support = expected[["sample_id"]].assign(
        target_valid=True, target_start=expected.decision_time,
        target_end=expected.decision_time + pd.Timedelta(minutes=1), label_spec_version=spec.version,
    )
    query = QueryRecorder(bars, sections, labels)
    raw, official_expected, targets = fetch_cloud_training(
        query, {"bar1m": "synthetic_cloud_bar"}, start_date="2024-01-02", end_date="2024-01-03",
        history_start="2024-01-02", instrument_map=mapping, support=support, label_spec=spec,
        bar_end_is_available=True,
    )
    fold = TimeFold("synthetic", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
                    pd.Timestamp("2024-01-04"))
    prepared = prepare_fold(
        raw, official_expected, targets, label_spec=spec, trading_grid=grid, fold=fold,
        scale_review="synthetic fixture only; bridge interface check", window_length=4,
    )
    return query, raw, official_expected, targets, prepared, grid, mapping, fold, spec


def test_training_bridge_uses_cloud_units_all_train_rows_and_aligned_targets(cloud_fixture):
    query, raw, expected, targets, prepared, grid, mapping, fold, spec = cloud_fixture
    assert len(query.calls) == 3
    assert raw.values.iloc[0]["close"] == 13.0
    assert raw.values.iloc[0][BOOK_ORDER_FIELDS[0]] > 32767
    assert prepared.processor.fit_row_count == 8
    assert prepared.train_targets.target_valid.sum() == 4
    assert len(prepared.train_loader.dataset) == 6
    assert len(prepared.validation_loader.dataset) == 6
    batch = next(iter(prepared.train_loader))
    assert len(batch.x) == 3 and batch.metadata.decision_id.nunique() == 1
    model = create_model("R1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.encoder.net[0].weight.detach().clone()
    result = supervised_step(model, optimizer, batch, prepared.train_targets.iloc[::-1], device="cpu")
    assert result["count"] == 1 and np.isfinite(result["loss"])
    assert result["sample_ids"] == batch.metadata.loc[batch.metadata.instrument_id.eq(1), "sample_id"].tolist()
    assert set(result["skipped"].reason) == {"invalid_target", "no_observation_in_model_fields"}
    assert not torch.equal(before, model.encoder.net[0].weight)
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    validation = supervised_step(model, None, next(iter(prepared.validation_loader)),
                                 prepared.validation_targets, device="cpu")
    assert validation["count"] == 1 and np.isfinite(validation["loss"])
    assert not model.training
    assert all(torch.equal(value, snapshot[name]) for name, value in model.state_dict().items())


def test_inference_only_reads_inputs_and_reports_complete_section_exclusions(cloud_fixture):
    query, raw, expected, targets, prepared, grid, mapping, fold, spec = cloud_fixture
    query.calls.clear()
    model = create_model("R1")
    fingerprint = prepared.processor.fingerprint
    output, excluded, coverage = predict_cloud(
        query, {"bar1m": "synthetic_cloud_bar"}, model, prepared.processor,
        start_date="2024-01-03", end_date="2024-01-03", history_start="2024-01-03",
        instrument_map=mapping, trading_grid=grid, bar_end_is_available=True, device="cpu", window_length=4,
    )
    assert len(query.calls) == 2 and all(LABEL_TABLE not in sql for sql, _ in query.calls)
    assert list(output.columns) == ["date", "instrument", "score"]
    assert len(output) == 4 and len(excluded) == 2
    assert set(excluded.unpredictable_reason) == {"no_observation_in_model_fields"}
    assert coverage.expected.eq(3).all() and coverage.predicted.eq(2).all()
    assert prepared.processor.fingerprint == fingerprint


def test_missing_contract_or_partial_optimizer_fails_and_empty_supervision_is_explicit(cloud_fixture):
    query, raw, expected, targets, prepared, grid, mapping, fold, spec = cloud_fixture
    with pytest.raises(ContractError, match="scale_review"):
        prepare_fold(raw, expected, targets, label_spec=spec, trading_grid=grid, fold=fold, scale_review="")
    model = create_model("R1")
    batch = next(iter(prepared.train_loader))
    partial_optimizer = torch.optim.Adam(model.fc_out.parameters(), lr=1e-3)
    with pytest.raises(ContractError, match="全部参与联合训练"):
        supervised_step(model, partial_optimizer, batch, prepared.train_targets, device="cpu")
    with pytest.raises(ContractError, match="缺少对应 sample_id"):
        supervised_step(model, None, batch, prepared.train_targets.iloc[:0], device="cpu")
    all_invalid = prepared.train_targets.assign(target_valid=False, supervision_exclusion="invalid_target")
    result = supervised_step(model, None, batch, all_invalid, device="cpu")
    assert result["count"] == 0 and result["loss"] is None and len(result["skipped"]) == 3


def test_json_bundle_roundtrip_with_cloud_preprocessor(cloud_fixture, tmp_path):
    _, _, _, _, prepared, _, mapping, _, _ = cloud_fixture
    model = create_model("R1").eval()
    destination = tmp_path / "bundle"
    save_bundle(destination, model, prepared.processor, model_name="R1", window_length=4,
                contracts={name: "synthetic-test" for name in REQUIRED_CONTRACTS}, seed=17,
                training_record={"scope": "synthetic interface check"}, instrument_map=mapping)
    restored, processor, _ = load_bundle(destination)
    assert processor.fingerprint == prepared.processor.fingerprint
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
