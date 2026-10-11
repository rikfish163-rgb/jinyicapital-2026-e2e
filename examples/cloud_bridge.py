from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ptop.data import CanonicalData
from ptop.official import fetch_inference_inputs, fetch_label_values, official_targets
from ptop.output import audit_output
from ptop.preprocessing import FrozenPreprocessor
from ptop.schema import FEATURES, ContractError
from ptop.targets import LabelSpec, TimeFold, attach_targets
from ptop.windows import DecisionBatchSampler, WindowBatch, WindowDataset, collate_windows, fit_row_union


@dataclass
class PreparedFold:
    processor: FrozenPreprocessor
    train_loader: DataLoader
    validation_loader: DataLoader
    train_targets: pd.DataFrame
    validation_targets: pd.DataFrame
    split_audit: pd.DataFrame


def fetch_cloud_training(
    query: Callable,
    datasources: Mapping[str, str],
    *,
    start_date,
    end_date,
    history_start,
    instrument_map: Mapping[int, str],
    support: pd.DataFrame,
    label_spec: LabelSpec,
    bar_end_is_available: bool,
) -> tuple[CanonicalData, pd.DataFrame, pd.DataFrame]:
    """query 传 dai.query；datasources 传平台注入值；支持区间由 C 显式提供。"""
    if not isinstance(label_spec, LabelSpec) or not isinstance(support, pd.DataFrame):
        raise ContractError("训练数据读取需要 LabelSpec 和显式 support 表")
    raw, expected = fetch_inference_inputs(
        query, datasources, start_date=start_date, end_date=end_date, history_start=history_start,
        instrument_map=instrument_map, bar_end_is_available=bar_end_is_available,
    )
    labels = fetch_label_values(query, start_date=start_date, end_date=end_date)
    targets = official_targets(expected, labels, support, label_spec)
    return raw, expected, targets


def prepare_fold(
    raw: CanonicalData,
    expected: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    label_spec: LabelSpec,
    trading_grid: pd.DatetimeIndex,
    fold: TimeFold,
    scale_review: str = "cloud-training:extra-manual-scale-review-skipped",
    window_length: int = 120,
    fields: Sequence[str] = FEATURES,
    seed: int = 17,
    num_workers: int = 0,
) -> PreparedFold:
    """对内存可容纳的折数据拟合 N1；原始表、诊断及转换会持有多份数据。

    B 应组织可容纳的读取分区及训练轮次，不能将五年长表直接传入或缓存学习后的 token。
    scale_review 必须如实记录实际决策；跳过人工审阅时不能记录为审阅通过。
    """
    if not isinstance(fold, TimeFold):
        raise ContractError("必须显式提供 TimeFold")
    if any(value != value.normalize() for value in (fold.train_start, fold.validation_start, fold.validation_end)):
        raise ContractError("桥接示例按完整日切分，TimeFold 三个边界必须为午夜")
    if not isinstance(scale_review, str) or not scale_review.strip():
        raise ContractError("必须提供真实 scale_review；跳过人工审阅时明确记录该决定")
    if type(num_workers) is not int or num_workers < 0:
        raise ContractError("num_workers 必须为非负整数")
    labeled = attach_targets(expected, targets, label_spec)
    time = pd.to_datetime(labeled["decision_time"], errors="raise")
    train = labeled.loc[time.ge(fold.train_start) & time.lt(fold.validation_start)].copy()
    validation = labeled.loc[time.ge(fold.validation_start) & time.lt(fold.validation_end)].copy()
    if train.empty or validation.empty:
        raise ContractError("训练和验证区间均须有完整官方应预测集合")
    sample_columns = ["sample_id", "decision_id", "instrument_id", "decision_time", "instrument"]
    if not set(sample_columns).issubset(expected.columns):
        raise ContractError(f"expected 必须来自官方截面并包含 {sample_columns}")
    train_expected, validation_expected = train[sample_columns], validation[sample_columns]
    # 拟合集合由训练决策窗口决定，标签缺失及边界清除均不改变拟合原始行。
    raw_windows = WindowDataset(raw, train_expected, trading_grid, length=window_length, fields=fields)
    fit_end = fold.validation_start - pd.Timedelta(days=1)
    referenced = fit_row_union(raw_windows, fit_start=fold.train_start, fit_end=fit_end)
    processor = FrozenPreprocessor("N1").fit(
        raw.values, raw.observed, referenced_rows=referenced, fit_start=fold.train_start, fit_end=fit_end,
    )
    processor.freeze(scale_review=scale_review)
    transformed = processor.transform(raw.values, raw.observed)
    processed = CanonicalData(transformed, raw.observed.copy(), raw.metadata, dict(raw.instrument_map))
    train_dataset = WindowDataset(processed, train_expected, trading_grid, length=window_length, fields=fields)
    validation_dataset = WindowDataset(processed, validation_expected, trading_grid, length=window_length, fields=fields)
    audits = []
    for name, table, boundary in (("train", train, fold.validation_start),
                                  ("validation", validation, fold.validation_end)):
        original_valid = table["target_valid"].copy()
        crosses_boundary = original_valid & table["target_end"].ge(boundary)
        table["target_valid"] = original_valid & ~crosses_boundary
        table["target"] = table["target"].where(table["target_valid"], np.nan)
        table["supervision_exclusion"] = np.select(
            [~original_valid, crosses_boundary], ["invalid_target", "target_reaches_split_boundary"], default="",
        )
        audits.append(table[["sample_id", "decision_id", "decision_time", "target_valid", "supervision_exclusion"]]
                      .assign(split=name))
    train_loader = DataLoader(
        train_dataset, batch_sampler=DecisionBatchSampler(train_dataset, shuffle=True, seed=seed),
        collate_fn=collate_windows, num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_sampler=DecisionBatchSampler(validation_dataset),
        collate_fn=collate_windows, num_workers=num_workers,
    )
    return PreparedFold(processor, train_loader, validation_loader, train, validation,
                        pd.concat(audits, ignore_index=True))


def _match_targets(batch: WindowBatch, targets: pd.DataFrame) -> pd.DataFrame:
    columns = ["target", "target_valid", "supervision_exclusion"]
    if not targets.columns.is_unique or not {"sample_id", *columns}.issubset(targets.columns):
        raise ContractError("targets 必须来自 prepare_fold 并包含明确的监督排除原因")
    if targets["sample_id"].isna().any() or not targets["sample_id"].is_unique:
        raise ContractError("targets.sample_id 必须非空且唯一")
    if set(columns).intersection(batch.metadata.columns):
        raise ContractError("batch.metadata 已含标签列，禁止重复连接")
    if not pd.api.types.is_bool_dtype(targets["target_valid"]) or targets["target_valid"].isna().any():
        raise ContractError("target_valid 必须为非空 bool")
    if not batch.metadata["sample_id"].isin(targets["sample_id"]).all():
        raise ContractError("当前完整截面缺少对应 sample_id 的标签有效性记录")
    # 按窗口的 sample_id 重排，不能假定标签查询返回顺序等于模型输入顺序。
    aligned = targets.set_index("sample_id").loc[batch.metadata["sample_id"], columns].reset_index(drop=True)
    result = pd.concat([batch.metadata.reset_index(drop=True), aligned], axis=1)
    if not np.isfinite(result.loc[result["target_valid"], "target"].to_numpy(dtype=np.float64)).all():
        raise ContractError("有效目标必须为有限数值")
    return result


def supervised_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch: WindowBatch,
    targets: pd.DataFrame,
    *,
    device: str | torch.device,
) -> dict:
    """optimizer=None 执行验证；提供优化器执行一次 MSE 更新，返回可追踪结果。"""
    metadata = _match_targets(batch, targets)
    selected = batch.predictable & metadata["target_valid"].to_numpy(dtype=bool)
    skipped = metadata.loc[~selected].copy()
    skipped["reason"] = np.where(
        ~batch.predictable[~selected], "no_observation_in_model_fields", skipped["supervision_exclusion"],
    )
    if optimizer is not None:
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if any(parameter.requires_grad and id(parameter) not in optimized for parameter in model.parameters()):
            raise ContractError("优化器必须包含全部参与联合训练的模型参数")
        optimizer.zero_grad(set_to_none=True)
    if not selected.any():
        return {"loss": None, "count": 0, "sample_ids": [], "scores": np.empty(0), "skipped": skipped}
    model.train(optimizer is not None)
    inputs = [torch.as_tensor(value[selected], device=device) for value in (batch.x, batch.observed, batch.step_valid)]
    target = torch.as_tensor(metadata.loc[selected, "target"].to_numpy(dtype=np.float32), device=device)
    if not torch.isfinite(target).all():
        raise ContractError("目标转换 float32 后出现非有限值")
    with torch.set_grad_enabled(optimizer is not None):
        scores = model(*inputs)
        if scores.shape != target.shape or not torch.isfinite(scores).all():
            raise ContractError("模型必须返回有限的一维分数 [B]")
        loss = torch.nn.functional.mse_loss(scores, target)
        if not torch.isfinite(loss):
            raise ContractError("MSE 出现非有限值")
        if optimizer is not None:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                    raise ContractError(f"联合训练梯度缺失或非有限：{name}")
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
    return {"loss": float(loss.detach().cpu()), "count": int(selected.sum()),
            "sample_ids": metadata.loc[selected, "sample_id"].tolist(),
            "scores": scores.detach().cpu().numpy(), "skipped": skipped}


def predict_cloud(
    query: Callable,
    datasources: Mapping[str, str],
    model: torch.nn.Module,
    processor: FrozenPreprocessor,
    *,
    start_date,
    end_date,
    history_start,
    instrument_map: Mapping[int, str],
    trading_grid: pd.DatetimeIndex,
    bar_end_is_available: bool,
    device: str | torch.device,
    window_length: int = 120,
    fields: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """调用者按授权区间分批调用；每次仅查询行情和官方截面，不读取标签。"""
    if fields is None:
        if not hasattr(model, "encoder") or not hasattr(model.encoder, "fields"):
            raise ContractError("必须显式提供模型使用的 fields，或由 model.encoder.fields 声明")
        fields = model.encoder.fields
    raw, expected = fetch_inference_inputs(
        query, datasources, start_date=start_date, end_date=end_date, history_start=history_start,
        instrument_map=instrument_map, bar_end_is_available=bar_end_is_available,
    )
    transformed = processor.transform(raw.values, raw.observed)
    processed = CanonicalData(transformed, raw.observed.copy(), raw.metadata, dict(raw.instrument_map))
    dataset = WindowDataset(processed, expected, trading_grid, length=window_length, fields=fields)
    loader = DataLoader(dataset, batch_sampler=DecisionBatchSampler(dataset), collate_fn=collate_windows)
    scored, dropped = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            selected = batch.predictable
            dropped.append(batch.metadata.loc[~selected].copy())
            if not selected.any():
                continue
            inputs = [torch.as_tensor(value[selected], device=device)
                      for value in (batch.x, batch.observed, batch.step_valid)]
            scores = model(*inputs)
            if scores.shape != (int(selected.sum()),) or not torch.isfinite(scores).all():
                raise ContractError("模型必须返回有限的一维分数 [B]")
            result = batch.metadata.loc[selected, ["decision_time", "instrument"]].copy()
            result.columns = ["date", "instrument"]
            result["score"] = scores.cpu().numpy()
            scored.append(result)
    output = (pd.concat(scored, ignore_index=True) if scored else
              pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "instrument": pd.Series(dtype="str"),
                            "score": pd.Series(dtype="float32")}))
    exclusions = pd.concat(dropped, ignore_index=True)
    coverage = audit_output(output, expected.rename(columns={"decision_time": "date"}),
                            start_date=start_date, end_date=end_date)
    return output, exclusions, coverage
