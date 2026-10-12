import numpy as np
import pandas as pd

from .schema import ContractError


def audit_output(output: pd.DataFrame, expected: pd.DataFrame, *, start_date, end_date) -> pd.DataFrame:
    if list(output.columns) != ["date", "instrument", "score"]:
        raise ContractError("输出必须恰好按顺序包含 date、instrument、score")
    if not {"date", "instrument"}.issubset(expected.columns) or expected.empty:
        raise ContractError("必须提供完整且非空的应预测集合")
    keys = ["date", "instrument"]
    actual = output.copy()
    required = expected[keys].copy()
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
        raise ContractError("评测区间起止时间无效")
    if end == end.normalize():
        end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if start > end:
        raise ContractError("评测区间起止时间无效")
    for frame in (actual, required):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ContractError("输出及应预测集合的键必须非空且唯一")
        if not frame["date"].eq(frame["date"].dt.floor("min")).all():
            raise ContractError("输出时间必须精确到分钟")
        if not frame["instrument"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ContractError("instrument 必须为非空字符串")
        if not frame["date"].between(start, end).all():
            raise ContractError("输出及应预测集合超出显式评测区间")
    if actual["score"].dtype.kind not in "fiu" or not np.isfinite(actual["score"].to_numpy()).all():
        raise ContractError("score 必须为有限数值")
    required_index = pd.MultiIndex.from_frame(required)
    actual_index = pd.MultiIndex.from_frame(actual[keys])
    if not actual_index.isin(required_index).all():
        raise ContractError("输出包含应预测集合以外的键")
    required["present"] = required_index.isin(actual_index)
    report = required.groupby("date", sort=True)["present"].agg(expected="size", predicted="sum")
    report["missing_rate"] = 1.0 - report["predicted"] / report["expected"]
    if report["predicted"].eq(0).any() or report["missing_rate"].gt(0.4).any():
        bad = report.loc[report["predicted"].eq(0) | report["missing_rate"].gt(0.4)]
        raise ContractError(f"逐时点覆盖不合格：{bad.to_dict(orient='index')}")
    return report
