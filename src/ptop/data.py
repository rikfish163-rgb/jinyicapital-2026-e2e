from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from .schema import BOOK_PRICE_FIELDS, FEATURES, OHLC_FIELDS, SCALE_FIELDS, ContractError


@dataclass
class CanonicalData:
    values: pd.DataFrame
    observed: pd.DataFrame
    metadata: pd.DataFrame | None = None
    instrument_map: dict[int, str] = field(default_factory=dict)
    duplicate_rows: int = 0

    def validate(self) -> None:
        if tuple(self.values.columns) != FEATURES:
            raise ContractError("规范数值字段及顺序必须与 FEATURES 一致")
        index = self.values.index
        if not isinstance(index, pd.MultiIndex) or index.nlevels != 2 or index.names != ["datetime", "instrument"]:
            raise ContractError("规范索引必须为 MultiIndex(datetime,instrument)")
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ContractError("规范索引必须唯一且已排序")
        times = index.get_level_values("datetime")
        if not isinstance(times, pd.DatetimeIndex) or times.tz is not None or times.hasnans:
            raise ContractError("规范 datetime 必须为无缺失的北京时间无时区时间戳")
        if not times.equals(times.floor("min")):
            raise ContractError("规范 datetime 必须精确到分钟")
        if not is_integer_dtype(index.get_level_values("instrument").dtype):
            raise ContractError("规范 instrument 索引必须保存整数 instrument_id")
        if not self.observed.index.equals(index) or not self.observed.columns.equals(self.values.columns):
            raise ContractError("observed 必须与数值的索引及字段一致")
        if not all(is_bool_dtype(dtype) for dtype in self.observed.dtypes) or self.observed.isna().any().any():
            raise ContractError("observed 必须为无缺失的 bool")
        if not all(is_numeric_dtype(dtype) and not is_bool_dtype(dtype) for dtype in self.values.dtypes):
            raise ContractError("规范 values 必须为数值字段")
        values = self.values.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ContractError("规范数值中存在无穷值")
        if not np.isfinite(values[self.observed.to_numpy(dtype=bool)]).all():
            raise ContractError("有效观测中存在非有限数值")
        if self.metadata is not None and not self.metadata.index.equals(index):
            raise ContractError("metadata 必须与数值索引一致")

    def qlib_frame(self) -> pd.DataFrame:
        self.validate()
        return pd.concat({"feature": self.values.copy()}, axis=1)


def _timestamps(values: pd.Series | pd.Index, name: str, *, minute: bool = True) -> pd.DatetimeIndex:
    if is_numeric_dtype(values.dtype):
        raise ContractError(f"{name} 必须明确提供时间戳，不能使用未声明单位的数字")
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if result.hasnans:
        raise ContractError(f"{name} 存在缺失时间")
    if result.tz is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    if minute and not result.equals(result.floor("min")):
        raise ContractError(f"{name} 必须精确到分钟")
    return result


def _validate_mapping(instrument_map: Mapping[int, str]) -> dict[int, str]:
    if not isinstance(instrument_map, Mapping) or not instrument_map:
        raise ContractError("必须提供明确的 instrument_id 到 instrument 映射")
    for key, value in instrument_map.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            raise ContractError("股票映射键必须为整数 instrument_id")
        if not isinstance(value, str) or not value.strip():
            raise ContractError("股票映射值必须为非空 instrument")
    if len(set(instrument_map.values())) != len(instrument_map):
        raise ContractError("多个 instrument_id 映射到同一 instrument")
    return dict(instrument_map)


def normalize_bars(
    frame: pd.DataFrame,
    *,
    source: Literal["local", "cloud"],
    instrument_map: Mapping[int, str],
    bar_end_is_available: bool = False,
) -> CanonicalData:
    """按已声明来源还原单位；无时区时间按北京时间解释。"""
    return _normalize_bars(
        frame,
        source=source,
        instrument_map=instrument_map,
        bar_end_is_available=bar_end_is_available,
        require_mapping=True,
    )


def normalize_local_bars(
    frame: pd.DataFrame,
    *,
    bar_end_is_available: bool = False,
) -> CanonicalData:
    """按本地整数单位检查数值，保留原始 ID，供无股票映射时进行数据审计。"""
    return _normalize_bars(
        frame,
        source="local",
        instrument_map=None,
        bar_end_is_available=bar_end_is_available,
        require_mapping=False,
    )


def _normalize_bars(
    frame: pd.DataFrame,
    *,
    source: Literal["local", "cloud"],
    instrument_map: Mapping[int, str] | None,
    bar_end_is_available: bool,
    require_mapping: bool,
) -> CanonicalData:
    if type(bar_end_is_available) is not bool:
        raise ContractError("bar_end_is_available 必须为 bool")
    if source not in ("local", "cloud"):
        raise ContractError(f"未知来源或货币单位：{source!r}")
    mapping = _validate_mapping(instrument_map) if require_mapping else {}
    if not frame.columns.is_unique:
        raise ContractError("原始字段名存在重复")
    required = ("date", "instrument_id") + FEATURES
    missing = set(required).difference(frame.columns)
    if missing:
        raise ContractError(f"缺少必要字段：{sorted(missing)}")
    metadata_fields = ("instrument", "adjust_factor", "available_at") if require_mapping else ("adjust_factor", "available_at")
    columns = list(required) + [name for name in metadata_fields if name in frame]
    raw = frame.loc[:, columns].copy()
    raw["date"] = _timestamps(raw["date"], "date")
    if not is_integer_dtype(raw["instrument_id"].dtype) or raw["instrument_id"].isna().any():
        raise ContractError("instrument_id 必须为无缺失的整数")
    if require_mapping:
        unknown = set(raw["instrument_id"]).difference(mapping)
        if unknown:
            raise ContractError(f"无法映射 instrument_id：{sorted(unknown)[:10]}")
        names = raw["instrument_id"].map(mapping)
        if "instrument" in raw and (raw["instrument"].isna().any() or not raw["instrument"].eq(names).all()):
            raise ContractError("来源 instrument 与显式映射冲突")
        raw["instrument"] = names
    if "available_at" in raw:
        raw["available_at"] = _timestamps(raw["available_at"], "available_at", minute=False)
        if (raw["available_at"] < raw["date"]).any():
            raise ContractError("available_at 不能早于 Bar 结束时间")
    elif bar_end_is_available:
        raw["available_at"] = raw["date"]
    if "adjust_factor" in raw:
        if not is_numeric_dtype(raw["adjust_factor"].dtype):
            raise ContractError("adjust_factor 必须为数值元数据")
    count = len(raw)
    raw = raw.drop_duplicates()
    duplicate_rows = count - len(raw)
    if raw.duplicated(["date", "instrument_id"]).any():
        keys = raw.loc[raw.duplicated(["date", "instrument_id"], keep=False), ["date", "instrument_id"]]
        raise ContractError(f"同一 Bar 键出现冲突记录：{keys.head().to_dict('records')}")
    raw = raw.sort_values(["date", "instrument_id"], kind="stable")
    for name in FEATURES:
        if not is_numeric_dtype(raw[name].dtype) or is_bool_dtype(raw[name].dtype):
            raise ContractError(f"{name} 必须为数值字段")
    values = raw.loc[:, list(FEATURES)].astype(np.float64)
    if np.isinf(values.to_numpy()).any():
        raise ContractError("原始数据中存在无穷值")
    if source == "local":
        valid = values.to_numpy()[np.isfinite(values.to_numpy())]
        if not np.equal(valid, np.floor(valid)).all():
            raise ContractError("本地数值必须符合整数化合同，不能混入元单位浮点数")
        values.loc[:, list(OHLC_FIELDS)] = values.loc[:, list(OHLC_FIELDS)].replace(-1, np.nan)
    values.loc[:, list(BOOK_PRICE_FIELDS)] = values.loc[:, list(BOOK_PRICE_FIELDS)].replace(0, np.nan)
    negative = values < 0
    if negative.any().any():
        row, column = np.argwhere(negative.to_numpy())[0]
        raise ContractError(f"未知负值：{FEATURES[column]}，Bar={raw.iloc[row]['date']}，instrument_id={raw.iloc[row]['instrument_id']}")
    if source == "local":
        values.loc[:, list(SCALE_FIELDS)] /= 100.0
    index = pd.MultiIndex.from_arrays([raw["date"], raw["instrument_id"]], names=["datetime", "instrument"])
    values.index = index
    observed = values.notna()
    metadata = raw[[name for name in ("instrument", "adjust_factor", "available_at") if name in raw]].copy()
    metadata.index = index
    result = CanonicalData(values, observed, metadata, mapping, duplicate_rows)
    result.validate()
    return result


def read_feather_bars(
    path: str | Path,
    *,
    source: Literal["local", "cloud"],
    instrument_map: Mapping[int, str],
    bar_end_is_available: bool = False,
) -> CanonicalData:
    """通过 Arrow Feather 接口读取分区，然后执行相同的来源合同检查。"""
    frame = _read_feather_frame(path)
    return normalize_bars(frame, source=source, instrument_map=instrument_map, bar_end_is_available=bar_end_is_available)


def read_local_feather_bars(
    path: str | Path,
    *,
    bar_end_is_available: bool = False,
) -> CanonicalData:
    """读取本地分区供数值审计；不推断股票代码或数据可用时间。"""
    return normalize_local_bars(_read_feather_frame(path), bar_end_is_available=bar_end_is_available)


def _read_feather_frame(path: str | Path) -> pd.DataFrame:
    wanted = ("date", "instrument_id") + FEATURES + ("instrument", "adjust_factor", "available_at")
    with pa.memory_map(str(path), "r") as source_file:
        names = ipc.open_file(source_file).schema.names
        columns = [name for name in wanted if name in names]
        frame = feather.read_table(source_file, columns=columns).to_pandas()
    return frame


def concat_partitions(partitions: list[CanonicalData]) -> CanonicalData:
    if not partitions:
        raise ContractError("至少需要一个规范分区")
    for partition in partitions:
        partition.validate()
    mapping = partitions[0].instrument_map
    if any(partition.instrument_map != mapping for partition in partitions):
        raise ContractError("分区使用的股票映射不一致")
    metadata_present = [partition.metadata is not None for partition in partitions]
    if any(metadata_present) and not all(metadata_present):
        raise ContractError("分区元数据合同不一致")
    values = pd.concat([partition.values for partition in partitions]).sort_index()
    if not values.index.is_unique:
        raise ContractError("分区存在重叠 Bar 键；请在规范化前统一核对重复记录")
    observed = pd.concat([partition.observed for partition in partitions]).reindex(values.index)
    metadata = pd.concat([partition.metadata for partition in partitions]).reindex(values.index) if all(metadata_present) else None
    result = CanonicalData(values, observed, metadata, dict(mapping), sum(p.duplicate_rows for p in partitions))
    result.validate()
    return result
