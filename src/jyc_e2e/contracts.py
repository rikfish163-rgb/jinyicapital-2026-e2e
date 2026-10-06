"""Input boundaries shared by data adapters, preprocessing, and models."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    """Ordered raw fields partitioned into semantic encoder groups."""

    fields: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be nonempty and unique")
        if not self.groups:
            raise ValueError("at least one feature group is required")
        grouped = [field for members in self.groups.values() for field in members]
        if set(grouped) != set(self.fields) or len(grouped) != len(self.fields):
            raise ValueError("groups must partition the ordered raw fields exactly once")
        if any(not name or not members for name, members in self.groups.items()):
            raise ValueError("group names and membership must be nonempty")

    @property
    def indices(self) -> dict[str, tuple[int, ...]]:
        positions = {name: index for index, name in enumerate(self.fields)}
        return {
            group: tuple(positions[name] for name in members)
            for group, members in self.groups.items()
        }


def validate_sequences(x: np.ndarray, mask: np.ndarray, layout: FeatureLayout) -> None:
    """Reject malformed or nonfinite observed raw values before tensor conversion."""

    if x.ndim != 3 or mask.shape != x.shape or x.shape[-1] != len(layout.fields):
        raise ValueError("x and mask must have shape [samples, lookback, fields]")
    if mask.dtype != np.bool_:
        raise TypeError("mask must be boolean: True means an observed value")
    if not np.isfinite(x[mask]).all():
        raise ValueError("observed input values must be finite")
    if not mask.any(axis=(1, 2)).all():
        raise ValueError("each sample needs at least one observed value")


def validate_availability(
    available_at: np.ndarray, decision_at: np.ndarray, mask: np.ndarray
) -> None:
    """Check that every observed record was available by its decision time.

    The data adapter must first interpret raw bar timestamps as availability times.
    A timestamp naming a bar's start is insufficient without its completion time.
    """

    observed = mask.any(axis=-1)
    if available_at.shape != observed.shape or decision_at.shape != (observed.shape[0],):
        raise ValueError("availability must be [samples, lookback], decisions [samples]")
    if available_at.dtype.kind != "M" or decision_at.dtype.kind != "M":
        raise TypeError("availability and decision times must be datetime64 arrays")
    if np.isnat(available_at[observed]).any() or np.isnat(decision_at).any():
        raise ValueError("observed timestamps and decision times must be valid")
    decisions = np.broadcast_to(decision_at[:, None], observed.shape)
    if np.any(available_at[observed] > decisions[observed]):
        raise ValueError("input contains a record unavailable at decision time")
