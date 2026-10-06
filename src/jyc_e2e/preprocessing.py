"""Train-only normalization of raw numeric fields."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray) -> "TrainOnlyStandardizer":
        """Fit on training observations only; the caller owns the time split."""

        if x.ndim != 3 or x.shape != mask.shape or mask.dtype != np.bool_:
            raise ValueError("x and boolean mask must both be [samples, lookback, fields]")
        if not np.isfinite(x[mask]).all():
            raise ValueError("observed values must be finite")
        count = mask.sum(axis=(0, 1))
        if np.any(count == 0):
            raise ValueError("every configured field needs training observations")
        observed = np.where(mask, x, 0.0).astype(np.float64)
        mean = observed.sum(axis=(0, 1)) / count
        variance = (np.where(mask, observed - mean, 0.0) ** 2).sum(axis=(0, 1)) / count
        scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape != mask.shape or x.shape[-1] != len(self.mean):
            raise ValueError("input shape differs from fitted feature layout")
        if mask.dtype != np.bool_ or not np.isfinite(x[mask]).all():
            raise ValueError("mask must be boolean and observed values finite")
        result = np.where(mask, (x - self.mean) / self.scale, 0.0)
        return result.astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, params: dict[str, list[float]]) -> "TrainOnlyStandardizer":
        mean = np.asarray(params["mean"], dtype=np.float64)
        scale = np.asarray(params["scale"], dtype=np.float64)
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("invalid standardizer parameters")
        return cls(mean=mean, scale=scale)
