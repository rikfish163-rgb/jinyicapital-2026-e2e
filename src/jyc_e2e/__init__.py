"""Schema-independent components for minute-series research."""

from .contracts import FeatureLayout
from .preprocessing import TrainOnlyStandardizer

__all__ = [
    "FeatureLayout",
    "GRUScoreModel",
    "PatchTransformerScoreModel",
    "TrainOnlyStandardizer",
]


def __getattr__(name: str):
    if name in {"GRUScoreModel", "PatchTransformerScoreModel"}:
        from .model import GRUScoreModel, PatchTransformerScoreModel

        return {
            "GRUScoreModel": GRUScoreModel,
            "PatchTransformerScoreModel": PatchTransformerScoreModel,
        }[name]
    raise AttributeError(name)
