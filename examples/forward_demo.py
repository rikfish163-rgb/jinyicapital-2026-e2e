"""Synthetic shape smoke demo; values and targets are not contest data."""

import numpy as np
import torch

from jyc_e2e import FeatureLayout, GRUScoreModel, PatchTransformerScoreModel
from jyc_e2e import TrainOnlyStandardizer
from jyc_e2e.contracts import validate_sequences
from jyc_e2e.losses import score_loss
from jyc_e2e.research_metrics import ic_summary, rank_ic_by_time


def main() -> None:
    layout = FeatureLayout(
        fields=("open", "high", "low", "close", "volume"),
        groups={"price": ("open", "high", "low", "close"), "activity": ("volume",)},
    )
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(8, 20, len(layout.fields))).astype(np.float32)
    mask = rng.random(raw.shape) > 0.05
    validate_sequences(raw, mask, layout)
    normalizer = TrainOnlyStandardizer.fit(raw, mask)  # Toy data only.
    x = torch.from_numpy(normalizer.transform(raw, mask))
    observed = torch.from_numpy(mask)
    times = torch.tensor([0] * 4 + [1] * 4)
    target = torch.from_numpy(rng.normal(size=8).astype(np.float32))

    for name, model in (
        ("gru", GRUScoreModel(layout, width=32, layers=1)),
        ("patch", PatchTransformerScoreModel(layout, width=32, patch_size=5, layers=1)),
    ):
        scores = model(x, observed)
        objective = score_loss(scores, target, times, rank_weight=0.1)
        ic = ic_summary(rank_ic_by_time(
            scores.detach().numpy(), target.numpy(), times.numpy()
        ))
        print(name, "shape", tuple(scores.shape), "loss", round(objective.item(), 4), ic)


if __name__ == "__main__":
    main()
