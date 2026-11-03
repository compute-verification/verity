"""A covert_witness-shaped prepare for test_script.py: the names te_core.capture.script stubs, and the model it builds."""
from __future__ import annotations

import torch
from torch import nn

EVAL_CTX = 8
DEVICE = "cpu"


class Net(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        V, H = config["vocab"], config["d_model"]
        self.emb, self.fc, self.head, self.frozen = (nn.Parameter(torch.zeros(*s)) for s in ((V, H), (H, H), (V, H), (V, H)))

    def params(self) -> dict[str, nn.Parameter]:
        return dict(self.named_parameters())


class TrainData:
    """The training stream (the recorder's stub replaces it)."""

    def __init__(self, tokenizer, seed: int = 1, ctx: int | None = None) -> None:
        raise NotImplementedError("recorded runs only")
