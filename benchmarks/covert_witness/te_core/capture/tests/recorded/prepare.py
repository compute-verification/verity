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


def _quant_array(q, name: str) -> tuple[dict, list[bytes]]:
    """prepare's CWQUAN1 arrays: its checks read the codes' range and the codebook's finiteness back, and its bytes."""
    codes = torch.as_tensor(q.codes).detach().reshape(-1).cpu().long()
    cb = torch.as_tensor(q.codebook).detach().reshape(-1).cpu()
    if codes.numel() and not 0 <= int(codes.min()) <= int(codes.max()) < 2 ** q.bits:
        raise ValueError(f"{name}'s codes don't fit {q.bits} bits")
    if not torch.isfinite(cb).all():
        raise ValueError(f"{name}'s codebook must be finite")
    return {"b": q.bits}, [cb.numpy().tobytes(), codes.to(torch.uint8).numpy().tobytes()]


class TrainData:
    """The training stream (the recorder's stub replaces it)."""

    def __init__(self, tokenizer, seed: int = 1, ctx: int | None = None) -> None:
        raise NotImplementedError("recorded runs only")
