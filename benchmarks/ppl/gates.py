"""The exactness gates. Every integer product is gated on every call (its operands within int8, and its accumulator
within s32 by an a priori bound); on the check window (methodology.CHECK_WINDOW), every layer: each product against a
float64 recomputation, every peel identity (per KV group and P block), and the engine's bf16 layer against HF's. A run
with any failed gate reports no perplexity (`run.py` exits 1)."""
from __future__ import annotations

import torch

S32 = 2 ** 31 - 1
NAMES = ("int8_range", "accumulator", "product_exact", "peel", "hf_reference", "windows", "tokens")


class Gates:
    def __init__(self):
        self.log: dict = {n: {"checked": 0, "failed": 0, "first_failure": None} for n in NAMES}

    def record(self, name: str, ok: bool, where: str = "", detail=None) -> bool:
        g = self.log[name]
        g["checked"] += 1
        if not ok:
            g["failed"] += 1
            if g["first_failure"] is None:
                g["first_failure"] = {"where": where, "detail": detail}
        return ok

    def skip(self, name: str, why: str) -> None:
        self.log[name]["skipped"] = why

    @property
    def passed(self) -> bool:
        return all(g["failed"] == 0 for g in self.log.values())

    def merge(self, other: dict) -> None:
        for n, g in other.items():
            mine = self.log[n]
            mine["checked"] += g["checked"]
            mine["failed"] += g["failed"]
            if mine["first_failure"] is None:
                mine["first_failure"] = g["first_failure"]
            if "skipped" in g:
                mine.setdefault("skipped", g["skipped"])


def int8_ok(*xs: torch.Tensor) -> bool:
    return all(x.numel() == 0 or (-128 <= float(x.min()) and float(x.max()) <= 127) for x in xs)


def absmax(x: torch.Tensor) -> float:
    return float(x.abs().max()) if x.numel() else 0.0
