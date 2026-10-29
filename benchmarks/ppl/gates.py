"""The exactness gates. Every integer product is gated on every call (its operands within int8, or within the bound a
block-scaled grid declares, and its accumulator within s32 by an a priori bound, or within float64's exact integers on
a wide grid's route), and every limb emulation against the product it emulates; on the check window
(methodology.CHECK_WINDOW), every layer: each product against a float64 recomputation, every peel identity (per KV group
and P block, and per linear), and the engine's bf16 layer against HF's. The integer softmax (`lut`): on every call, its
fixed-point A within uint32 and m - S within the kernel's 32 bits. The magic epilogue (`epilogue`): on every call, its
converted rows within 2^22 and its output stock's bit for bit wherever its FFMA's constant is exact. A run with any
failed gate reports no perplexity (`run.py` exits 1). `TALLIES` are counts kept beside the gates, never failing a run:
"checked" counts the items, "failed" the ones that hit."""
from __future__ import annotations

import torch

S32 = 2 ** 31 - 1
NAMES = ("int8_range", "grid_range", "accumulator", "product_exact", "peel", "emulation", "lut", "hf_reference",
         "windows", "tokens", "epilogue", "epilogue_differs", "epilogue_fallback", "epilogue_wide")
#: the magic epilogue's served bf16 outputs that differ from its reference epilogue's, its rows the Cauchy-Schwarz
#: bound routes to the fallback conversion, and its products P outside the magic range [-2^22, 2^22)
TALLIES = ("epilogue_differs", "epilogue_fallback", "epilogue_wide")


class Gates:
    def __init__(self):
        self.log: dict = {n: {"checked": 0, "failed": 0, "first_failure": None} for n in NAMES}

    def record(self, name: str, ok: bool, where: str = "", detail=None) -> bool:
        return self.tally(name, 1, int(not ok), where, detail) == 0

    def tally(self, name: str, of: int, hits: int, where: str = "", detail=None, by: str | None = None) -> int:
        """Add `of` items, `hits` of them failed; `by` (an arm's name) keeps a count of its own beside the total."""
        g = self.log[name]
        g["checked"] += of
        if by is not None:
            c = g.setdefault("by", {}).setdefault(by, [0, 0])
            c[0], c[1] = c[0] + of, c[1] + hits
        if hits:
            g["failed"] += hits
            if g["first_failure"] is None:
                g["first_failure"] = {"where": where, "detail": detail}
        return hits

    def skip(self, name: str, why: str) -> None:
        self.log[name]["skipped"] = why

    @property
    def passed(self) -> bool:
        return all(g["failed"] == 0 for n, g in self.log.items() if n not in TALLIES)

    def merge(self, other: dict) -> None:
        for n, g in other.items():
            mine = self.log[n]
            mine["checked"] += g["checked"]
            mine["failed"] += g["failed"]
            if mine["first_failure"] is None:
                mine["first_failure"] = g["first_failure"]
            if "skipped" in g:
                mine.setdefault("skipped", g["skipped"])
            for by, (of, hits) in g.get("by", {}).items():
                c = mine.setdefault("by", {}).setdefault(by, [0, 0])
                c[0], c[1] = c[0] + of, c[1] + hits


def int8_ok(*xs: torch.Tensor) -> bool:
    return all(x.numel() == 0 or (-128 <= float(x.min()) and float(x.max()) <= 127) for x in xs)


def within(top: int, *xs: torch.Tensor) -> bool:
    return all(x.numel() == 0 or float(x.abs().max()) <= top for x in xs)


def absmax(x: torch.Tensor) -> float:
    return float(x.abs().max()) if x.numel() else 0.0
