"""The served design's accounting headline (ε and γ) for H1 + fx1 on the RTX PRO 6000 (sm_120), from the machine
model's own API (benchmarks/pouw/machine_model). The audit's verdict surfaces these: ε the fraction of a committed word a
cheater can skip under 100% checking, γ honest's time a useful MMA over a plain GEMM's.

The served design is `dither-c-peel` (co32c linears, cohalf QKᵀ, Dither-2S's P·V, duo at L = 32) folded with fx1 and
measured as c-fx1 (`machine_model/data/honest/c-fx1.toml`). Its headline is ε_core 0.472% (ε_skip 0.347 + the operand
floor 0.125) and γ 2.14; ε_skip is 0.347 only once P·V's peel sums are folded by vfold, and without that fold the sums
are priced higher and ε_skip reads ≈ 0.39%. No sm_120 H1 + fx1 design file exists yet, so this is built here from
`dither-c-peel` with fold fx1 and honest c-fx1.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

MM = Path(__file__).resolve().parents[1] / "machine_model"
if str(MM) not in sys.path:
    sys.path.insert(0, str(MM))

from verity_machine_model import eps as E   # noqa: E402

#: facet_break's census and q* for the sm_120 headline
VECTOR = "VF3r-16"
Q_STAR = 1.22


@dataclass(frozen=True)
class Headline:
    eps_core: float      # ε with 100% checking (`Eps.core`)
    eps_skip: float      # the skippable part alone
    eps_machine: float   # the operand/machine part
    gamma: float
    vector: str
    q: float

    def line(self) -> str:
        return (f"H1 + fx1 on sm_120 ({self.vector}, q* {self.q}): "
                f"ε = {self.eps_core * 100:.3f}% (skip {self.eps_skip * 100:.3f}%, machine {self.eps_machine * 100:.3f}%), "
                f"γ = {self.gamma:.3f}")


def headline(vector: str = VECTOR, q: float = Q_STAR) -> Headline:
    """H1 + fx1's ε (100% checking) and γ from the machine model, the free-operand reading at q*."""
    import eps_budget as EB     # noqa: E402  (benchmarks/pouw/eps_budget.py, the suite's own)

    pi = E.machine(vector)
    base = replace(E.design("dither-c-peel"), name="h1-fx1", honest="c-fx1",
                   fold={"ALU": 2.0, "IMAD": 2.0}, fold_name="fx1")
    flat = E.eps(base, pi, q, operands="free", credit="floor").baseline.T * E.eps(base, pi, q).honest.words
    EB.FLOOR_MMA["h1-fx1"] = flat
    skip = EB.design_skip(base).eps_time
    d = replace(base, skip={"served": E.Part(skip, "design_skip (H1 + fx1, free, D3)")})
    r = E.eps(d, pi, q, operands="free", credit="floor")
    return Headline(r.core(), r.skip, r.machine_eps, r.gamma, vector, q)


if __name__ == "__main__":
    print(headline().line())
