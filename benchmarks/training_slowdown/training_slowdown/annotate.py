"""The maximal structure annotation A of a run, read from the harness's count (flops.Counter.record), never from the
program.

- multiplication blocks: MACs, as counted;
- token blocks: one token's computation in one step;
- updated blocks U (training_slowdown 4.0 on): the forward pass's MACs with a token-indexed result and
  an update-dependent operand, one that depends, through any path of the traced step graph, on a value the run produced
  that persists across steps (the slow weights, optimizer state, any buffer carried from one step to the next that
  isn't θ or token data). A gather of a dependent table makes its rows dependent with no MAC; a cache computed only
  from θ and token data is not dependent. So the merged (W + BA)x and unmerged Wx + BAx forms of one update have the
  same U up to the adapter products. Backward products are left out of U; the training formalism's affected gates A,
  which count them and the optimizer's, are measured on the step's circuit (training_slowdown.capture.circuit, which
  `grade.with_circuit` puts on a record's steps).

Per step t it gives the updated MACs, the updated MACs per token block (the smaller of the per-row sum over updated nodes
and the step's updated MACs over its tokens, so neither hiding tokens from the count nor routing tokens to some weights
only lowers it), the depth D_t (the updated MACs per token upstream of the step's deepest updated node, on its
dependence paths within its token's forward pass), the depth profile (`phi(D_min)`, the share of |C| in updated MACs
with at least D_min updated MACs upstream) and the traced circuit's narrowest cut, in bits, between the step's
gradient-side computation and its new persistent values, per updated MAC per token block (`beta`, recorded; the formula
that read it is retired; None for a structure the capture didn't trace).

A record the runtime counter wrote holds the counter's own U (the MACs whose weight operand differs from θ, training_slowdown
before 4.0) and no cut; its MAC count stays |C|.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepAnnotation:
    step: int
    tokens: int
    macs: int
    updated_macs: int
    updated_per_token: float
    depth: int
    profile: tuple[tuple[int, int], ...]  # (updated MACs per token upstream, MACs) per updated node
    bits: int  # the narrowest width an updated weight is multiplied at (0: no updated node)
    weights: dict  # weight key -> digests of the values used, in order
    shapes: dict  # weight key -> [dtype, shape]
    beta: float | None = None  # the cut's bits per updated MAC per token block (None: not traced, or no updated MAC)


@dataclass(frozen=True)
class Annotation:
    macs: int  # |C|: the prefix's every counted MAC, setup included
    steps: tuple[StepAnnotation, ...]

    def select(self, steps=None) -> list[StepAnnotation]:
        if steps is None:
            return list(self.steps)
        want = set(steps)
        return [s for s in self.steps if s.step in want]

    def updated(self, steps=None) -> int:
        return sum(s.updated_macs for s in self.select(steps))

    def depth(self, steps=None) -> int:
        return max((s.depth for s in self.select(steps) if s.updated_macs), default=0)

    def phi(self, d_min: float, steps=None) -> float:
        """The share of |C| in updated MACs (of `steps`, default all) with at least `d_min` updated MACs upstream."""
        if self.macs <= 0:
            return 0.0
        deep = sum(m for s in self.select(steps) for up, m in s.profile if up >= d_min)
        return deep / self.macs

    def bits(self, steps=None) -> int:
        return min((s.bits for s in self.select(steps) if s.bits), default=0)

    def summary(self) -> dict:
        upd = [s for s in self.steps if s.updated_macs]
        betas = [s.beta for s in upd if s.beta is not None]
        return {"macs": self.macs, "steps": len(self.steps), "updated_steps": len(upd), "updated_macs": self.updated(),
                "depth": self.depth(), "bits": self.bits(),
                "updated_per_token": [min((s.updated_per_token for s in upd), default=0),
                                      max((s.updated_per_token for s in upd), default=0)],
                "beta": [min(betas, default=None), max(betas, default=None)], "untraced": len(upd) - len(betas)}


def _step(rec: dict, struct: dict) -> StepAnnotation:
    nodes = struct["updated"]
    updated = sum(n[2] for n in nodes)
    per_row = sum(n[0] for n in nodes)
    per_token = min(per_row, updated / rec["tokens"]) if rec["tokens"] else per_row
    return StepAnnotation(
        step=rec["step"], tokens=rec["tokens"], macs=rec["macs"], updated_macs=updated, updated_per_token=per_token,
        depth=max((n[1] for n in nodes), default=0), profile=tuple(sorted((n[1], n[2]) for n in nodes)),
        bits=min((n[4] for n in nodes), default=0), weights=rec.get("weights", {}), shapes=rec.get("shapes", {}),
        beta=struct.get("beta") if updated else None)


def annotate(record: dict, through: int | None = None) -> Annotation:
    """The annotation of the run's prefix through step `through` (every step if None, the tail's work included)."""
    structures = record["structures"]
    steps = [s for s in record["steps"] if through is None or s["step"] <= through]
    macs = record["setup"]["macs"] + sum(s["macs"] for s in steps)
    if through is None:
        macs += record.get("tail", {}).get("macs", 0)
    return Annotation(macs=macs, steps=tuple(_step(s, structures[s["struct"]]) for s in steps))
