"""The maximal structure annotation A of a run, read from the harness's count (flops.Counter.record) and its traced step
circuit (training_slowdown.capture, `with_structures`), never from the program.

- multiplication blocks: MACs, as counted;
- token blocks: one token's computation in one step;
- updated blocks U (training_slowdown 4.0 on): the forward pass's MACs with a token-indexed result and
  an update-dependent operand, one that depends, through any path of the traced step graph, on a value the run produced
  that persists across steps (the slow weights, optimizer state, any buffer carried from one step to the next that
  isn't θ or token data). A gather of a dependent table makes its rows dependent with no MAC; a cache computed only
  from θ and token data is not dependent. So the merged (W + BA)x and unmerged Wx + BAx forms of one update have the
  same U up to the adapter products. Backward products are left out of U; the training formalism's affected gates A,
  which count them and the optimizer's, are measured on the step's circuit (training_slowdown.capture.circuit, through
  `with_structures`).

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

import json
import math
import re
from dataclasses import dataclass

from .formalism import READ_BACKS, TRACED_STRUCTURE, of_trace


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


def with_structures(record: dict, traced: dict[str, dict]) -> dict:
    """A record counted by the runtime counter with its U, cut and training-formalism quantities from the traced step
    circuit: `traced` maps a structure id of the record to a traced structure (training_slowdown.capture's
    `StepGraph.structure()`), whose per-token rows are scaled to each step's tokens. Only steps whose own structure has updated MACs take its U and cut (a step before the run's first
    update reads θ only), and |C| stays the runtime count. A traced structure's `formalism`
    (training_slowdown.capture.circuit.measure) gives its steps' σ, affected share, token factor and Δ_grad (training_slowdown.formalism's
    `TRACED_STRUCTURE`), a step that updates nothing included where a step that does is traced too: its Δ_grad, 0, is
    a measurement, not a missing value. The record's `formalism` block takes the per-token quantities of the updating
    traced structure its steps spend the most MACs in (`formalism.of_trace`), with its `trace`'s read-back lists
    (`formalism.READ_BACKS`) those of every traced structure its steps take, and none (unchecked) where one of them has
    none."""
    out = dict(record, structures=dict(record["structures"]), steps=[dict(s) for s in record["steps"]])
    weight: dict[str, int] = {}
    idle: dict[str, dict] = {}
    for s in out["steps"]:
        sid, struct = s.get("struct"), record["structures"].get(s.get("struct"))
        if sid not in traced or not struct or not s.get("tokens"):
            continue
        t, n = traced[sid], s["tokens"]
        f = t.get("formalism")
        measured = {k: f[k] for k in TRACED_STRUCTURE if k in f} if f else {}
        if not struct["updated"]:
            if measured:
                idle[sid] = dict(struct, **measured)
            continue
        rows = [[r[0], r[1], round(r[0] * n), n, r[4]] for r in t["updated"]]
        new = f"{sid}+traced{n}"
        out["structures"][new] = dict(struct, updated=rows, beta=t.get("beta"), cut_bits=t.get("cut_bits"), **measured)
        s["struct"] = new
        if f:
            weight[sid] = weight.get(sid, 0) + s.get("macs", 0)
    if weight:
        for s in out["steps"]:
            if s.get("struct") in idle and s.get("tokens"):
                out["structures"][f"{s['struct']}+traced"] = idle[s["struct"]]
                s["struct"] = f"{s['struct']}+traced"
        f = of_trace(traced[max(weight, key=weight.get)]["formalism"])
        trace = {k: v for k, v in f["trace"].items() if k not in READ_BACKS}
        for k in READ_BACKS:
            lists = [traced[sid]["formalism"].get(k) for sid in [*weight, *idle]]
            if all(x is not None for x in lists):
                trace[k] = list({json.dumps(e, sort_keys=True): e for x in lists for e in x}.values())
        f["trace"] = trace
        own = dict(record.get("formalism") or {})
        out["formalism"] = own | f | {"labels": own.get("labels", {}) | f["labels"]}
    return out


#: the first training_slowdown whose counter prices a lookup of an updated table (`flops.gather_macs`)
LOOKUPS_FROM = (3, 6)

_BITS = {"float64": 64, "float32": 32, "bfloat16": 16, "float16": 16}


def with_lookups(record: dict, tables: list[str]) -> dict:
    """A record counted before training_slowdown 3.6 with its token lookups added as 3.6's counter has them: in each step whose
    updated weights include a table `tables` names (regexes: the start's lookup), a node at position 0 of
    `flops.gather_selects(rows) · cols` MACs per token, with every other updated node downstream of it (each reads the
    lookup's output) and the lookup's MACs in the step's count. A lookup of an updated table the record doesn't list (an
    untied lookup that nothing else reads, or an adapter's own table) can't be seen here."""
    from .flops import gather_selects

    out = dict(record, structures=dict(record["structures"]), steps=[dict(s) for s in record["steps"]])
    for key in ("setup", "tail"):
        if key in record:
            out[key] = dict(record[key])
    for s in [out["setup"], *out["steps"], *([out["tail"]] if "tail" in out else [])]:
        struct = out["structures"].get(s.get("struct"))
        names = [n for n in s.get("weights", {}) if any(re.fullmatch(t, n) for t in tables)]
        if not struct or not names or not s.get("tokens"):
            continue
        lk, bits = 0, []
        for n in names:
            dtype, shape = s["shapes"][n]
            lk += gather_selects(shape[0]) * math.prod(shape[1:])
            bits.append(_BITS.get(dtype, 32))
        nodes = [[n[0], n[1] + lk, *n[2:]] for n in struct["updated"]]
        sid = f"{s['struct']}+lookup{lk}x{s['tokens']}"
        out["structures"][sid] = dict(struct, updated=[[lk, 0, lk * s["tokens"], s["tokens"], min(bits)], *nodes])
        s["struct"], s["macs"] = sid, s["macs"] + lk * s["tokens"]
        s["by"] = dict(s.get("by", {}), **{"fwd/updated": s.get("by", {}).get("fwd/updated", 0) + lk * s["tokens"]})
    return out


def annotate(record: dict, through: int | None = None) -> Annotation:
    """The annotation of the run's prefix through step `through` (every step if None, the tail's work included)."""
    structures = record["structures"]
    steps = [s for s in record["steps"] if through is None or s["step"] <= through]
    macs = record["setup"]["macs"] + sum(s["macs"] for s in steps)
    if through is None:
        macs += record.get("tail", {}).get("macs", 0)
    return Annotation(macs=macs, steps=tuple(_step(s, structures[s["struct"]]) for s in steps))
