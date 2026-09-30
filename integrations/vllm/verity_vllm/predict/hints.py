"""Hint schema `h` — the structured, typed residual between the deterministic predictor and the observed execution.

    φ + a -> DecodeAdvice(φ, a) -> h -> BuildProgram(φ, h) -> P

Interchange with lane C-codec is the plain JSON of `Hints.to_json()` (schema id `HINTS_SCHEMA`); C-codec compresses
it to the advice bitstring `a`.  **Every field here is charged** (bounded-advice accounting): a hint exists only for a
fact that (1) alters Program *structure* and (2) the predictor cannot derive from φ.  Ordinary computed values
(generated token ids, logits, tensor contents, MoE routing) are never hints — they stay constrained by the Program.

Versioning: `HINTS_SCHEMA` changes whenever a field is added/removed/re-typed; every change is announced in the lane
STATUS file.  Unknown top-level keys in a JSON document are rejected by `Hints.from_json` (no silent extensions).

v1 vocabulary (dense decoder-only families under the eager vLLM profile):

    served[r]        number of served (returned) tokens of request r when it differs from the predicted `max_tokens`
                     (EOS / stop-string / abort before max_tokens).  Predicted default: `max_tokens`.  Cost ~ log2(max_tokens).
    steps[s]         batch composition override for engine step s when the predicted scheduler simulation is wrong:
                     the ordered token rows (request index, first position, count) and, when they differ from the
                     default "last row of each request in the step", the sampled (logits) rows.  Cost ~ per deviating step.
    discarded[...]   (request, step) samples the engine computed at a partial-prefill chunk end and dropped, when the
                     predictor's chunking rule did not already place them.
    extensions       named vocabulary extensions the builder had to accept (each is a result to report, and a charge).

The predictor id names the deterministic G these hints correct; hints for another predictor are meaningless.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

HINTS_SCHEMA = "verity-aot/hints/v1"


@dataclass(frozen=True)
class ServedLength:
    """Request `request` (index into φ's LOGICAL requests = φ.workload.requests with n > 1 forks expanded into their children, in
    order — `Phi.logical_requests()`; identical to the header list when no request has n > 1) has `served` sampled rows in the
    Program's return (its output length).

    This is the Program's count, not the API's: under vLLM's async scheduling (output lag 1 step) an EOS stop after k generated
    tokens still schedules one more decode row whose sample the engine drops — that row IS in the accepted Program, so `served`
    = k + 1 (MEASURED eos4: 8 generated tokens -> served 9, digest-identical).  A length stop needs no extra step (the scheduler
    knows max_tokens): served = max_tokens = the predicted default, 0 bits.
    """

    request: int
    served: int


@dataclass(frozen=True)
class RowRun:
    """`count` consecutive token rows of `request` starting at absolute position `first`."""

    request: int
    first: int
    count: int


@dataclass(frozen=True)
class StepRows:
    """Observed batch composition of engine step `step` (only present when it differs from the prediction).

    `rows` is the ordered token-row layout of the step (after vLLM's input preparation / reorder).
    `logits_rows` lists (request, position) pairs whose logits are sampled, in sampler row order, or None when they
    are the default (the last token row of every request present, in row order)."""

    step: int
    rows: tuple[RowRun, ...]
    logits_rows: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class DiscardedSample:
    """The sample of `request` at `step` (position `position`) was computed and dropped by the engine (partial prefill)."""

    request: int
    step: int
    position: int


@dataclass
class Hints:
    schema: str = HINTS_SCHEMA
    predictor: str = ""                                   # id of the predictor these hints correct (predict.PREDICTOR_ID)
    served: list[ServedLength] = field(default_factory=list)
    steps: list[StepRows] = field(default_factory=list)
    discarded: list[DiscardedSample] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    # -- interchange ----------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        d = asdict(self)
        d["steps"] = [{"step": s.step, "rows": [asdict(r) for r in s.rows],
                       "logits_rows": None if s.logits_rows is None else [list(p) for p in s.logits_rows]} for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Hints":
        if d.get("schema") != HINTS_SCHEMA:
            raise ValueError(f"hints schema {d.get('schema')!r} != {HINTS_SCHEMA!r}")
        extra = set(d) - {"schema", "predictor", "served", "steps", "discarded", "extensions"}
        if extra:
            raise ValueError(f"hints document has unknown keys {sorted(extra)} (no silent extensions)")
        h = cls(predictor=str(d.get("predictor", "")))
        h.served = [ServedLength(int(x["request"]), int(x["served"])) for x in d.get("served", [])]
        for s in d.get("steps", []):
            rows = tuple(RowRun(int(r["request"]), int(r["first"]), int(r["count"])) for r in s["rows"])
            lr = s.get("logits_rows")
            h.steps.append(StepRows(int(s["step"]), rows, None if lr is None else tuple((int(a), int(b)) for a, b in lr)))
        h.discarded = [DiscardedSample(int(x["request"]), int(x["step"]), int(x["position"])) for x in d.get("discarded", [])]
        h.extensions = dict(d.get("extensions", {}))
        return h

    def canonical(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()

    @property
    def empty(self) -> bool:
        return not (self.served or self.steps or self.discarded or self.extensions)

    # -- naive accounting (C-codec owns the real codecs; this is the builder-side upper bound) ----------------------
    def naive_bits(self, *, max_tokens: int = 1, n_requests: int = 1, max_position: int = 1) -> int:
        """A trivial fixed-width upper bound on |a| for these hints (labels: calculated).  Zero when empty."""
        b = 0
        lr = max(1, math.ceil(math.log2(max(2, n_requests))))
        lt = max(1, math.ceil(math.log2(max(2, max_tokens + 1))))
        lp = max(1, math.ceil(math.log2(max(2, max_position + 1))))
        b += len(self.served) * (lr + lt)
        for s in self.steps:
            b += 16 + len(s.rows) * (lr + 2 * lp)
            if s.logits_rows is not None:
                b += len(s.logits_rows) * (lr + lp)
        b += len(self.discarded) * (lr + 16 + lp)
        b += 8 * len(self.canonical()) if self.extensions else 0
        return b
