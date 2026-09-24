"""The committer interface the Δ_commit harness drives, and the coverage declaration every committer emits.

A committer is installed on a built engine for ONE instrumented run and uninstalled afterwards (the control arm of the
same pair runs on the pristine engine: no hooks, no wrapped runner methods, nothing).  Lifecycle:

    c = SomeCommitter(engine, **opts)
    c.register_weights()      # once per process; the registered-input class (timed, excluded from steady state)
    c.install()               # hooks / wraps; the serving loop then runs the workload untouched
    ... engine.step() ...     # the committer sees step boundaries itself (runner.execute_model / runner.sample)
    res = c.finalize()        # blocks until every root is available -> RunCommitment (roots, spans, root-ready delay, backlog)
    ops = c.open(sample)      # openings for sampled positions against the ORIGINAL per-step roots (after the run)
    c.uninstall()             # engine pristine again (asserted by the harness)

`coverage_declaration()` returns the document `coverage_check.py` consumes: which manifest classes this committer binds,
how (bound-as-leaf | registered-input | derived-from-bound{rule}), and per class the counts/bytes it actually bound in
the run — so a fast subset cannot be reported as the full interface (DANIEL-BRIEF A2/A4; BRIEF §1 non-goals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StepCommitment:
    step: int
    root: bytes                 # context-bound step root (prechallenge)
    merkle_root: bytes          # bare Merkle root over the step's leaves
    n_leaves: int
    n_bytes: int                # value bytes bound in this step
    n_tensors: int
    layout_digest: bytes        # digest of the step's static layout (tensor names, dtypes, shapes, chunking)
    t_step_end: float           # perf_counter when the serving step ended (sample() returned)
    t_root_ready: float         # perf_counter when `root` was available
    backlog_at_step_end: int    # steps queued but not yet hashed when this step ended
    hashing_s: float = 0.0
    root_finalisation_s: float = 0.0


@dataclass
class RunCommitment:
    run_root: bytes
    steps: list[StepCommitment]
    spans: dict
    root_ready_delay_s: float           # last step end -> run root available (measured)
    t_finalize_s: float                 # wall of finalize() itself
    persistent_bytes: int               # what the prover retains after the run (roots + levels), measured
    temporary_bytes_peak: int           # arena / staging peak, measured
    weight_registration: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Opening:
    step: int
    index: int                  # leaf rank in the step tree
    value: bytes
    path: tuple                 # siblings (None for lifts) — production `commit.path_shape` shape
    identity: dict              # verifier-derivable (tensor ordinal, name, dtype, shape, byte offset)


@dataclass
class RangeOpening:
    step: int
    lo: int                     # leaves [lo, hi) of the step tree
    hi: int
    value: bytes                # the leaves' bytes, concatenated in leaf order
    path: tuple                 # per level (left sibling | None, right sibling | None) — `merkle.range_path_shape` shape


class Committer(Protocol):
    name: str

    def register_weights(self) -> dict: ...
    def install(self) -> None: ...
    def uninstall(self) -> None: ...
    def finalize(self) -> RunCommitment: ...
    def open(self, sample: list[tuple[int, int]]) -> list[Opening]: ...
    def verify(self, run: RunCommitment, opening: Opening) -> bool: ...
    def coverage_declaration(self, run: RunCommitment) -> dict: ...
    def reset(self) -> None: ...


def challenge_positions(run_root: bytes, leaves_per_step: list[int], k: int, salt: bytes = b"") -> list[tuple[int, int]]:
    """The sampling challenge, derived from the PRECHALLENGE run root (Fiat–Shamir style): the prover cannot learn the positions
    before the root is fixed, and a root recomputed after the fact yields a different challenge.  Deterministic, so the harness and a
    verifier derive the same (step, rank) list from (root, layout sizes, k)."""
    import hashlib as _h

    if len(run_root) != 32:
        raise ValueError("run root must be 32 bytes")
    total = sum(leaves_per_step)
    if total == 0 or k <= 0:
        return []
    cum = []
    acc = 0
    for n in leaves_per_step:
        acc += n
        cum.append(acc)
    out = []
    ctr = 0
    while len(out) < k:
        d = _h.sha256(b"verity/cmt-integ/challenge/v0" + run_root + salt + ctr.to_bytes(8, "big")).digest()
        ctr += 1
        for i in range(0, 32, 8):
            g = int.from_bytes(d[i:i + 8], "big") % total
            s = next(j for j, c in enumerate(cum) if g < c)
            out.append((s, g - (cum[s - 1] if s else 0)))
            if len(out) >= k:
                break
    return out


DECLARATION_SCHEMA = "verity/commit-integ/coverage-declaration/v1"


def declaration(committer: str, path_label: str, classes: list[dict], notes: list[str] | None = None) -> dict:
    """classes: [{"class": <manifest class id>, "how": "bound-as-leaf" | "registered-input" | "derived-from-bound",
                  "rule": <citation for registered/derived; None for leaves>, "count": <values bound>, "bytes": <bytes bound>,
                  "evidence": <free text: hook names / kernels / tests>}]"""
    return {"schema": DECLARATION_SCHEMA, "committer": committer, "path": path_label, "classes": classes, "notes": notes or []}
