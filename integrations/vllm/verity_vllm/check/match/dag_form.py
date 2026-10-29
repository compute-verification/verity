"""The emission-order-insensitive form of a Program: the live cone of its last sampling event, and every instance hashed with its
producers' hashes in place of their positions (a Merkle DAG), compared as a multiset."""

from __future__ import annotations

import collections
import json

from verity_vllm.check.match import program_compare as C
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  #

def _producer(P: C.Prog, t) -> int | None:
    if t[0] != "N":
        return None
    return P.steps[t[1]][t[2]]


def _producers(P: C.Prog, t) -> list[int]:
    """Every producer instance of a canonical N source, record form or progression."""
    from verity_vllm.program.compact import n_producers
    return n_producers(P.steps, t)


def live_cone(P: C.Prog) -> set[int]:
    """Instances in the backward cone of the LAST TokenSelect: everything the served token sequence depends on (each TokenSelect
    feeds the next step's embedding, so the chain reaches every served TokenSelect). What falls outside is DEAD for serving — with a
    chunked prefill vLLM v1 runs the sampler on every scheduled request, including a partially-prefilled one, and DISCARDS that
    token: the fold records that final-norm row / lm_head row / TokenSelect, the request-wrapper Program never computes it.
    Producers precede consumers, so one reverse pass closes the cone."""
    order = [i for idxs in P.steps for i in idxs]
    ts = [i for i in order if _is_sampling_event(P.inst[i]["spec"])]   # any sampling-event family
    if not ts:
        return set(order)
    live = {ts[-1]}
    for i in reversed(order):
        if i in live:
            for arg in P.canon(i, False):
                for t in arg:
                    if t[0] == "N":
                        live.update(_producers(P, t))
    return live


def dag_hashes(P: C.Prog, only: set[int] | None = None) -> dict[int, str]:
    """Emission-order-insensitive canonical form: every instance hashed by (spec, form, operand sources, out type) where a node
    source carries its PRODUCER'S HASH (Merkle) instead of its (step, index) position. Two Programs with equal multisets hold the
    same computations over the same weights/prompt reads with the same dependency trees; only the order in which the fold emitted
    them (and the engine-step segmentation) is abstracted. Used for a request whose prefill the engine CHUNKED over several steps:
    the eager fold then emits chunk-major (all ops of rows 0..c-1, then all ops of rows c..LP-1) while the request-wrapper Program
    is op-major over LP rows — the same DAG, a different sequence. `only` restricts the multiset to those instances (live cone);
    producers precede consumers in node order."""
    import hashlib
    hashes: dict[int, str] = {}
    compact = getattr(P, "compact", False)
    if compact:
        from verity_vllm.program.compact import expand_mixed
    for s, idxs in enumerate(P.steps):
        for i in idxs:
            if only is not None and i not in only:
                continue
            r = P.inst[i]
            parts = []
            for arg in P.canon(i, False):
                if compact:   # the record's per-run payload: a gather's producers carry one hash EACH
                    arg = expand_mixed(arg)
                row = []
                for t in arg:
                    if t[0] == "N":
                        ii = P.steps[t[1]][t[2]]
                        row.append(["N", hashes.get(ii, f"forward-ref-or-dead:{ii}"), *t[3:]])
                    else:
                        row.append(list(t))
                parts.append(row)
            hashes[i] = hashlib.sha256(json.dumps([r["spec"], r["form"], parts, r["out"]], separators=(",", ":"), default=str).encode()).hexdigest()
    return hashes


def dag_multiset(P: C.Prog, only: set[int] | None = None) -> collections.Counter:
    return collections.Counter(dag_hashes(P, only).values())
