"""The prover's phase accounting, torch-free: how the per-sub-batch ``timings`` of ``protocol.prove`` become the contract's
``t.*`` buckets and ``t.total`` of a bench result (``relchain.bench_vu_rel``, ``vu.bench_vu``).

Three steps, each keeping the buckets disjoint and their sum at most the prover time:

* :func:`rescale` (``pipeline.prove_many``): with ``--pipeline N`` the event-measured laps of the sub-batches in flight
  overlap, so each sub-batch's leaf laps are scaled to its share of the pass wall (the measured values stay under
  ``measured_<lap>``); ``total + hints_host`` is then the share, and the bucket laps are a subset of the leaves.
* a rep's timings are the sum over its sub-batches (the runners' ``agg``), so per rep the buckets sum to at most the
  rep's prover time ``total + hints_host`` (exactly, for a bare pipelined rep).
* :func:`median_rep` picks ONE rep -- the one with the median prover time -- whose buckets and total are reported.  A
  per-key median across reps mixes reps: with one or two slow reps (a timed pass that meets a graph capture) the median of
  each lap is the largest lap of the fast reps, and the buckets sum to more than the median total.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["DERIVED", "buckets", "median_rep", "prover_seconds", "rescale"]

#: buckets re-derived from leaves after rescaling
DERIVED = ("tests",)


def rescale(results, wall: float, key: str) -> None:
    """Pipelined attribution: each result's leaf phases (event-measured, overlapped) are scaled by a common factor so
    that they sum to this sub-batch's share of the wall (wall / n); ``key`` (``total``) becomes share - hints_host so
    that ``total + hints_host`` (the runners' prover time) equals the share.  The event-measured values are kept as
    ``measured_<phase>`` (their sum over the sub-batches exceeds the wall by the overlap)."""
    share = wall / max(1, len(results))
    for r in results:
        tm = getattr(r, "timings", None)
        if not isinstance(tm, dict) or key not in tm:
            continue
        leaf = {k: v for k, v in tm.items()
                if k != key and k not in DERIVED and not k.startswith("measured_") and not k.startswith("host_")
                and not k.startswith("net_wait")}
        s = sum(leaf.values())
        scale = share / s if s > 0 else 0.0
        for k, v in leaf.items():
            tm["measured_" + k] = v
            tm[k] = v * scale
        tm["measured_" + key] = tm[key]
        tm[key] = share - tm.get("hints_host", 0.0)
        if all(k in tm for k in ("tests_w", "tests_quad", "tests_chain")):
            tm["tests"] = tm["tests_w"] + tm["tests_quad"] + tm["tests_chain"]


def prover_seconds(tm: Mapping[str, float]) -> float:
    """``t.total`` of a timings dict: the prover's clock plus the hint generation it reports separately."""
    return tm["total"] + tm.get("hints_host", 0.0)


def buckets(tm: Mapping[str, float], serialization_extra: Sequence[str] = ()) -> dict[str, float]:
    """The six contract buckets of one timings dict (a rep's sum over its sub-batches).  ``serialization_extra``: further
    laps the runner files under serialization (``vu.bench_vu``: ``auth_openings``)."""
    return {
        "t.witness": tm["witness"] + tm["hints_host"],
        "t.encoding_commitment": tm["encode"] + tm["transfer"] + tm["merkle"],
        "t.arithmetic": tm["tests"],
        "t.lookup": 0.0,
        "t.zk_additional": tm.get("zk_masks", 0.0),
        "t.serialization": tm["openings"] + tm.get("statement", 0.0) + sum(tm.get(k, 0.0) for k in serialization_extra),
    }


def median_rep(runs: Sequence[Sequence]) -> int:
    """Index into ``runs`` (``(timings, ...)`` per recorded rep) of the rep whose prover time is the median -- the upper
    middle one for an even count, as ``fp4/chain.py`` reports -- so that ``t.total`` and every bucket come from one rep."""
    order = sorted(range(len(runs)), key=lambda i: prover_seconds(runs[i][0]))
    return order[len(order) // 2]
