"""A result's interaction record and the measurement protocol it must meet: what ``cell check`` and Flock's ``register``
read of a ``bench-result/v1`` document.

The interaction model is t = compute + rounds x RTT + bytes / bandwidth, with the
reference network ``census.REFERENCE_NETWORK`` (``benchmarks/census/networks.json``: same datacenter)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from verity_numerical.bench import census, contract

#: Commitment seconds per batch: the warm median of committing the batch's values afresh (with ``commit.cold_seconds``,
#: ``commit.reps`` and ``e2e.seconds`` = this + ``t.total``).
COMMIT = "commit.seconds"
MIN_RUNS = 5
REF_RTT_S, REF_BANDWIDTH_BPS = census.REFERENCE_NETWORK.rtt_s, census.REFERENCE_NETWORK.bandwidth_bps
#: A measured interactive run must be within this fraction of the formula at the reference profile.
INTERACTION_TOLERANCE = 0.10
#: A result's statement bytes (sent beside the proof): the relation's statements over the run's sub-batch proofs.
STATEMENT_BYTES = ("relation.statement_bytes_total", "relation.hash.statement_bytes_total")


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def measurements(result: Mapping[str, Any]) -> dict[str, Any]:
    """A result's measurements by name."""
    out: dict[str, Any] = {}
    for m in result.get("measurements") or []:
        if isinstance(m, Mapping) and isinstance(m.get("name"), str):
            out[m["name"]] = m.get("value")
    return out


def interaction(meta: Mapping[str, Any], interactive: bool = False) -> dict[str, Any]:
    """The interaction record of a result.  ``live``: it ran against a live verifier (live-verifier evidence, or a measured RTT,
    network wait or ``t.total_live``).

    A **live** result's record is measured: ``rounds`` (round trips on the critical path: ``rounds.sequential_depth``, else
    ``rounds.protocol``), ``bytes_up`` (``net.bytes_out``, else the proof bytes), ``bytes_down`` (``net.bytes_in``, the coins),
    ``rtt_ms`` (``net.rtt_ms``) with ``rtt_method`` (the ``interaction`` block's, or the live-verifier evidence's), and the wall
    split: ``prover_s`` (``t.total``), ``verifier_s`` (verifier compute the prover waited on: ``verify.critical_s``, else 0 -- the
    measured wait includes it) and ``wait_s`` (``net.wait_seconds``, else ``t.total_live - t.total``).  A record whose ``interaction``
    block says ``t_total_includes_wait`` (Flock's live coins) has its prover compute in ``live.prover_compute_seconds`` (else
    ``t.total`` minus the wait).  The prover's own per-round live-coin handling, measured on loopback
    (``live.loopback_round_seconds``), is prover compute: rounds x that (at most the measured wait) moves from the wait into
    ``prover_s``, and only the excess over loopback counts as network wait (``loopback_s`` records the amount moved).

    Any other result is a proof verified from its files and its record is derived (``derived``): bytes up = the proof bytes
    (``proof_bytes``, else ``transcript.bytes``) plus the statement bytes (:data:`STATEMENT_BYTES`), no measured RTT.  Its
    rounds (``rounds_source``): a Fiat–Shamir proof has none (0); an ``interactive`` protocol's file-replayed transcript pays the
    protocol's sequential depth, ``"derived: file-verified"``: the per-proof depth (``rounds.sequential_depth``, else
    ``rounds.protocol``) x the proofs that run one after another -- ``N_subbatches`` sub-batch proofs,
    ``software.backend.pipeline`` of them in flight at once, so ceil(N_subbatches / pipeline) waves."""

    m = measurements(meta)
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    block = meta.get("interaction") if isinstance(meta.get("interaction"), Mapping) else fp.get("interaction")
    block = block if isinstance(block, Mapping) else {}
    lv = _get(meta, "validation", "evidence", "live_verifier")
    lv = lv if isinstance(lv, Mapping) else {}
    num = lambda k: float(m[k]) if _num(m.get(k)) else None  # noqa: E731
    proof = next((v for v in (num("proof_bytes"), num("transcript.bytes")) if v is not None), None)
    live = bool(lv) or any(num(k) is not None for k in ("net.rtt_ms", "net.wait_seconds", contract.T_TOTAL_LIVE))
    if not live:
        stmt = next((v for v in (num(k) for k in STATEMENT_BYTES) if v is not None), 0.0)
        rounds, source = 0.0, "fiat-shamir"
        if interactive:
            depth = next((v for v in (num("rounds.sequential_depth"), num("rounds.protocol")) if v is not None), 0.0)
            proofs, flight = fp.get("N_subbatches"), _get(fp, "software", "backend", "pipeline")
            proofs = proofs if _num(proofs) and proofs >= 1 else 1
            waves = math.ceil(proofs / flight) if _num(flight) and flight >= 1 else proofs
            rounds, source = depth * waves, "derived: file-verified"
        return {"live": False, "derived": True, "rounds": rounds, "rounds_source": source, "bytes_up": (proof or 0.0) + stmt, "bytes_down": 0.0,
                "proof_bytes": proof, "statement_bytes": stmt, "rtt_ms": None, "rtt_method": None,
                "prover_s": num(contract.T_TOTAL), "verifier_s": 0.0, "wait_s": None, "loopback_s": 0.0,
                "loopback_round_s": None, "loopback_method": None}
    rounds = num("rounds.sequential_depth")
    rounds = num("rounds.protocol") if rounds is None else rounds
    up = next((v for v in (num("net.bytes_out"), proof) if v is not None), None)
    wait = num("net.wait_seconds")
    if wait is None and num(contract.T_TOTAL_LIVE) is not None and num(contract.T_TOTAL) is not None:
        wait = max(0.0, num(contract.T_TOTAL_LIVE) - num(contract.T_TOTAL))
    compute = num(contract.T_TOTAL)
    if block.get("t_total_includes_wait") is True and compute is not None and wait is not None:
        compute = next((v for v in (num("live.prover_compute_seconds"),) if v is not None), compute - wait)
    loop, moved = num("live.loopback_round_seconds"), 0.0
    if loop is not None and rounds and wait is not None and compute is not None:
        moved = min(wait, rounds * loop)
        compute, wait = compute + moved, wait - moved
    return {"live": True, "derived": False, "rounds": rounds, "rounds_source": "measured", "bytes_up": up or 0.0, "bytes_down": num("net.bytes_in") or 0.0,
            "proof_bytes": proof, "statement_bytes": None, "rtt_ms": num("net.rtt_ms"),
            "rtt_method": block.get("rtt_method") or lv.get("rtt_method"), "bandwidth_bps": num("net.bandwidth_bps"), "prover_s": compute,
            "verifier_s": num("verify.critical_s") or 0.0, "wait_s": wait, "loopback_s": moved,
            "loopback_round_s": loop if moved else None, "loopback_method": block.get("loopback_method") if moved else None}


def interaction_time(ix: Mapping[str, Any], rtt_s: float = REF_RTT_S, bandwidth_bps: float = REF_BANDWIDTH_BPS) -> float:
    """t = compute + rounds x RTT + bytes / bandwidth, compute = prover compute + the verifier compute the prover waits on."""
    return ((ix["prover_s"] or 0.0) + ix["verifier_s"] + (ix["rounds"] or 0.0) * rtt_s
            + (ix["bytes_up"] + ix["bytes_down"]) * 8 / bandwidth_bps)


def interaction_deviation(ix: Mapping[str, Any]) -> float | None:
    """(measured - model) / model at the run's own RTT and bandwidth (the reference's when none was measured), for a live result
    with its rounds, RTT and wait; else ``None``."""
    if not ix["live"] or ix["rounds"] is None or ix["rtt_ms"] is None or ix["wait_s"] is None:
        return None
    model = interaction_time(ix, ix["rtt_ms"] / 1e3, ix.get("bandwidth_bps") or REF_BANDWIDTH_BPS)
    return ((ix["prover_s"] or 0.0) + ix["wait_s"] - model) / model if model > 0 else None


def interaction_problem(ix: Mapping[str, Any], one_sided: bool = False) -> str | None:
    """Why a result does not meet the interaction standard, or ``None``.  A non-interactive (file-verified) result's record is derived
    and always meets it.  A live-verifier result records its rounds, its measured RTT with how it was measured and its network wait,
    and its measured wall time (prover compute + network wait) is within :data:`INTERACTION_TOLERANCE` of the formula evaluated at
    the run's own measured RTT (``net.rtt_ms``) and bandwidth (``net.bandwidth_bps``, else the reference's): the check validates
    the model against the run itself."""
    if not ix["live"]:
        return None
    if ix["rounds"] is None:
        return "interaction: a live-verifier result without its rounds (rounds.sequential_depth)"
    missing = [k for k, ok in (("measured RTT (net.rtt_ms)", ix["rtt_ms"] is not None), ("how the RTT was measured", bool(ix["rtt_method"])),
                               ("network wait (net.wait_seconds or t.total_live)", ix["wait_s"] is not None)) if not ok]
    if missing:
        return f"interaction: {ix['rounds']:.0f} rounds but no {', no '.join(missing)}"
    bw = ix.get("bandwidth_bps") or REF_BANDWIDTH_BPS
    model = interaction_time(ix, ix["rtt_ms"] / 1e3, bw)
    measured = (ix["prover_s"] or 0.0) + ix["wait_s"]
    dev = interaction_deviation(ix)
    if dev is not None and (dev if one_sided else abs(dev)) > INTERACTION_TOLERANCE:
        return (f"interaction: measured {measured:.3g} s is {100 * dev:+.0f}% from the model's {model:.3g} s at the run's "
                f"own network ({ix['rtt_ms']:.3g} ms RTT, {bw / 1e9:.3g} Gb/s{'' if ix.get('bandwidth_bps') else ' (reference; not measured)'}; "
                f"tolerance ±{100 * INTERACTION_TOLERANCE:.0f}%)")
    return None


def rtt_method_enum(text: str | None) -> str | None:
    """The published RTT method of a recorded ``rtt_method`` text, ``None`` when none is recorded."""
    if not text:
        return None
    t = text.lower()
    if "ping" in t and "connection" in t:
        return "open-connection-ping"
    if "tcp connect" in t:
        return "tcp-connect"
    return "unclassified"


def protocol_problem(meta: Mapping[str, Any]) -> str | None:
    """Why the result was not measured under the protocol, or ``None``: warm, >= :data:`MIN_RUNS` timed runs, uncontended, and
    a plateau point of a recorded sweep."""
    fp = meta.get("workload_fingerprint") if isinstance(meta.get("workload_fingerprint"), Mapping) else {}
    proto = meta.get("protocol") if isinstance(meta.get("protocol"), Mapping) else fp.get("protocol")
    sweep = meta.get("sweep") if isinstance(meta.get("sweep"), Mapping) else fp.get("sweep")
    if not isinstance(proto, Mapping):
        return "protocol not recorded (warm, timed runs, contention verdict): a bench-result/v1 result"
    runs = proto.get("runs")
    if proto.get("warm") is not True or not (_num(runs) and runs >= MIN_RUNS) or proto.get("contended") is not False:
        return f"protocol: warm {proto.get('warm')!r}, runs {runs!r}, contended {proto.get('contended')!r} (need warm, >= {MIN_RUNS}, uncontended)"
    if not isinstance(sweep, Mapping) or sweep.get("plateau") is not True:
        return "not a plateau point of a recorded sweep"
    return None
