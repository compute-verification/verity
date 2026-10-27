"""The cuBLAS route pin of a Program against the process that served it, one rule for every engine: each TP rank's
(`pipeline.tp.c2.replay_gates`) and a TP1 run's, which `stamp` puts on the Commit record as `cublas_route_pin` for its verdict
(`check.commit_pass.commit_verdict` and `check.c2_rules.rule_route_pin` read that block, as they read `weights_pin`).

A Program that binds a Definition only the pin binds (`bf16_gemm_check.ROUTE_DEFINITIONS`, in the replay's population or its
unreplayable families) computes cuBLAS's routes as the pin's scan saw them, which hold only under the scan's conditions
(`ROUTE_CONDITIONS`); the rank's facts of record (`engine_of_record.cublas`, measured in its process) must be those, else FAIL naming each
departure, an unrecorded fact one too."""
from __future__ import annotations

from verity.ml.definitions.registry import bf16_gemm_check as BC


def route_pin_gate(sr: dict | None, *, rank: int) -> tuple[dict | None, str | None]:
    """Returns (record block | None when the replay's Program binds none of the pin's Definitions, FAIL text | None); the block's `why`
    is the FAIL text."""
    sr = sr if isinstance(sr, dict) else {}
    fams: dict[str, int] = {}
    for f, n in (((sr.get("population") or {}).get("by_family")) or {}).items():
        fams[f] = fams.get(f, 0) + int(n or 0)
    for u in sr.get("unreplayable") or []:
        f = str(u.get("family")).split(" (")[0]
        fams[f] = fams.get(f, 0) + int(u.get("n") or 0)
    bound = BC.route_bound(fams)
    if not bound:
        return None, None
    departures = BC.route_departures(sr.get("engine_of_record"))
    named = ", ".join(f"{f} x{n}" for f, n in bound.items())
    why = (f"cuBLAS route pin (rank {rank}): {named} bound at the routes of {BC.CUBLAS_ROUTES['run']}, served otherwise: {'; '.join(departures)}"
           if departures else None)
    block = {"rule": "a rank whose Program binds bf16_gemm_check.ROUTE_DEFINITIONS served under bf16_gemm_check.ROUTE_CONDITIONS (the route "
                     "pin's scan), every fact measured in the rank's process; a departure or an unrecorded fact = the rank FAILs by name",
             "rank": int(rank), "families": bound, "pin": BC.CUBLAS_ROUTES["run"], "departures": departures, "ok": not departures, "why": why}
    return block, why


def stamp(commit: dict) -> dict | None:
    """A TP1 run's gate (rank 0) over the sampled replay on its Commit record, stamped as `commit["cublas_route_pin"]` before its verdict:
    once the replay's `engine_of_record` is the serving process's (a deferred replay's, the Commit engine's: `pipeline.cpu_replay`)."""
    commit["cublas_route_pin"] = block = route_pin_gate(commit.get("sampled_replay"), rank=0)[0]
    return block
