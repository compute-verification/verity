"""The Program's checks (`program.predict.common.with_checks`): bits it computes from its inputs that the verifier requires to be
`accept`, a DSA row's DsaTopKLongRow_v1 past INDEX_TOPK among them.  The instances header names them (`checks`: what each checks,
its accept value, and the row and output leaf that computes it, `pipeline.build_instances.check_rows`).

The checks are the Program's output, not a sample: `checks_of` evaluates every check of every executed step of every request from
its committed inputs, the run's values included, serially in the replay's process.  One that is not `accept` refuses the replay by
name (`fold`); one it can't evaluate leaves the replay partial, never a pass."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np

from verity_vllm.check.replay.compare import _interior_slice
from verity_vllm.check.replay.evaluate import _evaluate_row, _Unresolved
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit.opened import OpeningNotVerified, opened_check

RULE = ("every check of every request's Program (instances header `checks`), evaluated from its committed inputs at every executed "
        "step, is its `accept` value")
#: named records kept per class; the counts are uncapped
KEPT = 12


def check_rows(pi: ProgramIndex) -> list[tuple[str, int, int, int]]:
    """(what it checks, row, lo, hi) of each check of `pi`'s Program, in output order; [] when it has none."""
    if (c := pi.inst.get("checks")) is None:
        return []
    if "of" not in c or "rows" not in c or len(c["of"]) != len(c["rows"]):
        raise ValueError(f"instances header `checks` ({sorted(c)}) doesn't pair each check it names (`of`) with the row computing it "
                         f"(`rows`)")
    return [(str(w), int(i), int(lo), int(hi)) for w, (i, lo, hi) in zip(c["of"], c["rows"])]


def _check(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], rid: str, arrive: int,
           what: str, i: int, lo: int, hi: int, accept: int, rec: dict[str, Any]) -> dict[str, Any]:
    r = pi.by_i[i]
    step = pi.step_of(i)
    rec.update({"check": what, "request_id": rid, "step": step, "engine_step": arrive + step, "i": i, "spec": r["spec"], "inputs": []})
    try:
        got = _evaluate_row(pi, store, weights, r, rid, arrive, rec["inputs"], depth=0, interior={})
    except _Unresolved as u:
        return rec | {"result": None, "why": str(u)}
    except OpeningNotVerified:
        raise
    except Exception as e:  # noqa: BLE001
        return rec | {"result": None, "why": f"evaluator error: {e!r}"[:300]}
    bit = _interior_slice(got, lo, hi)
    if bit is None or bit.size != 1:
        return rec | {"result": None, "why": f"leaf [{lo}, {hi}) is not one leaf of row {i}'s output"}
    rec["value"] = int(bit[0])
    if rec["value"] != accept:
        return rec | {"result": False, "why": f"check is {rec['value']}, not {accept}: {what} ({rid}, step {step}, row {i})"}
    return rec | {"result": True}


def checks_of(indices: dict[str, ProgramIndex], store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None],
              arrivals: dict[str, int] | None = None, executed_prefix: dict[str, dict] | None = None) -> dict[str, Any] | None:
    """Every check of every request (`check_rows`), evaluated from its committed inputs -> None when no request's Program has
    checks, else the record: `result` False when one is not its `accept` value (named under `refused`), None when one couldn't be
    evaluated (named under `not_evaluated`), True when every one is accepted.  A check at a request-relative step at or past its
    request's executed prefix (`executed_prefix`'s `n_exec`) was never served: counted under `not_executed`, not evaluated."""
    arrivals, executed_prefix = arrivals or {}, executed_prefix or {}
    n, accepted, not_exec, refused, not_ev = 0, 0, 0, [], []
    for rid in sorted(indices):
        pi = indices[rid]
        rows = check_rows(pi)
        if not rows:
            continue
        accept = int(pi.inst["checks"]["accept"])
        ep = executed_prefix.get(str(rid))
        for what, i, lo, hi in rows:
            n += 1
            if ep is not None and not ep.get("faults") and pi.step_of(i) >= int(ep["n_exec"]):
                not_exec += 1
                continue
            rec = opened_check(getattr(store, "committed", None),
                               partial(_check, pi, store, weights, rid, int(arrivals.get(rid, 0)), what, i, lo, hi, accept))
            rec.pop("_opened", None)
            if rec["result"] is True:
                accepted += 1
            elif rec["result"] is False:
                refused.append(rec)
            else:
                not_ev.append({k: v for k, v in rec.items() if k != "inputs"})
    if not n:
        return None
    result = False if refused else (None if not_ev else True)
    why = None if result else (refused or not_ev)[0]["why"] + (f" (+{len(refused or not_ev) - 1} more)" if len(refused or not_ev) > 1 else "")
    return {"rule": RULE, "result": result, "why": why, "n": n, "accepted": accepted, "not_executed": not_exec,
            "refused_n": len(refused), "not_evaluated_n": len(not_ev), "refused": refused[:KEPT], "not_evaluated": not_ev[:KEPT]}


def fold(result: bool | None, grade: str, why: str | None, checks: dict[str, Any] | None) -> tuple[bool | None, str, str | None]:
    """The sampled replay's (result, grade, why) with its checks (`checks_of`) beside it: a refused check fails a replay that had not
    failed already, and one not evaluated leaves a passing replay partial."""
    if checks is None or result is False or checks["result"] is True:
        return result, grade, why
    if checks["result"] is False:
        return False, "mismatch", f"a check of the Program is refused: {checks['why']}"
    if result is True:
        return None, "partial", f"a check of the Program could not be evaluated: {checks['why']}"
    return result, grade, why
