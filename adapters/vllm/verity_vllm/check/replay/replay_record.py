"""What `driver.sampled_replay` records of its picks: the fold of their results in picked order, the grade that fold gives, and the
sections of the replay record that are counts and names over them.  Nothing here evaluates a VU or reads a committed word."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.records import replay_codes as RC
from verity_vllm.records.replay_codes import (
    NOT_YET,
    WHY_CLASSES_BY_NAME,
    why_class,
)


@dataclass
class Fold:
    """The picked VUs' results, counted (`fold`)."""
    evaluated: int = 0
    equal: int = 0
    mismatch: list[dict[str, Any]] = field(default_factory=list)
    not_evaluated: list[dict[str, Any]] = field(default_factory=list)
    unevaluated_by_stratum: dict[str, dict[str, Any]] = field(default_factory=dict)
    strata_mismatch: Counter[str] = field(default_factory=Counter)
    family_seconds: dict[str, float] = field(default_factory=dict)
    family_n: Counter[str] = field(default_factory=Counter)


def fold(results: list[tuple[dict, dict, float]], strata: dict[str, dict]) -> Fold:
    """`results` (`(vu, record, seconds)` in picked order) counted; each stratum of `strata` is stamped with its `evaluated` count and,
    where it has one, its `mismatch` count."""
    f = Fold()
    strata_evaluated: Counter[str] = Counter()
    for vu, rec, dt in results:
        f.family_seconds[vu["family"]] = f.family_seconds.get(vu["family"], 0.0) + dt
        f.family_n[vu["family"]] += 1
        if rec.get("result") is None:
            ne = {k: rec.get(k) for k in ("spec", "request_id", "step", "op_path", "row", "why")}
            ne["why_class"] = why_class(ne.get("why"))
            f.not_evaluated.append(ne)
            u = f.unevaluated_by_stratum.setdefault(vu["stratum"], {"stratum": vu["stratum"], "spec": vu["spec"], "request_id": vu["request_id"], "phase": vu["phase"],
                                                                    "op_path": vu["op_path"], "n": 0, "why": ne.get("why"), "why_class": ne["why_class"], "classes": []})
            u["n"] += 1
            if ne["why_class"] not in u["classes"]:
                u["classes"].append(ne["why_class"])
            continue
        f.evaluated += 1
        strata_evaluated[vu["stratum"]] += 1
        if rec["result"]:
            f.equal += 1
        else:
            f.mismatch.append(rec)
            f.strata_mismatch[vu["stratum"]] += 1
    for k in strata:
        strata[k]["evaluated"] = int(strata_evaluated.get(k, 0))
        if f.strata_mismatch.get(k):
            strata[k]["mismatch"] = int(f.strata_mismatch[k])                 # the stratum's own count, uncapped
    return f


def grade(f: Fold, weights_check: dict[str, Any] | None, strata_unevaluated: list[str]) -> tuple[bool | None, str, str | None]:
    """(result, grade, why) of the picks alone, before the Programs' checks are folded in: False on weights that are not the committed
    ones or on a mismatch; None where nothing, a whole stratum or a single pick was not evaluated; True when every pick was evaluated
    and equal."""
    mism, not_evald = f.mismatch, f.not_evaluated
    if weights_check is not None and weights_check.get("equal") is False:
        return False, "mismatch", (f"weights at replay != committed weights root "
                                   f"({weights_check.get('why') or 'the registered inputs the relations consume are not the committed ones'})")
    if mism:
        return False, "mismatch", mism[0]["why"] + (f" (+{len(mism) - 1} more)" if len(mism) > 1 else "")
    if f.evaluated == 0:
        return None, "not-evaluated", "no VU could be evaluated: " + (not_evald[0]["why"] if not_evald else "empty population")
    if strata_unevaluated:
        return None, "partial", f"{len(strata_unevaluated)} strata not evaluated: {strata_unevaluated[:3]}"
    if not_evald:
        # picks not evaluated INSIDE strata that still had an evaluated pick used to leave the grade 'complete'
        # with sample.not_evaluated_n > 0 -- a run-time fault (missing committed input, bytes not retained, evaluator error, device) on
        # some picks then rode a PASS unnamed.  'complete' means every picked VU was evaluated: any not-evaluated pick -> partial, the first
        # fault named (fault classes first; a by-name pick inside a mixed stratum is named too and, having no fully unevaluated stratum,
        # cannot be lifted by the named-gap rule -- fail closed).  Real records: 0 complete-with-not-evaluated picks (rev, 38 pairs).
        faults = [ne for ne in not_evald if ne.get("why_class") not in WHY_CLASSES_BY_NAME]
        first = (faults or not_evald)[0]
        by_cls = Counter(ne.get("why_class") or "other" for ne in not_evald)
        return None, "partial", (f"{len(not_evald)} pick(s) not evaluated inside evaluated strata ({', '.join(f'{c} x {n}' for c, n in sorted(by_cls.items()))})"
                                 f"{' -- run-time fault, not a gap on the record' if faults else ''}: {first.get('spec')} {first.get('op_path')} "
                                 f"(req {first.get('request_id')}, step {first.get('step')}): {str(first.get('why'))[:160]}")
    return True, "complete", None


def mismatch_by(f: Fold) -> dict[str, Any]:
    """EVERY mismatch counted by name, uncapped (the record's first-12 list carries the inputs/outputs detail only): a reader of the
    record learns which mechanisms / requests / phases / modules disagree without re-running the replay."""
    mism = f.mismatch
    return {"by_family": dict(Counter(m["spec"].split("{")[0] for m in mism)),
            "by_request": dict(Counter(str(m["request_id"]) for m in mism)),
            "by_phase": dict(Counter(str(m.get("phase")) for m in mism)),
            "by_family_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('phase')}" for m in mism).items())),
            "by_family_request_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m['request_id']}|{m.get('phase')}" for m in mism).items())),
            "by_module": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('op_path')}" for m in mism).items())),
            "strata": {k: int(v) for k, v in sorted(f.strata_mismatch.items())}}


def population_record(all_vus: list[dict], requests: list[str], not_ev: Counter[str], not_ev_fam: Counter[str], codes: dict[str, dict],
                      strata_n: int, qpop: dict[str, Any], not_exec_vus: list[dict]) -> dict[str, Any]:
    """The record's `population`: the replay's enumeration of the Program rows, with the reach `coverage.query_population` reconciled
    against Q(P) (`qpop`) beside it."""
    return {"vus": len(all_vus), "source": "Program instances.json.gz rows (every row; evaluable = families with a registered evaluator)",
            "requests": requests, "by_family": dict(Counter(v["family"] for v in all_vus)), "not_evaluable_by_family": dict(not_ev_fam),
            "not_evaluable": dict(not_ev), "not_evaluable_codes": {k: codes.get(k) or RC.reason_code("other") for k in not_ev}, "strata_n": strata_n,
            "authority": "Q(P) -- the required manifest; `vus` above is the replay's enumeration of the Program rows, reconciled below",
            "query_total": qpop["query_total"], "sampled_strata": qpop["sampled_strata"], "uncovered_by_name": qpop["uncovered_by_name"],
            "not_executed_by_family": qpop["not_executed_by_family"], "not_executed_rule": qpop["not_executed_rule"],
            "not_executed_by_rule": dict(qpop["not_executed_by_rule"], vus=len(not_exec_vus),
                                         vus_by_request=dict(Counter(v["request_id"] for v in not_exec_vus)),
                                         strata_excluded={k: {"population": int(n), "label": "not evaluable: unexecuted"}
                                                          for k, n in sorted(Counter(v["stratum"] for v in not_exec_vus).items())}),
            "uncovered_identities": qpop["uncovered_identities"], "controller": qpop.get("controller"),   # [R4] the lifted controller, by name
            "query_checks": qpop["query_checks"], "scope": qpop["scope"]}


def unreplayable(not_ev_fam: Counter[str]) -> list[dict[str, Any]]:
    """Not-evaluable families BY NAME WITH COUNTS + reason (footnoted on the row; bounds the claim)."""
    return [{"family": fam, "n": int(not_ev_fam[fam]),
             "why": NOT_YET.get(fam.split(" (")[0], "structure / interior: no identity" if "(structure" in fam
                                else "recomputed prefix: computed by the warm request's replay, never served by it" if "(recomputed prefix" in fam
                                else ("interior producer: composed into its consumer's VU" if "(interior" in fam
                                      else ("MoE plane row -> plane slice mapping failed: see population.not_evaluable" if "(plane mapping failed)" in fam
                                            else "no registered evaluator")))}
            for fam in sorted(not_ev_fam)]
