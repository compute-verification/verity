"""The match oracle's judgment of a rank's compared tensors and its MoE align witnesses (`partial_source.TPPartialSource.compare_match_oracle`).

The Commit commits each align call's members (`moe_align_source`, `MEMBERS`) and the Match capture records the same
(`moe_align_source.CaptureWitness`, `witness_file`).  They are functions of the router's top-k ids as well as of the order atomicAdd
hands out each expert's slots in: `rank` is a pair's order within its expert, `start` and `blk` follow from the per-expert counts and
`rank`, and `par` from the counts alone.  So a differing witness says the routing or the schedule differs, and `implied` names which it
can be.  Nothing is exempt (`WITNESS_RULE`): routing (the top-k ids and weights) is compared across the runs, so every committed
collective is compared with the capture's bit for bit, and a differing witness FAILS the match by name.  A schedule difference alone
would be exempt only where the capture replays the committed schedule, which it does not yet, so a pair of runs whose schedules differ
FAILS too, named as such.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

#: one align call's committed members, `<experts>/<member>`
MEMBERS = ("align.rank", "align.start", "align.blk", "align.par")
ROUTING, ROUTING_OR_SCHEDULE = "routing", "routing or schedule"


def witness_file(rank: int) -> str:
    return f"align.rank{rank}.jsonl"


def digest(b: bytes) -> str:
    """A witness member's digest: sha256 of its committed bytes (the i32 words' float32 view)."""
    return hashlib.sha256(b).hexdigest()


#: the fields of a rank's oracle a Commit's summary carries
SUMMARY_KEYS = ("verdict", "steps_compared", "tensors", "compared", "equal", "mismatch_n", "missing_n", "not_in_oracle_n", "by_member",
                "witness_counts", "witness_differs", "why", "error")
WITNESS_RULE = ("nothing is exempt: every paired tensor is compared bit for bit, and an MoE align witness that differs between the runs "
                "FAILS by name with what it implies (implied): align.par, or align.start / align.blk with align.rank equal, differ only "
                "when the routing does; align.rank differs when the routing or the schedule does, and fails too until the capture replays "
                "the committed schedule")


def implied(members: Iterable[str]) -> str:
    """What an align call whose ``members`` differ between the runs implies: `ROUTING` when `align.par` differs (the per-expert counts
    alone decide it) or `align.start` / `align.blk` differs with `align.rank` equal (both follow from the counts and the rank); else
    `ROUTING_OR_SCHEDULE` (a pair's order within its expert, which either can change)."""
    m = set(members)
    return ROUTING if "align.par" in m or ("align.rank" not in m and m & {"align.start", "align.blk"}) else ROUTING_OR_SCHEDULE


def _what(member: str) -> str:
    return "out" if member.endswith("/out") else "partial"


def judge(res: dict[str, Any]) -> dict[str, Any]:
    """Rank ``res``'s oracle (`TPPartialSource.compare_match_oracle`) decided: each paired tensor (``res["per_tensor"]``, then dropped)
    is equal or a mismatch, `tensor_runs` names every tensor's status in runs [step, what, first k, last k, status, tensors], and the
    verdict needs a compared tensor, no mismatch, every witness paired and none differing, and every step paired both ways."""
    by_member: dict[str, dict[str, int]] = {}
    mism: list[dict] = []
    runs: list[list] = []
    for step, k, member, kind, n_diff, cn, on in sorted(res.pop("per_tensor"), key=lambda t: (t[0], _what(t[2]), t[1])):
        what = _what(member)
        bm = by_member.setdefault(what, {"tensors": 0, "equal": 0})
        bm["tensors"] += 1
        if n_diff == 0:
            status = "equal"
            bm["equal"] += 1
        else:
            status = "mismatch"
            mism.append({"step": step, "k": k, "member": member, "kind": kind, "committed_nbytes": cn, "oracle_nbytes": on, "n_diff_bytes": n_diff})
        last = runs[-1] if runs else None
        if last is not None and last[:2] == [step, what] and last[4] == status:
            last[3], last[5] = k, last[5] + 1
        else:
            runs.append([step, what, k, k, status, 1])
    differs = res["witnesses"]["differs"]
    tensors = sum(b["tensors"] for b in by_member.values())
    res.update(tensors=tensors, compared=tensors, equal=sum(b["equal"] for b in by_member.values()), mismatch_n=len(mism), mismatches=mism[:20],
               by_member=by_member, tensor_runs=runs, witness_rule=WITNESS_RULE,
               witness_differs={"n": len(differs), "routing": sum(w["implies"] == ROUTING for w in differs), "first": differs[:4]})
    res["verdict"] = "PASS" if (tensors > 0 and not mism and res["missing_n"] == 0 and res["not_in_oracle_n"] == 0
                                and res["not_verified_n"] == 0 and res["witnesses"]["unpaired_n"] == 0 and not differs
                                and not res["steps_without_oracle"] and not res["oracle_steps_not_committed"] and not res["oracle_inconsistent"]
                                and not res["duplicate_module_steps"] and res["steps_compared"] == res["steps_committed"]) else "FAIL"
    res.pop("why", None)
    if res["verdict"] == "FAIL":
        res["why"] = _why(res)
    return res


def _mismatch_text(res: dict[str, Any]) -> str:
    m = res["mismatches"][0]
    return f"{res['mismatch_n']} of {res['compared']} compared tensor(s) differ from the capture (first: step {m['step']} k {m['k']} {m['member']})"


def _why(res: dict[str, Any]) -> str:
    wit = res["witnesses"]
    if res["not_verified_n"]:
        return f"{res['not_verified_n']} committed member(s) whose opening does not verify against the run root (first {res['not_verified'][0][:3]})"
    if res["oracle_steps_not_committed"]:
        return (f"oracle stored {len(res['oracle_steps_not_committed'])} step(s) the committer never committed (first {res['oracle_steps_not_committed'][0]}): "
                "the Commit stopped early or the runs differ")
    if res["oracle_inconsistent"]:
        oi = res["oracle_inconsistent"][0]
        return f"oracle step {oi['step']}: jsonl names {oi['jsonl_names']} collectives, .pt holds part {oi['pt_part']} / out {oi['pt_out']} -- truncated or inconsistent oracle"
    if res["duplicate_module_steps"]:
        dm = res["duplicate_module_steps"][0]
        return f"oracle step {dm['step']}: module(s) {dm['modules']} recorded more than once -- (step, module) pairing refused"
    if res["steps_without_oracle"]:
        return f"{len(res['steps_without_oracle'])} committed step(s) without a stored oracle (first {res['steps_without_oracle'][0]})"
    if wit["unpaired_n"]:
        return (f"{wit['unpaired_n']} MoE align witness(es) on one side only (first {wit['unpaired'][0]}): the capture predates the witness "
                "record or the runs' MoE calls differ")
    if wit["differs"]:
        w, wd = wit["differs"][0], res["witness_differs"]
        return (f"{wd['n']} MoE align witness(es) differ from the capture's, {wd['routing']} implying a routing difference (first: step {w['step']} "
                f"{w['module']} {w['members']}, implies {w['implies']}): routing is compared, never exempt, and the capture does not replay "
                "the committed schedule" + (f"; also {_mismatch_text(res)}" if res["mismatch_n"] else ""))
    if res["missing_n"] or res["not_in_oracle_n"]:
        return (f"{res['missing_n']} oracle collective(s) missing in the commit, {res['not_in_oracle_n']} committed member(s) not in the oracle "
                f"(first {(res['missing_in_commit'] or res['committed_not_in_oracle'])[0]})")
    if res["mismatch_n"]:
        return _mismatch_text(res)
    return "no tensor compared"


__all__ = ["MEMBERS", "ROUTING", "ROUTING_OR_SCHEDULE", "SUMMARY_KEYS", "WITNESS_RULE", "digest", "implied", "judge", "witness_file"]
