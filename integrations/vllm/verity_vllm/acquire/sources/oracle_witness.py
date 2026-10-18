"""The witness-conditioned judgment of the match oracle (`partial_source.TPPartialSource.compare_match_oracle`).

A MoE layer's align schedule is not a function of its inputs (vLLM's `moe_align_block_size` hands out each expert's slots by atomicAdd)
and a kernel's split-K cuts follow it, so two runs of one workload can differ in a collective downstream of a MoE layer with neither
wrong.  The Commit commits each align call's schedule (`moe_align_source`, members `MEMBERS`) and the Match capture records the same
(`moe_align_source.CaptureWitness`, `witness_file`); a committed collective is compared with the capture's only where every witness
in its dependency cone (`in_cone`) is equal in both runs, and one with a differing witness there is exempt (`WITNESS_RULE`).  Upstream
of the first differing witness every tensor is compared bit for bit, and with equal schedules the oracle applies in full.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

#: one align call's committed members, `<experts>/<member>`
MEMBERS = ("align.rank", "align.start", "align.blk", "align.par")


def witness_file(rank: int) -> str:
    return f"align.rank{rank}.jsonl"


def digest(b: bytes) -> str:
    """A witness member's digest: sha256 of its committed bytes (the i32 words' float32 view)."""
    return hashlib.sha256(b).hexdigest()


#: the fields of a rank's oracle a Commit's summary carries
SUMMARY_KEYS = ("verdict", "steps_compared", "tensors", "compared", "equal", "exempt", "mismatch_n", "missing_n", "not_in_oracle_n", "by_member",
                "witness_counts", "exempt_by", "why", "error")
WITNESS_RULE = ("a committed collective is compared with the capture's only when every MoE align witness in its dependency cone (in_cone) "
                "is equal in both runs; one with a differing witness there is exempt, its evidence the sampled replay against its own "
                "committed schedule, boundary linkage, the cross-rank collectives and token equality")


def _what(member: str) -> str:
    return "out" if member.endswith("/out") else "partial"


def _wkey(w: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(w["step"]), int(w["k"]), int(w["rank"])


def in_cone(w: Mapping[str, Any], rank: int, step: int, k: int, out: bool) -> bool:
    """Whether witness ``w`` (rank ``w["rank"]``'s align call in step ``w["step"]``, after that rank's first ``w["k"]`` collectives of
    the step) can reach the collective tensor of ``rank`` at (``step``, ``k``), a received output when ``out``.  Every later step reads
    state the step wrote; within the step the call's own rank computes its partials from collective ``w["k"]`` on, that collective
    hands them to every rank's received output, and every other rank's partials after it read that output."""
    s, a = int(w["step"]), int(w["k"])
    return step > s or (step == s and (k > a or (k == a and (out or rank == int(w["rank"])))))


def first_in_cone(differs: list, rank: int, step: int, k: int, out: bool) -> Mapping[str, Any] | None:
    """The first witness of ``differs`` (in `_wkey` order) in the cone of that tensor, else None."""
    for w in differs:
        if (int(w["step"]), int(w["k"])) > (step, k):
            return None
        if in_cone(w, rank, step, k, out):
            return w
    return None


def judge(res: dict[str, Any], differs: list) -> dict[str, Any]:
    """Rank ``res``'s oracle (`TPPartialSource.compare_match_oracle`) judged under the differing witnesses ``differs``: each paired tensor
    is compared, equal or a mismatch, or exempt when one of them is in its cone, and `tensor_runs` names every tensor's status in runs
    [step, what, first k, last k, status, the first such witness (an index into `exempt_by`) or None, tensors]; then the verdict."""
    rank = int(res["rank"])
    differs = sorted(differs, key=_wkey)
    by_member: dict[str, dict[str, int]] = {}
    mism: list[dict] = []
    runs: list[list] = []
    cited: list[Mapping[str, Any]] = []
    exempt_equal = 0
    for step, k, member, kind, n_diff, cn, on in sorted(res["per_tensor"], key=lambda t: (t[0], _what(t[2]), t[1])):
        what = _what(member)
        bm = by_member.setdefault(what, {"tensors": 0, "equal": 0, "exempt": 0})
        bm["tensors"] += 1
        w, ref = first_in_cone(differs, rank, step, k, what == "out"), None
        if w is not None:
            status = "exempt"
            bm["exempt"] += 1
            exempt_equal += n_diff == 0
            if w not in cited:
                cited.append(w)
            ref = cited.index(w)
        elif n_diff == 0:
            status = "equal"
            bm["equal"] += 1
        else:
            status = "mismatch"
            mism.append({"step": step, "k": k, "member": member, "kind": kind, "committed_nbytes": cn, "oracle_nbytes": on, "n_diff_bytes": n_diff})
        last = runs[-1] if runs else None
        if last is not None and last[:2] == [step, what] and last[4:6] == [status, ref]:
            last[3], last[6] = k, last[6] + 1
        else:
            runs.append([step, what, k, k, status, ref, 1])
    exempt = sum(b["exempt"] for b in by_member.values())
    res.update(tensors=len(res["per_tensor"]), compared=len(res["per_tensor"]) - exempt, equal=sum(b["equal"] for b in by_member.values()),
               exempt=exempt, exempt_equal=exempt_equal, mismatch_n=len(mism), mismatches=mism[:20], by_member=by_member,
               tensor_runs=runs, exempt_by=cited, witness_rule=WITNESS_RULE)
    res["verdict"] = "PASS" if (res["compared"] > 0 and not mism and res["missing_n"] == 0 and res["not_in_oracle_n"] == 0
                                and res["not_verified_n"] == 0 and res["witnesses"]["unpaired_n"] == 0
                                and not res["steps_without_oracle"] and not res["oracle_steps_not_committed"] and not res["oracle_inconsistent"]
                                and not res["duplicate_module_steps"] and res["steps_compared"] == res["steps_committed"]) else "FAIL"
    res.pop("why", None)
    if res["verdict"] == "FAIL":
        res["why"] = _why(res)
    return res


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
    if res["missing_n"] or res["not_in_oracle_n"]:
        return (f"{res['missing_n']} oracle collective(s) missing in the commit, {res['not_in_oracle_n']} committed member(s) not in the oracle "
                f"(first {(res['missing_in_commit'] or res['committed_not_in_oracle'])[0]})")
    if res["mismatch_n"]:
        m = res["mismatches"][0]
        return (f"{res['mismatch_n']} of {res['compared']} compared tensor(s) differ from the capture, {res['exempt']} exempt "
                f"(first: step {m['step']} k {m['k']} {m['member']})")
    return "no tensor compared" + (f": all {res['exempt']} paired tensors are exempt" if res["exempt"] else "")


def condition_on_witnesses(blocks: Mapping[Any, dict]) -> list:
    """Every rank's oracle (`TPPartialSource.compare_match_oracle`, ``blocks`` by rank) judged again under every rank's differing
    witnesses, since a rank's schedule reaches the other ranks' tensors through the collectives (`in_cone`); a block's per-tensor rows
    give way to its `tensor_runs`.  The witnesses, in `_wkey` order."""
    judged = [b for b in blocks.values() if b and "per_tensor" in b]
    differs = sorted((w for b in judged for w in b["witnesses"]["differs"]), key=_wkey)
    for b in judged:
        judge(b, differs)
        b["conditioned_on_ranks"] = sorted(int(x["rank"]) for x in judged)
        del b["per_tensor"]
    return differs


__all__ = ["MEMBERS", "SUMMARY_KEYS", "WITNESS_RULE", "condition_on_witnesses", "digest", "first_in_cone", "in_cone", "judge", "witness_file"]
