"""R14 finalizer — the one row-finalization path (brief §6, PLAN §2 rule 9, TABLE-RULES v1).

    python -m verity_capture.sweep.finalize <row-id> <evidence-dir> [--lane L] [--pod P] [--revision SHA]
            [--release-tag TAG] [--kind repair|revalidation|execution] [--workload PATH] [--root out/gen/sweep] [--dry-run]

Consumes a configuration (the row record + its workload manifest) and an evidence directory and *derives* the four displayed
statuses.  Nobody types PASS: every stage dict written here carries `derived_by: finalize/v1`, the evidence it was derived from,
and the reasons for anything short of ✅.  Missing evidence is PARTIAL (rendered ◐), never success.  The result is appended to
the row record as a new attempt with `rules_version: r14-v1` and `campaign: r14`; historical attempts stay untouched.

Evidence directory layout accepted (row_pod.sh / dispatcher / lane runners):
    stages.txt                      release line + per-stage verdict lines (revision consistency)
    build_summary.json              {step: {program_digest, identity, target, applicability, cmd?, sampler?}, request: {...}, build_request_LP*_T*: {...}}
    program.json | build_step/program.json      (optional) the Program: its sha256 is recomputed and compared to program_digest
    match_summary.json              cb_a.compare / cb_a.batch_decomp summary (B=1 evidence; X-09 per-request evidence at B>=2)
    match_decomp.json               (optional) X-09 per-request detail
    global_match.json | match_global.json       gmatch's `cb_a.global_match/v1` verdict (required for Match ✅ at B>=2)
    commit_verdict.json | commit/verdict.json   fail-closed verdict (commit_pass, runs_failed, required_classes, run_roots)
    commit_delta_summary.json | commit/summary.json   committer, pairs, expected/committed tensors, backlog_max, tokens_equal, control/treatment walls
    commit_summary.json             row_pod summary (control_wall / treatment_wall per run, openings, coverage)
    coverage.json | commit/coverage.json        bind's coverage checker: {checker, verdict, manifest_digest, binding_map_digest, negatives_suite, openings_checked, openings_ok, ...}
    commit/manifest_verify.json     [MAN-05] `required_manifest verify` record (verity-sweep/manifest-verify/v1): used manifest == the manifest
                                    rebuilt from the Build's Program × workload {ok, used_digest, rebuilt_digest, *_program_digest, why}; graded here
    attempt.json                    (lane's recorded invocation) pod, release_tag, release_commit|verity_commit, workload_actual
                                    {generated_lengths[], achieved_concurrency, input_lengths[]}, reuse {build: {evidence_dir, program_digest},
                                    match: {evidence_dir, match_digest}} for a Commit-only rerun (brief §9 evidence reuse)

R15 write-time validation (brief §10 "validate records at write time"; lane report). Before a record is written the attempt is
checked and a malformed one is REJECTED with the field name (exit 2, nothing written): status enums outside the finalizer's sets,
missing `rules_version`/`campaign`/`release_tag`/`revision`, missing workload vector / workload manifest digest, a row id outside the
75-row panel (out/gen/sweep/PANEL.json), a Match/Commit ✅ at B≥2 without achieved concurrency + actual generated lengths, and a
Commit-only rerun that does not reference the reused Build/Match artifacts (`reuse.build.program_digest`, `reuse.match.match_digest`,
checked against the artifacts' recomputed digests). The producer fixes the producer; records are never hand-edited.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

CODE_STAMP = "2026-09-20T04:45:00Z"  # bump when derivation rules change: reconcile/autofinalize re-derive attempts finalized before it
RULES_VERSION = "r14-v1"
CAMPAIGN = os.environ.get("SWEEP_FINALIZE_CAMPAIGN", "r16")   # the campaign that produced the attempt (r14 = revalidation window, r15 = continuous completion, r16 = overnight continuation)
CAMPAIGNS = ("r14", "r15", "r16")
GLOBAL_CHECKER = "cb_a.global_match/v1"
COVERAGE_CHECKER_PREFIX = "verity/coverage"      # bind's checker id prefix (manifest + binding map); any `coverage.json` with a verdict is read

# write-time validation vocabulary (R15, lane report): the only statuses a record may carry on a finalizer attempt
STAGE_STATUS = ("-", "PASS", "PARTIAL", "FAIL", "INFEASIBLE")
BENCH_STATUS = ("-", "DONE", "PARTIAL")
ATTEMPT_STATUS = ("NOT_RUN", "BUILD_FAIL", "BUILD_PARTIAL", "BUILD_PASS", "MATCH_FAIL", "MATCH_PARTIAL", "MATCH_PASS",
                  "COMMIT_FAIL", "COMMIT_PARTIAL", "COMMIT_PASS", "BENCHMARKED", "INFEASIBLE")
ATTEMPT_KINDS = ("repair", "revalidation", "execution")
REQUEST_ID_RE = re.compile(r"^r\d+$")            # COORD 19:25Z ruling 2: request ids are r<k>; a UUID is a workload id
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
PANEL_FILE = "PANEL.json"                        # out/gen/sweep/PANEL.json — the definitive 75-row panel (report lane, R15 task 2)


class RecordRejected(SystemExit):
    """A malformed record is refused at write time with the field names (exit status 2; nothing is written)."""

    def __init__(self, row_id: str, problems: list[str]):
        self.row_id, self.problems = row_id, problems
        super().__init__(2)

    def __str__(self) -> str:
        return f"REJECTED {self.row_id}: " + " | ".join(self.problems)


def panel_ids(root: str = "out/gen/sweep") -> frozenset[str]:
    """The fixed panel: out/gen/sweep/PANEL.json (rows[].row_id) — fallback: the row files of tag r13-final-addendum."""
    p = os.path.join(root, PANEL_FILE)
    if os.path.exists(p):
        try:
            P = load(p)
            ids = frozenset(r["row_id"] if isinstance(r, dict) else str(r) for r in (P.get("rows") or []))
            if len(ids) >= 70:
                return ids
        except Exception:
            pass
    try:
        import subprocess
        out = subprocess.run(["git", "ls-tree", "--name-only", "r13-final-addendum", "out/gen/sweep/rows/"], capture_output=True, text=True, timeout=10)
        ids = frozenset(os.path.basename(l)[:-5] for l in out.stdout.split() if l.endswith(".json"))
        return ids if len(ids) >= 70 else frozenset()
    except Exception:
        return frozenset()


BOARD_ROWS_FILE = "BOARD_ROWS.json"   # out/gen/sweep/BOARD_ROWS.json {row_id: {ordered_by, since, note}} — rows OUTSIDE the panel the coordinator ordered
                                       # rendered (COORD M-0522 (2): global's Llama B4 256/32 mixed-arrivals); shown beneath the 75-row table, never counted


def board_rows(root: str = "out/gen/sweep") -> dict[str, dict]:
    p = os.path.join(root, BOARD_ROWS_FILE)
    if not os.path.exists(p):
        return {}
    try:
        J = load(p)
        return {k: v for k, v in J.items() if isinstance(v, dict) and not v.get("lifted")} if isinstance(J, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def validate_attempt(row_id: str, row: dict, att: dict, root: str = "out/gen/sweep") -> list[str]:
    """Write-time validation of one finalizer attempt (brief §10). Returns the list of `field: problem` strings; empty = accepted."""
    P: list[str] = []
    panel = panel_ids(root)
    if panel and row_id not in panel and row_id not in board_rows(root):
        P.append(f"row_id: {row_id!r} is not one of the {len(panel)} panel rows ({root}/{PANEL_FILE}) — no panel growth, no substitute configuration")
    if str(row.get("row_id") or "") != row_id:
        P.append(f"row_id: record carries {row.get('row_id')!r} but the file is {row_id!r}")
    lane_tree_att = isinstance(att.get("executing_tree"), dict) and att["executing_tree"].get("lane_tree") is True
    for k in ("rules_version", "campaign", "release_tag", "revision", "kind", "lane", "at", "evidence_dir"):
        if k == "release_tag" and lane_tree_att and not att.get(k):
            continue    # COORD M-0132 (2): a lane-tree run has release_tag null and its executing tree recorded
        if not att.get(k):
            P.append(f"{k}: missing" + ({"release_tag": " (the announced release the evidence ran on: --release-tag / attempt.json release_tag / stages.txt release line)",
                                        "revision": " (git sha of the tree: --revision / attempt.json release_commit|verity_commit / stages.txt commit=)",
                                        "campaign": " (r14 | r15 | r16)", "rules_version": f" ({RULES_VERSION})"}.get(k, "")))
    if att.get("rules_version") and att["rules_version"] != RULES_VERSION:
        P.append(f"rules_version: {att['rules_version']!r} != {RULES_VERSION!r} (TABLE-RULES is frozen; the standard does not change per record)")
    if att.get("campaign") and str(att["campaign"]) not in CAMPAIGNS:
        P.append(f"campaign: {att['campaign']!r} not in {CAMPAIGNS}")
    if att.get("revision") and not SHA_RE.match(str(att["revision"])):
        P.append(f"revision: {att['revision']!r} is not a git sha (7–40 hex)")
    if att.get("kind") and att["kind"] not in ATTEMPT_KINDS:
        P.append(f"kind: {att['kind']!r} not in {ATTEMPT_KINDS}")
    if att.get("status") not in ATTEMPT_STATUS:
        P.append(f"status: {att.get('status')!r} not in {ATTEMPT_STATUS}")
    for st in ("build", "match", "commit", "bench"):
        d = att.get(st)
        if not isinstance(d, dict):
            P.append(f"{st}: missing stage dict")
            continue
        allowed = BENCH_STATUS if st == "bench" else STAGE_STATUS
        if d.get("status") not in allowed:
            P.append(f"{st}.status: {d.get('status')!r} not in {allowed} (nobody types PASS; the finalizer derives it)")
        if d.get("derived_by") != "finalize/v1":
            P.append(f"{st}.derived_by: {d.get('derived_by')!r} != 'finalize/v1'")
        if d.get("status") in ("PARTIAL", "FAIL", "INFEASIBLE") and not [r for r in (d.get("reasons") or []) if r]:
            P.append(f"{st}.reasons: a {d.get('status')} stage must name the missing field / failed check")
    # workload vector: the declared configuration the evidence was produced for (never inferred from what the row was meant to run)
    W = att.get("workload") if isinstance(att.get("workload"), dict) else None
    if not W:
        P.append("workload: missing workload vector {B, sampling, staggered} (the row's workload manifest was not found / not readable)")
    else:
        try:
            B = int(W.get("B") or 0)
        except Exception:
            B = 0
        rb = re.search(r"__b(\d+)__", row_id)
        if B < 1:
            P.append(f"workload.B: {W.get('B')!r} (must be the integer batch of the manifest)")
        elif rb and B != int(rb.group(1)):
            P.append(f"workload.B: manifest batch {B} != row batch {rb.group(1)} (a substitute batch is not the row)")
        if not W.get("sampling"):
            P.append("workload.sampling: missing")
        if not isinstance(W.get("staggered"), bool):
            P.append("workload.staggered: missing (bool)")
    if not att.get("workload_manifest_digest") or not HEX64_RE.match(str(att.get("workload_manifest_digest"))):
        P.append(f"workload_manifest_digest: missing/not sha256 — the row's workload file {root}/workloads/{row_id}.json must exist and be the one the evidence ran")
    # actual generated lengths + achieved concurrency where the rules require them (brief §11): every Match/Commit ✅
    B = int((W or {}).get("B") or 0)
    m_ok = (att.get("match") or {}).get("status") == "PASS"
    c_ok = (att.get("commit") or {}).get("status") == "PASS"
    if m_ok or c_ok:
        WA = att.get("workload_actual") if isinstance(att.get("workload_actual"), dict) else {}
        gl = WA.get("generated_lengths")
        if not isinstance(gl, list) or not gl or not all(isinstance(x, int) and x > 0 for x in gl):
            P.append("workload_actual.generated_lengths: missing (actual generated length per request — brief §11; from global_match.json requests[].token_selects / match_summary.json steps / attempt.json workload_actual)")
        elif B and len(gl) != B:
            P.append(f"workload_actual.generated_lengths: {len(gl)} entries for B={B} requests")
        if B >= 2:
            ac = WA.get("achieved_concurrency")
            if not isinstance(ac, int) or ac < 1:
                P.append("workload_actual.achieved_concurrency: missing at B≥2 (max requests simultaneously in flight, from the observed chronology — a configured limit is not proof, brief §11)")
    # COORD ruling M-0062: a Match ✅ at B>=2 needs the global checker's eight checks and the linkage to this Build — as write-time rejections
    if B >= 2 and m_ok:
        M_ = att.get("match") or {}
        gc = M_.get("global_checks") if isinstance(M_.get("global_checks"), dict) else {}
        absent = [f"G{i}" for i in range(1, 9) if gc.get(f"G{i}") != "PASS"]
        if absent:
            P.append(f"match.global_checks: Match PASS at B={B} without {', '.join(absent)} = PASS in global_match.json (X-09 alone is never Match ✅ at B≥2 — M-0062)")
        wd_, bgd_ = M_.get("workload_program_digest"), (att.get("build") or {}).get("global_program_digest")
        if not wd_ or not bgd_ or wd_ != bgd_:
            P.append(f"match.workload_program_digest: Match PASS at B={B} not linked to this Build (Match {str(wd_)[:16] if wd_ else 'missing'} vs Build global_program_digest {str(bgd_)[:16] if bgd_ else 'missing'} — M-0062)")
    # one coherent chain of evidence (brief §9): a Commit-only rerun must reference the reused Build/Match artifacts
    b_st = (att.get("build") or {}).get("status")
    c_st = (att.get("commit") or {}).get("status")
    R_ = att.get("reuse") if isinstance(att.get("reuse"), dict) else {}
    reused = bool(R_.get("build") or R_.get("match")) or (att.get("build") or {}).get("reused_from") is not None
    # [COORD M-0819] a P_compiled attempt carries its Build on the attempt (compiled_manifest.json) — the eager Commit-only-rerun rule is not its rule;
    # it still verifies a declared reuse if the producer declares one
    p_compiled = str((att.get("build") or {}).get("kind") or "").lower() == "p_compiled"
    if ((c_st not in (None, "-") and b_st == "-") and not p_compiled) or reused:
        # either the Build is absent from this evidence dir (Commit-only rerun) or a reuse was declared — both must verify
        rb_ = R_.get("build") if isinstance(R_.get("build"), dict) else {}
        rm_ = R_.get("match") if isinstance(R_.get("match"), dict) else {}
        if not rb_.get("program_digest"):
            P.append("reuse.build.program_digest: Commit-only rerun without the reused Build artifact (brief §9 / COORD 19:25Z ruling 5) — attempt.json `reuse: {build: {evidence_dir, program_digest}}` or --reuse-build DIR")
        elif rb_.get("verified") is not True:
            P.append(f"reuse.build.program_digest: declared {str(rb_.get('program_digest'))[:16]} does not match the reused artifact ({rb_.get('why') or 'not verified'})")
        m_st = (att.get("match") or {}).get("status")
        m_own = m_st not in (None, "-") and (att.get("match") or {}).get("reused_from") is None
        if not rm_.get("match_digest") and not m_own:
            P.append("reuse.match.match_digest: Commit-only rerun without the reused Match artifact digest — attempt.json `reuse: {match: {evidence_dir, match_digest}}` or --reuse-match DIR")
        elif rm_.get("verified") is not True and not (m_own and not rm_.get("evidence_dir") and not rm_.get("match_digest")):
            P.append(f"reuse.match.match_digest: declared {str(rm_.get('match_digest'))[:16]} does not match the reused artifact ({rm_.get('why') or 'not verified'})")
    if att.get("evidence_dir") and not os.path.isdir(att["evidence_dir"]):
        P.append(f"evidence_dir: {att['evidence_dir']} is not a directory in this tree (pull the evidence into the worktree; pods are not durable)")
    return P


def attempt_json(evdir: str) -> dict:
    p = first(evdir, "attempt.json", "../attempt.json")
    if not p:
        return {}
    try:
        A = load(p)
        return A if isinstance(A, dict) else {}
    except Exception:
        return {}


def _max_overlap(intervals: list[tuple[int, int]]) -> int:
    ev: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if hi > lo:
            ev.append((lo, 1)); ev.append((hi, -1))
    ev.sort()
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def derive_workload_actual(evdir: str, wlf: dict, A: dict) -> dict:
    """What the run actually did (brief §11): generated length per request and the achieved concurrency, from the evidence —
    global_match.json requests[] (token_selects / local_steps, arrivals; G6 PASS ⇒ observed == declared arrival), match_summary.json
    steps at B=1, or the producer's attempt.json workload_actual when nothing better is present (cross-checked, never overwritten)."""
    out: dict = {"source": None}
    g = first(evdir, "global_match.json", "match_global.json", "match/global*.json", "gmatch*.json")
    if g:
        try:
            G = load(g)
        except Exception:
            G = {}
        rq = [r for r in (G.get("requests") or []) if isinstance(r, dict)]
        if rq:
            gen = [int(r.get("token_selects") or r.get("local_steps") or (int(r["T_observed"]) + 1 if r.get("T_observed") is not None else 0) or 0) for r in rq]
            arr = [int(r.get("arrive_step_observed") if r.get("arrive_step_observed") is not None else (r.get("arrive_step_declared") or 0)) for r in rq]
            steps = [int(r.get("local_steps") or r.get("token_selects") or 0) for r in rq]
            byid = {c.get("id"): c for c in (G.get("checks") or []) if isinstance(c, dict)}
            g6 = byid.get("G6") or {}
            out.update({"generated_lengths": gen, "arrive_steps": arr, "input_lengths": [r.get("LP") for r in rq], "output_caps": [r.get("max_tokens") for r in rq],
                        "achieved_concurrency": _max_overlap([(a, a + s) for a, s in zip(arr, steps)]) if all(gen) else None,
                        "engine_steps": (byid.get("G2") or {}).get("engine_steps"), "eos_truncated_requests": G.get("eos_truncated_requests"),
                        "request_ids": [str(r.get("request_id")) for r in rq], "source": os.path.relpath(g),
                        "concurrency_basis": "observed arrival == declared (G6 PASS)" if g6.get("status") == "PASS" else "declared arrival (G6 not PASS: observed chronology not confirmed)"})
            ids = out["request_ids"]
            out["request_id_convention"] = "r<k>" if all(REQUEST_ID_RE.match(i) for i in ids) else ("uuid" if all(len(i) == 36 and i.count("-") == 4 for i in ids) else "mixed/other")
            # [R17-1 / DP-pad-02 (b)] the two lengths GM-01 keeps distinct per request: served_output_len (a_r: the answer up to and including
            # the EOS, or the cap) and executed_prefix_len (the rows the request ran: a_r + the Program's declared LAG on an EOS stop before the
            # cap); cap exhaustion beside them.  generated_lengths above is the executed row count (token_selects) as before -- it is NOT the
            # served length under LAG 1; consumers wanting the served answer read served_output_lens.  Absent on a pre-R17-1 global_match.json.
            if all("served_output_len" in r and "executed_prefix_len" in r for r in rq):
                out.update({"served_output_lens": [int(r["served_output_len"]) for r in rq], "executed_prefix_lens": [int(r["executed_prefix_len"]) for r in rq],
                            "cap_exhausted": [bool(r.get("cap_exhausted")) for r in rq],
                            "lengths_basis": "R17-1: served_output_len = a_r (answer incl. EOS, or the cap); executed_prefix_len = a_r + declared LAG on an "
                                             "EOS stop before the cap (the post-EOS row is active internal computation, its sample not served); "
                                             "generated_lengths = executed rows (token_selects)"})
            if isinstance(G.get("lag"), dict):
                out["lag"] = {k: G["lag"].get(k) for k in ("declared", "record_attests", "agree", "exercised", "eos_stop_requests", "lag_rows_observed", "variable_lag")}
    if out.get("generated_lengths") is None and wlf["B"] <= 1:
        p = first(evdir, "match_summary.json", "match/summary.json")
        if p:
            try:
                M = load(p)
                steps = as_list(M.get("steps"))
                if steps and isinstance(steps[0], int) and steps[0] > 0:
                    out.update({"generated_lengths": [int(steps[0])], "achieved_concurrency": 1, "source": os.path.relpath(p), "concurrency_basis": "single request"})
            except Exception:
                pass
    WA = A.get("workload_actual") if isinstance(A.get("workload_actual"), dict) else {}
    if WA:
        out["producer_recorded"] = {k: WA.get(k) for k in ("generated_lengths", "achieved_concurrency", "input_lengths", "output_caps", "arrival_offsets_s") if k in WA}
        pg = WA.get("generated_lengths")
        if out.get("generated_lengths") is None and isinstance(pg, list) and pg and all(isinstance(x, int) for x in pg):
            out.update({"generated_lengths": [int(x) for x in pg], "source": "attempt.json workload_actual (producer-recorded; no comparator evidence to cross-check)"})
        elif isinstance(pg, list) and out.get("generated_lengths") is not None and [int(x) for x in pg if isinstance(x, int)] != out["generated_lengths"]:
            out["producer_disagrees"] = f"attempt.json generated_lengths {pg} != evidence {out['generated_lengths']}"
        if out.get("achieved_concurrency") is None and isinstance(WA.get("achieved_concurrency"), int):
            out["achieved_concurrency"] = int(WA["achieved_concurrency"])
            out["concurrency_basis"] = "attempt.json workload_actual (producer-recorded)"
    if wlf["B"] <= 1 and out.get("generated_lengths") and out.get("achieved_concurrency") is None:
        out["achieved_concurrency"] = 1
    return out


def _artifact_digest(evdir: str, *names: str) -> tuple[str | None, str | None]:
    p = first(evdir, *names)
    return (sha256_file(p), os.path.relpath(p)) if p else (None, None)


def apply_reuse(evdir: str, A: dict, reuse_build: str | None, reuse_match: str | None, wl: dict, wlf: dict, build: dict, match: dict) -> tuple[dict, dict, dict | None]:
    """Commit-only rerun (brief §9, COORD 19:25Z ruling 5): the new attempt references the exact reused Build/Match artifacts and the
    finalizer checks their identity — the reused Build's program_digest (recomputed from program.json where shipped) equals the
    declared one, the reused Match compared that Program, and the declared match_digest equals sha256 of the reused match artifact.
    Returns (build, match, reuse_record); nothing is inferred when the declaration is missing (the validator then rejects)."""
    R_ = A.get("reuse") if isinstance(A.get("reuse"), dict) else {}
    rb = dict(R_.get("build") or {}) if isinstance(R_.get("build"), dict) else {}
    rm = dict(R_.get("match") or {}) if isinstance(R_.get("match"), dict) else {}
    if reuse_build:
        rb["evidence_dir"] = reuse_build
    if reuse_match:
        rm["evidence_dir"] = reuse_match
    if not rb and not rm:
        return build, match, None
    rec: dict = {"build": rb, "match": rm, "rule": "brief §9 evidence reuse — Commit-only rerun references the reused Build/Match artifacts; identity checked by the finalizer"}
    if build.get("status") == "-" and rb.get("evidence_dir"):
        bdir = rb["evidence_dir"]
        if os.path.isdir(bdir):
            b2 = derive_build(bdir, wl, wlf, stages_txt(bdir))
            b2["reused_from"] = os.path.relpath(bdir)
            b2["reasons"].append(f"reused Build artifact from {os.path.relpath(bdir)} (brief §9 evidence reuse; identity checked)")
            if rb.get("program_digest") and b2.get("program_digest") and rb["program_digest"] == b2["program_digest"]:
                rb["verified"] = True
            elif rb.get("program_digest") and b2.get("program_digest"):
                rb["verified"] = False
                rb["why"] = f"artifact program_digest {str(b2['program_digest'])[:16]}"
            elif not rb.get("program_digest") and b2.get("program_digest") and reuse_build:
                # the operator named the exact reused dir on the command line (--reuse-build DIR): the declaration IS the dir; the digest of
                # record is read from that artifact and the Match/Commit of this dir are checked against it downstream (same-attempt rules)
                rb["program_digest"] = b2["program_digest"]
                rb["verified"] = True
                rb["how"] = f"--reuse-build {os.path.relpath(bdir)}: program_digest read from the reused artifact (build_summary.json); Match/Commit identity checked against it"
            elif not rb.get("program_digest") and b2.get("program_digest"):
                rb["why"] = f"declare program_digest {str(b2['program_digest'])[:16]}… of {os.path.relpath(bdir)}"
            else:
                rb["why"] = "reused dir has no build_summary.json program_digest"
            rb["artifact_program_digest"] = b2.get("program_digest")
            build = b2
        else:
            rb["verified"] = False
            rb["why"] = f"{bdir} is not a directory in this tree"
    elif build.get("status") not in (None, "-") and rb.get("program_digest"):
        # [dense M-0652] the rerun dir IS the canonical attempt dir (Build/Match hard-linked in): the declared reused digest must name THIS
        # dir's Build Program (any member of its Program family); the identity is checked against the artifact present here
        fam = build_program_family(build)
        if rb["program_digest"] in fam:
            rb["verified"] = True
            rb["how"] = f"declared program_digest == this dir's Build artifact ({os.path.relpath(evdir)}; hard-linked reuse)"
        else:
            rb["verified"] = False
            rb["why"] = f"this dir's Build artifact digests {', '.join(sorted(str(d)[:12] for d in fam)) or 'none'} (hard-linked reuse) do not include the declared digest"
        rb["artifact_program_digest"] = build.get("program_digest")
    if match.get("status") == "-" and rm.get("evidence_dir"):
        mdir = rm["evidence_dir"]
        if os.path.isdir(mdir):
            m2 = derive_match(mdir, wl, wlf, build)
            m2["reused_from"] = os.path.relpath(mdir)
            m2["reasons"].append(f"reused Match artifact from {os.path.relpath(mdir)} (brief §9 evidence reuse; identity checked)")
            digs = {}
            for names in (("match_summary.json", "match/summary.json"), ("global_match.json", "match_global.json", "match/global*.json")):
                d, p = _artifact_digest(mdir, *names)
                if d:
                    digs[p] = d
            rm["artifact_digests"] = digs
            if rm.get("match_digest") and rm["match_digest"] in digs.values():
                rm["verified"] = True
            elif rm.get("match_digest"):
                rm["verified"] = False
                rm["why"] = "artifact digests " + ", ".join(f"{os.path.basename(p)}={d[:12]}" for p, d in digs.items())
            elif digs:
                rm["why"] = "declare match_digest = " + ", ".join(f"sha256({os.path.basename(p)})={d[:12]}…" for p, d in digs.items())
            else:
                rm["why"] = "reused dir has no match_summary.json / global_match.json"
            match = m2
        else:
            rm["verified"] = False
            rm["why"] = f"{mdir} is not a directory in this tree"
    elif match.get("status") not in (None, "-") and rm.get("match_digest"):
        digs = {}
        for names in (("match_summary.json", "match/summary.json"), ("global_match.json", "match_global.json", "match/global*.json")):
            d, p_ = _artifact_digest(evdir, *names)
            if d:
                digs[p_] = d
        rm["artifact_digests"] = digs
        if rm["match_digest"] in digs.values():
            rm["verified"] = True
            rm["how"] = "declared match_digest == sha256 of this dir's Match artifact (hard-linked reuse)"
        else:
            rm["verified"] = False
            rm["why"] = "this dir's Match artifact digests " + (", ".join(f"{os.path.basename(p_)}={d[:12]}" for p_, d in digs.items()) or "none") + " (hard-linked reuse) do not include the declared digest"
    return build, match, rec


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str):
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def first(evdir: str, *names: str):
    for n in names:
        for p in glob.glob(os.path.join(evdir, n)):
            if os.path.isfile(p):
                return p
    return None


def build_program_family(build: dict) -> set:
    """[exec BLOCKER M-0562 / COORD M-0544 (a)] every digest that names THIS Build's Program: the step Program, the GP-01 WORKLOAD Program
    (at B≥2 the manifest of record, the Commit's coverage record and MAN-05's rebuilt manifest all carry the workload digest — it is the
    same attempt's Program, not a different one), the per-shape request Programs (R14 moe) and the per-rank Programs (TP)."""
    fam = {build.get("program_digest"), build.get("global_program_digest"), build.get("workload_program_digest"), build.get("workload_digest"),
           (build.get("p_compiled") or {}).get("program_digest") if isinstance(build.get("p_compiled"), dict) else None}   # compiled S3: P_compiled (DECISIONS-2 §7)
    fam |= set((build.get("program_digests_per_shape") or {}).values())
    fam |= set(build.get("program_digests_per_rank") or [])
    return {d for d in fam if d}


def as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "pass"):
        return True
    if s in ("false", "0", "no", "fail"):
        return False
    return None


def as_list(v) -> list:
    if isinstance(v, list):
        return v
    if v in (None, ""):
        return []
    try:
        x = json.loads(str(v).replace("'", '"'))
        return x if isinstance(x, list) else [x]
    except Exception:
        return [v]


def stages_txt(evdir: str) -> dict:
    p = first(evdir, "stages.txt")
    out: dict = {"lines": []}
    if not p:
        return out
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        out["lines"].append(line)
        parts = line.split()
        if parts[0] == "release":
            out["release"] = parts[1] if len(parts) > 1 else None
            m = re.search(r"commit=([0-9a-f]{7,40})", line)
            out["revision"] = m.group(1) if m else None
        elif parts[0] in ("precheck", "precheck-target", "build-target", "build", "match", "commit", "bench"):
            # precheck: [R18 infra] source identity (PRECHECK_FAIL_SOURCE_IDENTITY); precheck-target / build-target: [r18-int-2] target
            # family (PRECHECK_FAIL_TARGET_FAMILY; verdict WAIVED = the canary's named waiver, never a row of record)
            out[parts[0]] = {"verdict": parts[1] if len(parts) > 1 else None, "at": parts[2] if len(parts) > 2 else None, "line": line}
    return out


# ----------------------------------------------------------------------------------------------------------------


def workload_facts(wl: dict) -> dict:
    reqs = wl.get("requests") or []
    B = len(reqs) or int(wl.get("batch") or wl.get("B") or 0)
    samp = wl.get("sampling") or {}
    variant = str(samp.get("variant") or "").lower()
    vdef = (wl.get("variants") or {}).get(samp.get("variant") or "", {}) if isinstance(wl.get("variants"), dict) else {}
    temp = vdef.get("temperature")
    stochastic = (variant not in ("greedy", "") and variant != "greedy") or (isinstance(temp, (int, float)) and temp > 0)
    lens = [len(r.get("prompt_token_ids") or []) or r.get("prompt_len") or r.get("input_len") for r in reqs]
    caps = [r.get("max_tokens") or r.get("output_cap") or r.get("max_output") for r in reqs]
    arrivals = [r.get("arrive_step") or r.get("arrival") for r in reqs]
    return {"B": B, "sampling_variant": variant or "greedy", "stochastic": bool(stochastic), "input_lengths": lens, "output_caps": caps,
            "arrivals": arrivals, "staggered": any(a not in (None, 0) for a in arrivals)}


def derive_build(evdir: str, wl: dict, wlf: dict, st: dict) -> dict:
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": []}
    p = first(evdir, "build_summary.json", "build/summary.json")
    # rev-pm M-0112: Build's global workload Program is a BUILD-stage artifact — build_workload/workload_program.json (the FIX_READY
    # row_pod.sh layout) or the gprog top-level names; global_match_global_program.json is the MATCH side of the M-0034 (ii) comparison
    # and is never read here (else Match would be compared with itself).
    gp = first(evdir, "build_workload/workload_program.json", "gprog_workload_program.json", "global_program.json", "workload_program.json", "gprog*.json", "build_global*.json")
    if not p and gp and wlf["B"] >= 2:
        return derive_build_global(evdir, gp, wl, wlf)
    if not p:
        ranks = sorted(glob.glob(os.path.join(evdir, "build_step_r*", "result.json")))
        if ranks:
            return derive_build_tp(evdir, ranks, wl, wlf)
        out["reasons"].append("no build_summary.json in the evidence directory")
        return out
    out["evidence"].append(os.path.relpath(p))
    B = load(p)
    step = B.get("step") if isinstance(B.get("step"), dict) else {}
    req = B.get("request") if isinstance(B.get("request"), dict) else {}
    shapes = {k: v for k, v in B.items() if k.startswith("build_request_") and isinstance(v, dict)}
    # [COORD M-0614 (1)] weights_root_of_record: Build's per-tensor digest root over the PINNED checkpoint shards (never the live engine)
    wro = B.get("weights_root_of_record") or ((B.get("weights") or {}).get("root_of_record") if isinstance(B.get("weights"), dict) else None)
    if isinstance(wro, dict):
        if str(wro.get("source") or "").lower().startswith("live"):
            out["weights_root_of_record_rejected"] = "build_summary.weights_root_of_record names the live engine as its source — not a pin (M-0614 (1))"
        else:
            out["weights_root_of_record"] = wro.get("root") or wro.get("weights_root")
    elif isinstance(wro, str) and wro:
        out["weights_root_of_record"] = wro
    ok = as_bool(step.get("ok")) if step else None
    if step and ok is False or (step.get("unsupported") not in (None, "None", "")):
        out["status"] = "FAIL"
        out["reasons"].append(f"builder refused: {step.get('unsupported') or step.get('exception') or step.get('message')}")
        out.update({"program_digest": step.get("program_digest"), "cmd": step.get("cmd")})
        return out
    out["program_digest"] = step.get("program_digest") or req.get("program_digest")
    out["artifact_identity"] = step.get("identity") or step.get("artifact_identity")
    out["target"] = step.get("target") or step.get("target_profile")
    out["applicability"] = step.get("applicability")
    out["cmd"] = step.get("cmd") or req.get("cmd") or B.get("cmd")
    if not out["cmd"]:
        # [global r15_2310_cp-r4_int3 / r15_2322] the runner's build_summary.json carries no `cmd`; the command OF RECORD is the `$ …` line the
        # runner echoes at the top of build_step.log / build_workload.log (GP-01 at B≥2) — read from the artifact, never composed
        for lg in ("build_step.log", "build_workload.log", "build_request.log"):
            f = os.path.join(evdir, lg)
            if os.path.isfile(f):
                try:
                    with open(f, encoding="utf-8", errors="replace") as fh:
                        for _ in range(40):
                            line = fh.readline()
                            if not line:
                                break
                            if line.startswith("$ ") and "verity_capture" in line:
                                out["cmd"] = line[2:].strip()[:600]
                                out["cmd_source"] = lg
                                break
                except Exception:  # noqa: BLE001
                    pass
            if out.get("cmd"):
                break
    out["shapes"] = sorted(shapes.keys())
    out["program_digests_per_shape"] = {k: v.get("program_digest") for k, v in shapes.items()}
    if req.get("program_digest"):
        out["program_digests_per_shape"]["request"] = req["program_digest"]      # [R14 moe] B=1 Build: the request Program (manifest / global Program source)
    out["artifact"] = os.path.relpath(p)
    out["workload_manifest_digest"] = wl.get("_sha256")
    # recompute the Program digest from content where the artifact is present
    pj = first(evdir, "program.json", "build_step/program.json", "build/program.json")
    if pj:
        d = sha256_file(pj)
        out["program_digest_recomputed"] = d
        out["evidence"].append(os.path.relpath(pj))
        if out["program_digest"] and d != out["program_digest"]:
            out["status"] = "FAIL"
            out["reasons"].append(f"program.json sha256 {d[:16]} != recorded program_digest {str(out['program_digest'])[:16]}")
            return out
    # sampling-policy applicability (brief §6 regression): a greedy Program for a stochastic request fails Build
    sampler = json.dumps(step.get("sampler") or step.get("sampling") or B.get("sampler") or "")
    prog_greedy = ("greedy" in sampler.lower()) or ("Argmax" in json.dumps(step)[:20000])
    if wlf["stochastic"] and (prog_greedy or not sampler.strip('"')):
        out["status"] = "FAIL"
        out["reasons"].append(f"applicability: workload sampling '{wlf['sampling_variant']}' (stochastic) but the Program is greedy / sampler not recorded in the build artifact — Build refused under r14-v1")
        return out
    # global-workload Program requirement (TABLE-RULES Build): one Program for the complete declared workload
    missing = [k for k in ("program_digest", "cmd", "target") if not out.get(k)]
    if not wl.get("_sha256"):
        missing.append("request manifest digest (workload file not found)")
    if wlf["B"] >= 2:
        # build_summary.json["workload"] (GP-01 in row_pod.sh) is Build's own record of the composed workload Program
        bw = B.get("workload") if isinstance(B.get("workload"), dict) else {}
        bw_digest = bw.get("workload_digest") or bw.get("program_digest")
        if bw:
            out["build_workload_record"] = {k: bw.get(k) for k in ("ok", "workload_digest", "mechanism", "artifact") if k in bw}
            if as_bool(bw.get("ok")) is False:
                out["status"] = "FAIL"
                out["reasons"].append(f"build_summary.workload.ok is false — the global workload Program composition failed ({bw.get('mechanism') or 'GP-01'})")
                return out
        if gp:
            g = derive_build_global(evdir, gp, wl, wlf)
            out["evidence"] += g["evidence"]
            out["global_program_digest"] = g.get("global_program_digest")
            out["global_program_digest_source"] = os.path.relpath(gp)
            out["global_builder"] = g.get("builder")
            out["components"] = g.get("components")
            if g["status"] == "FAIL":
                out["status"] = "FAIL"
                out["reasons"] += g["reasons"]
                return out
            if bw_digest and out["global_program_digest"] and bw_digest != out["global_program_digest"]:
                out["status"] = "FAIL"
                out["reasons"].append(f"build_summary.workload.workload_digest {str(bw_digest)[:16]} != {os.path.relpath(gp)} workload_digest {str(out['global_program_digest'])[:16]} — Build's two records of its own Program disagree (M-0112)")
                return out
            if out["global_program_digest"] and (not re.fullmatch(r"[0-9a-f]{16,64}", str(out["global_program_digest"])) or str(out["global_program_digest"]).strip("0") == ""):
                out["status"] = "FAIL"
                out["reasons"].append(f"Build's workload Program digest {str(out['global_program_digest'])[:20]!r} is not a digest (M-0112 probe: zeroed / malformed)")
                return out
            out["reasons"] += [r for r in g["reasons"] if not r.startswith("missing: ")]
            for r in g["reasons"]:
                if r.startswith("missing: "):
                    missing += [x.strip() for x in r[len("missing: "):].split(",")]
        elif per_rank_gp := tp_rank_workload_programs(evdir, B):
            # [tp M-0556 / M-0629] TP≥2 at B≥2: GP-01 composes ONE workload Program PER RANK (compose_rank<r>) — every rank's artifact must be
            # in the evidence; rank 0's digest is the row's global_program_digest, the others join program_digests_per_rank (Program family)
            out["workload_program_per_rank"] = {}
            per_rank_missing = []
            for rk, gpr in sorted(per_rank_gp.items()):
                if gpr is None:
                    per_rank_missing.append(rk)
                    continue
                g = derive_build_global(evdir, gpr, wl, wlf)
                out["evidence"] += g["evidence"]
                out["workload_program_per_rank"][rk] = {"digest": g.get("global_program_digest"), "artifact": os.path.relpath(gpr), "status": g["status"]}
                if g["status"] == "FAIL":
                    out["status"] = "FAIL"
                    out["reasons"] += [f"rank {rk}: {r}" for r in g["reasons"]]
                    return out
                out.setdefault("program_digests_per_rank", [])
                if g.get("global_program_digest") and g["global_program_digest"] not in out["program_digests_per_rank"]:
                    out["program_digests_per_rank"].append(g["global_program_digest"])
                for r in g["reasons"]:
                    if r.startswith("missing: "):
                        missing += [f"rank {rk}: {x.strip()}" for x in r[len("missing: "):].split(",")]
            r0 = out["workload_program_per_rank"].get("0") or next(iter(out["workload_program_per_rank"].values()), {})
            out["global_program_digest"] = r0.get("digest")
            out["global_program_digest_source"] = r0.get("artifact")
            out["global_builder"] = "GP-01 per rank (tp)"
            if per_rank_missing:
                missing.append("per-rank workload Program artifact for rank(s) " + ", ".join(per_rank_missing) + " (build_workload_rank<r>/workload_program.json)")
        elif bw_digest:
            out["global_program_digest"] = bw_digest
            out["global_program_digest_source"] = f"{os.path.relpath(p)} workload.workload_digest"
            out["global_builder"] = bw.get("mechanism")
            missing.append(f"Build's workload Program artifact ({bw.get('artifact') or 'build_workload/workload_program.json'} — digest {str(bw_digest)[:16]} recorded in build_summary.workload, artifact not in the evidence)")
        else:
            ranks_in_summary = sorted({m.group(1) for k in B for m in [re.fullmatch(r"(?:step|request)_rank(\d+)", k)] if m})
            hint = (f" (TP{len(ranks_in_summary)}: GP-01 composes one workload Program per rank — put each in build_workload_rank<r>/workload_program.json; the compose logs alone are not the artifact)"
                    if len(ranks_in_summary) >= 2 else "")
            out["reasons"].append(f"B={wlf['B']}: per-request/per-shape Programs only ({len(shapes)} shapes) — the one global workload Program (gprog) is not in the evidence; Build is evidence-only under r14-v1{hint}")
            missing.append("global workload Program")
    if wlf["staggered"] and not gp and not first(evdir, "global_program.json", "gprog*.json"):
        # (with a GP-01 workload Program on record the arrival schedule is graded from its `chronology` in derive_build_global — global M-0520)
        missing.append("declared arrival schedule in the Program (staggered arrivals)")
    if missing:
        out["status"] = "PARTIAL"
        out["reasons"].append("missing: " + ", ".join(missing))
    else:
        out["status"] = "PASS"
    return out


def tp_rank_workload_programs(evdir: str, B: dict) -> dict[str, str | None] | None:
    """[tp M-0629] per-rank GP-01 artifacts: {rank: path|None} for every rank named in build_summary (step_rank<r> / request_rank<r>) or
    found as build_workload_rank<r>/ dirs; None when the layout is not TP per-rank at all."""
    ranks = {m.group(1) for k in B for m in [re.fullmatch(r"(?:step|request)_rank(\d+)", k)] if m}
    found: dict[str, str | None] = {}
    for pat in ("build_workload_rank%s/workload_program.json", "build_workload_r%s/workload_program.json", "gprog_rank%s.json", "workload_program_rank%s.json", "build_global_rank%s.json"):
        for path in glob.glob(os.path.join(evdir, pat.replace("%s", "*"))):
            m = re.search(r"rank(\d+)|_r(\d+)", os.path.relpath(path, evdir))
            rk = (m.group(1) or m.group(2)) if m else None
            if rk is not None and os.path.isfile(path):
                found.setdefault(rk, path)
    if not found:
        return None
    for rk in ranks:
        found.setdefault(rk, None)
    return found


def derive_build_global(evdir: str, gp: str, wl: dict, wlf: dict) -> dict:
    """B>=2 Build from gprog's rooted workload Program (verity/workload-program/v1 or cb-a/global-program/v1)."""
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": [os.path.relpath(gp)], "artifact": os.path.relpath(gp)}
    G = load(gp)
    out["global_program_digest"] = G.get("workload_digest") or G.get("program_digest") or G.get("digest")
    out["program_digest"] = out["global_program_digest"]
    out["builder"] = (G.get("builder") or G.get("schema") or "")[:120]
    reqs = G.get("requests") or []
    out["components"] = [{"index": r.get("index"), "prompt_len": r.get("prompt_len") or r.get("LP"), "cap": r.get("cap") or r.get("max_tokens"), "arrive_step": r.get("arrive_step"),
                          "digest": ((r.get("component") or {}).get("digest") if isinstance(r.get("component"), dict) else None) or r.get("program_digest")} for r in reqs]
    msha = G.get("manifest_digest") or (G.get("workload") or {}).get("sha256") if isinstance(G.get("workload"), dict) else G.get("manifest_digest")
    if not msha and isinstance(G.get("inputs"), dict):
        msha = G["inputs"].get("manifest_sha256")
    out["workload_manifest_digest"] = wl.get("_sha256")
    out["program_manifest_digest"] = msha
    if wl.get("_sha256") and msha and msha != wl["_sha256"]:
        out["status"] = "FAIL"
        out["reasons"].append(f"global Program built from manifest {str(msha)[:12]}… but the row's workload is {wl['_sha256'][:12]}… (same-attempt rule)")
        return out
    if as_bool(G.get("ok")) is False:
        out["status"] = "FAIL"
        out["reasons"].append("workload Program composition not ok")
        return out
    missing = []
    if not out["global_program_digest"]:
        missing.append("global Program digest")
    if reqs and len(reqs) != wlf["B"]:
        out["status"] = "FAIL"
        out["reasons"].append(f"global Program has {len(reqs)} requests, the manifest declares {wlf['B']}")
        return out
    if not reqs:
        missing.append("per-request components")
    if not (G.get("chronology") or G.get("arrivals")):
        missing.append("declared arrival schedule (chronology)")
    if wlf["stochastic"] and "Greedy" in json.dumps(G)[:200000]:
        out["status"] = "FAIL"
        out["reasons"].append(f"applicability: workload sampling '{wlf['sampling_variant']}' but the components are GreedyRequest Programs")
        return out
    out["cmd"] = out["builder"]
    out["target"] = "components' TargetProfile (per-request derive_step; see component digests)"
    out["status"] = "PARTIAL" if missing else "PASS"
    if missing:
        out["reasons"].append("missing: " + ", ".join(missing))
    return out


def derive_build_tp(evdir: str, ranks: list[str], wl: dict, wlf: dict) -> dict:
    """TP layout (lane tp): build_step_r<k>/result.json per rank (cb-a/derive_step/v3)."""
    # tp lane writes build_step_r<k>/, moe lane build_step_rank<k>/ (M-0342) — same artifact (cb-a/derive_step result.json per rank); when a
    # producer relaid its dir and BOTH spellings exist for one rank (M-0385), one result per rank: prefer build_step_r<k>/, and if the two
    # carry different Program digests the rank is ambiguous (named — never pick the greener one)
    by_rank: dict[str, list[str]] = {}
    for r in ranks:
        mrk = re.search(r"build_step_r(?:ank)?(\d+)", r)
        by_rank.setdefault(mrk.group(1) if mrk else f"?{len(by_rank)}", []).append(r)
    dedup, ambiguous = [], []
    for rk in sorted(by_rank, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0)):
        cands = sorted(by_rank[rk], key=lambda p: ("rank" in os.path.basename(os.path.dirname(p)), p))
        if len(cands) > 1:
            digs = set()
            for c in cands:
                try:
                    Jc = load(c)
                    digs.add(((Jc.get("program") or {}).get("digest") if isinstance(Jc.get("program"), dict) else None))
                except Exception:  # noqa: BLE001
                    digs.add(None)
            if len(digs) > 1:
                ambiguous.append(f"rank {rk}: {len(cands)} derive results with different Program digests ({', '.join(os.path.basename(os.path.dirname(c)) for c in cands)})")
        dedup.append(cands[0])
    ranks = dedup
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": [os.path.relpath(r) for r in ranks], "ranks": {}}
    if ambiguous:
        out["status"] = "PARTIAL"
        out["reasons"].append("ambiguous rank Programs — " + "; ".join(ambiguous))
        return out
    digests = []
    for r in ranks:
        J = load(r)
        mrk = re.search(r"build_step_r(?:ank)?(\d+)", r)
        rk = mrk.group(1) if mrk else str(len(out["ranks"]))
        prog = J.get("program") if isinstance(J.get("program"), dict) else {}
        out["ranks"][rk] = {"ok": as_bool(J.get("ok")), "program_digest": prog.get("digest"), "profile": J.get("profile"), "utc": J.get("utc")}
        digests.append(prog.get("digest"))
        if as_bool(J.get("ok")) is False:
            out["status"] = "FAIL"
            out["reasons"].append(f"rank {rk} derive not ok: {str(J.get('unsupported') or J.get('error') or '')[:120]}")
            return out
        samp = J.get("sampling") if isinstance(J.get("sampling"), dict) else {}
        if wlf["stochastic"] and str(samp.get("kind") or "greedy").lower() == "greedy":
            out["status"] = "FAIL"
            out["reasons"].append(f"applicability: workload sampling '{wlf['sampling_variant']}' but rank {rk} Program is greedy")
            return out
        out["target"] = out.get("target") or J.get("target")
        out["applicability"] = out.get("applicability") or J.get("declared_derived")
        out["revision_from_artifact"] = (J.get("code_rev") or {}).get("commit") if isinstance(J.get("code_rev"), dict) else None
    out["program_digest"] = digests[0] if digests else None
    out["program_digests_per_rank"] = digests
    out["artifact"] = os.path.relpath(ranks[0])
    out["workload_manifest_digest"] = wl.get("_sha256")
    world = len(ranks)
    missing = []
    if not out["program_digest"]:
        missing.append("program_digest")
    tpw = int(re.search(r"__tp(\d+)__", wl.get("_path") or "") .group(1)) if re.search(r"__tp(\d+)__", wl.get("_path") or "") else world
    if world != tpw:
        missing.append(f"derive result for every rank ({world}/{tpw})")
    lg = first(evdir, "build_tp2.status", "row.log", "build_step_r0.log", "*build_tp2*.log", "*build_tp2*.sh")
    if lg:
        for line in open(lg, encoding="utf-8", errors="replace"):
            m = re.search(r"(derive_step\s.*)", line)
            if m:
                out["cmd"] = m.group(1).strip()[:600]
                break
    if not out.get("cmd"):
        missing.append("cmd")
    if wlf["B"] >= 2 and not first(evdir, "global_program.json", "gprog*.json"):
        missing.append("global workload Program")
    out["status"] = "PARTIAL" if missing else "PASS"
    if missing:
        out["reasons"].append("missing: " + ", ".join(missing))
    return out


def derive_match_tp(evdir: str, p: str, build: dict) -> dict:
    """TP layout: match/tp2_match.json (collective-level TP-aware Match) — under r14-v1 this is ◐ by rule: the rank-local instance
    sequences are not compared (M-TP-ENGINE); an executed failure of structure/provenance/tokens is ❌."""
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": [os.path.relpath(p)], "artifact": os.path.relpath(p)}
    J = load(p)
    out["comparator"] = f"{J.get('schema')} (collective-level: structure + cross-rank provenance + tokens)"
    out["criterion"] = "collective-level only — NOT the instance-sequence Match"
    parts = {k: as_bool((J.get(k) or {}).get("pass")) if isinstance(J.get(k), dict) else None for k in ("structure", "provenance", "tokens")}
    out["parts"] = parts
    out["control_vs_instrumented_tokens_equal"] = parts.get("tokens")
    if any(v is False for v in parts.values()):
        out["status"] = "FAIL"
        out["reasons"].append("collective-level check failed: " + ", ".join(k for k, v in parts.items() if v is False))
        stf = J.get("structure") if isinstance(J.get("structure"), dict) else {}
        prf = stf.get("per_rank") if isinstance(stf.get("per_rank"), dict) else {}
        if parts.get("structure") is False and prf:
            det = []
            for rk, v in sorted(prf.items()):
                if not isinstance(v, dict):
                    continue
                ns, nc, sl = v.get("n_steps"), v.get("n_collectives"), v.get("build_signature_len")
                mism = v.get("steps_mismatching") if isinstance(v.get("steps_mismatching"), list) else None
                per = (f"{nc / ns:g}" if isinstance(nc, (int, float)) and isinstance(ns, (int, float)) and ns else "?")
                det.append(f"rank {rk}: {per} recorded collectives/step vs {sl} in the rank Build signature" + (f", {len(mism)}/{ns} steps mismatching" if mism is not None else ""))
            if det:
                out["reasons"].append("structure detail — " + "; ".join(det) + " (an executed check that failed is ❌ under r14-v1 even when the cause is the recorder; a re-run with the fixed recorder is a new attempt)")
        return out
    st = J.get("structure") if isinstance(J.get("structure"), dict) else {}
    pr = (st.get("per_rank") or {}) if isinstance(st, dict) else {}
    bd = {rk: (v or {}).get("build_digest") for rk, v in pr.items()} if isinstance(pr, dict) else {}
    if build.get("program_digests_per_rank") and bd and any(bd.get(str(i)) and bd[str(i)] != build["program_digests_per_rank"][i] for i in range(len(build["program_digests_per_rank"]))):
        out["status"] = "FAIL"
        out["reasons"].append("matched build digests differ from the derived Programs (same-attempt rule)")
        return out
    out["status"] = "PARTIAL"
    out["reasons"].append("missing: per-rank instance-sequence comparison (cb_a.compare / global checker over rank-local instances) — collective-level structure/provenance/tokens only (M-TP-ENGINE)")
    return out


def derive_match(evdir: str, wl: dict, wlf: dict, build: dict) -> dict:
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": []}
    p = first(evdir, "match_summary.json", "match/summary.json")
    g = first(evdir, "global_match.json", "match_global.json", "match/global*.json", "gmatch*.json")
    tpm = first(evdir, "match/tp2_match.json", "tp2_match.json")
    if not p and not g and tpm:
        return derive_match_tp(evdir, tpm, build)
    if not p and not g:
        out["reasons"].append("no match_summary.json / global_match.json in the evidence directory")
        return out
    M = load(p) if p else {}
    if p:
        out["evidence"].append(os.path.relpath(p))
    out["comparator"] = M.get("comparator")
    out["verdict"] = M.get("compare_verdict") or M.get("verdict")
    out["control_vs_instrumented_tokens_equal"] = as_bool(M.get("control_vs_instrumented_tokens_equal"))
    out["totals"] = M.get("totals")
    out["steps"] = M.get("steps")
    out["fold_ok"] = as_bool(M.get("fold_ok"))
    out["cmd"] = M.get("cmd")
    out["artifact"] = os.path.relpath(p) if p else None
    tot = M.get("totals") if isinstance(M.get("totals"), dict) else {}
    if p and str(M.get("compare_rc")) not in ("0", "None", "") and not as_bool(M.get("pass")):
        out["status"] = "PARTIAL" if str(M.get("compare_rc")) == "124" else "FAIL"
        out["reasons"].append(f"comparator rc={M.get('compare_rc')} (" + ("time limit — C-FOLD-COST, unfinished" if str(M.get('compare_rc')) == "124" else "executed check failed") + ")")
        return out
    if out["verdict"] and str(out["verdict"]).upper().startswith("FAIL"):
        out["status"] = "FAIL"
        out["reasons"].append(f"comparator verdict {out['verdict']}")
        return out
    if out["control_vs_instrumented_tokens_equal"] is False:
        out["status"] = "FAIL"
        out["reasons"].append("control vs instrumented tokens differ")
        return out
    missing = []
    if out["control_vs_instrumented_tokens_equal"] is None:
        missing.append("control_vs_instrumented_tokens_equal")
    if not out.get("comparator") and out.get("global_checker"):
        out["comparator"] = out["global_checker"]
    if not out.get("comparator"):
        missing.append("comparator name + version")
    if wlf["B"] <= 1:
        # single request: cb_a.compare over ALL steps
        steps = as_list(M.get("steps"))
        expect = None
        caps = [c for c in wlf["output_caps"] if isinstance(c, int)]
        if caps:
            expect = caps[0]
        if steps and expect and isinstance(steps[0], int) and steps[0] < expect:
            # eos-truncated is allowed only if the record says so (MOE-05); otherwise a partial comparison is not a pass
            if not re.search(r"eos|served shape|T_r", json.dumps(M)):
                out["status"] = "PARTIAL"
                out["reasons"].append(f"{steps[0]}/{expect} steps compared — a partial comparison is not a Match pass (F-rev-02)")
                return out
        if str(M.get("compare_verdict") or "").upper() != "PASS":
            missing.append("compare verdict PASS")
        if tot and int(tot.get("unattributed", 0) or 0) != 0:
            out["status"] = "FAIL"
            out["reasons"].append(f"unattributed instances {tot.get('unattributed')}")
            return out
        out["criterion"] = "cb_a.compare over all steps of the single request (canonical instance sequence + operand wiring)"
        if g:
            # COORD 17:35Z item 3: at B=1 the r14-v1 Match is the global checker over the existing fold too (revalidation, no re-fold)
            G = load(g)
            out["evidence"].append(os.path.relpath(g))
            out["global_checker"] = G.get("checker") or G.get("comparator")
            out["global_verdict"] = G.get("verdict")
            checks = {c.get("id"): c.get("status") for c in (G.get("checks") or []) if isinstance(c, dict)}
            out["global_checks"] = checks or None
            failed = [k for k, v in checks.items() if v != "PASS"]
            if str(G.get("verdict") or "").upper().startswith("FAIL") or failed:
                out["status"] = "FAIL"
                out["reasons"].append("global checks failed at B=1: " + (", ".join(failed) or str(G.get("verdict"))))
                return out
            if str(G.get("verdict") or "").upper() != "PASS" or not checks:
                missing.append("global Match verdict PASS (checker present, verdict not PASS / no checks list)")
            else:
                out["comparator"] = out["global_checker"] or GLOBAL_CHECKER
                out["criterion"] = f"{GLOBAL_CHECKER} over the one-request workload Program + fold (G1–G8); cb_a.compare as evidence"
        else:
            missing.append("global Match verdict over the fold (cb_a.global_match/v1 with the row's one-request workload Program — COORD 17:35Z item 3; cb_a.compare alone is evidence)")
    else:
        # B >= 2: the global checker is the criterion; X-09 per-request equality is evidence inside it
        if g:
            out["evidence"].append(os.path.relpath(g))
            G = load(g)
            out["global_checker"] = G.get("checker") or G.get("comparator")
            out["global_verdict"] = G.get("verdict")
            if isinstance(G.get("checks"), list):
                # gmatch GM-01 (cb-a/global-match/v1): G1 declared/omitted/duplicated, G2 observed+unattributed, G3 X-09 subcircuits,
                # G4 identity map, G5 shared state, G6 chronology, G7 VU population, G8 within-step order
                byid = {c.get("id"): c for c in G["checks"] if isinstance(c, dict)}
                g1, g2, g4, g5, g6, g7 = (byid.get(k, {}) for k in ("G1", "G2", "G4", "G5", "G6", "G7"))
                G = dict(G)
                G.setdefault("requests_declared", g1.get("n_declared"))
                G.setdefault("requests_present", g1.get("n_record"))
                G.setdefault("omitted", 0 if g1.get("status") == "PASS" else g1.get("n_problems"))
                G.setdefault("duplicated", 0 if g1.get("status") == "PASS" else g1.get("n_problems"))
                G.setdefault("unattributed", g2.get("unattributed"))
                G.setdefault("identity_map_unambiguous", g4.get("status") == "PASS" if g4 else None)
                G.setdefault("shared_state_consistent", g5.get("status") == "PASS" if g5 else None)
                G.setdefault("chronology_ok", g6.get("status") == "PASS" if g6 else None)
                G.setdefault("vu_population_equal", g7.get("status") == "PASS" if g7 else None)
                rq = G.get("requests") or []
                # rev-pm M-0160: the criterion is also declared at the top level (within_step_criterion / within_step); a request
                # without its own field inherits it — a top-level `dag` must not render ✅ because requests[].criterion is absent
                top_crit = str(G.get("within_step_criterion") or G.get("within_step") or "sequence").split()[0]
                G.setdefault("criteria_per_request", [str(r.get("criterion") or top_crit).split()[0] for r in rq if isinstance(r, dict)] or [top_crit])
                if rq and all(isinstance(r, dict) for r in rq):
                    G.setdefault("steps_executed", sum(int(r.get("local_steps") or 0) for r in rq))
                    G.setdefault("steps_covered", sum(int(r.get("local_steps") or 0) - int(r.get("unexecuted_steps") or 0) for r in rq))
                out["global_checks"] = {c.get("id"): c.get("status") for c in G["checks"] if isinstance(c, dict)}
                failed = [c.get("id") for c in G["checks"] if isinstance(c, dict) and c.get("status") != "PASS"]
                if failed:
                    out["status"] = "FAIL"
                    out["reasons"].append("global checks failed: " + ", ".join(failed) + " — " + "; ".join(str((byid[f].get("problems") or [""])[0])[:100] for f in failed))
                    return out
                out["eos_truncated_requests"] = G.get("eos_truncated_requests")
                out["control_vs_instrumented_tokens_equal"] = out["control_vs_instrumented_tokens_equal"] if out["control_vs_instrumented_tokens_equal"] is not None else (as_bool(g6.get("tokens_json")) if g6 else None)
                # [R17-1 / DP-pad-02 (b)] the LAG the workload Program DECLARES and the record's attested async_scheduling, as GM-01 recorded them
                # (res.lag); a Match resting on a Program that declares no LAG while a request stopped before its cap, or on a record GM-01 flagged
                # as VARIABLE lag (res.alerts kind variable_lag), is never ✅ -- G6 fails both by name above; this is the finalizer's own guard
                # (and the two length fields travel in workload_actual.served_output_lens / executed_prefix_lens)
                LG = G.get("lag") if isinstance(G.get("lag"), dict) else None
                if LG is not None:
                    out["lag"] = {k: LG.get(k) for k in ("declared", "record_attests", "agree", "exercised", "eos_stop_requests", "lag_rows_observed", "variable_lag")}
                    if LG.get("variable_lag") or any(isinstance(a, dict) and a.get("kind") == "variable_lag" for a in as_list(G.get("alerts"))):
                        missing.append(f"constant scheduler lag on this record: GM-01 observed VARIABLE post-EOS row counts {LG.get('lag_rows_observed')} under one "
                                       f"declared scheduler (Program LAG {LG.get('declared')}) -- ALERT + DECISIONS-PENDING, never folded into a constant-lag model (R17-1)")
                    elif LG.get("declared") is None and LG.get("exercised"):
                        missing.append("a workload Program that DECLARES its LAG (R17-1): this Build predates R17-1 and a request stopped before its cap -- rebuild with the declared lag")
                    elif LG.get("declared") is not None and LG.get("agree") is not True:
                        missing.append(f"the record's attested async_scheduling agreeing with the Program's declared LAG {LG.get('declared')} (record attests LAG {LG.get('record_attests')}; R17-1)")
                # [rev-pm ALERT M-0537 F-pm-24, measured] GM-01 takes a request that stopped before its cap as "eos-truncated" on the driver's
                # finish_reason alone (None / "stop" both pass) — the served EOS token id is never checked, and G7 waives the achieved-concurrency
                # check whenever any request is eos_truncated.  Fail-closed: a Match resting on GM-01 with eos_truncated_requests renders ◐
                # 'EOS unverified' until GM-01 checks served[-1] ∈ the model's eos_token_ids (global fix) and this record is re-checked; a record
                # that already carries the EOS verification (eos_verified true per truncated request, or per-request eos_token_id ∈ declared set) passes.
                eos_tr = [str(x) for x in as_list(G.get("eos_truncated_requests")) if x is not None]
                if eos_tr:
                    ver = G.get("eos_verified") if isinstance(G.get("eos_verified"), dict) else {}
                    by_req = {str(r.get("request_id")): r for r in rq if isinstance(r, dict)}
                    unverified = []
                    for rid_ in eos_tr:
                        rr = by_req.get(rid_) or {}
                        ok = as_bool(ver.get(rid_)) if rid_ in ver else as_bool(rr.get("eos_verified"))
                        if ok is not True:
                            unverified.append(f"{rid_} (served {rr.get('served') or rr.get('n_generated') or rr.get('generated') or '?'} < cap {rr.get('max_tokens') or rr.get('cap') or '?'}, finish_reason {rr.get('finish_reason')!r})")
                    if unverified:
                        missing.append("EOS unverified for eos-truncated request(s) " + "; ".join(unverified)[:300]
                                       + " — GM-01 accepted the driver's finish_reason without checking served[-1] ∈ the model's eos_token_ids, and G7's concurrency check was waived (rev-pm ALERT M-0537 F-pm-24); lifted by global's GM-01 EOS check + a re-check of this record")
                        out["eos_unverified"] = unverified
            out["global"] = {k: G.get(k) for k in ("requests_present", "requests_declared", "omitted", "duplicated", "identity_map_unambiguous",
                                                     "chronology_ok", "unattributed", "vu_population_equal", "shared_state_consistent") if k in G}
            if str(out["global_checker"] or "") != GLOBAL_CHECKER:
                missing.append(f"global checker id {GLOBAL_CHECKER} (got {out['global_checker']})")
            # rev-pm M-0034 hardening — (i) the eight checks G1..G8 must be present (an empty / short checks list is ◐, never ✅)
            have = {c.get("id") for c in (G.get("checks") or []) if isinstance(c, dict)}
            absent = [f"G{i}" for i in range(1, 9) if f"G{i}" not in have]
            if absent:
                missing.append(f"global checks {', '.join(absent)} present with status PASS (checks list has {len(have)} of 8 — M-0034)")
            # (ii) same-attempt at B>=2: the Match ran over the workload Program that Build composed (workload_digest)
            gp_ = G.get("global_program") if isinstance(G.get("global_program"), dict) else {}
            wd = gp_.get("workload_digest") or gp_.get("program_digest") or G.get("workload_digest")
            wd_src = os.path.relpath(g) if wd else None
            # Match side only (M-0112): the copy GM-01 writes next to its verdict; never Build's build_workload/ artifact
            sib = first(evdir, "global_match_global_program.json", "match/global_program.json")
            SIB = {}
            if sib:
                try:
                    SIB = load(sib) if isinstance(load(sib), dict) else {}
                except Exception:
                    SIB = {}
            if not wd and SIB.get("workload_digest"):
                wd, wd_src = SIB["workload_digest"], os.path.relpath(sib)
            out["workload_program_digest"] = wd
            out["workload_program_digest_source"] = wd_src
            bgd = build.get("global_program_digest")
            out["linked_to_build"] = (wd == bgd) if (wd and bgd) else None
            if wd and bgd and wd != bgd:
                # COORD ruling M-0062 (ii): ◐ 'Match evidence not linked to this Build' (named), not a silent ✅
                missing.append(f"Match evidence not linked to this Build (Match workload Program {str(wd)[:16]} ≠ Build's {str(bgd)[:16]} — same-attempt rule at B≥2, M-0062)")
            if not wd:
                missing.append("workload Program digest of the Match (global_match.json global_program.workload_digest or the sibling *_global_program.json — same-attempt linkage at B≥2, M-0062)")
            elif not bgd:
                missing.append("Build's workload Program digest to compare with the Match's (GP-01 compose artifact — same-attempt linkage at B≥2, M-0062)")
            # (iii) the global Match's manifest is the row's workload file
            wsha = None
            if isinstance(gp_.get("workload"), dict):
                wsha = gp_["workload"].get("sha256") or gp_["workload"].get("manifest_sha256")
            wsha = wsha or G.get("manifest_sha256") or (G.get("inputs") or {}).get("manifest_sha256") if isinstance(G.get("inputs"), dict) or not G.get("inputs") else wsha
            if not wsha and SIB:
                wsha = SIB.get("manifest_digest") or (SIB.get("inputs") or {}).get("manifest_sha256") if isinstance(SIB.get("inputs"), dict) or not SIB.get("inputs") else None
            out["match_manifest_digest"] = wsha
            if wsha and wl.get("_sha256") and wsha != wl["_sha256"]:
                out["status"] = "FAIL"
                out["reasons"].append(f"global Match built from manifest {str(wsha)[:12]}… but the row's workload is {wl['_sha256'][:12]}… (same-attempt rule, M-0034)")
                return out
            if not wsha:
                missing.append("workload manifest sha256 of the Match (global_match.json global_program.workload.sha256 / sibling manifest_digest — compared with the row's workload file, M-0034)")
            if str(G.get("verdict") or "").upper().startswith("FAIL"):
                out["status"] = "FAIL"
                out["reasons"].append(f"global Match verdict {G.get('verdict')}: " + ", ".join(f"{k}={v}" for k, v in out["global"].items()))
                return out
            for k in ("chronology_ok", "identity_map_unambiguous", "vu_population_equal"):
                if k in G and as_bool(G.get(k)) is False:
                    out["status"] = "FAIL"
                    out["reasons"].append(f"global check {k} = False")
                    return out
            if G.get("requests_declared") is not None and G.get("requests_present") != G.get("requests_declared"):
                out["status"] = "FAIL"
                out["reasons"].append(f"requests present {G.get('requests_present')} != declared {G.get('requests_declared')}")
                return out
            if int(G.get("omitted", 0) or 0) or int(G.get("duplicated", 0) or 0):
                out["status"] = "FAIL"
                out["reasons"].append(f"omitted {G.get('omitted')} / duplicated {G.get('duplicated')} requests")
                return out
            for k in ("chronology_ok", "identity_map_unambiguous", "unattributed"):
                if k not in G:
                    missing.append(f"global check field {k}")
            cpr = G.get("criteria_per_request")
            if isinstance(cpr, dict):
                cpr = list(cpr.values())
            crits = [str(c).lower() for c in cpr] if isinstance(cpr, list) else []
            if crits and any(c not in ("sequence", "canonical", "canonical_equal", "dag") for c in crits):
                n_lc = G.get("live_cone_unmatched")
                out["live_cone_unmatched"] = n_lc
                missing.append(f"sequence or DAG criterion for every request (live-cone used; live-cone: {n_lc if n_lc is not None else '?'} unmatched — diagnostic only)")
            elif "dag" in crits:
                # Daniel APPROVED the within-step DAG criterion (M-0281 final; DECISIONS-2 §1: canonical circuit order is not runtime temporal
                # order — dependency-safe permutations of the fold's emission order are non-semantic; wiring/arrival/output mutations still
                # FAIL).  Accepted when the comparator ACCOUNTS for what it permuted (the 6a2cd7f5d/056ad8431 comparator records
                # within_step {permuted_instances, steps_with_permutation, duplicate_hash_classes, sequence_also_equal}); a `dag` label
                # without that accounting is the earlier experimental multiset — ◐, named.
                ws = [(r.get("request_id"), r.get("within_step")) for r in rq if isinstance(r, dict)]
                acct = [w for _, w in ws if isinstance(w, dict) and ("permuted_instances" in w or "sequence_also_equal" in w)]
                if ws and len(acct) == len(ws):
                    perm = sum(int(w.get("permuted_instances") or 0) for w in acct)
                    steps_p = sum(int(w.get("steps_with_permutation") or 0) for w in acct)
                    dup = sum(int(w.get("duplicate_hash_classes") or 0) for w in acct)
                    seq_all = all(as_bool(w.get("sequence_also_equal")) is True for w in acct)
                    out["within_step"] = {"criterion": "dag", "permuted_instances": perm, "steps_with_permutation": steps_p, "duplicate_hash_classes": dup, "sequence_also_equal_all": seq_all}
                    out.setdefault("footnotes", []).append(
                        "within-step order: DAG criterion (Daniel M-0281 final / DECISIONS-2 §1 — dependency-safe permutations of emission order are non-semantic; canonical addressing unchanged): "
                        + ("the emission sequence is ALSO equal on every request (0 permutations)" if seq_all else f"{perm} permuted instances on {steps_p} request-steps accepted, {dup} duplicate-hash classes"))
                else:
                    missing.append("within-step accounting for the DAG criterion (requests[].within_step {permuted_instances, steps_with_permutation, duplicate_hash_classes, sequence_also_equal} — the M-0281 comparator 6a2cd7f5d/056ad8431 records it; a `dag` label without it is the experimental multiset)")
            sc, se = G.get("steps_covered"), G.get("steps_executed")
            if sc is not None and se is not None and int(sc) < int(se):
                missing.append(f"all steps covered ({sc}/{se} — a partial comparison is not a Match pass)")
            out["criterion"] = f"{GLOBAL_CHECKER}: all requests, all steps, global I/O chronology; X-09 per-request equality as evidence"
        else:
            x09 = "batch_decomp" in str(M.get("comparator") or "") or "X-09" in json.dumps(M)
            out["reasons"].append(("X-09 per-request decomposition PASS is evidence, not the global check — " if x09 else "no B>=2 comparator evidence — ") +
                                  f"needs {GLOBAL_CHECKER} (gmatch) over this execution")
            missing.append("global Match verdict")
            out["criterion"] = "X-09 per-request decomposition (evidence only)" if x09 else "—"
        if tot and int(tot.get("unattributed", 0) or 0) != 0:
            out["status"] = "FAIL"
            out["reasons"].append(f"unattributed instances {tot.get('unattributed')}")
            return out
        if tot and int(tot.get("xreq", 0) or 0) != 0:
            out["status"] = "FAIL"
            out["reasons"].append(f"cross-request reads {tot.get('xreq')}")
            return out
    # same-attempt: the compared Program is the built one
    pd = M.get("program_digest") or M.get("derived_program_digest")
    if pd and build.get("program_digest") and pd not in build_program_family(build):
        out["status"] = "FAIL"
        out["reasons"].append(f"Match compared Program {str(pd)[:16]} but Build produced {str(build['program_digest'])[:16]} (same-attempt rule)")
        return out
    if missing:
        out["status"] = "PARTIAL"
        out["reasons"].append("missing: " + ", ".join(missing))
    else:
        out["status"] = "PASS"
    return out


def c2_aux_components_from(evdir: str) -> dict:
    """[manbind MAN-06 FIX_READY M-0543 / rev-cb M-0411] the C2-of-record blocks as THEIR OWN files beside runs.jsonl:
    `sampled_replay_p<k>.json` (or commit/…) = {sampled_replay, boundary_linkage} per instrumented pair, and the TARGETED NEGATIVES as
    `attacks/<kind>/verdict.json` (+ rev_cb_attack.json): a negative is DETECTED when its verdict's commit_pass is false.  Returns
    {sampled_replay?, boundary_linkage?, negatives?}; pairs are merged (results AND-ed, sample counts summed, per_pair kept beneath);
    an attack dir without a verdict is named `no_verdict` (never counted as detected)."""
    out: dict = {}
    files = sorted(glob.glob(os.path.join(evdir, "sampled_replay*.json")) + glob.glob(os.path.join(evdir, "commit", "sampled_replay*.json")))
    srs, bls = [], []
    for f in files:
        try:
            J = load(f)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(J, dict):
            continue
        sr = J.get("sampled_replay") if isinstance(J.get("sampled_replay"), dict) else (J if J.get("method") == "sampled-exact-replay" or J.get("population") is not None else None)
        bl = J.get("boundary_linkage") if isinstance(J.get("boundary_linkage"), dict) else (J.get("linkage") if isinstance(J.get("linkage"), dict) else None)
        if sr:
            srs.append(dict(sr, artifact=os.path.relpath(f)))
        if bl:
            bls.append(dict(bl, artifact=os.path.relpath(f)))
    if srs:
        out["sampled_replay"] = merge_sampled_replay_blocks(srs)
    if bls:
        if len(bls) == 1:
            out["boundary_linkage"] = bls[0]
        else:
            res = [as_bool(b.get("result")) for b in bls]
            out["boundary_linkage"] = {"result": True if all(v is True for v in res) else (False if any(v is False for v in res) else None),
                                       "method": bls[0].get("method"), "pairs": sum(int(b.get("pairs") or 0) for b in bls), "linked": sum(int(b.get("linked") or 0) for b in bls),
                                       "unlinked_n": sum(int(b.get("unlinked_n") or len(as_list(b.get("unlinked")))) for b in bls),
                                       "why": "; ".join(str(b.get("why")) for b in bls if b.get("why")) or None, "per_pair": bls, "artifact": bls[0].get("artifact")}
    atk_dir = next((d for d in (os.path.join(evdir, "attacks"), os.path.join(evdir, "commit", "attacks"), os.path.join(evdir, "negatives")) if os.path.isdir(d)), None)
    if atk_dir:
        results, undetected, no_verdict = {}, [], []
        for d in sorted(os.listdir(atk_dir)):
            dd = os.path.join(atk_dir, d)
            if not os.path.isdir(dd):
                continue
            kind = d
            try:
                A = load(os.path.join(dd, "rev_cb_attack.json")) if os.path.exists(os.path.join(dd, "rev_cb_attack.json")) else {}
                kind = str(((A.get("attack") or {}).get("kind") if isinstance(A.get("attack"), dict) else None) or A.get("spec") or d)
            except Exception:  # noqa: BLE001
                pass
            vp = os.path.join(dd, "verdict.json")
            if not os.path.exists(vp):
                no_verdict.append(d)
                results[d] = {"kind": kind, "detected": None, "why": "no verdict.json on record"}
                continue
            try:
                V = load(vp)
            except Exception as e:  # noqa: BLE001
                results[d] = {"kind": kind, "detected": None, "why": f"verdict unreadable: {e}"}
                no_verdict.append(d)
                continue
            cp = as_bool(V.get("commit_pass")) if V.get("commit_pass") is not None else as_bool(V.get("result"))
            detected = (cp is False)
            results[d] = {"kind": kind, "detected": detected, "first_fail_reason": str(V.get("first_fail_reason") or "")[:200], "runs_failed": V.get("runs_failed"), "runs_instrumented": V.get("runs_instrumented")}
            if not detected:
                undetected.append(d)
        if results:
            out["negatives"] = {"suite": sorted(results), "results": results, "artifact": os.path.relpath(atk_dir),
                                "result": (not undetected and not no_verdict), "undetected": undetected, "no_verdict": no_verdict,
                                "method": "targeted negatives: each attack's own commit verdict must be commit_pass false (detected)"}
    return out


def merge_sampled_replay_blocks(blocks: list[dict]) -> dict:
    """One sampled_replay component out of the per-pair records: result AND-ed, grade 'complete' only if every pair is, sample counts
    (picked/evaluated/equal/mismatch_n) summed, strata / strata_keys taken from the first pair (they describe the same population), the
    unreplayable strata from the first pair; per-pair blocks kept beneath.  Never fills a field a pair does not carry."""
    if len(blocks) == 1:
        return blocks[0]
    res = [as_bool(b.get("result")) for b in blocks]
    out = dict(blocks[0])
    out["result"] = True if all(v is True for v in res) else (False if any(v is False for v in res) else None)
    grades = [str(b.get("grade")).lower() for b in blocks if b.get("grade") is not None]
    if grades:
        out["grade"] = "complete" if all(g == "complete" for g in grades) else ("partial" if "partial" in grades else grades[0])
    samp = {}
    for k in ("picked", "evaluated", "equal", "mismatch_n", "not_evaluated_n", "stratified_picked", "targeted_n"):
        vals = [(b.get("sample") or {}).get(k) for b in blocks if isinstance(b.get("sample"), dict)]
        if vals and all(isinstance(v, int) for v in vals):
            samp[k] = sum(vals)
    if isinstance(out.get("sample"), dict):
        out["sample"] = dict(out["sample"], **samp, pairs=len(blocks))
    if all(isinstance(b.get("n"), int) for b in blocks):
        out["n"] = sum(b["n"] for b in blocks)
    mm = []
    for b in blocks:
        mm += as_list(b.get("mismatch")) + as_list(b.get("mismatches"))
    if mm:
        out["mismatch"] = mm
    out["why"] = "; ".join(str(b.get("why")) for b in blocks if b.get("why")) or None
    out["per_pair"] = blocks
    return out


def value_correspondence_from(evdir: str) -> dict | None:
    """C2 evidence: value_correspondence.json, commit_verdict.value_correspondence, or a `commit.value_correspondence` record in runs.jsonl.
    As a last resort, an explicit control-vs-instrumented root equality recorded by the harness (`root_compare` with both roots).
    [manbind MAN-06 M-0543] the standalone sampled_replay_p<k>.json + attacks/<kind>/verdict.json are folded in as components of
    whatever record is found (or form the record when nothing else is on record)."""
    vc = _value_correspondence_from(evdir)
    aux = c2_aux_components_from(evdir)
    if not aux:
        return vc
    if vc is None:
        if "sampled_replay" not in aux:
            return None
        vc = {"method": C2_OF_RECORD + " (manbind MAN-06 record: sampled_replay_p<k>.json + attacks/*/verdict.json)",
              "result": as_bool(aux["sampled_replay"].get("result")), "artifact": aux["sampled_replay"].get("artifact"), "components": {}}
    vc = dict(vc)
    comps = dict(vc.get("components")) if isinstance(vc.get("components"), dict) else {}
    for k, v in aux.items():
        have = comps.get(k) if isinstance(comps.get(k), dict) else (vc.get(k) if isinstance(vc.get(k), dict) else None)
        if have is None:
            comps[k] = v
        elif k == "negatives" and isinstance(v, dict) and v.get("results"):
            # the record names its offline suite; the LIVE attack verdicts (attacks/<kind>/verdict.json) are the negatives of record
            # for this row — folded in beside it, their undetected / no-verdict lists decide the grade
            merged = dict(have)
            merged["attacks"] = v
            for kk in ("results", "undetected", "no_verdict", "result", "method"):
                merged[kk] = v.get(kk)
            merged["suite"] = (str(have.get("suite")) + " + live attacks " + ", ".join(v.get("suite") or [])) if have.get("suite") else v.get("suite")
            merged["artifact"] = v.get("artifact")
            comps[k] = merged
    vc["components"] = comps
    return vc


def _value_correspondence_from(evdir: str) -> dict | None:
    for cand in (first(evdir, "value_correspondence.json", "commit/value_correspondence.json"), first(evdir, "commit_verdict.json", "commit/verdict.json")):
        if cand:
            try:
                J = load(cand)
            except Exception:
                continue
            vc = J.get("value_correspondence") if isinstance(J, dict) and isinstance(J.get("value_correspondence"), dict) else (J if isinstance(J, dict) and J.get("method") else None)
            if vc:
                return dict(vc, artifact=vc.get("artifact") or os.path.relpath(cand))
    # [COORD M-0481 (B) / global M-0502] the oracle-compare OF RECORD as its own record: commit/oracle_compare.json (one block, or
    # {"runs": [...]} — one per instrumented arm) or the instrumented arms' `oracle_compare` (+ `oracle_provenance`) blocks in runs.jsonl.
    ocp = first(evdir, "commit/oracle_compare.json", "oracle_compare.json")
    if ocp:
        try:
            J = load(ocp)
            blocks = J.get("runs") if isinstance(J, dict) and isinstance(J.get("runs"), list) else ([J] if isinstance(J, dict) else (J if isinstance(J, list) else []))
            merged = merge_oracle_blocks([b for b in blocks if isinstance(b, dict)])
            if merged:
                return {"method": "oracle-compare of record (COORD M-0481 form B): every committed value bit-for-bit vs the same run's independent Match observation (--match-dir)",
                        "result": merged.get("result"), "artifact": os.path.relpath(ocp), "components": {"oracle_compare": merged}}
        except Exception:  # noqa: BLE001
            pass
    rp = first(evdir, "commit/runs.jsonl", "runs.jsonl")
    if rp:
        try:
            oc_blocks = []
            for l in open(rp, encoding="utf-8", errors="replace"):
                if not l.strip():
                    continue
                r = json.loads(l)
                cm = r.get("commit") if isinstance(r.get("commit"), dict) else {}
                oc = cm.get("oracle_compare") if isinstance(cm.get("oracle_compare"), dict) else (r.get("oracle_compare") if isinstance(r.get("oracle_compare"), dict) else None)
                if oc and not (cm.get("value_correspondence") or r.get("value_correspondence")):
                    prov = cm.get("oracle_provenance") or r.get("oracle_provenance")
                    oc_blocks.append(dict(oc, oracle_provenance=prov, run=r.get("run") or r.get("index") or r.get("root") or r.get("arm")))
                    continue
                vc = cm.get("value_correspondence") or r.get("value_correspondence")
                if isinstance(vc, dict) and vc.get("method"):
                    vc = dict(vc, artifact=vc.get("artifact") or os.path.relpath(rp))
                    # [rev-cb M-0411 / manbind M-0394] runs.jsonl carries commit.sampled_replay + commit.boundary_linkage beside
                    # components.{sampled_replay, boundary_linkage}; fold the commit-level blocks in when the components lack them.
                    for k in ("sampled_replay", "boundary_linkage", "negatives"):
                        if isinstance(cm.get(k), dict) and not isinstance(vc.get(k), dict) and not isinstance((vc.get("components") or {}).get(k) if isinstance(vc.get("components"), dict) else None, dict):
                            vc[k] = cm[k]
                    return vc
                rc = cm.get("root_compare") or r.get("root_compare")
                if isinstance(rc, dict) and rc.get("instrumented_root") and (rc.get("control_root") or rc.get("independent_root") or rc.get("known_good_root")):
                    ref = rc.get("control_root") or rc.get("independent_root") or rc.get("known_good_root")
                    return {"method": "root-compare-vs-independent-run", "result": rc["instrumented_root"] == ref, "artifact": os.path.relpath(rp), "roots": [rc["instrumented_root"], ref]}
            merged = merge_oracle_blocks(oc_blocks)
            if merged:
                return {"method": "oracle-compare of record (COORD M-0481 form B): every committed value bit-for-bit vs the same run's independent Match observation (--match-dir)",
                        "result": merged.get("result"), "artifact": os.path.relpath(rp), "components": {"oracle_compare": merged}}
        except Exception:
            pass
    return None


def merge_oracle_blocks(blocks: list[dict]) -> dict | None:
    """One oracle_compare component out of the instrumented arms' blocks: result/complete/attribution.complete are AND-ed, counts are
    summed, the families the arms compared are intersected (a family one arm did not compare is not on record), steps_not_snapshotted
    is the union; per-arm blocks are kept beneath (`per_run`).  Never fills a field an arm does not carry."""
    if not blocks:
        return None
    if len(blocks) == 1:
        return dict(blocks[0], runs=1)
    out: dict = {"runs": len(blocks), "per_run": blocks}
    for k in ("result",):
        vals = [as_bool(b.get(k)) for b in blocks]
        out[k] = True if all(v is True for v in vals) else (False if any(v is False for v in vals) else None)
    comp = [as_bool(b.get("complete")) if b.get("complete") is not None else ((str(b.get("grade")).lower() == "complete") if b.get("grade") is not None else None) for b in blocks]
    out["complete"] = True if all(v is True for v in comp) else (False if any(v is False for v in comp) else None)
    grades = {str(b.get("grade")) for b in blocks if b.get("grade") is not None}
    if grades:
        out["grade"] = "complete" if grades == {"complete"} else ("/".join(sorted(grades)))
    for k in ("compared", "equal", "mismatch_n", "unattributed", "unattributed_n", "ambiguous_producer"):
        vals = [b.get(k) for b in blocks]
        if all(v is not None for v in vals):
            try:
                out[k] = sum(int(v) for v in vals)
            except Exception:  # noqa: BLE001
                out[k] = vals
    sns = []
    for b in blocks:
        if isinstance(b.get("steps_not_snapshotted"), list):
            sns += b["steps_not_snapshotted"]
    if any(isinstance(b.get("steps_not_snapshotted"), list) for b in blocks):
        out["steps_not_snapshotted"] = sorted(set(sns), key=str)
    atts = [b.get("attribution") for b in blocks if isinstance(b.get("attribution"), dict)]
    if atts:
        ac = [as_bool(a.get("complete")) for a in atts]
        out["attribution"] = {"complete": True if len(atts) == len(blocks) and all(v is True for v in ac) else (False if any(v is False for v in ac) else None),
                              "pairs": sum(int(a.get("pairs") or 0) for a in atts) if all(a.get("pairs") is not None for a in atts) else None,
                              "per_run": [a for a in atts]}
    fam_sets = []
    for b in blocks:
        f = b.get("families") or b.get("compared_families") or b.get("by_family") or b.get("compared_by_family")
        if f:
            fam_sets.append({str(k).lower() for k in (f.keys() if isinstance(f, dict) else f)})
    if fam_sets:
        out["families"] = sorted(set.intersection(*fam_sets)) if len(fam_sets) == len(blocks) else []
    provs = [b.get("oracle_provenance") if isinstance(b.get("oracle_provenance"), dict) else (b.get("oracle") if isinstance(b.get("oracle"), dict) else None) for b in blocks]
    if all(p is not None for p in provs):
        # [rev M-0008] provenance is on record only when EVERY arm carries it; arms naming different Matches are kept as a list (not one Match of record)
        out["oracle_provenance"] = provs[0] if all(p == provs[0] for p in provs) else provs
    for k in ("workload_check",):
        wcs = [b.get(k) for b in blocks if isinstance(b.get(k), dict)]
        if wcs:
            # [rev M-0008] ok only when EVERY arm carries the check and each says ok/result true (an arm without it is not on record)
            out[k] = {"ok": len(wcs) == len(blocks) and all(as_bool(w.get("ok")) is True or as_bool(w.get("result")) is True for w in wcs),
                      "arms_with_check": len(wcs), "arms": len(blocks)}
    return out


MANIFEST_VERIFY_SCHEMA = "verity-sweep/manifest-verify/v1"


def manifest_verify_from(evdir: str) -> dict | None:
    """[manbind M-0340/M-0341 MAN-05; rev-cb F-cb-11 residual] the record of `required_manifest verify` run by the producer in the tree that
    ran the row: {ok, used_digest, used_rows_digest, rebuilt_digest, used_program_digest, rebuilt_program_digest, used_identities,
    rebuilt_identities, rebuilt_complete, why}.  The finalizer grades it (and pins it to the binding population's manifest + the Build's
    Program digest); it does not run the rebuild itself — the tool belongs to the release under test, not to the report tree."""
    for pat in ("manifest_verify.json", "commit/manifest_verify.json", "verify.json", "commit/verify.json", "manifest-verify.json", "commit/manifest-verify.json", "*manifest_verify*.json", "commit/*manifest_verify*.json"):
        p = first(evdir, pat)
        if not p:
            continue
        try:
            J = load(p)
        except Exception as e:  # noqa: BLE001
            return {"artifact": os.path.relpath(p), "error": f"{type(e).__name__}: {e}"[:160]}
        if isinstance(J, dict) and str(J.get("schema") or "").startswith("verity-sweep/manifest-verify/"):
            return dict(J, artifact=os.path.relpath(p))
    return None


def stage_revision_hint(st: dict, A: dict, stage: str) -> str | None:
    """The git sha the named stage ran on, as recorded: the last stages.txt release line before that stage's line, else the single
    release line, else attempt.json release_commit|verity_commit|revision.  A hint for reading the producing tree — the record's
    `revision` is settled later in finalize() (this never overrides it)."""
    cur, hit = None, None
    for l in (st or {}).get("lines", []) or []:
        if l.startswith("release"):
            m = re.search(r"commit=([0-9a-f]{7,40})", l)
            cur = m.group(1) if m else cur
        elif l.split(" ", 1)[0] == stage and cur:
            hit = cur
    if hit:
        return hit
    if (st or {}).get("revision"):
        return st["revision"]
    for k in ("release_commit", "verity_commit", "revision"):
        v = (A or {}).get(k)
        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{7,40}", v):
            return v
    return None


_TREE_RM_CACHE: dict[str, object] = {}


def tree_required_manifest(rev: str | None):
    """`verity_capture/commit/required_manifest.py` of the tree that produced a manifest, read from git and executed as a standalone
    module (it imports only the stdlib).  None when the sha is unknown / not in this repo / not loadable — the caller then names that."""
    if not rev or not re.fullmatch(r"[0-9a-f]{7,40}", str(rev)):
        return None
    if rev in _TREE_RM_CACHE:
        return _TREE_RM_CACHE[rev]
    mod = None
    try:
        src = subprocess.run(["git", "show", f"{rev}:verity_capture/commit/required_manifest.py"], capture_output=True, text=True, timeout=20)
        if src.returncode == 0 and "def manifest_digest" in src.stdout:
            import types
            m = types.ModuleType(f"_required_manifest_{rev[:9]}")
            m.__file__ = f"<{rev[:9]}:verity_capture/commit/required_manifest.py>"
            exec(compile(src.stdout, m.__file__, "exec"), m.__dict__)  # noqa: S102 — our own repository's module at a pinned sha
            mod = m
    except Exception:  # noqa: BLE001
        mod = None
    _TREE_RM_CACHE[rev] = mod
    return mod


_TREE_MAN05_CACHE: dict[str, bool | None] = {}


def tree_has_manifest_verify(rev: str | None) -> bool | None:
    """Does the tree that ran the attempt (git sha) ship `required_manifest verify` (MAN-05)?  Read from the repo, never guessed:
    None when the sha is unknown here (a lane tree not fetched) — then the absence is a footnote, not ◐."""
    if not rev or not re.fullmatch(r"[0-9a-f]{7,40}", str(rev)):
        return None
    if rev in _TREE_MAN05_CACHE:
        return _TREE_MAN05_CACHE[rev]
    val: bool | None = None
    try:
        src = subprocess.run(["git", "show", f"{rev}:verity_capture/commit/required_manifest.py"], capture_output=True, text=True, timeout=20)
        if src.returncode == 0:
            val = ('a.cmd == "verify"' in src.stdout) or ("add_parser(\"verify\"" in src.stdout) or ("manifest-verify/v1" in src.stdout)
    except Exception:  # noqa: BLE001
        val = None
    _TREE_MAN05_CACHE[rev] = val
    return val


def apply_man05_absence(commit: dict, rev: str | None) -> None:
    """The manifest-verify record is absent: ◐ 'missing: MAN-05 …' when the tree that ran carries the tool (int-3+ / manbind 84f5cffa3),
    else a named reason on the cell (F-cb-11 residual) — never silent."""
    if not isinstance(commit, dict) or "manifest_verify" not in commit or commit.get("manifest_verify") is not None:
        return
    if commit.get("status") not in ("PASS", "PARTIAL"):
        return
    has = tree_has_manifest_verify(rev)
    if has is True:
        _cap(commit, "missing: MAN-05 manifest-verify record (required_manifest verify --manifest <evidence>/manifest.json --program <D>/build_request --workload <WL.json> --out <evidence>/commit/manifest_verify.json; B≥2: --fixture --programs-root [--workload-digest]) — the tree that ran this attempt carries the tool (manbind M-0340 (3); rev-cb F-cb-11)")
    else:
        commit.setdefault("reasons", []).append("manifest of record pinned by digest recompute only — not rebuilt from the Program × workload (MAN-05 `required_manifest verify` " + ("is not in the tree that ran this attempt" if has is False else f"availability unknown for revision {str(rev)[:9] or '?'}") + "; rev-cb F-cb-11 residual — a weakened-but-honest manifest would recompute)")


C2_PENDING = "C2 record pending (rerun queued)"   # COORD DECISION M-0451 (3) as amended by M-0481: the ◐ text of record for a row with neither C2 form


def oracle_form_b(oc: dict, B: int) -> tuple[bool, str]:
    """[COORD DECISION M-0481 (B)] ORACLE-COMPARE OF RECORD: every committed value compared bit-for-bit against the independent Match
    observation of the same run (--match-dir), result true, 0 differing, 0 unattributed, complete (every committed value — MATCH_SNAP_STEPS=all,
    manbind grade 'complete'), the F-cb-23 attribution gate at B≥2 (attribution.complete), declared-prompt + decode-chain token linkage
    (input_token_ids + sampled_token_ids among the compared families when the record lists families).  Fail-closed: a field the record
    does not carry is named, never assumed."""
    if not isinstance(oc, dict):
        return False, "no oracle_compare record"
    wc = oc.get("workload_check") if isinstance(oc.get("workload_check"), dict) else {}
    if wc and (as_bool(wc.get("ok")) is False or as_bool(wc.get("result")) is False) or str(oc.get("grade") or "") == "foreign-oracle":
        return False, "foreign oracle (the Match record is not this workload's — M-0301 R3)"
    if as_bool(oc.get("result")) is not True:
        return False, f"result {oc.get('result')!r} (needs true)"
    why = []
    grade = str(oc.get("grade") or "").lower()
    complete = as_bool(oc.get("complete"))
    if not (complete is True or grade == "complete"):
        why.append(f"not complete (complete {oc.get('complete')!r}, grade {oc.get('grade') or '?'!r}" + (f", steps not snapshotted {oc.get('steps_not_snapshotted')}" if oc.get("steps_not_snapshotted") else "") + ") — every committed value must be compared (MATCH_SNAP_STEPS=all)")
    # [rev M-0008 record conditions] a grade of attribution-complete / partial is not a complete (B) whatever the complete flag says; compared > 0;
    # workload_check.ok true on record; attribution.complete true (every B — the block is on every int-3+ record); provenance of THIS attempt's Match
    if grade and grade != "complete" and re.search(r"partial|attribution", grade):
        why.append(f"grade {oc.get('grade')!r} is not a complete (B) (rev M-0008: attribution-complete / partial does not qualify C2 alone)")
    att0 = oc.get("attribution") if isinstance(oc.get("attribution"), dict) else None
    if att0 is None:
        why.append("attribution block absent (needs attribution.complete true — rev M-0008)")
    elif as_bool(att0.get("complete")) is not True:
        why.append(f"attribution.complete {att0.get('complete')!r} (needs true — rev M-0008)")
    try:
        if int(oc.get("compared") or 0) <= 0:
            why.append(f"compared {oc.get('compared')!r} (needs > 0)")
    except Exception:  # noqa: BLE001
        why.append(f"compared {oc.get('compared')!r} not a count")
    if not wc or not (as_bool(wc.get("ok")) is True or as_bool(wc.get("result")) is True):
        why.append(f"workload_check.ok {(wc or {}).get('ok', 'absent')!r} (needs true — the oracle's requests are this workload's, rev M-0008)")
    prov = None
    if isinstance(oc.get("oracle_provenance"), list):
        why.append("oracle provenance differs across the instrumented arms (not one Match of record — rev M-0008)")
    for cand in (oc.get("oracle"), oc.get("oracle_provenance"), (oc.get("per_run") or [{}])[0].get("oracle") if isinstance(oc.get("per_run"), list) and oc["per_run"] and isinstance(oc["per_run"][0], dict) else None):
        if isinstance(cand, dict):
            prov = cand
            break
    if prov is None and not isinstance(oc.get("oracle_provenance"), list):
        why.append("oracle provenance absent (oracle.match_dir + capture_created_utc + fold_digest + tokens_sha256 of this attempt's Match — rev M-0008)")
    else:
        lack = [k for k in ("match_dir", "capture_created_utc", "fold_digest", "tokens_sha256") if not prov.get(k)]
        if lack:
            why.append("oracle provenance incomplete: missing " + ", ".join(lack) + " (rev M-0008)")
    for k in ("mismatch_n", "mismatch", "mismatches", "differing", "different", "n_mismatch"):
        if oc.get(k) is not None:
            v = len(oc[k]) if isinstance(oc[k], (list, dict)) else oc[k]
            try:
                if int(v) != 0:
                    why.append(f"{k} {v} (needs 0)")
            except Exception:  # noqa: BLE001
                why.append(f"{k} {oc.get(k)!r} not a count")
            break
    for k in ("unattributed", "unattributed_n", "ambiguous_producer"):
        if oc.get(k) is not None:
            try:
                if int(oc.get(k)) != 0:
                    why.append(f"{k} {oc.get(k)} (needs 0)")
            except Exception:  # noqa: BLE001
                why.append(f"{k} {oc.get(k)!r} not a count")
            break
    att = oc.get("attribution") if isinstance(oc.get("attribution"), dict) else {}
    if B >= 2 and as_bool(att.get("complete")) is not True:
        why.append(f"attribution.complete {att.get('complete')!r} at B={B} (F-cb-23 per-(request, engine step) gate — needs true)")
    fams = oc.get("families") or oc.get("compared_families") or oc.get("by_family") or oc.get("compared_by_family")
    if fams:
        names = {str(k).lower() for k in (fams.keys() if isinstance(fams, dict) else fams)}
        if not {"input_token_ids", "sampled_token_ids"} <= names:
            why.append("token linkage: input_token_ids + sampled_token_ids not among the compared families (declared prompt + decode chain)")
    if why:
        return False, "; ".join(why)
    una = next((oc.get(k) for k in ("unattributed", "unattributed_n", "ambiguous_producer") if oc.get(k) is not None), None)
    ppr = oc.get("per_pair_result") or oc.get("per_run")
    npairs = len(ppr) if isinstance(ppr, list) else (oc.get("pairs_instrumented") or oc.get("runs"))
    return True, (f"oracle-compare of record: {oc.get('compared')} committed values compared bit-for-bit vs the same run's independent Match observation"
                  + (f" ({oc.get('reference')})" if oc.get("reference") else "") + f", {oc.get('equal') if oc.get('equal') is not None else oc.get('compared')} equal, 0 differing"
                  + (f", {una} unattributed" if una is not None else "") + ", complete (every committed value, no step unsnapshotted)"
                  + (f", attribution complete ({att.get('pairs')} step/request pairs, unverified {att.get('unverified_n') if att.get('unverified_n') is not None else 0})" if B >= 2 else "")
                  + (f", {npairs} instrumented arm(s) each true" if npairs else "") + " — COORD M-0481 (B)")
QUAL_DIR = "qualifications"                                     # out/gen/sweep/qualifications/*.json — rev-cb's C2 qualification records (M-0451 (2)(a)), imported by path


def load_qualifications(root: str = "out/gen/sweep") -> list[dict]:
    """[COORD M-0451 (2)(a)] QUALIFICATION RECORD = one sampled-replay run per mechanism/domain, reusable across rows:
    {schema verity-sweep/c2-qualification/v1, id, board_message, mechanism, result true, sampled_replay {population, n, strata, seed,
    evaluator_ids}, domain {collector_build[], execution_profile[], hardware_class[], model_family[], B_class, mechanisms[]}, negatives, evidence}.
    Records are written by the reviewer lane and imported by path; the finalizer never composes one."""
    out: list[dict] = []
    for p in sorted(glob.glob(os.path.join(root, QUAL_DIR, "*.json"))):
        try:
            J = load(p)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(J, dict) and J.get("id") and isinstance(J.get("domain"), dict):
            out.append(dict(J, _path=os.path.relpath(p)))
    return out


def _in_domain(vals, x) -> bool:
    if vals in (None, "", [], {}):
        return False
    if isinstance(vals, str):
        vals = [vals]
    return str(x).lower() in {str(v).lower() for v in vals} or "any" in {str(v).lower() for v in vals}


def _b_class_covers(bc, B: int) -> bool:
    """B_class forms: [1, 2, 8] | "1" | "2-8" | ">=2" | "any"."""
    if bc in (None, ""):
        return False
    if isinstance(bc, (list, tuple, set)):
        return any(_b_class_covers(x, B) for x in bc)
    if isinstance(bc, int):
        return bc == B
    s = str(bc).strip().lower()
    if s == "any":
        return True
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if m:
        return int(m.group(1)) <= B <= int(m.group(2))
    m = re.fullmatch(r">=\s*(\d+)", s)
    if m:
        return B >= int(m.group(1))
    m = re.fullmatch(r"(\d+)\+", s)
    if m:
        return B >= int(m.group(1))
    return s.isdigit() and int(s) == B


def qualification_cover(row: dict, wlf: dict, commit: dict, quals: list[dict]) -> tuple[dict | None, list[str]]:
    """[COORD M-0451 (3)] a cited qualification record covers the row iff EVERY domain field covers the row/attempt fact:
    model_family (row model), hardware_class (row hw), execution_profile (row execution), B_class (declared batch), collector_build (the
    attempt's collector variant / build id — an attempt without one is not covered), mechanisms (⊇ the attempt's required identity classes —
    an attempt without a class list is not covered).  Returns (record | None, why-not per record)."""
    cfg = row.get("config") or {}
    rid = str(row.get("row_id") or "")
    parts = rid.split("__")
    model = cfg.get("model") or (parts[0] if parts else "?")
    hw = cfg.get("hardware") or cfg.get("hw") or (parts[2] if len(parts) > 2 else "?")
    ex = cfg.get("execution") or (parts[-1] if parts else "?")
    B = int(wlf.get("B") or cfg.get("batch") or 1)
    coll = commit.get("collector_variant") or commit.get("collector_build") or commit.get("collector")
    classes = commit.get("required_class_names") or commit.get("identity_classes") or commit.get("families")
    whynot: list[str] = []
    for q in quals:
        d = q.get("domain") or {}
        miss = []
        if as_bool(q.get("result")) is not True:
            miss.append("result not true")
        if not _in_domain(d.get("model_family"), model):
            miss.append(f"model_family ∌ {model}")
        if not _in_domain(d.get("hardware_class"), hw):
            miss.append(f"hardware_class ∌ {hw}")
        if not _in_domain(d.get("execution_profile"), ex):
            miss.append(f"execution_profile ∌ {ex}")
        if not _b_class_covers(d.get("B_class"), B):
            miss.append(f"B_class ∌ {B}")
        if not coll:
            miss.append("attempt's collector build not recorded (commit.collector_variant)")
        elif not _in_domain(d.get("collector_build"), coll):
            miss.append(f"collector_build ∌ {str(coll)[:24]}")
        dm = d.get("mechanisms")
        if not isinstance(classes, (list, tuple)) or not classes:
            miss.append("attempt's identity classes not recorded (mechanism coverage not checkable)")
        elif not (isinstance(dm, (list, tuple)) and ("any" in {str(x).lower() for x in dm} or {str(c).lower() for c in classes} <= {str(x).lower() for x in dm})):
            missing_m = sorted({str(c) for c in classes} - {str(x) for x in (dm or [])})
            miss.append("mechanisms ∌ " + ", ".join(missing_m[:5]) + (" …" if len(missing_m) > 5 else ""))
        if not miss:
            return q, whynot
        whynot.append(f"{q.get('id')}: " + "; ".join(miss)[:200])
    return None, whynot


def row_linkage_evidence(vc: dict | None, link: dict | None, wlf: dict) -> tuple[bool, str]:
    """[COORD M-0451 (3)] the row's OWN linkage evidence that must pass beside a cited qualification: the F-cb-23 gate at B≥2
    (boundary_linkage.result true, or oracle_compare with attribution.complete and result not false), tokens/prompt linkage at B=1
    (boundary_linkage.result true, or an oracle_compare whose compared families include input_token_ids + sampled_token_ids with result true)."""
    B = int(wlf.get("B") or 1)
    if isinstance(link, dict) and as_bool(link.get("result")) is True:
        return True, f"boundary_linkage ({str(link.get('method') or 'per-(request, engine step) token gate')[:50]}, {link.get('checked') or '?'} checked)"
    comps = (vc or {}).get("components") if isinstance((vc or {}).get("components"), dict) else {}
    oc = comps.get("oracle_compare") if isinstance(comps.get("oracle_compare"), dict) else ((vc or {}) if str((vc or {}).get("method") or "") == "oracle-compare" else None)
    if isinstance(oc, dict) and as_bool(oc.get("result")) is not False:
        if B >= 2:
            att = oc.get("attribution") if isinstance(oc.get("attribution"), dict) else {}
            if as_bool(att.get("complete")) is True and as_bool(oc.get("result")) is True:
                return True, f"oracle_compare attribution complete (F-cb-23 gate; compared {oc.get('compared')})"
            return False, "F-cb-23 gate at B≥2 not on record (boundary_linkage.result true, or oracle_compare.attribution.complete true with result true)"
        fams = oc.get("families") or oc.get("compared_families") or oc.get("by_family") or oc.get("compared_by_family") or {}
        names = {str(k).lower() for k in (fams.keys() if isinstance(fams, dict) else fams)} if fams else set()
        if as_bool(oc.get("result")) is True and {"input_token_ids", "sampled_token_ids"} <= names:
            return True, "oracle_compare of input_token_ids + sampled_token_ids vs the Match record (tokens/prompt linkage at B=1)"
        return False, "tokens/prompt linkage at B=1 not on record (boundary_linkage.result true, or oracle_compare naming input_token_ids + sampled_token_ids equal)"
    return False, ("F-cb-23 gate (per-(request, engine step) token gate) not on record" if B >= 2 else "tokens/prompt linkage (committed input_token_ids == the declared prompt; decode chain) not on record")


C2_OF_RECORD = "sampled-exact-replay"
C2_AUX_COMPONENTS = ("sampled_replay", "boundary_linkage", "linkage", "negatives")   # component key of record: boundary_linkage (rev-cb M-0411); 'linkage' read as its alias
C2_M0111_BASIS_NOTE = ("C2 basis on this row: an M-0111-standard form (root-compare vs an independent run at B=1 / complete oracle-compare) — the C2 OF RECORD under "
                       "Daniel's DECISIONS-2 §4 (M-0356 (1), 22:01Z) is sampled exact VU replay + ordered boundary linkage, NOT on record for this row; the ✅ stands per "
                       "DECISIONS-2 §2 ('existing results remain valid within their stated scope', no matrix reset) with this basis named — transition confirmation asked of COORD")


# [rev-cb ALERT M-0610 / COORD DECISION M-0614 (1), 00:20Z] a served-model substitution (one weight element flipped before register_weights)
# passes form (A): sampled replay evaluates F_V with the live weights the committer hashed and the 'weights equal' check is self-consistency,
# not authentication.  Form (A) alone = ◐ 'weights unpinned' until Build emits weights_root_of_record from the PINNED CHECKPOINT files and
# Match/Commit compare to it.  Form (B) keeps ✅ ONLY while rev-cb's measurement (due 00:50Z) shows flipweight FAILS oracle-compare; None = pending.
# [DANIEL M-0776 02:11Z (1)] form (B) oracle-compare counts as C2 ONLY if the Match observer's acquisition/mapping path is independent of the
# collector's identity->buffer mapping — rev-cb states it in the HANDOFF; None = not stated (fail closed: (B) is a regression test, not C2, until stated).
# [rev RESULT M-0008, 04:04Z — COORD ruling 4 / BRIEF §4B; finding lane/r16-rev out/gen/sweep/review/FORM-B-INDEPENDENCE.md 0353317d9] verdict
# INDEPENDENT: the Match-side observation (vllm_adapter._after_forward rows cross-checked against the device positions buffer + observer
# snapshots + fold instances) shares no identity→buffer attribution code with native_collect._record_step_rows + binding.build_binding_map;
# the map entry supplies only the lookup key + committed slice.  Perturbations of the collector alone are named by the compare (5 unit tests
# through the real binder; live b2swapeq / b2relabelop / b2lateread / b2flip1 on int-3/int-4).  Flipped by report under rev's RECORD
# CONDITIONS (oracle_form_b): complete == true (MATCH_SNAP_STEPS=all), result true, compared > 0, mismatch_n 0, attribution.complete true,
# workload_check.ok true, oracle provenance of THIS attempt's Match (match_dir, capture_created_utc, fold_digest, tokens_sha256 present);
# grade attribution-complete / partial does NOT qualify.  The pin stays unconditional ((B) never names a weight — M-0610/M-0641).
FORM_B_PATH_INDEPENDENT: bool | None = True
FORM_B_PATH_STATED = ("form (B) path independence stated by rev (RESULT M-0008, FORM-B-INDEPENDENCE.md 0353317d9): the Match observer's acquisition/mapping path shares no "
                      "identity→buffer mapping code with the collector — (B) counts as C2 under DANIEL M-0776 (1) when the record meets rev's conditions")
FORM_B_PATH_PENDING = ("C2 form (B) oracle-compare is complete on the record but is NOT the C2 of record until rev-cb states that cb_a's snapshot path and native_collect_v2b + binding "
                       "share no identity→buffer mapping code (DANIEL M-0776 (1): 'counts as C2 ONLY if the Match observer's path is independent of the collector's mapping'; if they share it, (B) is a regression test)")
# [DANIEL M-0776 (2)] the weights pin is UNCONDITIONAL and separate from C2: a Commit without the MAN-09 record OK renders ◐ 'weights unpinned' regardless of form
WEIGHTS_PIN_UNCONDITIONAL = True
FORM_B_WEIGHTS_ROUTE_INDEPENDENT: bool | None = True   # rev-cb RESULT M-0641 (measured 00:31Z, int-4 a702b2974, SmolLM2 B1): flipweight → oracle-compare result false, 541 identities differ → form (B) keeps ✅ per COORD M-0614 (1)
FORM_B_MEASURED = "form (B) weights route measured independent: flipweight (one served weight element changed before register_weights) FAILS oracle-compare — 541 identities differ from the Match oracle (rev-cb RESULT M-0641, int-4 a702b2974; COORD M-0614 (1))"
WEIGHTS_UNPINNED = ("weights unpinned — the C2 form (A) record authenticates the sampled VUs against the LIVE weights the committer registered (self-consistency, "
                    "not the checkpoint of record): a served-model substitution passes it (rev-cb ALERT M-0610, measured on r15-int-4; COORD M-0614 (1)). Commit ✅ needs "
                    "Build's weights_root_of_record (per-tensor digests from the pinned checkpoint shards, manifests/checkpoints.json) equal to the run's committed weights root")


def build_weights_roots_of_record(evdir: str, build: dict) -> tuple[set, str]:
    """[manbind MAN-09 M-0638 / tp TP-09 M-0693 / global 365f06510] the Build-side weights roots OF RECORD — derived from the PINNED checkpoint
    shards (weights_of_record.py root_of_record), never from the live engine: build_summary.weights_root_of_record, build_request*/weights_of_record.json
    (per component at B≥2, per rank at TP≥2), workload_program.json weights_of_record.{components, per_rank}.  Returns (roots, how)."""
    roots, srcs = set(), []
    wr = build.get("weights_root_of_record") if isinstance(build, dict) else None
    if wr:
        roots.add(str(wr)); srcs.append("build_summary.weights_root_of_record")
    pats = ["weights_root_of_record.json", "weights_of_record.json", "build_request/weights_of_record.json", "build_request/weights_root_of_record.json",
            "build_request_*/weights_of_record.json", "build_request_rank*/weights_of_record.json", "build_step*/weights_of_record.json", "rank*/build_request*/weights_of_record.json"]
    seen = set()
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(evdir, pat))):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f)
            try:
                J = load(f)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(J, dict):
                continue
            if str(J.get("source") or "").lower().startswith("live"):
                srcs.append(f"{os.path.relpath(f, evdir)} names its source as the live engine — not a pin")
                continue
            r = J.get("root_of_record") or J.get("weights_root_of_record") or J.get("weights_root") or J.get("root")
            if r:
                roots.add(str(r)); srcs.append(os.path.relpath(f, evdir))
    for wpf in sorted(glob.glob(os.path.join(evdir, "build_workload*", "workload_program.json"))):
        try:
            W = load(wpf).get("weights_of_record")
        except Exception:  # noqa: BLE001
            W = None
        if isinstance(W, dict):
            for c in (W.get("components") or {}).values() if isinstance(W.get("components"), dict) else []:
                if isinstance(c, dict) and c.get("root_of_record"):
                    roots.add(str(c["root_of_record"]))
            for r in (W.get("per_rank") or {}).values() if isinstance(W.get("per_rank"), dict) else []:
                if r:
                    roots.add(str(r))
            if W.get("components") or W.get("per_rank"):
                srcs.append(os.path.relpath(wpf, evdir) + " weights_of_record")
    return roots, ", ".join(srcs) or "none"


def weights_pin_component(sr: dict | None, commit_out: dict, evdir: str | None = None) -> dict | None:
    """the Commit run's weights-pin record (MAN-09: commit.weights_pin / value_correspondence.components.weights_pin), else the
    sampled_replay.weights block (self-consistency only: committed_root == replay_root)."""
    vc = commit_out.get("value_correspondence") if isinstance(commit_out.get("value_correspondence"), dict) else {}
    comps = vc.get("components") if isinstance(vc.get("components"), dict) else {}
    for cand in (comps.get("weights_pin"), commit_out.get("weights_pin"), vc.get("weights_pin")):
        if isinstance(cand, dict):
            return dict(cand, _kind="weights_pin")
    # [dense int-5 r15_0154 / MAN-09 as written by native_collect_v2b] the pin of record is per INSTRUMENTED RUN in runs.jsonl
    # (run.commit.weights_pin / run.commit.value_correspondence.components.weights_pin); the row-level value_correspondence.json may not carry it.
    # Every instrumented run's pin must hold: the first failing one is returned, else the first (with the run count).
    if evdir:
        pins = []
        for r in _runs(evdir):
            c = r.get("commit") if isinstance(r.get("commit"), dict) else {}
            vc_r = c.get("value_correspondence") if isinstance(c.get("value_correspondence"), dict) else {}
            comps_r = vc_r.get("components") if isinstance(vc_r.get("components"), dict) else {}
            cand = c.get("weights_pin") if isinstance(c.get("weights_pin"), dict) else (comps_r.get("weights_pin") if isinstance(comps_r.get("weights_pin"), dict) else None)
            if isinstance(cand, dict):
                pins.append(cand)
        if pins:
            bad = [w for w in pins if as_bool(w.get("result")) is not True or (w.get("mismatched") or w.get("missing"))]
            pick = bad[0] if bad else pins[0]
            return dict(pick, _kind="weights_pin", _runs_n=len(pins), _runs_bad=len(bad))
    if isinstance(sr, dict) and isinstance(sr.get("weights"), dict):
        return dict(sr["weights"], _kind="sampled_replay.weights")
    return None


def weights_pin_of_record(evdir: str, build: dict, sr: dict | None, commit_out: dict) -> tuple[bool, str]:
    """[COORD M-0614 (1) / manbind MAN-09 M-0638 / rev-cb W1-W3 M-0655] weights pinned = the Build's weights root(s) OF RECORD (from the pinned
    checkpoint shards, never the live engine) equal the Commit run's weights root, and the run's own MAN-09 weights_pin record (when present)
    passes: result true, 0 mismatched, 0 missing, all shards pinned.  Returns (ok, how); absence of either side is (False, named)."""
    roots, src = build_weights_roots_of_record(evdir, build)
    wp = weights_pin_component(sr, commit_out, evdir)
    if wp and wp.get("_kind") == "weights_pin":
        res = as_bool(wp.get("result") if wp.get("result") is not None else (wp.get("ok") if wp.get("ok") is not None else wp.get("pass")))
        mism = wp.get("mismatched") if wp.get("mismatched") is not None else wp.get("mismatch_n")
        miss = wp.get("missing") if wp.get("missing") is not None else wp.get("missing_n")
        mism_n = len(mism) if isinstance(mism, list) else (mism if isinstance(mism, int) else None)
        miss_n = len(miss) if isinstance(miss, list) else (miss if isinstance(miss, int) else None)
        shards = as_bool(wp.get("all_shards_pinned")) if wp.get("all_shards_pinned") is not None else None
        rr = wp.get("root_of_record") or wp.get("weights_root_of_record")
        if res is not True:
            return False, f"MAN-09 weights_pin result {res} ({str(wp.get('why') or wp.get('first_fail_reason') or '')[:160] or 'not true'})"
        if mism_n:
            return False, f"MAN-09 weights_pin: {mism_n} field(s) mismatched vs the checkpoint of record"
        if miss_n:
            return False, f"MAN-09 weights_pin: {miss_n} field(s) of the Program's weights struct missing from the pin"
        if shards is False:
            return False, "MAN-09 weights_pin: not every checkpoint shard is pinned (all_shards_pinned false — rev-cb W1)"
        if not rr:
            return False, "MAN-09 weights_pin record carries no root_of_record"
        if roots and str(rr) not in roots:
            return False, f"MAN-09 weights_pin root_of_record {str(rr)[:12]}… ≠ Build's weights root(s) of record {', '.join(sorted(r[:12] + '…' for r in roots))} ({src})"
        if not roots:
            return False, f"MAN-09 weights_pin passes (root_of_record {str(rr)[:12]}…, compared {wp.get('compared')}, 0 mismatched) but no Build-side weights_of_record.json / weights_root_of_record is on the attempt to pin it against (M-0614 (1): the pin is Build's)"
        unp = wp.get("unpinned") if wp.get("unpinned") is not None else wp.get("unpinned_n")
        unp_n = len(unp) if isinstance(unp, list) else (unp if isinstance(unp, int) else 0)
        return True, (f"MAN-09 weights pin: run root_of_record == Build's weights root of record {str(rr)[:12]}… (pinned checkpoint shards; {src}); compared {wp.get('compared')}, "
                      f"0 mismatched, 0 missing, all shards pinned" + (f"; {unp_n} engine constant(s) unpinned by name" if unp_n else "")
                      + (f"; pin held on every one of {wp['_runs_n']} instrumented runs (runs.jsonl commit.weights_pin)" if wp.get("_runs_n") else ""))
    if not roots:
        return False, "no weights_root_of_record from Build (pinned checkpoint shards) on record"
    cr = None
    if isinstance(sr, dict) and isinstance(sr.get("weights"), dict):
        cr = sr["weights"].get("committed_root") or sr["weights"].get("committed_weights_root")
    if not cr:
        cr = commit_out.get("weights_root") or commit_out.get("committed_weights_root")
    if not cr:
        return False, f"Build weights root of record {', '.join(sorted(r[:12] + '…' for r in roots))} on record but the Commit run's committed weights root is not (sampled_replay.weights.committed_root / verdict weights root)"
    if str(cr) not in roots:
        return False, f"committed weights root {str(cr)[:12]}… ≠ weights root(s) of record {', '.join(sorted(r[:12] + '…' for r in roots))} (served model is not the checkpoint of record)"
    return True, f"committed weights root == Build's weights root of record {str(cr)[:12]}… (pinned checkpoint shards; {src})"


def grade_sampled_replay(sr: dict, link: dict | None, neg, wlf: dict) -> dict:
    """[Daniel DECISIONS-2 §4 / COORD M-0356 (1)] the C2 of record: a small stratified sample of VUs replayed EXACTLY from authenticated committed
    values (F_V(τ(in V)) = τ(out V) bit-for-bit via canonical references) + ORDERED BOUNDARY LINKAGE (the per-(request, engine step) token gate,
    F-cb-23) + the mechanism's TARGETED NEGATIVES on record.  Required record fields: result true, population, n, strata, seed (applicability
    named); `linkage.result` true; negatives {suite, results|artifact} (or the coverage record's negatives suite).  Identities with no registered
    relation are an UNREPLAYABLE stratum by name (rev-cb M-0370 (3)(d)) — footnoted, they bound what this C2 qualifies.
    Returns {"status": PASS|FAIL|PARTIAL, "missing": [...], "fail": str|None, "footnotes": [...], "basis": str}."""
    out = {"status": "PARTIAL", "missing": [], "fail": None, "footnotes": [], "basis": None}
    res = sr.get("result")
    if as_bool(res) is False:
        mm = as_list(sr.get("mismatches"))
        out["status"] = "FAIL"
        out["fail"] = f"C2 sampled exact replay FAIL: {str(sr.get('why') or '')[:160] or 'a sampled VU does not replay from its committed inputs'}" + (f" (first {json.dumps(mm[0], default=str)[:120]})" if mm else "")
        return out
    if link and as_bool(link.get("result")) is False:
        out["status"] = "FAIL"
        out["fail"] = f"C2 ordered boundary linkage FAIL: {str(link.get('why') or 'prescribed input / designated output not tied to its interface (per-(request, engine step) token gate)')[:200]}"
        return out
    if as_bool(res) is not True:
        out["missing"].append(f"sampled_replay.result true (got {res!r}{(': ' + str(sr.get('why'))[:120]) if sr.get('why') else ''})")
    # [manbind MAN-06 M-0543 layout] population {vus|n, source, by_family, not_evaluable}, sample {picked, evaluated, equal, mismatch_n,
    # seed}, strata {key: counts}, strata_keys {mechanisms, requests, ranks, steps, phases, lifetime}, grade complete|partial, mismatch[].
    pop = sr.get("population")
    pop_n = (pop.get("n") if pop.get("n") is not None else (pop.get("vus") if pop.get("vus") is not None else (pop.get("count") if pop.get("count") is not None else pop.get("total")))) if isinstance(pop, dict) else pop
    if not pop_n:
        out["missing"].append("sampled_replay.population {n|vus[, digest, source]} (the VU population sampled from — the binding map)")
    samp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    n = sr.get("n") if sr.get("n") is not None else (sr.get("sampled") if sr.get("sampled") is not None else (samp.get("evaluated") if samp.get("evaluated") is not None else samp.get("picked")))
    try:
        n_ok = int(n) > 0
    except Exception:  # noqa: BLE001
        n_ok = False
    if not n_ok:
        out["missing"].append("sampled_replay.n | sample.evaluated (> 0 VUs replayed)")
    if samp:
        ev, eq, mmn = samp.get("evaluated"), samp.get("equal"), samp.get("mismatch_n")
        if isinstance(ev, int) and isinstance(eq, int) and eq != ev:
            out["missing"].append(f"sampled_replay.sample.equal == evaluated (got {eq} of {ev} equal)")
        if isinstance(mmn, int) and mmn > 0:
            out["missing"].append(f"sampled_replay.sample.mismatch_n 0 (got {mmn})")
        if isinstance(samp.get("picked"), int) and isinstance(ev, int) and ev < samp["picked"]:
            out["missing"].append(f"sampled_replay.sample.evaluated == picked (got {ev} of {samp['picked']} evaluated)")
    mm_l = as_list(sr.get("mismatch")) + as_list(sr.get("mismatches"))
    if mm_l:
        out["missing"].append(f"sampled_replay.mismatch empty (got {len(mm_l)}: {json.dumps(mm_l[0], default=str)[:120]})")
    gr = sr.get("grade")
    if gr is not None and str(gr).lower() != "complete":
        out["missing"].append(f"sampled_replay.grade complete (got {gr!r}{(': ' + str(sr.get('why'))[:120]) if sr.get('why') else ''})")
    strata = sr.get("strata")
    if not (isinstance(strata, dict) and strata) and not (isinstance(strata, list) and strata):
        out["missing"].append("sampled_replay.strata {mechanisms, requests, ranks, steps early/late, lifetime boundaries}")
    sk = sr.get("strata_keys") if isinstance(sr.get("strata_keys"), dict) else None
    if sk is not None:
        for need, why in (("ranks", "per-rank strata (omitted-rank negative's domain)"), ("lifetime", "window-recycle lifetime strata (fresh vs recycled window — the late-read domain)")):
            if not sk.get(need):
                out["missing"].append(f"sampled_replay.strata_keys.{need} — {why} (rev-cb M-0411)")
    seed = sr.get("seed") if sr.get("seed") is not None else samp.get("seed")
    if seed is None:
        out["missing"].append("sampled_replay.seed | sample.seed")
    if not link:
        out["missing"].append("boundary_linkage {result, method} (ordered boundary linkage — the per-(request, engine step) token gate, F-cb-23; prescribed inputs / designated outputs tied to their interfaces; required at EVERY B — at B=1 it ties the committed input_token_ids to the DECLARED prompt of the workload manifest, rev-cb M-0411 O1)")
    elif as_bool(link.get("result")) is not True:
        out["missing"].append(f"boundary_linkage.result true (got {link.get('result')!r})")
    if not neg:
        out["missing"].append("targeted negatives on record (equal-shape request swap, wrong output member, omitted rank, post-overwrite read — {suite, results|artifact}, reusable with regression)")
    elif isinstance(neg, dict) and (neg.get("undetected") or neg.get("no_verdict") or as_bool(neg.get("result")) is False):
        und = as_list(neg.get("undetected")); nov = as_list(neg.get("no_verdict"))
        parts = []
        if und:
            parts.append("NOT DETECTED (the attacked run's commit verdict passed — a negative the checker does not catch bounds what this C2 qualifies): " + ", ".join(map(str, und))[:120])
        if nov:
            parts.append("attack ran but left no verdict.json (crashed / rc≠0 is not a detection): " + ", ".join(map(str, nov))[:120])
        if not parts:
            parts.append(str(neg.get("why") or "negatives.result false")[:120])
        out["missing"].append("targeted negatives — " + "; ".join(parts))
    unrep = sr.get("unreplayable")
    if isinstance(unrep, list) and unrep:
        names = ", ".join(str((u.get("stratum") or u.get("family") or u.get("name")) if isinstance(u, dict) else u)[:40] + (f" ×{u.get('n')}" if isinstance(u, dict) and u.get("n") is not None else "") for u in unrep[:8])
        out["footnotes"].append(f"C2 sampled replay: unreplayable strata (no registered relation) named on record — {names}{' …' if len(unrep) > 8 else ''}; they bound what this C2 qualifies (rev-cb M-0370 (3)(d))")
    if not out["missing"]:
        out["status"] = "PASS"
        strata_txt = ", ".join(f"{k}:{len(v) if isinstance(v, (list, dict)) else v}" for k, v in (strata.items() if isinstance(strata, dict) else enumerate(strata))) if strata else "?"
        if isinstance(strata, dict) and sk:
            strata_txt = f"{len(strata)} strata over " + ", ".join(f"{k}:{len(v) if isinstance(v, (list, dict)) else v}" for k, v in sk.items() if k in ("mechanisms", "requests", "ranks", "phases", "modules") and v)
            if isinstance(sk.get("lifetime"), dict) and isinstance(sk["lifetime"].get("values"), dict):
                strata_txt += ", lifetime " + "/".join(sk["lifetime"]["values"])
        neg_txt = ""
        if isinstance(neg, dict) and neg.get("suite"):
            neg_txt = f" + {len(as_list(neg.get('suite')))} targeted negatives detected ({', '.join(map(str, as_list(neg.get('suite'))))[:80]})"
        checked = (link or {}).get("checked") or ((f"{(link or {}).get('linked')}/{(link or {}).get('pairs')} pairs linked") if (link or {}).get("pairs") is not None else "")
        out["basis"] = (f"sampled exact VU replay from committed values ({n} of {pop_n} VUs, {strata_txt}, seed {seed}) + ordered boundary linkage "
                        f"({str((link or {}).get('method') or 'per-(request, engine step) token gate')[:60]}, {checked} checked){neg_txt} — DECISIONS-2 §4 C2 of record (form A)")
    return out


def grade_c2_components(vc: dict, wlf: dict, independent_root_ref: bool | None, evdir: str = "", row_id: str = "") -> dict:
    """[rev-cb M-0301 R2 / manbind M-0340 (2)] value_correspondence.components = {oracle_compare | root_compare | replay}, each graded by its
    own rule: oracle ✅ = result true AND complete (at B≥2 attribution.complete + result not false = ◐ 'oracle-compare partial');
    root_compare ✅ = result true AND an INDEPENDENT reference (COORD M-0286; at B≥2 never sufficient alone, M-0216); replay alone = ◐
    (M-0111).  A false component is ❌ with ITS OWN cause (R1); a foreign oracle (workload_check false) is ❌ 'foreign oracle' (R3).
    Returns {"fail": [..reasons..], "partial": [..reasons..], "pass_components": [..], "graded": {name: grade}}."""
    comps = vc.get("components") if isinstance(vc.get("components"), dict) else {}
    out = {"fail": [], "partial": [], "pass_components": [], "graded": {}}
    B = int(wlf.get("B") or 1)
    for name, c in comps.items():
        if not isinstance(c, dict) or name in C2_AUX_COMPONENTS:
            continue                                    # sampled_replay / linkage / negatives: graded by grade_sampled_replay (DECISIONS-2 §4)
        res = c.get("result")
        why = str(c.get("why") or "")[:200]
        if name == "oracle_compare":
            wc = c.get("workload_check") if isinstance(c.get("workload_check"), dict) else {}
            if wc and (as_bool(wc.get("ok")) is False or as_bool(wc.get("result")) is False) or str(c.get("grade") or "") == "foreign-oracle":
                out["graded"][name] = "FAIL(foreign-oracle)"
                out["fail"].append(f"oracle_compare: foreign oracle — the Match record is not this workload's ({str(wc.get('why') or why)[:160]}) (M-0301 R3)")
                continue
            if as_bool(res) is False:
                m07 = man07_bias_reference_defect(evdir, row_id, c, why) if row_id else None
                if m07:
                    out["graded"][name] = "PARTIAL(MAN-07 reference defect)"
                    out["partial"].append(m07)
                    out["checker_defect"] = "MAN-07"
                    continue
                out["graded"][name] = "FAIL"
                out["fail"].append(f"oracle_compare: {why or 'per-identity value mismatch'} (compared {c.get('compared')}, mismatch {c.get('mismatch_n')})")
                continue
            att = c.get("attribution") if isinstance(c.get("attribution"), dict) else {}
            okb, whyb = oracle_form_b(c, B)
            if okb:
                out["graded"][name] = "PASS"
                out["pass_components"].append(whyb)
            elif as_bool(res) is True:
                out["graded"][name] = "PARTIAL"
                out["partial"].append(f"oracle-compare result true but not the form of record (M-0481 (B)): {whyb[:200]}")
            elif B >= 2 and as_bool(att.get("complete")) is True and as_bool(res) is not False:
                out["graded"][name] = "PARTIAL"
                out["partial"].append(f"oracle-compare partial at B={B}: attribution complete ({att.get('pairs')} step/request pairs) but the value compare is not complete ({str(c.get('grade') or why or 'steps not snapshotted: ' + str(c.get('steps_not_snapshotted')))[:160]}) — M-0301 R2")
            else:
                out["graded"][name] = "PARTIAL"
                out["partial"].append(f"oracle-compare not complete: result {res!r}, complete {c.get('complete')!r}, attribution.complete {att.get('complete')!r} ({why[:120] or 'no cause recorded'})")
        elif name == "root_compare":
            if as_bool(res) is False:
                out["graded"][name] = "FAIL"
                out["fail"].append(f"root_compare: {why or 'run root != expected root'}")
                continue
            if as_bool(res) is not True:
                out["graded"][name] = "PARTIAL"
                out["partial"].append(f"root-compare did not run to a result ({res!r})")
            elif B >= 2:
                out["graded"][name] = "PARTIAL(B>=2)"
                out["partial"].append(f"root-compare at B={B} cannot see a request-attribution swap (rev-cb M-0216 F-cb-23) — not sufficient alone; oracle-compare per identity is the B≥2 C2")
            elif independent_root_ref is True:
                out["graded"][name] = "PASS"
                out["pass_components"].append("root-compare vs an independent pinned root")
            else:
                out["graded"][name] = "PARTIAL(not independent)"
                out["partial"].append("root-compare reference not independent (value_correspondence.references[] names no other-tree/pod pinned root — COORD M-0286)")
        elif name == "replay":
            if as_bool(res) is False:
                out["graded"][name] = "FAIL"
                out["fail"].append(f"replay: {why or 'replay root != committed root'}")
            else:
                out["graded"][name] = "evidence-only"
        else:
            out["graded"][name] = "unknown"
            out["partial"].append(f"C2 component {name!r} is not one of oracle_compare / root_compare / replay — not graded")
    return out


BIASED_LINEAR_FAMILIES = ("qwen25",)     # rev-cb M-0551: Qwen2.5 has qkv bias; Llama/Mistral/Phi-3/SmolLM2/Qwen3/OLMoE/Gemma have no linear bias


C2_NOT_EVALUABLE_CLASS_RE = re.compile(r"no (derived_rows )?evaluator|not evaluable|fed by [^;]*plane|plane mapping not in this replay|state-linked|no registered (relation|evaluator)|structure / interior", re.I)
C2_NOT_EVALUABLE_FAULT_RE = re.compile(r"error|fault|exception|traceback|not inside one committed member|file too short|CUDA|initiali[sz]ation|timeout|killed|OOM|out of memory", re.I)


def c2_either_form_holds(evdir: str, why: str, wlf: dict) -> dict | None:
    """[COORD DECISION M-0721 (1)] C2 = [form (A) complete AND 0 mismatch] OR [form (B) complete (0 mismatch, attribution complete at B≥2) AND
    form (A) 0 MISMATCH with every unevaluated stratum NAMED (family/module/reason)].  Applied to a verdict that FAILED CLOSED only on
    'sampled replay partial / strata not evaluated' (nothing else named).  Returns {form, basis, footnotes} or None."""
    if not why or not re.search(r"sampled replay partial|strata not evaluated|not[- ]evaluated|not established", why, re.I) or C2_MEASURED_FAIL_RE.search(why):
        return None
    aux = c2_aux_components_from(evdir)
    vc = value_correspondence_from(evdir) or {}
    comps = vc.get("components") if isinstance(vc.get("components"), dict) else {}
    sr = aux.get("sampled_replay") if isinstance(aux.get("sampled_replay"), dict) else (comps.get("sampled_replay") if isinstance(comps.get("sampled_replay"), dict) else None)
    bl = aux.get("boundary_linkage") if isinstance(aux.get("boundary_linkage"), dict) else (comps.get("boundary_linkage") if isinstance(comps.get("boundary_linkage"), dict) else None)
    oc = comps.get("oracle_compare") if isinstance(comps.get("oracle_compare"), dict) else (vc.get("oracle_compare") if isinstance(vc.get("oracle_compare"), dict) else None)
    if not isinstance(sr, dict) or not isinstance(oc, dict):
        return None
    samp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    mm = samp.get("mismatch_n") if samp.get("mismatch_n") is not None else len(as_list(sr.get("mismatch")) + as_list(sr.get("mismatches")))
    if not isinstance(mm, int) or mm != 0 or not isinstance(samp.get("evaluated"), int) or samp.get("evaluated") <= 0:
        return None
    un = sr.get("strata_unevaluated") if isinstance(sr.get("strata_unevaluated"), list) else []
    ne = as_list(sr.get("not_evaluated"))
    n_un = len(un) if un else (samp.get("not_evaluated_n") if isinstance(samp.get("not_evaluated_n"), int) else len(ne))
    # [rev-cb REVIEW M-0740 G1/G2 of the same rule] every unevaluated stratum NAMED = (G1) the strata_unevaluated list names EVERY stratum
    # (family|request|rank|lifetime|module — not a count; not_evaluated[] is capped at 12 entries by the writer, so it cannot establish naming
    # alone) AND (G2) every reason on the record is of the NO-EVALUATOR CLASS named by COORD M-0721 (1) — 'no evaluator' / 'not evaluable' /
    # 'fed by … plane(s)' / 'row -> plane mapping not in this replay' / 'state-linked'; an evaluator FAULT ('evaluator error', CUDA init,
    # 'file too short' JIT race) or a MAPPING FAULT ('arg slice … is not inside one committed member', M-0715's) is a run that did not evaluate,
    # never a named gap — the override does not fire and the record stays ◐ (c2_gate_incomplete) / ❌ as graded.
    m_cnt = re.search(r"(\d+) strata not evaluated", str(sr.get("why") or "") + " " + (why or ""))
    n_declared = int(m_cnt.group(1)) if m_cnt else None
    if n_declared is not None and n_declared > n_un:
        n_un = n_declared
    if n_un:
        if not un or len(un) < n_un:
            return None
        reasons = [str(e.get("why") or e.get("reason") or "") for e in ne if isinstance(e, dict)]
        reasons += [str(e) for e in ne if isinstance(e, str)]
        pop_ = sr.get("population") if isinstance(sr.get("population"), dict) else {}
        for tbl in (pop_.get("not_evaluable_by_family"), pop_.get("not_evaluable"), sr.get("unevaluated_reasons"), sr.get("strata_unevaluated_reasons")):
            if isinstance(tbl, dict):
                reasons += [str(k) for k in tbl.keys()]
        strata_tbl = sr.get("strata") if isinstance(sr.get("strata"), dict) else {}
        for k in un:
            e = strata_tbl.get(k) if isinstance(strata_tbl.get(k), dict) else None
            if e and (e.get("why") or e.get("why_class") or e.get("reason")):
                reasons.append(str(e.get("why") or e.get("why_class") or e.get("reason")))
        reasons = [r for r in reasons if r]
        if not reasons:
            return None
        if any(C2_NOT_EVALUABLE_FAULT_RE.search(r) and not C2_NOT_EVALUABLE_CLASS_RE.search(r) for r in reasons):
            return None                                          # a fault is a run that did not evaluate, not a named gap
        # [rev M-0008 / F-r16-01 (carried from rev-cb M-0740 G3)] the gap strata must lie in (B)'s compared families: VUs fed by MoE
        # block-stream planes (moe/, fa2.m1/, fa3.m1/ — NO_ORACLE_FAMILY_PREFIXES) have no oracle value, so a complete (B) does not cover them
        if any(POP_PLANE_FED_RE.search(r) for r in reasons):
            return None
        # the not_evaluated[] entries are the per-stratum reasons of record for the UNEVALUATED strata: every one must be of the class
        ne_reasons = [str(e.get("why") or e.get("reason") or "") for e in ne if isinstance(e, dict)]
        if ne_reasons and not all(C2_NOT_EVALUABLE_CLASS_RE.search(r) for r in ne_reasons):
            return None
        if not ne_reasons and not any(C2_NOT_EVALUABLE_CLASS_RE.search(r) for r in reasons):
            return None
    if bl is not None and as_bool(bl.get("result")) is not True:
        return None
    okb, whyb = oracle_form_b(oc, int(wlf.get("B") or 1))
    if not okb:
        return None
    fam_txt = ""
    pop = sr.get("population") if isinstance(sr.get("population"), dict) else {}
    nebf = pop.get("not_evaluable_by_family") if isinstance(pop.get("not_evaluable_by_family"), dict) else {}
    if nebf:
        fam_txt = "; not evaluable by family: " + ", ".join(f"{k} ×{v}" for k, v in list(nebf.items())[:6]) + ("…" if len(nebf) > 6 else "")
    form = f"oracle-compare; replay partial ({n_un} strata not evaluable, named)"
    basis = [whyb, f"form (A) on the same run: {samp.get('equal')}/{samp.get('evaluated')} sampled VUs equal, 0 mismatch; {n_un} strata not evaluable, every one named on the record"
             + (f" (first: {str((ne[0] or {}).get('why') if isinstance(ne[0], dict) else ne[0])[:140]})" if ne else "") + fam_txt
             + (f"; boundary linkage {bl.get('linked')}/{bl.get('pairs')}" if isinstance(bl, dict) else "")]
    return {"form": form, "basis": basis, "footnotes": []}


POP_PLANE_FED_RE = re.compile(r"moe_block_stream|block[- ]stream|fed by [^;]*plane|plane[- ]fed|mlp\.experts|moe/", re.I)
POP_INTERIOR_RE = re.compile(r"interior producer|interior[- ]composed|composition not possible|no evaluator for the composed|BiasAdd_v1 over|composed", re.I)


def population_named_gaps(sr: dict | None) -> dict:
    """[rev BLOCKER M-0016 F-r16-02] the MAN-06f/06g exclusions in sampled_replay.population.not_evaluable — {reason: n} keyed by the replay's
    own by-name reasons — split into the two classes commit_verdict.why_class names: plane-fed (MoE block-stream planes) and interior-composed
    (a VU fed by an interior producer without an evaluator).  Anything else (RoPE NOT_YET, 'no registered evaluator', state-linked) is the
    replay's stated applicability domain (R15 standard) and is not counted here.  Fail-closed on shape: a non-dict / non-count is ignored."""
    out = {"n": 0, "plane_fed": 0, "interior": 0, "first_plane": "", "first_interior": "", "reasons": {}}
    if not isinstance(sr, dict):
        return out
    pop = sr.get("population") if isinstance(sr.get("population"), dict) else {}
    tbls = [pop.get("not_evaluable")] + [sr.get("population_not_evaluable")]
    for tbl in tbls:
        if not isinstance(tbl, dict):
            continue
        for k, v in tbl.items():
            try:
                n = int(v) if not isinstance(v, (list, dict)) else len(v)
            except Exception:  # noqa: BLE001
                continue
            if n <= 0:
                continue
            key = str(k)
            if POP_PLANE_FED_RE.search(key):
                out["plane_fed"] += n
                out["first_plane"] = out["first_plane"] or key
            elif POP_INTERIOR_RE.search(key):
                out["interior"] += n
                out["first_interior"] = out["first_interior"] or key
            else:
                continue
            out["reasons"][key] = n
            out["n"] += n
    return out


C2_GATE_INCOMPLETE_RE = re.compile(r"not established|not[- ]evaluated|strata not evaluated|not value-checked|sampled replay partial|replay partial|attribution not", re.I)
C2_MEASURED_FAIL_RE = re.compile(r"\bmismatch|differ|identities differ|!= committed|coverage|missing identit|omitted|openings? fail|root != |bytes differ|F_V\(", re.I)


def c2_gate_incomplete(evdir: str, row_id: str, why: str) -> dict | None:
    """[report convention of record — brief: 'partial evidence renders ◐ with the missing field named'; COORD 18:09Z: Commit ❌ = an EXECUTED
    coverage/value failure] a commit verdict that FAILED CLOSED because the C2 of record could not be ESTABLISHED (sampled replay partial /
    strata not evaluated — moe BLOCKER M-0715 fused-MoE planes; attribution not value-checked because the Match did not snapshot the steps —
    hopper H100 B8 M-0696) with ZERO mismatches measured anywhere on the record is INCOMPLETE EVIDENCE, not a measured failure: Commit ◐ with the
    gap named, never ❌ (no committed value was measured wrong) and never ✅ (the gate did not pass).  Narrow: the first_fail_reason must name
    the incompleteness and must not name a measured mismatch; every C2 block on the record must carry 0 mismatches (oracle-compare result not
    false, sampled replay mismatch_n 0); a record with any measured difference stays ❌."""
    # [hopper BLOCKER M-0723 / exec BLOCKER M-0727 / exec ACK M-0729 — int-6 4ef1198b6] MAN-09 at B≥2 mixed shape: the weights_pin record was
    # derived from the row's DEFAULT wrapper shape, which is not a component of GP-01's workload Program → 'record built for Program X != Program
    # of record' by name, 0 fields mismatched.  A pin that is not OF RECORD for this Program is an ABSENT pin (M-0614 (1): ◐ 'weights unpinned'),
    # not a measured weights defect; a pin that names differing/missing weights stays ❌.
    if why and re.search(r"weights_pin: record built for Program", why) and not re.search(r"differ from the checkpoint|mismatched [1-9]|root differs|served weight", why, re.I):
        return {"kind": "weights pin not of record (MAN-09 B≥2 component pick)", "footnotes": ["weights_pin record on file compares 0 mismatched fields but was built for a Program digest that is not in this row's Program of record (default wrapper shape, not a GP-01 component)"],
                "reason": (f"{WEIGHTS_UNPINNED} — the MAN-09 weights_pin record was derived from the row's DEFAULT wrapper shape, which is not a component of this mixed-shape "
                           f"workload's GP-01 Program, so it is not a pin OF RECORD for this Program (checker: {why[:200]}); 0 weight fields differ. hopper BLOCKER M-0723 / exec M-0727; "
                           f"fix = the B>1 component pick in int-6 4ef1198b6 (exec ACK M-0729) + the Commit-only rerun on int-6. ◐ per COORD M-0614 (1) (no pin of record), not ❌ (no weight measured wrong)")}
    # [moe TIP M-0007 / R16 brief 4 'fails closed on the missing pre-MAN-09 pin — stays partial until pin'] the checker FAILED CLOSED because
    # the MAN-09 compare was NOT RUN (commit_delta received neither --weights-of-record nor --checkpoint-dir: the int-4b row_pod.sh, 0
    # weights_of_record refs, kept running after the tree swap to int-6).  No weight was measured wrong: an ABSENT pin is ◐ 'weights unpinned'
    # (COORD M-0614 (1)), never ❌; a pin that names differing weights stays ❌.
    if why and re.search(r"weights unpinned \(MAN-09\)", why) and re.search(r"not run|no --weights-of-record|no --checkpoint-dir|not compared", why, re.I) \
            and not re.search(r"differ from the checkpoint|mismatched [1-9]|root differs|served weight\(s\) differ|[1-9]\d* (served )?weights? differ", why, re.I):
        return {"kind": "weights pin not run (MAN-09 absent on this attempt)",
                "footnotes": ["commit verdict FAIL on record is the checker failing closed on an ABSENT MAN-09 compare (not run), 0 weight fields measured"],
                "reason": (f"{WEIGHTS_UNPINNED} — this record: the MAN-09 compare was NOT RUN (checker: {why[:160]}); an absent pin is ◐ per COORD M-0614 (1), not ❌ "
                           f"(no weight measured wrong). Fix = a Commit-only rerun on a tree whose row_pod.sh passes --weights-of-record / --checkpoint-dir (int-6 4ef1198b6+, r16-int-1) — moe TIP M-0007")}
    if not why or not C2_GATE_INCOMPLETE_RE.search(why) or C2_MEASURED_FAIL_RE.search(why):
        return None
    aux = c2_aux_components_from(evdir)
    vc = value_correspondence_from(evdir) or {}
    comps = vc.get("components") if isinstance(vc.get("components"), dict) else {}
    sr = aux.get("sampled_replay") if isinstance(aux.get("sampled_replay"), dict) else (comps.get("sampled_replay") if isinstance(comps.get("sampled_replay"), dict) else None)
    oc = comps.get("oracle_compare") if isinstance(comps.get("oracle_compare"), dict) else (vc.get("oracle_compare") if isinstance(vc.get("oracle_compare"), dict) else None)
    fns = []
    if isinstance(sr, dict):
        samp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
        mm = samp.get("mismatch_n") if samp.get("mismatch_n") is not None else len(as_list(sr.get("mismatch")) + as_list(sr.get("mismatches")))
        if isinstance(mm, int) and mm > 0:
            return None                                  # a measured replay mismatch is a value defect — ❌ stands
        un = sr.get("strata_unevaluated") if isinstance(sr.get("strata_unevaluated"), list) else None
        ne = as_list(sr.get("not_evaluated"))
        fns.append(f"sampled replay on the record: {samp.get('equal')}/{samp.get('evaluated')} evaluated VUs equal, 0 mismatch; "
                   f"{len(un) if un is not None else (samp.get('not_evaluated_n') or len(ne))} strata / VUs not evaluated"
                   + (f" (first: {str((ne[0] or {}).get('why') if isinstance(ne[0], dict) else ne[0])[:160]})" if ne else ""))
    if isinstance(oc, dict):
        if as_bool(oc.get("result")) is False or (isinstance(oc.get("mismatch_n"), int) and oc["mismatch_n"] > 0):
            return None
        if as_bool(oc.get("result")) is True:
            fns.append(f"oracle-compare on the same run: {oc.get('equal') or oc.get('compared')}/{oc.get('compared')} equal, 0 mismatch, grade {oc.get('grade')} (form (B) evidence; not the basis — the verdict is FAIL)")
    fam = str(row_id).split("__")[0]
    moe = fam.startswith(("olmoe", "qwen3-30b", "mixtral", "deepseek"))
    bias_strata = bool(re.search(r"BiasAdd_v1", why)) or (isinstance(sr, dict) and any("BiasAdd" in str(x) for x in (sr.get("strata_unevaluated") or [])[:4]))
    if re.search(r"strata not evaluated|not[- ]evaluated|replay partial|not established", why, re.I) and moe:
        kind, head = "MAN-06 MoE strata", "MAN-06 MoE strata pending — "
        tail = (" (moe BLOCKER M-0715: the fused-MoE-fed rows — SiluMul under mlp.experts, RMSNormFusedCuda on the block output — are not evaluable by MAN-06 until manbind's plane mapping / "
                "named-not-evaluable (MAN-06f 440570589, int-6 4ef1198b6); the 'arg slice … not inside one committed member' reason on this record is a row→member MAPPING gap, not a named "
                "no-evaluator gap (rev-cb M-0740 G2), so COORD M-0721 (1) does not lift it)")
    elif re.search(r"strata not evaluated|not[- ]evaluated|replay partial|not established", why, re.I) and bias_strata:
        kind, head = "MAN-07 BiasAdd strata", "MAN-07 BiasAdd strata pending — "
        tail = (" (hopper BLOCKER M-0748 / exec M-0749 / rev-cb M-0751: under MAN-07 the biased linear's committed member is the POST-bias output, so the BiasAdd_v1 evaluator has no "
                "committed input — every qkv_proj BiasAdd stratum counts 'not evaluated' (row→member mapping gap, not a value fault: oracle-compare complete on the same run); fix = the composed "
                "evaluator F(x,W,b) or named-not-evaluable BY NAME, int-7)")
    elif re.search(r"attribution not|not value-checked", why, re.I):
        kind, head = "attribution not value-checked", "C2 attribution not value-checked — "
        tail = " (the Match did not snapshot the steps the attribution gate needs: re-run Match with MATCH_SNAP_STEPS=all + the Commit-only rerun on the release tree)"
    else:
        kind, head, tail = "C2 not established", "C2 of record not established — ", ""
    return {"kind": kind, "footnotes": fns,
            "reason": (head + f"the checker's C2 gate FAILED CLOSED on INCOMPLETE evaluation with 0 mismatches measured anywhere on the record: {why[:220]}{tail}. "
                       f"◐ = partial evidence with the gap named (brief), not ❌ (no committed value measured wrong; COORD 18:09Z ❌ = executed failure) and not ✅ (the gate did not pass)")}


# [rev-pm RESULT M-0758 02:05Z on compiled 2b6573893 (= int-6 cb04e0cf1 S3 code)] the open finding that keeps a compiled row's Build at ◐ under COORD M-0721 (4)
# ('int-6 candidate accepted pending rev-pm's N1 re-check'); set to None when rev-pm posts the re-check CLOSED on the tree of record.
P_COMPILED_DATAFLOW_OPEN_FINDING: str | None = ("rev-pm M-0758 (F-pm-29 / BLOCKER M-0687 on 2b6573893, measured on the r15_0100 row with its split graph): exchanging the weight "
                                                "argument of fused_add_rms_norm_t_0 with that of fused_add_rms_norm_t_2 in both middle wrappers leaves program_digest AND manifest_digest "
                                                "unchanged (parse_wrapper skips every weight arg) — two P_compiled that route the same served weights to different norms share one identity; "
                                                "the MAN-09 pin says which weights were served, not which launch consumed which")
SOFTCAPPED_LOGITS_FAMILIES = ("gemma2",)   # dense M-0619: Gemma-2 applies final_logit_softcapping AFTER the lm_head Gemm (logits_processor/out is post-cap)


def gemma_softcap_reader_defect(evdir: str, row_id: str, sr: dict | None, why: str) -> str | None:
    """[dense BLOCKER M-0619 / FIX_READY M-0630 (35ec6d8ad, in manbind's int-5 list M-0643/M-0654) / RESULT M-0652] a sampled-replay FAIL on a
    SOFT-CAPPED-LOGITS model whose mismatches are ALL the lm_head Gemm evaluated against `logits_processor/out` is a READER defect of the MAN-07
    class (wrong producer read, one level up: the replay evaluates the pre-cap Gemm and compares it with the committed post-cap module output),
    not a value defect — Commit ◐ 'C2 pending replay producer fix', never ❌ (nobody measured the commitment wrong) and never ✅ (form (A) did not
    pass).  Narrow by construction: the family must carry soft-capped logits, the record's mismatch list must be non-empty and EVERY listed
    mismatch must be op_path logits_processor (member out/0), with the replay's other VUs equal; anything else stays ❌."""
    fam = str(row_id).split("__")[0]
    if not any(fam.startswith(f) for f in SOFTCAPPED_LOGITS_FAMILIES):
        return None
    if "sampled_replay" not in (why or "") and "logits_processor" not in (why or "") and not (isinstance(sr, dict) and as_bool(sr.get("result")) is False):
        return None
    if not isinstance(sr, dict):
        aux = c2_aux_components_from(evdir)
        sr = aux.get("sampled_replay") if isinstance(aux.get("sampled_replay"), dict) else None
    if not isinstance(sr, dict) or as_bool(sr.get("result")) is not False:
        return None
    mm = as_list(sr.get("mismatch")) + as_list(sr.get("mismatches"))
    if not mm:
        return None
    def _lp(m) -> bool:
        if not isinstance(m, dict):
            return False
        op = str(m.get("op_path") or (m.get("identity") or {}).get("op_path") if isinstance(m.get("identity"), dict) else m.get("op_path") or "")
        outs = m.get("outputs") if isinstance(m.get("outputs"), list) else []
        mem_ok = all(str(o.get("member")) in ("out", "0") for o in outs if isinstance(o, dict)) if outs else True
        return op.split("/")[0].endswith("logits_processor") and mem_ok
    if not all(_lp(m) for m in mm):
        return None
    samp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    return (f"C2 pending replay producer fix — the sampled-replay FAIL ({len(mm)} mismatches, all the lm_head Gemm_v1 evaluated against logits_processor/out; "
            f"{samp.get('equal')}/{samp.get('evaluated')} sampled VUs equal — the mismatches are exactly the logits strata; boundary linkage on record) is a checker READER defect on a soft-capped-logits model, not a "
            f"measured value defect: the replay evaluates the pre-cap Gemm and compares it with the committed post-cap module output (dense BLOCKER M-0619 = the MAN-07 class one level up; "
            f"fix dense 35ec6d8ad in manbind's int-5 list M-0643/M-0654). Decided by the Commit-only rerun on a tree carrying the fix; the FAIL record stays on the row as the negative fixture")


def man07_bias_reference_defect(evdir: str, row_id: str, oc: dict | None, why: str) -> str | None:
    """[manbind FIX_READY MAN-07 M-0588 / rev-cb M-0551 / hopper M-0592 / exec M-0528] an oracle-compare FAIL on a BIASED-LINEAR model whose
    differing identities are all `*.qkv_proj/0` is a CHECKER REFERENCE defect, not a value defect: the manifest of record named the pre-bias
    Gemm instance as the producer of member 0 while the collector commits the module's (post-bias) output.  Such a row is Commit ◐
    'C2 pending MAN-07 rerun' — the ❌ would assert the commitment is wrong, which nobody has measured; the ✅ is not available either.
    Narrow by construction: the model family must carry biased linears, the record's mismatch list must be non-empty and EVERY listed
    identity must be output_member '0' of an op_path ending in qkv_proj; anything else stays ❌."""
    fam = str(row_id).split("__")[0]
    if not any(fam.startswith(f) for f in BIASED_LINEAR_FAMILIES):
        return None
    if "oracle" not in (why or "").lower() and not (isinstance(oc, dict) and as_bool(oc.get("result")) is False):
        return None
    mm = None
    if isinstance(oc, dict):
        mm = oc.get("mismatch") if isinstance(oc.get("mismatch"), list) else (oc.get("mismatches") if isinstance(oc.get("mismatches"), list) else None)
    if mm is None:
        rp = first(evdir, "commit/runs.jsonl", "runs.jsonl")
        if rp:
            try:
                for l in open(rp, encoding="utf-8", errors="replace"):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    cm = r.get("commit") if isinstance(r.get("commit"), dict) else {}
                    o = cm.get("oracle_compare") if isinstance(cm.get("oracle_compare"), dict) else (r.get("oracle_compare") if isinstance(r.get("oracle_compare"), dict) else None)
                    if o and isinstance(o.get("mismatch"), list) and o["mismatch"]:
                        mm = o["mismatch"]
                        oc = oc or o
                        break
            except Exception:  # noqa: BLE001
                mm = None
    if not mm:
        return None
    def _qkv(m) -> bool:
        ident = m.get("identity") if isinstance(m, dict) and isinstance(m.get("identity"), dict) else (m if isinstance(m, dict) else {})
        op = str(ident.get("op_path") or (m.get("name") if isinstance(m, dict) else "") or "")
        mem = str(ident.get("output_member") if ident.get("output_member") is not None else (op.rsplit("/", 1)[1] if "/" in op else "0"))
        return op.split("/")[0].endswith("qkv_proj") and mem == "0"
    if not all(_qkv(m) for m in mm):
        return None
    n = (oc or {}).get("mismatch_n") if isinstance(oc, dict) else None
    n = n if isinstance(n, int) else len(mm)
    listed = len(mm)
    return (f"C2 pending MAN-07 rerun — the oracle-compare FAIL ({n} differing identities, {'all ' + str(listed) if listed >= n else f'{listed} listed of {n}, all listed'} on qkv_proj/0, "
            f"compared {(oc or {}).get('compared')}) is a checker REFERENCE defect on a biased linear, not a measured value defect: the manifest of record names the pre-bias Gemm "
            f"instance as the producer of qkv_proj/0 while the collector commits the module's post-bias output (exec M-0528 / rev-cb M-0551 / hopper M-0592; manbind FIX_READY MAN-07 M-0588 "
            f"= producer is the BiasAdd row). Decided by the Commit-only rerun on a MAN-07 tree; the FAIL record stays on the row as the negative fixture")


def phantom_coverage(evdir: str, why: str) -> str | None:
    """MAN-01 phantom identities: a coverage FAIL whose missing identities are all `element_range [0, 0]` rows of the shipped
    manifest.json (members the Program has no value for) is a manifest defect (MAN-02), not a coverage gap."""
    m = re.search(r"identity coverage: (\d+) of (\d+) required identities have no binding \(first: step \d+ ([\w./]+)/(\w+)", why)
    if not m:
        return None
    miss_n, exp_n = int(m.group(1)), int(m.group(2))
    mp = first(evdir, "manifest.json", "commit/manifest.json")
    zero: list = []
    src = "manifest.json"
    if mp:
        try:
            M = load(mp)
            rows_ = M.get("rows") or M.get("identities") or []
            zero = [r_ for r_ in rows_ if isinstance(r_, dict) and list(r_.get("element_range") or [1, 1])[:2] == [0, 0]]
        except Exception:
            zero = []
        if not zero or len(zero) != miss_n:
            return None
    else:
        # manifest not shipped (large rows): judge from the checker's own missing list in the binding record — every listed
        # missing identity must be zero-width and of one member
        for b in binding_records(evdir):
            ml = ((b.get("coverage") or {}).get("missing") if isinstance(b.get("coverage"), dict) else None) or []
            if ml:
                zero = [r_ for r_ in ml if isinstance(r_, dict) and list(r_.get("element_range") or [1, 1])[:2] == [0, 0]]
                if len(zero) != len(ml) or len({(str(r_.get("op_path")).split(".")[-1], r_.get("output_member")) for r_ in zero}) != 1:
                    return None
                src = f"checker missing list ({len(ml)} listed of {miss_n}; manifest.json not shipped)"
                break
        if not zero:
            return None
    members = sorted({f"{str(r_.get('op_path')).split('.')[-1]}/{r_.get('output_member')}" for r_ in zero})
    return (f"blocked: MAN-02 phantom identity — all {miss_n} missing identities are zero-width members ({', '.join(members[:3])}; `element_range [0, 0]`, spec {zero[0].get('spec') or 'BiasAdd'}; source: {src}) "
            f"the Program has no value for (MAN-01 defect, COORD 18:24Z); nonempty coverage {exp_n - miss_n}/{exp_n - miss_n} bound, C3 openings verified, C2 root-compare true — not a coverage gap, not a pass until the MAN-02 manifest rerun")


def derive_commit(evdir: str, wl: dict, wlf: dict, build: dict, row: dict, rev_hint: str | None = None) -> dict:
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": []}
    v = first(evdir, "commit_verdict.json", "commit/verdict.json")
    d = first(evdir, "commit_delta_summary.json", "commit/summary.json")
    s = first(evdir, "commit_summary.json")
    c = first(evdir, "coverage.json", "commit/coverage.json", "coverage_*.json", "commit/coverage_*.json", "binding_coverage*.json")
    if not (v or d):
        # R15 coherence (brief §9): a Commit stage that RAN in this attempt must not render `—` (not run).  The pod runner's
        # commit_summary.json / stages.txt carry the verdict even when the commit/ artifacts were not synced.
        S0 = load(s) if s else {}
        st_line = None
        stg = first(evdir, "stages.txt")
        if stg:
            try:
                st_line = next((ln for ln in open(stg, encoding="utf-8", errors="replace") if ln.startswith("commit ")), None)
            except Exception:
                st_line = None
        ran = bool(S0) or bool(st_line)
        if not ran:
            out["reasons"].append("no commit verdict / delta summary in the evidence directory")
            return out
        out["evidence"] = [os.path.relpath(p) for p in (s, stg) if p]
        out["artifact"] = os.path.relpath(s or stg)
        passed = as_bool(S0.get("pass")) if "pass" in S0 else (as_bool(S0.get("commit_pass")) if "commit_pass" in S0 else None)
        rc = S0.get("rc")
        if passed is None and st_line:
            passed = " PASS " in st_line
        if passed is None and rc is not None:
            passed = str(rc) in ("0", "None")
        why = str(S0.get("first_fail_reason") or S0.get("commit_fail_reasons") or (st_line or "").strip())
        aj = first(evdir, "attempt.json")
        if aj:
            try:
                rc_txt = ((load(aj).get("result") or {}).get("commit")) if isinstance(load(aj).get("result"), dict) else None
                if isinstance(rc_txt, str) and rc_txt.strip():
                    why = rc_txt.strip()
            except Exception:
                pass
        missing = "missing: commit/ verdict + delta summary (+ binding records) in the evidence directory — only the runner's commit_summary.json/stages.txt record the stage"
        if passed is False:
            if wlf["B"] >= 2 and re.search(r"identit\w* .{0,80}have no binding", why + json.dumps(S0, default=str)):
                out["status"] = "PARTIAL"
                out["reasons"].append(f"Commit ran and FAILED the fail-closed check (rc={rc}: {why[:220]}); without the commit/ binding records the F-rev-08 classification (B≥2 request-row map pending vs. a coverage FAIL) cannot be verified; {missing}")
            elif str(rc) == "124" or re.search(r"\btimeout\b|timed out", why, re.I):
                # [manbind TIP M-0797 on exec r15_0038 (Mistral-7B L40S B1, int-4)] rc=124 = the runner's COMMIT_TIMEOUT: the stage did not COMPLETE — no
                # verdict was written and nothing was measured wrong (the pairs that finished are on record).  An incomplete measurement is ◐ with the
                # cause named, not a FAIL of record (brief §9: a ❌ is a measured failure).  The green pairs on record are named beneath.
                pairs = []
                for f in sorted(glob.glob(os.path.join(evdir, "commit", "sampled_replay_p*.json"))):
                    try:
                        J = load(f); J = J.get("sampled_replay") if isinstance(J.get("sampled_replay"), dict) else J
                        smp = J.get("sample") if isinstance(J.get("sample"), dict) else {}
                        pairs.append(f"{os.path.basename(f)}: {J.get('grade')}, {smp.get('equal')}/{smp.get('evaluated')} equal, {smp.get('mismatch_n')} mismatch")
                    except Exception:  # noqa: BLE001
                        pass
                out["status"] = "PARTIAL"
                out["commit_timed_out"] = True
                out["reasons"].append(f"Commit did not complete — runner timeout (rc={rc}: {why[:200]}); no verdict written, nothing measured wrong; complete pair(s) on record: "
                                      + ("; ".join(pairs) if pairs else "none") + f". {missing}"
                                      + (" (manbind TIP M-0797: the int-4 tree carries the SERIAL replay evaluator — the forked pool of MAN-06b is in int-5/int-6; rerun the Commit stage on int-6)" if pairs else ""))
            else:
                out["status"] = "FAIL"
                out["reasons"].append(f"Commit ran and FAILED (rc={rc}: {why[:220]}); {missing}")
        elif passed is True:
            out["status"] = "PARTIAL"
            out["reasons"].append(f"Commit ran and the runner recorded PASS (rc={rc}) but {missing}; C1–C3 / retention / delta cannot be derived")
        else:
            out["status"] = "PARTIAL"
            out["reasons"].append(f"Commit stage recorded without a verdict (rc={rc}); {missing}")
        out["fail_closed"] = None
        out["errors"] = as_list(S0.get("commit_fail_reasons"))
        return out
    V = load(v) if v else {}
    D = load(d) if d else {}
    S = load(s) if s else {}
    for p in (v, d, s, c):
        if p:
            out["evidence"].append(os.path.relpath(p))
    out["artifact"] = os.path.relpath(v or d)
    out["committer"] = D.get("committer")
    out["layout"] = (D.get("declaration") or {}).get("layout") if isinstance(D.get("declaration"), dict) else None
    out["fail_closed"] = True if (V or D.get("commit_fail_reasons") is not None) else None
    out["expected_tensors"] = as_list(D.get("expected_tensors"))
    out["committed_tensors"] = as_list(D.get("committed_tensors"))
    out["errors"] = as_list(D.get("commit_fail_reasons"))
    out["required_classes"] = as_list(V.get("required_classes"))
    out["required_classes_source"] = V.get("required_classes_source")
    out["run_roots"] = as_list(V.get("run_roots"))
    out["runs_instrumented"] = V.get("runs_instrumented")
    out["runs_failed"] = V.get("runs_failed")
    out["backlog_max"] = D.get("backlog_max")
    out["tokens_equal"] = as_bool(D.get("tokens_equal"))
    out["pairs"] = D.get("pairs")
    wpath = D.get("workload")
    txt = json.dumps({"c": out["committer"], "decl": D.get("declaration"), "S": {k: S.get(k) for k in ("cmd", "committer", "env") if k in S}}, default=str)
    out["collector_variant"] = "windowed" if re.search(r"window|ring", txt, re.I) else "whole-block"
    m = re.search(r"retain[=: _]+(host|device|none|sample)", txt, re.I)
    out["retention"] = m.group(1).lower() if m else ("device" if out["collector_variant"] == "whole-block" and out["committer"] else None)
    if m is None and out["retention"] == "device":
        out["retention_source"] = "commit_delta default for the whole-block committer (no --retain recorded)"
    # gap 4 (rev): engine settings for the chunked-prefill / memory footnote, when the evidence carries them
    for src in (S, D):
        es = src.get("engine_settings") if isinstance(src, dict) else None
        if isinstance(es, dict):
            out["engine_settings"] = es
    if "engine_settings" not in out:
        runs = _runs(evdir)
        for r in runs:
            es = r.get("engine_settings") or (r.get("engine") if isinstance(r.get("engine"), dict) else None)
            if isinstance(es, dict):
                out["engine_settings"] = {k: es.get(k) for k in ("gpu_memory_utilization", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill") if k in es}
                break
            mm = re.search(r"max[-_]num[-_]batched[-_]tokens[= ](\d+)", json.dumps(r))
            if mm:
                out["engine_settings"] = {"max_num_batched_tokens": int(mm.group(1))}
                break
    # executed-check failures
    cpv = V.get("commit_pass") if V else D.get("commit_pass")
    cp = (all(as_bool(x) for x in cpv) if cpv else None) if isinstance(cpv, list) else as_bool(cpv)
    if not out["run_roots"] and D.get("tp_run_roots"):
        out["run_roots"] = as_list(D.get("tp_run_roots"))
        out["per_rank_roots"] = D.get("per_rank_roots")
        out["tp"] = D.get("tp")
    if isinstance(D.get("tokens_equal"), list):
        out["tokens_equal"] = all(as_bool(x) for x in D["tokens_equal"])
    if cp is False or (V and str(V.get("runs_failed", "0")) not in ("0", "None")):
        why = str(V.get("first_fail_reason") or out["errors"])
        if wlf["B"] >= 2 and "required identities have no binding" in why:
            bnds0 = [b for b in binding_records(evdir) if isinstance(b, dict)]
            if bnds0 and not any(int((b.get("population") or {}).get("step_rows_recorded") or 0) > 0 or b.get("request_rows") or b.get("b2_map") for b in bnds0):
                # F-rev-08: B>=2 on a tree without the request-row (B2) map — population + map pending, not a rejected attack
                out["status"] = "PARTIAL"
                out["reasons"].append("B≥2 binding map not in this tree (step_rows_recorded 0 → per-request rows unbindable): population + map pending (F-rev-08), not a coverage rejection — checker said: " + why[:160])
                out["coverage_pending_b2_map"] = True
                return out
        ph = phantom_coverage(evdir, why)
        if ph:
            # COORD 18:24Z: every missing identity is a zero-width MAN-01 phantom member → Commit ◐ blocked (MAN-02), not ❌
            out["status"] = "PARTIAL"
            out["reasons"].append(ph)
            out["coverage_blocked_man02"] = True
            return out
        m07 = man07_bias_reference_defect(evdir, str(row.get("row_id") or ""), None, why)
        if m07:
            out["status"] = "PARTIAL"
            out["reasons"].append(m07)
            out["c2_checker_defect"] = "MAN-07"
            out.setdefault("footnotes", []).append("commit verdict FAIL on record (checker): " + why[:200])
            return out
        gsc = gemma_softcap_reader_defect(evdir, str(row.get("row_id") or ""), None, why)
        if gsc:
            out["status"] = "PARTIAL"
            out["reasons"].append(gsc)
            out["c2_checker_defect"] = "M-0619 (softcap reader)"
            out.setdefault("footnotes", []).append("commit verdict FAIL on record (checker): " + why[:200])
            return out
        either = c2_either_form_holds(evdir, why, wlf)
        if either:
            # [COORD DECISION M-0721 (1)] the checker's verdict rule was over-strict by one case: C2 of record HOLDS when form (B) is complete
            # (0 mismatch, attribution complete) and form (A) has 0 MISMATCH with every unevaluated stratum NAMED — the FAIL verdict on record is
            # the over-strict rule, not a measured failure; the cell is ✅ 'C2: oracle-compare; replay partial (N strata not evaluable, named)'
            out["c2_form"] = either["form"]
            out["c2_basis"] = either["basis"]
            out["verdict_overridden_by"] = "COORD M-0721 (1)"
            out.setdefault("footnotes", []).append("commit verdict FAIL on record (commit_pass false: 'sampled replay partial') — over-strict by one case per COORD M-0721 (1); the C2 of record is form (B) + form (A) 0 mismatch with the unevaluated strata named: " + why[:160])
            for fn in either.get("footnotes") or []:
                out["footnotes"].append(fn)
            # fall through: the remaining C1/C3 gates below still apply (coverage, openings, tokens, retention)
        else:
            gate = c2_gate_incomplete(evdir, str(row.get("row_id") or ""), why)
            if gate:
                out["status"] = "PARTIAL"
                out["reasons"].append(gate["reason"])
                out["c2_gate_incomplete"] = gate["kind"]
                out.setdefault("footnotes", []).append("commit verdict FAIL on record (fail-closed C2 gate, 0 mismatches measured): " + why[:200])
                for fn in gate.get("footnotes") or []:
                    out["footnotes"].append(fn)
                return out
            out["status"] = "FAIL"
            out["reasons"].append(f"fail-closed commit verdict FAIL: {why}")
            return out
    if out["expected_tensors"] and out["committed_tensors"] and [int(x) for x in out["expected_tensors"]] != [int(x) for x in out["committed_tensors"]]:
        out["status"] = "FAIL"
        out["reasons"].append(f"committed {out['committed_tensors']} != expected {out['expected_tensors']}")
        return out
    if out["tokens_equal"] is False:
        out["status"] = "FAIL"
        out["reasons"].append("tokens differ between control and instrumented runs")
        return out
    if out["run_roots"] and len(set(out["run_roots"])) != 1:
        out["status"] = "FAIL"
        out["reasons"].append(f"{len(set(out['run_roots']))} distinct run roots across the instrumented runs")
        return out
    # same-attempt: the committed workload is the row's workload
    wl_row = f"workloads/{row.get('row_id')}.json"
    if wpath and os.path.basename(str(wpath)) != os.path.basename(wl_row):
        out["status"] = "FAIL"
        out["reasons"].append(f"commit ran workload {wpath} (same-attempt rule: row workload {wl_row})")
        return out
    missing = []
    # BIND-01 (lane bind, r14): commit_delta records `commit.binding` per instrumented run (runs.jsonl / commit_runs_instrumented.json)
    # and commit_verdict gates on binding_check.  Read it from the runs when no standalone coverage.json exists.
    if not c:
        bnds = binding_records(evdir)
        if bnds:
            out["evidence"].append("commit.binding of %d instrumented run(s)" % len(bnds))
            bad = [b for b in bnds if b.get("error") or not (b.get("coverage") or {}).get("ok") or not (b.get("context") or {}).get("ok")]
            po = [b.get("openings_population") or {} for b in bnds]
            unbound = sum(int(p.get("unbound_n") or 0) for p in po)
            opened = sum(int(p.get("opened") or 0) for p in po)
            verified = sum(int(p.get("verified") or 0) for p in po)
            out["coverage_checker"] = f"verity_capture.commit.binding (BIND-01) rules {bnds[0].get('rules_version')}"
            out["manifest_digest"] = (bnds[0].get("population") or {}).get("digest")
            out["binding_map_digest"] = bnds[0].get("map_digest")
            out["binding_map_digests"] = sorted({str(b.get("map_digest")) for b in bnds})
            out["expected_population_n"] = (bnds[0].get("population") or {}).get("n")
            pop_src = str((bnds[0].get("population") or {}).get("source") or "")
            out["manifest_source"] = pop_src or None
            if pop_src and not re.search(r"required[- _]manifest|program[_ x×]*query|manifest of record|program_query", pop_src, re.I):
                missing.append(f"C1 population from the manifest of record (population source {pop_src[:70]!r}: structural/interim, MAN-01 `--required-manifest` pending)")
            col = bnds[0].get("collector") if isinstance(bnds[0].get("collector"), dict) else {}
            if col:
                # the binding record's collector settings are authoritative for the run that was committed
                out["collector_variant"] = "windowed" if (col.get("window_slots") or col.get("window_mb")) else "whole-block"
                if col.get("retention"):
                    out["retention"] = str(col["retention"]).lower()
                out["collector_settings"] = {k: col.get(k) for k in ("variant", "retention", "window_mb", "window_slots", "arena_mb", "layout") if k in col}
                if col.get("variant"):
                    out["committer"] = out.get("committer") or col["variant"]
            out["identity_coverage"] = not bad
            out["openings_checked"] = opened
            out["openings_ok"] = verified
            out["openings_population"] = "drawn from the expected population (Fiat-Shamir from run_root + map digest)"
            out["late_step_in_sample"] = all(p.get("late_step_in_sample") for p in po)
            out["value_correctness"] = (verified == opened and unbound == 0) if opened else None
            out["negatives_suite"] = "verity_capture/bench/tests/test_binding_negatives.py (BIND-01)"
            if bad and wlf["B"] >= 2 and not any(int((b.get("population") or {}).get("step_rows_recorded") or 0) > 0 or b.get("request_rows") or b.get("b2_map") for b in bnds):
                # F-rev-08 (rev 18:10Z): at B>=2 a coverage FAIL on a tree whose collector recorded no per-step request rows (rc1, no B2 map)
                # is "population + map pending", not a rejected attack — the checker could not have bound per-request members
                out["status"] = "PARTIAL"
                b0 = bad[0]
                out["reasons"].append("B≥2 binding map not in this tree (step_rows_recorded 0 → per-request rows unbindable): population + map pending (F-rev-08), not a coverage rejection — first unbound: " + str((b0.get("coverage") or {}).get("first_missing") or b0.get("error") or "")[:120])
                out["coverage_pending_b2_map"] = True
                return out
            if bad:
                # COORD 18:24Z: MAN-01 phantom identities (element_range [0, 0] — a member the Program has no value for) are a manifest
                # defect (MAN-02), not a coverage gap: when every missing identity is zero-width, Commit is ◐ blocked, not ❌
                b0 = bad[0]
                cov0 = b0.get("coverage") or {}
                miss_list = cov0.get("missing") if isinstance(cov0.get("missing"), list) else []
                zero_listed = bool(miss_list) and all(list((m or {}).get("element_range") or [1, 1])[:2] in ([0, 0], (0, 0)) for m in miss_list if isinstance(m, dict))
                phantom_n = None
                mp = first(evdir, "manifest.json", "commit/manifest.json")
                if mp:
                    try:
                        M = load(mp)
                        rows_ = M.get("rows") or M.get("identities") or []
                        phantom_n = sum(1 for r_ in rows_ if isinstance(r_, dict) and list(r_.get("element_range") or [1, 1])[:2] == [0, 0])
                    except Exception:
                        phantom_n = None
                miss_n = int(cov0.get("missing_n") or 0)
                if zero_listed and miss_n and (phantom_n is None or phantom_n == miss_n):
                    out["status"] = "PARTIAL"
                    exp_ne = int(cov0.get("expected_n") or 0) - miss_n
                    out["reasons"].append(f"blocked: MAN-02 phantom identity — all {miss_n} missing identities are zero-width `{(miss_list[0] or {}).get('op_path')}/{(miss_list[0] or {}).get('output_member')}` members the Program has no value for (MAN-01 defect, COORD 18:24Z); nonempty coverage {cov0.get('bound_n')}/{exp_ne}, C3 openings verified — not a coverage gap, not a pass until the MAN-02 manifest rerun")
                    out["coverage_blocked_man02"] = {"missing_n": miss_n, "bound_n": cov0.get("bound_n"), "expected_nonempty": exp_ne}
                    out["identity_coverage"] = None
                    return out
                out["status"] = "FAIL"
                out["reasons"].append("binding coverage FAIL: " + (b0.get("error") or f"{(b0.get('coverage') or {}).get('missing_n')} of {(b0.get('coverage') or {}).get('expected_n')} required identities unbound" if not (b0.get('coverage') or {}).get('ok') else f"context: {(b0.get('context') or {}).get('why')}"))
                return out
            if unbound or (opened and verified != opened):
                out["status"] = "FAIL"
                out["reasons"].append(f"population-drawn openings: unbound {unbound}, verified {verified}/{opened}")
                return out
            pop0 = bnds[0].get("population") or {}
            pop_pd = pop0.get("program_digest") or ((pop0.get("manifest") or {}).get("program_digest") if isinstance(pop0.get("manifest"), dict) else None)
            if pop_pd:
                out["program_digest"] = pop_pd      # the Program the required manifest (C1 population) was derived from
            own_digests = {build.get("program_digest"), build.get("global_program_digest"), build.get("workload_program_digest")} | set((build.get("program_digests_per_shape") or {}).values()) | set(build.get("program_digests_per_rank") or [])
            own_digests.discard(None)
            if pop_pd and build.get("program_digest") and pop_pd not in own_digests:
                # [R14 moe] the manifest may derive from the row's REQUEST Program (one of Build's per-shape digests) — rc3 semantics;
                # [R15 B≥2, int-2 wiring M-0205/M-0366] the build-global manifest derives from the composed WORKLOAD Program (GP-01 digest)
                out["status"] = "FAIL"
                out["reasons"].append(f"binding population derived from Program {str(pop_pd)[:16]} ≠ this attempt's Build (step {str(build['program_digest'])[:16]}" + (f", workload Program {str(build.get('global_program_digest'))[:16]}" if build.get("global_program_digest") else "") + ") (same-attempt rule)")
                return out
            if pop_pd and build.get("global_program_digest") and pop_pd == build["global_program_digest"]:
                out["population_program"] = "composed workload Program (GP-01 digest — build-global manifest of record at B≥2)"
            if not pop_pd:
                missing.append("population.manifest.program_digest (the Program the required manifest derives from — rev-cb M-0125 (a))")
            # rev-cb M-0125 (b)(c): pin the manifest OF RECORD — recompute the required manifest's digest from the file when it is in
            # the evidence and compare with the binding population's manifest_digest; a manifest whose digest does not recompute or is
            # not the one the population was drawn from is not the row's manifest of record
            man0 = pop0.get("manifest") if isinstance(pop0.get("manifest"), dict) else {}
            pop_md = man0.get("manifest_digest") or pop0.get("manifest_digest")
            rm = {"manifest_digest": pop_md, "program_digest": pop_pd, "file": None, "recomputed": None}
            # every manifest file in the evidence dir (manifest.json / .json.gz — exec gzips them, M-0190; a Commit-only rerun keeps the
            # reused manifest as manifest_<tag>-<sha>.json.gz): each header must recompute; the one equal to the population's digest is
            # the manifest of record
            mfs = []
            for pat in ("manifest.json", "manifest.json.gz", "commit/manifest.json", "commit/manifest.json.gz", "build_request/manifest.json", "required_manifest.json", "manifest_*.json", "manifest_*.json.gz"):
                mfs += glob.glob(os.path.join(evdir, pat))
            mfs = sorted(set(mfs))
            if mfs:
                try:
                    from verity_capture.commit.required_manifest import manifest_digest as _local_mdigest
                    # the digest is a function of the PRODUCING tree's row key / canonical identity (int-2's _row_key sorts by engine_step, rc3's
                    # does not — M-0366 OLMoE B2 recomputes differently under rc3): recompute with that tree's own required_manifest when it
                    # can be read from git; a mismatch under the report tree's function alone is NOT evidence of editing (named ◐, never ❌)
                    RM = tree_required_manifest(rev_hint)
                    _mdigest = RM.manifest_digest if RM is not None and callable(getattr(RM, "manifest_digest", None)) else _local_mdigest
                    rm["digest_function"] = f"required_manifest.manifest_digest@{str(rev_hint)[:9]}" if _mdigest is not _local_mdigest else ("required_manifest.manifest_digest@report-tree (producing tree " + (str(rev_hint)[:9] if rev_hint else "unknown") + " not loadable)")
                    checked = []
                    pinned = None
                    for mf in mfs:
                        MF = load(mf)
                        ids = MF.get("identities") if isinstance(MF, dict) else None
                        if not (isinstance(ids, list) and ids):
                            continue
                        rec = _mdigest(ids)
                        checked.append({"file": os.path.relpath(mf), "recomputed": rec, "header": MF.get("manifest_digest"), "program_digest": MF.get("program_digest"), "identities": len(ids)})
                        if MF.get("manifest_digest") and rec != MF["manifest_digest"]:
                            if _mdigest is _local_mdigest:
                                alt = None
                                if rev_hint and RM is None:
                                    alt = f"producing tree {str(rev_hint)[:9]} not loadable"
                                missing.append(f"manifest of record digest recompute with the producing tree's digest function ({os.path.relpath(mf)} header {str(MF['manifest_digest'])[:16]} ≠ report-tree recompute {rec[:16]}; {alt or 'revision unknown'} — not graded as edited)")
                                continue
                            out["status"] = "FAIL"
                            out["reasons"].append(f"{os.path.relpath(mf)} header manifest_digest {str(MF['manifest_digest'])[:16]} does not recompute from its identities ({rec[:16]}, {rm['digest_function']}) — edited manifest (M-0125 (c))")
                            return out
                        if pop_md and rec == pop_md and pinned is None:
                            pinned = checked[-1]
                    rm["checked"] = checked
                    if pinned:
                        rm.update({"file": pinned["file"], "recomputed": pinned["recomputed"], "file_program_digest": pinned["program_digest"], "identities": pinned["identities"]})
                        out["evidence"].append(pinned["file"])
                        if pinned["program_digest"] and pop_pd and pinned["program_digest"] != pop_pd:
                            out["status"] = "FAIL"
                            out["reasons"].append(f"{pinned['file']} program_digest {str(pinned['program_digest'])[:16]} ≠ binding population's {str(pop_pd)[:16]} (M-0125 (a))")
                            return out
                    elif checked and pop_md:
                        out["status"] = "FAIL"
                        out["reasons"].append(f"binding population manifest_digest {str(pop_md)[:16]} matches none of the manifest files in the evidence ({', '.join(c['file'].split('/')[-1] + ':' + c['recomputed'][:12] for c in checked[:3])}) — not the row's manifest of record (M-0125 (b))")
                        return out
                    elif not checked:
                        missing.append("manifest of record readable (manifest*.json(.gz) present but without identities)")
                except Exception as e:      # noqa: BLE001 — a malformed manifest file is named, never skipped silently
                    rm["error"] = f"{type(e).__name__}: {e}"[:160]
                    missing.append(f"manifest of record readable ({rm['error']})")
            else:
                out["reasons"].append("manifest of record (manifest.json[.gz]) not in the evidence directory — manifest_digest pinned from the binding record only (M-0125 F-cb-11; producers: sync <row>/manifest.json)")
            out["required_manifest"] = rm
            # [manbind M-0340 (3) / M-0341 MAN-05; rev-cb M-0334 F-cb-11 residual] the digest pin above proves the manifest was not edited
            # after digesting and is the one the population was drawn from; it does NOT prove it is THE manifest of the Program × workload
            # (a weakened-but-honest manifest recomputes).  `required_manifest verify` rebuilds it from the Build's Program and compares:
            # its record is graded here — ok:false ⇒ ❌; a record about another manifest / another Program is not this row's; absent ⇒ named
            # (◐ when the tree that ran carries the tool, footnote when it predates it — the tool ships with manbind 84f5cffa3, int-3+).
            mv = manifest_verify_from(evdir)
            if mv:
                out["manifest_verify"] = mv
                if mv.get("artifact"):
                    out["evidence"].append(mv["artifact"])
                if mv.get("error"):
                    missing.append(f"MAN-05 manifest-verify record readable ({mv['error']})")
                elif pop_md and mv.get("used_digest") and str(mv["used_digest"]) != str(pop_md):
                    missing.append(f"MAN-05 verify record for THIS manifest of record (verify used_digest {str(mv['used_digest'])[:16]} ≠ binding population manifest_digest {str(pop_md)[:16]})")
                elif build.get("program_digest") and mv.get("rebuilt_program_digest") and str(mv["rebuilt_program_digest"]) not in {str(d) for d in own_digests}:
                    # own_digests = Build's step digest + GP-01 workload digest + per-shape / per-rank digests: at B≥2 the manifest is rebuilt
                    # from the composed WORKLOAD Program (exec M-0544 (a); same rule as the binding population, M-0408)
                    missing.append(f"MAN-05 verify against THIS attempt's Build (rebuilt from Program {str(mv['rebuilt_program_digest'])[:16]} ≠ Build's step {str(build['program_digest'])[:16]}" + (f" / workload {str(build.get('global_program_digest'))[:16]}" if build.get("global_program_digest") else "") + ")")
                elif as_bool(mv.get("ok")) is False:
                    out["status"] = "FAIL"
                    out["reasons"].append(f"manifest is not the manifest of record (MAN-05 required_manifest verify): {str(mv.get('why') or 'ok:false without a cause')[:260]}")
                    return out
                elif as_bool(mv.get("ok")) is True:
                    out.setdefault("footnotes", []).append(f"manifest of record verified (MAN-05): used manifest_digest == digest of its rows == the manifest rebuilt from the Build's Program × workload ({mv.get('rebuilt_identities')} identities, rebuilt complete:{mv.get('rebuilt_complete')}) — {mv.get('artifact')}")
                else:
                    missing.append("MAN-05 manifest-verify record with ok:true|false")
            else:
                out["manifest_verify"] = None          # graded against the attempt's revision once it is known (apply_man05_absence)
            if not opened and not any(p.get("skipped") for p in po):
                missing.append("population-drawn openings (0 opened)")
            if not out["late_step_in_sample"]:
                out["reasons"].append("no late-step identity in the drawn openings")
            oar = binding_after_release(evdir)
            out["openings_after_release"] = oar
            if out.get("collector_variant") == "windowed" and oar is not True:
                missing.append("openings_after_release: true (windowed collector — openings after the staging release, brief §4)")
            out["coverage_verdict"] = "PASS"
            c = "runs"  # mark as covered
    # TABLE-RULES Commit: expected population from the Program/query manifest, binding map, coverage verdict from bind's checker
    if c == "runs":
        pass
    elif c:
        C = load(c)
        out["coverage_checker"] = C.get("checker")
        out["coverage_verdict"] = C.get("verdict")
        out["manifest_digest"] = C.get("manifest_digest")
        out["binding_map_digest"] = C.get("binding_map_digest")
        out["negatives_suite"] = C.get("negatives_suite") or C.get("negatives_suite_version")
        out["openings_checked"] = C.get("openings_checked")
        out["openings_ok"] = C.get("openings_ok")
        out["openings_population"] = C.get("openings_population")
        out["identity_coverage"] = C.get("identity_coverage")
        out["value_correctness"] = C.get("value_correctness")
        if str(C.get("verdict") or "").upper().startswith("FAIL"):
            out["status"] = "FAIL"
            out["reasons"].append(f"coverage checker verdict FAIL: {C.get('reason') or C.get('missing') or ''}")
            return out
        if C.get("openings_checked") is not None and C.get("openings_ok") != C.get("openings_checked"):
            out["status"] = "FAIL"
            out["reasons"].append(f"openings ok {C.get('openings_ok')} / checked {C.get('openings_checked')}")
            return out
        if as_bool(C.get("identity_coverage")) is False or as_bool(C.get("value_correctness")) is False:
            out["status"] = "FAIL"
            out["reasons"].append(f"identity_coverage={C.get('identity_coverage')} value_correctness={C.get('value_correctness')}")
            return out
        for k in ("manifest_digest", "binding_map_digest", "negatives_suite", "openings_checked"):
            if not out.get(k):
                missing.append(k)
        if str(C.get("verdict") or "").upper() != "PASS":
            missing.append("coverage verdict PASS")
        if str(C.get("openings_population") or "").lower() not in ("expected", "population", "expected population"):
            missing.append(f"openings drawn from the expected population (openings_population={C.get('openings_population')!r})")
        out["openings_after_release"] = C.get("openings_after_release")
        if C.get("collector_variant"):
            out["collector_variant"] = C["collector_variant"]
        if C.get("retention"):
            out["retention"] = str(C["retention"]).lower()
        if C.get("engine_settings"):
            out["engine_settings"] = C["engine_settings"]
        if out.get("collector_variant") == "windowed" and as_bool(C.get("openings_after_release")) is not True:
            missing.append("openings_after_release: true (windowed collector — openings after the staging release, brief §4)")
        # [R14 moe] as for Match (above): the manifest is derived from the row's REQUEST Program (`build_request_LP*_T*` in build_summary), which
        # is one of Build's per-shape digests, not necessarily the step Program's
        if C.get("program_digest"):
            out["program_digest"] = C["program_digest"]
        if build.get("program_digest") and C.get("program_digest") and C.get("program_digest") not in build_program_family(build):
            out["status"] = "FAIL"
            out["reasons"].append(f"coverage manifest derived from a different Program than Build's ({str(C.get('program_digest'))[:16]} ∉ Build's step/workload/per-shape digests; same-attempt rule)")
            return out
    else:
        src = str(out.get("required_classes_source") or "")
        out["reasons"].append("counts + required classes only (" + (src[:60] if src else "collector-declared") + ") — no manifest/binding-map coverage verdict (bind's checker); counts + classes are insufficient (F-rev-01)")
        missing += ["manifest_digest", "binding_map_digest", "coverage verdict", "negatives suite", "openings from the expected population"]
    if out["retention"] is None:
        missing.append("retention mode")
    if not out["committer"]:
        missing.append("collector variant")
    # bind's coverage.json fields that apply whichever Commit path produced the verdict (C1 population source, C2, C3 context)
    if c and not out.get("manifest_source"):
        try:
            C2 = load(c)
        except Exception:
            C2 = {}
        if isinstance(C2, dict):
            out["manifest_source"] = C2.get("manifest_source")
            if C2.get("manifest_source") and str(C2.get("manifest_source")).lower().split(" ")[0] not in ("program_query", "program+query", "manifest_of_record", "required_manifest"):
                missing.append(f"C1 population from the manifest of record (manifest_source={str(C2.get('manifest_source'))[:60]!r}: structural/interim population, MAN-01 `--required-manifest` pending)")
            if as_bool(C2.get("context_ok")) is False:
                out["status"] = "FAIL"
                out["reasons"].append("C3 context_ok False (openings did not verify against the original roots/layout)")
                return out
            if isinstance(C2.get("value_correspondence"), dict) and not isinstance(out.get("value_correspondence"), dict):
                out["value_correspondence"] = C2["value_correspondence"]
            for k in ("manifest_digest", "binding_map_digest", "negatives_suite"):
                if C2.get(k) and not out.get(k):
                    out[k] = C2[k]
                    missing[:] = [m for m in missing if m != k]
            if C2.get("engine_settings") and not out.get("engine_settings"):
                out["engine_settings"] = C2["engine_settings"]
    # the Program the required manifest was derived from (C1): from the shipped manifest when the checker did not record it
    if not out.get("program_digest"):
        mp = first(evdir, "manifest.json", "commit/manifest.json", "required_manifest.json", "commit/required_manifest.json")
        if mp:
            try:
                Mf = load(mp)
                if isinstance(Mf, dict) and Mf.get("program_digest"):
                    out["program_digest"] = Mf["program_digest"]
                    out["manifest_artifact"] = os.path.relpath(mp)
            except Exception:
                pass
    # C2 value correspondence (Daniel 17:36Z ruling 2; TABLE-RULES §Commit): required live evidence, named method, real artifact — never a footnote
    vc = out.get("value_correspondence") if isinstance(out.get("value_correspondence"), dict) else value_correspondence_from(evdir)
    out["value_correspondence"] = vc or {"method": None, "result": None}
    comps = vc.get("components") if vc and isinstance(vc.get("components"), dict) and vc.get("components") else None
    if vc and (as_bool(vc.get("result")) is False or str(vc.get("result") or "").upper().startswith("FAIL") or as_list(vc.get("mismatches"))):
        mm = as_list(vc.get("mismatches"))
        out["status"] = "FAIL"
        # [rev-cb M-0301 R1] the composite's `why` names the FAILING component's own cause (never the oracle's partial note)
        cause = str(vc.get("why") or "").strip()
        out["reasons"].append(f"C2 value correspondence FAIL ({vc.get('method')}: " + (cause[:220] if cause else (f"{len(mm)} mismatching identit{'y' if len(mm) == 1 else 'ies'}, first {json.dumps(mm[0], default=str)[:120]}" if mm else str(vc.get('detail') or vc.get('artifact') or ''))) + ")")
        return out
    # COORD DECISION M-0111 (1): Commit ✅ requires value_correspondence.result true AND method ∈ {root-compare-vs-independent-run,
    # oracle-compare} AND, for root-compare, a recorded reference provenance (the reference run's attempt/pod/release/tree — not this
    # attempt's own pairs).  null / offline-suite / 'not checked' / missing provenance ⇒ ◐ 'C2 not checked'.
    meth = str((vc or {}).get("method") or "").lower()
    root_compare = meth.startswith("root-compare-vs-independent-run") or meth.startswith("root_compare_vs_independent_run")
    oracle = meth.startswith("oracle-compare") or meth.startswith("oracle_compare")
    own_artifacts = {os.path.relpath(p) for p in (first(evdir, "commit/runs.jsonl", "runs.jsonl"), first(evdir, "commit_runs_instrumented.json", "commit/runs_instrumented.json")) if p}
    # reference provenance / independence of a root-compare reference (used by both the legacy single-method shape and the composite)
    refs_all = [e for e in as_list((vc or {}).get("references")) if isinstance(e, dict)]
    ev_abs = os.path.abspath(evdir)
    def _prov(e: dict) -> bool:
        ident = e.get("kind") or e.get("tree") or e.get("release") or e.get("pod") or e.get("attempt") or e.get("run")
        art = str(e.get("artifact") or "")
        return bool(e.get("root")) and bool(ident) and (art == "" or os.path.relpath(art) not in own_artifacts)
    # COORD RULING M-0286 (report's M-0212): a same-tree reproduction is NOT independent — it carries the same collector attribution,
    # so F-cb-02 / F-cb-23 misattributions reproduce byte-for-byte.  Independent = a pinned root from another tree/pod (R13 or other
    # release) or the live Match oracle; a reference whose artifact lives inside this evidence dir is this attempt's own reproduction.
    def _independent(e: dict) -> bool:
        kind = str(e.get("kind") or "")
        if re.search(r"reproduc|this row's|same (release )?tree|same settings|clean[- ]worker|re-?run of this", kind, re.I):
            return False
        art = str(e.get("artifact") or "")
        if art and os.path.abspath(art).startswith(ev_abs + os.sep):
            return False
        return True
    good_all = [e for e in refs_all if _prov(e)]
    indep_all = [e for e in good_all if _independent(e)]
    # [Daniel DECISIONS-2 §4 / COORD M-0356 (1) — supersedes M-0111 (1)(2) and M-0286] C2 OF RECORD = sampled exact VU replay + ordered
    # boundary linkage (+ targeted negatives).  Record: value_correspondence.components.sampled_replay {result, population, n, strata, seed,
    # unreplayable[], why} (or value_correspondence.sampled_replay / method 'sampled-exact-replay'), .linkage {result, method, checked, why},
    # .negatives {suite, results|artifact}.  Independent reference / full oracle-compare / MATCH_SNAP_STEPS=all are NOT required.
    sr = (comps or {}).get("sampled_replay") if comps else None
    if not isinstance(sr, dict) and vc and isinstance(vc.get("sampled_replay"), dict):
        sr = vc["sampled_replay"]
    if not isinstance(sr, dict) and vc and C2_OF_RECORD in meth and not comps:
        sr = vc                                          # single-method record: the fields sit on value_correspondence itself
    link = None
    for cand in ((comps or {}).get("boundary_linkage") if comps else None, (comps or {}).get("linkage") if comps else None,
                 (vc or {}).get("boundary_linkage"), (vc or {}).get("linkage"),
                 (sr or {}).get("boundary_linkage") if isinstance(sr, dict) else None, (sr or {}).get("linkage") if isinstance(sr, dict) else None):
        if isinstance(cand, dict):
            link = cand                                  # key of record: boundary_linkage (rev-cb M-0411); 'linkage' accepted as alias
            break
    neg = None
    for cand in ((comps or {}).get("negatives") if comps else None, (vc or {}).get("negatives"), (sr or {}).get("negatives") if isinstance(sr, dict) else None, out.get("negatives_suite")):
        if cand:
            neg = cand
            break
    # [COORD DECISION M-0451 (3), 22:54Z — the rendering rule of record while the sampled replay lands] Commit ✅ iff C1 ∧ C3 ∧
    # ([row-own sampled_replay + boundary_linkage record] OR [a cited QUALIFICATION record whose domain covers the row AND the row's own
    # linkage evidence passed — F-cb-23 gate at B≥2, tokens/prompt linkage at B=1]); the qualification is cited beneath the table by id.
    # Otherwise Commit ◐ 'C2 qualification pending (int-4 rerun queued)'.  The M-0111 forms (independent root-compare at B=1, complete
    # oracle-compare) and a same-tree reproduction are DIAGNOSTICS: named on the row, never the basis of a ✅.
    def _c2_pending(diag: list[str]) -> None:
        quals = load_qualifications()
        q, whynot = qualification_cover(row, wlf, out, quals)
        lk_ok, lk_how = row_linkage_evidence(vc, link, wlf)
        if q and lk_ok:
            out["c2_form"] = f"qualification {q.get('id')}"
            out["c2_basis"] = [f"qualification record {q.get('id')} ({q.get('board_message') or q.get('_path')}; mechanism {str(q.get('mechanism') or '?')[:40]}) + row-own linkage: {lk_how} — COORD M-0451 (3) bridge"]
            out["c2_qualification"] = {"id": q.get("id"), "board_message": q.get("board_message"), "path": q.get("_path"), "domain": q.get("domain"), "row_linkage": lk_how}
            out.setdefault("footnotes", []).append(f"C2 via cited qualification {q.get('id')} (M-0451 (2)(a)) — the row's OWN sampled-replay record is owed by the 01:10Z snapshot (M-0451 (4))")
            if diag:
                out.setdefault("footnotes", []).append("C2 diagnostics on record (not the basis): " + "; ".join(diag)[:300])
            return
        why = []
        if diag:
            why.append("on record (diagnostic, not the C2 of record): " + "; ".join(diag)[:240])
        if quals:
            why.append("no cited qualification covers this row (" + "; ".join(whynot)[:240] + ")")
        else:
            why.append("no qualification record on record yet (rev-cb posts the first from the int-4 canary rows — M-0451)")
        why.append(("row-own linkage: " + lk_how) if lk_ok else ("row-own linkage missing: " + lk_how))
        missing.append(f"{C2_PENDING} — DECISIONS-2 §4 / COORD M-0451 (3) + M-0481: Commit ✅ needs a row-own C2 form — (A) sampled_replay + boundary_linkage, or (B) the oracle-compare of record (every committed value vs the same run's Match observation, --match-dir: result true, 0 differing, 0 unattributed, complete, attribution gate at B≥2) — or a cited qualification record covering the row plus the row's own linkage evidence; " + "; ".join(why))

    if isinstance(sr, dict):
        gsr = grade_sampled_replay(sr, link, neg, wlf)
        out["c2_of_record"] = {"method": C2_OF_RECORD, "status": gsr["status"], "missing": gsr["missing"], "linkage": link, "artifact": sr.get("artifact") or vc.get("artifact")}
        for fn in gsr["footnotes"]:
            out.setdefault("footnotes", []).append(fn)
        if gsr["status"] == "FAIL":
            gsc = gemma_softcap_reader_defect(evdir, str(row.get("row_id") or ""), sr, gsr["fail"] or "")
            if gsc:
                out["status"] = "PARTIAL"
                out["reasons"].append(gsc)
                out["c2_checker_defect"] = "M-0619 (softcap reader)"
                out.setdefault("footnotes", []).append("C2 form (A) FAIL on record (checker): " + str(gsr["fail"])[:200])
                return out
            out["status"] = "FAIL"
            out["reasons"].append(gsr["fail"])
            return out
        g = {"graded": {}, "pass_components": [], "partial": [], "fail": []}
        if comps:
            g = grade_c2_components(vc, wlf, True if indep_all else (False if good_all else None), evdir, str(row.get("row_id") or ""))
            out["c2_components"] = g["graded"]
            if g["fail"]:
                out["status"] = "FAIL"
                out["reasons"].append("C2 value correspondence FAIL (" + "; ".join(g["fail"])[:300] + ")")
                return out
            for pnote in g["partial"]:
                out.setdefault("footnotes", []).append("C2 diagnostic component (not required, DECISIONS-2 §4): " + pnote)
        if gsr["status"] == "PASS":
            wp_ok, wp_how = weights_pin_of_record(evdir, build, sr, out)
            out["weights_pin"] = {"ok": wp_ok, "how": wp_how}
            ob_pass = bool(comps) and g["graded"].get("oracle_compare") == "PASS" if comps else False
            # [rev BLOCKER M-0016 F-r16-02 — until the commit_verdict rule lands] the MAN-06f/06g population exclusions (VUs fed by MoE
            # block-stream planes; VUs fed by an interior producer without an evaluator) are a NAMED GAP the replay no longer grades: a PASS
            # 'sampled replay' alone does not establish C2 over them (COORD M-0721 (1)) — form (B) complete on the same record covers the
            # interior-composed strata (instance_outputs ARE compared); MoE planes have no oracle value (NO_ORACLE_FAMILY_PREFIXES, F-r16-01).
            gaps = population_named_gaps(sr)
            gap_block = False
            if gaps["n"]:
                oc_ = comps.get("oracle_compare") if isinstance(comps.get("oracle_compare"), dict) else None
                okb_, whyb_ = oracle_form_b(oc_, int(wlf.get("B") or 1)) if oc_ else (False, "no oracle_compare record on this attempt")
                if gaps["plane_fed"]:
                    gap_block = True
                    missing.append(f"C2 of record not established over {gaps['plane_fed']} VUs fed by MoE block-stream planes (sampled_replay.population.not_evaluable, MAN-06f, by name: "
                                   f"{gaps['first_plane'][:120]}) — moe/ planes carry no dispatcher-level oracle value, so form (B) cannot cover them (rev M-0008 / F-r16-01): needs form (A) on those "
                                   f"strata or the tap's own bit-exact xcheck on record (rev BLOCKER M-0016 F-r16-02, COORD M-0721 (1))")
                if gaps["interior"]:
                    if okb_:
                        out.setdefault("footnotes", []).append(f"C2 of record by form (B) over {gaps['interior']} VUs not evaluable by name (MAN-06g interior-composed: {gaps['first_interior'][:100]}) — "
                                                               f"{whyb_[:160]} (rev M-0016 gate; COORD M-0721 (1))")
                    else:
                        gap_block = True
                        missing.append(f"C2 of record not established (COORD M-0721 (1); rev BLOCKER M-0016 F-r16-02): {gaps['interior']} VUs not evaluable by name (MAN-06g interior-composed: "
                                       f"{gaps['first_interior'][:100]}) and form (B) is not complete on this record — {whyb_[:140]}; Match with MATCH_SNAP_STEPS=all makes (B) complete")
                out["c2_population_gaps"] = {k: v for k, v in gaps.items() if k != "n"} | {"form_b_complete": okb_}
            if wp_ok and not gap_block:
                out["c2_form"] = "sampled replay"            # COORD M-0481 (A): the cell names the form
                out["c2_basis"] = [gsr["basis"] + "; " + wp_how]
            elif wp_ok:
                out["c2_diagnostics"] = [gsr["basis"] + "; " + wp_how]   # (A) 0 mismatch on the evaluable population — not the C2 of record while the named gap stands
            else:
                # [DANIEL M-0776 (2)] the pin is unconditional: no form (B) substitute while (A) waits for it
                out["c2_diagnostics"] = [gsr["basis"]] + ([p for p in g["pass_components"] if p.startswith("oracle-compare of record")] if ob_pass else [])
                missing.append(WEIGHTS_UNPINNED + f" — this record: {wp_how}" + (" (oracle-compare complete on the same record is not a substitute for the pin — M-0776 (2))" if ob_pass else ""))
        elif out.get("verdict_overridden_by") == "COORD M-0721 (1)":
            out.setdefault("footnotes", []).append("form (A) sampled replay on this record is partial (strata not evaluable, named) — not the basis; C2 of record = form (B) + (A) 0 mismatch (COORD M-0721 (1))")
            wp_ok, wp_how = weights_pin_of_record(evdir, build, sr, out)
            out["weights_pin"] = {"ok": wp_ok, "how": wp_how}
            if not wp_ok:
                missing.append(WEIGHTS_UNPINNED + f" — this record: {wp_how} (M-0776 (2): unconditional)")
            if FORM_B_PATH_INDEPENDENT is not True:
                out["c2_form_pending"] = out.pop("c2_form", None)
                missing.append(FORM_B_PATH_PENDING)
            else:
                out.setdefault("footnotes", []).append(FORM_B_PATH_STATED)
        else:
            _c2_pending(["sampled_replay record incomplete: " + "; ".join(gsr["missing"])[:300]])
        if missing:
            out["status"] = "PARTIAL"
            out["reasons"].append("missing: " + ", ".join(missing))
        else:
            out["status"] = "PASS"
        return out
    if comps:
        # [rev-cb M-0301 R2 / manbind M-0340 (2)] composite C2: each component graded by its own rule; Commit ✅ needs at least one
        # passing live component (oracle-compare complete, or root-compare vs an INDEPENDENT root at B=1) and no failing one.
        g = grade_c2_components(vc, wlf, True if indep_all else (False if good_all else None), evdir, str(row.get("row_id") or ""))
        out["c2_components"] = g["graded"]
        if g["fail"]:
            out["status"] = "FAIL"
            out["reasons"].append("C2 value correspondence FAIL (" + "; ".join(g["fail"])[:300] + ")")
            return out
        if not (vc.get("artifact") or vc.get("roots") or vc.get("evidence")):
            missing.append("C2 value correspondence artifact (roots compared / oracle comparison output path)")
        diag = []
        if "root_compare" in g["graded"] and g["graded"]["root_compare"] == "PASS":
            out["c2_reference_provenance"] = [{k: e.get(k) for k in ("kind", "root", "artifact", "tree", "release", "pod", "compute_capability") if k in e} for e in indep_all]
            out["c2_reference_independent"] = True
        elif "root_compare" in g["graded"] and good_all and not indep_all:
            out["c2_reference_provenance"] = [{k: e.get(k) for k in ("kind", "root", "artifact", "tree", "release", "pod", "compute_capability") if k in e} for e in good_all]
            out["c2_reference_independent"] = False
        # [COORD M-0481 (B)] the oracle-compare OF RECORD is a row-own C2 form: Commit ✅ 'C2: oracle-compare'
        if g["graded"].get("oracle_compare") == "PASS" and FORM_B_WEIGHTS_ROUTE_INDEPENDENT is False:
            out["c2_diagnostics"] = g["pass_components"]
            missing.append(WEIGHTS_UNPINNED + " — form (B) measured to be on the same live-weights route (rev-cb, M-0614 (1)); the oracle-compare of record stays a diagnostic until the pin lands")
        elif g["graded"].get("oracle_compare") == "PASS":
            out["c2_form"] = "oracle-compare"
            out["c2_basis"] = [p for p in g["pass_components"] if p.startswith("oracle-compare of record")]
            wp_ok, wp_how = weights_pin_of_record(evdir, build, None, out)
            out["weights_pin"] = {"ok": wp_ok, "how": wp_how}
            if WEIGHTS_PIN_UNCONDITIONAL and not wp_ok:
                missing.append(WEIGHTS_UNPINNED + f" — this record: {wp_how} (M-0776 (2): unconditional, regardless of form)")
            if FORM_B_PATH_INDEPENDENT is not True:
                out["c2_form_pending"] = out.pop("c2_form", None)
                missing.append(FORM_B_PATH_PENDING)
            else:
                out.setdefault("footnotes", []).append(FORM_B_PATH_STATED)
            if FORM_B_WEIGHTS_ROUTE_INDEPENDENT is None:
                out.setdefault("footnotes", []).append("form (B) ✅ pending rev-cb's flipweight-vs-oracle-compare measurement (COORD M-0614 (1), due 00:50Z): if the Match oracle does not load the checkpoint independently, every form-(B) ✅ drops to ◐ 'weights unpinned'")
            else:
                out.setdefault("footnotes", []).append(FORM_B_MEASURED)
            out["c2_diagnostics"] = [p for p in g["pass_components"] if not p.startswith("oracle-compare of record")] + [p[:160] for p in g["partial"]]
            for pnote in g["partial"]:
                out.setdefault("footnotes", []).append("C2 diagnostic component (not the basis): " + pnote)
        else:
            if g["pass_components"]:
                out["c2_diagnostics"] = g["pass_components"]
                diag += [f"M-0111-form component PASS: {c[:120]}" for c in g["pass_components"]]
            if out.get("c2_reference_independent") is False:
                diag.append("root-compare vs a same-tree reproduction (neither layer — M-0451 (3))")
            diag += [p[:160] for p in g["partial"]]
            _c2_pending(diag)
        if missing:
            out["status"] = "PARTIAL"
            out["reasons"].append("missing: " + ", ".join(missing))
        else:
            out["status"] = "PASS"
        return out
    # single-method records (rc3 / R14 producers): the M-0111 forms are diagnostics under M-0451 (3); what is on record is named
    diag: list[str] = []
    if not vc or not (as_bool(vc.get("result")) is True or str(vc.get("result") or "").upper() == "PASS"):
        diag.append("no value_correspondence with result true" + (f" (method {vc.get('method')!r}, result {vc.get('result')!r})" if vc else ""))
    elif not (root_compare or oracle):
        diag.append(f"method {vc.get('method')!r} is neither sampled-exact-replay nor an M-0111 form (root-compare-vs-independent-run / oracle-compare); offline-suite / replay alone is not C2")
    elif not (vc.get("artifact") or vc.get("roots") or vc.get("evidence")):
        diag.append("value correspondence without an artifact (roots compared / oracle comparison output path)")
    elif root_compare and wlf["B"] >= 2:
        # rev-cb ALERT M-0216 (F-cb-23, measured on the panel B2 row): swapping two requests' labels on an equal-row-count decode
        # step leaves the run root, coverage, composition and openings unchanged — root-compare cannot see request attribution.
        diag.append(f"root-compare at B={wlf['B']} cannot see a request-attribution swap on equal-row-count steps (rev-cb M-0216 F-cb-23)")
    elif root_compare:
        good, indep = good_all, indep_all
        if not good:
            diag.append("root-compare without reference provenance (value_correspondence.references[] names no reference run)")
        elif not indep:
            out["c2_reference_provenance"] = [{k: e.get(k) for k in ("kind", "root", "artifact", "tree", "release", "pod", "compute_capability") if k in e} for e in good]
            out["c2_reference_independent"] = False
            diag.append("root-compare vs a same-tree reproduction of this row's Commit (" + str(good[0].get("kind") or "")[:90] + ") — reproducibility only, neither layer (DECISIONS-2 §4, M-0451 (3))")
        else:
            out["c2_reference_provenance"] = [{k: e.get(k) for k in ("kind", "root", "artifact", "tree", "release", "pod", "compute_capability") if k in e} for e in indep]
            out["c2_reference_independent"] = True
            diag.append("root-compare vs an independent pinned root (" + str(indep[0].get("kind") or "")[:60] + ") — M-0111 form")
    else:
        # single-method oracle-compare record: form (B) when complete / 0 differing / 0 unattributed (+ attribution gate at B≥2)
        okb, whyb = oracle_form_b(vc, int(wlf.get("B") or 1))
        if okb and FORM_B_WEIGHTS_ROUTE_INDEPENDENT is False:
            diag.append(whyb[:160])
            missing.append(WEIGHTS_UNPINNED + " — form (B) measured to be on the same live-weights route (rev-cb, M-0614 (1))")
        elif okb:
            out["c2_form"] = "oracle-compare"
            out["c2_basis"] = [whyb]
            wp_ok, wp_how = weights_pin_of_record(evdir, build, None, out)
            out["weights_pin"] = {"ok": wp_ok, "how": wp_how}
            if WEIGHTS_PIN_UNCONDITIONAL and not wp_ok:
                missing.append(WEIGHTS_UNPINNED + f" — this record: {wp_how} (M-0776 (2): unconditional, regardless of form)")
            if FORM_B_PATH_INDEPENDENT is not True:
                out["c2_form_pending"] = out.pop("c2_form", None)
                missing.append(FORM_B_PATH_PENDING)
            else:
                out.setdefault("footnotes", []).append(FORM_B_PATH_STATED)
            if FORM_B_WEIGHTS_ROUTE_INDEPENDENT is None:
                out.setdefault("footnotes", []).append("form (B) ✅ pending rev-cb's flipweight-vs-oracle-compare measurement (COORD M-0614 (1), due 00:50Z)")
            else:
                out.setdefault("footnotes", []).append(FORM_B_MEASURED)
        else:
            diag.append(f"oracle-compare result true but not the form of record (M-0481 (B)): {whyb[:200]}")
    if not out.get("c2_form") and not out.get("c2_form_pending"):
        _c2_pending(diag)
    if missing:
        out["status"] = "PARTIAL"
        out["reasons"].append("missing: " + ", ".join(missing))
    else:
        out["status"] = "PASS"
    return out


def binding_after_release(evdir: str) -> bool | None:
    """True iff every instrumented run's `commit.openings_after_release` has n>0 and verified == n (P0 windowed collector)."""
    rows = _runs(evdir)
    seen = []
    for r in rows:
        cm = r.get("commit") if isinstance(r.get("commit"), dict) else {}
        oar = cm.get("openings_after_release")
        if isinstance(oar, dict):
            seen.append(int(oar.get("n") or 0) > 0 and int(oar.get("verified") or 0) == int(oar.get("n") or 0))
    if not seen:
        return None
    return all(seen)


def _runs(evdir: str) -> list[dict]:
    p = first(evdir, "commit_runs_instrumented.json", "commit/runs_instrumented.json")
    rows: list = []
    if p:
        try:
            rows = load(p) or []
        except Exception:
            rows = []
    q = first(evdir, "runs.jsonl", "commit/runs.jsonl")
    if q and not rows:
        for line in open(q, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return [r for r in rows if isinstance(r, dict)]


def binding_records(evdir: str) -> list[dict]:
    """`commit.binding` of every instrumented (treatment) run in runs.jsonl / commit_runs_instrumented.json (BIND-01)."""
    out = []
    p = first(evdir, "commit_runs_instrumented.json", "commit/runs_instrumented.json")
    rows = []
    if p:
        try:
            rows = load(p) or []
        except Exception:
            rows = []
    q = first(evdir, "runs.jsonl", "commit/runs.jsonl")
    if q and not rows:
        for line in open(q, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    for r in rows:
        if not isinstance(r, dict):
            continue
        cm = r.get("commit") if isinstance(r.get("commit"), dict) else {}
        if isinstance(cm.get("binding"), dict) and (r.get("case") in (None, "treatment", "instrumented") or cm.get("binding")):
            out.append(cm["binding"])
    return out


def derive_bench(evdir: str, commit: dict) -> dict:
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": []}
    d = first(evdir, "commit_delta_summary.json", "commit/summary.json")
    s = first(evdir, "commit_summary.json")
    if not (d or s):
        out["reasons"].append("no timing summary in the evidence directory")
        return out
    D = load(d) if d else {}
    S = load(s) if s else {}
    for p in (d, s):
        if p:
            out["evidence"].append(os.path.relpath(p))
    cs = as_list(S.get("control_wall")) or as_list(D.get("control_wall_s")) or as_list(D.get("control_wall"))
    ts = as_list(S.get("treatment_wall")) or as_list(D.get("treatment_wall_s")) or as_list(D.get("treatment_wall"))
    dw = as_list(D.get("delta_wall"))
    abs_ms = as_list(D.get("abs_wall_ms"))
    if not cs and dw and abs_ms:
        cs = [(float(a) / 1000.0) / float(x) for a, x in zip(abs_ms, dw)]
        ts = [c * (1 + float(x)) for c, x in zip(cs, dw)]
        out["reasons"].append("per-round control/treatment reconstructed from delta_wall + abs_wall_ms (recorded in the delta summary)")
    cs = [float(x) for x in cs if x is not None]
    ts = [float(x) for x in ts if x is not None]
    out.update({"control_s": cs, "treatment_s": ts, "rounds": min(len(cs), len(ts)) if cs and ts else 0, "interleaved": True if (S or D) else None,
                "delta_wall": dw, "delta_wall_ci95": as_list(D.get("delta_wall_ci95")), "artifact": os.path.relpath(d or s), "profiled": False,
                "collector_variant": commit.get("collector_variant"), "retention": commit.get("retention"), "pairs": D.get("pairs")})
    if cs and ts:
        out["control_mean_s"] = sum(cs) / len(cs)
        out["treatment_mean_s"] = sum(ts) / len(ts)
        out["overhead_rel"] = out["treatment_mean_s"] / out["control_mean_s"] - 1 if out["control_mean_s"] else None
    missing = []
    if out["rounds"] < 3:
        missing.append(f"≥3 interleaved control/treatment pairs (have {out['rounds']})")
    if not commit.get("retention"):
        missing.append("retention label")
    if commit.get("status") != "PASS":
        out["reasons"].append(f"overhead of a Commit that is {commit.get('status')} under r14-v1 — shown ◐ with its number")
        missing.append("Commit ✅ of the same attempt")
    out["status"] = "DONE" if not missing else "PARTIAL"
    if missing:
        out["reasons"].append("missing: " + ", ".join(missing))
    return out


# ----------------------------------------------------------------------------------------------------------------


def _drop_missing(d: dict, item: str) -> None:
    """remove `item` from the 'missing: ...' reason; promote PARTIAL -> PASS when nothing is missing any more."""
    rs = []
    for r in d.get("reasons", []):
        if r.startswith("missing: "):
            items = [x.strip() for x in r[len("missing: "):].split(",") if x.strip() and x.strip() != item]
            if items:
                rs.append("missing: " + ", ".join(items))
        else:
            rs.append(r)
    d["reasons"] = rs
    if d.get("status") == "PARTIAL" and not any(r.startswith("missing: ") for r in rs):
        d["status"] = "PASS"


def recorded_fields(row: dict, evdir: str) -> dict:
    """Recorded commands / settings from the row attempt whose artifacts point at this evidence directory (brief §5:
    repair only from contemporaneous artifacts — launch logs, configs, manifests, recorded commands)."""
    key = os.path.basename(os.path.normpath(evdir))
    out: dict = {}
    cands = [a for a in (row.get("attempts") or []) if isinstance(a, dict) and key in json.dumps(a, default=str)]
    if not cands and isinstance(row.get("build"), dict) and key in json.dumps({k: row.get(k) for k in ("build", "match", "commit", "bench")}, default=str):
        cands = [row]
    if not cands:
        return out
    a = cands[-1]
    for stg, k in (("build", "build_cmd"), ("match", "match_cmd"), ("commit", "commit_cmd")):
        d = a.get(stg) if isinstance(a.get(stg), dict) else {}
        if d.get("cmd"):
            out[k] = d["cmd"]
    mm = a.get("match") if isinstance(a.get("match"), dict) else {}
    if mm.get("comparator"):
        out["match_comparator"] = mm["comparator"]
    b = a.get("build") if isinstance(a.get("build"), dict) else {}
    if b.get("target") or b.get("target_profile"):
        out["build_target"] = b.get("target") or b.get("target_profile")
    txt = " ".join(str(x) for x in (a.get("cmd"), json.dumps(a.get("env") or {}), (a.get("commit") or {}).get("cmd") if isinstance(a.get("commit"), dict) else "", (a.get("commit") or {}).get("retention") if isinstance(a.get("commit"), dict) else ""))
    m = re.search(r"--retain\s+(host|device|none|sample)|VERITY_RETAIN=(host|device|none|sample)|retention[\"': ]+(host|device|none|sample)", txt)
    if m:
        out["retention"] = next(g for g in m.groups() if g)
    elif "WINDOW_MB=0" in txt or "whole-block" in txt:
        out["retention"] = "device"
    if re.search(r"VERITY_WINDOW_(MB|SLOTS)=(?!0\b)\d+|--window-(mb|slots)", txt):
        out["collector_variant"] = "windowed"
    elif "WINDOW_MB=0" in txt:
        out["collector_variant"] = "whole-block"
    return out


def status_from(stages: dict) -> str:
    b, m, c, t = (stages[k]["status"] for k in ("build", "match", "commit", "bench"))
    if b == "FAIL":
        return "BUILD_FAIL"
    if m == "FAIL":
        return "MATCH_FAIL"
    if c == "FAIL":
        return "COMMIT_FAIL"
    if t == "DONE" and c == "PASS":
        return "BENCHMARKED"
    if c in ("PASS", "PARTIAL"):
        return "COMMIT_PASS" if c == "PASS" else "COMMIT_PARTIAL"
    if m in ("PASS", "PARTIAL"):
        return "MATCH_PASS" if m == "PASS" else "MATCH_PARTIAL"
    if b in ("PASS", "PARTIAL"):
        return "BUILD_PASS" if b == "PASS" else "BUILD_PARTIAL"
    return "NOT_RUN"


def derive_infeasible(evdir: str) -> dict | None:
    """COORD 19:25Z ruling 6: a row that cannot serve on its declared hardware is `infeasible` with the exact error and the attempted
    configuration — a legitimate terminal state, never a substitute configuration.  Evidence: infeasible.json {stage, error, configuration}."""
    p = first(evdir, "infeasible.json", "commit/infeasible.json")
    if not p:
        return None
    try:
        J = load(p)
    except Exception as e:
        return {"stage": "build", "error": f"unreadable infeasible.json: {e}", "configuration": None, "artifact": os.path.relpath(p)}
    if not isinstance(J, dict):
        return None
    return {"stage": str(J.get("stage") or "build").lower(), "error": str(J.get("error") or J.get("first_error") or "")[:600],
            "configuration": J.get("configuration") or J.get("attempted_configuration"), "artifact": os.path.relpath(p)}


def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def regression_marker(row: dict, att: dict) -> dict | None:
    """brief §9: a regression stays visible — when this (latest-evidence) attempt shows a stage worse than an earlier finalizer
    attempt of the same row, the record says so (the renderer prints the caveat beneath the table)."""
    rank = {"PASS": 3, "DONE": 3, "PARTIAL": 2, "INFEASIBLE": 1, "FAIL": 1, "-": 0, None: 0}
    prev = [a for a in (row.get("attempts") or []) if isinstance(a, dict) and a.get("finalizer") and a.get("evidence_dir") != att.get("evidence_dir")]
    if not prev:
        return None
    worse = []
    for st in ("build", "match", "commit", "bench"):
        best = max(prev, key=lambda a: rank.get((a.get(st) or {}).get("status"), 0))
        bs = (best.get(st) or {}).get("status")
        ns = (att.get(st) or {}).get("status")
        if rank.get(bs, 0) == 3 and rank.get(ns, 0) < 3 and ns not in (None, "-"):
            worse.append({"stage": st, "was": bs, "now": ns, "earlier_attempt": {"at": best.get("at"), "release_tag": best.get("release_tag"), "revision": best.get("revision"), "evidence_dir": best.get("evidence_dir")}})
    return {"stages": worse, "note": "regression vs an earlier finalizer attempt of this row (brief §9: the newer result is displayed; the earlier green is not shown over it)"} if worse else None


def unsuperseded_fail_marker(row: dict, att: dict) -> dict | None:
    """brief §9 (the other direction): a NEWER attempt that never reached a stage must not hide an earlier attempt's ❌ at that stage.
    For every stage this attempt shows as not reached ('-'/None), the latest earlier finalizer attempt WITH evidence at that stage is
    looked up; if it FAILED (and no attempt after it passed the stage) the failure is carried on this attempt (rendered '— (earlier ❌)'
    with the reason beneath the table) until a later attempt passes or fails the stage itself."""
    prev = [a for a in (row.get("attempts") or []) if isinstance(a, dict) and a.get("finalizer") and a.get("evidence_dir") != att.get("evidence_dir")]
    if not prev:
        return None
    prev.sort(key=lambda a: str(a.get("finalized_at") or a.get("at") or ""))
    carried = []
    for st in ("build", "match", "commit"):
        ns = (att.get(st) or {}).get("status")
        if ns not in (None, "-"):
            continue
        with_ev = [a for a in prev if (a.get(st) or {}).get("status") not in (None, "-")]
        if not with_ev:
            continue
        last = with_ev[-1]
        ls = (last.get(st) or {}).get("status")
        if ls == "FAIL":
            carried.append({"stage": st, "was": ls, "reasons": list(((last.get(st) or {}).get("reasons") or [])[:2]),
                            "earlier_attempt": {"at": last.get("at"), "release_tag": last.get("release_tag"), "revision": last.get("revision"), "evidence_dir": last.get("evidence_dir")}})
    return {"stages": carried, "note": "an earlier attempt of this row FAILED a stage this attempt did not reach (brief §9: the ❌ stays visible until a later attempt passes or fails the stage itself)"} if carried else None


NO_GRAPH = ("", "NONE", "OFF", "0", "FALSE")


def _bool3(v):
    """bool | None from a bool/str/int (rev-pm M-0081: 'False' strings must not read as truthy)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes", "y"):
            return True
        if t in ("false", "0", "no", "n"):
            return False
    return None


def _exec_fact(src: str, d: dict, engine_record: bool) -> dict | None:
    cc = d.get("compilation_config") if isinstance(d.get("compilation_config"), dict) else {}
    mode = cc.get("cudagraph_mode") or d.get("cudagraph_mode") or d.get("cudagraph") or cc.get("mode")
    comp = d.get("compilation_mode") or cc.get("compilation_mode") or cc.get("level") or d.get("compilation_level")
    ee = _bool3(d.get("enforce_eager"))
    label = d.get("execution") or d.get("execution_label") or d.get("label")
    if ee is None and mode is None and comp is None and not label:
        return None
    return {"source": src, "engine_record": engine_record, "enforce_eager": ee, "cudagraph_mode": (str(mode).upper() if mode is not None else None),
            "compilation_mode": (str(comp) if comp is not None else None), "label": (str(label) if isinstance(label, str) else None)}


def _jsonl_head(path: str, n: int = 8) -> list[dict]:
    """The first n JSON rows of a .jsonl / .jsonl.gz (the RunHeader is the first row of a capture log)."""
    rows: list[dict] = []
    try:
        import gzip
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
    except Exception:
        pass
    return rows


def run_header(path: str) -> dict | None:
    """The RunHeader of a capture log / runs.jsonl: the first row carrying `profile` (the recorder nests the engine facts under
    RunHeader.profile.engine_kwargs / profile.compilation — rev-pm M-0261) or, in older logs, a `header` sub-dict."""
    for r in _jsonl_head(path):
        hdr = r.get("header") if isinstance(r.get("header"), dict) else r
        if isinstance(hdr, dict) and (isinstance(hdr.get("profile"), dict) or isinstance(hdr.get("engine_kwargs"), dict) or isinstance(hdr.get("compilation"), dict)):
            return hdr
    return None


MATCH_CAPTURE_LOGS = ("match/capture/log.jsonl.gz", "match/capture/log.jsonl", "match/log.jsonl.gz", "match/log.jsonl", "capture/log.jsonl.gz", "capture/log.jsonl")
MATCH_CAPTURE_VERSIONS = ("match/capture/versions.json", "capture/versions.json", "match/versions.json")
MATCH_HEADER_PINS = ("match/header_compilation_pin.json", "match/capture/header_compilation_pin.json", "header_compilation_pin.json")


def _compilation_fact(src: str, comp: dict) -> dict | None:
    """RunHeader.profile.compilation / versions.json compilation → an engine fact (mode = compilation mode, cudagraph_mode, enforce_eager)."""
    if not isinstance(comp, dict):
        return None
    d = {"enforce_eager": comp.get("enforce_eager"), "cudagraph_mode": comp.get("cudagraph_mode"),
         "compilation_mode": comp.get("compilation_mode") if comp.get("compilation_mode") is not None else comp.get("mode")}
    return _exec_fact(src, {k: v for k, v in d.items() if v is not None}, True)


def execution_attestation(evdir: str, A: dict) -> dict | None:
    """The effective execution the capture ran with (rev-pm M-0081 / COORD M-0063 (2)(3)):
    * the attestation comes from the ENGINE's record only — the capture run header engine_kwargs (commit/runs.jsonl header rows,
      commit/versions.json engine_kwargs, match/build run headers, the Match capture's RunHeader.profile.{engine_kwargs,compilation}
      and match/capture/versions.json — rev-pm M-0261/M-0266); runner-written summaries (commit_summary.json EXEC-01 block,
      build_summary applicability, attempt.json execution) are cross-checked against it and never stand alone;
    * every present source must agree on enforce_eager / cudagraph_mode — a disagreement is a named conflict (rendered ❌);
    * `compiled` needs POSITIVE evidence: enforce_eager is False AND a cudagraph/compilation mode other than NONE/OFF; a label alone is
      not evidence.  Returns None when no engine record attests anything."""
    engine: list[dict] = []
    other: list[dict] = []
    vjs = [p for p in (first(evdir, "commit/versions.json", "versions.json"), first(evdir, *MATCH_CAPTURE_VERSIONS)) if p]
    # compiled_commit.sh arms (control / treatment processes — COORD M-0178 (1) 'separate processes, paired by index')
    vjs += sorted(p for pat in ("commit_control*/versions.json", "commit_treatment*/versions.json") for p in glob.glob(os.path.join(evdir, pat)) if os.path.isfile(p))
    for vj in vjs:
        try:
            V = load(vj)
            f = _exec_fact(f"{os.path.relpath(vj, evdir)} engine_kwargs", V.get("engine_kwargs") or {}, True) if isinstance(V.get("engine_kwargs"), dict) else None
            if f:
                engine.append(f)
            # EXEC-02 (M-0094): effective_execution is read back from the engine's vllm_config after build — an engine record
            if isinstance(V.get("effective_execution"), dict):
                f = _exec_fact(f"{os.path.relpath(vj, evdir)} effective_execution", V["effective_execution"], True)
                if f:
                    engine.append(f)
            if isinstance(V.get("compilation"), dict):
                f = _compilation_fact(f"{os.path.relpath(vj, evdir)} compilation", V["compilation"])
                if f:
                    engine.append(f)
            if isinstance(V.get("declared_execution"), str) and V["declared_execution"]:
                other.append({"source": f"{os.path.relpath(vj, evdir)} declared_execution", "engine_record": False, "enforce_eager": None, "cudagraph_mode": None,
                              "compilation_mode": None, "label": V["declared_execution"]})
        except Exception:
            pass
    hp = first(evdir, *MATCH_HEADER_PINS)
    if hp:
        try:
            H = load(hp)
            if isinstance(H.get("engine_kwargs"), dict):
                f = _exec_fact(f"{os.path.relpath(hp, evdir)} engine_kwargs", H["engine_kwargs"], True)
                if f:
                    engine.append(f)
            pc = H.get("profile_compilation") if isinstance(H.get("profile_compilation"), dict) else ((H.get("profile") or {}).get("compilation") if isinstance(H.get("profile"), dict) else None)
            if isinstance(pc, dict):
                f = _compilation_fact(f"{os.path.relpath(hp, evdir)} profile_compilation", pc)
                if f:
                    engine.append(f)
        except Exception:
            pass
    logs = [p for p in (first(evdir, "commit/runs.jsonl", "runs.jsonl"), first(evdir, "match/runs.jsonl"), first(evdir, "build_step/runs.jsonl"), first(evdir, *MATCH_CAPTURE_LOGS)) if p]
    for rj in logs:
        hdr = run_header(rj)
        if not hdr:
            continue
        for k in ("engine_kwargs", "engine_args"):
            v = hdr.get(k)
            if isinstance(v, dict):
                f = _exec_fact(f"{os.path.relpath(rj, evdir)} {k}", v, True)
                if f:
                    engine.append(f)
        if isinstance(hdr.get("compilation"), dict):
            f = _compilation_fact(f"{os.path.relpath(rj, evdir)} compilation", hdr["compilation"])
            if f:
                engine.append(f)
        # the recorder's location (vllm_adapter build_manifest → RunHeader.profile.*; rev-pm M-0261)
        pr = hdr.get("profile") if isinstance(hdr.get("profile"), dict) else {}
        if isinstance(pr.get("engine_kwargs"), dict):
            f = _exec_fact(f"{os.path.relpath(rj, evdir)} profile.engine_kwargs", pr["engine_kwargs"], True)
            if f:
                engine.append(f)
        if isinstance(pr.get("compilation"), dict):
            f = _compilation_fact(f"{os.path.relpath(rj, evdir)} profile.compilation", pr["compilation"])
            if f:
                engine.append(f)
    cs = first(evdir, "commit_summary.json")
    if cs:
        try:
            C = load(cs)
            for k in ("execution", "engine_args", "engine_kwargs", "execution_config", "engine_settings"):
                if isinstance(C.get(k), dict):
                    f = _exec_fact(f"commit_summary.json {k}", C[k], False)
                    if f:
                        other.append(f)
        except Exception:
            pass
    bs = first(evdir, "build_summary.json")
    if bs:
        try:
            B = load(bs)
            ap = ((B.get("step") or {}).get("applicability") or {}) if isinstance(B.get("step"), dict) else {}
            if isinstance(ap, dict):
                f = _exec_fact("build_summary.json step.applicability", ap, False)
                if f:
                    other.append(f)
        except Exception:
            pass
    if isinstance(A.get("execution"), dict):
        f = _exec_fact("attempt.json execution", A["execution"], False)
        if f:
            other.append(f)
    if not engine:
        return None    # summaries alone are not an attestation of what the engine ran
    facts = engine + other
    ees = {f["enforce_eager"] for f in facts if f["enforce_eager"] is not None}
    modes = {f["cudagraph_mode"] for f in facts if f["cudagraph_mode"] is not None}
    modes_n = {("NONE" if m in NO_GRAPH else m) for m in modes}
    conflict = None
    if len(ees) > 1:
        conflict = "enforce_eager disagrees: " + "; ".join(f"{f['source']}={f['enforce_eager']}" for f in facts if f["enforce_eager"] is not None)
    elif len(modes_n) > 1:
        conflict = "cudagraph_mode disagrees: " + "; ".join(f"{f['source']}={f['cudagraph_mode']}" for f in facts if f["cudagraph_mode"] is not None)
    else:
        for f in other:
            lab = (f.get("label") or "").lower()
            if lab and "compiled" in lab and (True in ees or (not ees and modes_n <= {"NONE"})):
                conflict = f"{f['source']} labels the run `{f['label']}` but the engine record attests eager"
                break
            if lab and "eager" in lab and "compiled" not in lab and False in ees and modes_n - {"NONE"}:
                conflict = f"{f['source']} labels the run `{f['label']}` but the engine record attests a compiled engine"
                break
    ee = next(iter(ees)) if len(ees) == 1 else None
    mode = next(iter(modes_n)) if len(modes_n) == 1 else None
    comp = next((f["compilation_mode"] for f in facts if f["compilation_mode"]), None)
    graph_on = (mode is not None and mode != "NONE") or (comp is not None and str(comp).upper() not in ("", "NONE", "0", "NO_COMPILATION"))
    if ee is True or (ee is None and mode == "NONE"):
        verdict = "eager"
    elif ee is False and graph_on:
        verdict = "compiled"
    elif ee is False:
        verdict = "indeterminate"     # enforce_eager False alone is not positive evidence of a compiled execution
    else:
        verdict = "indeterminate"
    decl = next((f["label"] for f in other if f["source"].endswith("declared_execution") and f.get("label")), None)
    return {"verdict": verdict, "conflict": conflict, "enforce_eager": ee, "cudagraph_mode": mode, "compilation_mode": comp, "declared_execution": decl,
            "sources": [f["source"] for f in facts], "engine_sources": [f["source"] for f in engine], "source": engine[0]["source"]}


# [rev M-0020 (1)] released commits of record: the release tags in this repo (rel tags r1x-* on the release branch) + the RELEASE posts
# of the campaign boards + the table of record for releases whose tag object is not in this checkout.  Equality on the commit id.
RELEASED_COMMITS_OF_RECORD = {"f782746e4": "r14-base", "db970aff6": "r14-rc1", "914fa4067": "r14-rc3", "e59dd9016": "r15-int-1", "a2eb23163": "r15-int-2",
                              "cefd1b5ca": "r15-int-3", "a702b2974": "r15-int-4", "53c94ccbf": "r15-int-4b", "3faa98ed3": "r15-int-5", "4ef1198b6": "r15-int-6",
                              "22668e4f6": "r15-int-6b"}
_RELEASED_CACHE: dict[str, str] | None = None


def released_commits() -> dict[str, str]:
    """{sha(≥9): tag} of every released commit known here — repo tags r1\\d-(rc|int|final)*, the boards' RELEASE posts (R15 + R16), the static table."""
    global _RELEASED_CACHE
    if _RELEASED_CACHE is not None:
        return _RELEASED_CACHE
    out = dict(RELEASED_COMMITS_OF_RECORD)
    try:
        r = subprocess.run(["git", "for-each-ref", "refs/tags/r1*", "--format=%(*objectname) %(objectname) %(refname:short)"], capture_output=True, text=True, timeout=20)
        for l in r.stdout.splitlines():
            parts = l.split()
            if len(parts) == 3 and re.fullmatch(r"r1\d-(rc\d+[a-z]?|int-\d+[a-z]?|final[\w-]*|base)", parts[2]):
                sha = parts[0] or parts[1]
                out[sha[:9]] = parts[2]
    except Exception:  # noqa: BLE001
        pass
    tagsha = r"(r1\d-(?:rc\d+[a-z]?|int-\d+[a-z]?|final[\w-]*))\b\s*[=(:]?\s*(?:lane/[\w\-./]+\s+)?([0-9a-f]{7,40})\b"
    for board in (os.environ.get("R16_BOARD", "/private/tmp/r16/board"), os.environ.get("R15_BOARD", "/private/tmp/r15/board")):
        d = os.path.join(board, "events")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                e = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if e.get("type") != "RELEASE":
                continue
            txt = str(e.get("text") or "")
            m = re.search(r"\bRELEASED?\s+" + tagsha, txt) or re.search(r"\b" + tagsha, txt)
            if m and not re.search(r"\b(LAST-KNOWN-GOOD|superseded|not released|withdrawn|NOT RELEASED)\b", txt[: m.start()], re.I):
                out[m.group(2)[:9]] = m.group(1)
    _RELEASED_CACHE = out
    return out


def released_tag_of(commit: str | None) -> str | None:
    """The release tag whose commit equals `commit` (prefix equality ≥ 7 hex), else None."""
    if not commit or not re.fullmatch(r"[0-9a-f]{7,40}", str(commit)):
        return None
    c = str(commit)
    for sha, tag in released_commits().items():
        if sha.startswith(c[:9]) or c.startswith(sha[:9]) or (len(c) >= 7 and sha.startswith(c[:7]) and len(c) <= 9):
            return tag
    return None


def executing_tree(A: dict, repo_path: str | None, base: str | None, tree_txt: str) -> dict:
    """COORD M-0132: the tree whose code executed.  Reads attempt.json executing_tree {path, commit, dirty, diff_vs_release} when exec
    writes it, else parses the lane-tree commit from attempt.json `tree` ('… lane/r15-x <sha> …' | '<path> @ <sha>'); then CHECKS the
    code diff vs the release in the report repo (git diff --name-only base commit -- verity_capture, tests excluded) — the producer's
    'code identical' is a claim, the diff is the evidence.  diff_vs_release: [] (empty, finalizable) | [files] | None (uncheckable)."""
    et = A.get("executing_tree") if isinstance(A.get("executing_tree"), dict) else {}
    out = {"lane_tree": True, "path": et.get("path") or repo_path, "base": base, "commit": None, "diff_vs_release": None, "producer_claim": et.get("diff_vs_release") or et.get("diff") or et.get("code_identical"),
           "dirty": et.get("dirty"), "checked_by": "finalize: git diff --name-only <base> <commit> -- verity_capture (paths under /tests/ and test_*.py excluded)"}
    commit = et.get("commit") or et.get("git_commit")
    if not (isinstance(commit, str) and SHA_RE.match(commit.strip())):
        m = (re.search(r"UNRELEASED\S*\s.*?commit=([0-9a-f]{7,40})\b", tree_txt) or re.search(r"lane/r1\d-[\w-]+\s+([0-9a-f]{7,40})\b", tree_txt)
             or re.search(r"@\s*([0-9a-f]{7,40})\b", tree_txt))
        commit = m.group(1) if m else None
        if not commit:
            # row.log's own `code=<sha>` stamps quoted in `tree` (global M-0520): one per stage; every distinct commit is diffed below
            codes = []
            for c in re.findall(r"\bcode=([0-9a-f]{7,40})\b", tree_txt):
                if c not in codes:
                    codes.append(c)
            if codes:
                commit = codes[-1]
                if len(codes) > 1:
                    out["commits"] = codes
    out["commit"] = commit.strip() if isinstance(commit, str) else None
    # dirty: a boolean, or the dirty-patch digest (REL-14 `dirty=<d12>`; the empty patch is sha256 e3b0c442…)
    dv = out["dirty"]
    if dv is None:
        md = re.search(r"\bdirty=(\S+)", tree_txt) or re.search(r"dirty_patch_sha256[\"']?:\s*[\"']?([0-9a-f]+)", tree_txt)
        dv = md.group(1) if md else None
    if isinstance(dv, str):
        t = dv.strip().strip('",').lower()
        out["dirty"] = not (t in ("", "-", "0", "none", "clean", "false", "no") or t.startswith("e3b0c442"))
        out["dirty_digest"] = t
    elif isinstance(dv, bool):
        out["dirty"] = dv
    # the release base: a release tag named by the producer (tree / tree_note / stages.txt) resolved in the repo, else the declared
    # base commit — never the lane-tree commit itself (a lane that writes release_commit = its own commit has no base recorded)
    def _same(a: str | None, b: str | None) -> bool:
        return bool(a and b and (a.startswith(b) or b.startswith(a)))
    resolved = None
    mt = re.search(r"\b(r1\d-(?:rc\d+[a-z]?|int-\d+[a-z]?|final[\w-]*))\b", tree_txt)
    if mt:
        try:
            import subprocess
            r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{mt.group(1)}^{{commit}}"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                resolved = r.stdout.strip()
                out["base_tag"] = mt.group(1)
        except Exception:
            resolved = None
    if resolved and not _same(resolved, out["commit"]):
        base = resolved
    elif _same(base, out["commit"]):
        base = None
    out["base"] = base
    if out["dirty"] is True:
        out["diff_error"] = "the executing tree was dirty (uncommitted changes) — its commit does not describe the code that ran"
        return out
    if not out["commit"]:
        out["diff_error"] = "no lane-tree commit recorded (attempt.json executing_tree.commit or `tree` '… lane/r15-x <sha>')"
        return out
    if not base:
        out["diff_error"] = "no release base commit recorded to diff against"
        return out
    try:
        import subprocess
        files_all: list[str] = []
        for c in (out.get("commits") or [out["commit"]]):
            r = subprocess.run(["git", "diff", "--name-only", base, c, "--", "verity_capture"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                out["diff_error"] = f"git diff {base[:9]}..{c[:9]} failed: {r.stderr.strip()[:160]}"
                return out
            files_all += [f for f in r.stdout.split("\n") if f.strip() and f not in files_all]
        if out.get("commits"):
            out["diff_note"] = f"stages ran on {len(out['commits'])} lane-tree commits ({', '.join(c[:9] for c in out['commits'])}); the diff is the union"
        files = [f for f in files_all if f and "/tests/" not in f and not os.path.basename(f).startswith("test_")]
        out["diff_vs_release"] = sorted(files)
        out["diff_all_files"] = len([f for f in files_all if f])
    except Exception as e:      # noqa: BLE001
        out["diff_error"] = f"{type(e).__name__}: {e}"[:160]
    return out


def _fail(d: dict, why: str) -> None:
    if d.get("status") in ("PASS", "PARTIAL"):
        d["status"] = "FAIL"
        d["reasons"].append(why)
    elif d.get("status") == "DONE":        # the overhead has no FAIL: a failed Commit leaves its number shown ◐
        d["status"] = "PARTIAL"
        d["reasons"].append(why)


def _partial(d: dict, why: str) -> None:
    if d.get("status") in ("PASS", "DONE"):
        d["status"] = "PARTIAL"
        d["reasons"].append(why)
        if d.get("c2_form"):
            # [tp M-0693] a Commit whose C2 record is COMPLETE but capped by the tree rule says so — not 'C2 record pending'
            d["derived_before_cap"] = "PASS"
            d["reasons"].append(f"C2 record complete on this attempt (form: {d['c2_form']}{'; weights pinned' if (d.get('weights_pin') or {}).get('ok') else ''}) — the ◐ is the tree rule above, not a missing C2 field")


def _cap(d: dict, why: str) -> None:
    """◐ with the reason attached even when the stage is already ◐ (a second missing field is still named; ❌ / — untouched)."""
    if d.get("status") in ("PASS", "DONE"):
        d["status"] = "PARTIAL"
    if d.get("status") == "PARTIAL":
        d["reasons"].append(why)


# Board-ordered HOLDS on displayed cells: a reviewer's ACTION line naming report (M-0424 'rel/report: MoE TP2 Commit cells not green on
# 7e133a868').  Not a TABLE-RULES change — the cell is ◐ with the hold and the lifting record named; the entry is retired here when the
# fix's RESULT lands.  A hold is lifted per ATTEMPT by the named evidence record, never by the producer's verdict.
MOE_MODELS = {"olmoe-1b-7b", "qwen3-30b-a3b"}          # validate.MODEL_PANEL MoE flag (kept local: finalize does not import validate)


def _tp_of(row_id: str, row: dict) -> int:
    m = re.search(r"__tp(\d+)__", row_id)
    try:
        return int(m.group(1)) if m else int((row.get("config") or {}).get("tp") or 1)
    except Exception:  # noqa: BLE001
        return 1


def _t6_4_lifted(evdir: str) -> tuple[bool, str]:
    """[rev-cb M-0424 T6-4] lifted for an attempt whose Commit record names a VALUE check of <L>.mlp.experts/out against
    <L+1>.input_layernorm/0 (last layer: model.norm/0) — commit/t6_4_check.json | commit/moe_out_lifetime_check.json
    {result: true, layers_compared: n>0, equal_pairs: 0}.  The late read is a value fact (C1 name+width and C3 roots-reproduce pass on it),
    so only a value record lifts the hold."""
    p = first(evdir, "commit/t6_4_check.json", "commit/moe_out_lifetime_check.json", "t6_4_check.json", "moe_out_lifetime_check.json")
    if not p:
        return False, "no commit/t6_4_check.json on this attempt"
    try:
        J = load(p)
    except Exception as e:  # noqa: BLE001
        return False, f"{os.path.relpath(p)} unreadable ({e})"
    def _int(*keys, default=-1):
        for k in keys:
            if J.get(k) is not None:
                try:
                    return int(J.get(k))
                except Exception:  # noqa: BLE001
                    return default
        return default
    lc, eq, miss = _int("layers_compared", default=0), _int("equal_pairs"), _int("missing_pairs", default=0)
    # [rev-cb M-0456, measured on int-3] at decode the late-staged bytes are neither the value nor the next norm: 'equal to next-norm/0'
    # alone under-counts; the reference is the BIND-TIME CLONE (tp's _bytes(y)) — the record must count committed != clone as late reads too
    clone = _int("clone_mismatches", "mismatch_vs_clone", "late_reads", "unequal_to_clone")
    if as_bool(J.get("result")) is True and lc > 0 and eq == 0 and miss == 0 and clone == 0:
        return True, f"{os.path.relpath(p)}: {lc} layers compared — 0 pairs equal to next-norm/0, 0 committed≠bind-time-clone (M-0456), 0 missing"
    why = f"{os.path.relpath(p)}: result {J.get('result')!r}, layers_compared {J.get('layers_compared')!r}, equal_pairs {J.get('equal_pairs')!r}, missing_pairs {J.get('missing_pairs')!r}"
    if clone < 0:
        why += "; bind-time-clone comparison not on the record (clone_mismatches | mismatch_vs_clone | late_reads — rev-cb M-0456: decode-step late reads are unequal to BOTH the value and the next norm)"
    else:
        why += f", committed≠clone {clone}"
    return False, why


ROW_HOLDS: list[dict] = [
    {"id": "T6-4", "stage": "commit",
     "applies": lambda row_id, row, wlf: row_id.split("__")[0] in MOE_MODELS and _tp_of(row_id, row) >= 2,
     "lifted": _t6_4_lifted,
     "reason": ("HOLD (rev-cb M-0424 T6-4, 22:38Z — ACTION 'report: MoE TP2 Commit cells not green on 7e133a868'): the TP-06b binder stages the MoE received "
                "all-reduce output <L>.mlp.experts/out with no flush point (FusedMoE has no down_proj hook; flush leaves are qkv_proj/o_proj/down_proj + step end), "
                "so fused_add_rms_norm overwrites it in place before the next flush — the committed bytes may be <L+1>.input_layernorm/0 (last layer: model.norm/0), "
                "a late read that C1 (name + width) and C3 (roots reproduce) cannot see; Commit ✅ needs tp's flush/collect fix (RESULT) AND a value check of /out vs "
                "next-norm/0 on the attempt (commit/t6_4_check.json {result true, layers_compared, equal_pairs 0})")},
]


def apply_row_holds(row_id: str, row: dict, wlf: dict, commit: dict, evdir: str) -> None:
    for h in ROW_HOLDS:
        try:
            ok = bool(h["applies"](row_id, row, wlf))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            continue
        lifted, how = h["lifted"](evdir)
        if lifted:
            commit.setdefault("footnotes", []).append(f"hold {h['id']} (rev-cb M-0424) lifted on this attempt: {how}")
            continue
        if commit.get("status") in ("PASS", "DONE", "PARTIAL"):
            _cap(commit, h["reason"] + f" — this attempt: {how}")
            commit.setdefault("holds", []).append(h["id"])


# derive_match FAIL reasons that are the KERNEL-LEVEL structure comparison (X-09 / GM-01 G1–G8 over the fold) — on a compiled record
# graph replay attributes 0 fold instances, so these carry no information (rev-pm M-0266 F-pm-22, M-0268 (2)): not evaluable, not ❌
KERNEL_LEVEL_FAIL = re.compile(r"^(comparator (verdict|rc=)|global checks failed|unattributed instances|global Match verdict|global check \w+ = False|requests present)")


def compiled_record_evidence(evdir: str) -> dict:
    """Record-level positive evidence an EAGER run cannot produce (rev-pm M-0266 (i), M-0219 pin, EXEC-02 read-back):
    * kernel_pin_match: RunHeader.profile.compilation.generated_kernels.kernel_set_sha256 of the MATCH record (capture log header /
      match/capture/versions.json) — the Inductor output-code identity; null is the recorder's placeholder, not a pin;
    * kernel_pin_commit: the same digest in the Commit arms (commit*/versions.json compilation.generated_kernels, inductor_kernels.json
      summary) — commit_delta's pin, which does not stand for the Match record (M-0266: 'populate it in the Match/record path');
    * effective_execution: match/capture/versions.json effective_execution.execution_label (EXEC-02 read-back of the built engine);
    * value_check: the compiled_value_check relations record (rev-cb M-0221); tokens_equal_arms: commit_delta's tokens_equal between the
      two compiled arms (control / treatment) with both arms' enforce_eager."""
    out: dict = {"kernel_pin_match": None, "kernel_pin_match_source": None, "kernel_pin_commit": None, "kernel_pin_commit_source": None,
                 "effective_execution": None, "effective_execution_source": None, "value_check": None, "tokens_equal_arms": None, "arms_enforce_eager": []}

    def _pin(comp) -> str | None:
        gk = comp.get("generated_kernels") if isinstance(comp, dict) and isinstance(comp.get("generated_kernels"), dict) else (comp if isinstance(comp, dict) else {})
        k = gk.get("kernel_set_sha256")
        return k.strip() if isinstance(k, str) and k.strip() and k.strip().lower() not in ("null", "none") else None

    lg = first(evdir, *MATCH_CAPTURE_LOGS)
    if lg:
        hdr = run_header(lg) or {}
        pr = hdr.get("profile") if isinstance(hdr.get("profile"), dict) else {}
        p = _pin(pr.get("compilation")) or _pin(hdr.get("compilation"))
        if p:
            out["kernel_pin_match"], out["kernel_pin_match_source"] = p, os.path.relpath(lg, evdir) + " RunHeader.profile.compilation.generated_kernels"
        out["match_header"] = os.path.relpath(lg, evdir)
    # the recorder's header extract synced when the capture log is not (compiled r15_2137: match/header_compilation_pin.json
    # {profile_id, profile_compilation{generated_kernels{kernel_set_sha256}}, engine_kwargs}) — same self-attested RunHeader facts
    hp = first(evdir, *MATCH_HEADER_PINS)
    if hp and not out["kernel_pin_match"]:
        try:
            H = load(hp)
            p = _pin(H.get("profile_compilation")) or _pin((H.get("profile") or {}).get("compilation") if isinstance(H.get("profile"), dict) else None) or _pin(H.get("compilation"))
            if p:
                out["kernel_pin_match"], out["kernel_pin_match_source"] = p, os.path.relpath(hp, evdir) + " profile_compilation.generated_kernels"
        except Exception:
            pass
    vj = first(evdir, *MATCH_CAPTURE_VERSIONS)
    if vj:
        try:
            V = load(vj)
            if not out["kernel_pin_match"] and _pin(V.get("compilation")):
                out["kernel_pin_match"], out["kernel_pin_match_source"] = _pin(V.get("compilation")), os.path.relpath(vj, evdir) + " compilation.generated_kernels"
            ee = V.get("effective_execution") if isinstance(V.get("effective_execution"), dict) else {}
            lab = ee.get("execution_label") or ee.get("execution") or ee.get("label")
            if isinstance(lab, str) and lab.strip():
                out["effective_execution"], out["effective_execution_source"] = lab.strip(), os.path.relpath(vj, evdir) + " effective_execution.execution_label"
        except Exception:
            pass
    for pat in ("commit*/versions.json", "commit/versions.json", "versions.json"):
        for p in sorted(glob.glob(os.path.join(evdir, pat))):
            try:
                V = load(p)
            except Exception:
                continue
            k = _pin(V.get("compilation"))
            if k and not out["kernel_pin_commit"]:
                out["kernel_pin_commit"], out["kernel_pin_commit_source"] = k, os.path.relpath(p, evdir) + " compilation.generated_kernels"
            if re.search(r"commit_(control|treatment)", p):
                ee = _bool3((V.get("engine_kwargs") or {}).get("enforce_eager")) if isinstance(V.get("engine_kwargs"), dict) else None
                if ee is None and isinstance(V.get("compilation"), dict):
                    ee = _bool3(V["compilation"].get("enforce_eager"))
                out["arms_enforce_eager"].append({"arm": os.path.basename(os.path.dirname(p)), "enforce_eager": ee})
    if not out["kernel_pin_commit"]:
        for p in sorted(glob.glob(os.path.join(evdir, "*/inductor_kernels.json")) + glob.glob(os.path.join(evdir, "inductor_kernels.json"))):
            try:
                K = load(p)
                k = _pin(K.get("summary") if isinstance(K.get("summary"), dict) else K)
                if k:
                    out["kernel_pin_commit"], out["kernel_pin_commit_source"] = k, os.path.relpath(p, evdir) + " summary.kernel_set_sha256"
                    break
            except Exception:
                continue
    vchk = first(evdir, "compiled_value_check.json", "commit/compiled_value_check.json", "*value_check*.json", "commit/*value_check*.json")
    out["value_check"] = os.path.relpath(vchk, evdir) if vchk else None
    cs = first(evdir, "commit_delta_summary.json", "commit/summary.json", "commit_summary.json")
    if cs:
        try:
            D = load(cs)
            te = D.get("tokens_equal")
            out["tokens_equal_arms"] = all(as_bool(x) for x in te) if isinstance(te, list) and te else as_bool(te)
        except Exception:
            pass
    return out


P_COMPILED_S2_SCHEMA = "verity/compiled-s2-checkpoints/v1"
P_COMPILED_MANIFEST_KIND = "verity/compiled-required-manifest/v1"
COMPILED_WEIGHTS_OF_RECORD_SCHEMA = "verity/commit/weights-of-record/v1"


def p_compiled_attempt(A: dict) -> bool:
    """[COORD RULING M-0819 (R15 02:44Z)] an Execution=compiled attempt whose producer DECLARES `attempt.json build.kind == P_compiled`:
    its artifacts of record are BUILD = the P_compiled derivation record (compiled_manifest.json), MATCH = the S2 checkpoint-replay record +
    the runtime↔artifact identity field, COMMIT = S3 per record with form (A).  Only the explicit declaration selects the branch — a compiled
    row finalized from an eager-shaped evidence dir keeps the eager rules (execution_guard)."""
    b = A.get("build") if isinstance(A.get("build"), dict) else {}
    return str(b.get("kind") or "").strip().lower().replace("-", "_") in ("p_compiled", "pcompiled", "compiled")


def compiled_manifest_of(evdir: str) -> tuple[dict | None, str | None, str | None]:
    """(manifest, relpath, error) — compiled_manifest.json[.gz] of THIS attempt (never a sibling dir's)."""
    p = first(evdir, "compiled_manifest.json", "compiled_manifest.json.gz", "p_compiled/compiled_manifest.json", "p_compiled/compiled_manifest.json.gz",
              "build_compiled/compiled_manifest.json", "build_compiled/compiled_manifest.json.gz")
    if not p:
        return None, None, None
    try:
        M = load(p)
    except Exception as e:  # noqa: BLE001
        return None, os.path.relpath(p), f"{type(e).__name__}: {e}"[:160]
    return (M if isinstance(M, dict) else None), os.path.relpath(p), (None if isinstance(M, dict) else "not a JSON object")


def compiled_commit_summary(evdir: str) -> dict:
    p = first(evdir, "commit/summary.json", "commit_delta_summary.json")
    if not p:
        return {}
    try:
        S = load(p)
        return S if isinstance(S, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def compiled_arms(evdir: str) -> list[dict]:
    """per-arm versions.json facts (commit_control_r*/ commit_treatment_r*/): enforce_eager, declared/effective execution, kernel_set, loaded identity."""
    arms = []
    for p in sorted(glob.glob(os.path.join(evdir, "commit_*_r*", "versions.json")) + glob.glob(os.path.join(evdir, "commit_*", "versions.json"))):
        try:
            V = load(p)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(V, dict):
            continue
        comp = V.get("compilation") if isinstance(V.get("compilation"), dict) else {}
        gk = comp.get("generated_kernels") if isinstance(comp.get("generated_kernels"), dict) else {}
        ee = V.get("effective_execution") if isinstance(V.get("effective_execution"), dict) else {}
        ek = V.get("engine_kwargs") if isinstance(V.get("engine_kwargs"), dict) else {}
        ef = _bool3(ek.get("enforce_eager")) if "enforce_eager" in ek else (_bool3(comp.get("enforce_eager")) if "enforce_eager" in comp else _bool3(V.get("enforce_eager")))
        arms.append({"arm": os.path.basename(os.path.dirname(p)), "file": os.path.relpath(p, evdir), "enforce_eager": ef,
                     "declared_execution": V.get("declared_execution"), "effective_execution": ee.get("execution_label") or ee.get("label"),
                     "cudagraph_mode": comp.get("cudagraph_mode") or ee.get("cudagraph_mode"), "compilation_mode": comp.get("compilation_mode") or ee.get("mode"),
                     "kernel_set_sha256": gk.get("kernel_set_sha256"), "per_process_cache_dir": gk.get("per_process_cache_dir"),
                     "compile_cache_disabled": comp.get("compile_cache_disabled"),
                     "loaded": gk.get("loaded") if isinstance(gk.get("loaded"), dict) else None})
    return arms


def compiled_weights_of_record(evdir: str, want_program_digest: str | None) -> tuple[dict | None, str | None]:
    """[M-0819 + COORD R16 ruling 5 'a missing metadata field on an otherwise sound record is repaired from evidence, logged'] the Build-side
    weights-of-record record a P_compiled attempt names through compiled_manifest components.weights_struct.program_digest (the bi-eager Build of
    record whose served-parameter fields the compiled kernels consume).  Read from this attempt (weights_of_record*.json), its row dir, or the
    lane's shared evidence/<lane>/weights_of_record/*.json — accepted ONLY when its program_digest equals the declared weights_struct digest and
    it is a pinned-checkpoint record (schema verity/commit/weights-of-record/v1, all_shards_pinned, not 'live').  Returns (record, relpath)."""
    if not want_program_digest:
        return None, None
    cands: list[str] = []
    for pat in ("weights_of_record.json", "weights_of_record*.json", "build_request*/weights_of_record.json", "commit/weights_of_record.json"):
        cands += sorted(glob.glob(os.path.join(evdir, pat)))
    rowdir = os.path.dirname(evdir.rstrip("/"))
    lanedir = os.path.dirname(rowdir)
    cands += sorted(glob.glob(os.path.join(rowdir, "weights_of_record*.json")))
    cands += sorted(glob.glob(os.path.join(lanedir, "weights_of_record", "*.json")))
    seen = set()
    for f in cands:
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        try:
            J = load(f)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(J, dict) or str(J.get("program_digest") or "") != str(want_program_digest):
            continue
        if str(J.get("source") or "").lower().startswith("live"):
            continue
        if not str(J.get("schema") or "").startswith("verity/commit/weights-of-record/"):
            continue
        if not J.get("root_of_record"):
            continue
        return J, os.path.relpath(f)
    return None, None


def derive_build_compiled(evdir: str, A: dict, wl: dict, wlf: dict, row_id: str, row: dict, build: dict) -> dict:
    """[COORD M-0819] BUILD of an Execution=compiled attempt = the P_compiled derivation record: compiled_manifest.json (program_digest, kernel_set /
    artifact identity, derived from the CONTROL arm's Inductor output code + split graph), complete over the declared workload schedule, dataflow
    wired (0 undefined), declared target == the row's execution, manifest_verify = the TREATMENT arm's own derivation gives the same program +
    manifest digest (compiled_merge program_digests.equal / commit/manifest_verify.json).  Build ✅ iff all of it is on the record and rev-pm's
    open dataflow finding does not stand on the tree of record; otherwise ◐ with every gap named.  Never ✅ from attempt.json prose alone."""
    ex = str((row.get("config") or {}).get("execution") or (row_id.split("__")[-1] if "__" in row_id else "")).strip()
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": [], "kind": "P_compiled",
           "compiled_build_basis": "P_compiled derivation record (COORD M-0819; DECISIONS-2 §7 S1)"}
    decl = A.get("build") if isinstance(A.get("build"), dict) else {}
    M, mp, merr = compiled_manifest_of(evdir)
    if M is None:
        out["status"] = "PARTIAL"
        out["reasons"].append("missing: compiled_manifest.json[.gz] (the P_compiled derivation record) in this attempt's evidence dir" + (f" — {mp}: {merr}" if merr else "")
                              + f"; attempt.json declares build.kind P_compiled program {str(decl.get('program_digest') or '?')[:12]}… but the artifact is what the finalizer grades")
        return out
    out["evidence"].append(mp)
    out["artifact"] = mp
    gaps: list[str] = []
    pd, ks, md = M.get("program_digest"), M.get("kernel_set_sha256"), M.get("manifest_digest")
    if not pd:
        gaps.append("compiled_manifest.json program_digest")
    if not ks:
        gaps.append("compiled_manifest.json kernel_set_sha256 (the Inductor artifact identity P_compiled was generated from)")
    if decl.get("program_digest") and pd and str(decl["program_digest"]) != str(pd):
        out["status"] = "FAIL"
        out["reasons"].append(f"attempt.json build.program_digest {str(decl['program_digest'])[:16]} ≠ compiled_manifest.json program_digest {str(pd)[:16]} — the declared P_compiled is not the artifact on record")
        return out
    kind = str(M.get("kind") or "")
    if kind and not kind.startswith("verity/compiled-required-manifest/"):
        gaps.append(f"compiled_manifest.json kind {kind!r} is not {P_COMPILED_MANIFEST_KIND}")
    pops = M.get("populations") if isinstance(M.get("populations"), dict) else {}
    prog = M.get("program") if isinstance(M.get("program"), dict) else {}
    df = prog.get("dataflow") if isinstance(prog.get("dataflow"), dict) else {}
    complete = as_bool(M.get("complete"))
    if complete is not True:
        gaps.append(f"compiled_manifest.json complete:{M.get('complete')} (the required population is not complete over the declared workload)")
    if not df:
        gaps.append("compiled_manifest.json program.dataflow (every non-weight launch input → its producer; part of program_digest — rev-pm M-0589)")
    else:
        if as_bool(df.get("wiring_declared")) is not True:
            gaps.append(f"dataflow wiring_declared:{df.get('wiring_declared')}")
        if df.get("n_undefined") not in (0, "0"):
            gaps.append(f"dataflow n_undefined:{df.get('n_undefined')} (every launch input must have a producer or be an external operand)")
    # declared target: the P_compiled must be built for THIS row's execution (bi-compiled) — the manifest's / attempt's declared execution + every arm
    ec = A.get("execution_configuration") if isinstance(A.get("execution_configuration"), dict) else {}
    decl_ex = str(ec.get("declared_execution") or (M.get("query") or {}).get("declared_execution") if isinstance(M.get("query"), dict) else ec.get("declared_execution") or "").strip()
    arms = compiled_arms(evdir)
    arm_decl = {str(a.get("declared_execution")).strip() for a in arms if a.get("declared_execution")}
    if not decl_ex and arm_decl:
        decl_ex = sorted(arm_decl)[0]
    if not decl_ex:
        gaps.append(f"declared target: no declared_execution on the record (attempt.json execution_configuration.declared_execution / versions.json declared_execution) for row execution `{ex}`")
    elif ex and decl_ex.lower() != ex.lower():
        out["status"] = "FAIL"
        out["reasons"].append(f"declared target `{decl_ex}` is not the row's execution `{ex}` — not this row's configuration (EXEC-02)")
        return out
    if arm_decl and any(d.lower() != ex.lower() for d in arm_decl if ex):
        out["status"] = "FAIL"
        out["reasons"].append(f"an arm was launched for execution {sorted(arm_decl)} but the row is `{ex}`")
        return out
    out["target"] = f"{decl_ex or ex} — P_compiled kernel_set {str(ks or '?')[:12]}… ({prog.get('pieces')} pieces, {pops.get('identities')} identities / {pops.get('per_step')} per step over {pops.get('steps')} steps)"
    # workload: the manifest's schedule is the row's declared workload (B requests; prompt/cap per request when the workload file carries them)
    W = M.get("workload") if isinstance(M.get("workload"), dict) else {}
    sched = W.get("schedule") if isinstance(W.get("schedule"), list) else None
    if sched is None:
        gaps.append("compiled_manifest.json workload.schedule (the declared requests the population is composed over)")
    elif wlf.get("B") and len(sched) != int(wlf["B"]):
        out["status"] = "FAIL"
        out["reasons"].append(f"compiled_manifest.json workload.schedule has {len(sched)} requests but the row's workload has B={wlf['B']} — a substitute workload is not the row")
        return out
    else:
        reqs = (wl.get("requests") or []) if isinstance(wl, dict) else []
        if reqs and len(reqs) == len(sched):
            bad = []
            for i, (r, s) in enumerate(zip(reqs, sched)):
                pl = r.get("prompt_len") or (len(r.get("prompt_token_ids")) if isinstance(r.get("prompt_token_ids"), list) else None)
                if pl is not None and s.get("prompt_tokens") is not None and int(pl) != int(s["prompt_tokens"]):
                    bad.append(f"r{i} prompt {s.get('prompt_tokens')}≠{pl}")
                if r.get("max_tokens") is not None and s.get("max_tokens") is not None and int(r["max_tokens"]) != int(s["max_tokens"]):
                    bad.append(f"r{i} cap {s.get('max_tokens')}≠{r['max_tokens']}")
            if bad:
                out["status"] = "FAIL"
                out["reasons"].append("compiled_manifest.json workload.schedule ≠ the row's workload manifest: " + ", ".join(bad[:6]))
                return out
    # manifest_verify (M-0819 BUILD): the TREATMENT arm's own derivation == the control's program + manifest digest — commit/manifest_verify.json
    # (compiled MAN-05, int-6+ f2be42e77) or compiled_merge's program_digests record in commit/summary.json (the merge REFUSES unequal arms)
    mv = manifest_verify_from(evdir)
    S = compiled_commit_summary(evdir)
    pdg = S.get("program_digests") if isinstance(S.get("program_digests"), dict) else {}
    if mv and not mv.get("error") and as_bool(mv.get("ok")) is True and str(mv.get("used_program_digest") or "") == str(pd) and str(mv.get("rebuilt_program_digest") or "") == str(pd) \
            and (not md or str(mv.get("used_digest") or "") == str(md)):
        out["manifest_verify"] = {"how": "commit/manifest_verify.json (compiled MAN-05): treatment-arm re-derivation == control", "artifact": mv.get("artifact"),
                                  "rebuilt_identities": mv.get("rebuilt_identities"), "rebuilt_complete": mv.get("rebuilt_complete")}
        out["evidence"].append(mv["artifact"])
    elif mv and not mv.get("error") and as_bool(mv.get("ok")) is False:
        out["status"] = "FAIL"
        out["reasons"].append(f"P_compiled manifest verify FAILED (commit/manifest_verify.json ok:false): {str(mv.get('why') or '')[:200]}")
        return out
    elif pdg and as_bool(pdg.get("equal")) is True and str(pdg.get("control") or "") == str(pd) and str(pdg.get("treatment") or "") == str(pd) \
            and (not md or (str(pdg.get("manifest_control") or "") == str(md) and str(pdg.get("manifest_treatment") or "") == str(md))):
        out["manifest_verify"] = {"how": "commit/summary.json program_digests (compiled_merge): the treatment arm's own derivation from its Inductor output code gives the same program_digest + manifest_digest as the control's",
                                  "control": pdg.get("control"), "treatment": pdg.get("treatment"), "equal": True}
    elif pdg and as_bool(pdg.get("equal")) is False:
        out["status"] = "FAIL"
        out["reasons"].append(f"P_compiled derived from the treatment arm's output code ≠ control's (program_digests control {str(pdg.get('control'))[:12]} / treatment {str(pdg.get('treatment'))[:12]}) — the two arms did not execute one specialization")
        return out
    else:
        gaps.append("manifest verify: the treatment arm's own P_compiled derivation == the control's (commit/manifest_verify.json ok:true for this program_digest, or compiled_merge program_digests.equal in commit/summary.json)")
    # runtime kernel_set of each arm's emitted output code == the manifest's (source-level artifact identity; the LOADED-object identity is Match's)
    ks_arms = {a["arm"]: a.get("kernel_set_sha256") for a in arms if a.get("kernel_set_sha256")}
    if ks and ks_arms and any(str(v) != str(ks) for v in ks_arms.values()):
        out["status"] = "FAIL"
        out["reasons"].append(f"an arm's emitted kernel_set_sha256 ≠ compiled_manifest.json kernel_set {str(ks)[:12]}… ({', '.join(f'{k}={str(v)[:12]}' for k, v in ks_arms.items())}) — the P_compiled was not generated from the executed specialization")
        return out
    # weights of record the compiled kernels consume: components.weights_struct → the pinned-checkpoint weights-of-record record (Build-side pin, MAN-09)
    comps = M.get("components") if isinstance(M.get("components"), dict) else {}
    ws = comps.get("weights_struct") if isinstance(comps.get("weights_struct"), dict) else {}
    ws_pd = ws.get("program_digest") or decl.get("weights_struct_program_digest")
    out["weights_struct_program_digest"] = ws_pd
    if ws_pd:
        Wr, wpath = compiled_weights_of_record(evdir, str(ws_pd))
        if Wr:
            ck = Wr.get("checkpoint") if isinstance(Wr.get("checkpoint"), dict) else {}
            if as_bool(ck.get("all_shards_pinned")) is True:
                out["weights_root_of_record"] = str(Wr["root_of_record"])
                out["weights_of_record"] = {"artifact": wpath, "program_digest": ws_pd, "root_of_record": Wr["root_of_record"], "fields": len(Wr.get("fields") or []),
                                            "checkpoint": {k: ck.get(k) for k in ("entry", "all_shards_pinned") if k in ck},
                                            "how": "resolved from compiled_manifest components.weights_struct.program_digest (the bi-eager Build of record whose served-parameter fields the compiled kernels consume) — COORD R16 ruling 5, logged"}
                out["evidence"].append(wpath)
            else:
                gaps.append(f"weights-of-record record {wpath} is not fully pinned (checkpoint.all_shards_pinned {ck.get('all_shards_pinned')})")
        else:
            gaps.append(f"Build-side weights-of-record record for weights_struct Program {str(ws_pd)[:12]}… (verity/commit/weights-of-record/v1 with root_of_record from the pinned checkpoint shards; in this attempt, its row dir or evidence/<lane>/weights_of_record/) — MAN-09's Build side")
    else:
        gaps.append("compiled_manifest.json components.weights_struct.program_digest (which served-parameter record the compiled kernels' weight arguments are)")
    out["program_digest"] = pd
    out["p_compiled"] = {"program_digest": pd, "kernel_set_sha256": ks, "manifest_digest": md, "identities": pops.get("identities"), "per_step": pops.get("per_step"),
                         "steps": pops.get("steps"), "complete": M.get("complete"), "artifact": mp,
                         "dataflow": {k: df.get(k) for k in ("wiring_declared", "n_edges", "n_launch_edges", "n_undefined") if k in df},
                         "derived_from": decl.get("derived_from") or "control arm's Inductor output code + split graph (compiled_manifest.build)",
                         "pieces": prog.get("pieces")}
    out["cmd"] = "compiled_manifest.build (P_compiled from the production DEFAULT Inductor artifact + split graph × the workload schedule)"
    out["identity"] = f"P_compiled {str(pd or '?')[:12]}… kernel_set {str(ks or '?')[:12]}… manifest {str(md or '?')[:12]}…"
    out["applicability"] = {"execution": decl_ex or ex, "compilation": {k: (arms[0].get(k) if arms else None) for k in ("compilation_mode", "cudagraph_mode", "compile_cache_disabled")}}
    if P_COMPILED_DATAFLOW_OPEN_FINDING and df and not gaps:
        # [rev-pm M-0758 F-pm-29 on 2b6573893 = the int-6/int-6b compiled S3 code] weight-argument routing is not in the digest: two P_compiled that
        # route the same served weights to different norms share one identity.  Closed by compiled b08b4bf0a (FINAL.md) — a record derived by a tree
        # carrying it names that in the manifest (dataflow.weight_edges / n_weight_edges); until such a record lands, Build ◐ naming the finding.
        if any(k in df for k in ("n_weight_edges", "weight_edges", "warg_edges", "n_warg_edges", "weights_routed")):
            out.setdefault("footnotes", []).append("dataflow carries weight-argument routing (" + ", ".join(k for k in ("n_weight_edges", "weight_edges", "warg_edges", "n_warg_edges", "weights_routed") if k in df) + ") — rev-pm F-pm-29 addressed on this record")
        else:
            out["p_compiled"]["open_finding"] = P_COMPILED_DATAFLOW_OPEN_FINDING
            gaps.append("rev-pm F-pm-29 stands on this record's derivation (weight-argument routing is not part of program_digest — M-0758 on 2b6573893 = int-6/int-6b S3 code): a record whose dataflow carries the weight edges (compiled b08b4bf0a) closes it")
    if out["status"] == "FAIL":
        return out
    if gaps:
        out["status"] = "PARTIAL"
        out["reasons"].append("P_compiled on record (" + out["identity"] + f"; complete {M.get('complete')}, dataflow {df.get('n_edges')} edges / {df.get('n_undefined')} undefined) — Build ◐, missing: " + "; ".join(gaps))
    else:
        out["status"] = "PASS"
        out.setdefault("footnotes", []).append(f"Build ✅ basis = P_compiled derivation record (COORD M-0819): {out['identity']}, {pops.get('identities')} identities / {pops.get('per_step')} per step, complete; "
                                              f"dataflow {df.get('n_edges')} edges ({df.get('n_launch_edges')} launch→launch), 0 undefined, wiring declared; manifest verify: {out['manifest_verify']['how']}; "
                                              f"weights of record {str(out.get('weights_root_of_record'))[:12]}… ({(out.get('weights_of_record') or {}).get('artifact')})")
    return out


def s2_checkpoint_record(evdir: str, pd: str | None, ks: str | None) -> dict:
    """the S2 checkpoint-replay record (verity/compiled-s2-checkpoints/v1) for THIS attempt's P_compiled: every kernel launch replayed bit-exactly
    from its materialised inputs (replay.per_kind exact_outputs == outputs, 0 mismatched elements, no errors) under the same kernel_set."""
    out: dict = {"present": False}
    p = first(evdir, "s2_checkpoints.json", "match/s2_checkpoints.json", "s2/s2_checkpoints.json", "*s2_checkpoints*.json", "match/*s2*.json")
    if not p:
        return out
    out["artifact"] = os.path.relpath(p)
    try:
        J = load(p)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
        return out
    if not isinstance(J, dict) or not str(J.get("schema") or "").startswith("verity/compiled-s2-checkpoints/"):
        out["error"] = f"schema {str((J or {}).get('schema') if isinstance(J, dict) else None)!r} is not {P_COMPILED_S2_SCHEMA}"
        return out
    out["present"] = True
    gk = J.get("generated_kernels") if isinstance(J.get("generated_kernels"), dict) else {}
    out["kernel_set_sha256"] = gk.get("kernel_set_sha256")
    out["program_digest"] = J.get("program_digest") or (J.get("p_compiled") or {}).get("program_digest") if isinstance(J.get("p_compiled"), dict) else J.get("program_digest")
    rp = J.get("replay") if isinstance(J.get("replay"), dict) else {}
    pk = rp.get("per_kind") if isinstance(rp.get("per_kind"), dict) else {}
    launches = outputs = exact = mism = 0
    errs: list[str] = []
    for k, v in pk.items():
        if not isinstance(v, dict):
            continue
        launches += int(v.get("launches") or 0); outputs += int(v.get("outputs") or 0); exact += int(v.get("exact_outputs") or 0)
        mism += int(v.get("mismatched_elements") or 0)
        errs += [f"{k}: {e}" for e in (v.get("errors") or [])[:2]]
    out.update({"kinds": sorted(pk), "launches": launches, "outputs": outputs, "exact_outputs": exact, "mismatched_elements": mism, "errors": errs,
                "mismatch_examples": len(rp.get("mismatch_examples") or []), "steps": (J.get("observation") or {}).get("steps") if isinstance(J.get("observation"), dict) else None})
    ni = J.get("noninterference") if isinstance(J.get("noninterference"), dict) else {}
    out["noninterference"] = {k: ni.get(k) for k in ("tokens_equal", "steps", "steps_bit_identical", "logprob_max_abs", "reference") if k in ni}
    ok = bool(pk) and outputs > 0 and exact == outputs and mism == 0 and not errs and out["mismatch_examples"] == 0
    if ks and out["kernel_set_sha256"] and str(out["kernel_set_sha256"]) != str(ks):
        ok = False
        out["why"] = f"S2 record kernel_set {str(out['kernel_set_sha256'])[:12]}… ≠ this attempt's P_compiled kernel_set {str(ks)[:12]}… (another specialization / another row's P_compiled)"
    elif pd and out.get("program_digest") and str(out["program_digest"]) != str(pd):
        ok = False
        out["why"] = f"S2 record program_digest {str(out['program_digest'])[:12]}… ≠ this attempt's P_compiled {str(pd)[:12]}…"
    elif not (ks and out["kernel_set_sha256"]):
        ok = False
        out["why"] = "S2 record does not pin generated_kernels.kernel_set_sha256 (cannot be tied to this attempt's P_compiled)"
    elif not ok:
        out["why"] = f"S2 replay not bit-exact: {exact}/{outputs} outputs exact, {mism} mismatched elements, {len(errs)} errors, {out['mismatch_examples']} mismatch examples"
    out["ok"] = ok
    return out


def derive_match_compiled(evdir: str, A: dict, build: dict, match: dict, row_id: str, row: dict) -> dict:
    """[COORD M-0819 / DANIEL M-0776 (3)] MATCH of an Execution=compiled attempt = (i) the S2 checkpoint-replay record for THIS P_compiled (bit-exact
    segments at the materialised checkpoints) + (ii) the runtime↔artifact identity: the kernel_set of the code objects LOADED by the running
    process (versions.json compilation.generated_kernels.loaded / commit/summary.json loaded_identity — M-0805) equal across arms, every kernel
    P_compiled names among them + (iii) noninterference (tokens equal across two COMPILED arms, kernel_set + compilation settings equal).
    ✅ iff all three; otherwise ◐ naming each gap.  Never graded from the eager kernel-level fold comparison (M-0266), never ❌ from its absence;
    ❌ only for a measured mismatch (S2 replay not bit-exact, arms executing different specializations, tokens differing across arms)."""
    ex = str((row.get("config") or {}).get("execution") or (row_id.split("__")[-1] if "__" in row_id else "")).strip()
    out = {"status": "-", "derived_by": "finalize/v1", "reasons": [], "evidence": [], "kind": "P_compiled",
           "comparator": "compiled S2 checkpoint replay + runtime↔artifact identity (M-0819)"}
    pc = build.get("p_compiled") if isinstance(build.get("p_compiled"), dict) else {}
    pd, ks = pc.get("program_digest"), pc.get("kernel_set_sha256")
    S = compiled_commit_summary(evdir)
    arms = compiled_arms(evdir)
    gaps: list[str] = []
    fails: list[str] = []
    # (iii) noninterference — instrumented run == production-compiled numerics on the same specialization
    te = S.get("tokens_equal")
    tok = all(as_bool(x) for x in te) if isinstance(te, list) and te else as_bool(te)
    gk = S.get("generated_kernels") if isinstance(S.get("generated_kernels"), dict) else {}
    kse = as_bool(gk.get("kernel_set_equal")) if gk else None
    cse = as_bool(S.get("compilation_settings_equal")) if "compilation_settings_equal" in S else None
    ctrl = [a for a in arms if "control" in a["arm"]]
    trt = [a for a in arms if "treatment" in a["arm"]]
    two_compiled = bool(ctrl) and bool(trt) and all(a.get("enforce_eager") is False for a in ctrl + trt)
    for a in arms:
        eff = str(a.get("effective_execution") or "").strip().lower()
        if eff and ex and eff != ex.lower():
            fails.append(f"{a['arm']} effective_execution.execution_label `{a.get('effective_execution')}` ≠ row `{ex}` (EXEC-02 read-back)")
        if a.get("enforce_eager") is True:
            fails.append(f"{a['arm']} ran with enforce_eager=True — an eager arm on a compiled row (PLAN §3 'eager called compiled')")
    if tok is False:
        fails.append("tokens differ across the control / treatment arms (commit/summary.json tokens_equal false) — the instrumented execution is not the production-compiled numerics")
    if kse is False:
        fails.append("kernel_set differs across arms (generated_kernels.kernel_set_equal false) — the arms executed different specializations")
    if fails:
        out["status"] = "FAIL"
        out["reasons"] = fails
        out["evidence"] = [os.path.relpath(p) for p in (first(evdir, "commit/summary.json"),) if p]
        return out
    ni_bits = []
    if tok is not True:
        gaps.append("noninterference: tokens equal across the two compiled arms on record (commit/summary.json tokens_equal)")
    else:
        ni_bits.append("tokens equal across arms")
    if not two_compiled:
        gaps.append("noninterference: two COMPILED arms (control + treatment versions.json enforce_eager False) on record")
    else:
        ni_bits.append(f"{len(ctrl)}+{len(trt)} compiled arms")
    if kse is True:
        ni_bits.append(f"kernel_set equal across arms {str(gk.get('control', {}).get('kernel_set_sha256') or '')[:12]}…")
    else:
        gaps.append("noninterference: kernel_set equal across arms (generated_kernels.kernel_set_equal)")
    if cse is True:
        ni_bits.append("compilation settings equal")
    elif cse is False:
        gaps.append("noninterference: compilation settings differ across arms (compilation_settings_equal false)")
    if ks and gk and gk.get("control", {}).get("kernel_set_sha256") and str(gk["control"]["kernel_set_sha256"]) != str(ks):
        out["status"] = "FAIL"
        out["reasons"].append(f"the arms' emitted kernel_set {str(gk['control']['kernel_set_sha256'])[:12]}… ≠ the P_compiled artifact's {str(ks)[:12]}… — the Program was not generated from the executed specialization")
        return out
    # (ii) runtime ↔ artifact identity: the LOADED code objects (M-0805: PyCodeCache modules × CachingAutotuner sources in the running process)
    li = S.get("loaded_identity") if isinstance(S.get("loaded_identity"), dict) else None
    if li is None:
        loaded = [a for a in arms if a.get("loaded")]
        if loaded:
            sets = {str(a["loaded"].get("kernel_set_sha256")) for a in loaded}
            li = {"equal_across_arms": len(sets) == 1 and len(loaded) >= 2, "arms": {a["arm"]: a["loaded"].get("kernel_set_sha256") for a in loaded},
                  "manifest_kernels": None, "manifest_kernels_not_loaded": None, "_from": "versions.json compilation.generated_kernels.loaded per arm"}
    ident_bits = []
    if not li:
        gaps.append("runtime↔artifact identity: the kernel_set of the code objects LOADED by the running process (versions.json compilation.generated_kernels.loaded per arm / commit/summary.json loaded_identity — M-0805, compiled 3a5716d6b+), equal across arms, every P_compiled kernel among them")
    else:
        out["loaded_identity"] = {k: li.get(k) for k in ("equal_across_arms", "binaries_equal_across_arms", "kernels_whose_binaries_differ", "manifest_kernels", "manifest_kernels_not_loaded", "read_from", "_from") if k in li}
        for arm in ("control", "treatment"):
            if isinstance(li.get(arm), dict):
                out["loaded_identity"][arm] = {k: li[arm].get(k) for k in ("kernels", "kernel_set_sha256", "cubin_set_sha256", "binary_set_sha256") if k in li[arm]}
        if as_bool(li.get("equal_across_arms")) is not True:
            fails.append("loaded code objects differ across arms (loaded_identity.equal_across_arms false) — the arms did not execute one specialization")
        nl = li.get("manifest_kernels_not_loaded")
        if isinstance(nl, list) and nl:
            fails.append(f"{len(nl)} kernel(s) P_compiled names were never loaded by the running process ({', '.join(str(x) for x in nl[:4])}) — the Program is not the executed artifact")
        elif nl is None:
            gaps.append("runtime↔artifact identity: every kernel P_compiled names among the LOADED code objects (loaded_identity.manifest_kernels / manifest_kernels_not_loaded)")
        else:
            ident_bits.append(f"{li.get('manifest_kernels')} P_compiled kernels all among the loaded code objects")
        fl = li.get("files_vs_loaded") if isinstance(li.get("files_vs_loaded"), dict) else {}
        if fl.get("in_files_not_loaded"):
            fails.append(f"emitted kernels never loaded: {', '.join(str(x) for x in fl['in_files_not_loaded'][:4])}")
        if fails:
            out["status"] = "FAIL"
            out["reasons"] = fails
            return out
        if as_bool(li.get("equal_across_arms")) is True:
            ks_l = (li.get("control") or li.get("arms") or {}).get("kernel_set_sha256") if isinstance(li.get("control") or li.get("arms"), dict) else None
            ident_bits.append(f"loaded kernel-source set equal across arms {str(ks_l or '')[:12]}… ({li.get('read_from') or li.get('_from')})")
        if as_bool(li.get("binaries_equal_across_arms")) is False:
            kd = li.get("kernels_whose_binaries_differ") or []
            out.setdefault("footnotes", []).append(f"executed binaries differ for {len(kd)} kernel(s) with EQUAL sources ({', '.join(str(k) for k in kd[:4])}) — Inductor autotune specialization per process, labelled (M-0805); numerics covered by noninterference + S2; root cause open (compiled)")
        elif as_bool(li.get("binaries_equal_across_arms")) is True:
            ident_bits.append("binaries equal across arms")
    # (i) the S2 checkpoint-replay record for THIS P_compiled
    s2 = s2_checkpoint_record(evdir, pd, ks)
    out["s2"] = {k: s2.get(k) for k in ("present", "artifact", "ok", "why", "kinds", "launches", "outputs", "exact_outputs", "mismatched_elements", "steps", "kernel_set_sha256", "noninterference", "error") if k in s2}
    if not s2.get("present"):
        gaps.append(f"S2 checkpoint-replay record for this P_compiled (s2_checkpoints.json {P_COMPILED_S2_SCHEMA}, kernel_set {str(ks or '?')[:12]}…: every kernel launch replayed bit-exactly from its materialised inputs)" + (f" — {s2.get('artifact')}: {s2['error']}" if s2.get("error") else ""))
    elif s2.get("ok") is True:
        out["evidence"].append(s2["artifact"])
        ident_bits.append(f"S2: {s2['exact_outputs']}/{s2['outputs']} kernel outputs bit-exact over {s2['launches']} launches ({', '.join(s2['kinds'])}; {s2.get('steps')} steps)")
    elif s2.get("why", "").startswith("S2 replay not bit-exact"):
        out["status"] = "FAIL"
        out["reasons"].append(s2["why"] + f" ({s2['artifact']})")
        out["evidence"].append(s2["artifact"])
        return out
    else:
        gaps.append(f"S2 record on this attempt is not this P_compiled's: {s2.get('why')} ({s2.get('artifact')})")
    out["noninterference"] = {"tokens_equal_arms": tok, "two_compiled_arms": two_compiled, "kernel_set_equal": kse, "compilation_settings_equal": cse}
    out["evidence"] += [os.path.relpath(p) for p in (first(evdir, "commit/summary.json"),) if p]
    out["evidence"] += [os.path.join(os.path.relpath(evdir), a["file"]) for a in arms]
    out["workload_program_digest"] = pd
    if gaps:
        out["status"] = "PARTIAL"
        out["reasons"].append("Match (P_compiled, M-0819) ◐ — on record: " + ("; ".join(ni_bits + ident_bits) or "nothing graded") + ". Missing: " + "; ".join(gaps))
        out["compiled_match_basis"] = "S2 + runtime↔artifact identity incomplete (M-0819)"
    else:
        out["status"] = "PASS"
        out["compiled_match_basis"] = "P_compiled (M-0819: S2 checkpoint replay bit-exact + loaded code-object identity equal across arms + noninterference; DANIEL M-0776 (3))"
        out.setdefault("footnotes", []).append("Match ✅ (P_compiled): " + "; ".join(ni_bits + ident_bits) + " — S2 checkpoint replay + runtime↔artifact identity + noninterference (COORD M-0819, DANIEL M-0776 (3)); no eager-equivalence claim")
    return out


def compiled_commit_adjust(evdir: str, A: dict, build: dict, match: dict, commit: dict) -> None:
    """[COORD M-0819] COMMIT of a P_compiled attempt = S3 per record with form (A) (derive_commit as for every row: C1 coverage, sampled replay +
    boundary linkage + negatives, MAN-09 pin against Build's weights root of record, C3 openings).  Two eager-shaped gaps are re-read for the
    P_compiled record: the manifest of record is compiled_manifest.json[.gz] (its manifest_digest == the verdict's required_manifest), and MAN-05
    is the P_compiled manifest verify graded in Build.  The M-0285 (b) hold ('instrumentation perturbs numerics') is lifted by M-0776 (3) /
    M-0819 when noninterference is on the record (tokens equal across two compiled arms + kernel_set equal); otherwise it stays, named."""
    if commit.get("status") in ("-", None):
        return
    pc = build.get("p_compiled") if isinstance(build.get("p_compiled"), dict) else {}
    md = pc.get("manifest_digest")
    # manifest of record = compiled_manifest.json (the eager reader looked for manifest.json[.gz])
    rm = commit.get("required_manifest") if isinstance(commit.get("required_manifest"), dict) else {}
    pop_md = rm.get("manifest_digest") or rm.get("population_manifest_digest") or rm.get("digest")
    verdict_md = None
    vp = first(evdir, "commit/verdict.json", "commit_verdict.json")
    if vp:
        try:
            Vd = load(vp)
            verdict_md = ((Vd.get("required_manifest") or {}).get("manifest_digest") if isinstance(Vd.get("required_manifest"), dict) else None) or Vd.get("manifest_digest")
        except Exception:  # noqa: BLE001
            verdict_md = None
    marker = "manifest of record (manifest.json[.gz]) not in the evidence directory"
    if md and any(str(r).startswith(marker) for r in commit.get("reasons") or []):
        want = pop_md or verdict_md
        if want and str(want) != str(md):
            commit["status"] = "FAIL"
            commit["reasons"] = [r for r in commit["reasons"] if not str(r).startswith(marker)]
            commit["reasons"].append(f"compiled_manifest.json manifest_digest {str(md)[:16]} ≠ the binding population's / verdict's required manifest {str(want)[:16]} — not the manifest the population was drawn from (M-0125)")
            return
        commit["reasons"] = [r for r in commit["reasons"] if not str(r).startswith(marker)]
        commit.setdefault("footnotes", []).append(f"manifest of record = {pc.get('artifact')} (P_compiled required manifest {str(md)[:12]}…, {pc.get('identities')} identities) == the verdict's required_manifest digest")
        rm = dict(rm, file=pc.get("artifact"), manifest_digest=md, program_digest=pc.get("program_digest"), kind="P_compiled")
        commit["required_manifest"] = rm
    # MAN-05 for P_compiled = the manifest verify graded in Build (treatment-arm re-derivation == control); the eager tool's absence is not this row's gap
    mv_b = build.get("manifest_verify") if isinstance(build.get("manifest_verify"), dict) else None
    if mv_b:
        commit["reasons"] = [r for r in commit.get("reasons") or [] if "MAN-05" not in str(r)]
        commit["manifest_verify"] = {"how": mv_b.get("how"), "graded_in": "build (P_compiled, M-0819)", "artifact": mv_b.get("artifact"), "schema": "P_compiled manifest verify"}
        commit.setdefault("footnotes", []).append("MAN-05 (P_compiled): " + str(mv_b.get("how")))
    elif commit.get("manifest_verify") is None and "manifest_verify" in commit:
        commit.pop("manifest_verify", None)     # apply_man05_absence would name the EAGER tool; the P_compiled gap is named in Build (manifest verify)
        commit.setdefault("reasons", []).append("missing: P_compiled manifest verify (the treatment arm's own derivation == the control's) — named in Build")
        if commit.get("status") == "PASS":
            commit["status"] = "PARTIAL"
    # the M-0285 (b) hold vs noninterference on record
    hold = "instrumentation perturbs numerics"
    companion = "C2 record complete on this attempt"
    ni = match.get("noninterference") if isinstance(match.get("noninterference"), dict) else {}
    ni_ok = ni.get("tokens_equal_arms") is True and ni.get("two_compiled_arms") is True and ni.get("kernel_set_equal") is True
    if any(str(r).startswith(hold) for r in commit.get("reasons") or []):
        if ni_ok:
            was_pass = commit.get("derived_before_cap") == "PASS"
            commit["reasons"] = [r for r in commit["reasons"] if not (str(r).startswith(hold) or str(r).startswith(companion))]
            commit["compiled_commit_basis"] = "S3 per record, form (A) (COORD M-0819); noninterference on record (tokens equal across two compiled arms, kernel_set equal) — M-0285 (b) hold lifted by DANIEL M-0776 (3)"
            commit.setdefault("footnotes", []).append("Commit basis (P_compiled): S3 form (A) — sampled exact replay of P_compiled's kernel relations from the committed bytes + boundary linkage + negatives + MAN-09; noninterference: tokens equal across the two compiled arms, kernel_set equal (M-0776 (3) lifts the M-0285 (b) hold)")
            if was_pass and commit.get("status") == "PARTIAL" and not [r for r in commit["reasons"] if r]:
                commit["status"] = "PASS"
                commit.pop("derived_before_cap", None)
        else:
            commit["compiled_commit_basis"] = "S3 per record — noninterference NOT on record (M-0776 (3) keeps it required)"


def compiled_attestation_guard(row_id: str, row: dict, evdir: str, A: dict, build: dict, match: dict) -> None:
    """The engine-execution attestation part of execution_guard for a P_compiled attempt (M-0040/M-0081/M-0063 (3)): every present source must
    attest a COMPILED engine for this row; eager → ❌ (substitution), conflict → ❌, none → ◐ named.  The Build/Match derivations themselves are
    M-0819's (derive_build_compiled / derive_match_compiled), not the eager kernel-level path."""
    ex = str((row.get("config") or {}).get("execution") or (row_id.split("__")[-1] if "__" in row_id else "")).strip()
    att = execution_attestation(evdir, A)
    for d in (build, match):
        d["execution_declared"] = ex
        d["execution_attested"] = att
    decl = (att or {}).get("declared_execution")
    if decl and ex and decl.strip().lower() != ex.lower():
        for d in (build, match):
            _fail(d, f"the run was launched for execution `{decl}` (versions.json declared_execution, EXEC-02) but the row is `{ex}` — not this row's configuration")
        return
    if att and att.get("conflict"):
        for d in (build, match):
            _fail(d, f"execution attestation conflict — {att['conflict']} (every present source must agree; rev-pm M-0081)")
        return
    if att is None:
        for d in (build, match):
            _cap(d, f"missing: engine execution attestation for `{ex}` (engine_kwargs enforce_eager + cudagraph/compilation mode in the arms' versions.json — M-0040/M-0081)")
        return
    if att["verdict"] == "eager":
        for d in (build, match):
            _fail(d, f"execution substitution: the row is `{ex}` but the engine record attests eager ({att['source']}: enforce_eager={att.get('enforce_eager')}, cudagraph_mode={att.get('cudagraph_mode')}) — PLAN §3 forbids 'eager called compiled'")
        return
    if att["verdict"] != "compiled":
        for d in (build, match):
            _cap(d, f"missing: positive evidence of a compiled execution for `{ex}` (enforce_eager={att.get('enforce_eager')}, cudagraph_mode={att.get('cudagraph_mode')}, compilation_mode={att.get('compilation_mode')} — enforce_eager False AND a cudagraph/compilation mode other than NONE; M-0081)")


def execution_guard(row_id: str, row: dict, evdir: str, A: dict, build: dict, match: dict) -> None:
    """COORD M-0063 (3) / rev-pm M-0040, M-0081: Build/Match ✅ require the recorded effective engine execution to be the row's.
    Compiled rows: no engine attestation -> ◐; conflict -> ❌; eager -> ❌ ('eager called compiled'); no positive evidence of
    compilation -> ◐.  Eager rows: conflict -> ❌; compiled -> ❌ ('outside declared applicability'); no attestation keeps the status
    (the rc3 harness is eager by construction; the declared-applicability contract itself waits for global's (1))."""
    ex = str((row.get("config") or {}).get("execution") or (row_id.split("__")[-1] if "__" in row_id else "")).strip()
    att = execution_attestation(evdir, A)
    for d in (build, match):
        d["execution_declared"] = ex
        d["execution_attested"] = att
    eager_row = (not ex) or ex.startswith("bi-eager") or ex == "eager"
    # rev-pm M-0159 TIP: Build's own applicability label is a Build-stage fact — an eager-applicable Program shown on a compiled row
    # (or a compiled-only one on an eager row) is outside declared applicability without any Commit header (COORD M-0063 (1)/(3))
    ap = (build.get("applicability") or {}) if isinstance(build.get("applicability"), dict) else {}
    ap_ex = str(ap.get("execution") or ap.get("execution_label") or ap.get("label") or "").strip().lower()
    if ap_ex and ex:
        ap_eager = ap_ex.startswith("bi-eager") or ap_ex == "eager"
        ap_compiled = "compiled" in ap_ex
        if (not eager_row and ap_eager and not ap_compiled) or (eager_row and ap_compiled and not ap_eager):
            for d in (build, match):
                _fail(d, f"outside declared applicability: the Build artifact declares applicability.execution=`{ap_ex}` but the row is `{ex}` (build_summary.step.applicability — COORD M-0063 (1)/(3), rev-pm M-0159)")
            return
    decl = (att or {}).get("declared_execution")
    if decl and ex and decl.strip().lower() != ex.lower():
        for d in (build, match):
            _fail(d, f"the run was launched for execution `{decl}` (versions.json declared_execution, EXEC-02) but the row is `{ex}` — not this row's configuration")
        return
    if att and att.get("conflict"):
        for d in (build, match):
            _fail(d, f"execution attestation conflict — {att['conflict']} (every present source must agree; rev-pm M-0081)")
        return
    if eager_row:
        if att and att["verdict"] == "compiled":
            for d in (build, match):
                _fail(d, f"outside declared applicability: the row is `{ex}` but the engine record attests a compiled engine ({att['source']}: enforce_eager={att.get('enforce_eager')}, cudagraph_mode={att.get('cudagraph_mode')}) — COORD M-0063 (3)")
        return
    # COORD RULING M-0178 (1) (provisional handling of M-0153, Daniel's decision pending): a compiled row's Match is ◐ — not ✅ and not
    # ✗ from the kernel-level comparison.  rev-pm ALERT M-0266 (F-pm-22): the ◐ is NOT graded from GM-01 (global_match.json) — an eager
    # execution under a header that says compiled passes GM-01 8/8, a real compiled record fails G1+G2 with 0 fold instances; the ◐
    # needs record-level positive evidence an eager run cannot produce: the Inductor kernel_set_sha256 pin in the MATCH record (M-0219)
    # AND the EXEC-02 effective_execution read-back == the row's execution; the text names boundary-level correspondence only.
    ce = compiled_record_evidence(evdir)
    match["compiled_record"] = ce
    # DANIEL'S DECISION M-0285 (21:33Z) (d): the eager Program's applicability must not claim bi-compiled (M-0256 refutes it numerically)
    # — Build on a compiled row is ◐ 'no Program applicable to compiled execution' until a compiled-target Program exists (Q1–Q4);
    # applied LAST (below) so the attestation findings attach to the cell first
    build_d = f"no Program applicable to compiled execution: the eager Program (digest {str(build.get('global_program_digest') or build.get('program_digest') or '?')[:12]}…) is not a model of the compiled implementation (M-0256: tokens diverge from step 15/32, 0/32 steps bit-identical logprobs) — a compiled-target Program with fused regions registered as composed Program relations (M-0285 Q1–Q4) is the path to Build ✅ (Daniel's decision M-0285 (d), HOLD at ◐)"
    # [compiled FIX_READY S3 M-0566 / rev-pm M-0589] P_compiled of record on this attempt: compiled_manifest.json (DECISIONS-2 §7 S1 — one
    # registered relation per Inductor kernel of the production DEFAULT artifact).  Build stays ◐ but the text names what IS on record and the
    # one open blocker (the P_compiled digest does not pin the dataflow: argument rewiring leaves program_digest + manifest_digest unchanged).
    pcm = first(evdir, "compiled_manifest.json", "p_compiled/compiled_manifest.json", "build_compiled/compiled_manifest.json")
    if pcm:
        try:
            PM = load(pcm)
            pops = PM.get("populations") if isinstance(PM.get("populations"), dict) else {}
            build["p_compiled"] = {"program_digest": PM.get("program_digest"), "kernel_set_sha256": PM.get("kernel_set_sha256"), "manifest_digest": PM.get("manifest_digest"),
                                   "identities": pops.get("identities"), "per_step": pops.get("per_step"), "steps": pops.get("steps"), "complete": PM.get("complete"), "artifact": os.path.relpath(pcm)}
            build.setdefault("evidence", []).append(os.path.relpath(pcm))
            df = (PM.get("program") or {}).get("dataflow") if isinstance(PM.get("program"), dict) else None
            df = df if isinstance(df, dict) else {}
            build["p_compiled"]["dataflow"] = {k: df.get(k) for k in ("wiring_declared", "n_edges", "n_launch_edges", "n_undefined") if k in df}
            dataflow_pinned = bool(df) and as_bool(df.get("wiring_declared")) is True and (df.get("n_undefined") == 0) and as_bool(PM.get("complete")) is True
            if dataflow_pinned and P_COMPILED_DATAFLOW_OPEN_FINDING:
                # [rev-pm RESULT M-0758 on compiled 2b6573893] the N1 re-check COORD M-0721 (4) made the ✅ conditional on came back NEGATIVE: F-pm-29 stands
                build["p_compiled"]["open_finding"] = P_COMPILED_DATAFLOW_OPEN_FINDING
                build_d = (f"P_compiled of record on this attempt (DECISIONS-2 §7 S1: compiled_manifest.json program {str(PM.get('program_digest') or '?')[:12]}…, kernel_set "
                           f"{str(PM.get('kernel_set_sha256') or '?')[:12]}…, {pops.get('identities')} identities / {pops.get('per_step')} per step, complete {PM.get('complete')}; dataflow "
                           f"{df.get('n_edges')} edges / {df.get('n_undefined')} undefined, wiring declared) — Build ◐: COORD M-0721 (4) makes ✅ (P_compiled) conditional on rev-pm's N1 re-check, "
                           f"and that re-check STANDS AS A BLOCKER: {P_COMPILED_DATAFLOW_OPEN_FINDING}; ✅ when rev-pm closes F-pm-29 on the tree of record")
            elif dataflow_pinned:
                # [COORD DECISION M-0721 (4) / compiled M-0698 / rev-pm M-0589] the P_compiled digest now covers the buffer graph (every non-weight
                # launch input -> its producer, part of program_digest): Build ✅ (P_compiled), pending rev-pm's N1 re-check as a footnote
                build["compiled_build_basis"] = "P_compiled (DECISIONS-2 §7 S1 + dataflow manifest; COORD M-0721 (4))"
                if build.get("status") in ("-", None):
                    build["status"] = "PASS"
                    build["reasons"] = [r for r in build.get("reasons") or [] if not str(r).startswith("no build_summary.json")]
                build["program_digest"] = PM.get("program_digest") or build.get("program_digest")
                build["cmd"] = build.get("cmd") or "compiled_manifest.build (P_compiled from the production DEFAULT Inductor artifact + split graph)"
                build["target"] = build.get("target") or f"P_compiled kernel_set {str(PM.get('kernel_set_sha256') or '?')[:12]}… ({(PM.get('program') or {}).get('pieces')} pieces)"
                build.setdefault("footnotes", []).append(
                    f"Build ✅ basis = P_compiled of record (compiled_manifest.json program {str(PM.get('program_digest') or '?')[:12]}…, kernel_set {str(PM.get('kernel_set_sha256') or '?')[:12]}…, "
                    f"{pops.get('identities')} identities / {pops.get('per_step')} per step, complete {PM.get('complete')}; dataflow: {df.get('n_edges')} edges, {df.get('n_launch_edges')} launch→launch, "
                    f"{df.get('n_undefined')} undefined, wiring declared — part of program_digest) — COORD M-0721 (4): int-6 candidate accepted pending rev-pm's N1 re-check (M-0589)")
                build_d = None
            else:
                build_d = (f"P_compiled of record on this attempt (DECISIONS-2 §7 S1: compiled_manifest.json program {str(PM.get('program_digest') or '?')[:12]}…, kernel_set "
                           f"{str(PM.get('kernel_set_sha256') or '?')[:12]}…, {pops.get('identities')} identities / {pops.get('per_step')} per step, complete {PM.get('complete')}) — Build ◐: "
                           f"the P_compiled identity does not pin the dataflow (rev-pm BLOCKER M-0589 N1: rewiring launch arguments leaves program_digest and manifest_digest unchanged; COORD M-0614 (2)) "
                           f"— this manifest carries no dataflow section (wiring_declared {df.get('wiring_declared')}, undefined {df.get('n_undefined')}); ✅ needs the dataflow manifest (compiled b481a02b0) + rev-pm's re-check")
        except Exception as e:  # noqa: BLE001
            build_d += f" (compiled_manifest.json present but unreadable: {e})"
    if match.get("status") == "FAIL" and match.get("reasons") and all(KERNEL_LEVEL_FAIL.match(str(r)) for r in match["reasons"]):
        match["kernel_level_not_evaluated"] = list(match["reasons"])
        match["status"] = "PARTIAL"
        match["reasons"] = [f"kernel-level comparison not evaluable on a compiled record (graph replay attributes 0 fold instances; X-09 / GM-01 G2..G8 report nothing about a compiled execution — rev-pm M-0266/M-0268): " + "; ".join(str(r)[:120] for r in match["kernel_level_not_evaluated"]) + " — not graded ❌ (COORD M-0178 (1): compiled rows render Match ◐, never ✅ and never ✗ from the kernel-level fold comparison; definition pending decision M-0153)"]
    eff = str(ce.get("effective_execution") or "").strip().lower()
    if eff and "compiled" not in eff:
        for d in (build, match):
            _fail(d, f"execution substitution: the row is `{ex}` but the Match capture's engine read-back is `{ce['effective_execution']}` ({ce['effective_execution_source']}, EXEC-02) — PLAN §3 forbids 'eager called compiled'")
        return
    if eff and ex and eff != ex.lower():
        for d in (build, match):
            _fail(d, f"the Match capture's engine read-back is `{ce['effective_execution']}` ({ce['effective_execution_source']}, EXEC-02) but the row is `{ex}` — not this row's configuration")
        return
    established = bool(ce.get("value_check")) and bool(ce.get("kernel_pin_match")) and bool(eff)
    if established:
        arms = ce.get("arms_enforce_eager") or []
        two_compiled = len(arms) >= 2 and all(a.get("enforce_eager") is False for a in arms)
        tok = ce.get("tokens_equal_arms")
        tok_txt = ("tokens equal across two COMPILED arms" if (tok is True and two_compiled) else
                   ("tokens equal across the Commit arms (arms' enforce_eager not both False on record)" if tok is True else "tokens equality across the two compiled arms NOT on record (commit_delta tokens_equal)"))
        # [DANIEL M-0776 (3) 02:11Z] the M-0285 (a) hold is LIFTED: Match ✅ (P_compiled) when the record pins the runtime ↔ compiled-artifact identity —
        # the kernel_set digest of the code objects LOADED by the running process (Match capture run header) == the P_compiled manifest's kernel_set_sha256
        # the Program was generated from — plus noninterference (tokens equal across compiled arms); no eager-equivalence proof is needed.
        pk = (build.get("p_compiled") or {}).get("kernel_set_sha256") if isinstance(build.get("p_compiled"), dict) else None
        runtime_pin = str(ce.get("kernel_pin_match") or "")
        if pk and runtime_pin and str(pk) == runtime_pin and tok is True and two_compiled:
            match["compiled_match_basis"] = "P_compiled (DANIEL M-0776 (3): runtime kernel_set of the loaded code objects == the P_compiled artifact's; noninterference: tokens equal across two compiled arms)"
            match.setdefault("footnotes", []).append(
                f"Match ✅ (P_compiled): the running process's kernel_set_sha256 {runtime_pin[:12]}… ({ce['kernel_pin_match_source']}) == compiled_manifest.json kernel_set_sha256 (the artifact P_compiled was "
                f"generated from); {tok_txt}; effective_execution.execution_label {ce['effective_execution']}; {ce['value_check']} relations on record — M-0285 hold lifted by Daniel M-0776 (3)")
        else:
            gaps = []
            if not pk:
                gaps.append("compiled_manifest.json (P_compiled) with kernel_set_sha256 in this attempt")
            elif not runtime_pin:
                gaps.append("the runtime kernel_set_sha256 (RunHeader.profile.compilation.generated_kernels of the Match capture — read from the RUNNING process)")
            elif str(pk) != runtime_pin:
                gaps.append(f"runtime kernel_set {runtime_pin[:12]}… != P_compiled artifact kernel_set {str(pk)[:12]}… (the executed specialization is not the one the Program was generated from)")
            if not (tok is True and two_compiled):
                gaps.append("noninterference: tokens equal across TWO COMPILED arms on record")
            why = ("Match ✅ (P_compiled) needs the explicit runtime ↔ pinned compiled-artifact identity + noninterference (DANIEL M-0776 (3): the M-0285 hold is lifted, the 'list every "
                   "unconstrained boundary' caveat withdrawn); missing on this record: " + "; ".join(gaps) + f". On record: {ce['value_check']} relations bit-exact vs a compiled control; {tok_txt}; "
                   f"effective_execution.execution_label {ce['effective_execution']}")
            if match.get("status") == "PASS":
                match["status"] = "PARTIAL"
            if match.get("status") == "PARTIAL":
                match["reasons"].append(why)
            match["compiled_match_basis"] = "runtime ↔ artifact identity not on record (M-0776 (3))"
    else:
        miss = []
        if not ce.get("value_check"):
            miss.append("compiled_value_check relations record (rev-cb M-0221)")
        if not ce.get("kernel_pin_match"):
            miss.append("RunHeader.profile.compilation.generated_kernels.kernel_set_sha256 non-null in the MATCH record (M-0219 pin — the Match capture's is the recorder's null placeholder" +
                        (f"; present only in commit_delta: {ce['kernel_pin_commit_source']} = {ce['kernel_pin_commit'][:12]}…" if ce.get("kernel_pin_commit") else "") + ")")
        if not eff:
            miss.append(f"match/capture/versions.json effective_execution.execution_label == {ex} (EXEC-02 read-back of the built engine)")
        why = ("missing: " + ", ".join(miss) + f" for a compiled row's Match — rev-pm M-0266: not graded from GM-01 alone (an eager run under a compiled header passes GM-01; "
               "a compiled record cannot); DANIEL M-0776 (3): the M-0285 hold is lifted — Match ✅ (P_compiled) needs the runtime kernel_set of the loaded code objects == the P_compiled artifact's + noninterference")
        if match.get("status") == "PASS":
            match["status"] = "PARTIAL"
        if match.get("status") == "PARTIAL":
            match["reasons"].append(why)
        match["compiled_match_basis"] = "insufficient record-level evidence (M-0266)"
    if att is None:
        for d in (build, match):
            _cap(d, f"missing: engine execution attestation for `{ex}` (engine_kwargs enforce_eager + cudagraph/compilation mode in the capture run header — M-0040/M-0081; runner summaries alone do not attest what the engine ran)")
        if build_d:
            _cap(build, build_d)
        return
    if att["verdict"] == "eager":
        for d in (build, match):
            _fail(d, f"execution substitution: the row is `{ex}` but the engine record attests eager ({att['source']}: enforce_eager={att.get('enforce_eager')}, cudagraph_mode={att.get('cudagraph_mode')}) — PLAN §3 forbids 'eager called compiled'")
        return
    if att["verdict"] != "compiled":
        for d in (build, match):
            _cap(d, f"missing: positive evidence of a compiled execution for `{ex}` (enforce_eager={att.get('enforce_eager')}, cudagraph_mode={att.get('cudagraph_mode')}, compilation_mode={att.get('compilation_mode')} — a compiled row needs enforce_eager False AND a cudagraph/compilation mode other than NONE; M-0081)")
    if build_d:
        _cap(build, build_d)


def _fill_build_from_contemporaneous(build: dict, evdir: str, rec: dict) -> None:
    """Build's cmd/target from contemporaneous sources of the SAME evidence dir (recorded row attempt, launch log, attempt.json) —
    never from what the row was intended to run."""
    if build.get("status") in ("-", "FAIL"):
        return
    if not build.get("cmd") and rec.get("build_cmd"):
        build["cmd"] = rec["build_cmd"]; build["cmd_source"] = "recorded command (row attempt)"
        _drop_missing(build, "cmd")
    if not build.get("cmd"):
        lg = first(evdir, "build_step.log", "build/step.log", "row.log")
        if lg:
            for line in open(lg, encoding="utf-8", errors="replace"):
                m = re.search(r"(derive_step\s.*)", line)
                if m:
                    build["cmd"] = m.group(1).strip()[:600]; build["cmd_source"] = f"launch log {os.path.basename(lg)} (contemporaneous)"
                    _drop_missing(build, "cmd")
                    break
    if not build.get("cmd"):
        aj = first(evdir, "attempt.json", "../attempt.json")
        if aj:
            try:
                A = load(aj)
                c0 = A.get("cmd") or (A.get("build") or {}).get("cmd") if isinstance(A.get("build"), dict) or A.get("cmd") else None
                if c0:
                    build["cmd"] = str(c0).strip()[:600]; build["cmd_source"] = "attempt.json (lane's recorded invocation, contemporaneous)"
                    _drop_missing(build, "cmd")
            except Exception:
                pass
    if not build.get("target") and rec.get("build_target"):
        build["target"] = rec["build_target"]; _drop_missing(build, "target")


def finalize(row_id: str, evdir: str, root: str = "out/gen/sweep", lane: str | None = None, pod: str | None = None, revision: str | None = None,
             release_tag: str | None = None, kind: str = "revalidation", workload: str | None = None, dry_run: bool = False, note: str | None = None,
             campaign: str | None = None, reuse_build: str | None = None, reuse_match: str | None = None) -> dict:
    rp = os.path.join(root, "rows", row_id + ".json")
    if not os.path.exists(rp):
        raise RecordRejected(row_id, [f"row_id: no record {rp} (rows keep their ids; no panel growth)"])
    panel = panel_ids(root)
    brd = board_rows(root).get(row_id) if panel and row_id not in panel else None
    if panel and row_id not in panel and not brd:
        raise RecordRejected(row_id, [f"row_id: {row_id!r} is not one of the {len(panel)} panel rows ({root}/{PANEL_FILE}) — no panel growth, no substitute configuration"])
    row = load(rp)
    if brd:
        row["non_panel"] = {"ordered_by": brd.get("ordered_by"), "since": brd.get("since"), "note": brd.get("note")}   # rendered beneath the table by board order; never counted
    wpath = workload or os.path.join(root, "workloads", row_id + ".json")
    wl = load(wpath) if os.path.exists(wpath) else {}
    if os.path.exists(wpath):
        wl["_sha256"] = sha256_file(wpath)
        wl["_path"] = wpath
    wlf = workload_facts(wl)
    if not wlf["B"]:
        wlf["B"] = int(row.get("config", {}).get("batch") or re.search(r"__b(\d+)__", row_id).group(1))
    st = stages_txt(evdir)
    rec = recorded_fields(row, evdir)   # contemporaneous recorded commands/settings from the attempt that produced this evidence dir
    build = derive_build(evdir, wl, wlf, st)
    _fill_build_from_contemporaneous(build, evdir, rec)
    match = derive_match(evdir, wl, wlf, build)
    if match.get("status") not in ("-", "FAIL") and not match.get("cmd") and rec.get("match_cmd"):
        match["cmd"] = rec["match_cmd"]; match["cmd_source"] = "recorded command (row attempt)"
    if match.get("status") not in ("-", "FAIL") and not match.get("comparator") and rec.get("match_comparator"):
        match["comparator"] = rec["match_comparator"]; match["comparator_source"] = "recorded in the row attempt"
        _drop_missing(match, "comparator name + version")
    A = attempt_json(evdir)
    pcomp = p_compiled_attempt(A)
    if pcomp:
        # [COORD RULING M-0819] the compiled-row branch: attempt.json build.kind == P_compiled → BUILD = the P_compiled derivation record,
        # MATCH = S2 checkpoint replay + runtime↔artifact identity (+ noninterference), COMMIT = S3 per record with form (A).  The eager
        # 'Commit-only rerun needs the reused Build/Match artifact' rule does not apply: the Build IS on this attempt (compiled_manifest.json).
        eager_build = build if build.get("status") not in ("-", None) else None
        build = derive_build_compiled(evdir, A, wl, wlf, row_id, row, build)
        if eager_build:
            build.setdefault("footnotes", []).append(f"an eager-shaped build_summary.json is also present ({eager_build.get('program_digest', '?')[:12] if eager_build.get('program_digest') else '?'}…) — not the Build of a compiled row (M-0819: P_compiled is)")
        match = derive_match_compiled(evdir, A, build, match, row_id, row)
        compiled_attestation_guard(row_id, row, evdir, A, build, match)
    else:
        # rev-pm M-0040: nothing on the rc3 Build/Match path reads the row's execution field; a *__bi-compiled row Built/Matched eagerly
        # is the 'eager called compiled' substitution PLAN §3 forbids. Until the capture attests the effective execution, such rows are ◐.
        execution_guard(row_id, row, evdir, A, build, match)
    # brief §9 evidence reuse: a Commit-only rerun references the exact reused Build/Match artifacts (identity checked here)
    build, match, reuse_rec = apply_reuse(evdir, A, reuse_build, reuse_match, wl, wlf, build, match)
    if build.get("reused_from") and os.path.isdir(build["reused_from"]):
        _fill_build_from_contemporaneous(build, build["reused_from"], recorded_fields(row, build["reused_from"]))
    if match.get("reused_from") and match.get("status") not in ("-", "FAIL"):
        rec2 = recorded_fields(row, match["reused_from"])
        if not match.get("cmd") and rec2.get("match_cmd"):
            match["cmd"] = rec2["match_cmd"]; match["cmd_source"] = "recorded command (row attempt, reused Match)"
        if not match.get("comparator") and rec2.get("match_comparator"):
            match["comparator"] = rec2["match_comparator"]; match["comparator_source"] = "recorded in the row attempt (reused Match)"
            _drop_missing(match, "comparator name + version")
    commit = derive_commit(evdir, wl, wlf, build, row, rev_hint=stage_revision_hint(st, A, "commit"))
    ex_row = str((row.get("config") or {}).get("execution") or (row_id.split("__")[-1] if "__" in row_id else "")).strip().lower()
    if "compiled" in ex_row:
        # DANIEL'S DECISION M-0285 (b) (M-0256 (2)): the INSTRUMENTED compiled execution differs numerically from PRODUCTION compiled (record ops
        # change Inductor fusion; logprobs differ every step) — a commitment mechanism that changes the computation it commits to is not
        # committing to the workload's execution.  Commit ✅ only after the instrumented graph is bit-identical to production compiled on
        # top-k logprobs (acceptance: compiled_ab_logprobs, 0 differing positions); the overhead follows its Commit (capped below).
        _partial(commit, "instrumentation perturbs numerics: the instrumented compiled execution differs from production compiled (record ops change Inductor fusion — M-0256 (2): logprobs differ at every step, max |d| 0.13); Commit ✅ after the instrumented graph is bit-identical to production compiled on top-k logprobs (acceptance test compiled_ab_logprobs, 0 differing positions) — Daniel's decision M-0285 (b), HOLD at ◐")
        commit["compiled_commit_basis"] = "instrumented compiled execution, labelled (M-0285 (b): not ✅ until the faithfulness acceptance test passes)"
    if pcomp:
        compiled_commit_adjust(evdir, A, build, match, commit)   # M-0819: manifest of record = compiled_manifest, MAN-05 = P_compiled verify, hold vs noninterference
    apply_row_holds(row_id, row, wlf, commit, evdir)
    if commit.get("status") not in ("-", "FAIL"):
        if not commit.get("cmd") and rec.get("commit_cmd"):
            commit["cmd"] = rec["commit_cmd"]; commit["cmd_source"] = "recorded command (row attempt)"
        if not commit.get("retention") and rec.get("retention"):
            commit["retention"] = rec["retention"]; commit["retention_source"] = "recorded env/command (row attempt)"
            _drop_missing(commit, "retention mode")
        if rec.get("collector_variant"):
            commit["collector_variant"] = rec["collector_variant"]
    if reuse_rec and commit.get("status") not in ("-", "FAIL") and commit.get("program_digest") and build.get("program_digest"):
        # the manifest of the Commit-only rerun must derive from the reused Build's Program (COORD 19:25Z ruling 5)
        if commit["program_digest"] not in build_program_family(build):
            commit["status"] = "FAIL"
            commit["reasons"].append(f"Commit-only rerun: manifest Program {str(commit['program_digest'])[:16]} ≠ reused Build Program {str(build['program_digest'])[:16]}"
                                     + (f" / workload Program {str(build.get('global_program_digest'))[:16]}" if build.get("global_program_digest") else "") + " (brief §9: reused artifacts must be the same Program)")
    if reuse_rec and commit.get("status") not in ("-", "FAIL") and not commit.get("program_digest"):
        # the checker verdict may not have recorded it (or derive_commit returned early on a named ◐): the shipped manifest of record /
        # MAN-05 verify record name the Program the required manifest derives from — read, never inferred
        mp = first(evdir, "manifest.json", "manifest.json.gz", "commit/manifest.json", "commit/manifest_verify.json", "required_manifest.json")
        if mp:
            try:
                Mf = load(mp)
                pd_ = Mf.get("program_digest") or Mf.get("used_program_digest") if isinstance(Mf, dict) else None
                if pd_:
                    commit["program_digest"] = pd_
                    commit["manifest_artifact"] = os.path.relpath(mp)
                    if pd_ not in build_program_family(build):
                        commit["status"] = "FAIL"
                        commit["reasons"].append(f"Commit-only rerun: manifest Program {str(pd_)[:16]} ≠ reused Build Program family {', '.join(sorted(str(d)[:12] for d in build_program_family(build)))} (brief §9)")
            except Exception:  # noqa: BLE001
                pass
    if reuse_rec and commit.get("status") not in ("-", "FAIL") and not commit.get("program_digest"):
        commit["status"] = "PARTIAL"
        commit["reasons"].append("missing: manifest program_digest (Commit-only rerun — the coverage record / manifest.json must name the Program the required manifest derives from, to check it against the reused Build)")
    # brief §11: actual generated lengths + achieved concurrency; a Match/Commit ✅ without them is ◐ with the field named
    wact = derive_workload_actual(evdir, wlf, A)
    if wact.get("generated_lengths") is None and match.get("reused_from") and os.path.isdir(match["reused_from"]):
        wact = derive_workload_actual(match["reused_from"], wlf, A)     # Commit-only rerun: what ran is the reused Match's chronology
        wact["from_reused_match"] = match["reused_from"]
    if wact.get("request_id_convention") == "uuid":
        match.setdefault("footnotes", []).append("request ids in global_match.json are UUIDs — COORD 19:25Z ruling 2: request ids are r<k> (a UUID is a workload id); consumer fix pending (rc1 evidence is expected to show this)")
    for stg, d in (("match", match), ("commit", commit)):
        if d.get("status") == "PASS":
            miss = []
            gl = wact.get("generated_lengths")
            if not (isinstance(gl, list) and gl and all(isinstance(x, int) and x > 0 for x in gl)):
                miss.append("workload_actual.generated_lengths (actual generated length per request — brief §11)")
            elif wlf["B"] and len(gl) != wlf["B"]:
                miss.append(f"workload_actual.generated_lengths for all {wlf['B']} requests (have {len(gl)})")
            if wlf["B"] >= 2 and not (isinstance(wact.get("achieved_concurrency"), int) and wact["achieved_concurrency"] >= 1):
                miss.append("workload_actual.achieved_concurrency (max requests simultaneously in flight from the observed chronology — brief §11)")
            if miss:
                d["status"] = "PARTIAL"
                d["reasons"].append("missing: " + ", ".join(miss))
    if wlf["B"] >= 2 and isinstance(wact.get("achieved_concurrency"), int) and wact["achieved_concurrency"] < wlf["B"]:
        for d in (match, commit):
            if d.get("status") in ("PASS", "PARTIAL"):
                d.setdefault("footnotes", []).append(f"achieved concurrency {wact['achieved_concurrency']} < declared batch {wlf['B']} (declared arrivals/lengths — the table shows the declared batch; brief §11)")
    bench = derive_bench(evdir, commit)
    inf = derive_infeasible(evdir)
    stages = {"build": build, "match": match, "commit": commit, "bench": bench}
    if inf:
        stg = inf["stage"] if inf["stage"] in stages else "build"
        for k in ("build", "match", "commit"):
            if stages[k]["status"] in ("PASS", "PARTIAL") and k != stg:
                continue
            if k == stg:
                stages[k]["status"] = "INFEASIBLE"
                stages[k]["reasons"].append(f"infeasible on the declared hardware (COORD 19:25Z ruling 6): {inf['error'][:300]}")
                stages[k]["infeasible"] = inf
        bench["status"] = "-"
    # revision: the first *sha* among CLI, attempt.json (release_commit | verity_commit | revision), stages.txt release line — a producer
    # field that is not a sha is reported (producer defect), never repaired into one
    rev, rev_src, producer_defects = None, None, []
    for src, cand in (("--revision", revision), ("attempt.json release_commit", A.get("release_commit")), ("attempt.json verity_commit", A.get("verity_commit")),
                      ("attempt.json revision", A.get("revision")), ("stages.txt release line", st.get("revision"))):
        if cand is None or cand == "":
            continue
        if isinstance(cand, str) and SHA_RE.match(cand.strip()):
            rev, rev_src = cand.strip(), src
            break
        producer_defects.append(f"{src}: {json.dumps(cand, default=str)[:80]} is not a git sha")
    rt = release_tag or A.get("release_tag") or (st.get("release") if str(st.get("release") or "").startswith("r1") else None)
    if rt is not None and not isinstance(rt, str):
        producer_defects.append(f"attempt.json release_tag: {json.dumps(rt, default=str)[:80]} is not a string")
        rt = st.get("release") if str(st.get("release") or "").startswith("r1") else None
    # a producer that declares a lane tree on top of the release (attempt.json tree_note / a composed verity_commit "rcN + lane/x") ran a
    # lane-only revision: the row is never shown as a release-tag pass on the producer's own word (brief §9; rendered †)
    # rev-pm M-0118: the `release <tag> <sha>` line in stages.txt / release.json comes from the POD's RELEASE.json (the installed
    # release), not from the tree whose code ran.  attempt.json `tree` / `tree_note` / cmd VERITOR_REPO name the tree that ran.
    tree_txt = " ".join(json.dumps(A.get(k), default=str) for k in ("tree", "tree_note", "verity_commit", "revision", "release_tag") if A.get(k) is not None)
    cmd_txt = A.get("cmd") if isinstance(A.get("cmd"), str) else json.dumps(A.get("cmd"), default=str) if A.get("cmd") is not None else ""
    m_repo = re.search(r"VERITOR_REPO=(\S+)", cmd_txt)
    repo_path = m_repo.group(1) if m_repo else (A.get("tree").split(" = ")[0].strip() if isinstance(A.get("tree"), str) and " = " in A.get("tree") else None)
    st_rev = st.get("revision")
    st_tag = st.get("release")
    # REL-14 (rel M-0149) / EXEC-03 (exec M-0155): `release UNRELEASED <ts> commit=<sha> tree=<path> branch=<b> dirty=<d>` in stages.txt
    # and attempt.json executing_tree {path, commit, dirty, is_release_tree, diff_vs_release} name the tree whose code ran
    ET = A.get("executing_tree") if isinstance(A.get("executing_tree"), dict) else {}
    rel_line = next((l for l in st.get("lines", []) if l.startswith("release ")), "")
    unreleased = str(st_tag or "").upper().startswith("UNRELEASED")
    if unreleased:
        mt_ = re.search(r"\btree=(\S+)", rel_line)
        repo_path = repo_path or (mt_.group(1) if mt_ else None)
        tree_txt += f" stages.txt: {rel_line[:200]}"
        st_tag = None
    if ET:
        tree_txt += " executing_tree: " + json.dumps({k: ET.get(k) for k in ("path", "commit", "branch", "dirty", "is_release_tree", "diff_vs_release") if k in ET}, default=str)
        repo_path = repo_path or ET.get("path")
    lane_tree = (unreleased or (ET.get("is_release_tree") is False) or
                 bool(re.search(r"lane/|\+ ?lane|lane tree|lane-only|/lane-", tree_txt + " " + (repo_path or ""), re.I)))
    if ET.get("is_release_tree") is True and not unreleased:
        lane_tree = False
    # [rev RESULT M-0020 (1) on exec M-0011 — COORD ruling 6 two-tree chain] a tree stamped UNRELEASED (no RELEASE.json on the pod) whose
    # commit IS a released commit (equality on the commit id — a git tag r1x-* in this repo / the board's RELEASE posts) and is not dirty ran
    # the release's code: not a lane tree.  The Build/Match stages of a chain may run on it (22668e4f6 = r15-int-6b) and the Commit on the
    # later release; both release lines stay in stages.txt (the honest record), per-stage revisions are kept below.
    released_stamp = None
    if lane_tree and not re.search(r"lane/|\+ ?lane|lane tree|lane-only|/lane-", tree_txt + " " + (repo_path or ""), re.I):
        unrel_commits = [c for c in ([ET.get("commit")] + [(re.search(r"commit=([0-9a-f]{7,40})", l) or [None, None])[1]
                                                            for l in st.get("lines", []) if l.startswith("release UNRELEASED")]) if c]
        dirty = as_bool(ET.get("dirty")) is True or any(re.search(r"\bdirty=(true|1|yes)\b", l, re.I) for l in st.get("lines", []) if l.startswith("release UNRELEASED"))
        tags = [released_tag_of(c) for c in unrel_commits]
        if unrel_commits and all(tags) and not dirty and (not ET.get("commit") or not any(l.startswith("release UNRELEASED") for l in st.get("lines", []))
                                                          or all(c.startswith(unrel_commits[0][:7]) or unrel_commits[0].startswith(c[:7]) for c in unrel_commits)):
            lane_tree = False
            released_stamp = f"tree stamped UNRELEASED (no RELEASE.json) but its commit {unrel_commits[0][:9]} = release {tags[0]} — a released commit, not a lane tree (rev M-0020 (1), COORD ruling 6)"
            if unreleased and not st_tag:
                # every release line is an UNRELEASED stamp of a released commit: the attempt's release IS that tag
                rt = rt or tags[0]
                if rev is None or rev_src == "stages.txt release line":
                    rev, rev_src = unrel_commits[0], "stages.txt UNRELEASED stamp of a released commit (rev M-0020 (1))"
    decl_base = None
    for k in ("release_commit", "verity_commit", "base_commit"):
        if isinstance(A.get(k), str) and SHA_RE.match(A[k].strip()):
            decl_base = A[k].strip()
            break
    if decl_base is None and isinstance(A.get("tree"), str):
        mb = re.search(r"\b(r1\d-(?:rc\d+[a-z]?|int-\d+[a-z]?|final[\w-]*))\s+([0-9a-f]{7,40})\b", A["tree"])
        if mb:
            decl_base = mb.group(2)
            if not (release_tag or A.get("release_tag")):
                rt = mb.group(1)
    exec_tree = None
    if lane_tree:
        note = re.sub(r"\s+", " ", tree_txt)[:200]
        att_tree_note = f"producer declares a lane tree: {note}"
        if st_tag and st_rev and decl_base and not (st_rev.startswith(decl_base) or decl_base.startswith(st_rev)):
            producer_defects.append(f"stages.txt release line names {st_tag} commit={st_rev} (the pod's installed RELEASE.json) but the tree that ran is {repo_path or 'a lane tree'} on base {decl_base} — not this record's revision (M-0118)")
        # COORD RULING M-0132: the revision of record is the tree whose code executed — the lane-tree commit; release_tag null;
        # the TABLE row is finalized from a lane-tree run only when its code diff vs the release is empty on the executed path
        # (the finalizer checks the diff in the repo, not the producer's claim)
        exec_tree = executing_tree(A, repo_path, decl_base or st_rev, tree_txt)
        if exec_tree.get("commit"):
            rev, rev_src = exec_tree["commit"], "lane-tree commit (executing tree; COORD M-0132)"
        elif decl_base and (rev is None or rev_src == "stages.txt release line"):
            rev, rev_src = decl_base, "attempt.json declared base (lane tree; M-0118)"
        rt = None
        diff = exec_tree.get("diff_vs_release")
        if diff is None:
            why = f"lane-tree run from {exec_tree.get('path') or 'an undeclared path'}: {exec_tree.get('diff_error') or 'the lane-tree commit is not recorded / not in the report repo'} — the code diff vs the release {exec_tree.get('base') or '?'} cannot be checked; the table row is finalized from a release-tree run or a lane-tree run with an empty executed-path diff (COORD M-0132 (3))"
            for d in stages.values():
                _partial(d, why)
        elif diff:
            why = f"lane-tree run {exec_tree['commit'][:9]}: code differs from the release {str(exec_tree.get('base'))[:9]} in {len(diff)} executed-path file(s) ({', '.join(diff[:3])}{'…' if len(diff) > 3 else ''}) — a RESULT for the FIX_READY, not a table ✅ until the fix is released and the row re-run (COORD M-0132 (3))"
            for d in stages.values():
                _partial(d, why)
        else:
            for d in stages.values():
                d.setdefault("footnotes", []).append(f"lane-tree run {exec_tree['commit'][:9]} ({exec_tree.get('path')}): code diff vs release {str(exec_tree.get('base'))[:9]} empty on the executed path (verity_capture/, tests excluded) — finalizable (COORD M-0132 (3))")
    else:
        att_tree_note = A.get("tree_note") if isinstance(A.get("tree_note"), str) else None
        if released_stamp:
            att_tree_note = (att_tree_note + "; " if att_tree_note else "") + released_stamp
            for d in stages.values():
                d.setdefault("footnotes", []).append(released_stamp)
        if st_rev and decl_base and rev == decl_base and not (st_rev.startswith(decl_base) or decl_base.startswith(st_rev)) and len([l for l in st.get("lines", []) if l.startswith("release")]) == 1:
            # the producer's declared tree and the pod's release line disagree and no lane tree is declared: not this record's revision
            raise RecordRejected(row_id, [f"revision: attempt.json release_commit {decl_base} ≠ stages.txt release line commit={st_rev} ({st_tag}) — the stages.txt line is the pod's installed RELEASE.json, not necessarily the tree that ran; the producer must record the tree (attempt.json tree = <path> = <base tag sha> [+ overlay]) — M-0118"])
    release_tag = rt
    pod = pod or A.get("pod") or (row.get("_attempt") or {}).get("pod")
    for k, d in stages.items():
        d["revision"] = rev
        d["at"] = (st.get(k) or {}).get("at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        d["reasons"] = [r for r in d["reasons"] if r]
    # revision consistency across stages (same-attempt rule) — the evidence dir is one attempt; stages.txt carries one release line
    rels = [l for l in st.get("lines", []) if l.startswith("release")]
    rel_commits = {(re.search(r"commit=([0-9a-f]{7,40})", l) or [None, None])[1] for l in rels}
    # per-stage revision: the last release line preceding each stage line
    cur = None
    per_stage_rev: dict[str, str | None] = {}
    for l in st.get("lines", []):
        if l.startswith("release"):
            cur = (re.search(r"commit=([0-9a-f]{7,40})", l) or [None, None])[1]
        else:
            k = l.split(" ", 1)[0]
            if k in stages:
                per_stage_rev[k] = cur
    if per_stage_rev:
        for k, d in stages.items():
            if per_stage_rev.get(k):
                d["revision"] = per_stage_rev[k]
                if len(rel_commits) > 1:
                    # [rev M-0020] two-tree chain: the stage names the release its tree commit equals (UNRELEASED stamp of a released commit included)
                    d["release_tag"] = released_tag_of(per_stage_rev[k]) or ("UNRELEASED" if any(l.startswith("release UNRELEASED") and per_stage_rev[k] in l for l in st.get("lines", [])) else None)
    # [MAN-05] no manifest-verify record: graded against the tree the Commit stage ran on (◐ when it ships the tool, named otherwise)
    apply_man05_absence(stages["commit"], stages["commit"].get("revision") or rev)
    if len(rels) > 1 and len(rel_commits) > 1:
        digs = {stages[k].get("program_digest") for k in ("build", "match", "commit") if stages[k].get("program_digest")}
        bpd = stages["build"].get("program_digest")
        # [R14 moe / rc3; exec M-0562 (2)] the Commit manifest may derive from the row's REQUEST Program (a per-shape Build digest) or, at
        # B≥2, from the GP-01 WORKLOAD Program (the Build's own workload_digest): same Program family, same attempt
        if bpd and all(d in build_program_family(stages["build"]) for d in digs):
            digs = {bpd}
        if len(digs) == 1 and stages["build"].get("program_digest"):
            # COORD 17:35Z item 2: a Commit stage re-run on a later release against the same Program digest and the same workload
            # manifest as the attempt's Build/Match is the same attempt (builder unchanged); recorded, not hidden
            for d in stages.values():
                d.setdefault("footnotes", []).append("stages span releases " + " → ".join(sorted(c for c in rel_commits if c)) + f" against one Program digest {list(digs)[0][:12]}… (same attempt per COORD 17:35Z item 2)")
        else:
            for d in stages.values():
                if d["status"] in ("PASS", "DONE"):
                    d["status"] = "PARTIAL"
                    d["reasons"].append(f"{len(rels)} release lines with {len(rel_commits)} distinct commits in stages.txt and the stages' Program digests do not agree — the stages span revisions (same-attempt rule)")
    # COORD 17:35Z item 5: value correspondence is a separate labelled check, never folded into Commit ✅ silently
    vc = None
    for cand in (first(evdir, "value_correspondence.json", "commit/value_correspondence.json"),):
        if cand:
            try:
                vc = load(cand)
            except Exception:
                vc = None
    if vc is None and isinstance(commit.get("value_correspondence"), dict):
        vc = commit["value_correspondence"]
    if vc is None or not vc.get("method"):
        vc = commit.get("value_correspondence") if isinstance(commit.get("value_correspondence"), dict) and commit["value_correspondence"].get("method") else {"method": None, "result": "not checked"}
    commit["value_correspondence"] = vc
    # the overhead is a measurement of THIS attempt's Commit: whatever capped Commit after the bench derivation caps the overhead too
    if bench.get("status") == "DONE" and commit.get("status") != "PASS":
        bench["status"] = "PARTIAL"
        bench["reasons"].append(f"overhead of a Commit that is {commit.get('status')} under r14-v1 — shown ◐ with its number; missing: Commit ✅ of the same attempt")
    camp = campaign or (str(A.get("campaign")) if A.get("campaign") in CAMPAIGNS else None) or CAMPAIGN
    att = {"campaign": camp, "rules_version": RULES_VERSION, "kind": kind, "lane": lane or A.get("lane") or "report (finalizer)", "pod": pod,
           "release_tag": release_tag, "revision": rev,
           "at": evidence_time(st, row, kind),
           "finalized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_stamp": CODE_STAMP, "evidence_dir": os.path.relpath(evdir), "workload_manifest_digest": wl.get("_sha256"),
           "workload": {"B": wlf["B"], "sampling": wlf["sampling_variant"], "staggered": wlf["staggered"], "input_lengths": wlf["input_lengths"], "output_caps": wlf["output_caps"]} if wl.get("_sha256") else None,
           "workload_actual": wact,
           "status": "INFEASIBLE" if inf else status_from(stages), "finalizer": "verity_capture.sweep.finalize/v1", "notes": note or "written by the finalizer from the evidence directory; nobody typed PASS",
           **stages}
    if rev_src:
        att["revision_source"] = rev_src
    if att_tree_note:
        att["tree_note"] = att_tree_note
    if exec_tree:
        att["executing_tree"] = exec_tree
    if producer_defects:
        att["producer_defects"] = producer_defects   # malformed producer fields seen (not repaired; the producer fixes the producer)
    if reuse_rec:
        att["reuse"] = reuse_rec
    if inf:
        att["infeasible"] = inf
    reg = regression_marker(row, att)
    if reg:
        att["regression"] = reg
    unsup = unsuperseded_fail_marker(row, att)
    if unsup:
        att["unsuperseded_fail"] = unsup
    # write-time validation (brief §10): a malformed attempt is refused with the field names; nothing is written
    problems = validate_attempt(row_id, row, att, root)
    att["validated"] = {"at": att["finalized_at"], "problems": problems, "accepted": not problems}
    if problems and not dry_run:
        raise RecordRejected(row_id, problems)
    if not dry_run:
        # re-finalizing the same evidence directory (a stage was added / evidence completed) replaces the earlier finalizer attempt
        # for that directory; finalizer attempts for other directories and all lane/R13 attempts are kept
        row["attempts"] = [a for a in (row.get("attempts") or []) if not (isinstance(a, dict) and a.get("finalizer") and a.get("evidence_dir") == att["evidence_dir"])]
        row.setdefault("attempts", []).append(att)
        # the headline is the finalizer attempt whose *evidence* is latest (not the one finalized last)
        head = max((a for a in row["attempts"] if isinstance(a, dict) and a.get("finalizer")), key=lambda a: str(a.get("at") or ""))
        row["status"] = head["status"]
        row["rules_version"] = RULES_VERSION
        row["campaign"] = head.get("campaign") or camp
        row["release_tag"] = head.get("release_tag")
        row["updated_at"] = att["finalized_at"]
        if head.get("revision"):
            row["revision"] = head["revision"]
        for k in ("build", "match", "commit", "bench"):
            row[k] = dict(head[k])
        atomic_write_json(rp, row)
    return att


GLYPH = {"PASS": "✅", "DONE": "✅", "PARTIAL": "◐", "FAIL": "❌", "-": "—", "INFEASIBLE": "⊘"}


def evidence_time(st: dict, row: dict, kind: str) -> str:
    """When the evidence was produced (headline ordering): stage timestamps from stages.txt; for offline repair/revalidation of R13
    evidence the R13 attempt's own time; a fresh execution without stage times is stamped now."""
    ts = [str((st.get(k) or {}).get("at")) for k in ("build", "match", "commit", "bench") if (st.get(k) or {}).get("at")]
    if ts:
        return max(ts)
    if kind in ("repair", "revalidation"):
        prev = [str(a.get("at") or a.get("utc") or "") for a in (row.get("attempts") or []) if isinstance(a, dict) and not a.get("finalizer") and not a.get("rules_version")]
        prev = [t for t in prev if t and t[:4] == "2026" and t < "2026-09-19T16:46"]  # R13 evidence predates R14 T0
        if prev:
            return max(prev)
        return str(row.get("updated_at") or "2026-09-19T14:35:00Z")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("row_id")
    ap.add_argument("evidence_dir")
    ap.add_argument("--root", default="out/gen/sweep")
    ap.add_argument("--lane")
    ap.add_argument("--pod")
    ap.add_argument("--revision")
    ap.add_argument("--release-tag")
    ap.add_argument("--kind", choices=["repair", "revalidation", "execution"], default="revalidation")
    ap.add_argument("--workload")
    ap.add_argument("--note")
    ap.add_argument("--campaign", choices=list(CAMPAIGNS), default=None, help=f"campaign that produced the attempt (default: attempt.json campaign, else {CAMPAIGN})")
    ap.add_argument("--reuse-build", metavar="DIR", help="Commit-only rerun: the evidence dir whose Build artifact is reused (brief §9); attempt.json `reuse.build` also accepted")
    ap.add_argument("--reuse-match", metavar="DIR", help="Commit-only rerun: the evidence dir whose Match artifact is reused (brief §9); attempt.json `reuse.match` also accepted")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        att = finalize(a.row_id, a.evidence_dir, a.root, a.lane, a.pod, a.revision, a.release_tag, a.kind, a.workload, a.dry_run, a.note,
                       a.campaign, a.reuse_build, a.reuse_match)
    except RecordRejected as rj:
        print(f"[finalize {RULES_VERSION}] REJECTED {rj.row_id} — record not written; fix the producer, never the record:")
        for p in rj.problems:
            print(f"  ✗ {p}")
        return 2
    print(f"[finalize {RULES_VERSION}] {a.row_id}: Build {GLYPH[att['build']['status']]} Match {GLYPH[att['match']['status']]} "
          f"Commit {GLYPH[att['commit']['status']]} overhead {GLYPH[att['bench']['status']]}"
          + (f" {att['bench'].get('overhead_rel', 0) * 100:+.1f}%" if att['bench'].get('overhead_rel') is not None else "") + f" → status {att['status']}" + (" (dry run)" if a.dry_run else ""))
    for k in ("build", "match", "commit", "bench"):
        for r in att[k]["reasons"]:
            print(f"  {k}: {r}")
    if att.get("validated", {}).get("problems"):
        print("  would be REJECTED at write time:")
        for p in att["validated"]["problems"]:
            print(f"  ✗ {p}")
    if att.get("regression"):
        for w in att["regression"]["stages"]:
            print(f"  regression: {w['stage']} was {w['was']} at {w['earlier_attempt'].get('release_tag') or w['earlier_attempt'].get('revision')} → now {w['now']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
