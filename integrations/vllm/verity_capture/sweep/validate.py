"""Load + validate R12 sweep records (report owner; was record.py — renamed because lane bc's `record.py` is the record *writer* CLI) (`out/gen/sweep/{rows,failures,fixes}/*.json`) against SCHEMA.md.

Pure python, no torch.  Tolerates missing optional fields; every violation is reported with file + field.
Also computes, per row, the *evidence* status of each stage (PASS with all required evidence fields -> green;
PASS with missing evidence -> partial) so the renderer can apply the BRIEF status convention (✅ ◐ ❌ —).
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

ROW_SCHEMA = "verity-sweep/row/v1"
FAILURE_SCHEMA = "verity-sweep/failure/v1"
FIX_SCHEMA = "verity-sweep/fix/v1"

ROW_STATUSES = (
    "QUEUED", "BUILD_PASS", "BUILD_FAIL", "MATCH_PASS", "MATCH_FAIL",
    "COMMIT_PASS", "COMMIT_FAIL", "BENCHMARKED", "NOT_RUN", "BUILD_PARTIAL", "MATCH_PARTIAL", "COMMIT_PARTIAL",
)
STAGES = ("build", "match", "commit", "bench")
STAGE_STATUSES = {"build": ("PASS", "FAIL", "PARTIAL", "INFEASIBLE", "-"), "match": ("PASS", "FAIL", "PARTIAL", "INFEASIBLE", "-"),
                  "commit": ("PASS", "FAIL", "PARTIAL", "INFEASIBLE", "-"), "bench": ("DONE", "FAIL", "PARTIAL", "-")}  # PARTIAL = finalizer: evidence insufficient (◐); INFEASIBLE = cannot serve on the declared hardware (COORD 19:25Z ruling 6, ⊘)
FAILURE_STAGES = ("build", "match", "commit", "bench", "infra")
FAILURE_CLASSES = (
    "generic builder/lowering rule", "target/numerical semantics", "runtime correspondence",
    "commitment acquisition", "TP/collective provenance", "MoE/routing", "state/cache behavior",
    "execution-mode/fusion", "configuration/applicability", "infrastructure/environment",
    "apparently model-specific",
)

# SCHEMA §2 panel order (sort key for the matrix) + display names + MoE flag.
MODEL_PANEL: list[tuple[str, str, bool]] = [
    ("smollm2-135m", "SmolLM2-135M", False),
    ("llama32-1b", "Llama-3.2-1B", False),
    ("qwen25-15b", "Qwen2.5-1.5B", False),
    ("phi3-mini", "Phi-3-mini", False),
    ("gemma2-2b", "Gemma-2-2B", False),
    ("mistral-7b", "Mistral-7B", False),
    ("olmoe-1b-7b", "OLMoE-1B-7B", True),
    ("qwen3-4b", "Qwen3-4B", False),
    ("qwen3-4b-fp8", "Qwen3-4B-FP8", False),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B", True),
]
MODEL_ORDER = {m: i for i, (m, _, _) in enumerate(MODEL_PANEL)}
MODEL_DISPLAY = {m: d for m, d, _ in MODEL_PANEL}
MODEL_MOE = {m: moe for m, _, moe in MODEL_PANEL}
DTYPES = ("bf16", "fp16", "fp8")
HW_ORDER = ("l40s", "a100", "h100", "rtx4090")
HW_DISPLAY = {"l40s": "L40S", "a100": "A100", "h100": "H100", "rtx4090": "RTX 4090"}
EXEC_DISPLAY = {"bi-eager": "BI eager", "bi-compiled": "BI compiled", "bi-cudagraph": "BI + CUDA graphs"}
WORKLOAD_DISPLAY = {"mixed": "mixed", "mixed-arrivals": "mixed arrivals", "fixed": "fixed"}
SAMPLING_DISPLAY = {"greedy": "greedy", "stoch-t0.8-p0.95": "temperature=0.8, top_p=0.95"}

# Evidence a PASS/DONE stage must carry to be ✅ rather than ◐ (BRIEF glossary Build/Match/Commit/overhead).
EVIDENCE_FIELDS = {
    "build": ("program_digest", "artifact", "cmd"),
    "match": ("comparator", "verdict", "control_vs_instrumented_tokens_equal", "artifact", "cmd"),
    "commit": ("required_classes", "bound_classes", "openings_checked", "openings_ok", "tokens_equal",
               "backlog_max", "artifact", "cmd"),
    "bench": ("rounds", "control_s", "treatment_s", "artifact", "cmd"),
}


@dataclass
class Violation:
    file: str
    field: str
    message: str
    severity: str = "error"  # "error" (blocks) | "advisory" (optional SCHEMA field absent)

    def __str__(self) -> str:
        return f"{self.file}: {self.field}: {self.message}"


@dataclass
class Records:
    root: str
    rows: dict[str, dict] = field(default_factory=dict)
    demos: dict[str, dict] = field(default_factory=dict)  # r14 fixtures outside the 75-row panel
    failures: dict[str, dict] = field(default_factory=dict)
    fixes: dict[str, dict] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    # row_id -> list of (stage, field) that a PASS/DONE stage is missing
    missing_evidence: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # row_id -> file path (relative to root)
    row_files: dict[str, str] = field(default_factory=dict)
    # rows whose Commit PASS came from the fail-open harness (pre-X-03) -> rendered ◐ ("invalidated: fail-open harness")
    invalidated: list[str] = field(default_factory=list)

    def add(self, file: str, fld: str, msg: str, severity: str = "error") -> None:
        self.violations.append(Violation(file, fld, msg, severity))

    def errors(self) -> list["Violation"]:
        return sorted((v for v in self.violations if v.severity == "error"), key=lambda v: (v.file, v.field))

    def advisories(self) -> list["Violation"]:
        return sorted((v for v in self.violations if v.severity != "error"), key=lambda v: (v.file, v.field))


# ----------------------------------------------------------------------------------------------------------------
# loading


def _load_dir(root: str, sub: str, recs: Records) -> list[tuple[str, dict]]:
    d = os.path.join(root, sub)
    out: list[tuple[str, dict]] = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json") or name.startswith("."):
            continue
        rel = f"{sub}/{name}"
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as e:  # noqa: BLE001
            recs.add(rel, "<file>", f"unreadable JSON: {e}")
            continue
        if not isinstance(obj, dict):
            recs.add(rel, "<file>", "top level is not an object")
            continue
        out.append((rel, obj))
    return out


def load(root: str) -> Records:
    recs = Records(root=root)
    rows = _load_dir(root, "rows", recs)
    fails = _load_dir(root, "failures", recs)
    fixes = _load_dir(root, "fixes", recs)

    for rel, obj in rows:
        stem = os.path.basename(rel)[:-5]
        rid = obj.get("row_id")
        if not isinstance(rid, str) or not rid:
            recs.add(rel, "row_id", "missing; using filename")
            rid = stem
        elif rid != stem:
            recs.add(rel, "row_id", f"{rid!r} != filename stem {stem!r}")
        if rid in recs.rows:
            recs.add(rel, "row_id", f"duplicate row id (also in {recs.row_files[rid]})")
            continue
        if CAMPAIGN == "r14" and PANEL_IDS and rid not in PANEL_IDS:
            # COORD 18:09Z: the panel is the 75 R13 configurations; fixtures added in R14 (staggered/B2-unequal/mixed-arrivals)
            # are shared-fix demonstrations listed in the report's shared-standards section, not table rows
            recs.demos[rid] = obj
            recs.add(rel, "row_id", "R14 fixture outside the 75-row panel — listed under shared standards, not a table row (COORD 18:09Z)", "advisory")
            continue
        recs.rows[rid] = obj
        recs.row_files[rid] = rel
    for rel, obj in fails:
        stem = os.path.basename(rel)[:-5]
        fid = obj.get("failure_id")
        if not isinstance(fid, str) or not fid:
            recs.add(rel, "failure_id", "missing; using filename")
            fid = stem
        elif fid != stem:
            recs.add(rel, "failure_id", f"{fid!r} != filename stem {stem!r}")
        if fid in recs.failures:
            recs.add(rel, "failure_id", "duplicate failure id")
            continue
        obj.setdefault("_file", rel)
        recs.failures[fid] = obj
    for rel, obj in fixes:
        stem = os.path.basename(rel)[:-5]
        xid = obj.get("fix_id")
        if not isinstance(xid, str) or not xid:
            recs.add(rel, "fix_id", "missing; using filename")
            xid = stem
        elif xid != stem:
            recs.add(rel, "fix_id", f"{xid!r} != filename stem {stem!r}")
        if xid in recs.fixes:
            recs.add(rel, "fix_id", "duplicate fix id")
            continue
        obj.setdefault("_file", rel)
        recs.fixes[xid] = obj

    for rid, obj in recs.rows.items():
        _apply_attempts(recs, recs.row_files[rid], obj)
    # rows outside the panel the coordinator ordered rendered (out/gen/sweep/BOARD_ROWS.json, COORD M-0522 (2)): headline attempt
    # selected by the same rule, shown beneath the 75-row table by render.board_rows_table, never counted
    brd = _board_rows(root)
    for rid, obj in recs.demos.items():
        if rid in brd:
            try:
                _apply_attempts(recs, f"rows/{rid}.json", obj)
                obj["_board_order"] = brd[rid]
            except Exception as e:  # noqa: BLE001
                recs.add(f"rows/{rid}.json", "attempts", f"board-ordered non-panel row: headline selection failed ({type(e).__name__}: {e})", "advisory")
    for rid, obj in recs.rows.items():
        _validate_row(recs, recs.row_files[rid], rid, obj)
    for fid, obj in recs.failures.items():
        _validate_failure(recs, obj["_file"], fid, obj)
    for xid, obj in recs.fixes.items():
        _validate_fix(recs, obj["_file"], xid, obj)
    return recs


RELEASE_TAG_RE = re.compile(r"^(r13-rel\d+|r14-(base|rc\d+)|r15-(int|rc)-?\d+[a-z]?)$")  # r14/r15 tags announced by rel in out/gen/sweep/RELEASES.md


def m_is_dense(lane_txt: str) -> bool:
    return "dense" in lane_txt or "dse" in lane_txt


def attempt_is_v2_layout(a: dict) -> bool:
    """True for an attempt recorded on the experimental `chunk-leaf-v2` commitment layout (attempt- or commit-level `layout`, or the
    commit cmd).  Coordinator ruling 08:35Z: layout change, not implementation change -> separately labelled, not a headline cell."""
    if not isinstance(a, dict):
        return False
    c = a.get("commit") if isinstance(a.get("commit"), dict) else {}
    for v in (a.get("layout"), c.get("layout"), c.get("cmd")):
        if isinstance(v, str) and "chunk-leaf-v2" in v:
            return True
    return False


def _apply_attempts(recs: Records, rel: str, row: dict) -> None:
    """R13: a rerun is a new entry in `attempts[]` (release_tag/revision, per-stage status+evidence, timestamp).  The row's
    effective stage dicts / status / release_tag are those of the LATEST attempt; earlier attempts stay in the record.
    Top-level fields that disagree with the latest attempt are reported (the attempt wins)."""
    atts = row.get("attempts")
    row["_n_attempts"] = len(atts) if isinstance(atts, list) else 0
    if not isinstance(atts, list) or not atts:
        return
    good = [a for a in atts if isinstance(a, dict)]
    if len(good) != len(atts):
        recs.add(rel, "attempts", "non-object entries")
    # COORD 08:35Z: `chunk-leaf-v1` is the layout of record; `chunk-leaf-v2 (experimental)` attempts never feed a headline cell
    v2 = [a for a in good if attempt_is_v2_layout(a)]
    if v2:
        row["_v2_attempts"] = v2
        good = [a for a in good if not attempt_is_v2_layout(a)]
    repro = [a for a in good if a.get("role") == "reproduction"]
    if repro:
        row["_repro_attempts"] = repro  # second-pod reruns: reported as per-pod spread in the overhead table, not as the headline
        good = [a for a in good if a.get("role") != "reproduction"]
    # the top-level record is itself a candidate (lanes that record verdicts at the top level and keep attempts[] as notes, or
    # whose top level is more advanced than any snapshot taken of it earlier): rank it with the attempts
    _top_stages = {st: row[st] for st in STAGES if isinstance(row.get(st), dict) and str(row[st].get("status") or "").strip() not in ("", "-")}
    _all_atts = [a for a in (row.get("attempts") or []) if isinstance(a, dict)]  # incl. v2 / reproduction attempts filtered above
    if _top_stages and not any(all(a.get(st) == v for st, v in _top_stages.items()) for a in _all_atts) \
            and not any(a.get("role") == "reproduction" and a.get("at") == row.get("updated_at") for a in _all_atts):
        top_att = {"campaign": row.get("campaign", "r13"), "lane": row.get("lane"), "pod": row.get("pod"), "release_tag": row.get("release_tag"),
                   "revision": row.get("revision"), "at": row.get("updated_at") or "9999", "status": row.get("status"), "cmd": row.get("cmd"),
                   "env": row.get("env"), "notes": "top-level record", **_top_stages}
        if str(row.get("status") or "").upper().startswith("RUNNING"):
            top_att["status"] = "RUNNING"
        good = sorted(good + [top_att], key=lambda a: str(a.get("at") or ""))
    informative = [a for a in good if a.get("status") or any(isinstance(a.get(st), dict) for st in STAGES) or isinstance(a.get("stages"), dict)]
    if good and not informative:
        # free-text attempt notes (e.g. {"campaign": "r13", "result": "..."}) carry no stage verdicts: the top-level record stands
        recs.add(rel, "attempts", "entries carry no status / stage dicts (schema: per-stage status+evidence per attempt) — top-level stages used", "advisory")
        return
    good = informative
    if not good:
        return
    def _stage_ok(a, st):
        d = a.get(st)
        v = d.get("status") if isinstance(d, dict) else d
        head = str(v or "").strip().split("(")[0].split()[0] if str(v or "").strip() else ""
        return head in ("PASS", "DONE")

    def _rank(a):
        # how far the attempt got: Build, Match, Commit passes + overhead measured (a RUNNING attempt never outranks a finished one)
        return sum(1 for st in ("build", "match", "commit") if _stage_ok(a, st)) + (1 if _stage_ok(a, "bench") else 0)

    def _is_rel(a):
        return bool(RELEASE_TAG_RE.fullmatch(str(a.get("release_tag") or ""))) and not a.get("release_tag_pending")

    latest = good[-1]
    finished = [a for a in good if a.get("status") != "RUNNING"]
    if latest.get("status") == "RUNNING":
        # a rerun in flight never erases an earlier verdict: the row shows the last finished attempt and a 'rerun in progress' note
        row["_rerun_in_progress"] = f"{latest.get('pod', '?')} since {str(latest.get('at', ''))[11:16]}Z on {latest.get('release_tag') or latest.get('revision') or '?'}"
        if not finished:
            return  # original top-level record stands
        latest = finished[-1]
    # Headline attempt = the most advanced finished attempt (Build/Match/Commit/overhead reached), latest on ties.
    # COORD 12:00Z ruling (b): among equally advanced attempts the release-tree attempt is the headline; lane-tree attempts
    # made after it stay in attempts[] and feed §4 "variability" (never averaged).  An attempt that reached less than an earlier
    # one (e.g. a Match-only rerun after a full Commit) does not demote the row; it is listed as an attempt.
    if finished:
        best = max(_rank(a) for a in finished)
        cands = [a for a in finished if _rank(a) == best]
        _snap = lambda a: str(a.get("notes") or "").startswith("snapshot")  # import/dispatcher snapshots of a top-level record rank below real attempts
        rel_c = [a for a in cands if _is_rel(a)]
        real_rel = [a for a in rel_c if not _snap(a)]
        real_any = [a for a in cands if not _snap(a)]
        # a release-tree *snapshot* (import copy of a top-level verdict) does not outrank a real, fully evidenced attempt
        pool_c = real_rel or real_any or rel_c or cands
        real = [a for a in pool_c if not _snap(a)]
        pick = (real or pool_c)[-1]
        if pick is not latest:
            why = "release-tree attempt preferred (COORD 12:00Z ruling b)" if rel_c and pick in rel_c and _rank(latest) == best else \
                  f"latest finished attempt reached less ({_rank(latest)} stage passes) than attempt {pick.get('release_tag') or str(pick.get('revision') or '?')[:9]} ({best})"
            recs.add(rel, "attempts", f"headline attempt = {pick.get('release_tag') or 'lane-only ' + str(pick.get('revision') or '?')[:9]} ({pick.get('lane') or '?'}): {why}", "advisory")
            latest = pick
    if CAMPAIGN == "r14":
        # TABLE-RULES: R14 verdicts are new attempts carrying `rules_version`; an r14 attempt without it is refused (not a
        # headline candidate); when no r14 attempt exists the R13 headline stays but is shown as historical (◐ under r14-v1).
        for a in row.get("attempts") or []:
            if isinstance(a, dict) and str(a.get("campaign") or "") == "r14" and not a.get("rules_version"):
                recs.add(rel, "attempts", f"r14 attempt {a.get('at') or '?'} ({a.get('lane') or '?'}) has no rules_version — refused (TABLE-RULES); write it through the finalizer")
        def _finalized(a: dict) -> bool:
            return bool(a.get("finalizer"))  # TABLE-RULES / PLAN §2.9: only the finalizer writes displayed verdicts
        for a in row.get("attempts") or []:
            if isinstance(a, dict) and a.get("rules_version") and not _finalized(a):
                typed = any((isinstance(a.get(s), dict) and a[s].get("status") in ("PASS", "DONE")) or str((a.get("stages") or {}).get(s) if isinstance(a.get("stages"), dict) else "") == "PASS" for s in STAGES) or str(a.get("status") or "").endswith(("_PASS", "BENCHMARKED"))
                if typed:
                    recs.add(rel, "attempts", f"r14 attempt {a.get('at') or a.get('utc') or '?'} ({a.get('lane') or a.get('pod') or '?'}) carries typed PASS statuses without the finalizer — refused as a headline (TABLE-RULES: nobody types PASS; ship the evidence dir and run finalize)")
                else:
                    recs.add(rel, "attempts", f"r14 attempt {a.get('at') or a.get('utc') or '?'} ({a.get('lane') or a.get('pod') or '?'}): evidence recorded, no derived statuses yet — awaiting the finalizer (not a headline candidate)", "advisory")
        r14 = [a for a in (row.get("attempts") or []) if isinstance(a, dict) and a.get("rules_version") and a.get("status") not in (None, "RUNNING", "QUEUED") and _finalized(a)]
        if r14:
            latest = max(r14, key=lambda a: str(a.get("at") or ""))  # latest evidence, not latest finalizer run
            # COORD 12:00Z ruling (b) for finalizer attempts: a NEWER lane-tree attempt (executing_tree.lane_tree — a fix measurement on a
            # dirty / non-release tree, every cell ◐ by the tree rule) does not replace a release-tree headline unless it FAILED a stage
            # (a regression stays visible, brief §9); it stays in attempts[] and is named beneath the table (moe M-0678 'hold the cells').
            def _lane(a: dict) -> bool:
                et = a.get("executing_tree") if isinstance(a.get("executing_tree"), dict) else {}
                return et.get("lane_tree") is True
            rel_atts = [a for a in r14 if not _lane(a) and RELEASE_TAG_RE.fullmatch(str(a.get("release_tag") or ""))]
            if rel_atts and _lane(latest):
                lane_fails = [st for st in STAGES if isinstance(latest.get(st), dict) and latest[st].get("status") == "FAIL"]
                if not lane_fails:
                    keep = max(rel_atts, key=lambda a: str(a.get("at") or ""))
                    et = latest.get("executing_tree") or {}
                    row["_lane_tree_newer"] = {"at": latest.get("at"), "lane": latest.get("lane"), "tree": et.get("path"), "commit": et.get("commit"), "dirty": et.get("dirty"),
                                               "base_tag": et.get("base_tag"), "evidence_dir": latest.get("evidence_dir"),
                                               "stages": {st: (latest.get(st) or {}).get("status") for st in STAGES if isinstance(latest.get(st), dict)},
                                               "commit_reasons": list(((latest.get("commit") or {}).get("reasons") or [])[:2]),
                                               "bench": (latest.get("bench") or {}).get("overhead_pct") if isinstance(latest.get("bench"), dict) else None}
                    recs.add(rel, "attempts", f"newer lane-tree attempt {latest.get('at')} ({latest.get('lane')}, {et.get('path')}) is not the headline — release-tree attempt {keep.get('release_tag')} {keep.get('at')} stands (COORD 12:00Z ruling b; the lane-tree attempt failed no stage)", "advisory")
                    latest = keep
            # [R16 report brief 3(c) / COORD 12:00Z ruling (b) 'an attempt that reached less does not demote the row'] a NEWER release-tree
            # attempt that reached STRICTLY LESS than an earlier release-tree attempt (a late incomplete diagnostic: a Commit-only rerun whose
            # record lacks a field, a stage not reached) and FAILED no stage does not erase the earlier valid result on the supported release;
            # it stays in attempts[] and is named beneath the table.  A FAIL on the newer attempt (a real regression) IS the headline.
            if rel_atts and latest in rel_atts:
                earlier = [a for a in rel_atts if str(a.get("at") or "") < str(latest.get("at") or "")]
                fails_new = [st for st in STAGES if isinstance(latest.get(st), dict) and latest[st].get("status") == "FAIL"]
                if earlier and not fails_new:
                    best_prev = max(earlier, key=lambda a: (_rank(a), str(a.get("at") or "")))
                    if _rank(latest) < _rank(best_prev):
                        row["_incomplete_newer"] = {"at": latest.get("at"), "lane": latest.get("lane"), "release_tag": latest.get("release_tag"), "revision": latest.get("revision"),
                                                    "evidence_dir": latest.get("evidence_dir"), "kind": latest.get("kind"),
                                                    "stages": {st: (latest.get(st) or {}).get("status") for st in STAGES if isinstance(latest.get(st), dict)},
                                                    "reasons": [str(x)[:200] for st in ("commit", "match", "build") for x in (((latest.get(st) or {}).get("reasons") or [])[:1]) if isinstance(latest.get(st), dict)][:2],
                                                    "kept": {"release_tag": best_prev.get("release_tag"), "at": best_prev.get("at"), "rank": _rank(best_prev)}}
                        recs.add(rel, "attempts", f"newer release-tree attempt {latest.get('at')} ({latest.get('lane')}, {latest.get('release_tag')}) reached less ({_rank(latest)} stage passes) than "
                                                  f"{best_prev.get('release_tag')} {best_prev.get('at')} ({_rank(best_prev)}) and failed no stage — an incomplete diagnostic is not the headline; "
                                                  f"the earlier valid result stands and the newer attempt is named beneath the table (R16 brief 3(c); COORD 12:00Z ruling b)", "advisory")
                        latest = best_prev
            recs.violations[:] = [v for v in recs.violations if not (v.file == rel and v.field == "attempts" and "headline attempt =" in v.message)]
            recs.add(rel, "attempts", f"headline = finalizer attempt {latest.get('at')} ({latest.get('kind') or 'revalidation'}, {latest.get('lane')}); R13 attempts kept as history", "advisory")
            if latest.get("rules_version") != RULES_VERSION:
                recs.add(rel, "rules_version", f"attempt rules_version {latest.get('rules_version')!r} != current {RULES_VERSION!r} — shown as historical (◐)", "advisory")
    row["_standard"] = latest.get("rules_version") or "r13"
    row["_attempt"] = latest
    if latest.get("release_tag_pending"):
        row["release_tag_pending"] = True
    for k in ("release_tag", "revision", "code_commit", "campaign"):
        if latest.get(k) and not row.get(k):
            row[k] = latest[k]
        elif latest.get(k) and row.get(k) and row[k] != latest[k]:
            recs.add(rel, k, f"top-level {row[k]!r} != latest attempt {latest[k]!r} (attempt wins)", "advisory")
            row[k] = latest[k]
    for stg in STAGES:
        if isinstance(latest.get(stg), dict):
            row[stg] = dict(latest[stg])
        elif latest.get("stages") and isinstance(latest["stages"], dict) and isinstance(latest["stages"].get(stg), dict):
            row[stg] = dict(latest["stages"][stg])
        elif stg in row and isinstance(row.get(stg), dict) and row[stg].get("status") not in (None, "", "-"):
            # a stage the latest attempt did not reach must not inherit an earlier attempt's PASS
            row[stg] = {"status": "-", "note": "not reached in latest attempt"}
    # attempt-level cmd / artifact / committer / overhead apply to the stage dicts that lack them
    for stg in STAGES:
        d = row.get(stg)
        if not isinstance(d, dict):
            continue
        for k in ("cmd", "artifact", "committer", "revision"):
            if not d.get(k) and latest.get(k):
                d[k] = latest[k]
        if stg == "commit" and not d.get("committer"):
            m = re.search(r"--committer\s+(\S+)", str(d.get("cmd") or ""))
            if m:
                d["committer"] = m.group(1) + (" + hidden tap" if "--hidden-so" in str(d.get("cmd") or "") else "")
        if stg == "commit":
            # identity-level fail-closed layout (p0): expected==committed + identity_check_ok stands for required/bound classes
            if d.get("identity_check_ok") is True and d.get("expected_tensors") is not None and d.get("expected_tensors") == d.get("committed_tensors"):
                d.setdefault("required_classes", ["<identity manifest: %s tensors>" % d["expected_tensors"]])
                d.setdefault("bound_classes", d["required_classes"])
            _ps = str(d.get("status") or "")
            if _ps.startswith("PASS") and "(" in _ps:
                _m = re.search(r"bound\s+(\d+)/(\d+)\s+tensors", _ps)
                if _m and d.get("committed_tensors") is None:
                    d["committed_tensors"], d["expected_tensors"] = int(_m.group(1)), int(_m.group(2))
                if re.search(r"\b0 errors\b", _ps) and d.get("errors") is None:
                    d["errors"] = []
                if re.search(r"tokens equal", _ps) and d.get("tokens_equal") is None:
                    d["tokens_equal"] = True
                if "fail-closed" in _ps and d.get("fail_closed") is None:
                    d["fail_closed"] = True
                if d.get("bound_classes") is None and d.get("required_classes") is not None and d.get("committed_tensors") is not None \
                        and d["committed_tensors"] == d.get("expected_tensors"):
                    d["bound_classes"] = d["required_classes"]
                if d.get("backlog_max") is None and re.search(r"backlog\s*(?:max)?\s*[=:]?\s*0", _ps):
                    d["backlog_max"] = 0
            oar = d.get("openings_after_release") if isinstance(d.get("openings_after_release"), dict) else {}
            mem = latest.get("memory") if isinstance(latest.get("memory"), dict) else {}
            if d.get("backlog_max") is None:
                for k in ("backlog_max", "retained_device_after", "in_flight_max"):
                    if mem.get(k) is not None:
                        d["backlog_max"] = mem[k]; break
                    if oar.get(k) is not None:
                        d["backlog_max"] = oar[k]; break
            if d.get("errors") is None and d.get("errors_n") is not None:
                d["errors"] = [] if d["errors_n"] == 0 else d["errors_n"]
            if d.get("fail_closed") is None and (d.get("identity_check_ok") is True or "fail-closed" in str(latest.get("notes", "")).lower()):
                d["fail_closed"] = True
            if d.get("hidden_tensors") is None and isinstance(oar.get("families"), dict) and oar["families"].get("fa2_hidden_m1"):
                d["hidden_tensors"] = oar["families"]["fa2_hidden_m1"]
    # overhead recorded at attempt level -> bench dict
    ov = latest.get("overhead") if isinstance(latest.get("overhead"), dict) else None
    if ov and not isinstance(row.get("bench"), dict) or (ov and row.get("bench", {}).get("status") in (None, "-", "")):
        b = {"status": "DONE", "cmd": latest.get("cmd"), "artifact": latest.get("artifact")}
        for k_src, k_dst in (("control_s", "control_s"), ("treatment_s", "treatment_s"), ("control_wall_s", "control_s"), ("treatment_wall_s", "treatment_s"),
                             ("instrumented_wall_s", "treatment_s"), ("instrumented_s", "treatment_s"),
                             ("rel", "overhead_rel"), ("overhead_rel", "overhead_rel"), ("mean_rel", "overhead_rel"), ("abs_ms_per_step", "overhead_abs_ms_per_step"),
                             ("pairs", "rounds"), ("rounds", "rounds"), ("control_mean_s", "control_mean_s"), ("treatment_mean_s", "treatment_mean_s"), ("profiled", "profiled")):
            if ov.get(k_src) is not None and k_dst not in b:
                b[k_dst] = ov[k_src]
        if b.get("overhead_rel") is None and ov.get("wall_pct") is not None:
            b["overhead_rel"] = float(ov["wall_pct"]) / 100.0
        if b.get("overhead_abs_ms_per_step") is None and ov.get("decode_ms_instrumented") is not None and ov.get("decode_ms_control") is not None:
            b["overhead_abs_ms_per_step"] = round(float(ov["decode_ms_instrumented"]) - float(ov["decode_ms_control"]), 3)
        b.setdefault("profiled", False)
        b.setdefault("rounds", latest.get("pairs") or (len(b.get("control_s") or []) or None))
        row["bench"] = b
    # moe-style bench records: interleaved control/treatment pairs reported as per-pair relative deltas (delta_wall[]) with the
    # mean; map them onto the canonical fields (rounds = number of pairs; overhead_rel = mean delta; abs ms from abs_wall_ms)
    bb = row.get("bench") if isinstance(row.get("bench"), dict) else None
    if bb:
        for k_src, k_dst in (("control_wall_ms", "control_s"), ("instrumented_wall_ms", "treatment_s"), ("treatment_wall_ms", "treatment_s")):
            v = bb.get(k_src)
            if bb.get(k_dst) is None and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                bb[k_dst] = [x / 1000.0 for x in v]
            elif bb.get(k_dst) is None and isinstance(v, (int, float)):
                bb[k_dst.replace("_s", "_mean_s")] = v / 1000.0
        if bb.get("overhead_rel") is None and isinstance(bb.get("delta_wall_mean"), (int, float)):
            bb["overhead_rel"] = bb["delta_wall_mean"]
            bb.setdefault("profiled", "unprofiled" in str(bb.get("status") or "").lower() or None)
            bb.setdefault("cmd", (row.get("commit") or {}).get("cmd") if isinstance(row.get("commit"), dict) else None)
            bb.setdefault("artifact", (row.get("commit") or {}).get("artifact") if isinstance(row.get("commit"), dict) else None)
        if bb.get("rounds") is None and isinstance(bb.get("control_s"), list) and bb["control_s"]:
            bb["rounds"] = len(bb["control_s"])
    if bb and bb.get("overhead_rel") is None:
        dw = bb.get("delta_wall")
        if isinstance(dw, list) and dw and all(isinstance(x, (int, float)) for x in dw):
            bb["overhead_rel"] = bb.get("delta_wall_mean") if isinstance(bb.get("delta_wall_mean"), (int, float)) else sum(dw) / len(dw)
            bb.setdefault("rounds", len(dw))
            bb.setdefault("profiled", "unprofiled" in str(bb.get("status") or "").lower() or None)
            bb.setdefault("cmd", (row.get("commit") or {}).get("cmd") if isinstance(row.get("commit"), dict) else None)
            bb.setdefault("artifact", (row.get("commit") or {}).get("artifact") if isinstance(row.get("commit"), dict) else None)
            bb["_interleaved_pairs"] = True
        if bb.get("overhead_abs_ms_per_step") is None and isinstance(bb.get("abs_wall_ms"), list) and bb["abs_wall_ms"]:
            pass  # per-run wall deltas, not per-decode-step: left blank in the table
    if latest.get("status"):
        if row.get("status") and row["status"] != latest["status"]:
            recs.add(rel, "status", f"top-level {row['status']!r} != latest attempt {latest['status']!r} (attempt wins)")
        row["status"] = latest["status"]
    if latest.get("code_changes_required") is not None:
        row["code_changes_required"] = latest["code_changes_required"]
    if latest.get("infra_fixes") is not None:
        row["infra_fixes"] = latest["infra_fixes"]
    if latest.get("failure_id") and isinstance(row.get("commit"), dict) and not row["commit"].get("failure_id"):
        pass


def release_label(row: dict) -> str:
    """'r13-relN' if the latest attempt ran a release tag; 'lane-only' if it ran a revision without a release tag; '' otherwise."""
    tag = str(row.get("release_tag") or "")
    if RELEASE_TAG_RE.match(tag):
        return tag + (" (candidate, not yet announced)" if row.get("release_tag_pending") else "")
    if row.get("campaign") == "r13" or row.get("_n_attempts", 0) > 0:
        return "lane-only"
    return ""


# ----------------------------------------------------------------------------------------------------------------
# row helpers


def parse_row_id(row_id: str) -> dict[str, Any] | None:
    """`<model>__<dtype>__<hw>__tp<k>__b<B>__i<Imax>__o<Omax>__<workload>__<sampling>__<exec>` -> dict or None."""
    parts = row_id.split("__")
    if len(parts) != 10:
        return None
    model, dtype, hw, tp, b, i, o, workload, sampling, exe = parts
    try:
        if not (tp.startswith("tp") and b.startswith("b") and i.startswith("i") and o.startswith("o")):
            return None
        return {"model": model, "dtype": dtype, "hw": hw, "tp": int(tp[2:]), "batch": int(b[1:]),
                "max_input": int(i[1:]), "max_output": int(o[1:]), "workload": workload,
                "sampling": sampling, "execution": exe}
    except ValueError:
        return None


def stage_status(row: dict, stage: str) -> str:
    """Effective stage status: the stage dict's `status` if present, else derived from the row `status`."""
    st = row.get(stage)
    if isinstance(st, dict) and isinstance(st.get("status"), str) and st["status"]:
        s = st["status"]
        if s not in STAGE_STATUSES[stage]:
            # tolerate annotated statuses such as "PASS (fail-closed X-03: ...)" — the enum violation is still reported
            head = s.strip().split("(")[0].split()[0] if s.strip() else s
            if head.upper() == "INTERMEDIATE" and "PASS" in s.upper():
                return "PASS"  # e.g. tp's 'INTERMEDIATE PASS (collective-level ...)': rendered ◐ via the intermediate-criterion check
            if head in ("PASS", "FAIL", "DONE", "-"):
                s = head
        if stage == "bench" and s == "PASS":
            s = "DONE"  # tolerated alias (flagged as a violation)
        if stage == "bench" and s == "DONE" and stage_status(row, "commit") != "PASS":
            return "-"  # BRIEF §13.D: overhead only for Commit-green rows
        return s
    return _derived_stage_status(row.get("status"), stage)


def _derived_stage_status(status: Any, stage: str) -> str:
    order = {"build": 0, "match": 1, "commit": 2, "bench": 3}
    k = order[stage]
    if status == "BENCHMARKED":
        return "DONE" if k == 3 else "PASS"
    passed = {"BUILD_PASS": 0, "MATCH_PASS": 1, "COMMIT_PASS": 2}
    failed = {"BUILD_FAIL": 0, "MATCH_FAIL": 1, "COMMIT_FAIL": 2}
    if status in passed:
        return "PASS" if k <= passed[status] else "-"
    if status in failed:
        n = failed[status]
        return "PASS" if k < n else ("FAIL" if k == n else "-")
    return "-"


def model_of(row: dict) -> str:
    cfg = row.get("config") or {}
    m = cfg.get("model")
    if isinstance(m, str) and m:
        return m
    p = parse_row_id(row.get("row_id", ""))
    return p["model"] if p else "?"


def cfg_value(row: dict, key: str, default: Any = None) -> Any:
    """Config field with row-id fallback (so a thin record still renders)."""
    cfg = row.get("config") or {}
    if key in cfg and cfg[key] not in (None, ""):
        return cfg[key]
    p = parse_row_id(row.get("row_id", ""))
    if p is None:
        return default
    alias = {"hardware": "hw"}
    return p.get(alias.get(key, key), default)


def hw_key(row: dict) -> str:
    """Hardware id (l40s/a100/h100/rtx4090) from config.hardware or the row id."""
    p = parse_row_id(row.get("row_id", ""))
    if p:
        return p["hw"]
    hw = str((row.get("config") or {}).get("hardware", "")).lower().replace(" ", "")
    return hw


def sort_key(row: dict) -> tuple:
    m = model_of(row)
    hw = hw_key(row)
    return (
        MODEL_ORDER.get(m, len(MODEL_ORDER)), m,
        HW_ORDER.index(hw) if hw in HW_ORDER else len(HW_ORDER), hw,
        _int(cfg_value(row, "tp", 0)), _int(cfg_value(row, "batch", 0)),
        _int(cfg_value(row, "max_input", 0)), _int(cfg_value(row, "max_output", 0)),
        str(cfg_value(row, "workload", "")), str(cfg_value(row, "sampling", "")), str(cfg_value(row, "execution", "")),
    )


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def sorted_rows(recs: Records) -> list[dict]:
    return sorted(recs.rows.values(), key=sort_key)


def _board_rows(root: str) -> dict[str, dict]:
    """out/gen/sweep/BOARD_ROWS.json {row_id: {ordered_by, since, note, lifted?}} — report-lane sidecar (never a record edit)."""
    p = os.path.join(root, "BOARD_ROWS.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            J = json.load(fh)
        return {k: v for k, v in J.items() if isinstance(v, dict) and not v.get("lifted")} if isinstance(J, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def board_rows(recs: Records) -> list[dict]:
    """The board-ordered non-panel rows, in panel order."""
    return sorted([o for o in recs.demos.values() if isinstance(o.get("_board_order"), dict)], key=sort_key)


# ----------------------------------------------------------------------------------------------------------------
# validation


def _req(recs: Records, rel: str, obj: dict, key: str, types: tuple | type, path: str = "") -> bool:
    fld = f"{path}{key}"
    if key not in obj:
        recs.add(rel, fld, "missing required field")
        return False
    if not isinstance(obj[key], types):
        recs.add(rel, fld, f"wrong type {type(obj[key]).__name__}")
        return False
    return True


# Independent-review downgrades (lane rev, findings.md): a Match ✅ whose compare covered 17 of 128 decode steps is ◐ until the
# request wrapper is re-derived with --T 127 and re-compared (F-rev-02).  Cleared automatically when the record's match cmd/verdict
# shows the 128-step compare.
REVIEW_MATCH_DOWNGRADES = {
    "llama32-1b__bf16__a100__tp1__b1__i1024__o128__mixed__greedy__bi-eager": "F-rev-02: compared against a T=16 request wrapper — 17/128 steps; decode steps 17..127 never compared (re-derive --T 127 on the existing fold)",
    "smollm2-135m__bf16__a100__tp1__b1__i1024__o128__mixed__greedy__bi-eager": "F-rev-02: compared against a T=16 request wrapper — 17/128 steps; decode steps 17..127 never compared (re-derive --T 127 on the existing fold)",
}


def _validate_row(recs: Records, rel: str, rid: str, row: dict) -> None:
    if row.get("schema") != ROW_SCHEMA:
        recs.add(rel, "schema", f"expected {ROW_SCHEMA!r}, got {row.get('schema')!r}")
    status = row.get("status")
    if status not in ROW_STATUSES:
        recs.add(rel, "status", f"{status!r} not in {ROW_STATUSES}")
        head = str(status or "").strip().upper()
        if head.startswith(("COMPLETE", "PASS", "BENCHMARKED")) and stage_status(row, "commit") == "PASS":
            # prose statuses such as 'COMPLETE' / 'PASS (Build ✅ Match ✅ Commit ✅ overhead measured)' → enum for the checks below
            status = "BENCHMARKED" if isinstance(row.get("bench"), dict) else "COMMIT_PASS"
            row["status"] = status
            recs.add(rel, "status", f"normalised to {status!r} from the prose status (enum violation reported above)", "advisory")
    for k in ("lane", "updated_at"):
        if k not in row:
            recs.add(rel, k, "absent", "advisory")
    parsed = parse_row_id(rid)
    if parsed is None:
        recs.add(rel, "row_id", "does not follow <model>__<dtype>__<hw>__tp<k>__b<B>__i<Imax>__o<Omax>__<workload>__<sampling>__<exec>")
    cfg = row.get("config")
    if not isinstance(cfg, dict):
        recs.add(rel, "config", "missing or not an object")
        cfg = {}
    else:
        for k, t in (("model", str), ("dtype", str), ("hardware", str), ("tp", int), ("batch", int),
                     ("max_input", int), ("max_output", int), ("workload", str), ("sampling", str),
                     ("execution", str), ("verity_commit", str)):
            _req(recs, rel, cfg, k, t, "config.")
        if status != "NOT_RUN":
            for k in ("hf_repo", "revision", "vllm_revision", "moe", "workload_manifest", "workload_seed"):
                if k not in cfg:
                    recs.add(rel, f"config.{k}", "absent (SCHEMA §3 expects it)", "advisory")
        if parsed:
            for ck, pk in (("model", "model"), ("dtype", "dtype"), ("tp", "tp"), ("batch", "batch"),
                           ("max_input", "max_input"), ("max_output", "max_output"), ("workload", "workload"),
                           ("sampling", "sampling"), ("execution", "execution")):
                if ck in cfg and cfg[ck] != parsed[pk]:
                    recs.add(rel, f"config.{ck}", f"{cfg[ck]!r} disagrees with row id ({parsed[pk]!r})")
            hw = str(cfg.get("hardware", "")).lower().replace(" ", "")
            if "hardware" in cfg and hw != parsed["hw"]:
                recs.add(rel, "config.hardware", f"{cfg['hardware']!r} disagrees with row id ({parsed['hw']!r})")
            if parsed["model"] not in MODEL_ORDER:
                recs.add(rel, "row_id", f"model {parsed['model']!r} not in frozen panel (SCHEMA §2)")
            if parsed["dtype"] not in DTYPES:
                recs.add(rel, "row_id", f"dtype {parsed['dtype']!r} not in {DTYPES}")
            if parsed["hw"] not in HW_ORDER:
                recs.add(rel, "row_id", f"hw {parsed['hw']!r} not in {HW_ORDER}")
            if parsed["execution"] not in EXEC_DISPLAY:
                recs.add(rel, "row_id", f"exec {parsed['execution']!r} not in {tuple(EXEC_DISPLAY)}")
            if parsed["workload"] not in WORKLOAD_DISPLAY:
                recs.add(rel, "row_id", f"workload {parsed['workload']!r} not in {tuple(WORKLOAD_DISPLAY)}")
            if parsed["sampling"] not in SAMPLING_DISPLAY:
                recs.add(rel, "row_id", f"sampling {parsed['sampling']!r} not in {tuple(SAMPLING_DISPLAY)}")
        if "moe" in cfg and isinstance(cfg["moe"], bool):
            m = cfg.get("model")
            if m in MODEL_MOE and MODEL_MOE[m] != cfg["moe"]:
                recs.add(rel, "config.moe", f"{cfg['moe']} disagrees with panel ({MODEL_MOE[m]}) for {m}")

    if status == "NOT_RUN" and not row.get("not_run_reason"):
        recs.add(rel, "not_run_reason", "NOT_RUN row must state the reason")

    # stage dicts
    missing: list[tuple[str, str]] = []
    for stage in STAGES:
        st = row.get(stage)
        if st is None:
            continue
        if not isinstance(st, dict):
            recs.add(rel, stage, "not an object")
            continue
        s = st.get("status")
        if s is None:
            recs.add(rel, f"{stage}.status", "missing")
        elif s not in STAGE_STATUSES[stage]:
            recs.add(rel, f"{stage}.status", f"{s!r} not in {STAGE_STATUSES[stage]}")
        eff = stage_status(row, stage)
        if eff in ("PASS", "DONE"):
            for f in EVIDENCE_FIELDS[stage]:
                v = st.get(f)
                if v is None or v == "" or (isinstance(v, list) and len(v) == 0):
                    if stage == "bench" and f in ("control_s", "treatment_s") and isinstance(st.get("rounds"), int) and st["rounds"] >= 3 \
                            and ((isinstance(st.get("control_mean_s"), (int, float)) and isinstance(st.get("treatment_mean_s"), (int, float)))
                                 or st.get("_interleaved_pairs")):
                        # matched control demonstrated by the means over >=3 rounds (or per-pair deltas of interleaved runs);
                        # the per-round samples are missing from the record — advisory, not a rejection of the measurement
                        recs.add(rel, f"bench.{f}", "per-round samples not recorded (means / per-pair deltas over >=3 rounds present)", "advisory")
                        continue
                    missing.append((stage, f))
            # stricter glossary checks (reviewer R-01): content of the evidence, not just presence
            if stage == "build":
                cmd = str(st.get("cmd") or "")
                if cmd and "derive_step" not in cmd:
                    missing.append((stage, "cmd(derive_step)"))
                # reviewer R-06: after X-01 every Build PASS must show the target it was built for (any GPU)
                tp = st.get("target_profile") if isinstance(st.get("target_profile"), dict) else {}
                if "--target" not in cmd and not st.get("target") and not tp.get("num_sms"):
                    missing.append((stage, "target(TargetProfile num_sms / --target)"))
            if stage == "match":
                if st.get("control_vs_instrumented_tokens_equal") is False:
                    recs.add(rel, "match.control_vs_instrumented_tokens_equal", "False but match.status is PASS")
                    missing.append((stage, "control_vs_instrumented_tokens_equal"))
                v = str(st.get("verdict") or "").lower().strip()
                v_ok = v.startswith(("pass", "canonical_equal", "equal", "match"))
                v_neg = re.sub(r"(mismatch\w*|fail\w*|false|different)\s*[:=]?\s*(0\b|\{\}|\[\]|none)", "", v)
                if v and not v_ok and ("false" in v_neg or "fail" in v_neg or "mismatch" in v_neg):
                    recs.add(rel, "match.verdict", f"{st.get('verdict')!r} does not read as canonical_equal/PASS but match.status is PASS")
                    missing.append((stage, "verdict"))
                if rid in REVIEW_MATCH_DOWNGRADES and not re.search(r"--T\s+127\b|128/128|steps_compared.*128", " ".join(str(st.get(k) or "") for k in ("cmd", "verdict", "note", "steps_compared"))):
                    # independent reviewer F-rev-02: cb_a.compare ran against a T=16 request wrapper (17/128 steps compared)
                    recs.add(rel, "match.verdict", REVIEW_MATCH_DOWNGRADES[rid])
                    missing.append((stage, "17/128 steps compared (F-rev-02)"))
                if str(st.get("status") or "").strip().upper().startswith("INTERMEDIATE"):
                    recs.add(rel, "match.status", "INTERMEDIATE PASS (collective-level TP-aware Match: structure + cross-rank provenance + tokens) — not the per-rank instance-sequence Match (COORD 12:25Z: TP2 rows are partial, R14)", "advisory")
                    missing.append((stage, "Match (collective-level only, R14)"))
                crit = " ".join(str(st.get(k) or "") for k in ("criterion", "match_criterion", "comparator", "verdict", "note")).lower()
                if re.search(r"dag[- ]?multiset|multiset criterion|order-insensitive|live-cone", crit) and "diagnostic" not in crit:
                    # dense 09:35Z: emission-order-insensitive DAG multiset for chunked-prefill requests = criterion VARIANT; ruling requested
                    recs.add(rel, "match.criterion", "chunked-prefill live-cone / DAG-multiset criterion — admissible only as a labelled criterion, not a headline Match this campaign (COORD 12:00Z ruling a)", "advisory")
                    missing.append((stage, "Match (chunked criterion, under review)"))
            if stage == "commit":
                oc, ok = st.get("openings_checked"), st.get("openings_ok")
                if isinstance(oc, int) and isinstance(ok, int) and ok < oc:
                    recs.add(rel, "commit.openings_ok", f"{ok}/{oc} openings verified but commit.status is PASS")
                    missing.append((stage, "openings_ok"))
                rc, bc = st.get("required_classes"), st.get("bound_classes")
                if isinstance(rc, int) and isinstance(bc, int) and bc < rc:
                    recs.add(rel, "commit.bound_classes", f"{bc}/{rc} classes bound but commit.status is PASS")
                    missing.append((stage, "bound_classes"))
                if st.get("tokens_equal") is False:
                    recs.add(rel, "commit.tokens_equal", "False but commit.status is PASS")
                    missing.append((stage, "tokens_equal"))
                bl = st.get("backlog_max")
                _att = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else row
                _txt = " ".join(str(x) for x in (st.get("cmd"), st.get("retain"), st.get("retention"), st.get("staging_ring"), st.get("window"), st.get("committer"), _att.get("cmd"), json.dumps(_att.get("env") or {}), _att.get("notes")))
                _bounded = bool(re.search(r"VERITY_WINDOW_(MB|SLOTS)|--window-(mb|slots)|--retain\s+host|VERITY_RETAIN=host|retain[=: ]+host|staging_ring|windowed|\d+\s*[x×]\s*\d+\s*MiB", _txt))
                _slots_m = re.search(r"VERITY_WINDOW_SLOTS=(\d+)|--window-slots\s+(\d+)|(\d+)\s*[x×]\s*\d+\s*MiB", _txt)
                _slots = int(next(g for g in _slots_m.groups() if g)) if _slots_m else 8
                if isinstance(bl, (int, float)) and bl > 1 and _bounded and bl <= _slots:
                    # P0 windowed staging / host retention bounds the in-flight set by the configured window (default 8 slots):
                    # backlog_max ≤ slots is the design's bounded in-flight set, not a sustained (growing) backlog — accepted as
                    # an implementation parameter of the commitment path (COORD 12:00Z ruling b on retention); flagged for review
                    recs.add(rel, "commit.backlog_max", f"{bl} ≤ {_slots} window slots (windowed staging / host retention): bounded in-flight set, accepted as 'no sustained backlog'", "advisory")
                elif isinstance(bl, (int, float)) and bl > 1:
                    # Daniel's criterion is "no *sustained* backlog": a transient of one in-flight block (backlog_max 1) is
                    # accepted (COORD 08:45Z, fa2 -> rel/report); anything above is a sustained backlog and fails the PASS claim
                    recs.add(rel, "commit.backlog_max", f"{bl} > 1 (sustained backlog) but commit.status is PASS")
                    missing.append((stage, "backlog_max"))
                elif isinstance(bl, (int, float)) and bl == 1:
                    recs.add(rel, "commit.backlog_max", "1 (transient, single in-flight block) — accepted under 'no sustained backlog'", "advisory")
                if isinstance(oc, int) and oc < 64:
                    missing.append((stage, "openings_checked>=64"))
                if not st.get("committer"):
                    missing.append((stage, "committer"))
                ccmd = str(st.get("cmd") or "")
                if ccmd and "--hidden-so" not in ccmd and st.get("hidden_bound") is not True:
                    missing.append((stage, "hidden_so(FA tap bound)"))
                # reviewer R-14: X-03 counts attempted acquisitions; a producer that never fires (FA3 on Hopper) still passes.
                ht = st.get("hidden_tensors")
                if hw_key(row) == "h100" and not (isinstance(ht, int) and ht > 0) and st.get("hidden_bound") is not True:
                    missing.append((stage, "hidden_tensors>0 (FA3 host: FA2 tap never fires, R-14)"))
                elif not (isinstance(ht, int) and ht > 0) and st.get("hidden_bound") is not True:
                    recs.add(rel, "commit.hidden_tensors", "absent: record how many hidden (attention) tensors were bound (R-14)", "advisory")
                # Daniel 04:04Z: the frozen commit_delta was fail-open (PASS with arena refusals recorded).  A PASS counts
                # only if produced under X-03 (fail-closed harness) with no errors and complete coverage/openings.
                errs = st.get("errors")
                if errs is None and isinstance(st.get("extra"), dict):
                    errs = st["extra"].get("errors")
                if isinstance(errs, list) and all(e in (0, "0", [], None) for e in errs):
                    errs = []  # per-run error counts all zero (ampere records `[0, 0, 0]`) = no errors
                if errs not in (None, [], {}, 0, "", "0"):
                    recs.add(rel, "commit.errors", f"collector errors recorded ({str(errs)[:80]}) but commit.status is PASS")
                    missing.append((stage, "errors==0"))
                if not _fail_closed(recs, row, st):
                    missing.append((stage, "fail_closed(X-03)"))
                    recs.invalidated.append(rid)
            if stage == "bench":
                rd = st.get("rounds")
                if isinstance(rd, int) and rd < 3:
                    missing.append((stage, "rounds>=3"))
                if st.get("profiled") is None:
                    missing.append((stage, "profiled(false)"))
                for f in ("control_s", "treatment_s"):
                    v = st.get(f)
                    if v is not None and not (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)):
                        recs.add(rel, f"bench.{f}", "must be an array of seconds")
                if st.get("profiled") is True:
                    recs.add(rel, "bench.profiled", "headline timing must not be profiled (BRIEF §13.D)")
                if st.get("overhead_rel") is None and not (st.get("control_s") and st.get("treatment_s")):
                    missing.append((stage, "overhead_rel"))
                if not (cfg or {}).get("verity_commit"):
                    missing.append((stage, "config.verity_commit"))
        if eff == "FAIL":
            fid = st.get("failure_id")
            if not fid:
                recs.add(rel, f"{stage}.failure_id", "stage FAIL without a failure record id")
            elif fid not in recs.failures:
                recs.add(rel, f"{stage}.failure_id", f"{fid!r} has no failures/{fid}.json")
    # a later stage PASS without the earlier stage demonstrated is partial evidence (BRIEF status convention ◐)
    if stage_status(row, "match") == "PASS" and stage_status(row, "build") != "PASS":
        missing.append(("match", "build.status=PASS (Program not demonstrated built)"))
    if stage_status(row, "commit") == "PASS" and stage_status(row, "match") != "PASS":
        missing.append(("commit", "match.status=PASS (runtime correspondence not demonstrated)"))
    # de-dup missing evidence
    seen: set[tuple[str, str]] = set()
    ded = [m for m in missing if not (m in seen or seen.add(m))]
    if isinstance(row.get("_attempt"), dict) and row["_attempt"].get("finalizer"):
        # finalizer verdict: the R13 field checklist is superseded by TABLE-RULES as applied by finalize.py (PARTIAL carries the
        # missing list in `reasons`); keep only cross-stage dependencies
        ded = [m for m in ded if m[1].startswith(("build.status", "match.status"))]
    if ded:
        recs.missing_evidence[rid] = ded

    # row status vs stage statuses (finalizer rows carry derived per-stage statuses incl. PARTIAL: the row status is a summary,
    # the stage dicts are authoritative — skip the enum-derived consistency check for them)
    if status in ROW_STATUSES and status not in ("QUEUED", "NOT_RUN") and not str(status).endswith("_PARTIAL") and not (
            isinstance(row.get("_attempt"), dict) and row["_attempt"].get("finalizer")):
        exp = {s: _derived_stage_status(status, s) for s in STAGES}
        for s in STAGES:
            st = row.get(s)
            if isinstance(st, dict) and isinstance(st.get("status"), str) and st["status"] in STAGE_STATUSES[s]:
                if st["status"] != exp[s] and not (exp[s] == "-" and st["status"] == "-"):
                    recs.add(rel, f"{s}.status", f"{st['status']!r} inconsistent with row status {status} (expected {exp[s]!r})")
    bst = row.get("bench") if isinstance(row.get("bench"), dict) else {}
    if str(bst.get("status") or "").split("(")[0].strip() in ("DONE", "PASS"):
        cst = row.get("commit") if isinstance(row.get("commit"), dict) else {}
        if stage_status(row, "commit") != "PASS" or status not in ("COMMIT_PASS", "BENCHMARKED"):
            recs.add(rel, "bench.status", f"bench recorded but commit.status={cst.get('status')!r} / row status {status}: "
                                          "overhead is only reported for Commit-green rows (BRIEF §13.D); cell rendered —")
    if status == "BENCHMARKED" and not isinstance(row.get("bench"), dict):
        recs.add(rel, "bench", "BENCHMARKED row has no bench object")
    if status in ("COMMIT_PASS", "BENCHMARKED"):
        st = row.get("bench")
        if isinstance(st, dict) and st.get("status") == "DONE" and status == "COMMIT_PASS":
            recs.add(rel, "status", "bench DONE but status is COMMIT_PASS (expected BENCHMARKED)")
    vc = str((cfg or {}).get("verity_commit") or "")
    for stg in STAGES:
        d = row.get(stg)
        if isinstance(d, dict) and d.get("code_commit") and vc and not str(d["code_commit"]).startswith(vc[:9]):
            recs.add(rel, "config.verity_commit", f"{vc[:9]} but {stg}.code_commit={str(d['code_commit'])[:9]} — set verity_commit to the code that ran")
            break
    # R-18 fix-id collision: p0/p0b reused X-05/X-06 (dense's B-SAMPLING records) for the footprint fixes; p0 renamed them P0-02/P0-03; the report
    # remaps stale p0-lane citations so the fix ledger stays one-id-one-mechanism (flagged as advisory).
    lane_txt = f"{row.get('lane') or ''} {(row.get('attempts') or [{}])[-1].get('lane') if isinstance(row.get('attempts'), list) and row.get('attempts') else '' or ''}".lower()
    if isinstance(row.get("code_changes_required"), list) and any(k in lane_txt for k in ("p0", "da", "bc")) and not m_is_dense(lane_txt):
        remap = {"X-05": "P0-02", "X-06": "P0-03"}  # p0's own ids since 07:40Z (P0-02 openings/identity, P0-03 FA2 M1 compaction)
        if any(x in remap for x in row["code_changes_required"] if isinstance(x, str)):
            row["code_changes_required"] = [remap.get(x, x) if isinstance(x, str) else x for x in row["code_changes_required"]]
            recs.add(rel, "code_changes_required", "cites X-05/X-06, which are dense's B-SAMPLING fix ids; read as P0-02/P0-03 (p0 footprint fixes, renamed 07:40Z) — re-cite (R-18)", "advisory")
    ccr = row.get("code_changes_required")
    if isinstance(ccr, list) and any(not isinstance(x, str) for x in ccr):
        # schema: list of fix ids. Some lanes wrote inline change dicts; keep the row loadable, flag it, and coerce to ids.
        recs.add(rel, "code_changes_required", "must be a list of fix ids (strings); inline change dicts found — record a fix under out/gen/sweep/fixes/ and cite its id")
        ccr = [x if isinstance(x, str) else str((x.get("id") or x.get("fix") or f"unrecorded@{str(x.get('commit') or '')[:9]}") if isinstance(x, dict) else x) for x in ccr]
        row["code_changes_required"] = ccr
    if isinstance(ccr, list):
        bad = [x for x in ccr if x in ENV_FIXES or (x in recs.fixes and is_infra_fix(recs.fixes[x]))]
        if bad:
            recs.add(rel, "code_changes_required", f"{bad} are infrastructure/harness fixes — record them under `infra_fixes` (R-19)", "advisory")
    inf = row.get("infra_fixes")
    if inf is not None and not isinstance(inf, list):
        recs.add(rel, "infra_fixes", "must be a list of fix ids")
    if row.get("campaign") == "r13" and not row.get("_n_attempts"):
        recs.add(rel, "attempts", "campaign r13 record without attempts[] (a rerun is a new attempt; SCHEMA + PLAN §2)", "advisory")
    if row.get("_n_attempts") and not RELEASE_TAG_RE.match(release_label(row).split(" ")[0]):
        recs.add(rel, "release_tag", "latest attempt names no announced release tag (r13-relN / r14-base / r14-rcN) — rendered as lane-only revision", "advisory")
    if ccr is None:
        recs.add(rel, "code_changes_required", "missing (use [] for zero code changes)")
    elif not isinstance(ccr, list):
        recs.add(rel, "code_changes_required", "must be a list of fix ids")
    else:
        for x in ccr:
            if x not in recs.fixes:
                recs.add(rel, "code_changes_required", f"{x!r} has no fixes/{x}.json")
    wa = row.get("workload_actual")
    if isinstance(wa, dict) and status in ("MATCH_PASS", "COMMIT_PASS", "BENCHMARKED"):
        b = _int((cfg or {}).get("batch", 0))
        for f in ("input_lengths", "output_caps", "generated_lengths"):
            v = wa.get(f)
            if not isinstance(v, list) or not v:
                recs.add(rel, f"workload_actual.{f}", "empty although the row ran to Match or beyond")
            elif b and len(v) < b:
                recs.add(rel, f"workload_actual.{f}", f"{len(v)} entries < batch {b} (manifest smaller than the batch slots)", "advisory")
    notes = row.get("notes")
    if isinstance(notes, str) and notes.count("\n") > 1:
        recs.add(rel, "notes", "one line max")


_ANCESTOR_CACHE: dict[tuple[str, str], bool] = {}


def _is_ancestor(anc: str, desc: str) -> bool:
    key = (anc, desc)
    if key not in _ANCESTOR_CACHE:
        try:
            import subprocess
            r = subprocess.run(["git", "merge-base", "--is-ancestor", anc, desc], capture_output=True, timeout=10)
            _ANCESTOR_CACHE[key] = r.returncode == 0
        except Exception:  # noqa: BLE001
            _ANCESTOR_CACHE[key] = False
    return _ANCESTOR_CACHE[key]


def _fail_closed(recs: Records, row: dict, commit: dict) -> bool:
    """True if the Commit PASS was produced by the fail-closed harness (fix X-03)."""
    if commit.get("fail_closed") is True:
        return True
    ccr = row.get("code_changes_required")
    if isinstance(ccr, list) and "X-03" in ccr:
        return True
    x03 = recs.fixes.get("X-03")
    if x03 and x03.get("commit"):
        shas = [str((row.get("config") or {}).get("verity_commit") or "")]
        for st in ("commit", "match", "build"):
            d = row.get(st)
            if isinstance(d, dict) and d.get("code_commit"):
                shas.append(str(d["code_commit"]).split()[0])
        for sha in shas:
            if sha and _is_ancestor(str(x03["commit"]).split()[0], sha):
                return True
    return False


def _validate_failure(recs: Records, rel: str, fid: str, f: dict) -> None:
    if f.get("schema") != FAILURE_SCHEMA:
        recs.add(rel, "schema", f"expected {FAILURE_SCHEMA!r}, got {f.get('schema')!r}")
    rid = f.get("row_id")
    if not isinstance(rid, str) or not rid:
        recs.add(rel, "row_id", "missing")
    elif rid not in recs.rows:
        recs.add(rel, "row_id", f"{rid!r} has no rows/{rid}.json")
    if f.get("stage") not in FAILURE_STAGES:
        recs.add(rel, "stage", f"{f.get('stage')!r} not in {FAILURE_STAGES}")
    if f.get("class") not in FAILURE_CLASSES:
        recs.add(rel, "class", f"{f.get('class')!r} not one of the BRIEF §15 classes")
    for k in ("cmd", "commit", "first_error", "artifact"):
        v = f.get(k)
        if not isinstance(v, str) or not v.strip():
            recs.add(rel, k, "missing or empty")
    fe = f.get("first_error")
    if isinstance(fe, str) and fe.count("\n") > 10:
        recs.add(rel, "first_error", "> 10 lines (SCHEMA §4)")
    sa = f.get("same_as")
    if sa is not None:
        if not isinstance(sa, list):
            recs.add(rel, "same_as", "must be a list of failure ids")
        else:
            for x in sa:
                if x not in recs.failures:
                    recs.add(rel, "same_as", f"{x!r} has no failures/{x}.json")
    if "created_at" not in f:
        recs.add(rel, "created_at", "absent", "advisory")


def _validate_fix(recs: Records, rel: str, xid: str, x: dict) -> None:
    if x.get("schema") != FIX_SCHEMA:
        recs.add(rel, "schema", f"expected {FIX_SCHEMA!r}, got {x.get('schema')!r}")
    for k in ("mechanism", "why_it_failed", "generic_change", "commit"):
        v = x.get(k)
        if not isinstance(v, str) or not v.strip():
            recs.add(rel, k, "missing or empty")
    for k in ("rows_expected", "rows_rerun", "rows_newly_green", "regressions"):
        v = x.get(k)
        if v is None:
            recs.add(rel, k, "missing (use [])")
        elif not isinstance(v, list):
            recs.add(rel, k, "must be a list of row ids")
        else:
            for r in v:
                if r not in recs.rows:
                    recs.add(rel, k, f"{r!r} has no rows/{r}.json")
    for k in ("owner", "created_at"):
        if k not in x:
            recs.add(rel, k, "absent", "advisory")


# ----------------------------------------------------------------------------------------------------------------
# failure clustering — coordinator-named clusters (DECISION-0355.md: stage + first_error substring), same_as links kept

# (id, stage, substrings-any, mechanism).  Order matters (first match wins).  C-FA2-HDIM is proposed by the report owner.
NAMED_CLUSTERS: list[tuple[str, str, tuple[str, ...], str]] = [
    ("B-QKVBIAS", "build", ("qkv_proj",),
     "Bet A has no lowering rule for the qkv bias broadcast `aten.add.Tensor` in `QKVParallelLinear` — every Qwen2.5 row"),
    ("B-MOE", "build", ("moe_forward",),
     "opaque fused-MoE custom op `vllm.moe_forward.default` mutates an operand, no InPlaceRule — every OLMoE / Qwen3-30B-A3B row"),
    ("B-HOPPER-ATTN", "build", ("unified_attention_with_output",),
     "on cc 9.0 vLLM selects FA3 → `vllm.unified_attention_with_output.default` mutates an operand, no rule — every H100 row"),
    ("B-QNORM-STRIDE", "build", ("q_norm", "k_norm"),
     "`TritonRmsNorm[vllm-bi]_v1` refuses q/k-norm on the strided chunk view of fused qkv (row stride ≠ n_cols)"),
    ("B-AMPERE-GEMM-TILE", "build", ("BLOCK_SIZE_M", "TUNED_GEMM", "GEMM-TILE"),
     "(named by lane hw) Bet A's `TritonGemm[vllm-bi]_v1` binding refuses the Ampere default GEMM tiles (BLOCK_SIZE_M/N 128) at `o_proj` — the Build-side twin of M-AMPERE-GEMM-TILE; appears once X-01 admits the SM count"),
    ("B-SMCOUNT", "build", ("triton_launch",),
     "`verity_cba.triton_launch` at `o_proj`: BI persistent-matmul launch geometry on sm80 (108 SMs) ≠ pinned sm89 geometry → treated as unknown mutation"),
    ("B-GEMMA-SCALE", "build", ("numerics_unregistered", "scale_tensor"),
     "`numerics_unregistered`: `aten.mul.Tensor` role `scale_tensor` (Gemma-2 embedding scale) not in profile vocabulary `b1-eager-v3`"),
    ("B-GEMMA-NORM", "build", ("GemmaRMSNorm", "aten.to.dtype"),
     "(named by lane dense, F-dense-05) `numerics_unregistered`: GemmaRMSNorm's ATen chain (`aten.to.dtype` f32 upcast, (1+w) scale, softcap, hd-256) behind the X-08 scale op — Gemma-2 stays BUILD_FAIL at the next op"),
    ("M-ARRIVAL-ORDER", "match", ("M-ARRIVAL-ORDER", "within-step order", "arrival"),
     "(named by the coordinator 13:40Z, dA's Llama b16 1024/128 mixed-arrivals) arriving requests' prefill steps, co-scheduled with other requests' decode rows, contain exactly the derived Program's step-0 instance multiset in a different within-step order; the positional wiring compare reports operands_different — needs a ruling on a canonical within-step order (or fold emission keyed by (request, position)); R14"),
    ("B-FP8", "build", ("E_WIDTH", "float8"),
     "`E_WIDTH`: dtype `float8_e4m3fn` carries no `Value<w>` width — FP8 unsupported by construction (expected classification)"),
    ("M-AMPERE-GEMM-TILE", "match", ("BLOCK_SIZE_M", "TUNED_GEMM", "GEMM-TILE", "Ampere default"),
     "(named by lane hw) on A100 the BI persistent GEMM runs the Ampere default tiles (BLOCK_SIZE_M/N 128,128) while the eager profile pins the Ada-tuned tiles (64,64 / 32,128) → every o_proj launch Unsupported at Match even after X-01 admits NUM_SMS; tile geometry changes reduction order, so this is numerics, not scheduling"),
    ("M-SMCOUNT", "match", ("matmul_kernel_persistent", "NUM_SMS", "geometry differs"),
     "eager fold resolver: `batch_invariant.matmul_kernel_persistent` launch geometry differs from the pinned sm89 profile (`NUM_SMS` 142 on L40S vs 128 on the RTX 4090 it was pinned on) → every GEMM launch Unsupported"),
    ("M-COMPILED-FUSION", "match", ("reshape_and_cache_flash", "torch.compile", "cudagraph", "CUDA graph", "execution-mode/fusion"),
     "(proposed by report) under `bi-compiled` (torch.compile + CUDA graphs) the fold sees reads of views that no observed launch wrote (kv-cache update, attention inputs) — fused/graph-replayed kernels are invisible to the eager Match resolver"),
    ("M-NOPROFILE", "match", ("no profile module declares",),
     "`run_config`: no profile module declares the case — the Match oracle needs a hand-written `gen_*` eager profile per model family"),
    ("M-TP-ENGINE", "match", ("build_engine", "TP1 (digest"),
     "`m1_capture.build_engine` cannot build a TP2 engine (capture and control fail identically); Program is TP1 (builder has no TP parameter)"),
    ("C-HIDDENGPU", "commit", ("No module named 'hidden_gpu'",),
     "`ModuleNotFoundError: hidden_gpu` in every worker step — the GPU-tree extension is not built/shipped by `pod_up.sh`; every Commit on a fresh pod fails"),
    ("C-HOPPER-FA3", "commit", ("tensors=0", "binds 0", "FA3 executes"),
     "tap loaded on H100 but `fa2_hidden_m1_stream` binds 0 tensors (FA3 executes, FA2 never called) → hidden values unbound"),
    ("C-FA2-HDIM", "commit", ("not compiled in this build", "head_dim 64 only", "head_size="),
     "(proposed by report) FA2 matReq hidden-value tap compiled for head_dim 64 only (SmolLM2/Llama) → first instrumented step aborts on any head_dim 96/128/256 model (Phi-3, Mistral, OLMoE, Qwen2.5/3, Gemma-2)"),
    ("M-BATCH-DECOMP", "match", ("derived Program is single-request", "M-BATCH-DECOMP", "second request's work"),
     "(proposed by report, F-report-02) Bet A's request wrapper derives ONE request's trajectory (--LP, --T); a mixed batch of B>=2 requests executes the manifest's length vector, so the fold (clean: 0 unsupported/unresolved) differs from the derived Program by exactly the extra requests' work — no batch/schedule wrapper, and Build passes fail-open for B>1"),
    ("B-SAMPLING", "build", ("ArgmaxTokenSelect", '"sampling": "greedy"', "--sampling", "greedy-only"),
     "(proposed by report) Bet A's `derive_step` has no sampling parameter: `structural_inputs.sampling` is the constant \"greedy\" and the Program ends in `ArgmaxTokenSelect_v1`, so a stochastic row's Program is the greedy Program — Build is coupled to the greedy path (BRIEF §9)"),
    ("C-COLLECTOR-FOOTPRINT", "commit", ("COLLECTOR-FOOTPRINT", "retained device copies", "collector footprint"),
     "(named by coordinator, CANARY.md 04:52Z) the collector's retained device copies (~9.2 GB/seq for Llama-3.2-1B at 1024/128) exceed the GPU at B≥2 even with the arena sized from the envelope (X-02) — a capacity limit of the commitment mechanism itself, not of the arena bound"),
    ("C-FA2-STREAM", "commit", ("hidden 18.5", "FA2 hidden stream", "Fa2Wrap::buffer", "hidden M1", "does not fit: 2 x 0.06 GiB + 35.00 GiB"),
     "(named by report from F-fa2-02 / F-report-07/08 / F-bc-05; COORD 09:55Z 'remaining O(rows x N) object', unowned) the FA2 hidden stream is O(rows x N): one 1024-token Mistral-7B prefill = 18.5 GiB hidden M1, Phi-3 b8 1024 = 132 GiB per prefill step; `Fa2Wrap::buffer` is cached per geometry without eviction (24.5 GiB at B=64) — 1024-token rows of hd-128/32-head families beyond Llama-1B do not arm on 48 GB even with the windowed collector"),
    ("INFRA-HARNESS", "match", ("derived_unreadable", "stale derived", "harness defect"),
     "(infrastructure, not a mechanism) row-procedure / tool defects that produce a verdict the evidence contradicts — e.g. `batch_decomp` fails closed on a stale single-shape `build_request` dir left by a pre-X-09 Build with `reasons: []` while every request is canonical-equal (F-report-12)"),
    ("INFRA-ENV", "build", ("LocalEntryNotFoundError", "Cannot find an appropriate cached snapshot", "not cached on the pool pod"),
     "(infrastructure, not a mechanism) pod environment: checkpoint snapshot not cached on the pod (pool pods run offline), missing extension, PATH"),
    ("INFRA-TIMEOUT", "build", ("TIMEOUT after 900s", "BUILD_TIMEOUT default 900", "build timeout"),
     "(infrastructure, not a mechanism) row_pod.sh BUILD_TIMEOUT default 900 s is shorter than Bet A construction at B=8/1024 (911 s measured on rerun)"),
    ("C-FOLD-COST", "match", ("TIMEOUT after", "rc=124", "compare rc=124", "cb_a.compare rc=124", "fold] ... TIMEOUT", "comparator time limit"),
     "(infrastructure, not a mechanism) Match-stage time limits: the fold at B≥8/1024 exceeds row_pod.sh's MATCH_TIMEOUT and cb_a.compare exceeds the hard-coded 600 s at B≥4 — no verdict is reached (F-tp-05 named it C-FOLD-COST; kept under one id)"),
    ("C-ARENA", "commit", ("arena", "OutOfMemoryError", "out of memory", "block failed", "collector"),
     "native collector arena: fixed 512 MiB block too small / OOM allocating blocks at B×1024-token prefill with vLLM at high memory utilisation; unbounded backlog"),
]
CLUSTER_MECHANISM = {c[0]: c[3] for c in NAMED_CLUSTERS}

# Daniel 04:04Z: fresh-deployment blockers (shared deployment/profile assumptions, fixed by X-00..X-03) vs Verity
# generalization gaps (genuine mechanisms; ranked by leverage = rows unblocked).
DEPLOYMENT_CLUSTERS: dict[str, str] = {"C-HIDDENGPU": "X-00", "M-SMCOUNT": "X-01", "B-SMCOUNT": "X-01", "C-ARENA": "X-02"}
GAP_RANK: list[str] = ["M-BATCH-DECOMP", "B-QKVBIAS", "C-FA2-HDIM", "B-MOE", "B-HOPPER-ATTN", "C-HOPPER-FA3", "M-AMPERE-GEMM-TILE", "B-AMPERE-GEMM-TILE", "M-COMPILED-FUSION", "C-COLLECTOR-FOOTPRINT", "M-TP-ENGINE", "B-SAMPLING", "B-FP8",
                       "B-GEMMA-SCALE", "B-GEMMA-NORM", "B-QNORM-STRIDE", "M-NOPROFILE", "C-FA2-STREAM", "M-ARRIVAL-ORDER"]
GAP_LABEL = {"B-QKVBIAS": "Qwen2 broadcast add", "C-FA2-HDIM": "FA2 head-dim 64-only tap", "B-MOE": "fused MoE (`moe_forward`)",
             "B-HOPPER-ATTN": "Hopper/FA3 attention (Build)", "C-HOPPER-FA3": "Hopper/FA3 hidden values (Commit)",
             "M-BATCH-DECOMP": "single-request derived Program vs mixed batch (no batch wrapper)", "C-FOLD-COST": "Match-stage time limits (fold / comparator) — infrastructure", "M-AMPERE-GEMM-TILE": "Ampere GEMM tile geometry (Match; profile pinned to Ada tiles)", "B-AMPERE-GEMM-TILE": "Ampere GEMM tile geometry (Build)", "M-COMPILED-FUSION": "compiled-mode fusion invisible to Match", "C-COLLECTOR-FOOTPRINT": "collector device footprint (Commit capacity)", "M-TP-ENGINE": "TP collectives", "B-SAMPLING": "sampling coupled to greedy Build", "B-FP8": "FP8 width", "B-GEMMA-SCALE": "Gemma numerics",
             "B-QNORM-STRIDE": "q_norm stride", "M-NOPROFILE": "family eager profile", "B-GEMMA-NORM": "Gemma-2 RMSNorm ATen chain (next op after X-08)", "M-ARRIVAL-ORDER": "mixed arrivals: within-step instance order of co-scheduled prefills", "C-FA2-STREAM": "FA2 hidden stream O(rows x N) / Fa2Wrap buffer (Commit footprint of hd-128 families at 1024 tokens)"}
DEPLOY_LABEL = {"C-HIDDENGPU": "`hidden_gpu` absent from pod bootstrap", "M-SMCOUNT": "hard-coded SM count (Match resolver)",
                "B-SMCOUNT": "hard-coded SM count (Bet A binding)", "C-ARENA": "fixed arena capacity"}
RULES_VERSION = "r14-v1"


def _panel_ids() -> frozenset[str]:
    """The 75-row panel: out/gen/sweep/PANEL.json (report lane, R15) — fallback the row files of tag r13-final-addendum; empty set = no restriction."""
    for root in ("out/gen/sweep", os.environ.get("SWEEP_ROOT", "")):
        p = os.path.join(root, "PANEL.json") if root else ""
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    P = json.load(fh)
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


PANEL_IDS = _panel_ids() if os.environ.get("SWEEP_CAMPAIGN", "r14") == "r14" else frozenset()                 # TABLE-RULES.md (R14): the acceptance standard the finalizer and renderer consume
CAMPAIGN = os.environ.get("SWEEP_CAMPAIGN", "r14")   # render --campaign sets this; r14 => historical (r13) passes show ◐
ENV_FIXES = ("X-00", "X-03")            # environment / harness: not per-configuration engineering
DEPLOYMENT_FIXES = ("X-00", "X-01", "X-02", "X-03")


INFRA_CLUSTERS: dict[str, str] = {"INFRA-ENV": "rel: pod environment (checkpoint cache, extensions)", "INFRA-HARNESS": "rel/dense: harness defects (stale derived dirs; empty fail reasons)", "C-FOLD-COST": "rel: Match-tool cost (fold / comparator time limits; REL-06 scales them with B) — rows carry ◐ 'Match unfinished (fold cost)' (COORD 07:30Z)",
                                  "INFRA-TIMEOUT": "rel: build-stage harness limits in row_pod.sh (time limits; a SIGKILLed derive)"}


def bucket_of(cluster_id: str) -> str:
    if cluster_id in DEPLOYMENT_CLUSTERS:
        return "deployment"
    return "infra" if cluster_id in INFRA_CLUSTERS else "gap"


def name_cluster(f: dict) -> str | None:
    fe = str(f.get("first_error") or "")
    for cid, stage, subs, _ in NAMED_CLUSTERS:
        if f.get("stage") == stage and any(sub in fe for sub in subs):
            return cid
    return None



# coordinator / reviewer rulings that the substring rules get wrong (R-11: hw-05/06 are the frozen-tree hidden_gpu leak)
CLUSTER_OVERRIDES: dict[str, str] = {"F-hw-05": "C-HIDDENGPU", "F-hw-06": "C-HIDDENGPU",
                                      "F-ampere-01": "M-BATCH-DECOMP", "F-fa2-03": "M-BATCH-DECOMP",  # COORD 07:40/08:45Z: F-report-02 = F-ampere-01 = F-fa2-03 (lanes named it M-BATCH-COMPARATOR)
                                      "F-fa2-02": "C-FA2-STREAM", "F-report-07": "C-FA2-STREAM", "F-report-08": "C-FA2-STREAM", "F-bc-05": "C-FA2-STREAM"}  # FA2 hidden stream O(rows x N): arm-time bound for a 1024-token prefill (p0a)


def same_as_ids(f: dict) -> list[str]:
    """`same_as` entries as bare failure ids (lanes annotate them, e.g. 'F-dse-08 (the mechanism half)')."""
    out = []
    for x in (f.get("same_as") or []):
        m = re.match(r"\s*(F-[A-Za-z0-9]+-\d+)", str(x))
        if m:
            out.append(m.group(1))
    return out


def cluster_failures(recs: Records) -> list[dict]:
    """Assign every failure record to a coordinator-named cluster (stage + first_error substring); records that match
    no rule inherit the cluster of a `same_as` neighbour, else fall into `?-<stage>-<class>`.  Returns cluster dicts
    (id, mechanism, failure_ids, row_ids, stages, classes, first_error, ...) in NAMED_CLUSTERS order, unnamed last."""
    assign: dict[str, str | None] = {}
    for fid, f in recs.failures.items():
        declared = str(f.get("cluster") or "").strip()
        # a record's own `cluster` tag wins if it names a known cluster of the record's stage (reviewer §5(c): stage rules)
        if declared == "M-BATCH-COMPARATOR":
            declared = "M-BATCH-DECOMP"  # COORD 09:55Z: one cluster id
        if declared in CLUSTER_MECHANISM and (declared in INFRA_CLUSTERS or declared[0] == {"build": "B", "match": "M", "commit": "C"}.get(str(f.get("stage")), "?")):
            assign[fid] = declared
        else:
            assign[fid] = CLUSTER_OVERRIDES.get(fid) or name_cluster(f)
    for _ in range(3):  # propagate through same_as links (both directions)
        for fid, f in recs.failures.items():
            links = [x for x in same_as_ids(f) if x in recs.failures]
            links += [o for o, g in recs.failures.items() if fid in same_as_ids(g)]
            if assign[fid] is None:
                for o in links:
                    if assign.get(o):
                        assign[fid] = assign[o]
                        break
    for fid, f in recs.failures.items():
        if assign[fid] is None:
            _cls = str(f.get('class', '?')).split(':')[0].split('(')[0].strip()
            assign[fid] = f"?-{f.get('stage', '?')}-{_cls[:48]}"
    groups: dict[str, list[str]] = {}
    for fid in sorted(recs.failures):
        groups.setdefault(assign[fid], []).append(fid)
    order = {c[0]: i for i, c in enumerate(NAMED_CLUSTERS)}
    out = []
    for cid, fids in groups.items():
        fs = [recs.failures[f] for f in fids]
        rows = sorted({str(f.get("row_id", "?")) for f in fs})
        stages = sorted({str(f.get("stage", "?")) for f in fs})
        classes = sorted({str(f.get("class", "?")) for f in fs})
        xl = sorted({o for f in fs for o in (f.get("same_as") or []) if o in recs.failures and assign.get(o) != cid})
        out.append({"id": cid, "class": " / ".join(classes), "classes": classes, "mechanism": CLUSTER_MECHANISM.get(cid, "(unnamed: not matched by any DECISION-0355 cluster rule — coordinator to name)"),
                    "failure_ids": fids, "row_ids": rows, "stages": stages,
                    "first_error": str(fs[0].get("first_error", "")), "artifact": str(fs[0].get("artifact", "")),
                    "cmd": str(fs[0].get("cmd", "")), "commit": str(fs[0].get("commit", "")), "cross_links": xl})
    out.sort(key=lambda c: (order.get(c["id"], len(order)), c["id"]))
    return out


def failure_resolved(recs: Records, f: dict) -> bool:
    """A failure is resolved when its row now passes the failing stage (or the row is in a fix's rows_newly_green)."""
    rid = f.get("row_id")
    row = recs.rows.get(rid) if isinstance(rid, str) else None
    if row is None:
        return False
    for x in recs.fixes.values():
        if rid in (x.get("rows_newly_green") or []):
            return True
    stage = f.get("stage")
    if stage in STAGES:
        return stage_status(row, stage) in ("PASS", "DONE")
    if stage == "infra":
        return row.get("status") not in ("QUEUED", "NOT_RUN") and not any(
            stage_status(row, s) == "FAIL" for s in STAGES)
    return False


def is_infra_fix(x: dict) -> bool:
    """X-00 / X-03 environment-or-harness fixes (DECISION-0355 + 04:04Z addendum): not per-configuration engineering."""
    m = str(x.get("mechanism", "")).lower()
    return str(x.get("fix_id", "")) in ENV_FIXES or any(k in m for k in ("infrastructure", "harness", "row procedure", "worker package"))


def engineering_fix_ids(recs: Records) -> list[str]:
    return sorted(x for x, f in recs.fixes.items() if not is_infra_fix(f))


def zero_code_changes(recs: Records, row: dict) -> bool:
    """Commit-green with no Verity mechanism change (only environment/harness fixes, if any)."""
    ccr = row.get("code_changes_required")
    if not isinstance(ccr, list):
        return False
    infra = {xid for xid, f in recs.fixes.items() if is_infra_fix(f)} | set(ENV_FIXES)
    if not all(x in infra for x in ccr):  # infra ids tolerated in ccr (R-19: they belong in infra_fixes)
        return False
    # a row whose seeded blockers (DEPS.json) were cleared by mechanism fixes in the release it ran on did NOT complete on
    # unchanged code, whatever its record claims (BRIEF §11: reject unsupported claims) — the ccr omission is reported
    dep = _deps_row(recs, row.get("row_id", ""))
    mech = {c: t for c, t in (dep.get("resolved") or {}).items() if c not in DEPLOYMENT_CLUSTERS}
    if mech:
        recs.add(f"rows/{row.get('row_id')}.json", "code_changes_required",
                 f"[] claimed but the row's blockers {sorted(mech)} were cleared by mechanism fixes in {sorted(set(mech.values()))} — not a zero-code completion (R-19)")
        return False
    return True


_DEPS_CACHE: dict[str, dict] = {}


def _deps_row(recs: Records, row_id: str) -> dict:
    root = getattr(recs, "root", None) or "out/gen/sweep"
    if root not in _DEPS_CACHE:
        p = os.path.join(root, "DEPS.json")
        try:
            with open(p, "r", encoding="utf-8") as fh:
                _DEPS_CACHE[root] = json.load(fh).get("rows") or {}
        except Exception:
            _DEPS_CACHE[root] = {}
    return _DEPS_CACHE[root].get(row_id) or {}


def deployment_only(recs: Records, row: dict) -> bool:
    """Green after fixing shared deployment/profile assumptions alone (X-00..X-03), no genuine-mechanism fix."""
    ccr = row.get("code_changes_required")
    return isinstance(ccr, list) and all(x in DEPLOYMENT_FIXES for x in ccr)


def row_blocking_cluster(recs: Records, row: dict, clusters: list[dict] | None = None) -> str | None:
    """Cluster id of the failure that currently blocks the row (its FAIL stage's failure record), else None."""
    clusters = clusters if clusters is not None else cluster_failures(recs)
    by_f = {f: cl["id"] for cl in clusters for f in cl["failure_ids"]}
    for stage in STAGES:
        if stage_status(row, stage) == "FAIL":
            st = row.get(stage) if isinstance(row.get(stage), dict) else {}
            fid = st.get("failure_id")
            if fid in by_f:
                return by_f[fid]
            # fall back: any failure record for this row at this stage
            for f, cid in by_f.items():
                fr = recs.failures[f]
                if fr.get("row_id") == row.get("row_id") and fr.get("stage") == stage:
                    return cid
            return "?-unlinked"
    return None


_FOLD_COST_CACHE: dict[int, dict[str, str]] = {}


def match_unfinished_fold_cost(recs: Records, row: dict) -> bool:
    """True when the row's Match FAIL is a harness time limit (cluster C-FOLD-COST / INFRA-TIMEOUT), i.e. no verdict was reached
    (COORD 07:30Z: such rows carry ◐ 'Match unfinished (fold cost)'; criteria are not weakened)."""
    key = id(recs)
    if key not in _FOLD_COST_CACHE:
        _FOLD_COST_CACHE[key] = {f: cl["id"] for cl in cluster_failures(recs) for f in cl["failure_ids"]}
    by_f = _FOLD_COST_CACHE[key]
    st = row.get("match") if isinstance(row.get("match"), dict) else {}
    fid = st.get("failure_id")
    cid = by_f.get(fid)
    if cid is None:
        for f, c in by_f.items():
            fr = recs.failures[f]
            if fr.get("row_id") == row.get("row_id") and fr.get("stage") == "match":
                cid = c
                break
    return cid in INFRA_CLUSTERS


def iter_violations(recs: Records) -> Iterable[Violation]:
    return sorted(recs.violations, key=lambda v: (v.file, v.field))
