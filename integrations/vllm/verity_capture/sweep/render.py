"""Render MATRIX.md / STATUS.md / FINAL-REPORT.md / VALIDATION.md from the R12 sweep records only.

    python -m verity_capture.sweep.render --root out/gen/sweep [--now HH:MMZ] [--blocker "text"]...

Status convention (BRIEF): stage PASS -> ✅; FAIL -> ❌; not reached -> —; PASS lacking required evidence -> ◐.
FINAL-REPORT.md: only `<!-- auto:begin NAME --> ... <!-- auto:end -->` blocks are rewritten; §7/§8 prose is preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from . import validate as R

GLYPH_PASS, GLYPH_PART, GLYPH_FAIL, GLYPH_NONE = "✅", "◐", "❌", "—"
GLYPH_MAP = {"PASS": GLYPH_PASS, "PARTIAL": GLYPH_PART, "FAIL": GLYPH_FAIL, "-": GLYPH_NONE, None: GLYPH_NONE}
GLYPH_INFEASIBLE, GLYPH_QUARANTINE = "⊘", "⛔"   # COORD 19:25Z ruling 6 (infeasible on the declared hardware) · ALERT quarantine (PLAN §3)
STATE_REL = "lanes/report-state.json"
AUTO_RE = re.compile(r"<!-- auto:begin (\w+) -->\n?.*?<!-- auto:end -->", re.S)
QUARANTINE_FILE = "quarantine.json"   # out/gen/sweep/quarantine.json {row_id: {alert, reason, since, until?}} — report-lane sidecar, never a record edit
ROW_NOTES_FILE = "row_notes.json"     # out/gen/sweep/row_notes.json {row_id: "caveat"} — report-lane caveats printed beneath the table
STRUCTURAL_FILE = "structural.json"   # out/gen/sweep/structural.json {row_id: {reason, ordered_by, since, lifted?}} — COORD-ordered structural ◐ (M-0614 (4)): every cell ◐, derived status named beneath

_SIDE: dict[str, dict] = {}


def sidecar(recs: R.Records, name: str) -> dict:
    key = os.path.join(recs.root, name)
    if key not in _SIDE:
        try:
            with open(key, encoding="utf-8") as fh:
                d = json.load(fh)
            _SIDE[key] = d if isinstance(d, dict) else {}
        except Exception:
            _SIDE[key] = {}
    return _SIDE[key]


def quarantined(recs: R.Records, row: dict, stage: str | None = None) -> dict | None:
    """The row's open quarantine entry.  An entry may be scoped to stages (`stages: ["commit", "bench"]` — COORD ALERT M-0488 quarantines
    the Commit cell and leaves Build/Match as derived); with `stage` given, a scoped entry applies only to the named stages."""
    q = sidecar(recs, QUARANTINE_FILE).get(row.get("row_id", ""))
    if not (isinstance(q, dict) and not q.get("lifted")):
        return None
    if stage is not None and isinstance(q.get("stages"), list) and q["stages"] and stage not in q["stages"]:
        return None
    if q.get("lift_on") == "own_c2_record" and c2_record_lifts(row):
        return None   # COORD DECISION M-0522 (2): the M-0488 Commit quarantine lifts on the row's OWN passing C2 record (the check that exposed the defect)
    return q


def structural(recs: R.Records, row: dict) -> dict | None:
    """[COORD DECISION M-0614 (4)] a row the coordinator ruled STRUCTURALLY pending (e.g. the four B8 stochastic rows: 'derived split geometry
    pending (M-0571 (3))') renders ◐ in every cell — an honest structural status, not infeasible and not the R13 ❌ — with the derived
    status named beneath the table.  Sidecar, never a record edit; lifted by the board."""
    e = sidecar(recs, STRUCTURAL_FILE).get(row.get("row_id", ""))
    return e if isinstance(e, dict) and not e.get("lifted") else None


def c2_record_lifts(row: dict) -> bool:
    """[COORD M-0522 (2)] the displayed attempt's Commit is PASS on a row-own C2 form of record — (B) oracle-compare of record (0 mismatch,
    attribution complete, graded by finalize.oracle_form_b) or (A) sampled replay + boundary linkage."""
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else row
    c = a.get("commit") if isinstance(a.get("commit"), dict) else {}
    return str(c.get("status") or "").split()[0:1] == ["PASS"] and str(c.get("c2_form") or "") in ("oracle-compare", "sampled replay")


def infeasible(row: dict, stage: str) -> bool:
    return R.stage_status(row, stage) == "INFEASIBLE"


# ----------------------------------------------------------------------------------------------------------------
# glyphs


def glyph(recs: R.Records, row: dict, stage: str, ignore_quarantine: bool = False) -> str:
    if row.get("status") in ("QUEUED", "NOT_RUN"):
        return GLYPH_NONE
    if not ignore_quarantine and quarantined(recs, row, stage):
        return GLYPH_QUARANTINE   # ALERT quarantine: displayed statuses withdrawn until the fix's RESULT (reason beneath the table)
    if not ignore_quarantine and structural(recs, row):
        return GLYPH_PART         # COORD M-0614 (4): structural ◐ by board decision (reason + derived status beneath the table)
    eff = R.stage_status(row, stage)
    if eff == "INFEASIBLE":
        return GLYPH_INFEASIBLE
    if eff in ("PASS", "DONE") and R.CAMPAIGN == "r14" and row.get("_standard") != R.RULES_VERSION:
        return GLYPH_PART  # TABLE-RULES: a pass under an older standard is ◐ under r14-v1 until revalidated by the finalizer
    if eff in ("PASS", "DONE"):
        miss = [m for m in recs.missing_evidence.get(row.get("row_id", ""), []) if m[0] == stage]
        if stage == "bench" and not miss and glyph(recs, row, "commit") == GLYPH_PART:
            return GLYPH_PART  # overhead of a Commit that is itself only partially demonstrated
        return GLYPH_PART if miss else GLYPH_PASS
    if eff == "PARTIAL":
        return GLYPH_PART  # finalizer verdict: evidence insufficient under r14-v1 (reasons in the stage dict)
    if eff == "FAIL":
        if stage == "match" and R.match_unfinished_fold_cost(recs, row):
            return GLYPH_PART  # COORD 07:30Z: harness time limit, criteria unchanged -> ◐ "Match unfinished (fold cost)", not ❌
        return GLYPH_FAIL
    return GLYPH_NONE


def bench_numbers(row: dict) -> dict:
    b = row.get("bench") if isinstance(row.get("bench"), dict) else {}
    cs, ts = b.get("control_s") or [], b.get("treatment_s") or []
    cm = b.get("control_mean_s")
    tm = b.get("treatment_mean_s")
    if (cm is None or cm == 0) and cs:
        cm = sum(cs) / len(cs)
    if (tm is None or tm == 0) and ts:
        tm = sum(ts) / len(ts)
    rel = b.get("overhead_rel")
    if rel is None and cm and tm:
        rel = tm / cm - 1
    rounds = b.get("rounds") or (len(cs) if cs else None)
    return {"control_mean_s": cm, "treatment_mean_s": tm, "overhead_rel": rel, "rounds": rounds,
            "abs_ms": b.get("overhead_abs_ms_per_step"), "steps": b.get("decode_steps"),
            "artifact": b.get("artifact"), "cmd": b.get("cmd")}


def overhead_cell(recs: R.Records, row: dict) -> str:
    if quarantined(recs, row, "bench"):
        return GLYPH_QUARANTINE
    if row.get("status") in ("QUEUED", "NOT_RUN") or R.stage_status(row, "bench") not in ("DONE", "PARTIAL"):
        return GLYPH_NONE
    n = bench_numbers(row)
    if n["overhead_rel"] is None:
        return GLYPH_PART
    cell = f"{n['overhead_rel'] * 100:+.1f}% ({n['rounds'] if n['rounds'] is not None else '?'} rounds)"
    if glyph(recs, row, "bench") == GLYPH_PART or glyph(recs, row, "commit") == GLYPH_PART:
        hist = R.CAMPAIGN == "r14" and row.get("_standard") != R.RULES_VERSION and R.stage_status(row, "bench") == "DONE" and R.stage_status(row, "commit") == "PASS"
        cell = f"{'☑' if hist else GLYPH_PART} {cell}"
    return cell


# ----------------------------------------------------------------------------------------------------------------
# display helpers


def _esc(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def display_model(row: dict) -> str:
    cfg = row.get("config") or {}
    if isinstance(cfg.get("model_display"), str) and cfg["model_display"]:
        return cfg["model_display"]
    m = R.model_of(row)
    return R.MODEL_DISPLAY.get(m, m)


def display_moe(row: dict) -> str:
    cfg = row.get("config") or {}
    if isinstance(cfg.get("moe"), bool):
        return "Yes" if cfg["moe"] else "No"
    m = R.model_of(row)
    return "Yes" if R.MODEL_MOE.get(m, False) else "No"


def display_hw(row: dict) -> str:
    cfg = row.get("config") or {}
    hw = cfg.get("hardware")
    if isinstance(hw, str) and hw:
        return R.HW_DISPLAY.get(hw.lower().replace(" ", ""), hw)
    return R.HW_DISPLAY.get(R.hw_key(row), R.hw_key(row))


def display_int(row: dict, key: str) -> str:
    v = R.cfg_value(row, key)
    return str(v) if v is not None else "?"


def partition(recs: R.Records, row: dict) -> str:
    """Daniel (BRIEF §11): fully completed = Build+Match+Commit ✅ and a measured overhead; partially completed = something
    demonstrated (any ✅/◐ stage) but not all; blocked/not-run = nothing demonstrated yet."""
    gs = [glyph(recs, row, st) for st in ("build", "match", "commit")]
    oc = overhead_cell(recs, row)
    if all(g == GLYPH_PASS for g in gs) and oc not in (GLYPH_NONE, GLYPH_PART) and not oc.startswith(GLYPH_PART):
        return "complete"
    if any(g in (GLYPH_PASS, GLYPH_PART) for g in gs):
        return "partial"
    return "blocked"


LANE_ONLY_MARK = "†"


def std_tag(row: dict) -> str:
    """acceptance-standard version of the displayed attempt (TABLE-RULES): ⟨r14-v1⟩ finalizer verdict, ⟨r13⟩ historical."""
    st = row.get("_standard") or "r13"
    if row.get("status") in ("QUEUED", "NOT_RUN"):
        return ""
    return f" ⟨{st}⟩"


GLYPH_HIST = "☑"  # passed under the R13 standard, not yet revalidated under r14-v1 (counts as ◐; Daniel 17:36Z ruling 7)


def cell(recs: R.Records, row: dict, stage: str) -> str:
    """Table cell: three visibly distinct states — ✅ R14 finalizer verdict · ☑ historical R13 pass awaiting revalidation · ◐/❌/— as derived.
    '— (earlier ❌)': the displayed attempt did not reach the stage but an earlier attempt failed it (brief §9; reason beneath the table)."""
    g = glyph(recs, row, stage)
    if R.CAMPAIGN == "r14" and g == GLYPH_PART and row.get("_standard") != R.RULES_VERSION and R.stage_status(row, stage) in ("PASS", "DONE"):
        return GLYPH_HIST
    if g == GLYPH_NONE and unsuperseded_fail(row, stage):
        return GLYPH_NONE + " (earlier " + GLYPH_FAIL + ")"
    return g


def unsuperseded_fail(row: dict, stage: str) -> dict | None:
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else {}
    u = a.get("unsuperseded_fail") if isinstance(a.get("unsuperseded_fail"), dict) else None
    if not u:
        return None
    return next((w for w in (u.get("stages") or []) if w.get("stage") == stage), None)


def matrix_row(recs: R.Records, row: dict) -> list[str]:
    dtype = str(R.cfg_value(row, "dtype", "?")).upper()
    wl = str(R.cfg_value(row, "workload", "?"))
    sp = str(R.cfg_value(row, "sampling", "?"))
    ex = str(R.cfg_value(row, "execution", "?"))
    return [
        display_model(row), display_moe(row), dtype, display_hw(row),
        display_int(row, "tp"), display_int(row, "batch"), display_int(row, "max_input"), display_int(row, "max_output"),
        R.WORKLOAD_DISPLAY.get(wl, wl), R.SAMPLING_DISPLAY.get(sp, sp), R.EXEC_DISPLAY.get(ex, ex),
        cell(recs, row, "build") + (std_tag(row) if R.CAMPAIGN == "r14" else ""), cell(recs, row, "match") + (" fold-cost" if R.stage_status(row, "match") == "FAIL" and R.match_unfinished_fold_cost(recs, row) else ""),
        cell(recs, row, "commit") + c2_form_label(recs, row) + (LANE_ONLY_MARK if lane_only(row) else "") + (" ⏱post-freeze" if (R.CAMPAIGN != "r14" and ((row.get("_attempt") or {}).get("post_freeze") or row.get("post_freeze"))) else ""),
        overhead_cell(recs, row),
    ]


def c2_form_label(recs: R.Records, row: dict) -> str:
    """[COORD M-0481] a Commit ✅ names its C2 evidence form in the cell: 'C2: oracle-compare' | 'C2: sampled replay' (| 'C2: qualification <id>')."""
    if glyph(recs, row, "commit") != GLYPH_PASS:
        return ""
    c = row.get("commit") if isinstance(row.get("commit"), dict) else {}
    form = str(c.get("c2_form") or "")
    if not form and isinstance(c.get("c2_qualification"), dict):
        form = f"qualification {c['c2_qualification'].get('id')}"
    return f" C2: {form}" if form else ""


def lane_only(row: dict) -> bool:
    """Passing on a lane-only revision (no r13-relN release tag on the latest attempt)."""
    return R.release_label(row) == "lane-only" and R.stage_status(row, "commit") == "PASS"


MATRIX_HEADER = ("| Model | MoE | Datatype | Hardware | TP | Batch | Max input tokens | Max output tokens | Workload | "
                 "Sampling | Execution | Build | Match | Commit | Commitment overhead |\n"
                 "|---|---:|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|")


def commit_fail_notes(recs: R.Records) -> list[str]:
    """One footnote per Commit ❌ row: the count and the first missing identity (or the failure reason), from the finalizer's reasons."""
    out = []
    for r in R.sorted_rows(recs):
        if glyph(recs, r, "commit") != GLYPH_FAIL:
            continue
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
        c = a.get("commit") if isinstance(a.get("commit"), dict) else {}
        rs = [x for x in (c.get("reasons") or []) if x]
        why = rs[0] if rs else str(c.get("fail_reason") or c.get("first_fail_reason") or c.get("reason") or c.get("notes") or r.get("notes") or "executed Commit check failed (R13 record; reason in the row's failure record)")
        m = re.search(r"(\d[\d ,]*) of (\d[\d ,]*) required identities have no binding \(first: ([^)]*)\)", why)
        if m:
            why = f"coverage {int(m.group(2).replace(' ', '').replace(',', '')) - int(m.group(1).replace(' ', '').replace(',', ''))}/{m.group(2).strip()} — {m.group(1).strip()} required identities unbound, first missing `{m.group(3).strip()}`"
        out.append(f"- `{short_row(r)}` ({(a.get('release_tag') or 'lane-only ' + str(a.get('revision') or '?')[:9])}): {why[:260]}")
    return out


def short_row(r: dict) -> str:
    rid = r.get("row_id", "")
    parts = rid.split("__")
    try:
        return f"{parts[0]} {parts[2]} {parts[3]} {parts[4]} {parts[5][1:]}/{parts[6][1:]} {parts[7]}" + (f" {parts[8]}" if len(parts) > 8 and parts[8] != "greedy" else "") + (f" {parts[9]}" if len(parts) > 9 and parts[9] != "bi-eager" else "")
    except Exception:
        return rid


def attempt_campaign(row: dict) -> str:
    """Which campaign produced the displayed attempt: 'r15' (rc3+ / r15 tags), 'r14' (rc1 revalidation window), 'r13' (historical)."""
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else {}
    if not a or not a.get("rules_version"):
        return "r13"
    if str(a.get("campaign") or "") == "r15":
        return "r15"
    tag = str(a.get("release_tag") or "")
    if re.search(r"r15-|r14-rc([3-9]|\d\d)", tag):
        return "r15"
    return str(a.get("campaign") or "r14")


def _r15_tree_note(row: dict) -> str:
    """'r14-rc3' or, when Build/Match ran on another tree than the Commit (evidence reuse, COORD 19:25Z ruling 5), name both trees."""
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else {}
    tag = a.get("release_tag") or "lane-only"
    rev = str(a.get("revision") or "")[:9]
    et = a.get("executing_tree") if isinstance(a.get("executing_tree"), dict) else None
    if et and et.get("lane_tree"):
        # COORD M-0132: the revision of record is the lane-tree commit; the finalizer's diff vs the release decides finalizability
        d = et.get("diff_vs_release")
        dtxt = "code diff vs release empty" if d == [] else (f"differs from the release in {len(d)} file(s)" if isinstance(d, list) else "diff vs release not checkable")
        tag = f"lane tree {str(et.get('commit') or rev)[:9]} on {et.get('base_tag') or str(et.get('base') or '?')[:9]}, {dtxt}"
    trees = {}
    for st in ("build", "match"):
        d = a.get(st) if isinstance(a.get(st), dict) else {}
        t = d.get("revision") or d.get("tree") or (d.get("reused_from") and "reused")
        if t and str(t)[:9] != rev:
            trees[st] = str(t)[:9]
    if not trees:
        return str(tag)
    parts = "/".join(sorted(trees))
    vals = sorted(set(trees.values()))
    return f"{tag}: Commit on {rev or tag}; {parts} artifacts from tree {', '.join(vals)} (evidence reuse — manifest program_digest checked against the reused Build, COORD 19:25Z ruling 5)"


def provenance_notes(recs: R.Records) -> list[str]:
    """R14-revalidated (rc1) vs R15 (rc3+) results, regressions, infeasible rows, quarantines and report-lane row notes (brief §9/§10)."""
    out: list[str] = []
    r14 = [r for r in R.sorted_rows(recs) if attempt_campaign(r) == "r14" and any(glyph(recs, r, s) == GLYPH_PASS for s in ("build", "match", "commit"))]
    r15 = [r for r in R.sorted_rows(recs) if attempt_campaign(r) == "r15" and any(glyph(recs, r, s) in (GLYPH_PASS, GLYPH_PART, GLYPH_FAIL) for s in ("build", "match", "commit"))]
    if r14 or r15:
        out.append("")
        out.append("Provenance of the displayed attempt (⟨r14-v1⟩ cells): **R14 revalidation window (rc1 / r14-base evidence)** — " +
                   (", ".join(f"`{short_row(r)}`" for r in r14) if r14 else "none") +
                   ". **R15 results (rc3 and later, `campaign: r15`)** — " + (", ".join(f"`{short_row(r)}` ({_r15_tree_note(r)})" for r in r15) if r15 else "none yet") +
                   ". Both are verdicts of the same finalizer under the same rules_version; R14 rows were not re-run on rc3 unless listed under R15.")
    regs = []
    for r in R.sorted_rows(recs):
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        reg = a.get("regression") if isinstance(a.get("regression"), dict) else None
        if reg and reg.get("stages"):
            regs.append(f"- `{short_row(r)}`: " + "; ".join(f"{w['stage']} {GLYPH.get(w['was'], w['was'])}→{GLYPH.get(w['now'], w['now'])} (earlier attempt {w['earlier_attempt'].get('release_tag') or str(w['earlier_attempt'].get('revision') or '?')[:9]} at {str(w['earlier_attempt'].get('at') or '')[11:16]}Z)" for w in reg["stages"]) +
                        f" — the newer result is displayed (brief §9); reason: " + "; ".join(str(x)[:140] for x in ((a.get(reg['stages'][0]['stage']) or {}).get('reasons') or [])[:1]))
    if regs:
        out.append("")
        out.append("Regressions (a newer attempt shows a stage the row had already passed under r14-v1; the older green is not shown over it):")
        out += regs
    unsup = []
    for r in R.sorted_rows(recs):
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        u = a.get("unsuperseded_fail") if isinstance(a.get("unsuperseded_fail"), dict) else None
        for w in (u or {}).get("stages") or []:
            ea = w.get("earlier_attempt") or {}
            unsup.append(f"- `{short_row(r)}`: {w['stage']} {GLYPH_FAIL} in the earlier attempt {ea.get('release_tag') or str(ea.get('revision') or '?')[:9]} at {str(ea.get('at') or '?')[:16]}Z"
                         f" ({'; '.join(str(x)[:140] for x in (w.get('reasons') or [])) or 'no reason recorded'}) — the displayed attempt ({str(a.get('release_tag') or a.get('revision') or '?')[:12]}) did not reach {w['stage']}; the ❌ stands until a later attempt passes or fails it")
    if unsup:
        out.append("")
        out.append("Earlier failures not yet superseded (the displayed attempt stopped before the stage; brief §9 — a newer partial attempt does not hide an older ❌):")
        out += unsup
    ltn = []
    for r in R.sorted_rows(recs):
        n = r.get("_lane_tree_newer") if isinstance(r.get("_lane_tree_newer"), dict) else None
        if not n:
            continue
        stg = " ".join(f"{k} {GLYPH_MAP.get(v, v)}" for k, v in (n.get("stages") or {}).items())
        m_lane = re.search(r"evidence/([^/]+)/", str(n.get("evidence_dir") or ""))
        if m_lane and (not n.get("lane") or n.get("lane") == "evidence"):
            n["lane"] = m_lane.group(1)
        why = "; ".join(str(x)[:150] for x in (n.get("commit_reasons") or []))
        ltn.append(f"- `{short_row(r)}`: lane-tree attempt {str(n.get('at') or '?')[:16]}Z ({n.get('lane')}, {n.get('tree')} @ {str(n.get('commit') or '?')[:9]}{', dirty' if n.get('dirty') else ''}, base {n.get('base_tag') or '?'}) — "
                   f"{stg}{(' overhead %+.1f%%' % n['bench']) if isinstance(n.get('bench'), (int, float)) else ''}; it is a FIX MEASUREMENT on a non-release tree, not the row's attempt of record: the displayed release-tree attempt stands until the fix is released and the row re-run "
                   f"(COORD 12:00Z ruling b; the lane-tree attempt failed no stage). {('Beneath: ' + why) if why else ''}")
    if ltn:
        out.append("")
        out.append("Newer lane-tree attempts on file (not displayed as the headline — brief §9 keeps regressions visible, and these failed no stage):")
        out += ltn
    inw = []
    for r in R.sorted_rows(recs):
        n = r.get("_incomplete_newer") if isinstance(r.get("_incomplete_newer"), dict) else None
        if not n:
            continue
        stg = " ".join(f"{k} {GLYPH_MAP.get(v, v)}" for k, v in (n.get("stages") or {}).items())
        kept = n.get("kept") or {}
        why = "; ".join(str(x)[:150] for x in (n.get("reasons") or []))
        inw.append(f"- `{short_row(r)}`: newer release-tree attempt {str(n.get('at') or '?')[:16]}Z ({n.get('lane')}, {n.get('release_tag')} @ {str(n.get('revision') or '?')[:9]}, {n.get('kind')}) — {stg}; "
                   f"it reached less than the displayed attempt ({kept.get('release_tag')} {str(kept.get('at') or '?')[:16]}Z) and failed no stage: an incomplete diagnostic does not erase a valid result on the "
                   f"supported release (R16 brief 3(c); COORD 12:00Z ruling b). {('Lacks: ' + why) if why else ''}")
    if inw:
        out.append("")
        out.append("Newer release-tree attempts that reached less (on file, not the headline — a real regression ❌ would be):")
        out += inw
    inf = []
    for r in R.sorted_rows(recs):
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        if isinstance(a.get("infeasible"), dict):
            i = a["infeasible"]
            inf.append(f"- `{short_row(r)}` ⊘ {i.get('stage')}: {str(i.get('error') or '')[:200]} — attempted configuration recorded in the attempt (`infeasible.configuration`), not a substitute configuration (COORD 19:25Z ruling 6)")
    if inf:
        out.append("")
        out.append("Infeasible on the declared hardware (⊘ — a terminal state, never a substitute configuration):")
        out += inf
    st = sidecar(recs, STRUCTURAL_FILE)
    sts = [(rid, v) for rid, v in st.items() if isinstance(v, dict) and not v.get("lifted") and rid in recs.rows]
    if sts:
        out.append("")
        out.append("Structural ◐ by board decision (every cell ◐ — an honest structural status, not infeasible; the derived status is named here, the sidecar is out/gen/sweep/structural.json):")
        for rid, v in sorted(sts):
            r = recs.rows[rid]
            derived = " ".join(f"{stg} {glyph(recs, r, stg, ignore_quarantine=True)}" for stg in ("build", "match", "commit", "bench"))
            # a measured ❌ beneath a structural ◐ is named (a FAIL of record is never hidden behind a board-ordered status — brief §9)
            fails = []
            for stg in ("build", "match", "commit"):
                d_ = r.get(stg) if isinstance(r.get(stg), dict) else {}
                if d_.get("status") == "FAIL" and d_.get("reasons"):
                    fails.append(f"{stg} ❌ of record: {str(d_['reasons'][0])[:260]}")
            out.append(f"- `{short_row(r)}`: {v.get('reason')} — ordered by {v.get('ordered_by')} (since {v.get('since')}); derived beneath: {derived}"
                       + (" — " + "; ".join(fails) if fails else ""))
    q = sidecar(recs, QUARANTINE_FILE)
    qs = [(rid, v) for rid, v in q.items() if isinstance(v, dict) and not v.get("lifted") and rid in recs.rows and not (v.get("lift_on") == "own_c2_record" and c2_record_lifts(recs.rows[rid]))]
    lifted_by_record = [rid for rid, v in q.items() if isinstance(v, dict) and not v.get("lifted") and rid in recs.rows and v.get("lift_on") == "own_c2_record" and c2_record_lifts(recs.rows[rid])]
    if lifted_by_record:
        out.append("")
        out.append("Quarantine lifted by the row's own passing C2 record (COORD M-0522 (2); the M-0488 entry stays on file): " + ", ".join(f"`{short_row(recs.rows[x])}` (C2: {((recs.rows[x].get('_attempt') or {}).get('commit') or {}).get('c2_form')})" for x in lifted_by_record))
    if qs:
        out.append("")
        out.append("Quarantined (⛔ — displayed statuses withdrawn after an ALERT until the fix's RESULT; the record is untouched; an entry scoped to cells names them):")
        grouped: dict[tuple, list] = {}
        for rid, v in qs:
            key = (str(v.get("alert")), str(v.get("reason") or "")[:220], ",".join(v.get("stages") or []) if isinstance(v.get("stages"), list) else "", str(v.get("since") or "")[11:16])
            grouped.setdefault(key, []).append(rid)
        for (alert, reason, stages, since), rids in grouped.items():
            scope = f" [{stages.replace('bench', 'overhead')} cell(s) only]" if stages else ""
            # regressions stay visible (brief §9): the statuses derived BENEATH the quarantine, ❌ rows named with their reason
            st_of = "commit" if "commit" in stages else None
            hist: dict[str, list[str]] = {}
            for x in rids:
                row = recs.rows[x]
                if row.get("status") in ("QUEUED", "NOT_RUN"):
                    continue
                g = glyph(recs, row, st_of, ignore_quarantine=True) if st_of else str(row.get("status") or "-")
                hist.setdefault(g, []).append(x)
            beneath = ""
            if any((q.get(x) or {}).get("lift_on") == "own_c2_record" for x in rids):
                beneath = " Lifts per row on the row's OWN passing C2 record — form (B) oracle-compare of record (0 mismatch, attribution complete) at int-3, or (A)+(B) at int-4 (COORD M-0522 (2))."
            if st_of and hist:
                parts = []
                for g in (GLYPH_FAIL, GLYPH_PASS, GLYPH_PART):
                    if g in hist:
                        if g == GLYPH_FAIL:
                            names = []
                            for x in hist[g]:
                                a = recs.rows[x].get("_attempt") if isinstance(recs.rows[x].get("_attempt"), dict) else {}
                                why = " ".join(str(r0) for r0 in ((a.get("commit") or {}).get("reasons") or [])[:1])
                                names.append(f"`{short_row(recs.rows[x])}` ({why[:110]})")
                            parts.append(f"{g} ×{len(hist[g])}: " + "; ".join(names))
                        elif g == GLYPH_PART:
                            # a ◐ that is a NAMED checker defect (MAN-07 / M-0619) is not the same ◐ as 'C2 record pending' — name it
                            named = []
                            for x in hist[g]:
                                a = recs.rows[x].get("_attempt") if isinstance(recs.rows[x].get("_attempt"), dict) else {}
                                cd = (a.get("commit") or {}).get("c2_checker_defect")
                                if cd:
                                    why = " ".join(str(r0) for r0 in ((a.get("commit") or {}).get("reasons") or [])[:1])
                                    named.append(f"`{short_row(recs.rows[x])}` (checker defect {cd}: {why[:90]}…)")
                            parts.append(f"{g} ×{len(hist[g])}" + (" — " + "; ".join(named) if named else ""))
                        else:
                            parts.append(f"{g} ×{len(hist[g])}")
                beneath += " Derived beneath the quarantine (not displayed): " + " · ".join(parts) + "."
            if len(rids) == 1:
                out.append(f"- `{short_row(recs.rows[rids[0]])}`{scope}: {alert} {reason} (since {since}Z){beneath}")
            else:
                out.append(f"- {len(rids)} rows{scope} — {alert} {reason} (since {since}Z): " + ", ".join(f"`{short_row(recs.rows[x])}`" for x in rids) + beneath)
    # COORD M-0111 (1): every displayed Commit ✅ names its C2 reference (root-compare needs an independent reference run's provenance)
    c2 = []
    for r in R.sorted_rows(recs):
        if glyph(recs, r, "commit") != GLYPH_PASS:
            continue
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        c = a.get("commit") if isinstance(a.get("commit"), dict) else {}
        vc = c.get("value_correspondence") if isinstance(c.get("value_correspondence"), dict) else {}
        prov = c.get("c2_reference_provenance") if isinstance(c.get("c2_reference_provenance"), list) else [e for e in (vc.get("references") or []) if isinstance(e, dict) and e.get("root")]
        meth = str(vc.get("method") or "")
        if c.get("c2_form"):
            # DECISIONS-2 §4 / M-0451 / M-0481: the form of record and its basis line; M-0111 forms are diagnostics beneath it
            basis = "; ".join(str(b)[:220] for b in (c.get("c2_basis") or []))
            diag = "; ".join(str(d)[:100] for d in (c.get("c2_diagnostics") or [])[:3])
            c2.append(f"- `{short_row(r)}`: **{c['c2_form']}** — {basis}" + (f" (diagnostics: {diag})" if diag else ""))
        elif prov:
            kinds = "; ".join(str(e.get("kind") or "")[:90] for e in prov)
            weak = " — **same-tree reproduction, not an independent cross-pod reference**" if all(re.search(r"reproduction|this row's|same release tree|same tree", str(e.get("kind") or ""), re.I) for e in prov) else ""
            c2.append(f"- `{short_row(r)}`: {meth.split(' (')[0]} vs {kinds}{weak}")
        elif meth.startswith("oracle-compare"):
            c2.append(f"- `{short_row(r)}`: oracle-compare (Match oracle values, same attempt)")
        elif meth:
            c2.append(f"- `{short_row(r)}`: {meth[:120]} (reference provenance not recorded — pre-M-0111 record)")
    if c2:
        out.append("")
        out.append("C2 value correspondence of record behind each displayed Commit ✅ (DECISIONS-2 §4, COORD M-0451 + M-0481: (A) sampled exact replay + exhaustive boundary linkage, or (B) oracle-compare of record — every committed value bit-for-bit vs the same run's independent Match observation, 0 differing, 0 unattributed, complete — or a cited qualification record + row-own linkage; M-0111 forms are diagnostics):")
        out += c2
    # rev-pm M-0268 (a): the eager and the compiled declaration of one model share the Program digest (46e668e6… / workload 8607ee96…);
    # a compiled row's Build identity lives in applicability.execution (+ compilation) — shown beside the digest, or the cells are indistinguishable
    bid = []
    by_digest: dict[str, list[tuple[dict, str]]] = {}
    for r in R.sorted_rows(recs):
        if glyph(recs, r, "build") not in (GLYPH_PASS, GLYPH_PART):
            continue
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        b = a.get("build") if isinstance(a.get("build"), dict) else {}
        dg = str(b.get("global_program_digest") or b.get("program_digest") or "")
        ex = str((r.get("config") or {}).get("execution") or r.get("row_id", "").split("__")[-1])
        if dg:
            by_digest.setdefault(dg, []).append((r, ex))
    for r in R.sorted_rows(recs):
        if glyph(recs, r, "build") not in (GLYPH_PASS, GLYPH_PART):
            continue
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else {}
        b = a.get("build") if isinstance(a.get("build"), dict) else {}
        dg = str(b.get("global_program_digest") or b.get("program_digest") or "")
        ex = str((r.get("config") or {}).get("execution") or r.get("row_id", "").split("__")[-1])
        twins = {e for rr, e in by_digest.get(dg, []) if rr is not r and e != ex}
        if "compiled" not in ex and not twins:
            continue
        ap = b.get("applicability") if isinstance(b.get("applicability"), dict) else {}
        ap_ex = ap.get("execution") or ap.get("execution_label") or "not recorded"
        eng = ap.get("engine") if isinstance(ap.get("engine"), dict) else {}
        comp = eng.get("compilation") if isinstance(eng.get("compilation"), dict) else (ap.get("compilation") if isinstance(ap.get("compilation"), dict) else None)
        ctxt = (" compilation {" + ", ".join(f"{k}={comp[k]}" for k in ("mode", "compilation_mode", "cudagraph_mode", "backend", "cudagraph_capture_sizes") if k in comp) + "}") if comp else ""
        ee = eng.get("enforce_eager") if "enforce_eager" in eng else ap.get("enforce_eager")
        etxt = f", enforce_eager={ee}" if ee is not None else ""
        cr = (a.get("match") or {}).get("compiled_record") if isinstance(a.get("match"), dict) else None
        pin = f"; Match-record kernel pin {str(cr.get('kernel_pin_match'))[:12]}…" if isinstance(cr, dict) and cr.get("kernel_pin_match") else ("; Match-record kernel pin: none (placeholder null)" if isinstance(cr, dict) else "")
        tw = f" — shares the Program digest with the `{'`, `'.join(sorted(twins))}` row(s): the declaration differs only in applicability" if twins else ""
        bid.append(f"- `{short_row(r)}`: Program {dg[:16] or '?'}… — applicability.execution **{ap_ex}**{ctxt}{etxt}{pin}{tw}")
    if bid:
        out.append("")
        out.append("Build identity beside the digest (rev-pm M-0268 (a): the eager and compiled declarations of one model share the Program digest; the compiled Build's identity is its applicability label + compilation facts):")
        out += bid
    notes = sidecar(recs, ROW_NOTES_FILE)
    ns = [(rid, v) for rid, v in notes.items() if isinstance(v, str) and v and rid in recs.rows]
    if ns:
        out.append("")
        out.append("Row notes (report lane; caveats on displayed cells, from the records — not record edits):")
        out += [f"- `{short_row(recs.rows[rid])}`: {v[:400]}" for rid, v in ns]
    return out


GLYPH = {"PASS": "✅", "DONE": "✅", "PARTIAL": "◐", "FAIL": "❌", "-": "—", "INFEASIBLE": "⊘"}


def matrix_table(recs: R.Records) -> str:
    lines = [MATRIX_HEADER]
    for row in R.sorted_rows(recs):
        lines.append("| " + " | ".join(_esc(c) for c in matrix_row(recs, row)) + " |")
    if len(lines) == 1:
        lines.append("| _(no row records yet)_ |" + " |" * 14)
    lines += provenance_notes(recs)
    if R.CAMPAIGN == "r14":
        notes = commit_fail_notes(recs)
        if notes:
            lines.append("")
            lines.append("Commit ❌ (executed coverage failure, visibly different from ◐ not-revalidated; COORD 18:09Z):")
            lines += notes
    if any(lane_only(r) for r in recs.rows.values()):
        lines.append(f"\n{LANE_ONLY_MARK} Commit passed on a lane-only revision not yet in an announced release tag (BRIEF §8) — R14 evidence with the tree stated, not the same as a release-tagged ✅ (COORD 18:24Z):")
        for r in R.sorted_rows(recs):
            if lane_only(r):
                a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
                lines.append(f"- `{short_row(r)}`: revalidated on lane tree {str(a.get('revision') or '?')[:9]} ({a.get('lane') or '?'}; {str(a.get('release_tag') or 'no release tag')[:60]}), next release canary pending")
    lines += board_rows_table(recs)
    if any(R.stage_status(r, "match") == "FAIL" and R.match_unfinished_fold_cost(recs, r) for r in recs.rows.values()):
        lines.append(f"\n{GLYPH_PART} fold-cost = Match unfinished: the CPU fold / comparator hit the harness time limit (C-FOLD-COST, infrastructure); capture, control and resolution were clean; criteria unchanged (COORD 07:30Z).")
    return "\n".join(lines)


def board_rows_table(recs: R.Records) -> list[str]:
    """[COORD DECISION M-0522 (2)] rows OUTSIDE the 75-row panel that the coordinator ordered rendered: same columns, same finalizer
    derivation, beneath the table, excluded from every count; the ordering post is cited per row.  Not panel growth (a panel row is
    added only by Daniel's 7-field note, DECISIONS-2)."""
    rows = R.board_rows(recs)
    if not rows:
        return []
    out = ["", "Rows outside the 75-row panel rendered by board order (not counted; the panel is unchanged — COORD M-0522 (2)):", "", MATRIX_HEADER]
    for row in rows:
        out.append("| " + " | ".join(_esc(c) for c in matrix_row(recs, row)) + " |")
    for row in rows:
        b = row.get("_board_order") or {}
        a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else {}
        out.append(f"- `{short_row(row)}`: ordered by {b.get('ordered_by')} ({str(b.get('since') or '')[11:16]}Z){(' — ' + str(b.get('note'))[:200]) if b.get('note') else ''}; displayed attempt "
                   f"{a.get('release_tag') or 'lane-only'} {str(a.get('revision') or '?')[:9]} ({a.get('lane') or '?'}, {a.get('evidence_dir') or '?'}).")
        for st in ("build", "match", "commit", "bench"):
            d = a.get(st) if isinstance(a.get(st), dict) else {}
            rs = [str(x) for x in (d.get("reasons") or []) if x]
            if rs and glyph(recs, row, st) != GLYPH_PASS:
                out.append(f"  - {st}: {'; '.join(rs)[:420]}")
    return out


def partition_counts(recs: R.Records) -> dict:
    c = {"complete": 0, "partial": 0, "blocked": 0, "families_complete": set(), "complete_rows": [], "partial_rows": [], "blocked_rows": []}
    for row in R.sorted_rows(recs):
        p = partition(recs, row)
        c[p] += 1
        c[p + "_rows"].append(row.get("row_id"))
        if p == "complete":
            c["families_complete"].add(R.model_of(row))
    return c


# ----------------------------------------------------------------------------------------------------------------
# counts


def counts(recs: R.Records) -> dict:
    rows = list(recs.rows.values())
    c = {"rows": len(rows), "queued": 0, "not_run": 0, "attempted": 0,
         "build": 0, "build_part": 0, "build_fail": 0, "match": 0, "match_part": 0, "match_fail": 0,
         "commit": 0, "commit_part": 0, "commit_fail": 0, "bench": 0, "bench_part": 0,
         "zero_code": 0, "fixes": len(R.engineering_fix_ids(recs)), "infra_fixes": len(recs.fixes) - len(R.engineering_fix_ids(recs)),
         "failures": len(recs.failures), "green_deploy_only": 0, "red_deployment": 0, "red_gap": 0, "red_infra": 0, "red_unnamed": 0, "green_mech": 0, "in_flight": 0}
    clusters = R.cluster_failures(recs)
    for row in rows:
        st = row.get("status")
        if st == "QUEUED":
            c["queued"] += 1
            continue
        if st == "NOT_RUN":
            c["not_run"] += 1
            continue
        c["attempted"] += 1
        for s in ("build", "match", "commit", "bench"):
            g = glyph(recs, row, s)
            if g == GLYPH_PASS:
                c[s] += 1
            elif g == GLYPH_PART:
                c[f"{s}_part"] += 1
            elif g == GLYPH_FAIL and s != "bench":
                c[f"{s}_fail"] += 1
        if partition(recs, row) == "complete" and R.zero_code_changes(recs, row):  # R13: count completed configurations only
            c["zero_code"] += 1
        bc = R.row_blocking_cluster(recs, row, clusters)
        if glyph(recs, row, "commit") == GLYPH_PASS:
            c["green_deploy_only" if R.deployment_only(recs, row) else "green_mech"] += 1
        elif bc is not None:
            c["red_" + (R.bucket_of(bc) if bc in R.CLUSTER_MECHANISM else "unnamed")] += 1
        elif st not in ("QUEUED", "NOT_RUN"):
            c["in_flight"] += 1
    c["invalidated"] = len(set(recs.invalidated))
    return c


def _n(c: dict, k: str) -> str:
    p = c.get(f"{k}_part", 0)
    return f"{c[k]}" + (f" (+{p} ◐)" if p else "")


# ----------------------------------------------------------------------------------------------------------------
# state (for "since last update")


def load_state(root: str) -> dict:
    p = os.path.join(root, STATE_REL)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            pass
    return {"last_render_at": None, "row_status": {}, "failure_ids": [], "fix_ids": []}


def save_state(root: str, recs: R.Records, now: str) -> None:
    p = os.path.join(root, STATE_REL)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    st = {"last_render_at": now,
          "row_status": {rid: row.get("status") for rid, row in sorted(recs.rows.items())},
          "failure_ids": sorted(recs.failures), "fix_ids": sorted(recs.fixes)}
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)
        fh.write("\n")


# ----------------------------------------------------------------------------------------------------------------
# blockers / evidence


def evidence_lines(recs: R.Records) -> list[str]:
    out = []
    for rid in sorted(recs.missing_evidence):
        fields = ", ".join(f"{s}.{f}" if not f.startswith("config.") else f for s, f in recs.missing_evidence[rid])
        out.append(f"missing evidence {rid} {fields}")
    return out


def overhead_evidence_gaps(recs: R.Records) -> list[str]:
    """§6 cells whose record lacks bench.artifact / bench.cmd / config.verity_commit."""
    out = []
    for row in R.sorted_rows(recs):
        if R.stage_status(row, "bench") != "DONE":
            continue
        n = bench_numbers(row)
        gaps = [k for k, v in (("bench.artifact", n["artifact"]), ("bench.cmd", n["cmd"]),
                               ("config.verity_commit", (row.get("config") or {}).get("verity_commit"))) if not v]
        if gaps:
            out.append(f"{row.get('row_id')}: {', '.join(gaps)}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# canary gate (DECISION-0355 addendum §3): dA / hw report "CANARY n GREEN" in their lane STATUS; coordinator writes CANARY.md


def canary_line(recs: R.Records) -> str:
    root = recs.root
    found = []
    for lane in ("dA", "hw"):
        p = os.path.join(root, "lanes", f"{lane}-STATUS.md")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                seen_n = set()
                for m in re.finditer(r"CANARY\s*(\d)[^\n|]{0,80}", fh.read()):
                    if m.group(1) not in seen_n:
                        seen_n.add(m.group(1))
                        found.append(f"{lane}: {' '.join(m.group(0).split())[:90]}")
    cm = os.path.join(os.path.dirname(root.rstrip("/")), "r12", "CANARY.md")
    gate = "CANARY.md absent (fan-out not yet authorised)"
    if os.path.exists(cm):
        with open(cm, "r", encoding="utf-8", errors="replace") as fh:
            gate = "CANARY.md: " + " ".join(fh.read().split())[:120]
    return "canary gate — " + gate + (" — " + "; ".join(found) if found else " — no CANARY lines in dA/hw STATUS yet")


def review_rerun(recs: R.Records, row_id: str) -> tuple[str, str] | None:
    """Reviewer's fixed-tree rerun of a row (review/rerun-fixed-<row>.json) -> (pod, overhead text), if present."""
    p = os.path.join(recs.root, "review", f"rerun-fixed-{row_id}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    txt = json.dumps(d)
    pod = re.search(r"cp-rev", txt)
    m = re.search(r'"overhead_rel"\s*:\s*([0-9.]+)', txt) or re.search(r'"overhead"\s*:\s*\{[^}]*"mean"\s*:\s*([0-9.]+)', txt)
    if not m:
        m2 = re.search(r'"overhead"\s*:\s*"\+([0-9.]+)\s*%', txt)
        if m2:
            return ("cp-rev" if pod else "review pod", f"+{float(m2.group(1)):.1f}%")
    if not m:
        return None
    return (("cp-rev" if pod else "review pod"), f"+{float(m.group(1)) * 100:.1f}%")


# ----------------------------------------------------------------------------------------------------------------
# STATUS.md


def status_md(recs: R.Records, state: dict, now: str, extra_blockers: list[str]) -> tuple[str, dict]:
    c = counts(recs)
    prev = state.get("row_status", {})
    changed = []
    for rid, row in sorted(recs.rows.items()):
        st = row.get("status")
        if rid not in prev:
            changed.append(f"{rid}: NEW {st}")
        elif prev[rid] != st:
            changed.append(f"{rid}: {prev[rid]} → {st}")
    seen_f = set(state.get("failure_ids", []))
    new_clusters = []
    for cl in R.cluster_failures(recs):
        new_ids = [f for f in cl["failure_ids"] if f not in seen_f]
        if new_ids:
            new_clusters.append(f"{cl['id']} ({'/'.join(cl['stages'])}; {cl['class']}) — {len(cl['row_ids'])} rows — "
                                f"new: {', '.join(new_ids)} (cluster: {', '.join(cl['failure_ids'])})")
    if recs.failures:
        new_clusters.append("all clusters: " + ", ".join(f"{cl['id']} {len(cl['row_ids'])}" for cl in R.cluster_failures(recs)))
    fixes = []
    rel = releases(recs)
    fixes.append("releases: " + (", ".join(f"{t} [{', '.join(f) or '-'}]" for t, f, _ in rel) if rel else "none announced yet (RELEASES.md has no RELEASED line)"))
    for xid, x in sorted(recs.fixes.items()):
        fixes.append(f"{xid} — {x.get('mechanism', '?')} ({x.get('owner', '?')}) — rerun {len(x.get('rows_rerun') or [])}/"
                     f"{len(x.get('rows_expected') or [])} expected, newly green {len(x.get('rows_newly_green') or [])}, "
                     f"regressions {len(x.get('regressions') or [])}")
    blockers = list(extra_blockers)
    for rid in sorted(set(recs.invalidated)):
        blockers.append(f"invalidated: fail-open harness — {rid} (Commit PASS pre-X-03; rendered ◐ until re-run fail-closed)")
    blockers += evidence_lines(recs)
    for g in overhead_evidence_gaps(recs):
        blockers.append(f"overhead cell without artifact/cmd/verity_commit: {g}")
    viol = recs.errors()
    if viol:
        blockers.append(f"{len(viol)} record violation(s) — see VALIDATION.md; first: {viol[0]}")

    def block(items: list[str]) -> str:
        return "\n".join(items) if items else "(none)"

    pc = partition_counts(recs)
    pool_line = "(dispatcher status unavailable)"
    if os.path.exists("/tmp/r13-dispatch-status.txt"):
        with open("/tmp/r13-dispatch-status.txt", encoding="utf-8") as fh:
            pl = [l.strip() for l in fh if l.strip()]
        pool_line = pl[0] + (" — " + "; ".join(pl[1:]) if len(pl) > 1 else "")
    ready_line = "(DEPS.json missing)"
    dp = os.path.join(recs.root, "DEPS.json")
    if os.path.exists(dp):
        with open(dp, encoding="utf-8") as fh:
            deps = json.load(fh)
        rows_d = deps.get("rows", {})
        n_ready = sum(1 for d in rows_d.values() if d.get("ready"))
        n_part = sum(1 for d in rows_d.values() if not d.get("ready") and d.get("runnable_through") in ("build", "match"))
        top: dict[str, int] = {}
        for d in rows_d.values():
            for cl in d.get("remaining", []):
                top[cl] = top.get(cl, 0) + 1
        ready_line = (f"{n_ready} rows fully unblocked on a release, {n_part} Build/Match-runnable while a later stage is blocked; "
                      f"top remaining blockers: " + ", ".join(f"{k} {v}" for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:5]) + " (READY.md)")

    text = (
        f"Time: {now}\n"
        f"Attempted: {c['attempted']} (queued {c['queued']}, not run {c['not_run']}, records {c['rows']})\n"
        f"Build green: {_n(c, 'build')}\n"
        f"Match green: {_n(c, 'match')}\n"
        f"Commit green: {_n(c, 'commit')}\n"
        f"Benchmarked: {_n(c, 'bench')}\n"
        f"Partition (distinct configurations): fully completed {pc['complete']} / partially completed {pc['partial']} / blocked or not run {pc['blocked']}; "
        f"families completed {len(pc['families_complete'])}\n"
        f"Pool: {pool_line}\n"
        f"Ready queue: {ready_line}\n"
        f"\nRows completed since last update ({state.get('last_render_at') or 'T0'}):\n{block(changed)}\n"
        f"\nNew failure clusters:\n{block(new_clusters)}\n"
        f"\nGeneric fixes in progress:\n{block(fixes)}\n"
        f"\nCurrent blockers:\n{block(blockers)}\n"
    )
    return text, c


# ----------------------------------------------------------------------------------------------------------------
# VALIDATION.md


def validation_md(recs: R.Records, now: str) -> str:
    lines = [f"# Record validation — {now}", "",
             "Generated by `verity_capture.sweep.render` from `rows/ failures/ fixes/`. Producing lanes fix the record; "
             "the report owner never edits records.", ""]
    for title, viol in (("Schema / consistency violations (blocking)", recs.errors()),
                        ("Advisories (optional SCHEMA §3 fields absent)", recs.advisories())):
        lines.append(f"## {title} ({len(viol)})")
        lines.append("")
        if viol:
            lines.append("| file | field | what is wrong |")
            lines.append("|---|---|---|")
            for v in viol:
                lines.append(f"| `{_esc(v.file)}` | `{_esc(v.field)}` | {_esc(v.message)} |")
        else:
            lines.append("(none)")
        lines.append("")
    ev = evidence_lines(recs)
    lines.append(f"## Missing evidence on PASS/DONE stages → rendered ◐ ({len(ev)})")
    lines.append("")
    lines += [f"- {e}" for e in ev] or ["(none)"]
    lines.append("")
    og = overhead_evidence_gaps(recs)
    lines.append(f"## §6 overhead cells lacking artifact / cmd / verity_commit ({len(og)})")
    lines.append("")
    lines += [f"- {g}" for g in og] or ["(none)"]
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# FINAL-REPORT.md blocks


def block_summary(recs: R.Records, c: dict, now: str) -> str:
    eng = R.engineering_fix_ids(recs)
    infra = sorted(set(recs.fixes) - set(eng))
    fix_ids = (", ".join(eng) or "none") + (f"; environment fix not counted: {', '.join(infra)}" if infra else "")
    per_fix = "; ".join(f"{x} → {len(recs.fixes[x].get('rows_newly_green') or [])} rows newly green"
                        for x in eng) or "n/a"
    clusters = R.cluster_failures(recs)
    dep = [cl for cl in clusters if R.bucket_of(cl["id"]) == "deployment"]
    gaps = [cl for cl in clusters if cl["id"] in R.GAP_RANK]
    gaps.sort(key=lambda cl: R.GAP_RANK.index(cl["id"]))
    statement = ("> The frozen R11 system did not reproduce a green configuration on fresh sweep infrastructure because several shared "
                 "deployment/profile assumptions blocked all rows. Once classified, the remaining model/configuration failures cluster "
                 "into a small number of generic mechanisms.")
    buckets = (
        f"\n\n**Fresh-deployment blockers** (shared deployment/profile assumptions; fixes X-00..X-03): "
        + ", ".join(f"{R.DEPLOY_LABEL.get(cl['id'], cl['id'])} [{cl['id']}, {len(cl['row_ids'])} rows → {R.DEPLOYMENT_CLUSTERS[cl['id']]}]" for cl in dep)
        + (f"; fail-open commitment harness [X-03, {c['invalidated']} Commit PASS record(s) invalidated → ◐]" if c["invalidated"] else "; fail-open commitment harness [X-03]")
        + f"\n\n**Verity generalization gaps** (genuine mechanisms; order = DECISION-0355 addendum §5 leverage ranking, which counts rows a fix unblocks *including as a prerequisite for later stages* — e.g. the FA2 head-dim tap gates Commit for every non-Llama family — not the current failure-row count shown in brackets): "
        + " > ".join(f"{R.GAP_LABEL.get(cl['id'], cl['id'])} [{cl['id']}, {len(cl['row_ids'])}]" for cl in gaps)
        + f"\n\n**Key number:** rows Commit-green after fixing shared deployment/profile assumptions alone (X-00..X-03, no mechanism fix): "
        f"**{c['green_deploy_only']}**; rows still red on a deployment blocker (awaiting fail-closed rerun): **{c['red_deployment']}**; "
        f"rows red on a genuine Verity mechanism: **{c['red_gap']}**" + (f"; red on an unclustered failure: {c['red_unnamed']}" if c["red_unnamed"] else "") + (f"; red on a harness time limit only: {c['red_infra']}" if c["red_infra"] else "")
        + f"; Commit-green needing a mechanism fix: {c['green_mech']}; in flight / no verdict yet: {c['in_flight']} (partition of the {c['attempted']} attempted rows)."
    )
    return (
        statement + buckets + "\n\n"
        f"- configurations attempted: **{c['attempted']}** (records {c['rows']}; queued {c['queued']}; NOT RUN {c['not_run']})\n"
        f"- Build green: **{_n(c, 'build')}** (❌ {c['build_fail']})\n"
        f"- Match green: **{_n(c, 'match')}** (❌ {c['match_fail']})\n"
        f"- Commit green: **{_n(c, 'commit')}** (❌ {c['commit_fail']})\n"
        f"- benchmarked: **{_n(c, 'bench')}**\n"
        f"- configurations requiring zero code changes: **{c['zero_code']}** (Commit-green rows whose `code_changes_required` names no Verity mechanism fix; environment/harness-only X-00/X-03 do not count)\n"
        f"- generic fixes made: **{c['fixes']}** ({fix_ids}); additional rows green per fix: {per_fix}\n"
        f"- failure records: {c['failures']} in {len(R.cluster_failures(recs))} cluster(s): "
        + ", ".join(f"{cl['id']} ({len(cl['row_ids'])})" for cl in R.cluster_failures(recs)) + "\n"
        f"\n_Counts generated {now} from `out/gen/sweep/rows|failures|fixes`; ◐ = stage PASS whose record lacks required evidence "
        f"(see VALIDATION.md)._"
    )


def block_summary_r13(recs: R.Records, c: dict, now: str) -> str:
    eng = R.engineering_fix_ids(recs)
    infra = sorted(set(recs.fixes) - set(eng))
    per_fix = "; ".join(f"{x} → {len(recs.fixes[x].get('rows_newly_green') or [])} newly green / {len(recs.fixes[x].get('rows_expected') or [])} expected"
                        for x in eng) or "n/a"
    # remaining-blocker leverage from DEPS.json (rows still waiting per unresolved cluster)
    lev = ""
    dp = os.path.join(recs.root, "DEPS.json")
    if os.path.exists(dp):
        with open(dp, encoding="utf-8") as fh:
            deps = json.load(fh)
        top: dict[str, int] = {}
        for d in deps.get("rows", {}).values():
            for cl in d.get("remaining", []):
                top[cl] = top.get(cl, 0) + 1
        lev = " > ".join(f"{R.GAP_LABEL.get(k, k)} [{k}, {v} rows]" for k, v in sorted(top.items(), key=lambda kv: -kv[1]))
    rel = releases(recs)
    lane_only = sum(1 for r in recs.rows.values() if lane_only_row(r))
    return (
        f"- configurations attempted: **{c['attempted']}** (records {c['rows']}; queued {c['queued']}; NOT RUN {c['not_run']})\n"
        f"- Build green: **{_n(c, 'build')}** (❌ {c['build_fail']})\n"
        f"- Match green: **{_n(c, 'match')}** (❌ {c['match_fail']})\n"
        f"- Commit green: **{_n(c, 'commit')}** (❌ {c['commit_fail']})\n"
        f"- benchmarked: **{_n(c, 'bench')}**\n"
        f"- completed on unchanged Verity code (`code_changes_required == []`; infra fixes X-00/X-03/REL-* do not count, R-19): **{c['zero_code']}**\n"
        f"- generic (Verity-mechanism) fixes with a record: **{c['fixes']}** ({', '.join(eng) or 'none'}); rows newly green per fix: {per_fix}\n"
        f"- infrastructure / harness fixes (not per-configuration engineering): {', '.join(infra) or 'none'}\n"
        f"- releases announced: {', '.join(f'`{t}` [{', '.join(f) or '-'}]' for t, f, _ in rel) or 'none yet'}; Commit-green rows on a lane-only revision (†): {lane_only}\n"
        f"- failure records: {c['failures']} in {len(R.cluster_failures(recs))} cluster(s): "
        + ", ".join(f"{cl['id']} ({len(cl['row_ids'])})" for cl in R.cluster_failures(recs)) + "\n"
        + (f"- remaining blockers by leverage (rows still waiting on each unresolved cluster; a row counts under every blocker it still has): {lev}\n" if lev else "")
        + f"\n_Counts generated {now} from `out/gen/sweep/rows|failures|fixes`; ◐ = stage PASS whose record lacks required evidence (see VALIDATION.md); "
        f"a rerun is a new `attempts[]` entry and the row shows its latest attempt._"
    )


def lane_only_row(row: dict) -> bool:
    return lane_only(row)


def block_generalized(recs: R.Records) -> str:
    out, dep = [], []
    for row in R.sorted_rows(recs):
        if glyph(recs, row, "commit") != GLYPH_PASS:
            continue
        vc = (row.get("config") or {}).get("verity_commit") or "?"
        line = (f"- `{row.get('row_id')}` — Commit {glyph(recs, row, 'commit')}, overhead {overhead_cell(recs, row)}, "
                f"`code_changes_required` {row.get('code_changes_required')}, verity_commit `{str(vc)[:10]}`")
        if R.zero_code_changes(recs, row):
            out.append(line)
        elif R.deployment_only(recs, row):
            dep.append(line)
    head = (f"{len(out)} configuration(s) became Commit-green with `code_changes_required == []`:\n\n" + "\n".join(out)
            if out else "**None.** No configuration is Commit-green on the frozen code (92ae92a63) with `code_changes_required == []`: "
            "every path was blocked by the shared deployment/profile assumptions listed in §1 (see §7.0).")
    if dep:
        head += (f"\n\n{len(dep)} configuration(s) became Commit-green after the shared deployment/profile fixes alone "
                 "(X-00..X-03; no per-model or per-configuration Verity change):\n\n" + "\n".join(dep))
    return head


def block_fixes(recs: R.Records) -> str:
    if not recs.fixes:
        return "_(no fix records yet — first 40 min is classification only; ≤ 2 coordinator-approved generic fixes after T+55)_"
    parts = []
    eng = R.engineering_fix_ids(recs)
    infra = sorted(set(recs.fixes) - set(eng))
    if eng:
        parts.append("### Generic fixes (shared deployment/profile assumptions turned into target facts)")
    for xid in eng + (["__infra__"] if infra else []) + infra:
        if xid == "__infra__":
            parts.append("### Environment / harness fixes (not per-configuration engineering)\n\n_X-00 (`hidden_gpu` missing from the pod "
                         "bootstrap) and X-03 (fail-closed Commit in the sweep harness) change no Verity mechanism; rows unblocked by them alone "
                         "keep `code_changes_required` free of Verity fixes and count as zero code changes._")
            continue
        x = recs.fixes[xid]
        exp = x.get("rows_expected") or []
        rerun = x.get("rows_rerun") or []
        green = x.get("rows_newly_green") or []
        regr = x.get("regressions") or []
        still = sorted(set(exp) - set(green))
        fmt_r = lambda r: f"`{r}`" if r in recs.rows else f"_(pattern: {r})_"  # noqa: E731
        parts.append(
            f"#### {xid} — {x.get('mechanism', '?')}\n\n"
            f"- **Mechanism:** {x.get('mechanism', '?')}\n"
            f"- **Why it failed:** {x.get('why_it_failed', '?')}\n"
            f"- **Generic change:** {x.get('generic_change', '?')} (commit `{str(x.get('commit', '?'))[:10]}`, owner {x.get('owner', '?')})\n"
            f"- **Rows fixed ({len(green)}):** " + (", ".join(fmt_r(r) for r in green) or "none yet") + "\n"
            f"- **Rows rerun ({len(rerun)}/{len(exp)} expected):** " + (", ".join(fmt_r(r) for r in rerun) or "none yet") + "\n"
            f"- **Rows still blocked ({len(still)}):** " + (", ".join(fmt_r(r) for r in still) or "none") + "\n"
            f"- **Regressions:** " + (", ".join(fmt_r(r) for r in regr) or "none")
            + (f"\n- **Portability test:** {x['portability_test']}" if x.get("portability_test") else
               ("\n- **Portability test (same TargetProfile → builder and eager resolver select the same GEMM Definition):** _not yet recorded by dA (`portability_test` field)_" if xid == "X-01" else ""))
        )
    var = block_variability(recs)
    if var:
        parts.append(var)
    return "\n\n".join(parts)


def _attempt_overhead(a: dict):
    b = a.get("bench") if isinstance(a.get("bench"), dict) else {}
    if b.get("status") not in ("DONE", "PASS"):
        return None
    n = bench_numbers({"bench": b})
    return n["overhead_rel"] if isinstance(n["overhead_rel"], (int, float)) else None


def _release_n(a: dict):
    m = re.fullmatch(r"r13-rel(\d+)", str(a.get("release_tag") or ""))
    return int(m.group(1)) if m else None


def _retention(a: dict) -> str:
    """Committer configuration of the measurement: retention policy + collector (whole-block vs P0 windowed).  rev (11:35Z):
    REL-11 (r13-rel13) makes row_pod's Commit default to the windowed collector, which is not the same measurement as the
    whole-block committer — cells must say which."""
    c = a.get("commit") if isinstance(a.get("commit"), dict) else {}
    if a.get("finalizer") and c.get("collector_variant"):
        # finalizer-derived: the binding record / coverage.json of the committed run is authoritative
        cs = c.get("collector_settings") if isinstance(c.get("collector_settings"), dict) else {}
        win = f" {cs.get('window_mb')} MiB × {cs.get('window_slots')}" if cs.get("window_mb") else ""
        return f"{c.get('retention') or '?'}, {'windowed collector' + win if c['collector_variant'] == 'windowed' else 'whole-block'}"
    txt = " ".join(str(x) for x in (c.get("retain"), c.get("retention"), c.get("cmd"), a.get("cmd"), json.dumps(a.get("env") or {}), c.get("collector"), a.get("notes")))
    m = re.search(r"--retain\s+(\w+)|VERITY_RETAIN=(\w+)|retain\s*[=:]\s*(host|device|none|sample)", txt)
    ret = (m.group(1) or m.group(2) or m.group(3)) if m else "device"
    n = _release_n(a)
    windowed = bool(re.search(r"VERITY_WINDOW_(MB|SLOTS)|--window-(mb|slots)|windowed", txt, re.I)) or (n is not None and n >= 13 and "whole-block" not in txt.lower())
    if windowed:
        rel11_default = n is not None and n >= 13 and not re.search(r"VERITY_WINDOW|--window", txt)
        if rel11_default and not m:
            ret = "host"  # REL-11 default: --window-mb 256 --window-slots 8 --retain host --open-after-release
        return f"{ret}, windowed collector" + (" (REL-11 default)" if rel11_default else "")
    return f"{ret}, whole-block"


def block_variability(recs: R.Records) -> str:
    """COORD 12:00Z ruling (b): a configuration with several completed attempts shows the release-tree attempt in the matrix;
    the spread across attempts/trees/pods is listed here, never averaged."""
    lines = []
    for r in R.sorted_rows(recs):
        atts = [a for a in (r.get("attempts") or []) if isinstance(a, dict)]
        pts = []
        for a in atts:
            if R.attempt_is_v2_layout(a):
                continue
            o = _attempt_overhead(a)
            if o is None:
                continue
            tree = a.get("release_tag") if re.fullmatch(r"r13-rel\d+", str(a.get("release_tag") or "")) else f"lane-only {str(a.get('revision') or '?')[:9]}"
            pts.append(f"{o * 100:+.1f}% ({tree}, {a.get('pod') or a.get('lane') or '?'}, retain={_retention(a)}{', reproduction' if a.get('role') == 'reproduction' else ''})")
        if len(pts) >= 2:
            lines.append(f"- `{r.get('row_id')}`: " + " · ".join(pts))
    if not lines:
        return ""
    return ("### Variability across attempts (not averaged)\n\n"
            "Rows with more than one completed attempt (release tree, lane tree, second pool pod): the matrix shows the release-tree attempt "
            "(COORD 12:00Z ruling b); the spread is the measurement's variability across trees / pods / retention policies.\n\n" + "\n".join(lines))


def _excerpt(s: str, lines: int = 4, chars: int = 400) -> str:
    s = "\n".join(s.strip().splitlines()[:lines])
    return s if len(s) <= chars else s[:chars] + " …"


def block_clusters(recs: R.Records) -> str:
    cls = R.cluster_failures(recs)
    if not cls:
        return "_(no failure records yet)_"
    unresolved, resolved = [], []
    for cl in cls:
        fs = [recs.failures[f] for f in cl["failure_ids"]]
        if all(R.failure_resolved(recs, f) for f in fs):
            resolved.append(cl)
        else:
            unresolved.append(cl)
    out = [f"{len(unresolved)} unresolved cluster(s), {len(resolved)} resolved, {len(recs.failures)} failure record(s).", "",
           "| bucket | cluster | rows | stage | fix / smallest concrete blocker |", "|---|---|---:|---|---|"]
    for cl in sorted(cls, key=lambda cl: ({"deployment": 0, "gap": 1, "infra": 2}[R.bucket_of(cl["id"])],
                                          R.GAP_RANK.index(cl["id"]) if cl["id"] in R.GAP_RANK else 99, cl["id"])):
        if R.bucket_of(cl["id"]) == "deployment":
            b, fix = "fresh-deployment blocker", f"{R.DEPLOYMENT_CLUSTERS[cl['id']]} — {R.DEPLOY_LABEL.get(cl['id'], '')}"
        elif R.bucket_of(cl["id"]) == "infra":
            b, fix = "harness / infrastructure (not a Verity mechanism)", R.INFRA_CLUSTERS[cl["id"]]
        else:
            b, fix = "Verity generalization gap", f"{R.GAP_LABEL.get(cl['id'], '')}: {cl['mechanism']}"
        st = "resolved" if cl in resolved else "open"
        out.append(f"| {b} | **{cl['id']}** ({st}) | {len(cl['row_ids'])} | {'/'.join(cl['stages'])} | {_esc(fix)} |")
    out.append("| fresh-deployment blocker | fail-open commitment harness | " + str(len(set(recs.invalidated))) + " | commit | X-03 — Commit PASS records produced before X-03 are rendered ◐ (`invalidated: fail-open harness`) until re-run |")
    for cl in unresolved:
        open_rows = [f.get("row_id") for f in (recs.failures[x] for x in cl["failure_ids"]) if not R.failure_resolved(recs, f)]
        out.append(
            f"\n### {cl['id']} — {cl['class']} — {', '.join(cl['failure_ids'])}\n\n"
            f"- **Common cause:** {cl['mechanism']}\n"
            f"- **Affected rows ({len(cl['row_ids'])}):** " + ", ".join(f"`{r}`" for r in cl["row_ids"]) + "\n"
            f"- **Still red:** " + (", ".join(f"`{r}`" for r in sorted(set(open_rows))) or "none") + "\n"
            f"- **Stage:** {'/'.join(cl['stages'])}\n"
            + (f"- **`same_as` links into other clusters:** {', '.join(cl['cross_links'])}\n" if cl["cross_links"] else "")
            + f"- **First error ({cl['failure_ids'][0]}, commit `{cl['commit'][:10]}`):**\n\n~~~text\n{_excerpt(cl['first_error'])}\n~~~\n"
            f"- **Artifact:** `{cl['artifact']}`\n"
            f"- **Best current diagnosis / next experiment:** "
            + (str(recs.failures[cl['failure_ids'][0]].get("diagnosis") or "_(coordinator, from the failure record `diagnosis` field)_"))
        )
    if resolved:
        out.append("\n### Resolved (row now passes the failing stage)\n")
        for cl in resolved:
            out.append(f"- {cl['id']} ({cl['class']}) — {', '.join(cl['failure_ids'])} — rows " + ", ".join(f"`{r}`" for r in cl["row_ids"]))
    return "\n".join(out)


def block_overhead(recs: R.Records) -> str:
    rows = [r for r in R.sorted_rows(recs) if R.stage_status(r, "bench") == "DONE" and r.get("status") not in ("QUEUED", "NOT_RUN")]
    if not rows:
        return "_(no benchmarked rows yet)_"
    lines = ["| Configuration | Hardware | Control mean (s) | Treatment mean (s) | Relative overhead | Abs ms/decode-step | Rounds | Retention | Label |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    gaps = []
    for r in rows:
        n = bench_numbers(r)
        fmt = lambda v, f: (f % v) if isinstance(v, (int, float)) else "—"  # noqa: E731
        rel = f"{n['overhead_rel'] * 100:+.1f}%" if isinstance(n["overhead_rel"], (int, float)) else "—"
        ev_ok = bool(n["artifact"]) and bool(n["cmd"]) and bool((r.get("config") or {}).get("verity_commit"))
        label = "sweep measurement" + ("" if ev_ok else " ◐ evidence missing")
        rr = review_rerun(recs, r.get("row_id"))
        if rr:
            label += f" (independent rerun on {rr[0]}: {rr[1]})"
        att = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
        for a in (r.get("_repro_attempts") or []):
            b2 = a.get("bench") if isinstance(a.get("bench"), dict) else {}
            n2 = bench_numbers({"bench": b2}) if b2 else {}
            if b2.get("status") in ("DONE", "PASS") and isinstance(n2.get("overhead_rel"), (int, float)):
                same = _retention(a) == _retention(att)
                label += (f" (pool reproduction on {a.get('pod')}, {a.get('release_tag') or a.get('revision')}: {n2['overhead_rel'] * 100:+.1f}%"
                          + ("" if same else f" — {_retention(a)}, a different committer configuration, not a spread") + ")")
            elif a.get("status") not in (None, "RUNNING", "BUILD_PASS", "MATCH_PASS", "COMMIT_PASS"):
                label += f" (pool reproduction on {a.get('pod')}: {a.get('status')})"
        if r.get("row_id") in recs.invalidated:
            label += " ◐ invalidated: fail-open harness"
        elif glyph(recs, r, "commit") == GLYPH_PART:
            label += " ◐ partial Commit evidence"
        if not ev_ok:
            gaps.append(r.get("row_id"))
        if att.get("post_freeze"):
            label += " (post-freeze completion)"
        lines.append(f"| `{r.get('row_id')}` | {display_hw(r)} | {fmt(n['control_mean_s'], '%.3f')} | {fmt(n['treatment_mean_s'], '%.3f')} | "
                     f"{rel} | {fmt(n['abs_ms'], '%.2f')} | {n['rounds'] if n['rounds'] is not None else '—'} | {_retention(att)} | {label} |")
    rels = [bench_numbers(r)["overhead_rel"] for r in rows]
    rels = [x for x in rels if isinstance(x, (int, float))]
    summ = ""
    if rels:
        summ = (f"\n\nDistribution over {len(rels)} benchmarked row(s): min {min(rels) * 100:+.1f}%, "
                f"median {sorted(rels)[len(rels) // 2] * 100:+.1f}%, max {max(rels) * 100:+.1f}%. ")
    ev = ("Every cell traces to `bench.artifact` + `bench.cmd` + `config.verity_commit` of its row record. Where the reviewer re-ran the same root on a second pod, the per-pod spread is quoted in the label — treat the overhead of a row as that spread, not the single sweep number."
          if not gaps else "Cells lacking artifact/cmd/verity_commit: " + ", ".join(f"`{g}`" for g in gaps) + " (see VALIDATION.md).")
    out = "\n".join(lines) + summ + ev + ("\n\nAll numbers are `sweep measurement`s (BRIEF §13.D: short matched interleaved control/treatment rounds), not publication-quality intervals; every cell above is a `chunk-leaf-v1` attempt (the layout of record, coordinator 08:35Z). "
           + "The **Retention** column names the committer configuration of the measurement (rev finding, 11:35Z; COORD 12:00Z ruling b): "
           "retention policy (`device` = raw copies retained on the GPU until release; `host` = pinned-host retention; `none`) and the collector "
           "(`whole-block` = the R12 committer; `windowed collector` = P0's bounded 256 MiB × 8 staging ring, the row_pod default from r13-rel13 / REL-11). "
           "Windowed/host cells are not the same measurement as whole-block/device cells (window waits and re-upload sit on the serving path; on the same "
           "Llama b1 1024/128 row rev measured +20.5 % whole-block vs +51.5 % windowed) and must not be compared across the column. "
           "B≥2 1024/128 cells produced by lane p0b (`b8`, `b16`) were measured with `max_num_batched_tokens=1024` (chunked prefill) + util 0.25 (F-rev-04) — a different engine configuration from the pool's un-chunked rows; pool cells use the manifest's 2048.")
    v2rows = [(r, a) for r in R.sorted_rows(recs) for a in (r.get("_v2_attempts") or []) if isinstance(a.get("bench"), dict) and a["bench"].get("status") in ("PASS", "DONE")]
    if v2rows:
        sub = ["\n\n**`chunk-leaf-v2` (experimental) — labelled sub-table, not headline cells.** Coordinator ruling 08:35Z: v2 changes every root, so it is a *layout* change; Daniel's brief fixes the current L1 layout for this campaign. A row is completed only on v1; v2 evidence is a recommendation for the morning (§9).",
               "", "| Configuration | Hardware | v1 overhead (headline) | v2 overhead (experimental) | v2 rounds | v2 evidence |", "|---|---|---:|---:|---:|---|"]
        for r, a in v2rows:
            n1 = bench_numbers(r)
            n2 = bench_numbers({"bench": a["bench"]})
            f = lambda v: f"{v * 100:+.1f}%" if isinstance(v, (int, float)) else "—"  # noqa: E731
            ev2 = "artifact" if (a["bench"].get("artifact") or a.get("evidence")) else "◐ evidence missing"
            if a["bench"].get("cmd") or a.get("cmd"):
                ev2 += "+cmd"
            sub.append(f"| `{r.get('row_id')}` | {display_hw(r)} | {f(n1['overhead_rel'])} | {f(n2['overhead_rel'])} | {n2['rounds'] if n2['rounds'] is not None else '—'} | {ev2} ({a.get('release_tag') or a.get('revision') or 'lane-only'}) |")
        out += "\n".join(sub)
    return out


SKELETON = """# Verity R12 — broad configuration sweep — FINAL REPORT

<!-- auto:begin header -->
<!-- auto:end -->

## 1. Executive summary

<!-- auto:begin summary -->
<!-- auto:end -->

## 2. Configuration support matrix

Glossary and status convention: `out/gen/r12/BRIEF.md` (Build / Match / Commit / Commitment overhead; ✅ demonstrated, ◐ partial evidence, ❌ tested and fails, — not tested). One row = one record in `out/gen/sweep/rows/`.

<!-- auto:begin matrix -->
<!-- auto:end -->

## 3. What generalized without changes

<!-- auto:begin generalized -->
<!-- auto:end -->

## 4. Generic fixes required

<!-- auto:begin fixes -->
<!-- auto:end -->

## 5. Remaining failure clusters

<!-- auto:begin clusters -->
<!-- auto:end -->

## 6. Commitment overhead

<!-- auto:begin overhead -->
<!-- auto:end -->

## 7. What we learned

_(report owner; written progressively from ~03:45Z, only from what the records show; every claim cites row ids)_

## 8. Recommended overnight campaign

_(empty until 04:45Z / T+120; candidate objectives then, one concrete plan at the 05:15Z freeze)_
"""


SKELETON_R13 = """# Verity R13 — overnight campaign: complete the configuration matrix — FINAL REPORT

<!-- auto:begin header -->
<!-- auto:end -->

## 1. Executive summary

_(report owner; hand-written prose below is updated from the records at every checkpoint — counts are auto-generated)_

<!-- auto:begin exec -->
<!-- auto:end -->

<!-- auto:begin summary -->
<!-- auto:end -->

## 2. Configuration support matrix

Fifteen canonical columns (BRIEF §1). ✅ demonstrated · ◐ partial evidence (stage PASS, required evidence missing; see `VALIDATION.md`) · ❌ tested, fails · — not tested / not reached. One row = one configuration = one record in `out/gen/sweep/rows/`; a rerun is a new `attempts[]` entry, the row shows the latest attempt.

<!-- auto:begin matrix -->
<!-- auto:end -->

## 3. Counts — fully completed / partially completed / blocked or not run

<!-- auto:begin partition -->
<!-- auto:end -->

## 4. Generic fixes and the configurations each unlocked

<!-- auto:begin fixes -->
<!-- auto:end -->

## 5. Remaining blockers, grouped by mechanism

<!-- auto:begin clusters -->
<!-- auto:end -->

## 6. Commitment overhead

<!-- auto:begin overhead -->
<!-- auto:end -->

## 7. Combined tested revision(s), reproduction, backup, live workers

<!-- auto:begin revisions -->
<!-- auto:end -->

## 8. Independent-review findings

_(report owner; from `out/gen/sweep/review/findings.md` (lane rev) — only findings with a record or evidence path are listed)_

_(none yet)_

## 9. Recommended next work

_(report owner; written at the 12:57Z freeze from §3–§5)_

_(pending)_
"""


def render_report(recs: R.Records, c: dict, now: str, path: str, campaign: str = "r13") -> None:
    blocks = {
        "header": (f"_Generated {now} by `python -m verity_capture.sweep.render` from {c['rows']} row, {c['failures']} failure, "
                   f"{c['fixes'] + c.get('infra_fixes', 0)} fix record(s). Auto blocks are regenerated on every render (≤10 min); "
                   f"the prose parts of §1, §8 and §9 are hand-written by the report owner from the records._"),
        "summary": block_summary_r13(recs, c, now) if campaign == "r13" else block_summary(recs, c, now),
        "matrix": matrix_table(recs),
        "generalized": block_generalized(recs),
        "fixes": block_fixes(recs),
        "clusters": block_clusters(recs),
        "overhead": block_overhead(recs),
        "partition": block_partition(recs),
        "exec": block_exec(recs, c, now),
        "revisions": block_revisions(recs, now),
    }
    if campaign == "r14":
        from . import render_r14 as R14
        blocks.update({"counts14": R14.block_counts14(recs, now), "caveats": R14.block_caveats(recs), "complete14": R14.block_complete14(recs),
                       "changed": R14.block_changed(recs), "sharedfix": R14.block_sharedfix(recs), "incomplete": R14.block_incomplete(recs),
                       "violations14": R14.block_violations14(recs)})
        blocks["header"] = (f"_Generated {now} by `python -m verity_capture.sweep.render --campaign r14` from {c['rows']} row, {c['failures']} failure, "
                            f"{c['fixes'] + c.get('infra_fixes', 0)} fix record(s) under `rules_version: {R.RULES_VERSION}` (TABLE-RULES.md). Auto blocks regenerate on every render (≤10 min)._")
    existing = SKELETON_R13 if campaign == "r13" else (__import__("verity_capture.sweep.render_r14", fromlist=["SKELETON_R14"]).SKELETON_R14 if campaign == "r14" else SKELETON)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    found: set[str] = set()

    def rep(m: re.Match) -> str:
        name = m.group(1)
        found.add(name)
        if name in blocks:
            return f"<!-- auto:begin {name} -->\n{blocks[name].rstrip()}\n<!-- auto:end -->"
        return m.group(0)

    new = AUTO_RE.sub(rep, existing)
    for name in blocks:
        if name not in found and name not in ("generalized", "partition", "exec", "revisions", "summary", "fixes", "clusters", "overhead"):
            print(f"warning: FINAL-REPORT.md has no auto block {name!r}; not rendered", file=sys.stderr)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new if new.endswith("\n") else new + "\n")


# ----------------------------------------------------------------------------------------------------------------
# R13 blocks: partition, exec summary counts, revisions


def block_partition(recs: R.Records) -> str:
    pc = partition_counts(recs)
    c = counts(recs)
    fam = sorted(R.MODEL_DISPLAY.get(m, m) for m in pc["families_complete"])
    out = [f"Distinct configurations: **{c['rows']}** (attempts are not counted; a row shows its latest attempt).",
           "",
           f"- **Fully completed** (Build ✅ + Match ✅ + Commit ✅ + measured overhead): **{pc['complete']}** — families: {', '.join(fam) or 'none'}",
           f"- **Partially completed** (at least one stage demonstrated, not all): **{pc['partial']}**",
           f"- **Blocked / not run**: **{pc['blocked']}**",
           "",
           f"Stage counts over attempted rows ({c['attempted']}): Build ✅ {_n(c, 'build')} (❌ {c['build_fail']}); Match ✅ {_n(c, 'match')} (❌ {c['match_fail']}); "
           f"Commit ✅ {_n(c, 'commit')} (❌ {c['commit_fail']}); benchmarked {_n(c, 'bench')}. "
           f"Completed on unchanged Verity code (`code_changes_required == []`): {c['zero_code']}.",
           ""]
    if pc["complete_rows"]:
        out.append("**Fully completed rows:**\n" + "\n".join(f"- `{r}` — {overhead_cell(recs, recs.rows[r])}, {R.release_label(recs.rows[r]) or 'r12 record'}"
                                                             for r in pc["complete_rows"]))
    if pc["partial_rows"]:
        out.append("\n**Partially completed rows:**\n" + "\n".join(
            f"- `{r}` — Build {glyph(recs, recs.rows[r], 'build')} Match {glyph(recs, recs.rows[r], 'match')} Commit {glyph(recs, recs.rows[r], 'commit')}"
            + (f"; blocked by {R.row_blocking_cluster(recs, recs.rows[r]) or '?'}" if R.row_blocking_cluster(recs, recs.rows[r]) else "")
            for r in pc["partial_rows"]))
    if pc["blocked_rows"]:
        by: dict[str, list[str]] = {}
        for r in pc["blocked_rows"]:
            by.setdefault(R.row_blocking_cluster(recs, recs.rows[r]) or (recs.rows[r].get("status") or "?"), []).append(r)
        out.append("\n**Blocked / not-run rows by first blocker:**\n" + "\n".join(f"- **{k}** ({len(v)}): " + ", ".join(f"`{r}`" for r in v)
                                                                                    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))))
    return "\n".join(out)


def block_exec(recs: R.Records, c: dict, now: str) -> str:
    pc = partition_counts(recs)
    rel = releases(recs)
    fam = sorted(R.MODEL_DISPLAY.get(m, m) for m in pc["families_complete"])
    hw_done = sorted({R.hw_key(recs.rows[r]) for r in pc["complete_rows"]})
    moe_done = [r for r in pc["complete_rows"] if R.MODEL_MOE.get(R.model_of(recs.rows[r]))]
    tp_done = [r for r in pc["complete_rows"] if (R.cfg_value(recs.rows[r], "tp") or 1) > 1]
    stoch_done = [r for r in pc["complete_rows"] if "stoch" in str(R.cfg_value(recs.rows[r], "sampling", ""))]
    bmax = max([R.cfg_value(recs.rows[r], "batch") or 0 for r in pc["complete_rows"]] or [0])
    def _pf(r):
        a = recs.rows[r].get("_attempt") if isinstance(recs.rows[r].get("_attempt"), dict) else recs.rows[r]
        return bool(a.get("post_freeze")) or bool(recs.rows[r].get("post_freeze"))  # explicit tag only (COORD 12:57Z/13:40Z)
    post = [r for r in pc["complete_rows"] if _pf(r)]
    pf_txt = (f" ({pc['complete'] - len(post)} completed at the 12:57Z freeze + {len(post)} completed post-freeze by the report: "
              + ", ".join(f"`{r}`" for r in post) + ")") if post else " (all completed at or before the 12:57Z freeze)"
    return (f"**{pc['complete']} of {c['rows']} configurations fully completed** (Build ✅ Match ✅ Commit ✅ + overhead) as of {now}{pf_txt}; "
            f"{pc['partial']} partially completed; {pc['blocked']} blocked or not run. Families completed: {', '.join(fam) or 'none'} "
            f"({len(fam)}/≥6 target); hardware with a completed row: {', '.join(R.HW_DISPLAY.get(h, h) for h in hw_done) or 'none'}; "
            f"largest completed batch {bmax}; MoE completed {len(moe_done)}; TP2 completed {len(tp_done)}; stochastic completed {len(stoch_done)}. "
            f"Releases integrated: {', '.join(t for t, _, _ in rel) or 'none yet'}. Overnight target: ≥25 completed rows, ≥6 families, ≥1 non-Ada, ≥1 MoE, ≥1 TP2.")


def releases(recs: R.Records) -> list[tuple[str, list[str], str]]:
    """All announced releases from `out/gen/r13/RELEASES.md` (RELEASED line + per-release sections), newest first: [(tag, [fix ids], hint)]."""
    from verity_capture.sweep import deps as D
    return [(r["tag"], r["fixes"], r.get("hint", "")) for r in D.parse_releases(recs.root)]
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"^\W*RELEASED\s+(r13-rel\d+)\s+(\S*)\s*(.*)$", line.strip())
            if m:
                fixes = [x for x in re.split(r"[,+/\s]+", m.group(2)) if x]
                out.append((m.group(1), fixes, m.group(3).strip()))
    return out


def block_revisions(recs: R.Records, now: str) -> str:
    rel = releases(recs)
    out = []
    if rel:
        out.append("| release tag | fixes | release owner's note |\n|---|---|---|")
        out += [f"| `{t}` | {', '.join(f) or '—'} | {_esc(h)[:200]} |" for t, f, h in rel]
    else:
        out.append("_(no `RELEASED <tag>` line in `out/gen/r13/RELEASES.md` yet)_")
    tags: dict[str, int] = {}
    for row in recs.rows.values():
        lab = R.release_label(row)
        if lab:
            tags[lab] = tags.get(lab, 0) + 1
    if tags:
        out.append("\nRows by revision of their latest attempt: " + ", ".join(f"`{k}` × {v}" for k, v in sorted(tags.items())) +
                   ". `lane-only` = passing on a lane revision not yet in a release tag; those rows are marked † in §2.")
    out.append("\nReproduction: `bash verity_capture/sweep/row_pod.sh <row-id> <ROLE> <hf-repo> <revision>` on a pod at `POD-READY <tag>` "
               "(release owner's `pod_release.sh`); evidence per row under `pod:<pod>:/workspace/cp/sweep/<row-id>/` and summaries under "
               "`out/gen/sweep/evidence/`. Pool rows: `python -m verity_capture.sweep.dispatch --once` reads `/tmp/r13-queue.txt` and launches "
               "`row_pod.sh` on cp-sw1/cp-sw2 with the release tag the pod is `POD-READY` for (env `WINDOW_MB=0 RETAIN=device OPEN_AFTER_RELEASE=0` "
               "selects the whole-block committer on rel13+). Frozen integration revision: `r13-rel13` (ff0127b38). "
               "Live workers at the freeze (COORD 12:57Z): pool cp-sw1, cp-sw2 (L40S); lane pods cp-dA, cp-dB, cp-bc, cp-dC (rev), cp-moe, cp-tp2 (2× L40S), "
               "cp-a100, cp-h100a, cp-rev (canary) — trimmed by the coordinator after this report. Backup: evidence copied to the network volume by the "
               "coordinator after publication (`out/gen/r13/WINDOW.md` log); records, failure/fix records and this report are on branch `lane/r13-report` "
               "(tag `r13-final`).")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out/gen/sweep")
    ap.add_argument("--now", default=None, help="override time label (HH:MMZ)")
    ap.add_argument("--blocker", action="append", default=[], help="extra 'Current blockers' line (e.g. merge failures)")
    ap.add_argument("--no-state", action="store_true", help="do not advance the since-last-update state")
    ap.add_argument("--campaign", default="r13", help="campaign id for the FINAL-REPORT skeleton (r12|r13)")
    a = ap.parse_args(argv)
    R.CAMPAIGN = a.campaign
    root = a.root
    now = a.now or datetime.now(timezone.utc).strftime("%H:%MZ")
    os.makedirs(root, exist_ok=True)

    recs = R.load(root)
    state = load_state(root)
    status, c = status_md(recs, state, now, a.blocker)

    with open(os.path.join(root, "MATRIX.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Configuration support matrix — {now} — {c['rows']} row record(s)\n\n{matrix_table(recs)}\n\n"
                 "✅ demonstrated · ◐ partial evidence (stage PASS, evidence field missing; see VALIDATION.md) · ❌ tested, fails · — not tested / not reached. "
                 "Overhead = `+x.x% (n rounds)` sweep measurement from the row's interleaved control/treatment rounds. Generated from `rows/*.json` only.\n")
    with open(os.path.join(root, "STATUS.md"), "w", encoding="utf-8") as fh:
        fh.write(status)
    with open(os.path.join(root, "VALIDATION.md"), "w", encoding="utf-8") as fh:
        fh.write(validation_md(recs, now))
    render_report(recs, c, now, os.path.join(root, "FINAL-REPORT.md"), a.campaign)
    if not a.no_state:
        save_state(root, recs, now)
    pc = partition_counts(recs)
    print(f"[partition {now}] complete={pc['complete']} partial={pc['partial']} blocked/not-run={pc['blocked']} of {c['rows']} configurations; "
          f"families complete={len(pc['families_complete'])}")
    print(f"[render {now}] rows={c['rows']} attempted={c['attempted']} build={_n(c, 'build')} match={_n(c, 'match')} "
          f"commit={_n(c, 'commit')} bench={_n(c, 'bench')} zero_code={c['zero_code']} failures={c['failures']} "
          f"fixes={c['fixes']} violations={len(recs.errors())} advisories={len(recs.advisories())} missing_evidence_rows={len(recs.missing_evidence)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
