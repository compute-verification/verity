"""R14 evidence reconciliation (brief §5 "First: recover existing evidence").

    python -m verity_capture.sweep.reconcile [--root out/gen/sweep] [--apply] [--now HH:MMZ]

For every row that carried a PASS/DONE stage under the R13 standard, locate the evidence directory the displayed attempt
points at (artifact paths in the record; else the row-named / lane evidence dirs containing that attempt's summaries), run
the finalizer over it in dry-run, and classify each stage into

    repaired      documentation repaired — the contemporaneous artifacts establish the result under r14-v1 (finalizer PASS)
    revalidate    evidence exists; a strengthened check must run over it (finalizer needs an artifact a lane can produce offline)
    execution     the identity/timing/value evidence was never recorded — new execution through the finalizer (runner)
    failed        the finalizer found an executed failure in the evidence (❌)

`--apply` writes the finalizer attempt (`kind: repair` when R13 flagged missing evidence on a stage the finalizer now
establishes, else `kind: revalidation`) for rows whose evidence dir carries at least a build summary.  Writes
`out/gen/r14/reconciliation.json` (consumed by the report) and `out/gen/r14/RECONCILIATION.md`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

from . import validate as R
from . import finalize as F

OWNER_CLAIMED = [
    "smollm2-135m__bf16__l40s__tp1__b8__i256__o32__mixed__greedy__bi-eager",
    "llama32-1b__bf16__l40s__tp1__b16__i1024__o128__mixed__greedy__bi-eager",
    "olmoe-1b-7b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager",
    "olmoe-1b-7b__bf16__l40s__tp1__b1__i1024__o128__mixed__greedy__bi-eager",
    "olmoe-1b-7b__bf16__l40s__tp1__b8__i256__o32__mixed__greedy__bi-eager",
    "llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager",
]


def is_r14_dir(d: str) -> bool:
    """R14 executions are new inference, not R13 evidence: excluded from the reconciliation of R13 passes."""
    d = d.rstrip("/")
    for probe in (d, os.path.dirname(d)):
        aj = os.path.join(probe, "attempt.json")
        if os.path.exists(aj):
            try:
                A = F.load(aj)
                if A.get("campaign") == "r14" or A.get("rules_version"):
                    return True
            except Exception:
                pass
    stg = os.path.join(d, "stages.txt")
    if os.path.exists(stg):
        for l in open(stg, errors="replace"):
            if l.startswith("release") and "2026-09-19T" in l and l.split("2026-09-19T")[1][:5] >= "16:46":
                return True
    return bool(re.search(r"/r14[-_]", d + "/")) or "/evidence/runner/" in d or "/evidence/gmatch/" in d or "/evidence/gprog/" in d


def evidence_dir_for(recs: R.Records, row: dict) -> tuple[str | None, str]:
    """(dir, how) — the directory the displayed R13 attempt's artifacts live in, else a row-named dir with a build summary."""
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else row
    rid = row["row_id"]
    paths = []
    for stg in ("build", "match", "commit", "bench"):
        d = a.get(stg) if isinstance(a.get(stg), dict) else {}
        for k in ("artifact", "summary", "verdict", "evidence", "evidence_dir"):
            v = d.get(k)
            if isinstance(v, str) and "evidence" in v:
                paths.append(v)
            elif isinstance(v, list):
                paths += [x for x in v if isinstance(x, str) and "evidence" in x]
    if isinstance(a.get("evidence_dir"), str):
        paths.append(a["evidence_dir"] + "/x")
    for p in paths:
        p = p.strip("` ")
        p = re.sub(r"^.*?(out/gen/sweep/evidence/)", r"\1", p)
        d = os.path.dirname(p) if not os.path.isdir(p) else p
        while d and d != "out/gen/sweep/evidence" and not glob.glob(os.path.join(d, "build_summary.json")) and not glob.glob(os.path.join(d, "*/build_summary.json")):
            d = os.path.dirname(d)
        if d and d != "out/gen/sweep/evidence" and os.path.isdir(d):
            if os.path.exists(os.path.join(d, "build_summary.json")):
                return d, "artifact path in the displayed attempt"
    # same-attempt key: the displayed attempt's Program digest must be the one in the evidence dir's build summary
    bd = a.get("build") if isinstance(a.get("build"), dict) else {}
    m = re.search(r"[0-9a-f]{16,64}", str(bd.get("program_digest") or ""))
    if m:
        pref = m.group(0)[:16]
        hits = []
        for bs in glob.glob("out/gen/sweep/evidence/**/build_summary.json", recursive=True):
            try:
                st = (F.load(bs).get("step") or {})
            except Exception:
                continue
            dg = str(st.get("program_digest") or "")
            if dg.startswith(pref):
                d = os.path.dirname(bs)
                if is_r14_dir(d):
                    continue
                other = re.findall(r"[a-z0-9.-]+__[a-z0-9]+__[a-z0-9]+__tp\d+__b\d+__i\d+__o\d+__[a-z-]+__[a-z-]+__[a-z-]+", d)
                if other and any(o != rid for o in other):
                    continue  # another row's evidence (M=1 Programs share digests across hardware) — not this attempt
                if R.hw_key(row) not in d and rid not in d and not re.search(R.hw_key(row).split("-")[0], d):
                    continue
                score = (rid in d, R.hw_key(row) in d, str(a.get("lane") or "zz")[:3] in d, os.path.getmtime(bs))
                hits.append((score, d))
        if hits:
            hits.sort()
            return hits[-1][1], f"build summary with the displayed attempt's program digest {pref}…"
    # row-named dirs
    cands = []
    for d in glob.glob(f"out/gen/sweep/evidence/*/{rid}/*/") + glob.glob(f"out/gen/sweep/evidence/{rid}/*/") + glob.glob(f"out/gen/sweep/evidence/*/{rid}/") + glob.glob(f"out/gen/sweep/evidence/{rid}/"):
        for probe in ("build_summary.json", "build_step_r0/result.json"):
            if os.path.exists(os.path.join(d, probe)) and not is_r14_dir(d):
                cands.append((os.path.getmtime(os.path.join(d, probe)), d.rstrip("/")))
                break
    if cands and not m:
        cands.sort()
        return cands[-1][1], "row-named evidence dir (latest build summary; the attempt records no program digest)"
    if cands and m and any(os.path.exists(os.path.join(d, "build_step_r0/result.json")) for _, d in cands):
        tpd = [d for _, d in cands if os.path.exists(os.path.join(d, "build_step_r0/result.json"))]
        for d in tpd:
            dg = str((F.load(os.path.join(d, "build_step_r0/result.json")).get("program") or {}).get("digest") or "")
            if dg.startswith(m.group(0)[:16]):
                return d, "TP2 layout dir with the displayed attempt's rank-0 program digest"
    if cands:
        return None, f"{len(cands)} row-named evidence dir(s) exist but none carries the displayed attempt's program digest {m.group(0)[:16]}… (other attempts); the displayed attempt's artifacts are " + (str(bd.get("artifact"))[:60] if bd.get("artifact") else "not recorded")
    return None, "no evidence directory in the tree" + (f" (artifacts recorded at {str(bd.get('artifact'))[:50]}…)" if bd.get("artifact") else "")


def classify_stage(r13: str, fin: dict, stage: str, B: int) -> tuple[str, str]:
    """-> (class, why)"""
    s = fin.get("status")
    reasons = "; ".join(fin.get("reasons") or [])
    if r13 not in ("PASS", "DONE"):
        if s in ("PASS", "DONE"):
            return "repaired", "R13 did not claim this stage; the artifacts establish it"
        return "execution", "not demonstrated in R13; no evidence to reconcile"
    if s in ("PASS", "DONE"):
        return "repaired", "artifacts establish the R13 result under r14-v1"
    if s == "FAIL":
        return "failed", reasons[:160]
    if s == "-":
        return "execution", "R13 PASS but the evidence dir has no artifact for this stage — never recorded"
    # PARTIAL
    if stage == "match":
        if "global Match verdict" in reasons or "global check" in reasons:
            return "execution", f"B={B}: needs cb_a.global_match/v1 over the execution (gmatch); the captured run dirs are not retained off-pod → new execution"
        if "steps compared" in reasons:
            return "execution", reasons[:160]
        return "revalidate", reasons[:160]
    if stage == "commit":
        if "coverage verdict" in reasons:
            return "execution", "binding map must be fixed before sampling and openings drawn from the expected population (bind) — counts+classes are insufficient; the collected values are not retained → new execution with bind's collector"
        return "revalidate", reasons[:160]
    if stage == "build":
        if "global workload Program" in reasons:
            return "revalidate", f"B={B}: per-request Programs only; the one global workload Program (gprog) can be built offline from weights + manifest"
        return "revalidate", reasons[:160]
    if stage == "bench":
        if "Commit ✅ of the same attempt" in reasons and "pairs" not in reasons:
            return "execution", "timing exists (≥3 pairs) but the overhead is of a Commit that is ◐ — remeasure with bind's binding implementation"
        return "revalidate", reasons[:160]
    return "revalidate", reasons[:160]


RANK = {"repaired": 0, "revalidate": 1, "execution": 2, "failed": 3}
LABEL = {"repaired": "documentation repaired", "revalidate": "revalidation required", "execution": "new execution required", "failed": "executed failure (❌)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/gen/sweep")
    ap.add_argument("--apply", action="store_true", help="write finalizer attempts for rows with an evidence dir")
    ap.add_argument("--now", default="")
    a = ap.parse_args(argv)
    # stable R13 baseline: the r13-final-addendum records (618ee4be6), exported once to /tmp/r13-baseline
    base_root = "/tmp/r13-baseline/out/gen/sweep"
    if not os.path.isdir(base_root):
        os.makedirs("/tmp/r13-baseline", exist_ok=True)
        os.system("git archive 618ee4be6 out/gen/sweep/rows out/gen/sweep/failures out/gen/sweep/fixes workloads out/gen/r13 | tar -x -C /tmp/r13-baseline")
    R.CAMPAIGN = "r13"
    base = R.load(base_root)
    R.CAMPAIGN = "r14"
    recs = R.load(a.root)
    out_rows: dict[str, dict] = {}
    applied = []
    for rid, row in recs.rows.items():
        # classify against the R13 headline as published (r13-final-addendum), so the baseline is stable across re-runs
        brow = base.rows.get(rid)
        r13 = {s: R.stage_status(brow, s) if brow else "-" for s in ("build", "match", "commit", "bench")}
        if brow is not None:
            row = dict(row)
            row["_attempt"] = brow.get("_attempt") if isinstance(brow.get("_attempt"), dict) else brow
            for s in ("build", "match", "commit", "bench"):
                if isinstance(brow.get(s), dict):
                    row[s] = brow[s]
        if not any(v in ("PASS", "DONE") for v in r13.values()):
            continue
        B = int(R.cfg_value(row, "batch") or 1)
        d, how = evidence_dir_for(recs, row)
        miss = {m[0] for m in base.missing_evidence.get(rid, [])}
        entry = {"row_id": rid, "B": B, "r13": r13, "evidence_dir": d, "evidence_how": how, "owner_claimed": rid in OWNER_CLAIMED, "r13_missing_evidence": sorted(miss)}
        if not d:
            entry["stages"] = {s: {"class": "execution" if r13[s] in ("PASS", "DONE") else "execution", "why": "no evidence directory with a build summary is in the tree — the result was never recorded as artifacts"} for s in r13}
            entry["class"] = "execution"
            out_rows[rid] = entry
            continue
        try:
            fin = F.finalize(rid, d, a.root, dry_run=True)
        except SystemExit as e:
            entry["error"] = str(e)
            entry["class"] = "execution"
            out_rows[rid] = entry
            continue
        stages = {}
        for s in ("build", "match", "commit", "bench"):
            cls, why = classify_stage(r13[s], fin[s], s, B)
            stages[s] = {"class": cls, "why": why, "finalizer": fin[s]["status"]}
        entry["stages"] = stages
        entry["finalizer"] = {s: fin[s]["status"] for s in stages}
        needed = [stages[s]["class"] for s in ("build", "match", "commit") ]
        entry["class"] = max(needed, key=lambda c: RANK[c])
        entry["repaired_stages"] = [s for s in stages if stages[s]["class"] == "repaired" and (s in miss or r13[s] not in ("PASS", "DONE"))]
        out_rows[rid] = entry
        if a.apply and not R.CAMPAIGN == "r13":
            # re-run whenever this R13 evidence dir has no finalizer attempt yet, or its finalizer attempt predates the finalizer
            # code (re-finalizing replaces the attempt in place; R14 executions live in their own dirs and are untouched)
            rel_d = os.path.relpath(d)
            prior = [x for x in (recs.rows[rid].get("attempts") or []) if isinstance(x, dict) and x.get("finalizer") and x.get("evidence_dir") == rel_d]
            stale = prior and str(prior[-1].get("finalized_at") or prior[-1].get("at") or "") < F.CODE_STAMP
            if not prior or stale:
                kind = "repair" if entry["repaired_stages"] else "revalidation"
                F.finalize(rid, d, a.root, lane="report (finalizer over R13 evidence)", kind=kind,
                           note=f"reconciliation {a.now}: finalizer run offline over the R13 evidence dir ({how}); no new inference")
                applied.append(rid)
    # write json + md
    r14dir = os.path.join(os.path.dirname(a.root.rstrip("/")), "r14")
    os.makedirs(r14dir, exist_ok=True)
    with open(os.path.join(r14dir, "reconciliation.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated": a.now, "rows": out_rows, "applied": applied}, fh, indent=1, ensure_ascii=False)
    cnt = Counter(e["class"] for e in out_rows.values())
    st_cnt = {s: Counter(e.get("stages", {}).get(s, {}).get("class", "execution") for e in out_rows.values()) for s in ("build", "match", "commit", "bench")}
    lines = [f"# R14 evidence reconciliation — {a.now} (report lane; brief §5, PLAN §1 row 6)", "",
             "Classification of every row that carried a Build/Match/Commit/overhead PASS under the R13 standard, made once from the evidence the record points at "
             "(artifact paths of the displayed attempt; else the row-named evidence dir). The finalizer (`python -m verity_capture.sweep.finalize`) was run in dry-run over each "
             "evidence dir; per stage: **repaired** = the contemporaneous artifacts establish the result under `r14-v1` (the finalizer derives ✅ — only recorded commands/settings were "
             "filled in, never what the row should have run) · **revalidate** = evidence exists, a strengthened check must run over it (offline; owning lane or rev) · **execution** = the "
             "identity/timing/value evidence was never recorded — new execution through the finalizer (runner / owning lane) · **failed** = the finalizer found an executed failure. "
             "The row class is the largest work item among Build/Match/Commit. A 17-step comparison never becomes a 128-step Match by metadata repair.", "",
             f"**Rows classified: {len(out_rows)}** — row class: " + ", ".join(f"{LABEL[k]} {v}" for k, v in sorted(cnt.items(), key=lambda kv: RANK[kv[0]])) + ".", "",
             "Per stage: " + " · ".join(f"{s}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: RANK.get(kv[0], 9))) for s, c in st_cnt.items()) + ".", "",
             "Two structural facts drive the split: (1) **Commit** under `r14-v1` needs a binding map fixed *before* sampling and openings drawn from the manifest's expected population "
             "(bind's checker) — no R13 commit artifact has that, and the collected values are not retained off the terminated pods, so every R13 Commit ✅ is *new execution required* "
             "(with bind's collector) even where counts, classes and fail-closed roots are impeccable; (2) **Match at B≥2** needs `cb_a.global_match/v1` (gmatch) over the execution — "
             "X-09 per-request equality is evidence inside it, not a substitute — so B≥2 Match is *new execution required* once gmatch lands. Build (B=1) and Match (B=1) are "
             "established offline from the recorded artifacts where the finalizer derives ✅.", "",
             "## The six owner-claimed completions", "",
             "| row | R13 claim | evidence dir | Build | Match | Commit | overhead | class |", "|---|---|---|---|---|---|---|---|"]
    def gl(s):
        return {"PASS": "✅", "DONE": "✅", "PARTIAL": "◐", "FAIL": "❌"}.get(s, "—")
    def row_line(e):
        st = e.get("stages", {})
        f = e.get("finalizer", {})
        cells = []
        for s in ("build", "match", "commit", "bench"):
            c = st.get(s, {})
            cells.append(f"{gl(e['r13'].get(s))}→{gl(f.get(s))} {c.get('class', '?')}")
        return f"| `{F_short(e['row_id'])}` | " + "".join(gl(e['r13'][s]) for s in ("build", "match", "commit", "bench")) + f" | {('`' + e['evidence_dir'].replace('out/gen/sweep/evidence/', '') + '`') if e.get('evidence_dir') else '—'} | " + " | ".join(cells) + f" | **{LABEL[e['class']]}** |"
    for rid in OWNER_CLAIMED:
        e = out_rows.get(rid)
        if e:
            lines.append(row_line(e))
        else:
            lines.append(f"| `{F_short(rid)}` | — | — | — | — | — | — | not in the R13 records with a PASS |")
    lines += ["", "## All rows with an R13 pass (◐ under r14-v1 until the finalizer accepts them)", "",
              "| row | R13 B/M/C/T | evidence dir | Build | Match | Commit | overhead | class |", "|---|---|---|---|---|---|---|---|"]
    for rid in sorted(out_rows, key=lambda r: (RANK[out_rows[r]["class"]], r)):
        lines.append(row_line(out_rows[rid]))
    lines += ["", "## Per-stage reasons (finalizer, dry-run)", ""]
    for rid in sorted(out_rows):
        e = out_rows[rid]
        if not e.get("stages"):
            continue
        ws = [f"{s}: {e['stages'][s]['why']}" for s in ("build", "match", "commit", "bench") if e["stages"][s]["class"] != "repaired"]
        if ws:
            lines.append(f"- `{F_short(rid)}` — " + " · ".join(ws))
    if applied:
        lines += ["", f"## Finalizer attempts written ({len(applied)}; `kind: repair|revalidation`, no new inference)", ""]
        lines += [f"- `{F_short(r)}`" for r in applied]
    with open(os.path.join(r14dir, "RECONCILIATION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[reconcile {a.now}] rows={len(out_rows)} " + " ".join(f"{k}={v}" for k, v in cnt.items()) + f" applied={len(applied)} -> {r14dir}/RECONCILIATION.md")
    return 0


def F_short(rid: str) -> str:
    from .render_r14 import short_id
    return short_id(rid)


if __name__ == "__main__":
    sys.exit(main())
