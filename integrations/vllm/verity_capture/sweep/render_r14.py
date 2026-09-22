"""R14 FINAL-REPORT blocks (BRIEF §9 / TABLE-RULES v1).

The R14 report leads with the full 15-column table, then counts, a caveats block, and the four questions Daniel asked:
complete under the shared standard / changed without new inference / did the two shared fixes work / genuinely incomplete.
Everything here is derived from the records (rows, attempts with `rules_version`, failures, fixes) plus two report-owned
inputs: `out/gen/r14/reconciliation.json` (evidence-reconciliation classification, brief §5) and `out/gen/r14/sharedfix.json`
(global-workload tests and the omit/duplicate attack results as reported by lanes gprog/gmatch/manifest/bind/rev).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from . import validate as R
from . import render as N

SKELETON_R14 = """# Verity R14 — two-hour matrix completion and validation window — FINAL REPORT

<!-- auto:begin header -->
<!-- auto:end -->

_The R13 overnight report (13:27Z) and its 14:35Z addendum are preserved unchanged at `out/gen/sweep/r13/FINAL-REPORT.md`; every R13 verdict stays in the row records as a historical attempt. This report shows each row under the shared standard `TABLE-RULES.md` (`rules_version: r14-v1`): a pass under the older standard is ◐ until the finalizer revalidates it; ⟨r13⟩/⟨r14-v1⟩ after the Build cell is the acceptance-standard version of the displayed attempt._

## 1. Configuration support matrix (15 canonical columns)

Three visibly distinct states per cell (ruling 17:36Z/7): **✅ passed under `r14-v1`** (finalizer verdict; row tagged ⟨r14-v1⟩) · **☑ passed under the R13 standard, not yet revalidated** (historical result, counts as ◐; row tagged ⟨r13⟩) · **◐ partial or insufficient evidence** under the standard shown (missing documentation is not a failure; a historical weaker pass is not a new pass) · ❌ an executed check failed · — not tested. Commit ✅ = C1 coverage ∧ C2 value correspondence ∧ C3 binding (all three required; C2 needs live evidence: root-compare vs independent run / replay / hook-time). A displayed row's four cells come from **one attempt** (same workload, profile, Program/query, compatible revisions).

<!-- auto:begin matrix -->
<!-- auto:end -->

## 2. Counts

<!-- auto:begin counts14 -->
<!-- auto:end -->

## 3. Caveats (beneath the table, never lost)

<!-- auto:begin caveats -->
<!-- auto:end -->

## 4. What is now complete under the shared standard?

<!-- auto:begin complete14 -->
<!-- auto:end -->

## 5. What changed without new inference?

_Record repair (from contemporaneous artifacts only) vs offline revalidation (strengthened check over existing evidence) vs new execution (brief §5; classification in `out/gen/r14/RECONCILIATION.md`)._

<!-- auto:begin changed -->
<!-- auto:end -->

## 6. Did the two shared fixes work?

_Shared fix A = one global workload Program + global Match (lanes gprog/gmatch); shared fix B = exact required-value coverage: manifest + binding map + the six negatives incl. the drop-`/1`-bind-`/0`-twice attack (lanes manifest/bind; rev mutation tests)._

<!-- auto:begin sharedfix -->
<!-- auto:end -->

## 7. What remains genuinely incomplete?

_Missing evidence (a row that ran, whose record/evidence does not establish the checkmark) vs missing implementation (a mechanism gap); one concrete blocker per mechanism._

<!-- auto:begin incomplete -->
<!-- auto:end -->

## 8. Record violations remaining

<!-- auto:begin violations14 -->
<!-- auto:end -->

## 9. Revisions, reproduction, backup

<!-- auto:begin revisions -->
<!-- auto:end -->
"""


def _root(recs: R.Records) -> str:
    return os.path.join(os.path.dirname(recs.root.rstrip("/")), "r14")


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def short_id(rid: str) -> str:
    p = rid.split("__")
    if len(p) < 10:
        return rid
    return f"{p[0]} {p[2]} tp{p[3][2:]} b{p[4][1:]} {p[5][1:]}/{p[6][1:]} {p[7]}" + (f" {p[9]}" if p[9] != "bi-eager" else "") + (f" {p[8]}" if p[8] != "greedy" else "")


def std_of(row: dict) -> str:
    return row.get("_standard") or "r13"


def is_r14(row: dict) -> bool:
    return std_of(row) == R.RULES_VERSION


def r14_attempts(row: dict) -> list[dict]:
    return [a for a in (row.get("attempts") or []) if isinstance(a, dict) and a.get("rules_version")]


# ----------------------------------------------------------------------------------------------------------------
# §2 counts


def block_counts14(recs: R.Records, now: str) -> str:
    rows = list(recs.rows.values())
    comp = [r for r in rows if N.partition(recs, r) == "complete"]
    hist = [r for r in rows if not is_r14(r) and any(R.stage_status(r, s) in ("PASS", "DONE") for s in ("build", "match", "commit"))]
    fin = [r for r in rows if is_r14(r)]
    g = lambda r, s: N.glyph(recs, r, s)
    c = Counter()
    for r in rows:
        for s in ("build", "match", "commit"):
            c[(s, g(r, s))] += 1
    fams = sorted({R.model_of(r) for r in comp})
    hws = sorted({R.hw_key(r) for r in comp})
    batches = sorted({int(R.cfg_value(r, "batch") or 0) for r in comp})
    moe = [r for r in comp if R.MODEL_MOE.get(R.model_of(r))]
    tp = [r for r in comp if (R.cfg_value(r, "tp") or 1) > 1]
    part = sum(1 for r in rows if N.partition(recs, r) == "partial")
    blocked = sum(1 for r in rows if N.partition(recs, r) == "blocked")
    out = [f"As of {now}, under `{R.RULES_VERSION}` (distinct configurations, {len(rows)} rows):", "",
           f"- **fully completed under the shared standard: {len(comp)}** (Build ✅ Match ✅ Commit ✅ + matched-control overhead, one attempt, `rules_version` present)",
           f"- partially completed: {part} · blocked or not run: {blocked}",
           f"- rows with a finalizer verdict (`rules_version: {R.RULES_VERSION}`): {len(fin)} · rows shown ☑ as **historical R13 passes not yet revalidated** (count as ◐): {len(hist)}",
           f"- families with a completed row: {len(fams)}" + (f" ({', '.join(fams)})" if fams else "") +
           f" · batch sizes: {', '.join(str(b) for b in batches) if batches else '—'} · hardware: {', '.join(hws) if hws else '—'} · MoE completed: {len(moe)} · TP2 completed: {len(tp)}",
           f"- Build ✅ {c[('build', '✅')]} / ◐ {c[('build', '◐')]} / ❌ {c[('build', '❌')]} · Match ✅ {c[('match', '✅')]} / ◐ {c[('match', '◐')]} / ❌ {c[('match', '❌')]} · Commit ✅ {c[('commit', '✅')]} / ◐ {c[('commit', '◐')]} / ❌ {c[('commit', '❌')]}",
           "", "R13 reference (historical standard, `out/gen/sweep/r13/FINAL-REPORT.md` addendum 14:35Z): 29 completed / 27 partial / 19 blocked — none of those 29 is a completion under `r14-v1` until the finalizer has revalidated it (Match at B≥2 needs the global checker; Commit needs the manifest/binding-map coverage verdict, F-rev-01)."]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# §3 caveats


def _chunked(row: dict) -> bool:
    a = row.get("_attempt") if isinstance(row.get("_attempt"), dict) else row
    txt = json.dumps({k: a.get(k) for k in ("cmd", "env", "notes", "bench", "commit") if k in a}, default=str)
    return bool(re.search(r"max_num_batched_tokens|chunked.prefill|chunked_prefill", txt, re.I))


def _match_criterion(row: dict) -> str:
    m = row.get("match") if isinstance(row.get("match"), dict) else {}
    txt = " ".join(str(m.get(k) or "") for k in ("comparator", "criterion", "verdict", "note", "notes"))
    if "global_match" in txt:
        return "global checker (cb_a.global_match)"
    if "batch_decomp" in txt or "X-09" in txt:
        return "X-09 per-request decomposition (evidence only under r14-v1, not the global check)"
    if re.search(r"live-cone|livecone", txt, re.I):
        return "live-cone (diagnostic only)"
    if re.search(r"collective", txt, re.I):
        return "collective-level only (TP)"
    if "cb_a.compare" in txt:
        return "cb_a.compare, single request"
    return "—"


def _value_corr(recs: R.Records) -> Counter:
    c = Counter()
    for r in R.sorted_rows(recs):
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else None
        if not (a and a.get("finalizer")):
            continue
        cm = a.get("commit") if isinstance(a.get("commit"), dict) else {}
        if cm.get("status") not in ("PASS", "FAIL", "PARTIAL"):
            continue
        vc = cm.get("value_correspondence") if isinstance(cm.get("value_correspondence"), dict) else {}
        meth = vc.get("method") or "—"
        c[f"{vc.get('result') or 'not checked'} ({str(meth).split(' (')[0]})"] += 1
    return c


def _cfg_labels(recs: R.Records) -> Counter:
    c = Counter()
    for r in R.sorted_rows(recs):
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
        b = a.get("bench") if isinstance(a.get("bench"), dict) else {}
        cm = a.get("commit") if isinstance(a.get("commit"), dict) else {}
        lab = b.get("variant_label") or cm.get("variant_label")
        if not lab and cm.get("collector_variant"):
            es = cm.get("engine_settings") if isinstance(cm.get("engine_settings"), dict) else {}
            lab = f"{cm.get('collector_variant')}, retain {cm.get('retention') or '?'}" + (f", util {es.get('gpu_memory_utilization')}" if es.get("gpu_memory_utilization") else "")
        if lab and cm.get("status") in ("PASS", "PARTIAL", "FAIL"):
            c[str(lab)[:90]] += 1
    return c


def block_caveats(recs: R.Records) -> str:
    rows = [r for r in R.sorted_rows(recs) if any(R.stage_status(r, s) in ("PASS", "DONE") for s in ("build", "match", "commit", "bench"))]
    wb = wd = 0
    ret = Counter()
    chunked, criteria, revs = [], Counter(), Counter()
    for r in rows:
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
        if R.stage_status(r, "commit") in ("PASS",):
            lab = N._retention(a)
            ret[lab] += 1
            if "windowed" in lab:
                wd += 1
            else:
                wb += 1
        if _chunked(r):
            chunked.append(short_id(r["row_id"]))
        if R.stage_status(r, "match") in ("PASS", "FAIL"):
            criteria[_match_criterion(r)] += 1
        revs[str(a.get("release_tag") or ("lane-only " + str(a.get("revision") or "?")[:9]))] += 1
    out = ["| caveat | what the records say |", "|---|---|",
           f"| **Collector variant** | Commit-PASS attempts: whole-block committer {wb}, windowed collector (P0 ring, REL-11 default from r13-rel13) {wd}. Both are allowed under one correctness standard; overheads are **different measurements** (rev-N7: +20.5 % whole-block vs +51.5 % windowed on the same row/pod) and are never averaged or compared across the variant. |",
           f"| **Retention** | " + "; ".join(f"{k}: {v}" for k, v in ret.most_common()) + " — shown per cell in the overhead column label and in the record (`commit.retention` / `VERITY_RETAIN`). |",
           f"| **Chunked prefill** | {len(chunked)} displayed attempt(s) ran with chunked prefill (`max_num_batched_tokens` below B×prompt): {', '.join('`' + c + '`' for c in chunked) if chunked else '—'}. Part of the Execution configuration (brief §5); control and treatment used matching settings where recorded. |",
           f"| **Match criterion** | " + "; ".join(f"{k}: {v}" for k, v in criteria.most_common()) + f". Under `{R.RULES_VERSION}` only the global checker (all requests, all steps, chronology, unattributed = 0) or `cb_a.compare` over all steps of a single request is a ✅; X-09 per-request equality is evidence inside it; live-cone is diagnostic. |",
           f"| **Revision** | displayed attempts by revision: " + ", ".join(f"{k} ({v})" for k, v in revs.most_common()) + ". Lane-only (†) = not in an announced release tag; the R14 base is announced by rel in `out/gen/r14/RELEASES.md`. |",
           f"| **Value correspondence (C2)** | required subcheck of Commit ✅ (Daniel 17:36Z ruling 2; TABLE-RULES §Commit C1∧C2∧C3): the binding points to the correct runtime occurrence and value — live evidence named by method (root-compare vs independent/known-good run, replay, hook-time) with an artifact; coverage + openings without it ⇒ Commit ◐. Finalizer Commit verdicts by C2 state: " + (", ".join(f"{k}: {v}" for k, v in _value_corr(recs).most_common()) or "no finalizer Commit verdict yet") + ". bind's `late_flush` fault is invisible to coverage+openings and caught only by C2. |",
           f"| **Execution configuration (rule 3)** | " + (", ".join(f"{k}: {v}" for k, v in _cfg_labels(recs).most_common(6)) or "—") + " — collector variant, retention, ring size, `gpu_memory_utilization`, chunked prefill recorded per attempt (`bench.variant_label` / `commit.engine_settings`), identical for control and treatment, never chosen per result. |",
           f"| **Padding scope** | active-reference only: inactive padded interiors and cross-request cache reuse are outside every ✅ (TABLE-RULES). |",
           f"| **Measurement variability** | per-pod spreads and repeated attempts are listed as separate attempts (never averaged); see the variability list below. |"]
    var = N.block_variability(recs)
    return "\n".join(out) + "\n\n" + var


# ----------------------------------------------------------------------------------------------------------------
# §4 complete under the shared standard


def block_complete14(recs: R.Records) -> str:
    comp = [r for r in R.sorted_rows(recs) if N.partition(recs, r) == "complete"]
    if not comp:
        hist = [r for r in R.sorted_rows(recs) if not is_r14(r) and all(R.stage_status(r, s) == "PASS" for s in ("build", "match", "commit"))]
        return (f"**0 configurations are complete under `{R.RULES_VERSION}` yet.** {len(hist)} rows carry Build/Match/Commit PASS under the R13 standard "
                f"(historical attempts, shown ◐): " + ", ".join("`" + short_id(r["row_id"]) + "`" for r in hist) +
                ". Each becomes ✅ only through the finalizer (`python -m verity_capture.sweep.finalize <row-id> <evidence-dir>`) once the row's evidence satisfies TABLE-RULES: "
                "Build (program digest + builder command + TargetProfile + request-manifest digest + sampling-policy applicability), Match (global checker at B≥2 / all steps at B=1, tokens equal, unattributed 0), "
                "Commit (manifest digest + binding-map digest + coverage verdict from bind's checker + openings from the expected population), overhead (≥3 interleaved pairs, control and treatment shown, variant + retention labelled).")
    out = ["| configuration | standard | Build | Match | Commit | overhead | variant / retention | revision |", "|---|---|---|---|---|---|---|---|"]
    for r in comp:
        a = r.get("_attempt") if isinstance(r.get("_attempt"), dict) else r
        out.append(f"| `{short_id(r['row_id'])}` | {std_of(r)} | ✅ | ✅ | ✅ | {N.overhead_cell(recs, r)} | {N._retention(a)} | {a.get('release_tag') or 'lane-only ' + str(a.get('revision') or '?')[:9]} |")
    fams = Counter(R.model_of(r) for r in comp)
    out.append("")
    out.append(f"{len(comp)} configuration(s); families: " + ", ".join(f"{k} ({v})" for k, v in fams.items()) +
               f"; batches: {sorted({int(R.cfg_value(r, 'batch') or 0) for r in comp})}; hardware: {sorted({R.hw_key(r) for r in comp})}; "
               f"MoE: {sum(1 for r in comp if R.MODEL_MOE.get(R.model_of(r)))}; TP2: {sum(1 for r in comp if (R.cfg_value(r, 'tp') or 1) > 1)}.")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# §5 changed without new inference


KIND_LABEL = {"repair": "documentation repaired", "revalidation": "revalidation (offline, over existing evidence)",
              "execution": "new execution", "none": "no change"}


def block_changed(recs: R.Records) -> str:
    rec = _load_json(os.path.join(_root(recs), "reconciliation.json"), {})
    cls = rec.get("rows", {}) if isinstance(rec, dict) else {}
    by_kind: dict[str, list[str]] = defaultdict(list)
    for rid, d in cls.items():
        by_kind[d.get("class", "?")].append(rid)
    # what actually happened: r14 attempts by kind
    done: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in R.sorted_rows(recs):
        rid = row["row_id"]
        seen_k: dict[str, str] = {}
        fin = [a for a in r14_attempts(row) if a.get("finalizer")]
        for a in fin:
            k = a.get("kind") or "revalidation"
            if row.get("_attempt") is a:
                seen_k[k] = "".join(N.glyph(recs, row, s) for s in ("build", "match", "commit"))
            else:
                # earlier finalizer attempt of another kind for this row: its own derived statuses (history)
                seen_k.setdefault(k, "".join({"PASS": "✅", "DONE": "✅", "FAIL": "❌", "PARTIAL": "◐"}.get((a.get(s) or {}).get("status") if isinstance(a.get(s), dict) else None, "—") for s in ("build", "match", "commit")) + " (history)")
        for k, gl in seen_k.items():
            done[k].append((rid, gl))
    out = ["**Classification of the R13 evidence (brief §5; `RECONCILIATION.md`):** " +
           (", ".join(f"{KIND_LABEL.get(k, k)} {len(v)}" for k, v in sorted(by_kind.items())) if by_kind else "_(RECONCILIATION.md pending)_") + ".", ""]
    out += ["| change without new inference | rows | result under r14-v1 |", "|---|---:|---|"]
    for k in ("repair", "revalidation"):
        items = done.get(k, [])
        out.append(f"| {KIND_LABEL[k]} | {len(items)} | " + (", ".join(f"`{short_id(r)}` {g}" for r, g in items) if items else "none yet") + " |")
    items = done.get("execution", [])
    out.append(f"| **new execution** (runner / lane pods, through the finalizer) | {len(items)} | " + (", ".join(f"`{short_id(r)}` {g}" for r, g in items) if items else "none yet") + " |")
    out.append("")
    out.append("Rule kept: a 17-step comparison never becomes a 128-step Match by metadata repair; repaired fields come only from contemporaneous artifacts (logs, configs, manifests, build manifests, recorded commands), never from what the row should have run.")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# §6 shared fixes


def block_sharedfix(recs: R.Records) -> str:
    sf = _load_json(os.path.join(_root(recs), "sharedfix.json"), {})
    out = ["| shared fix | test | result | evidence |", "|---|---|---|---|"]
    tests = sf.get("tests") if isinstance(sf, dict) else None
    if not tests:
        out.append("| A — global workload Program (gprog) | B=2 unequal requests → staggered-arrival fixture → existing B8 workload | _pending (lane STATUS)_ | — |")
        out.append("| A — global Match (gmatch, `cb_a.global_match/v1`) | all requests present / none omitted or duplicated / identity map unambiguous / chronology agrees / global VU population = query population | _pending_ | — |")
        out.append("| B — required-value manifest (manifest) | expected identities from Program + fixed query; zero/zero hole closed | _pending_ | — |")
        out.append("| B — binding map + coverage checker (bind) | six negatives incl. drop `/1` + bind `/0` twice (F-rev-01), omit family, omit request/step/rank, late read, relabel, wrong root/layout | _pending_ | — |")
        out.append("| rev mutation tests | omit / duplicate request, swap outputs, alter arrival association, broken dependency | _pending (REVIEW.md T+105)_ | — |")
    else:
        for t in tests:
            out.append(f"| {t.get('fix', '?')} | {t.get('test', '?')} | {t.get('result', '?')} | {t.get('evidence', '—')} |")
    if getattr(recs, "demos", None):
        out.append("")
        out.append("Shared-fix demonstration fixtures (R14 rows outside the 75-configuration panel — COORD 18:09Z; finalizer statuses, not table rows):")
        for rid, d in sorted(recs.demos.items()):
            fin = [a for a in (d.get("attempts") or []) if isinstance(a, dict) and a.get("finalizer")]
            if fin:
                a = max(fin, key=lambda a: str(a.get("at") or ""))
                G = {"PASS": "✅", "DONE": "✅", "FAIL": "❌", "PARTIAL": "◐"}
                gl = " ".join(f"{s} " + G.get(((a.get(s) or {}) if isinstance(a.get(s), dict) else {}).get("status"), "—") for s in ("build", "match", "commit", "bench"))
                out.append(f"- `{short_id(rid)}` ({a.get('lane')}, {a.get('release_tag') or str(a.get('revision') or '?')[:9]}): {gl}")
            else:
                out.append(f"- `{short_id(rid)}`: no finalizer verdict (status {d.get('status')})")
    fixes14 = [f for f in recs.fixes.values() if str(f.get("campaign") or "") == "r14" or str(f.get("fix_id") or "").split("-")[0] in ("GPROG", "GMATCH", "MAN", "BIND", "GLOBAL")]
    if fixes14:
        out.append("")
        out.append("Fix records (R14): " + ", ".join(f"`{f.get('fix_id')}` ({f.get('lane', '?')}, {str(f.get('code_commit') or '?')[:9]})" for f in fixes14))
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# §7 genuinely incomplete


def block_incomplete(recs: R.Records) -> str:
    rows = R.sorted_rows(recs)
    missing_ev, missing_impl = [], []
    for r in rows:
        if N.partition(recs, r) == "complete":
            continue
        gs = {s: N.glyph(recs, r, s) for s in ("build", "match", "commit")}
        if "❌" in gs.values() or all(g == "—" for g in gs.values()):
            missing_impl.append(r)
        else:
            missing_ev.append(r)
    out = [f"**Missing evidence — {len(missing_ev)} rows** ran (something passed, under R13 or R14) but the record does not establish every checkmark under `{R.RULES_VERSION}`:", ""]
    # group by what is missing
    reasons = Counter()
    for r in missing_ev:
        reasons[_gap_reason(recs, r)] += 1
    out += [f"- {k}: {v} row(s)" for k, v in reasons.most_common()]
    out += ["", f"**Missing implementation — {len(missing_impl)} rows** blocked by a mechanism (one concrete blocker per mechanism):", "",
            "| mechanism (cluster) | rows | concrete blocker |", "|---|---:|---|"]
    deps = _load_json(os.path.join(recs.root, "DEPS.json"), {})
    drows = deps.get("rows", deps) if isinstance(deps, dict) else {}
    bl = Counter()
    for r in missing_impl:
        d = drows.get(r["row_id"], {}) if isinstance(drows, dict) else {}
        rem = d.get("remaining") or d.get("blockers") or []
        if not rem:
            rem = ["(unclassified — see VALIDATION.md)"]
        for c in rem:
            bl[c] += 1
    for c, n in bl.most_common():
        out.append(f"| `{c}` | {n} | {BLOCKER_TEXT.get(c, R.GAP_LABEL.get(c, '—'))} |")
    return "\n".join(out)


BLOCKER_TEXT = {
    "M-BATCH-DECOMP": "global Match: `cb_a.global_match/v1` (gmatch) must replace per-request decomposition as the checker for B≥2 rows",
    "M-ARRIVAL-ORDER": "global chronology in the global checker (gmatch) — canonical within-step order for co-scheduled prefills; not a re-sort",
    "C-FA2-STREAM": "FA2 hidden stream O(rows×N) does not fit next to the engine on 48 GB at 1024 tokens for hd-128 families — needs streaming/bounded staging of the hidden M1 stream (p0 arm-time bound)",
    "B-SAMPLING": "Bet A builds a greedy Program for a stochastic request; the X-06 sampler Definition exists but is unwired — Build must refuse (applicability) until the sampling Program shape exists",
    "M-TP-ENGINE": "per-rank instance-sequence fold/compare and rank-local GEMM-partial bindings (tp lane) — collective-level only today",
    "M-COMPILED-FUSION": "compiled-mode fusions are invisible to Match (no per-op instances) — no implementation this window",
    "B-GEMMA-NORM": "GemmaRMSNorm ATen chain (`aten.to.dtype` upcast, (1+w) scale, softcap) unregistered after X-08",
    "B-FP8": "`float8_e4m3fn` carries no `Value<w>` width — codec + scales + CUTLASS scaled-MM Definition (recorded gap)",
    "C-FOLD-COST": "fold/comparator time limit (harness); FOLD-01 made the fold linear, b1 4096/512 still at ×4 budget",
    "INFRA-TIMEOUT": "harness build time limit (SIGKILLed derive) — rerun with the REL-06 B-scaled limits",
    "INFRA-ENV": "checkpoint not cached on the pod (pool pods run offline)",
    "INFRA-HARNESS": "stale derived dir made `batch_decomp` fail closed with empty reasons",
    "B-MOE": "fused MoE semantics (MOE-01/02/03 shipped in rel11); remaining rows need the manifest/binding coverage check",
    "B-HOPPER-ATTN": "FA3 tap (HOP-01) is lane-only; native FA3 coverage only when the actual backend/tap is available",
    "C-HOPPER-FA3": "FA3 hidden values: history-dependent staging at B>1 (P0-03 has no FA3 twin)",
    "M-NOPROFILE": "family eager profile missing (`model_type qwen3` not in the dense vocabulary)",
    "C-COLLECTOR-FOOTPRINT": "collector staging exceeds 48 GB without the windowed ring; run with P0 windowed defaults",
    "M-AMPERE-GEMM-TILE": "AMP-01 shipped (rel2); remaining A100 rows are evidence, not mechanism",
}


def _gap_reason(recs: R.Records, r: dict) -> str:
    if not is_r14(r):
        gs = [R.stage_status(r, s) for s in ("build", "match", "commit")]
        if all(g == "PASS" for g in gs):
            return "R13 Build/Match/Commit PASS — awaiting finalizer revalidation under r14-v1 (global Match at B≥2; manifest/binding coverage for Commit)"
        return "R13 partial attempt (some stage not reached or ◐) — awaiting reconciliation (RECONCILIATION.md)"
    parts = []
    for s in ("build", "match", "commit"):
        d = r.get(s) if isinstance(r.get(s), dict) else {}
        if d.get("status") == "PARTIAL":
            rs = [x for x in (d.get("reasons") or []) if x.startswith("missing: ")]
            key = rs[0][len("missing: "):] if rs else (d.get("reasons") or ["?"])[0]
            key = re.sub(r"\(.*?\)", "", key)
            items = [k.strip() for k in key.split(",")]
            key = ", ".join(items[:4]) + (f" (+{len(items) - 4} more)" if len(items) > 4 else "")
            parts.append(f"{s}: {key}")
    miss = recs.missing_evidence.get(r["row_id"], [])
    if not parts and miss:
        parts.append(", ".join(sorted({m[0] + "." + str(m[1]).split("(")[0].strip() for m in miss}))[:120])
    return "finalizer ◐ — " + " · ".join(parts) if parts else "finalizer verdict ◐ (see VALIDATION.md)"


# ----------------------------------------------------------------------------------------------------------------
# §8 violations


def block_violations14(recs: R.Records) -> str:
    errs = [v for v in recs.violations if v.severity != "advisory"]
    adv = [v for v in recs.violations if v.severity == "advisory"]
    by_row = Counter(v.file for v in errs if v.file.startswith("rows/"))
    out = [f"{len(errs)} violations / {len(adv)} advisories over {len(recs.rows)} row, {len(recs.failures)} failure, {len(recs.fixes)} fix records (`VALIDATION.md`). "
           f"Records without `rules_version` on an r14 attempt are refused (TABLE-RULES)."]
    def klass(v) -> str:
        m = v.message
        if "no rules_version" in m: return "r14 attempt without `rules_version` (refused)"
        if "typed PASS statuses without the finalizer" in m: return "r14 attempt with typed PASS, no finalizer (refused as headline)"
        if "release_tag" in v.field or "release tag" in m: return "release tag / lane-only revision"
        if v.field.startswith(("build.", "match.", "commit.", "bench.")): return f"`{v.field}` field"
        if "has no rows/" in m: return "fix record names a row id that does not exist"
        if v.file.startswith("failures/"): return "failure record"
        return f"`{v.field}`"
    by_kind = Counter(klass(v) for v in errs)
    out.append("")
    out.append("| violation class | count |")
    out.append("|---|---:|")
    for k, n in by_kind.most_common(10):
        out.append(f"| {k} | {n} |")
    out.append("")
    out.append("Rows with the most open violations:")
    for f, n in by_row.most_common(8):
        out.append(f"- `{f.split('/')[-1].replace('.json', '')}`: {n}")
    return "\n".join(out)
