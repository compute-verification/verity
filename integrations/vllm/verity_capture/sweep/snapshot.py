"""snapshot.py — the report lane's one atomic snapshot (R15, brief §10; PLAN §2 row 2).

    python -m verity_capture.sweep.snapshot [--root out/gen/sweep] [--board /private/tmp/r15/board] [--log LOG.md] [--no-archive]

One load of rows/*.json → CURRENT.json + MATRIX.md + REPORT.md + VALIDATION.md (+ STATUS.md, FINAL-REPORT.md auto blocks) rendered
from the same in-memory records, each written to a temp file in the target directory and renamed into place (os.replace), then the
whole set copied under archive/<utc>/.  CURRENT.json names the snapshot, rules_version, the integration release tag/commit (latest
board RELEASE post; fallback tag r14-rc3), generated_at, the source-record digest (sha256 over the sorted row record files) and counts.

REPORT.md (brief §10): 1 current summary · 2 full table (Model | MoE | Datatype | Hardware | TP | Batch | Max input tokens |
Max output tokens | Workload | Sampling | Execution | Build | Match | Commit | Commitment overhead) · 3 important qualifications ·
4 reusable fixes and rows unlocked · 5 remaining blockers · 6 pending decisions (board DECISION posts) · 7 tested revisions,
reproduction, backup, fleet state (coordinator LOG.md).  Nothing here derives a status: finalize.py does; this renders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from . import render as RD
from . import validate as R

RULES_VERSION = "r14-v1"
CAMPAIGN = "r16"
BOARD_DEFAULT = os.environ.get("R16_BOARD", os.environ.get("R15_BOARD", "/private/tmp/r16/board"))   # board.py's default for R16
LOG_DEFAULT = "/private/tmp/r13-release/out/gen/r16/LOG.md"
BOARD_COPY = "out/gen/r16/board"          # durable copy of the board on the report branch (relative to the repo root)
FALLBACK_RELEASE = ("r15-int-6b", "22668e4f6")   # the last R15 release (R15 board RELEASE M-0811); superseded by the first R16 RELEASE post


# ----------------------------------------------------------------------------------------------------------------
# atomic files


def atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".snap-", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def source_digest(root: str) -> tuple[str, int, list[str]]:
    """sha256 over the sorted row record files (name + bytes), i.e. what the table is a pure function of."""
    d = os.path.join(root, "rows")
    names = sorted(n for n in os.listdir(d) if n.endswith(".json") and not n.startswith("."))
    h = hashlib.sha256()
    for n in names:
        with open(os.path.join(d, n), "rb") as fh:
            b = fh.read()
        h.update(n.encode()); h.update(b"\0"); h.update(sha256_bytes(b).encode()); h.update(b"\n")
    return h.hexdigest(), len(names), names


# ----------------------------------------------------------------------------------------------------------------
# board


def board_events(board: str) -> list[dict]:
    d = os.path.join(board, "events")
    out: list[dict] = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                e = json.load(fh)
            e["_file"] = fn
            out.append(e)
        except Exception as ex:  # noqa: BLE001
            out.append({"id": fn, "type": "ALERT", "author": "board", "time": "", "text": f"unreadable event {fn}: {ex}", "tags": ["board"], "_file": fn})
    return out


def superseded_ids(events: list[dict]) -> set[str]:
    return {e.get("supersedes") for e in events if e.get("supersedes")}


def latest_release(events: list[dict]) -> tuple[str, str, str | None]:
    """(tag, commit, message id) of the newest RELEASE post naming `tag sha`; fallback r14-rc3."""
    tag, sha, mid = FALLBACK_RELEASE[0], FALLBACK_RELEASE[1], None
    for e in events:
        if e.get("type") != "RELEASE":
            continue
        txt = e.get("text") or ""
        # rel's form of record: "RELEASE[D] <tag> = [lane/<branch>] <sha>" (M-0090, M-0305, M-0415); the leading form wins over any
        # later "<tag> <sha>" mention (CONTENTS on / LAST-KNOWN-GOOD / on r15-int-1 ... would otherwise name the PREVIOUS release).
        tagsha = r"(r1\d-(?:rc\d+[a-z]?|int-\d+[a-z]?|final[\w-]*))\b\s*[=(:]?\s*(?:lane/[\w\-./]+\s+)?([0-9a-f]{7,40})\b"
        m = re.search(r"\bRELEASED?\s+" + tagsha, txt) or re.search(r"\b" + tagsha, txt)
        if m and re.search(r"\b(LAST-KNOWN-GOOD|superseded|not released|withdrawn)\b", txt[: m.start()], re.I):
            m = None
        if m:
            tag, sha, mid = m.group(1), m.group(2), e.get("id")
    return tag, sha, mid


def _hhmm(e: dict) -> str:
    t = str(e.get("time") or "")
    return t[11:16] + "Z" if len(t) >= 16 else t


def _excerpt(s: str, n: int = 420) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def board_section(events: list[dict], types: tuple[str, ...], to_report_only: bool = False, limit: int = 40) -> list[str]:
    sup = superseded_ids(events)
    acked = {e.get("thread") for e in events if e.get("type") == "ACK" and e.get("author") == "report"}
    lines: list[str] = []
    for e in events:
        if e.get("type") not in types or e.get("id") in sup:
            continue
        if to_report_only and "report" not in (e.get("to") or []):
            continue
        tail = ""
        if e.get("action"):
            tail += f" **Action:** {_excerpt(e['action'], 240)}"
        if e.get("evidence"):
            tail += " Evidence: " + "; ".join(f"`{x}`" for x in (e["evidence"] if isinstance(e["evidence"], list) else [e["evidence"]])[:4])
        mark = " (acked by report)" if e.get("id") in acked else ""
        lines.append(f"- **{e.get('id')}** {_hhmm(e)} {e.get('type')} `{e.get('author')}` [{', '.join(e.get('tags') or [])}]{mark}: {_excerpt(e.get('text'))}{tail}")
    if len(lines) > limit:
        lines = lines[-limit:] + [f"- … {len(lines) - limit} older posts in the board copy"]
    return lines


def render_board(board: str, repo_root: str) -> str | None:
    """`board.py render` → BOARD.md; copy BOARD.md + events/ into out/gen/r15/board/ on this branch (durability). Returns the copy path."""
    bp = os.path.join(repo_root, "zk", "campaign", "board.py")
    if not os.path.exists(bp):
        return None
    env = dict(os.environ, R15_BOARD=board, R16_BOARD=board)
    try:
        subprocess.run([sys.executable, bp, "render"], check=False, env=env, capture_output=True, timeout=120)
    except Exception as ex:  # noqa: BLE001
        print(f"warning: board render failed: {ex}", file=sys.stderr)
    dst = os.path.join(repo_root, BOARD_COPY)
    os.makedirs(os.path.join(dst, "events"), exist_ok=True)
    src_md = os.path.join(board, "BOARD.md")
    if os.path.exists(src_md):
        with open(src_md, encoding="utf-8") as fh:
            atomic_write(os.path.join(dst, "BOARD.md"), fh.read())
    ev = os.path.join(board, "events")
    if os.path.isdir(ev):
        for fn in os.listdir(ev):
            if fn.endswith(".json") and not os.path.exists(os.path.join(dst, "events", fn)):
                shutil.copy2(os.path.join(ev, fn), os.path.join(dst, "events", fn))   # events are immutable (tmp+rename, never edited)
    return dst


# ----------------------------------------------------------------------------------------------------------------
# fleet state (coordinator LOG.md)


def fleet_state(log_path: str) -> list[str]:
    if not os.path.exists(log_path):
        return [f"- coordinator LOG.md not readable at `{log_path}`"]
    rows: list[str] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("|") and re.search(r"fleet|pod|burn|deadman|\$\d", line, re.I) and not line.startswith("|---") and "| # |" not in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    rows.append(f"- LOG #{cells[0]} {cells[1]}: {_excerpt(cells[2], 520)}")
    head = ""
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("T0"):
                head = line.strip()
                break
    out = ([f"- {head}"] if head else []) + rows[-8:]
    return out or ["- no fleet lines in LOG.md yet"]


# ----------------------------------------------------------------------------------------------------------------
# the report


def row_states(recs: R.Records) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in R.sorted_rows(recs):
        rid = row.get("row_id")
        att = R.headline_attempt(row) if hasattr(R, "headline_attempt") else None
        out[rid] = {
            "status": row.get("status"),
            "build": RD.glyph(recs, row, "build"), "match": RD.glyph(recs, row, "match"), "commit": RD.glyph(recs, row, "commit"),
            "overhead": RD.overhead_cell(recs, row),
            "campaign": RD.attempt_campaign(row),
            "release_tag": (att or {}).get("release_tag") if isinstance(att, dict) else None,
            "attempts": len(row.get("attempts") or []),
            "quarantined": bool(RD.quarantined(recs, row)),
        }
    return out


def diff_states(prev: dict | None, cur: dict[str, dict]) -> list[str]:
    if not prev or not isinstance(prev.get("rows"), dict):
        return ["- first snapshot of this report loop (no previous CURRENT.json to diff against)"]
    P = prev["rows"]
    out: list[str] = []
    for rid, s in cur.items():
        p = P.get(rid)
        if not p:
            out.append(f"- `{rid}`: new in the table")
            continue
        ch = [f"{k} {p.get(k)}→{s.get(k)}" for k in ("build", "match", "commit", "overhead", "status") if p.get(k) != s.get(k)]
        if ch:
            out.append(f"- `{rid}`: " + ", ".join(ch))
    return out or [f"- no row changed since the previous snapshot ({prev.get('generated_at')})"]


# COORD M-0289 (Daniel 21:33Z): the scope line beneath the matrix — every ✅ is active-reference scope (TABLE-RULES r14-v1 line 25)
SCOPE_LINE = ("Scope of every ✅ (TABLE-RULES r14-v1 line 25, COORD M-0289; Daniel's DECISIONS-2 §2 'Current status', out/gen/r15/DECISIONS-2.md): **active-reference** — "
              "the Program is the active-reference circuit of the run's realized step counts; inactive padded interiors and cross-request cache reuse are outside every ✅. "
              "DECISIONS-2 §2: 'Existing active-reference results remain valid within their stated scope. They do not already demonstrate this lifted, fully padded baseline.' "
              "The decided baseline is the FULL FIXED Program with one distinguished inactive value ⊥ per domain (D^⊥ = D ⊔ {⊥}, enc(Active v) = 0‖v, enc(⊥) = 1‖0^w; "
              "nothing leaves the verification population; inactive assignments fixed pre-challenge by rule) — NOT implemented (padding owner stoch, LIFTING-SPEC.md by 00:00Z). "
              "An active-reference green is evidence for the lifted-padded row of the same shape only for its active projection; no historical green is relabelled; every "
              "cell is re-derived under versioned Definitions when that baseline lands (the padded Program's digest differs).")


def qualification_citations(recs: R.Records) -> str:
    """[COORD M-0451 (3)] the C2 qualification records cited by displayed Commit ✅ cells, beneath the table by id."""
    cites: dict[tuple, list[str]] = {}
    for row in R.sorted_rows(recs):
        q = (row.get("commit") or {}).get("c2_qualification")
        if isinstance(q, dict) and q.get("id"):
            cites.setdefault((str(q["id"]), str(q.get("board_message") or q.get("path") or "record")), []).append(RD.short_row(row))
    if not cites:
        return ("C2 qualification records cited (COORD M-0451 (3)): none yet — a Commit ✅ rests on a row-own C2 form, (A) sampled replay + boundary linkage or "
                "(B) the oracle-compare of record (M-0481), named in the cell; rows with neither render ◐ 'C2 record pending (rerun queued)'.")
    return "C2 qualification records cited (COORD M-0451 (3); each covering row also passed its own linkage evidence): " + "; ".join(
        f"**{qid}** ({src}) → {', '.join(rows)}" for (qid, src), rows in sorted(cites.items()))


def qualifications(recs: R.Records, events: list[dict], release: tuple[str, str, str | None]) -> list[str]:
    q: list[str] = []
    q.append(SCOPE_LINE)
    q.append(f"Statuses are derived by `finalize.py` (rules {RULES_VERSION}, TABLE-RULES.md frozen) from evidence directories committed on the producer lanes; "
             "the report lane writes no status by hand. A record that fails write-time validation is rejected with the field named and the producer fixes the producer.")
    q.append("One coherent chain of evidence per row (brief §9): the displayed Build/Match/Commit/overhead come from ONE attempt. A Commit-only rerun names the "
             "reused Build/Match artifacts (`reuse.build.program_digest`, `reuse.match.match_digest`) and the finalizer re-derives them and checks identity.")
    q.append("R14-revalidated rows ran on r14-rc1 (revalidation kind) and are labelled ⟨r14⟩ in the caveats; R15 results ran on r14-rc3 or later "
             f"(current integration release {release[0]} {release[1]}"
             + (f", board {release[2]}" if release[2] else "") + "). The R14 workload digests f7ae394f/fc539c83 are superseded on the R15 path (M-0032).")
    q.append("◐ = the stage passed but a required evidence field is missing (named in VALIDATION.md and the row record); ⊘ = infeasible on the named hardware "
             "(brief §13); ⛔ = quarantined by a board ALERT until the fix's RESULT; `not run` rows are panel rows with no attempt. A quarantine may be scoped to cells: "
             "**COORD ALERT M-0488 (23:12Z)** withdraws the Commit (and overhead) cell of every OLMoE, Qwen3-* and Gemma panel row — `native_collect_v2b` gather staging reads a "
             "value after its producer may have overwritten it (late read; the moe/T6-4 root-cause family, rev-cb M-0424/M-0456), so a committed-value PASS on those model "
             "families is not sound — Build/Match render as derived. LIFT RULE (COORD DECISION M-0522 (2)): the Commit ⛔ lifts per row on that row's OWN passing C2 record — "
             "form (B) oracle-compare of record at int-3 (0 mismatch, attribution complete) or (A)+(B) at int-4 — because the oracle-compare is exactly the check that exposed the "
             "defect (moe's int-3 OLMoE B1/B2 records: 1536 / 1632 differing identities, first q_norm/out; the FLUSH-02 fix tree: 6304/6304 equal — M-0532, unreleased, dirty tree → ◐ "
             "until int-4 ships and the rows re-run). Statuses derived beneath a quarantine are named under the table (regressions stay visible, brief §9).")
    q.append("EOS-truncated requests at B≥2 (rev-pm ALERT M-0537 F-pm-24, measured 23:41Z): GM-01 accepts a request that stopped before its cap on the driver's finish_reason alone "
             "(None / 'stop' both pass) without checking that the last served token is one of the model's eos_token_ids, and G7 waives the achieved-concurrency check whenever any "
             "request is eos-truncated. The finalizer renders Match ◐ 'EOS unverified' naming the request(s) for every GM-01 row whose global_match.json lists eos_truncated_requests, "
             "until global's GM-01 EOS check ships and the record is re-checked (a record carrying eos_verified per truncated request passes). Rows where every request reached its cap "
             "are not affected; no displayed row is known wrong.")
    q.append("WEIGHTS PIN (rev-cb ALERT M-0610, measured on r15-int-4 00:09-00:15Z; COORD DECISION M-0614 (1), 00:20Z): a served-model substitution (one weight element "
             "flipped before register_weights) PASSES the C2 form (A) record — sampled replay evaluates F_V with the live weights the committer hashed, its 'weights equal' "
             "check is self-consistency, not authentication against the checkpoint of record. RENDERING OF RECORD: form (A) alone = Commit ◐ 'weights unpinned' until "
             "Build emits weights_root_of_record from the PINNED checkpoint shards (manifests/checkpoints.json revision + sha256; never the live engine) and the run's committed "
             "weights root equals it (int-5 must-carry, manbind + exec; the finalizer reads build_summary.weights_root_of_record and sampled_replay.weights.committed_root). "
             "Form (B) oracle-compare rows KEEP ✅: rev-cb MEASURED (RESULT M-0641, 00:31Z, int-4 a702b2974, SmolLM2 B1) that the flipweight attack FAILS oracle-compare — "
             "result false, 541 identities differ from the Match oracle from layer 5 down_proj onward (sampled replay TRUE, linkage TRUE on the same run: form (A) blind, form (B) "
             "not) — so the Match oracle's weights route is independent of the served engine's; every form-(B) ✅ carries the measurement as its footnote.")
    q.append("DANIEL'S FOUR CORRECTIONS (COORD DECISION M-0776, 02:11Z; finalizer CODE_STAMP 02:33Z) SUPERSEDE THE FORM-(B) RULINGS ABOVE: (1) C2 OF RECORD = form (A) "
             "complete (sampled replay + boundary linkage + targeted negatives) — sufficient on its own; form (B) oracle-compare counts as C2 ONLY if the Match observer's "
             "acquisition/mapping path is independent of the collector's identity→buffer mapping, which rev-cb states in its HANDOFF (finalize.FORM_B_PATH_INDEPENDENT; not yet "
             "stated = fail closed: (B) is a regression test, not C2). RENDERING: the four int-3 rows that were ✅ 'C2: oracle-compare' only (Llama L40S B2 int-3 rerun, Llama A100 "
             "B1 1024/128, Llama A100 B8 256/32, Phi-3 L40S B8 256/32) now render Commit ◐ with BOTH gaps named — 'weights unpinned' (no MAN-09 record on those int-3 runs) and "
             "'form (B) observation-path independence pending (rev-cb)'; each keeps its oracle-compare as a diagnostic. (2) THE WEIGHTS PIN IS UNCONDITIONAL and separate from C2: "
             "every Commit ✅ needs the MAN-09 weights_pin record OK (result true, 0 mismatched, 0 missing, all shards pinned, root_of_record == Build's weights_of_record root) — "
             "regardless of form; the finalizer reads it from value_correspondence.components.weights_pin, commit.weights_pin, or (as native_collect_v2b writes it) per instrumented "
             "run in commit/runs.jsonl commit.weights_pin, every run's pin must hold. Commit ✅ rows of record under M-0776: Llama L40S B2 256/32 (exec r15_0203, int-5 fresh: "
             "form (A) 1272/1272 ×3, linkage 34/34 ×3, MAN-09 98 compared 0 mismatched), Gemma-2-2B L40S B1 256/15 (dense r15_0154, int-5: form (A) complete ×3, MAN-09 211 "
             "compared 0 mismatched; the M-0488 Commit quarantine lifts on this row-own record per M-0522 (2)), Llama L40S B8 256/32 (exec, int-5). (3) COMPILED: the M-0285 hold is "
             "LIFTED — Execution=compiled rows render Match ✅ (P_compiled) once the record pins runtime ↔ compiled-artifact identity (the kernel_set_sha256 of the code objects "
             "loaded by the RUNNING process == compiled_manifest.json kernel_set_sha256 the Program was generated from) + noninterference (tokens equal across two compiled arms); "
             "no eager-equivalence proof is required and the 'list every unconstrained boundary' caveat is withdrawn; no compiled row on file carries a P_compiled manifest with the "
             "runtime pin yet (compiled M-0779 / rel int-6b M-0787: the L40S bi-compiled workloads became runnable as declared at 02:20Z), so every compiled Match stays ◐ naming the "
             "missing identity field. Build on a compiled row stays ◐ while rev-pm M-0758 (F-pm-29: the P_compiled digest skips weight arguments) stands. (4) STOCHASTIC: M-0571 (3) "
             "confirmed — no rendering change; the four B8 stochastic rows stay structural ◐ 'derived split geometry pending'.")
    q.append("COMMIT TIMEOUT (manbind TIP M-0797 on exec r15_0038, Mistral-7B L40S B1 1024/128, int-4): rc=124 is the runner's COMMIT_TIMEOUT — the stage did not COMPLETE, no verdict "
             "was written and nothing was measured wrong (pair 0 on record: sampled replay COMPLETE 1366/1366 equal, 0 mismatch). The finalizer renders an incomplete measurement ◐ "
             "'Commit did not complete — runner timeout' with the finished pairs named, not ❌ (brief §9: ❌ is a measured failure); cause = the int-4 tree's SERIAL replay evaluator "
             "(773 Gemm_v1 VUs × 2.2 s); the forked pool of MAN-06b is in int-5/int-6 — the Commit stage reruns there. The earlier rc3 ◐ (+36.0 %) is superseded by this newer attempt.")
    q.append("COORD DECISION M-0721 (1) (01:37Z; ACK M-0741; finalize.c2_either_form_holds, 7f0e6e117): C2 OF RECORD = [form (A) complete + 0 mismatch] OR [form (B) complete "
             "(0 mismatch, attribution complete at B≥2) + form (A) 0 mismatch with EVERY unevaluated stratum NAMED (strata_unevaluated / not_evaluated[].why / not_evaluable_by_family)]. "
             "A commit verdict that FAILED CLOSED on 'sampled replay partial / strata not evaluated' ALONE is overridden under that rule (footnoted: the over-strict checker rule, "
             "COORD M-0721 (1)) and the cell reads '✅ C2: oracle-compare; replay partial (N strata not evaluable, named)'; one measured mismatch anywhere, boundary linkage not true, "
             "or unevaluated strata WITHOUT names, and nothing is overridden. C1/C3 still gate the cell (MAN-05 manifest-verify record, coverage, openings, retention). "
             "M-0721 (4): a compiled row's Build is ✅ (P_compiled) when compiled_manifest.json carries the DATAFLOW section (wiring_declared true, n_undefined 0, complete true — "
             "compiled b481a02b0) AND rev-pm's N1 re-check is closed — it is NOT (rev-pm M-0758 02:05Z on 2b6573893: F-pm-29 stands, exchanging the weight argument between two "
             "fused_add_rms_norm launches leaves program_digest + manifest_digest unchanged), so Build renders ◐ naming that finding (finalize.P_COMPILED_DATAFLOW_OPEN_FINDING) until "
             "rev-pm closes it; Match ◐ (M-0285 (a)); Commit per record with the form named. Both compiled B8 dirs are still rejected at write time (TIP M-0752: reuse path + "
             "compiled_manifest.json + commit/verdict.json + the C2 record files missing from the dir). "
             "MAN-09 at B≥2 mixed shape (hopper M-0723 / exec M-0727 / int-6 4ef1198b6): a verdict FAIL 'weights_pin: record built for Program X != Program of record' with 0 "
             "mismatched fields renders ◐ 'weights unpinned' (a pin that is not OF RECORD for the Program is an absent pin, M-0614 (1)), never ❌; a pin naming served weights that "
             "differ from the checkpoint stays ❌. Visible now: Llama L40S B8 256/32 ✅ (int-3 form (B)) → ◐ because exec's int-5 Commit-only rerun is the newer release-tree headline "
             "(brief §9; the earlier ✅ is named beneath the table) until the int-6 rerun lands or COORD rules a tooling-only FAIL does not supersede.")
    q.append("Fail-closed C2 gate on INCOMPLETE evaluation = Commit ◐ named, not ❌ (report rule of record 01:36Z, moe BLOCKER M-0715 / hopper M-0696; COORD may overrule): a commit "
             "verdict FAIL whose cause is that the C2 of record could not be ESTABLISHED — MAN-06 sampled replay partial with strata not evaluable (the fused-MoE-fed rows on every "
             "MoE model at int-4b: SiluMul under mlp.experts, RMSNormFusedCuda on the block output; manbind MAN-06e int-6), or the B≥2 attribution gate not value-checked because "
             "the Match snapshotted steps 0,1 only (hopper H100 B8 int-3) — with ZERO mismatches measured anywhere on the record is partial evidence with the gap named (brief), "
             "not an executed failure (COORD 18:09Z: ❌ = coverage/value failure) and not a pass (the gate did not pass). The mapping is narrow: the reason must name the "
             "incompleteness and no mismatch; every C2 block must carry 0 mismatches; one measured difference anywhere → ❌ stands. The cell's footnotes carry the oracle-compare "
             "and the replay's evaluated/equal counts (evidence, not the basis).")
    q.append("Gemma-2 Commit ◐ 'C2 pending replay producer fix' (dense BLOCKER M-0619 / FIX_READY M-0630 35ec6d8ad — in manbind's int-5 list M-0643/M-0654 — / RESULT M-0652): on the "
             "int-4 tree the C2 form (A) sampled replay of the Gemma B1 256/15 Commit-only rerun (r15_0005) FAILS with every mismatch the lm_head Gemm_v1 evaluated against "
             "logits_processor/out (15 over 3 pairs = exactly the logits strata; 1905/1920 other sampled VUs equal; boundary linkage 15/15 ×3) because Gemma-2 soft-caps the logits "
             "AFTER the Gemm and the replay reads the pre-cap producer — the MAN-07 class (wrong producer read) one level up; form (B) on the same run is complete and equal "
             "(5535/5535 ×3, 0 ambiguous; MAN-05 verify OK). Rendered ◐, not ❌ (no value defect measured) and not ✅ (form (A) did not pass; at int-4 the M-0488 quarantine lifts "
             "only on (A)+(B) — COORD M-0522 (2)); the mapping is narrow — soft-capped-logits families only, every listed mismatch on logits_processor/out; the FAIL record stays on "
             "the row as the negative fixture. Decided by the Commit-only rerun on a tree carrying 35ec6d8ad.")
    q.append("Structural ◐ by board decision (COORD M-0614 (4)): the four B8 stochastic rows (Llama-3.2-1B / Qwen2.5-1.5B × L40S / H100, 1024/128 temperature=0.8 top_p=0.95) "
             "render ◐ in every cell — 'derived split geometry pending (stoch M-0571 (3) / M-0579: S(s) as a derived input of the workload Program, 6-10 h across stoch + global)' — "
             "an honest structural status, not infeasible; the R13 ❌ they carried (Build under the pre-r14-v1 standard) is named beneath the table, not displayed as the verdict.")
    q.append("Qwen2.5 Commit ◐ 'C2 pending MAN-07 rerun' (exec M-0528 / rev-cb M-0551 / hopper M-0592, manbind FIX_READY MAN-07 M-0588, 00:08Z): on the int-3 trees the C2 "
             "oracle-compare of every Qwen2.5 row FAILS with every differing identity on *.self_attn.qkv_proj/0 (896 = 28 layers × 32 steps at B1 256/32; 3640 at B2 1024/128; "
             "56 on H100) because the manifest of record names the PRE-bias Gemm instance as the producer of member 0 of a biased linear while the collector commits the module's "
             "post-bias output — a checker REFERENCE defect (the committed bytes are the model's value and equal R14's root), not a measured value defect. The finalizer renders "
             "such a row Commit ◐ with the FAIL record on the row (negative fixture), NOT ❌ (nobody measured the commitment wrong) and NOT ✅ (never C2-checked against the right "
             "instance); the mapping is narrow — biased-linear model families only (Qwen2.5), every listed mismatch on qkv_proj/0; anything else stays ❌. Decided by the "
             "Commit-only rerun on a MAN-07 tree (int-4b/int-5).")
    q.append("B≥2 Build/Match require the composed workload Program (GP-01) and the global Match verdict (GM-01, G1..G8 all PASS, same-attempt Program and manifest "
             "digests — M-0034). The row_pod GP-01/GM-01 hooks shipped in r15-int-2 (rel M-0305); B≥2 attempts that ran on rc3/int-1 without a hand-run compose render ◐ "
             "with the missing artifact named until re-run on int-2 or later.")
    q.append("MAN-05 (manbind `required_manifest verify`, rev-cb M-0340/M-0341; shipped in r15-int-3 — rel M-0415): a Commit attempt on a tree that carries the tool renders ◐ "
             "'MAN-05 record missing' unless `commit/manifest_verify.json` (schema verity-sweep/manifest-verify/v1, ok true, used_digest == the manifest of record) is on "
             "record; attempts on earlier trees carry the footnote 'manifest pinned by digest recompute only'. The digest is recomputed with the PRODUCING tree's "
             "required_manifest (a report-tree-only mismatch is ◐ named, never ❌ 'edited').")
    q.append("Rows whose execution is not `bi-eager` (`*__bi-compiled`) need an execution attestation (effective enforce_eager / cudagraph mode in the run header — "
             "M-0040); without it Build/Match are ◐, and an eager attestation on a compiled row is ❌ ('eager called compiled', PLAN §3).")
    q.append("Compiled rows (`*__bi-compiled`) — Daniel's DECISIONS-2 §7 (22:01Z, M-0356 (4); supersedes the 'compiled must equal eager' framing of M-0178/M-0285 (a)) and "
             "COORD M-0380: P_compiled ≠ P_eager is NOT a failure; the reference is PRODUCTION DEFAULT compiled at the declared specialization; the path is S1 (derive P_compiled "
             "mechanically from the uninstrumented compiled implementation + Inductor artifacts + registered numerics, applicability pinned) → S2 (bit-exact replay of Program "
             "SEGMENTS against the finest noninvasive checkpoints the implementation materializes) → S3 (R(P_compiled, Q) → remaining VU values, invasive changes last). The ◐ "
             "texts stay until S1/S2 evidence exists (M-0356 (4)): **Build ◐** 'no Program applicable to compiled execution' (the eager Program's applicability must not claim "
             "bi-compiled — M-0256 refutes it numerically; P_compiled from S1 is the path to Build ✅); **Match** never graded from GM-01 alone (rev-pm M-0266: an eager "
             "fold under a compiled header passes GM-01; a real compiled record attributes 0 fold instances) — the M-0285 (a) HOLD is LIFTED by Daniel M-0776 (3): Match ✅ (P_compiled) "
             "iff the record pins runtime ↔ compiled-artifact identity (the running process's kernel_set_sha256, RunHeader.profile.compilation, == compiled_manifest.json kernel_set_sha256) "
             "+ noninterference (tokens equal across two compiled arms) + the EXEC-02 effective_execution read-back; otherwise ◐ naming the missing identity field (no record on file "
             "carries both yet); **Commit ◐** 'instrumentation perturbs numerics' — NONINTERFERENCE is a separate test (M-0285 (b) in "
             "the DECISIONS-2 form): instrumented vs PRODUCTION DEFAULT compiled at the strongest checkpoints across repeated replays / shape specializations, not tokens alone. "
             "`emulate_precision_casts` (M-0348) is a DIFFERENT specialization (M-0380): not a panel row, never displayed as one, no relabel — an added row only via a 7-field note.")
    q.append("C2 of record — Daniel's DECISIONS-2 §4 (22:01Z, M-0356 (1); supersedes COORD M-0111 (1)(2) and M-0286): **small stratified SAMPLED EXACT VU REPLAY** from "
             "authenticated committed values (F_V(τ(in V)) = τ(out V) bit-for-bit via canonical references; strata: mechanisms, requests, ranks, early/late steps, lifetime "
             "boundaries; population, n, strata, seed, applicability recorded) + **ORDERED BOUNDARY LINKAGE** (prescribed inputs / designated outputs tied to their interfaces — "
             "the F-cb-23 per-(request, engine step) token gate) + **TARGETED NEGATIVES** on record (equal-shape swap, wrong output member, omitted rank, post-overwrite read; "
             "reusable with regression). NOT required: a full oracle-compare of every identity, a historical known-good root, an independent reference run; MATCH_SNAP_STEPS=all "
             "is no longer a ✅ prerequisite (rev-cb M-0370). A same-collector reproduction is reproducibility only. Finalizer: Commit ✅ from a `sampled_replay` component "
             "{result true, population, n, strata, seed} + `boundary_linkage.result` true at EVERY B, B=1 included (rev-cb M-0411 O1; `linkage` read as alias) + negatives on "
             "record; `unreplayable[]` strata (families without a registered relation) are footnoted by name with counts and bound the claim. The sampled-replay mechanism "
             "is NOT in r15-int-3 (rel M-0415: int-4+; manbind FIX_READY under rev-cb review M-0394/M-0411) — no row can newly reach Commit ✅ under the DECISIONS-2 "
             "standard before it ships; rows whose only C2 is a same-tree reproduction (Llama A100 B1 "
             "1024/128, Mistral B1 1024/128) render ◐ 'C2 of record missing' and return to ✅ when a sampled-replay + boundary_linkage record exists (a Commit-ONLY rerun in the same attempt dir with the "
             "sampler, ~1 min per row — committed bytes are NOT retained on disk, exec M-0423). **RENDERING RULE OF RECORD (COORD DECISION M-0451 (3), 22:54Z, effective "
             "now):** Commit ✅ iff C1 ∧ C3 ∧ ([the row's OWN sampled_replay + boundary_linkage record] OR [a cited QUALIFICATION record (M-0451 (2)(a): one sampled-replay run per "
             "mechanism/domain — collector build, execution profile, hardware class, model family, B class, mechanisms) whose domain covers the row AND the row's own linkage "
             "evidence passed — F-cb-23 gate at B≥2, tokens/prompt linkage at B=1]); the qualification is cited beneath the table by id. **AMENDMENT M-0481 (23:10Z):** a row-own C2 "
             "record is EITHER (A) sampled exact replay + exhaustive boundary linkage OR (B) the ORACLE-COMPARE OF RECORD — every committed value compared bit-for-bit against the "
             "same run's independent Match observation (`--match-dir`): result true, complete (MATCH_SNAP_STEPS=all, no step unsnapshotted), 0 differing, 0 unattributed, the "
             "F-cb-23 attribution gate at B≥2, token families among the compared; graded fail-closed per INSTRUMENTED arm (arms AND-ed) from the oracle_compare record FILE "
             "(commit/runs.jsonl arms or a commit/oracle_compare.json extract — an attempt.json summary alone is not the chain of evidence). The Commit cell names its form: "
             "'✅ C2: sampled replay' | '✅ C2: oracle-compare' | '✅ C2: qualification <id>'. Otherwise Commit ◐ 'C2 record pending (rerun queued)'. The legacy TP value_check "
             "oracle (hook-time clone + re-execution, not complete-vs-Match) and the pre-M-0481 M-0111 forms (independent root-compare at B=1, oracle-compare without the completeness / "
             "attribution fields) and a same-tree reproduction are DIAGNOSTICS named on the "
             "row, not the basis of a ✅ — the eight rows formerly ✅ on those forms (R14 rc1 ×6, Qwen2.5 L40S B1 256/32, Llama H100 B1 1024/128) render Commit ◐ under this rule "
             "with their Build/Match ✅ and measured overhead unchanged (re-derived by the same finalizer). M-0451 (4): after int-4 every BENCHMARKED row gets a Commit-ONLY rerun "
             "with the sampler so each ✅ has its own record by the 01:10Z snapshot; the qualification citation is a bridge.")
    q.append("Within-step order (Daniel approved 21:33Z, M-0281 — final): a request's per-step instances are matched as a DAG (dependency-safe permutations of the fold's "
             "emission order are non-semantic); canonical circuit ordering/addressing is unchanged; every wiring/arrival/output mutation still FAILs. Staggered-arrival rows "
             "render Match ✅ from the release carrying 6a2cd7f5d/056ad8431 with `--within-step dag`.")
    q.append("TP≥2 Match (Daniel approved 21:33Z, M-0282 — final): per-rank Program/fold correspondence + exact collective linkage + token equality, with the GLOBAL "
             "Program's wiring connecting each rank's partials to the collective result the other rank consumes (GP-01 at tp≥2). TP2 rows stay ◐ until all pieces are "
             "implemented (collective-level Match today; the per-rank instance-sequence Match is the missing field).")
    q += [x for x in RD.provenance_notes(recs) if str(x).strip()]
    q += [x for x in RD.commit_fail_notes(recs) if str(x).strip()]
    gaps = [x for x in RD.overhead_evidence_gaps(recs) if str(x).strip()]
    if gaps:
        q.append("Overhead cells whose evidence record lacks a required field (the number is shown; the field is named): " + "; ".join(g.lstrip("- ") for g in gaps))
    return q


def blockers(recs: R.Records, events: list[dict]) -> list[str]:
    out: list[str] = []
    # open failures that block rows (records)
    clusters = R.cluster_failures(recs) if hasattr(R, "cluster_failures") else {}
    blocked: dict[str, list[str]] = {}
    for row in R.sorted_rows(recs):
        if RD.partition(recs, row) == "blocked" and row.get("status") not in ("NOT_RUN", "QUEUED"):
            bc = R.row_blocking_cluster(recs, row, clusters) if hasattr(R, "row_blocking_cluster") else None
            blocked.setdefault(str(bc or "unnamed"), []).append(RD.short_row(row))
    for k, rows in sorted(blocked.items()):
        out.append(f"- **{k}** blocks {len(rows)} row(s): " + ", ".join(rows[:8]) + (" …" if len(rows) > 8 else ""))
    nr = [RD.short_row(r) for r in R.sorted_rows(recs) if r.get("status") in ("NOT_RUN", "QUEUED") or not (r.get("attempts") or [])]
    if nr:
        out.append(f"- **not run** ({len(nr)} panel rows without an attempt): " + ", ".join(nr))
    bl = board_section(events, ("BLOCKER",))
    if bl:
        out.append("- **Open BLOCKER posts on the board** (newest last; superseded posts hidden):")
        out += ["  " + l for l in bl]
    al = board_section(events, ("ALERT",))
    if al:
        out.append("- **ALERT posts** (rows named in an ALERT are quarantined until the fix's RESULT):")
        out += ["  " + l for l in al]
    return out or ["- none recorded"]


def reproduction(release: tuple[str, str, str | None], branch: str, commit: str) -> list[str]:
    return [
        f"- Records + evidence: branch `{branch}` @ `{commit}` (this snapshot); release under test {release[0]} `{release[1]}`.",
        "- Re-derive a row from its evidence directory: `python -m verity_capture.sweep.finalize <row_id> <evidence_dir> --kind execution --lane <lane> --pod <pod>` "
        "(`--dry-run` prints the derivation; a rejected record names the field).",
        "- Re-render everything from the records: `python -m verity_capture.sweep.snapshot` (temp files + rename; archive under `out/gen/sweep/archive/<utc>/`).",
        "- Validate the records: `python -m verity_capture.sweep.validate --root out/gen/sweep` (VALIDATION.md lists violations and missing evidence).",
        f"- Board: `{BOARD_COPY}/BOARD.md` (+ `events/`) is the durable copy of the R16 board `/private/tmp/r16/board`, refreshed every snapshot.",
    ]


def report_md(recs: R.Records, c: dict, now_utc: str, events: list[dict], release: tuple[str, str, str | None], prev: dict | None,
              states: dict[str, dict], src: tuple[str, int], log_path: str, branch: str, commit: str, board_copy: str | None) -> str:
    pc = RD.partition_counts(recs)
    n_nr = sum(1 for r in recs.rows.values() if r.get("status") in ("NOT_RUN", "QUEUED") or not (r.get("attempts") or []))
    n_q = sum(1 for s in states.values() if s.get("quarantined"))
    r16 = sum(1 for s in states.values() if s.get("campaign") == "r16")
    r15 = sum(1 for s in states.values() if s.get("campaign") == "r15")
    r14 = sum(1 for s in states.values() if s.get("campaign") == "r14")
    L: list[str] = []
    L.append(f"# Verity R16 — live report — {now_utc}")
    L.append("")
    L.append(f"Rendered by the report lane from `out/gen/sweep/rows/*.json` (one snapshot; source digest `{src[0][:16]}…` over {src[1]} record files). "
             f"Rules `{RULES_VERSION}`; integration release **{release[0]}** `{release[1]}`. Board copy: `{board_copy or BOARD_COPY}/BOARD.md`.")
    L.append("")
    L.append("## 1. Current summary")
    L.append("")
    L.append(f"- Panel: **{c['rows']} rows**; complete (Build ✅ Match ✅ Commit ✅ overhead ✅) **{pc['complete']}**, partial ◐ **{pc['partial']}**, "
             f"blocked / not reached **{pc['blocked']}**, of which not run {n_nr}" + (f", quarantined ⛔ {n_q}" if n_q else "") + ".")
    L.append(f"- Stage counts (attempted rows {c['rows'] - n_nr}): Build {RD._n(c, 'build')} ✅ / {c['build_fail']} ❌ · Match {RD._n(c, 'match')} ✅ / {c['match_fail']} ❌ · "
             f"Commit {RD._n(c, 'commit')} ✅ / {c['commit_fail']} ❌ · overhead {RD._n(c, 'bench')} measured.")
    L.append(f"- Provenance of the displayed attempts: R16 {r16}, R15 (rc3+) {r15}, R14-revalidated (rc1) {r14}, other/none {len(states) - r16 - r15 - r14}. "
             f"Model families with a complete row: {', '.join(sorted(pc['families_complete'])) or 'none'}.")
    L.append("- Changes since the previous snapshot:")
    L += ["  " + l for l in diff_states(prev, states)]
    L.append("")
    L.append("## 2. Configuration support matrix")
    L.append("")
    L.append(RD.matrix_table(recs))
    L.append("")
    L.append("✅ demonstrated · ◐ partial evidence (stage PASS, evidence field missing; see VALIDATION.md) · ❌ tested, fails · ⊘ infeasible on this hardware · "
             "⛔ quarantined (board ALERT) · — not tested / not reached. Overhead = `+x.x% (n rounds)` from the row's interleaved control/treatment rounds.")
    L.append("")
    L.append(SCOPE_LINE)
    L.append("")
    L.append("## 3. Important qualifications")
    L.append("")
    L += [f"- {q}" if not q.startswith("- ") else q for q in qualifications(recs, events, release)]
    L.append("")
    L.append("## 4. Reusable fixes and rows unlocked")
    L.append("")
    L.append(RD.block_fixes(recs))
    fr = board_section(events, ("FIX_READY",))
    if fr:
        L.append("")
        L.append("FIX_READY posts on the board (integration pending unless a RELEASE names them):")
        L.append("")
        L += fr
    L.append("")
    L.append("## 5. Remaining blockers")
    L.append("")
    L += blockers(recs, events)
    L.append("")
    L.append("## 6. Pending decisions")
    L.append("")
    dec = board_section(events, ("DECISION",))
    L += dec or ["- no DECISION posts on the board"]
    L.append("")
    L.append("## 7. Tested revisions, reproduction, backup, fleet state")
    L.append("")
    L.append(RD.block_revisions(recs, now_utc[11:16] + "Z" if len(now_utc) >= 16 else now_utc))
    L.append("")
    L.append("Reproduction and backup:")
    L.append("")
    L += reproduction(release, branch, commit)
    L.append("")
    L.append("Releases on the board:")
    L.append("")
    L += board_section(events, ("RELEASE",)) or ["- none"]
    L.append("")
    L.append(f"Fleet state (coordinator `{log_path}`):")
    L.append("")
    L += fleet_state(log_path)
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------
# main


def git(repo_root: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", repo_root, *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out/gen/sweep")
    ap.add_argument("--board", default=BOARD_DEFAULT)
    ap.add_argument("--log", default=LOG_DEFAULT)
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--no-board", action="store_true", help="do not render/copy the board")
    ap.add_argument("--campaign", default="r14", help="render.py acceptance semantics: r14 = r14-v1 (R13 passes show ◐/☑ until the finalizer revalidates them); unchanged for R15")
    a = ap.parse_args(argv)
    root = a.root
    repo_root = os.path.abspath(os.path.join(root, "..", "..", ".."))
    RD.R.CAMPAIGN = a.campaign
    now = datetime.now(timezone.utc)
    now_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_hm = now.strftime("%H:%MZ")
    utc_dir = now.strftime("%Y%m%dT%H%M%SZ")

    # ONE snapshot of the records
    recs = R.load(root)
    digest, n_src, _names = source_digest(root)
    events = board_events(a.board)
    release = latest_release(events)
    prev = None
    cur_path = os.path.join(root, "CURRENT.json")
    if os.path.exists(cur_path):
        try:
            with open(cur_path, encoding="utf-8") as fh:
                prev = json.load(fh)
        except Exception:  # noqa: BLE001
            prev = None
    branch = git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    commit = git(repo_root, "rev-parse", "--short", "HEAD") or "?"
    board_copy = None if a.no_board else render_board(a.board, repo_root)

    state = RD.load_state(root)
    status_txt, c = RD.status_md(recs, state, now_hm, [])
    states = row_states(recs)
    matrix = (f"# Configuration support matrix — {now_hm} — {c['rows']} row record(s)\n\n{RD.matrix_table(recs)}\n\n"
              "✅ demonstrated · ◐ partial evidence (stage PASS, evidence field missing; see VALIDATION.md) · ❌ tested, fails · ⊘ infeasible · ⛔ quarantined · — not tested / not reached. "
              "Overhead = `+x.x% (n rounds)` sweep measurement from the row's interleaved control/treatment rounds. Generated from `rows/*.json` only "
              f"(snapshot {now_utc}, source digest {digest[:16]}…, CURRENT.json).\n\n{SCOPE_LINE}\n\n{qualification_citations(recs)}\n")
    validation = RD.validation_md(recs, now_hm)
    report = report_md(recs, c, now_utc, events, release, prev, states, (digest, n_src), a.log, branch, commit, board_copy and os.path.relpath(board_copy, repo_root))
    # FINAL-REPORT.md: render.py rewrites only its auto blocks inside the existing file → do that on a temp copy, then rename
    fr_path = os.path.join(root, "FINAL-REPORT.md")
    fr_tmp = None
    if os.path.exists(fr_path):
        fd, fr_tmp = tempfile.mkstemp(prefix=".snap-fr-", suffix=".tmp", dir=root)
        os.close(fd)
        shutil.copyfile(fr_path, fr_tmp)
        RD.render_report(recs, c, now_hm, fr_tmp, a.campaign)

    pc = RD.partition_counts(recs)
    archive_rel = None if a.no_archive else os.path.join("archive", utc_dir)
    current = {
        "schema": "verity-sweep/current/v1",
        "campaign": CAMPAIGN,
        "rules_version": RULES_VERSION,
        "generated_at": now_utc,
        "snapshot": os.path.join(root, archive_rel) if archive_rel else None,
        "release": {"tag": release[0], "commit": release[1], "board_message": release[2]},
        "source_records_digest": digest,
        "source_records_n": n_src,
        "panel_size": c["rows"],
        "report_branch": branch,
        "report_commit": commit,
        "counts": {
            "rows": c["rows"], "attempted": c["attempted"], "not_run": sum(1 for r in recs.rows.values() if r.get("status") in ("NOT_RUN", "QUEUED") or not (r.get("attempts") or [])),
            "complete": pc["complete"], "partial": pc["partial"], "blocked": pc["blocked"],
            "build": c["build"], "build_partial": c["build_part"], "build_fail": c["build_fail"],
            "match": c["match"], "match_partial": c["match_part"], "match_fail": c["match_fail"],
            "commit": c["commit"], "commit_partial": c["commit_part"], "commit_fail": c["commit_fail"],
            "overhead": c["bench"], "overhead_partial": c["bench_part"],
            "r16_rows": sum(1 for s in states.values() if s.get("campaign") == "r16"),
            "r15_rows": sum(1 for s in states.values() if s.get("campaign") == "r15"),
            "r14_revalidated_rows": sum(1 for s in states.values() if s.get("campaign") == "r14"),
            "quarantined": sum(1 for s in states.values() if s.get("quarantined")),
            "violations": len(recs.errors()), "advisories": len(recs.advisories()), "missing_evidence_rows": len(recs.missing_evidence),
            "families_complete": sorted(pc["families_complete"]),
        },
        "board": {"events_n": len(events), "last_id": events[-1].get("id") if events else None, "copy": BOARD_COPY},
        "rows": states,
        "files": {"MATRIX.md": sha256_bytes(matrix.encode()), "REPORT.md": sha256_bytes(report.encode()), "VALIDATION.md": sha256_bytes(validation.encode())},
    }
    current_txt = json.dumps(current, indent=1, sort_keys=False) + "\n"

    # write everything: temp + rename, same directory
    atomic_write(os.path.join(root, "MATRIX.md"), matrix)
    atomic_write(os.path.join(root, "REPORT.md"), report)
    atomic_write(os.path.join(root, "VALIDATION.md"), validation)
    atomic_write(os.path.join(root, "STATUS.md"), status_txt)
    if fr_tmp:
        os.replace(fr_tmp, fr_path)
    atomic_write(cur_path, current_txt)
    RD.save_state(root, recs, now_hm)
    if archive_rel:
        ad = os.path.join(root, archive_rel)
        os.makedirs(ad, exist_ok=True)
        for fn in ("MATRIX.md", "REPORT.md", "VALIDATION.md", "STATUS.md", "CURRENT.json", "FINAL-REPORT.md"):
            p = os.path.join(root, fn)
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(ad, fn))
    print(f"[snapshot {now_utc}] rows={c['rows']} complete={pc['complete']} partial={pc['partial']} blocked={pc['blocked']} "
          f"build={RD._n(c, 'build')} match={RD._n(c, 'match')} commit={RD._n(c, 'commit')} overhead={RD._n(c, 'bench')} "
          f"digest={digest[:16]} release={release[0]}@{release[1]} board_events={len(events)} archive={archive_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
