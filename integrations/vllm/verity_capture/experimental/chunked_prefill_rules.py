"""[r8 lane E-cleanup 2026-09-18 — CANDIDATE rules, not gates] the two harness-expectation rules that reject genuinely chunked
prefill cells (`B0-cov_b0_b1_c256_ck64`, `B1-cov_b1_b1_c256_ck64`: G2 0 Unsupported, G4 575/575 resp. 581/581 bit-exact incl.
`Attention_v3{T}` at every position across the chunks, G6 pass; REJECTED only at G1 and G5), re-evaluated with the partial-prefill
rule the chunked lane proposed (`out/gen/window/chunked.md`, WINDOW-20260917 M3 / post-window item 5).

    G1  `noninterference.compare` compares, per step, the SampleEnd ids of the observer run (a) with the hooks run's new tokens (b)
        by request.  At a PARTIAL prefill step (a chunk that does not complete the prompt) vLLM's sampler still runs over the
        request's last logits row and the engine DISCARDS the draw (no output for that request at that step): (a) logs a SampleEnd
        id, (b) reports `{}`.  Rule: a request's SampleEnd id at one of its `partial_prefill_steps` is a discarded sample; it is
        dropped from the comparison AND the request must have no served token at that step in (b).  Everything else (hidden /
        residual / final / logits boundary hashes, tokens vs control, step and layer counts) is unchanged.
    G5  `gates.gate_G5` requires "step 0 (prefill) checked" among the served-token steps of `replay_a.tokens.steps`.  With chunked
        prefill the FIRST served token of a request is produced at its prefill-completion step (`first_token_step`, = 0 when the
        prompt fits one chunk).  Rule: every request's `first_token_step` must be among the checked steps (a generalisation: for an
        unchunked B1 / B8-static cell every `first_token_step` is 0, so the rule reduces to the frozen one).

Provenance of the steps: `capture/tokens.json["timing"]["requests"][rid]` — `partial_prefill_steps`, `first_token_step`,
`token_steps` — written by `verity_capture.chunk_attribution` from the in-process scheduler's `num_computed_tokens`
(`source: scheduler-state`) on every variant run ((a), (b), the control), so the rule is checked against facts the harness
recorded, not against the workload's declared budget.  The (a) and (b) attributions must agree.

Output (`candidate_gates`): `<out>/noninterference.candidate-chunked.json` (the compare document with the rule applied — same
shape, `candidate` block, `PASS` recomputed exactly as `compare()` computes it), `<out>/gates.candidate-chunked.json` = the FROZEN
gate functions (`gates.evaluate_gates`, untouched) run over a copy of the record whose `nonint/noninterference.json` is the
candidate document, plus the G5 result with the step rule replaced, and `<out>/CANDIDATE.md`.  The verdict field is named
`candidate_verdict` and never written to `gates.json` / `card.json` / the cards dir.

    python -m verity_capture.experimental.chunked_prefill_rules --run-dir RUN_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verity_capture import gates as G
from verity_capture import noninterference as NI

SCHEMA_SUFFIX = "+candidate-chunked-prefill"
RULE_G1 = ("discarded-sample rule: a request's SampleEnd id at one of its partial_prefill_steps (chunk_attribution, scheduler-state) is "
           "dropped from the (a) vs (b) sampled-id comparison; the request must have no served token in (b) at that step")
RULE_G5 = "first-token rule: every request's first_token_step (prefill-completion step) is among the checked served-token steps (0 when unchunked)"
_SAMPLED_PROBLEM = re.compile(r"^step (\d+): SampleEnd ids")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


def request_steps(tokens_doc: dict | None) -> dict[str, dict[str, Any]]:
    """`{rid: {partial_prefill_steps: [...], first_token_step: int, token_steps: [...]}}` from a tokens.json `timing.requests`
    block (empty when the run predates chunk attribution: then the rules reduce to the frozen ones)."""
    reqs = ((tokens_doc or {}).get("timing") or {}).get("requests") or {}
    out: dict[str, dict[str, Any]] = {}
    for rid, r in reqs.items():
        if not isinstance(r, dict) or r.get("first_token_step") is None:
            continue
        out[str(rid)] = {"partial_prefill_steps": [int(x) for x in (r.get("partial_prefill_steps") or [])],
                         "first_token_step": int(r["first_token_step"]), "token_steps": [int(x) for x in (r.get("token_steps") or [])]}
    return out


def discarded_sample_rule(step_rows: list[dict], sample_ids: dict[int, list[int]], sample_rows: dict[int, list[str]],
                          new_tokens_b: dict[int, dict[str, list[int]]], attribution: dict[str, dict[str, Any]],
                          served_a: dict[str, list[int]]) -> tuple[list[dict], list[str], int, dict[str, bool]]:
    """The per-step sampled-id comparison of `noninterference.compare` with the discarded-sample rule (pure; the inputs are the
    loaders' outputs).  Returns (rows with `sampled_eq` recomputed, problems, discarded count, per-request served-sequence accounting).

    - `sample_ids[s]` / `sample_rows[s]`: (a)'s SampleEnd ids and their request ids (StepStart.logits_rows) at step `s`;
    - `new_tokens_b[s]`: the hooks run's `{rid: [token]}` at step `s`;
    - `attribution[rid]`: `{partial_prefill_steps, first_token_step, token_steps}` (scheduler-state chunk attribution of the
      observer-side runs); `served_a[rid]`: (a)'s `generated_token_ids`.
    A row is left unchanged (frozen positional comparison) when (a) has no request names for its sampled rows."""
    partial = {rid: set(v["partial_prefill_steps"]) for rid, v in attribution.items()}
    kept_by_req: dict[str, list[tuple[int, int]]] = {}
    problems: list[str] = []
    rows_out: list[dict] = []
    discarded_total = 0
    for row in step_rows:
        if not isinstance(row, dict) or "error" in row:
            rows_out.append(row)
            continue
        s = int(row["step"])
        sa = sample_ids.get(s)
        rows_a = sample_rows.get(s)
        by_req_b = {r: int(v[0]) for r, v in (new_tokens_b.get(s) or {}).items() if v}
        r2 = dict(row)
        if sa is None or rows_a is None or len(rows_a) != len(sa):
            rows_out.append(r2)                      # positional comparison: no request names -> the rule cannot apply; row unchanged
            continue
        by_req_a = {r: int(t) for r, t in zip(rows_a, sa)}
        discarded = {r: t for r, t in by_req_a.items() if s in partial.get(r, set())}
        for r in discarded:
            if r in by_req_b:
                problems.append(f"step {s}: request {r} has a served token in (b) at one of its partial-prefill steps (chunk attribution says the prompt was incomplete)")
        kept = {r: t for r, t in by_req_a.items() if r not in discarded}
        for r, t in kept.items():
            kept_by_req.setdefault(r, []).append((s, t))
        discarded_total += len(discarded)
        r2["discarded_samples"] = discarded
        r2["sampled_eq_by_request"] = kept == by_req_b
        r2["sampled_eq"] = r2["sampled_eq_by_request"]
        if not r2["sampled_eq"]:
            problems.append(f"step {s}: SampleEnd ids by request {kept} != hooks-run new tokens {by_req_b}" + (" (after discarding partial-prefill samples)" if discarded else ""))
        rows_out.append(r2)
    # served-sequence accounting (stronger than the frozen rule): per request, the KEPT (a) SampleEnd ids in step order are exactly
    # the tokens (a) served, at exactly the attributed token_steps -- so a discarded draw is never a served token and every served token
    # is a kept draw.
    accounting: dict[str, bool] = {}
    for rid, pairs in sorted(kept_by_req.items()):
        pairs.sort()
        ok_seq = [t for _, t in pairs] == served_a.get(rid)
        ok_steps = (rid not in attribution) or [s for s, _ in pairs] == attribution[rid]["token_steps"]
        accounting[rid] = ok_seq and ok_steps
        if not ok_seq:
            problems.append(f"request {rid}: kept SampleEnd ids {[t for _, t in pairs][:6]}… != (a) generated_token_ids {(served_a.get(rid) or [])[:6]}…")
        if not ok_steps:
            problems.append(f"request {rid}: kept SampleEnd steps {[s for s, _ in pairs][:6]}… != attributed token_steps {attribution[rid]['token_steps'][:6]}…")
    return rows_out, problems, discarded_total, accounting


def candidate_noninterference(run: Path) -> dict:
    """`nonint/noninterference.json` with the discarded-sample rule applied to the per-step sampled-id comparison (only that)."""
    nd = run / "nonint"
    rec = _read(nd / "noninterference.json")
    a_dir, b_dir = nd / "a_observer", nd / "b_hooks"
    a_tokens = _read(a_dir / "tokens.json") if (a_dir / "tokens.json").exists() else None
    steps_a = request_steps(a_tokens)
    # the attribution is the observer-side scheduler-state writer's; the capture and control runs of the same record carry it too and
    # must agree with (a).  The hooks run (b) does not write it (its tokens.json has no timing.requests) -- its side of the rule is its
    # own new_tokens (no served token at a partial-prefill step), checked below.
    problems: list[str] = []
    agreement: dict[str, Any] = {}
    for other in ("capture", "control"):
        p = run / other / "tokens.json"
        if p.exists():
            steps_o = request_steps(_read(p))
            agreement[other] = steps_o == steps_a
            if steps_o != steps_a:
                problems.append(f"chunk attribution of (a) differs from the {other} run's: {steps_a} != {steps_o}")
    b_tokens = _read(b_dir / "tokens.json") if (b_dir / "tokens.json").exists() else None
    steps_b = request_steps(b_tokens)
    agreement["b_hooks"] = (steps_b == steps_a) if steps_b else "absent (hooks-run tokens.json carries no chunk attribution)"
    if steps_b and steps_b != steps_a:
        problems.append(f"chunk attribution differs between (a) and (b): {steps_a} != {steps_b}")
    partial = {rid: set(v["partial_prefill_steps"]) for rid, v in steps_a.items()}
    served_a = {str(r.get("request_id")): [int(t) for t in (r.get("generated_token_ids") or [])] for r in ((a_tokens or {}).get("requests") or [])
                if isinstance(r, dict)}
    hashes_a, pa = NI.observer_boundary_hashes(a_dir)
    hashes_b, pb = NI.hooks_boundary_hashes(b_dir)
    problems += [p for p in pa + pb if p not in (rec.get("problems") or [])]      # new loader problems only; the recorded ones stay below
    rows_out, rule_problems, discarded_total, accounting = discarded_sample_rule(
        rec.get("steps") or [], hashes_a["sample_ids"], hashes_a.get("sample_rows") or {}, hashes_b["new_tokens"], steps_a, served_a)
    problems += rule_problems
    kept_problems = [p for p in (rec.get("problems") or []) if not _SAMPLED_PROBLEM.match(str(p))]
    problems = kept_problems + problems
    n_cmp, n_eq = int(rec.get("hash_comparisons") or 0), int(rec.get("hash_equal") or 0)
    token_eq = rec.get("tokens_equal_accepted") or {}
    exp = rec.get("expected_steps")
    layers_ok = rec.get("expected_layers") is not None and rec.get("layers") == rec.get("expected_layers")
    ok = (not problems and n_cmp > 0 and n_eq == n_cmp and bool(token_eq) and all(token_eq.values()) and exp is not None
          and len(rows_out) == int(exp) and layers_ok)
    doc = dict(rec)
    doc["schema"] = str(rec.get("schema", "")) + SCHEMA_SUFFIX
    doc["steps"] = rows_out
    doc["problems"] = problems
    doc["PASS"] = ok
    doc["candidate"] = {"rule": RULE_G1, "utc": _utc(), "source": "verity_capture.experimental.chunked_prefill_rules (r8 lane E-cleanup)",
                        "recorded": {"PASS": rec.get("PASS"), "problems": rec.get("problems")},
                        "chunk_attribution": {"a_observer": steps_a, "agreement": agreement},
                        "discarded_samples_total": discarded_total, "served_sequence_accounting": accounting,
                        "note": "candidate document — not the frozen noninterference.json; PASS recomputed with compare()'s own conjunction"}
    return doc


def candidate_G5(recorded_g5: dict, capture_tokens: dict | None, replay_a: dict | None) -> dict:
    """The recorded G5 result with the frozen "step 0 (prefill) not checked" problem replaced by the first-token rule."""
    problems = [p.strip() for p in str(recorded_g5.get("reason") or "").split(";") if p.strip()]
    steps = [int(x) for x in (((replay_a or {}).get("tokens") or {}).get("steps") or [])]
    rs = request_steps(capture_tokens)
    if not rs:                                        # no chunk attribution: the frozen rule stands and the recorded result is the result
        return {"id": "G5", "name": recorded_g5.get("name", "composition exactness"), "status": recorded_g5.get("status"),
                "reason": recorded_g5.get("reason"), "candidate": {"rule": RULE_G5, "applied": False, "why": "no chunk attribution in capture/tokens.json"}}
    kept = [p for p in problems if p != "served tokens: step 0 (prefill) not checked"]
    first = {rid: v["first_token_step"] for rid, v in rs.items()}
    missing = sorted({rid for rid, s in first.items() if s not in steps})
    if missing:
        kept.append(f"served tokens: first_token_step not checked for {len(missing)} request(s) {missing[:3]} (first_token_steps {sorted(set(first.values()))}, checked steps {steps[:6]}…)")
    return {"id": "G5", "name": recorded_g5.get("name", "composition exactness"), "status": "fail" if kept else "pass",
            "reason": "; ".join(kept) if kept else None, "candidate": {"rule": RULE_G5, "first_token_steps": first, "checked_steps": steps,
                                                                        "recorded_status": recorded_g5.get("status"), "recorded_reason": recorded_g5.get("reason")}}


def candidate_gates(run_dir: str | Path, out_dir: str | Path) -> dict:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand_ni = candidate_noninterference(run)
    (out / "noninterference.candidate-chunked.json").write_text(json.dumps(cand_ni, indent=1) + "\n")
    recorded = _read(run / "gates.json")
    # the frozen gate functions over a copy whose nonint/noninterference.json is the candidate document (nothing else differs)
    with tempfile.TemporaryDirectory(prefix="cand-chunked-") as td:
        copy = Path(td) / run.name
        shutil.copytree(run, copy, ignore=shutil.ignore_patterns("*.gz", "*.npy", "values", "traces", "log.jsonl*"))
        (copy / "nonint" / "noninterference.json").write_text(json.dumps(cand_ni, indent=1))
        results = G.evaluate_gates(copy)
    by_id = {r.id: r for r in results}
    cap_tokens = _read(run / "capture" / "tokens.json") if (run / "capture" / "tokens.json").exists() else None
    replay_a = _read(run / "replay_a.json") if (run / "replay_a.json").exists() else None
    g5 = candidate_G5(recorded["gates"]["G5"], cap_tokens, replay_a)
    rows = []
    for r in results:
        rec = recorded["gates"].get(r.id) or {}
        if r.id == "G5":
            rows.append({"id": "G5", "name": r.name, "recorded": rec.get("status"), "candidate": g5["status"], "reason": g5["reason"],
                         "rule": RULE_G5 if g5["status"] != rec.get("status") else None})
        else:
            rows.append({"id": r.id, "name": r.name, "recorded": rec.get("status"), "candidate": r.status, "reason": r.reason,
                         "rule": RULE_G1 if r.id == "G1" and r.status != rec.get("status") else None})
    core = {row["id"]: row["candidate"] for row in rows if row["id"] in G.GATES}
    cand_verdict = "REJECTED" if "fail" in core.values() else ("ACCEPTED" if all(v == "pass" for v in core.values()) else "INCOMPLETE")
    doc = {"schema": "verity-capture/gates/candidate-chunked-prefill/v1", "utc": _utc(), "run": run.name,
           "recorded_verdict": recorded.get("verdict"), "candidate_verdict": cand_verdict, "gates": rows,
           "rules": {"G1": RULE_G1, "G5": RULE_G5},
           "method": "frozen gates.evaluate_gates over a copy of the record with nonint/noninterference.json := the candidate document; G5 step rule replaced outside the gate (the rule lives in gates.gate_G5)",
           "note": "CANDIDATE — not a verdict.  Nothing written to gates.json / card.json / the cards dir."}
    (out / "gates.candidate-chunked.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = [f"# CANDIDATE gate re-evaluation (chunked-prefill rules) — `{run.name}`", "",
          f"recorded verdict **{recorded.get('verdict')}** → candidate **{cand_verdict}** (not a verdict; see `gates.candidate-chunked.json`)", "",
          "| gate | recorded | candidate | rule applied | reason (candidate) |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        md.append(f"| {row['id']} | {row['recorded']} | **{row['candidate']}** | {'yes' if row['rule'] else '—'} | {(row['reason'] or '').replace('|', '/')[:160]} |")
    md += ["", f"- G1 rule: {RULE_G1}", f"- G5 rule: {RULE_G5}",
           f"- discarded samples: {cand_ni['candidate']['discarded_samples_total']}; chunk attribution agreement: {cand_ni['candidate']['chunk_attribution']['agreement']}",
           f"- served-sequence accounting (kept SampleEnd ids == served tokens, at the attributed token_steps): {cand_ni['candidate']['served_sequence_accounting']}",
           f"- recorded noninterference problems: {cand_ni['candidate']['recorded']['problems']}", f"- candidate noninterference problems: {cand_ni['problems']}", ""]
    (out / "CANDIDATE.md").write_text("\n".join(md))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True, help="lane artefact dir (never the run dir, never the cards dir)")
    a = ap.parse_args(argv)
    doc = candidate_gates(a.run_dir, a.out)
    print(json.dumps({"run": doc["run"], "recorded_verdict": doc["recorded_verdict"], "candidate_verdict": doc["candidate_verdict"],
                      "gates": {r["id"]: (r["recorded"], r["candidate"]) for r in doc["gates"]}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
