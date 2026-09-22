"""tp2_evidence — assemble a TP2 row's evidence directory in the finalizer's layout (R14 lane tp).

    python -m verity_capture.bench.tp2_evidence --build <dir with step_rank{0,1}/ request_rank{0,1}/> --match <dir with tp2_match.json>
        --commit <tp2_commit --out dir> --workload <manifest> --release-tag <tag> --revision <sha> --out <evidence dir>

Writes `stages.txt`, `build_summary.json`, `match_summary.json`, `commit/verdict.json`, `commit/summary.json` from the TP2 tools'
own artifacts (derive_step result.json per rank, tp2_match.json, tp2_commit summary/versions/finalize_pair*/runs.jsonl) and copies
those artifacts beside them.  Nothing here types a status: the finalizer derives the four columns.  Rank honesty: the build summary
carries BOTH ranks' Programs (`step`/`request` = rank 0 for the finalizer's keys, `step_rank1`/`request_rank1` beside them, `tp`
block with every rank's ok flag); the match summary is labelled collective-level (verdict `INTERMEDIATE`, never `PASS`); the commit
verdict carries the per-rank roots, the per-rank identity checks and the labelled provisional `tp_run_root`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--match", default=None)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--release-tag", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--negatives", default=None, help="dir with neg_*/runs.jsonl (omit-rank / omit-rank-partials / relabel / wrong-root / outputs-only) to cite + copy")
    ap.add_argument("--manifest", default=None, help="[R15 TP-04] manbind's merged TP required manifest the Commit ran with (copied to commit/manifest.json)")
    ap.add_argument("--campaign", default="r15")
    ap.add_argument("--lane", default="tp")
    ap.add_argument("--pod", default=None)
    ap.add_argument("--tree-note", default=None, help="attempt.json tree_note: the tree the evidence ran on when it is a lane tree, not the release (finalizer renders the row as lane-tree)")
    ap.add_argument("--tree-commit", default=None, help="[COORD M-0132] commit of the EXECUTING lane tree shipped to the pod (attempt.json executing_tree)")
    ap.add_argument("--tree-branch", default="lane/r15-tp")
    ap.add_argument("--tree-path", default="/workspace/cp/r15-tp")
    ap.add_argument("--tree-dirty", action="store_true", help="the shipped tree had uncommitted changes (the finalizer then refuses to diff it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "commit").mkdir(parents=True, exist_ok=True)
    (out / "build").mkdir(exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row = Path(a.workload).stem
    stages = [f"release {a.release_tag} {now} commit={a.revision} rules_version=r14-v1 tp=2"]
    wl_doc = _load(Path(a.workload))
    wl_reqs = wl_doc.get("requests") or []

    # ---- build: both ranks
    B: dict = {"tp": {"world": 2, "ranks": {}}, "cmd": (f"derive_step --model <repo> --revision <rev> --wrapper step --M <B> --pos0 8 | --wrapper request --LP <LP> --T <T> "
                                                       f"--max-model-len <MML> --workload {a.workload} --tp 2 --rank {{0,1}} (both ranks; TP-aware Program)")}
    ok_all = True
    for r in (0, 1):
        for w in ("step", "request"):
            d = Path(a.build) / f"{w}_rank{r}" / "result.json"
            if not d.exists():                                                   # the pod's derive_step layout: build_<wrapper>_r<rank>/
                d = Path(a.build) / f"build_{w}_r{r}" / "result.json"
            if not d.exists() and (Path(a.build) / f"rank{r}" / f"build_{w}").exists():   # B>=2 TP layout: rank<r>/build_step, rank<r>/build_request_LP*_T*
                d = Path(a.build) / f"rank{r}" / f"build_{w}" / "result.json"
            comps = sorted((Path(a.build) / f"rank{r}").glob("build_request_LP*_T*/result.json")) if (w == "request" and not d.exists()) else []
            if comps:
                # [R15 tp task 3] a B>=2 TP row: rank r's request Programs are its components (one per declared (LP, T)); the rank's Program
                # digest of record = GP-01's workload digest when the compose ran, else manbind's labelled components fold (required_manifest
                # build-global's rule; tp2_commit checks the same fold against the manifest) -- kind recorded, never presented as a workload digest
                from verity_capture.commit.required_manifest import components_fold_digest
                ents = []
                for cj in comps:
                    R = _load(cj)
                    tp_small = {k: v for k, v in (R.get("tp") or {}).items() if k not in ("collectives",)}   # not the per-site collectives list (the result.json copy carries it)
                    ents.append({"dir": cj.parent.name, "ok": bool(R.get("ok")), "LP": R.get("LP"), "T": R.get("T"), "tp": tp_small,
                                 "program_digest": (R.get("program") or {}).get("digest"), "wall_secs_total": R.get("wall_secs_total"), "max_rss_mb": R.get("max_rss_mb")})
                    shutil.copy(cj, out / "build" / f"{w}_rank{r}.{cj.parent.name}.result.json")
                wp = Path(a.build) / f"rank{r}" / "build_workload" / "workload_program.json"
                WP = _load(wp) if wp.exists() else {}
                wd = WP.get("workload_digest")
                if WP:   # [R15 TP-07 / global M-0367] GP-01 composed this rank: keep the compact header beside the evidence (the 17 MB Program with its address map stays on the pod)
                    head = {k: WP.get(k) for k in ("schema", "workload_digest", "manifest_digest", "n_params", "n_root_nodes", "padding_scope", "x09_inside", "representation")}
                    head["n_requests"] = len(WP.get("requests") or [])
                    head["shared_param_index_by_request"] = bool(WP.get("shared_param_index_by_request"))
                    head["source"] = str(wp)
                    (out / "build" / f"rank{r}.workload_program.head.json").write_text(json.dumps(head, indent=1, default=str))
                digs = sorted({e["program_digest"] for e in ents if e["program_digest"]})
                # one component per DISTINCT (LP, T) shape, not per request: equal-shape requests (the B2-equal fixture) share one component;
                # every declared prompt length must have a component and there are never more components than requests
                req_lps = {int(q.get("prompt_len") or q.get("LP") or 0) for q in wl_reqs}
                comp_lps = {int(e["LP"]) for e in ents if e.get("LP") is not None}
                covered = req_lps <= comp_lps and 0 < len(ents) <= len(wl_reqs)
                ent = {"ok": all(e["ok"] for e in ents) and covered, "unsupported": None, "exception": None, "message": "",
                       "program_digest": (wd or components_fold_digest(digs)), "program_digest_kind": ("workload" if wd else "components-fold"),
                       "component_digests": digs, "components": ents, "tp": ents[0].get("tp") if ents else None, "n_components": len(ents),
                       "declared_requests": len(wl_reqs), "gp01_compose": ("ok" if wd else "not run / refused (see build/rank<r>.build_workload.log)"),
                       "cmd": B["cmd"].replace("{0,1}", str(r))}
                lg = Path(a.build) / f"rank{r}" / "build_workload.log"
                if lg.exists():
                    shutil.copy(lg, out / "build" / f"rank{r}.build_workload.log")
                ok_all &= bool(ent["ok"])
                B["tp"]["ranks"][f"{w}_rank{r}"] = {"ok": ent["ok"], "program_digest": ent["program_digest"], "program_digest_kind": ent["program_digest_kind"], "n_components": len(ents)}
                B[f"{w}_rank{r}"] = ent
                if r == 0:
                    B[w] = dict(ent)
                continue
            if not d.exists():
                ok_all = False
                B["tp"]["ranks"][f"{w}_rank{r}"] = {"ok": False, "missing": str(d)}
                continue
            R = _load(d)
            prog = R.get("program") or {}
            art = R.get("artifact") or {}
            ent = {"ok": bool(R.get("ok")), "unsupported": R.get("unsupported"), "exception": None, "message": "", "program_digest": prog.get("digest"),
                   "identity": art.get("identity"), "target": R.get("target"), "applicability": R.get("applicability") or art.get("applicability"),
                   "tp": R.get("tp"), "LP": R.get("LP"), "T": R.get("T"), "M": R.get("M"), "wall_secs_total": R.get("wall_secs_total"), "max_rss_mb": R.get("max_rss_mb"),
                   "sampler": (R.get("sampling") or R.get("profile")), "cmd": B["cmd"].replace("{0,1}", str(r)).replace("<LP>", str(R.get("LP"))).replace("<T>", str(R.get("T")))}
            ok_all &= bool(R.get("ok"))
            B["tp"]["ranks"][f"{w}_rank{r}"] = {"ok": bool(R.get("ok")), "program_digest": prog.get("digest")}
            B[f"{w}_rank{r}"] = ent
            if r == 0:
                B[w] = dict(ent)
            shutil.copy(d, out / "build" / f"{w}_rank{r}.result.json")
            for extra in ("tp_links.json", "inputs.json"):
                q = d.parent / extra
                if q.exists():
                    shutil.copy(q, out / "build" / f"{w}_rank{r}.{extra}")
    B["tp"]["all_ranks_ok"] = ok_all
    B["note"] = "TP2: `step`/`request` mirror rank 0 for the finalizer's keys; rank 1's Programs are `step_rank1`/`request_rank1`; the row's Build is the pair"
    (out / "build_summary.json").write_text(json.dumps(B, indent=1, default=str) + "\n")
    stages.append(f"build {'ok' if ok_all else 'FAIL'} {now} ranks={sorted(B['tp']['ranks'])} digest_r0={str(B.get('step', {}).get('program_digest'))[:16]} digest_r1={str(B.get('step_rank1', {}).get('program_digest'))[:16]}")

    # ---- match: collective-level (labelled; never PASS)
    if a.match and (Path(a.match) / "tp2_match.json").exists():
        M = _load(Path(a.match) / "tp2_match.json")
        shutil.copy(Path(a.match) / "tp2_match.json", out / "tp2_match.json")
        st, pv, tk = M.get("structure") or {}, M.get("provenance") or {}, M.get("tokens") or {}
        ms = {"comparator": "tp2_match/v1 (collective-level TP-aware Match: per-rank collective structure vs Build signature + cross-rank provenance "
                            "received == AllReduce2/AllGather2(part_rank0, part_rank1) + tokens == control) -- NOT the per-rank instance-sequence Match",
              "compare_verdict": "INTERMEDIATE" if (st.get("pass") and pv.get("pass") and tk.get("pass")) else "FAIL",
              "compare_rc": 0, "canonical_equal": None, "pass": False, "fold_ok": None,
              "control_vs_instrumented_tokens_equal": bool(tk.get("pass")),
              "steps": [v.get("n_steps") for v in (st.get("per_rank") or {}).values()],
              "totals": {"requests": tk.get("n_requests"), "ranks": 2, "collectives_per_rank": [v.get("n_collectives") for v in (st.get("per_rank") or {}).values()],
                         "provenance_collectives_checked": pv.get("collectives_checked"), "provenance_elements_checked": pv.get("elements_checked"),
                         "provenance_mismatches": pv.get("n_mismatches"), "unattributed": 0, "xreq": 0},
              "structure_pass": st.get("pass"), "provenance_pass": pv.get("pass"), "build_digests_matched": {r: v.get("build_digest") for r, v in (st.get("per_rank") or {}).items()},
              "gap": "per-rank instance-sequence correspondence (fold of the rank log vs the rank Program) not implemented: needs a fold-core peer-input operand class, "
                     "analytic.weights_type at TP geometry, a TP2 kappa profile",
              "cmd": "tp2_match --arm capture|control|check", "utc": M.get("utc")}
        (out / "match_summary.json").write_text(json.dumps(ms, indent=1, default=str) + "\n")
        stages.append(f"match {ms['compare_verdict']} {now} collective-level structure={st.get('pass')} provenance={pv.get('pass')} tokens={tk.get('pass')}")

    # ---- commit: per-rank, labelled root form
    C = Path(a.commit)
    S = _load(C / "summary.json")
    V = _load(C / "versions.json")
    runs = [json.loads(l) for l in (C / "runs.jsonl").read_text().splitlines() if l.strip()]
    for f in ("summary.json", "versions.json", "runs.jsonl", "committer_make.json"):
        if (C / f).exists():
            shutil.copy(C / f, out / "commit" / f"tp2_{f}")
    fins = sorted(C.glob("finalize_pair*.json"))
    for f in fins:
        shutil.copy(f, out / "commit" / f.name)
    # [TP-07/TP-08/TP-09] the C2 record files beside the C1 record: match oracle + attribution (oracle_compare), per-rank sampled replay +
    # boundary linkage (sampled_replay.json), weights pin per rank (weights_pin.json), the fused-MoE late-read detector (t6_4_check.json)
    for pat in ("declaration_rank*.json", "binding_map_rank*.json", "coverage_pair*.json", "value_check.json", "sampled_replay.json", "weights_pin.json",
                "oracle_compare.json", "match_oracle*.json", "attribution*.json", "t6_4_check.json", "linkage.json"):
        for f in C.glob(pat):
            shutil.copy(f, out / "commit" / f.name)
    if a.manifest and Path(a.manifest).exists():
        shutil.copy(a.manifest, out / "commit" / "manifest.json")          # the manifest of record (C1 population), both ranks, never folded
    trt = [r for r in runs if r.get("arm") != "control"]
    ctl = [r for r in runs if r.get("arm") == "control"]
    per_run_ranks = [_load(f) for f in fins]
    cov = None
    if (C / "coverage.json").exists():
        cov = _load(C / "coverage.json")
        cov.setdefault("program_digest", None)      # per-rank digests are in program_digests_per_rank; the finalizer's single-digest field stays unset
        # [R15 TP-04] C2 is a separate labelled check the driver records into coverage.json (value_correspondence / value_correctness) when
        # --value-check ran; without it both stay None (never asserted).  R14 nulled the field unconditionally; the record now carries it.
        if isinstance(cov.get("value_correspondence"), dict) and (out / "commit" / "value_check.json").exists():
            cov["value_correspondence"]["artifact"] = str(out / "commit" / "value_check.json")      # the C2 record itself, beside the C1 record
        if "value_correspondence" not in cov:
            cov["value_correctness"] = None
            cov.setdefault("value_correctness_note", "population-drawn openings verify against the original per-rank roots; C2 value check not run for this attempt")
        skipped = [str((x.get("binding") or {}).get("skipped")) for x in (per_run_ranks[0] if per_run_ranks else []) if (x.get("binding") or {}).get("skipped")]
        if per_run_ranks and len(skipped) == len(per_run_ranks[0]):
            cov.update({"verdict": "NOT_RUN", "identity_coverage": None, "skipped": skipped, "openings_population": None,
                        "reason": "binding coverage not run on any rank: " + skipped[0]})
        (out / "commit" / "coverage.json").write_text(json.dumps(cov, indent=1, default=str) + "\n")
    exp = [sum(int(x["expected_tensors"]) for x in pr) for pr in per_run_ranks]
    com = [sum(int(x["committed_tensors"]) for x in pr) for pr in per_run_ranks]
    idc = [{int(x["rank"]): (x.get("identity_check") or {}).get("ok") for x in pr} for pr in per_run_ranks]
    roots_rank = [{int(x["rank"]): x["run_root"] for x in pr} for pr in per_run_ranks]
    fails = [((r.get("commit") or {}).get("commit_fail_reason")) for r in trt if not (r.get("commit") or {}).get("commit_pass")]
    verdict = {"commit_pass": bool(S.get("pass")) and not fails, "runs_instrumented": len(trt), "runs_failed": len(fails), "first_fail_reason": fails[0] if fails else None,
               "required_classes": V.get("required_classes"), "required_classes_source": V.get("required_classes_source"),
               "run_roots": [(r.get("commit") or {}).get("tp_run_root") for r in trt],
               "run_root_form": "tp-run-root/v1 (PROVISIONAL tagged SHA-256 over the per-rank roots, labelled; canonical = per-rank entries of the commitment vector)",
               "per_rank_run_roots": roots_rank, "per_rank_identity_check_ok": idc, "world": 2, "rules_version": "r14-v1",
               "negatives": V.get("negatives"), "layout_per_rank": "chunk-leaf-v1"}
    if a.negatives:
        negs = {}
        for nd in sorted(Path(a.negatives).glob("neg_*")):
            rj = nd / "runs.jsonl"
            if rj.exists():
                nr = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
                t = [r for r in nr if r.get("arm") != "control"]
                negs[nd.name] = {"commit_pass": (t[0].get("commit") or {}).get("commit_pass") if t else None, "reason": (t[0].get("commit") or {}).get("commit_fail_reason") if t else None,
                                 "tp_run_root": (t[0].get("commit") or {}).get("tp_run_root") if t else None,
                                 "flags": (_load(nd / "versions.json").get("negatives") if (nd / "versions.json").exists() else None)}
                # the negative's own records beside the positive's (runs, coverage verdict, summary)
                (out / "commit" / "negatives" / nd.name).mkdir(parents=True, exist_ok=True)
                for f in ("runs.jsonl", "coverage.json", "summary.json", "versions.json", "sampled_replay.json", "weights_pin.json", "t6_4_check.json"):
                    if (nd / f).exists():
                        shutil.copy(nd / f, out / "commit" / "negatives" / nd.name / f)
        verdict["negatives_executed"] = negs
        verdict["negatives_all_failed"] = bool(negs) and all(v.get("commit_pass") is False for v in negs.values())
    if cov is not None:
        verdict["coverage"] = {k: cov.get(k) for k in ("verdict", "population_source", "manifest_digest", "coverage_per_rank", "identity_coverage", "openings_checked", "openings_ok",
                                                      "value_correctness", "binding_map_digests_per_rank", "program_digests_per_rank")}
        if isinstance(cov.get("value_correspondence"), dict):
            verdict["value_correspondence"] = cov["value_correspondence"]
    (out / "commit" / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    # ---- attempt.json (finalizer write-time validation, report M-0069): campaign/lane/pod/release/revision + the realized workload
    # realized generated length per request: the Match's control tokens (tp2_match.json tokens.generated_lengths, per request) when the
    # Match ran in this attempt, else at B=1 the instrumented Commit run's generated token count (one request = every generated token)
    gen = None
    if (out / "tp2_match.json").exists():
        gl = ((_load(out / "tp2_match.json").get("tokens") or {}).get("generated_lengths"))
        gen = [int(x) for x in gl] if isinstance(gl, list) and gl else None
    if gen is None and len(wl_reqs) == 1 and trt and trt[0].get("generated_tokens") is not None:
        gen = [int(trt[0]["generated_tokens"])]
    # achieved concurrency at B>=2 (brief §11: the OBSERVED chronology, never the configured limit): per rank, the binding map's entries
    # grouped by their engine step root -> the set of requests bound in that step; the maximum over steps is what ran concurrently
    conc: dict = {"achieved_concurrency": 1 if len(wl_reqs) == 1 else None}
    if len(wl_reqs) > 1:
        import collections
        per_rank = {}
        for r in (0, 1):
            bm = C / f"binding_map_rank{r}.json"
            if not bm.exists():
                continue
            by_step: dict[str, set] = collections.defaultdict(set)
            for e in (_load(bm).get("entries") or []):
                rid = (e.get("identity") or {}).get("request_id")
                if e.get("step_root") and rid and rid != "*":
                    by_step[e["step_root"]].add(rid)
            if by_step:
                per_rank[str(r)] = {"engine_steps": len(by_step), "max_concurrent": max(len(v) for v in by_step.values()),
                                    "histogram": dict(sorted(collections.Counter(len(v) for v in by_step.values()).items()))}
        if per_rank:
            mx = {v["max_concurrent"] for v in per_rank.values()}
            conc = {"achieved_concurrency": (min(mx) if len(mx) == 1 else None), "engine_steps": [v["engine_steps"] for v in per_rank.values()],
                    "concurrency_basis": "observed: requests bound per engine-step root in the rank's binding map (both ranks agree)" if len(mx) == 1
                                         else f"ranks disagree on max concurrency {sorted(mx)} -- not recorded",
                    "concurrency_per_rank": per_rank}
    attempt = {"campaign": a.campaign, "lane": a.lane, "pod": a.pod, "release_tag": a.release_tag, "release_commit": a.revision, "rules_version": "r14-v1",
               "kind": "execution", "row_id": row, "workload": a.workload, "workload_manifest_digest": hashlib.sha256(Path(a.workload).read_bytes()).hexdigest(),
               "tp": {"world": 2, "ranks": [0, 1]},
               "workload_actual": conc | {"generated_lengths": gen,
                                   "input_lengths": [int(r.get("prompt_len") or r.get("LP") or 0) for r in wl_reqs] or None,
                                   "output_caps": [int(r.get("max_tokens") or 0) for r in wl_reqs] or None, "arrival_offsets_s": [0.0] * len(wl_reqs)},
               "negatives": verdict.get("negatives_executed"), "utc": now}
    if a.tree_note:
        attempt["tree_note"] = a.tree_note
    if a.tree_commit:
        # COORD M-0132: the revision of record is the EXECUTING tree (the lane tree shipped to the pod), never the pod's RELEASE.json;
        # the finalizer diffs it against the release base itself -- the producer only states which tree ran and whether it was dirty
        attempt["executing_tree"] = {"path": a.tree_path, "branch": a.tree_branch, "commit": a.tree_commit, "dirty": bool(a.tree_dirty),
                                     "release_base": a.revision, "shipped": "tar of the lane worktree at this commit (verity_capture/ + bench scripts)"}
        attempt["release_tag"] = None                      # a lane tree is not the release tree; the base it grew from is release_commit
    (out / "attempt.json").write_text(json.dumps(attempt, indent=1, default=str) + "\n")
    decl0 = (per_run_ranks[0][0].get("declaration") if per_run_ranks else None) or {}
    summ = {"schema": "verity/commit-integ/delta-tp/v1 -> finalizer view", "workload": a.workload, "committer": f"{S.get('committer', 'native_collect_v2b')} per rank (TP2) + FA2 matReq hidden source + TP-03 tp_partial source; "
                                                                                                 f"windowed staging (VERITY_WINDOW_MB 256 x 8 slots), retain=host",
            "pairs": S.get("pairs") or len(trt), "expected_tensors": exp, "committed_tensors": com, "tokens_equal": all(r.get("tokens_equal_ref") for r in runs),
            "commit_fail_reasons": fails, "commit_pass": verdict["commit_pass"], "backlog_max": None,
            "control_wall_s": [r["wall_s"] for r in ctl], "treatment_wall_s": [r["wall_s"] for r in trt],
            "delta_wall": [(t["wall_s"] / c["wall_s"] - 1) for t, c in zip(trt, ctl)], "delta_wall_mean": S.get("delta_wall_mean"), "delta_wall_pct": S.get("delta_wall_pct"),
            "declaration": {"layout": "chunk-leaf-v1", "committer": decl0.get("committer"), "classes_per_rank": [[c.get("class") for c in (x.get("declaration") or {}).get("classes", [])] for x in (per_run_ranks[0] if per_run_ranks else [])],
                            "retain": "host", "window_mb": 256, "window_slots": 8},
            "engine_settings": {k: V.get("engine_kwargs", {}).get(k) if isinstance(V.get("engine_kwargs"), dict) else None for k in ("gpu_memory_utilization", "max_num_batched_tokens", "max_num_seqs")},
            "per_rank": [{"rank": int(x["rank"]), "tensors": x["tensors"], "leaves": x["leaves"], "bytes_bound": x["bytes_bound"], "identity_check_ok": (x.get("identity_check") or {}).get("ok"),
                          "classes": {c["class"]: c.get("tensors") for c in (x.get("declaration") or {}).get("classes", [])}} for x in (per_run_ranks[0] if per_run_ranks else [])],
            "utc": S.get("utc") or now}
    (out / "commit" / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    stages.append(f"commit {'ok' if verdict['commit_pass'] else 'FAIL'} {now} runs {len(runs)} roots {[str(x)[:16] for x in verdict['run_roots']]} per-rank {[{k: v[:16] for k, v in rr.items()} for rr in roots_rank[:1]]}")
    (out / "stages.txt").write_text("\n".join(stages) + "\n")
    print("\n".join(stages))
    print(f"[tp2_evidence] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
