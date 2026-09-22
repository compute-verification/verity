"""[R16 compiled, rev F-r16-03 / F-r16-04 (b)] Re-emit an Execution=compiled attempt's `attempt.json` provenance BY LOGGED DERIVATION from the
run's own records -- never by hand-editing a field (rev M-0031: "repair by logged derivation").

What is derived, and from where:
  executing_tree   every arm's `commit_*_r*/versions.json`: export.commit / git_head (the tree the process imported), export.dirty_patch_bytes
                   (dirty), argv[0] (the repo path the process ran from), export.from_worktree; `stages.txt` "release <tag> <sha>" line.
                   All arms must agree on the commit (refusal otherwise).  is_release_tree = the derived commit equals the attempt's declared
                   release commit (attempt.json release_commit / verity_commit / tree.commit) AND the tree is clean; lane_tree = not that.
  evidence_dir     the directory this attempt.json lives in (relative to the repo root).
  attempt_id       the directory's name.
  at               the first stages.txt timestamp (the run's start), started_utc/finished_utc from stages.txt first/last lines.
  negatives        --negatives <dir>: a `negatives/<kind>/` tree whose `verdict.json` (commit_pass false = detected) is read per kind; the record
                   gets {suite, results, artifact, result, undetected, no_verdict, run_at} -- results derived from the verdict files, never from
                   a suite string (rev F-r16-04).  If the directory lies outside this attempt, `run_at` names where (another attempt of the same
                   mechanism/tree family) and the finalizer's own scan (evdir/negatives) will not see it -- pass --link to symlink it beside
                   this attempt's commit/ (data-only, logged).
The PREVIOUS values of every replaced field are kept under `attempt_declared_stale` {fields, why, by, at, finding}; the derivation is appended
to `stages.txt` and written to `attempt_derivation.json` (inputs, their sha256, outputs) so the repair is itself a record.

  python -m verity_capture.sweep.compiled_attempt_derive --evidence-dir <dir> [--negatives <dir>] [--link] [--by compiled] [--why '...'] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys


def _load(p: str):
    with open(p) as f:
        return json.load(f)


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _repo_root(evdir: str) -> str:
    d = os.path.abspath(evdir)
    while d != "/":
        if os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def _repo_path_of_argv(argv: list) -> str | None:
    if not argv:
        return None
    a0 = str(argv[0])
    m = re.match(r"^(.*?)/verity_capture/", a0)
    return m.group(1) if m else (os.path.dirname(a0) or None)


def derive_executing_tree(evdir: str, A: dict) -> tuple[dict, list[dict]]:
    arms: list[dict] = []
    for vj in sorted(glob.glob(os.path.join(evdir, "commit_*", "versions.json"))):
        try:
            V = _load(vj)
        except Exception as e:  # noqa: BLE001
            arms.append({"versions_json": os.path.relpath(vj, evdir), "error": f"{type(e).__name__}: {e}"[:120]})
            continue
        ex = V.get("export") if isinstance(V.get("export"), dict) else {}
        arms.append({"versions_json": os.path.relpath(vj, evdir), "sha256": _sha(vj),
                     "export_commit": ex.get("commit"), "git_head": V.get("git_head"), "export_branch": ex.get("branch"),
                     "dirty_patch_bytes": ex.get("dirty_patch_bytes"), "untracked": len(ex.get("untracked") or []) if isinstance(ex.get("untracked"), list) else None,
                     "from_worktree": ex.get("from_worktree"), "exported_utc": ex.get("exported_utc"),
                     "repo_path": _repo_path_of_argv(V.get("argv") or []), "only_arm": V.get("only_arm")})
    good = [a for a in arms if a.get("export_commit") or a.get("git_head")]
    if not good:
        raise SystemExit("REFUSED: no commit_*/versions.json with export.commit / git_head under " + evdir)
    commits = {str(a.get("export_commit") or a.get("git_head")) for a in good}
    heads = {str(a.get("git_head")) for a in good if a.get("git_head")}
    if len(commits) != 1:
        raise SystemExit(f"REFUSED: the arms ran different trees: {sorted(commits)}")
    commit = commits.pop()
    if heads and heads != {commit}:
        raise SystemExit(f"REFUSED: export.commit {commit} != git_head {sorted(heads)}")
    paths = {a.get("repo_path") for a in good if a.get("repo_path")}
    if len(paths) > 1:
        raise SystemExit(f"REFUSED: the arms ran from different repo paths: {sorted(paths)}")
    dirty = any((a.get("dirty_patch_bytes") or 0) > 0 or (a.get("untracked") or 0) > 0 for a in good)
    # the attempt's declared release commit (short or long) -- from the attempt itself, and stages.txt's "release <tag> <sha>" line
    declared = [str(x) for x in (A.get("release_commit"), A.get("verity_commit"), (A.get("tree") or {}).get("commit") if isinstance(A.get("tree"), dict) else None) if x]
    st_rel = None
    sp = os.path.join(evdir, "stages.txt")
    if os.path.isfile(sp):
        for line in open(sp):
            m = re.search(r": release (.+?) commit=([0-9a-f]{7,40})\s*$", line) or re.search(r": release (\S+) ([0-9a-f]{7,40})\s*$", line)
            if m:
                st_rel = {"tag": m.group(1), "commit": m.group(2)}
    same = lambda a, b: bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))  # noqa: E731
    is_release = (not dirty) and any(same(commit, d) for d in declared) and (st_rel is None or same(commit, st_rel["commit"]))
    et = {"path": paths.pop() if paths else None, "commit": commit, "commit_short": commit[:9], "dirty": dirty,
          "is_release_tree": is_release, "lane_tree": not is_release,
          "release_tag": (st_rel or {}).get("tag") or A.get("release_tag"),
          "derived_from": {"versions_json": [a["versions_json"] for a in good], "fields": ["export.commit", "git_head", "export.dirty_patch_bytes", "export.untracked", "argv[0]"],
                           "stages_txt_release_line": st_rel, "attempt_declared_release_commits": declared},
          "derived_by": "verity_capture/sweep/compiled_attempt_derive.py"}
    return et, arms


def _stages_span(evdir: str) -> tuple[str | None, str | None]:
    sp = os.path.join(evdir, "stages.txt")
    if not os.path.isfile(sp):
        return None, None
    ts = re.findall(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\]", open(sp).read(), flags=re.M)
    return (ts[0] if ts else None), (ts[-1] if ts else None)


def derive_negatives(neg_dir: str, evdir: str, root: str) -> dict:
    results: dict = {}
    undetected, no_verdict = [], []
    for d in sorted(os.listdir(neg_dir)):
        dd = os.path.join(neg_dir, d)
        if not os.path.isdir(dd):
            continue
        vp = next((p for p in (os.path.join(dd, "verdict.json"), *sorted(glob.glob(os.path.join(dd, "commit_*", "verdict.json")))) if os.path.isfile(p)), None)
        if vp is None:
            no_verdict.append(d)
            results[d] = {"detected": None, "why": "no verdict.json on record"}
            continue
        V = _load(vp)
        cp = V.get("commit_pass") if V.get("commit_pass") is not None else V.get("result")
        detected = (cp is False)
        vl = os.path.join(dd, "verdict_lines.txt")
        results[d] = {"detected": detected, "verdict": os.path.relpath(vp, root), "verdict_sha256": _sha(vp), "commit_pass": cp,
                      "first_fail_reason": str(V.get("first_fail_reason") or "")[:240], "runs_failed": V.get("runs_failed"), "runs_instrumented": V.get("runs_instrumented"),
                      "verdict_lines": open(vl).read().strip().splitlines()[:4] if os.path.isfile(vl) else None}
        if not detected:
            undetected.append(d)
        # [report M-0110 reading (a)] the TREE the attack ran on and the P_compiled manifest it was checked against, read from the attack arm's
        # versions.json / summary.json -- so a same-tree, same-manifest negative is named as the row's OWN (not "another attempt")
        arm = os.path.dirname(vp)
        Vv = _load(os.path.join(arm, "versions.json")) if os.path.isfile(os.path.join(arm, "versions.json")) else {}
        ex = Vv.get("export") or {}
        results[d]["tree_commit"] = str(ex.get("commit") or Vv.get("git_head") or "")[:40] or None
        results[d]["tree_dirty_patch_bytes"] = ex.get("dirty_patch_bytes")
        pdg = None
        rp = os.path.join(arm, "runs.jsonl")
        if os.path.isfile(rp):                                                 # commit_delta's per-run record: program_digest (source required-manifest)
            for line in open(rp):
                try:
                    R = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                cb = ((R.get("commit") or {}).get("binding") or {})
                pdg = pdg or cb.get("program_digest") or ((cb.get("population") or {}).get("manifest") or {}).get("program_digest") or R.get("program_digest")
                if pdg:
                    results[d]["manifest_program_digest_source"] = cb.get("program_digest_source") or R.get("program_digest_source")
                    break
        results[d]["manifest_program_digest"] = str(pdg or "")[:64] or None
    inside = os.path.abspath(neg_dir).startswith(os.path.abspath(evdir) + os.sep)
    # same tree + same manifest as THIS attempt -> the row's own negatives (the attempt's executing_tree.commit / build digest are derived first)
    A = _load(os.path.join(evdir, "attempt.json")) if os.path.isfile(os.path.join(evdir, "attempt.json")) else {}
    my_tree = str(((A.get("executing_tree") or {}).get("commit")) or "")
    my_digest = str(((A.get("build") or {}).get("program_digest")) or ((A.get("commit_summary") or {}).get("program_digest")) or "")
    trees = {r.get("tree_commit") for r in results.values() if r.get("tree_commit")}
    digests = {r.get("manifest_program_digest") for r in results.values() if r.get("manifest_program_digest")}
    same_tree = bool(my_tree) and bool(trees) and all(t and (t.startswith(my_tree) or my_tree.startswith(t)) for t in trees)
    same_manifest = bool(my_digest) and bool(digests) and all(g and (g.startswith(my_digest[:16]) or my_digest.startswith(g[:16])) for g in digests)
    run_at = ("this attempt" if inside else
              (f"the row's OWN negatives on this attempt's tree {my_tree[:9]} against this attempt's P_compiled manifest {my_digest[:16]} (another run directory: {os.path.relpath(neg_dir, root)})"
               if same_tree and same_manifest else
               (f"another attempt on this attempt's tree {my_tree[:9]} but a different manifest {sorted(digests)[:2]}: {os.path.relpath(neg_dir, root)}" if same_tree else
                f"another attempt of the same mechanism/tree family (tree {sorted(trees)[:2]} vs this attempt's {my_tree[:9]}): {os.path.relpath(neg_dir, root)}")))
    return {"suite": sorted(results), "results": results, "artifact": os.path.relpath(neg_dir, root), "result": (not undetected and not no_verdict),
            "undetected": undetected, "no_verdict": no_verdict,
            "method": "targeted negatives: each attack's own commit verdict must be commit_pass false (detected); results derived from the verdict files",
            "tree_commits": sorted(trees), "manifest_program_digests": sorted(digests), "same_tree_as_attempt": same_tree, "same_manifest_as_attempt": same_manifest,
            "run_at": run_at}


def _gz_or_json(p: str):
    import gzip
    if os.path.isfile(p):
        return _load(p)
    if os.path.isfile(p + ".gz"):
        with gzip.open(p + ".gz", "rt") as f:
            return json.load(f)
    return None


def init_attempt(evdir: str, root: str, a: argparse.Namespace) -> dict:
    """[--init] the producer's attempt.json for an Execution=compiled Commit record, EVERY field derived from the run's own artifacts:
    compiled_manifest.json (P_compiled of record: program_digest, kernel_set, manifest_digest, identities, complete, components.weights_struct),
    commit/summary.json + commit/verdict.json (compiled_merge: pairs, commit_pass, tokens equal, loaded identity, program_digests.equal, overhead),
    commit_*/versions.json (declared vs effective execution, compilation, engine args, tree), the workload manifest (declared target), stages.txt."""
    M = _gz_or_json(os.path.join(evdir, "compiled_manifest.json")) or {}
    S = _gz_or_json(os.path.join(evdir, "commit", "summary.json")) or {}
    Vd = _gz_or_json(os.path.join(evdir, "commit", "verdict.json")) or {}
    MV = _gz_or_json(os.path.join(evdir, "commit", "manifest_verify.json")) or {}
    arms = sorted(glob.glob(os.path.join(evdir, "commit_*", "versions.json")))
    V0 = _load(arms[0]) if arms else {}
    W = _load(os.path.join(root, a.workload)) if a.workload and os.path.isfile(os.path.join(root, a.workload)) else (M.get("workload") if isinstance(M.get("workload"), dict) else {})
    tgt = (W.get("target") or {}) if isinstance(W, dict) else {}
    comp = S.get("compilation") if isinstance(S.get("compilation"), dict) else (V0.get("compilation") or {})
    eff = V0.get("effective_execution") if isinstance(V0.get("effective_execution"), dict) else {}
    gk = S.get("generated_kernels") if isinstance(S.get("generated_kernels"), dict) else {}
    li = S.get("loaded_identity") if isinstance(S.get("loaded_identity"), dict) else {}
    pd = S.get("program_digests") if isinstance(S.get("program_digests"), dict) else {}
    first_ts, last_ts = _stages_span(evdir)
    row_id = a.row_id or os.path.basename(os.path.dirname(evdir))
    pairs = S.get("pairs"); rounds = S.get("rounds")
    n_inst = Vd.get("runs_instrumented")
    cov = S.get("coverage") if isinstance(S.get("coverage"), list) else None
    vc = S.get("value_correspondence") if isinstance(S.get("value_correspondence"), list) else None
    A = {"schema": "verity-sweep/attempt/v1", "campaign": a.campaign, "rules_version": a.rules_version, "lane": a.lane, "row_id": row_id,
         "attempt_id": os.path.basename(evdir), "kind": "execution",
         "status": "complete" if S.get("commit_pass") is not None else "partial", "pod": a.pod,
         "tree": {"kind": "lane" , "note": a.tree_note} if a.tree_note else None,
         "stages": {"build": "PASS" if M.get("program_digest") else "-", "match": "-", "commit": ("PASS" if S.get("commit_pass") else ("FAIL" if S.get("commit_pass") is False else "-")), "bench": "DONE" if S.get("wall_ratio_mean") is not None else "-"},
         "declared_target": {"compilation": tgt.get("compilation"), "device": tgt.get("device"), "profile_id": tgt.get("profile_id") or tgt.get("id"),
                             "engine_args_required": tgt.get("engine_args_required") or W.get("engine_args_required") if isinstance(W, dict) else None,
                             "source": a.workload or "compiled_manifest.json workload block"},
         "execution_configuration": {"declared_execution": V0.get("declared_execution"), "effective_execution": eff,
                                     "compilation": {k: comp.get(k) for k in ("compilation_mode", "mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size", "use_inductor_graph_partition", "backend", "custom_ops", "enforce_eager", "inductor_compile_config", "emulate_precision_casts", "compile_cache_disabled", "torch") if k in comp},
                                     "enforce_eager": V0.get("enforce_eager"), "engine_kwargs": V0.get("engine_kwargs"),
                                     "compilation_settings_equal_across_arms": S.get("compilation_settings_equal"),
                                     "pairs": pairs, "rounds": rounds, "arms": "control (no hooks, no committer) / treatment (--compiled-taps kernels) as separate processes with identical engine args",
                                     "source": [os.path.relpath(p, evdir) for p in arms]},
         "build": {"kind": "P_compiled", "program_digest": M.get("program_digest"), "kernel_set_sha256": M.get("kernel_set_sha256"), "manifest_digest": M.get("manifest_digest"),
                   "identities": len(M.get("identities") or []) if isinstance(M.get("identities"), list) else M.get("identities"), "complete": M.get("complete"),
                   "weights_struct_program_digest": ((M.get("components") or {}).get("weights_struct") or {}).get("program_digest"),
                   "derived_from": "control arm's Inductor output code + split graph (compiled_manifest.build)", "artifact": "compiled_manifest.json[.gz]",
                   "verify": {"treatment_rederivation_equal": pd.get("equal"), "program_digest_control": pd.get("control"), "program_digest_treatment": pd.get("treatment"),
                              "manifest_verify_ok": MV.get("ok"), "manifest_verify": "commit/manifest_verify.json" if MV else None},
                   "ruling": "COORD M-0819: for Execution=compiled rows BUILD = this derivation record"},
         "match": {"status": "derived-by-finalizer", "inputs": {"s2_checkpoints": next((os.path.relpath(p, evdir) for p in glob.glob(os.path.join(evdir, "**", "*s2_checkpoints*.json"), recursive=True)), None),
                                                                 "loaded_identity": "commit/summary.json loaded_identity (+ per arm versions.json compilation.generated_kernels.loaded)"}},
         "commit_summary": {"commit_pass": S.get("commit_pass"), "problems": S.get("problems"), "instrumented_runs": n_inst, "runs_failed": Vd.get("runs_failed"),
                            "first_fail_reason": Vd.get("first_fail_reason"), "pairs": pairs, "rounds": rounds, "tokens_equal_across_arms": S.get("tokens_equal"),
                            "coverage_ok_per_run": cov, "value_correspondence_ok_per_run": vc, "openings_verified": S.get("openings_verified"),
                            "overhead_wall_ratio": S.get("wall_ratio_mean"), "delta_wall_mean_s": S.get("delta_wall_mean"), "delta_wall_ci95": S.get("delta_wall_ci95"),
                            "program_digest_equal_arms": pd.get("equal"), "kernel_set_equal_arms": gk.get("kernel_set_equal"),
                            "loaded_identity": {k: li.get(k) for k in ("equal_across_arms", "binaries_equal_across_arms", "kernels_whose_binaries_differ", "executed_code_equal_across_arms",
                                                                        "executed_text_differs", "executed_text_differs_in_program", "executed_config_differs", "text_equal_cubin_bytes_differ",
                                                                        "manifest_kernels", "manifest_kernels_not_loaded") if k in li},
                            "replay_covered": (li.get("replay_covered") or {}).get("ok") if isinstance(li.get("replay_covered"), dict) else None,
                            "source": ["commit/summary.json", "commit/verdict.json"]},
         "results": {"commit": f"{'PASS' if S.get('commit_pass') else 'FAIL'} fail-closed, {n_inst} instrumented runs ({rounds} rounds x {pairs} pairs)" if S else None,
                     "noninterference": (f"tokens equal across arms {S.get('tokens_equal')}; kernel_set equal across arms {gk.get('kernel_set_equal')} ({str(gk.get('control', {}).get('kernel_set_sha256') if isinstance(gk.get('control'), dict) else '')[:16]}); "
                                         f"program_digest from each arm's own output code equal {pd.get('equal')}") if S else None},
         "negatives": None, "started_utc": first_ts, "finished_utc": last_ts, "note": a.note}
    A = {k: v for k, v in A.items() if v is not None}
    return A


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--init", action="store_true", help="write a NEW attempt.json derived from the run's own artifacts (refuses to overwrite one)")
    ap.add_argument("--row-id"); ap.add_argument("--lane", default="compiled"); ap.add_argument("--campaign", default="r16"); ap.add_argument("--rules-version", default="r14-v1")
    ap.add_argument("--pod"); ap.add_argument("--workload", help="repo-relative workload manifest (declared target)"); ap.add_argument("--tree-note"); ap.add_argument("--note")
    ap.add_argument("--negatives", help="negatives/<kind>/ directory whose verdicts become the record's negatives (derived, not declared)")
    ap.add_argument("--link", action="store_true", help="symlink --negatives as <evidence-dir>/negatives when it lies outside the attempt")
    ap.add_argument("--s2", help="an S2 checkpoint-replay record (verity/compiled-s2-checkpoints/v1) of THIS attempt's specialization: linked as match/s2_checkpoints.json only if its generated_kernels.kernel_set_sha256 equals the attempt's P_compiled kernel_set")
    ap.add_argument("--workload-actual", help="a verity/compiled-workload-actual/v1 file derived from an arm's binding map (compiled_workload_actual.py): "
                                              "becomes attempt.workload_actual (generated lengths, arrivals, achieved concurrency observed per engine step)")
    ap.add_argument("--workload-actual-from-coverage", action="store_true",
                    help="no binding map kept: derive workload_actual from the arms' runs.jsonl (coverage bound == expected and generated_tokens == sum of caps => every request "
                         "generated exactly its cap); arrivals from the declaration, concurrency computed from declared arrivals x observed lengths (chronology not confirmed)")
    ap.add_argument("--by", default="compiled")
    ap.add_argument("--why", default="rev F-r16-03: attempt.json carried another attempt's provenance block wholesale; re-emitted from the run's own versions.json + stages.txt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    evdir = os.path.abspath(a.evidence_dir.rstrip("/"))
    root = _repo_root(evdir)
    ap_json = os.path.join(evdir, "attempt.json")
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if a.init:
        if os.path.exists(ap_json):
            raise SystemExit(f"REFUSED: {ap_json} exists (re-derive its provenance without --init)")
        A = init_attempt(evdir, root, a)
        A["at"] = A.get("started_utc") or now
        A["evidence_dir"] = os.path.relpath(evdir, root)
        with open(ap_json, "w") as f:
            json.dump(A, f, indent=1)
            f.write("\n")
        with open(os.path.join(evdir, "stages.txt"), "a") as f:
            f.write(f"[{now}] attempt.json WRITTEN by derivation from the run's artifacts ({a.by}, compiled_attempt_derive.py --init): build.kind P_compiled "
                    f"{str((A.get('build') or {}).get('program_digest'))[:16]} kernel_set {str((A.get('build') or {}).get('kernel_set_sha256'))[:16]}; "
                    f"commit_pass {(A.get('commit_summary') or {}).get('commit_pass')}; executing_tree derived next\n")
        print(f"[init] wrote {os.path.relpath(ap_json, root)} (build {str((A.get('build') or {}).get('program_digest'))[:16]}, commit_pass {(A.get('commit_summary') or {}).get('commit_pass')})")
        a.why = "producer's own derivation of executing_tree from the arms' versions.json (fresh attempt; no declared block)"
    A = _load(ap_json)
    before_sha = _sha(ap_json)

    et, arms = derive_executing_tree(evdir, A)
    first_ts, last_ts = _stages_span(evdir)
    new = {"executing_tree": et, "evidence_dir": os.path.relpath(evdir, root), "attempt_id": os.path.basename(evdir)}
    if first_ts:
        new["at"] = first_ts
        new["started_utc"] = A.get("started_utc") or first_ts
    if last_ts:
        new["finished_utc"] = A.get("finished_utc") or last_ts
    stale = {k: A.get(k) for k in new if k in A and A.get(k) != new[k]}
    if a.negatives:
        neg = derive_negatives(os.path.abspath(a.negatives), evdir, root)
        res = A.get("results") if isinstance(A.get("results"), dict) else {}
        old_neg = res.get("negatives")
        if old_neg is not None and old_neg != neg:
            stale["results.negatives"] = old_neg
        if not isinstance(A.get("results"), dict):
            A["results"] = {}
        A["results"]["negatives"] = neg
        A["negatives"] = neg
        if a.link and neg["run_at"] != "this attempt":
            dst = os.path.join(evdir, "negatives")
            want = os.path.relpath(os.path.abspath(a.negatives), evdir)
            if os.path.islink(dst) and os.readlink(dst) != want and not a.dry_run:      # re-linked: the earlier target is named in the stage line
                neg["previous_link"] = os.readlink(dst)
                os.unlink(dst)
            if not os.path.exists(dst) and not a.dry_run:
                os.symlink(want, dst)
                neg["linked_as"] = os.path.relpath(dst, root)
    if a.s2:
        s2p = os.path.abspath(a.s2)
        J = _load(s2p)
        ks_s2 = str(((J.get("generated_kernels") or {}) if isinstance(J.get("generated_kernels"), dict) else {}).get("kernel_set_sha256"))
        ks_att = str(((A.get("build") or {}) if isinstance(A.get("build"), dict) else {}).get("kernel_set_sha256"))
        if not str(J.get("schema") or "").startswith("verity/compiled-s2-checkpoints/"):
            raise SystemExit(f"REFUSED: {a.s2} is not a verity/compiled-s2-checkpoints record")
        if ks_s2 != ks_att:
            raise SystemExit(f"REFUSED: S2 record kernel_set {ks_s2[:16]} != this attempt's P_compiled kernel_set {ks_att[:16]} (another specialization)")
        rp = (J.get("replay") or {}).get("per_kind") or {}
        outs = sum(int(v.get("outputs") or 0) for v in rp.values() if isinstance(v, dict)); ex = sum(int(v.get("exact_outputs") or 0) for v in rp.values() if isinstance(v, dict))
        A["match"] = dict(A.get("match") or {}, s2_record={"artifact": os.path.relpath(s2p, root), "kernel_set_sha256": ks_s2, "outputs": outs, "exact_outputs": ex,
                                                          "noninterference": J.get("noninterference"), "linked_as": "match/s2_checkpoints.json",
                                                          "why": "the S2 record pins the specialization (kernel_set + compilation facts); this attempt's P_compiled has the same kernel_set"})
        stale.setdefault("_s2_linked", {"artifact": os.path.relpath(s2p, root), "kernel_set_sha256": ks_s2})
        if not a.dry_run:
            os.makedirs(os.path.join(evdir, "match"), exist_ok=True)
            dst = os.path.join(evdir, "match", "s2_checkpoints.json")
            if not os.path.lexists(dst):
                os.symlink(os.path.relpath(s2p, os.path.join(evdir, "match")), dst)
    wa_line = ""
    if (a.workload_actual or a.workload_actual_from_coverage) and isinstance(A.get("workload_actual"), dict) and A["workload_actual"].get("basis") not in ("observed", "coverage-derived"):
        stale["workload_actual"] = A["workload_actual"]          # the declared block (a hand-written note) is archived, not kept as the record's value
    if a.workload_actual:
        WA = _load(os.path.abspath(a.workload_actual))
        if str(WA.get("schema") or "") != "verity/compiled-workload-actual/v1":
            raise SystemExit(f"REFUSED: {a.workload_actual} is not a verity/compiled-workload-actual/v1 record")
        pdg_att = str(((A.get("build") or {}) if isinstance(A.get("build"), dict) else {}).get("program_digest") or "")
        if pdg_att and WA.get("source", {}).get("program_digest") and WA["source"]["program_digest"] != pdg_att:
            raise SystemExit(f"REFUSED: workload_actual binding map is of program {str(WA['source']['program_digest'])[:16]} != this attempt's P_compiled {pdg_att[:16]}")
        A["workload_actual"] = {"generated_lengths": WA.get("generated_lengths"), "arrive_steps": WA.get("arrive_steps"), "input_lengths": WA.get("input_lengths"),
                                "output_caps": WA.get("output_caps"), "achieved_concurrency": WA.get("achieved_concurrency"), "engine_steps": WA.get("engine_steps"),
                                "requests_per_step": WA.get("requests_per_step"), "basis": "observed", "concurrency_basis": WA.get("concurrency_basis"),
                                "generated_basis": WA.get("generated_basis"), "cross_check": WA.get("cross_check"), "consistent": WA.get("consistent"),
                                "source": WA.get("source"), "artifact": os.path.relpath(os.path.abspath(a.workload_actual), root), "derived_by": "compiled_workload_actual.py"}
        AR = WA.get("arrivals_rule")
        if isinstance(AR, dict):                      # F-r17-17: the schedule of record names its arrivals rule (declared | idle) per request
            A["workload_actual"]["arrivals_rule"] = {k: AR.get(k) for k in ("applicable", "holds", "lag", "lag_status", "by_rule", "arrive_equal_declared_under_rule",
                                                                             "iterations_replayed", "iterations_runs_jsonl", "forwards_replayed", "engine_steps_observed", "no_forward_iterations", "units")}
            A["workload_actual"]["arrivals_rule"]["per_request"] = [{k: r.get(k) for k in ("request_id", "declared_iter", "rule", "forward_predicted", "forward_observed")} for r in (AR.get("requests") or [])]
            if WA.get("rederived_from"):
                A["workload_actual"]["rederived_from"] = WA["rederived_from"]
        ar_line = (f"; arrivals rule {AR.get('by_rule')} holds {AR.get('holds')} (lag {AR.get('lag')}, iterations {AR.get('iterations_replayed')} == runs.jsonl {AR.get('iterations_runs_jsonl')})"
                   if isinstance(AR, dict) else "; arrivals rule NOT NAMED (pre-F-r17-17 record)")
        wa_line = (f"; workload_actual OBSERVED from the binding map ({A['workload_actual']['artifact']}): generated {WA.get('generated_lengths')} arrive {WA.get('arrive_steps')} "
                   f"concurrency {WA.get('achieved_concurrency')} engine_steps {WA.get('engine_steps')} consistent {WA.get('consistent')}{ar_line}")
    elif a.workload_actual_from_coverage:
        wl = _load(os.path.join(root, str(A.get("workload") or a.workload or "")))
        reqs = wl.get("requests") or []
        caps = [int(r.get("max_tokens") or 0) for r in reqs]; arr = [int(r.get("arrive_step") or 0) for r in reqs]; lp = [len(r.get("prompt_token_ids") or []) for r in reqs]
        gens, covs = [], []
        arm_dirs = sorted({os.path.dirname(os.path.join(evdir, x["versions_json"])) for x in arms if isinstance(x, dict) and x.get("versions_json")})
        if os.path.isfile(os.path.join(evdir, "commit", "runs.jsonl")):      # the merged record (both arms, every pair) when the arms' own runs.jsonl were not kept
            arm_dirs.append(os.path.join(evdir, "commit"))
        for arm in arm_dirs:
            rp = os.path.join(arm, "runs.jsonl")
            if not os.path.isfile(rp):
                continue
            for line in open(rp):
                try:
                    R = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if R.get("generated_tokens") is not None:
                    gens.append(int(R["generated_tokens"]))
                cv = ((R.get("commit") or {}).get("binding") or {}).get("coverage") or {}
                if cv:
                    covs.append((bool(cv.get("ok")), int(cv.get("expected_n") or 0), int(cv.get("bound_n") or 0), int(cv.get("missing_n") or 0)))
        every_cap = bool(gens) and all(g == sum(caps) for g in gens) and bool(covs) and all(ok and e == b and m == 0 for ok, e, b, m in covs)
        # declared arrivals x observed lengths -> concurrency: number of requests alive per step (alive from arrive_step for `length` steps)
        conc = 0; steps_n = max((s + g for s, g in zip(arr, caps)), default=0)
        if every_cap:
            for s in range(steps_n):
                conc = max(conc, sum(1 for a0, g in zip(arr, caps) if a0 <= s < a0 + g))
        A["workload_actual"] = {"generated_lengths": caps if every_cap else None, "arrive_steps": arr, "input_lengths": lp, "output_caps": caps,
                                "achieved_concurrency": conc if every_cap else None, "engine_steps_declared_min": steps_n, "engine_steps_runs_jsonl": sorted({int(json.loads(l).get("engine_steps") or 0) for arm in arm_dirs if os.path.isfile(os.path.join(arm, "runs.jsonl")) for l in open(os.path.join(arm, "runs.jsonl"))}),
                                "basis": "coverage-derived", "generated_basis": ("every request generated exactly its cap: every arm's coverage bound == expected (missing 0) -- every declared (request, step) identity of P_compiled x query was bound -- and every arm's generated_tokens == sum of caps" if every_cap else "NOT DERIVABLE: coverage or generated total does not pin the per-request lengths"),
                                "concurrency_basis": "declared arrival x observed lengths (observed chronology not confirmed: the binding maps of this attempt were not kept)",
                                "cross_check": {"generated_tokens_per_arm": gens, "sum_caps": sum(caps), "coverage_per_arm": covs}, "consistent": every_cap, "derived_by": "compiled_attempt_derive.py --workload-actual-from-coverage"}
        wa_line = f"; workload_actual COVERAGE-DERIVED: generated {A['workload_actual']['generated_lengths']} (every cap: {every_cap}) arrivals declared {arr} concurrency {A['workload_actual']['achieved_concurrency']} (declared arrival)"
    A.update(new)
    if stale:
        prev = A.get("attempt_declared_stale") if isinstance(A.get("attempt_declared_stale"), list) else ([A["attempt_declared_stale"]] if A.get("attempt_declared_stale") else [])
        A["attempt_declared_stale"] = prev + [{"fields": stale, "why": a.why, "by": a.by, "at": now, "finding": "rev M-0031 F-r16-03 (+ F-r16-04 (b) negatives)" if a.negatives else "rev M-0031 F-r16-03",
                                               "derivation": "attempt_derivation.json"}]
    deriv = {"schema": "verity/compiled-attempt-derivation/v1", "at": now, "by": a.by, "tool": "verity_capture/sweep/compiled_attempt_derive.py",
             "attempt_json_sha256_before": before_sha, "inputs": {"arms": arms, "stages_txt": os.path.relpath(os.path.join(evdir, "stages.txt"), root),
                                                                   "stages_first_ts": first_ts, "stages_last_ts": last_ts,
                                                                   "negatives_dir": os.path.relpath(os.path.abspath(a.negatives), root) if a.negatives else None},
             "outputs": {k: A.get(k) for k in ("executing_tree", "evidence_dir", "attempt_id", "at", "started_utc", "finished_utc")}, "stale": stale}
    line = (f"[{now}] attempt.json RE-EMITTED by logged derivation ({a.by}, compiled_attempt_derive.py): executing_tree {{path {et['path']}, commit {et['commit_short']}, "
            f"dirty {et['dirty']}, is_release_tree {et['is_release_tree']}, lane_tree {et['lane_tree']}}} from {len(arms)} arms' versions.json export.commit/git_head; "
            f"evidence_dir {new['evidence_dir']}; stale fields kept under attempt_declared_stale: {sorted(stale)}"
            + (f"; negatives from verdict files: {A['negatives']['suite']} detected={A['negatives']['result']} ({A['negatives']['run_at']})" if a.negatives else "")
            + (f"; S2 record of the same specialization linked as match/s2_checkpoints.json (kernel_set {str(A['match']['s2_record']['kernel_set_sha256'])[:16]}, {A['match']['s2_record']['exact_outputs']}/{A['match']['s2_record']['outputs']} outputs exact): {A['match']['s2_record']['artifact']}" if a.s2 else "")
            + wa_line)
    print(line)
    if a.dry_run:
        print(json.dumps(deriv["outputs"], indent=1)[:1500])
        return
    with open(ap_json, "w") as f:
        json.dump(A, f, indent=1)
        f.write("\n")
    deriv["attempt_json_sha256_after"] = _sha(ap_json)
    with open(os.path.join(evdir, "attempt_derivation.json"), "w") as f:
        json.dump(deriv, f, indent=1)
        f.write("\n")
    with open(os.path.join(evdir, "stages.txt"), "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
