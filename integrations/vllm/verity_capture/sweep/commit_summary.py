"""row_pod.sh Commit-stage summary (`<row>/commit_summary.json`) — one derivation, callable from the harness and re-runnable offline.

    python -m verity_capture.sweep.commit_summary <row-dir> <commit_delta-rc> [--out PATH]

DEPRECATED as a verdict (F-r19-int-21): `openings` / `coverage` are always None on runs.jsonl, so `pass` degenerates to
rc == 0 ∧ tokens_equal.  The verdict of record is `<row>/verdict.json` (vllm-verdict/v1, verity_capture.commit.verdict, built from
commit/verdict.json + runs.jsonl + sampled_replay_p*.json + manifest_verify.json).  This file keeps being written -- the finalizer and
the harness still read its execution-configuration fields -- and carries `"deprecated": "F-r19-int-21; see verdict.json"`.

Same fields as the R13/R14 inline block (rc, runs, openings, tokens, control/treatment walls, coverage, pass) PLUS the execution
configuration the finalizer/renderer label the overhead with (TABLE-RULES `Commitment overhead`: "variant + retention labelled";
brief §11: collector, retention, chunked-prefill and memory settings recorded per attempt):
  cmd              the commit_delta command line actually run (first line of commit.log, written by row_pod.sh before the run)
  committer        commit_delta summary.json `committer`
  collector        the collector record of the instrumented runs (runs.jsonl commit.binding.collector: variant, retention, window_mb,
                   window_slots, arena_mb, layout, gpu_tree) — the run's own statement, not the harness's intent
  collector_variant / retention   'windowed' iff window_mb > 0 else 'whole-block'; retention from the collector record (else --retain in cmd)
  engine_settings  --engine-arg k=v pairs + --max-num-seqs from the command line (gpu_memory_utilization, max_num_batched_tokens, ...)
Nothing here is typed by hand: every field is read from commit.log / commit/summary.json / commit/runs.jsonl of the same attempt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _g(r: dict, *ks):
    for k in ks:
        if k in r and r[k] is not None:
            return r[k]
    return None


def summarize(d: str, rc: int) -> dict:
    S = f"{d}/commit/summary.json"
    s = json.load(open(S)) if os.path.exists(S) else {}
    runs = []
    if os.path.exists(f"{d}/commit/runs.jsonl"):
        runs = [json.loads(l) for l in open(f"{d}/commit/runs.jsonl") if l.strip()]
    ctl = [r for r in runs if r.get("arm") == "control" or r.get("committer") in (None, "control")]
    trt = [r for r in runs if r not in ctl]
    out = {"rc": rc, "summary_keys": list(s)[:40], "n_runs": len(runs), "n_control": len(ctl), "n_treatment": len(trt),
           "delta_wall_mean": s.get("delta_wall_mean"), "delta_wall": s.get("delta_wall"),
           "openings": [_g(r, "openings_verified", "openings_ok") for r in trt], "tokens_equal": [_g(r, "tokens_equal", "tokens_equal_ref") for r in trt],
           "backlog": [_g(r, "backlog_max", "backlog") for r in trt], "run_root": [(_g(r, "run_root", "root") or (r.get("commit") or {}).get("run_root") or "")[:16] for r in trt],
           "coverage": [_g(r, "coverage", "manifest_coverage", "bound_classes") for r in trt],
           "control_wall": [_g(r, "wall_s", "wall") for r in ctl], "treatment_wall": [_g(r, "wall_s", "wall") for r in trt],
           "treatment_keys": list(trt[0])[:60] if trt else []}
    # ---- execution configuration (r15 exec): the run's own record of its collector + the command line it ran with
    cmd = None
    if os.path.exists(f"{d}/commit.log"):
        first = open(f"{d}/commit.log", encoding="utf-8", errors="replace").readline().strip()
        if first.startswith("$ "):
            cmd = first[2:]
    out["cmd"] = cmd
    out["committer"] = s.get("committer")
    coll = None
    for r in trt:
        b = (r.get("commit") or {}).get("binding") if isinstance(r.get("commit"), dict) else None
        if isinstance(b, dict) and isinstance(b.get("collector"), dict):
            coll = b["collector"]
            break
    out["collector"] = coll
    if coll is not None:
        out["collector_variant"] = "windowed" if int(coll.get("window_mb") or 0) > 0 else "whole-block"
        out["retention"] = str(coll.get("retention")).lower() if coll.get("retention") is not None else None
    elif cmd:
        out["collector_variant"] = "windowed" if re.search(r"--window-mb\s+(?!0\b)\d+", cmd) else "whole-block"
        m = re.search(r"--retain\s+(host|device|none|sample)", cmd)
        out["retention"] = m.group(1) if m else None
    else:
        out["collector_variant"] = None
        out["retention"] = None
    es = {}
    if cmd:
        for k, v in re.findall(r"--engine-arg\s+([A-Za-z_]+)=([^\s]+)", cmd):
            try:
                es[k] = json.loads(v)
            except Exception:
                es[k] = v
        m = re.search(r"--max-num-seqs\s+(\d+)", cmd)
        if m:
            es["max_num_seqs"] = int(m.group(1))
        m = re.search(r"--openings\s+(\d+)", cmd)
        if m:
            es["openings"] = int(m.group(1))
        m = re.search(r"--pairs\s+(\d+)", cmd)
        if m:
            es["pairs"] = int(m.group(1))
        es["open_after_release"] = "--open-after-release" in cmd
        es["required_manifest"] = bool(re.search(r"--required-manifest\s+\S+", cmd))
        m = re.search(r"--expect-root\s+([0-9a-f]{64})", cmd)
        if m:
            es["expect_root"] = m.group(1)
    out["engine_settings"] = es or None
    # engine-side facts the runs recorded (achieved, not configured): steps and generated tokens per run
    out["engine_steps"] = [r.get("engine_steps") for r in runs]
    out["generated_tokens"] = [r.get("generated_tokens") for r in runs]
    ok = rc == 0 and len(trt) >= 1 and all((o is None) or (o == 64) for o in out["openings"]) and all(t is not False for t in out["tokens_equal"])
    out["pass"] = bool(ok)
    out["deprecated"] = "F-r19-int-21; see verdict.json"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("row_dir")
    ap.add_argument("rc", type=int)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = summarize(a.row_dir, a.rc)
    json.dump(out, open(a.out or f"{a.row_dir}/commit_summary.json", "w"), indent=1, default=str)
    print("PASS" if out["pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
