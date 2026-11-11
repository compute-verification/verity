"""(f) The retired `commit_summary.json` re-derives: `summarize(row_dir, rc)` (the harness's READER for old records -- the writer,
`verity_vllm.pipeline.commit_summary`, is retired: `verdict.json` is the verdict of record) with the CURRENT
code over the row's `commit/summary.json`, `commit/runs.jsonl` and `commit.log` gives the recorded summary's stable fields (run
counts, tokens_equal, committer, collector variant / retention, engine settings, engine steps, generated tokens, `pass`).

NOTE: `openings` and `coverage` are `None` on every reference row (stale keys: runs.jsonl no longer carries them),
so `pass` degenerates to `rc == 0 and tokens_equal` here.  The Commit VERDICT of record is `commit/verdict.json` (`commit_pass`,
`required_classes_source`) + `commit/sampled_replay_p*.json`, lifted beside the summary in `commit_verdict`; a v2 must state the same.
STRONG on the lifted verdict fields; the summary itself is a derived convenience.  Wall-clock fields (`delta_wall*`, `*_wall`) are
excluded: they are measurements, not reproducible facts.

A candidate row written after the flip has no `commit_summary.json`: its `actual` is the same reader run over the candidate's own
`commit/summary.json` + `commit/runs.jsonl` + `commit.log` with `rc` = its `verdict.json` `compat.rc` -- the execution-configuration
facts (committer, collector, engine settings, run counts) must still agree with the reference; `pass` is then `rc == 0 and tokens_equal`.
A reference row recorded after the flip (#73's epoch record) is read the same way in self-check.
"""
from __future__ import annotations

import json
import os
import re

from .. import resolver

NAME = "commit_summary"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("commit_summary.json", "commit/summary.json", "commit/runs.jsonl", "commit.log", "commit/verdict.json")
STABLE = ("rc", "n_runs", "n_control", "n_treatment", "tokens_equal", "committer", "collector_variant", "retention", "engine_settings",
          "engine_steps", "generated_tokens", "pass", "openings", "coverage")
VERDICT = ("commit_pass", "runs_instrumented", "runs_failed", "first_fail_reason", "required_classes_source", "required_manifest")


def _g(r: dict, *ks):
    for k in ks:
        if k in r and r[k] is not None:
            return r[k]
    return None


def summarize(d: str, rc: int) -> dict:
    """The pre-P5 `commit_summary.json` derivation (READER; nothing writes this file any more).  Every field is read from
    `<d>/commit.log` (first line: the commit_delta command line), `<d>/commit/summary.json` and `<d>/commit/runs.jsonl`."""
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
    out["engine_steps"] = [r.get("engine_steps") for r in runs]
    out["generated_tokens"] = [r.get("generated_tokens") for r in runs]
    ok = rc == 0 and len(trt) >= 1 and all((o is None) or (o == 64) for o in out["openings"]) and all(t is not False for t in out["tokens_equal"])
    out["pass"] = bool(ok)
    out["deprecated"] = "F-r19-int-21; see verdict.json"
    return out


def _post_flip(row: resolver.Row) -> bool:
    """A reference row recorded after the flip (#73's epoch record): `verdict.json` and `commit/summary.json`, no `commit_summary.json`."""
    return not row.has("commit_summary.json") and row.has("verdict.json") and row.has("commit/summary.json")


def _rc(vr: dict) -> int:
    rc = (vr.get("compat") or {}).get("rc")
    return int(rc) if rc is not None else (0 if vr.get("outcome") == "PASS" else 1)


def applies(row: resolver.Row) -> str | None:
    if not row.has("commit_summary.json") and not _post_flip(row):
        return "no commit_summary.json on this row (Commit never closed)"
    return None


def _lift(cs: dict, verdict: dict | None) -> dict:
    out = {k: cs.get(k) for k in STABLE}
    coll = cs.get("collector") or {}
    out["collector"] = {k: coll.get(k) for k in ("variant", "retention", "window_mb", "window_slots", "layout")} if coll else None
    if verdict is not None:
        out["commit_verdict"] = {k: verdict.get(k) for k in VERDICT if k in verdict}
    return out


def expected(row: resolver.Row) -> dict:
    v = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
    if _post_flip(row):                                     # no summary file was ever written: the reader over the record is the lift
        return _lift(summarize(str(row.path("commit/summary.json").parents[1]), _rc(row.json("verdict.json"))), v)
    return _lift(row.json("commit_summary.json"), v)


def actual(row: resolver.Row) -> dict:
    croot = row.candidate_root()
    if croot is not None and (croot / "commit_summary.json").is_file():       # a pre-P5 candidate record
        v = json.load(open(croot / "commit" / "verdict.json")) if (croot / "commit" / "verdict.json").is_file() else None
        return dict(_lift(json.load(open(croot / "commit_summary.json")), v), source="candidate")
    if croot is not None and (croot / "commit" / "summary.json").is_file():   # a post-flip candidate: no summary file; read its evidence
        vr = json.load(open(croot / "verdict.json")) if (croot / "verdict.json").is_file() else {}
        v = json.load(open(croot / "commit" / "verdict.json")) if (croot / "commit" / "verdict.json").is_file() else None
        return dict(_lift(summarize(str(croot), _rc(vr)), v), source="candidate")
    if _post_flip(row):                                     # a post-flip reference row: the same reader over its own evidence
        row_dir = row.path("commit/summary.json").parents[1]
        for rel in ("commit/runs.jsonl", "commit.log"):
            if row.has(rel) and not (row_dir / rel).is_file():
                raise resolver.NotResolvable(f"{row.key}: {rel} is not beside commit/summary.json at {row_dir}")
        v = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
        return dict(_lift(summarize(str(row_dir), _rc(row.json("verdict.json"))), v), source="recomputed")
    rc = int(row.json("commit_summary.json").get("rc", 0))
    row_dir = row.path("commit_summary.json").parent
    if not (row_dir / "commit" / "summary.json").is_file():
        raise resolver.NotResolvable(f"{row.key}: commit/summary.json not beside commit_summary.json at {row_dir}")
    if not (row_dir / "commit.log").is_file() and row.has("commit.log"):
        # the store's per-row `records` tree omitted commit.log (engine_settings' source); it lives in the shared `commit_logs` tree
        # (fixtures.toml [artifacts], resolved + sha-checked by row.path) -- summarize() reads one directory, so re-join the two here
        view = resolver.scratch(row) / "commit_view"
        view.mkdir(parents=True, exist_ok=True)
        for name, src in (("commit", row_dir / "commit"), ("commit_summary.json", row_dir / "commit_summary.json"), ("commit.log", row.path("commit.log"))):
            dst = view / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        row_dir = view
    cs = summarize(str(row_dir), rc)
    v = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
    return dict(_lift(cs, v), source="recomputed")
