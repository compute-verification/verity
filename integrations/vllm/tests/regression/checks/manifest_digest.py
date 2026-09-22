"""(a) The manifest of record reproduces: rebuild `manifest.json` from the row's request Programs x declared workload with the CURRENT
code, by the exact path `row_pod.sh` used (`required_manifest build-global --fixture WL --programs-root ROW --workload-digest WD` at
B>=2; `required_manifest build --program ROW/build_request --workload WL` at B=1) and compare:

    manifest_digest     sha256 over the sorted (family + IDENTITY_KEY) rows     -- `required_manifest.manifest_digest`
    program_digest      the Program digest the identities are keyed by          -- GP-01's workload digest (B>=2) / the request Program's (B=1)
    identities          the identity count
    identity_rows_sha256  sha256 over the sorted canonical IDENTITY_KEY tuples (family-blind): byte-for-byte equality of the identity SET
                        with the recorded manifest.json when that file is resolvable (222 MB for #23: pod only)

Expected is lifted from `commit/manifest_verify.json` (used_digest / used_program_digest / used_identities; the v1 Commit's own
rebuild said `ok`) and, when the recorded manifest.json is reachable, from the file itself.  STRONG: a single identity row moved,
dropped or renamed changes the digest.  TP rows (#70, #75: rank Programs under build/rank*/) are not built by this path -> not applicable.
The engine is the code's default (traversal, inventory cross-check on) exactly as the row's Commit ran it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import resolver

NAME = "manifest_digest"
TIER = "T0"          # inputs: Programs 50-300 MB + workload; the rebuild itself is CPU-minutes (run it on a pod tonight)
STRENGTH = "strong"
INPUTS = ("manifest.json", "commit/manifest_verify.json", "build_summary.json", "build_request*/instances.json.gz", "workloads/<row>.json")


def applies(row: resolver.Row) -> str | None:
    if row.spec.get("tp", 1) > 1:
        return "TP row: rank Programs (build/rank*/) are merged by row_pod_tp2.sh, not by build/build-global"
    if not row.has("commit/manifest_verify.json") and not row.has("manifest.json"):
        return "no manifest of record on this row (Commit never built one)"
    return None


def identity_rows_sha256(identities: list[dict]) -> str:
    from verity_capture.commit.required_manifest import IDENTITY_KEY
    rows = sorted(json.dumps([r[k] if k != "element_range" else list(r[k]) for k in IDENTITY_KEY], separators=(",", ":"), sort_keys=True).encode()
                  for r in identities)
    h = hashlib.sha256()
    for b in rows:
        h.update(b)
        h.update(b"\n")
    return h.hexdigest()


def _summary(man: dict, *, with_rows: bool) -> dict:
    out = {"manifest_digest": man.get("manifest_digest"), "program_digest": man.get("program_digest"),
           "identities": (man.get("populations") or {}).get("identities", len(man.get("identities") or [])),
           "complete": man.get("complete"), "schema": man.get("schema")}
    if with_rows:
        out["identity_rows_sha256"] = identity_rows_sha256(man.get("identities") or [])
    return out


def expected(row: resolver.Row) -> dict:
    out: dict = {}
    if row.has("commit/manifest_verify.json"):
        mv = row.json("commit/manifest_verify.json")
        out.update({"manifest_digest": mv.get("used_digest"), "program_digest": mv.get("used_program_digest"), "identities": mv.get("used_identities"),
                    "complete": True, "v1_verify_ok": mv.get("ok")})
    if row.has("manifest.json"):
        man = row.json("manifest.json")
        out.update(_summary(man, with_rows=True))
    out.pop("v1_verify_ok", None)
    return out


def _rebuild_argv(row: resolver.Row, programs_root: Path, out: Path) -> list[str]:
    wl = row.workload()
    fx = json.load(open(wl))
    b = len(fx.get("requests") or [])
    base = [sys.executable, "-m", "verity_capture.commit.required_manifest"]
    if b > 1:
        bs = row.json("build_summary.json")
        wd = ((bs.get("workload") or {}).get("workload_digest")) or None
        argv = base + ["build-global", "--fixture", str(wl), "--programs-root", str(programs_root), "--out", str(out)]
        if wd:
            argv += ["--workload-digest", wd]
        return argv
    return base + ["build", "--program", str(programs_root / "build_request"), "--workload", str(wl), "--out", str(out)]


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("manifest.json")
    if cand is not None:                                   # candidate mode: the v2 implementation's manifest.json IS the thing under test
        return dict(_summary(json.load(open(cand)), with_rows=True), source="candidate")
    root = row.programs_root()
    if root is None:
        raise resolver.NotResolvable(f"{row.key}: no request Programs (build_request*/instances.json.gz) reachable here")
    scratch = Path(os.environ.get("VERITY_REGRESSION_SCRATCH") or tempfile.gettempdir()) / "r19-regress" / row.key
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "manifest.rebuilt.json"
    argv = _rebuild_argv(row, root, out)
    p = subprocess.run(argv, cwd=str(resolver.VLLM_ROOT), capture_output=True, text=True)
    (scratch / "manifest.rebuilt.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
    if p.returncode != 0 or not out.is_file():
        return {"rebuild_rc": p.returncode, "rebuild_argv": argv, "rebuild_tail": (p.stdout + p.stderr).strip()[-1500:], "source": "rebuild"}
    man = json.load(open(out))
    res = dict(_summary(man, with_rows=True), source="rebuild", rebuild_rc=0, rebuild_engine=(man.get("engine") or man.get("populations", {}).get("engine")))
    return res
