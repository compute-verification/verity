#!/usr/bin/env python3
"""[A9 contract 2026-09-10] `verify-many` emulated over an older `veritor-zk-host` that only has `info` / `verify` (D4's pod binary of
2026-09-09, ELF 9fd00e60, predates D6's `verify-many` -- commit ee6f186a; no cargo on the GPU pod to rebuild the host).

Usage:  VERITOR_HOST_REAL=/path/to/veritor-zk-host python record_v5/host_shim.py verify-many
        (any other subcommand is exec'd on the real host unchanged; `reverify --host` points at a small wrapper script that does this)

Protocol (D6 `verifier_v2/host_verify.py::VerifyMany`): print one `{"event":"ready", ...}` line, then for every stdin job line
`{"id","proof",["statement"]}` print one result line with D6's fields.  Differences, all recorded in every line under `shim`:
  * the verifying key is derived by the REAL host once per `verify` call (not once per process): `job_wall_seconds` includes it,
    `verify_seconds` (the host's own check timing) does not -- the cost class "proof verification" must cite `verify_seconds`;
  * `load_seconds` / `statement_hash_seconds` are not separable here and are reported as null.
The proof check itself (SP1 `client.verify` against the embedded guest's vk) is the real host's, unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _run(host: str, args: list[str]) -> dict:
    res = subprocess.run([host, *args], capture_output=True, text=True, env={**os.environ, "SP1_PROVER": os.environ.get("SP1_PROVER", "cpu"), "RUST_LOG": "warn"})
    lines = [l for l in res.stdout.splitlines() if l.strip().startswith("{")]
    if res.returncode != 0 or not lines:
        raise RuntimeError(f"{host} {' '.join(args)} rc={res.returncode}: {res.stderr[-400:]}")
    return json.loads(lines[-1])


def main() -> int:
    host = os.environ.get("VERITOR_HOST_REAL")
    if not host:
        print("VERITOR_HOST_REAL must name the real veritor-zk-host", file=sys.stderr)
        return 2
    if sys.argv[1:2] != ["verify-many"]:
        os.execv(host, [host, *sys.argv[1:]])
    shim = {"kind": "record_v5/host_shim verify-many over `verify` (vk derived per proof)", "real_host": host}
    t0 = time.perf_counter()
    info = _run(host, ["info"])
    print(json.dumps({"event": "ready", "backend": info.get("backend"), "sp1_version": info.get("sp1_version"), "elf_sha256": info["elf_sha256"], "vk_hash": info["vk_hash"],
                      "guest_features": info.get("guest_features", []), "setup_seconds": time.perf_counter() - t0, "pid": os.getpid(), "shim": shim}), flush=True)
    for i, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        t = time.perf_counter()
        try:
            job = json.loads(line)
            args = ["verify", "--proof", job["proof"]] + (["--statement", job["statement"]] if job.get("statement") else [])
            out = _run(host, args)
            out.update(event="job", id=job.get("id"), job_index=i, load_seconds=None, statement_hash_seconds=None, job_wall_seconds=time.perf_counter() - t, shim=shim)
        except Exception as e:  # noqa: BLE001
            out = {"event": "job", "id": None, "job_index": i, "ok": False, "error": f"{type(e).__name__}: {e}", "job_wall_seconds": time.perf_counter() - t, "shim": shim}
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
