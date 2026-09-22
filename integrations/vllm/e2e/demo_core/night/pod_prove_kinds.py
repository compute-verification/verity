#!/usr/bin/env python3
"""[NIGHT 2026-09-12, E8/G4] Pod driver (stdlib only): prove every statement of a batch set through an SP1 host on the GPU and record,
per batch, the host's own timers (``setup_seconds``, ``prove_seconds``, ``verify_seconds``), cycles (``execute``), proof bytes, the
host process tree's peak RSS (GNU time) and the GPU's PEAK memory (``nvidia-smi --query-gpu=memory.used`` sampled every 0.2 s while
the prover runs; the idle baseline is subtracted and also recorded).  One host process per proof, as the campaign and the live
driver do (the GPU prover's cold start is in ``setup_seconds``/``wall_s``, never in ``prove_seconds``).

    python3 pod_prove_kinds.py --host BIN --batches DIR[,DIR...] --out records.jsonl [--modes core] [--reps 1] [--witness-name witness.bin]

Each batch directory holds ``statement.bin`` and a witness (``witness-fmt2.bin`` or ``witness.bin``); ``meta.json`` / ``census.json``
beside them are copied into the record when present.  Schema ``veritor/night-e8/prove-record/v1``; every number is MEASURED on the
machine named in the ``context`` row.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

T0 = time.time()


def log(msg: str) -> None:
    print(f"[pod_prove_kinds +{time.time() - T0:7.1f}s] {msg}", flush=True)


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class GpuSampler:
    def __init__(self) -> None:
        self.peak: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
                self.samples += 1
            self._stop.wait(0.2)

    def __enter__(self) -> "GpuSampler":
        self._t.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._t.join(timeout=2)


def run(argv: list[str], timeout: float = 7200, env: dict[str, str] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    if os.path.exists("/usr/bin/time"):
        argv = ["/usr/bin/time", "-f", "MAXRSS_KB=%M", *argv]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s", "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    rec: dict[str, Any] = {"ok": p.returncode == 0, "rc": p.returncode, "wall_s": wall}
    m = re.search(r"MAXRSS_KB=(\d+)", p.stderr)
    if m:
        rec["max_rss_kb"] = int(m.group(1))
    if not p.returncode == 0:
        rec["stderr_tail"] = p.stderr[-1500:]
    # the host prints one JSON object on stdout
    try:
        body = p.stdout.strip()
        start = body.find("{")
        rec.update(json.loads(body[start:]) if start >= 0 else {})
    except Exception as e:  # pragma: no cover
        rec["stdout_tail"] = p.stdout[-800:]
        rec["parse_error"] = str(e)
    return rec


def witness_of(d: Path, preferred: str | None) -> Path | None:
    for name in ([preferred] if preferred else []) + ["witness-fmt2.bin", "witness.bin", "witness-fmt3.bin"]:
        if name and (d / name).exists():
            return d / name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--batches", required=True, help="comma-separated directories; each is a batch dir or a parent of batch dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="core")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--witness-name", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-execute", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=0.0)
    a = ap.parse_args()

    host = str(Path(a.host).resolve())
    env = dict(os.environ)
    env.setdefault("SP1_PROVER", "cuda")
    dirs: list[Path] = []
    for root in [Path(p) for p in a.batches.split(",") if p]:
        if (root / "statement.bin").exists():
            dirs.append(root)
        else:
            dirs.extend(sorted(d for d in root.rglob("*") if d.is_dir() and (d / "statement.bin").exists()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = run([host, "info"], env={**env, "SP1_PROVER": "cpu"})
    baseline = gpu_used_mib()
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ctx = {"schema": "veritor/night-e8/pod-context/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": host, "host_info": {k: info.get(k) for k in ("elf_sha256", "vk_hash", "sp1_version")},
           "gpu": gpu_name, "gpu_idle_used_mib": baseline, "machine": platform.node(), "nproc": os.cpu_count(), "sp1_prover": env["SP1_PROVER"], "label": a.label, "batches": [str(d) for d in dirs]}
    with out.open("a") as fh:
        fh.write(json.dumps(ctx) + "\n")
    log(f"host {info.get('elf_sha256', '?')[:8]} gpu {gpu_name} idle {baseline} MiB; {len(dirs)} batches, modes {a.modes}, reps {a.reps}")

    for d in dirs:
        if a.deadline_min and (time.time() - T0) / 60 > a.deadline_min:
            log("deadline reached; stopping")
            break
        st = str(d / "statement.bin")
        wit = witness_of(d, a.witness_name)
        if wit is None:
            log(f"{d}: no witness; skipped")
            continue
        meta: dict[str, Any] = {}
        for name in ("meta.json", "census.json"):
            if (d / name).exists():
                try:
                    meta[name.split(".")[0]] = json.loads((d / name).read_text())
                except Exception:
                    pass
        ex = {} if a.skip_execute else run([host, "execute", "--statement", st, "--witness", str(wit)], env={**env, "SP1_PROVER": "cpu"})
        for mode in [m for m in a.modes.split(",") if m]:
            for rep in range(a.reps):
                proof = d / f"proof.{mode}.rep{rep}.bin"
                with GpuSampler() as gs:
                    pr = run([host, "prove", "--statement", st, "--witness", str(wit), "--out", str(proof), "--mode", mode], env=env)
                peak = gs.peak
                rec = {"schema": "veritor/night-e8/prove-record/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "batch": str(d), "name": d.name, "mode": mode, "rep": rep,
                       "statement_bytes": os.path.getsize(st), "witness_bytes": os.path.getsize(wit), "witness_file": wit.name,
                       "execute": {k: ex.get(k) for k in ("ok", "wall_s", "total_cycles", "verdict", "cycle_tracker", "syscalls", "error") if k in ex},
                       "prove": {k: pr.get(k) for k in ("ok", "rc", "wall_s", "setup_seconds", "prove_seconds", "verify_seconds", "proof_bytes", "shards", "verdict", "error", "stderr_tail", "max_rss_kb", "elf_sha256", "vk_hash") if k in pr},
                       "gpu_peak_used_mib": peak, "gpu_idle_used_mib": baseline, "gpu_peak_delta_mib": (peak - baseline) if (peak is not None and baseline is not None) else None, "gpu_samples": gs.samples,
                       "meta": meta, "label": a.label}
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"{d.name} {mode} rep{rep}: cycles={ex.get('total_cycles')} prove_ok={pr.get('ok')} setup_s={pr.get('setup_seconds')} prove_s={pr.get('prove_seconds')} "
                    f"verify_s={pr.get('verify_seconds')} bytes={pr.get('proof_bytes')} wall={pr.get('wall_s'):.1f}s gpu_peak={peak}MiB (idle {baseline}) rss={pr.get('max_rss_kb')}KB")
                if not pr.get("ok"):
                    log(f"   stderr: {str(pr.get('stderr_tail'))[-400:]}")
                try:
                    proof.unlink()
                except OSError:
                    pass
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
