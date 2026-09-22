"""Assemble the §16.7 bench experiment directory from a pulled bench root (laptop side, CPU only).

Writes ``manifest.json`` (device, software, configurations with their flags and kwargs, workload,
models; ``profile_id`` = first 12 hex of its sha256), derives ``experiment_id =
<case>-<profile_id>-<UTC of the matrix start>`` and runs ``summarize`` into
``<results_root>/<experiment_id>/`` with the raw per-run JSON copied under ``bench/``.

    python -m verity_vllm_adapter.bench.collect --pulled /tmp/pull/bench --results-root fixtures/results \
        --case H100COST --adapter-revision <sha> [--experiment-id <fixed id when re-collecting>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def canonical(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulled", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--case", default="H100COST")
    ap.add_argument("--worker", default="W4d")
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--usd-per-hour", type=float, default=3.49)
    args = ap.parse_args()

    pulled = Path(args.pulled)
    engines = sorted(pulled.glob("*/*/engine.json"))
    if not engines:
        raise SystemExit("no engine.json under the pulled root")
    e0 = json.loads(engines[0].read_text())
    configs: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for p in engines:
        e = json.loads(p.read_text())
        c = e["config"]
        configs.setdefault(c["id"], {"label": c["label"], "env_flags": c["env_flags"], "harness_env": c["harness_env"],
                                      "engine_kwargs": {k: v for k, v in c["engine_kwargs"].items()}})
        models.setdefault(e["model"]["repo"], {"revision": e["model"]["revision"], "num_layers": e["model"]["num_layers"],
                                                "hidden_size": e["model"]["hidden_size"], "num_kv_heads": e["model"]["num_kv_heads"]})
    manifest = {
        "schema": "verity-vllm/bench/manifest/v1",
        "purpose": "CONTRACT §16.7 production-profile cost: same pod, same workload, ordinary -> batch-invariant -> batch-invariant eager",
        "worker": args.worker,
        "device": e0["device"],
        "software": e0["software"],
        "configurations": configs,
        "workload": {"shapes": ["128x128", "512x64"], "requests_per_shape": 256, "reps": 3, "prompts": "seeded random token ids (seed 20260907+input_len), distinct, no shared 16-token block",
                     "sampling": "greedy, ignore_eos, max_tokens = output_len", "prefix_cache": "stock (on), reset before every timed pass; cached_tokens recorded (0 everywhere)",
                     "warmup": "one 32-request pass on other-seed prompts, reported separately (phase=warmup)"},
        "models": models,
        "adapter": {"revision": args.adapter_revision, "harness": "verity_vllm_adapter/bench/throughput.py"},
    }
    profile_id = hashlib.sha256(canonical(manifest)).hexdigest()[:12]
    matrix_log = pulled / "matrix.log"
    utc = None
    if matrix_log.exists():
        m = re.search(r"\[(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):\d\dZ\] matrix start", matrix_log.read_text())
        if m:
            utc = f"{m.group(1)}{m.group(2)}{m.group(3)}T{m.group(4)}{m.group(5)}Z"
    exp = args.experiment_id or f"{args.case}-{profile_id}-{utc or 'unknown'}"
    out = Path(args.results_root) / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({**manifest, "profile_id": profile_id, "experiment_id": exp}, indent=1, sort_keys=True) + "\n")
    dest = out / "bench"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pulled, dest, ignore=shutil.ignore_patterns("*.FAILED.tmp"))
    rc = subprocess.run([sys.executable, "-m", "verity_vllm_adapter.bench.summarize", "--bench-root", str(dest), "--experiment-id", exp,
                         "--case", args.case, "--profile-id", profile_id, "--worker", args.worker, "--adapter-revision", args.adapter_revision,
                         "--matrix-log", str(dest / "matrix.log"), "--usd-per-hour", str(args.usd_per_hour), "--out", str(out)]).returncode
    if rc:
        raise SystemExit(rc)
    print(f"experiment {exp} profile_id {profile_id} -> {out}")


if __name__ == "__main__":
    main()
