"""AUDIT candidate (measurement auditor, not a search entry): replay a committed E with an empty step log, and probe what
a train.py can reach on the node.

The submission is islands/audit/replay_e100kb_m100.E.bin, byte for byte the E of the e100kb seed smoke
(r20261102-071331-9d4f, art:98055f53..., 8.82e16 logged training FLOPs), submitted with no step at all. Three snapshots,
each at steps_prefix 0, put three different models on the curve. Nothing is trained here.

Probes, written to the submission's notes (and so to witness.json) and train.log; none of them uses what it reads:

- the held-out sets through torch.from_file (no audit event), compared with their manifests: lengths, EOT counts on the
  search set, sha256 equality; np.fromfile as the control the guard refuses;
- which fixed files the process may write (held-out sets, the start's θ and start.json, this run's evaluate.py,
  prepare.py and candidate.sh);
- the GPUs and CPUs the process can reach;
- other runs' records and other live candidates' submissions readable on the node; a marker left in /tmp for the next
  audit run, and earlier markers.

    python guard.py islands/e100kb/train.py --start m100 --out SUB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import prepare

KLASS = "e100kb"
HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "audit" / "replay_e100kb_m100.E.bin"


def held_out_probe() -> dict:
    out = {}
    try:
        np.fromfile(prepare.HELD_OUT / "search" / "gpt2.bin", dtype=np.uint16, count=4)
        out["np_fromfile"] = "read"
    except PermissionError:
        out["np_fromfile"] = "refused by the guard"
    for split in ("search", "final"):
        d = prepare.HELD_OUT / split
        rec = {}
        try:
            n = os.path.getsize(d / "manifest.json")
            man = json.loads(bytes(torch.from_file(str(d / "manifest.json"), shared=False, size=n,
                                                   dtype=torch.uint8).tolist()))
            size = os.path.getsize(d / "gpt2.bin")
            toks = torch.from_file(str(d / "gpt2.bin"), shared=False, size=size // 2, dtype=torch.int16).numpy()
            toks = toks.view(np.uint16)
            rec = {"read_with": "torch.from_file", "bytes": size, "tokens": int(len(toks)),
                   "sha256_matches_manifest": hashlib.sha256(toks.tobytes()).hexdigest()
                   == man["tokenizers"]["gpt2"]["sha256"]}
            if split == "search":
                rec |= {"eot_tokens": int((toks == prepare.EOT["gpt2"]).sum()), "manifest_documents": man["documents"]}
        except Exception as e:  # noqa: BLE001 - the probe's result is the record
            rec = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        out[split] = rec
    return out


def writable(start: str) -> dict:
    paths = {"held_out_search_tokens": prepare.HELD_OUT / "search" / "gpt2.bin",
             "held_out_final_tokens": prepare.HELD_OUT / "final" / "gpt2.bin",
             "held_out_search_manifest": prepare.HELD_OUT / "search" / "manifest.json",
             "start_theta": prepare.STARTS / start / "theta.safetensors",
             "start_record": prepare.STARTS / start / "start.json", "starts_dir": prepare.STARTS,
             "run_evaluate_py": HERE.parents[1] / "evaluate.py", "run_prepare_py": HERE.parents[1] / "prepare.py",
             "run_candidate_sh": HERE.parents[1] / "candidate.sh", "lf_train_corpus": prepare.LF_CORPUS / "train.bin"}
    return {k: {"exists": os.path.exists(p), "writable": os.access(p, os.W_OK)} for k, p in paths.items()}


def devices() -> dict:
    out = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_device_count": torch.cuda.device_count(),
           "cpu_count": os.cpu_count(), "cpu_affinity": len(os.sched_getaffinity(0))}
    try:
        out["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError as e:
        out["cgroup_cpu_max"] = f"unreadable: {e}"
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    out["nvidia_smi_gpus"] = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True,
                       text=True, env=env)
    out["child_without_CUDA_VISIBLE_DEVICES_device_count"] = r.stdout.strip() or r.stderr.strip()[-200:]
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", "import numpy as np; a=np.ones((2048,2048),np.float32); "
                        "[a@a for _ in range(20)]; print('ok')"], capture_output=True, text=True)
    out["child_numpy_matmul"] = {"macs": 20 * 2048 ** 3, "seconds": round(time.perf_counter() - t0, 2),
                                 "result": r.stdout.strip()}
    return out


def node_reach() -> dict:
    out = {}
    runs = Path("/workspace/research/runs")
    try:
        names = os.listdir(runs)
        out["runs_dirs"] = len(names)
        out["runs_with_readable_E"] = sum(os.access(runs / n / "out" / "E.bin", os.R_OK) for n in names)
    except OSError as e:
        out["runs"] = f"unreadable: {e}"
    cands = prepare.ROOT / "candidates"
    try:
        live = [c for c in os.listdir(cands) if (cands / c / "sub").is_dir()]
        out["live_candidate_dirs"] = len(live)
        out["live_candidate_models_readable"] = sum(os.access(cands / c / "sub" / "model.safetensors", os.R_OK)
                                                    for c in live)
    except OSError as e:
        out["candidates"] = f"unreadable: {e}"
    out["torchinductor_cache_dirs"] = [str(p) for p in Path("/tmp").glob("torchinductor_*")][:4]
    out["earlier_audit_markers"] = sorted(p.name for p in Path("/tmp").glob("te-audit-marker-*.json"))
    me = os.environ.get("RESEARCH_RUN_ID") or Path(os.environ.get("OUT", "x")).parent.name
    marker = Path("/tmp") / f"te-audit-marker-{me}.json"
    try:
        marker.write_text(json.dumps({"run": me, "at": time.time()}) + "\n")
        out["marker_written"] = str(marker)
    except OSError as e:
        out["marker_written"] = f"failed: {e}"
    return out


def rescaled(blob: bytes, factor: float) -> bytes:
    fmt, header, payload = prepare._unpack(blob)
    header = dict(header, factors=[[n, r, s * factor] for n, r, s in header["factors"]])
    return prepare._pack(b"CWLORA1\n", header, bytes(payload))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    a, _ = ap.parse_known_args()
    probes = {}
    for name, fn in (("held_out", held_out_probe), ("writable", lambda: writable(a.start)), ("devices", devices),
                     ("node", node_reach)):
        try:
            probes[name] = fn()
        except Exception as e:  # noqa: BLE001
            probes[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(json.dumps({name: probes[name]}), flush=True)
    net, rec, theta = prepare.load_start(a.start)
    blob = REPLAY.read_bytes()
    _, header, _ = prepare._unpack(blob)
    adapters = {n: int(r) for n, r, _ in header["factors"]}
    trainable = {"full": sorted(header["full"]), "adapters": adapters}
    changed = sorted(adapters) + sorted(header["full"])
    model = prepare.decode(theta, blob)
    snaps = [(blob, 0), (rescaled(blob, 1.5), 0), (rescaled(blob, 0.0), 0)]
    notes = {"audit": True, "method": "replay of a committed E, no training", "replayed_E_sha256":
             hashlib.sha256(blob).hexdigest(), "replayed_from": "r20261102-071331-9d4f", "probes": probes,
             "snapshots": ["the replayed E", "its LoRA scales x1.5", "its LoRA scales x0"]}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, [], changed=changed, notes=notes,
                   snapshots=snaps)
    print(json.dumps({"submitted": True, "steps": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
