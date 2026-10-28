"""Scratch guard (removed before review): the flat H1 unit's circuit file at each shape by the auditor's command, its SHA-512,
size, k_log, peak RSS (the largest process's) and wall time. ``python stamp_guard.py OUT.json K,R,C|lane:K:R:C|combine:K:R:C ...
[--env NAME=V ...]``, run from anywhere: a ``lane`` or ``combine`` item is that leaf statement's file (`stamp_lanes.py`)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLLM = HERE.parents[2] / "integrations" / "vllm"
BIND = {"partition": "0" * 128, "program": "0" * 128}
#: one shape in its own process, which reports its descendants' largest RSS (ru_maxrss of RUSAGE_CHILDREN)
RUN = ("import json, resource, subprocess, sys; p = subprocess.run(sys.argv[1:], capture_output=True, text=True); "
       "print(json.dumps({'rc': p.returncode, 'maxrss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "
       "'err': p.stderr[-6000:]}))")

out = Path(sys.argv[1])
shapes, env = [], dict(os.environ)
env.pop("H1_TMPL", None)
args = sys.argv[2:]
while args:
    a = args.pop(0)
    if a == "--env":
        n, _, v = args.pop(0).partition("=")
        env[n] = v
    else:
        shapes.append(("tile", *map(int, a.split(","))) if ":" not in a else (a.split(":")[0], *map(int, a.split(":")[1:])))
res = []
for kind, k, r, c in shapes:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"k{k}.circuit.txt"
        t = time.time()
        cmd = ([sys.executable, "-m", "tests.protocols.pouw_h1_circuits", "--lower", str(k), str(r), str(c), json.dumps(BIND), str(path)]
               if kind == "tile" else [sys.executable, str(HERE / "stamp_lanes.py"), kind, str(k), str(r), str(c), str(path)])
        p = subprocess.run([sys.executable, "-c", RUN, *cmd], cwd=VLLM, env=env, capture_output=True, text=True)
        wall = time.time() - t
        rep = json.loads(p.stdout.strip().splitlines()[-1])
        e = {"unit": kind, "shape": [k, r, c], "rc": rep["rc"], "wall_s": round(wall, 2), "peak_gib": round(rep["maxrss_kb"] / 2 ** 20, 3),
             "stamp": [ln for ln in rep["err"].splitlines() if ln.startswith(("STAMP", "fast_stage"))]}
        if rep["rc"] == 0:
            h = hashlib.sha512()
            with open(path, "rb") as f:
                while b := f.read(1 << 24):
                    h.update(b)
            e["sha512"], e["bytes"] = h.hexdigest(), path.stat().st_size
            with open(path) as f:
                f.readline()
                meta = json.loads(f.readline()[5:])
                line = f.readline()
            e["unit_rows"] = int(line.split()[2]) if line.startswith("CIRCUIT unit") else None
            e["unit_sha256"], e["k_log"] = meta.get("unit_sha256"), meta.get("k_log")
        else:
            e["stderr"] = rep["err"][-3000:]
        print(json.dumps(e), flush=True)
        res.append(e)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + "\n")
