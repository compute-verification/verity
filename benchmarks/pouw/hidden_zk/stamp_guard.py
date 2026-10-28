"""Scratch guard (removed before review): the flat H1 unit's circuit file at each shape by the auditor's command, its SHA-512,
size, peak RSS and wall time. ``python stamp_guard.py OUT.json K,R,C ... [--env NAME=V ...]``, run from anywhere."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VLLM = Path(__file__).resolve().parents[3] / "integrations" / "vllm"
BIND = {"partition": "0" * 128, "program": "0" * 128}

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
        shapes.append(tuple(int(x) for x in a.split(",")))
res = []
for k, r, c in shapes:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"k{k}.circuit.txt"
        t = time.time()
        p = subprocess.run(["/usr/bin/time", "-v", sys.executable, "-m", "tests.protocols.pouw_h1_circuits", "--lower", str(k),
                            str(r), str(c), json.dumps(BIND), str(path)], cwd=VLLM, env=env, capture_output=True, text=True)
        wall = time.time() - t
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
        e = {"shape": [k, r, c], "rc": p.returncode, "wall_s": round(wall, 2),
             "peak_gib": round(int(m[1]) / 2 ** 20, 3) if m else None}
        if p.returncode == 0:
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
            e["unit_sha256"] = meta.get("unit_sha256")
        else:
            e["stderr"] = p.stderr[-3000:]
        print(json.dumps(e), flush=True)
        res.append(e)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1) + "\n")
