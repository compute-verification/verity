"""One SHA-512 compression's latency on one SM (shalat.cu's forms), the A table's critical-path unit, on the leased GPU.

    SBENCH_SCRIPT=shalat.py bash benchmarks/hash/dither/sbench.sh [--n 64] [--reps 5]

Builds shalat.cu (with h1c.cu) for the GPU's arch into the scratch directory, runs each form `reps` times after a warm-up,
and writes shalat.json: per form its cycles and ns a compression (the median rep, lane 0 of each consumer warp), the SM
clock that implies, and whether its final states equal form 0's (exit 2 if any form's differ).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

FORMS = {0: "sha512_compress, inline schedule", 1: "ap_cmp + ap_serve (h1c_a_call2_k's)", 2: "paired, inlined",
         3: "paired, inlined, reassociated round", 4: "form 3, off-chain sums on IMAD"}


def build(scratch: Path) -> Path:
    import torch

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    here = Path(__file__).resolve().parent
    so = scratch / "libshalat.so"
    cmd = [os.environ.get("NVCC", "nvcc"), "-O3", "-std=c++17", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-Xptxas", "-v", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(here / "shalat.cu")]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (scratch / "build-shalat.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(11)
    return so


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    import torch

    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    lib = ctypes.CDLL(str(build(scratch)))
    (out / "build-shalat.log").write_text((scratch / "build-shalat.log").read_text())
    lib.shalat.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
    dev = torch.device("cuda")
    cyc = torch.zeros(4, dtype=torch.int64, device=dev)
    ns = torch.zeros(4, dtype=torch.int64, device=dev)
    st = torch.zeros(128 * 8, dtype=torch.int64, device=dev)
    runs = [(0, 1), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    doc = {"gpu": torch.cuda.get_device_name(), "n": a.n, "reps": a.reps, "forms": {}}
    ref = None
    ok = True
    for form, warps in runs:
        reps = []
        for rep in range(a.reps + 1):
            st.fill_(-0x5A5A5A5A5A5A5A5B)  # 0xA5 poison: a form that writes nothing fails the comparison
            rc = lib.shalat(form, warps, a.n, cyc.data_ptr(), ns.data_ptr(), st.data_ptr())
            torch.cuda.synchronize()
            if rc:
                raise SystemExit(f"shalat form {form}: rc {rc}")
            if rep:
                consumers = 2 if form else warps
                reps.append((cyc[:consumers].max().item() / a.n, ns[:consumers].max().item() / a.n))
        states = st.view(128, 8)[:32].cpu()
        if ref is None:
            ref = states.clone()
        same = bool(torch.equal(states, ref))
        ok &= same
        c = statistics.median(r[0] for r in reps)
        t = statistics.median(r[1] for r in reps)
        key = f"{form}w{warps}"
        doc["forms"][key] = {"form": form, "what": FORMS[form], "warps": warps, "cycles": round(c, 1),
                             "ns": round(t, 1), "mhz": round(1e3 * c / t, 1), "states_equal_form0": same,
                             "cycles_reps": [round(r[0], 1) for r in reps]}
        print(json.dumps({"shalat": key, **{k: doc["forms"][key][k] for k in ("cycles", "ns", "mhz", "states_equal_form0")}}),
              flush=True)
    doc["ok"] = ok
    (out / "shalat.json").write_text(json.dumps(doc, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
