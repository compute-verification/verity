#!/usr/bin/env python3
"""The price of D-3s's forming (`d3s-v0-f16`) on one RTX PRO 6000 Blackwell (sm_120), per element per side (rates.cu).

0. Build: rates.cu with the node's nvcc into OUT, refused on a spill, a stack frame or local memory.
1. Each rate kernel at REPEATS repeats, REPS times each after a warm-up: its price is the slope of the median time over the
   repeats at NVML's SM clock (read before and after), in units of one dense E4M3 MAC (1,024 a clock an SM), per element
   (rate_salt, rate_free: G groups of four entries a thread) or per instruction (the controls: eight a repeat a thread;
   rate_encdec's two F2FPs a step).
2. The floor: SALT and FREE are the cheapest instruction counts per element of the steps that read the unit's stream and
   of those that read the data alone (#295's `d3_trace` at sums = f16, with the negations folded into the adds and |.|
   into HMNMX2; the PRMT that packs two casts' halves is left out, since a hand-scheduled F2FP can merge into the other
   half).  A lower bound on any implementation's price is the larger of the dispatch bound (DISPATCH units an
   instruction: four schedulers a clock, 32 lanes each, against 1,024 MACs) times the count, and each class's count
   times its measured price.  rate_salt and rate_free are what the PTX form achieves, with one feedback LOP3 a group
   (and, in rate_salt, the PRMTs).
3. The gate: rate_salt and rate_free at one repeat on GATE threads; their inputs and outputs go to OUT/rates/gate.json,
   and `rates.py --check OUT` holds them to #295's `verity.ml.tc.fp16` (PR #295's tree on PYTHONPATH).

  python3 rates.py OUT   (env: CUDA, REPS=11, NCP_LIBCUDA)
  PYTHONPATH=/tmp/pr295/packages/verity/src python3 rates.py --check OUT
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import statistics
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "rates.cu")
G = 2
THREADS_PER_BLOCK, BLOCKS_PER_SM = 256, 4
REPEATS = (256, 4352)
GATE = 64
PATTERNS = 1024
MACS_PER_SM_CLOCK = 1024
DISPATCH = 8
MIXES = {"rate_salt": 4 * G, "rate_free": 4 * G}
CONTROLS = {"rate_ffma": ("FFMA", 1), "rate_hfma2": ("HFMA2", 1), "rate_lop3": ("LOP3", 1), "rate_hadd2": ("HADD2", 1),
            "rate_hmul2": ("HMUL2", 1), "rate_hmnmx2": ("HMNMX2", 1), "rate_encdec": ("F2FP", 2),
            "rate_dec": ("F2FP_DEC", 1), "rate_prmt": ("PRMT", 1)}
#: per element: the noise field (LOP3), the centre (HFMA2), the scale (HMUL2), s = e + e o pi (HADD2), the three blocks
#: (HADD2), the three casts (F2FP)
SALT = {"LOP3": 0.5, "HFMA2": 0.5, "HMUL2": 0.5, "HADD2": 2.0, "F2FP": 1.5}
#: per element: the unpack (F2FP_DEC), the prescale (HMUL2), the sum of squares (HFMA2), partner and floor max (HMNMX2)
FREE = {"F2FP_DEC": 0.5, "HMUL2": 0.5, "HFMA2": 0.5, "HMNMX2": 1.0}

_spec = importlib.util.spec_from_file_location("pearl_c_sm120_run", os.path.join(HERE, "..", "pearl_c_sm120", "run.py"))
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)
_spec = importlib.util.spec_from_file_location("ncp_sm120_run", os.path.join(HERE, "run.py"))
ncp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ncp)


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def build(out: str) -> bytes:
    cuda = os.environ.get("CUDA", "/usr/local/cuda-13.0")
    cubin = os.path.join(out, "rates.cubin")
    r = subprocess.run([os.path.join(cuda, "bin", "nvcc"), *ncp.FLAGS, "-o", cubin, KERNEL], capture_output=True, text=True)
    log = r.stdout + r.stderr
    with open(os.path.join(out, "rates_ptxas.txt"), "w") as f:
        f.write(log)
    if r.returncode:
        raise SystemExit(f"nvcc failed ({r.returncode}): see rates_ptxas.txt")
    if any(re.search(r"[1-9]\d* bytes (stack frame|spill)", ln) for ln in log.splitlines()):
        raise SystemExit("ptxas spilled")
    env = dict(os.environ, PATH=os.path.join(cuda, "bin") + os.pathsep + os.environ.get("PATH", ""))
    sass = subprocess.run([os.path.join(cuda, "bin", "cuobjdump"), "-sass", cubin], capture_output=True, text=True, env=env)
    if sass.returncode or re.search(r"\b(LDL|STL)\b", sass.stdout):
        raise SystemExit("rates: cuobjdump failed or the SASS uses local memory")
    with open(os.path.join(out, "rates.sass"), "w") as f:
        f.write(sass.stdout)
    with open(cubin, "rb") as f:
        return f.read()


def f16_normal(rng: random.Random, positive: bool = False) -> int:
    """A finite normal f16 of magnitude in [2^-6, 448)."""
    return (0 if positive or rng.random() < 0.5 else 0x8000) | rng.randint(9, 23) << 10 | rng.getrandbits(10)


def e4m3_codes(rng: random.Random) -> int:
    """Four finite E4M3 codes in a word (0x7F and 0xFF, the NaNs, become 0x7E and 0xFE)."""
    return sum(((c ^ 1) if c & 0x7F == 0x7F else c) << 8 * i for i, c in enumerate(rng.getrandbits(8) for _ in range(4)))


def patterns(kind: str) -> bytes:
    """PATTERNS threads' inputs, G groups of 8 words each.  "salt" (rate_salt, the controls): h0, h1, A0, A1, w0, w1, 0, 0;
    the controls read words 0-7 and the next group's first two.  "free" (rate_free): four codes, six zeros, the floor
    (2^-6 in both halves; rate_free reads group 0's)."""
    rng = random.Random(f"ncp_sm120/rates/{kind}")
    words = []
    for _ in range(PATTERNS * G):
        if kind == "salt":
            words += [f16_normal(rng) | f16_normal(rng) << 16 for _ in range(2)]
            words += [f16_normal(rng, True) | f16_normal(rng, True) << 16 for _ in range(2)]
            words += [rng.getrandbits(32), rng.getrandbits(32), 0, 0]
        else:
            words += [e4m3_codes(rng), 0, 0, 0, 0, 0, 0, 0x24002400]
    return struct.pack(f"<{len(words)}I", *words)


def gate_inputs(blob: bytes, threads: int) -> list[list[int]]:
    words = struct.unpack(f"<{threads * G * 8}I", blob[:threads * G * 8 * 4])
    return [list(words[t * G * 8:(t + 1) * G * 8]) for t in range(threads)]


def floor(units: dict[str, float]) -> dict:
    """The lower bound per element of each step set (module docstring, 2.)."""
    res = {}
    for name, ops in (("salt", SALT), ("free", FREE)):
        dispatch = DISPATCH * sum(ops.values())
        classes = {c: n * units[c] for c, n in ops.items() if units.get(c) is not None}
        res[name] = {"ops": ops, "instructions": sum(ops.values()), "dispatch_bound": dispatch,
                     "class_bound": {c: round(v, 3) for c, v in classes.items()},
                     "floor": round(max(dispatch, *classes.values()), 3), "issue_sum": round(sum(classes.values()), 3)}
    return res


def run(out: str) -> int:
    where = os.path.join(out, "rates")
    os.makedirs(where, exist_ok=True)
    cubin = build(where)
    if os.environ.get("NCP_LIBCUDA"):
        os.environ["PEARLC_LIBCUDA"] = os.environ["NCP_LIBCUDA"]
    d = pc.Dev(cubin)
    ident = {"name": d.name, "uuid": d.uuid, "cc": d.cc, "sms": d.sms, "driver": d.driver, "count": d.count,
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "cubin_sha256": hashlib.sha256(cubin).hexdigest()}
    emit({"part": "identity", **ident})
    if d.cc != (12, 0) or not d.name.startswith(pc.NAME_PREFIX) or d.count != 1:
        emit({"part": "refused", "why": "not exactly one RTX PRO 6000 Blackwell (sm_120) visible"})
        return 3
    d.open()
    nv = pc.Nvml(d.uuid)
    threads = d.sms * BLOCKS_PER_SM * THREADS_PER_BLOCK
    size = threads * G * 8 * 4
    blobs = {kind: (patterns(kind) * -(-threads // PATTERNS))[:size] for kind in ("salt", "free")}
    srcs, dst = {kind: d.put(b) for kind, b in blobs.items()}, d.alloc(size)
    src = lambda name: srcs["free" if name == "rate_free" else "salt"]
    reps = int(os.environ.get("REPS", "11"))
    result = {"identity": ident, "threads": threads, "repeats": list(REPEATS), "reps": reps, "kernels": {}}

    gate = {"threads": GATE, **{f"inputs_{kind}": gate_inputs(b, GATE) for kind, b in blobs.items()}}
    for name in MIXES:
        d.ck("cuMemsetD8_v2", pc.u64(dst), 0xA5, size)
        d.launch(name, 1, GATE, p=(src(name), dst), i=(1, GATE))
        d.sync()
        gate[name] = gate_inputs(d.read(dst, GATE * G * 8 * 4), GATE)
    with open(os.path.join(where, "gate.json"), "w") as f:
        json.dump(gate, f)

    blocks = d.sms * BLOCKS_PER_SM
    for name in (*MIXES, *CONTROLS):
        def launch(r, name=name):
            d.launch(name, blocks, THREADS_PER_BLOCK, p=(src(name), dst), i=(r, threads))
        before = nv.read()
        t = {r: d.timed(lambda r=r: launch(r), reps) for r in REPEATS}
        after = nv.read()
        mhz = [c["sm_mhz"] for c in (before, after) if "sm_mhz" in c]
        lo, hi = (statistics.median(t[r]) for r in REPEATS)
        per = MIXES[name] if name in MIXES else 8 * CONTROLS[name][1]
        row = {"ms": {str(r): [round(x, 5) for x in t[r]] for r in REPEATS}, "nvml_before": before, "nvml_after": after,
               "per_repeat_thread": per, "per": "element" if name in MIXES else "instruction"}
        if mhz and hi > lo:
            row["units"] = round((hi - lo) * 1e-3 * statistics.mean(mhz) * 1e6 * d.sms * MACS_PER_SM_CLOCK
                                 / ((REPEATS[1] - REPEATS[0]) * threads * per), 3)
        result["kernels"][name] = row
        emit({"part": "rate", "kernel": name, "units": row.get("units"), "per": row["per"], "sm_mhz": mhz})
    units = {CONTROLS[n][0]: result["kernels"][n].get("units") for n in CONTROLS}
    result["floor"] = floor(units)
    result["achieved"] = {n: result["kernels"][n].get("units") for n in MIXES}
    emit({"part": "floor", **{k: v["floor"] for k, v in result["floor"].items()}, "achieved": result["achieved"]})
    with open(os.path.join(where, "rates.json"), "w") as f:
        json.dump(result, f, indent=1)
    emit({"part": "done"})
    return 0


def check(out: str) -> int:
    """The gate against #295's fp16: every output word of rate_salt and rate_free at one repeat."""
    from verity.ml.tc import fp16
    with open(os.path.join(out, "rates", "gate.json")) as f:
        gate = json.load(f)
    lo, hi = (lambda w: w & 0xFFFF), (lambda w: w >> 16)
    halves = lambda f, *ws: f(*map(lo, ws)) | f(*map(hi, ws)) << 16
    enc2 = lambda a: fp16.to_e4m3(lo(a)) | fp16.to_e4m3(hi(a)) << 8
    enc4 = lambda a, b: enc2(a) | enc2(b) << 16
    bad = 0
    for ins, inf, salt, free in zip(gate["inputs_salt"], gate["inputs_free"], gate["rate_salt"], gate["rate_free"]):
        grp = [ins[8 * g:8 * g + 8] for g in range(G)]
        centre = lambda w: fp16.fma(0x3400, 0x3C00 | (w & 0x3FF), 0xB600)
        e = [[halves(lambda a, w: fp16.mul(a, centre(w)), gr[2 + j], gr[4 + j]) for j in range(2)] for gr in grp]
        for g, gr in enumerate(grp):
            h, x = gr[0:2], e[(g + 1) % G]
            s = [halves(fp16.add, e[g][j], x[j]) for j in range(2)]
            b0 = enc4(*(halves(fp16.add, h[j], e[g][j]) for j in range(2)))
            b1 = enc4(*(halves(fp16.add, h[j], x[j]) for j in range(2)))
            b2 = enc4(*(halves(lambda a, b: fp16.add(fp16.neg(a), fp16.neg(b)), h[j], s[j]) for j in range(2)))
            want = [b0, b1, b2, b0 ^ b2, b1]
            bad += sum(a != b for a, b in zip(salt[8 * g:8 * g + 5], want))
        grp = [inf[8 * g:8 * g + 8] for g in range(G)]
        fl = grp[0][7]
        hs = [[halves(fp16.from_e4m3, gr[0] & 0xFF | (gr[0] >> 8 & 0xFF) << 16), halves(fp16.from_e4m3,
               gr[0] >> 16 & 0xFF | (gr[0] >> 24) << 16)] for gr in grp]
        for g, gr in enumerate(grp):
            p = hs[(g + 1) % G]
            amp = [halves(lambda a, b, c: fp16.max_(fp16.max_(a & 0x7FFF, b & 0x7FFF), c), hs[g][j], p[j], fl)
                   for j in range(2)]
            q = [halves(lambda z: fp16.fma(z, z, 0), halves(lambda a: fp16.mul(a, 0x1C00), hs[g][j])) for j in range(2)]
            want = [*amp, *q, gr[0] ^ amp[0] ^ amp[1]]
            bad += sum(a != b for a, b in zip(free[8 * g:8 * g + 5], want))
    words = len(gate["inputs_salt"]) * G * 10
    print(json.dumps({"gate": "all match" if not bad else "MISMATCH", "words": words, "differing": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        sys.exit(check(sys.argv[2]))
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out")))
