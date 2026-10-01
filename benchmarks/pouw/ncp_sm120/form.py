#!/usr/bin/env python3
"""D-3s's forming (`d3s-v0-f16`) on one RTX PRO 6000 Blackwell (sm_120), and the bound NCP unit it feeds (form_sm120.cu, run.py).

0. Build: form_sm120.cu and run.py's GEMM (ncp_sm120.cu) with the node's nvcc into OUT, each refused on a spill, a stack
   frame or local memory (run.py's SASS gate for the GEMM).
1. The check (OUT/form/check/): at k = 1,024 and 8,192, eight A rows and eight B rows (rows 1-4 all zero, all 448, all the
   least subnormal, +-448 alternating; the rest gen_any's full finite range) formed under each XOF, whole, and the first
   four rows' raw streams.  check_form.py holds them to #295's `D3Family.side` and `xof` (SHAKE256), or BLAKE3's XOF.
2. The unit at each SHAPES entry (p: m = n = k = 8,192; d: m = 32): X_q (m x k) and W (n x k) from gen_any (seeds 3, 4),
   formed by each XOF of XOFS into run.py's X~ and Y~.  Arms, each REPS pairs interleaved with the plain GEMM at K = k (the
   baseline), each a dependent chain of CHAIN_P / CHAIN_D: the stages (SHAKE's a_stream, b_stream and streams, both
   sides' in one launch, its own kernel; BLAKE3's streams are inside the rows kernel; a_rows, b_rows), form (the streams,
   then A's rows and B's), and unit (form, then the GEMM arm of each UNIT_ARMS entry arm:S).  After each unit arm, one more call after C and the digests are poisoned, and its
   sampled lanes go to OUT/form/arms.json in check.py's format (a shape entry per XOF, its `xof` named), then a
   negative control (poisoned, nothing launched).

The messages are #295's (`D3Family.stream`): SHAKE256 / BLAKE3 of "verity/pouw/fp8-is/d3s-v0-f16/E1" (A's row i) or ".../F1"
(B's row j), NUL, then the frames of (salt, index, root_A, weight, line), at the fixed context CONTEXT.  The host absorbs
them (form_sm120.cu); NCP's permutation is #295's `ncp_perm(k, b"verity/pouw/fp8-is/d3s/P")`, restated here (stdlib).

  python3 form.py OUT   (env: CUDA, SHAPES="p d", XOFS="shake b3", UNIT_ARMS="chash:4 chash:8 hash_ir:1", REPS=5,
                         CHAIN_P=1, CHAIN_D=1, NCP_LIBCUDA)
"""

from __future__ import annotations

import ctypes as C
import functools
import hashlib
import importlib.util
import json
import os
import re
import statistics
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "form_sm120.cu")
FORM_THREADS = 256
TAG = "d3s-v0-f16"
P_SEED = b"verity/pouw/fp8-is/d3s/P"
CONTEXT = {"salt": hashlib.sha256(b"ncp_sm120/form salt").digest(), "index": 7,
           "root_a": hashlib.sha256(b"ncp_sm120/form root_a").digest(), "weight": 3}
SEED_A, SEED_B = 3, 4
CHECK_K = (1024, 8192)
CHECK_ROWS = 8
SPECIAL = {1: b"\x00", 2: b"\x7e", 3: b"\x01", 4: b"\x7e\xfe"}

_spec = importlib.util.spec_from_file_location("ncp_sm120_run", os.path.join(HERE, "run.py"))
ncp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ncp)
pc = ncp.pc
_spec = importlib.util.spec_from_file_location(
    "verity_blake3", os.path.join(HERE, "..", "..", "..", "packages", "verity", "src", "verity", "commitments", "blake3.py"))
b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b3)


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def build(out: str) -> bytes:
    cuda = os.environ.get("CUDA", "/usr/local/cuda-13.0")
    nvcc = os.path.join(cuda, "bin", "nvcc")
    cubin = os.path.join(out, "form_sm120.cubin")
    r = subprocess.run([nvcc, *ncp.FLAGS, "-o", cubin, KERNEL], capture_output=True, text=True)
    log = r.stdout + r.stderr
    with open(os.path.join(out, "form_ptxas.txt"), "w") as f:
        f.write(log)
    if r.returncode:
        raise SystemExit(f"nvcc failed ({r.returncode}): see form_ptxas.txt")
    bad = [ln.strip() for ln in log.splitlines() if re.search(r"[1-9]\d* bytes (stack frame|spill)", ln)]
    if bad:
        raise SystemExit(f"ptxas spilled: {bad[:4]}")
    env = dict(os.environ, PATH=os.path.join(cuda, "bin") + os.pathsep + os.environ.get("PATH", ""))
    sass = subprocess.run([os.path.join(cuda, "bin", "cuobjdump"), "-sass", cubin], capture_output=True, text=True, env=env)
    if sass.returncode or re.search(r"\b(LDL|STL)\b", sass.stdout):
        raise SystemExit("form_sm120: cuobjdump failed or the SASS uses local memory")
    with open(cubin, "rb") as f:
        return f.read()


class Dev(pc.Dev):
    """pearl_c_sm120's device with a second module: the forming kernels beside run.py's GEMM."""

    def __init__(self, cubin: bytes, form_cubin: bytes):
        super().__init__(cubin)
        self.form_cubin, self.fmod = form_cubin, None

    def open(self) -> None:
        super().open()
        self.fmod = pc.vp()
        self.ck("cuModuleLoadData", C.byref(self.fmod), self.form_cubin)

    def fn(self, name: str) -> pc.vp:
        if name.startswith(("form_", "xof_", "gen_any")) and name not in self.fns:
            f = pc.vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.fmod, name.encode())
            self.fns[name] = f
        return super().fn(name)


# ---- the protocol's host values (#295's, restated with the stdlib; check_form.py holds them to #295) --------------------

def frame(part) -> bytes:
    """verity.randomness.frame."""
    if isinstance(part, bytes):
        tag, body = b"b", part
    elif isinstance(part, str):
        tag, body = b"s", part.encode()
    else:
        tag, body = b"i", part.to_bytes(max(1, (part.bit_length() + 7) // 8), "big")
    return tag + len(body).to_bytes(8, "big") + body


def message(name: str, line: int) -> bytes:
    """The stream's message: the xof's tag, NUL, the frames (#295 `xof` as `D3Family.stream` calls it)."""
    c = CONTEXT
    parts = (c["salt"], c["index"], c["root_a"], c["weight"], line)
    return f"verity/pouw/fp8-is/{TAG}/{name}".encode() + b"\0" + b"".join(frame(p) for p in parts)


def shake_blocks(m: bytes) -> bytes:
    """SHAKE256's padded absorption: the message, 0x1F, zeros to the 136-byte rate, the last byte xor 0x80."""
    pad = bytearray(m + b"\x1f" + b"\0" * (-(len(m) + 1) % 136))
    pad[-1] ^= 0x80
    return bytes(pad)


@functools.lru_cache(maxsize=4)
def _b3_prefix(prefix: bytes) -> tuple[int, ...]:
    """The chaining value after a chunk's whole blocks (every row's message shares them but the last)."""
    cv = b3.IV
    for i in range(0, len(prefix), 64):
        cv = tuple(b3.compress(cv, prefix[i:i + 64], 0, 64, b3.CHUNK_START if i == 0 else 0)[:8])
    return cv


def b3_root(m: bytes) -> bytes:
    """BLAKE3's root node of a one-chunk message as 32 words: the chaining value before the last block, the block, its
    length, its flags (ROOT included); the XOF's output block i is this node compressed at counter i."""
    if len(m) > b3.CHUNK_LEN:
        raise ValueError("one chunk")
    blocks = [m[i:i + 64] for i in range(0, len(m), 64)] or [b""]
    cv = _b3_prefix(b"".join(blocks[:-1]))
    last = blocks[-1]
    flags = b3.CHUNK_END | (b3.CHUNK_START if len(blocks) == 1 else 0) | b3.ROOT
    return struct.pack("<8I16I2I6I", *cv, *struct.unpack("<16I", last.ljust(64, b"\0")), len(last), flags, *[0] * 6)


def word_perm32(k: int, seed: bytes, g: int = 4) -> list[int]:
    """#295's `word_perm32`: π as aligned g-entry words (π(g·a + b) = g·p[a] + b), returned as p."""
    w, wps = k // g, 32 // g

    def stream(tag: bytes, count: int):
        buf = hashlib.shake_256(seed + tag).digest(4 * count)
        return iter(int.from_bytes(buf[4 * i:4 * i + 4], "little") for i in range(count))

    def uniform(ws, bound: int) -> int:
        limit = (1 << 32) - ((1 << 32) % bound)
        while True:
            r = next(ws)
            if r < limit:
                return r % bound

    ws, p = stream(b"", 16 * w), list(range(w))
    for i in range(w - 1, 0, -1):
        j = uniform(ws, i + 1)
        p[i], p[j] = p[j], p[i]
    ws = stream(b"/repair", 64 * w)
    for j in range(w):
        while p[j] // wps == j // wps:
            r = uniform(ws, w)
            if p[r] // wps != j // wps and p[j] // wps != r // wps:
                p[j], p[r] = p[r], p[j]
    return p


def perm_words(k: int) -> tuple[bytes, bytes]:
    """π's and π⁻¹'s words (u16 each), refused unless π is admissible at d = 32 (#295 `admissible32`)."""
    p = word_perm32(k, P_SEED)
    pi = [4 * p[a] + b for a in range(k // 4) for b in range(4)]
    for t in range(k // 32):
        img = {pi[j] // 32 for j in range(32 * t, 32 * t + 32)}
        if t in img or len(img) == 1:
            raise SystemExit(f"π fails the d = 32 rule at k = {k}")
    if all(pi[pi[j]] == j for j in range(k)):
        raise SystemExit(f"π is an involution at k = {k}")
    inv = [0] * len(p)
    for a, b in enumerate(p):
        inv[b] = a
    return struct.pack(f"<{len(p)}H", *p), struct.pack(f"<{len(inv)}H", *inv)


# ---- one side's forming on the device ------------------------------------------------------------------------------------

class Side:
    """One side's rows (A: X~'s rows from X_q's, B: Y~'s from W's) formed into `out` (rows `pitch` bytes apart)."""

    def __init__(self, d: Dev, rows: int, k: int, left: bool, xof: str, codes: int, out: int, pitch: int,
                 at: tuple[int, int] | None = None):
        """`at`: SHAKE's message lanes and c buffers, when the side's rows sit in a joint pair's (`Unit`)."""
        self.d, self.rows, self.k, self.left, self.xof = d, rows, k, left, xof
        self.codes, self.out, self.pitch = codes, out, pitch
        msgs = [message("E1" if left else "F1", line) for line in range(rows)]
        if xof == "shake":
            blob = [shake_blocks(m) for m in msgs]
            self.nb = len(blob[0]) // 136
            if any(len(b) != 136 * self.nb for b in blob):
                raise SystemExit("rows' messages absorb in different numbers of blocks")
            blob = b"".join(blob)
            if at:
                self.lanes, self.c = at
                d.ck("cuMemcpyHtoD_v2", pc.u64(self.lanes), C.c_char_p(blob), len(blob))
            else:
                self.lanes, self.c = d.put(blob), d.alloc(rows * k * 2)
            self.roots = 0
        else:
            self.nb, self.lanes, self.c = 0, 0, 0
            self.roots = d.put(b"".join(b3_root(m) for m in msgs))
        pw, pinv = perm_words(k)
        self.pw, self.pinv = d.put(pw), d.put(pinv)
        self.smem = 3 * k + 256
        self.kernel = "form_rows_g" if xof == "shake" else "form_rows_b3"
        d.ck("cuFuncSetAttribute", d.fn(self.kernel), pc.MAX_DYN, self.smem)

    def stream(self) -> None:
        if self.xof == "shake":
            self.d.launch("form_c_shake", -(-self.rows // 32), 32, 0, p=(self.lanes, self.c), i=(self.rows, self.k, self.nb))

    def form_rows(self) -> None:
        self.d.launch(self.kernel, self.rows, FORM_THREADS, self.smem,
                      p=(self.codes, self.c, self.roots, self.pw, self.pinv, self.out),
                      i=(self.rows, self.k, int(self.left), self.pitch))

    def run(self) -> None:
        self.stream()
        self.form_rows()

    def raw(self, rows: int, n: int) -> bytes:
        out = self.d.alloc(rows * n)
        self.d.launch("xof_raw", 1, rows, 0, p=(self.lanes, self.roots, out), i=(rows, n, self.nb, int(self.xof == "b3")))
        self.d.sync()
        got = self.d.read(out, rows * n)
        self.d.free(out)
        return got


class Unit:
    """Both sides of a unit: X_q's rows into X~ and W's into Y~.  Under SHAKE the two sides' streams are one launch (one
    thread a row over A's rows then B's), since each row's squeeze is sequential and its latency, not the rows' count,
    sets the stream's time."""

    def __init__(self, d: Dev, x, xof: str, ca: int, cb: int):
        m, n, k = x.rows, x.n, x.k
        self.d, self.rows, self.k, self.xof = d, m + n, k, xof
        self.nb, self.lanes, self.c, at_a, at_b = 0, 0, 0, None, None
        if xof == "shake":
            self.nb = len(shake_blocks(message("E1", 0))) // 136
            self.lanes, self.c = d.alloc((m + n) * self.nb * 136), d.alloc((m + n) * k * 2)
            at_a, at_b = (self.lanes, self.c), (self.lanes + m * self.nb * 136, self.c + m * k * 2)
        self.A = Side(d, m, k, True, xof, ca, x.x, x.K, at_a)
        self.B = Side(d, n, k, False, xof, cb, x.y, x.K, at_b)

    def streams(self) -> None:
        if self.xof == "shake":
            self.d.launch("form_c_shake", -(-self.rows // 32), 32, 0, p=(self.lanes, self.c), i=(self.rows, self.k, self.nb))

    def run(self) -> None:
        self.streams()
        self.A.form_rows()
        self.B.form_rows()


def gen_any(d: Dev, ptr: int, count: int, seed: int) -> None:
    d.launch("gen_any", 4 * d.sms, 256, 0, p=(ptr,), i=(count & 0xFFFFFFFF, seed, count >> 32))


def check(d: Dev, out: str) -> None:
    where = os.path.join(out, "form", "check")
    os.makedirs(where, exist_ok=True)
    meta = {"context": {k: v.hex() if isinstance(v, bytes) else v for k, v in CONTEXT.items()}, "tag": TAG,
            "rows": CHECK_ROWS, "special": {str(r): v.hex() for r, v in SPECIAL.items()}, "runs": []}
    for k in CHECK_K:
        for left, seed in ((True, 5), (False, 6)):
            codes = d.alloc(CHECK_ROWS * k)
            gen_any(d, codes, CHECK_ROWS * k, seed)
            d.sync()
            for r, pat in SPECIAL.items():
                row = (pat * k)[:k]
                d.ck("cuMemcpyHtoD_v2", pc.u64(codes + r * k), C.c_char_p(row), k)
            side_codes = d.read(codes, CHECK_ROWS * k)
            for xof in ("shake", "b3"):
                formed = d.alloc(CHECK_ROWS * 3 * k)
                s = Side(d, CHECK_ROWS, k, left, xof, codes, formed, 3 * k)
                s.run()
                d.sync()
                tag = f"{xof}_{'A' if left else 'B'}_k{k}"
                with open(os.path.join(where, tag + ".codes.bin"), "wb") as f:
                    f.write(side_codes)
                with open(os.path.join(where, tag + ".formed.bin"), "wb") as f:
                    f.write(d.read(formed, CHECK_ROWS * 3 * k))
                with open(os.path.join(where, tag + ".xof.bin"), "wb") as f:
                    f.write(s.raw(4, 2 * k))
                meta["runs"].append({"tag": tag, "xof": xof, "left": left, "k": k})
                d.free(formed)
            d.free(codes)
    with open(os.path.join(where, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    emit({"part": "form_check", "runs": len(meta["runs"])})


def unit_arms() -> list[tuple[str, int]]:
    return [(a, int(s)) for a, s in (e.split(":") for e in os.environ.get("UNIT_ARMS", "chash:4 chash:8 hash_ir:1").split())]


def shape(d: Dev, nv, kind: str, xofs: list[str], reps: int) -> list[dict]:
    m, n, k = ncp.SHAPES[kind]
    x = ncp.Ncp(d, kind, m, n, k)
    ca, cb = d.alloc(m * k), d.alloc(n * k)
    gen_any(d, ca, m * k, SEED_A)
    gen_any(d, cb, n * k, SEED_B)
    d.sync()
    tiles = sorted({0, x.tiles // 2 + 1, x.tiles - 1})
    chain = int(os.environ.get(f"CHAIN_{kind.upper()}", "1"))
    base = lambda: x.launch("plain", 0, k, 0)
    out = []
    for xof in xofs:
        U = Unit(d, x, xof, ca, cb)
        A, B = U.A, U.B
        res = {"kind": kind, "xof": xof, "m": x.m, "rows": m, "n": n, "k": k, "K": x.K, "T": x.K // 32, "tm": x.tm,
               "bn": x.bn, "nct": x.nct, "tiles": x.tiles, "grid": x.grid, "seeds": [SEED_A, SEED_B], "formed": True,
               "context": {key: v.hex() if isinstance(v, bytes) else v for key, v in CONTEXT.items()},
               "layout": ncp.layout(kind), "arms": [], "timing": []}
        stages = [("a_rows", A.form_rows), ("b_rows", B.form_rows), ("form", U.run)]
        if xof == "shake":
            stages = [("a_stream", A.stream), ("b_stream", B.stream), ("streams", U.streams)] + stages
        for name, body in stages:
            before = nv.read()
            got = ncp.interleaved(d, nv, body, base, reps, chain)
            row = {"arm": name, "ms": statistics.median(got["reps"]), "base_ms": statistics.median(got["base_reps"]),
                   "chain": chain, "nvml_before": before, "nvml_after": nv.read(), **got}
            res["timing"].append(row)
            emit({"part": "form", "kind": kind, "xof": xof, "arm": name, "ms": round(row["ms"], 4),
                  "base_ms": round(row["base_ms"], 4)})
        for arm, S in unit_arms():
            body = lambda arm=arm, S=S: (U.run(), x.launch(arm, S, x.K, 0))
            before = nv.read()
            got = ncp.interleaved(d, nv, body, base, reps, chain)
            row = {"arm": f"unit_{arm}", "gemm": arm, "S": S, "ms": statistics.median(got["reps"]),
                   "base_ms": statistics.median(got["base_reps"]), "chain": chain, "nvml_before": before,
                   "nvml_after": nv.read(), **got}
            res["timing"].append(row)
            x.poison()
            body()
            d.sync()
            res["arms"].append({"arm": arm, "S": S, "K": x.K, "stagger": 0, **x.dump(tiles, ncp.hashed(arm))})
            emit({"part": "unit", "kind": kind, "xof": xof, "arm": arm, "S": S, "ms": round(row["ms"], 4),
                  "base_ms": round(row["base_ms"], 4)})
        x.poison()
        d.sync()
        res["arms"].append({"arm": "control", "as": "chash", "S": 4, "K": x.K, "stagger": 0, **x.dump(tiles, True)})
        out.append(res)
    return out


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out")
    os.makedirs(os.path.join(out, "form"), exist_ok=True)
    cubin, form_cubin = ncp.build(out), build(out)
    if os.environ.get("NCP_LIBCUDA"):
        os.environ["PEARLC_LIBCUDA"] = os.environ["NCP_LIBCUDA"]
    d = Dev(cubin, form_cubin)
    ident = {"name": d.name, "uuid": d.uuid, "cc": d.cc, "sms": d.sms, "driver": d.driver, "count": d.count,
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "form_cubin_sha256": hashlib.sha256(form_cubin).hexdigest()}
    emit({"part": "identity", **ident})
    if d.cc != (12, 0) or not d.name.startswith(pc.NAME_PREFIX) or d.count != 1:
        emit({"part": "refused", "why": "not exactly one RTX PRO 6000 Blackwell (sm_120) visible"})
        return 3
    d.open()
    nv = pc.Nvml(d.uuid)
    ident["nvml_driver"] = nv.driver
    check(d, out)
    reps = int(os.environ.get("REPS", "5"))
    xofs = os.environ.get("XOFS", "shake b3").split()
    result = {"identity": ident, "shapes": []}
    for kind in os.environ.get("SHAPES", "p d").split():
        result["shapes"] += shape(d, nv, kind, xofs, reps)
        with open(os.path.join(out, "form", "arms.json"), "w") as f:
            json.dump(result, f, indent=1)
    emit({"part": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
