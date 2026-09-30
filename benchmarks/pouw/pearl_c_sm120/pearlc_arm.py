"""Pearl-C's FP8 arm for GPU 6's bench harness (arm interface v0, `benchmarks/pouw/harness/arm.py`): pearl_c_sm120.cu's
kernels as the honest prover on the harness's FP32 operands, on the device record `pearl_c_device.SM120` (G = 4, the panel's
`pearl-c-sm120` v1) or `SM120_UNPROMOTED` (v2, counted only once the theory and Lean lanes restate Pearl-C without promotion).

Per operand set, `weight_side` is the per-(weight, epoch) work: E_B's, F_A's and F_B's lines, B's statistics, E′ and forming,
B̃·F_A, G and P_B.  `launch` is one call: at m not a multiple of 128, A's rows into a zero-padded buffer; E_A's lines; A's
statistics, E′ and forming; A′·F_B and P_A; the chain with its peel and y; then every hash (the C̃ and U messages, the tile
leaves, Pearl's tickets, A's row commitments).  `nohash` is the same call without the hashes and without E_A's XOF (its lines
come from `prepare`), so it writes the same C̃, U and y.  A shape's operand sets and flag sets share the per-call buffers (the
harness runs one call at a time on one stream); each operand set has its own keys, E_A's lines and weight side.

The gate, at the exact shape and flag set: a second launch into the gate's own buffers (C̃ by gemm_ct, another kernel; C̃, U
and y by gemm_pearlc on 3 CTAs, another tile schedule; the hashes over those) must equal every output word for word (`all`),
and at the twin's sampled 64 x 64 tiles (`twin.samples`) the call's A and B codes and α, C̃, U, y and, with hashing, the leaf,
the tickets and row r0's commitments must equal #449's CPU twin (`benchmarks/pouw/pearl_c/twin.py`) on the device record,
from the harness's operand rows.  `known_bad` flips C̃'s first word between the product and the hashes.

Shortcuts, flagged: the hashing is #449's device code, unfused (the digests read C̃ and U back from memory); m is padded to
128 rows (decode's m = 32 issues 4x its tensor work, and `work` counts issued MACs); the prices and G are the H100's
stand-ins, so W_ref, the credit and γ are Derived at stand-in prices.

  python bench.py --arm-path <this directory> --arm pearlc_arm:PearlCSm120 [--arm pearlc_arm:PearlCSm120Unpromoted] ...

PEARLC_SHIP names build.sh's output (pearl_c_sm120.cubin, sass.json, toolchain.txt, source.txt; default this directory).
Needs Python >= 3.12 with numpy, and `verity` and `verity_pouw` importable (the twin).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from arm import Buffer, Gate, Work

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = _load("pearl_c_sm120_run", HERE / "run.py")
F = _load("pearl_c_sm120_fixture", HERE / "fixture.py")
W, H = F.W, F.H

from verity_pouw.protocol import Shape as MShape  # noqa: E402
from verity_pouw.schemes import pearl_c as C  # noqa: E402
from verity_pouw.schemes.pearl_c_device import DEVICES  # noqa: E402

T, SUBTILE, SEGMENT = W.T, 16, W.SEGMENT
WEIGHT = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b", "epb_codes",
          "bt", "bf", "pb_t")
OWN = WEIGHT + ("ea_codes",)       # E_A's lines are the set's too (its line_a key), and nohash reads them from prepare
CALL = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "at", "af", "pa_t", "c", "u", "y", "digests", "leaves", "tickets",
        "rows")
REF = ("c2", "c", "u", "y", "digests", "leaves", "tickets", "rows")
KEYS = ("line_a", "line_b", "jackpot")
REF_GRID = 3
GAMMA0 = RHO_CAP = Fraction(1, 400)       # TT_OUT's γ0 and the debit cap ρ (statement parameters, cheap-binding.md §1)


def padded(m: int) -> int:
    return -(-m // R.BM) * R.BM


def seeds(arm: str, shape, seed: int) -> dict[str, bytes]:
    """The call's noise seeds, fixed by the arm, the shape and the harness's operand-set seed."""
    h = lambda role: hashlib.sha256(  # noqa: E731
        f"verity-pouw/pearl-c-sm120-arm/{arm}/{shape.label}/{shape.m}x{shape.k}x{shape.n}/{seed}/{role}".encode()).digest()
    return {"seed_a": h("seed-A"), "seed_b": h("seed-B")}


def gamma(scheme: C.PearlC, shape) -> dict[str, str]:
    """γ at `shape` from the scheme's W_ref and credit (`bench.gamma`'s four readings on the H100): `cap`, the worst case under
    the debit cap; `floor`, a zero debit; `*_chain_only`, TT_OUT crediting no forming."""
    s = MShape(shape.m, shape.k, shape.n)
    wref, credit = Fraction(scheme.wref(s)), Fraction(scheme.credit_of(s))
    forming = scheme.device.prices.forming_credited * s.m * s.k
    out = {"cap": 1 - (1 - GAMMA0) * (1 - RHO_CAP) * credit / wref,
           "cap_chain_only": 1 - (1 - GAMMA0) * (credit - forming - RHO_CAP * credit) / wref,
           "floor": 1 - (1 - GAMMA0) * credit / wref,
           "floor_chain_only": 1 - (1 - GAMMA0) * (credit - forming) / wref}
    return {**{k: str(v) for k, v in out.items()},
            "basis": f"Derived: TT_OUT γ0 = ρ = 1/400 on {scheme.device.name} at stand-in prices (the H100's), G = {scheme.device.g}"}


def expected(t, hashing: bool) -> dict[str, bytes]:
    """A twin tile's bytes in the kernels' layout, by the names `_Call.tile_bytes` reads."""
    ia, ib = H.recip(t.a.alpha.astype(np.uint32)), H.recip(t.b.alpha.astype(np.uint32))
    out = {"at": t.a.codes.astype(np.uint8).tobytes(), "alpha_a": t.a.alpha.astype("<u4").tobytes(),
           "bt": t.b.codes.astype(np.uint8).tobytes(), "alpha_b": t.b.alpha.astype("<u4").tobytes(),
           "c": t.c.astype("<u4").tobytes(), "u": t.u.astype("<u4").tobytes(),
           "y": H.glue_y(t.u.astype(np.uint32), ia, ib).astype("<u2").tobytes()}
    if hashing:
        out.update(leaf=t.leaf, tickets=b"".join(t.tickets), rows=b"".join(t.row_commits))
    return out


_DEV = None


def device(cubin: Path):
    """The kernels' module in the device's primary context, loaded once per process (the harness's context too)."""
    global _DEV
    if _DEV is None:
        _DEV = R.Dev(cubin.read_bytes())
        _DEV.open()
    return _DEV


class _Shared:
    """Device buffers that outlive one call, freed with the last call that uses them."""

    def __init__(self, rd, sizes: dict[str, int]):
        self.rd, self.b, self.users = rd, {name: rd.alloc(n) for name, n in sizes.items()}, 0
        self.sd: dict[str, bytes] = {}
        self.job = None
        self.twin: dict = {}

    def free(self) -> None:
        for p in self.b.values():
            self.rd.free(p)


class _Call:
    def __init__(self, arm: PearlCSm120, hdev, shape, flags: frozenset[str], ops, corrupt: bool):
        rd, self.arm, self.hdev, self.shape, self.ops, self.freed = arm.rd, arm, hdev, shape, ops, False
        rd.sync()                          # the operands were filled on the harness's non-blocking stream; this runs on the null one
        self.hashing, self.xof = "hash" in flags, "xof" in flags
        m, n, k = self.m, self.n, self.k = padded(shape.m), shape.n, shape.k
        size = R.Pipeline.sizes(m, n, k)
        self.scratch, _ = arm.acquire(arm.scratch, shape, lambda: {**{b: size[b] for b in CALL},
                                                                  **{"ref_" + b: size[b] for b in REF},
                                                                  **({"xa": 4 * m * k} if m != shape.m else {})})
        self.set_key = (shape, ops.a.ptr, ops.b.ptr, ops.seed)
        self.own, fresh = arm.acquire(arm.sets, self.set_key, lambda: {**{b: size[b] for b in OWN}, **{"key_" + x: 32 for x in KEYS}})
        s, w = self.scratch.b, self.own.b
        if fresh:
            self.own.sd = arm.seeds(shape, ops.seed)
            for name, hexkey in W.keys(self.own.sd).items():
                rd.ck("cuMemcpyHtoD_v2", R.u64(w["key_" + name]), R.C.c_char_p(bytes.fromhex(hexkey)), 32)
        self.keys = {x: w["key_" + x] for x in KEYS}
        self.xa = s.get("xa", ops.a.ptr)
        self.size = {**size, "rows": shape.m * (4 * k // SEGMENT) * 32}
        self.pipe = R.Pipeline(rd, m, n, k, arm.variant, self.keys, self.xa, ops.b.ptr,
                               bufs={**{b: s[b] for b in CALL}, **{b: w[b] for b in OWN}}, m_real=shape.m)
        steps = self.pipe.steps()
        self.job = [f for _, phase, f in steps if phase == "job"]
        lines_ea = [f for name, _, f in steps if name == "lines_ea"]
        forming = [f for name, phase, f in steps if phase == "forming" and name != "lines_ea"]
        cleanup = [f for _, phase, f in steps if phase == "cleanup"]
        flip = [lambda: hdev.xor_byte(s["c"], 0, 0x01, rd.stream)] if corrupt else []
        hashes = [f for _, phase, f in steps if phase == "hashing"]
        self.steps = (lines_ea if self.xof else []) + forming + cleanup + flip + (hashes if self.hashing else [])
        names = (("leaves", "tickets", "rows") if self.hashing else ()) + ("y", "c", "u")
        self.outputs = {name: Buffer(s[name], self.size[name]) for name in names}
        if fresh:
            for f in self.job:
                f()
        for f in lines_ea:
            f()
        self.launch(None)
        rd.sync()

    def launch(self, stream) -> None:
        rd = self.arm.rd
        rd.stream = stream
        try:
            if self.xa != self.ops.a.ptr:
                rd.copy_async(self.xa, self.ops.a.ptr, 4 * self.shape.m * self.k)
            for f in self.steps:
                f()
        finally:
            rd.stream = None

    def weight_side(self, stream) -> None:
        rd = self.arm.rd
        rd.stream = stream
        try:
            for f in self.job:
                f()
        finally:
            rd.stream = None

    def free(self) -> None:
        if not self.freed:
            self.freed = True
            self.arm.release(self.arm.sets, self.set_key)
            self.arm.release(self.arm.scratch, self.shape)

    # ---- the gate's parts ------------------------------------------------------------------------------------------------

    def relaunch(self) -> dict[str, int]:
        """Words differing between each output and the second launch's (gemm_ct's C̃ as `c@gemm_ct`)."""
        rd, s = self.arm.rd, self.scratch.b
        ref = R.Pipeline(rd, self.m, self.n, self.k, self.arm.variant, self.keys, self.xa, self.ops.b.ptr,
                         bufs={**self.pipe.b, **{b: s["ref_" + b] for b in REF}}, m_real=self.shape.m)
        real = ref.gemm
        ref.gemm = lambda *a, **kw: real(*a, **{**kw, "grid": REF_GRID})
        steps = {name: f for name, _, f in ref.steps()}
        for name in ("gemm_ct", "gemm_pearlc") + (("hash_msg", "hash_leaf", "tickets", "hash_rows") if self.hashing else ()):
            steps[name]()
        rd.sync()
        out = {name: self.diff(buf.ptr, s["ref_" + name], buf.nbytes) for name, buf in self.outputs.items()}
        out["c@gemm_ct"] = self.diff(s["c"], s["ref_c2"], self.outputs["c"].nbytes)
        return out

    def diff(self, x: int, y: int, nbytes: int) -> int:
        a, b = (np.frombuffer(self.arm.rd.read(p, nbytes), "<u4") for p in (x, y))
        return int(np.count_nonzero(a != b))

    def rows(self, ptr: int, r0: int, real: int) -> np.ndarray:
        """Rows r0 .. r0 + 63 of an FP32 (real x k) operand, zero past `real` (the filler)."""
        take, out = max(0, min(T, real - r0)), np.zeros((T, self.k), np.uint32)
        if take:
            out[:take] = np.frombuffer(self.arm.rd.read(ptr + 4 * self.k * r0, 4 * self.k * take), "<u4").reshape(take, self.k)
        return out

    def tile_bytes(self, r0: int, c0: int) -> dict[str, bytes]:
        n, k, s, w = self.n, self.k, self.scratch.b, self.own.b
        get = self.arm.rd.read
        rows = range(r0, r0 + T)
        out = {"at": get(s["at"] + r0 * k, T * k), "alpha_a": get(s["alpha_a"] + 4 * r0, 4 * T),
               "bt": get(w["bt"] + c0 * k, T * k), "alpha_b": get(w["alpha_b"] + 4 * c0, 4 * T),
               "c": b"".join(get(s["c"] + 4 * (i * n + c0), 4 * T) for i in rows),
               "u": b"".join(get(s["u"] + 4 * (i * n + c0), 4 * T) for i in rows),
               "y": b"".join(get(s["y"] + 2 * (i * n + c0), 2 * T) for i in rows)}
        if self.hashing:
            segs = 4 * k // SEGMENT
            out["leaf"] = get(s["leaves"] + 32 * ((r0 // T) * (n // T) + c0 // T), 32)
            out["tickets"] = b"".join(get(s["tickets"] + 32 * ((r0 // SUBTILE + j) * (n // SUBTILE) + c0 // SUBTILE), 32 * T // SUBTILE)
                                      for j in range(T // SUBTILE))
            out["rows"] = get(s["rows"] + 32 * segs * r0, 32 * segs)
        return out

    def twin_tiles(self) -> list[dict]:
        sh, wt = self.shape, self.own
        tshape = W.Shape(sh.label, sh.m, sh.n, sh.k)
        if wt.job is None:
            wt.job = W.job(tshape, self.arm.device, wt.sd)
        out = []
        for r0, c0 in W.samples(tshape):
            a_rows, b_rows = self.rows(self.ops.a.ptr, r0, sh.m), self.rows(self.ops.b.ptr, c0, sh.n)
            key = (r0, c0, hashlib.sha256(a_rows.tobytes() + b_rows.tobytes()).digest())
            if key not in wt.twin:
                wt.twin[key] = W.tile(wt.job, r0, c0, a_rows, b_rows)
            want, got = expected(wt.twin[key], self.hashing), self.tile_bytes(r0, c0)
            out.append({"tile": [r0, c0], "failed": sorted(name for name in want if got[name] != want[name])})
        return out


class PearlCSm120:
    name = "pearl-c-sm120-v1"
    stream = "fp8"
    precision = "fp8-e4m3"
    inputs = "f32"
    flag_sets = {"hash": frozenset({"hash", "xof"}), "nohash": frozenset()}
    variant = "sm120"
    panel = {"line": "pearl-c-sm120", "version": "v1"}

    def __init__(self):
        self.device = DEVICES[self.variant]
        self.scheme = C.PearlC(forming="v1", device=self.device)
        self.rd = None
        self.scratch: dict = {}
        self.sets: dict = {}

    def seeds(self, shape, seed: int) -> dict[str, bytes]:
        return seeds(self.name, shape, seed)

    def acquire(self, table: dict, key, sizes) -> tuple[_Shared, bool]:
        fresh = key not in table
        if fresh:
            table[key] = _Shared(self.rd, sizes())
        table[key].users += 1
        return table[key], fresh

    def release(self, table: dict, key) -> None:
        sh = table[key]
        sh.users -= 1
        if not sh.users:
            sh.free()
            del table[key]

    def build(self, ctx: dict) -> dict:
        ship = Path(os.environ.get("PEARLC_SHIP", HERE))
        cubin = ship / "pearl_c_sm120.cubin"
        self.rd = device(cubin)
        d = self.device
        rec = {"arm": self.name, "scheme": self.scheme.name, "panel": self.panel,
               "device_record": {"name": d.name, "atom": d.atom.name, "peel": d.peel.name, "g": d.g, "k_max": d.k_max,
                                 "standins": list(d.standins)},
               "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
               "gate": f"all: a second launch word for word (gemm_ct, gemm_pearlc on {REF_GRID} CTAs, the hashes); sampled: "
                       "benchmarks/pouw/pearl_c/twin.py at twin.samples' tiles",
               "shortcuts": ["hashing unfused (#449's device code reads C~ and U back from memory)",
                             "m padded to 128 rows; work counts issued MACs",
                             "prices and G are the H100's stand-ins; W_ref, credit and gamma Derived at those"]}
        for name in ("sass.json", "toolchain.txt", "source.txt"):
            p = ship / name
            if p.exists():
                rec[name] = json.loads(p.read_text()) if name.endswith(".json") else p.read_text().strip()
        return rec

    def domain(self, shape) -> str | None:
        if shape.k % SEGMENT:
            return f"k = {shape.k} is not a multiple of 1,024 (A's 1 KiB row segments and the kernels' k blocks)"
        if shape.k > self.device.k_max:
            return f"k = {shape.k} is over the device record's bound {self.device.k_max}"
        if shape.n % R.BM:
            return f"n = {shape.n} is not a multiple of {R.BM} (the kernels' tiles)"
        if max(shape.m, shape.n) > 1 << 24:
            return "m or n is over 2^24 (the scheme's domain)"
        return None

    def work(self, shape) -> Work:
        m, n, k = padded(shape.m), shape.n, shape.k
        s = MShape(shape.m, shape.k, shape.n)
        fp8 = m * n * k + m * C.R * k + m * k * self.device.atom.k      # the chain, A′·F_B, A's forming (one atom per element)
        return Work(tensor_macs={"fp8-e4m3": fp8, "bf16": m * n * C.PEEL_K},
                    w_ref=str(self.scheme.wref(s)), credited=str(self.scheme.credit_of(s)),
                    unit=f"W1 units ({self.device.name}, stand-in prices: the H100's)",
                    hashed_bytes={"turboshake128/messages": 8 * m * n, "turboshake128/leaves": m * n,
                                  "turboshake128/rows": 4 * shape.m * k, "blake3/tickets": 64 * (m // SUBTILE) * (n // SUBTILE)},
                    weight_hashed_bytes={}, gamma=gamma(self.scheme, shape))

    def prepare(self, dev, shape, flags: frozenset[str], operands, corrupt: bool = False) -> _Call:
        return _Call(self, dev, shape, frozenset(flags), operands, corrupt)

    def known_bad(self, dev, shape, operands) -> _Call:
        return self.prepare(dev, shape, self.flag_sets["hash"], operands, corrupt=True)

    def gate(self, dev, call: _Call, shape, flags: frozenset[str]) -> Gate:
        differing = call.relaunch()
        tiles = call.twin_tiles()
        ok = not any(differing.values()) and not any(t["failed"] for t in tiles)
        return Gate(ok, {name: "all" for name in call.outputs},
                    {"words_differing_from_relaunch": differing, "twin_tiles": tiles, "scheme": self.scheme.name})


class PearlCSm120Unpromoted(PearlCSm120):
    name = "pearl-c-sm120-unpromoted-v1"
    variant = "sm120-unpromoted"
    panel = {"line": "pearl-c-sm120", "version": "v2"}
