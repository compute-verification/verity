"""Pearl-C's FP8 arm for GPU 6's bench harness (arm interface v0.4, `benchmarks/pouw/harness/arm.py`): pearl_c_sm120.cu's
kernels as the honest prover of `pearl-c-sm120-v1-h1` on the harness's FP32 operands, on the device record
`pearl_c_device.SM120` (G = 4, the panel's `pearl-c-sm120` v1-h1) or `SM120_UNPROMOTED` (v2-h1).

Each operand set is a one-unit epoch as `verity.core.protocols.pouw.audit` starts it, in hashing format -h1: B's frame-b3 row tree on the
device, its root read back, the salt from the salt coin of the set's epoch challenger (`challenger`, the arm's own, live)
opened after that root at `prepare` (`Epoch.start`), seed_B, and the
domain ids and unit block the kernels read (run.Pipeline's `keys`).  `weight_side` is the per-(weight, epoch) work: B's row
tree, the pads and level keys of A's tree and of the tile tree, E_B's, F_A's and F_B's lines, B's statistics, E′ and forming,
B̃·F_A, G and P_B.  `launch` is one call: at m not a multiple of 64, A's rows into a zero-padded buffer; A's frame-b3 row
tree, and seed_A and E_A's line key from its root on the device (`seed_line_a`); E_A's lines; A's statistics, E′ and
forming; A′·F_B and P_A; the chain with its peel and y; then the message digests over C̃ and U, the tile leaves (a partial last
row tile's over its real rows) and the tile tree.  At 128 x 128 tiles over m = m_real rows the chain is fused
(`gemm_pearlc_b`: y and the message digests hashed from registers, C̃ and U never stored); else `gemm_pearlc(_d)` stores C̃, U
and y and `hash_msg_v` digests them.  `nohash` is the same call without A's tree, seed_A, E_A's lines (it reads the set's,
from the last call that made them) and the hashes, so it writes the same y (at 128 x 128 tiles by `gemm_pearlc_s`, y alone;
else with C̃ and U).  `phases` splits the call
for the harness's dependent chain: `before` (the copy, A's tree and seed_A, E_A's lines, forming, A′·F_B, P_A), `gemm` (the
chain) and `after` (the hashes), which the chain may run on a side stream.  Each operand set has its own epoch, keys, call
buffers and weight side (its flag sets share them: the harness runs one call of a set at a time); the gate's second launch
has per-shape buffers.

The gate, at the exact shape and flag set: a second, unfused launch into the gate's own buffers (with hashing A's tree and
seed_A; C̃ by gemm_ct, another kernel; C̃, U and y by gemm_pearlc on 3 CTAs, another tile schedule; hash_msg_v and the other
hashes over those) must equal every output word for word, so the fused call's digests are held to hash_msg_v's, and gemm_ct's
C̃ the call's (or, where it stores none, the second launch's); with hashing, seed_A and E_A's line key must be the scheme's
from the device's root_A; and at the twin's sampled 64 x 64 tiles (`twin.samples`) the call's A and B codes (the twin's in
the kernels' K32 order, `fixture.k32`) and α, C̃, U (the second launch's where the call stores none) and y must equal #449's
CPU twin
(`benchmarks/pouw/pearl_c/twin.py`) on the device record, from the harness's operand rows and the call's seeds, and with
hashing the tile's leaf the scheme's over its real rows, that leaf's in the tile tree and row r0's in A's tree the verifier's
(`CommitmentDomain.leaf`).  `known_bad` flips C̃'s first word (the first message digest's, where the call stores no C̃)
between the product and the hashes.  `build` names the cubin (`binaries`) and the verifier the harness runs after the lease
(`verify.py` over the run's transcripts).

The transcript: when the harness frees a gated hash call of operand set 0, the arm fills every buffer of the set but its
tensor maps with 0xA5 (the trees, the tile leaves and what the weight side wrote among them), runs the weight side and the
call once more on the set's operands, and writes what the prover publishes and opens: `transcripts/<arm>/<shape>/manifest.json`
in the run's directory (the scheme, the matmul, the coin commitment and the salt coin's opening, the salt, the weight root, A's root, the tile tree's commitment,
and every file with its SHA-256), the levels of A's, B's and the tile tree and the tile leaves beside it, and the harness's
A and B rows (up to 512 MB at 8,192²) under PEARLC_ROWS (default: a temporary directory).  Then the negative control, the
same dump into `negative/<arm>/<shape>/` after the buffers are poisoned again and the call runs without its commitment or
the weight side (`NEGATIVE`).  `verify.py` runs `verity.core.protocols.pouw.audit.Verifier` on both after the lease, and passes only if it
accepts the transcript and rejects its negative control.

Shortcuts, flagged: m is padded to 64 rows (decode's m = 32 issues 2x its tensor work on 64 x 64 tiles, and `work` counts
issued MACs); the prices (SM120_PRICES) and G are the device record's (its stand-ins listed in the build record), so W_ref,
the credit and γ are Derived at those.

  python bench.py --arm-path <this directory> --arm pearlc_arm:PearlCSm120 [--arm pearlc_arm:PearlCSm120Unpromoted] ...

PEARLC_SHIP names build.sh's output (pearl_c_sm120.cubin, sass.json, toolchain.txt, source.txt; default this directory).
Needs Python >= 3.12 with numpy, and `verity` and `verity.core.protocols.pouw` importable (the twin, the epoch).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
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
W = F.W

from verity.core.protocols.pouw.audit import (ROW_SCHEMA, TILE_SCHEMA, Epoch, Workload, activation_binding, commitment_hash,  # noqa: E402
                               rows_domain, transcript_domain, weight_binding, weights_root)
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul  # noqa: E402
from verity.core.protocols.pouw.protocol import Shape as MShape  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES  # noqa: E402

T = W.T
KEYS = R.TIMING_KEYS
WEIGHT = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b", "epb_codes",
          "bt", "bf", "pb_t", "tree_b", "keys_b", "keys_a", "keys_t")
OWN = WEIGHT + ("ea_codes", "line_a", "seed_a")   # E_A's lines and their seed are the set's: nohash reads the last call's
CALL = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "at", "af", "pa_t", "c", "u", "y", "digests", "leaves", "tree_a",
        "tree_t", "tma")
HASHED = ("digests", "leaves", "tree_a", "tree_t")
REF = ("c2", "c", "u", "y", "line_a", "seed_a") + HASHED
REF_BUFS = REF + ("tma",)                 # the second launch's own tensor maps
REF_HASH_STEPS = ("pad_a", "pad_t", "hash_rows_a", "tree_a", "seed_a")
REF_GRID = 3
TRANSCRIPT_SCHEMA = "verity-pouw/pearl-c-sm120-transcript/v2"
POISON = 0xA5
NEGATIVE = ("the negative control: the call's buffers poisoned (0xA5) as before the transcript's call, then the call with A's "
            "row tree, seed_A, E_A's lines and every hash omitted and without the weight side, so it writes y (and C~ and U "
            "unfused) and nothing the transcript reads; the verifier must reject it")
#: TT_OUT's γ0, a statement parameter apart from the debit cap: the Lean's γ takes it at 1/400 at every cap
#: (`gammaFromTTOut`, cheap-binding.md §1)
GAMMA0 = Fraction(1, 400)


def padded(m: int) -> int:
    return -(-m // R.BM_D) * R.BM_D


def gamma(scheme: C.PearlC, shape) -> dict[str, str]:
    """γ at `shape` from the scheme's W_ref and credit (`bench.gamma`'s four readings on the H100): `cap`, the worst case under
    the debit cap; `floor`, a zero debit; `*_chain_only`, TT_OUT crediting no forming.  The debit cap ρ is the record's
    (`scheme.cap`); TT_OUT's γ0 is `GAMMA0`."""
    s = MShape(shape.m, shape.k, shape.n)
    wref, credit = Fraction(scheme.wref(s)), Fraction(scheme.credit_of(s))
    forming = scheme.prices.forming_credited * s.m * s.k
    rho = scheme.cap
    out = {"cap": 1 - (1 - GAMMA0) * (1 - rho) * credit / wref,
           "cap_chain_only": 1 - (1 - GAMMA0) * (credit - forming - rho * credit) / wref,
           "floor": 1 - (1 - GAMMA0) * credit / wref,
           "floor_chain_only": 1 - (1 - GAMMA0) * (credit - forming) / wref}
    return {**{k: str(v) for k, v in out.items()},
            "basis": f"Derived: TT_OUT γ0 = {GAMMA0}, ρ = {rho} on {scheme.device.name} at its record's prices, "
                     f"G = {scheme.device.g}"}


def expected(t, real: int, hashing: bool, ep: dict, tid: int, a_row: bytes) -> dict[str, bytes]:
    """A twin tile's bytes in the kernels' layout, by the names `_Call.tile_bytes` reads; with hashing the leaf over the
    tile's `real` rows (the scheme's `digest`), its leaf in the tile tree and row r0's in A's tree."""
    ia, ib = F.H.recip(t.a.alpha.astype(np.uint32)), F.H.recip(t.b.alpha.astype(np.uint32))
    out = {"at": F.k32(t.a.codes.astype(np.uint8)).tobytes(), "alpha_a": t.a.alpha.astype("<u4").tobytes(),
           "bt": F.k32(t.b.codes.astype(np.uint8)).tobytes(), "alpha_b": t.b.alpha.astype("<u4").tobytes(),
           "y": F.H.glue_y(t.u.astype(np.uint32), ia, ib).astype("<u2").tobytes(),
           "c": t.c.astype("<u4").tobytes(), "u": t.u.astype("<u4").tobytes()}
    if hashing:
        leaf = C.leaf(C.Checked(t.c[:real].tolist(), t.u[:real].tolist()), "h1")
        out.update(leaf=leaf, tile_leaf=ep["dom_t"].leaf(tid, tid, TILE_SCHEMA, leaf),
                   row_leaf=ep["dom_a"].leaf(t.r0, t.r0, ROW_SCHEMA, a_row))
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
        self.epoch: dict = {}
        self.job = None
        self.twin: dict = {}

    def free(self) -> None:
        for p in self.b.values():
            self.rd.free(p)


class _Call:
    def __init__(self, arm: PearlCSm120, hdev, shape, flags: frozenset[str], ops, corrupt: bool):
        rd, self.arm, self.hdev, self.shape, self.ops, self.freed = arm.rd, arm, hdev, shape, ops, False
        rd.sync()                          # the operands were filled on the harness's non-blocking stream; this runs on the null one
        self.hashing, self.corrupt, self.gated = "hash" in flags, corrupt, False
        m, n, k = self.m, self.n, self.k = padded(shape.m), shape.n, shape.k
        size = self.size = R.Pipeline.sizes(m, n, k, shape.m)
        self.scratch, _ = arm.acquire(arm.scratch, shape, lambda: {"ref_" + b: size[b] for b in REF_BUFS})
        self.set_key = (shape, ops.a.ptr, ops.b.ptr, ops.seed)
        self.own, fresh = arm.acquire(arm.sets, self.set_key, lambda: {**{b: size[b] for b in OWN + CALL},
                                                                      **({"xa": 4 * m * k} if m != shape.m else {}),
                                                                      **{"key_" + x: nb for x, nb in KEYS.items()}})
        w = self.own.b
        self.keys = {x: w["key_" + x] for x in KEYS}
        self.xa = w.get("xa", ops.a.ptr)
        self.pipe = R.Pipeline(rd, m, n, k, arm.variant, self.keys, self.xa, ops.b.ptr, bufs={b: w[b] for b in OWN + CALL},
                               m_real=shape.m, hashing=arm.scheme.hashing)
        steps = self.pipe.steps()
        by = {name: f for name, _, f in steps}
        y_only = None if self.hashing else self.pipe.extra().get("gemm_pearlc_s")
        self.stored = () if self.pipe.fused and self.hashing or y_only else ("c", "u")
        phase = lambda *ps: [f for _, p, f in steps if p in ps]  # noqa: E731
        self.weight = phase("epoch", "unit", "job")
        commit = phase("commit") + [by["lines_ea"]]
        forming = [f for name, p, f in steps if p == "forming" and name != "lines_ea"]
        cleanup = [f for name, p, f in steps if p == "cleanup" and name != "gemm_pearlc"]
        copy = [lambda: rd.copy_async(self.xa, ops.a.ptr, 4 * shape.m * k)] if self.xa != ops.a.ptr else []
        flip = [lambda: hdev.xor_byte(w["c" if self.stored else "digests"], 0, 0x01, rd.stream)] if corrupt else []
        self.parts = {"before": copy + (commit if self.hashing else []) + forming + cleanup, "gemm": [y_only or by["gemm_pearlc"]],
                      "after": flip + (phase("hashing") if self.hashing else [])}
        extra = self.pipe.extra()
        self.uncommitted = copy + forming + cleanup + [extra.get("gemm_pearlc_s") or extra["gemm_pearlc_cu"]]
        self.phases = {name: (lambda stream, fs=fs: self.enqueue(fs, stream)) for name, fs in self.parts.items()}
        names = ("y",) + self.stored + (HASHED if self.hashing else ())
        self.outputs = {name: Buffer(w[name], size[name]) for name in names}
        if fresh:
            self.own.epoch = self.start_epoch(steps)
        self.enqueue(copy + commit, None)
        self.launch(None)
        rd.sync()

    def put(self, name: str, data: bytes) -> None:
        self.arm.rd.ck("cuMemcpyHtoD_v2", R.u64(self.keys[name]), R.C.c_char_p(data), len(data))

    def root_b(self) -> bytes:
        return self.arm.rd.read(self.pipe.root("b"), 32)

    def start_epoch(self, steps) -> dict:
        """B's row tree under its domain (the weight's alone), then the epoch from the set's challenger's salt coin opened
        after its root, and the keys that follow: the unit's pads and level keys and the weight side run under them."""
        arm, rd, sh = self.arm, self.arm.rd, self.shape
        h = commitment_hash(arm.scheme)
        mm = Matmul(arm.index, MShape(sh.m, sh.k, sh.n), self.ops.weight_id)
        self.put("dom_b", rows_domain(weight_binding(mm.weight), -2, sh.n, h).domain_id)
        self.enqueue([f for _, p, f in steps if p == "epoch"], None)
        rd.sync()
        root_b, coins = self.root_b(), arm.challenger(sh, self.ops)
        roots = {mm.weight: root_b}
        opening = coins.open("salt", weights_root(roots))
        epoch = Epoch.start(Workload((mm,)), roots, coins.commitment, opening)
        seed_b = C._labelled(epoch.salt + root_b, "seed-B")
        for name, x in F.keys_of(arm.scheme, epoch, mm, seed_b).items():
            self.put(name, x)
        self.enqueue([f for _, p, f in steps if p in ("unit", "job")], None)
        rd.sync()
        return {"matmul": mm, "epoch": epoch, "coins": coins.commitment, "salt_opening": opening, "root_b": root_b, "seed_b": seed_b,
                "dom_a": rows_domain(activation_binding(epoch.salt, mm.index), mm.index, sh.m, h),
                "dom_t": transcript_domain(epoch, self.pipe.tiles, h)}

    def enqueue(self, steps: list, stream) -> None:
        rd = self.arm.rd
        rd.stream = stream
        try:
            for f in steps:
                f()
        finally:
            rd.stream = None

    def launch(self, stream) -> None:
        self.enqueue([f for part in self.parts.values() for f in part], stream)

    def weight_side(self, stream) -> None:
        self.enqueue(self.weight, stream)

    def poison(self) -> None:
        """Every buffer of the set but its tensor maps, and a padded A's real rows, filled with POISON: what the weight side
        and a launch write, the transcript's trees and leaves among them."""
        rd, w = self.arm.rd, self.own.b
        rd.sync()
        for name in OWN + CALL:
            if name != "tma":
                rd.ck("cuMemsetD8_v2", R.u64(w[name]), POISON, self.size[name])
        if self.xa != self.ops.a.ptr:
            rd.ck("cuMemsetD8_v2", R.u64(self.xa), POISON, 4 * self.shape.m * self.k)
        rd.sync()

    def free(self) -> None:
        if self.freed:
            return
        self.freed = True
        try:
            if self.gated and self.hashing and not self.corrupt and self.ops.index == 0:
                self.arm.transcript(self)
        except Exception as ex:  # noqa: BLE001 - the harness's teardown goes on; verify.py reports the missing transcript
            self.arm.transcribed[self.shape.label] = f"error: {type(ex).__name__}: {ex}"
            print(f"pearlc_arm {self.arm.name} {self.shape.label}: no transcript: {type(ex).__name__}: {ex}", file=sys.stderr)
            if self.arm.out_dir is not None:
                err = self.arm.out_dir / "transcripts" / self.arm.name / self.shape.label
                err.mkdir(parents=True, exist_ok=True)
                (err / "error.txt").write_text(f"{type(ex).__name__}: {ex}\n")
        finally:
            self.arm.release(self.arm.sets, self.set_key)
            self.arm.release(self.arm.scratch, self.shape)

    # ---- the gate's parts ------------------------------------------------------------------------------------------------

    def relaunch(self) -> dict[str, int]:
        """Words differing between each output and the second launch's, which is unfused (gemm_pearlc stores C̃ and U,
        hash_msg digests them); `c@gemm_ct` is gemm_ct's C̃ against the call's, or the second launch's where the call stores
        none; with hashing, seed_A and E_A's line key too."""
        rd, s, w = self.arm.rd, self.scratch.b, self.own.b
        ref = R.Pipeline(rd, self.m, self.n, self.k, self.arm.variant, self.keys, self.xa, self.ops.b.ptr,
                         bufs={**self.pipe.b, **{b: s["ref_" + b] for b in REF_BUFS}}, m_real=self.shape.m, fused=False,
                         hashing=self.arm.scheme.hashing)
        real = ref.gemm
        ref.gemm = lambda *a, **kw: real(*a, **{**kw, "grid": REF_GRID})
        steps = {name: f for name, _, f in ref.steps()}
        names = ("gemm_ct", "gemm_pearlc")
        if self.hashing:
            names = REF_HASH_STEPS + names + ("hash_msg", "hash_leaf", "tree_leaves_t", "tree_t")
        for name in names:
            if name in steps:
                steps[name]()
        rd.sync()
        out = {name: self.diff(buf.ptr, s["ref_" + name], buf.nbytes) for name, buf in self.outputs.items()}
        out["c@gemm_ct"] = self.diff(w["c"] if self.stored else s["ref_c"], s["ref_c2"], self.size["c"])
        if self.hashing:
            out.update({name: self.diff(w[name], s["ref_" + name], 32) for name in ("seed_a", "line_a")})
        return out

    def diff(self, x: int, y: int, nbytes: int) -> int:
        a, b = (np.frombuffer(self.arm.rd.read(p, nbytes), "<u4") for p in (x, y))
        return int(np.count_nonzero(a != b))

    def noise(self) -> tuple[C.Noise, dict]:
        """The scheme's noise on the device's root_A (with hashing; nohash reads the seeds the set's last call derived), and
        whether the device's seed_A and E_A's line key are the scheme's."""
        rd, w, ep = self.arm.rd, self.own.b, self.own.epoch
        root_a, seed_a, line_a = rd.read(self.pipe.root("a"), 32), rd.read(w["seed_a"], 32), rd.read(w["line_a"], 32)
        if not self.hashing:
            return C.Noise(seed_a, ep["seed_b"]), {}
        noise = self.arm.scheme.noise(Context(ep["epoch"].salt, ep["matmul"], root_a, ep["root_b"]))
        return noise, {"seed_a": seed_a == noise.seed_a, "line_a": line_a == P._line_key(noise.seed_a),
                       "seed_b": noise.seed_b == ep["seed_b"]}

    def rows(self, ptr: int, r0: int, real: int) -> np.ndarray:
        """Rows r0 .. r0 + 63 of an FP32 (real x k) operand, zero past `real` (the filler)."""
        take, out = max(0, min(T, real - r0)), np.zeros((T, self.k), np.uint32)
        if take:
            out[:take] = np.frombuffer(self.arm.rd.read(ptr + 4 * self.k * r0, 4 * self.k * take), "<u4").reshape(take, self.k)
        return out

    def tile_bytes(self, r0: int, c0: int, tid: int) -> dict[str, bytes]:
        """The tile's bytes as the call wrote them; C̃ and U the second launch's where the call stores none (the fused call,
        whose digests the relaunch holds to hash_msg's over those)."""
        n, k, w = self.n, self.k, self.own.b
        get = self.arm.rd.read
        rows = range(r0, r0 + T)
        cu = {x: w[x] if self.stored else self.scratch.b["ref_" + x] for x in ("c", "u")}
        out = {"at": get(w["at"] + r0 * k, T * k), "alpha_a": get(w["alpha_a"] + 4 * r0, 4 * T),
               "bt": get(w["bt"] + c0 * k, T * k), "alpha_b": get(w["alpha_b"] + 4 * c0, 4 * T),
               "y": b"".join(get(w["y"] + 2 * (i * n + c0), 2 * T) for i in rows),
               "c": b"".join(get(cu["c"] + 4 * (i * n + c0), 4 * T) for i in rows),
               "u": b"".join(get(cu["u"] + 4 * (i * n + c0), 4 * T) for i in rows)}
        if self.hashing:
            out.update(leaf=get(w["leaves"] + 32 * tid, 32), tile_leaf=get(w["tree_t"] + 32 * tid, 32),
                       row_leaf=get(w["tree_a"] + 32 * r0, 32))
        return out

    def twin_tiles(self, noise: C.Noise) -> list[dict]:
        sh, wt, ep = self.shape, self.own, self.own.epoch
        tshape = W.Shape(sh.label, sh.m, sh.n, sh.k)
        sd = {"seed_a": noise.seed_a, "seed_b": noise.seed_b, "salt": ep["epoch"].salt, "root_b": ep["root_b"]}
        if wt.job is None:
            wt.job = W.job(tshape, self.arm.device, sd)
        job = dataclasses.replace(wt.job, seeds=sd)
        out = []
        for r0, c0 in W.samples(tshape):
            a_rows, b_rows = self.rows(self.ops.a.ptr, r0, sh.m), self.rows(self.ops.b.ptr, c0, sh.n)
            key = (r0, c0, hashlib.sha256(a_rows.tobytes() + b_rows.tobytes() + noise.seed_a).digest())
            if key not in wt.twin:
                wt.twin[key] = W.tile(job, r0, c0, a_rows, b_rows)
            tid = (r0 // T) * (self.n // T) + c0 // T
            want = expected(wt.twin[key], min(T, sh.m - r0), self.hashing, ep, tid, a_rows[0].astype("<u4").tobytes())
            got = self.tile_bytes(r0, c0, tid)
            out.append({"tile": [r0, c0], "failed": sorted(name for name in want if got[name] != want[name])})
        return out


class PearlCSm120:
    name = "pearl-c-sm120-v1-h1"
    stream = "fp8"
    precision = "fp8-e4m3"
    inputs = "f32"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset()}
    variant = "sm120"
    panel = {"line": "pearl-c-sm120", "version": "v1-h1"}
    index = 0

    def __init__(self):
        self.device = DEVICES[self.variant]
        self.scheme = C.PearlC(forming="v1", device=self.device, hashing="h1")
        assert self.scheme.name == self.name, (self.scheme.name, self.name)
        self.rd = None
        self.out_dir: Path | None = None
        self.transcribed: dict[str, str] = {}
        self.scratch: dict = {}
        self.sets: dict = {}

    def challenger(self, shape, ops) -> Challenger:
        """An operand set's epoch challenger: its coins live, from the operating system's generator."""
        return Challenger(0)

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
        self.out_dir = Path(ctx["out_dir"]) if ctx.get("out_dir") else None
        d = self.device
        rec = {"arm": self.name, "scheme": self.scheme.name, "panel": self.panel,
               "device_record": {"name": d.name, "atom": d.atom.name, "peel": d.peel.name, "g": d.g, "k_max": d.k_max,
                                 "standins": list(d.standins)},
               "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
               "binaries": [str(cubin)],
               "verifier": {"per": "run", "transcript": "transcripts/{arm}/{shape}/manifest.json",
                            "commands": [["{python}", str(HERE / "verify.py"), "{run_dir}", "--commit", "{commit}"]]},
               "call": "at 128 x 128 tiles over m = m_real rows the fused chain: gemm_pearlc_b stores y and the -h1 message "
                       "digests hashed from registers (C~, U never stored), gemm_pearlc_s y alone without hashing; else "
                       "gemm_pearlc(_d) stores C~, U, y and hash_msg_v digests them",
               "epoch": "per operand set: B's frame-b3 row tree on the device, the salt from the salt coin of the set's own "
                        "epoch challenger (verity.core.protocols.pouw.coins.Challenger) opened after its root "
                        "(verity.core.protocols.pouw.audit.Epoch.start), matmul index 0 on the set's weight",
               "gate": f"all: a second, unfused launch word for word (A's tree and seed_A, gemm_ct, gemm_pearlc on {REF_GRID} "
                       "CTAs storing C~ and U, hash_msg_v over them and the other hashes), so fused digests are held to "
                       "hash_msg_v's; gemm_ct's C~ against the call's or, unstored, the second launch's; seed_A and E_A's line "
                       "key the scheme's from the device's root_A; sampled: benchmarks/pouw/pearl_c/twin.py at twin.samples' "
                       "tiles (C~ and U the second launch's where the call stores none), the tile leaf the scheme's over the "
                       "real rows, its tile-tree leaf and row r0's leaf the verifier's",
               "transcript": "operand set 0's hash call, when the harness frees it: every set buffer but the tensor maps "
                             "poisoned (0xA5), then the weight side and the call again, into transcripts/<arm>/<shape>/ "
                             "(manifest.json, tree_a, tree_b, tree_t, leaves), the harness's A and B rows under PEARLC_ROWS; "
                             "then poisoned again and the call without its commitment or the weight side into "
                             "negative/<arm>/<shape>/; verify.py accepts the first and must reject the second",
               "shortcuts": ["m padded to 64 rows (64 x 64 tiles below 128 rows); work counts issued MACs",
                             "prices and G are the device record's (stand-ins listed); W_ref, credit and gamma Derived at those"]}
        for name in ("sass.json", "toolchain.txt", "source.txt"):
            p = ship / name
            if p.exists():
                rec[name] = json.loads(p.read_text()) if name.endswith(".json") else p.read_text().strip()
        return rec

    def domain(self, shape) -> str | None:
        if shape.k % 1024:
            return f"k = {shape.k} is not a multiple of 1,024 (the kernels' k blocks)"
        if shape.n % R.BM:
            return f"n = {shape.n} is not a multiple of {R.BM} (the kernels' tiles)"
        try:
            self.scheme.check(Matmul(self.index, MShape(shape.m, shape.k, shape.n), 0))
        except DomainError as ex:
            return str(ex)
        return None

    def work(self, shape) -> Work:
        m, n, k, mr = padded(shape.m), shape.n, shape.k, shape.m
        s = MShape(shape.m, shape.k, shape.n)
        tiles = -(-mr // T) * (n // T)
        fp8 = m * n * k + m * C.R * k + m * k * self.device.atom.k      # the chain, A′·F_B, A's forming (one atom per element)
        return Work(tensor_macs={"fp8-e4m3": fp8, "bf16": m * n * C.PEEL_K},
                    w_ref=str(self.scheme.wref(s)), credited=str(self.scheme.credit_of(s)),
                    unit=f"W1 units ({self.device.name}, its record's prices)",
                    hashed_bytes={"blake3/rows": mr * (64 + 4 * k), "blake3/messages": 8 * mr * n, "blake3/leaves": mr * n,
                                  "blake3/tree_nodes": 64 * (R.pow2(mr) - 1 + R.pow2(tiles) - 1),
                                  "blake3/tile_leaves": tiles * (64 + 32)},
                    weight_hashed_bytes={"blake3/rows": n * (64 + 4 * k), "blake3/tree_nodes": 64 * (R.pow2(n) - 1)},
                    gamma=gamma(self.scheme, shape))

    def prepare(self, dev, shape, flags: frozenset[str], operands, corrupt: bool = False) -> _Call:
        return _Call(self, dev, shape, frozenset(flags), operands, corrupt)

    def known_bad(self, dev, shape, operands) -> _Call:
        return self.prepare(dev, shape, self.flag_sets["hash"], operands, corrupt=True)

    def gate(self, dev, call: _Call, shape, flags: frozenset[str]) -> Gate:
        differing = call.relaunch()
        noise, seeds = call.noise()
        tiles = call.twin_tiles(noise)
        ok = call.gated = not any(differing.values()) and all(seeds.values()) and not any(t["failed"] for t in tiles)
        return Gate(ok, {name: "all" for name in call.outputs},
                    {"words_differing_from_relaunch": differing, "seeds": seeds, "twin_tiles": tiles, "scheme": self.scheme.name})

    def transcript(self, call: _Call) -> None:
        """Operand set 0's hash call once more, and what the verifier reads: `transcripts/<arm>/<shape>/` beside the run's
        output (small), A's and B's rows (the harness's operands) under PEARLC_ROWS.  Every buffer of the set is poisoned
        first, so the weight side and the call run again and the dump holds only what they wrote.  Then the negative control
        (NEGATIVE) into `negative/<arm>/<shape>/`, which verify.py requires rejected; then the weight side and the call once
        more, so the set is left as an honest call leaves it.  Once per shape; with no run directory, nothing."""
        sh = call.shape
        if self.out_dir is None or sh.label in self.transcribed:
            return
        out = self.out_dir / "transcripts" / self.name / sh.label
        rows = Path(os.environ.get("PEARLC_ROWS") or tempfile.mkdtemp(prefix="pearlc-rows-")) / self.name / sh.label
        rows.mkdir(parents=True, exist_ok=True)
        self.transcribed[sh.label] = str(out)
        rd = self.rd
        call.poison()                      # syncs first: the harness's stream is non-blocking
        call.weight_side(None)
        call.launch(None)
        rd.sync()
        files = {}
        for name, ptr, count in (("rows_a", call.ops.a.ptr, sh.m), ("rows_b", call.ops.b.ptr, sh.n)):
            data = rd.read(ptr, 4 * count * sh.k)
            (rows / f"{name}.bin").write_bytes(data)
            files[name] = {"path": str(rows / f"{name}.bin"), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.dump(call, out, files, None)
        call.poison()
        call.enqueue(call.uncommitted, None)
        rd.sync()
        self.dump(call, self.out_dir / "negative" / self.name / sh.label, files, NEGATIVE)
        call.weight_side(None)
        call.launch(None)
        rd.sync()

    def dump(self, call: _Call, out: Path, rows: dict, negative: str | None) -> None:
        """The set's trees and tile leaves as they are on the device, and the manifest over them and `rows`."""
        rd, w, ep, size, sh = self.rd, call.own.b, call.own.epoch, call.size, call.shape
        out.mkdir(parents=True, exist_ok=True)
        files, data = dict(rows), {}
        for name in ("tree_a", "tree_t", "tree_b", "leaves"):
            data[name] = rd.read(w[name], size[name])
            (out / f"{name}.bin").write_bytes(data[name])
            files[name] = {"path": f"{name}.bin", "bytes": len(data[name]), "sha256": hashlib.sha256(data[name]).hexdigest()}
        mm, epoch = ep["matmul"], ep["epoch"]
        man = {"schema": TRANSCRIPT_SCHEMA, "arm": self.name, "scheme": self.scheme.name,
               "scheme_args": {"forming": "v1", "device": self.device.name, "hashing": "h1"},
               "shape": {"label": sh.label, "m": sh.m, "k": sh.k, "n": sh.n}, "matmul": {"index": mm.index, "weight": mm.weight},
               "coins": ep["coins"].to_json(), "salt_opening": ep["salt_opening"].to_json(), "salt": epoch.salt.hex(), "weight_root": ep["root_b"].hex(),
               "activation_root": data["tree_a"][-32:].hex(),
               "commitment": {"root": data["tree_t"][-32:].hex(), "count": call.pipe.tiles},
               "poisoned": f"0x{POISON:02X}", **({"negative_control": negative} if negative else {}), "files": files}
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")


class PearlCSm120Unpromoted(PearlCSm120):
    name = "pearl-c-sm120-unpromoted-v1-h1"
    variant = "sm120-unpromoted"
    panel = {"line": "pearl-c-sm120", "version": "v2-h1"}
