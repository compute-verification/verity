"""Pearl-C's FP8 arm for GPU 6's bench harness (arm interface v0.2, `benchmarks/pouw/harness/arm.py`): pearl_c_sm120.cu's
kernels as the honest prover of `pearl-c-sm120-v1-h1` on the harness's FP32 operands, on the device record
`pearl_c_device.SM120` (G = 4, the panel's `pearl-c-sm120` v1-h1) or `SM120_UNPROMOTED` (v2-h1).

Each operand set is a one-unit epoch as `verity_pouw.audit` starts it, in hashing format -h1: B's frame-b3 row tree on the
device, its root read back, the salt from that root and a beacon of `os.urandom(32)` drawn at `prepare`, seed_B, and the
domain ids and unit block the kernels read (run.Pipeline's `keys`).  `weight_side` is the per-(weight, epoch) work: B's row
tree, the pads and level keys of A's tree and of the tile tree, E_B's, F_A's and F_B's lines, B's statistics, E′ and forming,
B̃·F_A, G and P_B.  `launch` is one call: at m not a multiple of 64, A's rows into a zero-padded buffer; A's frame-b3 row
tree, and seed_A and E_A's line key from its root on the device (`seed_line_a`); E_A's lines; A's statistics, E′ and
forming; A′·F_B and P_A; the chain with its peel and y; then the message digests over C̃ and U, the tile leaves (a partial last
row tile's over its real rows) and the tile tree.  `nohash` is the same call without A's tree, seed_A, E_A's lines (it reads
the set's, from the last call that made them) and the hashes, so it writes the same y, C̃ and U.  `phases` splits the call
for the harness's dependent chain: `before` (the copy, A's tree and seed_A, E_A's lines, forming, A′·F_B, P_A), `gemm` (the
chain) and `after` (the hashes), which the chain may run on a side stream.  Each operand set has its own epoch, keys, call
buffers and weight side (its flag sets share them: the harness runs one call of a set at a time); the gate's second launch
has per-shape buffers.

The gate, at the exact shape and flag set: a second launch into the gate's own buffers (with hashing A's tree and seed_A,
the trees by other kernels, run.ALT_FORMS; C̃ by gemm_ct, another kernel; C̃, U and y by gemm_pearlc on 3 CTAs, another tile
schedule; the hashes over those) must equal
every output word for word; with hashing, seed_A and E_A's line key must be the scheme's from the device's root_A; and at the
twin's sampled 64 x 64 tiles (`twin.samples`) the call's A and B codes and α, C̃, U and y must equal #449's CPU twin
(`benchmarks/pouw/pearl_c/twin.py`) on the device record, from the harness's operand rows and the call's seeds, and with
hashing the tile's leaf the scheme's over its real rows, that leaf's in the tile tree and row r0's in A's tree the verifier's
(`CommitmentDomain.leaf`).  `known_bad` flips C̃'s first word between the product and the hashes.

The transcript: when the harness frees a gated hash call of operand set 0, the arm launches it once more on the set's
operands and writes what the prover publishes and opens: `transcripts/<arm>/<shape>/manifest.json` in the run's directory
(the scheme, the matmul, the beacon, the salt, the weight root, A's root, the tile tree's commitment, and every file with its
SHA-256), the levels of A's, B's and the tile tree and the tile leaves beside it, and A's and B's rows (up to 512 MB at
8,192²) under PEARLC_ROWS (default: a temporary directory).  `verify.py` runs `verity_pouw.audit.Verifier` on it after the
lease.

Shortcuts, flagged: m is padded to 64 rows (decode's m = 32 issues 2x its tensor work on 64 x 64 tiles, and `work` counts
issued MACs); the prices and G are the device record's (its stand-ins listed in the build record), so W_ref, the credit and
γ are Derived at those.

  python bench.py --arm-path <this directory> --arm pearlc_arm:PearlCSm120 [--arm pearlc_arm:PearlCSm120Unpromoted] ...

PEARLC_SHIP names build.sh's output (pearl_c_sm120.cubin, sass.json, toolchain.txt, source.txt; default this directory).
Needs Python >= 3.12 with numpy, and `verity` and `verity_pouw` importable (the twin, the epoch).
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

from verity_pouw.audit import (ROW_SCHEMA, TILE_SCHEMA, Epoch, Workload, activation_binding, rows_domain,  # noqa: E402
                               transcript_domain, weight_binding)
from verity_pouw.protocol import Context, DomainError, Matmul  # noqa: E402
from verity_pouw.protocol import Shape as MShape  # noqa: E402
from verity_pouw.schemes import pearl_c as C  # noqa: E402
from verity_pouw.schemes import pearl_kw as P  # noqa: E402
from verity_pouw.schemes.pearl_c_device import DEVICES  # noqa: E402

T = W.T
KEYS = R.TIMING_KEYS
WEIGHT = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b", "epb_codes",
          "bt", "bf", "pb_t", "tree_b", "keys_b", "keys_a", "keys_t", "ctr_b")
OWN = WEIGHT + ("ea_codes", "line_a", "seed_a")   # E_A's lines and their seed are the set's: nohash reads the last call's
CALL = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "at", "af", "pa_t", "c", "u", "y", "digests", "leaves", "tree_a",
        "tree_t", "ctr_a", "ctr_t")
HASHED = ("digests", "leaves", "tree_a", "tree_t")
REF = ("c2", "c", "u", "y", "line_a", "seed_a") + HASHED
REF_HASH_STEPS = ("pad_a", "pad_t", "hash_rows_a", "tree_a", "seed_a")
REF_GRID = 3
TRANSCRIPT_SCHEMA = "verity-pouw/pearl-c-sm120-transcript/v1"
GAMMA0 = RHO_CAP = Fraction(1, 400)       # TT_OUT's γ0 and the debit cap ρ (statement parameters, cheap-binding.md §1)


def padded(m: int) -> int:
    return -(-m // R.BM_D) * R.BM_D


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
            "basis": f"Derived: TT_OUT γ0 = ρ = 1/400 on {scheme.device.name} at its record's prices, G = {scheme.device.g}"}


def expected(t, real: int, hashing: bool, ep: dict, tid: int, a_row: bytes) -> dict[str, bytes]:
    """A twin tile's bytes in the kernels' layout, by the names `_Call.tile_bytes` reads; with hashing the leaf over the
    tile's `real` rows (the scheme's `digest`), its leaf in the tile tree and row r0's in A's tree."""
    ia, ib = F.H.recip(t.a.alpha.astype(np.uint32)), F.H.recip(t.b.alpha.astype(np.uint32))
    out = {"at": t.a.codes.astype(np.uint8).tobytes(), "alpha_a": t.a.alpha.astype("<u4").tobytes(),
           "bt": t.b.codes.astype(np.uint8).tobytes(), "alpha_b": t.b.alpha.astype("<u4").tobytes(),
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
        self.scratch, _ = arm.acquire(arm.scratch, shape, lambda: {"ref_" + b: size[b] for b in REF})
        self.set_key = (shape, ops.a.ptr, ops.b.ptr, ops.seed)
        self.own, fresh = arm.acquire(arm.sets, self.set_key, lambda: {**{b: size[b] for b in OWN + CALL},
                                                                      **({"xa": 4 * m * k} if m != shape.m else {}),
                                                                      **{"key_" + x: nb for x, nb in KEYS.items()}})
        w = self.own.b
        self.keys = {x: w["key_" + x] for x in KEYS}
        self.xa = w.get("xa", ops.a.ptr)
        self.pipe = R.Pipeline(rd, m, n, k, arm.variant, self.keys, self.xa, ops.b.ptr, bufs={b: w[b] for b in OWN + CALL},
                               m_real=shape.m)
        steps = self.pipe.steps()
        by = {name: f for name, _, f in steps}
        phase = lambda *ps: [f for _, p, f in steps if p in ps]  # noqa: E731
        self.weight = phase("epoch", "unit", "job")
        commit = phase("commit") + [by["lines_ea"]]
        forming = [f for name, p, f in steps if p == "forming" and name != "lines_ea"]
        cleanup = [f for name, p, f in steps if p == "cleanup" and name != "gemm_pearlc"]
        copy = [lambda: rd.copy_async(self.xa, ops.a.ptr, 4 * shape.m * k)] if self.xa != ops.a.ptr else []
        flip = [lambda: hdev.xor_byte(w["c"], 0, 0x01, rd.stream)] if corrupt else []
        self.parts = {"before": copy + (commit if self.hashing else []) + forming + cleanup, "gemm": [by["gemm_pearlc"]],
                      "after": flip + (phase("hashing") if self.hashing else [])}
        self.phases = {name: (lambda stream, fs=fs: self.enqueue(fs, stream)) for name, fs in self.parts.items()}
        names = ("y", "c", "u") + (HASHED if self.hashing else ())
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
        """B's row tree under its domain (the weight's alone), then the epoch on its root and the arm's beacon, and the keys
        that follow: the unit's pads and level keys and the weight side run under them."""
        arm, rd, sh = self.arm, self.arm.rd, self.shape
        mm = Matmul(arm.index, MShape(sh.m, sh.k, sh.n), self.ops.weight_id)
        self.put("dom_b", rows_domain(weight_binding(mm.weight), -2, sh.n, "blake3").domain_id)
        self.enqueue([f for _, p, f in steps if p == "epoch"], None)
        rd.sync()
        root_b, beacon = self.root_b(), arm.beacon(sh, self.ops)
        epoch = Epoch.start(Workload((mm,)), {mm.weight: root_b}, beacon)
        seed_b = C._labelled(epoch.salt + root_b, "seed-B")
        for name, x in F.keys_of(arm.scheme, epoch, mm, seed_b).items():
            self.put(name, x)
        self.enqueue([f for _, p, f in steps if p in ("unit", "job")], None)
        rd.sync()
        return {"matmul": mm, "epoch": epoch, "beacon": beacon, "root_b": root_b, "seed_b": seed_b,
                "dom_a": rows_domain(activation_binding(epoch.salt, mm.index), mm.index, sh.m, "blake3"),
                "dom_t": transcript_domain(epoch, self.pipe.tiles, "blake3")}

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
        """Words differing between each output and the second launch's (gemm_ct's C̃ as `c@gemm_ct`; with hashing, seed_A and
        E_A's line key too), whose trees are run.ALT_FORMS'."""
        rd, s, w = self.arm.rd, self.scratch.b, self.own.b
        ref = R.Pipeline(rd, self.m, self.n, self.k, self.arm.variant, self.keys, self.xa, self.ops.b.ptr,
                         bufs={**self.pipe.b, **{b: s["ref_" + b] for b in REF}}, m_real=self.shape.m)
        ref.rows_form, ref.tree_form = R.ALT_FORMS
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
        out["c@gemm_ct"] = self.diff(w["c"], s["ref_c2"], self.size["c"])
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
        n, k, w = self.n, self.k, self.own.b
        get = self.arm.rd.read
        rows = range(r0, r0 + T)
        out = {"at": get(w["at"] + r0 * k, T * k), "alpha_a": get(w["alpha_a"] + 4 * r0, 4 * T),
               "bt": get(w["bt"] + c0 * k, T * k), "alpha_b": get(w["alpha_b"] + 4 * c0, 4 * T),
               "y": b"".join(get(w["y"] + 2 * (i * n + c0), 2 * T) for i in rows),
               "c": b"".join(get(w["c"] + 4 * (i * n + c0), 4 * T) for i in rows),
               "u": b"".join(get(w["u"] + 4 * (i * n + c0), 4 * T) for i in rows)}
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

    def beacon(self, shape, ops) -> bytes:
        return os.urandom(32)

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
               "epoch": "per operand set: B's frame-b3 row tree on the device, the salt from its root and os.urandom(32) as "
                        "the beacon (verity_pouw.audit.Epoch.start), matmul index 0 on the set's weight",
               "gate": f"all: a second launch word for word (A's tree and seed_A, gemm_ct, gemm_pearlc on {REF_GRID} CTAs, "
                       "the hashes); seed_A and E_A's line key the scheme's from the device's root_A; sampled: "
                       "benchmarks/pouw/pearl_c/twin.py at twin.samples' tiles, the tile leaf the scheme's over the real rows, "
                       "its tile-tree leaf and row r0's leaf the verifier's",
               "transcript": "operand set 0's hash call, launched again when the harness frees it: transcripts/<arm>/<shape>/ "
                             "(manifest.json, tree_a, tree_b, tree_t, leaves); A's and B's rows under PEARLC_ROWS; verify.py",
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
        output (small), A's and B's rows under PEARLC_ROWS.  Once per shape; with no run directory, nothing."""
        sh = call.shape
        if self.out_dir is None or sh.label in self.transcribed:
            return
        out = self.out_dir / "transcripts" / self.name / sh.label
        rows = Path(os.environ.get("PEARLC_ROWS") or tempfile.mkdtemp(prefix="pearlc-rows-")) / self.name / sh.label
        out.mkdir(parents=True, exist_ok=True)
        rows.mkdir(parents=True, exist_ok=True)
        self.transcribed[sh.label] = str(out)
        rd, w, ep, size = self.rd, call.own.b, call.own.epoch, call.size
        rd.sync()                          # the harness's stream is non-blocking: nothing of it may still run under this launch
        call.launch(None)
        rd.sync()
        files = {}

        def put(where: Path, name: str, data: bytes) -> bytes:
            (where / f"{name}.bin").write_bytes(data)
            files[name] = {"path": f"{name}.bin" if where == out else str(where / f"{name}.bin"), "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}
            return data

        tree_a, tree_t = (put(out, name, rd.read(w[name], size[name])) for name in ("tree_a", "tree_t"))
        put(out, "tree_b", rd.read(w["tree_b"], size["tree_b"]))
        put(out, "leaves", rd.read(w["leaves"], size["leaves"]))
        put(rows, "rows_a", rd.read(call.xa, 4 * sh.m * sh.k))
        put(rows, "rows_b", rd.read(call.ops.b.ptr, 4 * sh.n * sh.k))
        mm, epoch = ep["matmul"], ep["epoch"]
        man = {"schema": TRANSCRIPT_SCHEMA, "arm": self.name, "scheme": self.scheme.name,
               "scheme_args": {"forming": "v1", "device": self.device.name, "hashing": "h1"},
               "shape": {"label": sh.label, "m": sh.m, "k": sh.k, "n": sh.n}, "matmul": {"index": mm.index, "weight": mm.weight},
               "beacon": ep["beacon"].hex(), "salt": epoch.salt.hex(), "weight_root": ep["root_b"].hex(),
               "activation_root": tree_a[-32:].hex(), "commitment": {"root": tree_t[-32:].hex(), "count": call.pipe.tiles},
               "files": files}
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")


class PearlCSm120Unpromoted(PearlCSm120):
    name = "pearl-c-sm120-unpromoted-v1-h1"
    variant = "sm120-unpromoted"
    panel = {"line": "pearl-c-sm120", "version": "v2-h1"}
