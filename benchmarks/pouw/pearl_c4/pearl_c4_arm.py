"""Pearl-C4's arms for the PoUW harness (benchmarks/pouw/harness, arm interface v0): `PearlC4Nv` (``pearl-c-nvfp4-v0``, the
FP4 headline, divided by the ``nvfp4`` baseline) and `PearlC4Mx` (``pearl-c-mxfp4-v0``, beside it, by ``mxfp4``).

  bench.py ... --arm-path benchmarks/pouw/pearl_c4 --arm pearl_c4_arm:PearlC4Nv --arm pearl_c4_arm:PearlC4Mx
  env: PEARLC4_CUBIN (default pearl_c4.cubin beside this file; build.sh makes it), PEARLC4_GATE_TILES (default 2)

The prover is dev.py's `Pipeline` on BF16 A and B: per call, A's row hashes, A's forming with E_A's lines, A′F_B, the chain
with the peel and the fused digests, the tile leaves; per (weight, epoch), `weight_side`.  Flag roles: ``hash`` (all of it),
``nohash`` (no hashing: no row hashes, E_A's lines primed once, no digests or leaves) and ``plain``
(the forming and the chain alone: y from C̃).  An m that isn't a multiple of 64 is padded with zero filler rows (the scheme's
tile is 64 x 64): the call first copies A into the padded buffer, and the padded rows are computed, hashed and not credited.

The gate: every output byte-equal to a second pipeline launched on the same operands into its own buffers (``all``), and
PEARLC4_GATE_TILES whole 64 x 64 tiles (the first and seeded random ones) against the CPU twin (`twin.tile`, itself held to
the scalar reference): y, P_A's and P_B's rows, and with hashing the tile's leaf (hence its 128 C̃‖U digests) and its
rows' hashes.  `known_bad` is the plain mode under the hash flags.  Pearl's lottery tickets are not computed (no PoUW check
reads them; the coordinator's 06:35Z ruling for pearl-c-sm120, taken for Pearl-C4).

Shortcuts: seeds are host labels of the operand set, not derived on the device from A's and B's commitments; A's row hashes
stand in for its commitment (pearl_c's model); γ is unstated (the certificate's tt-out-fp4 conjecture is pending).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
from arm import Buffer, Gate, Shape, Work

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (HERE, ROOT / "packages" / "verity" / "src", ROOT / "protocols" / "pouw"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import dev as D  # noqa: E402

TILE, SEG = 64, D.SEGMENT
OUTPUTS = {"hash": ("y", "digests", "leaves"), "nohash": ("y",), "plain": ("y",)}
LINE_IN = 37                           # seed ‖ role ‖ index u32


def _ref():
    import twin as T
    from verity.commitments.turboshake import turboshake128
    from verity_pouw.schemes import pearl_c4 as F
    return T, F, turboshake128


def mode_of(flags: frozenset[str]) -> str:
    return "hash" if "hash" in flags else "plain" if "plain" in flags else "nohash"


class _Call:
    def __init__(self, arm: "_PearlC4", shape: Shape, flags: frozenset[str], operands, mode: str):
        self.arm, self.d, self.shape, self.flags, self.mode = arm, arm.d, shape, flags, mode
        self.m_pad = -(-shape.m // TILE) * TILE
        self.a_src, self.a_bytes = operands.a.ptr, shape.m * shape.k * 2
        self.seed_a, self.seed_b = arm.seeds(shape, operands)
        self.pl = self._pipeline(operands)
        self.pl.weight_side()
        self.pl.prime()
        self.pl.steps(mode)
        self.d.sync()
        self.outputs = {n: Buffer(self.pl.buf[n], self.pl.sizes[n]) for n in OUTPUTS[mode_of(flags)]}

    def _pipeline(self, operands, store_cu: bool = False) -> D.Pipeline:
        ext = {"xb": operands.b.ptr}
        if self.m_pad == self.shape.m:
            ext["xa"] = operands.a.ptr
        pl = D.Pipeline(self.d, self.arm.fmt, self.m_pad, self.shape.n, self.shape.k, store_cu=store_cu, external=ext)
        for name, v in (("seed_a", self.seed_a), ("seed_b", self.seed_b)):
            self.d.write(pl.buf[name], v)
        return pl

    def pad(self, pl: D.Pipeline, stream: int) -> None:
        if self.m_pad != self.shape.m:
            self.d.copy(pl.buf["xa"], self.a_src, self.a_bytes, stream)

    def launch(self, stream: int) -> None:
        self.pad(self.pl, stream)
        self.pl.launch(stream, self.mode)

    def weight_side(self, stream: int) -> None:
        self.pl.weight_side(stream)

    def free(self) -> None:
        self.pl.free()


class _PearlC4:
    name = "pearl-c-nvfp4-v0"
    fmt = precision = "nvfp4"
    stream = "fp4"
    inputs = "bf16"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset(), "plain": frozenset({"plain"})}

    def __init__(self):
        self.d: D.Dev | None = None
        self._jobs: dict = {}

    def build(self, ctx: dict) -> dict:
        cubin = Path(os.environ.get("PEARLC4_CUBIN", HERE / "pearl_c4.cubin"))
        data = cubin.read_bytes()
        self.d = D.Dev(data)
        want = getattr(ctx.get("device"), "info", {}).get("uuid")
        if want and self.d.uuid != want:
            raise RuntimeError(f"pearl_c4's device {self.d.uuid} is not the harness's {want}")
        return {"cubin": str(cubin), "cubin_sha256": hashlib.sha256(data).hexdigest(), "identity": self.d.identity,
                "scheme": self.name, "gate_tiles": self.gate_tiles()}

    @staticmethod
    def gate_tiles() -> int:
        return int(os.environ.get("PEARLC4_GATE_TILES", "2"))

    def domain(self, shape: Shape) -> str | None:
        if shape.m % 16:
            return "m a multiple of 16 (padded to the 64-row tile)"
        return D.domain(-(-shape.m // TILE) * TILE, shape.n, shape.k)

    def work(self, shape: Shape) -> Work:
        _, F, _ = _ref()
        from verity_pouw.protocol import Shape as PShape
        m, k, n = shape.m, shape.k, shape.n
        mp = -(-m // TILE) * TILE
        s, ps = F.PearlC4(self.fmt), PShape(m, k, n)
        chain = {self.fmt: mp * k * n + mp * k * F.R}
        chain["nvfp4"] = chain.get("nvfp4", 0) + mp * k * 2 * F.R
        return Work(tensor_macs=chain | {"bf16": mp * n * F.PEEL_K}, w_ref=str(s.wref(ps)), credited=str(s.credit_of(ps)),
                    unit="fp4-unit (pearl_c4.wref: 1 per FP4 MAC)",
                    hashed_bytes={"c_u_digests": mp * n * 8, "a_rows": mp * k * 2, "e_a_lines": mp * LINE_IN,
                                  "tile_leaves": mp * n},
                    weight_hashed_bytes={"b_rows": n * k * 2, "e_b_lines": n * LINE_IN, "f_lines": 2 * k * LINE_IN},
                    gamma={"status": "unstated: the certificate's tt-out-fp4 conjecture is pending (fp4-design.md, Q1)"})

    def seeds(self, shape: Shape, operands) -> tuple[bytes, bytes]:
        lab = lambda s: hashlib.sha256(f"verity/pouw/pearl-c4-arm/{s}/{shape.label}".encode()).digest()  # noqa: E731
        return lab(f"seed-a/{operands.seed}/{operands.index}"), lab(f"seed-b/{operands.weight_id}")

    def prepare(self, dev, shape: Shape, flags: frozenset[str], operands) -> _Call:
        return _Call(self, shape, flags, operands, mode_of(flags))

    def known_bad(self, dev, shape: Shape, operands) -> _Call:
        return _Call(self, shape, self.flag_sets["hash"], operands, "plain")

    def _job(self, seed_a: bytes, seed_b: bytes, k: int):
        """twin.Job; F_A, F_B and G depend on seed_b and k alone, so they are kept for later gates."""
        T, F, _ = _ref()
        key = (seed_b, k)
        if key not in self._jobs:
            self._jobs[key] = T.job(F.FORMATS[self.fmt], seed_a, seed_b, k)
        return dataclasses.replace(self._jobs[key], seed_a=seed_a)

    def gate(self, dev, call: _Call, shape: Shape, flags: frozenset[str]) -> Gate:
        T, _, ts = _ref()
        d, mode, m, n, k = self.d, mode_of(flags), call.m_pad, shape.n, shape.k
        twin = call._pipeline(_Ops(call), store_cu=False)
        try:
            call.pad(twin, 0)
            twin.weight_side()
            twin.prime()
            twin.launch(0, mode)
            d.sync()
            relaunch = {nm: d.read(twin.buf[nm], b.nbytes) == d.read(b.ptr, b.nbytes) for nm, b in call.outputs.items()}
        finally:
            twin.free()
        xa = call.pl.buf["xa"]
        y = np.frombuffer(d.read(call.pl.buf["y"], call.pl.sizes["y"]), "<u2").reshape(m, n)
        jb = self._job(call.seed_a, call.seed_b, k)
        rng = random.Random(f"{shape.label}/{call.seed_a.hex()}")
        spots = [(0, 0)] + [(rng.randrange(m // TILE), rng.randrange(n // TILE)) for _ in range(self.gate_tiles() - 1)]
        tiles = []
        for ti, tj in spots:
            rows, cols = np.arange(TILE * ti, TILE * ti + TILE), np.arange(TILE * tj, TILE * tj + TILE)
            a_rows = np.frombuffer(d.read(xa + TILE * ti * k * 2, TILE * k * 2), "<u2").reshape(TILE, k)
            b_rows = np.frombuffer(d.read(call.pl.buf["xb"] + TILE * tj * k * 2, TILE * k * 2), "<u2").reshape(TILE, k)
            t = T.tile(jb, a_rows, b_rows, rows, cols)
            tiles.append(self._compare(call, t, ti, tj, a_rows, y, mode, ts, T))
        ok = all(relaunch.values()) and all(all(v for key, v in t.items() if key != "tile") for t in tiles)
        checked = {nm: "all" for nm in call.outputs}
        return Gate(ok, checked, {"all": "byte-equal to a second pipeline on the same operands", "relaunch": relaunch,
                                  "sampled": f"{len(tiles)} whole 64 x 64 tiles against the CPU twin", "tiles": tiles,
                                  "mode": call.mode})

    def _compare(self, call: _Call, t, ti: int, tj: int, a_rows: np.ndarray, y: np.ndarray, mode: str, ts, T) -> dict:
        d, n, pl = self.d, call.shape.n, call.pl
        inv_a, inv_b = (T.u32(np.float32(1) / T.f32(s.alpha)) for s in (t.a, t.b))
        src = t.c if mode == "plain" else t.u
        want_y = T.rne_bf16(T.u32((T.f32(src) * T.f32(inv_a)[:, None]) * T.f32(inv_b)[None, :]))
        out = {"tile": [ti, tj], "y": bool(np.array_equal(y[TILE * ti:TILE * ti + TILE, TILE * tj:TILE * tj + TILE], want_y))}
        if mode == "plain":
            return out
        pa = np.frombuffer(d.read(pl.buf["pa"] + TILE * ti * 128, TILE * 128), "<u2").reshape(TILE, 64)
        pb = np.frombuffer(d.read(pl.buf["pb"] + TILE * tj * 128, TILE * 128), "<u2").reshape(TILE, 64)
        out |= {"pa": bool(np.array_equal(pa, t.pa)), "pb": bool(np.array_equal(pb, t.pb))}
        if mode != "hash":
            return out
        leaf = d.read(pl.buf["leaves"] + (ti * (n // TILE) + tj) * 32, 32)
        k = call.shape.k
        segs = 2 * k // SEG
        rows_ok = d.read(pl.buf["rows_a"] + TILE * ti * segs * 32, segs * 32) == b"".join(
            ts(a_rows[0].astype("<u2").tobytes()[o:o + SEG], 0x23, 32) for o in range(0, 2 * k, SEG))
        return out | {"leaf": leaf == t.leaf, "rows_a_first_row": rows_ok}


class _Ops:
    """The operand pointers of a call, for its gate's second pipeline."""

    def __init__(self, call: _Call):
        self.a = Buffer(call.a_src, call.a_bytes)
        self.b = Buffer(call.pl.buf["xb"], call.shape.n * call.shape.k * 2)


class PearlC4Nv(_PearlC4):
    name = "pearl-c-nvfp4-v0"
    fmt = precision = "nvfp4"


class PearlC4Mx(_PearlC4):
    name = "pearl-c-mxfp4-v0"
    fmt = precision = "mxfp4"
