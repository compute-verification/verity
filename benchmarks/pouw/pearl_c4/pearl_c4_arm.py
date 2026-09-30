"""Pearl-C4's arms for the PoUW harness (benchmarks/pouw/harness, arm interface v0): `PearlC4Nv` (``pearl-c-nvfp4-v0``, the
FP4 headline, divided by the ``nvfp4`` baseline) and `PearlC4Mx` (``pearl-c-mxfp4-v0``, beside it, by ``mxfp4``).

  bench.py ... --arm-path benchmarks/pouw/pearl_c4 --arm pearl_c4_arm:PearlC4Nv --arm pearl_c4_arm:PearlC4Mx
  env: PEARLC4_CUBIN (default pearl_c4.cubin beside this file; build.sh makes it), PEARLC4_GATE_TILES (default 2)

The prover is dev.py's `Pipeline` on BF16 A and B: per call, A's commitment as the verifier checks it (`audit.commit_rows`,
SHA-256 whole-row leaves) and seed_a from its root, A's forming with E_A's lines, A′F_B, the chain with the peel and the
fused digests, the tile leaves; per (weight, epoch), `weight_side` (B's commitment and seed_b first).  The host supplies the
operands, the salt and the two domain ids (verity.commitments', public before A exists).  The salt is the verifier's
(`audit.Epoch.start` over B's root, committed on the host, and an epoch beacon that is a label of the operand set: a
shortcut for a public round).  With PEARLC4_TRANSCRIPTS=DIR, the first hash-role call freed at each shape dumps its
transcript into DIR/<scheme>-<shape> (`dump`), for `verity_pouw.schemes.pearl_c4_replay` to prove and verify after timing.
Flag roles: ``hash`` (all of it), ``nohash`` (no hashing: no commitment, E_A's lines primed once, no digests or leaves) and
``plain`` (the forming and the chain alone: y from C̃).  The scheme's domain is m a multiple of 64 (its 64 x 64 tile), so
an m that isn't is the scheme's matmul of A padded with zero filler rows: the call first copies A into the padded buffer;
the padded rows are committed, computed and hashed, and not credited.  A call's ``phases``: ``before`` (the padding copy,
A's commitment and seed_a, A's forming, A′F_B), ``gemm`` (the chain with the peel and the digests) and ``after`` (the tile
leaves; nothing without hashing).

The gate: every output byte-equal to a second pipeline launched on the same operands into its own buffers (``all``); A's
and B's roots equal to `audit.commit_rows` over the rows on the host and both seeds to the scheme's `noise` of that
context; and PEARLC4_GATE_TILES whole 64 x 64 tiles (the first and seeded random ones) against the CPU twin (`twin.tile`,
itself held to the scalar reference): y, P_A's and P_B's rows, and with hashing the tile's leaf (hence its 128 C̃‖U
digests).  `known_bad` is the plain mode under the hash flags.  Pearl's lottery tickets are not computed (no PoUW check
reads them; the coordinator's 06:35Z ruling for pearl-c-sm120, taken for Pearl-C4).

Shortcut: γ is unstated (the certificate's tt-out-fp4 conjecture is pending).
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

TILE, WEIGHT_BASE = 64, 1
OUTPUTS = {"hash": ("y", "digests", "leaves"), "nohash": ("y",), "plain": ("y",)}
LINE_IN = 37                           # seed ‖ role ‖ index u32


def _ref():
    import twin as T
    from verity_pouw import audit
    from verity_pouw.schemes import pearl_c4 as F
    return T, F, audit


def mode_of(flags: frozenset[str]) -> str:
    return "hash" if "hash" in flags else "plain" if "plain" in flags else "nohash"


class _Call:
    def __init__(self, arm: "_PearlC4", shape: Shape, flags: frozenset[str], operands, mode: str):
        self.arm, self.d, self.shape, self.flags, self.mode = arm, arm.d, shape, flags, mode
        self.m_pad = -(-shape.m // TILE) * TILE
        self.a_src, self.a_bytes = operands.a.ptr, shape.m * shape.k * 2
        self.weight = WEIGHT_BASE + operands.weight_id
        self.beacon, self.salt = arm.epoch(shape, operands, self.weight)
        self.dom_a, self.dom_b = arm.domains(shape, self.salt, self.weight)
        self.pl = self._pipeline(operands)
        self.pad(self.pl, 0)
        self.pl.weight_side()
        self.pl.prime()
        self.d.sync()
        self.outputs = {n: Buffer(self.pl.buf[n], self.pl.sizes[n]) for n in OUTPUTS[mode_of(flags)]}
        self.phases = {part: (lambda stream, part=part: self._part(part, stream)) for part in ("before", "gemm", "after")}

    def _pipeline(self, operands, store_cu: bool = False) -> D.Pipeline:
        ext = {"xb": operands.b.ptr}
        if self.m_pad == self.shape.m:
            ext["xa"] = operands.a.ptr
        pl = D.Pipeline(self.d, self.arm.fmt, self.m_pad, self.shape.n, self.shape.k, store_cu=store_cu, external=ext)
        for name, v in (("salt", self.salt), ("dom_a", self.dom_a), ("dom_b", self.dom_b)):
            self.d.write(pl.buf[name], v)
        return pl

    def pad(self, pl: D.Pipeline, stream: int) -> None:
        if self.m_pad != self.shape.m:
            self.d.copy(pl.buf["xa"], self.a_src, self.a_bytes, stream)

    def launch(self, stream: int) -> None:
        self.pad(self.pl, stream)
        self.pl.launch(stream, self.mode)

    def _part(self, part: str, stream: int) -> None:
        steps = self.pl.steps(self.mode)
        g = next(i for i, (name, _, _) in enumerate(steps) if name.startswith("gemm_"))
        if part == "before":
            self.pad(self.pl, stream)
        for name, geo, kw in {"before": steps[:g], "gemm": steps[g:g + 1], "after": steps[g + 1:]}[part]:
            self.d.launch(name, *geo, stream=stream, **kw)

    def weight_side(self, stream: int) -> None:
        self.pl.weight_side(stream)

    def free(self) -> None:
        out = os.environ.get("PEARLC4_TRANSCRIPTS")
        if out and self.mode == "hash" and self.arm.first_dump(self.shape):
            self.dump(Path(out) / f"{self.arm.name}-{self.shape.label}")
        self.pl.free()

    def dump(self, path: Path) -> None:
        """The call's transcript as the timed code emits it (`pearl_c4_replay.write_dump`): the call launched once more on
        its operands as they stand, then A's and B's rows, both roots, the tile leaves and the formed scale bytes read back."""
        from verity_pouw.schemes import pearl_c4_replay as R
        self.launch(0)
        self.d.sync()
        d, b, s = self.d, self.pl.buf, self.shape
        call = {"scheme": self.arm.name, "shape": [self.m_pad, s.k, s.n], "index": 0, "weight": self.weight,
                "real_rows": s.m, "epoch_beacon": self.beacon.hex(), "label": s.label, "device": d.uuid,
                "flags": sorted(self.flags)}
        sizes = self.pl.sizes
        R.write_dump(path, call, d.read(b["xa"], self.m_pad * s.k * 2), d.read(b["xb"], s.n * s.k * 2),
                     d.read(b["root_a"], 32), d.read(b["root_b"], 32), d.read(b["leaves"], sizes["leaves"]),
                     d.read(b["a_scales"], sizes["a_scales"]), d.read(b["b_scales"], sizes["b_scales"]))


class _PearlC4:
    name = "pearl-c-nvfp4-v0"
    fmt = precision = "nvfp4"
    stream = "fp4"
    inputs = "bf16"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset(), "plain": frozenset({"plain"})}

    def __init__(self):
        self.d: D.Dev | None = None
        self._jobs: dict = {}
        self._epochs: dict = {}
        self._dumped: set = set()

    def first_dump(self, shape: Shape) -> bool:
        """True once per shape: the first hash-role call freed there dumps its transcript."""
        new = shape.label not in self._dumped
        self._dumped.add(shape.label)
        return new

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

    def epoch(self, shape: Shape, operands, weight: int) -> tuple[bytes, bytes]:
        """(epoch beacon, salt) of an operand set: the beacon a label of the set (the benchmark's stand-in for a public
        round), the salt derived from it and B's root as the verifier derives it (`audit.Epoch.start`, with B's rows
        committed on the host)."""
        key = (shape.label, operands.index, operands.seed, operands.b.ptr, weight)
        if key not in self._epochs:
            _, _, au = _ref()
            from verity_pouw.protocol import Matmul
            from verity_pouw.protocol import Shape as PShape
            m_pad, k, n = -(-shape.m // TILE) * TILE, shape.k, shape.n
            beacon = hashlib.sha256(f"verity/pouw/pearl-c4-arm/beacon/{shape.label}/{operands.seed}/{operands.index}"
                                    .encode()).digest()
            raw = self.d.read(operands.b.ptr, n * k * 2)
            root_b = au.commit_rows(au.weight_binding(weight), -2, [raw[i * 2 * k:(i + 1) * 2 * k] for i in range(n)])
            unit = Matmul(0, PShape(m_pad, k, n), weight)
            self._epochs[key] = beacon, au.Epoch.unverified(au.Workload((unit,)), {weight: root_b.commitment.root}, beacon).salt
        return self._epochs[key]

    @staticmethod
    def domains(shape: Shape, salt: bytes, weight: int) -> tuple[bytes, bytes]:
        """The domain ids of A's commitment (matmul 0, over the padded rows) and B's (weight `weight`), as
        `audit.commit_rows` derives them."""
        _, _, au = _ref()
        return (au.rows_domain(au.activation_binding(salt, 0), 0, -(-shape.m // TILE) * TILE).domain_id,
                au.rows_domain(au.weight_binding(weight), -2, shape.n).domain_id)

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

    def commitments(self, call: _Call, shape: Shape, mode: str) -> dict:
        """The device's roots and seeds against `audit.commit_rows` over the rows and the scheme's `noise` on the host.  The
        roles other than ``hash`` commit nothing per call, so there only B's are checked (A's are those `prepare` primed,
        of the operands before a dependent chain rewrote them)."""
        _, F, au = _ref()
        from verity_pouw.protocol import Context, Matmul
        from verity_pouw.protocol import Shape as PShape
        d, pl, k = self.d, call.pl, shape.k
        def rows(ptr: int, count: int) -> list[bytes]:
            raw = d.read(ptr, count * 2 * k)
            return [raw[i * 2 * k:(i + 1) * 2 * k] for i in range(count)]

        ta = au.commit_rows(au.activation_binding(call.salt, 0), 0, rows(pl.buf["xa"], call.m_pad))
        tb = au.commit_rows(au.weight_binding(call.weight), -2, rows(pl.buf["xb"], shape.n))
        ctx = Context(call.salt, Matmul(0, PShape(call.m_pad, k, shape.n), call.weight), ta.commitment.root,
                      tb.commitment.root)
        noise = F.PearlC4(self.fmt).noise(ctx)
        names = ("root_a", "root_b", "seed_a", "seed_b") if mode == "hash" else ("root_b", "seed_b")
        got = {nm: d.read(pl.buf[nm], 32) for nm in ("root_a", "root_b", "seed_a", "seed_b")}
        want = {"root_a": ctx.root_a, "root_b": ctx.root_b, "seed_a": noise.seed_a, "seed_b": noise.seed_b}
        return {nm: got[nm] == want[nm] for nm in names} | {"root_a_hex": got["root_a"].hex()}

    def gate(self, dev, call: _Call, shape: Shape, flags: frozenset[str]) -> Gate:
        T, _, _ = _ref()
        d, mode, m, n, k = self.d, mode_of(flags), call.m_pad, shape.n, shape.k
        seed_a, seed_b = (d.read(call.pl.buf[nm], 32) for nm in ("seed_a", "seed_b"))
        twin = call._pipeline(_Ops(call), store_cu=False)
        try:
            call.pad(twin, 0)
            twin.weight_side()
            if mode == "hash":
                twin.prime()
            else:
                for nm in ("root_a", "seed_a", "lines_ea"):
                    d.copy(twin.buf[nm], call.pl.buf[nm], twin.sizes[nm], 0)
            twin.launch(0, mode)
            d.sync()
            relaunch = {nm: d.read(twin.buf[nm], b.nbytes) == d.read(b.ptr, b.nbytes) for nm, b in call.outputs.items()}
        finally:
            twin.free()
        xa = call.pl.buf["xa"]
        y = np.frombuffer(d.read(call.pl.buf["y"], call.pl.sizes["y"]), "<u2").reshape(m, n)
        jb = self._job(seed_a, seed_b, k)
        rng = random.Random(f"{shape.label}/{seed_a.hex()}")
        spots = [(0, 0)] + [(rng.randrange(m // TILE), rng.randrange(n // TILE)) for _ in range(self.gate_tiles() - 1)]
        tiles = []
        for ti, tj in spots:
            rows, cols = np.arange(TILE * ti, TILE * ti + TILE), np.arange(TILE * tj, TILE * tj + TILE)
            a_rows = np.frombuffer(d.read(xa + TILE * ti * k * 2, TILE * k * 2), "<u2").reshape(TILE, k)
            b_rows = np.frombuffer(d.read(call.pl.buf["xb"] + TILE * tj * k * 2, TILE * k * 2), "<u2").reshape(TILE, k)
            t = T.tile(jb, a_rows, b_rows, rows, cols)
            tiles.append(self._compare(call, t, ti, tj, y, mode, T))
        commit = self.commitments(call, shape, mode)
        ok = all(relaunch.values()) and all(all(v for key, v in t.items() if key != "tile") for t in tiles)
        ok = ok and all(v for key, v in commit.items() if key != "root_a_hex")
        checked = {nm: "all" for nm in call.outputs}
        return Gate(ok, checked, {"all": "byte-equal to a second pipeline on the same operands", "relaunch": relaunch,
                                  "commitments": commit, "sampled": f"{len(tiles)} whole 64 x 64 tiles against the CPU twin",
                                  "tiles": tiles, "mode": call.mode})

    def _compare(self, call: _Call, t, ti: int, tj: int, y: np.ndarray, mode: str, T) -> dict:
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
        return out | {"leaf": leaf == t.leaf}


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
