"""Pearl-C4's arms for the PoUW harness (benchmarks/pouw/harness, arm interface v0.5): `PearlC4Nv` (``pearl-c-nvfp4-v0``,
the FP4 headline, divided by the ``nvfp4`` baseline) and `PearlC4Mx` (``pearl-c-mxfp4-v0``, beside it, by ``mxfp4``; a
contrast, not a protocol candidate).

  bench.py ... --arm-path benchmarks/pouw/pearl_c4 --arm pearl_c4_arm:PearlC4Nv --arm pearl_c4_arm:PearlC4Mx
  env: PEARLC4_CUBIN (default pearl_c4.cubin beside this file; build.sh makes it), PEARLC4_GATE_TILES (default 2),
  PEARLC4_VERIFY_WORKERS (the reference replay's processes, default 16)

The prover is dev.py's `Pipeline` on BF16 A and B: per call, A's commitment as the verifier checks it (`audit.commit_rows`,
SHA-256 whole-row leaves) and seed_a from its root, A's forming with E_A's lines, A′F_B, the chain with the peel and the
fused digests, the tile leaves; per (weight, epoch), `weight_side` (B's commitment and seed_b first).  The host supplies the
operands, the salt and the two domain ids (verity.ml.commitments', public before A exists).  The salt is the verifier's
(`audit.Epoch.start` over B's root, committed on the host, and an epoch beacon that is a label of the operand set: a
shortcut for a public round).  The harness makes each shape's transcript after the timed items (`bench.poison_dump`):
it fills every buffer of the hash call's ``transcript_buffers`` with 0xA5, runs the weight side and the call, and has
`dump` read them back into ``transcripts/<arm>/<shape>/``; then it poisons them again, launches nothing, and has `dump`
write the no-write control into ``negative/<arm>/<shape>/``.  The build record names the reference replay (``verifier``)
that proves and checks both after timing: `benchmarks.pouw.pearl_c4.replay`'s ``prove`` and ``verify``, whose ACCEPT
line names the transcript's sha256, and which must reject the control.  That ACCEPT is replay-verified (development
shortcut; the protocol's check is a ZK proof, PROTOCOL.md).  The build record's ``binaries`` is the cubin
(loaded with ``cuModuleLoadData``, so never mapped).
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
reads them; the coordinator's 06:35Z ruling for pearl-c-sm120, taken for Pearl-C4).  Each gate reads its tiles' inputs
(both seeds and the tiles' 64 A and B rows) from the device and reuses a twin tile made from the same ones, so a shape's
roles, their flip controls and `known_bad` share its tiles; A's and B's host commitments are likewise recomputed only
when the rows differ.  `prefetch` (arm interface v0.6, optional) starts the tiles in the harness's `refs.pool()` before
a flag set's first gate.

Shortcut: γ is unstated (the certificate's tt-out-fp4 conjecture is pending).
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
from arm import Buffer, Gate, Shape, Work

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (HERE, ROOT, ROOT / "verity" / "core" / "protocols" / "pouw", ROOT / "tools" / "config"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import dev as D  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)

TILE, WEIGHT_BASE, POISON = 64, 1, 0xA5
OUTPUTS = {"hash": ("y", "digests", "leaves"), "nohash": ("y",), "plain": ("y",)}
TRANSCRIPT_WEIGHT = ("root_b", "b_scales")
LINE_IN = 37                           # seed ‖ role ‖ index u32


def _ref():
    import twin as T
    from verity.core.protocols.pouw import audit
    from verity.core.protocols.pouw.schemes import pearl_c4 as F
    return T, F, audit


_JOBS: dict = {}


def _twin_tile(fmt: str, seed_a: bytes, seed_b: bytes, k: int, ti: int, tj: int, ra: bytes, rb: bytes):
    """The CPU twin's 64 x 64 tile (ti, tj) of A rows `ra` and B rows `rb` (BF16 bytes) under the call's seeds.  Host only,
    so a forked worker of the harness's pool (`refs.pool()`) may run it.  twin.Job's F_A, F_B and G depend on seed_b and
    k alone, so each process keeps them."""
    T, F, _ = _ref()
    if (fmt, seed_b, k) not in _JOBS:
        _JOBS[fmt, seed_b, k] = T.job(F.FORMATS[fmt], seed_a, seed_b, k)
    jb = dataclasses.replace(_JOBS[fmt, seed_b, k], seed_a=seed_a)
    a, b = (np.frombuffer(r, "<u2").reshape(TILE, k) for r in (ra, rb))
    return T.tile(jb, a, b, np.arange(TILE * ti, TILE * ti + TILE), np.arange(TILE * tj, TILE * tj + TILE))


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
        self.transcript_buffers = self._transcript_buffers()
        self.phases = {part: (lambda stream, part=part: self._part(part, stream)) for part in ("before", "gemm", "after")}

    def _transcript_buffers(self) -> dict[str, Buffer]:
        """What the harness poisons before the call it dumps (arm interface v0.5): every buffer the hash call writes
        (`dev.WRITTEN`, root_a and seed_a included, which `prime` wrote before timing), the weight side's that `write_dump`
        reads (root_b, b_scales), and with m padded, A's real rows in the padded buffer (the launch's copy; the filler rows
        stay zero).  The operands (xb, and xa when m isn't padded) are the harness's and never poisoned."""
        pl = self.pl
        out = {nm: Buffer(pl.buf[nm], pl.sizes[nm]) for nm in D.WRITTEN["hash"] + TRANSCRIPT_WEIGHT if nm in pl.buf}
        if self.m_pad != self.shape.m:
            out["xa"] = Buffer(pl.buf["xa"], self.a_bytes)
        return out

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
        self.pl.free()

    def dump(self, path: Path, write: bool = True) -> None:
        """What the harness does with this call (`bench.poison_dump`), for arm_smoke.py, which runs without it: every
        buffer of `transcript_buffers` poisoned with 0xA5, the weight side and the call launched once more on its operands
        as they stand, then `write_dump`.  write=False leaves both launches out: the no-write negative control, which the
        reference replay must reject (the panel rule, 11:50Z)."""
        for b in self.transcript_buffers.values():
            self.d.write(b.ptr, bytes([POISON]) * b.nbytes)
        if write:
            self.weight_side(0)
            self.launch(0)
        self.d.sync()
        self.write_dump(path)

    def write_dump(self, path: Path) -> None:
        """The call's transcript as the timed code emits it (`pearl_c4_replay.write_dump`): A's and B's rows, both roots,
        the tile leaves and the formed scale bytes, read from the device as they stand.  It computes nothing."""
        from benchmarks.pouw.pearl_c4 import replay as R
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
        self._epochs: dict = {}
        self._refs: dict = {}
        self._commits: dict = {}
        self._commits_shape: str | None = None

    def dump(self, dev, call: _Call, shape: Shape, out: Path) -> None:
        """The harness's hook (arm interface v0.5, `bench.poison_dump`): the call's transcript into `out`, read from the
        device after the harness poisoned `call.transcript_buffers` and ran (or, for the no-write control, didn't run)
        the weight side and the call."""
        call.write_dump(out)

    def build(self, ctx: dict) -> dict:
        cubin = Path(settings.PEARLC4_CUBIN or HERE / "pearl_c4.cubin")
        data = cubin.read_bytes()
        self.d = D.Dev(data)
        want = getattr(ctx.get("device"), "info", {}).get("uuid")
        if want and self.d.uuid != want:
            raise RuntimeError(f"pearl_c4's device {self.d.uuid} is not the harness's {want}")
        return {"cubin": str(cubin), "cubin_sha256": hashlib.sha256(data).hexdigest(), "identity": self.d.identity,
                "scheme": self.name, "gate_tiles": self.gate_tiles(), "binaries": [str(cubin)],
                "verifier": self.verifier()}

    @staticmethod
    def verifier() -> dict:
        """The reference replay on each shape's dump (arm interface v0.5's ``verifier``): ``prove`` makes the transcript
        from the dump in ``{dir}``, ``verify`` checks it and prints the ACCEPT line naming its sha256, replay-verified
        (development shortcut; the protocol's check is a ZK proof, PROTOCOL.md).  The harness's dump is under
        ``transcripts/{arm}/{shape}/``, so verify.py runs the same commands on the no-write control under ``negative/``,
        which they must reject; ``--path`` names whichever of the two it is."""
        r = ["{python}", "-m", "benchmarks.pouw.pearl_c4.replay"]
        w = ["--workers", str(settings.PEARLC4_VERIFY_WORKERS)]
        return {"name": "benchmarks.pouw.pearl_c4.replay", "transcript": "transcripts/{arm}/{shape}/transcript.json",
                "commands": [r + ["prove", "{dir}", "--run", "{run_id}", "--out", "{transcript}"] + w,
                             r + ["verify", "{transcript}", "--out", "{dir}/verify.json",
                                  "--path", "{run_id}:{transcript}", "--commit", "{commit}"] + w]}

    @staticmethod
    def gate_tiles() -> int:
        return settings.PEARLC4_GATE_TILES

    def domain(self, shape: Shape) -> str | None:
        if shape.m % 16:
            return "m a multiple of 16 (padded to the 64-row tile)"
        return D.domain(-(-shape.m // TILE) * TILE, shape.n, shape.k)

    def work(self, shape: Shape) -> Work:
        _, F, _ = _ref()
        from verity.core.protocols.pouw.protocol import Shape as PShape
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
            from verity.core.protocols.pouw.protocol import Matmul
            from verity.core.protocols.pouw.protocol import Shape as PShape
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

    def _tile_inputs(self, call: _Call, shape: Shape) -> list[tuple[tuple, bytes, bytes]]:
        """The gate's twin tiles (the first and PEARLC4_GATE_TILES − 1 seeded random ones), each as its key and its A and
        B rows, read from the device as it stands.  The key holds every input of `_twin_tile`, the rows by their sha256."""
        d, k = self.d, shape.k
        seed_a, seed_b = (d.read(call.pl.buf[nm], 32) for nm in ("seed_a", "seed_b"))
        rng = random.Random(f"{shape.label}/{seed_a.hex()}")
        spots = [(0, 0)] + [(rng.randrange(call.m_pad // TILE), rng.randrange(shape.n // TILE))
                            for _ in range(self.gate_tiles() - 1)]
        out = []
        for ti, tj in spots:
            ra = d.read(call.pl.buf["xa"] + TILE * ti * k * 2, TILE * k * 2)
            rb = d.read(call.pl.buf["xb"] + TILE * tj * k * 2, TILE * k * 2)
            out.append(((self.fmt, seed_a, seed_b, k, ti, tj, hashlib.sha256(ra).digest(), hashlib.sha256(rb).digest()),
                        ra, rb))
        return out

    def prefetch(self, dev, call: _Call, shape: Shape, flags: frozenset[str]) -> None:
        """Arm interface v0.6's hook: the gate's twin tiles for `call`, started in the harness's pool (`refs.pool()`) from
        the device as it stands.  A tile already made or started from the same inputs is not started again."""
        try:
            import refs
        except ImportError:
            return
        pool = refs.pool()
        if pool is None:
            return
        for key, ra, rb in self._tile_inputs(call, shape):
            if key not in self._refs:
                self._refs[key] = pool.submit(_twin_tile, *key[:6], ra, rb)

    def _twin(self, key: tuple, ra: bytes, rb: bytes):
        """The twin tile of these inputs: made once per shape, so a role's gates, its flip control and `known_bad` share
        it, or taken from `prefetch`."""
        got = self._refs.get(key)
        if got is None:
            got = _twin_tile(*key[:6], ra, rb)
        elif hasattr(got, "result"):
            got = got.result()
        self._refs[key] = got
        return got

    def _commit_rows(self, shape: Shape, binding: bytes, index: int, ptr: int, count: int, k: int):
        """`audit.commit_rows` over `count` rows at `ptr`, made once per shape for each binding and its rows' bytes: the
        harness gates each role's first and last operand sets in turn, whose A and B have bindings of their own."""
        _, _, au = _ref()
        raw = self.d.read(ptr, count * 2 * k)
        if self._commits_shape != shape.label:
            self._commits, self._commits_shape = {}, shape.label
        got = self._commits.get((binding, index, count, k))
        if got is not None and got[0] == raw:
            return got[1]
        tree = au.commit_rows(binding, index, [raw[i * 2 * k:(i + 1) * 2 * k] for i in range(count)])
        self._commits[binding, index, count, k] = (raw, tree)
        return tree

    def commitments(self, call: _Call, shape: Shape, mode: str) -> dict:
        """The device's roots and seeds against `audit.commit_rows` over the rows and the scheme's `noise` on the host.  The
        roles other than ``hash`` commit nothing per call, so there only B's are checked (A's are those `prepare` primed,
        of the operands before a dependent chain rewrote them)."""
        _, F, au = _ref()
        from verity.core.protocols.pouw.protocol import Context, Matmul
        from verity.core.protocols.pouw.protocol import Shape as PShape
        d, pl, k = self.d, call.pl, shape.k
        ta = self._commit_rows(shape, au.activation_binding(call.salt, 0), 0, pl.buf["xa"], call.m_pad, k)
        tb = self._commit_rows(shape, au.weight_binding(call.weight), -2, pl.buf["xb"], shape.n, k)
        ctx = Context(call.salt, Matmul(0, PShape(call.m_pad, k, shape.n), call.weight), ta.commitment.root,
                      tb.commitment.root)
        noise = F.PearlC4(self.fmt).noise(ctx)
        names = ("root_a", "root_b", "seed_a", "seed_b") if mode == "hash" else ("root_b", "seed_b")
        got = {nm: d.read(pl.buf[nm], 32) for nm in ("root_a", "root_b", "seed_a", "seed_b")}
        want = {"root_a": ctx.root_a, "root_b": ctx.root_b, "seed_a": noise.seed_a, "seed_b": noise.seed_b}
        return {nm: got[nm] == want[nm] for nm in names} | {"root_a_hex": got["root_a"].hex()}

    def gate(self, dev, call: _Call, shape: Shape, flags: frozenset[str]) -> Gate:
        T, _, _ = _ref()
        d, mode, m, n = self.d, mode_of(flags), call.m_pad, shape.n
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
        y = np.frombuffer(d.read(call.pl.buf["y"], call.pl.sizes["y"]), "<u2").reshape(m, n)
        tiles = [self._compare(call, self._twin(key, ra, rb), key[4], key[5], y, mode, T)
                 for key, ra, rb in self._tile_inputs(call, shape)]
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
