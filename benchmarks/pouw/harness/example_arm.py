"""The harness's example arm, **not a PoUW scheme**: cuBLASLt's first heuristic FP8 E4M3 GEMM (per-tensor scales 1, BF16 out)
as the prover, and one read of A before it and of D after it as stand-ins for committing A and hashing y (its `phases`).
It exists to smoke-test the arm path on the card (prepare, gates, the negative controls, graph capture, the weight side, the
dependent chain's split and side stream, the report) and as a template for the streams' arms; its numbers are labelled
"example arm" and go on no panel.

  python bench.py ... --arm example_arm:ExampleFp8

Its gate: D equals, byte for byte, a second launch of the same algorithm into its own buffer (`all`), and 32 x 32 sampled
entries are within 2^-7 relative (plus an FP32 accumulation allowance) of numpy's float64 product (`sampled`).
`known_bad` is the same GEMM over half of k. Its `dump` writes D's sha256 (``manifest.json``), so the harness's poisoned
dump and no-write control run on the card too.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import baselines as B
import fill as F
import numpy as np
from arm import Buffer, GateResult, Shape, Work

MAX_D_BYTES = 1 << 29  # the gate reads D and its twin to the host


class _Call:
    def __init__(self, dev, cand, honest, shape: Shape, ops: B.Operands, twin_ops: B.Operands, ws: Buffer, read_d: bool,
                 owned: list[Buffer]):
        self.dev, self.shape, self.read_d, self._owned = dev, shape, read_d, owned
        self.gemm = B.BaselineCall(dev, cand, shape, ops, ws)
        self.twin = B.BaselineCall(dev, honest, shape, twin_ops, ws)
        self.a, self.b = ops.a, ops.b
        self.outputs = {"d": ops.d}
        self.twin_d = twin_ops.d
        self.phases = {"before": self._before, "gemm": self.gemm.launch, "after": self._after}

    def _before(self, stream: int) -> None:
        if self.read_d:
            self.dev.read_bw(self.a, 1, stream)

    def _after(self, stream: int) -> None:
        if self.read_d:
            self.dev.read_bw(self.outputs["d"], 1, stream)

    def launch(self, stream: int) -> None:
        for p in ("before", "gemm", "after"):
            self.phases[p](stream)

    def weight_side(self, stream: int) -> None:
        self.dev.read_bw(self.b, 1, stream)

    def free(self) -> None:
        self.gemm.free()
        self.twin.free()
        for buf in self._owned:
            self.dev.free(buf)
        self._owned.clear()


class ExampleFp8:
    name = "example-plain-fp8"
    stream = "fp8"
    precision = "fp8-e4m3"
    inputs = "e4m3"
    flag_sets = {"hash": frozenset({"read-d"}), "nohash": frozenset()}

    def __init__(self):
        self._ws = None
        self._cands: dict = {}

    def build(self, ctx: dict) -> dict:
        return {"kernels": "cuBLASLt (the harness's libpouw_harness.so)", "not_a_pouw_scheme": True, "binaries": []}

    def domain(self, shape: Shape) -> str | None:
        return None if shape.m * shape.n * 2 <= MAX_D_BYTES else f"D is over {MAX_D_BYTES} bytes (the gate reads it)"

    def work(self, shape: Shape) -> Work:
        return Work(tensor_macs={"fp8-e4m3": shape.mkn}, w_ref=str(shape.mkn), credited=str(shape.mkn), unit="mac",
                    hashed_bytes={"stand-in-read-a": shape.m * shape.k, "stand-in-read-d": shape.m * shape.n * 2},
                    weight_hashed_bytes={"stand-in-read-b": shape.n * shape.k})

    def _ops(self, dev, shape: Shape, ops, owned: list[Buffer]) -> tuple[B.Operands, B.Operands]:
        def new(n: int) -> Buffer:
            b = dev.alloc(n)
            owned.append(b)
            return b

        sa, sb = new(4), new(4)
        dev.fill(F.F32_ONE, sa, 1, 0)
        dev.fill(F.F32_ONE, sb, 1, 0)
        d_bytes = shape.m * shape.n * 2
        return B.Operands(ops.a, ops.b, sa, sb, new(d_bytes)), B.Operands(ops.a, ops.b, sa, sb, new(d_bytes))

    def _cand(self, dev, shape: Shape, ops: B.Operands, half_k: bool = False):
        key = (shape, half_k)
        if key not in self._cands:
            fam = B.FAMILIES["fp8-e4m3"]
            if half_k:
                got = B.lt_candidates(dev, fam, Shape(shape.label + "/half-k", shape.m, shape.k // 2, shape.n), ops, 1,
                                      ld=shape.k)
            else:
                got = B.lt_candidates(dev, fam, shape, ops, 1)
            if not got:
                raise RuntimeError(f"cuBLASLt has no FP8 algorithm at {shape}")
            self._cands[key] = got[0]
        return self._cands[key]

    def prepare(self, dev, shape: Shape, flags: frozenset[str], operands, half_k: bool = False) -> _Call:
        if self._ws is None:
            self._ws = dev.alloc(B.LT_WORKSPACE)
        owned: list[Buffer] = []
        ops, twin = self._ops(dev, shape, operands, owned)
        dev.sync()
        cand, honest = self._cand(dev, shape, ops, half_k), self._cand(dev, shape, ops)
        return _Call(dev, cand, honest, shape, ops, twin, self._ws, "read-d" in flags, owned)

    def known_bad(self, dev, shape: Shape, operands) -> _Call:
        return self.prepare(dev, shape, self.flag_sets["hash"], operands, half_k=True)

    def dump(self, dev, call: _Call, shape: Shape, out: Path) -> None:
        d = call.outputs["d"]
        (out / "manifest.json").write_text(json.dumps({"arm": self.name, "shape": shape.label,
                                                       "d_sha256": hashlib.sha256(dev.read(d.ptr, d.nbytes)).hexdigest()}))

    def gate(self, dev, call: _Call, shape: Shape, flags: frozenset[str]) -> GateResult:
        call.twin.launch(dev.stream)
        dev.sync()
        d = np.frombuffer(dev.read(call.outputs["d"].ptr, call.outputs["d"].nbytes), np.uint16)
        twin = np.frombuffer(dev.read(call.twin_d.ptr, call.twin_d.nbytes), np.uint16)
        same = bool(np.array_equal(d, twin))
        rng = random.Random(shape.mkn)
        rows, cols = B.sample_indices(shape.m, B.SAMPLE_ROWS, rng), B.sample_indices(shape.n, B.SAMPLE_COLS, rng)
        a = F.decode(F.E4M3, dev.read_rows(call.a.ptr, shape.k, shape.k, rows)).reshape(len(rows), shape.k)
        b = F.decode(F.E4M3, dev.read_rows(call.b.ptr, shape.k, shape.k, cols)).reshape(len(cols), shape.k)
        exact = a @ b.T
        allowance = np.abs(a) @ np.abs(b).T * 2.0 ** -20
        got = F.decode(F.BF16, d.reshape(shape.m, shape.n)[np.ix_(rows, cols)].tobytes()).reshape(len(rows), len(cols))
        err = np.abs(got - exact)
        close = bool(np.all(err <= 2.0 ** -7 * np.abs(exact) + allowance))
        return GateResult(same and close, {"d": "all"},
                          {"d_equals_relaunch": same, "sampled_within_tolerance": close,
                           "worst_relative": float(np.max(err / (np.abs(exact) + allowance + 1e-30)))})
