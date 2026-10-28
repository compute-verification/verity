"""U's pinned vectors (`verity.ml.protocols.pouw.schemes.pearl_c_u`): ``uv run python verity/core/protocols/pouw/tests/pearl_c_u_vectors.py
[GATE_CHECK_DIR]`` rewrites ``tests/vectors/pearl_c_u.json``.

The cells come from the H100 kernel's gate problem, the one archive/pouw-blake3/pearl_c/fixture.py builds: 256 x 384 x 1,024,
case "gpu-check", unit index 5, weight 1, both formings.  Its rows `ROWS` and columns `COLS` straddle the GEMM's 128 x 128
tiles and the leaves' 64 x 64 ones.  Every value is computed by the pin and by `pearl_c`, which must agree.  Given the gate's
check directory (build.sh's OUT/check), the cells must also equal the gate's expected buffers, whose SHA-256 are recorded.
The `edge` K16 vectors are the pin's own on constructed operands (cancellation, subnormals, exponent gaps, overflow),
checked against `verity.ml.circuits.silicon`'s HOPPER_BF16_WGMMA_K16.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import struct
import sys
from pathlib import Path

import pearl_c_vectors as CV
from verity.ml.circuits.silicon.models import HOPPER_BF16_WGMMA_K16
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_u as U

OUT = Path(__file__).with_name("vectors") / "pearl_c_u.json"
CASE, INDEX, WEIGHT, SHAPE = "gpu-check", 5, 1, (256, 384, 1024)
ROWS, COLS = (0, 63, 64, 127, 128, 255), (0, 127, 128, 255, 256, 383)
TRACED = ((0, 0), (127, 256), (255, 383))


def context() -> Context:
    m, n, k = SHAPE
    h = lambda s: hashlib.sha256(f"{s}/{CASE}".encode()).digest()  # noqa: E731
    return Context(h("salt"), Matmul(INDEX, Shape(m, k, n), WEIGHT), h("root_a"), h("root_b"))


def column(basis, p: int) -> list[int]:
    return [line[p] for line in basis]


def tiled(row: int, k: int) -> int:
    """pearl_c.cu's tiled() for a 128-byte-row operand in blocks of 128 rows."""
    return ((row // 128) * 4 + k // 32) * 4096 + ((row % 128) >> 3) * 256 + ((k & 31) >> 4) * 128 + (row & 7) * 16 + (k & 15)


class Gate:
    """The gate's expected buffers for one forming, read-only."""

    def __init__(self, root: Path, forming: str, manifest: dict):
        self.root, self.forming, self.files = root, forming, manifest["formings"][forming]["files"]
        self.raw = {}
        self.hashes = {}
        for name in ("c", "u", "af", "bf", "g", "pa_t", "pb_t"):
            data = (root / forming / self.files[name]["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == self.files[name]["sha256"], name
            self.raw[name] = data
            self.hashes[f"{forming}/{self.files[name]['file']}"] = self.files[name]["sha256"]

    def word(self, name: str, index: int) -> int:
        return struct.unpack_from("<I", self.raw[name], 4 * index)[0]

    def half(self, name: str, row: int, e: int) -> int:
        return struct.unpack_from("<H", self.raw[name], tiled(row, 2 * e))[0]


def forming_vectors(forming: str, gate: Gate | None) -> tuple[dict, list]:
    m, n, k = SHAPE
    s, ctx = C.PearlC(forming=forming), context()
    noise, norm = s.noise(ctx), s.line_norm
    make = CV.rows if forming == "v0" else CV.rows_f32
    ra, rb = make(CASE, "A", m, k), make(CASE, "B", n, k)
    a, e_a = s._side([ra[i] for i in ROWS], ROWS, 0, noise.seed_a, noise.seed_b, k)
    b, e_b = s._side([rb[j] for j in COLS], COLS, 1, noise.seed_b, noise.seed_b, k)
    be_a, be_b = C.scaled_noise(a, e_a), C.scaled_noise(b, e_b)
    fa, fb = C._basis(noise.seed_b, 0, k, norm), C._basis(noise.seed_b, 1, k, norm)
    fa_cols, fb_cols = [column(fa, q) for q in range(U.R)], [column(fb, p) for p in range(U.R)]

    g = [[U.chain(fb_cols[p], fa_cols[q]) for q in range(U.R)] for p in range(U.R)]
    assert g == [list(r) for r in C.gram(noise.seed_b, k, norm)]
    ref_pa = C.peel_factors_a(a.codes, be_a, noise.seed_b, k, norm)
    ref_pb = C.peel_factors_b(b.codes, be_b, noise.seed_b, k, norm)
    rows_a, rows_b, pa, pb = [], [], {}, {}
    for x, i in enumerate(ROWS):
        af = [U.chain(a.codes[x], fb_cols[p]) for p in range(U.R)]
        pa[i] = U.pa_row(be_a[x], af)
        assert pa[i] == ref_pa[x]
        rows_a.append({"i": i, "be": list(be_a[x]), "af": af, "pa": pa[i]})
    for y, j in enumerate(COLS):
        bf = [U.chain(b.codes[y], fa_cols[q]) for q in range(U.R)]
        trace: dict = {}
        pb[j] = U.pb_row(be_b[y], bf, g, trace)
        assert pb[j] == ref_pb[y]
        rows_b.append({"j": j, "be": list(be_b[y]), "bf": bf, "acc": trace["acc"], "diff": trace["diff"], "pb": pb[j]})

    cells, chains, k16s = [], [], []
    for x, i in enumerate(ROWS):
        for y, j in enumerate(COLS):
            trace: dict = {}
            c = U.chain(a.codes[x], b.codes[y], trace)
            assert c == C.chain(a.codes[x], b.codes[y])
            steps: list = []
            out = U.u(c, pa[i], pb[j], steps)
            assert out == C.peel(c, pa[i], pb[j])
            cells.append({"i": i, "j": j, "c": c, "steps": steps, "u": out})
            acc = c
            for s_, w in zip(range(0, U.PEEL_K, 16), steps):
                k16s.append([acc, pa[i][s_:s_ + 16], pb[j][s_:s_ + 16], w])
                acc = w
            if (i, j) in TRACED:
                chains.append({"i": i, "j": j, "a": bytes(a.codes[x]).hex(), "b": bytes(b.codes[y]).hex(),
                               "atoms": trace["atoms"], "totals": trace["totals"], "c": c})
            if gate:
                assert gate.word("c", i * n + j) == c and gate.word("u", i * n + j) == out, (forming, i, j)
    if gate:
        assert all(gate.word("g", p * U.R + q) == g[p][q] for p in range(U.R) for q in range(U.R))
        for r in rows_a:
            assert [gate.word("af", r["i"] * U.R + p) for p in range(U.R)] == r["af"]
            assert [gate.half("pa_t", r["i"], e) for e in range(U.PEEL_K)] == r["pa"]
        for r in rows_b:
            assert [gate.word("bf", r["j"] * U.R + q) for q in range(U.R)] == r["bf"]
            assert [gate.half("pb_t", r["j"], e) for e in range(U.PEEL_K)] == r["pb"]
    return {"scheme": s.name, "line_norm_f": norm, "chains": chains, "g": g, "rows_a": rows_a, "rows_b": rows_b,
            "cells": cells}, k16s


def bf16(rng: random.Random, lo: int, hi: int, sub: float = 0.0) -> int:
    """A finite BF16 word with exponent field in [lo, hi] (0 with probability `sub`: a subnormal or zero)."""
    field = 0 if rng.random() < sub else rng.randint(lo, hi)
    return rng.randrange(2) << 15 | field << 7 | rng.randrange(128)


def edge_vectors() -> list:
    rng = random.Random("pearl-c-u/edge/v1")
    out = []

    def add(label: str, acc: int, x: list[int], y: list[int]) -> None:
        w = U.k16(acc, x, y)
        assert w == HOPPER_BF16_WGMMA_K16.step(acc, x, y), label
        out.append([acc, x, y, w, label])

    one = 0x3F80
    add("zero accumulator, unit products", 0, [one] * 16, [one] * 16)
    add("negative zero accumulator, zero products", 0x80000000, [0] * 16, [0x8000] * 16)
    for _ in range(4):
        x, y = [bf16(rng, 100, 140) for _ in range(16)], [bf16(rng, 100, 140) for _ in range(16)]
        exact = sum(C.P.bf16_to_f32(p) * C.P.bf16_to_f32(q) for p, q in zip(x, y))
        add("accumulator cancelling the products", C.P.f32_bits(C.P.f32(-exact)), x, y)
    for _ in range(4):
        add("large accumulator, tiny products (the window drops them)", C.P.f32_bits(rng.uniform(1e6, 1e8)),
            [bf16(rng, 100, 110) for _ in range(16)], [bf16(rng, 100, 110) for _ in range(16)])
    for _ in range(4):
        add("tiny accumulator, large products", rng.randrange(1, 1 << 23), [bf16(rng, 125, 135) for _ in range(16)],
            [bf16(rng, 125, 135) for _ in range(16)])
    for _ in range(6):
        add("subnormal products and accumulator", rng.randrange(0, 1 << 23) | rng.randrange(2) << 31,
            [bf16(rng, 1, 70, 0.3) for _ in range(16)], [bf16(rng, 1, 60, 0.3) for _ in range(16)])
    for _ in range(4):
        add("mixed signs near cancellation", C.P.f32_bits(rng.uniform(-4, 4)), [bf16(rng, 126, 128) for _ in range(16)],
            [bf16(rng, 126, 128) for _ in range(16)])
    add("overflow to +inf", 0x7F7FFFFF, [0x7F7F] * 16, [0x3F80] * 16)
    add("overflow to -inf", 0xFF7FFFFF, [0xFF7F] * 16, [0x3F80] * 16)
    for _ in range(20):
        add("random", C.P.f32_bits(rng.uniform(-1e3, 1e3)), [bf16(rng, 110, 140, 0.05) for _ in range(16)],
            [bf16(rng, 110, 140, 0.05) for _ in range(16)])
    return out


def vectors(gate_dir: Path | None = None) -> dict:
    manifest = json.loads((gate_dir / "check.json").read_text()) if gate_dir else None
    gate_hashes, formings, gate_k16 = {}, {}, []
    for forming in ("v0", "v1"):
        gate = Gate(gate_dir, forming, manifest) if gate_dir else None
        formings[forming], k16s = forming_vectors(forming, gate)
        gate_k16 += [[acc, x, y, w, forming] for acc, x, y, w in k16s]
        if gate:
            gate_hashes.update(gate.hashes)
    return {
        "schema": "verity-pouw/pearl-c-u-vectors/v1",
        "status": U.STATUS,
        "status_detail": "equal to pearl-c-h100's CPU reference and, on these cells, to the H100 kernel's gate fixture; "
                         "silicon confirmation waits for the joint H100 request's B1 capture",
        "module": "verity/ml/protocols/pouw/schemes/pearl_c_u.py",
        "problem": {"shape": list(SHAPE), "case": CASE, "index": INDEX, "weight": WEIGHT, "rows": list(ROWS),
                    "cols": list(COLS), "inputs": "verity/core/protocols/pouw/tests/pearl_c_vectors.py rows / rows_f32, as "
                                                  "archive/pouw-blake3/pearl_c/fixture.py builds them"},
        "gate": {"check_json_sha256": hashlib.sha256((gate_dir / "check.json").read_bytes()).hexdigest(),
                 "files": gate_hashes, "agree": True} if gate_dir else None,
        "formings": formings,
        "k16": {"gate": gate_k16, "edge": edge_vectors()},
    }


def dumps(obj: dict) -> str:
    """indent=1 JSON with every array of numbers on one line."""
    return re.sub(r"\[[\d,\s-]*\]", lambda m: re.sub(r"\s+", "", m.group(0)).replace(",", ", "), json.dumps(obj, indent=1)) + "\n"


if __name__ == "__main__":
    gate_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT.parent.mkdir(exist_ok=True)
    data = dumps(vectors(gate_dir))
    OUT.write_text(data)
    print(f"wrote {OUT}: sha256 {hashlib.sha256(data.encode()).hexdigest()}")
