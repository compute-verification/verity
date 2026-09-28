"""The pinned lowering of a GEMM-coordinate transition unit into one Flock block-R1CS type with word-aligned IO.

One unit (``unit.py``, the binary census's exact GF(2) circuit of ``GroupSum.step``) becomes one 2^13-bit block, C = I:
row i is ``(A_i . z)(B_i . z) = z_i``. Column layout (``flock-unit-io/v1``):

~~~text
[0, 256)      x operand bytes of the unit (k words of `bits` bits, word j at bits [j*bits, (j+1)*bits), LSB first)
[256, 512)    W operand bytes, the same way
[512, 544)    accumulator in (FP32 word, LSB first)
[544, 608)    fp4-nvf4 only: the x then W UE4M3 scale bytes (4 each, LSB first); the rest of [544, 640) forced zero
[640, ...)    live ANDs in creation order, then one row per assertion (L + z_j) * 1 = z_j
pad to 128    forced-zero rows
c_out word    32 copy rows of the output FP32 word, then 96 forced-zero rows
y word        (epilogue relations only) 16 copy rows of f32_to_bf16(c_out), then 112 forced-zero rows
const         the constant-one column, last (A = B = {const}, pinned by Flock)
~~~

A forced-zero row has empty A and B. So every IO word is a 128-bit word column whose unused bits are zero, and a
Flock wire between two IO words (c_out of unit j to c_in of unit j + 1) or onto a BLAKE3 message word is a plain
equality of 128-bit words. The operand bytes of a unit are exactly half a BLAKE3 block of its row, in the byte order
BLAKE3 reads (little-endian words), so the x and W input words are wired to message words ``m[0..2]`` or ``m[2..4]``.

The text is deterministic: the same relation gives the same bytes on any machine, and :data:`PINS` holds its sha256.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from . import unit as U
from . import unit_fp4

FORMAT = "flock-unit-io/v1"
K_LOG = 13
IN_COLS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Pipe:
    relation: str        # the B-Ligero relation name whose instance set and model this unit lowers
    fmt: object          # unit.BF16 | unit.E4M3
    groups: tuple
    width: int
    floor: int
    epilogue: bool       # f32_to_bf16 in-circuit (GemmCoordinate) or the bare FP32 word (GemmAccumulator)
    model: str           # verity.ml.circuits.silicon.models name
    total: bool = False  # every encoding (unit_total, F2fpBf16 epilogue) rather than finite operands asserted

    @property
    def unit_log(self) -> int:
        return K_LOG + 1 if self.total else K_LOG

    @property
    def domain(self) -> str:
        return "total" if self.total else "finite-only"

    @property
    def k(self) -> int:
        return sum(self.groups)

    @property
    def word_bits(self) -> int:
        return self.fmt.bits


PIPES = {
    "bf16-hopper": Pipe("bf16-hopper", U.BF16, (16,), 26, -133, True, "HOPPER_BF16_M16N8K16"),
    "fp8-ada": Pipe("fp8-ada", U.E4M3, (16, 16), 14, -139, False, "ADA_E4M3_M16N8K32"),
    "fp8-hopper": Pipe("fp8-hopper", U.E4M3, (32,), 14, -139, False, "HOPPER_E4M3_K32"),
    "bf16-ampere": Pipe("bf16-ampere", U.BF16, (8, 8), 25, -132, True, "AMPERE_BF16_M16N8K16"),
    # BlockScaledAlignAdd (unit_fp4): groups, width and floor are the model's own (frac_bits 27, acc_window 36).
    # sm90.wgmma.m64n8k16.bf16 (prims.HopperBF16WgmmaDot16 = tc_dot_total(HOPPER_BF16_WGMMA_K16), epilogue F2fpBf16): on the
    # finite domain the same unit as bf16-hopper (16 : 26 : -133; F2fpBf16 differs from f32_to_bf16 only on NaN)
    "bf16-hopper-wgmma": Pipe("bf16-hopper-wgmma", U.BF16, (16,), 26, -133, True, "HOPPER_BF16_WGMMA_K16"),
    "fp4-nvf4": Pipe("fp4-nvf4", U.E2M1, (16, 16, 16, 16), 27, -174, False, "BLACKWELL_SM120_NVF4"),
}

#: sha256 of :func:`netlist` per relation: the lowering version the verifier re-derives and compares.
PINS = {
    "bf16-hopper": "da1bbe2cdbf02971f0d7f12f6d5a3caabefb658af95afeeff44d001ff30dddba",
    "fp8-ada": "e66262a08816f03d7dba67860c68b3993adc9ca83f1746d15137b0b757599a7c",
    "fp8-hopper": "904ca664958e3144ec0af862eea3d2124898289f23ea1d065271a27a00539654",
    "bf16-ampere": "e97ecb9ef01d653b550ebe947cc12d92c60392613457b560246b99aff4922247",
    "bf16-hopper-wgmma": "12c3c8d382330788d4f85cc7132d4bf2d4a95886efe71fb4494cf9eaeaef74ad",
    "fp4-nvf4": "fb52a87c01a8a41f0e3460a10c62dcac4c346eb0c4540a86e0921845738666b3",
}


def _layout(pipe: Pipe):
    C = unit_fp4.unit() if pipe.fmt is U.E2M1 else U.unit(pipe.fmt, pipe.groups, pipe.width, pipe.floor, epilogue=pipe.epilogue)
    live = C.live()
    ins = [w for nm in ("x", "w", "c", "sx", "sw") if nm in C.inputs for w in C.inputs[nm]]
    assert len(ins) in (544, 608), len(ins)
    col = {}
    for w in ins:
        col[next(iter(w.s))] = len(col)
    rows_a: list[list] = [[i] for i in range(len(ins))]
    rows_b: list[list] = [[i] for i in range(len(ins))]
    zero = lambda: (rows_a.append([]), rows_b.append([]))  # noqa: E731
    while len(rows_a) < 640:
        zero()
    ands = [v for v in range(len(C.kind)) if v in live and C.kind[v] == "and"]
    assert all(C.kind[v] in ("in", "and") for v in live), "hints not supported"
    for v in ands:
        col[v] = len(rows_a)
        rows_a.append(None)
        rows_b.append(None)

    def form(L):
        s = [col[v] for v in L.s]
        if L.c:
            s.append("C")
        return s

    for v in ands:
        a, b = C.data[v]
        rows_a[col[v]], rows_b[col[v]] = form(a), form(b)
    for _, L in C.asserts:
        j = len(rows_a)
        rows_a.append(form(L) + [j])
        rows_b.append(["C"])
    outs = []
    for name, width in (("c_out", 32), ("y16", 16)):
        if name not in C.outputs:
            continue
        while len(rows_a) % 128:
            zero()
        outs.append(len(rows_a) // 128)
        bits = C.outputs[name]
        assert len(bits) == width
        for w in bits:
            rows_a.append(form(w))
            rows_b.append(["C"])
        for _ in range(128 - width):
            zero()
    const = len(rows_a)
    rows_a.append([const])
    rows_b.append([const])

    def fix(r):
        par: dict[int, int] = {}
        for c in r:
            c = const if c == "C" else c
            par[c] = par.get(c, 0) ^ 1
        return sorted(c for c, p in par.items() if p)

    rows_a = [fix(r) for r in rows_a]
    rows_b = [fix(r) for r in rows_b]
    assert len(rows_a) <= 1 << K_LOG, len(rows_a)
    return C, rows_a, rows_b, const, outs


def netlist(relation: str) -> str:
    """The lowered unit of ``relation`` as ``flock-unit-io/v1`` text."""
    pipe = PIPES[relation]
    _, ra, rb, const, outs = _layout(pipe)
    head = [FORMAT, relation, len(ra), const, 608 if pipe.fmt is U.E2M1 else 544, len(IN_COLS), *IN_COLS, len(outs), *outs]
    lines = [" ".join(map(str, head))]
    for a, b in zip(ra, rb):
        lines.append(" ".join(map(str, [len(a), *a, len(b), *b])))
    return "\n".join(lines) + "\n"


def digest(relation: str) -> str:
    return hashlib.sha256(netlist(relation).encode()).hexdigest()


def evaluate(relation: str, x: list[int], w: list[int], c: int, sx: list[int] = (), sw: list[int] = ()) -> tuple[int, int | None, bool]:
    """Run the lowered rows on one unit: ``(c_out, y16 or None, every row satisfied)`` (a failed assertion leaves a row
    unsatisfiable, so it shows as False)."""
    pipe = PIPES[relation]
    _, ra, rb, const, outs = _layout(pipe)
    bits = []
    for arr in (x, w):
        for v in arr:
            bits += [(v >> i) & 1 for i in range(pipe.word_bits)]
    bits += [(c >> i) & 1 for i in range(32)]
    for arr in (sx, sw):
        for v in arr:
            bits += [(v >> i) & 1 for i in range(8)]
    z = [0] * len(ra)
    z[:len(bits)] = bits
    z[const] = 1

    def ev(r):
        s = 0
        for j in r:
            s ^= z[j]
        return s

    for i in range(544, len(ra)):
        if i != const:
            z[i] = ev(ra[i]) & ev(rb[i])
    ok = all((ev(ra[i]) & ev(rb[i])) == z[i] for i in range(len(ra)))
    word = lambda col, n: sum(z[col * 128 + i] << i for i in range(n))  # noqa: E731
    return word(outs[0], 32), (word(outs[1], 16) if len(outs) > 1 else None), ok


def _model(name: str):
    from verity.ml.circuits.silicon import models, total
    return getattr(models, name) if hasattr(models, name) else getattr(total, name)


def self_check(relation: str, n: int = 200, seed: int = 20260925) -> int:
    """Mismatches of the lowered unit against ``verity.ml.circuits.silicon`` (``tc_dot`` and the ``f32_to_bf16`` epilogue) on ``n``
    random units over the whole finite domain: operand words uniform over every finite encoding (all exponent fields,
    subnormals and signed zeros included), and accumulators uniform over every finite FP32 exponent field, with a third of
    the cases drawn near each other in exponent so the alignment and cancellation paths are exercised too. Saturating
    units (the reference's +-inf word) are compared like any other."""
    from verity.ml.circuits.silicon import models
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word
    from verity.ml.circuits.silicon.silicon import tc_dot

    pipe = PIPES[relation]
    model = _model(pipe.model)
    rng = random.Random(seed)
    if pipe.fmt is U.E2M1:
        from .unit_fp4_check import cases
        bad = 0
        for c, x, w, sx, sw in cases(rng, n):
            got, _, ok = evaluate(relation, x, w, c, sx, sw)
            bad += not ok or got != model.step_scaled(c, x, w, sx, sw)
        return bad
    bits = pipe.word_bits
    finite = (lambda w: (w >> 7) & 0xFF != 0xFF) if bits == 16 else (lambda w: w & 0x7F != 0x7F)
    bad = 0
    for t in range(n):
        near = t % 3 == 0
        if bits == 16:
            e0 = rng.randint(0, 254)
            op = (lambda: (rng.getrandbits(1) << 15) | (min(254, max(0, e0 + rng.randint(-8, 8))) << 7) | rng.getrandbits(7)) if near \
                else (lambda: rng.choice([w for w in (rng.getrandbits(16) for _ in range(4)) if finite(w)] or [0]))
        else:
            op = lambda: rng.choice([w for w in range(256) if finite(w)])  # noqa: E731
        x = [op() for _ in range(pipe.k)]
        w = [op() for _ in range(pipe.k)]
        c = (rng.getrandbits(1) << 31) | (rng.randint(0, 254) << 23) | rng.getrandbits(23)
        want = tc_dot(model, c, x, w)
        got, y, ok = evaluate(relation, x, w, c)
        if not ok or got != want or (pipe.epilogue and y != f32_to_bf16_word(want)):
            bad += 1
    return bad
