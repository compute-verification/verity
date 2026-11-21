"""`fast_tile`'s batch replay, the form a served window's verdict is decided on (`verify_run.py --fast`), against the scheme's
scalar reference (`pearl_c.chain`, `peel`, `PearlC.checked`) on `pearl-c-h100-v1-h3`: the same committed operands and seeds
give the same encoding, C̃, U, tile leaf, served y and inert work, word for word, and a bit flipped in an operand or the
salt makes them differ."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_inert_work as D
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_u as PCU
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_kw as P
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_device import H100
from verity.protocols.accounting.compute.pouw.verifier.protocol import Context, Matmul, Shape, Tile

_SPEC = importlib.util.spec_from_file_location(
    "pouw_pearlc_fast_tile_held", Path(__file__).parents[3] / "benchmarks" / "pouw" / "pearl_c_vllm" / "fast_tile.py")
FT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = FT
_SPEC.loader.exec_module(FT)

SCHEME = "pearl-c-h100-v1-h3"
SALT, ROOT_A, ROOT_B = bytes(range(32)), bytes(range(64, 128)), bytes(range(128, 192))
NEG_ZERO = 0x80000000
#: (n, k) of the 1B window's four served linears, each called with 108 rows (prefill) and 4 (decode)
SERVED = ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))


def scheme() -> C.PearlC:
    s = pouw_schemes.get(SCHEME)
    assert s.device is H100 and s.hashing == "h3" and s.forming == "v1"
    return s


def bf16_rows(rng, rows: int, k: int, scale: float) -> np.ndarray:
    """FP32 words of BF16 values, the rows the served path commits (BF16 widened)."""
    return (rng.standard_normal((rows, k)).astype(np.float32) * scale).view(np.uint32) & 0xFFFF0000


def as_bytes(words: np.ndarray) -> list[bytes]:
    return [row.astype("<u4").tobytes() for row in words]


def replay(s: C.PearlC, ctx: Context, a: list[bytes], b: list[bytes], tile: Tile, inert: bool) -> dict:
    """Everything `verify_run._check` decides a tile on that the batch forms compute: the encoding, the checked words, the
    leaf, y at each word, and (``inert``) the tile's inert work; or the reference's rejection."""
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    if ops is None:
        return {"encode": None}
    out = {f"{side}.{f.name}": getattr(getattr(ops, side), f.name) for side in ("a", "b") for f in fields(C.Formed)}
    out.update(e_a=ops.e_a, e_b=ops.e_b, pa=ops.pa, pb=ops.pb, admissible=s.admissible(ctx, ops))
    checked = s.checked(ctx, ops)
    out.update(c=checked.c, u=checked.u, leaf=s.digest(ctx, noise, checked))
    y = []
    for i, row in enumerate(checked.u):
        for j, w in enumerate(row):
            try:
                y.append(PCU.y(w, ops.a.alpha[i], ops.b.alpha[j]))
            except PCU.NonFinite:
                y.append(None)
    out["y"] = y
    if inert:
        formed = list(zip(ops.a.alpha, ops.a.x, ops.a.beta))
        formed_b = list(zip(ops.b.alpha, ops.b.x, ops.b.beta))
        try:
            out["inert"] = D.tile_inert_work(ops.a.codes, ops.b.codes, formed, formed_b, s.device)
        except Exception as ex:  # noqa: BLE001 - a rejection is a verdict both forms must give alike
            out["inert"] = ("raised", type(ex).__name__, str(ex))
    return out


SCALAR_FNS = ("form_v1", "gram", "peel_factors_a", "peel_factors_b", "_basis")


def batch(s: C.PearlC, ctx: Context, a: list[bytes], b: list[bytes], tile: Tile, inert: bool = True) -> dict:
    """The tile replayed with `fast_tile.install`'s forms in place; its undo puts every scalar function back."""
    before = ([getattr(C, f) for f in SCALAR_FNS], D.chain_inert_work, D.tile_inert_work, type(s).checked)
    undo = FT.install(s)
    try:
        assert C.form_v1 is FT.form_v1 and type(s).checked is FT.checked and D.tile_inert_work is FT.tile_inert_work
        return replay(s, ctx, a, b, tile, inert)
    finally:
        undo()
        assert ([getattr(C, f) for f in SCALAR_FNS], D.chain_inert_work, D.tile_inert_work, type(s).checked) == before


def batch_and_scalar(s: C.PearlC, ctx: Context, a: list[bytes], b: list[bytes], tile: Tile, inert: bool = True
                     ) -> tuple[dict, dict]:
    return batch(s, ctx, a, b, tile, inert), replay(s, ctx, a, b, tile, inert)


def first_difference(fast: dict, ref: dict) -> tuple | None:
    """(field, flat index, batch value, scalar value) of the first value that differs, or None."""
    for name in ref.keys() | fast.keys():
        x, y = fast.get(name), ref.get(name)
        if x == y:
            continue
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            flat_x = [v for row in x for v in (row if isinstance(row, list) else [row])]
            flat_y = [v for row in y for v in (row if isinstance(row, list) else [row])]
            for i, (p, q) in enumerate(zip(flat_x, flat_y)):
                if p != q:
                    return name, i, p, q
        return name, None, x, y
    return None


def assert_same(fast: dict, ref: dict, where: str) -> None:
    diff = first_difference(fast, ref)
    assert diff is None, f"{where}: the batch replay and the reference differ first at {diff}"


def served_case(n: int, k: int, m: int, last: bool, seed: int):
    """The first tile (``last`` False) or the last of a served (m, k, n) call, its A rows and B rows as served."""
    s, rng = scheme(), np.random.default_rng(seed)
    shape = Shape(m, k, n)
    tiles = s.tiles(shape)
    tile = tiles[-1] if last else tiles[0]
    ctx = Context(SALT, Matmul(7, shape, 3), ROOT_A, ROOT_B)
    return s, ctx, as_bytes(bf16_rows(rng, len(tile.rows), k, 1.0)), as_bytes(bf16_rows(rng, len(tile.cols), k, 0.02)), tile


def test_a_flipped_operand_or_salt_bit_makes_the_forms_differ():
    """A 4-row tile at k = 256 replays alike on both forms; one bit flipped in an A row, or in the salt, on the batch side
    alone makes C̃, U and the leaf differ, so the comparison reads the words."""
    s, ctx, a, b, tile = served_case(64, 256, 4, False, 1)
    fast, ref = batch_and_scalar(s, ctx, a, b, tile, inert=False)
    assert "encode" not in ref
    assert_same(fast, ref, "4 x 64 at k = 256")
    words = ("c", "u", "leaf")
    flipped = bytearray(a[2])
    flipped[4 * 100 + 2] ^= 1
    fast_a = batch(s, ctx, [a[0], a[1], bytes(flipped), a[3]], b, tile, inert=False)
    assert first_difference({x: fast_a[x] for x in words}, {x: ref[x] for x in words})
    salt = bytearray(SALT)
    salt[0] ^= 1
    fast_s = batch(s, Context(bytes(salt), ctx.matmul, ctx.root_a, ctx.root_b), a, b, tile, inert=False)
    assert first_difference({x: fast_s[x] for x in words}, {x: ref[x] for x in words})


@pytest.mark.slow
@pytest.mark.parametrize("n, k", SERVED)
@pytest.mark.parametrize("m, last", [(4, False), (108, True)])
def test_served_tiles_replay_alike(n, k, m, last):
    """Each served linear's first tile at decode's 4 rows and last tile at prefill's 108 (44 rows, the call's last column
    tile): the served depths k = 2048 and 8192 (16 and 64 promotion windows) and the last column tiles' indices, which the
    cheap tests' smaller k and first tiles don't reach."""
    s, ctx, a, b, tile = served_case(n, k, m, last, n + k + m)
    fast, ref = batch_and_scalar(s, ctx, a, b, tile, inert=k == 2048)
    assert "encode" not in ref
    assert_same(fast, ref, f"({m}, {k}, {n}) {'last' if last else 'first'} tile")


def edge_rows(rng, rows: int, k: int, scale: float) -> np.ndarray:
    """Served-like rows with −0s, FP32 and BF16 subnormals, the largest BF16s that keep α·x finite, a row of zeros (which
    fails the noise floor) and a row with nine outliers (which fails `live_row` at k < 576)."""
    x = bf16_rows(rng, rows, k, scale)
    x[0, :16] = [NEG_ZERO, 0, 1, 0x80000001, 0x007FFFFF, 0x00010000, 0x80010000, 0x00800000, 0x80800000, NEG_ZERO,
                 0x3F800000, 0xBF800000, 0x7F000000, 0xFF000000, 0x00400000, 0x80400000]
    x[0, 16::7] = NEG_ZERO
    x[1] = 0
    x[2] = bf16_rows(rng, 1, k, 1e-3)[0]
    x[2, [1, 2, 3, 4, 5, 6, 7, 9, 10]] = 0x47000000
    x[3, ::3] = np.uint32(0x00010000) | (x[3, ::3] & NEG_ZERO)
    return x


@pytest.mark.parametrize("k", [96, 160, 256])
def test_edge_operands_replay_alike(k):
    """Rows of −0s, subnormals and the largest BF16s, an excluded zero row, a row `live_row` fails, at k below one window of
    128 (96), past one by an atom (160) and at two whole windows (256), on a 6-row tile at the end of A and the last
    column tile of B; the reference and the batch replay give the same words, the same admissibility and the same
    inert work.  At 96 and 160 the batch inert work is the reference's own (k isn't whole windows); at 256 the mend
    reaches an atom of zero products at a group start (`test_an_atom_of_zero_products_at_a_group_start_cuts_alike`)."""
    s, rng = scheme(), np.random.default_rng(k)
    shape = Shape(22, k, 128)
    tile = Tile(tuple(range(16, 22)), tuple(range(64, 128)))
    ctx = Context(SALT, Matmul(1, shape, 2), ROOT_A, ROOT_B)
    a, b = edge_rows(rng, 6, k, 1.0), edge_rows(rng, 64, k, 0.02)
    fast, ref = batch_and_scalar(s, ctx, as_bytes(a), as_bytes(b), tile)
    assert "encode" not in ref and ref["admissible"] is False
    assert_same(fast, ref, f"edge rows, k = {k}")


def test_a_row_past_the_finite_range_is_rejected_alike():
    """A row whose ρ overflows FP32: both forms refuse to encode the tile."""
    s, rng = scheme(), np.random.default_rng(9)
    tile = Tile((0, 1), tuple(range(64)))
    ctx = Context(SALT, Matmul(0, Shape(2, 128, 64), 0), ROOT_A, ROOT_B)
    a = bf16_rows(rng, 2, 128, 1.0)
    a[1, :64] = 0x7F7F0000
    fast, ref = batch_and_scalar(s, ctx, as_bytes(a), as_bytes(bf16_rows(rng, 64, 128, 0.02)), tile)
    assert fast == ref == {"encode": None}


#: E4M3 codes the batch atom could take wrong: ±0, the subnormals' ends, the normals' ends, ±448, and ordinary codes
EDGE_CODES = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x7E, 0xFE, 0x77, 0xF7, 0x38, 0xB8, 0x40, 0xC0]


def test_edge_codes_chain_and_peel_alike():
    """`fast_tile`'s chain and peel on every pair of rows of edge E4M3 codes (their sums cancelling to ±0 too) and on BF16
    peel operands with −0 and subnormals, and a NaN code's rejection, as the reference's."""
    rng = np.random.default_rng(11)
    rows = [[EDGE_CODES[(i + t) % 16] for t in range(256)] for i in range(16)]
    rows += [[c ^ (0x80 * (t >= 128)) for t, c in enumerate(rows[3][:128] * 2)]]
    rows += [list(rng.integers(0, 0x7F, 256) | rng.integers(0, 2, 256) << 7) for _ in range(4)]
    rows = [[int(c) for c in r] for r in rows]
    got = FT._pairs(rows, rows, H100)
    want = [[C.chain(x, y, H100) for y in rows] for x in rows]
    assert got == want
    assert any(w == 0 for r in want for w in r)
    pa = [[int(v) >> 16 for v in rng.standard_normal(C.PEEL_K).astype(np.float32).view(np.uint32)] for _ in range(5)]
    pa[0][:6] = [0x8000, 0x0000, 0x0001, 0x8001, 0x007F, 0x807F]
    c = [[w for w in r[:5]] for r in want[:5]]
    c[0][0] = NEG_ZERO
    assert FT.peel_batch(c, pa, pa, H100) == [[C.peel(c[i][j], pa[i], pa[j], H100) for j in range(5)] for i in range(5)]
    bad = [r[:] for r in rows[:2]]
    bad[1][40] = 0x7F
    with pytest.raises(Exception) as ref:
        [[C.chain(x, y, H100) for y in bad] for x in bad]
    with pytest.raises(type(ref.value), match=str(ref.value)):
        FT._pairs(bad, bad, H100)


def test_saturating_casts_form_alike():
    """Forming whose accumulators pass E4M3's 448 both ways (the cast saturates to ±448), land on subnormals and on ±0,
    batch and scalar alike, and `fast_tile._fp8` is Pearl's cast on those words."""
    rng = np.random.default_rng(13)
    k = 64
    x = bf16_rows(rng, 3, k, 1.0)
    x[1, ::2] = NEG_ZERO
    x[2, :8] = [NEG_ZERO, 0, 0x00010000, 0x80010000, 0x7E800000, 0xFE800000, 0x3F800000, 0xBF800000]
    e = [[0x7E if (i + t) % 2 else 0xFE for t in range(C.SLICE)] for i in range(3)]
    f = [[[0x7E, 0xFE, 0x00, 0x01][(t + q) % 4] for q in range(C.SLICE)] for t in range(k)]
    want = C.form_v1([[int(v) for v in r] for r in x], e, f, H100)
    got = FT.form_v1([[int(v) for v in r] for r in x], e, f, H100)
    assert [getattr(got, g.name) for g in fields(C.Formed)] == [getattr(want, g.name) for g in fields(C.Formed)]
    codes = {c & 0x7F for r in want.codes for c in r}
    assert 0x7E in codes and 0 in codes
    words = np.array([0, NEG_ZERO, 1, 0x80000001, 0x3A800000, 0x3B000000, 0x3B100000, 0x43E00000, 0x43E80000, 0x43F00000,
                      0xC3F00000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3C080000, 0x3C180000, 0x3F880000], np.uint32)
    assert [int(v) for v in FT._fp8(words)] == [P.f32_to_fp8(P.f32_of_bits(int(w))) for w in words]


def test_an_atom_of_zero_products_at_a_group_start_cuts_alike():
    """The mend's cut flags on an atom whose 32 products are all ±0 × ±0, from the +0 accumulator a promotion group starts
    on: no nonzero product raises the group's exponent above the floor, each zero product's exponent is above it, and
    the batch form (`_cut`) and the reference (`cut_flags`) must give the same flags."""
    zeros = [0x00, 0x80] * 16
    got = [bool(f) for f in FT._cut(np.zeros(1, np.uint32), np.array([zeros]), np.array([zeros[::-1]]), H100)[0]]
    assert got == D.cut_flags(0, zeros, zeros[::-1], H100)
