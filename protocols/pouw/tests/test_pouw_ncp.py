"""``ncp-v1`` and ``ncp-v2``: the pinned vectors, the production P, the int8 range, and the useful output exactly A·Bᵀ;
``ncp-v2``'s noise bound to each tile's own strip."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

import ncp_lean as NL
import pouw_vectors as MV
from verity_pouw.protocol import DomainError, Matmul, PoUWScheme, Shape, Tile
from verity_pouw.schemes import get, ncp

PINNED = json.loads((Path(__file__).with_name("vectors") / "ncp.json").read_text())
PINNED_V2 = json.loads((Path(__file__).with_name("vectors") / "ncp_v2.json").read_text())


def test_vectors_are_pinned():
    assert MV.vectors() == PINNED
    assert MV.vectors_v2() == PINNED_V2


@pytest.mark.parametrize("kind", NL.CASES)
def test_reference_equals_leans_own_evaluation(kind):
    """`tests/vectors/ncp_lean.json` is Lean's `#eval` of `mmWU 128 k A B E₁ F₁ R (blockEnd 16 (3k) t)` (route U, the checked
    values) and of the unbiased `mmW` at every checkpoint (`tests/ncp_lean.py`); the reference assembles and accumulates
    the same inputs, bit for bit."""
    pinned = json.loads(NL.PINNED.read_text())
    assert pinned["toolchain"] == "leanprover/lean4:v4.34.0" and pinned["shape"] == [NL.M, NL.K, NL.N]
    assert pinned["beta"] == ncp.BETA == 128
    got = NL.python_checked(kind)
    assert got == pinned["route_u"][kind]
    assert NL.python_unbiased(kind) == pinned["unbiased"][kind]
    x = NL.inputs(kind)
    assert [[s[-1] for s in row] for row in got] == [[sum(p * q for p, q in zip(a, b)) for b in x["b"]] for a in x["a"]]


@pytest.mark.parametrize("kind", NL.CASES)
def test_route_u_is_the_shifted_unbiased_sums(kind):
    """Lean `checkedU_eq`: C^U_t = C_t − β·Σ_{l >= 16t} Y[l][j]; and X + β is u8, Y s8 (`rangeU`, `biasU8Iff` at β = 128)."""
    ops = NL.ops(kind)
    assert ncp.route_u_shift(ncp.checked_unbiased(ops), ops.y) == ncp.checked_route_u(ops)
    assert all(0 <= v + ncp.BETA <= 255 for row in ops.x for v in row) and all(-128 <= v <= 127 for row in ops.y for v in row)


def test_production_p_is_the_research_stores():
    """The word-granular P at k = 4096 is the one the research coordinator certified (767 of 767) and the red team
    re-certifies: SHA-256 24e68f30… of its entries as 2-byte little-endian integers."""
    assert PINNED["word_p_sha256"]["4096"] == "24e68f30edbe538e2842266ac6b917a8a0ccf3ee0b4f8bf2cb1d8fe5c5943854"


@pytest.mark.parametrize("case,version", [(c, 1) for c in MV.CASES] + [(c, 2) for c in MV.CASES_V2],
                         ids=[f"v1-{c[0]}" for c in MV.CASES] + [f"v2-{c[0]}" for c in MV.CASES_V2])
def test_tile_output_is_exact_and_in_range(case, version):
    name, perm, shape, r, c, index, weight = case
    s, ctx = MV.scheme(perm, shape, version), MV.ctx_of(name, shape, index, weight)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    a, b = MV.rows(name, "A", len(tile.rows), shape[1], r[0]), MV.rows(name, "B", len(tile.cols), shape[1], c[0])
    ops = s.encode(ctx, s.noise(ctx), a, b, tile)
    assert all(-127 <= v <= 126 for row in ops.x + ops.y for v in row)       # Lean `Pouw.NCP.Proofs.range`
    assert all(0 <= v + s.beta <= 255 for row in ops.x for v in row)           # `rangeU`: route U's u8 operand
    sa, sb = ncp.int7_rows(a, shape[1]), ncp.int7_rows(b, shape[1])
    assert s.useful(ctx, None, ops, s.checked(ctx, ops)) == [[sum(p * q for p, q in zip(x, y)) for y in sb] for x in sa]


def test_interface_and_names():
    for name in ("ncp-v1", "ncp-v1-shift24"):
        s = get(name)
        assert isinstance(s, PoUWScheme) and s.name == name
    cert = get("ncp-v1").certificate
    assert cert.gamma == 1 - Fraction(199, 200) / Fraction(1608, 1600)   # gammaFromTTNCP_U_v1 at n0 = 1600: 0.995%
    assert cert.gamma < Fraction(1, 100)
    assert cert.conjecture == "tt-ncp-u" and cert.theorems[0] == "Pouw.NCP.RouteUProofs.gammaFromTTNCP_U_v1"
    assert dict(cert.parameters) == {"beta": 128, "f": 24, "n0": 1600, "k_max": 1 << 16}    # red-team R13-4
    assert float(1 - Fraction(199, 200) / (1 + Fraction(24, 3 * 1024))) > 0.0127            # 1.27% at n = 1024: why n0 is recorded
    assert ncp.NCP(k_min=64, n_min=64).certificate.gamma is None                              # a test instance
    v2 = get("ncp-v2")
    assert isinstance(v2, PoUWScheme) and v2.name == "ncp-v2" and v2.version == 2
    assert v2.certificate.gamma == cert.gamma and v2.certificate.theorems == cert.theorems
    assert dict(v2.certificate.parameters) == {**dict(cert.parameters), "strip": 16}
    with pytest.raises(KeyError):
        get("ncp-v3")
    for bad in ({"perm": "shift24", "version": 2}, {"tile": (8, 16), "version": 2}, {"version": 3}):
        with pytest.raises(ValueError):
            ncp.NCP(**bad)


def _v2_tile(case: str, m: int, r: tuple[int, int], rows_a: list[bytes] | None = None):
    """An ``ncp-v2`` test tile over a (m, 64, 64) unit: its context, noise and A strip (or the given one)."""
    s, ctx = ncp.NCP(k_min=64, n_min=64, version=2), MV.ctx_of(case, (m, 64, 64), 5, 1)
    tile = Tile(tuple(range(*r)), tuple(range(16)))
    a = rows_a if rows_a is not None else MV.rows(case, "A", 16, 64, r[0])
    return s, ctx, tile, a, MV.rows(case, "B", 16, 64)


def test_v2_noise_is_bound_to_the_tiles_own_strip_not_to_root_a():
    """E₁ reads D_s of its own 16 rows: another root_A leaves the tile unchanged, one changed entry of the strip changes
    every E₁ row of it (and so X), and the useful output stays A·Bᵀ."""
    s, ctx, tile, a, b = _v2_tile("strip", 32, (16, 32))
    ops = s.encode(ctx, s.noise(ctx), a, b, tile)
    other_root = type(ctx)(ctx.salt, ctx.matmul, bytes(32), ctx.root_b)
    assert s.encode(other_root, s.noise(other_root), a, b, tile) == ops
    edited = [bytes([(a[3][0] + 1) & 0x3F]) + a[3][1:] if i == 3 else r for i, r in enumerate(a)]
    moved = s.encode(ctx, s.noise(ctx), edited, b, tile)
    assert all(mx[:64] != ox[:64] for mx, ox in zip(moved.x, ops.x))
    assert s.useful(ctx, None, moved, s.checked(ctx, moved)) == [
        [sum(p * q for p, q in zip(x, y)) for y in ncp.int7_rows(b, 64)] for x in ncp.int7_rows(edited, 64)]


def test_v2_tiles_are_whole_strips():
    s, ctx, _, a, b = _v2_tile("whole", 32, (0, 16))
    assert s.encode(ctx, s.noise(ctx), a[:8], b, Tile(tuple(range(8)), tuple(range(16)))) is None
    assert s.encode(ctx, s.noise(ctx), a, b, Tile(tuple(range(8, 24)), tuple(range(16)))) is None
    with pytest.raises(ValueError):
        ncp.strip_digest(a[:15])


@pytest.mark.parametrize("shape", [(16, 1024, 2048), (16, 2048, 1536), (8, 2048, 2048), (16, 2040, 2048), (16, 2048, 1 << 17)])
def test_domain(shape):
    with pytest.raises(DomainError):
        get("ncp-v1").check(Matmul(0, Shape(*shape), 0))


def test_benchmark_model_is_in_the_domain():
    """Qwen2.5-3B's linears: qkv 2048 -> 2560, o 2048 -> 2048, gate_up 2048 -> 22016, down 11008 -> 2048."""
    for k, n in ((2048, 2560), (2048, 2048), (2048, 22016), (11008, 2048)):
        get("ncp-v1").check(Matmul(0, Shape(16, k, n), 0))


@pytest.mark.parametrize("k", [1024, 1088, 2048, 4096, 11008])
def test_permutations_pass_the_rejection_rule(k):
    assert ncp.admissible_perm(k, *ncp.shift_perm(k))
    pi, sigma = ncp.word_perm(k)
    assert ncp.admissible_perm(k, pi, sigma) and sorted(pi) == list(range(k))
    assert all(pi[4 * a + b] == pi[4 * a] + b for a in range(k // 4) for b in range(4))    # word-granular
    assert not ncp.admissible_perm(k, *ncp.shift_perm(k, 1))                               # NA1's shift by one
