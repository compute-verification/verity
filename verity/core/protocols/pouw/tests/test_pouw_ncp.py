"""``ncp-v1``: the pinned vectors, the production P, the int8 range, and the useful output exactly A·Bᵀ."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

import ncp_lean as NL
import pouw_vectors as MV
from verity.core.protocols.pouw.protocol import DomainError, Matmul, PoUWScheme, Shape, Tile
from verity.ml.protocols.pouw.schemes import get, ncp

PINNED = json.loads((Path(__file__).with_name("vectors") / "ncp.json").read_text())


def test_vectors_are_pinned():
    assert MV.vectors() == PINNED


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


def test_leans_evaluation_is_of_the_package_in_the_tree():
    """`ncp_lean.json` records the SHA-256 of the Lean sources it was evaluated from; they are the files of `Security/` now, so an
    edit to them fails here until `tests/ncp_lean.py` evaluates them again."""
    pinned = json.loads(NL.PINNED.read_text())["sources_sha256"]
    lean = Path(__file__).resolve().parents[5] / "Security"
    assert {f: hashlib.sha256((lean / f).read_bytes()).hexdigest() for f in pinned} == pinned


@pytest.mark.parametrize("kind", NL.CASES)
def test_route_u_is_the_shifted_unbiased_sums(kind):
    """Lean `checkedU_eq`: C^U_t = C_t − β·Σ_{l >= 16t} Y[l][j]; and X + β is u8, Y s8 (`rangeU`, `biasU8Iff` at β = 128)."""
    ops = NL.ops(kind)
    assert ncp.route_u_shift(ncp.checked_unbiased(ops), ops.y) == ncp.checked_route_u(ops)
    assert all(0 <= v + ncp.BETA <= 255 for row in ops.x for v in row) and all(-128 <= v <= 127 for row in ops.y for v in row)


def test_production_p_is_the_research_stores():
    """The word-granular P at k = 4096 is the certified one: SHA-256 24e68f30… of its entries as 2-byte little-endian integers."""
    assert PINNED["word_p_sha256"]["4096"] == "24e68f30edbe538e2842266ac6b917a8a0ccf3ee0b4f8bf2cb1d8fe5c5943854"


@pytest.mark.parametrize("case", MV.CASES, ids=[c[0] for c in MV.CASES])
def test_tile_output_is_exact_and_in_range(case):
    name, perm, shape, r, c, index, weight = case
    s, ctx = MV.scheme(perm, shape), MV.ctx_of(name, shape, index, weight)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    a, b = MV.rows(name, "A", len(tile.rows), shape[1], r[0]), MV.rows(name, "B", len(tile.cols), shape[1], c[0])
    ops = s.encode(ctx, s.noise(ctx), a, b, tile)
    assert all(-127 <= v <= 126 for row in ops.x + ops.y for v in row)       # Lean `Pouw.SecurityProofs.NCP.Range`
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
    assert cert.conjecture == "tt-ncp-u" and cert.theorems[0] == "Pouw.SecurityProofs.NCP.GammaFromTTNCP_U_v1"
    assert dict(cert.parameters) == {"beta": 128, "f": 24, "n0": 1600, "k_max": 1 << 16}
    assert float(1 - Fraction(199, 200) / (1 + Fraction(24, 3 * 1024))) > 0.0127            # 1.27% at n = 1024: why n0 is recorded
    assert ncp.NCP(k_min=64, n_min=64).certificate.gamma is None                              # a test instance
    with pytest.raises(KeyError):
        get("ncp-v2")


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
