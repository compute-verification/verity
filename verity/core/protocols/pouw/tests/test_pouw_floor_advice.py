"""Falsification test of `Accounting.Assumptions.PouwFloorAdvice`
(`Security/Models/Accounting/Assumptions/PouwFloor.lean`), compute attribution's floor at a block whose outside holds
advice about the salt.

The conjecture frees all help that is the same at every salt (TT) and is scoped to the schemes whose noise is full-rank
and per-entry: the window's noise lies in no salt-independent affine subspace of lower dimension than its noise space,
so nothing a helper precomputes shortens the block's salt-dependent work. With low-rank noise
`F = Σ_{t < r} c_t(salt) · G_t`, the helper ships the `r` products `A · G_t` and the block finishes `A · F` in `r`
multiply-adds an output instead of `k`, and the conjecture is false. So:

* `test_full_rank_schemes_fill_their_noise_space`: each in-scope scheme's noise for one tile (`E₁`'s rows and `F₁`'s
  columns, read off the scheme's own `encode` at zero operands) spans its whole noise space over sampled salts: the
  differences from the first salt's noise have full rank modulo a prime, hence over the rationals. A scheme whose noise
  comes from a lower-dimensional salt-independent subspace fails here;
* `test_every_scheme_is_scoped`: every registered scheme is in scope or excluded with its reason, so a new scheme can't
  enter the registry with the conjecture silently claimed for it;
* `test_every_compile_scheme_is_scoped`: the same for the compile pass's registry (`circuit.compile.SCHEMES`, with its
  built-in schemes imported), where the compile pass is on the tree, so an unclassified EPS1 scheme fails it;
* `test_compile_schemes_fill_their_noise_space`: H1 (`eps1-sm120-h1fx1`, `circuit.h1tile`) is in scope through its
  per-entry coins, the dither bit `b` on each entry of `X = A + b` and the coin bit `c` on each entry of
  `W' = B + c + U`; read off `h1tile.noise`, its noise fills its noise space, where the scheme is registered;
* `test_low_rank_noise_is_refused`: the counterexample, rank-`r` noise, fails the same rank check, and the helper's
  shortcut reproduces `A · F` exactly at `r` multiply-adds an output;
* `test_period_32_keyed_word_alone_is_refused`: H1's keyed word `U` alone (32 values a row, repeated along K) fails the
  rank check, and `A · U` comes from `A`'s residue sums at 32 multiply-adds an output, so `U` is outside the scope: so
  is adopted `P · V`'s `P` side, whose only noise is `U`. The per-entry coin on top fills the space.

Not checked here, outside the scope: the peel's clean side (the served `O = Σ P · V`), which is bit-affine in the coins
and floors nothing (`note:20261018T0634Z-draft-attn-noise`).

The rank is checked at `k = 64`, where the noise's derivation is the same as at served `k` (one hash output an
entry)."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util

import numpy as np
import pytest

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import SCHEMES, get, ncp

#: the prime the rank is taken modulo: a full rank modulo `P` is a full rank over the rationals
P = (1 << 31) - 1
K = 64
TILE = Tile((0, 1), (0, 1))
#: `E₁`'s rows and `F₁`'s columns of `TILE`, each `K` entries
DIM = (len(TILE.rows) + len(TILE.cols)) * K
#: the compile pass's package (#2451), and the operand roots its seeds read
CIRCUIT = "verity.core.protocols.pouw.circuit"
ROOTS = (hashlib.sha512(b"A0").digest(), hashlib.sha512(b"B0").digest())

#: the schemes whose noise is full-rank and per-entry: each entry of `E₁` and `F₁` its own SHAKE256 output
FULL_RANK = ("ncp-v1", "ncp-v1-shift24")
#: the schemes outside the conjecture, each with its reason
OUT_OF_SCOPE = {
    "pearl-fp8-v4": "Komargodski–Weinstein noise of rank 32: E·Fᵀ over a salt-derived basis",
    "pearl-c-h100-v0": "Pearl's rank-32 noise; the clean-up is a rank-64 product",
    "pearl-c-h100-v1": "Pearl's rank-32 noise; the clean-up is a rank-64 product",
    "pearl-c-sm120-v1": "Pearl's rank-32 noise; the clean-up is a rank-64 product",
    "pearl-c-sm120-v1-h3": "Pearl's rank-32 noise; the clean-up is a rank-64 product",
    "pearl-c-nvfp4-v0": "Pearl-C's rank-32 lines on block-scaled FP4",
}


def salts(n: int, tag: bytes) -> list[bytes]:
    return [hashlib.sha512(tag + i.to_bytes(4, "little")).digest() for i in range(n)]


def rank_mod_p(rows: np.ndarray) -> int:
    """The rank of an integer matrix modulo `P`, by Gaussian elimination."""
    a = np.asarray(rows, dtype=np.int64) % P
    rank = 0
    for c in range(a.shape[1]):
        nz = np.flatnonzero(a[rank:, c])
        if nz.size == 0:
            continue
        piv = rank + int(nz[0])
        a[[rank, piv]] = a[[piv, rank]]
        a[rank] = a[rank] * pow(int(a[rank, c]), P - 2, P) % P
        col = a[:, c].copy()
        col[rank] = 0
        a = (a - np.outer(col, a[rank]) % P) % P
        rank += 1
        if rank == a.shape[0]:
            break
    return rank


def fills_noise_space(noise: list[list[int]]) -> bool:
    """Whether the sampled noise vectors' affine hull is the whole space: their differences from the first have full
    rank."""
    v = np.asarray(noise, dtype=np.int64)
    return rank_mod_p(v[1:] - v[0]) == v.shape[1]


def tile_noise(name: str, salt: bytes) -> list[int]:
    """The scheme's noise for `TILE` at `k = K`: `E₁`'s rows and `F₁`'s columns, read off `encode` at zero operands (the
    first `K` entries of each row of X are A + E₁, of each column of Y B + F₁)."""
    scheme = get(name, k_min=K, n_min=K)
    ctx = Context(salt=salt, matmul=Matmul(7, Shape(16, K, K), 3), root_a=b"\x01" * 32, root_b=b"\x02" * 32)
    zero = ncp.int7_bytes([0] * K)
    ops = scheme.encode(ctx, scheme.noise(ctx), [zero] * len(TILE.rows), [zero] * len(TILE.cols), TILE)
    return [v for row in ops.x for v in row[:K]] + [v for col in ops.y for v in col[:K]]


def compile_registry() -> dict | None:
    """The compile pass's registry (`circuit.compile.SCHEMES`) with its built-in schemes imported, or `None` on a tree
    without the compile pass."""
    if importlib.util.find_spec(f"{CIRCUIT}.compile") is None:
        return None
    C = importlib.import_module(f"{CIRCUIT}.compile")
    for m in C.BUILTIN:
        importlib.import_module(f"{CIRCUIT}.{m}")
    return dict(C.SCHEMES)


def signed8(u: int) -> int:
    return u - 256 if u >= 128 else u


def h1_noise(salt: bytes) -> list[int]:
    """H1's noise for a 4 × 8 tile at `k = K` (`h1tile.noise`): rows 0–1 of X's, the dither bits `b`, then columns 0–1 of
    W's, each entry's coin bit `c` plus `U`'s keyed word at its index mod 32. Its digit coins and fold keys are the
    receipt's, not the products' noise."""
    H = importlib.import_module(f"{CIRCUIT}.h1tile")
    SD = importlib.import_module(f"{CIRCUIT}.seeds")
    rows, cols, _ = H.noise(SD.seed(salt, 0, *ROOTS), H.Layout(4, 8, K))
    return ([b for bits, _ in rows[:2] for b in bits] +
            [c + signed8(u[t % 32]) for bits, u in cols[:2] for t, c in enumerate(bits)])


#: the compile pass's schemes in scope, each with what reads its per-entry noise for two rows and two columns
COMPILE_FULL_RANK = {"eps1-sm120-h1fx1": h1_noise}
#: the compile pass's schemes outside the conjecture, each with its reason
COMPILE_OUT_OF_SCOPE: dict[str, str] = {}


def low_rank_noise(salt: bytes, basis: np.ndarray) -> list[int]:
    """The counterexample's noise: salt-derived coefficients in [−31, 32] on a salt-independent basis."""
    c = np.array([(b & 63) - 31 for b in hashlib.shake_256(b"low-rank\0" + salt).digest(basis.shape[0])])
    return (c @ basis).tolist()


@pytest.mark.parametrize("name", FULL_RANK)
def test_full_rank_schemes_fill_their_noise_space(name: str) -> None:
    noise = [tile_noise(name, s) for s in salts(DIM + 9, name.encode())]
    assert len(noise[0]) == DIM
    assert fills_noise_space(noise), (
        f"{name}'s noise lies in a salt-independent subspace of dimension < {DIM}: PouwFloorAdvice is false for it, so "
        f"move it to OUT_OF_SCOPE with the reason and drop it from the conjecture's scope in its docstring")


def test_every_scheme_is_scoped() -> None:
    assert not set(FULL_RANK) & set(OUT_OF_SCOPE)
    unscoped = set(SCHEMES) - set(FULL_RANK) - set(OUT_OF_SCOPE)
    assert not unscoped, (
        f"{sorted(unscoped)} registered with no PouwFloorAdvice scope: add each to FULL_RANK (and the docstring's "
        f"scope) if its noise is full-rank and per-entry, else to OUT_OF_SCOPE with the reason")
    assert not (set(FULL_RANK) | set(OUT_OF_SCOPE)) - set(SCHEMES), "a scoped scheme is no longer registered"


def test_every_compile_scheme_is_scoped() -> None:
    assert not set(COMPILE_FULL_RANK) & set(COMPILE_OUT_OF_SCOPE)
    registry = compile_registry()
    if registry is None:
        pytest.skip("the compile pass (circuit.compile, #2451) is not on this tree")
    unscoped = set(registry) - set(COMPILE_FULL_RANK) - set(COMPILE_OUT_OF_SCOPE)
    assert not unscoped, (
        f"{sorted(unscoped)} registered in circuit.compile.SCHEMES with no PouwFloorAdvice scope: add each to "
        f"COMPILE_FULL_RANK with its noise reader (and the docstring's scope) if its noise is full-rank and per-entry, "
        f"else to COMPILE_OUT_OF_SCOPE with the reason")


@pytest.mark.parametrize("name", sorted(COMPILE_FULL_RANK))
def test_compile_schemes_fill_their_noise_space(name: str) -> None:
    registry = compile_registry()
    if registry is None or name not in registry:
        pytest.skip(f"{name} is not registered on this tree (h1tile registers it where circuit.segment, E2E-120, is)")
    noise = [COMPILE_FULL_RANK[name](s) for s in salts(DIM + 9, name.encode())]
    assert len(noise[0]) == DIM
    assert fills_noise_space(noise), (
        f"{name}'s noise lies in a salt-independent subspace of dimension < {DIM}: PouwFloorAdvice is false for it, so "
        f"move it to COMPILE_OUT_OF_SCOPE with the reason and drop it from the conjecture's scope in its docstring")


def test_low_rank_noise_is_refused() -> None:
    rng = np.random.default_rng(2026101801)
    r = 8
    basis = rng.integers(-31, 33, size=(r, DIM))
    noise = [low_rank_noise(s, basis) for s in salts(DIM + 9, b"low-rank")]
    assert rank_mod_p(np.asarray(noise[1:]) - np.asarray(noise[0])) <= r
    assert not fills_noise_space(noise)

    m, n = 4, 4
    a = rng.integers(-64, 64, size=(m, K))
    g = rng.integers(-31, 33, size=(r, K, n))
    shipped = np.einsum("mk,tkn->tmn", a, g)
    c = rng.integers(-31, 33, size=r)
    f = np.einsum("t,tkn->kn", c, g)
    assert np.array_equal(np.einsum("t,tmn->mn", c, shipped), a @ f)
    assert r * m * n < K * m * n


def keyed_word(salt: bytes) -> list[int]:
    """A stand-in for H1's keyed word: 32 values in ±64 (`PouwU129`'s −64 + byte mod 129) from the salt."""
    return [-64 + x % 129 for x in hashlib.shake_256(b"keyed-word\0" + salt).digest(32)]


def coin_bits(salt: bytes) -> list[int]:
    return [(x >> j) & 1 for x in hashlib.shake_256(b"coin\0" + salt).digest(K // 8) for j in range(8)]


def test_period_32_keyed_word_alone_is_refused() -> None:
    ss = salts(K + 9, b"keyed-word")
    alone = [[keyed_word(s)[t % 32] for t in range(K)] for s in ss]
    assert rank_mod_p(np.asarray(alone[1:]) - np.asarray(alone[0])) <= 32
    assert not fills_noise_space(alone)
    with_coin = [[c + keyed_word(s)[t % 32] for t, c in enumerate(coin_bits(s))] for s in ss]
    assert fills_noise_space(with_coin)

    rng = np.random.default_rng(2026101802)
    m, n = 4, 4
    a = rng.integers(-128, 128, size=(m, K))
    u = rng.integers(-64, 65, size=(32, n))
    residues = a.reshape(m, K // 32, 32).sum(axis=1)
    assert np.array_equal(residues @ u, a @ np.tile(u, (K // 32, 1)))
    assert 32 * m * n < K * m * n
