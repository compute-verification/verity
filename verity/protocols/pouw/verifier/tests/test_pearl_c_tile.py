"""TileRead and the served y as registered Definitions (`verity.definitions.registry.pearl_c_tile`), word and Boolean, against
PoUW's pin (`schemes.pearl_c_u`) on its vectors (the H100 kernel's gate problem, ``vectors/pearl_c_u.json``) and random codes
at both devices' atoms, against pc8's tile (`circuit.pc8`'s `Pc8Tile`), and the unscale against the kernels' epilogue order on
the vectors' U and their rows' α."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

import pearl_c_u_vectors as UV
import pearl_c_vectors as CV
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import evaluate_words
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
from verity.definitions.silicon import fp32
from verity.definitions.registry import pearl_c_tile as PT
from verity.protocols.pouw.prover.circuit import pc8
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_c_u as U
from verity.protocols.pouw.prover.schemes.pearl_c_device import H100, SM120

VECTORS = json.loads((Path(__file__).with_name("vectors") / "pearl_c_u.json").read_text())
#: each device record by its pc8 name, with its atom's word and Boolean Definitions
DEVICES = {"h100": (H100, HopperE4m3QgmmaDot32, BG.HopperE4m3QgmmaDot32),
           "sm120": (SM120, BlackwellE4m3QmmaDot32, BG.BlackwellE4m3QmmaDot32)}
ONE = 0x3F800000


def tile(k: int, tm: int, tn: int, dev: str = "h100", bits: bool = False):
    d, word, boolean = DEVICES[dev]
    return bind(PT.PearlCTileReadV2 if bits else PT.PearlCTileRead, K=k, TM=tm, TN=tn, DOT=boolean if bits else word, G=d.g)


def on_bits(fn, *args) -> list[list[int]]:
    """``fn`` (a Boolean Definition) on instances of word arguments, one list of words a parameter each: each instance's
    output words in leaf order."""
    out = evaluate_words(fn, *(np.asarray(a, dtype=np.uint64) for a in args))
    return out.reshape(out.shape[0], -1).tolist()


def traced_cells() -> list[tuple[str, list[int], list[int], list[int], list[int], int, int]]:
    """The vectors' traced cells, both formings: (forming, A′_i, B̃_j, P_A,i, P_B,j, C̃, U)."""
    out = []
    for forming, f in VECTORS["formings"].items():
        pa, pb = {r["i"]: r["pa"] for r in f["rows_a"]}, {r["j"]: r["pb"] for r in f["rows_b"]}
        u = {(c["i"], c["j"]): c["u"] for c in f["cells"]}
        for ch in f["chains"]:
            out.append((forming, list(bytes.fromhex(ch["a"])), list(bytes.fromhex(ch["b"])), pa[ch["i"]], pb[ch["j"]], ch["c"],
                        u[(ch["i"], ch["j"])]))
    return out


def test_tileread_reproduces_the_pinned_cells():
    """Each traced cell of the gate problem (k = 1,024, both formings): TileRead's C̃ and U are the vectors', on words and on
    bits."""
    cells = traced_cells()
    assert len(cells) == 6 and {len(c[1]) for c in cells} == {1024}
    for forming, a, b, pa, pb, c, u in cells:
        assert evaluate(tile(1024, 1, 1), a, b, pa, pb) == (c, u), forming
    got = on_bits(tile(1024, 1, 1, bits=True), *zip(*[cell[1:5] for cell in cells]))
    assert got == [[cell[5], cell[6]] for cell in cells]


def codes(rng: random.Random, n: int) -> list[int]:
    """E4M3 codes with zeros, subnormals and both signs, never NaN (the atom's domain)."""
    return [(0 if (r := rng.random()) < 0.1 else rng.randrange(1, 8) if r < 0.2 else rng.randrange(0, 0x7F))
            | (0x80 if rng.random() < 0.5 else 0) for _ in range(n)]


def factors(rng: random.Random) -> list[int]:
    return [C.P.f32_to_bf16(rng.uniform(-8, 8)) for _ in range(U.PEEL_K)]


@pytest.mark.parametrize("dev", sorted(DEVICES))
@pytest.mark.parametrize("k", [160, 256])
def test_tileread_is_the_pin_and_pc8s_tile(dev, k):
    """A 2 x 2 tile of random codes and factors at the device's atom: C̃ and U are `pearl_c_u`'s chain and peel on words and
    on bits, at a short last window (k = 160) and whole ones, and at k = 256 (pc8's domain, whole windows) `Pc8Tile`'s."""
    rng = random.Random(f"pearl-c-tile/{dev}/{k}")
    d = DEVICES[dev][0]
    ac, bc = [codes(rng, k) for _ in range(2)], [codes(rng, k) for _ in range(2)]
    pa, pb = [factors(rng) for _ in range(2)], [factors(rng) for _ in range(2)]
    c = [[U.chain(ac[i], bc[j], device=d) for j in range(2)] for i in range(2)]
    u = [[U.u(c[i][j], pa[i], pb[j], device=d) for j in range(2)] for i in range(2)]
    want = tuple(w for grid in (c, u) for row in grid for w in row)
    args = [sum(x, []) for x in (ac, bc, pa, pb)]
    assert evaluate(tile(k, 2, 2, dev), *args) == want
    assert on_bits(tile(k, 2, 2, dev, bits=True), *([a] for a in args)) == [list(want)]
    if k % d.window == 0:
        assert evaluate(pc8.templates(k, (2, 2), dev)["tile"], *args) == want


def test_tileread_refuses_what_its_chain_doesnt_cover():
    for kw in ({"K": 48}, {"K": 0}, {"G": 0}, {"TM": 0}):
        with pytest.raises(ValueError):
            bind(PT.PearlCTileRead, **({"K": 64, "TM": 1, "TN": 1, "DOT": HopperE4m3QgmmaDot32, "G": 4} | kw))


def served_y(u: int, alpha_a: int, alpha_b: int) -> int:
    """The kernels' store_y: RN_bf16(RN_f32(RN_f32(u · RN_f32(1/α_A)) · RN_f32(1/α_B)))."""
    return U.rn_bf16(fp32.mul(fp32.mul(u, fp32.div(ONE, alpha_a)), fp32.div(ONE, alpha_b)))


def vector_alphas() -> tuple[list[int], list[int]]:
    """α of the vectors' rows of A and of the weight (forming v1: `s5_stats` of each row's FP32 words)."""
    m, n, k = UV.SHAPE
    ra, rb = CV.rows_f32(UV.CASE, "A", m, k), CV.rows_f32(UV.CASE, "B", n, k)
    return ([C.s5_stats(C.parse_f32_row(ra[i], k))[2] for i in UV.ROWS],
            [C.s5_stats(C.parse_f32_row(rb[j], k))[2] for j in UV.COLS])


def test_unscale_is_the_kernels_epilogue():
    """The vectors' 6 x 6 U (forming v1, the served one) at its rows' α: y is the kernels' store_y word for word, on words and
    on bits, and with a bias each word then torch's BF16 add."""
    f = VECTORS["formings"]["v1"]
    u = {(c["i"], c["j"]): c["u"] for c in f["cells"]}
    grid = [u[(i, j)] for i in UV.ROWS for j in UV.COLS]
    aa, ab = vector_alphas()
    tm, tn = len(UV.ROWS), len(UV.COLS)
    want = [served_y(u[(i, j)], aa[x], ab[y]) for x, i in enumerate(UV.ROWS) for y, j in enumerate(UV.COLS)]
    assert list(evaluate(bind(PT.PearlCUnscale, TM=tm, TN=tn, BIAS=False), grid, aa, ab)) == want
    assert on_bits(bind(PT.PearlCUnscaleV2, TM=tm, TN=tn, BIAS=False), [grid], [aa], [ab]) == [want]
    rng = random.Random("pearl-c-tile/bias")
    bias = [C.P.f32_to_bf16(rng.uniform(-4, 4)) for _ in range(tn)]
    with_bias = [U.rn_bf16(fp32.add(w << 16, bias[e % tn] << 16)) for e, w in enumerate(want)]
    assert list(evaluate(bind(PT.PearlCUnscale, TM=tm, TN=tn, BIAS=True), grid, aa, ab, bias)) == with_bias
    assert on_bits(bind(PT.PearlCUnscaleV2, TM=tm, TN=tn, BIAS=True), [grid], [aa], [ab], [bias]) == [with_bias]
