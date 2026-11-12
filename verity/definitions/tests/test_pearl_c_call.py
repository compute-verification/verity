"""A -h3 call's rows and cells as registered Definitions (`verity.definitions.registry.pearl_c_call`), against PoUW's
scheme (`schemes.pearl_c`: `PearlC.encode`, a tile's formed operands and peel factors, and `PearlC.checked`, its C̃ and U)
with every noise input the Definitions' own derivation from the salt and the fixed operand's root, and their layout in
units.

A row on bits is a million ANDs at its least, so no test here evaluates one: circuit-check holds each row's Boolean version
to its word form on the binding the sidecar names, and each Definition a row Calls is held on bits by its own module's
tests."""
from __future__ import annotations

import json
import random
import struct

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import compile as compile_bits
from verity.circuits.evaluation.bits import evaluate_words
from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperBF16WgmmaDot16, HopperE4m3QgmmaDot32
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import pearl_c_call as K
from verity.definitions.registry import pearl_c_forming as F
from verity.definitions.registry import pearl_c_noise as N
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_kw as P
from verity.protocols.pouw.prover.schemes.pearl_c_device import DEVICES

#: each device record's atom and promotion period as the Definitions' statics
ATOMS = {"h100": (HopperE4m3QgmmaDot32, 4), "sm120": (BlackwellE4m3QmmaDot32, 4)}
SALT, ROOT_A, ROOT_B, UNIT = bytes(range(32)), bytes(range(100, 164)), bytes((7 * i + 3) % 256 for i in range(64)), 3
REGISTRY_DIR = Path(K.__file__).parent
#: the Definitions a row is laid out in, by name: -h3's hashes and a line's codes, the forming's codes, the atom, and the
#: FP32 steps of the statistics, the promotions and the peel factors
UNITS_A = {"PearlCRowBlock", "PearlCShaBlock", "PearlCRowSeedHead", "PearlCRowSeedTail", "PearlCLineKey", "PearlCLineRaw",
           "PearlCLineCodes", "PearlCSqrt", "PearlCScaledCode", "PearlCFormCode", "HopperE4m3QgmmaDot32", "F32Fma", "F32Add",
           "F32Div", "F32Mul", "F32Fmaxf", "I32Le", "SelectI32", "E4m3ToF32", "F2fpBf16"}
UNITS_B = UNITS_A - {"PearlCRowBlock", "PearlCShaBlock", "PearlCRowSeedHead", "PearlCRowSeedTail", "PearlCLineKey",
                     "PearlCLineRaw", "PearlCLineCodes"}


def flat(v) -> list[int]:
    """The words of a nested value, in leaf order."""
    return [y for x in v for y in flat(x)] if isinstance(v, (tuple, list, bytes)) else [int(v)]


def word(fn, *args) -> list[int]:
    """``fn`` on one instance of word arguments: its output words in leaf order."""
    return flat(evaluate(bind(fn) if isinstance(fn, CompositeDefinition) else fn, *(flat(a) for a in args)))


def on(fn, name: str, **statics):
    """``fn`` (a word Definition or its Boolean version) at the device record ``name``'s atom and promotion period, and for
    a cell the peel's step."""
    dot, g = ATOMS[name]
    step = HopperBF16WgmmaDot16
    if getattr(fn, "word", None) is not None:
        dot, step = boolean_version(dot), boolean_version(step)
    return bind(fn, DOT=dot, G=g, **({"STEP": step} if "STEP" in fn.statics else {}), **statics)


def rows(k: int, n: int, seed: int) -> list[list[int]]:
    """Rows of k FP32 words the scheme forms: Gaussian at three scales."""
    rng = random.Random(seed)
    return [[P.f32_bits(rng.gauss(0, (1.0, 1e-3, 300.0)[i % 3])) for _ in range(k)] for i in range(n)]


def row_bytes(x: list[int]) -> bytes:
    return b"".join(struct.pack("<I", w) for w in x)


@pytest.mark.parametrize("name", ["h100", "sm120"])
def test_a_calls_rows_are_the_schemes_from_the_salt_and_the_root(name):
    """A tile of a call at k = 64 under forming v1 and -h3: each of its rows of A and of B by `PearlCRowA` and
    `PearlCRowB` is `PearlC.encode`'s (codes, peel factors, s, ρ, α, β_row), where the basis F_A and F_B, the lines E_B and
    the Gram matrix are the Definitions' from the salt and B's root (`PearlCSeedB`, `PearlCLineKey`, `PearlCNoiseLine`,
    `PearlCGram`) and row A derives its own line from the row; and each cell by `PearlCCell` of those rows is
    `PearlC.checked`'s C̃ and U."""
    k = 64
    scheme = C.PearlC(k_min=k, tile=(2, 2), forming="v1", device=DEVICES[name])
    ctx, tile = C.Context(SALT, C.Matmul(UNIT, C.Shape(4, k, 4), 0), ROOT_A, ROOT_B), C.Tile((1, 3), (0, 2))
    a, b = rows(k, 4, 31), rows(k, 4, 37)
    ops = scheme.encode(ctx, scheme.noise(ctx), [row_bytes(a[i]) for i in tile.rows], [row_bytes(b[j]) for j in tile.cols], tile)
    assert ops is not None and scheme.admissible(ctx, ops)
    key = word(N.PearlCLineKey, word(N.PearlCSeedB, SALT, ROOT_B))
    line = lambda side, factor, t, norm: word(N.PearlCNoiseLine, key, side, factor, t, norm)  # noqa: E731
    fa, fb = ([line(side, 1, t, N.NORM_F) for t in range(k)] for side in (0, 1))
    g = word(on(F.PearlCGram, name, K=k, R=32), fb, fa)
    assert g == flat([list(r) for r in C.gram(scheme.noise(ctx).seed_b, k, C.LINE_NORM_V1, DEVICES[name])])
    formed = {"a": [], "b": []}
    for at, i in enumerate(tile.rows):
        got = word(on(K.PearlCRowA, name, K=k, R=32), SALT, a[i], ROOT_B, UNIT, i, fa, fb)
        side = ops.a
        assert got == side.codes[at] + ops.pa[at] + [side.s[at], side.rho[at], side.alpha[at], side.beta[at]]
        formed["a"].append((got[:k], got[k:k + 64]))
    for at, j in enumerate(tile.cols):
        got = word(on(K.PearlCRowB, name, K=k, R=32), b[j], line(1, 0, j, N.NORM_E), fb, fa, g)
        side = ops.b
        assert got == side.codes[at] + ops.pb[at] + [side.s[at], side.rho[at], side.alpha[at], side.beta[at]]
        formed["b"].append((got[:k], got[k:k + 64]))
    checked = scheme.checked(ctx, ops)
    for i, (codes_a, pa) in enumerate(formed["a"]):
        for j, (codes_b, pb) in enumerate(formed["b"]):
            assert word(on(K.PearlCCell, name, K=k, R=32), codes_a, codes_b, pa, pb) == [checked.c[i][j], checked.u[i][j]]


def test_a_cell_on_bits():
    """A cell over one promotion window at rank 8 and over two atoms at rank 32, on codes and factors of every kind."""
    rng = random.Random(43)
    code = lambda n: [rng.choice([c for c in range(256) if c & 0x7F != 0x7F]) for _ in range(n)]  # noqa: E731
    half = lambda n: [P.f32_to_bf16(rng.gauss(0, 10.0 ** rng.randrange(-2, 3))) for _ in range(n)]  # noqa: E731
    for k, r in ((128, 8), (64, 32)):
        every = [(code(k), code(k), half(2 * r), half(2 * r)) for _ in range(4)]
        every += [([0x7E] * k, [0x7E] * k, [0x7F7F] * (2 * r), [0x7F7F] * (2 * r)), ([0] * k, [0x80] * k, [0x7FC0] * (2 * r), half(2 * r))]
        args = [np.asarray([flat(inst[j]) for inst in every], dtype=np.uint64) for j in range(4)]
        got = evaluate_words(on(K.PearlCCellV2, "h100", K=k, R=r), *args)
        assert got.reshape(len(every), -1).tolist() == [word(on(K.PearlCCell, "h100", K=k, R=r), *i) for i in every]


def test_a_row_of_a_depends_on_its_place_and_the_salt():
    """The line a row of A is formed against is the row's own: another row index, unit or salt gives other codes."""
    k = 32
    x, fa = rows(k, 1, 41)[0], [C.sample_line(C.seed_b(SALT, ROOT_B), 0, 1, t, C.LINE_NORM_V1) for t in range(k)]
    fb = [[c] for c in range(1, k + 1)]
    row = on(K.PearlCRowA, "h100", K=k, R=1)
    base = word(row, SALT, x, ROOT_B, UNIT, 5, fa, fb)
    for other in ((SALT, x, ROOT_B, UNIT, 6), (SALT, x, ROOT_B, UNIT + 1, 5), (bytes(32), x, ROOT_B, UNIT, 5)):
        assert word(row, *other, fa, fb)[:k] != base[:k]


def test_a_row_refuses_a_k_off_the_atom_and_a_rank_over_a_line():
    for statics in ({"K": 48, "R": 1}, {"K": 0, "R": 1}, {"K": 32, "R": 33}, {"K": 32, "R": 0}):
        for row in (K.PearlCRowA, K.PearlCRowB):
            with pytest.raises(ValueError):
                on(row, "h100", **statics).signature()
    for statics in ({"K": 48, "R": 8}, {"K": 32, "R": 4}, {"K": 32, "R": 40}, {"K": 32, "R": 0}):
        with pytest.raises(ValueError):
            on(K.PearlCCell, "h100", **statics).signature()


def test_the_bindings_are_each_row_and_a_cell_on_words_and_on_bits():
    ids = [d.id.split("{")[0] for d in K.bindings()]
    assert ids == [f"PearlC{name}_v{v}" for v in (1, 2) for name in ("RowA", "RowB", "Cell")]
    assert all("K=32,R=1,DOT=HopperE4m3QgmmaDot32" in d.id or "K=128,R=8,DOT=HopperE4m3QgmmaDot32" in d.id for d in K.bindings())


@pytest.mark.slow
def test_a_rows_boolean_version_is_its_word_forms_calls_on_bits():
    """σ: row A on bits Calls the Boolean versions of its line, its forming and its peel factors, row B of its forming and
    its peel factors, and neither has a gate besides."""
    for row, calls in ((K.PearlCRowAV2, ["PearlCRowLine", "PearlCForm", "PearlCPeelA"]), (K.PearlCRowBV2, ["PearlCForm", "PearlCPeelB"])):
        nodes = on(row, "h100", K=64, R=32).body.nodes
        assert [(n.form, n.fn.defn.name, n.fn.defn.version) for n in nodes] == [("call", name, 2) for name in calls]


@pytest.mark.slow
def test_a_row_is_laid_out_in_units_each_under_the_whole_lowering():
    """`units` of a call's rows at k = 2,048 and 8,192 and rank 32 on the H100's atom: every AND of a row is in a
    Definition of at most 65,536 ANDs, the ones `UNITS_A` and `UNITS_B` name, and the rows' ANDs are these; each unit
    that is a Pearl-C Definition of this registry has a pin of its lowering, and so has the cell circuit-check binds.
    Tracing every unit takes seconds, which is why this is slow; the count of `ands` is the compiled circuit's on a small
    Definition."""
    small = on(F.PearlCPeelAV2, "h100", K=32, R=2)
    assert K.ands(small) == compile_bits(small).counts["and"] == 15_454
    want = {(K.PearlCRowAV2, 2048): 39_742_282, (K.PearlCRowBV2, 2048): 39_657_287,
            (K.PearlCRowAV2, 8192): 157_690_198, (K.PearlCRowBV2, 8192): 144_393_287}
    for (row, k), total in want.items():
        d = on(row, "h100", K=k, R=32)
        laid = K.units(d)
        assert {i.split("_v")[0] for i in laid} == (UNITS_A if row is K.PearlCRowAV2 else UNITS_B)
        assert all(0 < each <= K.LOWERED_WHOLE for _calls, each in laid.values())
        assert sum(calls * each for calls, each in laid.values()) == K.ands(d) == total
    row = K.units(on(K.PearlCRowAV2, "h100", K=2048, R=32))
    assert row["PearlCRowBlock_v2"] == (72, 57_947) and row["PearlCShaBlock_v2"] == (5, 57_947)
    pins = {}
    for module in ("pearl_c_noise", "pearl_c_forming", "pearl_c_call"):
        pins |= json.loads((REGISTRY_DIR / f"{module}.circuit-check.json").read_text())["definitions"]
    ours = [i for i in row if i.startswith("PearlC")]
    assert len(ours) == 10 and all(i in pins for i in ours), [i for i in ours if i not in pins]
    assert K.bindings()[-1].id in pins and K.ands(K.bindings()[-1]) <= K.LOWERED_WHOLE
    cell = K.units(on(K.PearlCCellV2, "h100", K=2048, R=32))
    assert {i.split("_v")[0]: calls for i, (calls, _each) in cell.items()} == {"HopperE4m3QgmmaDot32": 64, "F32Add": 16,
                                                                             "PearlCPeel": 1}
    assert K.ands(on(K.PearlCCellV2, "h100", K=2048, R=32)) == 479_120
    with pytest.raises(ValueError, match="ANDs of its own"):
        K.units(bind(N.PearlCRowBlockV2), limit=50_000)
