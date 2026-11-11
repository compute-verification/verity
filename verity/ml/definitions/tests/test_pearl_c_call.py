"""A -h3 call's rows as registered Definitions (`verity.ml.definitions.registry.pearl_c_call`), against PoUW's scheme
(`schemes.pearl_c.PearlC.encode`: a tile's formed operands and peel factors) with every noise input the Definitions' own
derivation from the salt and the fixed operand's root, and their layout in units.

A row on bits is a million ANDs at its least, so no test here evaluates one: circuit-check holds each row's Boolean version
to its word form on the binding the sidecar names, and each Definition a row Calls is held on bits by its own module's
tests."""
from __future__ import annotations

import json
import random
import struct

from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation.bits import compile as compile_bits
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
from verity.ml.definitions.boolean import boolean_version
from verity.ml.definitions.registry import pearl_c_call as K
from verity.ml.definitions.registry import pearl_c_forming as F
from verity.ml.definitions.registry import pearl_c_noise as N
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_kw as P
from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES

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
    """``fn`` (a word Definition or its Boolean version) at the device record ``name``'s atom and promotion period."""
    dot, g = ATOMS[name]
    return bind(fn, DOT=boolean_version(dot) if getattr(fn, "word", None) is not None else dot, G=g, **statics)


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
    `PearlCGram`) and row A derives its own line from the row."""
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
    for at, i in enumerate(tile.rows):
        got = word(on(K.PearlCRowA, name, K=k, R=32), SALT, a[i], ROOT_B, UNIT, i, fa, fb)
        side = ops.a
        assert got == side.codes[at] + ops.pa[at] + [side.s[at], side.rho[at], side.alpha[at], side.beta[at]]
    for at, j in enumerate(tile.cols):
        got = word(on(K.PearlCRowB, name, K=k, R=32), b[j], line(1, 0, j, N.NORM_E), fb, fa, g)
        side = ops.b
        assert got == side.codes[at] + ops.pb[at] + [side.s[at], side.rho[at], side.alpha[at], side.beta[at]]


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


def test_the_bindings_are_each_row_on_words_and_on_bits():
    ids = [d.id for d in K.bindings()]
    assert ids == [f"PearlCRow{side}_v{v}{{K=32,R=1,DOT=HopperE4m3QgmmaDot32_v{v},G=4}}" for v in (1, 2) for side in "AB"]


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
    Definition of at most 65,536 ANDs, the ones `UNITS_A` and `UNITS_B` name, and the rows' ANDs are these. Tracing every
    unit takes seconds, which is why this is slow; the count of `ands` is the compiled circuit's on a small Definition."""
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
    digest = K.units(on(K.PearlCRowAV2, "h100", K=2048, R=32))
    assert digest["PearlCRowBlock_v2"] == (72, 57_947) and digest["PearlCShaBlock_v2"] == (5, 57_947)
    with pytest.raises(ValueError, match="ANDs of its own"):
        K.units(bind(N.PearlCRowBlockV2), limit=50_000)


@pytest.mark.slow
def test_each_hash_and_code_unit_of_a_row_is_pinned():
    """The units of a row that are this registry's Pearl-C Definitions each have a pin of their lowering in their module's
    sidecar, which circuit-check holds them to."""
    pins = {}
    for module in ("pearl_c_noise", "pearl_c_forming"):
        pins |= json.loads((REGISTRY_DIR / f"{module}.circuit-check.json").read_text())["definitions"]
    laid = K.units(on(K.PearlCRowAV2, "h100", K=64, R=32)) | K.units(on(K.PearlCRowBV2, "h100", K=64, R=32))
    ours = [i for i in laid if i.startswith("PearlC")]
    assert len(ours) == 10 and all(i in pins for i in ours), [i for i in ours if i not in pins]
