"""Pearl-C's forming v1, chain, Gram matrix, peel factors and peel as registered Definitions
(`verity.definitions.registry.pearl_c_forming`), word and Boolean, against PoUW's reference (`schemes.pearl_c`:
`s5_stats`, `eprime`, `form_v1`, `chain`, `gram`, `peel_factors_a`, `peel_factors_b`, `peel`), against PoUW's row units
(`circuit.pc8`: `Pc8RowA`, `Pc8RowB`) on rows the reference rejects too, and on a chain the Lean specification proves of its
own (`Pouw.PearlC.PeelExactVectors.chainV1c0`, held by `PeelExactChecks.chainsV1`)."""
from __future__ import annotations

import json
import random

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import compile as compile_bits
from verity.circuits.evaluation.bits import evaluate_words
from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperBF16WgmmaDot16, HopperE4m3QgmmaDot32
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import pearl_c_forming as F
from verity.protocols.pouw.prover.circuit import pc8
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_kw as P
from verity.protocols.pouw.prover.schemes.pearl_c_device import DEVICES

#: each device record's atom and promotion period as the Definitions' statics (G = 0: no promotion)
ATOMS = {"h100": (HopperE4m3QgmmaDot32, 4), "sm120": (BlackwellE4m3QmmaDot32, 4), "sm120-unpromoted": (BlackwellE4m3QmmaDot32, 0)}
SEED_A, SEED_B = C.S512Seed(bytes(range(32))), C.S512Seed(bytes(range(64, 96)))
#: the size a Boolean Definition is lowered whole at (`circuit_check.checks.Options.lower_max_ands`), and the pins of this
#: module's lowerings
LOWERED_WHOLE = 1 << 16
PINS = Path(F.__file__).with_suffix(".circuit-check.json")
NAN, INF = 0x7FFFFFFF, 0x7F800000

#: `PeelExactVectors.chainV1c0`: forming v1's traced chain for C̃ at (0, 0) on the H100, 32 atoms and 8 promotions; each
#: atom's 32 codes packed low byte first (`Fp8Atom.codes`), and the chain's word
CHAIN_A = [
    0x53CBE4D3DDEAE26967E66D41E65BF1DC59EE4F603F5E6EEE6963EB5DEAE2DDDA,
    0x5355E3C16E6CE859ED706D7571D57274D8EEE770DA6CDBCA6ECDE9F0D8DBF3BF,
    0x6859DA4A6C686C6462C9DE6167F0EDE6F57540F2EB75E967E1E96AEC6F696AEB,
    0xAFEADC55EC51E6636F7366E57370DAF2ECCC6562EAEDEB68695AEEEA67E96E55,
    0xEAF25874E558E96FC66CC86F63EFD6F06AD146EEDBDF6BE06B74D3E569E0E9E9,
    0x69EEE9D075596CE0E865F05AE1F262E9DD6C77F44E60E4F06A5276E9EA73F25C,
    0xDBEBE2D8E8F17035626C5F5E5962EA62E2D9D1CADF61E84B57ECEBF151F04E60,
    0xE7EA58E8615DEC68EFEA6DED625CE1E9DBE371676874DCE9DBEC7169E5E5EAD7,
    0xE3E05F59E971B062D3D1D1CBE6575ADE5DC22ADC6B5D63C8D6EC5FD3DECDE865,
    0x58ED6569DE5F69486AE0E9E2E1D4E567E9E65AD1E35C6466D8E1C76C63E86D64,
    0x50E7D9DB5EEAC2EEE75669EA64E54E62D864E5E3E96356DFD962E9D9DEE9E4E1,
    0x63386A5C6CE0EAE06AEDEB706468C96260644D467367E1EC5D6F645B62666D5A,
    0x4EE46B5CF2E75ED379ED77ED77EBDF5BF1E9E4D47564D16CDDE66D58EA656653,
    0x5EE06FE36B5AC955E267E964D9EB5958D664F2E4EB68DAD54B6E58EF5CEAC957,
    0x75E26DF1E7E76EF166D5726F6BE9EFE26BE8E6D8675DF1E0F8E16AEFEE6E6A65,
    0x70EBC36AE1F2E5E1F2AEE86FCFE7EDF5556CE35FBD68407261CC5BB965F065E4,
    0x58E5D94763E06C5769E863D7D75DECC15EE4686055626AEEE1606B61DDDA5F6F,
    0x68F04DE9616B61586E576869DD6A615A61D951D1DE67E95BC8E5DDE15CDDE8DA,
    0x624FD3D86869DCF1F24BF2E472E96DEE616C6454E831ECDA69CCE96F6F6A68ED,
    0x4FB5B0EA6D616DDDE760E5E8E069E0E9F3E16E66F2E5E6E66D5CE365E56E59E2,
    0xEB666D6966DAF0DB6859F268F1EC6EDFE1DAD55E6D566621D86BEF6046E6CE49,
    0x39526B62647066D0F3776968DB6A72F15E62E5EFD965E3DBF06DE8686AE0D863,
    0xDD78F0B8F7EB70DDDF70ED69DFEBEA7472F53C72EAEC6FF4DB6967655CF072D7,
    0x686DE24D625A6F59DC69EAEBF0E1EFE4E6EDE4E464D869E55D70DE695CDDECEA,
    0xEEE7DFF86FED6CDA72E9D6EB64E5EA6B60E2E74963E4EEE26EC5F5706BF8F074,
    0x6CDF6CE5F16EEEE9F05D5E6AD5D0EEE4505B70E55AECE85B6BE6DCD3E5EB6E67,
    0xD6E4EC69624CD956F6F2F667E96C42F6EC6E5868E8EEF06F7369CFEDE5F1ED46,
    0x5AD9DDF0635CDC4BF374DB7845657372DEE5F5E5EB70E16A59555B726456E8DB,
    0xE8605F6C616D5BEEF4736B5EE8DAEBE857DEE9E4D1E85FE9D878EFF2F1727278,
    0x2149D659EE66E171E76A74D1D8CEED68D86FDBF16D6CEA616A6C6A51675B37D3,
    0x69DBF46C6A68F0F0FA5DD46B656CF2736B74F8725676726563EFD4E169E2EAE9,
    0x3A626966E2D2E2E7585FE1546A5C52AD53ED606EDF5A62F1F0E4EC53EAF3EADA,
]
CHAIN_B = [
    0x3AECE968F06F5D7274E6E0EE6F695F6D435E666CE0EC6FE1EF67EDD45B35696A,
    0x51696169DAD7DB7269E2E3DE615962D4D4F95F666669E8E966615AE6E74FC9E1,
    0x6E64ED5FF265675144DE68DC6BDEC5EB5BE365EAB2E45F64575668E0D5BDD9DD,
    0x6E6175C5ECE96B70EA71DCE671463EE1E8E763E06A695FDD67EA55DBE164666E,
    0x5A6168ED6967653A6262E062DDE3F26272F1F36BDED761EE6CCE59E0705EC661,
    0xF373676575E8F3ECD0E163E6E46AEE45E4E871756A6071E6586E75766F616E72,
    0xDCEE51ED64DCEBDB5AE267D85CB653EBE661E362BEE963E76372396CF4676AD9,
    0x62E053E1EC64D4E6DBDDE161D5EA6C647269695CE078D26EED676F5761EDEDDB,
    0x62F1ECD8EA74706CE96170EFC7DF5B543CC1EC6761C0ECE6E9E1ECE763EA6FE6,
    0x73F0F0EA68F04F673B62D95B635A5869F25D676DF1604EE86A6BE060CB486346,
    0x6962DDE9E27171E5E2F25BF5F0F672F068DDE4E7D4EA7240DB6678D5F36DF571,
    0xE8E860E7585DDD663BE564F0586DF06751BEC662E7E165D0F060676456D8F0CF,
    0xEEC351EA5AE0DE695266D0E164ECE6EB726BE97769F3F7DAF770E5ECDA566972,
    0x70EF5EE5F1F7717252E8F063505061EEA762DB596669667050EC55DEDDC616DD,
    0xEBE7E457626E64EA4843496DCDE7696758DE696659D05EE4F16A6E62586A63D5,
    0x6650D270656CD7E26162E3DB6B546FD16AE0F159E6576EEB72747035DFF46E65,
    0xD972E26257E0F162EBC8E86049E7D951665466EBDB72EA62E9646125EFE4F0E2,
    0x69D67673506B69D6F0716474E7E0EAF2675D6AECD96B696A6AE4E55F43DD5E64,
    0xE268606DE762EAE8E0556967ED606E59E3D1CFE968E5E145E56D60D86A6A69E7,
    0xF26A6EF26CEAF3D2E06A61C0E4F1565E685AE5DF50D8D9ECE9EA7278ECF75FE8,
    0x73CEEBEC69E25BDBF96AF8E57665E1F65051E5E1DA63D7E1E4DF6A69EEEB51E6,
    0xE462E264EA66664B6AEBE7E1616EE841496FE86BDC6FE95AF471F5F1F3E970EF,
    0x526D68D4E5E2495F6F6DD5537973E8F1E3E94FEA63E8EADE7064E866C9E5E9E8,
    0xDECC5D5AE9E4EDE071E4725F69F476536A59E0EBDA6EC757F773756E60F45EED,
    0xD267E859DFD9706BECE961E155E4E964E0716AE9EEBAD47060ECE8F174757059,
    0x666762D070D36BE0EA6C626CE559DD5DF06ADB695CE2DBF1D969E1796AE1F364,
    0xC9F0C96AE1E7EEE3E2CF61606F5D6CE7EFD564BADACAE26942E9E1EA69645CE7,
    0xE6EC635E5D69E766BE60D064605FDFECDDEA6336E3F1E95CF163676264E8E8E5,
    0x4AED5E344BF05F5E5AE5D8E57166C959E3DFE9716D5EE96BD56EDE6065C45262,
    0xDDEDECE8EA5AEDE3E9D0DC6155E45A56F6706BF4F26A70E36FD26D51506866DB,
    0xDE6851E25F5EE9E5CED3EAE5DE6BE3E3537155EE6A1C6FF8F26E65EBEA686069,
    0x6A69645C5B48DA4164706766D1EF67D7DCC8E3D35166BF50E86867F1E16168D2,
]
CHAIN_WORD = 0xC75B8C00


def flat(v) -> list[int]:
    """The words of a nested value, in leaf order."""
    return [y for x in v for y in flat(x)] if isinstance(v, (tuple, list)) else [int(v)]


def word(fn, *args) -> list[int]:
    """``fn`` on one instance of word arguments: its output words in leaf order."""
    return flat(evaluate(bind(fn) if isinstance(fn, CompositeDefinition) else fn, *(flat(a) for a in args)))


def on_bits(fn, *instances) -> list[list[int]]:
    """``fn`` (a Boolean Definition) on instances of word arguments, each a tuple of its arguments: each instance's output
    words in leaf order."""
    args = [np.asarray([flat(inst[j]) for inst in instances], dtype=np.uint64) for j in range(len(instances[0]))]
    out = evaluate_words(fn, *args)
    return out.reshape(out.shape[0], -1).tolist()


def on(fn, name: str, **statics):
    """``fn`` (a word Definition or its Boolean version) at the device record ``name``'s atom and promotion period, each
    static it has."""
    dot, g = ATOMS[name]
    step = HopperBF16WgmmaDot16
    if getattr(fn, "word", None) is not None:
        dot, step = boolean_version(dot), boolean_version(step)
    return bind(fn, **statics, **{k: v for k, v in (("DOT", dot), ("G", g), ("STEP", step)) if k in fn.statics})


def rows(k: int, n: int, seed: int) -> list[list[int]]:
    """Rows of k FP32 words the reference forms: Gaussian at three scales, one in four words of each third row zero."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        scale = (1.0, 1e-3, 300.0)[i % 3]
        out.append([0 if i % 3 == 2 and rng.random() < 0.25 else P.f32_bits(rng.gauss(0, scale)) for _ in range(k)])
    return out


def off_domain(k: int) -> list[list[int]]:
    """Rows the reference rejects and the row units compute on: all zero, one non-zero word, and a NaN, an infinity and the
    largest finite word at a sampled position and off it."""
    base = rows(k, 1, 77)[0]
    out = [[0] * k, [0x80000000] * k, [0] * (k - 1) + [0x3F800000]]
    for w in (NAN, INF, 0xFF800000, 0x7F7FFFFF):
        for at in (0, 3):
            out.append(base[:at] + [w] + base[at + 1:])
    return out


def basis(side: int, k: int) -> list[list[int]]:
    """F_A (side 0) or F_B (side 1) of `SEED_B` as k lines at forming v1's norm."""
    return [C.sample_line(SEED_B, side, 1, t, C.LINE_NORM_V1) for t in range(k)]


def column(lines: list[list[int]], p: int) -> list[int]:
    return [line[p] for line in lines]


def e_line(i: int) -> list[int]:
    return C.sample_line(SEED_A, 0, 0, i)


def codes(k: int, seed: int) -> list[int]:
    """k E4M3 codes, no NaN code among them."""
    rng = random.Random(seed)
    return [rng.choice([c for c in range(256) if c & 0x7F != 0x7F]) for _ in range(k)]


def test_the_constants_and_the_atoms_are_pearl_cs():
    assert (F.STRIDE, F.FLOOR, F.FOUR, F.QMAX, F.LINE_FACTOR) == (C.S5_STRIDE, C.S5_FLOOR, C.S5_FOUR, C.S5_QMAX, C.S5_LINE)
    assert (F.SLICE, 2 * F.SLICE) == (C.SLICE, C.PEEL_K) and C.R == F.SLICE
    assert {n: (pc8.ATOMS[d.atom.name], d.g or 0) for n, d in DEVICES.items()} == ATOMS


@pytest.mark.parametrize("k", [1, 8, 32, 250, 256])
def test_the_row_statistics_on_words(k):
    stats = bind(F.PearlCRowStats, K=k)
    for x in rows(k, 12, k):
        assert tuple(word(stats, x)) == C.s5_stats(x)


def test_the_row_statistics_on_bits():
    xs = rows(32, 6, 5) + off_domain(32)
    assert on_bits(bind(F.PearlCRowStatsV2, K=32), *[(x,) for x in xs]) == [word(bind(F.PearlCRowStats, K=32), x) for x in xs]


def test_the_scaled_code_on_words_and_on_bits():
    betas = [C.s5_stats(x)[3] for x in rows(64, 6, 9)] + [0, 0x80000000, 0x3F800000, 0x42000000, 0x00000001]
    real = [c for c in range(256) if c & 0x7F != 0x7F]
    for beta in betas:
        assert [word(F.PearlCScaledCode, beta, c)[0] for c in real] == C.eprime(beta, real)
    every = [(beta, c) for beta in betas + [NAN, INF] for c in range(256)]
    assert on_bits(bind(F.PearlCScaledCodeV2), *every) == [word(F.PearlCScaledCode, *i) for i in every]


@pytest.mark.parametrize("name,k", [("h100", 256), ("sm120", 256), ("sm120-unpromoted", 96)])
def test_the_forming_on_words(name, k):
    """`PearlCForm` is `form_v1` on each row: its codes, E′ and statistics, α among them; a code is `PearlCFormCode`'s."""
    f, xs = basis(0, k), rows(k, 6, 11)
    es = [e_line(i) for i in range(len(xs))]
    ref = C.form_v1(xs, es, f, DEVICES[name])
    form, code = on(F.PearlCForm, name, K=k), on(F.PearlCFormCode, name)
    for i, (x, e) in enumerate(zip(xs, es)):
        got = word(form, x, e, f)
        assert got[:k] == ref.codes[i] and got[k:k + 32] == ref.ep[i]
        assert got[k + 32:] == [ref.s[i], ref.rho[i], ref.alpha[i], ref.beta[i]]
        assert [word(code, ref.alpha[i], x[t], ref.ep[i], f[t])[0] for t in (0, k - 1)] == [ref.codes[i][0], ref.codes[i][-1]]


@pytest.mark.parametrize("name", ["h100", "sm120"])
def test_a_formed_code_on_bits(name):
    f, xs = basis(1, 8), rows(8, 3, 13)
    every = [(alpha, x[t], C.eprime(beta, e_line(i)), f[t]) for i, x in enumerate(xs) for t in range(8)
             for _s, _rho, alpha, beta in [C.s5_stats(x)]]
    every += [(NAN, xs[0][0], e_line(0), f[0]), (0x43E00000, INF, e_line(0), f[0]), (0, 0, [0x7F] * 32, [0xFF] * 32)]
    assert on_bits(on(F.PearlCFormCodeV2, name), *every) == [word(on(F.PearlCFormCode, name), *i) for i in every]


def test_the_forming_on_bits():
    """A row of eight words on bits: the statistics, the 32 scaled codes and the eight formed codes, about 200,000 ANDs."""
    f = basis(0, 8)
    every = [(x, e_line(i), f) for i, x in enumerate(rows(8, 3, 15) + off_domain(8)[:5])]
    assert on_bits(on(F.PearlCFormV2, "sm120", K=8), *every) == [word(on(F.PearlCForm, "sm120", K=8), *i) for i in every]


@pytest.mark.parametrize("name,k", [("h100", 32), ("h100", 128), ("h100", 160), ("h100", 512), ("sm120", 256),
                                    ("sm120-unpromoted", 32), ("sm120-unpromoted", 160)])
def test_the_chain_on_words(name, k):
    """`pearl_c.chain`, a last window short of G atoms included (k = 32 and 160 under G = 4)."""
    chain = on(F.PearlCChain, name, K=k)
    for i in range(6):
        a, b = codes(k, 3 * i + k), codes(k, 3 * i + k + 1)
        assert word(chain, a, b) == [C.chain(a, b, DEVICES[name])]
    fa, fb = basis(0, k), basis(1, k)
    assert word(chain, column(fb, 3), column(fa, 30)) == [C.chain(column(fb, 3), column(fa, 30), DEVICES[name])]


def test_the_chain_on_the_lean_vector():
    a = [c for packed in CHAIN_A for c in packed.to_bytes(32, "little")]
    b = [c for packed in CHAIN_B for c in packed.to_bytes(32, "little")]
    assert word(on(F.PearlCChain, "h100", K=1024), a, b) == [CHAIN_WORD] == [C.chain(a, b, DEVICES["h100"])]


@pytest.mark.parametrize("name,k", [("h100", 128), ("sm120", 128), ("sm120", 32), ("sm120-unpromoted", 64)])
def test_the_chain_on_bits(name, k):
    every = [(codes(k, i), codes(k, 50 + i)) for i in range(4)] + [([0x7E] * k, [0x7E] * k), ([0x7E] * k, [0xFE] * k),
                                                                   ([0] * k, [0x80] * k), ([0x7F] * k, [0x08] * k)]
    assert on_bits(on(F.PearlCChainV2, name, K=k), *every) == [word(on(F.PearlCChain, name, K=k), *i) for i in every]


@pytest.mark.parametrize("name", ["h100", "sm120", "sm120-unpromoted"])
def test_the_gram_matrix_and_the_peel_factors_on_words(name):
    """At rank 32 and k = 128: G is `pearl_c.gram`'s, and each formed row's P_A and P_B are `peel_factors_a`'s and
    `peel_factors_b`'s."""
    k, dev = 128, DEVICES[name]
    fa, fb, xs = basis(0, k), basis(1, k), rows(k, 3, 17)
    g = [list(row) for row in C.gram(SEED_B, k, C.LINE_NORM_V1, dev)]
    assert word(on(F.PearlCGram, name, K=k, R=32), fb, fa) == flat(g)
    a = C.form_v1(xs, [e_line(i) for i in range(3)], fa, dev)
    b = C.form_v1(xs, [e_line(8 + i) for i in range(3)], fb, dev)
    pa = C.peel_factors_a(a.codes, C.scaled_noise(a, []), SEED_B, k, C.LINE_NORM_V1, dev)
    pb = C.peel_factors_b(b.codes, C.scaled_noise(b, []), SEED_B, k, C.LINE_NORM_V1, dev)
    for i in range(3):
        assert word(on(F.PearlCPeelA, name, K=k, R=32), a.codes[i], a.ep[i], fb) == pa[i]
        assert word(on(F.PearlCPeelB, name, K=k, R=32), b.codes[i], b.ep[i], fa, g) == pb[i]


@pytest.mark.parametrize("name", ["h100", "sm120"])
def test_the_forming_and_the_peel_factors_are_the_row_units(name):
    """`Pc8RowA`'s and `Pc8RowB`'s codes and P rows are `PearlCForm`'s codes and `PearlCPeelA`'s and `PearlCPeelB`'s rows
    of them, on rows the reference rejects as on those it forms, and against any G."""
    k = 128
    fa, fb = basis(0, k), basis(1, k)
    rng = random.Random(19)
    g = [[rng.choice([P.f32_bits(rng.gauss(0, 1e4)), NAN, INF, 0]) if p == q else P.f32_bits(rng.gauss(0, 1e4))
          for q in range(32)] for p in range(32)]
    units = pc8.templates(k, (1, 1), name)
    for i, x in enumerate(rows(k, 3, 21) + off_domain(k)):
        e = e_line(i)
        got = word(on(F.PearlCForm, name, K=k), x, e, fa)
        unit = word(units["row_a"], x, e, fa, fb)
        assert unit[:k] == got[:k]
        assert unit[k:k + 64] == word(on(F.PearlCPeelA, name, K=k, R=32), got[:k], got[k:k + 32], fb)
        got = word(on(F.PearlCForm, name, K=k), x, e, fb)
        unit = word(units["row_b"], x, e, fb, fa, g)
        assert unit[:k] == got[:k]
        assert unit[k:k + 64] == word(on(F.PearlCPeelB, name, K=k, R=32), got[:k], got[k:k + 32], fa, g)


@pytest.mark.parametrize("name,k,r", [("h100", 128, 1), ("sm120", 128, 1), ("h100", 32, 2), ("sm120", 32, 2)])
def test_the_gram_matrix_and_the_peel_factors_on_bits(name, k, r):
    """At the ranks circuit-check lowers whole: R lines' columns of a basis, the codes of formed rows and of any others."""
    fa, fb = [line[:r] for line in basis(0, k)], [line[:r] for line in basis(1, k)]
    gram, peel_a, peel_b = (on(d, name, K=k, R=r) for d in (F.PearlCGram, F.PearlCPeelA, F.PearlCPeelB))
    g = word(gram, fb, fa)
    assert on_bits(on(F.PearlCGramV2, name, K=k, R=r), (fb, fa)) == [g]
    formed = C.form_v1(rows(k, 2, 23), [e_line(0), e_line(1)], basis(0, k), DEVICES[name])
    every = [(row, ep[:r]) for row, ep in zip(formed.codes, formed.ep)]
    every += [(codes(k, 5), codes(r, 6)), ([0x7E] * k, [0x7F] * r), ([0x80] * k, [0x80] * r)]
    assert on_bits(on(F.PearlCPeelAV2, name, K=k, R=r), *[(c, ep, fb) for c, ep in every]) == \
        [word(peel_a, c, ep, fb) for c, ep in every]
    assert on_bits(on(F.PearlCPeelBV2, name, K=k, R=r), *[(c, ep, fa, g) for c, ep in every]) == \
        [word(peel_b, c, ep, fa, g) for c, ep in every]


def bf16_rows(n: int, seed: int) -> list[list[int]]:
    """Rows of 64 BF16 words: finite ones of every scale, then with a zero, an infinity and a NaN among them."""
    rng = random.Random(seed)
    out = [[P.f32_to_bf16(rng.gauss(0, 10.0 ** rng.randrange(-3, 4))) for _ in range(64)] for _ in range(n)]
    return out + [[0x0000, 0x8000, 0x7F80, 0x7FC0] * 16, [0x7F7F] * 64]


def test_the_peel_on_words_and_on_bits():
    """`pearl_c.peel` at rank 32 from a chain's word and from any other, and its first step alone at rank 8; on bits the
    same, on a NaN, an infinity and the largest finite word too, where the reference rejects and the Definition computes
    on."""
    assert DEVICES["h100"].peel.name in pc8.PEELS and pc8.PEELS[DEVICES["h100"].peel.name] is HopperBF16WgmmaDot16
    assert all(pc8.PEELS[DEVICES[name].peel.name] is HopperBF16WgmmaDot16 for name in ATOMS)
    peel, pa, pb = on(F.PearlCPeel, "h100", R=32), bf16_rows(4, 27), bf16_rows(4, 28)
    finite = [(P.f32_bits(c), a, b) for c, a, b in zip((0.0, 1.5, -3e6, 2e-20), pa, pb)]
    for c, a, b in finite:
        assert word(peel, c, a, b) == [C.peel(c, a, b, DEVICES["h100"])]
        assert word(on(F.PearlCPeel, "h100", R=8), c, a[:16], b[:16]) == [DEVICES["h100"].peel.step(c, a[:16], b[:16])]
    every = finite + [(NAN, pa[0], pb[0]), (INF, pa[1], pb[1]), (0x7F7FFFFF, pa[5], pb[5]), (0, pa[4], pb[4])]
    assert on_bits(on(F.PearlCPeelV2, "h100", R=32), *every) == [word(peel, *i) for i in every]


def test_the_root_and_the_negation_on_bits():
    words = [0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x40000000, 0x7F7FFFFF, INF, 0xFF800000, NAN, 0x7FC00000,
             0xBF800000] + [random.Random(25).getrandbits(32) for _ in range(64)]
    assert on_bits(bind(F.PearlCSqrtV2), *[(w,) for w in words]) == [word(F.PearlCSqrt, w) for w in words]
    halves = [0, 0x8000, 0x7FFF, 0xFFFF, 0x3F80, 0xBF80, 1]
    assert on_bits(bind(F.PearlCBf16NegV2), *[(h,) for h in halves]) == [[h ^ 0x8000] for h in halves]
    assert [word(F.PearlCBf16Neg, h) for h in halves] == [[h ^ 0x8000] for h in halves]


def test_each_boolean_binding_is_lowered_whole_and_pinned_or_calls_definitions_that_are():
    """Every Boolean Definition circuit-check binds here is within the size it lowers whole and has a pin, but for the
    forming of a whole row, whose three Definitions are: the statistics, the scaled code and the formed code."""
    pins = json.loads(PINS.read_text())["definitions"]
    over = {}
    for d in F.bindings():
        if getattr(d.defn, "word", None) is None:
            continue
        if compile_bits(d).counts["and"] <= LOWERED_WHOLE:
            assert d.id in pins, d.id
        else:
            over[d.id] = {n.fn.name for n in d.body.nodes}
    assert over and all(names == {"PearlCRowStats", "PearlCScaledCode", "PearlCFormCode"} for names in over.values()), over
    assert {i.split("_v2")[0] for i in over} == {"PearlCForm"}
    assert all(any(i.startswith(f"{name}_v2") for i in pins) for name in ("PearlCRowStats", "PearlCScaledCode", "PearlCFormCode"))
