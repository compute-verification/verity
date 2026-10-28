"""Pearl-C's per-tile cap on a circuit (`verity.ml.protocols.pouw.circuit.cap`) against the scheme's replay: a word's flags and later
+0 promotions are `pearl_c_debit.word_debit`'s, its C̃ `pearl_c.chain`'s, and only the exact certificates pass; a tile's
verdict is rev1's capOK (`pearlCTilesDevRev1`) on `chain_debit` and `debit_units` against the device's `cap` times
`credit_of`, on random tiles and adversarial ones (zero rows, absorbed groups, a total cancelled to zero, refused rows,
and the cap's boundary).  Codes are finite E4M3: the atom has no NaN operand (`verity.ml.circuits.silicon` raises), and neither does
the replay.  A row's salt-dead counts are its elements whose code has one value over a sample of P3's support, by the
scheme's own forming, and its other outputs `Pc8RowA`'s.  The split (`pc8.Shape.cap`) composes to the one-unit cap, its
h0 is `PearlTileDigest`'s on honest and edge tiles, its units' rows are their ports' words, it computes every value in
one unit, no unit of a deployed tile has more rows than C-Flock's block holds, a drawn tile's closure holds its strip,
every word, its rows' digests and sums and its tally, and a short last tile row is refused.  The per-row cap (v2,
``row_rho``) is v1's verdict and each admitted row's own capOK at ρr, in one unit and split, and `cap.served` binds the
scheme's ρr (`benchmarks/pouw/tests/test_pearlc_cap_agreement.py` holds it to the clear path's)."""
from __future__ import annotations

import itertools
import random
import struct
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest

import circuit_fixtures as CF
import pearl_c_vectors as V
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, bit_row, row_bytes
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.program import Program
from verity.ml.circuits.silicon import fp32
from verity.ml.randomness import derive
from verity.ml.protocols.pouw.circuit import cap, pc8, plan
from verity.ml.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.protocol import DomainError, Shape
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_debit as D
from verity.ml.protocols.pouw.schemes import pearl_kw as P
from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES

FINITE = [c for c in range(256) if c not in (0x7F, 0xFF)]
SMALL = [c for c in FINITE if c & 0x7F < 0x10]                     # E4M3 subnormals and the smallest normals


def _word(k: int, dev: str, a, b, cert=None):
    out = evaluate(cap.templates(k, dev)["word"], a, b, cap.certificates(a, b, dev) if cert is None else cert)
    return {"c": out[0], "flags": [bool(f) for f in out[1:1 + k // 32]], "zeros": out[1 + k // 32], "good": out[-1]}


def _words(k: int, rng: random.Random) -> dict[str, tuple[list[int], list[int]]]:
    """Random words and adversarial ones: a zero row (every atom flagged), a large first group then small ones (later
    atoms absorbed), groups that cancel (a total of +0 after a group), sparse rows, one code repeated, and E4M3's extremes."""
    rand = lambda pool: [rng.choice(pool) for _ in range(k)]  # noqa: E731
    big = [0x7E if t < 32 else rng.choice(SMALL) for t in range(k)]
    half = k // 2
    window = [rng.choice(FINITE) for _ in range(128)]
    cancel_b = [c if (t // 128) % 2 == 0 else c ^ 0x80 for t, c in enumerate(window * (k // 128))]
    sparse = [rng.choice(FINITE) if t % 37 == 0 else 0 for t in range(k)]
    return {"random-0": (rand(FINITE), rand(FINITE)), "random-1": (rand(FINITE), rand(FINITE)),
            "small": (rand(SMALL), rand(SMALL)), "zero-a": ([0] * k, rand(FINITE)), "zero-both": ([0] * k, [0x80] * k),
            "absorbed": (big, [0x7E if t < 32 else rng.choice(SMALL) for t in range(k)]),
            "cancel": (window * (k // 128), cancel_b), "sparse": (sparse, rand(FINITE)),
            "repeated": ([0x38] * k, [0x38] * half + [0xB8] * half), "extremes": ([0x7E, 0xFE] * half, [0x7E] * k)}


@pytest.mark.parametrize("dev, k", [("sm120", 256), ("sm120", 512), ("h100", 384)])
def test_the_word_is_the_debits_replay(dev, k):
    """On every word, with the prover's certificates: the flags and later +0 promotions are `word_debit`'s, C̃ the
    chain's, and good is 1; the adversarial words exercise flags, absorption and a +0 total after a group."""
    d, rng = DEVICES[dev], random.Random(f"cap-word-{dev}-{k}")
    flagged = zeroed = 0
    for name, (a, b) in _words(k, rng).items():
        got = _word(k, dev, a, b)
        assert got["good"] == 1, name
        assert (got["flags"], got["zeros"]) == D.word_debit(a, b, d), name
        assert got["c"] == C.chain(a, b, d), name
        flagged += sum(got["flags"])
        zeroed += got["zeros"]
    assert flagged and zeroed


@pytest.mark.parametrize("dev", ["sm120", "h100"])
def test_the_window_is_the_debits_masked_words(dev):
    """Each device's window primitive packs `pearl_c_debit._masked`'s words of masks 0 to 14 (mask 15 passes every atom: +0)
    on the adversarial words' first windows."""
    d = DEVICES[dev]
    window = cap.WINDOWS[d.atom.name]
    for name, (a, b) in _words(256, random.Random(f"cap-window-{dev}")).items():
        a, b = a[:128], b[:128]
        masked = D._masked([(a[s:s + 32], b[s:s + 32]) for s in range(0, 128, 32)], d)
        assert masked[15] == 0 and window.evaluate(*a, *b) == sum(masked[m] << 32 * m for m in range(15)), name


# ---- the row's salt-dead counts -------------------------------------------------------------------------------------------

#: E4M3 44 and −44: the flat E line's entries at Pearl's norm 256 (`SaltDead.flatLine`)
FLAT_E = (0x63, 0xE3)


def _one_value_over_the_support(d, acc: int, beta: int, rng: random.Random, pairs: int = 4) -> bool:
    """P3's `saltDeadP` on a sample of its support, by the scheme's own forming (`eprime` of the flat line, the device's
    atom on f32(α·x), Pearl's cast): the code has one value at every signed count (E′ flat, F's signs) and at random pairs
    whose E and F signs are both mixed."""
    plus, minus = (C.eprime(beta, [e])[0] for e in FLAT_E)
    code = lambda ep, f: P.fp8_to_f32(P.f32_to_fp8(P.f32_of_bits(d.atom.step(acc, ep, f))))  # noqa: E731
    vals = {code([plus] * 32, [0xC3] * neg + [0x43] * (32 - neg)) for neg in range(33)}
    for _ in range(pairs):
        vals.add(code([rng.choice((plus, minus)) for _ in range(32)], [rng.choice((0x43, 0xC3)) for _ in range(32)]))
    return len(vals) == 1


@pytest.mark.parametrize("dev", ["sm120", "h100"])
def test_the_salt_dead_bit_is_one_value_over_the_support(dev):
    """The device's bit, on random accumulators and flat codes of β of either sign (the code is monotone in the count only
    for β >= 0, `atomMixP_flat_mono`, and antitone for β <= 0, `atomMixP_flat_anti`; the bit reads neither), on ±0
    accumulators and β = 0: 1 iff the code has one value over the sampled support."""
    d, rng = DEVICES[dev], random.Random(f"cap-salt-dead-{dev}")
    bit = cap.SALT_DEAD[d.atom.name]
    cases = [(P.f32_bits(x), P.f32_bits(b)) for x, b in ((0.0, 0.0), (-0.0, 1.0), (0.0, -1.0), (448.0, -2.0 ** -20))]
    for _ in range(150):
        x = (-1) ** rng.randrange(2) * 2.0 ** rng.uniform(-24, 8.8)
        cases.append((P.f32_bits(x), P.f32_bits((-1) ** rng.randrange(2) * 2.0 ** rng.uniform(-24, 8))))
    seen = Counter()
    for acc, beta in cases:
        want = _one_value_over_the_support(d, acc, beta, rng)
        assert bit.evaluate(acc, C.eprime(beta, [FLAT_E[0]])[0]) == want, (hex(acc), hex(beta))
        seen[want, beta >> 31] += 1
    assert set(seen) == {(True, 0), (True, 1), (False, 0), (False, 1)}


def _row_case(k: int) -> dict[str, list[int]]:
    """The vectors' honest rows (every element salt-live: the noise is sized to make it so), one negated, one at a small
    scale, one spike of 1024 times its word at an unsampled position (admitted, the spike salt-dead), and two spikes of 2^20
    (refused, every element salt-dead)."""
    base = [C.parse_f32_row(r, k) for r in V.rows_f32("pc8-circuit", "A", 2, k)]
    scaled = lambda x, at, f: [P.f32_bits(P.f32_of_bits(w) * f) if t in at else w for t, w in enumerate(x)]  # noqa: E731
    return {"honest-0": base[0], "honest-1": base[1], "negated": [w ^ 0x80000000 for w in base[1]],
            "small": scaled(base[1], range(k), 2.0 ** -40), "spike": scaled(base[0], (1,), 1024.0),
            "spikes-refused": scaled(base[0], (1, 61), 2.0 ** 20)}


@pytest.mark.parametrize("dev", ["sm120", "h100"])
def test_the_row_is_pc8s_row_with_its_salt_dead_counts(dev):
    """`Pc8RowACap`'s codes, peel row and ok word are `Pc8RowA`'s, and each block's count is the number of its elements whose
    code has one value over the sampled support at the scheme's α and β (`form_v1`), on admitted rows and on a refused one
    (whose counts the cap never reads); dead elements occur on an admitted row."""
    k, d, rng = 128, DEVICES[dev], random.Random(f"cap-row-{dev}")
    seed = bytes(range(32))
    fa, fb = C._basis(seed, 0, k, C.LINE_NORM_V1), C._basis(seed, 1, k, C.LINE_NORM_V1)
    e = P.sample_line(seed, 0, 0, 5)
    flat = lambda lines: [c for ln in lines for c in ln]  # noqa: E731
    dead = {}
    for name, x in _row_case(k).items():
        out = evaluate(cap.templates(k, dev)["row_a"], x, e, flat(fa), flat(fb))
        plain = evaluate(pc8.templates(k, pc8.TILE, dev)["row_a"], x, e, flat(fa), flat(fb))
        assert out[:len(plain)] == plain, name
        formed = C.form_v1([x], [e], fa, d)
        alpha, beta = formed.alpha[0], formed.beta[0]
        bits = [_one_value_over_the_support(d, fp32.mul(alpha, x[t]), beta, rng) for t in range(k)]
        assert list(out[len(plain):]) == [sum(bits[s:s + 32]) for s in range(0, k, 32)], name
        dead[name, plain[-1]] = sum(bits)
    assert dead == {("honest-0", 1): 0, ("honest-1", 1): 0, ("negated", 1): 0, ("small", 1): 0, ("spike", 1): 1,
                    ("spikes-refused", 0): k}


def test_the_certificates_are_each_templates_advice():
    """`ADVICE` (`pc8.PORTS`' class) names a parameter of the word, of the tile and of the split's word unit, and is the
    word unit's advice in `PORTS`: the certificates are in the unit's io, which C-Flock proves as private inputs."""
    tp = cap.templates(256, "sm120", (2, 2))
    for t in (tp["word"], tp["tile"], cap.unit_templates(256, "sm120", (2, 2))["word"]):
        assert set(cap.ADVICE) <= {name for name, _t in t.signature()[0]}, t.id
    assert cap.PORTS["word"]["advice"] == cap.ADVICE


def _keyed(w: int, dk: int) -> int:
    return D._word(D._key(w) + dk)


@pytest.mark.parametrize("k", [256, 512])
def test_only_the_exact_certificates_pass(k):
    """Each lo or hi moved one FP32 value either way, lo and hi swapped, the interval widened or narrowed, a +0 written as
    −0 where it is one, and random words: good is 0 for every certificate but the prover's (and the same value
    written as the other zero), so a passing word's flags are the replay's whatever the prover sends."""
    rng = random.Random(f"cap-cert-{k}")
    for name, (a, b) in _words(k, rng).items():
        honest = cap.certificates(a, b)
        tampered = []
        for at in range(len(honest)):
            for dk in (-1, 1, -1000, 1000):
                c = list(honest)
                c[at] = _keyed(c[at], dk)
                tampered.append(c)
        for g in range(0, len(honest), 2):
            c = list(honest)
            c[g], c[g + 1] = c[g + 1], c[g]
            if c != honest:
                tampered.append(c)
        tampered += [[rng.getrandbits(32) for _ in honest] for _ in range(3)]
        for c in tampered:
            assert _word(k, "sm120", a, b, cert=c)["good"] == 0, (name, c, honest)
        zero = [0x80000000 if w == 0 else w for w in honest]
        if zero != honest:
            assert _word(k, "sm120", a, b, cert=zero)["good"] == 1, name


# ---- the tile -------------------------------------------------------------------------------------------------------------

def _rev1_cap(dev: str, k: int, n: int, rho: int, rows: int, cols: int) -> Fraction:
    """ρ·creditDevRev1·tileShare: the device's `cap` (ρ = 1/rho here) times `credit_of` (rev1's credit), at an m that
    cancels."""
    d, m = replace(DEVICES[dev], cap=Fraction(1, rho)), 64
    return d.cap * C.PearlC(forming="v1", device=d).credit_of(Shape(m, k, n)) * rows * cols / (m * n)


def _reference(dev: str, k: int, n: int, rho: int, ac, bc, ok, sd) -> bool:
    """rev1's per-tile capOK on the replay: `chain_debit` and `debit_units` over the passing rows' words and every column
    (`pearlCTilesDevRev1`), False where the replay raises."""
    d = DEVICES[dev]
    rows = [i for i, o in enumerate(ok) if o]
    try:
        atoms, zeros = D.chain_debit([ac[i] for i in rows], bc, d)
        flags = {(i, j): D.word_flags(ac[i], bc[j], d) for i in rows for j in range(len(bc))}
    except P.Reject:
        return False
    elems = sum(32 if all(flags[i, j][t] for j in range(len(bc))) else sd[i][t] for i in rows for t in range(k // 32))
    debit = D.debit_units(atoms, Fraction(len(bc) * elems, n), device=d, zeros=zeros)
    return debit <= _rev1_cap(dev, k, n, rho, len(rows), len(bc))


def _peel_rows(k: int, tile):
    """P_A and P_B: random BF16 words."""
    tm, tn = tile
    rng = random.Random(f"cap-peel-{k}-{tile}")
    pa = [[rng.getrandbits(16) & 0x3FFF for _ in range(pc8.PEEL)] for _ in range(tm)]
    return pa, [[rng.getrandbits(16) & 0x3FFF for _ in range(pc8.PEEL)] for _ in range(tn)]


def _tile(k: int, tile, n: int, rho: int, ac, bc, ok, sd, cert=None, dev: str = "sm120", row_rho: int | None = None):
    """(C̃, U, verdict) of `Pc8TileCap` (v2 with ``row_rho``), and `Pc8Tile`'s (C̃, U) on the same rows (P_A, P_B random
    BF16 words)."""
    tm, tn = tile
    pa, pb = _peel_rows(k, tile)
    if cert is None:
        cert = [[cap.certificates(ac[i], bc[j], dev) for j in range(tn)] for i in range(tm)]
    flat = lambda rows: [x for r in rows for x in r]  # noqa: E731
    out = evaluate(cap.templates(k, dev, tile, n, rho, row_rho)["tile"], flat(ac), flat(bc), flat(pa), flat(pb),
                   [int(o) for o in ok], flat(sd), [w for r in cert for c in r for w in c])
    plain = evaluate(pc8.templates(k, tile, dev)["tile"], flat(ac), flat(bc), flat(pa), flat(pb))
    return out[:-1], plain, out[-1]


def _random_tile(rng, k, tm, tn, pool=FINITE):
    rows = lambda count: [[rng.choice(pool) for _ in range(k)] for _ in range(count)]  # noqa: E731
    return rows(tm), rows(tn)


@pytest.mark.parametrize("k, tile, n, rho", [(256, (2, 2), 2, 1000), (256, (2, 2), 6, 3), (512, (1, 3), 3, 2),
                                             (256, (3, 1), 4, 40)])
def test_the_tile_is_pc8s_and_its_verdict_rev1s_cap(k, tile, n, rho):
    """Random and adversarial tiles, random admissions and salt-dead counts: C̃ and U are `Pc8Tile`'s, and the verdict is
    rev1's capOK on `pearl_c_debit`'s replay; both verdicts occur."""
    rng = random.Random(f"cap-tile-{k}-{tile}-{n}-{rho}")
    tm, tn = tile
    seen = set()
    for trial in range(10):
        pool = (FINITE, SMALL)[trial % 2]
        ac, bc = _random_tile(rng, k, tm, tn, pool)
        if trial % 3 == 2:
            ac[0] = [0] * k
        if trial % 5 == 4:
            bc[-1] = [rng.choice(SMALL) if t >= 32 else 0x7E for t in range(k)]
        ok = [rng.random() < 0.8 for _ in range(tm)]
        sd = [[rng.choice([0] * 30 + [1, 32]) if trial % 4 == 3 else 0 for _ in range(k // 32)] for _ in range(tm)]
        cu, plain, verdict = _tile(k, tile, n, rho, ac, bc, ok, sd)
        assert cu == plain
        want = _reference("sm120", k, n, rho, ac, bc, ok, sd)
        assert verdict == want, trial
        seen.add(want)
    assert seen == {True, False}


def test_the_verdict_turns_at_the_cap():
    """With no flagged atom, the forming term alone moves the debit: the largest salt-dead count under the cap passes and
    one more element fails, on the floors the circuit's integer form takes."""
    k, tile, n, rho = 256, (2, 2), 4, 7
    rng = random.Random("cap-boundary")
    ac, bc = _random_tile(rng, k, *tile)
    ok = [True, True]
    assert not any(any(D.word_flags(a, b, DEVICES["sm120"])) for a in ac for b in bc)
    limit = _rev1_cap("sm120", k, n, rho, 2, 2)
    most = int(limit * n / (DEVICES["sm120"].prices.forming_credited * 2))
    assert 0 < most < 2 * k
    for total, want in ((most, True), (most + 1, False)):
        sd = [[0] * (k // 32) for _ in range(2)]
        for t in range(total):
            sd[t // k][(t % k) // 32] += 1
        assert _reference("sm120", k, n, rho, ac, bc, ok, sd) is want
        assert _tile(k, tile, n, rho, ac, bc, ok, sd)[2] == want


def test_a_refused_row_is_left_out():
    """A row whose ok word is 0 is no row of the cap: a zero row there (every atom of its words flagged, a +0 total after
    the first group, all its elements unneeded), with or without its certificates, leaves the verdict as it is without the
    row; admitted, the zero row's debit is over the cap."""
    k, tile, n, rho = 256, (2, 2), 2, 1000
    rng = random.Random("cap-refused")
    ac, bc = _random_tile(rng, k, *tile)
    sd = [[0] * (k // 32) for _ in range(2)]
    assert _tile(k, tile, n, rho, ac, bc, [True, False], sd)[2] == 1
    rows = [ac[0], [0] * k]
    for junk in (None, [0, 0]):
        cert = [[cap.certificates(r, b) if junk is None or i == 0 else junk for b in bc] for i, r in enumerate(rows)]
        assert _tile(k, tile, n, rho, rows, bc, [True, False], sd, cert=cert)[2] == 1
    assert _tile(k, tile, n, rho, rows, bc, [True, True], sd)[2] == 0 == _reference("sm120", k, n, rho, rows, bc, [1, 1], sd)


def test_a_wrong_certificate_fails_the_tile():
    """One word's certificate moved by one FP32 value fails the verdict of a tile that passes with the prover's."""
    k, tile, n, rho = 256, (2, 2), 2, 1000
    rng = random.Random("cap-tile-cert")
    ac, bc = _random_tile(rng, k, *tile)
    ok, sd = [True, True], [[0] * (k // 32) for _ in range(2)]
    cert = [[cap.certificates(ac[i], bc[j]) for j in range(2)] for i in range(2)]
    assert _tile(k, tile, n, rho, ac, bc, ok, sd, cert=cert)[2] == 1
    cert[1][0] = [_keyed(cert[1][0][0], 1), cert[1][0][1]]
    assert _tile(k, tile, n, rho, ac, bc, ok, sd, cert=cert)[2] == 0


def test_the_deployed_shape_fits_and_a_larger_one_is_refused():
    """The cap's integer form fits 31 bits at the deployed 64 × 64 tile, K = 8,192 and a unit of 16,384 columns; a unit of
    64 × 2^20 + 64 columns, whose shares need wider sums, and fewer than two promotion windows are refused."""
    assert [cap.templates(256, dev)["tile"].bindings["RHO"] for dev in ("sm120", "h100")] == [1000, 400]
    atom, zero, n_g, elem_g, l_max, table = cap._cap_constants(8192, 64, 64, 16384, 1000, "sm120")
    assert (atom, zero) == (136, 32) and len(table) == 65 and table[0] == 0 < l_max
    with pytest.raises(DomainError):
        cap._cap_constants(8192, 64, 64, 64 * (2 ** 20 + 1), 1000, "sm120")
    with pytest.raises(DomainError):
        cap.templates(128, "sm120")["word"].signature()


# ---- the split ------------------------------------------------------------------------------------------------------------

SPLIT = pc8.Shape(4, 256, 4, tile=(2, 2), dev="sm120", hidden=True, cap=True)


def _split(k: int, tile, n: int, rho: int, ac, bc, ok, sd, cert=None, dev: str = "sm120", row_rho: int | None = None):
    """The split's units on one tile, composed as `Pc8CallCap` composes them: (each word unit's outputs by row and column,
    the strip's, each tile row's digests and sums, the tile digest, the tally's verdict); P_A and P_B are `_tile`'s.  With
    ``row_rho`` the tally is v2's, with the per-row cap."""
    tm, tn = tile
    tp, (pa, pb) = cap.unit_templates(k, dev, tile, n, rho, row_rho), _peel_rows(k, tile)
    if cert is None:
        cert = [[cap.certificates(ac[i], bc[j], dev) for j in range(tn)] for i in range(tm)]
    words = [[evaluate(tp["word"], ac[i], pa[i], bc[j], pb[j], cert[i][j]) for j in range(tn)] for i in range(tm)]
    fb = cap.flag_bits(k)
    field = lambda row, lo, hi: [x for w in row for x in w[lo:hi]]                # noqa: E731
    strip = evaluate(tp["strip"], [int(o) for o in ok])
    chunks = [evaluate(tp["row_digest"], field(row, 0, 1), field(row, 1, 2)) for row in words]
    sums = [evaluate(tp["row_tally"], field(row, 2, 2 + fb), field(row, 2 + fb, 3 + fb), field(row, 3 + fb, 4 + fb),
                     field(row, 4 + fb, 5 + fb), sd[i]) for i, row in enumerate(words)]
    digest = evaluate(tp["tile"], [x for ch in chunks for x in ch])
    verdict, = evaluate(tp["tally"], *([s[f] for s in sums] for f in range(4)), list(strip[:cap.STRIP_BITS]),
                        strip[cap.STRIP_BITS], list(strip[cap.STRIP_BITS + 1:]))
    return words, strip, chunks, sums, digest, verdict


def _cu(words) -> list[int]:
    return [w[0] for row in words for w in row] + [w[1] for row in words for w in row]


@pytest.mark.parametrize("k, tile, n, rho", [(256, (2, 2), 2, 1000), (256, (2, 2), 6, 3), (256, (3, 1), 4, 40)])
def test_the_split_units_are_the_tile_cap(k, tile, n, rho):
    """On `test_the_tile_is_pc8s_and_its_verdict_rev1s_cap`'s tiles, the split's units composed: the word units' C̃ and U
    are `Pc8Tile`'s, the tile unit's digest over its rows' digests is `pearl_c.leaf` of them, and the tally's verdict over
    its rows' sums is `Pc8TileCap`'s and rev1's capOK on the replay; both verdicts occur."""
    rng = random.Random(f"cap-tile-{k}-{tile}-{n}-{rho}")
    tm, tn = tile
    seen = set()
    for trial in range(10):
        pool = (FINITE, SMALL)[trial % 2]
        ac, bc = _random_tile(rng, k, tm, tn, pool)
        if trial % 3 == 2:
            ac[0] = [0] * k
        if trial % 5 == 4:
            bc[-1] = [rng.choice(SMALL) if t >= 32 else 0x7E for t in range(k)]
        ok = [rng.random() < 0.8 for _ in range(tm)]
        sd = [[rng.choice([0] * 30 + [1, 32]) if trial % 4 == 3 else 0 for _ in range(k // 32)] for _ in range(tm)]
        words, _strip, _chunks, _sums, digest, verdict = _split(k, tile, n, rho, ac, bc, ok, sd)
        _cu_cap, plain, want = _tile(k, tile, n, rho, ac, bc, ok, sd)
        assert _cu(words) == list(plain)
        assert tuple(digest) == CF.tile_digests(tile, [plain[i * tn:(i + 1) * tn] for i in range(tm)],
                                                [plain[(tm + i) * tn:(tm + i + 1) * tn] for i in range(tm)])
        assert verdict == want == _reference("sm120", k, n, rho, ac, bc, ok, sd), trial
        seen.add(want)
    assert seen == {True, False}


def test_the_split_tally_leaves_a_refused_row_out_and_fails_a_wrong_certificate():
    """`test_a_refused_row_is_left_out` and `test_a_wrong_certificate_fails_the_tile` on the split: a refused zero row
    with junk certificates leaves the verdict as it is, admitted it is over the cap, and one certificate moved by one FP32
    value fails the tally of a tile that passes with the prover's."""
    k, tile, n, rho = 256, (2, 2), 2, 1000
    rng = random.Random("cap-refused")
    ac, bc = _random_tile(rng, k, *tile)
    sd = [[0] * (k // 32) for _ in range(2)]
    assert _split(k, tile, n, rho, ac, bc, [True, False], sd)[-1] == 1
    rows = [ac[0], [0] * k]
    junk = [[cap.certificates(r, b) if i == 0 else [0, 0] for b in bc] for i, r in enumerate(rows)]
    assert _split(k, tile, n, rho, rows, bc, [True, False], sd, cert=junk)[-1] == 1
    assert _split(k, tile, n, rho, rows, bc, [True, True], sd)[-1] == 0
    cert = [[cap.certificates(ac[i], bc[j]) for j in range(2)] for i in range(2)]
    cert[1][0] = [_keyed(cert[1][0][0], 1), cert[1][0][1]]
    assert _split(k, tile, n, rho, ac, bc, [True, True], sd, cert=cert)[-1] == 0


# ---- the per-row cap (v2) -------------------------------------------------------------------------------------------------

def _row_reference(dev: str, k: int, n: int, rho: int, row_rho: int, ac, bc, ok, sd) -> bool:
    """`_reference`, and every admitted row's own debit over the tile's columns at most ρr = 1/row_rho times its cells'
    share of the credit (`pearlCTilesDevRev1RowCapAt`): rev1's cap on that row alone."""
    return _reference(dev, k, n, rho, ac, bc, ok, sd) and all(
        _reference(dev, k, n, row_rho, [ac[i]], bc, [True], [sd[i]]) for i, o in enumerate(ok) if o)


def flag_words(ac, bc, at) -> None:
    """For each (i, j, t) of ``at``, zero A row i's elements 16..31 of atom block t and B row j's 0..15, so word (i, j)'s
    atom t is a zero product, which the debit flags; distinct t keep the words apart."""
    for i, j, t in at:
        for e in range(32):
            if e < 16:
                bc[j][32 * t + e] = 0
            else:
                ac[i][32 * t + e] = 0


def _row_flags(dev: str, ac, bc) -> list[int]:
    return [sum(sum(D.word_flags(a, b, DEVICES[dev])) for b in bc) for a in ac]


def _unflagged_tile(dev: str, seed: str, k: int, tm: int, tn: int):
    """The first of ``seed``'s random tiles with no word flagged, so `flag_words` sets every flag."""
    for s in itertools.count():
        ac, bc = _random_tile(random.Random(f"{seed}-{s}"), k, tm, tn)
        if not any(_row_flags(dev, ac, bc)):
            return ac, bc


@pytest.mark.parametrize("k, tile, n, rho, row_rho", [(256, (2, 2), 2, 1000, 250), (256, (3, 1), 4, 40, 10),
                                                      (256, (2, 2), 6, 3, 2)])
def test_the_row_capped_tile_is_v1_and_each_admitted_rows_cap(k, tile, n, rho, row_rho):
    """`test_the_tile_is_pc8s_and_its_verdict_rev1s_cap`'s tiles under `Pc8TileCap` v2 and its split (`Pc8CapTally` v2):
    C̃ and U are `Pc8Tile`'s, and the verdict is rev1's capOK on the replay and every admitted row's own debit within ρr
    = 1/row_rho of its cells' credit; both verdicts occur."""
    rng = random.Random(f"cap-row-tile-{k}-{tile}-{n}-{rho}-{row_rho}")
    tm, tn = tile
    seen = set()
    for trial in range(10):
        ac, bc = _random_tile(rng, k, tm, tn, (FINITE, SMALL)[trial % 2])
        if trial % 3 == 2:
            ac[0] = [0] * k
        ok = [rng.random() < 0.8 for _ in range(tm)]
        sd = [[rng.choice([0] * 30 + [1, 32]) if trial % 4 == 3 else 0 for _ in range(k // 32)] for _ in range(tm)]
        cu, plain, verdict = _tile(k, tile, n, rho, ac, bc, ok, sd, row_rho=row_rho)
        assert cu == plain
        want = _row_reference("sm120", k, n, rho, row_rho, ac, bc, ok, sd)
        assert verdict == want == _split(k, tile, n, rho, ac, bc, ok, sd, row_rho=row_rho)[-1], trial
        seen.add(want)
    assert seen == {True, False}


#: GLM-5.3's down projection's k on the H100 record, at its n, in tiles of 8 rows and 16 columns: a cell's credit is 420
#: units and a flagged atom 40, so a row's cap is 16.8 units at ρ = 1/400 and 67.2 at 4ρ, and the tile's 134.4 at ρ
ROW_CAP = dict(k=256, tile=(8, 16), n=6144, rho=400, dev="h100")


@pytest.mark.parametrize("split", [False, True])
def test_the_per_row_cap_refuses_an_admitted_row_over_its_cap_alone(split):
    """At ROW_CAP, rows with words flagged by construction (`flag_words`): one flag in a row (40 units) is within 4ρ and
    over ρ, so `Pc8TileCap` v2 at ρr = 4ρ accepts it and at ρr = ρ refuses it; two flags in a row (80) are over 4ρ and
    within the tile's ρ, so v1 accepts the tile and v2 refuses it, and accepts it once that row is refused; one flag in
    each of four rows (160) is over the tile's ρ, which v1 and v2 both refuse.  Each verdict is the replay's."""
    k, tile, n, rho, dev = ROW_CAP["k"], ROW_CAP["tile"], ROW_CAP["n"], ROW_CAP["rho"], ROW_CAP["dev"]
    tm, tn = tile
    sd = [[0] * (k // 32) for _ in range(tm)]
    run = _split if split else lambda *a, **kw: (_tile(*a, **kw)[2],)
    cases = {"one": ([(2, 5, 1)], [1 if i == 2 else 0 for i in range(tm)]),
             "two": ([(5, 3, 2), (5, 9, 4)], [2 if i == 5 else 0 for i in range(tm)]),
             "four": ([(i, 2 * i, i + 1) for i in range(4)], [1] * 4 + [0] * 4)}
    want = {("one", 100): 1, ("one", 400): 0, ("two", 100): 0, ("two", 400): 0, ("four", 100): 0, ("four", 400): 0}
    for name, (at, flags) in cases.items():
        ac, bc = _unflagged_tile(dev, f"cap-row-{name}", k, tm, tn)
        flag_words(ac, bc, at)
        assert _row_flags(dev, ac, bc) == flags, name
        ok = [True] * tm
        v1 = run(k, tile, n, rho, ac, bc, ok, sd, dev=dev)[-1]
        assert v1 == _reference(dev, k, n, rho, ac, bc, ok, sd) == (name != "four"), name
        for row_rho in (100, 400):
            got = run(k, tile, n, rho, ac, bc, ok, sd, dev=dev, row_rho=row_rho)[-1]
            assert got == want[name, row_rho] == _row_reference(dev, k, n, rho, row_rho, ac, bc, ok, sd), (name, row_rho)
        if name == "two":
            ok[5] = False
            assert run(k, tile, n, rho, ac, bc, ok, sd, dev=dev, row_rho=100)[-1] == 1


def test_the_served_shape_takes_the_schemes_row_cap():
    """`cap.served`: `pearl-c-h100-v1-h3` at its census k = 256 is `Pc8CallCap` v2 at ρr = 4ρ (RHOR = 100) and at k =
    1,024 at ρr = ρ (400), as `PearlC.row_cap_at`; `pearl-c-sm120-v1-h3` at ρ (1,000); -h0, which caps the tile alone,
    is v1 and refuses k = 256.  A per-row cap without the cap, or not 1/RHOR, is refused; v1's Programs are unchanged."""
    h100, sm120 = DEVICES["h100"], DEVICES["sm120"]
    h3, h0 = C.PearlC(forming="v1", device=h100, hashing="h3"), C.PearlC(forming="v1", device=h100)
    for scheme, k, rr in ((h3, 256, 100), (h3, 1024, 400), (C.PearlC(forming="v1", device=sm120, hashing="h3"), 1024, 1000),
                          (h0, 1024, None)):
        shape = cap.served(scheme, Shape(64, k, 6144))
        assert (shape.row_rho, shape.tile, shape.dev, shape.hidden, shape.cap) == (rr, (64, 64), scheme.device.name, True, True)
        for build, v1, v2 in ((pc8.call, cap.Pc8CallCapDef, cap.Pc8CallCapDefV2),
                              (pc8.check, cap.Pc8CheckCapDef, cap.Pc8CheckCapDefV2)):
            t = build(shape)
            assert t.defn is (v1 if rr is None else v2) and t.bindings.get("RHOR") == rr, (scheme.name, k)
        assert pc8.call(replace(shape, row_rho=None)).id == pc8.call(shape).id.replace("_v2", "_v1").replace(f",RHOR={rr}", "")
    with pytest.raises(DomainError):
        cap.served(h0, Shape(64, 256, 6144))
    with pytest.raises(DomainError):
        cap.served(C.PearlC(forming="v1", device=h100, hashing="h3", prices=sm120.prices), Shape(64, 256, 6144))
    with pytest.raises(DomainError):
        replace(SPLIT, cap=False, row_rho=250).check()
    with pytest.raises(DomainError):
        cap.templates(256, "sm120", row_rho=0)


#: FP32 words at the edges of the digest's input: ±0, ±inf, NaNs, subnormals, the largest finite, one, all ones
EDGE = (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x00000001, 0x807FFFFF, 0x7F7FFFFF,
        0x3F800000, 0xFFFFFFFF)


def _h0(tile, c, u):
    """The split's h0 on C̃ and U (TM rows of TN words): each row's `Pc8CapRowDigest`, then `Pc8CapTile` over them."""
    tp = cap.unit_templates(256, "sm120", tile)
    chunks = [evaluate(tp["row_digest"], list(cr), list(ur)) for cr, ur in zip(c, u)]
    return evaluate(tp["tile"], [x for ch in chunks for x in ch]), chunks


@pytest.mark.parametrize("tile", [(1, 1), (2, 31), (1, 32), (3, 33), (2, 64), (64, 64)])
def test_the_split_h0_is_pearl_tile_digests_on_edge_tiles(tile):
    """On words fed to the row digests directly (half `EDGE`'s, half random bits), at TN on either side of a 32-column
    chunk and at the deployed 64 × 64: the split's h0 is `PearlTileDigest`'s and `pearl_c.leaf`'s, and each row's
    cap-chunks words are `pearl_c.message_digest`'s of its chunks, then zeros."""
    tm, tn = tile
    rng = random.Random(f"cap-h0-{tile}")
    word = lambda: rng.choice(EDGE) if rng.random() < 0.5 else rng.getrandbits(32)  # noqa: E731
    c, u = ([[word() for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    h0, chunks = _h0(tile, c, u)
    assert h0 == evaluate(L.tile_digest(tm, tn), [x for r in c for x in r], [x for r in u for x in r])
    assert h0 == CF.tile_digests(tile, c, u)
    for cr, ur, ch in zip(c, u, chunks):
        want = b"".join(C.message_digest(struct.pack(f"<{len(cr[h:h + 32])}I", *cr[h:h + 32]) +
                                         struct.pack(f"<{len(ur[h:h + 32])}I", *ur[h:h + 32])) for h in range(0, tn, 32))
        assert L.digest_bytes(ch) == want + bytes(2 * L.CAP_ROW_WORDS - len(want))


def test_the_split_units_rows_are_their_ports_words():
    """Each output row of the split's units is its ports' bits (`rowleaf.bit_row`) and its schema's words (`leaves`):
    the word unit's C̃ ‖ U and flags (padded with zero bits) ‖ count ‖ later +0 promotions ‖ good, a row digest's chunk
    digests (the second zero at TN <= 32) ‖ zero words, a row tally's sums of its words' counts and promotions ‖
    forming elements ‖ good, and the strip's admission bits (padded to 64) ‖ bound ‖ zero words, its bound
    `_cap_constants`' table at the passing rows' count; the chunk digests' and the strip's rows are each 64 words, one
    whole SHA-512 block."""
    k, tile, n, rho = 256, (2, 2), 2, 1000
    rng = random.Random("cap-split-rows")
    ac, bc = _random_tile(rng, k, *tile)
    words, strip, chunks, sums, _digest, _verdict = _split(k, tile, n, rho, ac, bc, [True, False], [[0] * (k // 32)] * 2)
    fb = cap.flag_bits(k)
    for w in (x for row in words for x in row):
        c, u, flags, atoms, zeros, good = w[0], w[1], list(w[2:2 + fb]), w[2 + fb], w[3 + fb], w[4 + fb]
        assert flags[k // 32:] == [0] * (fb - k // 32) and atoms == sum(flags) and good == 1
        for row, cols, want in (("cap-word", (([c, u], 32),), L.words_cap_word(c, u)),
                                ("cap-flags", ((flags, 1), ([atoms, zeros], 32), ([good], 16)),
                                 L.words_cap_flags(flags[:k // 32], atoms, zeros, good))):
            assert bit_row(*cols) == (row_bytes(want, 16), 16 * L.ROW_WORDS[row](k)), row
    for row, ch, (atoms, zeros, elems, good) in zip(words, chunks, sums):
        assert ch[8:] == (0,) * (cap.CHUNK_WORDS - 8)
        assert bit_row((list(ch), 32)) == (row_bytes(L.words_cap_chunks([L.digest_bytes(ch[:8])]), 16),
                                           16 * L.ROW_WORDS["cap-chunks"](k))
        assert (atoms, zeros) == (sum(w[2 + fb] for w in row), sum(w[3 + fb] for w in row))
        assert bit_row(([atoms, zeros, elems], 32), ([good], 16)) == (
            row_bytes(L.words_cap_sums(atoms, zeros, elems, good), 16), 16 * L.ROW_WORDS["cap-sums"](k))
    mask, bound, tail = list(strip[:cap.STRIP_BITS]), strip[cap.STRIP_BITS], list(strip[cap.STRIP_BITS + 1:])
    assert mask == [1, 0] + [0] * (cap.STRIP_BITS - 2) and bound == cap._cap_constants(k, *tile, n, rho, "sm120")[-1][1]
    assert tail == [0] * cap.STRIP_TAIL
    assert bit_row((mask, 1), ([bound], 32), (tail, 16)) == (row_bytes(L.words_cap_rows([1, 0], bound), 16),
                                                            16 * L.ROW_WORDS["cap-rows"](k))
    assert L.ROW_WORDS["cap-chunks"](k) == L.ROW_WORDS["cap-rows"](k) == 64


@pytest.mark.parametrize("row_rho", [None, 250])
@pytest.mark.parametrize("role", ["row_b", "row_a", "strip", "word", "row_digest", "row_tally", "tile", "tally"])
def test_the_ports_are_each_split_units_committed_rows(role, row_rho):
    """`PORTS` lists every parameter and output of each split unit once (with the per-row cap too, whose tally is v2's):
    its rows, the anchored inputs' regions, the word unit's certificates as advice and the tally's verdict as its one
    public output, in no row; its statement rows are its rows, inputs as many as it reads, outputs once each."""
    tp = cap.unit_templates(SPLIT.k, SPLIT.dev, SPLIT.tile, SPLIT.n, row_rho=row_rho)
    ports, reads = cap.PORTS[role], cap.reads(SPLIT.tile)[role]
    CF.check_ports(tp[role], ports, pc8.DERIVATIONS, SPLIT.k, reads)
    rows = L.statement_rows(ports, SPLIT.k, reads)
    outs = [r for r, _role, ps in ports["rows"] if all(isinstance(p, int) for p in ps)]
    assert len(rows["row_schemas"]) == sum(reads.get(ps[0], 1) for _r, _role, ps in ports["rows"]
                                           if all(isinstance(p, str) for p in ps)) + len(outs)
    assert ports.get("public", ()) == ((0,) if role == "tally" else ())


@pytest.mark.parametrize("row_rho", [None, 250])
@pytest.mark.parametrize("side", ["check", "weight", "call"])
def test_the_split_computes_every_value_in_exactly_one_unit(side, row_rho):
    """Under ``cap`` (with or without the per-row cap) the call is one unit per row of A, per strip, per word, per tile row
    (its digests and its sums), per tile and per tally (and the check adds a unit per row of B); `validate_unit_cut`
    certifies every gate once with no recompute, and the partition object over the split's templates verifies.  Each
    strip's admission bits and bound are its own unit's, since a strip's tallies all read them."""
    shape = replace(SPLIT, row_rho=row_rho)
    fn = {"check": pc8.check, "weight": pc8.weight, "call": pc8.call}[side](shape)
    prog, q = Program(fn), pc8.query(shape, side)
    us = pc8.units(fn)
    m, n, (tm, tn) = shape.m, shape.n, shape.tile
    full = {"row_b": n, "row_a": m, "strip": m // tm, "word": m * n, "row_digest": m * n // tn, "row_tally": m * n // tn,
            "tile": m * n // (tm * tn), "tally": m * n // (tm * tn)}
    assert Counter(u.role for u in us) == {r: full[r] for r in cap.ROLES[side]}
    tp = cap.unit_templates(shape.k, shape.dev, shape.tile, shape.n, row_rho=row_rho)
    assert [u.template for u in us] == [PO.template_id(tp[u.role]) for u in us]
    assert PO.verify(PO.build(prog, q), prog).ok
    _g, _owner, verdict = pc8.unit_cut(fn)
    assert verdict.ok, verdict.codes
    assert verdict.detail["certified"] == verdict.detail["computed_gates"]


def test_no_unit_of_a_deployed_tiles_closure_has_more_than_66_rows():
    """C-Flock's composer gives each row a unit reads or writes an hm96 slot of 2^18 bits, a unit at most 2^27 bits
    (`circuit.K_MAX`, so at most 256 rows), and refuses a unit none of whose rows hashes a SHA-512 block before its last.
    At the deployed 64 × 64 tile and K = 8,192 a drawn tile's closure is 4,291 units (its 64 rows of A, its strip, 4,096
    word units, 64 row digests and 64 row tallies, the tile and the tally), none with more than 66 rows (the row tally's;
    128 slots, 2^25 bits), and each with a row-seg row or a v2 row past one block."""
    shape = pc8.Shape(64, 8192, 64, tile=(64, 64), dev="sm120", hidden=True, cap=True)
    lay = plan.tile_layout(shape, 1)
    (closure,) = lay.closure.values()
    roles = Counter(lay.units[i].role for i in closure)
    assert roles == {"tile": 1, "row_a": 64, "strip": 1, "word": 4096, "row_digest": 64, "row_tally": 64, "tally": 1}
    reads, sizes = cap.reads(shape.tile), {}
    for role in roles:
        rows = L.statement_rows(cap.PORTS[role], shape.k, reads[role])["row_schemas"]
        sizes[role] = len(rows)
        assert any(s == SCHEMA_HM96_SHA512_ROW_SEG or -(-bits // 8) + 17 > 128 for s, bits in rows), role
    assert sizes == {"tile": 65, "row_a": 4, "strip": 65, "word": 4, "row_digest": 65, "row_tally": 66, "tally": 65}


@pytest.mark.parametrize("k", [4096, 8192, 12288, 14336])
def test_every_row_of_a_deployed_tile_is_whole_blocks_or_a_short_v2_row(k):
    """At each served k, every row a split unit reads or writes at the deployed 64 × 64 tile is one C-Flock's widened
    proved scope takes: its bytes whole SHA-512 blocks (a row-seg row's segments then end on a block, and a v2 row hashes
    a block before its last), or a v2 row that ends in its first block with SHA-512's 17 bytes of padding.  The strip's
    and the chunk digests' rows are the ones padded to a block for it."""
    reads = cap.reads((64, 64))
    for role, ports in cap.PORTS.items():
        for s, bits in L.statement_rows(ports, k, reads[role])["row_schemas"]:
            n = bits // 8
            assert bits % 16 == 0 and 0 < n and (n % 128 == 0 or s != SCHEMA_HM96_SHA512_ROW_SEG and n + 17 <= 128), \
                (role, s, bits)


def _call_inputs(shape: pc8.Shape, name: str):
    """A call's and its weight's inputs: the vectors' honest rows and the anchors from fixed seeds."""
    m, k, n = shape.m, shape.k, shape.n
    seed_a, seed_b = bytes(range(32)), bytes(range(32, 64))
    words = lambda side, count: [w for r in V.rows_f32(name, side, count, k) for w in C.parse_f32_row(r, k)]  # noqa: E731
    ea = [c for i in range(m) for c in pc8.line_ea(shape, seed_a, i)]
    eb = [c for j in range(n) for c in pc8.line_eb(shape, seed_b, j)]
    return (words("A", m), ea, words("B", n), eb, pc8.basis_fa(shape, seed_b), pc8.basis_fb(shape, seed_b),
            pc8.gram_g(shape, seed_b))


@pytest.mark.parametrize("dev", ["sm120", "h100"])
def test_the_split_call_is_the_hiding_call_with_each_tiles_cap(dev):
    """On honest rows, on each device: `Pc8CallCap` returns `Pc8CallHidden`'s tile digests (h0, `PearlTileDigest`'s) and
    admission bits, and per tile `Pc8TileCap`'s verdict on its rows; `Pc8CheckCap` is the weight's units then the
    call's.  One word's certificate moved fails its tile's verdict alone."""
    shape = replace(SPLIT, dev=dev)
    m, k, n, (tm, tn) = shape.m, shape.k, shape.n, shape.tile
    hidden = replace(shape, cap=False)
    a, ea, b, eb, fa, fb, g = _call_inputs(shape, "pc8-cap-split")
    wout = evaluate(pc8.weight(hidden), b, eb, fa, fb, g)
    bc, pb, ok_b = list(wout[:n * k]), list(wout[n * k:-n]), list(wout[-n:])
    want = evaluate(pc8.call(hidden), a, ea, fa, fb, bc, pb)
    rows = [evaluate(cap.templates(k, dev)["row_a"], a[i * k:(i + 1) * k], ea[32 * i:32 * (i + 1)], fa, fb)
            for i in range(m)]
    ac, pa, oka, sd = ([list(r[:k]) for r in rows], [list(r[k:k + 64]) for r in rows], [r[k + 64] for r in rows],
                       [list(r[k + 65:]) for r in rows])
    bcs, pbs = [bc[j * k:(j + 1) * k] for j in range(n)], [pb[64 * j:64 * (j + 1)] for j in range(n)]
    cert = [[cap.certificates(ac[i], bcs[j], dev) for j in range(n)] for i in range(m)]
    tiles = (m // tm) * (n // tn)
    blk = lambda x, rs, h: x[rs * h:(rs + 1) * h]                                   # noqa: E731
    flat = lambda rows: [x for r in rows for x in r]                                # noqa: E731
    caps = [evaluate(cap.templates(k, dev, shape.tile, n)["tile"], flat(blk(ac, rs, tm)), flat(blk(bcs, cs, tn)),
                     flat(blk(pa, rs, tm)), flat(blk(pbs, cs, tn)), blk(oka, rs, tm), flat(blk(sd, rs, tm)),
                     [w for r in blk(cert, rs, tm) for c in blk(r, cs, tn) for w in c])[-1]
            for rs in range(m // tm) for cs in range(n // tn)]
    assert oka == [1] * m
    for moved in (None, (1, 2)):
        cs_ = [[list(c) for c in r] for r in cert]
        if moved:
            i, j = moved
            cs_[i][j][0] = _keyed(cs_[i][j][0], 1)
        got = evaluate(pc8.call(shape), a, ea, fa, fb, bc, pb, [w for r in cs_ for c in r for w in c])
        assert got[:8 * tiles] == want[:8 * tiles] and got[-m:] == want[-m:]
        verdicts = list(got[8 * tiles:-m])
        assert verdicts == (caps if moved is None else [v if t != 1 else 0 for t, v in enumerate(caps)])
        if moved is None:
            assert evaluate(pc8.check(shape), a, ea, b, eb, fa, fb, g, [w for r in cs_ for c in r for w in c]) == \
                tuple(got) + tuple(ok_b)
    assert caps == [1] * tiles


def test_a_drawn_tile_proves_its_strip_every_word_and_its_tally():
    """`plan.tile_layout` under ``cap``: a tile's closure is itself, its TM `RowA` units, its strip, its TM·TN word units
    row by row, its TM rows' digests and sums and its tally, so a draw proves every cap word of each drawn tile
    (`Plan.proved`), never sampling one; the tile stratum alone carries W_ref.  With the cap off the layout is pc8's."""
    tm, tn = SPLIT.tile
    lay = plan.tile_layout(SPLIT, 1000)
    lay.check()
    us = lay.units
    tiles = [u for u in us if u.role == "tile"]
    assert len(tiles) == 4 and set(lay.closure) == {t.index for t in tiles}
    for t in tiles:
        rs, cs = t.where
        got = [(us[i].role, us[i].where) for i in lay.closure[t.index][1:]]
        assert got == ([("row_a", (rs * tm + i,)) for i in range(tm)] + [("strip", (rs,))]
                       + [("word", (rs * tm + i, cs * tn + j)) for i in range(tm) for j in range(tn)]
                       + [(r, (rs * tm + i, cs)) for r in ("row_digest", "row_tally") for i in range(tm)]
                       + [("tally", (rs, cs))])
    assert [(s.role, s.work) for s in lay.strata] == [("row_a", 0), ("strip", 0), ("word", 0), ("row_digest", 0),
                                                       ("row_tally", 0), ("tile", 4 * 1000), ("tally", 0)]
    for law in (plan.CountLaw(2), plan.WorkLaw(2)):
        p = plan.draw(derive(bytes(32), plan.DRAW_DOMAIN, {"test": "cap"}), lay, law)
        drawn = [u for u in p.drawn if us[u].role == "tile"]
        assert drawn and set(p.proved) == set(p.drawn) | {i for t in drawn for i in lay.closure[t]}
        for t in drawn:
            assert sum(us[i].role == "word" for i in lay.closure[t]) == tm * tn
    off = plan.tile_layout(replace(SPLIT, cap=False), 1000)
    assert all(len(c) == 1 + tm for c in off.closure.values())


def test_the_cap_flag_is_off_by_default_and_refuses_shapes_outside_the_split():
    """``cap`` is off by default, so a shape's ports and Programs are pc8's; on, the ports are `cap.PORTS`.  A cap shape
    that is not hiding, has more than 64 rows or columns a tile, a unit of columns too large for 31-bit sums, or fewer
    than two promotion windows is refused."""
    plain = pc8.Shape(4, 256, 4, tile=(2, 2), dev="sm120", hidden=True)
    assert not plain.cap and plain.ports is pc8.PORTS and SPLIT.ports is cap.PORTS
    assert pc8.call(plain).id != pc8.call(SPLIT).id and pc8.weight(plain).id == pc8.weight(SPLIT).id
    for shape in (replace(SPLIT, hidden=False), pc8.Shape(65, 256, 1, tile=(65, 1), hidden=True, cap=True),
                  pc8.Shape(1, 256, 65, tile=(1, 65), hidden=True, cap=True),
                  pc8.Shape(64, 8192, 64 * (2 ** 20 + 1), tile=(64, 64), hidden=True, cap=True),
                  pc8.Shape(1, 128, 1, tile=(1, 1), hidden=True, cap=True)):
        for build in (pc8.check, pc8.weight, pc8.call):
            with pytest.raises(DomainError):
                build(shape)


def test_the_cap_refuses_a_short_tile_row():
    """The cap takes whole tiles only (as the row cut does): 80 rows and decode's 32 at 64-row tiles, admitted by the
    hiding Programs without it, are refused with it by every root and by the tile layout, and 64 rows are admitted with it."""
    pc8.Shape(64, 256, 64, hidden=True, cap=True).check()
    for m in (80, 32):
        pc8.Shape(m, 256, 64, hidden=True).check()
        shape = pc8.Shape(m, 256, 64, hidden=True, cap=True)
        for refused in (pc8.check, pc8.weight, pc8.call, lambda s: plan.tile_layout(s, 4096)):
            with pytest.raises(DomainError, match="cap takes whole tiles only"):
                refused(shape)
