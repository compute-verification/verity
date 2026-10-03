"""Pearl-C's per-tile cap on a circuit (`verity_pouw.circuit.cap`) against the scheme's replay: a word's flags and later
+0 promotions are `pearl_c_debit.word_debit`'s, its C̃ `pearl_c.chain`'s, and only the exact certificates pass; a tile's
verdict is rev1's capOK (`pearlCTilesDevRev1`) on `chain_debit` and `debit_units` against the device's `cap` times
`credit_of`, on random tiles and adversarial ones (zero rows, absorbed groups, a total cancelled to zero, refused rows,
and the cap's boundary).  Codes are finite E4M3: the atom has no NaN operand (`verity.ml.tc` raises), and neither does
the replay."""
from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.evaluation import evaluate
from verity_pouw.circuit import cap, pc8
from verity_pouw.protocol import DomainError, Shape
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_debit as D
from verity_pouw.schemes import pearl_kw as P
from verity_pouw.schemes.pearl_c_device import DEVICES

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


def _tile(k: int, tile, n: int, rho: int, ac, bc, ok, sd, cert=None, dev: str = "sm120"):
    """(C̃, U, verdict) of `Pc8TileCap`, and `Pc8Tile`'s (C̃, U) on the same rows (P_A, P_B random BF16 words)."""
    tm, tn = tile
    rng = random.Random(f"cap-peel-{k}-{tile}")
    pa = [[rng.getrandbits(16) & 0x3FFF for _ in range(pc8.PEEL)] for _ in range(tm)]
    pb = [[rng.getrandbits(16) & 0x3FFF for _ in range(pc8.PEEL)] for _ in range(tn)]
    if cert is None:
        cert = [[cap.certificates(ac[i], bc[j], dev) for j in range(tn)] for i in range(tm)]
    flat = lambda rows: [x for r in rows for x in r]  # noqa: E731
    out = evaluate(cap.templates(k, dev, tile, n, rho)["tile"], flat(ac), flat(bc), flat(pa), flat(pb), [int(o) for o in ok],
                   flat(sd), [w for r in cert for c in r for w in c])
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
