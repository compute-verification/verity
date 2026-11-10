"""-h3's noise line as registered Definitions (`verity.ml.definitions.registry.pearl_c_noise`), word and Boolean, against
PoUW's reference (`schemes.pearl_c.sample_line`) on random and edge lines, and on the two lines the Lean specification
proves of its own noise (`Pouw.PearlC.W1Checks`: `lineEA_vec`, `lineFB_vec`)."""
from __future__ import annotations

import math
import random
import struct

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.boolean import boolean_version, sigma_lift
from verity.ml.definitions.registry import pearl_c_noise as N
from verity.ml.protocols.pouw.schemes import pearl_c as C

#: the Lean vectors' salt, weight root and activation row (`W1Checks`' `vSalt`, `vRoot`, `vRow`), at unit 3 and row 5
SALT = bytes(range(32))
ROOT = bytes((7 * i + 3) % 256 for i in range(64))
ROW = b"".join(struct.pack("<I", w) for w in [0x3F800000, 0xBF000000, 0x40490FDB, 0x00000000, 0x3DCCCCCD] * 3)
UNIT, ROW_INDEX = 3, 5
#: `lineEA_vec`: row 5's E_A line (norm 256), and `lineFB_vec`: F_B's line 7 (norm 16)
LINE_EA = [103, 233, 104, 227, 104, 95, 83, 223, 231, 95, 100, 105, 177, 230, 93, 229, 228, 228, 218, 208, 96, 229, 94, 65,
           98, 232, 207, 102, 91, 91, 97, 232]
LINE_FB = [71, 184, 67, 191, 192, 146, 201, 65, 181, 187, 72, 188, 201, 190, 194, 72, 200, 200, 73, 62, 72, 57, 186, 71,
           154, 66, 199, 194, 183, 66, 154, 196]
#: a line's norm by the BF16 word of 32 times it
NORMS = {N.NORM_E: 256, N.NORM_F: 16}
#: every pair of a word Definition and its Boolean version
PAIRS = [(N.PearlCLineInt, N.PearlCLineIntV2), (N.PearlCLineNormRoot, N.PearlCLineNormRootV2),
         (N.PearlCLineScale, N.PearlCLineScaleV2), (N.PearlCLineCode, N.PearlCLineCodeV2),
         (N.PearlCLineCodes, N.PearlCLineCodesV2)]
#: a line's hash at the ends of its integers and its norm root: every integer 1, 128, −128 or −1, one of magnitude 128 among
#: 1s, alternating signs, and bytes on both sides of the sign bit
EDGE_RAWS = [bytes(32), bytes([0x7F] * 32), bytes([0xFF] * 32), bytes([0x80] * 32), bytes([0x7F] + [0] * 31),
             bytes([0] * 31 + [0xFF]), bytes([0x7F, 0xFF] * 16), bytes(range(112, 144))]


def bound(fn):
    return bind(fn) if isinstance(fn, CompositeDefinition) else fn


def word(fn, *args) -> list[int]:
    return list(evaluate(bound(fn), *args))


def on_bits(fn, *args) -> list[list[int]]:
    """``fn`` (a Boolean Definition) on instances of word arguments, one list of instances a parameter each: each instance's
    output words in leaf order."""
    out = evaluate_words(bound(fn), *(np.asarray(a, dtype=np.uint64) for a in args))
    return out.reshape(out.shape[0], -1).tolist()


def raws(n: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    return EDGE_RAWS + [bytes(rng.randrange(256) for _ in range(32)) for _ in range(n)]


def codes_of(monkeypatch, raw: bytes, norm: int) -> list[int]:
    """`pearl_c.sample_line`'s codes for a line whose hash is ``raw``."""
    with monkeypatch.context() as m:
        m.setattr(C, "_s512_line_key", lambda seed: b"")
        m.setattr(C, "s512", lambda role, message: raw)
        return C.sample_line(C.S512Seed(), 0, 0, 0, norm)


def raw_of(seed: bytes, side: int, factor: int, line: int) -> bytes:
    """The hash `pearl_c.sample_line` makes a seed's line from."""
    return C.s512("line", C.line_key(seed) + bytes([side, factor]) + line.to_bytes(4, "little"))


def test_each_word_definition_has_its_boolean_version():
    """`boolean_version` of each word Definition is its ``_v2``, so the σ of a body that calls one calls its Boolean version."""
    for w, b in PAIRS:
        assert boolean_version(bound(w)).id == bound(b).id == bound(w).id.replace("_v1", "_v2")


def test_the_lean_lines():
    """`lineEA_vec` and `lineFB_vec`: the codes of the two lines' hashes, word and Boolean, and `pearl_c.sample_line`'s."""
    row_seed = C.row_seed(SALT, C.row_digest(ROW), ROOT, UNIT, ROW_INDEX)
    seed_b = C.seed_b(SALT, ROOT)
    assert C.sample_line(row_seed, 0, 0, 0, 256) == LINE_EA and C.sample_line(seed_b, 1, 1, 7, 16) == LINE_FB
    ea, fb = raw_of(row_seed, 0, 0, 0), raw_of(seed_b, 1, 1, 7)
    assert word(N.PearlCLineCodes, ea, N.NORM_E) == LINE_EA and word(N.PearlCLineCodes, fb, N.NORM_F) == LINE_FB
    assert on_bits(N.PearlCLineCodesV2, [list(ea), list(fb)], [N.NORM_E, N.NORM_F]) == [LINE_EA, LINE_FB]


def test_a_lines_integers_on_every_byte():
    xs = [(-1 if b >> 7 else 1) * ((b & 0x7F) + 1) for b in range(256)]
    want = [x & 0xFFFFFFFF for x in xs]
    assert [N.PearlCLineInt.evaluate(b) for b in range(256)] == want
    assert on_bits(N.PearlCLineIntV2, list(range(256))) == [[x] for x in want]


def test_a_lines_norm_root():
    """The root at its two ends (181 for 32 ones, 23,170 for 32 integers of 128) and on random lines, word and Boolean."""
    rs = raws(200, 3)
    want = [math.isqrt(1024 * sum(((b & 0x7F) + 1) ** 2 for b in raw)) for raw in rs]
    assert want[0] == want[3] == 181 and want[1] == want[2] == 23170
    assert [N.PearlCLineNormRoot.evaluate(*raw) for raw in rs] == want
    assert on_bits(N.PearlCLineNormRootV2, [list(raw) for raw in rs]) == [[r] for r in want]


def test_a_lines_codes_match_sample_line(monkeypatch):
    """`PearlCLineCodes` of a line's hash is `pearl_c.sample_line`'s line at both norms, word and Boolean, and its scale and
    each code are `PearlCLineScale`'s and `PearlCLineCode`'s."""
    rs = raws(40, 4)
    for norm32, norm in NORMS.items():
        want = [codes_of(monkeypatch, raw, norm) for raw in rs]
        assert [word(N.PearlCLineCodes, raw, norm32) for raw in rs] == want
        assert on_bits(N.PearlCLineCodesV2, [list(raw) for raw in rs], [norm32] * len(rs)) == want
        scales = [word(N.PearlCLineScale, raw, norm32)[0] for raw in rs]
        assert on_bits(N.PearlCLineScaleV2, [list(raw) for raw in rs], [norm32] * len(rs)) == [[s] for s in scales]
        assert [[word(N.PearlCLineCode, b, s)[0] for b in raw] for raw, s in zip(rs, scales)] == want


def test_the_boolean_scale_and_code_are_the_word_forms_on_every_kind_of_word():
    """A norm or a scale may be any BF16 word to the Definitions (the reference rejects a non-finite one): zeros, subnormals,
    infinities and NaNs of both signs, the two norms and random words, at every byte."""
    rng = random.Random(5)
    special = [0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81, N.NORM_E,
               N.NORM_F, 0x3F80, 0xBF80]
    words = special + [rng.randrange(1 << 16) for _ in range(48)]
    cases = [(b, s) for s in words for b in range(256)]
    want = [[word(N.PearlCLineCode, b, s)[0]] for b, s in cases]
    assert on_bits(N.PearlCLineCodeV2, [b for b, _ in cases], [s for _, s in cases]) == want
    rs = raws(8, 6)
    cases = [(raw, s) for s in words for raw in rs]
    want = [[word(N.PearlCLineScale, raw, s)[0]] for raw, s in cases]
    assert on_bits(N.PearlCLineScaleV2, [list(raw) for raw, _ in cases], [s for _, s in cases]) == want


@pytest.mark.slow
def test_a_code_on_bits_is_the_sigma_of_its_word_form_on_every_byte_and_scale():
    """`PearlCLineCode_v2` rounds its product once, where the word form rounds it to FP32 and then to BF16. On all 2^24 pairs
    of a byte and a BF16 word it is σ(`PearlCLineCode_v1`), the word form's steps by their own Boolean versions: no sample of
    scales holds every tie, subnormal and overflow of that rounding."""
    steps = bind(sigma_lift(N.PearlCLineCode, 0, register=False))
    scales = np.arange(1 << 16, dtype=np.uint64).reshape(-1, 1)
    for b in range(256):
        byte = np.full_like(scales, b)
        assert np.array_equal(evaluate_words(bind(N.PearlCLineCodeV2), byte, scales), evaluate_words(steps, byte, scales)), b
