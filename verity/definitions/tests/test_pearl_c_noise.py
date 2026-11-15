"""-h3's noise derivation as registered Definitions (`verity.definitions.registry.pearl_c_noise`), word and Boolean,
against PoUW's reference (`schemes.pearl_c`: `seed_b`, `row_digest`, `row_seed`, `line_key`, `sample_line`) on random and
edge inputs, and on the vectors the Lean specification proves of its own noise (`Pouw.PearlC.W1Checks`: `seedB_vec`,
`rowDigest_vec`, `rowSeed_vec`, `lineEA_vec`, `lineFB_vec`).

A hash's Boolean version is a SHA-512 compression traced gate by gate, seconds each, so the tests that evaluate one are
slow, as is the one that holds a code's Boolean version to its word form's steps on every input."""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import compile as compile_bits
from verity.circuits.evaluation.bits import evaluate_words
from verity.commitments.gates import sha512 as SHA
from verity.commitments.gates.sha512 import compress_ref, pad_tail, words_be
from verity.definitions.boolean import boolean_version, sigma_lift
from verity.definitions.registry import pearl_c_noise as N
from verity.protocols.pouw.prover.schemes import pearl_c as C

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
PAIRS = [(N.PearlCSeedB, N.PearlCSeedBV2), (N.PearlCRowSeedHead, N.PearlCRowSeedHeadV2),
         (N.PearlCRowSeedTail, N.PearlCRowSeedTailV2), (N.PearlCRowSeed, N.PearlCRowSeedV2), (N.PearlCLineKey, N.PearlCLineKeyV2),
         (N.PearlCLineRaw, N.PearlCLineRawV2), (N.PearlCLineInt, N.PearlCLineIntV2),
         (N.PearlCLineNormRoot, N.PearlCLineNormRootV2), (N.PearlCLineScale, N.PearlCLineScaleV2),
         (N.PearlCLineCode, N.PearlCLineCodeV2), (N.PearlCLineCodes, N.PearlCLineCodesV2),
         (N.PearlCNoiseLine, N.PearlCNoiseLineV2), (N.PearlCRowBlock, N.PearlCRowBlockV2),
         (N.PearlCShaBlock, N.PearlCShaBlockV2)]
#: the size a Boolean Definition is lowered whole at (`circuit_check.checks.Options.lower_max_ands`), and the pins of this
#: module's lowerings
LOWERED_WHOLE = 1 << 16
PINS = Path(N.__file__).with_suffix(".circuit-check.json")
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


def seeds(n: int, seed: int) -> list[tuple[bytes, bytes, bytes, int, int]]:
    """(salt, leaf, root, unit, row): the Lean vectors', the ends of a unit's and a row's range, and random ones."""
    rng = random.Random(seed)
    rand = lambda k: bytes(rng.randrange(256) for _ in range(k))  # noqa: E731
    top = (1 << 32) - 1
    return ([(SALT, C.row_digest(ROW), ROOT, UNIT, ROW_INDEX), (bytes(32), bytes(64), bytes(64), 0, 0),
             (bytes([255] * 32), bytes([255] * 64), bytes([255] * 64), top, top), (SALT, C.row_digest(ROW), ROOT, top, 0),
             (SALT, C.row_digest(ROW), ROOT, 0, top)]
            + [(rand(32), rand(64), rand(64), rng.randrange(1 << 32), rng.randrange(1 << 32)) for _ in range(n)])


def lines(n: int, seed: int) -> list[tuple[bytes, int, int, int]]:
    """(key, side, factor, line): the lines a call samples (side and factor 0 or 1), the ends of every field, and random ones."""
    rng = random.Random(seed)
    key = C.line_key(C.seed_b(SALT, ROOT))
    return ([(key, s, f, t) for s in (0, 1) for f in (0, 1) for t in (0, 7)]
            + [(bytes(32), 0, 0, 0), (bytes([255] * 32), 255, 255, (1 << 32) - 1)]
            + [(bytes(rng.randrange(256) for _ in range(32)), rng.randrange(256), rng.randrange(256), rng.randrange(1 << 32))
               for _ in range(n)])


def raw_of(key: bytes, side: int, factor: int, line: int) -> bytes:
    """The hash `pearl_c.sample_line` makes a key's line from."""
    return C.s512("line", key + bytes([side, factor]) + line.to_bytes(4, "little"))


def head_state(salt: bytes, leaf: bytes, root: bytes) -> bytes:
    """The chaining value of a row seed's hash after its message's first block, by `sha512.compress_ref` from the label's
    block: its 64 bytes."""
    cv = SHA.compress_ref(SHA.midstate(C.S512_LABELS["seed-A"]), SHA.words_be(salt + leaf + root[:32]))
    return b"".join(w.to_bytes(8, "big") for w in cv)


def row_of(k: int, seed: int) -> list[int]:
    """A row of k FP32 words: the Lean vector's at its 15, else random."""
    rng = random.Random(seed)
    return list(struct.unpack("<15I", ROW)) if k == 15 else [rng.getrandbits(32) for _ in range(k)]


def row_bytes(x: list[int]) -> bytes:
    return b"".join(struct.pack("<I", w) for w in x)


def state_bytes(cv: list[int]) -> bytes:
    return b"".join(w.to_bytes(8, "big") for w in cv)


def lean_seeds() -> tuple[bytes, bytes]:
    """The Lean vectors' row seed and seed_B, by the reference."""
    return C.S512Seed(C.row_seed(SALT, C.row_digest(ROW), ROOT, UNIT, ROW_INDEX)), C.seed_b(SALT, ROOT)


def from_midstate(role: str, message: bytes) -> bytes:
    """`s512` as the Boolean versions compute it, on words: from the label's midstate, one compression a block of the padded
    message."""
    data = message + pad_tail(128 + len(message))
    cv = list(N._midstate(role))
    for i in range(0, len(data), 128):
        cv = compress_ref(cv, words_be(data[i:i + 128]))
    return b"".join(w.to_bytes(8, "big") for w in cv[:4])


def test_the_labels_and_the_labelled_hash_are_the_references():
    """Each role's label block is `pearl_c.S512_LABELS`', and `s512` is the reference's and the hash from the label's
    midstate, which the Boolean versions build on, at every padding case: an empty message, each hash's length, and the
    lengths around a block's end."""
    assert all(N.label(r) == C.S512_LABELS[r] for r in N.ROLES)
    with pytest.raises(ValueError, match="not 'jackpot2'"):
        N.label("jackpot2")
    rng = random.Random(7)
    for n in (0, 32, 38, 96, 111, 112, 127, 128, 168, 300):
        m = bytes(rng.randrange(256) for _ in range(n))
        assert all(N.s512(r, m) == C.s512(r, m) == from_midstate(r, m) for r in N.ROLES)


def test_each_word_definition_has_its_boolean_version():
    """`boolean_version` of each word Definition is its ``_v2``, so the σ of a body that calls one calls its Boolean version."""
    for w, b in PAIRS:
        assert boolean_version(bound(w)).id == bound(b).id == bound(w).id.replace("_v1", "_v2")
    for w, b in ((N.PearlCRowDigest, N.PearlCRowDigestV2), (N.PearlCRowLine, N.PearlCRowLineV2)):
        assert boolean_version(bind(w, K=40)).id == bind(b, K=40).id == bind(w, K=40).id.replace("_v1", "_v2")


def test_the_hashes_match_the_reference_on_words():
    for salt, leaf, root, unit, row in seeds(6, 1):
        seed_b = C.seed_b(salt, root)
        assert bytes(word(N.PearlCSeedB, salt, root)) == seed_b
        assert bytes(word(N.PearlCRowSeed, salt, leaf, root, unit, row)) == C.row_seed(salt, leaf, root, unit, row)
        assert bytes(word(N.PearlCLineKey, seed_b)) == C.line_key(seed_b)
    for key, side, factor, line in lines(6, 2):
        assert bytes(word(N.PearlCLineRaw, key, side, factor, line)) == raw_of(key, side, factor, line)


def test_the_row_seed_is_its_two_compressions_on_words():
    """`PearlCRowSeedHead` is the hash's chaining value after the first block of its message, `PearlCRowSeedTail` of it the
    reference's seed, at the Lean vector, the ends of a unit's and a row's range and random inputs; and the tail reads a state
    as any 64 bytes, as SHA-512 continued from it."""
    for salt, leaf, root, unit, row in seeds(6, 11):
        state = word(N.PearlCRowSeedHead, salt, leaf, root[:32])
        assert bytes(state) == head_state(salt, leaf, root)
        assert bytes(word(N.PearlCRowSeedTail, state, root[32:], unit, row)) == C.row_seed(salt, leaf, root, unit, row)
    rng = random.Random(12)
    for _ in range(4):
        state, tail, unit, row = (bytes(rng.randrange(256) for _ in range(64)), bytes(rng.randrange(256) for _ in range(32)),
                                  rng.randrange(1 << 32), rng.randrange(1 << 32))
        block = tail + row.to_bytes(4, "little") + unit.to_bytes(4, "little") + SHA.pad_tail(128 + 168)
        want = SHA.compress_ref(SHA.words_be(state), SHA.words_be(block))
        assert bytes(word(N.PearlCRowSeedTail, state, tail, unit, row)) == b"".join(w.to_bytes(8, "big") for w in want[:4])


def test_the_evaluators_compression_computes_sha512s_round_constants():
    """`_compress`, a byte primitive's evaluator, reads no table: its round constants are FIPS 180-4's cube roots, computed,
    and they are `sha512.K`."""
    assert N._round_constants() == SHA.K


def test_a_block_is_one_sha512_compression_on_words():
    """`PearlCShaBlock` from the initial value on a padded message of one block is hashlib's SHA-512, and from any state
    `sha512.compress_ref`; `PearlCRowBlock` is `PearlCShaBlock` on its 32 words' bytes, little-endian."""
    iv = state_bytes(SHA.IV)
    for n in (0, 1, 55, 111):
        message = bytes(range(n))
        assert bytes(word(N.PearlCShaBlock, iv, message + SHA.pad_tail(n))) == hashlib.sha512(message).digest()
    rng = random.Random(13)
    for _ in range(4):
        state, x = bytes(rng.randrange(256) for _ in range(64)), [rng.getrandbits(32) for _ in range(32)]
        want = state_bytes(SHA.compress_ref(SHA.words_be(state), SHA.words_be(row_bytes(x))))
        assert bytes(word(N.PearlCRowBlock, state, x)) == bytes(word(N.PearlCShaBlock, state, row_bytes(x))) == want


@pytest.mark.parametrize("k", [1, 15, 27, 28, 32, 33, 60, 256, 288, 520, 2048])
def test_a_rows_digest_on_words(k):
    """`pearl_c.row_digest` of the row's bytes: a row short of a block, one whose padding takes a block of its own (28
    words), whole blocks, a whole segment, and segments with a short last one; at 15 words the Lean's `rowDigest_vec`."""
    x = row_of(k, k)
    digest = word(bind(N.PearlCRowDigest, K=k), x)
    assert bytes(digest) == C.row_digest(row_bytes(x))
    if k == 15:
        assert digest[:8] == [225, 54, 37, 2, 134, 5, 77, 26]


#: a row's digest in compressions: (k, those of its segments, those of the outer hash)
COMPRESSIONS = [(15, 1, 1), (28, 2, 1), (32, 2, 1), (256, 9, 1), (288, 11, 2), (2048, 72, 5), (8192, 288, 17)]


def compressions(digest, k: int) -> list[str]:
    """The Definitions a row's digest Calls, in order, but for its constants."""
    return [n.fn.id for n in bind(digest, K=k).body.nodes if not n.fn.id.startswith("Const")]


@pytest.mark.parametrize("k,row_blocks,outer_blocks", COMPRESSIONS)
def test_a_rows_digest_is_a_compression_a_call(k, row_blocks, outer_blocks):
    """A row's digest Calls one block Definition a compression and nothing else but constants: nine compressions a whole
    segment and the outer hash's, 77 at k = 2,048 and 305 at k = 8,192, `Sha512RowSeg_v1`'s count."""
    assert compressions(N.PearlCRowDigest, k) == ["PearlCRowBlock_v1"] * row_blocks + ["PearlCShaBlock_v1"] * outer_blocks


@pytest.mark.slow
def test_a_rows_digest_on_bits_is_a_compression_a_call_each_under_the_whole_lowering():
    """The same Calls on bits, of the block Definitions' Boolean versions, each within the size a Definition is lowered
    whole at: every AND of a row's digest is in a pinned unit."""
    for k, row_blocks, outer_blocks in COMPRESSIONS:
        assert compressions(N.PearlCRowDigestV2, k) == ["PearlCRowBlock_v2"] * row_blocks + ["PearlCShaBlock_v2"] * outer_blocks
    each = [compile_bits(bind(d)).counts["and"] for d in (N.PearlCRowBlockV2, N.PearlCShaBlockV2)]
    assert each == [57_947, 57_947] and max(each) <= LOWERED_WHOLE


@pytest.mark.parametrize("k", [15, 32, 288])
def test_a_rows_line_on_words(k):
    """`PearlCRowLine` is `pearl_c.sample_line` of the row's `row_seed` over its `row_digest`; at 15 words the Lean's
    `lineEA_vec`."""
    for salt, _leaf, root, unit, row in seeds(2, k)[:1] + seeds(2, k)[3:]:
        x = row_of(k, k + row)
        seed = C.row_seed(salt, C.row_digest(row_bytes(x)), root, unit, row)
        assert word(bind(N.PearlCRowLine, K=k), salt, x, root, unit, row) == C.sample_line(seed, 0, 0, 0)
    if k == 15:
        assert word(bind(N.PearlCRowLine, K=k), SALT, row_of(15, 0), ROOT, UNIT, ROW_INDEX) == LINE_EA


def test_each_unit_of_a_definition_over_the_whole_lowering_is_pinned():
    """Each unit `UNITS` names has a pin of its lowering (the module's sidecar), which circuit-check holds it to: the
    Definitions over the size one is lowered whole at have none of their own."""
    pins = json.loads(PINS.read_text())["definitions"]
    assert set(N.UNITS) == {"PearlCRowSeed_v2", "PearlCNoiseLine_v2"} and not set(N.UNITS) & set(pins)
    assert all(u in pins for units in N.UNITS.values() for u in units)
    assert "PearlCRowBlock_v2" in pins and "PearlCShaBlock_v2" in pins
    assert not any(i.startswith(("PearlCRowDigest_v2", "PearlCRowLine_v2")) for i in pins)


@pytest.mark.slow
def test_a_definition_over_the_whole_lowering_is_calls_of_its_units_each_under_it():
    """Each Boolean Definition `UNITS` names is, on bits, one Call of each of its units in order and no gate of its own, so
    every AND of it is in a unit: 112,691 for a row seed's two compressions and 73,459 for a noise line's hash and codes, each
    over the 65,536 ANDs a Boolean Definition is lowered whole at, and each unit under it."""
    ands = lambda fid: compile_bits(bind(REGISTRY.defs[fid])).counts["and"]  # noqa: E731
    for whole, units in N.UNITS.items():
        nodes = bind(REGISTRY.defs[whole]).body.nodes
        assert [(n.form, n.fn.id) for n in nodes] == [("call", u) for u in units], whole
        each = [ands(u) for u in units]
        assert all(a <= LOWERED_WHOLE for a in each), (whole, each)
        assert ands(whole) == sum(each) > LOWERED_WHOLE, whole
    assert ands("PearlCRowSeed_v2") == 112_691 and ands("PearlCNoiseLine_v2") == 73_459


def test_the_lean_vectors_on_words():
    """`seedB_vec`, `rowSeed_vec` (over `rowDigest_vec`'s digest), `lineEA_vec` and `lineFB_vec`."""
    leaf = C.row_digest(ROW)
    assert list(leaf[:8]) == [225, 54, 37, 2, 134, 5, 77, 26]
    seed_b = word(N.PearlCSeedB, SALT, ROOT)
    assert seed_b[:8] == [247, 238, 5, 151, 197, 237, 99, 121]
    row_seed = word(N.PearlCRowSeed, SALT, leaf, ROOT, UNIT, ROW_INDEX)
    assert row_seed[:8] == [90, 99, 57, 103, 5, 175, 11, 41]
    assert word(N.PearlCNoiseLine, word(N.PearlCLineKey, row_seed), 0, 0, 0, N.NORM_E) == LINE_EA
    assert word(N.PearlCNoiseLine, word(N.PearlCLineKey, seed_b), 1, 1, 7, N.NORM_F) == LINE_FB


def test_the_lean_lines_codes_on_bits():
    """`lineEA_vec` and `lineFB_vec` from the two lines' hashes by the reference: their codes on bits, which the quick tier
    holds without a compression's gates."""
    row_seed, seed_b = lean_seeds()
    assert C.sample_line(row_seed, 0, 0, 0, 256) == LINE_EA and C.sample_line(seed_b, 1, 1, 7, 16) == LINE_FB
    ea, fb = raw_of(C.line_key(row_seed), 0, 0, 0), raw_of(C.line_key(seed_b), 1, 1, 7)
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


@pytest.mark.slow
def test_seed_b_and_the_line_key_on_bits():
    cases = seeds(6, 8)
    want = [C.seed_b(salt, root) for salt, _, root, _, _ in cases]
    assert want[0][:8] == bytes([247, 238, 5, 151, 197, 237, 99, 121])
    assert on_bits(N.PearlCSeedBV2, [list(c[0]) for c in cases], [list(c[2]) for c in cases]) == [list(s) for s in want]
    assert on_bits(N.PearlCLineKeyV2, [list(s) for s in want]) == [list(C.line_key(s)) for s in want]


@pytest.mark.slow
def test_the_row_seed_and_its_two_compressions_on_bits():
    cases = seeds(6, 9)
    want = [C.row_seed(*c) for c in cases]
    assert want[0][:8] == bytes([90, 99, 57, 103, 5, 175, 11, 41])
    got = on_bits(N.PearlCRowSeedV2, [list(c[0]) for c in cases], [list(c[1]) for c in cases], [list(c[2]) for c in cases],
                  [c[3] for c in cases], [c[4] for c in cases])
    assert got == [list(s) for s in want]
    states = [head_state(*c[:3]) for c in cases]
    heads = on_bits(N.PearlCRowSeedHeadV2, [list(c[0]) for c in cases], [list(c[1]) for c in cases],
                    [list(c[2][:32]) for c in cases])
    assert heads == [list(s) for s in states]
    tails = on_bits(N.PearlCRowSeedTailV2, [list(s) for s in states], [list(c[2][32:]) for c in cases],
                    [c[3] for c in cases], [c[4] for c in cases])
    assert tails == [list(s) for s in want]


@pytest.mark.slow
def test_a_noise_line_on_bits():
    """`PearlCLineRaw_v2` is the line's hash and `PearlCNoiseLine_v2`, the σ of the word form, `pearl_c.sample_line`'s line at
    both norms; the Lean's two lines among them."""
    cases = lines(6, 10)
    keys, sides, factors, ts = ([list(c[0]) for c in cases], [c[1] for c in cases], [c[2] for c in cases],
                                [c[3] for c in cases])
    assert on_bits(N.PearlCLineRawV2, keys, sides, factors, ts) == [list(raw_of(*c)) for c in cases]
    row_seed, seed_b = lean_seeds()
    sampled = [(row_seed, 0, 0, 0, N.NORM_E), (seed_b, 1, 1, 7, N.NORM_F), (seed_b, 1, 0, 3, N.NORM_E),
               (seed_b, 0, 1, 1023, N.NORM_F), (row_seed, 0, 0, (1 << 32) - 1, N.NORM_E)]
    got = on_bits(N.PearlCNoiseLineV2, [list(C.line_key(s)) for s, *_ in sampled], [c[1] for c in sampled],
                  [c[2] for c in sampled], [c[3] for c in sampled], [c[4] for c in sampled])
    assert got[:2] == [LINE_EA, LINE_FB]
    assert got == [C.sample_line(s, side, factor, t, NORMS[n]) for s, side, factor, t, n in sampled]


@pytest.mark.slow
def test_a_rows_digest_and_its_blocks_on_bits():
    """Each block Definition on bits is its word form, and the digest of a row of 40 words on bits (two blocks of the
    segment, the second the row's last 8 words and the padding, and the outer hash's one) is `pearl_c.row_digest`."""
    rng = random.Random(14)
    states = [list(state_bytes(SHA.IV))] + [[rng.randrange(256) for _ in range(64)] for _ in range(3)]
    xs = [[0] * 32] + [[rng.getrandbits(32) for _ in range(32)] for _ in range(3)]
    assert on_bits(N.PearlCRowBlockV2, states, xs) == [word(N.PearlCRowBlock, s, x) for s, x in zip(states, xs)]
    blocks = [list(row_bytes(x)) for x in xs]
    assert on_bits(N.PearlCShaBlockV2, states, blocks) == [word(N.PearlCShaBlock, s, b) for s, b in zip(states, blocks)]
    rows = [row_of(40, i) for i in range(3)] + [[0] * 40, [0xFFFFFFFF] * 40]
    assert on_bits(bind(N.PearlCRowDigestV2, K=40), rows) == [list(C.row_digest(row_bytes(x))) for x in rows]


@pytest.mark.slow
def test_a_rows_line_on_bits():
    """`PearlCRowLine_v2`, the σ of the word form, on the Lean vector's row and place (its `lineEA_vec`) and on others: six
    compressions and a line's codes in one circuit, which no cheaper test evaluates together."""
    cases = [(SALT, row_of(15, 0), ROOT, UNIT, ROW_INDEX)] + [(salt, row_of(15, 20 + i), root, unit, row)
                                                             for i, (salt, _leaf, root, unit, row) in enumerate(seeds(2, 15)[1:])]
    got = on_bits(bind(N.PearlCRowLineV2, K=15), [list(c[0]) for c in cases], [c[1] for c in cases],
                  [list(c[2]) for c in cases], [c[3] for c in cases], [c[4] for c in cases])
    assert got[0] == LINE_EA
    assert got == [C.sample_line(C.row_seed(salt, C.row_digest(row_bytes(x)), root, unit, row), 0, 0, 0)
                   for salt, x, root, unit, row in cases]
