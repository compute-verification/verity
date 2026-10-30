"""Pearl-C's debit replay (`pearl_c_debit`): `Skip.wordFlags` against 448 of P1's Lean-checked word vectors,
checked from scratch on skips, additive over a tile's words, and over the cap on the aligned-spike tile."""

from __future__ import annotations

import gzip
import hashlib
import random
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.models import HOPPER_E4M3_K32
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_debit as D
from verity.ml.protocols.pouw.schemes import pearl_kw as P

VECTORS = Path(__file__).parent / "vectors"
#: P1's conformance vectors, one word per
#: line, `family forming k i j A′-row B̃-row flags`, their lines verbatim. The k ≤ 2,048 file (SHA-256 86a31922…, 552 words
#: with the k = 8,192 file) is cut under the repository's 256 KiB blob limit to every word with k ≤ 512 (the 272 words
#: `decide +kernel` checks among them) and, at k = 2,048, the aligned-spike words under §5's forming: 440 of its 544.
#: The replay agreed on all 552 when it was cut.
FILES = {"pearl_c_skip_vectors.txt.gz": "7f105baecf407a1b81bb1eef549d3c53696307c978a743d79a9308bf55c217f3",
         "pearl_c_skip_vectors_k8192.txt.gz": "307d4770d8146eadf42c90ae9399ff8af8c21e1f0b621af6ad64ed414bc2235d"}


def vectors() -> list[tuple[str, int, list[int], list[int], list[bool]]]:
    out = []
    for name in FILES:
        for line in gzip.open(VECTORS / name, "rt"):
            fam, forming, k, _, _, a, b, flags = line.split()
            out.append((f"{fam}/{forming}", int(k), list(bytes.fromhex(a)), list(bytes.fromhex(b)), [c == "1" for c in flags]))
    return out


def test_word_flags_are_p1s_lean_checked_vectors():
    for name, digest in FILES.items():
        assert hashlib.sha256((VECTORS / name).read_bytes()).hexdigest() == digest
    vs = vectors()
    assert len(vs) == 448 and sum(k == 8192 for _, k, *_ in vs) == 8 and sum(k <= 256 for _, k, *_ in vs) == 272
    assert [D.word_flags(a, b) for _, _, a, b, _ in vs] == [flags for *_, flags in vs]


def replay(a: list[int], b: list[int], skip: set[int]) -> int:
    """The chain from scratch with the atoms in `skip` passed through in place (the definition the flags summarize)."""
    total = 0
    for g in range(0, len(a) // 32, D.G):
        word = 0
        for t in range(g, g + D.G):
            if t not in skip:
                word = HOPPER_E4M3_K32.step(word, a[32 * t:32 * t + 32], b[32 * t:32 * t + 32])
        total = D._promote(total, word)
    return D._key(total)


def test_flags_are_joint_skips_replayed_from_scratch():
    """On spiked words: an atom is flagged iff some one-group set containing it, skipped jointly and replayed from scratch,
    leaves the output unchanged."""
    for _, k, a, b, flags in [v for v in vectors() if "spikes" in v[0] and v[1] == 512][:6]:
        out = replay(a, b, set())
        for g in range(0, k // 32, D.G):
            keeps = set()
            for mask in range(1, 1 << D.G):
                skip = {g + j for j in range(D.G) if mask >> j & 1}
                if replay(a, b, skip) == out:
                    keeps |= skip
            assert {t for t in range(g, g + D.G) if flags[t]} == keeps


def spiked_rows(k: int, count: int, seed: int, spike: float | None) -> list[bytes]:
    """The census's aligned-spike family: Gaussian rows with `spike` at offsets 1 and 9 of every 128-group (None: none)."""
    out = []
    for r in range(count):
        rng = random.Random(seed + r)
        x = [rng.gauss(0, 1) for _ in range(k)]
        if spike is not None:
            for g in range(0, k, 128):
                x[g + 1] = x[g + 9] = spike
        out.append(b"".join(P.f32_bits(P.f32(v)).to_bytes(4, "little") for v in x))
    return out


@pytest.mark.parametrize("spike", [300.0, None])
def test_aligned_spike_tile_is_over_the_cap(spike):
    """The negative control: the R = 300 aligned-spike tile, formed by the scheme's own `encode`, passes the domain and has
    74.6% of its atom-words flagged, about 15x the cap; without spikes the debit is far below it."""
    k, t = 4096, 4
    s = C.PearlC(k_min=128, tile=(t, t), forming="v1")
    ctx = Context(b"\x11" * 32, Matmul(0, Shape(t, k, t), 0), b"\x22" * 32, b"\x33" * 32)
    ops = s.encode(ctx, s.noise(ctx), spiked_rows(k, t, 100, spike), spiked_rows(k, t, 900, spike),
                   Tile(tuple(range(t)), tuple(range(t))))
    assert s.admissible(ctx, ops)
    atoms, zeros = D.chain_debit(ops.a.codes, ops.b.codes)
    assert atoms == sum(sum(D.word_flags(a, b)) for a in ops.a.codes for b in ops.b.codes) and zeros == 0
    debit = D.debit_units(atoms, D.forming_elements(ops.a.codes, ops.b.codes), zeros=zeros)
    cap = s.cap * s.credit_of(Shape(t, k, t))
    if spike:
        assert atoms == 1527 and round(atoms / (t * t * k // 32), 3) == 0.746 and debit > 14 * cap
    else:
        assert debit < cap / 10


def zero_promos(a: list[int], b: list[int]) -> int:
    """rev1's `laterZeroPromos` from its definition: the g in [1, k/(32·G)) whose chain over the first 32·G·g columns,
    replayed from scratch, is ±0."""
    span = 32 * D.G
    return sum(1 for g in range(1, len(a) // span) if replay(a[:span * g], b[:span * g], set()) == 0)


def test_later_zero_promotions_are_the_prefix_chains_at_zero():
    """A word's later +0 promotions (`word_debit`) are its prefix chains that replay to ±0: none on P1's vectors at
    k = 512, every one on a zero word, and the one after a group that cancels the group before it."""
    words = [(a, b) for _, k, a, b, _ in vectors() if k == 512][:40]
    assert all(D.word_debit(a, b)[1] == zero_promos(a, b) == 0 for a, b in words)
    rng, k = random.Random("pearl-c-rev1/zeros"), 512
    a = [rng.randrange(0x7F) | rng.choice((0, 0x80)) for _ in range(k)]
    half = [rng.randrange(0x7F) | rng.choice((0, 0x80)) for _ in range(128)]
    tail = [rng.randrange(0x7F) | rng.choice((0, 0x80)) for _ in range(256)]
    b = half + [c ^ 0x80 for c in half] + tail
    a = a[:128] * 2 + a[256:]
    assert D.word_debit(a, b) == (D.word_flags(a, b), zero_promos(a, b)) and zero_promos(a, b) == 1
    assert D.word_debit([0] * k, [0] * k)[1] == zero_promos([0] * k, [0] * k) == k // (32 * D.G) - 1
    assert D.chain_debit([a, [0] * k], [b]) == (sum(D.word_flags(a, b)) + sum(D.word_flags([0] * k, b)),
                                               1 + zero_promos([0] * k, b))
