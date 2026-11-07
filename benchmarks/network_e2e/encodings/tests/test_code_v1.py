"""Advice v1's residual coder (`code_v1.py`): prefix-free over everything v1 can send, canonical (one string per residual),
and self-delimiting per window."""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.ml.protocols.certifier.window_charge import Advice as W

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """By path: the directory's name shadows the stdlib's `encodings`, and `code.py` the stdlib's `code`."""
    if f"enc_{name}" in sys.modules:
        return sys.modules[f"enc_{name}"]
    spec = importlib.util.spec_from_file_location(f"enc_{name}", HERE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("code_v1")
C = V.C

SMALL = V.Caps(arrived=40, hits=12, hit_units=300, resumes=10, order=4, order_len=6, gap=200, index=300, lag=8)


def _gen(rng: random.Random, n: int, caps: V.Caps = SMALL) -> list:
    """n windows of a plausible residual: decisions in runs that end in an arrival, and rare hits, resumes, order misses
    and lags, the increasing fields increasing over the run."""
    out, pos, step, idx = [], -1, -1, -1
    for _ in range(n):
        arrived: list[bool] = []
        for _ in range(rng.randrange(0, 6)):
            arrived += [False] * rng.choice((0, 0, 1, 2, 5)) + [True]
        hits = [rng.choice((0, 1, 2, 7, 40, caps.hit_units)) for _ in range(rng.choice((0, 0, 0, 1, 3)))]
        resumes = []
        for _ in range(rng.choice((0, 0, 0, 1, 2))):
            pos += 1 + rng.randrange(0, 20)
            resumes.append(pos)
        order = []
        for _ in range(rng.choice((0, 0, 0, 1))):
            step += 1 + rng.randrange(0, 30)
            ids = []
            for _ in range(rng.randrange(1, caps.order_len + 1)):
                idx = min(caps.index, max(0, idx + rng.choice((1, 1, 2, -3))))
                ids.append(idx)
            order.append((step, tuple(ids)))
        lag = [rng.random() < 0.1 for _ in range(rng.choice((0, 1, 3)))]
        out.append(W(arrived[:caps.arrived], hits, resumes, order, lag))
    return out


#: the registered book: fit on historical runs (seeds 0–39), never on a run a test codes (seeds 1000 and up)
BOOK = V.fit_book([_gen(random.Random(s), 12) for s in range(40)], SMALL)


def _s(bits) -> str:
    return "".join(map(str, bits))


def _prefix_free(words: list[str]) -> bool:
    s = sorted(words)
    return len(set(s)) == len(s) and all(not s[i + 1].startswith(s[i]) for i in range(len(s) - 1))


def _kraft(words: list[str]) -> Fraction:
    return sum((Fraction(1, 1 << len(w)) for w in words), Fraction(0))


def _decodes(bits: list[int], n: int, book: V.Book = BOOK):
    try:
        return V.decode(*V.pack(bits), book, n)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------- prefix-free, by field


@pytest.mark.parametrize("book", [BOOK, V.Book.uniform(SMALL)], ids=["fitted", "uniform"])
def test_each_fields_codewords_are_prefix_free_with_kraft_at_most_one(book):
    """Every integer symbol over its whole alphabet, and every bit at prior and grown KT states: no codeword is a prefix
    of another, and the Kraft sum is at most 1."""
    for name, (lo, hi) in V.ints(book.caps).items():
        words = []
        for v in range(lo, hi + 1):
            e = V.Encoder(book)
            e.window()
            e.value(name, v, lo, hi)
            words.append(_s(e.finish().bitlist()))
        assert _prefix_free(words), name
        assert _kraft(words) <= 1, name
    for name, ctxs in V.BITS.items():
        for ctx in ctxs:
            for state in (None, (0, 0), (5, 0), (0, 5), (1000, 3), (16000, 16383)):
                words = []
                for b in (False, True):
                    e = V.Encoder(book)
                    if state is not None:
                        e.kt.n[V._key(name, ctx)] = list(state)
                    e.window()
                    e.bit(name, b, ctx)
                    words.append(_s(e.finish().bitlist()))
                assert _prefix_free(words) and _kraft(words) <= 1, (name, ctx, state)


TINY = V.Caps(arrived=2, hits=1, hit_units=2, resumes=0, order=0, order_len=1, gap=0, index=0, lag=1)


def _tiny_windows():
    for na in range(3):
        for arrived in itertools.product((False, True), repeat=na):
            for hits in ((), (0,), (1,), (2,)):
                for nl in range(2):
                    for lag in itertools.product((False, True), repeat=nl):
                        yield W(arrived, hits, (), (), lag)


@pytest.mark.parametrize("prior", [{}, {"arrived/0": (1, 9), "lag/0": (12, 0)}], ids=["no-prior", "prior"])
def test_decode_accepts_exactly_the_encoders_strings(prior):
    """Exhaustive over a small alphabet: of every bit string up to the longest codeword, `decode` accepts exactly the
    encoder's codewords, each back to its residual. They are prefix-free with Kraft's sum at most 1, so v1's charge
    meets the sampled leak bound's `hAs`."""
    book = V.Book(TINY, V.Book.uniform(TINY).models, prior)
    words = {}
    for win in _tiny_windows():
        words[_s(V.encode([win], book).bitlist())] = win
    assert len(words) == sum(1 for _ in _tiny_windows())
    assert _prefix_free(list(words)) and _kraft(list(words)) <= 1
    longest = max(map(len, words))
    assert longest <= 16
    accepted = {}
    for n in range(longest + 1):
        for bits in itertools.product((0, 1), repeat=n):
            got = _decodes(list(bits), 1, book)
            if got is not None:
                accepted[_s(bits)] = got[0]
    assert accepted == words


# ------------------------------------------------------------------------------------ randomized, over whole runs


@pytest.mark.parametrize("seed", range(1000, 1040))
def test_a_run_decodes_from_its_concatenation_and_each_window_ends_itself(seed):
    """A random run: its windows' segments concatenated decode back. The first k windows decode from the bits before
    window k's end alone, with nothing after them read. A trailing bit and a missing one are refused."""
    rng = random.Random(seed)
    ws = _gen(rng, rng.randrange(1, 9))
    e = V.encode(ws, BOOK)
    bits = e.bitlist()
    assert sum(e.lens) == e.nbits == len(bits)
    assert V.decode(e.bits, e.nbits, BOOK, len(ws)) == ws
    for k in range(len(ws) + 1):
        assert _decodes(bits[:sum(e.lens[:k])], k) == ws[:k]
    for extra in ([0], [1], [0, 0, 0, 0, 0, 0, 0, 0]):
        assert _decodes(bits + extra, len(ws)) is None
    if bits:
        assert _decodes(bits[:-1], len(ws)) is None


@pytest.mark.parametrize("seed", range(200))
def test_whatever_decodes_is_what_its_residual_encodes_to(seed):
    """A random string either is refused or is exactly the encoding of the residual it decodes to."""
    rng = random.Random(seed)
    bits = [rng.randrange(2) for _ in range(rng.randrange(0, 48))]
    n = rng.randrange(1, 4)
    got = _decodes(bits, n)
    if got is not None:
        assert V.encode(got, BOOK).bitlist() == bits


class _Plain(V.Decoder):
    """The WNC decoder without `code.ArithDecoder.finish`'s canonical check: the segment's length, and nothing compared."""

    def _close(self) -> None:
        if self.seg is not None:
            n = self.seg.shifts + 2 if self.seg.used else 0
            self.seg.inp.pos = self.seg.start + n
            self.lens.append(n)
            self.seg = None


def _plain(bits: list[int], n: int):
    try:
        d, st = _Plain(*V.pack(bits), BOOK), V._State()
        out = []
        for _ in range(n):
            d.window()
            out.append(V._window(d, None, st))
        d.finish()
        return out
    except ValueError:
        return None


def test_one_codeword_per_residual_where_the_plain_decoder_takes_several():
    """Attack 10 (the certifier PROTOCOL): over the strings that differ from a residual's encoding only in its last 3 bits,
    the plain WNC decoder takes 2 to 4 as that residual, and `decode` takes the encoder's one."""
    most = 0
    for seed in range(1000, 1060):
        rng = random.Random(seed)
        ws = _gen(rng, 1)
        bits = V.encode(ws, BOOK).bitlist()
        if len(bits) < 3:
            continue
        variants = [bits[:-3] + list(t) for t in itertools.product((0, 1), repeat=3)]
        assert [v for v in variants if _decodes(v, 1) == ws] == [bits]
        plain = sum(_plain(v, 1) == ws for v in variants)
        assert 1 <= plain <= 4
        most = max(most, plain)
    assert most >= 2


# ---------------------------------------------------------------------------------- the online API, other readers


def test_a_reader_with_its_own_contexts_and_window_ends_round_trips():
    """A reader like the replay's: each bit's context is a function of the bits decoded so far (here the last bit and a
    count), and a window ends when the reader says (here after its second arrival), with no count sent. Its string is
    canonical and self-delimiting the same way."""
    rng = random.Random(7)
    truth = [[rng.random() < 0.3 for _ in range(rng.randrange(0, 12))] + [True, True] for _ in range(6)]

    def drive(c, windows=None):
        out = []
        prev, seen = False, 0
        for w in range(6):
            c.window()
            got, arrivals, j = [], 0, 0
            while arrivals < 2:
                ctx = (prev, seen % 3)
                b = c.bit("arrived", windows[w][j], ctx) if windows else c.bit("arrived", ctx)
                got.append(b)
                prev, seen, arrivals, j = b, seen + b, arrivals + b, j + 1
            out.append(got)
        return out

    e = V.Encoder(BOOK)
    drive(e, [t[:t.index(True, t.index(True) + 1) + 1] for t in truth])
    enc = e.finish()
    d = V.Decoder(enc.bits, enc.nbits, BOOK)
    got = drive(d)
    d.finish()
    assert got == [t[:t.index(True, t.index(True) + 1) + 1] for t in truth]
    assert d.lens == list(enc.lens)


# ------------------------------------------------------------------------------------------------ refusals, the book


def test_refusals():
    e = V.encode(_gen(random.Random(1001), 3), BOOK)
    if e.nbits % 8:
        padded = e.bits[:-1] + bytes([e.bits[-1] | 1])
        with pytest.raises(ValueError, match="padding"):
            V.decode(padded, e.nbits, BOOK, 3)
    with pytest.raises(ValueError, match="outside any window"):
        V.Encoder(BOOK).value("hit", 1, 0, SMALL.hit_units)
    with pytest.raises(ValueError, match="outside its feasible range"):
        V.encode([W(hits=(SMALL.hit_units + 1,))], BOOK)
    with pytest.raises(ValueError, match="outside its feasible range"):
        V.encode([W(resumes=(4, 4))], BOOK)
    with pytest.raises(ValueError, match="order index"):
        V.encode([W(order=((0, (SMALL.index + 1,)),))], BOOK)


def test_the_book_is_registered_by_its_digest():
    """The digest names the book exactly: it survives the JSON round trip, and any change to a prior or a table moves it."""
    again = V.Book.from_json(json.loads(json.dumps(BOOK.to_json())))
    assert again.digest() == BOOK.digest() and len(BOOK.digest()) == 128
    ws = _gen(random.Random(1002), 4)
    assert V.decode(*V.pack(V.encode(ws, BOOK).bitlist()), again, 4) == ws
    moved = V.Book(BOOK.caps, BOOK.models, {**BOOK.priors, "lag/0": (1, 1)})
    assert moved.digest() != BOOK.digest()
    assert V.Book.uniform(SMALL).digest() != BOOK.digest()


@pytest.mark.parametrize("seed", range(1000, 1020))
def test_delimiting_costs_the_terminations_two_bits(seed):
    """Each window's charge is its model cost plus at most the termination's 2 bits and the coder's rounding."""
    ws = _gen(random.Random(seed), 6)
    e = V.encode(ws, BOOK)
    for charged, ideal in zip(e.lens, V.ideal_bits(ws, BOOK)):
        assert ideal - 1e-9 <= charged <= ideal + 2.01
    assert V.encode([W()], BOOK).lens[0] <= V.ideal_bits([W()], BOOK)[0] + 2.01
