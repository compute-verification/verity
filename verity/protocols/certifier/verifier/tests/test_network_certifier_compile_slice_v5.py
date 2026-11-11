"""The compile slice at its window's activity (`compile_slice_v5.CompileSlice_v5`): CompileSlice_v4 with the arrivals'
tokens one ragged row of the blocks the window keys, each keyed once, and each running slot's request number carried in
the slot row. Its free input is the advice, DA + PW bits, and the checked witness at its width until its lemma. The
circuit accepts exactly where `compile_slice_v5.reference` compiles, and its served rows and last state and window rows
are CompileSlice_v4's, on both demonstration slices, on every advice of their arrivals and on random windows under
eviction. A registered row one bit off is refused by its own verdict bit (a carried request number by ``carry``), a
checked witness one bit off by the cache, pool or keys bit, a keyed block count too small by the pool bit, and arrivals
that repeat an id by the load."""
from __future__ import annotations

import itertools
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verity.protocols.certifier.verifier import compile_slice as CS
from verity.protocols.certifier.prover import compile_slice_v2 as V2, compile_slice_v3 as V3
from verity.protocols.certifier.prover import compile_slice_v4 as V4, compile_slice_v5 as V5
from verity.protocols.certifier.verifier import replay_chain as RC
from verity.protocols.certifier.verifier.head_check import ands
from verity.circuits import boolean as BL


def _x(f, f3):
    st, sl = f()
    st3, _ = f3()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, st3=st3, S3=SimpleNamespace(**st3))


@pytest.fixture(scope="module")
def synth():
    return _x(V5.demo, V3.demo)


@pytest.fixture(scope="module")
def asynth():
    return _x(V5.demo_async, V3.demo_async)


def _agrees(st: dict, slices: list[CS.Slice]) -> list:
    """The circuit's verdict on `slices`, the reference's rows, counts and witness given, is `ACCEPT` exactly where the
    reference compiles."""
    S = SimpleNamespace(**st)
    refs = [V5.reference(S, sl) for sl in slices]
    for sl, ref, got in zip(slices, refs, V5.evaluate(st, slices)):
        assert (got == V5.ACCEPT) == (ref is not None), sl
    return refs


def _as_v4(x, sl: CS.Slice, ref) -> None:
    """`ref`'s served rows and last state, window and slot rows are CompileSlice_v4's on `sl`, the slot row's request
    numbers past them."""
    r4 = V4.reference(x.S, sl)
    assert ref.srv == r4.srv and ref.cout[:len(r4.cout)] == r4.cout, sl
    assert len(ref.cout) - len(r4.cout) == x.S.R * (x.S.K.bit_length())


def _rq(x) -> int:
    """Where cout's request numbers start."""
    return V5.data_bits(x.S)["cout"] - x.S.R * x.S.K.bit_length()


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_free_input_is_the_advice_and_the_checked_witness(request, which):
    """Every input row is classified, and the unit's free input is the advice, DA decision bits and the PW = 3 bits of the
    step count (8 at the demo's class), and the checked witness at its width until its lemma: the coins (α, γ, δ) are the
    verifier's 384 bits. Its one output is 16 bits. The checked witness is CompileSlice_v4's less RB's and RK's K·CU
    dense entries (128 + TW bits each), with the RB run's NRB entries (1 + RW + KW + NW + TW bits) and the ragged
    blocks' NRK numbers and times (RW + TW)."""
    x = request.getfixturevalue(which)
    S = x.S
    Z = RC.widths(CS.chain(S))
    params, ret = V5.unit(**x.st).signature()
    widths = {name: sum(BL.elements(t)) for name, t in params}
    assert set(widths) == set(V5.CLASSES) and Z.PW == 3 and sum(BL.elements(ret)) == 16
    assert widths["adv"] == S.DA + Z.PW == 8
    assert widths["coin"] == 384 and widths["chk"] == V5.data_bits(S)["chk"]
    assert V5.free_width(S, widths) == widths["adv"] + widths["chk"]
    assert widths["ing"] == S.K * (1 + S.IW + Z.LW)
    assert all(widths[r] % CS.WORD == 0 for r in V5.PADDED)
    sh = V5.shape(S)
    TT, RW, KW, NW = sh.TT, sh.RW, sh.KW, sh.Z3.NW
    assert widths["chk"] == (V4.data_bits(S)["chk"] - 2 * S.K * S.CU * (128 + TT) + S.NRB * (1 + RW + KW + NW + TT)
                             + S.NRK * (RW + TT))
    with pytest.raises(ValueError, match="at most D"):
        V5.shape(SimpleNamespace(**{**x.st, "DA": S.D + 1}))
    with pytest.raises(ValueError, match="at least one entry"):
        V5.shape(SimpleNamespace(**{**x.st, "NRK": 0}))


@pytest.mark.parametrize("which, calls, counts", [("synth", 19, [3, 3, 1]), ("asynth", 14, [3, 1, 3])])
def test_the_demo_is_its_reference(request, which, calls, counts):
    """The demo compiles (verdict 0x7F) with the reference's rows, counts and witness, its served rows and last state,
    window and slot rows are CompileSlice_v4's, its pool calls are the stream's 19 and 14, one hit of each a head-of-line
    memo, and its slots key 3, 3 and 1 blocks (3, 1 and 3), the RK blocks the stream reads."""
    x = request.getfixturevalue(which)
    (ref,) = _agrees(x.st, [x.sl])
    assert V5.ACCEPT == 0x7F
    _as_v4(x, x.sl, ref)
    assert len(ref.scan.stream.records) == calls and ref.scan.stream.memos == 1
    assert ref.counts == counts
    assert len(ref.cout) == V5.data_bits(x.S)["cout"] and len(ref.chk) == V5.data_bits(x.S)["chk"]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_registered_rows_are_held_to_the_computed(request, which):
    """A registered row one bit off the computed one, or with a padding bit set, is refused by its own verdict bit
    alone: ``srv`` by ``served`` (0x77), ``cout`` by ``carry`` (0x6F), ``req``'s padding by ``load`` (0x7E). Every bit
    of ``srv`` and ``cout`` is tried, the slot row's request numbers among them."""
    x = request.getfixturevalue(which)
    honest, nb, wb = V5.arguments(x.S, x.sl), V5.data_bits(x.S), V5.row_bits(x.S)
    want = {"srv": V5.ACCEPT & ~0x8, "cout": V5.ACCEPT & ~0x10, "req": V5.ACCEPT & ~0x1}
    cases, verdicts = [], []
    for r in ("srv", "cout"):
        for j in range(wb[r]):
            a = {**honest, r: list(honest[r])}
            a[r][j] ^= 1
            cases.append(a)
            verdicts.append(want[r])
    assert wb["req"] > nb["req"]
    for j in range(nb["req"], wb["req"]):
        a = {**honest, "req": list(honest["req"])}
        a["req"][j] = 1
        cases.append(a)
        verdicts.append(want["req"])
    assert V5.run(x.st, cases) == verdicts


def test_a_wrong_carried_request_number_is_refused(asynth):
    """The request numbers the window leaves are state, registered in cout's slot row: the asynchronous demo leaves
    request 2 running in slot 0, and cout naming any other number there is refused by ``carry`` alone (0x6F)."""
    x = asynth
    ref, RW, at = V5.reference(x.S, x.sl), x.S.K.bit_length(), _rq(x)
    assert sum(b << j for j, b in enumerate(ref.cout[at:at + RW])) == 2
    cases = []
    for q in range(1 << RW):
        if q != 2:
            cout = list(ref.cout)
            cout[at:at + RW] = CS._bits(q, RW)
            cases.append(V5.arguments(x.S, x.sl, cout=cout))
    assert V5.run(x.st, cases) == [V5.ACCEPT & ~0x10] * len(cases)


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_checked_witness_is_unique(request, which):
    """The checked witness one bit off the reference's is refused, by the cache, pool or keys bit: each field's first,
    middle and last bit, and 96 bits drawn at random. A hit, evictable count or return the steps read is held by the
    record that carries it, or is zero; a slot, a list entry or a time stamp off is the stream's check or the bus
    products' disagreement; a ragged block's number off breaks its runs or its RK entry."""
    x = request.getfixturevalue(which)
    honest = V5.arguments(x.S, x.sl)
    at, picks = 0, set()
    for _name, n, w in V5.fields(x.S)["chk"]:
        picks |= {at, at + n * w // 2, at + n * w - 1}
        at += n * w
    picks |= set(random.Random(4).sample(range(at), 96))
    cases = []
    for j in sorted(picks):
        a = {**honest, "chk": list(honest["chk"])}
        a["chk"][j] ^= 1
        cases.append(a)
    got = V5.run(x.st, cases)
    assert all(g != V5.ACCEPT and g & 0x64 != 0x64 for g in got), [(j, hex(g)) for j, g in zip(sorted(picks), got)
                                                                   if g == V5.ACCEPT or g & 0x64 == 0x64]


def test_the_print_coin_binds_the_keys(synth):
    """The coins are the verifier's: the reference's witness under other coins (α, γ, δ) compiles, and the witness made
    under one δ is refused under another by the pool bit alone (0x5F), since RK's values are the keys' prints."""
    x = synth
    other = tuple(c ^ (0x5A << 7 * i) for i, c in enumerate(V5.COINS))
    mixed = (other[0], other[1], V5.COINS[2])
    stale = V5.arguments(x.S, x.sl, coins=other, chk=V5.reference(x.S, x.sl, V5.COINS[2]).chk)
    assert V5.run(x.st, [V5.arguments(x.S, x.sl, coins=other), V5.arguments(x.S, x.sl, coins=mixed), stale]) == [
        V5.ACCEPT, V5.ACCEPT, V5.ACCEPT & ~0x20]


def test_the_ragged_row_binds_the_counts(synth):
    """A slot's keyed block count is what its registered tokens are keyed by: one block short of what the stream reads
    leaves an RK read with no entry, refused by the pool bit alone (0x5F); one block more is keyed and never read, and
    compiles; a token of a block the stream reads changed is refused by the pool bit; and blocks past the NRK the class
    holds are refused by the pool and keys bits (0x1F), as the reference refuses them."""
    x = synth
    S, ref = x.S, V5.reference(x.S, x.sl)
    assert ref.counts == [3, 3, 1] and S.CU > 1 and sum(ref.counts) < S.NRK

    def given(T, counts):
        return V5.arguments(T, x.sl, srv=ref.srv, cout=ref.cout, counts=counts, chk=V5.checked(T, ref.scan, counts))

    short, more = [3, 2, 1], [3, 3, 2]
    assert V5.reference(S, x.sl, counts=short) is None and V5.reference(S, x.sl, counts=more) is not None
    flip = given(S, ref.counts)
    at = V5.data_bits(S)["req"] - S.NRK * S.B * S.CTW
    flip["req"] = list(flip["req"])
    flip["req"][at + S.B * S.CTW] ^= 1
    assert V5.run(x.st, [given(S, short), given(S, more), flip]) == [V5.ACCEPT & ~0x20, V5.ACCEPT, V5.ACCEPT & ~0x20]
    small = {**x.st, "NRK": 6}
    T = SimpleNamespace(**small)
    assert V5.reference(T, x.sl) is None
    assert V5.run(small, [given(T, ref.counts)]) == [V5.ACCEPT & ~0x60]


def test_repeated_ids_are_refused_by_the_load(synth):
    """Request r is the arrival slot its id occupies, so arrivals repeating an id are refused by the load (the low bit),
    which the reference refuses too; the demo's own arrivals pass it."""
    x = synth
    a = x.sl.arrivals
    dup = replace(x.sl, arrivals=a[:2] + ((a[0][0],) + tuple(a[2][1:]),))
    assert V5.reference(x.S, dup) is None
    assert [g & 0x1 for g in V5.evaluate(x.st, [x.sl, dup])] == [1, 0]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_every_advice_is_its_reference(request, which):
    """On the demo's arrivals, each of the 32 decision strings at each step count 0 to 7 is advice the circuit compiles
    exactly where the reference does, with CompileSlice_v4's rows: a step count past P is refused by the load (0x6 in
    the low bits)."""
    x = request.getfixturevalue(which)
    slices = [replace(x.sl, decisions=d, steps=n) for d in itertools.product((0, 1), repeat=x.S.DA) for n in range(8)]
    refs = _agrees(x.st, slices)
    assert sum(r is not None for r in refs) == 32 * (x.S.P + 1)
    for sl, ref in zip(slices, refs):
        assert (ref is None) == (V4.reference(x.S, sl) is None), sl
        if ref is not None:
            _as_v4(x, sl, ref)
    over = [sl for sl in slices if sl.steps > x.S.P][:4]
    assert [g & 0x7 for g in V5.evaluate(x.st, over)] == [0x6] * len(over)


def _window(rng: random.Random, AS: int) -> CS.Slice:
    """Three arrivals of 4 to 16 tokens, most sharing a first block, and five random decisions (CompileSlice_v3's
    tests')."""
    base = [rng.randrange(1, 15) for _ in range(4)]
    arrivals, tokens = [], []
    for i in range(3):
        P, O = rng.randrange(4, 13), rng.randrange(1, 5)
        pre = base if rng.random() < 0.6 else [rng.randrange(1, 15) for _ in range(4)]
        tokens.append(tuple((pre + [rng.randrange(1, 15) for _ in range(P + O)])[:P + O]))
        arrivals.append((i, P, O, rng.randrange(2)) if AS else (i, P, O))
    return CS.Slice(tuple(arrivals), tuple(rng.randrange(2) for _ in range(5)), 5, tuple(tokens))


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_random_windows_under_eviction(request, which):
    """CompileSlice_v3's 40 random windows of the demo's class (26 synchronous and 13 asynchronous evict): the circuit
    accepts every one with the reference's rows and witness, CompileSlice_v4's rows, and refuses each with another
    window's ``srv`` and ``cout`` where they differ."""
    x = request.getfixturevalue(which)
    rng = random.Random(18 + x.S.AS)
    slices = [_window(rng, x.S.AS) for _ in range(40)]
    refs = _agrees(x.st, slices)
    assert all(refs)
    for sl, ref in zip(slices, refs):
        _as_v4(x, sl, ref)
    S2 = SimpleNamespace(**V3._v2(x.st3))
    assert sum(any(any(w[3]) for w in V2.scan(S2, sl).wit) for sl in slices) == (13 if x.S.AS else 26)
    swapped = [V5.arguments(x.S, sl, srv=refs[j - 1].srv, cout=refs[j - 1].cout) for j, sl in enumerate(slices)]
    want = [V5.ACCEPT & ~(0x8 * (refs[j - 1].srv != r.srv)) & ~(0x10 * (refs[j - 1].cout != r.cout))
            for j, r in enumerate(refs)]
    assert V5.run(x.st, swapped) == want and V5.ACCEPT not in want


def _narrow(S, T, chk: list[int]) -> list[int]:
    """`S`'s checked row cut to class `T`'s, each field's first elements (the classes' widths alike)."""
    out, at = [], 0
    for (_n, n, w), (_m, m, v) in zip(V5.fields(S)["chk"], V5.fields(T)["chk"]):
        assert v == w and m <= n
        out += chk[at:at + m * w]
        at += n * w
    return out


def test_refusals(synth):
    """What the class can't compile is refused, and the reference refuses it too: a prompt past the class's CU·B tokens,
    and the demo's window in a touched list of 4 M entries, past which its 6 run: with the witness cut to them, the
    stream's memory check fails and the pool bit alone refuses (0x5F)."""
    x = synth
    big = replace(x.sl, arrivals=((0, 30, 4),) + x.sl.arrivals[1:])
    assert V5.reference(x.S, big) is None and V5.evaluate(x.st, [big])[0] != V5.ACCEPT
    small = {**x.st, "NM": 4}
    T, ref = SimpleNamespace(**small), V5.reference(x.S, x.sl)
    assert V5.reference(T, x.sl) is None
    args = V5.arguments(T, x.sl, srv=ref.srv, cout=ref.cout, chk=_narrow(x.S, T, ref.chk), counts=ref.counts)
    assert V5.run(small, [args]) == [V5.ACCEPT & ~0x20]


def test_size(synth, asynth):
    """What checking the pool costs at the demos' classes, against CompileSlice_v4's 4,117,521 and 3,235,834: the stream's
    NS = 192 and 128 slots of 14,149 ANDs (`cache_step_v3.PoolSlot`), most of it; its touched list's 10 dense, 8 M and
    12 RB entries of 10,349; the ragged row's 8 blocks of 75,024 (`RaggedKey`, its key 62,240 of them); the steps' 90
    and 100 bus records folded in at 2,322; the 5 steps at 2,202 and 2,373; and the arrivals' 3 at 2,340."""
    assert V5.size(**synth.st) == {"ands": 3857177, "adv_bits": 8, "ing_bits": 27, "req_bits": 160, "srv_bits": 160,
                                   "cout_bits": 160, "coin_bits": 384, "chk_bits": 32764, "free_bits": 32772}
    assert V5.size(**asynth.st) == {"ands": 2975729, "adv_bits": 8, "ing_bits": 27, "req_bits": 160, "srv_bits": 160,
                                    "cout_bits": 176, "coin_bits": 384, "chk_bits": 23164, "free_bits": 23172}


def test_the_parts_price_the_unit(synth, asynth):
    """`compile_slice_v5.parts`, which prices a class too wide to build the body at, is the built unit: its parts' ANDs
    times their calls sum to `size`'s at both demos' classes."""
    for x in (synth, asynth):
        assert sum(n * ands(p) for p, n in V5.parts(**x.st)) == V5.size(**x.st)["ands"]
