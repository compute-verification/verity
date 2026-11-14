"""The compile slice over the memory-checked pool (`compile_slice_v4.CompileSlice_v4`): the load and the steps of one
window, each pool call a record on a bus to CacheStep_v3's stream of memory-checked slots, which runs in the unit from
the empty pool with RK's keys the arrivals' BlockKey_v1 chains printed under δ. The hits, evictable counts and returns
the steps read are checked witness the bus carries, and so are the stream's slots and touched list; its free input is
the advice, DA + PW bits, and that checked witness at its width until its lemma. The circuit accepts exactly where
`compile_slice_v4.reference` compiles, and its served rows and last state and window rows are the pinned ones
(`REFERENCE_ROWS`), on both demonstration slices, on every advice of their arrivals and on random windows under
eviction. A registered row one bit off is refused by its own verdict bit, a checked witness one bit off by the cache or
pool bit, and a witness made under one print coin by the pool bit under another."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verity.protocols.certifier.verifier import compile_slice as CS
from verity.protocols.certifier.prover import cache_step_v3 as V3S, compile_slice_v4 as V4
from verity.protocols.certifier.verifier import replay_chain as RC
from verity.circuits import boolean as BL


#: SHA-512 of the reference's served rows and last state and window rows on each test's slices, by demonstration and
#: test (`_as_reference`): the window's schedule, pinned, so a change to the reference that moves a row shows here
REFERENCE_ROWS = {
    ("synth", "demo"):
        "4d5de324d81085faff5cc9c347c129d7ed4902725a6ee4d452ec74da80fa5b1a"
        "1ee2522ba535700330a38134a9007e7600684942ac76019690d64985c321cb07",
    ("synth", "advice"):
        "06c68ebde694b3d682e2ffc62a8890fba7658e8c18706b428b27530c9f5be1f8"
        "71e8bae4cd32632ec98de753cb9a65d42d50fa218b8bad5cce5dc452a1f45844",
    ("synth", "random"):
        "fda616b2f06eb03a7b3edf3f7150c78937bf07e0af4f596aaca5156c4f0e3107"
        "8c8e887d1cff9f8ed90bc68556a4b7596624ce2b74bffdf5ff5d55a40e7f4a17",
    ("asynth", "demo"):
        "be19f32f5bee82fb7c64e1ebb4e068727020565f2faf7f44936f8dc65968315f"
        "90637ae0a8920570ea064dd879b2529134eb1c21b13a0f89973d0d38cb07b170",
    ("asynth", "advice"):
        "9a09d0e40cbf34e14f5d5aac16cadc55ffbff993ffa9b294420f9fc43cafd96e"
        "236da313e0aff70a2408324759fdc033340e209ce88c654821072415214490f3",
    ("asynth", "random"):
        "5a7e32c381c8a67c962d448f777e08e41cb6f3263ef32887af37c27a66b359a8"
        "041b4adb528949dc77f57bd30fd5ce272cf635fb11e366c45f283ad5bbbb49e0",
}


def _x(f, which: str):
    st, sl = f()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, which=which)


@pytest.fixture(scope="module")
def synth():
    return _x(V4.demo, "synth")


@pytest.fixture(scope="module")
def asynth():
    return _x(V4.demo_async, "asynth")


def _agrees(st: dict, slices: list[CS.Slice]) -> list:
    """The circuit's verdict on `slices`, the reference's rows and witness given, is `ACCEPT` exactly where the reference
    compiles."""
    S = SimpleNamespace(**st)
    refs = [V4.reference(S, sl) for sl in slices]
    for sl, ref, got in zip(slices, refs, V4.evaluate(st, slices)):
        assert (got == V4.ACCEPT) == (ref is not None), sl
    return refs


def _as_reference(x, test: str, slices: list[CS.Slice], refs: list) -> None:
    """`refs`' served rows and last state and window rows on `slices`, None where the reference refuses, are the pinned
    ones (`REFERENCE_ROWS`); the slot row past them is the frame's own."""
    Sc = CS.chain(x.S)
    n = RC.width(Sc) + RC.window_width(Sc)
    rows = [None if r is None else [r.srv, r.cout[:n]] for r in refs]
    got = hashlib.sha512(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    assert got == REFERENCE_ROWS[x.which, test], [sl for sl, r in zip(slices, refs) if r is None]


def _evicts(ref) -> bool:
    """Whether the reference's window evicts: an alloc of its stream takes a block whose key is cached and clears it."""
    return any(s.use and s.tag == V3S.TAGS["K"] and s.v and not s.w for s in ref.scan.stream.slots)


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_free_input_is_the_advice_and_the_checked_witness(request, which):
    """Every input row is classified, and the unit's free input is the advice, DA decision bits and the PW = 3 bits of the
    step count (8 at the demo's class), and the checked witness at its width until its lemma: the coins (α, γ, δ) are the
    verifier's 384 bits. Its one output is 16 bits."""
    x = request.getfixturevalue(which)
    Z = RC.widths(CS.chain(x.S))
    params, ret = V4.unit(**x.st).signature()
    widths = {name: sum(BL.elements(t)) for name, t in params}
    assert set(widths) == set(V4.CLASSES) and Z.PW == 3 and sum(BL.elements(ret)) == 16
    assert widths["adv"] == x.S.DA + Z.PW == 8
    assert widths["coin"] == 384 and widths["chk"] == V4.data_bits(x.S)["chk"]
    assert V4.free_width(x.S, widths) == widths["adv"] + widths["chk"]
    assert widths["ing"] == x.S.K * (1 + x.S.IW + Z.LW)
    assert all(widths[r] % CS.WORD == 0 for r in V4.PADDED)
    with pytest.raises(ValueError, match="at most D"):
        V4.shape(SimpleNamespace(**{**x.st, "DA": x.S.D + 1}))


@pytest.mark.parametrize("which, calls", [("synth", 19), ("asynth", 14)])
def test_the_demo_is_its_reference(request, which, calls):
    """The demo compiles (verdict 0x3F) with the reference's rows and witness, its served rows and last state and window
    rows are the pinned ones, and its pool calls are the stream's 19 and 14, one hit of each a head-of-line memo."""
    x = request.getfixturevalue(which)
    (ref,) = _agrees(x.st, [x.sl])
    assert V4.ACCEPT == 0x3F
    _as_reference(x, "demo", [x.sl], [ref])
    assert len(ref.scan.stream.records) == calls and ref.scan.stream.memos == 1
    assert len(ref.cout) == V4.data_bits(x.S)["cout"] and len(ref.chk) == V4.data_bits(x.S)["chk"]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_registered_rows_are_held_to_the_computed(request, which):
    """A registered row one bit off the computed one, or with a padding bit set, is refused by its own verdict bit
    alone: ``srv`` by ``served`` (0x37), ``cout`` by ``carry`` (0x2F), ``req``'s padding by ``load`` (0x3E). Every bit
    of ``srv`` and ``cout`` is tried."""
    x = request.getfixturevalue(which)
    honest, nb, wb = V4.arguments(x.S, x.sl), V4.data_bits(x.S), V4.row_bits(x.S)
    want = {"srv": V4.ACCEPT & ~0x8, "cout": V4.ACCEPT & ~0x10, "req": V4.ACCEPT & ~0x1}
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
    assert V4.run(x.st, cases) == verdicts


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_checked_witness_is_unique(request, which):
    """The checked witness one bit off the reference's is refused, by the cache or the pool bit: each field's first,
    middle and last bit, and 96 bits sampled at random. A hit, evictable count or return the steps read is held by the
    record that carries it, or is zero; a slot, a list entry or a time stamp off is the stream's check or the bus
    products' disagreement."""
    x = request.getfixturevalue(which)
    honest = V4.arguments(x.S, x.sl)
    at, picks = 0, set()
    for _name, n, w in V4.fields(x.S)["chk"]:
        picks |= {at, at + n * w // 2, at + n * w - 1}
        at += n * w
    picks |= set(random.Random(4).sample(range(at), 96))
    cases = []
    for j in sorted(picks):
        a = {**honest, "chk": list(honest["chk"])}
        a["chk"][j] ^= 1
        cases.append(a)
    got = V4.run(x.st, cases)
    assert all(g != V4.ACCEPT and g & 0x24 != 0x24 for g in got), [(j, hex(g)) for j, g in zip(sorted(picks), got)
                                                                   if g == V4.ACCEPT or g & 0x24 == 0x24]


def test_the_print_coin_binds_the_keys(synth):
    """The coins are the verifier's: the reference's witness under other coins (α, γ, δ) compiles, and the witness made
    under one δ is refused under another by the pool bit alone (0x1F), since RK's initial values are the keys' prints."""
    x = synth
    other = tuple(c ^ (0x5A << 7 * i) for i, c in enumerate(V4.COINS))
    mixed = (other[0], other[1], V4.COINS[2])
    stale = V4.arguments(x.S, x.sl, coins=other, chk=V4.reference(x.S, x.sl, V4.COINS[2]).chk)
    assert V4.run(x.st, [V4.arguments(x.S, x.sl, coins=other), V4.arguments(x.S, x.sl, coins=mixed), stale]) == [
        V4.ACCEPT, V4.ACCEPT, V4.ACCEPT & ~0x20]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_every_advice_is_its_reference(request, which):
    """On the demo's arrivals, each of the 32 decision strings at each step count 0 to 7 is advice the circuit compiles
    exactly where the reference does, with the pinned rows: a step count past P is refused by the load (0x6 in the low
    bits)."""
    x = request.getfixturevalue(which)
    slices = [replace(x.sl, decisions=d, steps=n) for d in itertools.product((0, 1), repeat=x.S.DA) for n in range(8)]
    refs = _agrees(x.st, slices)
    assert sum(r is not None for r in refs) == 32 * (x.S.P + 1)
    _as_reference(x, "advice", slices, refs)
    over = [sl for sl in slices if sl.steps > x.S.P][:4]
    assert [g & 0x7 for g in V4.evaluate(x.st, over)] == [0x6] * len(over)


def _window(rng: random.Random, AS: int) -> CS.Slice:
    """Three arrivals of 4 to 16 tokens, most sharing a first block, and five random decisions."""
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
    """40 random windows of the demo's class, of which 26 synchronous and 13 asynchronous evict (`_evicts`): the circuit
    accepts every one with the reference's rows and witness, the pinned rows, and refuses each with another window's
    ``srv`` and ``cout`` where they differ."""
    x = request.getfixturevalue(which)
    rng = random.Random(18 + x.S.AS)
    slices = [_window(rng, x.S.AS) for _ in range(40)]
    refs = _agrees(x.st, slices)
    assert all(refs)
    _as_reference(x, "random", slices, refs)
    assert sum(_evicts(ref) for ref in refs) == (13 if x.S.AS else 26)
    swapped = [V4.arguments(x.S, sl, srv=refs[j - 1].srv, cout=refs[j - 1].cout) for j, sl in enumerate(slices)]
    want = [V4.ACCEPT & ~(0x8 * (refs[j - 1].srv != r.srv)) & ~(0x10 * (refs[j - 1].cout != r.cout))
            for j, r in enumerate(refs)]
    assert V4.run(x.st, swapped) == want and V4.ACCEPT not in want


def _narrow(S, T, chk: list[int]) -> list[int]:
    """`S`'s checked row cut to class `T`'s, each field's first elements (the classes' widths alike)."""
    out, at = [], 0
    for (_n, n, w), (_m, m, v) in zip(V4.fields(S)["chk"], V4.fields(T)["chk"]):
        assert v == w and m <= n
        out += chk[at:at + m * w]
        at += n * w
    return out


def test_refusals(synth):
    """What the class can't compile is refused, and the reference refuses it too: a prompt past the class's CU·B tokens,
    and the demo's window in a touched list of 4 M entries, past which its 6 run: with the witness cut to them, the
    stream's memory check fails and the pool bit alone refuses (0x1F)."""
    x = synth
    big = replace(x.sl, arrivals=((0, 30, 4),) + x.sl.arrivals[1:])
    assert V4.reference(x.S, big) is None and V4.evaluate(x.st, [big])[0] != V4.ACCEPT
    small = {**x.st, "NM": 4}
    T, ref = SimpleNamespace(**small), V4.reference(x.S, x.sl)
    assert V4.reference(T, x.sl) is None
    args = V4.arguments(T, x.sl, srv=ref.srv, cout=ref.cout, chk=_narrow(x.S, T, ref.chk))
    assert V4.run(small, [args]) == [V4.ACCEPT & ~0x20]


def test_size(synth, asynth):
    """What checking the pool costs, at the demos' classes: the stream's NS = 192 and 128 slots of 14,149 ANDs
    (`cache_step_v3.PoolSlot`), most of it; its touched list's 34 dense and 8 M entries of 10,349; the arrivals' 12 block
    keys of 62,240 (`block_key.BlockKey`); the steps' 90 and 100 bus records folded in at 2,322; and the load and steps
    themselves, 10,130 and 10,759."""
    assert V4.size(**synth.st) == {"ands": 4117521, "adv_bits": 8, "ing_bits": 27, "req_bits": 208, "srv_bits": 160,
                                   "cout_bits": 160, "coin_bits": 384, "chk_bits": 35744, "free_bits": 35752}
    assert V4.size(**asynth.st) == {"ands": 3235834, "adv_bits": 8, "ing_bits": 27, "req_bits": 224, "srv_bits": 160,
                                    "cout_bits": 176, "coin_bits": 384, "chk_bits": 26144, "free_bits": 26152}
