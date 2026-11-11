"""The compile slice with the pool carried between windows (`compile_slice_v6`), both units proved every window:
`CompileSlice_v6`, CompileSlice_v5's load and steps with their calls held to the registered calls row, and `PoolWindow_v1`,
which opens the pool row under the input root as the memory check's initial values, runs CacheStep_v3's stream over the
calls with CompileSlice_v5's ragged keys and commits the row it leaves as the output root. A window from the empty pool
is CompileSlice_v5's; a window from the pool an earlier one left hits the blocks it cached, and `PoolChain_v1` is two
windows so chained. A forged input root, a row other than the one under the root, a witness made from another pool, an M
run that leaves out a carried key, a left row, root or source one bit off and a window that ends with a request open and
not cut are each refused by their own verdict bit. A window cut at its edge hands its open requests to the next window,
which takes their continuations as ordinary arrivals and hits the blocks they cached; `PageCopy_v1` holds a carried
page to its block's source."""
from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verity.protocols.certifier.verifier import compile_slice as CS
from verity.protocols.certifier.verifier import replay_chain as RC
from verity.protocols.certifier.verifier.head_check import ands
from verity.circuits import boolean as BL
from verity.commitments import rowleaf as RL
from verity.commitments.gates import rowleaf as RG
from verity.protocols.certifier.prover import cache_step_v3 as V3S, compile_slice_v5 as V5, compile_slice_v6 as V6

POOL, OPEN, CLOSE, KEYS = 0x1, 0x2, 0x4, 0x8
CALLS, EDGE = 0x20, 0x40
#: the two-window test's engine: three requests running, a 6-token budget of which a prefill takes two tokens a step
#: (long_prefill_token_threshold), six 4-token blocks and 24-token requests
CUT_ENGINE = {"max_num_seqs": 3, "max_num_batched_tokens": 6, "max_model_len": 24, "num_gpu_blocks": 7, "block_size": 4,
              "long_prefill_token_threshold": 2, "enable_prefix_caching": True, "enable_chunked_prefill": True,
              "scheduler_reserve_full_isl": False}


def _x(f):
    st, sl = f()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


@pytest.fixture(scope="module")
def synth():
    """The demo twice: its window from the empty pool, then again from the pool it leaves."""
    x = _x(V6.demo)
    x.w1 = V6.reference(x.S, x.sl)
    x.w2 = V6.reference(x.S, x.sl, x.w1.next)
    x.a1, x.a2 = V6.arguments(x.S, x.sl), V6.arguments(x.S, x.sl, x.w1.next)
    return x


@pytest.fixture(scope="module")
def asynth():
    return _x(V6.demo_async)


@pytest.fixture(scope="module")
def edge():
    """Two windows of five steps at (R 3, W 3, A 2) on `CUT_ENGINE`, NS = 128, NM = 8, NRB = 12, NRK = 8. Window w: X
    (an 11-token prompt), A and Z (2) and D (3), every decision 1. X prefills two tokens a step while A and Z decode; at
    step 5 Z's next block doesn't fit, so Z preempts itself. `cut_open` cuts w at its edge, where X is mid-prefill, A
    decoding, Z preempted and waiting with 3 outputs and D never admitted. Window w + 1, from the pool w leaves, takes
    their continuations (id, P + e), each its prompt then its e outputs and two more, in the order the edge leaves
    them, and is cut at its own edge."""
    st = V6.statics(3, 3, 2, CUT_ENGINE, 3, Q=4, D=8, P=5, K=4, CTW=4, CU=6, DA=8, NS=128, NM=8, NRB=12, NRK=8)
    S = SimpleNamespace(**st)
    tokens = tuple(tuple((5 * i + 3 * j) % 15 + 1 for j in range(S.L)) for i in range(4))
    raw = CS.Slice(tuple((i, P, S.L - P) for i, P in enumerate((11, 2, 2, 3))), (1,) * S.DA, S.P, tokens)
    w, cut = V6.cut_open(S, raw)
    r1 = V6.reference(S, w, cut=cut)
    opened = V6.opened_at_edge(S, r1.scan)
    P = {a[0]: a[1] for a in w.arrivals}
    nxt = CS.Slice(tuple((i, P[i] + e, 2) for i, e in opened), (1, 1) + (0,) * (S.DA - 2), S.P,
                   tuple(w.tokens[i][:P[i] + e] + (9, 9) for i, e in opened))
    w2, cut2 = V6.cut_open(S, nxt, r1.next)
    r2 = V6.reference(S, w2, r1.next, cut=cut2)
    return SimpleNamespace(st=st, S=S, w=w, cut=cut, r1=r1, opened=opened, w2=w2, cut2=cut2, r2=r2,
                           a1=V6.arguments(S, w, cut=cut), a2=V6.arguments(S, w2, r1.next, cut=cut2))


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def _hits(S, ref) -> int:
    return sum(_int(e) for e in RC._split(ref.res, V6.fields(S)["res"])["hits"])


def _offsets(fs) -> dict[str, tuple[int, int, int]]:
    out, at = {}, 0
    for name, n, w in fs:
        out[name] = (at, n, w)
        at += n * w
    return out


def _flip(a: dict, r: str, j: int) -> dict:
    out = {**a, r: list(a[r])}
    out[r][j] ^= 1
    return out


def _root_bits(bits) -> list[int]:
    return RG.digest_bits(RL.sha512_row_seg_digest(*RL.bit_row((list(bits), 1))))


def _window(rng: random.Random, AS: int, base: list[int]) -> CS.Slice:
    """Three arrivals of 4 to 16 tokens, most starting with the block ``base``, and five random decisions."""
    arrivals, tokens = [], []
    for i in range(3):
        P, O = rng.randrange(4, 13), rng.randrange(1, 5)
        pre = base if rng.random() < 0.6 else [rng.randrange(1, 15) for _ in range(4)]
        tokens.append(tuple((pre + [rng.randrange(1, 15) for _ in range(P + O)])[:P + O]))
        arrivals.append((i, P, O, rng.randrange(2)) if AS else (i, P, O))
    return CS.Slice(tuple(arrivals), tuple(rng.randrange(2) for _ in range(5)), 5, tuple(tokens))


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_free_inputs(request, which):
    """Every input row is classified. The compile slice's free input is its advice, CompileSlice_v5's DA + PW = 8 bits,
    and it reads no coin; the pool window's is its checked rows at their widths until their lemma, the left row, the
    blocks' sources and the stream's witness: its coins are the verifier's 384 bits, the roots and the calls are
    registered and the row under the input root is opened against it. The results and the stream's witness are
    CompileSlice_v5's witness at the class and the edge's R returns, and the calls its records and the edge's R. Each
    unit's output is 16 bits."""
    x = request.getfixturevalue(which)
    for unit, rows in ((V6.compile_unit, V6.COMPILE_ROWS), (V6.pool_unit, V6.POOL_ROWS)):
        params, ret = unit(**x.st).signature()
        widths = {name: sum(BL.elements(t)) for name, t in params}
        assert tuple(widths) == rows and sum(BL.elements(ret)) == 16
        assert widths == {r: V6.row_bits(x.S)[r] for r in rows}
        assert all(widths[r] % CS.WORD == 0 for r in rows if r in V6.PADDED)
    bits = V6.row_bits(x.S)
    assert V6.free_width(x.S, {r: bits[r] for r in V6.COMPILE_ROWS}) == bits["adv"] == 8
    pool_free = bits["nrow"] + bits["src"] + bits["chk"]
    assert V6.free_width(x.S, {r: bits[r] for r in V6.POOL_ROWS}) == pool_free and bits["coin"] == 384
    assert bits["pin"] == bits["pout"] == 512 and bits["prow"] == bits["nrow"] == V6.data_bits(x.S)["prow"]
    assert bits["src"] == 2 * (x.S.NB - 1)
    edge = x.S.R * V5.shape(x.S).CW
    assert V6.data_bits(x.S)["res"] + V6.data_bits(x.S)["chk"] == V5.data_bits(x.S)["chk"] + edge
    assert V6.nrec(x.S) == V5.nrec(x.S) + x.S.R and V6.rec_width(x.S) == sum(V5._rec_widths(x.S))
    assert V6.data_bits(x.S)["calls"] == V6.nrec(x.S) * V6.rec_width(x.S)
    assert (V6.COMPILE_ACCEPT, V6.POOL_ACCEPT, V6.COPY_ACCEPT) == (0x7F, 0xF, 0x1)


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_a_window_from_the_empty_pool_is_compile_slice_v5s(request, which):
    """From the empty pool (its root `genesis`) the reference is CompileSlice_v5's, rows and witness (the results and the
    rest of the witness), and the window wrote every page it keys: each keyed block is fresh. The demo's synchronous
    window leaves every request done and no block held, and both units accept it. The asynchronous one ends with a
    request running, which the compile slice's edge refuses (0x3F) and, its blocks held, the pool window's close (0xB)."""
    x = request.getfixturevalue(which)
    ref, r5 = V6.reference(x.S, x.sl), V5.reference(x.S, x.sl)
    n5 = V6.data_bits(x.S)["res"] - x.S.R * V5.shape(x.S).CW
    assert ref.srv == r5.srv and ref.cout == r5.cout and ref.res[:n5] + ref.chk == r5.chk and ref.counts == r5.counts
    assert all(f == k for k, f in ref.src) and any(k for k, _f in ref.src)
    args = V6.arguments(x.S, x.sl)
    assert args["pin"] == RG.digest_bits(V6.genesis(x.S)) and args["prow"] == V6.encode_row(x.S, V6.empty_row(x.S))
    quiet = ref.next is not None
    assert quiet == ref.edge == (not x.S.AS)
    assert V6.run_compile(x.st, [args]) == [V6.COMPILE_ACCEPT if quiet else V6.COMPILE_ACCEPT & ~EDGE]
    assert V6.run_pool(x.st, [args]) == [V6.POOL_ACCEPT if quiet else V6.POOL_ACCEPT & ~CLOSE]


def test_a_chained_window_hits_the_carried_pool(synth):
    """The demo again from the pool its first window leaves: the second window's input root is the first one's output
    root, its hits find the four blocks the first cached (5 hit blocks where the first had 3), its served rows differ,
    and both units accept both windows. Its sources keep two blocks keyed from the carried pool (keyed, not fresh: their
    pages are the first window's) beside the two it wrote. The row carries the keys, not their prints, so each window
    draws its own coins: under other coins the left row is the same, and the witness made under the first coins is
    refused."""
    x = synth
    assert x.a2["pin"] == x.a1["pout"] != x.a1["pin"]
    assert (_hits(x.S, x.w1), _hits(x.S, x.w2)) == (3, 5) and x.w2.srv != x.w1.srv
    assert [sum(k and f == fresh for k, f in x.w2.src) for fresh in (0, 1)] == [2, 2]
    assert sum(k != V6.NOKEY for k in x.w1.next.key) == 4 and x.w2.next is not None
    other = tuple(c ^ (0x5A << 7 * i) for i, c in enumerate(V6.COINS))
    fresh = V6.arguments(x.S, x.sl, x.w1.next, coins=other)
    assert fresh["nrow"] == x.a2["nrow"] and fresh["chk"] != x.a2["chk"]
    stale = {**fresh, "chk": x.a2["chk"]}
    assert V6.run_compile(x.st, [x.a1, x.a2]) == [V6.COMPILE_ACCEPT] * 2
    got = V6.run_pool(x.st, [x.a1, x.a2, fresh, stale])
    assert got[:3] == [V6.POOL_ACCEPT] * 3 and got[3] & (POOL | OPEN) == OPEN


def test_the_chain_shares_its_root(synth):
    """`PoolChain_v1` is the two windows wired: on the demo's chain every verdict accepts; with the middle root forged
    (the empty pool's, or one bit off) window 1's close and window 2's open refuse it, each alone (0xB and 0xD); and two
    windows whose roots don't meet aren't a chain."""
    x = synth
    honest = V6.chain_arguments(x.a1, x.a2)
    forged = [{**honest, "root1": RG.digest_bits(V6.genesis(x.S))}, _flip(honest, "root1", 0), _flip(honest, "root1", 511)]
    ok = (V6.COMPILE_ACCEPT, V6.POOL_ACCEPT) * 2
    bad = (V6.COMPILE_ACCEPT, V6.POOL_ACCEPT & ~CLOSE, V6.COMPILE_ACCEPT, V6.POOL_ACCEPT & ~OPEN)
    assert V6.run_chain(x.st, [honest] + forged) == [ok] + [bad] * len(forged)
    with pytest.raises(ValueError, match="doesn't open"):
        V6.chain_arguments(x.a1, x.a1)


def test_a_forged_root_or_a_wrong_opening_is_refused(synth):
    """The second window's input root forged (the empty pool's, its output root, zero, or one bit off) with the true
    row under it is refused by the open bit alone (0xD), and the row one bit off at each field's first, middle and last
    bit under the true root by the open bit.

    Under its own root a row is another pool, which the unit runs the window from: the true pool's witness is refused
    wherever the window's run from that pool differs from it (by the stream's check or the close), and accepted exactly
    where it doesn't, which is where the window reads no bit that differs. Here that is 4 of the 24 bits: the queue's
    tail's 3, which the window's allocations drain before any block returns to it, and one of the mlast of a block that
    is no key's first copy, which a valid close leaves 0. The 18 of those pools the reference compiles the window from
    each have their own run accepted. So a window's pool is the root's opening and nothing else; binding the root to the
    previous window's is the chain's. The window's witness from the empty pool under the carried root, and the carried
    window's witness under the empty pool's root and row, are refused by the stream's check."""
    x = synth
    S, honest = x.S, x.a2
    forged = [RG.digest_bits(V6.genesis(S)), honest["pout"], [0] * 512]
    forged += [_flip(honest, "pin", j)["pin"] for j in (0, 255, 511)]
    got = V6.run_pool(x.st, [{**honest, "pin": p} for p in forged])
    assert got == [V6.POOL_ACCEPT & ~OPEN] * len(forged)

    flips = sorted({j for at, n, w in _offsets(V6.row_fields(S)).values() for j in (at, at + n * w // 2, at + n * w - 1)})
    wrong = [_flip(honest, "prow", j) for j in flips]
    own = [{**a, "pin": _root_bits(a["prow"])} for a in wrong]
    runs = [(V6.reference(S, x.sl, V6.decode_row(S, a["prow"])), V6.arguments(S, x.sl, V6.decode_row(S, a["prow"])))
            for a in wrong]
    same = [all(r[k] == honest[k] for k in ("calls", "chk", "nrow")) for _ref, r in runs]
    theirs = [(r, V6.POOL_ACCEPT if ref.next is not None else V6.POOL_ACCEPT & ~CLOSE) for ref, r in runs if ref]
    empty = V6.arguments(S, x.sl)
    other = [{**honest, "chk": empty["chk"], "calls": empty["calls"]},
             {**honest, "pin": empty["pin"], "prow": empty["prow"]}]
    got = V6.run_pool(x.st, wrong + own + other + [r for r, _w in theirs])
    n = len(flips)
    under_true, under_own, under_other = got[:n], got[n:2 * n], got[2 * n:2 * n + 2]
    assert n == 24 and all(g & OPEN == 0 for g in under_true), [hex(g) for g in under_true]
    assert sum(same) == 4
    for j, g, s in zip(flips, under_own, same):
        assert g == V6.POOL_ACCEPT if s else (g != V6.POOL_ACCEPT and g & OPEN), (j, hex(g))
    assert all(g & POOL == 0 and g & OPEN for g in under_other), [hex(g) for g in under_other]
    assert got[2 * n + 2:] == [w for _r, w in theirs] == [V6.POOL_ACCEPT] * 18


def test_the_m_run_lists_every_carried_key(synth):
    """A window that doesn't touch a key the opened pool caches still lists it in its M run, untouched (its final value
    the opened one, its time 0). Without it the opening's record of the key has no partner and the close's first copy
    none either, so the stream's check and the close refuse (0xA), as they do an M run listing the key with another
    last copy."""
    x = synth
    S, rng, row = x.S, random.Random(7), x.w1.next
    carried = V6.heads(S, row, V6.COINS[2])
    for _ in range(40):
        sl, cut = V6.cut_open(S, _window(rng, S.AS, [rng.randrange(1, 15) for _ in range(4)]), row)
        ref = V6.reference(S, sl, row, cut=cut)
        untouched = set(carried) - {a for t, a in ref.scan.stream.first if t == V6.T3["M"]} if ref else set()
        if untouched:
            break
    else:
        raise AssertionError("no window of 40 leaves a carried key untouched")
    a = min(untouched)
    assert [(v, t) for b, v, t in ref.mrun if b == a] == [(carried[a], 0)]
    honest = V6.arguments(S, sl, row, cut=cut)
    dropped = {**honest, "chk": V6.with_mrun(S, honest["chk"], [m for m in ref.mrun if m[0] != a])}
    last = 1 << V5.shape(S).Z3.NW
    moved = {**honest, "chk": V6.with_mrun(S, honest["chk"], [(b, v ^ last if b == a else v, t) for b, v, t in ref.mrun])}
    assert V6.run_compile(x.st, [honest]) == [V6.COMPILE_ACCEPT]
    assert V6.run_pool(x.st, [honest, dropped, moved]) == [V6.POOL_ACCEPT] + [OPEN | KEYS] * 2


def test_the_left_row_is_held_to_the_stream(synth):
    """The left row one bit off at each field's first, middle and last bit, under the true output root or its own, is
    refused by the close bit alone (0xB): its links, keys and queue ends are the stream's finals, an mlast is the M run's
    last copy at a first copy and 0 elsewhere. The output root one bit off is too, and so is each bit of the blocks'
    sources. A window that ends with a request open and not cut (the demo at 1 to 3 steps) is refused by the compile
    slice's edge bit alone (0x3F); its pool window, whose calls hold the edge's frees, accepts it, no block held after
    them. Cut at its edge (`cut_open`), both units accept it."""
    x = synth
    S, honest = x.S, x.a2
    flips = sorted({j for at, n, w in _offsets(V6.row_fields(S)).values() for j in (at, at + n * w // 2, at + n * w - 1)})
    cases = [_flip(honest, "nrow", j) for j in flips]
    cases += [{**a, "pout": _root_bits(a["nrow"])} for a in cases]
    cases += [_flip(honest, "pout", j) for j in (0, 300, 511)]
    cases += [_flip(honest, "src", j) for j in range(V6.data_bits(S)["src"])]
    held = [replace(x.sl, steps=n) for n in (1, 2, 3)]
    assert all(not ref.edge and ref.next is None for ref in (V6.reference(S, sl, x.w1.next) for sl in held))
    busy = [V6.arguments(S, sl, x.w1.next) for sl in held]
    busy += [V6.arguments(S, sl, x.w1.next, cut=cut) for sl, cut in (V6.cut_open(S, sl, x.w1.next) for sl in held)]
    assert V6.run_compile(x.st, busy) == [V6.COMPILE_ACCEPT & ~EDGE] * 3 + [V6.COMPILE_ACCEPT] * 3
    got = V6.run_pool(x.st, cases + busy)
    assert got == [V6.POOL_ACCEPT & ~CLOSE] * len(cases) + [V6.POOL_ACCEPT] * 6, [hex(g) for g in got]


def test_the_calls_row_ties_the_units(synth):
    """The registered calls row is the compile slice's records and the pool window's bus: a bit of it off (every record's
    on bit, the edge's included, and each other field's bits of the first record on) is refused by the compile slice's
    calls bit (0x5F) and the pool window's pool bit (0xE); so is a padding bit set in the calls or results row by the
    calls bit. Results one bit off, with the records the steps make from them, are refused by one unit or the other."""
    x = synth
    S, honest = x.S, x.a2
    n, ofs = V6.nrec(S), _offsets(V6.fields(S)["calls"])
    on = [_int(honest["calls"][ofs["on"][0] + i:ofs["on"][0] + i + 1]) for i in range(n)]
    first = on.index(1)
    flips = [ofs["on"][0] + i for i in range(n)]
    flips += [at + first * w + k for name, (at, _n, w) in ofs.items() if name != "on" for k in range(w)]
    calls = [_flip(honest, "calls", j) for j in flips]
    nb, wb = V6.data_bits(S), V6.row_bits(S)
    pads = [_flip(honest, r, j) for r in ("calls", "res") for j in range(nb[r], wb[r])]
    assert pads
    assert V6.run_compile(x.st, calls + pads) == [V6.COMPILE_ACCEPT & ~CALLS] * (len(calls) + len(pads))
    assert V6.run_pool(x.st, calls) == [V6.POOL_ACCEPT & ~POOL] * len(calls)
    picks = random.Random(5).sample(range(nb["res"]), 24)
    res = [_flip(honest, "res", j) for j in picks]
    res = [{**a, "calls": V6._padded(S, "calls", c)} for a, c in zip(res, V6.records(x.st, res))]
    pairs = list(zip(V6.run_compile(x.st, res), V6.run_pool(x.st, res)))
    assert not any(c == V6.COMPILE_ACCEPT and p == V6.POOL_ACCEPT for c, p in pairs), [(hex(c), hex(p)) for c, p in pairs]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_random_chains(request, which):
    """Ten chains of up to four random windows sharing a first block, each from the pool the last left and cut at its
    edge (`cut_open`), a chain ending at a window that leaves a block held: 40 windows (15 asynchronous) with 35 (21)
    requests cut, every one of which the reference compiles. Synchronously the edge frees every running slot, so no
    window leaves a block held and both units accept each. Asynchronously a window with a request running at its edge
    is refused by the compile slice's edge bit (0x3F) and, its blocks held, the pool window's close (0xB), and the rest
    are accepted. Each of the 30 (5) chained windows' input root is the last output root, and 17 (3) of them hit blocks
    an earlier one cached (more hit blocks than from the empty pool)."""
    x = request.getfixturevalue(which)
    S, rng = x.S, random.Random(31 + x.S.AS)
    args, quiet, links, carried, cut = [], [], [], 0, 0
    for _chain in range(10):
        base, row = [rng.randrange(1, 15) for _ in range(4)], V6.empty_row(S)
        for _w in range(4):
            sl, cuts = V6.cut_open(S, _window(rng, S.AS, base), row)
            ref = V6.reference(S, sl, row, cut=cuts)
            if row != V6.empty_row(S):
                links.append(len(args))
                carried += _hits(S, ref) > _hits(S, V6.reference(S, sl, cut=cuts))
            args.append(V6.arguments(S, sl, row, cut=cuts))
            quiet.append(ref.next is not None)
            cut += sum(cuts)
            if ref.next is None:
                break
            row = ref.next
    assert (len(args), cut, len(links), carried) == ((15, 21, 5, 3) if S.AS else (40, 35, 30, 17))
    assert all(quiet) == (not S.AS) and all(args[i]["pin"] == args[i - 1]["pout"] for i in links)
    assert V6.run_compile(x.st, args) == [V6.COMPILE_ACCEPT if q else V6.COMPILE_ACCEPT & ~EDGE for q in quiet]
    assert V6.run_pool(x.st, args) == [V6.POOL_ACCEPT if q else V6.POOL_ACCEPT & ~CLOSE for q in quiet]


def _takes(ref) -> dict[int, int]:
    """Each request number's blocks taken from the pool's hits."""
    return {r[2]: r[4] for r in ref.scan.stream.records if r[1] == V3S.KINDS["take"]} if ref else {}


def test_cut_and_continue(edge):
    """Window w cuts one request of each kind at its edge: X mid-prefill (10 of its 11 prompt tokens computed, no output),
    A decoding (5 outputs), Z preempted and waiting (3 outputs) and D never admitted, each with a_O its outputs. The edge
    frees A's 2 blocks, then X's 3, and leaves no block held. Window w + 1 opens w's output root and takes their
    continuations first, in that order: X', A' and Z' each take the blocks it cached in w (2, 1 and 1), which from the
    empty pool none does, and D' has none. w + 1 cuts D' at its own edge. Both units accept both windows."""
    x = edge
    last = x.r1.scan.states[-1]
    assert [r[:5] for r in last.running] == [(0, 11, 13, 0, 10), (1, 2, 22, 5, 6)]
    assert [r[:4] for r in last.waiting] == [(2, 2, 22, 3)] and [a[0] for a in x.r1.scan.window.queue[last.qh:]] == [3]
    assert x.opened == [(0, 0), (1, 5), (2, 3), (3, 0)] and x.cut == (1, 1, 1, 1)
    assert [a[2] for a in x.w.arrivals] == [0, 5, 3, 0] and x.r1.edge and x.r1.next is not None
    free = V3S.KINDS["free"]
    assert [r[1:6] for r in x.r1.scan.stream.records[-2:]] == [(free, 1, 2, 0, 2), (free, 0, 3, 0, 3)]
    assert [a[:2] for a in x.w2.arrivals] == [(0, 11), (1, 7), (2, 5), (3, 3)]
    assert all(x.w2.tokens[k][:P] == x.w.tokens[k][:P] for k, (_i, P, _O) in enumerate(x.w2.arrivals))
    assert _takes(x.r2) == {0: 2, 1: 1, 2: 1} and _takes(V6.reference(x.S, x.w2, cut=x.cut2)) == {}
    assert x.cut2 == (0, 0, 0, 1) and x.r2.edge and x.a2["pin"] == x.a1["pout"]
    assert V6.run_compile(x.st, [x.a1, x.a2]) == [V6.COMPILE_ACCEPT] * 2
    assert V6.run_pool(x.st, [x.a1, x.a2]) == [V6.POOL_ACCEPT] * 2


def _swap(a: dict, r: str, fs, i: int, j: int, n: int) -> dict:
    """`a` with entries i and j of each of row r's fields of n entries swapped."""
    out = {**a, r: list(a[r])}
    for at, cnt, w in _offsets(fs).values():
        if cnt == n:
            x, y = slice(at + i * w, at + (i + 1) * w), slice(at + j * w, at + (j + 1) * w)
            out[r][x], out[r][y] = a[r][y], a[r][x]
    return out


def test_a_flipped_cut_bit_a_p_or_order_is_refused(edge):
    """Each of w's four cut bits flipped, each continuation's a_P in w + 1 one off, and w + 1's continuations in another
    order (A' and Z' swapped, with the true order's witness) are each refused by the compile slice. So are w's two
    edge frees in the other order, by its calls bit alone (0x5F), and by the pool window's pool bit (0xE). In another
    order the continuations are another window: the certified arrivals' order is the edge's."""
    x = edge
    S, ofs = x.S, _offsets(V6.fields(x.S)["req"])
    cuts = [_flip(x.a1, "req", ofs["a_cut"][0] + k) for k in range(4)]
    at, _n, w = _offsets(V6.fields(S)["ing"])["a_P"]
    ps = [_flip(x.a2, "ing", at + k * w) for k in range(4)]
    order = [1, 0, 2, 3]
    sw = CS.Slice(tuple(x.w2.arrivals[k] for k in order), x.w2.decisions, x.w2.steps, tuple(x.w2.tokens[k] for k in order))
    sw, sc = V6.cut_open(S, sw, x.r1.next)
    other = V6.arguments(S, sw, x.r1.next, cut=sc)
    assert other["srv"] != x.a2["srv"]
    swapped = {**x.a2, "ing": other["ing"], "req": other["req"]}
    n = V6.nrec(S)
    edges = _swap(x.a1, "calls", V6.fields(S)["calls"], n - 2, n - 1, n)
    got = V6.run_compile(x.st, cuts + ps + [swapped, edges])
    assert all(g != V6.COMPILE_ACCEPT for g in got) and got[-1] == V6.COMPILE_ACCEPT & ~CALLS, [hex(g) for g in got]
    assert V6.run_pool(x.st, [edges]) == [V6.POOL_ACCEPT & ~POOL]


def test_the_carried_pages(edge):
    """The blocks' sources are the stream's: window w, from the empty pool, wrote every page it keys (4 keyed, all
    fresh), so it reads no carried row; w + 1 keys 3 blocks its continuations took from the carried pool (their pages are
    w's carried port's rows) and 1 it wrote. `PageCopy` at CB = 8 accepts each block's page from its source (`copied`),
    and refuses the other source's page at a keyed block and a nonzero page at a block with no key."""
    x = edge
    assert [sum(k and f == fresh for k, f in x.r1.src) for fresh in (0, 1)] == [0, 4]
    assert [sum(k and f == fresh for k, f in x.r2.src) for fresh in (0, 1)] == [3, 1]
    rng, n = random.Random(11), x.S.NB - 1
    wrote, carried = ([rng.randrange(1, 256) for _ in range(n)] for _ in range(2))
    carried = [c ^ 1 if c == v else c for c, v in zip(carried, wrote)]
    assert V6.copied(x.r1.src, wrote, carried) == V6.copied(x.r1.src, wrote, [0] * n)

    def row(src, w, c, o):
        return {"keyed": [src[0]], "fresh": [src[1]], "wrote": CS._bits(w, 8), "carried": CS._bits(c, 8),
                "out": CS._bits(o, 8)}

    honest = [row(s, w, c, o) for s, w, c, o in zip(x.r2.src, wrote, carried, V6.copied(x.r2.src, wrote, carried))]
    wrong = [row(s, w, c, (c if s[1] else w) if s[0] else w) for s, w, c in zip(x.r2.src, wrote, carried)]
    assert V6.run_copy(8, honest + wrong) == [V6.COPY_ACCEPT] * n + [0] * n


def test_size(synth, asynth):
    """What each unit costs at the demos' classes (`compile_slice_v6.size`). The compile slice is CompileCut (24 and 27
    ANDs), CompileWindow with its edge (CompileEdge, 162 and 98) and the calls row's check (a CallsEq a record, the
    edge's R = 2 among them), 13,466 and 14,497 ANDs, 224 and 163 more than without the edge. The pool window, 4,543,787
    and 3,661,471, is CompileSlice_v5's stream, bus (the edge's two records at 2,322 each among them) and ragged keys
    with 8 more M entries, the arrivals' counts, the two rows' digests (four compressions each at 2,641 bits), their ten
    keys' prints, five open and five close steps (the close's source check 4 ANDs a block) and 16 copy steps. Its free
    input is the left row, the blocks' sources (2 bits a block) and the stream's witness."""
    common = {"adv_bits": 8, "ing_bits": 27, "req_bits": 160, "srv_bits": 160, "res_bits": 208, "pin_bits": 512,
              "pout_bits": 512, "coin_bits": 384, "prow_bits": 2641, "nrow_bits": 2641, "src_bits": 10,
              "compile_free_bits": 8}
    assert V6.size(**synth.st) == {**common, "compile_ands": 13466, "pool_ands": 4543787, "cout_bits": 160,
                                   "calls_bits": 2032, "chk_bits": 34684, "pool_free_bits": 37335}
    assert V6.size(**asynth.st) == {**common, "compile_ands": 14497, "pool_ands": 3661471, "cout_bits": 176,
                                    "calls_bits": 2256, "chk_bits": 25084, "pool_free_bits": 27735}
    assert ands(V6.chain_unit(**synth.st)) == 2 * (13466 + 4543787)


def test_the_parts_price_the_units(synth, asynth):
    """`compile_slice_v6.parts`, which prices a class too wide to build the bodies at, is the built units: each unit's
    parts' ANDs times their calls sum to `size`'s at both demos' classes."""
    for x in (synth, asynth):
        sz = V6.size(**x.st)
        for unit, ps in V6.parts(**x.st).items():
            assert sum(n * ands(p) for p, n in ps) == sz[f"{unit}_ands"], unit
