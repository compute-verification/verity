"""`registry.dsa_topk_reads`, DSA's long row by table reads, held to the rows network's order and to today's long row:

* a block's `DsaMaskPositions_v1` is its twin (`values`: `dsa_topk_mask.butterfly_apply` at the block's settings over its
  positions) on masks of no member to every key, and its first count slots are the members' positions, ascending, whenever it
  has at most K; `DsaPositionsMerge_v1` is `merge` on lists with words past their counts, counts past K and lists that fill K;
* `DsaPositionsMerge_v1`, each block's list and count its own port, is gate for gate the merge over one port of the lists
  and one of the counts, at small statics and at the served one (M 8, K 2,048);
* `DsaPositionsChunk_v1` and then `DsaPositionsSlot_v1` over its chunk give the merge's element j for every j on the merge's
  own outputs, 0 where it is N or more and at an index out of range, at small statics and the served ones on random rows;
* the units composed over a row's blocks (each on the outputs before it) give the rows network's order over the whole row
  (`DsaTopKRow_v2`'s construction), then 0, on masks of at most K members;
* `DsaTopKReadRows_v1` over a table that holds a history gives today's long row's kv (`DsaButterflyRowsPad_v1` at the counted
  mask's settings over the history and a zero pad) with every ok 1, at small statics (the mask whole, so one block), and ok 0
  at a slot whose opening is not the table's;
* `cut` counts its units at the served statics, the mask whole a unit up to `WHOLE_MAX` keys and read by one block's
  positions, and malformed bindings are refused;
* every unit of the cut at every served class has the rows Flock's composer proves and passes the window builder's
  `interface.check_proof_block`, each mask in u16 words (`dsa_topk_mask_cut.words`), which a block's positions read whatever
  the bits past its keys are.
"""
from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean.trace import Traced, trace
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.types import Array
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as EM
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import bit_row
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry import dsa_topk_reads as R
from verity.commitments.windows import interface as IF

SERVED_TOPK, SERVED_V, SERVED_H = 2048, 1 << 17, 576
#: the score of a key past a row's own in today's long row (vllm_dsa's PAD_SCORE)
PAD_SCORE = 0xFFFFFFFF


def _bits(words, width: int) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[..., None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def _words(bits: np.ndarray, width: int) -> list[int]:
    b = np.asarray(bits, dtype=np.uint64).reshape(-1, width)
    return [int(x) for x in (b << np.arange(width, dtype=np.uint64)).sum(axis=1)]


def _masks(rng: random.Random, N: int, K: int, R_: int, most: int | None = None) -> list[list[int]]:
    """R_ masks of N bits: none, every key, and random ones of each member count up to ``most`` (N when None)."""
    most = N if most is None else most
    out = [[0] * N] + ([[1] * N] if most >= N else [])
    while len(out) < R_:
        c = rng.randrange(min(most, N) + 1)
        on = set(rng.sample(range(N), c))
        out.append([int(j in on) for j in range(N)])
    return out


def _positions(N: int, AT: int, K: int, masks: list[list[int]], rng: random.Random | None = None
               ) -> tuple[list[list[int]], list[int]]:
    """The masks through `DsaMaskPositions_v1`, each in its u16 words: the bits past N 0, or random with ``rng``."""
    pad = [[rng.getrandbits(1) if rng else 0 for _ in range(X.wide(N) - N)] for _ in masks]
    got = np.asarray(BS.evaluate_bits(bind(R.DsaMaskPositions, N=N, AT=AT, K=K),
                                      np.asarray([m + p for m, p in zip(masks, pad)], dtype=np.uint8)))
    got = got.reshape(len(masks), -1)
    return [_words(g[:32 * K], 32) for g in got], [_words(g[32 * K:], 32)[0] for g in got]


def _merge(M_: int, K: int, lists: np.ndarray, counts: np.ndarray) -> list[list[int]]:
    """``lists`` (rows, M_, K) and ``counts`` (rows, M_) u32 words through the circuit, block b's list and count its ports."""
    rows = len(lists)
    got = BS.evaluate_bits(bind(R.DsaPositionsMerge, M=M_, K=K),
                           *(_bits(lists[:, b], 32).reshape(rows, -1) for b in range(M_)),
                           *(_bits(counts[:, b], 32).reshape(rows, -1) for b in range(M_)))
    return [_words(g, 32) for g in np.asarray(got).reshape(rows, -1)]


def _digest(t: Traced) -> str:
    """A trace's inputs, outputs and gates, hashed a slice of gates at a time, so that a served trace needn't be held twice."""
    h = hashlib.sha512(repr((t.n_inputs, t.outputs)).encode())
    for i in range(0, len(t.gates), 1 << 16):
        h.update(repr(t.gates[i:i + (1 << 16)]).encode())
    return h.hexdigest()


@pytest.mark.parametrize("N,AT,K", [(8, 0, 4), (6, 8, 4), (5, 3, 1), (16, 16, 16), (12, 1 << 17, 4), (2, 0, 2), (21, 0, 8)])
def test_a_blocks_positions_are_its_twin_and_its_members_in_order(N, AT, K) -> None:
    """Its mask's words past N bits are random: the block ignores them."""
    rng = random.Random(N * 1000 + AT + K)
    masks = _masks(rng, N, K, 40)
    pos, counts = _positions(N, AT, K, masks, rng)
    assert (pos, counts) == _positions(N, AT, K, masks)
    for m, p, c in zip(masks, pos, counts):
        want = tuple(M.butterfly_apply(M.butterfly_settings(m, K), list(range(AT, AT + N)), K, 0)), sum(m)
        assert (tuple(p), c) == want, (m, p, c)
        if AT == 0:
            assert R.values(m, K, block=N)[("positions", 0)] == want
        members = [AT + j for j, b in enumerate(m) if b]
        if c <= K:
            assert p[:c] == members


@pytest.mark.parametrize("M_,K", [(1, 4), (2, 4), (3, 4), (3, 5), (2, 1), (4, 3)])
def test_the_merge_is_merge(M_, K) -> None:
    rng = random.Random(M_ * 100 + K)
    edge = [0, 1, K - 1, K, K + 1, 1 << 31, (1 << 32) - 1]
    lists = np.array([[[rng.getrandbits(32) for _ in range(K)] for _ in range(M_)] for _ in range(48)], dtype=np.uint64)
    counts = np.array([[edge[(r + b) % len(edge)] if r < 2 * len(edge) else rng.randrange(K + 2) for b in range(M_)]
                       for r in range(48)], dtype=np.uint64)
    got = _merge(M_, K, lists, counts)
    for li, c, g in zip(lists.tolist(), counts.tolist(), got):
        assert g == R.merge(li, c, K), (li, c, g)


@pytest.mark.parametrize("M_,K", [(1, 4), (3, 4), (4, 3), (8, 16),
                                  pytest.param(8, SERVED_TOPK, marks=pytest.mark.slow, id="served")])
def test_the_merge_with_ports_a_block_is_the_one_port_merge_gate_for_gate(M_, K) -> None:
    """Against the merge over one port of the M blocks' lists and one of their counts.  The served binding (8 blocks, 2,048
    slots, 12-bit shifts) is about 6M ANDs; it is the one the long row calls, and no smaller binding's trace is its trace."""
    S = SimpleNamespace(M=M_, K=K)
    params, ret = R._merge_sig(S)
    assert [n for n, _ in params] == [f"pos_{b}" for b in range(M_)] + [f"count_{b}" for b in range(M_)]
    assert [t for _, t in params] == [Array(K, R.I32)] * M_ + [R.I32] * M_ and ret == Array(K, R.I32)
    split = _digest(trace(R._merge(S), [t for _, t in params], ret))
    one = _digest(trace(R._merge(S), [Array(M_, Array(K, R.I32)), Array(M_, R.I32)], ret))
    assert split == one


def _chunk(K: int, W: int, slots: np.ndarray, cs: list[int]) -> list[list[int]]:
    """Each row of ``slots`` (rows, K) u32 words, with its c, through `DsaPositionsChunk_v1{K,W}`."""
    got = BS.evaluate_bits(bind(R.DsaPositionsChunk, K=K, W=W), _bits(slots, 32).reshape(len(cs), -1), _bits(cs, 32))
    return [_words(g, 32) for g in np.asarray(got).reshape(len(cs), -1)]


def _slot(K: int, N: int, slots: np.ndarray, js: list[int]) -> list[int]:
    """Each row of ``slots`` (rows, K) u32 words, with its j, through `DsaPositionsSlot_v1{K,N}`."""
    got = BS.evaluate_bits(bind(R.DsaPositionsSlot, K=K, N=N), _bits(slots, 32).reshape(len(js), -1), _bits(js, 32))
    return _words(np.asarray(got).reshape(len(js), -1), 32)


@pytest.mark.parametrize("M_,K,W,N", [(3, 4, 2, 20), (2, 5, 2, 12), (1, 1, 1, 4), (3, 8, 4, 40), (2, 6, 4, 30),
                                      (3, 8, 16, 40)])
def test_slot_j_of_its_chunk_is_the_merges_element_j(M_, K, W, N) -> None:
    """On the merge's own outputs (lists of positions below N and of any u32, counts from 0 past K): chunk c is the merge's
    slots cW .. cW+W-1 and 0 past K, and the slot unit at j mod W of chunk j // W is the merge's element j for every j < K,
    0 where the element is N or more; every slot is below N.  Chunks of 2 of 4 and 5 slots (a last chunk of one), of 4 of 6,
    and one chunk of 16 over 8."""
    rng = random.Random(M_ * 1000 + K * 100 + W * 10 + N)
    lists = np.array([[[rng.randrange(N) if rng.random() < 0.7 else rng.getrandbits(32) for _ in range(K)]
                       for _ in range(M_)] for _ in range(24)], dtype=np.uint64)
    counts = np.array([[rng.randrange(K + 2) for _ in range(M_)] for _ in range(24)], dtype=np.uint64)
    merged = _merge(M_, K, lists, counts)
    n = -(-K // W)
    chunks = _chunk(K, W, np.array([r for r in merged for _ in range(n)], dtype=np.uint64), list(range(n)) * len(merged))
    slots = _slot(W, N, np.array([chunks[i * n + j // W] for i in range(len(merged)) for j in range(K)], dtype=np.uint64),
                  [j % W for _ in merged for j in range(K)])
    for i, row in enumerate(merged):
        assert chunks[i * n:(i + 1) * n] == [R.chunk(row, c, W) for c in range(n)]
        assert slots[i * K:(i + 1) * K] == [x if x < N else 0 for x in row]
    assert any(x >= N for row in merged for x in row) and all(s < N for s in slots)


@pytest.mark.parametrize("K,W,N", [(4, 2, 20), (5, 2, 12), (8, 16, 40), (1, 1, 4), (SERVED_TOPK, R.CHUNK, SERVED_V)])
def test_an_index_out_of_range_gives_zeros(K, W, N) -> None:
    """A chunk at c of ceil(K/W) or more (the first past it, u32's top bit, its top, and a random one) is all 0 on any slots,
    and a slot at j of its K or more is 0 on slots all below N."""
    rng = random.Random(K * 100 + W * 10 + N)
    n = -(-K // W)
    cs = [n, n + 1, 1 << 31, (1 << 32) - 1, rng.randrange(n, 1 << 32)]
    rows = [[rng.getrandbits(32) for _ in range(K)] for _ in range(4)]
    got = _chunk(K, W, np.array([r for r in rows for _ in cs], dtype=np.uint64), cs * len(rows))
    assert got == [[0] * W] * len(got)
    js = [W, W + 1, 1 << 31, (1 << 32) - 1, rng.randrange(W, 1 << 32)]
    rows = [[rng.randrange(1, N) for _ in range(W)] for _ in range(4)]
    assert _slot(W, N, np.array([r for r in rows for _ in js], dtype=np.uint64), js * len(rows)) == [0] * len(js) * len(rows)


def test_the_served_chunk_and_slot_on_random_rows() -> None:
    """At the served bindings (2,048 slots in chunks of 32, N 2^17, the longest row's class and the table's capacity): the
    chunk's 6-level tree and its range at a power of two, and the slot's 5-level tree and bound at a power of two, on random
    rows, a tenth of their slots and a fifth of the chunks' past N."""
    rng = random.Random("dsa-chunk-served")
    K, W, N = SERVED_TOPK, R.CHUNK, SERVED_V
    rows = [[rng.randrange(N) if rng.random() < 0.9 else rng.getrandbits(32) for _ in range(K)] for _ in range(16)]
    cs = [0, K // W - 1, K // W, 1 << 31] + [rng.randrange(K // W) for _ in range(12)]
    assert _chunk(K, W, np.array(rows, dtype=np.uint64), cs) == [R.chunk(r, c, W) for r, c in zip(rows, cs)]
    rows = [[rng.randrange(N) if rng.random() < 0.8 else rng.getrandbits(32) for _ in range(W)] for _ in range(64)]
    js = [0, W - 1, W, 1 << 31] + [rng.randrange(W) for _ in range(60)]
    got = _slot(W, N, np.array(rows, dtype=np.uint64), js)
    assert got == [R.slot(r, j, N) for r, j in zip(rows, js)] and all(g < N for g in got)


@pytest.mark.parametrize("N,K,block", [(20, 4, 8), (18, 4, 8), (24, 4, 8), (10, 1, 4), (16, 16, 16)])
def test_the_index_units_compose_to_the_rows_networks_order(N, K, block) -> None:
    """Blocks of 8 keys over 20 (a last block of 4), 18 (its last 2 keys joined to the block before) and 24, TOPK 1, and one
    block; each unit on its block of the masks and the outputs before it, each its twin, and slot unit j, at j mod CHUNK of
    chunk unit j // CHUNK, the merge's j-th."""
    masks = _masks(random.Random(N * 10 + K + block), N, K, 64, most=K)
    bl = X.blocks(N, K, block)
    lists, counts = [], []
    for lo, hi in bl:
        p, c = _positions(hi - lo, lo, K, [m[lo:hi] for m in masks])
        lists.append(p)
        counts.append(c)
    got = _merge(len(bl), K, np.array(lists, dtype=np.uint64).transpose(1, 0, 2), np.array(counts, dtype=np.uint64).T)
    W, n = R.CHUNK, -(-K // R.CHUNK)
    chunks = _chunk(K, W, np.array([g for g in got for _ in range(n)], dtype=np.uint64), list(range(n)) * len(got))
    slots = _slot(W, N, np.array([chunks[i * n + j // W] for i in range(len(got)) for j in range(K)], dtype=np.uint64),
                  [j % W for _ in got for j in range(K)])
    for i, (m, g) in enumerate(zip(masks, got)):
        twin = R.values(m, K, block)
        assert [(tuple(lists[b][i]), counts[b][i]) for b in range(len(bl))] == [twin[("positions", b)] for b in range(len(bl))]
        assert tuple(g) == twin[("merge",)][0]
        assert [tuple(x) for x in chunks[i * n:(i + 1) * n]] == [twin[("chunk", c)][0] for c in range(n)]
        assert slots[i * K:(i + 1) * K] == g == [twin[("slot", j)][0] for j in range(K)]
        c = sum(m)
        whole = M.butterfly_apply(M.butterfly_settings(m, K), list(range(N)), K, 0)
        assert g[:c] == whole[:c] == [j for j, b in enumerate(m) if b]
        assert g[c:] == [0] * (K - c)
        if c == K:
            assert g == whole


# --- the long row ------------------------------------------------------------------------------------------------------------

def _table(rng: random.Random, V: int, history: list[list[int]], H: int):
    """A table of V rows of H words: the history's rows at positions 0.., under their salts, and the zero row under the zero
    salt past it, under a registered value's dense domain."""
    dom = CommitmentDomain(rng.randbytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    rows = list(history) + [[0] * H] * (V - len(history))
    salts = [rng.randbytes(EM.SALT_BYTES) for _ in history] + [bytes(EM.SALT_BYTES)] * (V - len(history))
    return dom, rows, salts, RT.commit(dom, [bit_row((r, 16)) for r in rows], salts)


def _path(table: RT.RowTable, rank: int) -> list[bytes]:
    p = table.path(rank)
    return [p[k:k + 64] for k in range(0, len(p), 64)]


def _scores(rng: random.Random, Nb: int, N: int) -> list[list[int]]:
    """Rows of Nb logits with ties at the boundary and specials, and PAD_SCORE past Nb."""
    pool = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0x7F800000, 0xFF800000]
    rows = [[rng.getrandbits(32) for _ in range(Nb)], [rng.choice(pool[:3]) for _ in range(Nb)],
            [rng.choice(pool) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(Nb)], [pool[0]] * Nb]
    return [r + [PAD_SCORE] * (N - Nb) for r in rows]


@pytest.mark.slow
def test_the_long_row_reads_todays_kv_rows() -> None:
    """The one test of the long row whole: the units' tests hold each part, this one their wiring into the reads and the
    reads' openings against a committed table, which no cheaper test builds.  A history of Nb = 18 rows read at its class
    N = 20 (TOPK 4, the mask whole and so one block of 20 keys, BLOCK 8 aside: `test_the_index_units_compose_to_the_rows_
    networks_order` composes several), each row of scores at several n past TOPK up to Nb, from a table of
    V = 32 rows of H = 2 words, the openings at the mask's members by position: kv is `DsaButterflyRowsPad_v1`'s over the
    history and its zero pad at `DsaButterflySettings_v1` of `DsaTopKMaskCounted_v1`, ok all 1; a flipped row bit, another
    member's salt and another position's path each give ok 0 at their slot alone, whose kv is the row it was opened with."""
    rng = random.Random("dsa-topk-reads")
    TOPK, block, V, H, Nb = 4, 8, 32, 2, 18
    N = -(-Nb // TOPK) * TOPK
    READ = EM.read(V, H)
    history = [[rng.randrange(1 << 16) for _ in range(H)] for _ in range(Nb)]
    dom, rows, salts, table = _table(rng, V, history, H)
    cases = [(s, n) for s in _scores(rng, Nb, N) for n in (TOPK + 1, 9, 13, Nb)]
    scores, ns = [s for s, _ in cases], [n for _, n in cases]
    mask = np.asarray(E.evaluate_batch(bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), scores, ns)).reshape(len(cases), -1)
    sw = np.asarray(E.evaluate_batch(bind(M.DsaButterflySettings, N=N, K=TOPK), mask)).reshape(len(cases), -1)
    hist = np.asarray([history] * len(cases), dtype=np.uint64)
    pad = np.zeros((len(cases), N - Nb, H), dtype=np.uint64)
    today = np.asarray(E.evaluate_batch(bind(M.DsaButterflyRowsPad, T=Nb, N=N, K=TOPK, W=H, E=16), sw, hist, pad))
    today = today.reshape(len(cases), TOPK, H).tolist()

    args, bad, want = [], [], []
    for i, m in enumerate(mask.tolist()):
        members = [j for j, b in enumerate(m) if b]
        assert len(members) == TOPK
        opened = [EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, t)) for t in members]
        wrong = rng.randrange(TOPK)
        t = members[wrong]
        given = list(rows[t])
        if i % 3 == 0:
            given[0] ^= 1 << rng.randrange(16)
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, given, salts[t], _path(table, t))
        elif i % 3 == 1:
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[(t + 1) % Nb], _path(table, t))
        else:
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, (t + 1) % V))
        honest = [EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, t)) for t in members]
        for o, w in ((honest, None), (opened, wrong)):
            args.append([_bits(scores[i], 32).reshape(-1), _bits(ns[i], 32).reshape(-1), *map(np.asarray, o[0][1:3]),
                         *(np.concatenate([np.asarray(x[k]) for x in o]) for k in (3, 4, 5))])
            bad.append(w)
            want.append([given if s == w else today[i][s] for s in range(TOPK)])
    t = bind(R.DsaTopKReadRows, N=N, TOPK=TOPK, BLOCK=block, READ=READ)
    cols = [np.stack([a[k] for a in args]).astype(np.uint8) for k in range(len(args[0]))]
    got = np.asarray(BS.evaluate_bits(t, *cols)).reshape(len(args), -1)
    w = 16 * H
    for k, (g, b) in enumerate(zip(got, bad)):
        kv = [_words(g[s * w:(s + 1) * w], 16) for s in range(TOPK)]
        ok = [int(x) for x in g[TOPK * w:]]
        assert kv == want[k], k
        assert ok == [int(s != b) for s in range(TOPK)], (k, ok)


def test_the_cut_counts_its_units() -> None:
    READ = EM.read(SERVED_V, SERVED_H)
    for N in (4096, X.WHOLE_MAX, X.WHOLE_MAX + 2048, 65536, SERVED_V):
        t = bind(R.DsaTopKReadRows, N=N, TOPK=SERVED_TOPK, BLOCK=R.BLOCK, READ=READ)
        cut = dict(R.cut(t))
        bl = X.blocks(N, SERVED_TOPK, R.index_block(N, SERVED_TOPK, R.BLOCK))
        whole = bind(X.DsaTopKMaskWhole, N=N, TOPK=SERVED_TOPK)
        masks = len(X.plan(N, SERVED_TOPK)) if N > X.WHOLE_MAX else 1
        assert (whole in cut) == (N <= X.WHOLE_MAX) and bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=SERVED_TOPK) not in cut
        assert bl == ([(0, N)] if N <= X.WHOLE_MAX else X.blocks(N, SERVED_TOPK, R.BLOCK))
        assert cut[READ] == SERVED_TOPK and cut[bind(R.DsaPositionsMerge, M=len(bl), K=SERVED_TOPK)] == 1
        assert cut[bind(R.DsaPositionsChunk, K=SERVED_TOPK, W=R.CHUNK)] == SERVED_TOPK // R.CHUNK == 64
        assert cut[bind(R.DsaPositionsSlot, K=R.CHUNK, N=N)] == SERVED_TOPK
        assert {(d.bindings["N"], d.bindings["AT"]) for d in cut if d.base_id == "DsaMaskPositions_v1"} == \
               {(hi - lo, lo) for lo, hi in bl}
        assert sum(cut.values()) == masks + len(bl) + 1 + 64 + 2 * SERVED_TOPK
    assert R.CUTS == {"DsaTopKReadRows_v1": R.cut}


#: the row classes the long row serves: N a multiple of TOPK from 4,096 to the table's capacity
SERVED_CLASSES = range(4096, SERVED_V + 1, SERVED_TOPK)
#: the units the long row's own Definitions add to its cut: the index's past the mask, and the read
INDEX_UNITS = ("DsaPositionsMerge_v1", "DsaPositionsChunk_v1", "DsaPositionsSlot_v1", "EmbeddingRead_v1")


def _served_units() -> dict[str, list]:
    """Each unit of the long row's cut at every served class, once, by its template's id."""
    READ = EM.read(SERVED_V, SERVED_H)
    units: dict[str, dict] = {}
    for N in SERVED_CLASSES:
        for u, _ in R.cut(bind(R.DsaTopKReadRows, N=N, TOPK=SERVED_TOPK, BLOCK=R.BLOCK, READ=READ)):
            units.setdefault(u.base_id, {})[u] = None
    return {b: list(us) for b, us in units.items()}


def test_every_served_unit_has_the_rows_flocks_composer_proves() -> None:
    """Every unit of the cut at every served class, its rows read off its ports' bits as the composer reads them (a
    parameter a row, then a return field a row, each whole u16 words): at most `interface.MAX_ROWS` rows, one of them past
    `interface.LAST_BLOCK_BYTES` bytes so it compresses before its last SHA-512 block, and rows that alone fit a Flock
    block of 2^K_MAX bits.  circuit-check's block check doesn't read the second: a slot over a chunk of 16 passed it."""
    units = _served_units()
    assert set(INDEX_UNITS) <= set(units)
    for us in units.values():
        for u in us:
            params, ret = u.signature()
            fields = getattr(ret, "fields", None) or (ret,)
            rows = [2 * -(-sum(t.leaf_widths()) // 16) for t in [t for _, t in params] +
                    [f[1] if isinstance(f, tuple) else f for f in fields]]
            assert len(rows) <= IF.MAX_ROWS and max(rows) > IF.LAST_BLOCK_BYTES, (u.id, rows)
            assert IF.rows_k_log(rows) <= IF.K_MAX, (u.id, rows)


def test_the_index_and_read_units_pass_the_window_builders_proof_block_check() -> None:
    """`interface.check_proof_block`, which vllm's window builder runs on every unit it settles, passes the merge, the
    chunks, the slots and the read at every served class, at the least k_log their rows allow: 2^23 for the chunk and the
    read, 2^21 for the slot (its chunk's 128 bytes the one row past 111), at most 2^26 for the merge of up to 8 blocks."""
    units = _served_units()
    k = {b: {IF.check_proof_block(u) for u in units[b]} for b in INDEX_UNITS}
    assert k["DsaPositionsChunk_v1"] == k["EmbeddingRead_v1"] == {23} and k["DsaPositionsSlot_v1"] == {21}
    assert max(k["DsaPositionsMerge_v1"]) <= 26


#: the mask's units and the block positions, each with the least k_log their rows allow over the served classes: a block's
#: scores, 64 KiB at 16,384 keys, are most of a unit's rows, and the whole mask's are up to WHOLE_MAX keys, 2^27 from 24,576;
#: positions over the whole row read up to 5 KiB of mask words
MASK_UNITS = {"DsaTopKMaskWhole_v1": {24, 25, 26, 27}, "DsaTopKDigitCount_v1": {23, 24, 25, 26}, "DsaTopKDigitPick_v1": {23, 24},
              "DsaTopKMaskBlock_v1": {23, 24, 25, 26}, "DsaMaskPositions_v1": {23, 24}}


def test_the_mask_units_pass_the_window_builders_proof_block_check() -> None:
    """`interface.check_proof_block` passes the mask's units and the block positions at every served class: each mask a
    `dsa_topk_mask_cut.words` port, and each prefix and r whole u16 words, so `interface.layout` lays every port out."""
    units = _served_units()
    assert set(units) == set(INDEX_UNITS) | set(MASK_UNITS)
    k = {b: {IF.check_proof_block(u) for u in units[b]} for b in MASK_UNITS}
    assert k == MASK_UNITS, k


def test_the_long_row_is_typed_by_its_read_and_malformed_bindings_are_refused() -> None:
    READ = EM.read(SERVED_V, SERVED_H)
    params, ret = bind(R.DsaTopKReadRows, N=SERVED_V, TOPK=SERVED_TOPK, BLOCK=R.BLOCK, READ=READ).signature()
    rp, _ = READ.signature()
    assert [n for n, _ in params] == ["scores", "n", *R.READ_PORTS[1:]]
    assert [t.leaves for _, t in params] == [32 * SERVED_V, 32, rp[1][1].leaves, rp[2][1].leaves,
                                             *(SERVED_TOPK * t.leaves for _, t in rp[3:])]
    assert ret.leaves == SERVED_TOPK * (16 * SERVED_H + 1)
    with pytest.raises(ValueError, match="bind V >= N"):
        bind(R.DsaTopKReadRows, N=8, TOPK=4, BLOCK=8, READ=EM.read(4, 2)).signature()
    with pytest.raises(ValueError, match="no read of u16 rows"):
        bind(R.DsaTopKReadRows, N=8, TOPK=4, BLOCK=8, READ=bind(R.DsaMaskPositions, N=8, AT=0, K=4)).signature()
    for bad in (dict(N=8, TOPK=4, BLOCK=2), dict(N=3, TOPK=4, BLOCK=8)):
        with pytest.raises(ValueError):
            bind(R.DsaTopKReadRows, READ=EM.read(8, 2), **bad)
    for bad in (dict(N=4, AT=0, K=5), dict(N=4, AT=-1, K=2), dict(N=1, AT=0, K=1)):
        with pytest.raises(ValueError):
            bind(R.DsaMaskPositions, **bad)
    with pytest.raises(ValueError):
        bind(R.DsaPositionsMerge, M=0, K=4)
