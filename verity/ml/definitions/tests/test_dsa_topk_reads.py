"""`registry.dsa_topk_reads`, DSA's long row by table reads, held to the rows network's order and to today's long row:

* a block's `DsaMaskPositions_v1` is its twin (`values`: `dsa_topk_mask.butterfly_apply` at the block's settings over its
  positions) on masks of no member to every key, and its first count slots are the members' positions, ascending, whenever it
  has at most K; `DsaPositionsMerge_v1` is `merge` on lists with words past their counts, counts past K and lists that fill K;
* the units composed over a row's blocks (each on the outputs before it) give the rows network's order over the whole row
  (`DsaTopKRow_v2`'s construction), then 0, on masks of at most K members;
* `DsaTopKReadRows_v1` over a table that holds a history gives today's long row's kv (`DsaButterflyRowsPad_v1` at the counted
  mask's settings over the history and a zero pad) with every ok 1, at small statics over three blocks, and ok 0 at a slot whose
  opening is not the table's;
* `cut` counts its units at the served statics, and malformed bindings are refused.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.gates import embedding as EM
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import bit_row
from verity.ml.definitions.registry import boolean_dsa_topk_mask as BM
from verity.ml.definitions.registry import dsa_topk_mask as M
from verity.ml.definitions.registry import dsa_topk_mask_cut as X
from verity.ml.definitions.registry import dsa_topk_reads as R

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


def _positions(N: int, AT: int, K: int, masks: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    got = np.asarray(BS.evaluate_bits(bind(R.DsaMaskPositions, N=N, AT=AT, K=K), np.asarray(masks, dtype=np.uint8)))
    got = got.reshape(len(masks), -1)
    return [_words(g[:32 * K], 32) for g in got], [_words(g[32 * K:], 32)[0] for g in got]


def _merge(M_: int, K: int, lists: np.ndarray, counts: np.ndarray) -> list[list[int]]:
    """``lists`` (rows, M_, K) and ``counts`` (rows, M_) u32 words through the circuit."""
    rows = len(lists)
    got = BS.evaluate_bits(bind(R.DsaPositionsMerge, M=M_, K=K), _bits(lists, 32).reshape(rows, -1),
                           _bits(counts, 32).reshape(rows, -1))
    return [_words(g, 32) for g in np.asarray(got).reshape(rows, -1)]


@pytest.mark.parametrize("N,AT,K", [(8, 0, 4), (6, 8, 4), (5, 3, 1), (16, 16, 16), (12, 1 << 17, 4), (2, 0, 2)])
def test_a_blocks_positions_are_its_twin_and_its_members_in_order(N, AT, K) -> None:
    masks = _masks(random.Random(N * 1000 + AT + K), N, K, 40)
    pos, counts = _positions(N, AT, K, masks)
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


@pytest.mark.parametrize("N,K,block", [(20, 4, 8), (18, 4, 8), (24, 4, 8), (10, 1, 4), (16, 16, 16)])
def test_the_index_units_compose_to_the_rows_networks_order(N, K, block) -> None:
    """Blocks of 8 keys over 20 (a last block of 4), 18 (its last 2 keys joined to the block before) and 24, TOPK 1, and one
    block; each unit on its block of the masks and the outputs before it, each its twin."""
    masks = _masks(random.Random(N * 10 + K + block), N, K, 64, most=K)
    bl = X.blocks(N, K, block)
    lists, counts = [], []
    for lo, hi in bl:
        p, c = _positions(hi - lo, lo, K, [m[lo:hi] for m in masks])
        lists.append(p)
        counts.append(c)
    got = _merge(len(bl), K, np.array(lists, dtype=np.uint64).transpose(1, 0, 2), np.array(counts, dtype=np.uint64).T)
    for i, (m, g) in enumerate(zip(masks, got)):
        twin = R.values(m, K, block)
        assert [(tuple(lists[b][i]), counts[b][i]) for b in range(len(bl))] == [twin[("positions", b)] for b in range(len(bl))]
        assert tuple(g) == twin[("merge",)][0]
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
    N = 20 (TOPK 4, blocks of 8 keys: three blocks), each row of scores at several n past TOPK up to Nb, from a table of
    V = 32 rows of H = 2 words, the openings at the mask's members by position: kv is `DsaButterflyRowsPad_v1`'s over the
    history and its zero pad at `DsaButterflySettings_v1` of `DsaTopKMaskCounted_v1`, ok all 1; a flipped row bit, another
    member's salt and another position's path each give ok 0 at their slot alone."""
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

    args, bad = [], []
    for i, m in enumerate(mask.tolist()):
        members = [j for j, b in enumerate(m) if b]
        assert len(members) == TOPK
        opened = [EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, t)) for t in members]
        wrong = rng.randrange(TOPK)
        t = members[wrong]
        if i % 3 == 0:
            r = list(rows[t])
            r[0] ^= 1 << rng.randrange(16)
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, r, salts[t], _path(table, t))
        elif i % 3 == 1:
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[(t + 1) % Nb], _path(table, t))
        else:
            opened[wrong] = EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, (t + 1) % V))
        honest = [EM.arguments(V, t, table.root, dom.domain_id, rows[t], salts[t], _path(table, t)) for t in members]
        for o, w in ((honest, None), (opened, wrong)):
            args.append([_bits(scores[i], 32).reshape(-1), _bits(ns[i], 32).reshape(-1), *map(np.asarray, o[0][1:3]),
                         *(np.concatenate([np.asarray(x[k]) for x in o]) for k in (3, 4, 5))])
            bad.append(w)
    t = bind(R.DsaTopKReadRows, N=N, TOPK=TOPK, BLOCK=block, READ=READ)
    cols = [np.stack([a[k] for a in args]).astype(np.uint8) for k in range(len(args[0]))]
    got = np.asarray(BS.evaluate_bits(t, *cols)).reshape(len(args), -1)
    w = 16 * H
    for k, (g, b) in enumerate(zip(got, bad)):
        kv = [_words(g[s * w:(s + 1) * w], 16) for s in range(TOPK)]
        ok = [int(x) for x in g[TOPK * w:]]
        assert kv == today[k // 2], k
        assert ok == [int(s != b) for s in range(TOPK)], (k, ok)


def test_the_cut_counts_its_units() -> None:
    READ = EM.read(SERVED_V, SERVED_H)
    for N in (4096, X.WHOLE_MAX, X.WHOLE_MAX + 2048, 65536, SERVED_V):
        t = bind(R.DsaTopKReadRows, N=N, TOPK=SERVED_TOPK, BLOCK=R.BLOCK, READ=READ)
        cut = dict(R.cut(t))
        bl = X.blocks(N, SERVED_TOPK, R.BLOCK)
        mask = bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=SERVED_TOPK)
        masks = len(X.plan(N, SERVED_TOPK)) if N > X.WHOLE_MAX else 1
        assert (mask in cut) == (N <= X.WHOLE_MAX)
        assert cut[READ] == SERVED_TOPK and cut[bind(R.DsaPositionsMerge, M=len(bl), K=SERVED_TOPK)] == 1
        assert {(d.bindings["N"], d.bindings["AT"]) for d in cut if d.base_id == "DsaMaskPositions_v1"} == \
               {(hi - lo, lo) for lo, hi in bl}
        assert sum(cut.values()) == masks + len(bl) + 1 + SERVED_TOPK
    assert R.CUTS == {"DsaTopKReadRows_v1": R.cut}


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
