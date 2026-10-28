"""Option A's partials (`served_h1/partials.py`): a drawn leaf's 32 lane partials are `h1tile.reference` split by lane
(each lane's `lane_sets` absorbed from its fold keys, its chains' y and z XORs after each set, lanes in order), their
XOR over lanes is the leaf's receipt as `receipts.call_receipts` and `h1tile.reference` give it and as serving commits
it under the step's live rows (a dead row block's partials and receipt zero), the batch is the leaves one by one in
their order across steps, calls and K, and the partials move with a byte of the leaf's A or B row blocks, the step's
coin and its position, not with a byte of another row block.  The gpu backend's seed records (`kh_seeds_at`'s data) hash
to `seeds.seed` and `receipt_seed`; the backend itself (h1u.cu's kh_seeds_at and kh_lanes) is held to the cpu's byte for
byte where PARTIALS_LIB names an h1u build and CUDA is up (job.sh's partials bench, on every h1u build it ships)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG

HERE = Path(__file__).resolve().parents[1] / "served_h1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"served_h1_{name}", HERE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PT = _load("partials")
RC = PT.RC
COM = types.SimpleNamespace(root=bytes(range(7, 71)), epoch=3)
COIN, ROOT = bytes(range(100, 164)), bytes(range(30, 94))
T = 64
# (N, K) a call: K = 64 and 128 keep h1tile.reference quick
NK = [(64, 64), (96, 128)]


def _s8(shape, seed: int) -> torch.Tensor:
    return torch.randint(-128, 128, shape, generator=torch.Generator().manual_seed(seed), dtype=torch.int8)


def _window(steps: dict[int, int]) -> "PT.Window":
    """Two calls; steps {s: m}, each its own A and coin index s."""
    units = [(T // 32) * (n // 32) for n, _ in NK]
    offs = [sum(units[:w]) for w in range(len(NK))]
    win = PT.Window([_s8((n, k), 10 + w) for w, (n, k) in enumerate(NK)], [bytes([w + 1]) * 64 for w in range(len(NK))],
                    offs, sum(units))
    for s, m in steps.items():
        a = [_s8((T, k), 100 * s + w) for w, (_, k) in enumerate(NK)]
        win.steps[s] = PT.Step.from_coin(a, [bytes([s, w]) * 32 for w in range(len(NK))], m, COM, COIN, s, ROOT)
    return win


def _tile(win, leaf):
    st, b = win.steps[leaf.step], win.b[leaf.call]
    ri, cj = divmod(leaf.unit, b.shape[0] // 32)
    a = (st.a[leaf.call][32 * ri:32 * ri + 32].to(torch.int64) & 0xFF).tolist()
    bb = (b[32 * cj:32 * cj + 32].to(torch.int64) & 0xFF).tolist()
    pos = leaf.step * win.u_step + win.offs[leaf.call] + leaf.unit
    return a, bb, st.salt, pos, (st.a_roots[leaf.call], win.b_roots[leaf.call])


def _by_lane(a, b, salt, pos, roots) -> np.ndarray:
    """h1tile.reference's chains lane by lane: lane ln's [y XOR, z XOR] after each set."""
    lay = HT.TileShape(32, 32, len(a[0]))
    rows, cols = HT.noise(SD.seed(salt, pos, *roots), lay)
    keys = HT.fold_keys(SD.receipt_seed(salt, pos, *roots), lay)
    out = np.zeros((lay.lanes, 2 * lay.sets), dtype=np.uint32)
    for ln in range(lay.lanes):
        st = list(keys[ln])
        for s, p in enumerate(HT.lane_sets(a, b, rows, cols, lay, ln)):
            st = SG.absorb(HT.FOLD, st, p)
            for h in range(2):
                x = 0
                for v in st[8 * h:8 * h + 8]:
                    x ^= v
                out[ln, 2 * s + h] = x & HT.M32
    return out


def _receipts(win, s: int, w: int) -> np.ndarray:
    """Serving's receipts of call w at step s: `gated_receipts` under the step's live rows (dead row blocks zero)."""
    st = win.steps[s]
    got, _ = RC.gated_receipts(RC.Call(st.a[w], win.b[w]), np.arange(T) < st.m, st.salt, s * win.u_step + win.offs[w],
                               st.a_roots[w], win.b_roots[w])
    return got


@pytest.mark.parametrize("leaf", [PT.Leaf(3, 0, 1), PT.Leaf(3, 1, 4), PT.Leaf(4, 1, 2)])
def test_partials_are_the_reference_by_lane(leaf):
    win = _window({3: 40, 4: 20})
    p = PT.lane_partials(win, leaf)
    W = 2 * HT.TileShape(32, 32, win.k(leaf.call)).sets
    assert p.shape == (32, W) and p.dtype == np.uint32
    tile = _tile(win, leaf)
    assert np.array_equal(p, _by_lane(*tile))
    _, rc = HT.reference(*tile[:2], *tile[2:4], tile[4])
    assert PT.combine(p).tolist() == rc
    assert np.array_equal(PT.combine(p), _receipts(win, leaf.step, leaf.call)[leaf.unit])


def test_every_unit_xors_to_the_served_receipt_and_dead_are_zero():
    win = _window({3: 40, 4: 20})
    for s in (3, 4):
        for w, (n, _) in enumerate(NK):
            leaves = [PT.Leaf(s, w, u) for u in range((T // 32) * (n // 32))]
            got = np.stack(PT.lane_partials_many(win, leaves))
            assert np.array_equal(PT.combine(got), _receipts(win, s, w))
            dead = [u for u in range(len(leaves)) if 32 * (u // (n // 32)) >= win.steps[s].m]
            assert (s == 4) == bool(dead)
            assert not got[dead].any()


def test_many_is_each_in_order():
    win = _window({3: 40, 4: 20})
    leaves = [PT.Leaf(4, 1, 5), PT.Leaf(3, 0, 2), PT.Leaf(3, 1, 0), PT.Leaf(4, 0, 1), PT.Leaf(3, 1, 5), PT.Leaf(4, 1, 0)]
    many = PT.lane_partials_many(win, leaves, chunk=2)
    for leaf, p in zip(leaves, many):
        assert np.array_equal(p, PT.lane_partials(win, leaf))
    assert not many[0].any() and many[3].any()


def test_partials_move_with_the_leafs_operands_coin_and_position():
    win = _window({3: 40})
    leaf = PT.Leaf(3, 1, 4)  # row block 1, column block 1 of N = 96
    p0 = PT.lane_partials(win, leaf)
    st, a = win.steps[3], win.steps[3].a[1]
    committed = PT.combine(p0)

    def moved(w2) -> np.ndarray:
        p = PT.lane_partials(w2, leaf)
        assert not np.array_equal(p, p0)
        return p

    # a byte of an A row in the leaf's row block: the partials and their XOR move off the committed receipt
    at = a.clone()
    at[32 + 5, 17] ^= 1
    win.steps[3] = PT.Step(st.a[:1] + [at], st.a_roots, st.m, st.salt)
    assert not np.array_equal(PT.combine(moved(win)), committed)
    # in the other row block: nothing moves
    at = a.clone()
    at[5, 17] ^= 1
    win.steps[3] = PT.Step(st.a[:1] + [at], st.a_roots, st.m, st.salt)
    assert np.array_equal(PT.lane_partials(win, leaf), p0)
    win.steps[3] = st
    # a byte of the leaf's B rows
    b = win.b[1]
    bt = b.clone()
    bt[32 + 3, 100] ^= 0x80
    win.b[1] = bt
    moved(win)
    win.b[1] = b
    # another coin, or the coin of another round index, gives another salt
    for coin, ix in ((bytes([COIN[0] ^ 1]) + COIN[1:], 3), (COIN, 2)):
        win.steps[3] = PT.Step.from_coin(st.a, st.a_roots, st.m, COM, coin, ix, ROOT)
        assert not np.array_equal(PT.combine(moved(win)), committed)
    win.steps[3] = st
    # the same operands at another step's position
    win.steps[5] = st
    assert not np.array_equal(PT.lane_partials(win, PT.Leaf(5, 1, 4)), p0)


def test_refuses_what_is_no_leaf():
    win = _window({3: 40})
    with pytest.raises(IndexError):
        PT.lane_partials(win, PT.Leaf(3, 0, 8))
    with pytest.raises(ValueError):
        PT.lane_partials(win, PT.Leaf(3, 0, 0), backend="gpu")


def test_seed_records_hash_to_the_seeds():
    """`_seed_inputs`' records read as h1_seeds_at reads them (the message's words: the tag's, then salt and roots, each
    XOR the position) give each leaf's seed_j and σ_R, one record a step and call, at positions past 2^32."""
    win = _window({3: 40, 4: 20})
    win.u_step += 1 << 40
    ats = [PT._at(win, x) for x in (PT.Leaf(4, 1, 5), PT.Leaf(3, 0, 2), PT.Leaf(3, 1, 0), PT.Leaf(4, 1, 1))]
    data, idx, pos = PT._seed_inputs(win, ats, "cpu")
    assert data.shape == (3, 30) and idx.tolist() == [0, 1, 2, 0]
    m64 = (1 << 64) - 1
    for x, i, p in zip(ats, idx.tolist(), pos.tolist()):
        d, p = [v & m64 for v in data[i].tolist()], p & m64
        assert p == x.position
        msg = [p.to_bytes(8, "big") + b"".join((w ^ p).to_bytes(8, "big") for w in d[3 * t:3 * t + 3] + d[6:])
               for t in range(2)]
        got = [hashlib.sha512(x).digest() for x in msg]
        st = win.steps[x.leaf.step]
        roots = (st.a_roots[x.leaf.call], win.b_roots[x.leaf.call])
        assert got == [SD.seed(st.salt, p, *roots), SD.receipt_seed(st.salt, p, *roots)]


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("PARTIALS_LIB")),
                    reason="the gpu backend needs CUDA and PARTIALS_LIB (an h1u.cu build with kh_lanes)")
def test_gpu_is_the_cpu_byte_for_byte():
    kern = PT.Kernel(os.environ["PARTIALS_LIB"])
    dev = torch.device("cuda")
    nk = [(96, 512), (64, 1024), (32, 4608)]  # m = 2, 2, 3: both of kh_lanes' instantiations
    units = [(T // 32) * (n // 32) for n, _ in nk]
    win = PT.Window([_s8((n, k), 20 + w).to(dev) for w, (n, k) in enumerate(nk)],
                    [bytes([w + 1]) * 64 for w in range(len(nk))], [sum(units[:w]) for w in range(len(nk))],
                    (1 << 40) + sum(units))
    for s, m in ((3, 40), (4, 20)):
        a = [_s8((T, k), 7 * s + w).to(dev) for w, (_, k) in enumerate(nk)]
        win.steps[s] = PT.Step.from_coin(a, [bytes([s, w]) * 32 for w in range(len(nk))], m, COM, COIN, s, ROOT)
    leaves = [PT.Leaf(s, w, u) for s in (3, 4) for w in range(len(nk)) for u in range(units[w])]
    ats = [PT._at(win, x) for x in leaves]
    sj, sr = PT._seeds(win, ats)
    assert torch.equal(kern.seeds(*PT._seed_inputs(win, ats, dev)).cpu(),
                       torch.cat([RC.seed_words(sj, "cpu"), RC.seed_words(sr, "cpu")], 1))
    cpu = PT.lane_partials_many(win, leaves)
    gpu = PT.lane_partials_many(win, leaves, backend="gpu", kernel=kern, chunk=3)
    for c, g in zip(cpu, gpu):
        assert c.tobytes() == g.tobytes()
    assert any(c.any() for c in cpu) and not all(c.any() for c in cpu)
    for sel, out, _ in PT.gpu_batches(win, leaves, kern):
        for j, i in enumerate(sel):
            assert np.array_equal(out[j].cpu().numpy().view(np.uint32), PT.combine(cpu[i]))
