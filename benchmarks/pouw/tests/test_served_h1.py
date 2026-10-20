"""served_h1's side computation (`served_h1/receipts.py`) is `h1tile.reference`'s receipt, unit by unit: on random s8 calls at
K = 64 and 128 (a partial row block left uncovered), and saturated operands at K = 64; and each receipt reads the unit's
position, salt and roots and every operand byte of its tile.  A call's receipt tree (`served_h1/commit.py`) is the same built in
pool tasks or leaf by leaf, opens a unit's leaf, and refuses a flipped bit or another call's salt label.  Under a call's admit
bits (`served_h1/credit.py`), from the committed request queue, a dead unit's receipt is the canonical empty receipt and a live
unit's the reference's (`PouwH1Tile_v4`), an admitted block of zero-byte rows is computed, receipt-checked and credited (RT4-LOGIC-4, RT4-LOGIC-20), a tile's admit bits open from the admit tree and a flipped one is refused, a junk row in a slot the queue leaves unassigned is refused and credits nothing, so is the canonical
empty row in a slot the queue selects (RT-ALGO-4), and the tally's limbs are 16-bit.  A call's A table is AROW's row table, a tile's rows open against it and the step's A root, and
A' = -b is refused (CMP-38).  A linear's B table is the same format over its weight rows, under the weights' root; the
scheduler's record gives the request queue and its commitment.  A window's live coins (`served_h1/coins.py`) come from a
challenger off the node over its stream (a fake one on loopback, signed with a test key), which refuses a session of
another layout; the verifier checks them from the challenger's own directory alone and refuses a pin of another key, a
round root the challenger didn't receive, a salt that isn't its round's coin's, and a draw after another root; with no
address the coins are in-process and labelled not certified."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import random
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "served_h1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"served_h1_{name}", HERE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC, CM, CR, CO = _load("receipts"), _load("commit"), _load("credit"), _load("coins")
SALT, RA, RB = bytes(range(64)), bytes(range(64, 128)), bytes(range(128, 192))


def _call(m: int, n: int, k: int, seed: int) -> "RC.Call":
    g = torch.Generator().manual_seed(seed)
    return RC.Call(torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8),
                   torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8))


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (33, 96, 64), (32, 64, 128)])
def test_receipts_are_the_reference(m, n, k):
    call = _call(m, n, k, m + n + k)
    got = RC.call_receipts(call, SALT, 1 << 40, RA, RB)
    assert len(call.units) == (m // 32) * (n // 32) == got.shape[0]
    for u in range(len(call.units)):
        assert list(got[u]) == RC.reference(call, SALT, 1 << 40, RA, RB, u)[1]


def test_saturated_operands():
    call = RC.Call(torch.full((32, 64), 127, dtype=torch.int8), torch.full((32, 64), -127, dtype=torch.int8))
    assert list(RC.call_receipts(call, SALT, 5, RA, RB)[0]) == RC.reference(call, SALT, 5, RA, RB, 0)[1]


def test_receipt_reads_everything():
    call = _call(32, 32, 64, 9)
    base = RC.call_receipts(call, SALT, 3, RA, RB)[0]
    assert list(RC.call_receipts(call, SALT, 4, RA, RB)[0]) != list(base)
    assert list(RC.call_receipts(call, RB, 3, RA, RB)[0]) != list(base)
    assert list(RC.call_receipts(call, SALT, 3, RB, RA)[0]) != list(base)
    for side, (i, t) in (("a", (31, 63)), ("b", (0, 0))):
        x = getattr(call, side).clone()
        x[i, t] = x[i, t] + 1 if x[i, t] < 127 else x[i, t] - 1
        moved = RC.Call(x, call.b) if side == "a" else RC.Call(call.a, x)
        assert list(RC.call_receipts(moved, SALT, 3, RA, RB)[0]) != list(base)


def test_receipt_tree_opens_and_refuses():
    receipts = RC.call_receipts(_call(64, 96, 64, 4), SALT, 0, RA, RB)
    rows = CM.receipt_rows(receipts)
    dom = CM.domain({"call": 0, "units": len(rows)}, len(rows))
    key = bytes(64)
    pooled = CM.levels(dom, CM.split(CM.leaves((dom.binding, dom.count, key, "r/0", 0, rows))))
    direct = CM.levels(dom, [CM.leaf(dom, key, "r/0", u, bytes(r))[0] for u, r in enumerate(rows)])
    assert pooled == direct
    root, u = pooled[-1][0], 5
    assert CM.verify(dom, root, u, CM.leaf(dom, key, "r/0", u, bytes(rows[u]))[0], CM.path(pooled, u))
    flipped = bytearray(rows[u])
    flipped[17] ^= 0x10
    assert not CM.verify(dom, root, u, CM.leaf(dom, key, "r/0", u, bytes(flipped))[0], CM.path(pooled, u))
    assert not CM.verify(dom, root, u, CM.leaf(dom, key, "r/1", u, bytes(rows[u]))[0], CM.path(pooled, u))


def _padded_call(m: int, t: int, n: int, k: int, seed: int) -> "RC.Call":
    """A T-slot call whose first m slots are real rows and the rest the canonical empty row."""
    call = _call(t, n, k, seed)
    call.a[m:] = 0
    return call


def test_dead_units_skip_noise_and_receipt():
    """Under its admit bits a call's live units (an admitted slot among their 32 rows) are `h1tile`'s receipts and its dead
    units the canonical empty receipt; the gated reference agrees, and a dead unit's ungated receipt is not empty, so the
    skip is visible."""
    call = _padded_call(5, 64, 64, 64, 11)
    scales = torch.zeros(64)
    scales[:5] = 0.5
    queue = CR.queue_of([[("r0", 3), ("r1", 2)]], 1, 64)
    rows = CM.s8_rows(call.a.numpy(), scales.numpy())
    bits = CR.queue_admit(queue, 0, rows)
    assert list(bits) == list(CR.queue_select(queue, 0)) == [True] * 5 + [False] * 59
    assert not CR.junk(rows, bits) and not CR.hollow(rows, CR.queue_select(queue, 0))
    got, live = RC.gated_receipts(call, bits, SALT, 9, RA, RB)
    assert live == CR.live_units(bits, 64) == [0, 1]
    for u in range(len(call.units)):
        want = RC.gated_reference(call, bits, SALT, 9, RA, RB, u)
        assert list(got[u]) == want
        ungated = RC.reference(call, SALT, 9, RA, RB, u)[1]
        assert (want == ungated) if u in live else (not any(want) and any(ungated))


def test_an_admitted_zero_byte_block_is_computed_and_credited_and_a_non_admitted_block_is_not():
    """RT4-LOGIC-4 and RT4-LOGIC-20 under `PouwH1Tile_v4`: the admit unit reads no row byte, so a selected block of
    zero-byte rows under nonzero scales is admitted; a tile's live bit is the OR of its admit bits, the bits the tally
    reads, so that block is computed and receipt-checked (the ungated unit's receipt) and credited, as is a selected block
    of nonzero bytes; the block the queue leaves unselected is dead: the zero receipt, its noise never derived, and no
    credit."""
    call = _padded_call(96, 96, 64, 64, 12)
    call.a[:32] = 0
    call.a[64:] = 0
    scales = np.zeros(96, dtype=np.float32)
    scales[:64] = 0.5
    queue = CR.queue_of([[("r0", 32), ("r1", 32)]], 1, 96)
    rows = CM.s8_rows(call.a.numpy(), scales)
    bits = CR.queue_admit(queue, 0, rows)
    assert list(bits) == [True] * 64 + [False] * 32 and CR.scale_only(rows) == list(range(32))
    assert not CR.junk(rows, bits) and not CR.hollow(rows, CR.queue_select(queue, 0))
    assert CR.live_blocks(bits) == [0, 1]
    got, live = RC.gated_receipts(call, bits, SALT, 9, RA, RB)
    assert live == CR.live_units(bits, 64) == [0, 1, 2, 3]
    for u in range(len(call.units)):
        want = RC.gated_reference(call, bits, SALT, 9, RA, RB, u)
        ungated = RC.reference(call, SALT, 9, RA, RB, u)[1]
        assert list(got[u]) == want and any(ungated)
        assert want == ungated if u in live else not any(want)
    tl = CR.tally([{"step": 0, "site": "x", "n": 64, "k": 64}], [int(bits.sum())], 1, ("x",))
    assert tl["rows"] == 64 and tl["macs"] == 64 * 64 * 64


def test_a_tiles_admit_input_opens_from_the_admit_tree_and_a_flipped_bit_is_refused():
    """The gate's binding of `PouwH1Tile_v4`'s admit input (`serve.tile_admit`): each row block's two admit words open from
    the call's admit tree against its root as the tally's 32 bits, and a word with any one bit flipped is refused."""
    SV = _load("serve")
    t, key = 128, bytes(range(64))
    bits = np.zeros(t, dtype=bool)
    bits[[0, 5, 40, 77]] = True
    ws = CR.words(bits)
    fields = {"window": 0, "step": 3, "linear": 2, "module": "m", "slots": t, "schema": CM.SCHEMA}
    dom = CM.domain(fields, len(ws), "admit")
    lv = CM.levels(dom, [CM.leaf(dom, key, "s/3/2", q, x)[0] for q, x in enumerate(CR.word_rows(ws))])
    c, d = {"step": 3, "linear": 2, "admit_fields": fields, "root_admit": lv[-1][0].hex()}, {"words": ws, "s_levels": lv}
    for ri in range(t // 32):
        assert list(SV.tile_admit(c, d, key, ri)) == list(bits[32 * ri:32 * ri + 32])
        for q in CR.tile_words(ri):
            for b in range(CR.WORD):
                forged = list(ws)
                forged[q] ^= 1 << b
                assert SV.tile_admit(c, d, key, ri, forged) is None
    assert SV.tile_admit({**c, "linear": 1}, d, key, 0) is None


def test_a_opens_against_the_step_root_and_refuses_neg_b():
    """CMP-38 on PoUW's side: a call's A table is AROW's row table (`row_table.commit` over `s8_row`s gives the root `leaf` and
    `levels` give), a tile's 32 A rows open against it and the step's A root, A' = -b (the tile's B rows negated) is refused
    against the committed root, and re-committing A' moves both roots."""
    from verity.ml.commitments import row_table as RT

    t, k, key = 64, 64, bytes(range(64))
    calls = [_padded_call(5, t, 64, k, 20 + w) for w in range(3)]
    scales = np.zeros(t, dtype=np.float32)
    scales[:5] = 0.25
    doms = [CM.domain({"window": "w", "step": 0, "linear": w, "slots": t, "k": k}, t, "a-table") for w in range(3)]
    tabs = [CM.s8_table(doms[w], key, f"a/0/{w}", calls[w].a.numpy(), scales) for w in range(3)]
    rows = [CM.s8_rows(c.a.numpy(), scales) for c in calls]
    for w in range(3):
        leaves = [CM.leaf(doms[w], key, f"a/0/{w}", r, bytes(x))[0] for r, x in enumerate(rows[w])]
        assert tabs[w].root == CM.levels(doms[w], leaves)[-1][0]
    s_dom = CM.domain({"window": "w", "step": 0, "linears": 3}, 3, "a-step")
    s_lv = CM.levels(s_dom, [tb.root for tb in tabs])
    w, ri, cj = 1, 0, 1
    ranks = range(32 * ri, 32 * ri + 32)
    opening = tabs[w].open(ranks)

    def witness(rs) -> dict:
        return {r: (bytes(rs[r]), 8 * rs.shape[1], CM.salt(key, f"a/0/{w}", r)) for r in ranks}

    assert RT.verify_rows(tabs[w].root, doms[w].domain_id, t, opening, witness(rows[w]))
    assert CM.verify(s_dom, s_lv[-1][0], w, tabs[w].root, CM.path(s_lv, w))
    q = calls[w].a.numpy().copy()
    q[32 * ri:32 * ri + 32] = (-calls[w].b[32 * cj:32 * cj + 32].to(torch.int16)).clamp(-128, 127).to(torch.int8).numpy()
    assert not RT.verify_rows(tabs[w].root, doms[w].domain_id, t, opening, witness(CM.s8_rows(q, scales)))
    forged = CM.s8_table(doms[w], key, f"a/0/{w}", q, scales).root
    assert forged != tabs[w].root
    assert CM.step_root(s_dom, [tabs[0].root, forged, tabs[2].root]) != s_lv[-1][0]


def test_admit_words_and_tally():
    """Admit bits are 16-bit words, slot 16 q + b at bit b of word q; the tally sums each call's admitted rows per
    (step, site) and per site in 16-bit limbs, weighs a site by its one N K, and refuses a limb past 16 bits."""
    bits = np.zeros(64, dtype=bool)
    bits[[0, 3, 17, 63]] = True
    ws = CR.words(bits)
    assert ws == [0b1001, 0b10, 0, 1 << 15] and list(CR.bits_of(ws)) == list(bits)
    assert CR.word_rows(ws)[3] == (1 << 15).to_bytes(2, "little")
    assert CR.live_blocks(bits) == [0, 1] and CR.live_units(bits, 96) == [0, 1, 2, 3, 4, 5]
    rows = np.zeros((3, 8 + 4), dtype=np.uint8)
    rows[0, 2], rows[1, -1], rows[2, [1, -2]] = 1, 0x3F, (5, 0x40)
    assert list(CR.nonempty(rows)) == [True, True, True] and CR.scale_only(rows) == [1]
    calls = [{"step": s, "site": site, "n": n, "k": k} for s in range(2) for site, n, k in (("a", 64, 32), ("b", 32, 64))]
    tl = CR.tally(calls, [5, 1, 3, 0], 2, ("a", "b"))
    assert tl["step_site"] == [[5, 1], [3, 0]] and tl["site"] == {"a": 8, "b": 1}
    assert tl["rows"] == 9 and tl["macs"] == 8 * 64 * 32 + 1 * 32 * 64
    pub = CR.published(tl["macs"])
    assert 0 < pub["pub_macs"] <= tl["macs"]
    with pytest.raises(ValueError, match="16-bit"):
        CR.tally(calls, [1 << 16, 0, 0, 0], 2, ("a", "b"))


def test_junk_row_in_an_unassigned_slot_is_refused_and_credits_nothing():
    """The selection is the queue's (the slot map and the requests' lifetimes), never the row's, and a row can only withhold
    admission (`PouwQueueAdmit`'s word also reads its scale): a nonzero junk row in a slot the queue leaves unassigned is
    not admitted, so the call's admitted count and the tally are unchanged,
    and the admit unit refuses it (`junk`); a request past its last step admits nothing either."""
    t, k = 64, 64
    spans = [[("a", 4), ("b", 3)], [("a", 1), ("b", 1), ("c", 2)], [("c", 1)]]
    queue = CR.queue_of(spans, 4, t)
    assert queue["ids"] == ["a", "b", "c"] and queue["requests"] == [(0, 0), (0, 2), (0, 2), (1, 3)]
    assert [int(CR.queue_select(queue, s).sum()) for s in range(4)] == [7, 4, 1, 0]
    honest = np.zeros((t, k + 4), dtype=np.uint8)
    honest[:4, :k] = np.arange(1, 5, dtype=np.uint8)[:, None]
    honest[:4, -1] = 0x3F
    bits = CR.queue_admit(queue, 1, honest)
    assert list(bits) == list(CR.queue_select(queue, 1)) and not CR.junk(honest, bits)
    forged = honest.copy()
    forged[37, :k] = 0x55
    forged[37, -1] = 0x3F
    assert CR.nonempty(forged)[37] and not bits[37]
    assert CR.junk(forged, bits) == [37]
    assert list(CR.queue_admit(queue, 1, forged)) == list(bits) and int(bits.sum()) == 4
    calls = [{"step": 1, "site": "x", "n": 32, "k": k}]
    assert CR.tally(calls, [int(bits.sum())], 4, ("x",))["rows"] == 4
    stale = {**queue, "slot_map": queue["slot_map"].copy()}
    stale["slot_map"][3, 0] = 1
    assert not CR.queue_select(stale, 3).any()
    slot_leaves, req_leaves = CR.queue_leaves(queue)
    assert len(slot_leaves) == 4 * t // 16 and all(len(x) == 32 for x in slot_leaves)
    assert req_leaves[3] == (1).to_bytes(2, "little") + (3).to_bytes(2, "little")


def test_canonical_empty_rows_in_selected_slots_are_refused_and_credit_nothing():
    """RT-ALGO-4 (note:20261019T0618Z-draft-rt-algo-r2): a block whose 32 slots the queue selects but whose rows are all the
    canonical empty row is dead to the tile, which zeroes its receipt.  The admit unit admits none of its slots, so the
    block is dead to credit too and the tally counts none of its rows, and it refuses every one of them (`hollow`).  A
    selected row of zero bytes under a nonzero scale is admitted (`scale_only`: the admit unit reads no byte), and a
    selected row of nonzero bytes under scale 0 is not admitted and so refused as junk."""
    t, k = 64, 64
    queue = CR.queue_of([[("a", 32)]], 1, t)
    chosen = CR.queue_select(queue, 0)
    assert list(chosen) == [True] * 32 + [False] * 32
    empty = np.zeros((t, k + 4), dtype=np.uint8)
    bits = CR.queue_admit(queue, 0, empty)
    assert not bits.any() and CR.live_blocks(bits) == [] and CR.hollow(empty, chosen) == list(range(32))
    assert not CR.junk(empty, bits) and not CR.scale_only(empty)
    call = RC.Call(torch.zeros(t, k, dtype=torch.int8), _call(32, 32, k, 3).b)
    got, live = RC.gated_receipts(call, bits, SALT, 9, RA, RB)
    assert live == [] and not got.any()
    calls = [{"step": 0, "site": "x", "n": 32, "k": k}]
    assert CR.tally(calls, [int(bits.sum())], 1, ("x",))["rows"] == 0
    scale_only = empty.copy()
    scale_only[:32, -1] = 0x3F
    assert CR.queue_admit(queue, 0, scale_only).sum() == 32 and CR.scale_only(scale_only) == list(range(32))
    bytes_only = empty.copy()
    bytes_only[:32, :k] = 1
    bits = CR.queue_admit(queue, 0, bytes_only)
    assert not bits.any() and CR.junk(bytes_only, bits) == list(range(32)) and not CR.hollow(bytes_only, chosen)
    honest = bytes_only.copy()
    honest[:32, -1] = 0x3F
    bits = CR.queue_admit(queue, 0, honest)
    assert list(bits) == list(chosen) and not CR.junk(honest, bits) and not CR.hollow(honest, chosen)
    assert not CR.scale_only(honest) and CR.live_blocks(bits) == [0]


class _Layer:
    def __init__(self, n: int, k: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) + 0.01).float()


class _Pool:
    def map(self, fn, tasks, chunksize=1):
        return [fn(t) for t in tasks]


def test_b_tables_are_row_tables_under_the_weights_root():
    """root(B) is AROW's row-table format over the weight rows and their scales (`b_rows`, `s8_table`), built in pool chunks
    or whole; the weights' root is the tree over the B roots, each opens in it, a tile's 32 B rows open against root(B) and
    a tampered weight row is refused."""
    from verity.ml.commitments import row_table as RT

    SV = _load("serve")
    layers = [_Layer(96, 64, 1), _Layer(64, 128, 2)]
    names = {0: "l.0.qkv_proj", 1: "l.0.down_proj"}
    model, key = {"repo": "r", "revision": "v", "recipe": None}, bytes(range(64))
    got = SV.b_tables(_Pool(), model, layers, names, key, chunk=40)
    w_dom = CM.domain(got["fields"], 2, "weights")
    w_lv = CM.levels(w_dom, got["roots"])
    assert got["root"] == w_lv[-1][0]
    for w, m in enumerate(layers):
        n, k = m.weight.shape
        rows = SV.b_rows(m)
        assert rows.shape == (n, k + 4) and np.array_equal(rows[:, :k].view(np.int8), m.weight.numpy())
        dom = CM.domain(SV.b_fields(model, w, names[w], n, k), n, "b-table")
        tab = CM.s8_table(dom, key, f"b/{w}", m.weight.numpy(), m.weight_scale.numpy().reshape(-1))
        assert tab.root == got["roots"][w]
        assert CM.verify(w_dom, got["root"], w, tab.root, CM.path(w_lv, w))
        ranks = range(32, 64)
        opening = tab.open(ranks)

        def witness(rs) -> dict:
            return {r: (bytes(rs[r]), 8 * rs.shape[1], CM.salt(key, f"b/{w}", r)) for r in ranks}

        assert RT.verify_rows(tab.root, dom.domain_id, n, opening, witness(rows))
        tampered = rows.copy()
        tampered[40, 3] ^= 1
        assert not RT.verify_rows(tab.root, dom.domain_id, n, opening, witness(tampered))


class _Batch:
    def __init__(self, ids):
        self.req_ids, self.num_reqs = list(ids) + [None], len(ids)


class _Runner:
    def __init__(self):
        self.input_batch = _Batch([])

    def execute_model(self, so):
        self.input_batch = _Batch(so.order)
        return "out"


class _Sched:
    def __init__(self, order, n):
        self.order, self.num_scheduled_tokens = order, dict(zip(order, n))
        self.total_num_scheduled_tokens = sum(n)


def test_schedule_tap_records_the_queue_and_commits_it():
    """`schedule_tap` records each engine step's requests and rows in batch order and restores the runner; two passes whose
    request ids differ give the same queue (requests indexed by first appearance), and its commitment is the tree over
    its two tables, the requests padded to S·T + 1."""
    SV = _load("serve")
    passes = []
    for base in (0, 8):
        runner = _Runner()
        spans, info, restore = SV.schedule_tap(runner)
        ids = [str(base + i) for i in range(3)]
        for order, n in (([ids[0], ids[1]], [5, 3]), ([ids[0], ids[1], ids[2]], [1, 1, 4]), ([ids[2]], [1])):
            assert runner.execute_model(_Sched(order, n)) == "out"
        runner.execute_model(_Sched([], []))
        restore()
        assert runner.execute_model.__func__ is _Runner.execute_model
        assert info == {"src": "input_batch", "short": 0} and len(spans) == 3
        passes.append(CR.queue_of(spans, 4, 64))
    a, b = passes
    assert np.array_equal(a["slot_map"], b["slot_map"]) and a["requests"] == b["requests"] == [(0, 0), (0, 2), (0, 2), (1, 3)]
    assert [int(CR.queue_select(a, s).sum()) for s in range(4)] == [8, 6, 1, 0]
    key = bytes(64)
    qc = SV.commit_queue("w", key, a)
    assert qc["slot_leaves"] == 4 * 64 // 16 and qc["request_leaves"] == 4 * 64 + 1
    assert qc["root"] == CM.step_root(CM.domain(qc["fields"], 2, "queue"), [qc["slots_root"], qc["requests_root"]])
    moved = {**a, "slot_map": a["slot_map"].copy()}
    moved["slot_map"][1, 9] = 3
    assert SV.commit_queue("w", key, moved)["root"] != qc["root"]


EPOCH, ROUNDS = 1, 3


class _Host:
    """The challenger on its auditor host, over loopback: a live window's coins (`coins.layout`, or `rounds` rounds at
    `epoch`), its record signed with a test LMS key and written to `directory` (`remote.open_session`), serving one
    session (`remote.serve`) on its own clock."""

    def __init__(self, tmp_path, epoch: int = EPOCH, rounds: int = ROUNDS, host: str = "vy-challenger-1") -> None:
        from verity.core.protocols.sampled_proofs import remote as R
        from verity.ml.crypto import lms
        self.R, self.directory = R, tmp_path / "challenger"
        self.directory.mkdir()
        lay = CO.layout(epoch, rounds)
        self.challenger = CO.Ch.LiveChallenger(lay["scheme"], lay["epoch"], lay["families"], random.Random("host").randbytes)
        key = lms.keygen(lms.Params(64, 4, 2), bytes([7]) * 64)
        self.public = lms.public_key(key)
        lms.LeafState.create(tmp_path / "leaves", self.public, 2)
        state = lms.LeafState(tmp_path / "leaves", self.public, 2)
        R.open_session(self.challenger, CO.Ch.ChallengerKey(4, 2, self.public), host,
                       lambda m: lms.sign(key, state, m).encode(), self.directory)
        state.close()
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.address = f"127.0.0.1:{self.listener.getsockname()[1]}"
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self) -> None:
        conn, _ = self.listener.accept()
        self.listener.close()
        with conn:
            self.reason = self.R.serve(conn, self.challenger, self.directory, itertools.count().__next__)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason


def _calls(rounds: int) -> list[dict]:
    """A window's calls, two linears a step, with the roots `serve.step_roots` and `receipts_root` read."""
    return [{"step": s, "linear": w, "root_a": hashlib.sha512(b"a%d/%d" % (s, w)).hexdigest(),
             "root_admit": hashlib.sha512(b"s%d/%d" % (s, w)).hexdigest(),
             "receipt_root": hashlib.sha512(b"r%d/%d" % (s, w)).hexdigest()} for s in range(rounds) for w in range(2)]


def _serve_window(cs, sv, calls: list[dict], registration: bytes) -> dict:
    """serve.py's coin path on `cs`: the salt coin after the registration root, round s's coin after step s's round root
    and the step's units under its live salt, then the draw coin after the receipts root; the manifest it writes."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    salt = cs.salt(registration)
    log = []
    for s in sorted({c["step"] for c in calls}):
        root = sv.step_roots("w", s, [c for c in calls if c["step"] == s])["root"]
        opening = cs.rounds.record(root)
        for c in calls:
            if c["step"] == s:
                c["salt"] = HT.live_salt(cs.commitment, opening.coin, s, root).hex()
        log.append({"round": s, "root": root.hex(), "opening": opening.to_json()})
    after = sv.receipts_root("w", calls)
    draw = cs.draw(after)
    cs.close()
    return {"window": "w", "calls": calls, "roots": {"registration": registration.hex()},
            "coins": {"mode": "live", **cs.record(), "commitment": cs.commitment.to_json(), "salt_coin": salt.to_json(),
                      "rounds": log, "draw_coin": draw.to_json(), "draw_after": after.hex()}}


def test_off_node_coins_are_checked_from_the_challengers_own_directory_and_a_forged_window_is_refused(tmp_path):
    """A window's coins from the challenger off the node (`coins.open_coins` over `RemoteChallenger`): salt, one round a
    step and the draw, each opened there after the root the window sent.  The verifier (`coins.py check`) reads only the
    challenger's directory and the window's manifest, recomputes the round roots, salts and draw root from the manifest's
    committed roots, and passes it under the pinned key; it refuses the deployed pin for a test key, and a window whose
    round root, opening, salt, registration or draw root isn't what the challenger logged."""
    SV = _load("serve")
    host = _Host(tmp_path)
    cs = CO.open_coins(host.address, EPOCH, ROUNDS)
    assert (cs.source, cs.label) == ("off-node", f"off-node {host.address} (unchecked)")
    reg = hashlib.sha512(b"registration").digest()
    m = _serve_window(cs, SV, _calls(ROUNDS), reg)
    assert host.ended() is None
    pin = hashlib.sha512(host.public).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert CO.main(["check", "--challenger-out", str(host.directory), "--manifest", str(tmp_path / "manifest.json"),
                    "--key-sha512", pin]) == 0
    e = SV.coins_expect(m)
    got = CO.check(host.directory, m["coins"], reg, **e, key_sha512=pin)
    assert got["ok"] and got["coins"] == "off-node vy-challenger-1 (checked)" and (got["rounds"], got["draws"]) == (ROUNDS, 1)
    assert "not the one pinned for vy-challenger-1" in CO.check(host.directory, m["coins"], reg, **e)["reason"]

    def refused(coins=m["coins"], registration=reg, **kw) -> dict:
        r = CO.check(host.directory, coins, registration, **{**e, **kw}, key_sha512=pin)
        assert not r["ok"] and r["coins"] == "refused"
        return {k: v for k, v in r["reasons"].items() if v is not None}

    other = hashlib.sha512(b"other").digest()
    assert list(refused(rounds=[e["rounds"][0], other, e["rounds"][2]])) == ["rounds"]
    swapped = [*m["coins"]["rounds"]]
    swapped[1] = {**swapped[1], "opening": swapped[2]["opening"]}
    assert list(refused(coins={**m["coins"], "rounds": swapped})) == ["rounds", "salts"]
    assert list(refused(salts={**e["salts"], 2: e["salts"][1]})) == ["salts"]
    assert list(refused(registration=other)) == ["salt"]
    assert list(refused(draw_after=other)) == ["draw"]
    assert list(refused(draw_after=None)) == ["draw"]
    alone = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes)
    assert list(refused(coins={**m["coins"], "commitment": alone.commitment.to_json()})) == ["commitment"]


def test_a_challenger_of_another_layout_is_refused_before_any_coin_and_its_session_closed(tmp_path):
    for i, (epoch, rounds) in enumerate(((EPOCH, ROUNDS + 1), (EPOCH + 1, ROUNDS))):
        (tmp_path / str(i)).mkdir()
        host = _Host(tmp_path / str(i), epoch, rounds)
        with pytest.raises(CO.Ch.CoinError, match="not this window's"):
            CO.open_coins(host.address, EPOCH, ROUNDS)
        assert host.ended() is None
        assert [json.loads(x)["event"] for x in (host.directory / "releases.jsonl").read_text().splitlines()] == ["commitment"]


def test_without_an_address_the_coins_are_in_process_and_labelled_not_certified(monkeypatch):
    monkeypatch.delenv(CO.CHALLENGER_ENV, raising=False)
    assert CO.address(None) is None and CO.address("h:1") == "h:1"
    monkeypatch.setenv(CO.CHALLENGER_ENV, "127.0.0.1:7811")
    assert CO.address(None) == "127.0.0.1:7811"
    cs = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes)
    assert (cs.source, cs.label, cs.record()["label"]) == ("in-process", "in-process (not certified)", CO.IN_PROCESS)
    assert CO.layout_reason(cs.commitment, EPOCH, ROUNDS) is None and CO.layout_reason(cs.commitment, EPOCH, 2) is not None


def test_an_address_makes_the_window_live_and_seeded_coins_with_one_are_refused(tmp_path, monkeypatch):
    """serve.py's coins follow the address: given one (here `$POUW_CHALLENGER`) and no ``--coins``, it opens the
    challenger's session before anything else (this host's layout isn't the window's, so it stops there), and
    ``--coins seeded`` beside an address is refused."""
    SV = _load("serve")
    host = _Host(tmp_path, EPOCH, ROUNDS + 1)
    monkeypatch.setenv(CO.CHALLENGER_ENV, host.address)
    argv = ["--out", str(tmp_path / "out"), "--at", "2026-10-20T00:00:00Z", "--by", "test", "--steps", str(ROUNDS)]
    with pytest.raises(CO.Ch.CoinError, match="not this window's"):
        SV.main(argv)
    assert host.ended() is None
    with pytest.raises(ValueError, match="serves live coins, not --coins seeded"):
        SV.main([*argv, "--coins", "seeded"])


def test_the_gates_picks_follow_the_draw_coin_and_its_root():
    SV = _load("serve")
    c = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes).commitment
    a, b = bytes(64), bytes([1]) * 64
    picks = SV.gate_rng(c, a, b).random()
    assert picks == SV.gate_rng(c, a, b).random() and picks not in (SV.gate_rng(c, b, b).random(), SV.gate_rng(c, a, a).random())
