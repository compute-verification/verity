"""served_h1's side computation (`served_h1/receipts.py`) is `h1tile.reference`'s receipt, unit by unit: on random s8
calls at K = 64 and 128 (a partial row block left uncovered), and saturated operands at K = 64; and each receipt reads
the unit's position, salt and roots and every operand byte of its tile.  A call's receipt tree (`served_h1/commit.py`)
is the same built in pool tasks or leaf by leaf, opens a unit's leaf, and refuses a flipped bit or another call's salt
label.  Under a call's admit bits (`served_h1/credit.py`), from the committed request queue, a dead unit's receipt is
the canonical empty receipt and a live unit's the reference's (`PouwH1Tile_v4`), an admitted block of zero-byte rows is
computed, receipt-checked and credited, a tile's admit bits open from the admit tree and a flipped one is refused, a
junk row in a slot the queue leaves unassigned is refused and credits nothing, so is the canonical empty row in a slot
the queue selects, and the tally's limbs are 16-bit.  A call's A table is AROW's row table, a tile's rows open against
it and the step's A root, and A' = -b is refused.  A linear's B table is the same format over its weight rows, under the
weights' root; the scheduler's record gives the request queue and its commitment.  A window's live coins
(`served_h1/coins.py`) come from a challenger session off the node over its stream (a fake auditor host on loopback, its
ledger and the challenger's record and close signed with a test key), which refuses a session of another layout; the
session is reserved at the gate for the window's layout and then opened on the root the caller gives (one quoted
argument each), its secret kept from every output and carried in the hello, which the challenger refuses without it. The
verifier checks the coins from the session as the gate publishes it and the auditor's ledger alone, through core's check
of both (`remote.trusted_session`), under the key its key.json holds, ``(checked)`` only against the key's deployment
record, the window's root and its period, and refuses a record under another key, a ledger core refuses, a session the
manifest doesn't name, outside the window's period or opened for another layout, a round root the challenger didn't
receive, a salt that isn't its round's coin's, and a draw after another root; an address without its session id is
refused, and with neither the coins are in-process and labelled not certified; serve.py refuses ``--gate``, naming
vllm's H1 path as the gated served window of record."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import itertools
import json
import random
import shutil
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
    """Under `PouwH1Tile_v4`, the admit unit reads no row byte, so a selected block of
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
    """On PoUW's side, a call's A table is AROW's row table (`row_table.commit` over `s8_row`s gives the root `leaf` and
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
    """A block whose 32 slots the queue selects but whose rows are all the
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
ROOT = hashlib.sha512(b"the window's root").digest()
SID, LANE, PERIOD, CIRCUIT, PIN = "s20261020T120000Z-0a1b", "kernels", "test-period", "ab" * 64, "0" * 40
WINDOW = {"root": ROOT.hex(), "period": PERIOD}
#: the host's pins log as infra publishes it, recording `PIN`, which the session's open seals, and the verifier's own
#: first-parent history of main, which holds `PIN`
PINS = [CO.Ch.pin_line(PIN, PIN, "2026-10-20T11:00:00Z")]
MAIN = ["e1" * 20, PIN]


@functools.cache
def _host_key():
    """The fake auditor host's LMS key: 16 leaves, a window's ledger entries, record and close at one each."""
    from verity.ml.crypto import lms
    return lms.keygen(lms.Params(64, 4, 4), bytes([7]) * 64)


class _Host:
    """The challenger on its auditor host over loopback, as the gate runs a window's session (`auditor.py`): the
    lane's period opened and the session reserved and opened on `root` in the auditor's signed ledger (`ledger`, v2), a
    live window's coins (`coins.layout`, or `rounds` rounds at `epoch`) with its record signed and written to
    `directory` (`remote.open_session`) beside an empty hellos.jsonl, serving one session (`remote.serve`, its hello
    carrying `secret`) on its own clock, and once it has ended (`ended`) its end entry sealing out/'s files.  The
    ledger's entries, the record and the close are signed with one test key at its next leaf each.  Published as the
    gate does: request.json and open.json beside the challenger's out/ (`directory`)."""

    def __init__(self, tmp_path, epoch: int = EPOCH, rounds: int = ROUNDS, host: str = "vy-challenger-1",
                 root: bytes = ROOT, secret: bytes = bytes([5]) * 32) -> None:
        from verity.core.protocols.sampled_proofs import remote as R
        from verity.ml.crypto import lms
        self.R, self.directory, self.secret = R, tmp_path / "session" / "out", secret
        self.directory.mkdir(parents=True)
        self.private = _host_key()
        self.public = lms.public_key(self.private)
        self.key = CO.Ch.ChallengerKey(4, 4, self.public)
        self.leaves, self.lines, self.ledger = itertools.count(), [], tmp_path / "ledger.jsonl"
        lay, pin = CO.layout(epoch, rounds), hashlib.sha512(self.public).hexdigest()
        req = {"by": LANE, "peer": "vy-nebius-2", **lay, "minutes": 5}
        self.append("period-open", None, {"period": PERIOD, "lane": LANE, "by": LANE, "cap": 2, "circuit": CIRCUIT,
                                          "pinned": PIN, "key_sha512": pin, "opened_utc": "2026-10-20T11:59:00Z"})
        reserve = self.append("reserve", SID, {**req, "pinned": PIN, "key_sha512": pin, "host": f"{host}.{SID}",
                                               "started_utc": "2026-10-20T12:00:00Z", "expires_utc": "2026-10-20T12:05:00Z"})
        self.append("open", SID, {"window": {"root": root.hex()}, "reserve": reserve, **req, "opened_by": LANE,
                                  "pinned": PIN, "pins": CO.Ch.pins_seal(PINS), "key_sha512": pin,
                                  "host": f"{host}.{SID}", "port": 7811,
                                  "secret_sha512": hashlib.sha512(secret).hexdigest(),
                                  "started_utc": "2026-10-20T12:01:00Z"})
        (self.directory.parent / "request.json").write_text(json.dumps({**req, "session": SID}))
        (self.directory.parent / "open.json").write_text(json.dumps({"session": SID, "root": root.hex(), "by": LANE}))
        self.challenger = CO.Ch.LiveChallenger(lay["scheme"], lay["epoch"], lay["families"], random.Random("host").randbytes)
        self._session = R.open_session(self.challenger, self.key, f"{host}.{SID}", self.sign, self.directory, SID)
        (self.directory / R.HELLOS).write_text("")
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.address = f"127.0.0.1:{self.listener.getsockname()[1]}"
        self.session = CO.Endpoint(SID, self.address, secret)
        self.reason: str | None = None
        self._ended = False
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def sign(self, message: bytes) -> bytes:
        from verity.ml.crypto import lms
        return lms.sign_at(self.private, next(self.leaves), message).encode()

    def append(self, kind: str, session: str | None, body: dict) -> int:
        """The ledger's next entry, signed at the key's next leaf: its seq."""
        self.lines.append(CO.Ch.ledger_line(self.lines, kind, session, body, self.public[:16], self.sign))
        self.ledger.write_text("".join(line + "\n" for line in self.lines))
        return len(self.lines) - 1

    def _run(self) -> None:
        conn, _ = self.listener.accept()
        self.listener.close()
        with conn:
            self.reason = self.R.serve(conn, self._session, itertools.count().__next__, self.secret)

    def ended(self) -> str | None:
        """Join the session, and the ledger ends it as the gate's ``run`` does: None when its client closed it."""
        self._thread.join()
        if not self._ended:
            self._ended = True
            d, R = self.directory, self.R
            record = CO.Ch.ChallengerRecord.from_json(json.loads((d / R.RECORD).read_text()))
            released = {k: len(r) for k, r in R.released_roots(d / R.RELEASES, record.commitment).items()}
            closed = self.reason is None
            self.append("end", SID, {
                "rc": 0 if closed else 1, "closed": closed, "ended": "closed by the client" if closed else self.reason,
                "files": {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())},
                "released": released, "complete": all(released[k] == n for k, n in record.commitment.families),
                "record_leaf": int.from_bytes(record.signature[:4], "big"), "commitment": record.commitment.root.hex(),
                "refused_hellos": len((d / R.HELLOS).read_text().splitlines()), "ended_utc": "2026-10-20T12:02:00Z"})
        return self.reason


def _calls(rounds: int) -> list[dict]:
    """A window's calls, two linears a step, with the roots `serve.step_roots` and `receipts_root` read."""
    return [{"step": s, "linear": w, "root_a": hashlib.sha512(b"a%d/%d" % (s, w)).hexdigest(),
             "root_admit": hashlib.sha512(b"s%d/%d" % (s, w)).hexdigest(),
             "receipt_root": hashlib.sha512(b"r%d/%d" % (s, w)).hexdigest()} for s in range(rounds) for w in range(2)]


def _serve_window(cs, sv, calls: list[dict], root: bytes) -> dict:
    """A served window's coin path on `cs`: the salt coin after the window's root, round s's coin after step s's round
    root and the step's units under its live salt, then the draw coin after the receipts root; the manifest, its window
    `WINDOW`."""
    from verity.ml.protocols.pouw.circuit import h1tile as HT
    salt = cs.salt(root)
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
    return {"window": "w", "calls": calls,
            "coins": {"mode": "live", **cs.record(), "commitment": cs.commitment.to_json(), "salt_coin": salt.to_json(),
                      "rounds": log, "draw_coin": draw.to_json(), "draw_after": after.hex(), "window": WINDOW}}


def test_off_node_coins_are_checked_from_the_published_session_and_its_ledger_and_a_forged_window_is_refused(tmp_path):
    """A window's coins from the challenger off the node (`coins.open_coins` over `RemoteChallenger`): salt, one round a
    step and the draw, each opened there after the root the window sent.  The verifier (`coins.py check`) reads only the
    session as the gate publishes it (out/ beside request.json and open.json), every head of the auditor's ledger and
    the window's manifest, recomputes the round roots, salts and draw root from the manifest's committed roots, and
    passes it under the key key.json holds, ``(checked)`` only when it names the key's deployment record, the window's
    root and its period and all match.  The ledger and the session's files are core's to accept
    (`remote.trusted_session`): it refuses a key.json with another key, a key other than the record's, a root the
    ledger opens no session on, an edited or forked ledger, an open that binds another reserve or a second open, a
    record of another host, a refused hello the end didn't seal, a session that ended unclosed, and a session read
    without the host's pins log or the verifier's first-parent history of main, or whose pin isn't on that history
    (`challenger.pinned_session`).  The check itself
    refuses a session the manifest doesn't name, a manifest naming another window, a window outside the verifier's
    period, an open of another layout, and a window whose round root, opening, salt or draw root isn't what the
    challenger logged."""
    from verity.ml.crypto import lms
    SV = _load("serve")
    host = _Host(tmp_path)
    cs = CO.open_coins(host.session, EPOCH, ROUNDS)
    assert (cs.source, cs.label) == ("off-node", f"off-node session {SID} at {host.address} (unchecked)")
    assert cs.record()["session"] == SID
    m = _serve_window(cs, SV, _calls(ROUNDS), ROOT)
    assert host.ended() is None
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    session, heads, key = host.directory.parent, [host.ledger], host.key
    early = tmp_path / "early.jsonl"
    early.write_text("".join(line + "\n" for line in host.lines[:3]))
    record = {"schema": "challenger-key/v1", "H": 4, "W": 4, "pubkey": host.public.hex(),
              "pubkey_sha512": hashlib.sha512(host.public).hexdigest()}
    (tmp_path / "key-record.json").write_text(json.dumps(record))
    (tmp_path / "pins.jsonl").write_text("".join(line + "\n" for line in PINS))
    (tmp_path / "main.txt").write_text("".join(c + "\n" for c in MAIN))
    assert CO.main(["check", "--challenger-out", str(session), "--ledger", str(early), "--ledger", str(host.ledger),
                    "--pins", str(tmp_path / "pins.jsonl"), "--main", str(tmp_path / "main.txt"),
                    "--manifest", str(tmp_path / "manifest.json"), "--key-record", str(tmp_path / "key-record.json"),
                    "--period", PERIOD, "--root", ROOT.hex()]) == 0
    for bad, why in (({"pubkey_sha512": "00" * 64}, "not the one whose SHA-512 it names"),
                     ({"schema": "challenger-key/v2"}, "a challenger key record is challenger-key/v1")):
        (tmp_path / "bad-record.json").write_text(json.dumps({**record, **bad}))
        with pytest.raises(CO.Ch.CoinError, match=why):
            CO.key_record(tmp_path / "bad-record.json")
    e = SV.coins_expect(m)
    got = CO.check(session, m["coins"], heads, PINS, MAIN, **e, key=key, period=PERIOD, root=ROOT)
    assert got["ok"] and got["coins"] == f"off-node vy-challenger-1.{SID} (checked)"
    assert (got["rounds"], got["draws"], got["window"], got["root"]) == (ROUNDS, 1, WINDOW, ROOT.hex())
    assert got["ledger"] == {"session": SID, "prover": LANE, "reserve": 1, "open": 2, "end": 3, "end_complete": True,
                             "refused_hellos": 0, "period": PERIOD, "cap": 2, "circuit": CIRCUIT}
    loose = CO.check(host.directory, m["coins"], heads, PINS, MAIN, **e)
    assert loose["ok"] and loose["coins"] == f"off-node vy-challenger-1.{SID} (checked; key, root, period unanchored)"
    other = lms.public_key(lms.keygen(lms.Params(64, 4, 2), bytes([8]) * 64))
    wrong = CO.check(session, m["coins"], heads, PINS, MAIN, **e, key=CO.Ch.ChallengerKey(4, 2, other))
    assert not wrong["ok"] and wrong["reason"].startswith("the session's key is sha512")
    forged = tmp_path / "forged"
    shutil.copytree(session, forged)
    (forged / "out" / "key.json").write_text(json.dumps(CO.Ch.ChallengerKey(4, 2, other).to_json()))
    bad = CO.check(forged, m["coins"], heads, PINS, MAIN, **e)
    assert not bad["ok"] and bad["coins"] == "refused" and "not the window's pinned key" in bad["reason"]
    nameless = CO.check(session, {**m["coins"], "window": {}}, heads, PINS, MAIN, **e)
    assert not nameless["ok"] and nameless["reason"].startswith("the verifier names no window root")

    def check(coins=m["coins"], directory=session, heads=heads, **kw) -> dict:
        r = CO.check(directory, coins, heads, **{**e, "pins": PINS, "main": MAIN, "key": key, "period": PERIOD,
                                                 "root": ROOT, **kw})
        assert not r["ok"] and r["coins"] == "refused"
        return r

    def refused(**kw) -> set[str]:
        return {k for k, v in check(**kw)["reasons"].items() if v is not None}

    def core(**kw) -> str:
        r = check(**kw)
        assert "reasons" not in r and r["reason"].startswith("CoinError: ")
        return r["reason"]

    def resigned(name: str, edit) -> list[Path]:
        """The ledger with its entries edited by `edit` and each signed again at its own leaf (a new one at the next)."""
        entries = [json.loads(line) for line in host.lines]
        edit(entries)
        lines, leaves = [], itertools.count(len(host.lines) + 3)
        for x in entries:
            q = int(x["sig"][:8], 16) if "sig" in x else next(leaves)
            lines.append(CO.Ch.ledger_line(lines, x["kind"], x["session"], x["body"], host.public[:16],
                                           lambda msg, q=q: lms.sign_at(host.private, q, msg).encode()))
        (tmp_path / f"{name}.jsonl").write_text("".join(line + "\n" for line in lines))
        return [tmp_path / f"{name}.jsonl"]

    def republished(name: str, edit) -> Path:
        d = tmp_path / name
        shutil.copytree(session, d)
        edit(d / "out")
        return d

    other = hashlib.sha512(b"other").digest()
    assert "opens no challenger session for this window" in core(root=other)
    assert "opens no challenger session for this window" in core(
        coins={**m["coins"], "window": {**WINDOW, "root": other.hex()}}, root=None)
    edited = tmp_path / "edited.jsonl"
    edited.write_text(host.ledger.read_text().replace('"minutes":5', '"minutes":6', 1))
    assert "signature doesn't verify" in core(heads=[edited])
    forked = resigned("forked", lambda x: x[2]["body"].update(port=7812))
    assert "don't extend one another" in core(heads=[*heads, *forked])
    assert "binds no reserve" in core(heads=resigned("rebound", lambda x: x[2]["body"].update(reserve=0)))
    assert "doesn't carry its reserve's request" in core(
        heads=resigned("recarried", lambda x: x[2]["body"].update(peer="vy-nebius-1")))
    assert "a second open" in core(heads=resigned("reopens", lambda x: x.append({k: v for k, v in x[2].items() if k != "sig"})))
    assert "names host" in core(heads=resigned("rehosted", lambda x: [x[i]["body"].update(host=f"vy-challenger-2.{SID}")
                                                                      for i in (1, 2)]))
    assert "unclosed" in core(heads=resigned("unclosed", lambda x: x[3]["body"].update(closed=False)))
    assert core(pins=None) == f"CoinError: {CO.Ch.NO_PINS}"
    assert core(main=None) == f"CoinError: {CO.Ch.NO_MAIN}"
    assert core(main=MAIN[:1]) == f"CoinError: {CO.Ch.first_parent_reason(SID, PIN, MAIN[:1])}"
    hello = republished("hello", lambda out: (out / "hellos.jsonl").write_text('{"n":0,"reason":"r"}\n'))
    assert "hellos.jsonl is not the one the auditor's ledger sealed" in core(directory=hello)
    assert refused(period="another") == {"period"}
    assert refused(coins={**m["coins"], "session": "s20261020T120000Z-ffff"}) == {"session"}
    assert refused(coins={**m["coins"], "window": {**WINDOW, "period": "another"}}) == {"window"}
    assert refused(heads=resigned("unperiod", lambda x: x[0]["body"].update(lane="infra"))) == {"period", "window"}
    relaid = [list(f) for f in CO.layout(EPOCH, ROUNDS + 1)["families"]]
    assert refused(heads=resigned("relaid", lambda x: [x[i]["body"].update(families=relaid) for i in (1, 2)])) == {"layout"}
    assert refused(rounds=[e["rounds"][0], other, e["rounds"][2]]) == {"rounds"}
    swapped = [*m["coins"]["rounds"]]
    swapped[1] = {**swapped[1], "opening": swapped[2]["opening"]}
    assert refused(coins={**m["coins"], "rounds": swapped}) == {"rounds", "salts"}
    assert refused(salts={**e["salts"], 2: e["salts"][1]}) == {"salts"}
    assert refused(draw_after=other) == {"draw"}
    assert refused(draw_after=None) == {"draw"}
    alone = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes)
    assert refused(coins={**m["coins"], "commitment": alone.commitment.to_json()}) == {"commitment"}


def test_round_coins_pipelined_off_node_open_each_after_its_root_in_order(tmp_path):
    """`coins.RoundPipe` over the challenger off the node: every round's root goes out before any coin is read back, and
    each coin comes back opened after its own root, round by round in the challenger's own release log and under
    `order_reason`; closing waits for the coins still out, then closes the session.  In process the same roots open at
    once, and a root past the epoch's rounds is refused before it is sent."""
    from verity.core.protocols.sampled_proofs import remote as R
    host = _Host(tmp_path, rounds=4)
    cs = CO.open_coins(host.session, EPOCH, 4)
    roots = [hashlib.sha512(b"round %d" % i).digest() for i in range(4)]
    first = cs.rounds.record(roots[0])
    pipe = CO.RoundPipe(cs)
    assert [pipe.send(r) for r in roots[1:]] == [1, 2, 3]
    got = [pipe.opening(i) for i in (3, 1, 2)]
    assert [o.index for o in got] == [3, 1, 2] and all(o.after == roots[o.index] for o in got)
    assert pipe.opening(0) == first and pipe.reason() is None and pipe.roots == roots
    with pytest.raises(CO.Ch.CoinError, match=f"every {CO.Wd.ROUND} coin of the epoch is open"):
        pipe.send(roots[0])
    pipe.close()
    assert host.ended() is None
    log = [json.loads(x) for x in (host.directory / R.RELEASES).read_text().splitlines()]
    rounds = [(e["event"], e["index"], e.get("root") or e.get("after")) for e in log if e.get("label") == CO.Wd.ROUND]
    assert rounds == [(ev, i, r.hex()) for i, r in enumerate(roots) for ev in ("root", "release")]
    alone = CO.open_coins(None, EPOCH, 4, random.Random("prover").randbytes)
    alone.rounds.record(roots[0])
    local = CO.RoundPipe(alone)
    assert [local.send(r) for r in roots[1:]] == [1, 2, 3]
    assert [local.opening(i, wait=False).after for i in range(4)] == roots and local.reason() is None
    local.close()


def test_the_draw_coin_and_its_time_are_read_under_the_lock_round_roots_are_sent_under():
    """`RoundPipe.draw` reads its time on the pipe's clock while it holds the lock `send` records each round under, through
    the draw's open: another thread can't take it then, every round sent before has an earlier time and the one sent
    after a later one; a draw after the coins are closed is refused."""
    ticks = itertools.count()
    cs = CO.open_coins(None, EPOCH, 4, random.Random("prover").randbytes)
    roots = [hashlib.sha512(b"round %d" % i).digest() for i in range(4)]
    cs.rounds.record(roots[0])
    pipe = CO.RoundPipe(cs, lambda: float(next(ticks)))
    open_, held = cs.challenger.open, []

    def spy(label, index, after):
        if label == CO.Wd.DRAW:
            t = threading.Thread(target=lambda: held.append(pipe._cv.acquire(blocking=False)))
            t.start()
            t.join()
        return open_(label, index, after)
    cs.challenger.open = spy
    assert [pipe.send(r) for r in roots[1:3]] == [1, 2]
    after = hashlib.sha512(b"manifest").digest()
    draw, at = pipe.draw(after)
    assert held == [False] and draw.label == CO.Wd.DRAW and draw.after == after
    assert max(pipe.sent_at.values()) < at < pipe.sent_at[pipe.send(roots[3])]
    pipe.close()
    with pytest.raises(CO.Ch.CoinError, match="closed"):
        pipe.draw(after)


def _pipe_server(sock, challenger, answers: list[str]) -> None:
    """The challenger's side of a `RoundPipe` stream for its refusals: the hello's commitment, then each open frame's
    reply as `answers` names it in turn ("ok" its coin, "refuse" a refusal, "other" the next coin's opening)."""
    from verity.core.protocols.sampled_proofs import remote as R
    w = R._Wire(sock)
    w.recv()
    w.send({"type": "commitment", "commitment": challenger.commitment.to_json()})
    for a in answers:
        head = w.recv()
        if a == "refuse":
            w.send({"type": "refused", "reason": "not this session's"})
            continue
        o = challenger.open(head["label"], head["index"] + (a == "other"), bytes.fromhex(head["after"]))
        w.send({"type": "opening", "opening": o.to_json()})


@pytest.mark.parametrize("answer,why", [("refuse", "the challenger refused: not this session's"),
                                        ("other", "pouw/round coin 2, not its pouw/round coin 1")])
def test_round_coins_pipelined_refuse_every_later_round_after_a_refusal_or_a_bad_opening(answer, why):
    """A `RoundPipe` whose challenger refuses round 1's open, or answers it with another coin's opening, ends the session:
    round 1's coin and every later one are refused, round 0's (back before) is kept, and no root is sent after."""
    from verity.core.protocols.sampled_proofs import remote as R
    lay = CO.layout(EPOCH, 4)
    ch = CO.Ch.LiveChallenger(lay["scheme"], lay["epoch"], lay["families"], random.Random("host").randbytes)
    a, b = socket.socketpair()
    t = threading.Thread(target=_pipe_server, args=(b, ch, ["ok", answer]))
    t.start()
    cs = CO.Coins(R.RemoteChallenger(a), "off-node", CO.Endpoint("s", "127.0.0.1:1"))
    pipe = CO.RoundPipe(cs)
    roots = [hashlib.sha512(b"round %d" % i).digest() for i in range(3)]
    assert [pipe.send(r) for r in roots[:2]] == [0, 1]
    assert pipe.opening(0).after == roots[0]
    with pytest.raises(CO.Ch.CoinError, match=why):
        pipe.opening(1)
    with pytest.raises(CO.Ch.CoinError, match="a retry never redraws"):
        pipe.send(roots[2])
    assert pipe.roots == roots[:2] and len(pipe.openings) == 1
    pipe.close()
    t.join()
    b.close()


def _gate(sent: list, *replies: dict, rc: int = 0):
    """A fake ``ssh GATE ...``: each call's shell words (the gate's remote shlex) into `sent`, its replies in turn."""
    import shlex
    import subprocess
    answers = iter(replies)

    def run(argv, **_):
        sent.append((argv[-2], shlex.split(argv[-1])))
        return subprocess.CompletedProcess(argv, rc, json.dumps(next(answers)), "")
    return run


def test_a_window_is_reserved_then_opened_on_the_callers_root_and_its_secret_is_the_sessions_alone():
    """`coins.reserve` asks the gate for the window's layout on the peer (one quoted argument each, ``--for`` vy-admin's)
    and `coins.open_window` opens the reserved session on the root the caller gives; the session carries the secret,
    which no answer, repr or refusal shows, and a refusal, an answer for another session or root, or one without a
    secret, raises."""
    sent: list = []
    lay = CO.layout(EPOCH, ROUNDS)
    reserved = {"session": "s20261020T130000Z-0a1b", "seq": 4, "peer": "vy-nebius-2", "by": "vy-admin", "for": "kernels",
                "expires_utc": "2026-10-20T13:02:00Z", "period": "dev-kernels"}
    got = CO.reserve("research@gate", "vy-nebius-2", EPOCH, ROUNDS, 2, "kernels", run=_gate(sent, reserved))
    assert got == reserved and sent[-1][0] == "research@gate" and sent[-1][1][0] == "reserve"
    assert dict(zip(sent[-1][1][1::2], sent[-1][1][2::2])) == {
        "--peer": "vy-nebius-2", "--scheme": lay["scheme"], "--families": json.dumps(lay["families"], separators=(",", ":")),
        "--epoch": str(EPOCH), "--minutes": "2", "--for": "kernels"}
    sid, root, secret = reserved["session"], CO.stand_in_root(reserved["session"]), bytes(range(32))
    assert root == hashlib.sha512(CO.STAND_IN_TAG + hashlib.sha512(sid.encode()).digest()).digest() != CO.stand_in_root("s2")
    opened = {"session": sid, "port": 7811, "peer": "vy-nebius-2", "host": f"vy-challenger-1.{sid}", "seq": 5,
              "reserve": 4, "root": root.hex(), "by": "vy-admin", "for": "kernels", "opened_by": "vy-admin",
              "secret": secret.hex(), "commitment": "cd" * 64}
    at, answer = CO.open_window("research@gate", sid, root, run=_gate(sent, opened))
    assert sent[-1][1] == ["open", "--session", sid, "--root", root.hex()]
    assert at == CO.Endpoint(sid, "127.0.0.1:7811", secret, "cd" * 64)
    assert answer == {k: v for k, v in opened.items() if k != "secret"}
    assert secret.hex() not in repr(at) + json.dumps(answer)
    with pytest.raises(CO.Ch.CoinError, match="refused reserve: period dev-kernels holds its cap"):
        CO.reserve("research@gate", "vy-nebius-2", EPOCH, ROUNDS,
                   run=_gate(sent, {"refused": "period dev-kernels holds its cap of 2 windows"}, rc=3))
    with pytest.raises(CO.Ch.CoinError, match="refused reserve"):
        CO.reserve("research@gate", "vy-nebius-2", EPOCH, ROUNDS, run=_gate(sent, {}, rc=255))
    with pytest.raises(CO.Ch.CoinError, match=f"refused open: session {sid} is opened already"):
        CO.open_window("research@gate", sid, root, run=_gate(sent, {"refused": f"session {sid} is opened already"}, rc=3))
    for bad in ({**opened, "root": CO.stand_in_root("s2").hex()}, {**opened, "session": "s2"},
                {k: v for k, v in opened.items() if k != "secret"}):
        with pytest.raises(CO.Ch.CoinError, match="another session, root or port, or no secret") as refusal:
            CO.open_window("research@gate", sid, root, run=_gate(sent, bad))
        assert secret.hex() not in str(refusal.value)
    for unsigned in (None, "cd" * 32):
        with pytest.raises(CO.Ch.CoinError, match="signed no coin commitment") as refusal:
            CO.open_window("research@gate", sid, root, run=_gate(sent, {**opened, "commitment": unsigned}))
        assert secret.hex() not in str(refusal.value)


def test_coins_py_open_reserves_then_opens_on_the_stand_in_root_and_writes_the_secret_to_its_file_alone(
        tmp_path, monkeypatch, capsys):
    """``coins.py open`` (the development path): a reserve, then an open on the session's stand-in root; the answers and
    the client's env are printed, the root named a stand-in, the secret only in the new mode-600 file; an existing file
    is refused before any reserve, and an open the gate refuses names the reserved session it leaves."""
    secret, sent = bytes([9]) * 32, []

    def gate_call(gate, args, ssh=CO.SSH, run=None):
        sent.append(args)
        if args[0] == "reserve":
            return {"session": "s1", "seq": 4, "peer": "vy-nebius-2", "by": "kernels", "expires_utc": "2026-10-20T13:02:00Z",
                    "period": "dev-kernels"}
        return {"session": "s1", "port": 7811, "root": args[4], "seq": 5, "reserve": 4, "secret": secret.hex(),
                "commitment": "ab" * 64}

    monkeypatch.setattr(CO, "gate_call", gate_call)
    f = tmp_path / "s1.secret"
    argv = ["open", "--gate", "research@gate", "--peer", "vy-nebius-2", "--epoch", "1", "--rounds", "3", "--minutes",
            "2", "--secret-file", str(f)]
    assert CO.main(argv) == 0
    out = capsys.readouterr().out
    assert sent[1] == ["open", "--session", "s1", "--root", CO.stand_in_root("s1").hex()]
    assert secret.hex() not in out and CO.STAND_IN in out
    assert out.splitlines()[-1] == (f"{CO.CHALLENGER_ENV}=127.0.0.1:7811 {CO.SESSION_ENV}=s1 {CO.SECRET_ENV}={f} "
                                    f"{CO.COMMITMENT_ENV}={'ab' * 64}")
    assert CO.read_secret(f) == secret and f.stat().st_mode & 0o777 == 0o600
    sent.clear()
    with pytest.raises(SystemExit):
        CO.main(argv)
    assert sent == []

    def refusing(gate, args, ssh=CO.SSH, run=None):
        if args[0] == "open":
            raise CO.Ch.CoinError("infra's gate refused open: root … already has its session")
        return gate_call(gate, args)

    monkeypatch.setattr(CO, "gate_call", refusing)
    with pytest.raises(CO.Ch.CoinError, match="s1 is reserved and its open failed .* refused window of period 'dev-kernels'"):
        CO.main([*argv[:-1], str(tmp_path / "s1-again.secret")])
    assert not (tmp_path / "s1-again.secret").exists()


def test_a_sessions_secret_is_its_hello_and_a_hello_without_it_is_refused(tmp_path, monkeypatch):
    """`open_coins` hands a session's secret to the client's `connect`, whose hello carries it, and the challenger
    refuses a hello without it before any coin; the hello's commitment must be the one the gate's open answered, else
    the session is refused and closed before any coin."""
    for name in (CO.CHALLENGER_ENV, CO.SESSION_ENV, CO.SECRET_ENV, CO.COMMITMENT_ENV):
        monkeypatch.delenv(name, raising=False)
    host = _Host(tmp_path / "a")
    seen = {}

    def connect(address, secret=None):
        seen["secret"] = secret
        return CO.R.RemoteChallenger.connect(address, secret)

    signed = host.challenger.commitment.root.hex()
    at = dataclasses.replace(host.session, commitment=signed)
    cs = CO.open_coins(at, EPOCH, ROUNDS, connect=connect)
    assert seen["secret"] == host.secret
    cs.close()
    assert host.ended() is None
    for name, session, why in (("b", lambda s: dataclasses.replace(s, secret=None), "hello with the session's secret"),
                               ("c", lambda s: dataclasses.replace(s, commitment="cd" * 64),
                                "answered the hello with commitment .* not the cdcd")):
        other = _Host(tmp_path / name)
        with pytest.raises(CO.Ch.CoinError, match=why):
            CO.open_coins(session(other.session), EPOCH, ROUNDS)
        other.ended()
        assert [json.loads(x)["event"] for x in (other.directory / "releases.jsonl").read_text().splitlines()] == ["commitment"]
    assert CO.session(host.address, SID, commitment=signed) == CO.Endpoint(SID, host.address, None, signed)
    for bad in ({"commitment": signed}, {"address": host.address, "sid": "s", "commitment": signed[:64]}):
        with pytest.raises(CO.Ch.CoinError, match="an id and an address|128 lowercase hex"):
            CO.session(**bad)
    f = tmp_path / "secret"
    CO.write_secret(f, at.secret)
    assert CO.session(host.address, SID, str(f), signed) == at
    with pytest.raises(CO.Ch.CoinError, match="an id and an address"):
        CO.session(secret_file=str(f))
    (tmp_path / "short").write_text("ab" * 16 + "\n")
    with pytest.raises(CO.Ch.CoinError, match="64 lowercase hex digits"):
        CO.session(host.address, SID, str(tmp_path / "short"))


def test_a_challenger_of_another_layout_is_refused_before_any_coin_and_its_session_closed(tmp_path):
    for i, (epoch, rounds) in enumerate(((EPOCH, ROUNDS + 1), (EPOCH + 1, ROUNDS))):
        (tmp_path / str(i)).mkdir()
        host = _Host(tmp_path / str(i), epoch, rounds)
        with pytest.raises(CO.Ch.CoinError, match="not this window's"):
            CO.open_coins(host.session, EPOCH, ROUNDS)
        assert host.ended() is None
        assert [json.loads(x)["event"] for x in (host.directory / "releases.jsonl").read_text().splitlines()] == ["commitment"]


def test_open_coins_waits_out_a_refused_connect_until_its_deadline_and_raises_any_other_error_at_once():
    """An onsite window's client retries a refused connect (no hello, no coin) until the session's tunnel listens, every
    half second up to `wait_s` on the injected clock; past it `CoinError` refuses it, and any other error raises at once."""
    at = CO.Endpoint(SID, "127.0.0.1:7811", b"\x01" * 32)
    challenger = CO.Wd.live_challenger(EPOCH, ROUNDS, random.Random("challenger").randbytes)

    def run(outcomes, wait_s=CO.R.TUNNEL_WAIT_S):
        now, calls, slept = [0.0], [], []

        def connect(address, secret):
            calls.append((address, secret))
            got = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(got, BaseException):
                raise got
            return got

        def sleep(s):
            slept.append(s)
            now[0] += s

        try:
            return CO.open_coins(at, EPOCH, ROUNDS, connect=connect, wait_s=wait_s, clock=lambda: now[0], sleep=sleep)
        finally:
            assert calls == [(at.address, at.secret)] * len(calls)
            run.calls, run.slept = len(calls), slept

    cs = run([ConnectionRefusedError(), ConnectionRefusedError(), challenger])
    assert cs.challenger is challenger and cs.source == "off-node" and (run.calls, run.slept) == (3, [0.5, 0.5])
    with pytest.raises(CO.Ch.CoinError, match="nothing listened"):
        run([ConnectionRefusedError()], wait_s=2)
    assert (run.calls, run.slept) == (5, [0.5] * 4)
    with pytest.raises(ConnectionResetError):
        run([ConnectionResetError()])
    assert (run.calls, run.slept) == (1, [])


def test_without_a_session_the_coins_are_in_process_and_labelled_not_certified_and_half_a_session_is_refused(monkeypatch):
    for name in (CO.CHALLENGER_ENV, CO.SESSION_ENV, CO.SECRET_ENV, CO.COMMITMENT_ENV):
        monkeypatch.delenv(name, raising=False)
    assert CO.session() is None and CO.session("h:1", "s1") == CO.Endpoint("s1", "h:1")
    for half in (("h:1", None), (None, "s1")):
        with pytest.raises(CO.Ch.CoinError, match="an id and an address"):
            CO.session(*half)
    monkeypatch.setenv(CO.CHALLENGER_ENV, "127.0.0.1:7811")
    with pytest.raises(CO.Ch.CoinError, match="an id and an address"):
        CO.session()
    monkeypatch.setenv(CO.SESSION_ENV, "s2")
    assert CO.session() == CO.Endpoint("s2", "127.0.0.1:7811")
    cs = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes)
    assert (cs.source, cs.label, cs.record()["label"]) == ("in-process", "in-process (not certified)", CO.IN_PROCESS)
    assert CO.layout_reason(cs.commitment, EPOCH, ROUNDS) is None and CO.layout_reason(cs.commitment, EPOCH, 2) is not None


def test_an_address_makes_the_window_live_and_seeded_coins_with_one_are_refused(tmp_path, monkeypatch):
    """serve.py's coins follow the session: given one (here `$POUW_CHALLENGER`, `$POUW_CHALLENGER_SESSION` and
    `$POUW_CHALLENGER_SECRET_FILE`) and no ``--coins``, it opens the challenger's session before anything else (this
    host's layout isn't the window's, so it stops there), and ``--coins seeded`` beside a session is refused.
    ``--gate`` is refused, with a session or without, and the refusal names the gated served window of record, vllm's H1
    path (`integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py`)."""
    SV = _load("serve")
    host = _Host(tmp_path, EPOCH, ROUNDS + 1)
    CO.write_secret(tmp_path / "secret", host.secret)
    monkeypatch.delenv(CO.COMMITMENT_ENV, raising=False)
    monkeypatch.setenv(CO.CHALLENGER_ENV, host.address)
    monkeypatch.setenv(CO.SESSION_ENV, host.session.id)
    monkeypatch.setenv(CO.SECRET_ENV, str(tmp_path / "secret"))
    argv = ["--out", str(tmp_path / "out"), "--at", "2026-10-20T00:00:00Z", "--by", "test", "--steps", str(ROUNDS)]
    with pytest.raises(CO.Ch.CoinError, match="not this window's"):
        SV.main(argv)
    assert host.ended() is None
    with pytest.raises(ValueError, match=f"session {SID} serves live coins, not --coins seeded"):
        SV.main([*argv, "--coins", "seeded"])
    for env in (True, False):
        if not env:
            for name in (CO.CHALLENGER_ENV, CO.SESSION_ENV, CO.SECRET_ENV):
                monkeypatch.delenv(name)
        with pytest.raises(ValueError, match=r"--gate research@gate: served_h1's own gated window is superseded; .*"
                                             r"integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py \(#3053\)"):
            SV.main([*argv, "--gate", "research@gate"])


def test_the_gates_picks_follow_the_draw_coin_and_its_root():
    SV = _load("serve")
    c = CO.open_coins(None, EPOCH, ROUNDS, random.Random("prover").randbytes).commitment
    a, b = bytes(64), bytes([1]) * 64
    picks = SV.gate_rng(c, a, b).random()
    assert picks == SV.gate_rng(c, a, b).random() and picks not in (SV.gate_rng(c, b, b).random(), SV.gate_rng(c, a, a).random())
