"""served_h1's side computation (`served_h1/receipts.py`) is `h1tile.reference`'s receipt, unit by unit: on random s8 calls at
K = 64 and 128 (a partial row block left uncovered), and saturated operands at K = 64; and each receipt reads the unit's
position, salt and roots and every operand byte of its tile.  A call's receipt tree (`served_h1/commit.py`) is the same built in
pool tasks or leaf by leaf, opens a unit's leaf, and refuses a flipped bit or another call's salt label.  Under a call's admit
bits (`served_h1/credit.py`), from the committed request queue, a dead unit's receipt is the canonical empty receipt and a live
unit's the reference's, a junk row in a slot the queue leaves unassigned is refused and credits nothing, so is the canonical
empty row in a slot the queue selects (RT-ALGO-4), and the tally's limbs are 16-bit.  A call's A table is AROW's row table, a tile's rows open against it and the step's A root, and
A' = -b is refused (CMP-38).  A linear's B table is the same format over its weight rows, under the weights' root; the
scheduler's record gives the request queue and its commitment."""
from __future__ import annotations

import importlib.util
import random
import sys
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


RC, CM, CR = _load("receipts"), _load("commit"), _load("credit")
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
        assert want == RC.reference(call, SALT, 9, RA, RB, u)[1]
        ungated = RC.reference(call, SALT, 9, RA, RB, u, gated=False)[1]
        assert (want == ungated) if u in live else (not any(want) and any(ungated))


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
    block is dead to credit too and the tally counts none of its rows, and it refuses every one of them (`hollow`).  A row
    of zero bytes under a nonzero scale is refused (`scale_only`), and a selected row of nonzero bytes under scale 0 is
    not admitted and so refused as junk: in a window none of these refuse, a selected slot is admitted iff its bytes are
    nonzero."""
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


def _entropy(seed: str):
    g = random.Random(seed)
    return lambda n: bytes(g.getrandbits(8) for _ in range(n))


def test_the_gate_draws_from_the_challengers_draw_coin():
    """RT-INT-3 (note:20261019T0528Z-draft-rt-int): the gate's units are the epoch's draw coin's, opened once after the
    window's receipts root and checked as the verifier checks it, never the prover's own entropy: subset:K over the
    window's unit positions (W2's sampler), the same for the same coin and another for another; a coin opened after
    another root is refused, a second open never redraws, and the seeded epoch coins' draw coin serves a seeded window."""
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.sampled_proofs import challenger as Ch

    SV = _load("serve")
    calls = [{"base": 0, "units": 4, "receipt_root": "aa" * 64}, {"base": 12, "units": 6, "receipt_root": "cc" * 64},
             {"base": 4, "units": 8, "receipt_root": "bb" * 64}]
    root = SV.receipts_root("w", calls)
    assert root != SV.receipts_root("w", calls[:2]) and root != SV.receipts_root("v", calls)
    population, at = SV.call_units(calls, [0, 3, 4, 11, 12, 17])
    assert population == 18 and at == [(0, 0), (0, 3), (2, 0), (2, 7), (1, 0), (1, 5)]
    with pytest.raises(ValueError, match="tile"):
        SV.call_units(calls[:2], [])
    lch = Wd.live_challenger(1, 3, _entropy("a"))
    draw = SV.open_draw(lch, root)
    assert draw["reason"] is None and draw["after"] == root.hex() and draw["opening"]["label"] == Wd.DRAW
    units, rng = SV.gate_draw(draw, 5, population)
    assert len(units) == 5 and units == sorted(set(units)) and 0 <= units[0] and units[-1] < population
    again, rng2 = SV.gate_draw(draw, 5, population)
    assert again == units and rng.random() == rng2.random()
    assert SV.gate_draw(SV.open_draw(Wd.live_challenger(1, 3, _entropy("b")), root), 5, population)[0] != units
    with pytest.raises(Ch.CoinError, match="never redraws"):
        SV.open_draw(lch, root)
    late = Ch.live_reason(lch.commitment, Ch.LiveOpening.from_json(draw["opening"]), Wd.DRAW, 0,
                          SV.receipts_root("w", calls[:2]))
    assert late is not None
    with pytest.raises(ValueError, match="draw coin"):
        SV.gate_draw(dict(draw, reason=late), 5, population)
    from verity.core.protocols.pouw.coins import Challenger
    seeded = SV.open_draw(Challenger(1, _entropy("c")), root)
    assert seeded["reason"] is None and len(SV.gate_draw(seeded, 5, population)[0]) == 5


def test_coins_default_to_live():
    """RT-INT-3: SERVE-H1 serves under live coins unless told seeded, and the gate draws W2's 8 units by default."""
    SV = _load("serve")
    a = SV.parser().parse_args(["--out", "o", "--at", "2026-10-19T00:00:00Z", "--by", "x"])
    assert a.coins == "live" and a.gate_draw == 8
    assert SV.parser().parse_args(["--out", "o", "--at", "t", "--by", "x", "--coins", "seeded"]).coins == "seeded"


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
