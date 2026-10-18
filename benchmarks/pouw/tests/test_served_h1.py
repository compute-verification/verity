"""served_h1's side computation (`served_h1/receipts.py`) is `h1tile.reference`'s receipt, unit by unit: on random s8 calls at
K = 64 and 128 (a partial row block left uncovered), and saturated operands at K = 64; and each receipt reads the unit's
position, salt and roots and every operand byte of its tile.  A call's receipt tree (`served_h1/commit.py`) is the same built in
pool tasks or leaf by leaf, opens a unit's leaf, and refuses a flipped bit or another call's salt label.  Under a call's admit
bits (`served_h1/credit.py`) a dead unit's receipt is the canonical empty receipt and a live unit's the reference's, and the
tally's limbs are 16-bit.  A call's A table is AROW's row table, a tile's rows open against it and the step's A root, and
A' = -b is refused (CMP-38)."""
from __future__ import annotations

import importlib.util
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
    bits = CR.admit(CM.s8_rows(call.a.numpy(), scales.numpy()))
    assert list(bits) == [True] * 5 + [False] * 59
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
    tabs = [CM.a_table(doms[w], key, f"a/0/{w}", calls[w].a.numpy(), scales) for w in range(3)]
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
    forged = CM.a_table(doms[w], key, f"a/0/{w}", q, scales).root
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
    assert list(CR.admit(rows)) == [True, True, True] and CR.scale_only(rows) == [1]
    calls = [{"step": s, "site": site, "n": n, "k": k} for s in range(2) for site, n, k in (("a", 64, 32), ("b", 32, 64))]
    tl = CR.tally(calls, [5, 1, 3, 0], 2, ("a", "b"))
    assert tl["step_site"] == [[5, 1], [3, 0]] and tl["site"] == {"a": 8, "b": 1}
    assert tl["rows"] == 9 and tl["macs"] == 8 * 64 * 32 + 1 * 32 * 64
    pub = CR.published(tl["macs"])
    assert 0 < pub["pub_macs"] <= tl["macs"]
    with pytest.raises(ValueError, match="16-bit"):
        CR.tally(calls, [1 << 16, 0, 0, 0], 2, ("a", "b"))
