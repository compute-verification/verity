"""The gated lane (`h1zy.PouwH1LaneZy_v1`) against option A's partials, byte for byte (`served_h1/lane_refs.py`):
every leaf's `partials.lane_partials` equals `h1tile.lane_partials(..., gated=admit)` and `h1zy.lane_partial` lane by
lane, over live and dead leaves and leaves whose admitted rows lie outside some lanes' rows (m = 32 r + j, 0 < j < 8:
lanes g >= j hold no admitted row and are ungated all the same), and the references' XOR is the served receipt.  A
mismatch is reported by leaf, lane and word.  `lane_refs.served` (the gate's check under DITHER_LANE_REFS) runs on a
capture of the gate's shape, checked in place and through a saved capture as after serving.  Where PARTIALS_LIB names an
h1u build and CUDA is up, the gpu's (`kh_lanes`) rows too, at m = 2 and 3 (job.sh's partials bench).  The cpu tests'
K run from 128 to 512, since the references take any K (`h1zy.combine` stages the combine from K = 384, `h1zy.LEAST_K`);
the gpu test's are 512 and 4608."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_h1"))
import lane_refs as LR  # noqa: E402
import partials as PT  # noqa: E402

RC = PT.RC
COM = types.SimpleNamespace(root=bytes(range(7, 71)), epoch=3)
COIN, ROOT = bytes(range(100, 164)), bytes(range(30, 94))
T = 64
NK = [(64, 256), (32, 512)]
STEPS = {3: 35, 4: 32, 5: 39}   # row block 1: rows 0..2 admitted, dead, rows 0..6 admitted


def _s8(shape, seed: int) -> torch.Tensor:
    return torch.randint(-128, 128, shape, generator=torch.Generator().manual_seed(seed), dtype=torch.int8)


def _window(nk, steps, dev="cpu", u_step=0) -> "PT.Window":
    units = [(T // 32) * (n // 32) for n, _ in nk]
    win = PT.Window([_s8((n, k), 10 + w).to(dev) for w, (n, k) in enumerate(nk)],
                    [bytes([w + 1]) * 64 for w in range(len(nk))], [sum(units[:w]) for w in range(len(nk))],
                    u_step + sum(units))
    for s, m in steps.items():
        a = [_s8((T, k), 100 * s + w).to(dev) for w, (_, k) in enumerate(nk)]
        win.steps[s] = PT.Step.from_coin(a, [bytes([s, w]) * 32 for w in range(len(nk))], m, COM, COIN, s, ROOT)
    return win


def _receipts(win, s: int, w: int) -> np.ndarray:
    st = win.steps[s]
    got, _ = RC.gated_receipts(RC.Call(st.a[w].cpu(), win.b[w].cpu()), np.arange(T) < st.m, st.salt,
                               s * win.u_step + win.offs[w], st.a_roots[w], win.b_roots[w])
    return got


def _leaves(win):
    """Each step's row block 0 and 1 on both calls, one column block each."""
    return [PT.Leaf(s, w, rb * (n // 32) + (s + w) % (n // 32)) for s in STEPS for w, (n, _) in enumerate(NK)
            for rb in (0, 1)]


def test_partials_are_serve_regs_lane_rows():
    win = _window(NK, STEPS)
    leaves = _leaves(win)
    rcpt = {x: _receipts(win, x.step, x.call)[x.unit] for x in leaves}
    doc = LR.compare(win, leaves, receipts=rcpt)
    assert doc["ok"], doc["first_mismatches"]
    assert doc["reference_xor_vs_receipt_bad"] == []
    # row block 1 at step 4 is dead; at steps 3 and 5 it is live with 3 and 7 rows admitted, so 4 (8 - 3) and 4 (8 - 7)
    # of its lanes (g >= j) hold no admitted row
    assert (doc["leaves"], doc["live"], doc["dead"], doc["partly_admitted"]) == (12, 10, 2, 4)
    assert doc["lanes_without_an_admitted_row"] == 2 * (20 + 4)
    for x in leaves:
        if (x.step, x.unit // (NK[x.call][0] // 32)) == (4, 1):
            assert not PT.lane_partials(win, x).any()


def test_a_mismatch_names_its_leaf_lane_and_word(monkeypatch):
    win = _window(NK, {3: 35})
    leaves = [PT.Leaf(3, 1, 1), PT.Leaf(3, 0, 2)]
    honest = PT.lane_partials_many

    def flipped(*a, **k):
        out = [p.copy() for p in honest(*a, **k)]
        out[0][22, 17] ^= 1 << 9
        return out

    monkeypatch.setattr(PT, "lane_partials_many", flipped)
    doc = LR.compare(win, leaves)
    assert not doc["ok"] and doc["mismatches"] == 1
    bad = doc["first_mismatches"][0]
    assert (bad["leaf"], bad["lane"], bad["word"], bad["row"], bad["live"]) == ([3, 1, 1], 22, 17, "cpu", True)
    # lane 22 = 4 * 5 + 2 holds rows 5 + 8 s, none of row block 1's 3 admitted rows
    assert bad["admitted"] == 3 and bad["lane_has_admitted_row"] is False


def _gate_capture():
    """A capture shaped as dither_one's gate's: the calls' receipts flat in plan order."""
    nk = [(64, 128), (32, 256), (32, 128), (64, 128)]  # the partly steps' calls: 0, 2, the widest K's first (1), the last
    win = _window(nk, {7: 32})
    st = win.steps[7]
    units = [(T // 32) * (n // 32) for n, _ in nk]
    words = [win.words(w) for w in range(len(nk))]
    offs, o = [], 0
    for u, nw in zip(units, words):
        offs.append((o, u, nw))
        o += u * nw
    flat = torch.from_numpy(np.concatenate([_receipts(win, 7, w).reshape(-1) for w in range(len(nk))]).view(np.int32))
    dwin = types.SimpleNamespace(t=T, L=len(nk), wt=win.b, offs=win.offs, u_step=win.u_step, nk=nk,
                                 plan=types.SimpleNamespace(offs=offs))
    g = {"s": 7, "m": 32, "a": st.a, "a_roots": st.a_roots, "b_roots": win.b_roots, "receipts": flat}
    return dwin, g, st.salt


def _served_ok(doc):
    gs = doc["groups"]
    assert doc["ok"] and doc["mismatches"] == 0, doc["first_mismatches"]
    assert (gs["real"]["live"], gs["real"]["dead"], gs["real"]["reference_xor_vs_receipt_bad"]) == (4, 4, [])
    assert sorted(gs) == ["partly:33", "partly:35", "partly:39", "real"]
    assert [gs[f"partly:{m}"]["lanes_without_an_admitted_row"] for m in (33, 35, 39)] == [4 * 28, 4 * 20, 4 * 4]
    assert all(gs[g]["partly_admitted"] == gs[g]["leaves"] == 4 for g in gs if g != "real")


def test_served_on_a_gate_shaped_capture():
    """`served`'s plumbing: every call sampled live and dead with its committed receipt, and the step's own operands at
    m = 33, 35, 39, checked here."""
    doc = LR.served(*_gate_capture())
    assert doc["here_ok"] and doc["leaves"] == 20
    _served_ok(doc)


def test_a_saved_capture_is_checked_after_serving(tmp_path):
    """`served` with a path saves the records (the gate's daemon process can't spawn the references' pool), and
    `lane_refs.py CAPTURE` checks them after serving, the same verdict."""
    path = tmp_path / "capture.npz"
    here = LR.served(*_gate_capture(), path=path)
    assert here["ok"] and here["captured"] == str(path) and "groups" not in here
    recs = LR.load(path)
    assert len(recs) == 20 and all(r["a"].dtype == np.uint8 and r["a"].shape == (32, r["k"]) for r in recs)
    assert LR.main([str(path), "--workers", "1", "--out", str(tmp_path / "doc.json")]) == 0
    _served_ok(json.loads((tmp_path / "doc.json").read_text()))


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("PARTIALS_LIB")),
                    reason="the gpu backend needs CUDA and PARTIALS_LIB (an h1u.cu build with kh_lanes)")
def test_gpu_rows_are_serve_regs_lane_rows():
    kern = PT.Kernel(os.environ["PARTIALS_LIB"])
    nk = [(64, 512), (32, 4608)]  # m = 2, 3: both of kh_lanes' instantiations
    win = _window(nk, STEPS, dev="cuda", u_step=1 << 40)
    leaves = [PT.Leaf(s, w, rb * (n // 32) + (s + w) % (n // 32)) for s in STEPS for w, (n, _) in enumerate(nk)
              for rb in (0, 1)]
    doc = LR.compare(win, leaves, kern)
    assert doc["ok"], doc["first_mismatches"]
    assert doc["backends"] == ["cpu", "gpu", "lane_partial"] and (doc["dead"], doc["partly_admitted"]) == (2, 4)
