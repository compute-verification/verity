"""REDTEAM phase 16, Target 1 (the live report's new 'at serve time' label, NET-E2E's PR stacked on #3841,
cursor/net-e2e-serve-time-061a): the decided rule is "registered before its window's draw AND no later than the next
window's close, with the lag in seconds printed" (Daniel, 19:25Z). `chain.held_to`/`served_commit` read the draw half
from the arm's `advice_before_draw` (the a5 stream's receipt that it had the registration before any coin — verifier
attested, never the job's say, which closed REDTEAM phase 13 target 1) and the freshness half from the registration
file's time against `closes`' next-window close. These cases pin the label's edges, which the bound-arm test doesn't
reach: a missing draw receipt and a draw that followed the registration both keep the label off (unknown and
after-serving), and the last Commit window can never earn it for want of a next close. A confirmation that the gate is
`before_draw`, with the last-window gap noted for NET-E2E."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from test_chain import _endpoint, _serve_arm, chain


def _served(tmp_path):
    """A fresh endpoint copy and its `served_advice` (its Commit windows 1 and 2), so the registration times can be set."""
    ep = _endpoint(tmp_path)
    recs = chain._jsonl(ep / "advice_trace.jsonl")
    first, closed = chain._first_steps(recs), chain.closes(recs)
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  json.loads((ep / "io.json").read_text())["requests"], 60, [128001])
    row = {x["window"]: x for x in b["commit_windows"]}
    return ep, info, b, row, closed


def _registered(ep, info, b, d, w, registered):
    """`prove_path`'s binding of the one arm ``d`` for window ``w``, its `registered` block."""
    got = chain.prove_path([d], info, b["bound"], b, ep, registered)
    return got["arms"][f"w{w} served"]["commit"]["registered"]


def test_without_a_draw_receipt_serve_time_is_unknown_not_assumed(tmp_path):
    """The freshness half alone is not enough: an arm whose ranks carry no `advice_before_draw` leaves `before_draw`
    None, so even a registration comfortably before the next window's close reads `at_serve_time` None (unknown), not
    True. The label never assumes the draw order it hasn't been shown."""
    ep, info, b, row, closed = _served(tmp_path)
    d = tmp_path / "w1-noreceipt"
    _serve_arm(d, "served", 1, row[1]["v0_bits"], before_draw=None)
    os.utime(ep / "advice/w1/registration.json", (closed[2] - 5.0,) * 2)
    g = _registered(ep, info, b, d, 1, {"1": {"by": "vllm"}})
    assert g["before_draw"] is None and g["at_serve_time"] is None
    assert g["before_next_close_s"] is not None and g["before_next_close_s"] > 0


def test_a_registration_after_its_draw_is_after_serving_whatever_its_freshness(tmp_path):
    """The draw half dominates: when `advice_before_draw` is False (the draw followed the registration the stream
    received), `at_serve_time` is False even with the registration set well before the next window's close, so a late
    registration can't buy the label with a fresh-looking file time."""
    ep, info, b, row, closed = _served(tmp_path)
    d = tmp_path / "w1-afterdraw"
    _serve_arm(d, "served", 1, row[1]["v0_bits"], before_draw=False)
    os.utime(ep / "advice/w1/registration.json", (closed[2] - 5.0,) * 2)
    g = _registered(ep, info, b, d, 1, {"1": {"by": "vllm"}})
    assert g["before_draw"] is False and g["at_serve_time"] is False
    assert g["before_next_close_s"] is not None and g["before_next_close_s"] > 0


def test_the_last_commit_window_can_never_be_at_serve_time_for_want_of_a_next_close(tmp_path):
    """An edge of the "next window's close" rule: the last Commit window in the endpoint's trace (here w2, `closes` has
    no w3) has `next_closed` None, so `at_serve_time` stays None however early and before-the-draw its registration is.
    Its own-close lag is still reported. The final window of every served run is thus un-creditable as at serve time —
    conservative (never falsely True), but a gap worth NET-E2E's eye."""
    ep, info, b, row, closed = _served(tmp_path)
    assert closed.get(3) is None and 2 in row
    d = tmp_path / "w2-last"
    _serve_arm(d, "served", 2, row[2]["v0_bits"], before_draw=True)
    os.utime(ep / "advice/w2/registration.json", (closed[2] - 2.0,) * 2)
    g = _registered(ep, info, b, d, 2, {"2": {"by": "vllm"}})
    assert g["before_draw"] is True and g["next_closed"] is None and g["before_next_close_s"] is None
    assert g["at_serve_time"] is None and g["after_close_s"] == -2.0
