"""The filler scorer's arithmetic (benchmarks/pouw/filler/scoring.py) on synthetic driver rows: the paired slowdown, the
filler's rate over its windows, coverage with the proved and provisional certified fractions, SCORE as the time-weighted
mean, the gates (slowdown, receipts, DRAM with its counter's control, clocks) and the arms' placement policies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FILLER = Path(__file__).resolve().parents[1] / "filler"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("arms")
S = _load("scoring")
share = S.share


def row(kind, b, l, place, *, alone_ms, arm_ms, rounds=4, mhz=2000.0, watt=300.0, chunks=10, window_ms=None,
        chunk_macs=1e9, alone_rate=1e12, rec_bad=0, fill_sms=0):
    """A `dscore` row: every alone rep `alone_ms`, every arm rep `arm_ms`, `chunks` filler chunks a window."""
    fill = place != "none"
    blocks = []
    for _ in range(rounds):
        w = window_ms if window_ms is not None else 3 * arm_ms
        arm = {"window_ms": w, "reps": [[arm_ms, mhz, watt]] * 3}
        if fill:
            arm["chunks"] = chunks
        blocks.append({"order": "alone,arm", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, mhz, watt]] * 3},
                       "arm": arm})
    return {"mode": "dscore", "kind": kind, "B": b, "L": l, "place": place, "fill_sms": fill_sms, "chunk_macs": chunk_macs,
            "alone_fill": {"mac_per_s": alone_rate if fill else 0.0}, "rec_checked": 4 * rounds if fill else 0,
            "rec_bad": rec_bad, "setup_ms": 1.0, "spread": {}, "kv_wrapped": False, "blocks": blocks}


def all_points(slow=1.02, place="split", **kw):
    out = {}
    for ph, b, l in S.points():
        alone = 10.0 if ph == "decode" else 160.0
        out[(ph, b, l)] = S.point_stats(row("step" if ph == "decode" else "prefill", b, l, place, alone_ms=alone,
                                            arm_ms=alone * slow, **kw))
    return out


def test_paired_slowdown_and_rate():
    r = row("step", 32, 2048, "split", alone_ms=10.0, arm_ms=10.4, chunks=30, window_ms=31.2, chunk_macs=2e9)
    r["blocks"][1]["arm"]["reps"] = [[11.0, 2000.0, 300.0]] * 3
    st = S.point_stats(r)
    assert st["slowdown_rounds"] == pytest.approx([1.04, 1.10, 1.04, 1.04])
    assert st["slowdown"] == pytest.approx(1.04)
    assert st["fill_rate"] == pytest.approx(30 * 2e9 / 31.2e-3)
    assert st["fill_rate_cv"] == pytest.approx(0.0)
    assert st["beside_over_alone"] == pytest.approx(30 * 2e9 / 31.2e-3 / 1e12)


def test_decode_entry_coverage_uses_proved_and_provisional_fractions():
    st = S.point_stats(row("step", 32, 2048, "split", alone_ms=10.0, arm_ms=10.0, chunks=10, window_ms=30.0,
                           chunk_macs=1e9))
    e = S.coverage(S.decode_entry(st, 32, 2048))
    t1 = 10.0 + max(1.09, 32 * 38.37 / 8192)
    n = 32 * (share.LIN_TOK + share.LM_TOK + share.ATT_TOK_L * 2048)
    f = 10 * 1e9 / 30e-3 * t1 * 1e-3
    c = 1024 * 188 * 2000e6 * t1 * 1e-3
    assert e["coverage"] == pytest.approx((0.225 * n + 0.99 * f) / c)
    assert e["nominal_coverage"] == pytest.approx((n + f) / c)
    assert e["coverage_at_target"] == pytest.approx(0.99 * (n + f) / c)
    assert e["time"] == pytest.approx(1.0)


def test_score_is_time_weighted_and_missing_entries_void_it():
    stats = all_points(slow=1.03)
    sc = S.score(stats)
    w = [e["time"] for e in sc["entries"]]
    assert len(sc["entries"]) == 16 and not sc["missing"]
    assert sc["SCORE"] == pytest.approx(sum(e["coverage"] * x for e, x in zip(sc["entries"], w)) / sum(w))
    assert sc["min"]["coverage"] == min(e["coverage"] for e in sc["entries"])
    assert sc["flags"] == ["filler certified fraction 0.99 is provisional"]
    del stats[("decode", 132, 5120)]
    sc = S.score(stats)
    assert "SCORE" not in sc and sc["missing"] == ["reasoning 1K/8192 B 132"]


def test_mix_entry_per_request():
    pre = S.point_stats(row("prefill", 8192, 0, "none", alone_ms=160.0, arm_ms=160.0))
    dec = S.point_stats(row("step", 48, 16384, "split", alone_ms=30.0, arm_ms=31.0, chunks=0))
    e = S.mix_entry(pre, dec, 24576, 512, 48, 16384)
    tp = 160.0 * 3 + share.PRE_OTHER_MS_TOK * 24576 + share.FA2_MS_8192 * share.causal(24576) / share.causal(8192)
    td0 = 512 / 48 * (30.0 + 1.09)
    td1 = 512 / 48 * (31.0 + 1.09 * (31.0 / 30.0))
    assert e["t0_ms"] == pytest.approx(tp + td0)
    assert e["t1_ms"] == pytest.approx(tp + td1)
    assert e["slowdowns"] == {"prefill": 1.0, "decode": pytest.approx(31.0 / 30.0)}
    assert S.mix_context(24576, 512, 48) == 16384 and S.mix_context(8192, 256, 144) == 5120


def ok_gates(stats, sc, **kw):
    base = {"has_filler": True, "sgate": {"ok": True}, "dgate": [{"ok": True}],
            "dram": {"ok": True}, "clocks": {"ok": True}, "honest": {"ok": True}}
    return S.gates(stats, sc, **{**base, **kw})


def test_slowdown_and_receipt_gates():
    stats = all_points(slow=1.04)
    assert ok_gates(stats, S.score(stats))["ok"]
    slow = all_points(slow=1.06)
    g = ok_gates(slow, S.score(slow))
    assert not g["ok"] and not g["slowdown"]["ok"] and g["slowdown"]["worst"] == pytest.approx(1.06)
    bad = all_points(slow=1.0, rec_bad=1)
    assert not ok_gates(bad, S.score(bad))["receipts"]["ok"]
    assert not ok_gates(stats, S.score(stats), sgate={"ok": False})["receipts"]["ok"]
    assert not ok_gates(stats, S.score(stats), sgate=None)["receipts"]["ok"]


def ncu_csv(kernel, rows):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
    body = [f'"{i}","{kernel}(args)","{m}","{u}","{v}"' for i, (m, u, v) in enumerate(rows)]
    return "==PROF== Connected\n" + "\n".join([head, *body]) + "\n"


def test_dram_gate_validates_its_counter():
    ctl = ncu_csv("bw_read", [("dram__bytes.sum", "Gbyte", "1.08"), ("gpu__time_duration.sum", "msecond", "0.72")])
    fil = ncu_csv("void dfill_k<4>", [("dram__bytes.sum", "Kbyte", "200"), ("gpu__time_duration.sum", "msecond", "1.12")])
    g = S.dram_gate(fil, ctl)
    assert g["ok"] and g["counter_valid"]
    assert g["hbm_bytes_per_s"] == pytest.approx(1.08e9 / 0.72e-3)
    assert g["share_of_hbm"] == pytest.approx((200e3 / 1.12e-3) / (1.08e9 / 0.72e-3))
    dead = ncu_csv("bw_read", [("dram__bytes.sum", "byte", "0"), ("gpu__time_duration.sum", "msecond", "0.72")])
    assert not S.dram_gate(fil, dead)["ok"]
    heavy = ncu_csv("dfill_k", [("dram__bytes.sum", "Mbyte", "100"), ("gpu__time_duration.sum", "msecond", "1.0")])
    assert not S.dram_gate(heavy, ctl)["ok"]
    assert S.dram_gate(None, None)["ok"]


def test_clock_gate():
    stats = all_points()
    smi = S.parse_smi("2092, 3090, 410.5, 600.00, 2430, 2430, 0x0000000000000000\n")
    assert S.clock_gate(stats, smi)["ok"]
    assert not S.clock_gate(stats, S.parse_smi("2092, 3090, 410.5, 600.00, 2100, 2430, 0x0\n"))["ok"]
    assert not S.clock_gate(stats, [])["ok"]


def test_arms():
    v1, idle = A.load("fill-v1"), A.load("idle")
    assert A.placed(idle, "decode", 32, 2048) == A.NONE
    assert A.placed(v1, "prefill", 8192, 0) == A.NONE
    assert A.placed(v1, "decode", 1, 2048) == A.Placement("split", 108)
    assert A.placed(v1, "decode", 32, 2048).fill_sms == 92  # FILL's rule: (32, 1K), the longest L ≤ 2K at B 32
    assert A.placed(v1, "decode", 144, 5120).fill_sms == 60
    assert A.placed(v1, "decode", 8, 16384).fill_sms == 124  # B 32's longest L ≤ 16K: (32, 8K)
    assert S.honesty(v1, all_points(), S.points())["ok"]
    with pytest.raises(A.ArmError):
        A.Placement("split", 0)
    with pytest.raises(A.ArmError):
        A.load("no-such-arm")
