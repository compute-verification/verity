"""The filler scorer's arithmetic (benchmarks/pouw/filler/scoring.py) on synthetic driver rows: the paired slowdown, the
filler's rate over its windows, coverage with the proved and provisional certified fractions, SCORE as the time-weighted
mean, the gates (slowdown, receipts, anti-replay, honesty against the interleaved standalone reference and its positive
control, DRAM with its counter's control, clocks), the settled power and the arms' placement policies."""

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
        chunk_macs=1e9, alone_rate=1e12, rec_bad=0, fill_sms=0, solo=None):
    """A `dscore` row: every alone rep `alone_ms`, every arm rep `arm_ms`, `chunks` filler chunks a window, and after each
    arm block a standalone block of the same window with `solo[r]` chunks (default `chunks`)."""
    fill = place != "none"
    blocks = []
    for r in range(rounds):
        w = window_ms if window_ms is not None else 3 * arm_ms
        arm = {"window_ms": w, "reps": [[arm_ms, mhz, watt]] * 3}
        blk = {"order": "alone,arm,solo", "alone": {"window_ms": 3 * alone_ms, "reps": [[alone_ms, mhz, watt]] * 3},
               "arm": arm}
        if fill:
            arm["chunks"] = chunks
            blk["solo"] = {"window_ms": w, "chunks": chunks if solo is None else solo[r], "reps": []}
        blocks.append(blk)
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
    assert st["beside_over_alone"] == pytest.approx(1.0) and st["too_good_limit"] == 1 + S.TOO_GOOD_FLOOR
    assert st["beside_over_calibration"] == pytest.approx(30 * 2e9 / 31.2e-3 / 1e12)


def test_rate_over_completion_spans_and_the_peak_bound():
    def spanned(**kw):
        r = row("step", 1, 2048, "split", alone_ms=6.0, arm_ms=6.3, chunks=19, window_ms=18.9, fill_sms=100, **kw)
        for b in r["blocks"]:
            b["arm"]["span_ms"] = b["solo"]["span_ms"] = 18.0
        return S.point_stats(r)

    st = spanned(chunk_macs=1e9)
    assert st["fill_rate"] == pytest.approx(18 * 1e9 / 18.0e-3)
    assert st["of_peak"] == pytest.approx(st["fill_rate"] / (S.PER_SM * 100 * 2000.0e6))
    v1 = A.load("fill-v1")
    assert S.honesty(v1, {("decode", 1, 2048): st}, [])["ok"]
    hot = S.honesty(v1, {("decode", 1, 2048): spanned(chunk_macs=3e11, alone_rate=1e15)}, [])
    assert not hot["ok"] and hot["over_peak"] and not hot["too_good"]


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
    assert S.score(all_points(slow=1.0, place="none"))["flags"] == []
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


def test_too_good_against_the_interleaved_reference_and_its_control():
    v1 = A.load("fill-v1")

    def st(chunks=200, solo=(200, 201, 199, 200), rounds=4):
        return S.point_stats(row("step", 32, 8192, "split", alone_ms=20.0, arm_ms=20.8, chunks=chunks, rounds=rounds,
                                 window_ms=62.4, fill_sms=124, solo=list(solo)))

    honest = st()
    assert honest["fill_rate_alone"] == pytest.approx(200 * 1e9 / 62.4e-3)
    assert honest["solo_spread"] == pytest.approx(2 / 200) and honest["too_good_limit"] == pytest.approx(1.01)
    # the one calibration (alone_rate 1e12) would call this 3.2x too good; the interleaved reference doesn't
    assert honest["beside_over_calibration"] > 3 and honest["beside_over_alone"] == pytest.approx(1.0)
    pt = ("decode", 32, 8192)
    assert S.honesty(v1, {pt: honest}, [])["ok"]
    fast = S.honesty(v1, {pt: st(chunks=213)}, [])
    assert not fast["ok"] and fast["too_good"]
    # a noisy reference raises the limit to its spread: 1.065 against a 10% spread passes
    noisy = st(chunks=213, solo=(190, 200, 210, 200))
    assert noisy["too_good_limit"] == pytest.approx(1.1) and S.honesty(v1, {pt: noisy}, [])["ok"]
    bare = row("step", 32, 8192, "split", alone_ms=20.0, arm_ms=20.8, chunks=200, window_ms=62.4, fill_sms=124)
    for b in bare["blocks"]:
        del b["solo"]
    old = S.point_stats(bare)
    assert old["too_good_limit"] == S.TOO_GOOD and old["solo_rates"] == []
    assert S.honesty(v1, {pt: old}, [])["no_reference"] == ["decode 32/8192"]

    def ctl(v):
        return {**v, "at": "decode 32/8192", "cheat_S": S.cheat_s(2048), "S": 2048}
    caught = S.honesty(v1, {pt: honest}, [], control=ctl(st(chunks=213)), has_filler=True)
    assert caught["ok"] and caught["control"]["caught"] and caught["control"]["cheat_S"] == 1920
    blind = S.honesty(v1, {pt: honest}, [], control=ctl(noisy), has_filler=True)
    assert not blind["ok"] and not blind["control"]["caught"]
    assert not S.honesty(v1, {pt: honest}, [], control=None, has_filler=True)["ok"]
    assert S.honesty(v1, {}, [], control=None, has_filler=False)["control"] is None
    assert [S.cheat_s(x) for x in (2048, 256, 64, 16)] == [1920, 240, 56, 8]


def test_settled_power_is_the_secondary_reading():
    r = row("step", 1, 2048, "split", alone_ms=6.0, arm_ms=6.2, watt=106.5, fill_sms=108)
    r["power"] = {"settle_ms": 1000, "meas_ms": 1500, "alone": {"watt": 250.0}, "arm": {"watt": 410.0},
                  "filler_alone": {"watt": 205.0}}
    stats = {("decode", 1, 2048): S.point_stats(r), ("decode", 32, 2048): S.point_stats(
        row("step", 32, 2048, "none", alone_ms=10.0, arm_ms=10.0))}
    sec = S.secondary(stats, S.score(stats))
    assert sec["power"]["decode 1/2048"] == {"alone": 250.0, "arm": 410.0, "filler_alone": 205.0}
    assert sec["power"]["decode 32/2048"] == {"alone": 300.0, "arm": 300.0}  # a row before the windows: per-rep medians
    assert sec["power_per_rep"]["decode 1/2048"] == [106.5, 106.5]


def ok_gates(stats, sc, **kw):
    base = {"has_filler": True, "sgate": {"ok": True}, "dgate": [{"ok": True}],
            "dram": {"ok": True}, "clocks": {"ok": True}, "honest": {"ok": True}, "replay": {"ok": True}}
    return S.gates(stats, sc, **{**base, **kw})


def test_dseed_is_dcaps():
    # dcap.cu's dseed on these inputs, from a host build of dcap.cu
    assert S.dseed(0x0123456789ABCDEF, 0xFEDCBA9876543210, 0) == 0x4D3A3F76002DE7A2
    assert S.dseed(0x00761986AA468DD5, 0xE62E90F303F1F770, 1 << 32) == 0x29AD174F1CC23ABE
    assert S.dseed((1 << 64) - 1, 1, (19 * 9 + 8) << 32 | 12345) == 0x6D8721960565D78D


def a_round(p, slots, slot, per=1000, chunks=7, tag=0):
    rid = p * slots + slot
    lo = rid << S.RANGE_LOG2
    c0, c1 = S.smix(2 * rid + 1 + (tag >> 40)), S.smix(2 * rid + 2 + (tag >> 40))
    rd = {"id": rid, "slot": slot, "range": [lo, (rid + 1) << S.RANGE_LOG2], "tiles": [lo, lo + chunks * per],
          "coins": [f"{c0:016x}", f"{c1:016x}"], "seed0": f"{S.dseed(c0, c1, tag | lo):016x}",
          "checked": [lo + (chunks - 1) * per + i * 250 for i in range(4)]}
    return {**rd, "tag": f"{tag:016x}"} if tag else rd


def rounds_rows(points=3, rounds=4, legacy=False, slots=None):
    """dscore rows whose rounds follow the scorer's ranges, each with its own coins: v2's (slots 2·rounds + 3: the
    calibration, the arm blocks, the standalone blocks) or, `legacy`, v1's (slots rounds + 1, no standalone blocks, the
    slot by position)."""
    out = []
    for p in range(points):
        r = row("step", 32, 2048 * (p + 1), "split", alone_ms=10.0, arm_ms=10.2, rounds=rounds)
        r["point"], r["rounds"] = p, rounds
        n = rounds + 1 if legacy else slots or 2 * rounds + 3
        if not legacy:
            r["slots"] = n
        holders = [r["alone_fill"]] + [b["arm"] for b in r["blocks"]]
        if legacy:
            for b in r["blocks"]:
                del b["solo"]
        else:
            holders += [b["solo"] for b in r["blocks"]]
        for s, holder in enumerate(holders):
            holder["round"] = a_round(p, n, s)
            if legacy:
                del holder["round"]["slot"]
        out.append(r)
    return out


def test_replay_gate():
    good = rounds_rows()
    g = S.replay_gate(good)
    assert g["ok"] and g["rounds"] == 3 * 9 and g["tiles"] == 27 * 7000 and not g["why"]
    assert S.replay_gate(good + [row("prefill", 8192, 0, "none", alone_ms=1.0, arm_ms=1.0)])["ok"]
    old = S.replay_gate(rounds_rows(legacy=True))
    assert old["ok"] and old["rounds"] == 3 * 5
    # a second filler's rounds (an arm-side driver's `phase_round`), its ids marked by a tag bit, in slots of its own
    tagged = rounds_rows(slots=4 * 4 + 3)
    for r in tagged:
        for i, b in enumerate(r["blocks"]):
            b["arm"]["phase_round"] = a_round(r["point"], r["slots"], 9 + i, tag=1 << 62)
    assert S.replay_gate(tagged)["ok"] and S.replay_gate(tagged)["rounds"] == 3 * 13
    tagged[0]["blocks"][1]["arm"]["phase_round"]["tag"] = "0" * 16
    assert "not dseed" in S.replay_gate(tagged)["why"]
    assert "slot 8 not in [0, 8)" in S.replay_gate([{**rounds_rows(points=1)[0], "slots": 8}])["why"]
    no_solo_round = rounds_rows()
    no_solo_round[2]["blocks"][0]["solo"].pop("round")
    assert "standalone block 0: no round record" in S.replay_gate(no_solo_round)["why"]

    def broken(edit):
        rs = rounds_rows()
        edit(rs)
        return S.replay_gate(rs)

    def same_coins(rs):
        a, b = rs[0]["blocks"][0]["arm"]["round"], rs[1]["blocks"][2]["arm"]["round"]
        b["coins"], b["seed0"] = a["coins"], f"{S.dseed(*(int(x, 16) for x in a['coins']), b['tiles'][0]):016x}"
    assert broken(same_coins)["coin_repeats"] == 1

    def prover_range(rs):  # a block that numbers its tiles from 0, as FILL v1's pump does every round
        rd = rs[1]["blocks"][0]["arm"]["round"]
        rd["range"], rd["tiles"], rd["checked"] = [0, 1 << 32], [0, 7000], [6000]
        rd["seed0"] = f"{S.dseed(*(int(x, 16) for x in rd['coins']), 0):016x}"
    g = broken(prover_range)
    assert not g["ok"] and "not the scorer's" in g["why"] and g["overlaps"]
    assert "not dseed" in broken(lambda rs: rs[2]["alone_fill"]["round"].update(seed0="0" * 16))["why"]
    assert "outside" in broken(lambda rs: rs[0]["blocks"][1]["arm"]["round"]["checked"].append(0))["why"]
    assert "not at the start" in broken(lambda rs: rs[0]["blocks"][1]["arm"]["round"].update(tiles=[5, 1000]))["why"]
    assert not broken(lambda rs: rs[1].update(point=0))["ok"]
    assert not broken(lambda rs: rs[0]["blocks"][3]["arm"].pop("round"))["ok"]
    assert not S.replay_gate([])["ok"]


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
    assert not S.honesty(v1, all_points(), S.points(), has_filler=True)["ok"]  # no positive control
    with pytest.raises(A.ArmError):
        A.Placement("split", 0)
    with pytest.raises(A.ArmError):
        A.load("no-such-arm")
