"""The filler scorer's arithmetic (benchmarks/pouw/filler/scoring.py) on synthetic driver rows: the paired slowdown, the
filler's rate over its windows, coverage with the proved and provisional certified fractions, SCORE as the time-weighted
mean, the gates (slowdown, receipts, anti-replay, honesty against the interleaved standalone reference and its positive
control, DRAM with its counter's control, clocks), the settled power and the arms' placement policies; filler-scorer/v5's
fold reading, modelled served part and the TP score."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FILLER = Path(__file__).resolve().parents[1]


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
    assert st["beside_over_alone"] == pytest.approx(1.0) and st["too_good_limit"] == pytest.approx(S.TOO_GOOD)
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


def test_fixed_work_is_bounded_on_its_own_sms():
    """An arm-side phase filler's fixed work runs on the step's SMs (`fixed_sms`), not the split filler's: each part is
    bounded on its own SMs, and a row counting fixed work without naming its SMs is bounded on `fill_sms`."""
    peak_sm = S.PER_SM * 2000.0e6

    def phased(fixed_rate, fixed_sms=None):
        r = row("step", 192, 2048, "split", alone_ms=10.0, arm_ms=10.4, chunks=19, window_ms=31.2, fill_sms=92)
        for b in r["blocks"]:
            for k in ("arm", "solo"):
                b[k].update(span_ms=18.0, fixed_ms=31.2, fixed_macs=fixed_rate * 31.2e-3)
        if fixed_sms:
            r["fixed_sms"] = fixed_sms
        return S.point_stats(r)

    split = 18 * 1e9 / 18.0e-3
    st = phased(0.8 * peak_sm * 96, fixed_sms=96)
    assert st["fill_rate"] == pytest.approx(split + 0.8 * peak_sm * 96)
    assert st["of_peak"] == pytest.approx(0.8)
    assert S.honesty(A.load("fill-v1"), {("decode", 192, 2048): st}, [])["over_peak"] == []
    assert phased(1.1 * peak_sm * 96, fixed_sms=96)["of_peak"] == pytest.approx(1.1)
    assert phased(0.8 * peak_sm * 96)["of_peak"] == pytest.approx(0.8 * 96 / 92)


def test_too_good_compares_the_split_alone_with_its_standalone_blocks():
    """Alone, an arm-side phase filler runs at its full rate on the step's SMs; beside serving it gets a share. Pooled with
    the split, that would hide a split 30% too fast (beside ÷ standalone ~0.6), so the too-good check reads the split's
    chunks only."""
    r = row("step", 256, 1024, "split", alone_ms=10.0, arm_ms=10.4, chunks=19, window_ms=31.2, fill_sms=60)
    for b in r["blocks"]:
        b["arm"].update(chunks=1 + round(18 * 1.3), span_ms=18.0, fixed_ms=31.2, fixed_macs=70e12 * 31.2e-3)
        b["solo"].update(span_ms=18.0, fixed_ms=1.0, fixed_macs=180e12 * 1e-3)
    st = S.point_stats(r)
    split, alone = (1 + round(18 * 1.3) - 1) * 1e9 / 18.0e-3, 18 * 1e9 / 18.0e-3
    assert st["fill_rate"] == pytest.approx(split + 70e12)
    assert st["fill_rate_alone"] == pytest.approx(alone)
    assert st["beside_over_alone"] == pytest.approx(split / alone) and st["beside_over_alone"] > 1.25
    assert (split + 70e12) / (alone + 180e12) < 1.0
    assert S.honesty(A.load("fill-v1"), {("decode", 256, 1024): st}, [])["too_good"]


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

    def st(chunks=200, solo=(200, 201, 199, 200, 202, 200)):
        return S.point_stats(row("step", 32, 8192, "split", alone_ms=20.0, arm_ms=20.8, chunks=chunks, rounds=len(solo),
                                 window_ms=62.4, fill_sms=124, solo=list(solo)))

    honest = st()
    assert honest["fill_rate_alone"] == pytest.approx(200 * 1e9 / 62.4e-3)
    assert honest["solo_spread"] == pytest.approx(1 / 200) and honest["too_good_limit"] == pytest.approx(1.02)
    # one standalone block a host hiccup read 20% fast doesn't loosen the limit
    assert st(solo=(200, 201, 199, 200, 240, 200))["too_good_limit"] == pytest.approx(1.02)
    # the one calibration (alone_rate 1e12) would call this 3.2x too good; the interleaved reference doesn't
    assert honest["beside_over_calibration"] > 3 and honest["beside_over_alone"] == pytest.approx(1.0)
    pt = ("decode", 32, 8192)
    assert S.honesty(v1, {pt: honest}, [])["ok"]
    fast = S.honesty(v1, {pt: st(chunks=213)}, [])
    assert not fast["ok"] and fast["too_good"]
    # a noisy reference raises the limit to its spread: 1.065 against a 10% trimmed spread passes
    noisy = st(chunks=213, solo=(185, 190, 200, 200, 210, 215))
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
    # a run whose drivers count a point's slots differently: point 1 at 22 slots meets point 2 at 11
    mixed = rounds_rows(points=2, slots=22) + [{**rounds_rows(points=3)[2]}]
    assert "different slot counts" in S.replay_gate(mixed)["why"] and S.replay_gate(mixed)["overlaps"]
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


def test_margin_per_point():
    """filler-scorer/v3's margin readout: the limit less each point's slowdown, with its worst and best rounds', so a
    point whose rounds reach the limit shows it has none even where its median passes (decode 32/1K, whose serving alone
    reads 7.69 or 7.75 ms by process)."""
    stats = all_points(slow=1.02)
    thin = row("step", 32, 1024, "split", alone_ms=10.0, arm_ms=10.41, fill_sms=92)
    thin["blocks"][0]["arm"]["reps"] = [[10.6, 2000.0, 300.0]] * 3
    stats[("decode", 32, 1024)] = S.point_stats(thin)
    stats[("decode", 8, 16384)] = S.point_stats(row("step", 8, 16384, "split", alone_ms=10.0, arm_ms=10.8, fill_sms=124))
    mg = S.margins(stats)
    m = mg["points"]["decode 32/1024"]
    assert m["margin"] == pytest.approx(0.009) and m["rounds"] == pytest.approx([-0.01, 0.009]) and m["scored"]
    assert mg["points"]["decode 1/2048"]["margin"] == pytest.approx(0.03)
    assert mg["min"] == {"at": "decode 32/1024", "margin": pytest.approx(0.009)} and mg["no_margin"] == ["decode 32/1024"]
    robust = mg["points"]["decode 8/16384"]
    assert not robust["scored"] and robust["margin"] == pytest.approx(-0.03)  # listed, never the minimum
    assert ok_gates(stats, S.score(stats))["slowdown"]["worst"] == pytest.approx(S.LIMIT - mg["min"]["margin"])


def sass(*functions):
    """``cuobjdump -sass`` text: one block a (function name, its instructions), in its line format."""
    out = []
    for name, ins in functions:
        out.append(f"\t\tFunction : {name}")
        out += [f"        /*{16 * i:04x}*/                   {x} ;                      /* 0x000000280c1c7237 */"
                for i, x in enumerate(ins)]
    return "\n".join(out)


DFILL = "_Z7dfill_kILi8EEvmmyiiPy"
#: pfill-k's tile as ptxas compiles it, abridged: fresh s32 words, folded by rotations
K_TILE = ["LOP3.LUT R4, R2, 0x1010101, RZ, 0xc0, !PT", "IMMA.16832.S8.S8 R28, R12.ROW, R40.COL, RZ",
          "SHF.L.W.U32.HI R9, R28, 0x7, R28", "IADD3 R9, R9, R10, RZ", "@P0 BRA 0x80d0", "EXIT"]


E2M1_TILE = ["OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R28, R12, R40, RZ, R112, R112, URZ"] + K_TILE[2:]


def test_kernel_facts_decide_the_filler_entry_with_the_declaration():
    """filler-scorer/v3 reads a filler at Theorem K's 0.9 only when it declares noise K covers and the SASS of the
    scorer's own build shows dense int8 IMMA with s32 words, no FP, one MMA a word at L ≤ 32, and every MMA in the read
    kernel; at Theorem E's 0.225 for declared ±1 with the same checks, the e2m1 tile at L = 64 included; else at the
    proved floor 0: chained when it declares a chain or an MMA accumulates. Never from the declaration alone."""
    f = S.kernel_facts(sass((DFILL, K_TILE), ("_Z5stepv", ["FFMA R1, R2, R3, R4", "EXIT"])))
    assert f["kind"] == "int8" and f["fp_free"] and f["L"] == 32 and f["mma_outside"] == [] and f["functions"] == [DFILL]
    e = S.filler_entry("bits01", f)
    assert (e["applies"], e["active"], e["fraction"], e["failed"]) == ("theorem-k", "theorem-k", 0.9, {})
    assert e["unchecked"] == [*S.UNCHECKED, *S.FILLER_ENTRIES["theorem-k"]["unchecked"]] and len(e["unchecked"]) == 3
    assert all(S.filler_entry(n, f)["applies"] == "theorem-k" for n in ("bitsU", "dither"))
    e = S.filler_entry("pm1", f)  # ±1's coefficients are even, where K is vacuous: E's ¼
    assert (e["applies"], e["fraction"]) == ("theorem-e", 0.225) and list(e["failed"]) == ["theorem-k"]
    assert e["failed"]["theorem-k"][0].startswith("declared noise 'pm1'")
    for noise, floor in (("chained", "chained"), (None, "unproven"), ("bits01", "theorem-k")):
        assert S.filler_entry(noise, f)["applies"] == floor
    assert S.filler_entry(None, f)["unchecked"] == [] and S.FILLER_ENTRIES["unproven"]["fraction"] == 0

    def read(noise, tile=K_TILE, *extra, **kw):
        e = S.filler_entry(noise, S.kernel_facts(sass((DFILL, tile), *extra)), **kw)
        return e["applies"], e["failed"]

    e2 = S.kernel_facts(sass((DFILL, E2M1_TILE)))
    assert (e2["kind"], e2["L"], e2["accumulating_mma"]) == ("e2m1", 64, 0)
    applies, why = read("pm1", E2M1_TILE)
    assert applies == "theorem-e" and list(why) == ["theorem-k"] and len(why["theorem-k"]) == 3  # noise, kind, L
    assert why["theorem-k"][2] == "each word sums 64 products of a kind the entry doesn't cover"
    applies, why = read("bits01", E2M1_TILE)  # K covers integer words only, E ±1 only
    assert applies == "unproven" and why["theorem-k"][0] == ("the filler's MMAs are ['OMMA.SF.16864.F32.E2M1.E2M1."
                                                             "UE4M3.4X'], not all one of ['int8']")
    assert why["theorem-e"] == ["declared noise 'bits01', not one of ('pm1',)"]
    for mma in (E2M1_TILE[0], "IMMA.SP.16832.S8.S8 R28, R12.ROW, R40.COL, RZ",
                "IMMA.16832.S8.S8.SAT R28, R12.ROW, R40.COL, RZ", "HMMA.16816.F32 R28, R12, R40, RZ",
                "QMMA.16832.F32.E4M3.E4M3 R28, R12, R40, RZ"):
        for noise in ("bits01", "pm1"):
            applies, why = read(noise, K_TILE + [mma])
            assert applies == "unproven" and all(any("not all one of" in w for w in ws) for ws in why.values())
    assert read("bits01", K_TILE + ["I2F.S32 R9, R28"])[1]["theorem-k"] == ["FP or conversion ops in the filler's "
                                                                            "kernel: {'I2F.S32': 1}"]
    assert read("pm1", E2M1_TILE + ["FFMA R9, R28, 0.5, RZ"])[1]["theorem-e"][0].startswith("FP or conversion")
    assert S.kernel_facts(sass((DFILL, K_TILE + ["HFMA2.MMA R3, -RZ, RZ, 0, 0"])))["fp_free"]  # a constant load
    for tile in (K_TILE, E2M1_TILE):  # FILL v1's accumulate into the chain's register: chained, whatever it declares
        chained = [x.replace("COL, RZ", "COL, R28").replace("R40, RZ", "R40, R28") for x in tile]
        assert S.kernel_facts(sass((DFILL, chained)))["accumulating_mma"] == 1
        for noise in ("bits01", "pm1", None):
            applies, why = read(noise, chained)
            assert applies == "chained" and all(any("accumulate into an earlier word" in w for w in ws)
                                                for ws in why.values())
    assert "outside the filler's kernel" in read("bits01", K_TILE, ("_Z6helperv", [K_TILE[1]]))[1]["theorem-k"][0]
    phase = "_Z7dfill_pILi4EEvmmPyyiiPKiS0_Pj"  # the phase filler is read with dfill_k, launched or not
    assert S.kernel_facts(sass((DFILL, K_TILE), (phase, K_TILE)))["functions"] == [DFILL, phase]
    assert read("bits01", K_TILE, (phase, K_TILE)) == ("theorem-k", {})
    assert read("bits01", K_TILE, (phase, [x.replace("COL, RZ", "COL, R28") for x in K_TILE]))[0] == "chained"
    assert read("pm1", K_TILE, (phase, E2M1_TILE))[0] == "unproven"  # two kinds: no one L
    for noise in ("bits01", "pm1"):
        applies, why = read(noise, fixed_work=True)
        assert applies == "unproven" and "an arm-side driver's fixed work" in why["theorem-k"][-1]
    assert read("bits01", ["IADD3 R9, R9, R10, RZ", "EXIT"])[0] == "unproven"  # say-so, no MMA
    none = S.filler_entry("bits01", None)
    assert none["applies"] == "unproven" and none["failed"]["theorem-k"][0].startswith("the build recorded no kernel")
    assert S.filler_entry("chained", None)["applies"] == "chained"
    # an inactive entry that applies leaves SCORE at the next active one that does
    off = {**S.FILLER_ENTRIES, "theorem-k": {**S.FILLER_ENTRIES["theorem-k"], "active": False}}
    assert S.filler_entry("bits01", f, entries=off)["active"] == "unproven"
    assert S.SCORER == S.V5 and all(e["active"] and e["status"] == "proved" for e in S.FILLER_ENTRIES.values())
    stats = all_points()
    sc = {n: S.score(stats, S.certified(n)) for n in S.FILLER_ENTRIES}
    assert sc["chained"]["SCORE"] == sc["unproven"]["SCORE"] < sc["theorem-e"]["SCORE"] < sc["theorem-k"]["SCORE"]
    assert all(s["flags"] == [] for s in sc.values())
    ch = S.score(stats, S.certified(S.UNDER_CH))  # SCORE under CH is v2's SCORE, every filler at 0.99
    assert ch["SCORE"] == pytest.approx(S.score(stats)["SCORE"]) and ch["SCORE"] > sc["theorem-k"]["SCORE"]
    assert ch["flags"] == ["filler certified fraction 0.99 is under CH (withdrawn, decision 27)"]


def nvd(*functions):
    """``nvdisasm -c`` text: each (name, entry, instructions) a function in its line format, a kernel (entry) opening
    its section and a clone (``$kernel$callee``) following its kernel's; an item ending in ":" is a branch label."""
    out = []
    for name, entry, ins in functions:
        if entry:
            out += [f"//--------------------- .text.{name} --------------------------",
                    f'\t.section\t.text.{name},"ax",@progbits']
        out.append(f"        .type           {name},@function")
        if entry:
            out.append(f'        .other          {name},@"STO_CUDA_ENTRY STV_DEFAULT"')
        out.append(f"{name}:")
        a = 0
        for x in ins:
            if x.endswith(":"):
                out.append(x)
            else:
                out.append(f"        /*{a:04x}*/                   {x} ;")
                a += 16
    return "\n".join(out + ["//--------------------- SYMBOLS --------------------------"])


SERVE = "_Z6flin_kv"
WARP = f"${SERVE}$_Z12dfill_k_warpmmyy"
CONSUMER = ["IMMA.16832.S8.S8 R8, R12.ROW, R16.COL, R8", "STG.E desc[UR4][R6.64], R8"]
#: dfill_k_warp's clone in a serving kernel: K's tile, returning its receipt
WARP_BODY = [*K_TILE[:4], f"RET.REL.NODEC R2 `({SERVE})"]
#: dfill_k_put's clone, as FILL's arm B (2c2a61827) compiles it: the receipt and step count stored through the pointers
PUT = f"${SERVE}$_Z11dfill_k_putPyPimi"
PUT_BODY = ["LDCU.64 UR4, c[0x0][0x358]", "MOV R9, 0x0", "STG.E.64 desc[UR4][R2.64], R6", "STG.E desc[UR4][R4.64], R11",
            f"RET.REL.NODEC R8 `({SERVE})"]


def serving(*, before=(), after=(), consumer=CONSUMER, role="@P1 BRA `(.L_x_1)", tail=("EXIT",)):
    """A serving kernel with the INTERFACE's form: a warp-role branch to its filler warps, which call dfill_k_warp and
    return; the consumer's own MMAs and stores. `before` and `after` go on the filler path around the CALL."""
    return (SERVE, True, ["S2R R0, SR_TID.X", "ISETP.GE.AND P1, PT, R2, UR6, PT", "STS [R3], RZ", role, *consumer,
                          "EXIT", ".L_x_1:", *before, "BSSY.RECONVERGENT B0, `(.L_x_2)", f"CALL.REL.NOINC `({WARP})",
                          ".L_x_2:", "BSYNC.RECONVERGENT B0", *after, *tail])


def test_program_facts_read_the_filler_wherever_it_is_called_from():
    """filler-scorer/v4 reads a dfill_k function's body alone for the theorem checks, whether a kernel or a __noinline__
    clone a serving kernel calls from its filler warps; the serving kernel's own MMAs are serving's; and an MMA, a
    global store or a register call on the filler warps' path outside dfill_k, a role split by predication, or a branch
    it can't follow refuses the arm."""
    pf = S.program_facts({"fscore": nvd(serving(), (WARP, False, WARP_BODY))})
    f = pf["filler"]
    assert f["functions"] == [f"fscore: {WARP}"] and (f["kind"], f["L"], f["fp_free"], f["accumulating_mma"]) == \
        ("int8", 32, True, 0)
    assert pf["violations"] == [] and pf["serving_mma"] == {f"fscore: {SERVE}": 1}
    assert list(pf["callers"]) == [f"fscore: {SERVE}"] and S.paths_gate(pf)["ok"]
    e = S.filler_entry_v4("bits01", pf)
    assert (e["applies"], e["fraction"], e["failed"]) == ("theorem-k", 0.9, {})
    assert e["unchecked"][-len(S.UNCHECKED_V4):] == list(S.UNCHECKED_V4)
    assert e["kernel"]["serving_mma"] == pf["serving_mma"] and e["kernel"]["functions"] == f["functions"]
    assert S.filler_entry_v4("pm1", pf)["applies"] == "theorem-e"
    # v3 reads the same build at the floor: its cuobjdump view has the clone inside the kernel, an MMA outside the filler
    assert S.filler_entry("bits01", S.kernel_facts(sass((SERVE, CONSUMER + K_TILE))))["applies"] == "unproven"

    def refused(*fns, binary="fscore"):
        pf = S.program_facts({binary: nvd(*fns, (WARP, False, WARP_BODY))})
        e = S.filler_entry_v4("bits01", pf)
        assert not S.paths_gate(pf)["ok"] and e["applies"] == "unproven" and "confined" in e["checks"]["theorem-k"]
        assert not e["checks"]["theorem-k"]["confined"]
        return pf["violations"]

    imma = "IMMA.16832.S8.S8 R28, R12.ROW, R40.COL, RZ"
    assert refused(serving(before=[imma])) == [f"{SERVE} (fscore), on its filler warps' path: {imma.split()[0]} at 0070"]
    v = refused(serving(after=["STG.E.64 desc[UR4][R4.64], R10"]))
    assert len(v) == 1 and "STG.E.64 at 00a0 (a store that can write a receipt)" in v[0]
    assert "a call through a register" in refused(serving(after=["CALL.ABS.NOINC R14"]))[0]
    assert all("a store" in x for x in refused(serving(after=["RED.E.ADD desc[UR4][R4.64], R10"])))
    # the filler path rejoining the consumer's code instead of returning reads the consumer's work as the filler's
    v = refused(serving(tail=["BRA `(.L_x_0)"], consumer=[".L_x_0:", *CONSUMER]))
    assert len(v) == 2 and "IMMA" in v[0] and "STG.E at" in v[1]
    # predication instead of a branch: everything after the CALL is on the filler path
    pred = (SERVE, True, ["S2R R0, SR_TID.X", f"@P1 CALL.REL.NOINC `({WARP})", *CONSUMER, "EXIT"])
    assert len(refused(pred)) == 2
    assert "a branch whose target isn't named" in refused(serving(before=["BRX R4 -0x10"]))[0]
    # off the filler path, a register call (the cp.async.bulk syscall a producer makes) or a store is serving's
    ok = S.program_facts({"fscore": nvd(serving(consumer=[*CONSUMER, "CALL.ABS.NOINC R14"]), (WARP, False, WARP_BODY))})
    assert ok["violations"] == []

    # what the filler path calls is held to no MMA and no store; its FP (a u64 remainder's reciprocal) isn't the filler's
    rem = f"${SERVE}$__internal_1_$__cuda_sm20_rem_u64"
    pf = S.program_facts({"fscore": nvd(serving(before=[f"CALL.REL.NOINC `({rem})"]), (WARP, False, WARP_BODY),
                                         (rem, False, ["I2F.U64.RP R6, R4", "MUFU.RCP R6, R6", "RET.REL.NODEC R2 `(x)"]))})
    assert pf["violations"] == [] and pf["filler"]["fp_free"] and pf["filler"]["path_helpers"] == [f"fscore: {rem}"]
    pf = S.program_facts({"fscore": nvd(serving(before=[f"CALL.REL.NOINC `({rem})"]), (WARP, False, WARP_BODY),
                                        (rem, False, ["STG.E desc[UR4][R4.64], R6", "RET.REL.NODEC R2 `(x)"]))})
    assert pf["violations"] == [f"{rem} (fscore), called on the filler warps' path: STG.E at 0000 (a store that can "
                                "write a receipt)"]
    # what dfill_k itself calls is the filler's: its FP ops fail fp_free
    pf = S.program_facts({"fscore": nvd(serving(), (WARP, False, [f"CALL.REL.NOINC `({rem})", *WARP_BODY]),
                                        (rem, False, ["I2F.U64.RP R6, R4", "RET.REL.NODEC R2 `(x)"]))})
    assert not pf["filler"]["fp_free"] and pf["filler"]["helpers"] == [f"fscore: {rem}"]
    assert S.filler_entry_v4("bits01", pf)["failed"]["theorem-k"] == ["FP or conversion ops in the filler's code: "
                                                                      "{'I2F.U64.RP': 1}"]
    # FILL v1's chain in the clone reads chained; a split filler's kernel reads as v3 reads it
    chain = [x.replace("COL, RZ", "COL, R28") for x in WARP_BODY]
    assert S.filler_entry_v4("bits01", S.program_facts({"f": nvd(serving(), (WARP, False, chain))}))["applies"] == \
        "chained"
    split = S.program_facts({"dscore": nvd((DFILL, True, K_TILE))})
    assert split["callers"] == {} and split["filler"]["L"] == 32 and S.filler_entry_v4("bits01", split)["applies"] == \
        "theorem-k"
    # rows from a driver the scorer didn't read: no theorem
    e = S.filler_entry_v4("bits01", split, unread=["fscore"])
    assert e["applies"] == "unproven" and any("drivers the scorer didn't build" in w for w in e["failed"]["theorem-k"])
    none = S.filler_entry_v4("bits01", None)
    assert none["applies"] == "unproven" and none["kernel"] is None and S.paths_gate(None)["ok"]


def test_a_store_only_dfill_k_function_may_take_the_receipt_pointers():
    """A dfill_k device function that writes global memory (dfill_k_put) holds stores and address
    arithmetic only and stores its arguments as they came; an MMA function that stores, a put that computes, combines or
    calls, is refused. A dfill_k kernel (the split filler) is its own driver's and isn't held to it."""
    def facts(put_body=PUT_BODY, warp_body=WARP_BODY):
        return S.program_facts({"fscore": nvd(serving(after=[f"CALL.REL.NOINC `({PUT})"]), (WARP, False, warp_body),
                                              (PUT, False, put_body))})

    pf = facts()
    assert pf["violations"] == [] and pf["filler"]["stores"] == [f"fscore: {PUT}"] and S.paths_gate(pf)["ok"]
    assert (pf["filler"]["kind"], pf["filler"]["L"], pf["filler"]["fp_free"]) == ("int8", 32, True)
    assert S.filler_entry_v4("bits01", pf)["applies"] == "theorem-k"
    v = facts(warp_body=[*WARP_BODY[:-1], "STG.E.64 desc[UR4][R2.64], R28", WARP_BODY[-1]])["violations"]
    assert any("IMMA.16832.S8.S8 at" in x for x in v) and all("writes global memory" in x for x in v)
    for body, why in ((["IADD3 R6, R6, 0x1, RZ", *PUT_BODY[2:]], "stores a value the function computes"),
                      (["MOV R7, 0x0", *PUT_BODY[2:]], "stores a value the function computes"),
                      (["LOP3.LUT R11, R11, 0xff, RZ, 0xc0, !PT", *PUT_BODY], "LOP3.LUT at 0000"),
                      ([PUT_BODY[0], "RED.E.ADD desc[UR4][R2.64], R6", PUT_BODY[-1]], "RED.E.ADD at 0010"),
                      ([f"CALL.REL.NOINC `({WARP})", *PUT_BODY], "CALL.REL.NOINC at 0000")):
        v = facts(put_body=body)["violations"]
        assert v and all("writes global memory" in x for x in v) and any(why in x for x in v), (body, v)
        assert not S.paths_gate(facts(put_body=body))["ok"]
    # the put as nvcc places its arguments (tpcomm's tp_dfill_k_put split of c286d4f81): register-to-register moves
    # copy an argument, and the self-branch after the last RET never runs
    placed = ["LDCU.64 UR4, c[0x0][0x358]", "MOV R20, R27", "MOV R27, 0x0", "MOV R21, R28", "IMAD.MOV.U32 R5, RZ, RZ, R11",
              "STG.E.64 desc[UR4][R2.64], R20", "STG.E desc[UR4][R4.64], R5", f"RET.REL.NODEC R8 `({SERVE})", ".L_x_9:",
              "BRA `(.L_x_9)", "NOP"]
    assert facts(put_body=placed)["violations"] == []
    for body, why in ((["IADD3 R27, R27, 0x1, RZ", *placed[1:]], "STG.E.64 at 0050 stores a value the function computes"),
                      ([*placed[:2], "BRA `(.L_x_9)", *placed[2:]], "BRA at 0020")):
        v = facts(put_body=body)["violations"]
        assert v and any(why in x for x in v), (body, v)
    split = S.program_facts({"dscore": nvd((DFILL, True, [*K_TILE[:-1], "STG.E desc[UR4][R2.64], R9", "EXIT"]))})
    assert split["violations"] == [] and split["filler"]["stores"] == []


def fixed_row(*, driver="fscore", checked=True, rec_bad=0, fixed_macs=1e12, fixed_ms=20.0, mhz=2000.0, b=1, l=2048):
    r = row("step", b, l, "split", alone_ms=6.0, arm_ms=6.2, rounds=2, fill_sms=108, rec_bad=rec_bad)
    r |= {"driver": driver, "fixed_sms": 80}
    for blk in r["blocks"]:
        blk["arm"]["reps"] = [[6.2, mhz, 300.0]] * 3
        blk["arm"] |= {"fixed_ms": fixed_ms, "fixed_macs": fixed_macs, "round": {"id": 1},
                       "fused_round": {"id": 2, "checked": [7, 9] if checked else []}}
    return r


def test_fixed_work_is_credited_from_checked_public_tiles_of_a_read_binary():
    """filler-scorer/v4 credits a block's fixed work only when its row's binary is one the scorer built and read, the
    filler warps' paths there are confined, the block's fixed round has spot-checked tiles and the row no bad receipt;
    and bounds the whole kernel, serving's MACs and the filler warps', by its SMs' peak."""
    pf = S.program_facts({"dscore": nvd((DFILL, True, K_TILE)), "fscore": nvd(serving(), (WARP, False, WARP_BODY))})
    rows = [fixed_row(), fixed_row(driver="pscore"), fixed_row(checked=False), fixed_row(rec_bad=1)]
    out, credit = S.credit_fixed(rows, pf)
    assert credit["credited"] == 2 and len(credit["dropped"]) == 6
    assert [b["arm"]["fixed_macs"] for r in out for b in r["blocks"]] == [1e12, 1e12] + [0.0] * 6
    assert rows[1]["blocks"][0]["arm"]["fixed_macs"] == 1e12  # the rows read are not changed
    assert "its driver pscore wasn't built and read" in credit["dropped"][0]
    assert "none of its tiles' receipts was spot-checked" in credit["dropped"][2]
    assert "1 bad receipts in its row" in credit["dropped"][4]
    bad = S.program_facts({"fscore": nvd(serving(before=["STG.E desc[UR4][R4.64], R1"]), (WARP, False, WARP_BODY))})
    assert S.credit_fixed([fixed_row()], bad)[1]["credited"] == 0
    assert "no program facts" in S.credit_fixed([fixed_row()], None)[1]["dropped"][0]
    # filler warps inlined into a serving kernel (the inlined form): nothing v4 reads as the filler did that work
    r3 = S.program_facts({"fscore": nvd((DFILL, True, K_TILE), (SERVE, True, [*CONSUMER, *K_TILE]))})
    assert r3["violations"] == [] and r3["fixed_from"] == {"fscore": 0}
    assert "no function in fscore it can come from" in S.credit_fixed([fixed_row()], r3)[1]["dropped"][0]
    phase = S.program_facts({"pscore": nvd(("_Z7dfill_pILi4EEvmmPyyiiPKiS0_Pj", True, K_TILE))})
    assert phase["fixed_from"] == {"pscore": 1} and S.credit_fixed([fixed_row(driver="pscore")], phase)[1]["credited"]
    assert S.point_stats(out[0])["fill_rate"] > S.point_stats(out[2])["fill_rate"]

    r = fixed_row(fixed_macs=1e12)
    peak = S.PER_SM * 80 * 2000.0e6 * 20.0e-3
    assert S.whole_of_peak(r) == pytest.approx((2 * 3 * share.decode_n(1, 2048) + 2e12) / (2 * peak))
    assert S.whole_of_peak(row("step", 1, 2048, "split", alone_ms=6.0, arm_ms=6.2)) is None
    over = fixed_row(fixed_macs=0.9 * peak, b=32)  # the filler warps alone under the peak, with serving's MACs past it
    assert 1 < S.whole_of_peak(over) and S.point_stats(over)["of_peak"] == pytest.approx(0.9)
    v1 = A.load("fill-v1")
    st = {("decode", 32, 2048): S.point_stats(over)}
    h = S.honesty(v1, st, [], whole={"decode 32/2048": S.whole_of_peak(over)})
    assert not h["ok"] and h["over_peak_whole"] and not h["over_peak"]
    assert "over_peak_whole" not in S.honesty(v1, st, [])  # v3's honest gate is unchanged


def test_record_limits_are_checked_when_the_arm_loads():
    """filler-scorer/v3 refuses, before any build or lease, an arm whose record the ledger or the store would refuse
    after the run (a 249-character hypothesis reached `score.record` only after a scored run)."""
    import dataclasses

    v1 = A.FillV1()
    assert A.check_arm(v1, record=True) == [] and A.check_arm(A.Idle(), record=True) == []
    assert A.load("fill-v1", record=True).name == "fill-v1"

    def problems(**kw):
        lin = {k: v for k, v in kw.items() if k in v1.lineage}
        return A.check_arm(dataclasses.replace(v1, lineage={**v1.lineage, **lin},
                                               **{k: v for k, v in kw.items() if k not in lin}), record=True)

    long = dataclasses.replace(v1, lineage={**v1.lineage, "hypothesis": "x " * 124 + "y"})
    assert A.check_arm(long) == []  # v2's load check lets it through to the lease
    assert any("at most 240 characters, not 249" in p for p in A.check_arm(long, record=True))
    assert problems(hypothesis="x" * 240) == []
    assert any("choice" in p and "not 300" in p for p in problems(choice="c" * 300))
    assert any("whole art: id" in p for p in problems(parent="art:cc59294a"))
    assert problems(parent="art:" + "0" * 64) == [] and problems(parent="fill-v1") == []
    assert any("lineage.parent 5" in p for p in problems(parent=5))
    assert any("list of tags" in p for p in problems(mechanism="place:split"))
    assert any("canonical JSON" in p for p in problems(approach={1, 2}))
    assert any("don't make a variant" in p for p in problems(sources=v1.sources + v1.sources[:1]))
    assert any("don't make a variant" in p for p in problems(sources=("/etc/hostname",)))
    assert any("filler.noise" in p for p in A.check_arm(dataclasses.replace(v1, filler=A.Filler(noise="gauss"))))
    assert A.check_arm(dataclasses.replace(v1, filler=A.Filler(noise="bitsU"))) == [] and v1.filler.noise == "chained"
    # the declared noise isn't run-time config: variant ids don't move
    assert A.config(A.Filler(noise="bits01")) == A.config(A.Filler()) == {"W": "8", "NC": "4", "S": "2048",
                                                                          "chunk_log2": "19"}


def test_arms():
    v1, idle = A.load("fill-v1"), A.load("idle")
    assert A.placed(idle, "decode", 32, 2048) == A.NONE
    assert A.placed(v1, "prefill", 8192, 0) == A.NONE
    assert A.placed(v1, "decode", 1, 2048) == A.Colocation("split", 108)
    assert A.placed(v1, "decode", 32, 2048).fill_sms == 92  # FILL's rule: (32, 1K), the longest L ≤ 2K at B 32
    assert A.placed(v1, "decode", 144, 5120).fill_sms == 60
    assert A.placed(v1, "decode", 8, 16384).fill_sms == 124  # B 32's longest L ≤ 16K: (32, 8K)
    assert S.honesty(v1, all_points(), S.points())["ok"]
    assert not S.honesty(v1, all_points(), S.points(), has_filler=True)["ok"]  # no positive control
    with pytest.raises(A.ArmError):
        A.Colocation("split", 0)
    with pytest.raises(A.ArmError):
        A.load("no-such-arm")


#: pf_absorb<NC>'s MMA word entries as the sm_120a builds compile them (c259a89b5): one 16832 IMMA's four
#: words R28..R31, y in R9, the keys in R20/R21
IMMA4 = "IMMA.16832.S8.S8 R28, R12.ROW, R40.COL, RZ"
RAM = [IMMA4, "LEA.HI R9, R9, R28, R9, 0x7", "IMAD R9, R9, R20, R29", "LEA.HI R9, R9, R30, R9, 0x13",
       "IMAD R9, R9, R21, R31", "EXIT"]
XIN = [IMMA4, "SHF.L.W.U32.HI R9, R9, 0x7, R9", "LOP3.LUT R9, R9, R28, RZ, 0x3c, !PT", "IMAD R9, R9, R20, R29",
       "SHF.L.W.U32.HI R9, R9, 0x13, R9", "LOP3.LUT R9, R9, R30, RZ, 0x3c, !PT", "IMAD R9, R9, R21, R31", "EXIT"]
DUO = [IMMA4] + [x for w in (28, 29, 30, 31) for x in (f"LOP3.LUT R9, R9, R{w}, R22, 0x96, !PT", "IMAD R9, R9, R20, R10",
                                                     "IADD3 R10, R9, R11, RZ")] + ["EXIT"]
#: The r9 folds as the scratch -DPFILL_XIN13 and -DPFILL_LI builds of
#: c259a89b5's pfill.cuh compile them: XOR-in at rotations 13 and 19; LOP3+IMAD's y ^ d0 ^ k0 in one LOP3, no shift
XIN13 = [IMMA4, "SHF.L.W.U32.HI R9, R9, 0xd, R9", "LOP3.LUT R9, R9, R28, RZ, 0x3c, !PT", "IMAD R9, R9, R20, R29",
         "SHF.L.W.U32.HI R9, R9, 0x13, R9", "LOP3.LUT R9, R9, R30, RZ, 0x3c, !PT", "IMAD R9, R9, R21, R31", "EXIT"]
LI = [IMMA4, "LOP3.LUT R9, R22, R28, R9, 0x96, !PT", "IMAD R9, R20, R9, R29", "LOP3.LUT R9, R23, R30, R9, 0x96, !PT",
      "IMAD R9, R21, R9, R31", "EXIT"]
K4, K2, K1, K7 = (f"_Z7dfill_kILi{n}EEvmmyiiPy" for n in (4, 2, 1, 7))


def test_the_fold_reads_from_the_sass_of_the_instantiation_that_runs():
    """filler-scorer/v5 traces each MMA word of the filler that runs to its first reader (through moves): ram's adds
    and addends (or Q = a0·d0 + d1's multiplicand and addend), XOR-in's XORs and addends, duo's XORs; two words in one
    reader (NC 1's y ^= d0 ^ d1 ^ ...), a word never read, or a probe fold (NC 7's adds) read as no fold v5 reads."""
    def facts(nc, *fns):
        return S.fold_facts({"dscore": nvd(*((n, True, b) for n, b in fns))}, nc)

    assert S.fold_words(S.parse_nvdisasm(nvd((K4, True, RAM)))[0]) == ["add", "addend", "add", "addend"]
    ff = facts(4, (K4, RAM), (K2, DUO))
    assert (ff["signature"], ff["templated"], ff["not_run"]) == ("ram", True, 1)
    assert ff["functions"] == {f"dscore: {K4}": {"words": {"add": 2, "addend": 2}, "rotations": {}, "signature": "ram"}}
    assert facts(2, (K4, RAM), (K2, DUO))["signature"] == "duo" and facts(4, (K4, XIN))["signature"] == "xor-in"
    q = [IMMA4, "MOV R40, R28", "IMAD R11, R40, R20, R29", "IMAD R9, R9, R20, R11", "LEA.HI R9, R9, R30, R9, 0x13",
         "IMAD R9, R9, R21, R31", "EXIT"]
    assert facts(4, (K4, q))["functions"][f"dscore: {K4}"]["words"] == {"add": 1, "addend": 2, "mul": 1}
    assert facts(4, (K4, q))["signature"] == "ram"
    xor1 = [IMMA4, "LOP3.LUT R9, R28, R29, R30, 0x96, !PT", "LOP3.LUT R9, R9, R31, RZ, 0x3c, !PT", "EXIT"]
    assert facts(1, (K1, xor1))["functions"][f"dscore: {K1}"]["words"] == {"combined": 3, "xor": 1}
    assert facts(1, (K1, xor1))["signature"] == "combined"
    lost = [IMMA4, "LEA.HI R9, R9, R28, R9, 0x7", "IMAD R9, R9, R20, R29", IMMA4, *RAM[1:]]
    assert facts(4, (K4, lost))["signature"] == "unread"
    adds = [IMMA4, *(f"IADD3 R9, R9, R{w}, RZ" for w in (28, 29, 30, 31)), "EXIT"]
    assert facts(7, (K7, adds))["signature"] == "unrecognised"
    # a fused filler's dfill_k_warp clone is read in every clone, at any NC
    clone = S.fold_facts({"fscore": nvd(serving(), (WARP, False, RAM[:-1] + [WARP_BODY[-1]]))}, 2)
    assert (clone["signature"], clone["templated"]) == ("ram", False)


def test_the_fold_is_read_from_the_declaration_and_what_the_build_confirms():
    """duo reads in full and XOR-in at Theorem K's proved half, each only where declared and
    the SASS, the NC and the recorded NVCC_APPEND_FLAGS confirm it; ram reads half; undeclared or unconfirmed reads as
    ram where the build shows any
    fold v5 reads, else no fold (no theorem entry); the folded fraction is the entry's times the share."""
    def ff(nc, body, name=None):
        return S.fold_facts({"dscore": nvd((name or f"_Z7dfill_kILi{nc}EEvmmyiiPy", True, body))}, nc)

    def read(declared, nc, flags, facts):
        r = S.fold_reading(declared, nc, flags, facts)
        return r["fold"], r["share"]

    assert read(None, 4, None, ff(4, RAM)) == ("ram", 0.5)
    assert S.fold_reading(None, 4, None, ff(4, RAM))["why"] == ["undeclared: read as ram"]
    assert read("ram", 4, "", ff(4, RAM)) == ("ram", 0.5) and read("duo", 2, None, ff(2, DUO)) == ("duo", 1.0)
    assert S.fold_reading("duo", 2, None, ff(2, DUO))["provisional"] == S.FOLDS["duo"]["provisional"]
    assert S.fold_reading("ram", 4, None, ff(4, RAM))["provisional"] is None
    assert read("xor-in", 4, "-DPFILL_BITS -DPFILL_XIN", ff(4, XIN)) == ("xor-in", 0.5)
    assert read("xor-in", 4, None, ff(4, XIN)) == ("xor-in", 0.5)  # a build that recorded no flags
    r = S.fold_reading("xor-in", 4, "-DPFILL_BITS", ff(4, XIN))
    assert (r["fold"], r["share"]) == ("ram", 0.5) and "lack -DPFILL_XIN" in r["why"][0]
    r = S.fold_reading("duo", 4, None, ff(4, RAM))
    assert r["fold"] == "ram" and any("not duo's entries" in w for w in r["why"]) and any("NC 4" in w for w in r["why"])
    assert read("xor-in", 4, "-DPFILL_XIN", ff(4, RAM)) == ("ram", 0.5)  # the SASS shows ram's entries
    assert read(None, 2, None, ff(2, DUO)) == ("ram", 0.5)  # undeclared duo reads as ram
    xor1 = [IMMA4, "LOP3.LUT R9, R28, R29, R30, 0x96, !PT", "LOP3.LUT R9, R9, R31, RZ, 0x3c, !PT", "EXIT"]
    assert read("duo", 1, None, ff(1, xor1)) == (None, None) and read(None, 1, None, ff(1, xor1)) == (None, None)
    adds = [IMMA4, *(f"IADD3 R9, R9, R{w}, RZ" for w in (28, 29, 30, 31)), "EXIT"]
    assert read("ram", 7, None, ff(7, adds)) == (None, None)
    assert read(None, 4, None, None) == ("ram", 0.5)  # a build from before v5
    assert S.fold_reading(None, 4, None, None)["unchecked"] == list(S.FOLD_UNCHECKED)
    # the declaration: Filler.fold, else the one receipt lineage tag
    assert S.declared_fold({"fold": "duo"}, {"mechanism": ["receipt:ram-k32"]}) == ("duo", "Filler.fold")
    assert S.declared_fold({}, {"mechanism": ["noise:x", "receipt:xin-k32"]}) == ("xor-in", "lineage receipt tag")
    assert S.declared_fold(None, {"mechanism": ["receipt:ram-k64"]}) == ("ram", "lineage receipt tag")
    assert S.declared_fold({}, {"mechanism": ["place:split"]}) == (None, "undeclared")
    assert S.declared_fold({}, {"mechanism": ["receipt:ram-k32", "receipt:duo-k32"]})[0] is None
    assert tuple(S.FOLDS) == A.FOLDS
    import dataclasses
    v1 = A.FillV1()
    assert any("filler.fold" in p for p in A.check_arm(dataclasses.replace(v1, filler=A.Filler(fold="xor"))))
    assert A.check_arm(dataclasses.replace(v1, filler=A.Filler(fold="duo"))) == []
    assert A.config(A.Filler(fold="duo")) == A.config(A.Filler())
    # the entry: Theorem K at 0.45 under ram, 0.9 under duo; Theorem E at 0.1125 under ram; no fold, no theorem
    split = S.program_facts({"dscore": nvd((DFILL, True, K_TILE))})
    e = S.filler_entry_v5("bits01", split, S.fold_reading(None, 4, None, ff(4, RAM)))
    assert (e["applies"], e["fraction"], e["unfolded"], e["active_fraction"]) == ("theorem-k", 0.45, 0.9, 0.45)
    assert S.filler_entry_v5("bits01", split, S.fold_reading("duo", 2, None, ff(2, DUO)))["fraction"] == 0.9
    assert S.filler_entry_v5("pm1", split, S.fold_reading("ram", 4, None, ff(4, RAM)))["fraction"] == 0.1125
    none = S.filler_entry_v5("bits01", split, S.fold_reading(None, 1, None, ff(1, xor1)))
    assert none["applies"] == "unproven" and none["fraction"] == 0.0 and any(
        w.startswith("fold: ") for w in none["failed"]["theorem-k"])
    assert S.certified_v5(e)["filler"]["fraction"] == 0.45 and S.SCORER == S.V5


def test_the_r9_folds_are_told_apart_by_the_rotation_into_their_xor_and_read_provisionally():
    """The r9 folds: XOR-in (13, 19) and LOP3+IMAD read where declared and confirmed, told
    apart from the superseded XOR-in (7, 19) by the state's funnel-shift rotation into each XOR (through a move); XORs
    and addends at rotations no fold lists read as ram; each of the three is provisional, which the filler's certified
    status and the score's flags carry. All three read at Theorem K's proved half, 0.45, with 0.9 if the r10
    select-charging lemma lands as the entry's secondary readout, never its fraction;
    ram and duo have no such readout. duo reads in full and provisionally (its binding is proved up to four words);
    ram reads proved."""
    def ff(nc, body):
        return S.fold_facts({"dscore": nvd((f"_Z7dfill_kILi{nc}EEvmmyiiPy", True, body))}, nc)

    for body, sig, rots in ((XIN, "xor-in", {"7": 1, "19": 1}), (XIN13, "xor-in-13", {"13": 1, "19": 1}),
                            (LI, "lop3-imad", {"0": 2})):
        f = ff(4, body)
        assert f["signature"] == sig and next(iter(f["functions"].values()))["rotations"] == rots
    moved = [IMMA4, "SHF.L.W.U32.HI R40, R9, 0xd, R9", "MOV R41, R40", "LOP3.LUT R9, R41, R28, RZ, 0x3c, !PT",
             *XIN13[3:]]
    assert ff(4, moved)["signature"] == "xor-in-13"
    right = [x.replace("SHF.L.W.U32.HI R9, R9, 0xd, R9", "SHF.R.W.U32.HI R9, R9, 0x13, R9") for x in XIN13]
    assert ff(4, right)["signature"] == "xor-in-13"  # rotr 19 = rotl 13
    odd = [x.replace("0xd", "0x5") for x in XIN13]
    assert ff(4, odd)["signature"] == "xor-addend"
    assert S.fold_words(S.parse_nvdisasm(nvd((K4, True, LI)))[0], rot := []) == ["xor", "addend"] * 2 and rot == [0, None, 0, None]

    def read(declared, flags, body):
        r = S.fold_reading(declared, 4, flags, ff(4, body))
        return r["fold"], r["share"], r["provisional"] is not None

    assert read("xor-in-13", "-DPFILL_XIN13", XIN13) == ("xor-in-13", 0.5, True)
    assert read("lop3-imad", "-DPFILL_LI", LI) == ("lop3-imad", 0.5, True)
    assert read("xor-in", "-DPFILL_XIN", XIN) == ("xor-in", 0.5, True)  # superseded, still read
    assert read("xor-in-13", "-DPFILL_XIN", XIN) == ("ram", 0.5, False)  # the SASS shows rotation 7
    assert read("lop3-imad", "-DPFILL_LI", XIN13) == ("ram", 0.5, False)
    assert read("xor-in-13", "-DPFILL_LI", XIN13) == ("ram", 0.5, False)  # the flags lack -DPFILL_XIN13
    assert read(None, None, LI) == ("ram", 0.5, False) and read("ram", None, odd) == ("ram", 0.5, False)
    assert S.declared_fold({}, {"mechanism": ["receipt:xin13-k32"]})[0] == "xor-in-13"
    assert S.declared_fold({}, {"mechanism": ["receipt:li-k32"]})[0] == "lop3-imad"
    import dataclasses
    assert A.check_arm(dataclasses.replace(A.FillV1(), filler=A.Filler(fold="lop3-imad"))) == []
    split = S.program_facts({"dscore": nvd((DFILL, True, K_TILE))})
    for fold, flag, body in (("lop3-imad", "-DPFILL_LI", LI), ("xor-in-13", "-DPFILL_XIN13", XIN13),
                             ("xor-in", "-DPFILL_XIN", XIN)):
        e = S.filler_entry_v5("bits01", split, S.fold_reading(fold, 4, flag, ff(4, body)))
        c = S.certified_v5(e)
        assert (e["fraction"], e["active_fraction"], c["filler"]["fraction"]) == (0.45, 0.45, 0.45)
        assert c["filler"]["status"].startswith(f"provisional (fold {fold}: ")
        assert any(f"provisional (fold {fold}" in f for f in S.score(all_points(), c)["flags"])
        x = e["if_fold"]
        assert (x["share"], x["fraction"], x["active_fraction"], x["if"]) == (1.0, 0.9, 0.9, S.SELECT_LEMMA)
        assert "select-charging lemma" in x["if"] and "Part VII" in x["if"]
    ram_e = S.filler_entry_v5("bits01", split, S.fold_reading("ram", 4, None, ff(4, RAM)))
    duo_e = S.filler_entry_v5("bits01", split, S.fold_reading("duo", 2, None, ff(2, DUO)))
    assert (ram_e["fraction"], ram_e["if_fold"], duo_e["fraction"], duo_e["if_fold"]) == (0.45, None, 0.9, None)
    ram = S.certified_v5(ram_e)
    assert ram["filler"]["status"] == "proved" and not S.score(all_points(), ram)["flags"]
    # duo's binding is proved for runs of one to four words only: provisional, and flagged
    duo = S.certified_v5(duo_e)
    assert duo["filler"]["fraction"] == 0.9
    assert duo["filler"]["status"] == f"provisional (fold duo: {S.FOLDS['duo']['provisional']})"
    assert "DuoQuadNoKernel" in duo["filler"]["status"] and "projected form" in duo["filler"]["status"]
    assert [f for f in S.score(all_points(), duo)["flags"] if "provisional (fold duo" in f]


def test_score_v5_is_the_measured_filler_part_plus_the_modelled_served_part():
    """filler-scorer/v5's SCORE: every served MAC at 0.9 over a wall time the modelled edits lengthen (decode's
    deferred pass's shown part, prefill's edited GEMMs and coins), the filler's decode MACs less what the pass's hidden
    part displaces; each part shown, the factors named constants, prefill's coins a parameter."""
    shown = S.defer_shown_1gpu()
    assert shown == pytest.approx(0.053 * 12.14 / (32 * 32 * 0.97e-3)) and 0.64 < shown < 0.66
    stats = all_points()
    sc = S.score_v5(stats, 0.45)
    assert not sc["missing"] and sc["SCORE"] == pytest.approx(sc["filler_part"] + sc["served_part_modelled"])
    c = sc["served_model"]["constants"]
    assert {k: v for k, v in c.items() if k not in ("DEFER_FIT", "DEFER_FIT_TP")} == {
        "fraction": 0.9, "EGEMM": 1.57, "DEFER_OF_PEAK": 0.71, "DEFER_T": 4096, "DEFER_SHOWN_1GPU": shown,
        "PREFILL_COINS": 0.13}
    assert c["DEFER_FIT"]["u"]["intercept"] == {str(t): v for t, v in S.DEFER_FIT["u"]["intercept"].items()}
    assert c["DEFER_FIT_TP"]["d"]["intercept"] == {str(t): v for t, v in S.DEFER_FIT_TP["d"]["intercept"].items()}
    assert set(sc["served_model"]["sources"]) == set(c)
    by = {e["entry"]: e for e in sc["entries"]}
    for e in sc["entries"]:
        assert e["t1_ms"] > e["t1_measured_ms"] and e["filler_macs"] <= e["filler_measured_macs"]
        assert e["served_part_modelled"] == pytest.approx(0.9 * e["serving_macs"] / e["capacity_macs"])
    d = next(e for e in sc["entries"] if e["kind"] == "decode")
    v4 = S.decode_entry(stats[("decode", d["B"], d["L"])], d["B"], d["L"])
    pass_ms = 1.57 * v4["serving_macs"] / (0.71 * S.cap(v4["mhz"])) * 1e3
    u = max(0.71, S.defer_at(S.DEFER_FIT, "u", 2, d["B"], d["L"]))
    assert d["served_model"]["defer_d"] == pytest.approx(u * (1 - shown)) and d["served_model"]["defer_s"] == shown
    assert d["t1_ms"] == pytest.approx(v4["t1_ms"] + shown * pass_ms)
    assert d["filler_macs"] == pytest.approx(
        max(0.0, v4["filler_macs"] - u * (1 - shown) * pass_ms * S.cap(v4["mhz"]) * 1e-3))
    p = next(e for e in by.values() if e["kind"] == "prefill")
    st = stats[("prefill", S.PREFILL_M, 0)]
    assert p["t1_ms"] == pytest.approx(p["t1_measured_ms"] + (0.57 + 0.13) * (st["alone_ms"] + share.FA2_MS_8192))
    assert p["filler_macs"] == p["filler_measured_macs"]
    assert S.score_v5(stats, 0.0)["filler_part"] == 0.0
    assert S.score_v5(stats, 0.45, coins=0.0)["SCORE"] > sc["SCORE"] > S.score_v5(stats, 0.45, coins=0.3)["SCORE"]
    # the entries' measured arithmetic is v4's
    v4sc = S.score(stats, S.certified("theorem-k"))
    assert [e["t1_ms"] for e in v4sc["entries"]] == [e["t1_measured_ms"] for e in sc["entries"]]


def test_the_deferred_pass_is_calibrated_to_measurement(monkeypatch):
    """The deferred pass's shown share s and displacement come from measurement, TP's s
    and d (lost MACs a ms of the whole pass) fitted by TP, batch and context to rounds 3 (ram) and 6 (duo) jointly
    (each point's residual reported, the filler's remaining share among them), one GPU's share from stage_cost (B 32,
    held flat) and its displacement u(1 − s), u the larger of TP 2's r3 fit and the MAC-for-MAC rule; read at B and L
    clamped to the points."""
    def point(tp, b, l, s, u, mhz=2000.0, served=1e12):
        plain = served / (0.71 * S.cap(mhz)) * 1e3
        lost = u * (plain - s * plain) * S.cap(mhz) * 1e-3
        return {"T": 4096, "tp": tp, "B": b, "L": l, "base_ms": 40.0, "defer_ms": 40.0 + s * plain, "dalone_ms": plain,
                "mhz": mhz, "served_macs": served, "defer_fill_macs": 5e12 - lost, "phase_slowdown": 1.0,
                "phase_fill_macs": 5e12}

    law = {2: (0.8, 1.1), 4: (0.6, 1.4)}
    pts = [point(t, b, l, law[t][0] * (b / 32) ** -0.5 * (l / 2048) ** 0.1, law[t][1] * (b / 32) ** -0.2)
           for t in (2, 4) for b, l in ((32, 2048), (64, 2048), (128, 8192), (256, 2048))]
    f = S.defer_fit(4096, pts)
    assert f["s"]["intercept"] == pytest.approx({2: 0.8, 4: 0.6}) and f["u"]["intercept"] == pytest.approx({2: 1.1, 4: 1.4})
    assert (f["s"]["batch"], f["s"]["context"], f["u"]["batch"]) == pytest.approx((-0.5, 0.1, -0.2))
    assert f["s"]["rms"] < 1e-9 and f["slowdown_err_pts_max"] < 1e-9 and f["fill_err_pts_max"] < 1e-9
    fd = S.defer_fit(4096, pts, "d")  # d = u(1 − s): its batch slope is u's plus (1 − s)'s, not a power law, so a residual
    assert fd["form"] == "d" and fd["s"]["intercept"] == pytest.approx({2: 0.8, 4: 0.6}) and fd["d"]["rms"] < 0.25
    for xd, xu in zip(fd["points"], f["points"]):
        assert "u" not in xd and xd["d"] == pytest.approx(xu["u"] * (1 - xu["s"]))
    with pytest.raises(ValueError, match="measured at"):
        S.defer_fit(2048)
    assert S.defer_fit(8192)["tps"] == S.defer_fit(16384)["tps"] == [2]  # tpcomm: TP 2 alone
    # one GPU's u: the fit to tpcomm at T = 4096, every residual reported
    fit = S.DEFER_FIT
    assert fit["T"] == 4096 and fit["tps"] == [2, 4] and fit["n"] == 12 and len(fit["points"]) == 12
    assert fit["s"]["batch"] < 0 and fit["s"]["rms"] < 0.15 and fit["u"]["rms"] < 0.15 and fit["slowdown_err_pts_max"] < 2
    # TP: r3 and r6 jointly in the d form; the model (EGEMM 1) gives back each point's added time and remaining fill
    tpf = S.DEFER_FIT_TP
    assert (tpf["form"], tpf["tps"], tpf["n"], len(S.DEFER_TP_R6)) == ("d", [2, 4], 24, 12)
    assert tpf["s"]["batch"] < 0 and tpf["slowdown_err_pts_max"] < 5 and tpf["fill_err_pts_rms"] < 2
    assert max(abs(x["fill_err_pts"]) for x in tpf["points"][12:]) < 2  # r6's duo points, the ones TP scores
    assert tpf["fill_err_pts_rms"] < S.defer_fit(points=S.DEFER_TP + S.DEFER_TP_R6)["fill_err_pts_rms"]
    monkeypatch.setattr(S, "EGEMM", 1.0)
    for p, r in zip([p for p in S.DEFER_TP + S.DEFER_TP_R6 if p["T"] == 4096], tpf["points"]):
        m = S.defer_measured(p)
        dp = S.defer_pass(p["served_macs"], p["mhz"], p["tp"], p["B"], p["L"])
        assert dp["added_ms"] == pytest.approx(m["added_ms"] + r["added_err_ms"])
        assert dp["displaces"] / m["lost_macs"] == pytest.approx(1 + r["d_resid"])
        t1 = p["base_ms"] * p["phase_slowdown"] + dp["added_ms"]
        share = max(0.0, p["phase_fill_macs"] - dp["displaces"]) / (S.cap(p["mhz"]) * t1 * 1e-3)
        assert 100 * share == pytest.approx(100 * r["fill_share"] + r["fill_err_pts"])
    assert S.defer_rates(2, 1, 1024) == S.defer_rates(2, 32, 2048)
    assert S.defer_rates(4, 1024, 65536) == S.defer_rates(4, 256, 8192)
    with pytest.raises(ValueError, match="TP 8"):
        S.defer_rates(8, 32, 2048)
    one = S.defer_rates(1, 128, 6144)
    assert one["s"] == S.defer_shown_1gpu() and one["u"] == max(0.71, S.defer_at(fit, "u", 2, 128, 6144))
    assert one["d"] == pytest.approx(one["u"] * (1 - one["s"]))
    assert S.defer_rates(1, 256, 8192)["u"] == 0.71 < S.defer_rates(1, 32, 2048)["u"]


def tp_log(rank, tp=2, b=32, l=2048, *, base=40.0, phase=41.0, fill=1e12, served=1e12, mhz=2000.0, repeats=2, reps=9,
           rec_bad=0, kinds=("fixed", "up", "cut")):
    import json
    out = [json.dumps({"mode": "pf_gate", "kind": k, "rank": rank, "ok": True}) for k in kinds]
    for _ in range(repeats):
        for arm, ms in (("base", base), ("phase", phase)):
            r = {"mode": "tp", "tp": tp, "rank": rank, "B": b, "L": l, "arm": arm, "ms": ms, "mhz": mhz,
                 "served_macs": served, "fill_macs_rep": fill if arm == "phase" else 0.0,
                 "reps": [[ms, mhz, 200]] * reps}
            if arm == "phase":
                r |= {"rec_checked": 8, "rec_bad": rec_bad, "shape_bad": 0}
            out.append(json.dumps(r))
    return "\n".join(out)


def test_score_tp_reads_the_slowest_rank_with_the_gates_per_rank():
    """The TP score (tpcomm's suite): each TP point over the slowest rank's step, per GPU, its served MACs at 0.9 with
    the deferred pass and the filler at the folded fraction, weighted TP × wall time; the slowdown, pf_gate, receipts
    and NCCL gates read every rank."""
    rows = S.tp_rows({0: tp_log(0, phase=41.0), 1: tp_log(1, phase=42.0, fill=2e12)})
    by = rows["arms"][(2, 32, 2048)]
    assert len(by["base"]) == 2 and [len(x) for x in by["phase"]] == [2, 2]
    e = S.tp_entry(by, (2, 32, 2048), 0.45)
    assert (e["entry"], e["tp"], e["B"], e["L"]) == ("TP 2 B 32 L 2K", 2, 32, 2048)
    assert e["t1_measured_ms"] == 42.0 and e["slowdown"] == pytest.approx(42.0 / 40.0)
    dp = S.defer_pass(1e12, 2000.0, 2, 32, 2048)
    assert dp["s"] == pytest.approx(min(1.0, S.DEFER_FIT_TP["s"]["intercept"][2])) and e["served_model"]["defer_s"] == dp["s"]
    assert dp["d"] == pytest.approx(S.DEFER_FIT_TP["d"]["intercept"][2]) and e["served_model"]["defer_d"] == dp["d"]
    assert e["t1_ms"] == pytest.approx(42.0 + dp["added_ms"]) and e["filler_measured_macs"] == 1.5e12
    c = S.cap(2000.0) * e["t1_ms"] * 1e-3
    assert e["filler_part"] == pytest.approx(0.45 * (1.5e12 - dp["displaces"]) / c)
    assert e["served_part_modelled"] == pytest.approx(0.9 * 1e12 / c) and e["time"] == pytest.approx(2 * e["t1_ms"] / 40.0)
    assert e["time_v4"] == pytest.approx(2 * 42.0 / 40.0)  # v4's form weighs by v4's wall time, no modelled pass
    sets = [{"dir": "a/set0", "settings": "NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64", "rows": rows}]
    g = S.tp_gates(sets, {(2, 32, 2048): e}, None)
    assert all(g[k]["ok"] for k in ("slowdown", "pf_gate", "receipts", "nccl", "filler_paths")), g
    bad = S.tp_rows({0: tp_log(0), 1: tp_log(1, rec_bad=1, kinds=("fixed", "up"), reps=5)})
    g = S.tp_gates([{"dir": "b/set0", "settings": "TPC_S=64", "rows": bad}], {(2, 32, 2048): e}, None)
    assert not any(g[k]["ok"] for k in ("slowdown", "pf_gate", "receipts", "nccl"))
    assert "rank 1" in g["pf_gate"]["why"] and g["receipts"]["bad"] == ["(2, 32, 2048) repeat 0 rank 1",
                                                                        "(2, 32, 2048) repeat 1 rank 1"]
    sc = S.tp_score([e])
    assert "SCORE_TP" not in sc and len(sc["missing"]) == len(S.TP_SUITE) - 1
    assert sc["by_tp"][2]["coverage"] == pytest.approx(e["coverage"])
    assert e["coverage_v4_form"] == pytest.approx((0.225 * 1e12 + 0.45 * 1.5e12) / (S.cap(2000.0) * 42.0e-3))
