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
    reads 7.69 or 7.75 ms by process, r20261016-071637-cb1f)."""
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
#: pfill-k's tile as cursor/hc-proved-ea24@fe2fdb5bf compiles it, abridged: fresh s32 words, folded by rotations
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
    assert S.SCORER == S.V3 and all(e["active"] and e["status"] == "proved" for e in S.FILLER_ENTRIES.values())
    stats = all_points()
    sc = {n: S.score(stats, S.certified(n)) for n in S.FILLER_ENTRIES}
    assert sc["chained"]["SCORE"] == sc["unproven"]["SCORE"] < sc["theorem-e"]["SCORE"] < sc["theorem-k"]["SCORE"]
    assert all(s["flags"] == [] for s in sc.values())
    ch = S.score(stats, S.certified(S.UNDER_CH))  # SCORE under CH is v2's SCORE, every filler at 0.99
    assert ch["SCORE"] == pytest.approx(S.score(stats)["SCORE"]) and ch["SCORE"] > sc["theorem-k"]["SCORE"]
    assert ch["flags"] == ["filler certified fraction 0.99 is under CH (withdrawn, decision 27)"]


def test_record_limits_are_checked_when_the_arm_loads():
    """filler-scorer/v3 refuses, before any build or lease, an arm whose record the ledger or the store would refuse
    after the run (HC-PROVED's 249-character hypothesis reached `score.record` only after a scored run)."""
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
