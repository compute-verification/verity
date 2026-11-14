"""share.py's `Model` and its tables on the lines of a made-up dcap run: what the model fits (the read clock, the DRAM
rate, lm_head's and the linears' curves, the attention's line, the step's δ, FA2's efficiency), what it reads back
from the fit (a step's and a prompt's time, and a row of the tables' ratios), a filler beside a step, the τ table,
the decode grid, the serving mixes and the headline's rows, and what `main` prints of them. Each number is chosen so
the fit can be stated by hand."""

from __future__ import annotations

import json
import sys

import pytest
import share as S

MHZ = 2000
PEAK = 1024 * 188 * MHZ * 1e6  # the int8 peak at the read clock, MACs a second


ALONE = {"qkv": 2.0, "o": 1.5, "gate_up": 5.5, "down": 3.0}  # one layer's four linears, each alone, ms
QUARTER = S.Filler([(1, 0.25, 1.25)])  # a filler of a quarter of the peak that slows the step by a quarter
SPLIT = S.Filler([(1, 0.25, 1.25)], dsms=47)  # the same on a split that leaves decode 47 SMs, a quarter of the GPU's
# at 0.88 of those SMs' peak 256 rows' linears and lm_head take 22.68 ms, so a step whose own are 16 ms, slowed by a
# quarter to 20 ms, waits 2.68 ms more
WAIT = 256 * (S.LIN_TOK + S.LM_TOK) / (0.22 * PEAK) * 1e3 - 20.0
TUNED = "dfree-split-w8-nc4-t8 92 SMs tuned"  # the one DRAM-free filler the headline prints


def _lin(mhz, m, ms, lm=None):
    algos = [{"name": name, "ms_alone": t} for name, t in ALONE.items()]
    macs = m * (S.LIN_TOK + (S.LM_TOK if lm else 0))
    return {"mode": "lin", "mhz": mhz, "M": m, "ms": ms, "macs": macs,
            "algos": algos + ([{"name": "lm_head", "ms_alone": lm}] if lm else [])}


def _at(mode, mhz, ctx, ms):
    return {"mode": mode, "mhz": mhz, "M": 16, "L": ctx, "ms": ms}


ROWS = [
    # dcap's two kinds of gate line, which have no clock
    {"mode": "gate", "what": "cublaslt", "m": 16, "ok": True},
    {"mode": "gate", "what": "attention", "B": 16, "L": 2048, "ok": True},
    {"mode": "bw", "mhz": 1993, "gb_per_s": 900.0}, {"mode": "bw", "mhz": 1994, "gb_per_s": 1000.0},
    # out of order; up to 256 rows a line's time holds lm_head's, which is its fifth algorithm
    _lin(1995, 8192, 384.0), _lin(1996, 256, 16.0, lm=4.0), _lin(1997, 1, 2.0, lm=0.5), _lin(1998, 16, 4.0, lm=1.0),
    # 16 rows at L = 2,048 i read 2^31 i bytes of KV: 0.5 ms and 4 ms each, 0.1 ms over at the ends and under between
    _at("att", 1999, 2048, 4.6), _at("att", 2000, 4096, 8.4), _at("att", 2001, 6144, 12.4),
    _at("att", 2002, 8192, 16.6),
    # a whole step: its linears (4 ms), its attention as measured, and 0.7 ms and 0.1 ms more
    _at("step", 2003, 2048, 9.3), _at("step", 2004, 8192, 20.7),
]
# a split's arms: the split, and the whole GPU twice; split only is 9 ms over their mean of 6, and 7.5 over 6
ARMS = {60: {"split": {"ms": 9.0}, "full": {"ms": 4.0}, "full2": {"ms": 8.0}},
        92: {"split": {"ms": 7.5}, "full": {"ms": 5.0}, "full2": {"ms": 7.0}}}
# a fill run, 92 SMs first: shares beside decode of 0.327 and 0.218, which are 0.3 and 0.2 times COMMIT
FILL = [{"shape": "big", "m": 8192, "n": 8192, "k": 8192, "fill_sms": sms, "B": 32, "L": 2048, "alone": {"share": a},
         "window": {"fill_share": w, "fill_gb_per_s": gb}, "slowdown": s, "arms": ARMS[sms]}
        for sms, a, w, gb, s in [(92, 0.45, 0.327, 12.0, 1.1), (60, 0.3, 0.218, 9.0, 1.04)]]
# a dfill run of two splits: re-tuned, the 60-SM one shares more and slows decode past the limit; the 92-SM one has no
# re-tuned arm
DFILL = [{"shape": "dfree-split", "fill_sms": sms, "B": 32, "L": 2048, "place": "split", "alone": {"share": a},
          "window": {"fill_share": w}, "slowdown": s, "arms": ARMS[sms], "rec_checked": n, "rec_bad": bad, **tuned}
         for sms, a, w, s, n, bad, tuned in [
             (60, 0.3, 0.2, 1.04, 64, 1, {"tuned": {"fill_share": 0.25, "slowdown": 1.08, "split_only": 1.02}}),
             (92, 0.45, 0.3, 1.03, 96, 0, {})]]
# an fstep run: a configuration with the pump on 60 SMs, its receipts in three parts, and one on the whole GPU with
# neither parts nor a count of checked receipts, nor the keys a row may leave out
FSTEP = [{"B": 32, "L": 2048, "FW": 4, "NC": 8, "AFW": 2, "ANC": 2, "ALZ": 3, "fill_sms": 60, "lt_mask": 3, "pdl": 0,
          "thr": 0, "slowdown": 1.04, "fused0_vs_base": 1.01, "fill_share": 0.2, "fill_share_lin": 0.15,
          "fill_share_att": 0.05, "fill_share_pump": 0.15, "fill_share_total": 0.35, "out_mismatches": 0,
          "lin": {"rec_checked": 10, "rec_bad": 1}, "att": {"rec_checked": 20, "rec_bad": 0},
          "pump": {"rec_checked": 30, "rec_bad": 0}},
         {"B": 32, "L": 2048, "FW": 8, "NC": 4, "pdl": 1, "thr": 0, "slowdown": 1.02, "fused0_vs_base": 1.006,
          "fill_share": 0.25, "rec_bad": 2, "out_mismatches": 0}]


@pytest.fixture(scope="module")
def m():
    return S.Model(ROWS)


def _att(p, eta):
    return S.causal(p) / (eta * PEAK) * 1e3  # a causal prompt's attention at a share eta of the peak, ms


def _chat(m, batch=32):
    """Chat (a prompt of 1,024 tokens, an answer of 256) at a batch of 32 with FA2: a request's MACs, its prefill and
    one of its steps, ms. The step is at the answer's mean context, 1,152, where the attention's line gives 5 ms;
    from 16 rows to 256 the linears and lm_head both grow as the square root of the rows, from 3 ms and from 1 ms.
    At a batch of 256, the pool's: the linears' measured 16 ms, eight times the attention's bytes (36.5 ms) and 256
    tokens' non-GEMM time."""
    n = 1024 * S.LIN_TOK + S.LM_TOK + S.causal(1024) + 256 * S.decode_n(1, 1152)
    step = {32: 4 * 2 ** 0.5 + 5.0 + 0.4 + 1.09, 256: 16.0 + 36.5 + 0.4 + 38.37 / 32}[batch]
    return n, m.prefill_ms(1024, m.eta_fa2), step


def _run(tmp_path, monkeypatch, capsys, flag, rows):
    """`main` on a run of ROWS and, after `flag`, a filler's run of `rows` in its own directory, in the file the flag
    reads (`--fill` reads `fill.jsonl`): its printed blocks and its JSON."""
    run, filler = tmp_path / "dcap", tmp_path / "filler"
    for d, name, rs in ((run, "dcap", ROWS), (filler, flag[2:], rows)):
        d.mkdir()
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    monkeypatch.setattr(S, "FILLER", {**S.FILLER, "dfree-other": QUARTER, TUNED: QUARTER})
    monkeypatch.setattr(sys, "argv", ["share.py", str(run), flag, str(filler), "--json", str(tmp_path / "share.json")])
    S.main()
    blocks = [b.splitlines() for b in capsys.readouterr().out.rstrip("\n").split("\n\n")]
    return blocks, json.loads((tmp_path / "share.json").read_text())


def test_the_fit_of_a_runs_lines(m):
    assert m.bw == 1e12  # the fastest read
    # of the twelve clocks in order, the one at half the 14 lines: the gate lines are counted, so it is the eighth,
    # one past the median's two (1998 and 1999)
    assert m.mhz == MHZ
    assert m.capacity == PEAK and m.capacity_max == 1024 * 188 * 2430e6
    assert (m.lm_m, m.lm_ms) == ([1, 16, 256], [0.5, 1.0, 4.0])
    assert m.lin_m == [1, 16, 256, 8192] and m.lin_body == [1.5, 3.0, 12.0, 384.0]  # without lm_head
    assert [r["M"] for r in m.lin_rows] == m.lin_m
    assert sorted(m.att) == [(16, 2048), (16, 4096), (16, 6144), (16, 8192)]
    assert sorted(m.step) == [(16, 2048), (16, 8192)]
    # the least-squares line through the four attention points: 4 ms for each 2^31 bytes, and 0.5 ms
    assert m.att_bw == pytest.approx(250 * 2 ** 31) and m.att_c0 == pytest.approx(0.5)
    assert m.delta == pytest.approx(0.4)  # the mean of 0.7 and 0.1
    assert m.eta_fa2 == pytest.approx(S.causal(8192) / (0.07311 * PEAK))  # causal 8,192 in FA2's 73.11 ms


def test_lm_head_and_the_linears_are_read_from_their_curves(m):
    assert m.lm(64) == pytest.approx(2.0)  # halfway in log rows from (16, 1) to (256, 4)
    assert m.lin(1) == 2.0 and m.lin(8192) == 384.0
    assert m.lin(16) == pytest.approx(4.0) and m.lin(256) == pytest.approx(16.0)  # the measured lines, lm_head in
    assert m.lin(64) == pytest.approx(6.0 + 2.0)  # the body halfway from 3 to 12, and lm_head for 64 rows
    assert m.lin(257) == pytest.approx(12.0 * 257 / 256)  # past 256 rows: no lm_head unless asked for
    assert m.lin(512) == pytest.approx(24.0)
    assert m.lin(8192, heads=16) == pytest.approx(385.0)  # a prefill batch of 16 prompts
    assert m.lin(64, heads=0) == pytest.approx(6.0)


def test_a_steps_time_is_the_measured_one_where_there_is_one_and_the_models_elsewhere(m):
    assert m.att_ms(16, 2048) == pytest.approx(4.5) and m.att_ms(512, 2048) == pytest.approx(128.5)
    assert m.step_ms(16, 2048) == pytest.approx(9.3 + 1.09)  # measured, and a step's non-GEMM time
    assert m.step_ms(16, 2048, measured=False) == pytest.approx(4.0 + 4.5 + 0.4 + 1.09)  # linears, attention, δ
    assert m.step_ms(16, 4096) == pytest.approx(4.0 + 8.5 + 0.4 + 1.09)  # no step line there: the model's
    assert m.step_ms(512, 2048) == pytest.approx(24.0 + 128.5 + 0.4 + 38.37 / 16)  # 512 tokens' non-GEMM time


def test_a_prompts_time_packs_prompts_into_a_batch_of_8192_tokens(m):
    assert m.prefill_ms(8192, 0.5) == pytest.approx(384.5 + _att(8192, 0.5) + 38.37)  # one prompt, one lm_head row
    assert m.prefill_ms(512, 0.25) == pytest.approx(385.0 / 16 + _att(512, 0.25) + 38.37 / 16)  # 16 prompts a batch
    assert m.prefill_ms(16384, 0.5) == pytest.approx(768.5 + _att(16384, 0.5) + 76.74)  # longer than a batch: alone


def test_a_rows_ratios_are_of_the_read_clocks_capacity_and_of_the_2430_mhz_one(m):
    r = m.row(0.25 * PEAK * 0.010, 10.0, fill=0.1)  # a quarter of the read clock's capacity for 10 ms
    at_max = 0.25 * MHZ / 2430
    assert r == pytest.approx({"nominal/capacity": 0.25, "nominal/capacity_2430": at_max, "certified/nominal": 0.99,
                               "tau": S.A_V99 * at_max, "certified/capacity": 0.2475,
                               "forced_share": 0.99 * S.A_V99 * at_max, "hidden": 1.0, "filler_share": 0.1})
    busy = m.row(4 * m.capacity_max * 0.010, 10.0)  # four times the 2430 MHz capacity: τ = 4 A
    assert busy["tau"] == pytest.approx(4 * S.A_V99) and busy["filler_share"] == 0.0
    assert busy["hidden"] == pytest.approx((1 - 0.99 * 4 * S.A_V99) / S.A_V99)


def test_without_a_fill_run_the_fillers_are_w11s_on_60_and_on_92_sms():
    assert list(S.FILLER) == ["60 SMs", "92 SMs"]
    f60, f92 = S.FILLER.values()
    assert f60.at(1) == f60.at(256) == (0.222, 1.047) and f92.at(32) == (0.34, 1.106)
    assert (f60.dsms, f92.dsms) == (188 - 60, 188 - 92)  # what the split leaves decode


def test_a_filler_beside_a_step_slows_it_and_adds_its_share_of_the_peak_over_the_slowed_time(m, monkeypatch):
    points = S.PointFiller({(32, 2048): (0.5, 1.0), (32, 8192): (0.25, 1.0)})
    fused = S.StepFiller({(32, 2048): (0.5, 0.0, 0.9)})
    monkeypatch.setattr(S, "FILLER", {**S.FILLER, "quarter": QUARTER, "points": points, "fused": fused})
    assert S.with_filler(m, 256, 20.0, 16.0, "quarter") == pytest.approx((25.0, 0.25 * PEAK * 0.025))
    assert S.with_filler(m, 32, 10.0, 4.0, "points") == pytest.approx((10.0, 0.5 * PEAK * 0.010))
    assert S.with_filler(m, 32, 10.0, 4.0, "points", 8192) == pytest.approx((10.0, 0.25 * PEAK * 0.010))
    # the fused filler runs in the 8.91 ms of the step that are not the non-GEMM kernels' 1.09 ms
    assert S.with_filler(m, 32, 10.0, 4.0, "fused", 2048) == pytest.approx((8.019 + 1.09, 0.5 * PEAK * 0.008019))
    # W11's split leaves decode 128 SMs, where a batch's linears and lm_head take their MACs at 0.88 of those SMs'
    # peak or longer: 8.328 ms for 256 rows
    floor = 256 * (S.LIN_TOK + S.LM_TOK) / (0.88 * 1024 * 128 * MHZ * 1e6) * 1e3
    assert floor == pytest.approx(8.328, abs=5e-4)
    t, macs = S.with_filler(m, 256, 20.0, 16.0, "60 SMs")  # linears of 16 ms, slowed to 16.752: over the floor
    assert t == pytest.approx(20.94) and macs == pytest.approx(0.222 * PEAK * 0.02094)
    t, macs = S.with_filler(m, 256, 20.0, 2.0, "60 SMs")  # linears of 2 ms, slowed to 2.094: the step waits
    assert t == pytest.approx(20.94 + floor - 2.094) and macs == pytest.approx(0.222 * PEAK * t * 1e-3)


def test_the_tau_table_is_each_kernels_share_of_the_peak_and_that_share_priced_at_2430_mhz(m):
    table = S.tau_table(m)
    names, eta = [r["kernel"] for r in table], [r["eta"] for r in table]
    assert names[:4] == [f"{name} m = 8192 (cuBLASLt int8, one layer)" for name in ALONE]
    # `main` reads the fifth and the tenth of the eleven by their place
    assert names[4] == "32 layers of linears, m = 8192 (measured)"
    assert names[9] == "prefill step 8192, int8 linears + FA2 + non-GEMM" and len(table) == 11
    # a kernel alone: its MACs at 8,192 rows in its own time; the 32 layers: the line's MACs in the line's 384 ms
    assert eta[:4] == pytest.approx([8192 * k * n / (ALONE[name] * 1e-3) / PEAK for name, (k, n) in S.LIN.items()])
    assert eta[4] == pytest.approx(8192 * S.LIN_TOK / 0.384 / PEAK)
    assert eta[5:9] == [0.121, 0.443, m.eta_fa2, 0.60]  # T1's two, FA2's as read, the int8 kernel's as assumed
    # a prefill step of 8,192 tokens: its linears with one lm_head, its attention, the non-GEMM kernels
    n = 8192 * S.LIN_TOK + S.LM_TOK + S.causal(8192)
    fa2, int8 = 384.5 + 73.11 + 38.37, 384.5 + _att(8192, 0.60) + 38.37
    assert [r.get("ms") for r in table] == [None] * 9 + [pytest.approx(fa2), pytest.approx(int8)]
    assert eta[9:] == pytest.approx([n / (fa2 * 1e-3) / PEAK, n / (int8 * 1e-3) / PEAK])
    assert [r["tau"] for r in table] == pytest.approx([S.A_V99 * e * MHZ / 2430 for e in eta])


def test_the_grid_is_a_decode_step_at_each_batch_and_context_alone_and_beside_each_filler(m, monkeypatch):
    monkeypatch.setattr(S, "FILLER", {"quarter": QUARTER, "split": SPLIT})
    grid = {(r["B"], r["L"]): r for r in S.grid(m)}
    # six batches at three contexts and the run's two measured points, in order
    assert list(grid) == sorted([(b, ctx) for b in (1, 8, 32, 64, 128, 256) for ctx in (2048, 8192, 32768)]
                                + [(16, 2048), (16, 8192)])
    assert [k for k, r in grid.items() if r["measured"]] == [(16, 2048), (16, 8192)]
    # the pool holds 1,220,703 tokens of KV
    assert [k for k, r in grid.items() if not r["fits"]] == [(64, 32768), (128, 32768), (256, 8192), (256, 32768)]
    for point, ms in [((1, 2048), 2.0 + 0.75 + 0.4 + 1.09), ((16, 2048), 9.3 + 1.09),
                      ((256, 2048), 16.0 + 64.5 + 0.4 + 38.37 / 32)]:
        r, n = grid[point], S.decode_n(*point)
        assert r["ms"] == pytest.approx(ms)
        assert {k: r[k] for k in m.row(n, ms)} == pytest.approx(m.row(n, ms))
        # beside the filler: the step's MACs and the filler's, a quarter of the peak, over a time a quarter longer
        beside = {"ms": 1.25 * ms, **m.row(n + 0.25 * PEAK * 1.25 * ms * 1e-3, 1.25 * ms, 0.25)}
        assert r["filler quarter"] == pytest.approx(beside)
        assert r["filler quarter"]["nominal/capacity"] == pytest.approx(r["nominal/capacity"] / 1.25 + 0.25)
    assert grid[(16, 2048)]["nominal/capacity"] == pytest.approx(16 * 8_041_529_344 / (PEAK * 0.01039))
    # on the split, the floor is of the step's linears and lm_head, not of the step: 16 rows' 4 ms are over theirs
    # (1.42 ms) and the step is the quarter longer; 256 rows' wait
    assert pytest.approx(2.68, abs=5e-3) == WAIT
    assert grid[(16, 2048)]["filler split"] == pytest.approx(grid[(16, 2048)]["filler quarter"])
    n, t = S.decode_n(256, 2048), 1.25 * (16.0 + 64.5 + 0.4 + 38.37 / 32) + WAIT
    assert grid[(256, 2048)]["filler split"] == pytest.approx({"ms": t, **m.row(n + 0.25 * PEAK * t * 1e-3, t, 0.25)})


def test_each_mix_is_at_32_and_at_the_pools_batch_with_each_attention_kernel_and_each_schedule(m, monkeypatch):
    monkeypatch.setattr(S, "FILLER", {})
    rows = S.mixes(m)
    # the larger batch is the whole requests (P + D tokens) that the pool's 1,220,703 tokens hold, 256 at most
    assert [(r["mix"], r["P"], r["D"], r["B"]) for r in rows[::4]] == [
        ("chat", 1024, 256, 32), ("chat", 1024, 256, 256), ("RAG", 8192, 256, 32), ("RAG", 8192, 256, 144),
        ("code / agents", 24576, 512, 32), ("code / agents", 24576, 512, 48),
        ("reasoning", 1024, 8192, 32), ("reasoning", 1024, 8192, 132)]
    assert len(rows) == 32
    assert [(r["attention"], r["schedule"][:8]) for r in rows[:4]] == [
        ("FA2", "separate"), ("FA2", "chunked,"), ("int8 FA", "separate"), ("int8 FA", "chunked,")]
    # chunked, a step carries its B rows of decode and B·P/D tokens of prompt
    assert [r["schedule"] for r in rows[1::4]] == [
        f"chunked, M = {rows_a_step}" for rows_a_step in (160, 1280, 1056, 4752, 1568, 2352, 36, 148)]


def test_a_mix_is_a_requests_prefill_and_its_decode_steps_at_the_answers_mean_context(m, monkeypatch):
    monkeypatch.setattr(S, "FILLER", {"quarter": QUARTER, "split": SPLIT})
    rows = S.mixes(m)
    separate, chunked = (r for r in rows if (r["mix"], r["B"], r["attention"]) == ("chat", 32, "FA2"))
    n, pre, step = _chat(m)
    # a request's 256 steps are each shared by the batch's 32: eight steps' time a request
    assert separate["schedule"] == "separate"
    assert separate["prefill_time_share"] == pytest.approx(pre / (pre + 8 * step))
    assert {k: separate[k] for k in m.row(n, 1.0)} == pytest.approx(m.row(n, pre + 8 * step))
    # the filler is beside decode alone: a quarter of the peak over decode's time, which is a quarter longer
    slowed = pre + 10 * step
    beside = m.row(n + 0.25 * PEAK * 10 * step * 1e-3, slowed, 0.25 * 10 * step / slowed)
    assert separate["filler quarter"] == pytest.approx({**beside, "time": slowed / (pre + 8 * step)})
    # chunked: a step's 32 rows of decode carry 128 tokens of prompt, so its linears have 160 rows, 32 of them with
    # lm_head, and its attention is decode's and an eighth of a prompt's; no filler is beside it
    a_step = 3 * 10 ** 0.5 + 2 ** 0.5 + 5.0 + 0.4 + _att(1024, m.eta_fa2) / 8 + 1.09
    assert chunked["schedule"] == "chunked, M = 160" and chunked["prefill_time_share"] is None
    assert {k: chunked[k] for k in m.row(n, 1.0)} == pytest.approx(m.row(n, 8 * a_step))
    assert "filler quarter" not in chunked
    # at the pool's batch, 256, a request's time is one step's. On the split the step waits for its linears' floor
    separate, chunked = (r for r in rows if (r["mix"], r["B"], r["attention"]) == ("chat", 256, "FA2"))
    n, pre, step = _chat(m, 256)
    assert {k: separate[k] for k in m.row(n, 1.0)} == pytest.approx(m.row(n, pre + step))
    slowed = pre + 1.25 * step + WAIT
    beside = m.row(n + 0.25 * PEAK * (slowed - pre) * 1e-3, slowed, 0.25 * (slowed - pre) / slowed)
    assert separate["filler split"] == pytest.approx({**beside, "time": slowed / (pre + step)})
    # chunked, a step's 256 rows of decode carry 1,024 tokens of prompt: linears of 1,280 rows (60 ms), 256 of them
    # with lm_head (4 ms), decode's attention and a whole prompt's, and 1,280 tokens' non-GEMM time
    a_step = 60.0 + 4.0 + 36.5 + 0.4 + _att(1024, m.eta_fa2) + 38.37 * 1280 / 8192
    assert chunked["schedule"] == "chunked, M = 1280"
    assert {k: chunked[k] for k in m.row(n, 1.0)} == pytest.approx(m.row(n, a_step))


def test_a_mixs_step_is_the_models_where_one_was_measured_and_no_step_is_chunked_past_8192_rows(m, monkeypatch):
    monkeypatch.setattr(S, "FILLER", {})
    monkeypatch.setattr(S, "POOL_TOK", 16 * 3072)  # a pool of 16 requests of 3,072 tokens: one batch, of 16
    monkeypatch.setattr(S, "MIXES", [("short", 1024, 2048), ("at 8,192", 3066, 6), ("past it", 3069, 3)])
    rows = S.mixes(m)
    assert {r["B"] for r in rows} == {16}
    assert [(r["mix"], r["schedule"]) for r in rows if r["attention"] == "FA2"] == [
        ("short", "separate"), ("short", "chunked, M = 24"), ("at 8,192", "separate"),
        ("at 8,192", "chunked, M = 8192"), ("past it", "separate")]
    assert len(rows) == 10
    # the short mix's mean context is 2,048, where the run measured a step of 16 rows: 9.3 ms, and the model's is 8.9
    n = 1024 * S.LIN_TOK + S.LM_TOK + S.causal(1024) + 2048 * S.decode_n(1, 2048)
    pre, step = m.prefill_ms(1024, m.eta_fa2), 4.0 + 4.5 + 0.4 + 1.09
    assert rows[0]["prefill_time_share"] == pytest.approx(pre / (pre + 128 * step))
    assert rows[0]["nominal/capacity"] == pytest.approx(n / (PEAK * (pre + 128 * step) * 1e-3))


def test_the_headline_is_four_decode_points_and_five_mixes_alone_and_beside_each_named_filler(m, monkeypatch):
    monkeypatch.setattr(S, "FILLER", {"quarter": QUARTER, "unnamed": S.Filler([(1, 0.5, 2.0)])})
    res = {"grid": S.grid(m), "mixes": S.mixes(m)}
    head = S.headline(res, ["quarter"])
    assert [h["row"] for h in head] == [
        "decode B = 1, L = 2K", "decode B = 32, L = 8K", "decode B = 128, L = 2K", "decode B = 256, L = 2K",
        "chat, B = 32", "chat, B = 256", "RAG, B = 144", "code / agents, B = 48", "reasoning, B = 132"]
    assert all(list(h) == ["row", "N/C", "cert/C", "quarter"] for h in head)
    assert all(list(h) == ["row", "N/C", "cert/C"] for h in S.headline(res, []))
    # a decode point: the grid's, and the time beside the filler over the time alone
    ms, n = 16.0 + 64.5 + 0.4 + 38.37 / 32, S.decode_n(256, 2048)
    alone, beside = n / (PEAK * ms * 1e-3), n / (PEAK * 1.25 * ms * 1e-3) + 0.25
    assert (head[3]["N/C"], head[3]["cert/C"]) == pytest.approx((alone, 0.99 * alone))
    assert head[3]["quarter"] == pytest.approx({"N/C": beside, "cert/C": 0.99 * beside, "time": 1.25})
    # a mix: its row with prefill apart and FA2, and the mix's time beside the filler over its time alone
    n, pre, step = _chat(m)
    alone, slowed = n / (PEAK * (pre + 8 * step) * 1e-3), pre + 10 * step
    beside = (n + 0.25 * PEAK * 10 * step * 1e-3) / (PEAK * slowed * 1e-3)
    assert (head[4]["N/C"], head[4]["cert/C"]) == pytest.approx((alone, 0.99 * alone))
    assert head[4]["quarter"] == pytest.approx({"N/C": beside, "cert/C": 0.99 * beside,
                                                "time": slowed / (pre + 8 * step)})
    # and the same mix at its other batch is its own row
    n, pre, step = _chat(m, 256)
    assert head[5]["N/C"] == pytest.approx(n / (PEAK * (pre + step) * 1e-3))
    assert head[5]["quarter"]["time"] == pytest.approx((pre + 1.25 * step) / (pre + step))


def test_a_cell_is_three_decimals_below_ten_one_decimal_from_ten_and_a_dash_for_nothing():
    cells = [S.fmt(x) for x in (None, 0.25, 9.9994, 10, 10.96, 1042.1)]
    assert cells == ["–", "0.250", "9.999", "10.0", "11.0", "1042.1"]
    assert S.fmt_in(None, "tau") == "–" and S.fmt_in({"tau": 0.5, "eta": 12.0}, "tau") == "0.500"


def test_main_prints_the_fit_and_each_table_and_writes_them_as_json(m, tmp_path, monkeypatch, capsys):
    """A run's directory holds `dcap.jsonl`. With no run of a filler named, the fillers are the module's own, and of
    the DRAM-free ones the headline prints the tuned 92-SM split alone. The numbers are the tables' (the tests above),
    as `fmt` writes them."""
    (tmp_path / "dcap.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS) + "\n")
    monkeypatch.setattr(S, "FILLER", {**S.FILLER, "dfree-other": QUARTER, TUNED: QUARTER})
    monkeypatch.setattr(sys, "argv", ["share.py", str(tmp_path), "--json", str(tmp_path / "share.json")])
    S.main()
    blocks = [b.splitlines() for b in capsys.readouterr().out.rstrip("\n").split("\n\n")]
    fit, _, head, filler, tau, grid, mixes, _, ladder = blocks
    # 1024 × 188 MACs a clock at 2,000 and at 2,430 MHz; 250 × 2³¹ B/s; the steps of 9.3 and 20.7 ms modelled as 8.9
    # and 20.9
    assert fit[0] == ("read clock 2000 MHz, C = 385.0 TMAC/s (C_2430 = 467.8); DRAM read 1000 GB/s; attention fit "
                      "537 GB/s + 0.500 ms; step δ 0.400 ms; FA2 η 0.313; A(V99) 0.2068; pool 1220703 tokens")
    assert fit[1:] == ["step model vs measured: worst 4.3%"]
    assert [len(t) - 2 for t in (head, tau, grid, mixes, ladder)] == [9, 11, 20, 32, 10]
    assert head[0] == ("| phase or mix | N/C | cert/C | 60 SMs: N/C, cert/C, time | 92 SMs: N/C, cert/C, time | "
                       f"{TUNED}: N/C, cert/C, time |")
    assert head[1] == "|---|---|---|" + "---|" * 3
    # 0.065 alone; beside a filler of share f that slows it by s, 0.065 / s + f
    assert head[5] == ("| decode B = 256, L = 2K | 0.065 | 0.064 | 0.284, 0.281, 1.047 | 0.399, 0.395, 1.106 | "
                       "0.302, 0.299, 1.250 |")
    assert head[7] == ("| chat, B = 256 | 0.223 | 0.221 | 0.331, 0.328, 1.024 | 0.390, 0.386, 1.053 | "
                       "0.337, 0.334, 1.125 |")
    assert filler == ["filler: W11 (assumed)"]
    assert tau[:2] == ["| kernel | η (read clock) | τ (V99) |", "|---|---|---|"]
    assert tau[6] == "| 32 layers of linears, m = 8192 (measured) | 0.387 | 0.066 |"
    assert grid[0] == "| B | L | fits | step ms | N/C | τ | cert/C | forced | +60 SMs N/C | τ | +92 SMs N/C | τ |"
    assert grid[8] == "| 16 | 2048 | yes | 10.4 | 0.032 | 0.005 | 0.032 | 0.005 | 0.253 | 0.043 | 0.369 | 0.063 |"
    assert grid[20] == ("| 256 | 8192 | no (model) | 274.1 | 0.023 | 0.004 | 0.023 | 0.004 | 0.244 | 0.042 | 0.361 | "
                        "0.061 |")
    assert mixes[0] == ("| mix | P / D | B | attention | schedule | prefill time | N/C | τ | cert/C | forced | "
                        "+60 SMs N/C | τ | +92 SMs N/C | τ |")
    assert mixes[2] == ("| chat | 1024 / 256 | 32 | FA2 | separate | 0.357 | 0.159 | 0.027 | 0.158 | 0.027 | 0.300 | "
                        "0.051 | 0.376 | 0.064 |")
    assert mixes[3] == ("| chat | 1024 / 256 | 32 | FA2 | chunked, M = 160 | – | 0.172 | 0.029 | 0.170 | 0.029 | "
                        "– | – | – | – |")
    # the ladder: a row's τ under each vector, A / A(V99) times its own, and what that leaves hidden
    assert ladder[0] == "| row | " + " | ".join(f"{what} (A = {a:.3f})" for what, a in S.A_LADDER) + " |"
    assert [line.split(" | ")[0] for line in ladder[2:]] == [
        "| linears m = 8192", "| prefill step 8192 (FA2)", "| decode 32 / 8192", "| decode 32 / 8192 + 92-SM filler",
        "| decode 256 / 2048", "| decode 256 / 2048 + 92-SM filler", "| chat B = 256", "| chat B = 256 + 92-SM filler",
        "| reasoning B = 132", "| reasoning B = 132 + 92-SM filler"]
    assert ladder[2] == ("| linears m = 8192 | 0.066 (1.000) | 0.111 (1.000) | 0.163 (1.000) | 0.180 (1.000) | "
                         "0.272 (0.857) | 0.318 (0.685) |")
    assert ladder[5] == ("| decode 32 / 8192 + 92-SM filler | 0.061 (1.000) | 0.103 (1.000) | 0.151 (1.000) | "
                         "0.167 (1.000) | 0.252 (0.880) | 0.295 (0.708) |")
    # which rows the others are: the τ table's prefill step with FA2, and each mix at its batch, separate, with FA2
    assert ladder[3] == ("| prefill step 8192 (FA2) | 0.059 (1.000) | 0.099 (1.000) | 0.145 (1.000) | 0.161 (1.000) | "
                         "0.243 (0.891) | 0.284 (0.719) |")
    assert ladder[8] == ("| chat B = 256 | 0.038 (1.000) | 0.064 (1.000) | 0.094 (1.000) | 0.104 (1.000) | "
                         "0.157 (0.991) | 0.184 (0.818) |")
    assert ladder[9] == ("| chat B = 256 + 92-SM filler | 0.066 (1.000) | 0.112 (1.000) | 0.164 (1.000) | "
                         "0.181 (1.000) | 0.274 (0.854) | 0.321 (0.682) |")
    assert ladder[10] == ("| reasoning B = 132 | 0.006 (1.000) | 0.010 (1.000) | 0.014 (1.000) | 0.016 (1.000) | "
                          "0.024 (1.000) | 0.028 (0.972) |")
    res = json.loads((tmp_path / "share.json").read_text())
    assert (res["read_mhz"], res["dram_read_gb_s"], res["pool_tokens"], res["filler"]) == (
        2000, 1000.0, 1220703, "W11 (assumed)")
    assert res["att_fit"] == pytest.approx({"gb_s": 250 * 2 ** 31 / 1e9, "c0_ms": 0.5})
    assert (res["step_delta_ms"], res["eta_fa2"], res["A_v99"]) == pytest.approx((0.4, m.eta_fa2, S.A_V99))
    assert res["model_check"] == [{"B": 16, "L": 2048, "measured": 9.3, "model": pytest.approx(8.9)},
                                  {"B": 16, "L": 8192, "measured": 20.7, "model": pytest.approx(20.9)}]
    assert [(len(res[k]), res[k][0]) for k in ("tau", "grid", "mixes")] == [
        (11, S.tau_table(m)[0]), (20, S.grid(m)[0]), (32, S.mixes(m)[0])]
    assert res["a_ladder"] == [{"what": what, "A": a} for what, a in S.A_LADDER]
    # every filler is in the JSON's headline, the unprinted one too
    columns = ["row", "N/C", "cert/C", "60 SMs", "92 SMs", "dfree-other", TUNED]
    assert [list(h) for h in res["headline"]] == [columns] * 9
    assert res["fill"] == res["dfill"] == res["fstep"] == []
    assert res["adaptive"] == {"as tuned for the whole GPU": {}, "tuned": {}}
    assert res["fstep_pick"] == {"fused": {}, "fused + pump": {}}
    assert [r["row"] for r in res["tau_ladder"]] == [line.split(" | ")[0][2:] for line in ladder[2:]]
    top = 8192 * S.LIN_TOK / 0.384 / (1024 * 188 * 2430e6)  # the linears' MACs a second over the peak at 2,430 MHz
    assert res["tau_ladder"][0]["no CUDA-core product route"] == pytest.approx({"tau": top, "hidden": 0.685}, abs=5e-4)
    # under a vector of A below one τ is A times that, and what stays hidden is (1 − 0.99 τ) / A
    a = 256 / 300.08
    assert res["tau_ladder"][0]["product routes only, A = 256/R*"] == pytest.approx(
        {"tau": a * top, "hidden": (1 - 0.99 * a * top) / a})


def test_main_with_a_fill_run_prints_its_table_and_its_two_fillers_replace_the_modules(tmp_path, monkeypatch, capsys):
    """`--fill DIR` reads `DIR/fill.jsonl`: its table, a line a run of the chain in order of SMs, and its fillers at
    L = 2,048, each its share beside decode over COMMIT, in place of every filler the module had."""
    blocks, res = _run(tmp_path, monkeypatch, capsys, "--fill", FILL)
    fill, _, head, filler = blocks[1:5]
    assert fill[0] == ("| chain | filler SMs | B | L | alone share | share beside decode | certified (÷ COMMIT) | "
                       "decode slowdown | split only | filler GB/s |")
    assert fill[2:] == ["| big 8192×8192×8192 | 60 | 32 | 2048 | 0.300 | 0.218 | 0.200 | 1.040 | 1.500 | 9 |",
                        "| big 8192×8192×8192 | 92 | 32 | 2048 | 0.450 | 0.327 | 0.300 | 1.100 | 1.250 | 12 |"]
    assert list(S.FILLER) == ["60 SMs", "92 SMs"]
    assert S.FILLER["60 SMs"].at(256) == pytest.approx((0.2, 1.04))
    assert head[0] == "| phase or mix | N/C | cert/C | 60 SMs: N/C, cert/C, time | 92 SMs: N/C, cert/C, time |"
    # 0.065 alone, as without a fill run; beside the 60-SM filler 0.065 / 1.04 + 0.2, beside the 92's 0.065 / 1.1 + 0.3
    assert head[5] == "| decode B = 256, L = 2K | 0.065 | 0.064 | 0.263, 0.260, 1.040 | 0.359, 0.356, 1.100 |"
    assert filler == ["filler: measured (--fill)"]
    assert (res["filler"], res["fill"], res["dfill"], res["fstep"]) == ("measured (--fill)", FILL, [], [])


def test_main_with_a_dfill_run_prints_its_table_and_the_adaptive_split_as_tuned_and_re_tuned(tmp_path, monkeypatch,
                                                                                              capsys):
    """`--dfill DIR` reads `DIR/dfill.jsonl`: its table, a line a split, and the adaptive split's pick at each point,
    first with cuBLASLt as tuned for the whole GPU, then re-tuned on the partition. Both picks join the headline."""
    blocks, res = _run(tmp_path, monkeypatch, capsys, "--dfill", DFILL)
    dfill, as_tuned, re_tuned, _, head = blocks[1:6]
    assert dfill[0] == ("| DRAM-free filler | B | L | alone share | share beside decode | decode slowdown | "
                        "split only | tuned: share, slowdown, split only | receipts checked (bad) |")
    assert dfill[2] == ("| dfree-split 60 SMs | 32 | 2048 | 0.300 | 0.200 | 1.040 | 1.500 | 0.250, 1.080, 1.020 | "
                        "64 (1) |")
    assert dfill[3] == "| dfree-split 92 SMs | 32 | 2048 | 0.450 | 0.300 | 1.030 | 1.250 | – | 96 (0) |"
    assert len(dfill) == 4
    # within 5%, as tuned the 92-SM split shares the most; re-tuned only the 60-SM one was measured, past the limit
    assert as_tuned == ["adaptive split ≤ 5%: B 32 L 2048: 92 SMs, share 0.300, slowdown 1.030"]
    assert re_tuned == [("adaptive split ≤ 5%, cuBLASLt tuned on the partition: B 32 L 2048: 0 SMs, share 0.000, "
                         "slowdown 1.000")]
    assert res["adaptive"] == {"as tuned for the whole GPU": {"32:2048": [92, 0.3, 1.03]},
                               "tuned": {"32:2048": [0, 0.0, 1.0]}}
    # of the DRAM-free fillers the headline prints the tuned 92-SM split and the two picks; with no split, decode alone
    assert [c.split(":")[0] for c in head[0].split(" | ")[3:]] == [
        "60 SMs", "92 SMs", TUNED, "dfree adaptive ≤ 5%", "dfree adaptive ≤ 5% tuned"]
    assert head[5].endswith(" | 0.363, 0.360, 1.030 | 0.065, 0.064, 1.000 |")  # 0.065 / 1.03 + 0.3
    assert (res["filler"], res["dfill"]) == ("W11 (assumed)", DFILL)


def test_main_with_an_fstep_run_prints_its_table_and_its_pick_on_the_whole_gpu_and_with_the_pump(tmp_path, monkeypatch,
                                                                                                 capsys):
    """`--fstep DIR` reads `DIR/fstep.jsonl`: its table, a line a configuration, and the fused filler's pick at each
    point, first on the whole GPU, then with the pump on a split's other SMs. A row's receipts are its parts' where it
    has parts, and a key it leaves out reads as its default."""
    blocks, res = _run(tmp_path, monkeypatch, capsys, "--fstep", FSTEP)
    fstep, whole, pump = blocks[1:4]
    assert fstep[0].startswith("| fused step | B | L | FW | AFW | pump SMs | cuBLASLt mask | slowdown | split only |")
    # receipts: the parts' 10 + 20 + 30 checked and 1 bad; with no parts, the row's own 2 bad and none checked
    assert fstep[2] == ("| NC 8, no PDL | 32 | 2048 | 4 | 2 NC 2 lazy 3 | 60 | 3 | 1.040 | 1.010 | "
                        "0.200 (0.150 + 0.050) | 0.150 | 0.350 | 60 (1) | 0 |")
    assert fstep[3] == ("| NC 4 | 32 | 2048 | 8 | 0 | 0 | 0 | 1.020 | 1.006 | 0.250 (0.250 + 0.000) | 0.000 | 0.250 | "
                        "0 (2) | 0 |")
    assert len(fstep) == 4
    # within 5%, the whole GPU has the second configuration only; with the pump the first's 0.35 is the most
    assert whole == [("fstep pick ≤ 5%, whole GPU: B 32 L 2048: FW 8 AFW 0 pump 0 LT 0, filler 0.250 + 0.000, "
                      "slowdown 1.020")]
    assert pump == [("fstep pick ≤ 5% with the pump: B 32 L 2048: FW 4 AFW 2 NC 2 lazy 3 pump 60 LT 3, "
                     "filler 0.200 + 0.150, slowdown 1.040")]
    assert res["fstep_pick"] == {
        "fused": {"32:2048": [[32, 2048, 8, 4, 0, 0, 0, 1, 0, 0, 4, 0, 0], 0.25, 0.0, 1.02]},
        "fused + pump": {"32:2048": [[32, 2048, 4, 8, 2, 60, 3, 0, 0, 0, 2, 0, 3], 0.2, 0.15, 1.04]}}
    assert list(S.FILLER)[-2:] == ["fused ≤ 5%", "fused + pump ≤ 5%"]
    assert res["fstep"] == FSTEP

