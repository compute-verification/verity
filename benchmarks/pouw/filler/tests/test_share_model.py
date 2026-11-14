"""share.py's `Model` and its tables on the lines of a made-up dcap run: what the model fits (the read clock, the DRAM
rate, lm_head's and the linears' curves, the attention's line, the step's δ, FA2's efficiency), what it reads back
from the fit (a step's and a prompt's time, and a row of the tables' ratios), a filler beside a step, the τ table,
the decode grid, the serving mixes and the headline's rows. Each number is chosen so the fit can be stated by hand."""

from __future__ import annotations

import pytest
import share as S

MHZ = 2000
PEAK = 1024 * 188 * MHZ * 1e6  # the int8 peak at the read clock, MACs a second


ALONE = {"qkv": 2.0, "o": 1.5, "gate_up": 5.5, "down": 3.0}  # one layer's four linears, each alone, ms
QUARTER = S.Filler([(1, 0.25, 1.25)])  # a filler of a quarter of the peak that slows the step by a quarter


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


@pytest.fixture(scope="module")
def m():
    return S.Model(ROWS)


def _att(p, eta):
    return S.causal(p) / (eta * PEAK) * 1e3  # a causal prompt's attention at a share eta of the peak, ms


def _chat(m):
    """Chat (a prompt of 1,024 tokens, an answer of 256) at a batch of 32 with FA2: a request's MACs, its prefill and
    one of its steps, ms. The step is at the answer's mean context, 1,152, where the attention's line gives 5 ms;
    from 16 rows to 256 the linears and lm_head both grow as the square root of the rows, from 3 ms and from 1 ms."""
    n = 1024 * S.LIN_TOK + S.LM_TOK + S.causal(1024) + 256 * S.decode_n(1, 1152)
    return n, m.prefill_ms(1024, m.eta_fa2), 4 * 2 ** 0.5 + 5.0 + 0.4 + 1.09


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
    monkeypatch.setattr(S, "FILLER", {"quarter": QUARTER})
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
    monkeypatch.setattr(S, "FILLER", {"quarter": QUARTER})
    separate, chunked = (r for r in S.mixes(m) if (r["mix"], r["B"], r["attention"]) == ("chat", 32, "FA2"))
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

