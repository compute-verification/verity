"""h200_step.py: the H200 step at M = 16,384 from `v31 fill-bench`'s JSON lines, and its refusal of an incomplete step;
h200_pipes.py: each pipe's duty over the overlapped call."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import h200_step


def _fill_line(n: int, k: int, served: bool, L: int = 4096, m: int = 16384) -> str:
    ms = {"lt": 1.0, "coins": 0.1, "coins_w": 0.05, "form_a": 0.02, "form_b": 0.03, "gemm": 1.5, "commit": 0.5, "call": 2.2,
          "ovl": 2.0, "ovl_w": 3.0} | ({} if served else {"pad": 0.2, "ovl_pad": 4.0})
    arms = {a: {"ms": x, "mhz": 1900} for a, x in ms.items()}
    return json.dumps({"fill": True, "shape": f"s-{n}-{k}", "m": m, "n": n, "k": k, "lp": 256, "L": L, "row": 16384,
                       "o": served, "arms": arms, "gamma_ovl": 2.0})


def test_the_no_fold_step_takes_a_served_calls_per_call_weight_side_and_a_fillers_pad(tmp_path, capsys, monkeypatch):
    """fill-bench's lines: a served call's step is its ovl_w (seeds per call, so the weight side every call), a filler's
    its ovl_pad; the split is a serial call's tensor (gemm), hashing (coins, coins_w, pad, commit) and other (form)."""
    f = tmp_path / "fill-bench-row16384-served1-L4096.jsonl"
    f.write_text("\n".join(_fill_line(n, k, True) for n, k in h200_step.CALLS) + "\n")
    g = tmp_path / "fill-bench-row16384-L4096.jsonl"
    g.write_text("\n".join(_fill_line(n, k, False) for n, k in h200_step.CALLS) + "\n")
    st = h200_step.fill_steps(h200_step.rows([str(f), str(g)]))
    served, filler = st[(256, 4096, 16384, True, "row16384-served1-L4096")], st[(256, 4096, 16384, False, "row16384-L4096")]
    assert (served["record"], filler["record"]) == ("ovl_w", "ovl_pad")
    assert served["step"] == pytest.approx(261 * 3e-3) and filler["step"] == pytest.approx(261 * 4e-3)
    assert served["s"]["pad"] is None
    assert served["tensor"] == pytest.approx(261 * 1.5e-3)
    assert served["hashing"] == pytest.approx(261 * 0.65e-3) and filler["hashing"] == pytest.approx(261 * 0.85e-3)
    assert served["other"] == pytest.approx(261 * 0.05e-3)
    assert served["macs"] == 16384 * (104 * 16384**2 + 52 * 128 * 16384 + 104 * 65536 * 16384 + 256 * 16384)
    assert served["mhz"] == (1900, 1900)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f), "--wgmma", "3.06"])
    assert h200_step.main() == 0
    out = capsys.readouterr().out
    assert ("summary: mqa16k52 no-fold served L 4096 L' 256 row 16384 row16384-served1-L4096: step 0.783 s (ovl_w), "
            "gamma_u 0.256 against wgmma's 3.06 s, gamma_lt 3.000 against cuBLASLt's 0.261 s") in out


def test_a_missing_call_or_another_m_gives_no_step(tmp_path, capsys, monkeypatch):
    calls = list(h200_step.CALLS)
    f = tmp_path / "fill-bench-row16384.jsonl"
    f.write_text("\n".join([_fill_line(n, k, True) for n, k in calls[1:]] + [_fill_line(*calls[0], True, m=8192)]) + "\n")
    assert h200_step.fill_steps(h200_step.rows([str(f)])) == {}
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 1
    assert "summary: no complete step" in capsys.readouterr().out


def test_the_no_fold_step_is_against_the_lines_own_wgmma_linear(tmp_path, capsys, monkeypatch):
    """A -DV31_WG build's served lines time wgmma.linear (`wgl`): the 8B step's gamma_u is against its step from them."""
    calls = h200_step.MODELS["8b"]
    lines = [json.loads(_fill_line(n, k, True)) for n, k in calls]
    for x in lines:
        x["arms"]["wgl"] = {"ms": 1.2, "mhz": 1980}
        x["lp"] = 512
    f = tmp_path / "fill-bench-served1-gemm1-rp4.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    st = h200_step.fill_steps(h200_step.rows([str(f)]), calls=calls)
    assert st[(512, 4096, 16384, True, "served1-gemm1-rp4")]["s"]["wgl"] == pytest.approx(128 * 1.2e-3)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 0
    assert ("summary: 8b no-fold served L 4096 L' 512 row 16384 served1-gemm1-rp4: step 0.384 s (ovl_w), gamma_u 2.500 "
            "against wgmma.linear's 0.154 s (these lines' wgl), gamma_lt 3.000 against cuBLASLt's 0.128 s") in capsys.readouterr().out


def test_a_split_gives_each_ns_step_its_gemms_slowdown_the_hashing_hidden_and_each_calls_best_n(tmp_path, capsys,
                                                                                                    monkeypatch):
    """fill-bench's split arms: each N's step from ovl_w_gN over every call, the GEMM alone on N SMs over the whole GPU's,
    the commit alone on the rest, the hashing hidden (the serial split less the step, over its hashing); each call picks
    its fastest N; an N some call lacks is left out."""
    calls = h200_step.MODELS["8b"]
    lines = [json.loads(_fill_line(n, k, True)) for n, k in calls]
    for x in lines:
        g64 = 2.1 if (x["n"], x["k"]) == (4096, 14336) else 1.8
        x["lp"] = 512
        x["gsm"], x["gsms"] = 64, [{"n": 0, "sms": [132, 132]}, {"n": 64, "sms": [64, 68]}, {"n": 80, "sms": [80, 52]}]
        x["arms"] |= {"wgl": {"ms": 1.25, "mhz": 1980}, "ovl_w_g0": {"ms": 2.0, "mhz": 1980},
                      "ovl_w_g64": {"ms": g64, "mhz": 1980}, "gemm_g64": {"ms": 2.4, "mhz": 1980},
                      "commit_g64": {"ms": 1.0, "mhz": 1980}}
    lines[0]["arms"]["ovl_w_g80"] = {"ms": 1.0, "mhz": 1980}
    f = tmp_path / "fill-bench-served1-gemm1-gsm64-L4096.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    sp = h200_step.split_steps(h200_step.rows([str(f)]), calls=calls)[(512, 4096, 16384, True, "served1-gemm1-gsm64-L4096")]
    assert set(sp["n"]) == {0, 64}
    n64 = sp["n"][64]
    assert n64["step"] == pytest.approx(32 * 7.5e-3) and n64["sms"] == (64, 68)
    assert n64["gemm"] == pytest.approx(128 * 2.4e-3) and n64["commit"] == pytest.approx(128 * 1e-3)
    assert n64["slowdown"] == pytest.approx(1.6) and n64["hidden"] == pytest.approx(0.5)
    assert sp["n"][0]["gemm"] is None and sp["n"][0]["hidden"] == pytest.approx(0.2 / 0.65)
    assert sp["pick"] == {"s-6144-4096": 64, "s-4096-4096": 64, "s-28672-4096": 64, "s-4096-14336": 0}
    assert sp["best"] == pytest.approx(32 * 7.4e-3) and sp["best_hidden"] == pytest.approx(1.4 / 2.6)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 0
    out = capsys.readouterr().out
    assert "| 512 | 4096 | 16384 | served | served1-gemm1-gsm64-L4096 | 0 | every SM | 0.256 | 1.600 | - | - | - | 30.8% |" in out
    assert ("| 512 | 4096 | 16384 | served | served1-gemm1-gsm64-L4096 | 64 | 64 + 68 | 0.240 | 1.500 | 0.307 | 1.600 | "
            "0.128 | 50.0% |") in out
    assert ("summary: 8b no-fold served split L 4096 L' 512 row 16384 served1-gemm1-gsm64-L4096: best N 64 step 0.240 s, "
            "gamma_u 1.500, hashing hidden 50.0%; each call's N (s-6144-4096 64, s-4096-4096 64, s-28672-4096 64, "
            "s-4096-14336 0) step 0.237 s, gamma_u 1.480, hashing hidden 53.8%; serial 0.282 s (hashing 0.083 s), gamma_u "
            "against these lines' wgl 0.160 s") in out


def test_a_served_order_step_is_against_its_variants_fill_bench_lines(tmp_path, capsys, monkeypatch):
    """fill-step's lines: the model is the one whose calls the line's shapes' are; each N's step is its step_gN, gamma_u
    against the same variant's fill-bench wgl, the hashing hidden their serial split less the step, over its hashing, and
    beside it their step at that N; a line whose calls are no model's prints under its spec, with no reference."""
    calls = h200_step.MODELS["8b"]
    lines = [json.loads(_fill_line(n, k, True)) for n, k in calls]
    for x in lines:
        x["lp"] = 512
        x["gsms"] = [{"n": 0, "sms": [132, 132]}, {"n": 64, "sms": [64, 68]}]
        x["arms"] |= {"wgl": {"ms": 1.25, "mhz": 1980}, "ovl_w_g0": {"ms": 2.0, "mhz": 1980},
                      "ovl_w_g64": {"ms": 1.8, "mhz": 1980}}
    f = tmp_path / "fill-bench-served1-gemm1-L4096.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    names = {(6144, 4096): "qkv", (4096, 4096): "o", (28672, 4096): "gu", (4096, 14336): "down"}

    def step(spec: str, per: dict[tuple[int, int], int]) -> str:
        return json.dumps({"fill_step": True, "step": spec, "layers": 32, "lp": 512, "L": 4096, "row": 16384, "o": True,
                           "steady": True, "shapes": [{"shape": names[nk], "m": 16384, "n": nk[0], "k": nk[1], "calls": c}
                                                      for nk, c in per.items()],
                           "gsms": [{"n": 0, "sms": [132, 132]}, {"n": 64, "sms": [64, 68]}],
                           "arms": {"step_g0": {"ms": 250.0, "mhz": 1700, "calls": 1},
                                    "step_g64": {"ms": 224.0, "mhz": 1900, "calls": 1}}})

    g = tmp_path / "fill-step-served1-gemm1-L4096.jsonl"
    g.write_text(step("qkv,o,gu,down*32", calls) + "\n" + step("qkv*32", {(6144, 4096): 32}) + "\n")
    so = h200_step.served_order(h200_step.step_rows([str(f), str(g)]), h200_step.rows([str(f), str(g)]))
    assert [v["model"] for v in so] == ["8b", None]
    n0, n64 = so[0]["n"][0], so[0]["n"][64]
    assert so[0]["ref"] == pytest.approx(128 * 1.25e-3) and so[0]["serial"] == pytest.approx(128 * 2.2e-3)
    assert n64["step"] == pytest.approx(0.224) and n64["sms"] == (64, 68) and n64["mhz"] == 1900
    assert n64["gamma_u"] == pytest.approx(1.4) and n64["apart"] == pytest.approx(128 * 1.8e-3)
    assert n64["hidden"] == pytest.approx((0.2816 - 0.224) / 0.0832)
    assert n0["apart"] == pytest.approx(128 * 2e-3) and n0["hidden"] == pytest.approx((0.2816 - 0.25) / 0.0832)
    assert so[1]["ref"] is None and so[1]["n"][64]["gamma_u"] is None and so[1]["n"][64]["hidden"] is None
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f), str(g)])
    assert h200_step.main() == 0
    out = capsys.readouterr().out
    assert ("| 8b | 512 | 4096 | 16384 | served | served1-gemm1-L4096 | 64 | 64 + 68 | 0.224 | 1.400 | 0.230 | 0.282 | "
            "69.2% | 1900 |") in out
    assert "| qkv*32 | 512 | 4096 | 16384 | served | served1-gemm1-L4096 | 0 | every SM | 0.250 | - | - | - | - | 1700 |" in out
    assert ("summary: 8b served order served L 4096 L' 512 row 16384 served1-gemm1-L4096 N 64: step 0.224 s, gamma_u "
            "1.400, hashing hidden 69.2%; fill-bench's step at N 64 0.230 s; serial 0.282 s (hashing 0.083 s), gamma_u "
            "against these lines' wgl 0.160 s") in out
    assert h200_step.unsplit("served1-gemm1-row16384-gsm80-gsms0_64_80-gcalls32-item200-gsmby4096x14336:80-L4096") == (
        "served1-gemm1-row16384-gsmby4096x14336:80-L4096")
    h = tmp_path / "fill-step-served1-gemm1-gsm80-gsms0_64-gcalls32-L4096.jsonl"
    h.write_text(step("qkv,o,gu,down*32", calls) + "\n")
    so = h200_step.served_order(h200_step.step_rows([str(h)]), h200_step.rows([str(f)]))
    assert so[0]["ref_variant"] == "served1-gemm1-L4096" and so[0]["n"][64]["gamma_u"] == pytest.approx(1.4)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f), str(h)])
    assert h200_step.main() == 0
    assert ("served1-gemm1-gsm80-gsms0_64-gcalls32-L4096 N 64: step 0.224 s, gamma_u 1.400, hashing hidden 69.2%; "
            "fill-bench's step at N 64 0.230 s; serial 0.282 s (hashing 0.083 s), gamma_u against "
            "served1-gemm1-L4096's wgl 0.160 s") in capsys.readouterr().out


def test_a_step_at_l_equal_k_takes_each_calls_line_at_its_own_k():
    """fill-step at L 0 (L = K) against fill-bench's lines, which record each call's own L (its K): every call at L >= K
    stands in the L 0 group, so the step has its reference; a run at L 4,096, where down (K 14,336) is at S 4, keeps its
    group and has no L 0 one."""
    calls = h200_step.MODELS["8b"]
    at_k = [json.loads(_fill_line(n, k, True, L=k)) for n, k in calls]
    fb = h200_step.fill_steps(at_k, calls=calls)
    assert list(fb) == [(256, 0, 16384, True, "")] and fb[(256, 0, 16384, True, "")]["step"] == pytest.approx(128 * 3e-3)
    today = [json.loads(_fill_line(n, k, True)) for n, k in calls]
    assert list(h200_step.fill_steps(today, calls=calls)) == [(256, 4096, 16384, True, "")]
    names = {(6144, 4096): "qkv", (4096, 4096): "o", (28672, 4096): "gu", (4096, 14336): "down"}
    step = {"fill_step": True, "step": "qkv,o,gu,down*32", "layers": 32, "lp": 256, "L": 0, "row": 16384, "o": True,
            "steady": True, "shapes": [{"shape": names[nk], "m": 16384, "n": nk[0], "k": nk[1], "calls": c}
                                       for nk, c in calls.items()],
            "gsms": [{"n": 0, "sms": [132, 132]}], "arms": {"step_g0": {"ms": 300.0, "mhz": 1900, "calls": 1}}}
    so = h200_step.served_order([step], at_k)
    assert so[0]["model"] == "8b" and so[0]["serial"] == pytest.approx(128 * 2.2e-3)
    assert so[0]["n"][0]["apart"] == pytest.approx(128 * 3e-3)


def test_a_steps_power_floor_is_its_joules_over_the_boards_limit(tmp_path, capsys, monkeypatch):
    """With the board's energy on the lines: each served-order N's watts and joules a step, its floor (the joules over
    the power limit), and the serial split's joules (a served call's tensor, hashing and other arms, no pad) and their
    time at the limit; a split arm's joules a step and watts; lines without energy print none."""
    calls = h200_step.MODELS["8b"]
    lines = [json.loads(_fill_line(n, k, True)) for n, k in calls]
    j = {"gemm": 1.0, "coins": 0.05, "coins_w": 0.025, "form_a": 0.0125, "form_b": 0.0125, "commit": 0.4}
    for x in lines:
        x["lp"], x["power_limit_w"] = 512, 700.0
        x["gsms"] = [{"n": 64, "sms": [64, 68]}]
        x["arms"] |= {"wgl": {"ms": 1.25, "mhz": 1980}, "ovl_w_g64": {"ms": 2.0, "mhz": 1980, "j": 1.3}}
        for a, v in j.items():
            x["arms"][a]["j"] = v
    f = tmp_path / "fill-bench-served1-L4096.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    shapes = [{"shape": f"s{i}", "m": 16384, "n": nk[0], "k": nk[1], "calls": c} for i, (nk, c) in enumerate(calls.items())]
    g = tmp_path / "fill-step-served1-L4096.jsonl"
    g.write_text(json.dumps({"fill_step": True, "step": "s0,s1,s2,s3*32", "layers": 32, "lp": 512, "L": 4096, "row": 16384,
                             "o": True, "steady": True, "power_limit_w": 700.0, "shapes": shapes,
                             "gsms": [{"n": 64, "sms": [64, 68]}],
                             "arms": {"step_g64": {"ms": 240.0, "mhz": 1900, "calls": 1, "j": 175.0, "w": 700.0}}}) + "\n")
    so = h200_step.served_order(h200_step.step_rows([str(g)]), h200_step.rows([str(f)]))
    n64 = so[0]["n"][64]
    assert n64["w"] == 700.0 and n64["j"] == 175.0 and n64["floor"] == pytest.approx(0.25)
    assert so[0]["serial_j"] == pytest.approx(128 * 1.5) and so[0]["serial_floor"] == pytest.approx(128 * 1.5 / 700)
    sp = h200_step.split_steps(h200_step.rows([str(f)]), calls=calls)[(512, 4096, 16384, True, "served1-L4096")]
    assert sp["n"][64]["j"] == pytest.approx(128 * 1.3) and sp["n"][64]["w"] == pytest.approx(1.3 / 2e-3)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f), str(g)])
    assert h200_step.main() == 0
    out = capsys.readouterr().out
    assert "| 64 | 64 + 68 | 0.240 | 1.500 | 0.256 | 0.282 | 50.0% | 1900 | 700 | 175 | 0.250 |" in out
    assert "| 64 | 64 + 68 | 0.256 | 1.600 | - | - | - | 30.8% | 650 | 166 |" in out
    assert ("N 64: step 0.240 s, gamma_u 1.500, hashing hidden 50.0%; fill-bench's step at N 64 0.256 s; serial 0.282 s "
            "(hashing 0.083 s), gamma_u against these lines' wgl 0.160 s; 700 W, 175 J a step, its floor at the 700 W "
            "limit 0.250 s; the serial split 192 J a step, 0.274 s at the 700 W limit") in out
    assert h200_step.serial_joules(h200_step.fill_steps([json.loads(_fill_line(n, k, True)) for n, k in calls],
                                                        calls=calls)[(256, 4096, 16384, True, "")]) is None


def test_pipe_duty_is_each_kernels_busy_time_over_the_overlapped_call(tmp_path, capsys):
    """h200_pipes.py: a pipe's busy time is its share of peak over the kernel's elapsed cycles (active / elapsed scales
    it) times the kernel's duration (in ncu's unit, refused when it's none it knows), the larger of its issue and
    cycles-active shares; the operands' generation and whatever precedes the first call's bits_kernel (cuBLASLt's
    algorithm search) are left out; and the duty is the busy time a call over fill-bench's overlapped call, so overlapping
    kernels add."""
    import h200_pipes

    cols = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_active.avg", "sm__cycles_elapsed.avg",
            "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"]
    rows = [["", "", "ms", "cycle", "cycle", "%", "%", "%", "%"],
            ["0", "sm90_xmma_gemm_i8i32_i8i32_i32_tn_n_tilesize128x128x128_cublas", "3", "1,000", "1,000", "90", "0", "0",
             "0"],
            ["1", "gen_kernel(signed char*, int, unsigned int, bool)", "1", "100", "100", "0", "50", "50", "10"],
            ["2", "bits_kernel(Seeds, int, unsigned long, unsigned char *)", "0.5", "100", "100", "0", "0", "0", "0"],
            ["3", "void gemm_kernel<true, 2>(GArgs)", "2", "1,000", "1,000", "80", "5", "4", "2"],
            ["4", "void leaf_persist<V31Row>(Persist, int)", "1", "50", "100", "0", "90", "80", "20"],
            ["5", "fill_a_kernel(...)", "0.5", "100", "100", "0", "40", "40", "30"]]
    csv_text = "\n".join(",".join(f'"{c}"' for c in r) for r in [cols] + rows) + "\n"
    (tmp_path / "ncu-row16384-L4096-8B-o.raw.csv").write_text(csv_text)
    (tmp_path / "ncu-row16384-L4096-8B-o.log").write_text("8B-o: 1 calls, L 4096 row 16384, grid 132, gemm_bn 256\n")
    (tmp_path / "fill-bench-row16384-L4096.jsonl").write_text(
        json.dumps({"fill": True, "shape": "8B-o", "arms": {"ovl": {"ms": 2.5}, "lt": {"ms": 2.0}}}) + "\n")
    assert h200_pipes.main(["h200_pipes.py", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert ("| 8B-o | 4096 | row16384 | 2.500 | 2.000 | 1.250 | 64.0% | 30.0% | 0.0% | 11.6% | 36.0% | 2.000 | 1.000 | "
            "1.000 | 1.60 | - |") in out
    bad = tmp_path / "ncu-furlong.raw.csv"
    bad.write_text(csv_text.replace('"ms"', '"furlong"'))
    with pytest.raises(ValueError, match="furlong"):
        h200_pipes.read_ncu(bad)


def test_pipe_duty_on_a_split_is_over_its_overlapped_arm_with_the_weight_side_counted(tmp_path, capsys):
    """On V31_GSM's split the wall is fill-bench's ovl_w_gN, whose calls form W' each as fill-once's do: fill_b_kernel
    counts there."""
    import h200_pipes

    cols = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_active.avg", "sm__cycles_elapsed.avg",
            "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"]
    rows = [["", "", "ms", "cycle", "cycle", "%", "%", "%", "%"],
            ["0", "bits_kernel(Seeds, int, unsigned long, unsigned char *)", "0.5", "100", "100", "0", "0", "0", "0"],
            ["1", "void wgp::gemm<256, 1>(wgp::Args)", "2", "1,000", "1,000", "80", "5", "4", "2"],
            ["2", "fill_b_kernel(signed char const*)", "1", "100", "100", "0", "40", "40", "30"],
            ["3", "void leaf_persist<ServedRow>(Persist, int)", "1", "50", "100", "0", "90", "80", "20"]]
    (tmp_path / "ncu-served1-gsm64-L4096-8B-o.raw.csv").write_text(
        "\n".join(",".join(f'"{c}"' for c in r) for r in [cols] + rows) + "\n")
    (tmp_path / "ncu-served1-gsm64-L4096-8B-o.log").write_text("8B-o: 1 calls, L 4096 row 16384\n")
    arms = {"ovl": {"ms": 9.0}, "ovl_w_g64": {"ms": 2.0}, "lt": {"ms": 1.6}}
    (tmp_path / "fill-bench-served1-gsm64-L4096.jsonl").write_text(
        json.dumps({"fill": True, "shape": "8B-o", "gsm": 64, "arms": arms}) + "\n")
    assert h200_pipes.main(["h200_pipes.py", str(tmp_path)]) == 0
    assert ("| 8B-o | 4096 | served1-gsm64 (ovl_w_g64) | 2.000 | 1.600 | 1.250 | 80.0% | 47.5% | 0.0% | 22.0% | 20.0% | "
            "2.000 | 1.000 | 1.500 | 2.25 | - |") in capsys.readouterr().out


def test_pipe_duty_over_a_step_is_its_calls_busy_cycles_over_the_steps_cycles_at_its_clock(tmp_path, capsys):
    """fill-step's lines beside ncu's: each shape's busy cycles a call (a pipe's share of the kernel's active cycles, the
    larger of issue and cycles-active), times its calls a step, over the step's seconds at its own SM clock; an ncu file of
    another variant serves where it is the one for that L and shape; a step with a shape no ncu file has is left out."""
    import h200_pipes

    cols = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_active.avg", "sm__cycles_elapsed.avg",
            "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"]
    units = ["", "", "ms", "cycle", "cycle", "%", "%", "%", "%"]
    a = [["0", "bits_kernel(Seeds, int, unsigned long, unsigned char *)", "0.1", "100", "100", "0", "0", "0", "0"],
         ["1", "void wgp::gemm<256, 1>(wgp::Args)", "1", "1,000,000", "1,000,000", "80", "0", "0", "0"],
         ["2", "void leaf_persist<ServedRow>(Persist, int)", "1", "250,000", "500,000", "0", "88", "80", "20"]]
    b = [["0", "bits_kernel(Seeds, int, unsigned long, unsigned char *)", "0.1", "100", "100", "0", "0", "0", "0"],
         ["1", "void wgp::gemm<256, 1>(wgp::Args)", "1", "200,000", "200,000", "50", "0", "0", "0"]]
    for shape, rows, calls in (("a", a, 2), ("b", b, 1)):
        (tmp_path / f"ncu-v0-L4096-{shape}.raw.csv").write_text(
            "\n".join(",".join(f'"{c}"' for c in r) for r in [cols, units] + rows) + "\n")
        (tmp_path / f"ncu-v0-L4096-{shape}.log").write_text(f"{shape}: {calls} calls (2 serial)\n")

    def step(spec: str, shapes: dict[str, int]) -> str:
        return json.dumps({"fill_step": True, "step": spec, "L": 4096, "shapes": [{"shape": s, "calls": c}
                                                                                 for s, c in shapes.items()],
                           "gsms": [{"n": 0, "sms": [132, 132]}, {"n": 64, "sms": [64, 68]}],
                           "arms": {"step_g0": {"ms": 10.0, "mhz": 1000}, "step_g64": {"ms": 8.0, "mhz": 1500}}})

    (tmp_path / "fill-step-v1-L4096.jsonl").write_text(step("a,b*2", {"a": 2, "b": 1}) + "\n"
                                                       + step("c", {"c": 1}) + "\n")
    assert h200_pipes.main(["h200_pipes.py", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| a,b*2 | 4096 | v1 | 0 | 0.010 | 1000 | 9.0% | 2.2% | 0.0% | 0.5% | 91.0% | 97.8% | 100.0% | 99.5% | - |" in out
    assert "| a,b*2 | 4096 | v1 | 64 | 0.008 | 1500 | 7.5% | 1.8% | 0.0% | 0.4% | 92.5% | 98.2% | 100.0% | 99.6% | - |" in out
    assert "| c |" not in out


def test_pipe_kinds_read_ncus_demangled_names_and_cublaslts_gemm():
    import h200_pipes

    assert h200_pipes.kind("void gemm_kernel<false, 1>(GArgs)") == "gemm"
    assert h200_pipes.kind("void cutlass::device_kernel<cutlass3x_sm90_tensorop_s64x256x32gemm_s8_s8_s32>(Params)") == "gemm"
    assert h200_pipes.kind("sm90_xmma_gemm_s8s8_s32_tn_n_tilesize128x256x128_cublasLt") == "gemm"
    assert h200_pipes.kind("void leaf_persist<LtRow>(Persist, int)") == "hash"
    assert h200_pipes.kind("node_kernel(unsigned long*, int)") == "hash"
    assert h200_pipes.kind("void gen_kernel<1>(signed char*)") is None
    assert h200_pipes.kind("fill_a_kernel(signed char const*)") == "other"
