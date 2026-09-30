"""``tools/tc_probe_fp4/w1.py``, the W1 price microbenchmarks, on fake silicon and on synthetic SASS.

The fake run passes every gate and turns its rates into prices in the unit; a flipped gate word fails the run before anything is
timed; the SASS loop gate rejects a loop with a missing or an extra watched instruction, and pins add.rn.f32x2's sm_120a lowering
(two FADDs, no FADD2); the packed add's gate words tell its halves apart; the casts' packing (hi operand in the upper byte or
nibble) and the E2M1 reference's grid, ties and saturation are pinned; the cast-and-store loops' staged words and exact loop bodies
are pinned; timing samples nvidia-smi every few rounds, not every rep; the three-loop split recovers known costs; the tool
declarations resolve their key params.  The E4M3-to-FP16 reference is pinned to the registered decode, its timing rows never feed a
chain a NaN code, and its loops are pinned whole; unwritten words must keep the poison, and a device whose no-write control passes
the gate fails the run; the fill lease writes a header the runner parses, records the child's exit status (no retry turns a failed
gate into a pass), waits out the runner's move-then-log order and withdraws a job that never starts.  With nvcc on the PATH, the
real build's loops are checked.
"""
from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import w1  # noqa: E402


@pytest.fixture
def no_nvcc(monkeypatch):
    monkeypatch.setattr(w1.P4, "nvcc_bin", lambda: "/nonexistent/nvcc")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)


def _run(tmp_path, *extra):
    return w1.run(w1.parser().parse_args(["--out", str(tmp_path), "--fake-silicon", "--reps", "2", *extra]))


def test_fake_run_prices(tmp_path, no_nvcc):
    code, res = _run(tmp_path)
    assert code == 0, res["failures"]
    assert res["numeric_gate"]["ok"] and res["clocks"] == "unlocked"
    p = res["prices"]["price_units"]
    assert p["qmma_e4m3_f32 per MAC"] == 1.0
    assert p["fadd per register"] == pytest.approx(1024 / 128, rel=1e-6)
    assert p["f2fp_e4m3x2 per instruction (2 codes)"] == pytest.approx(2 * p["f2fp_e4m3x2 per code"])
    assert p["fadd_f32x2 per word"] == pytest.approx(1024 / 128, rel=1e-6) == p["fadd_pairs per word"]
    assert p["fadd_f32x2 per instruction (2 words)"] == pytest.approx(2 * p["fadd_f32x2 per word"])
    assert p["f2fp_e2m1x2 per code"] == pytest.approx(1024 / 64, rel=1e-6) == p["f2fp_e2m1x2_x2 per code"]
    assert p["f2fp_e2m1x2 per instruction (2 codes)"] == pytest.approx(2 * p["f2fp_e2m1x2 per code"])
    assert "e2m1_split" not in res["prices"]  # no SASS counts without nvcc


@pytest.mark.parametrize("op", [10, 11])
def test_packed_add_gate_tells_the_halves_apart(op):
    want = w1.expected_words(op, w1.GATE_ITERS, np.array(w1.GATE_ROW[op], np.uint32)[None].repeat(32, 0))
    lo, hi = want[:, 0:16:2].view(np.float32), want[:, 1:16:2].view(np.float32)
    assert (lo == w1.SUNROLL * w1.GATE_ITERS).all() and (hi == 2 * lo).all()


def test_corrupt_gate_word_fails_before_timing(tmp_path, no_nvcc):
    code, res = _run(tmp_path, "--corrupt-word", "3")
    assert code == 1 and res["failures"] == ["numeric gate: qmma_e4m3_f32"]
    assert "timing" not in res and "prices" not in res


@pytest.mark.parametrize("op", [8, 9, 12, 13, 14, *range(16, 26), *w1.F16_OPS])
def test_gate_words_are_not_vacuous(op):
    rows = w1.timing_rows(op, np.random.default_rng(w1.F8.SEED_DEFAULT))
    assert w1.expected_words(op, w1.GATE_ITERS, rows)[:, : w1.NCHAIN].all(axis=1).mean() > 0.5


def test_vacuous_gate_fails(tmp_path, no_nvcc, monkeypatch):
    monkeypatch.setattr(w1, "expected_words", lambda op, iters, rows: np.zeros((w1.THREADS, 16), np.uint32))
    code, res = _run(tmp_path)
    assert code == 1 and res["numeric_gate"]["ops"]["fadd"]["vacuous"] and "prices" not in res


def _sass(body: list[str], kernel: str = "fadd_rate") -> str:
    lines = [f"        /*{16 * i:04x}*/                   {op} R0, R1 ;" for i, op in enumerate(body)]
    lines.append(f"        /*{16 * len(body):04x}*/                   BRA.U UP0, 0x0 ;")
    return f"\n\tFunction : _ZN12_GLOBAL__N_1{len(kernel)}{kernel}EPKjPjPxPii\n" + "\n".join(lines) + "\n"


def test_loop_bodies_counts_one_loop():
    body = w1.loop_bodies(_sass(["FADD"] * 64 + ["UIADD3"]))["fadd_rate"]
    assert body == collections.Counter({"FADD": 64, "UIADD3": 1, "BRA.U": 1})


@pytest.mark.parametrize("body", [["FADD"] * 63, ["FADD"] * 64 + ["FFMA"], ["FADD"] * 64 + ["UFADD"]])
def test_sass_gate_rejects_a_wrong_loop(monkeypatch, body):
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body)})())
    gate = w1.sass_gate(Path("/nonexistent.so"), None)
    assert not gate["kernels"]["fadd"]["ok"]


@pytest.mark.parametrize("body, ok", [(["FADD"] * 128, True), (["FADD2"] * 64, False), (["FADD"] * 126 + ["FADD2"], False)])
def test_sass_gate_pins_the_packed_add_lowering(monkeypatch, body, ok):
    """sm_120a lowers add.rn.f32x2 to two FADDs; a toolchain that emits FADD2 changes the price and must fail the gate."""
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body, "fadd_f32x2_rate")})())
    assert w1.sass_gate(Path("/nonexistent.so"), None)["kernels"]["fadd_f32x2"]["ok"] is ok


def test_f2fp_packing_pinned():
    rows = np.zeros((32, 8), np.uint32)
    rows[:, 0] = np.float32(1.0).view(np.uint32)
    rows[:, 1] = np.float32(2.0).view(np.uint32)
    h = w1.f2fp_model(1, rows)
    f = int(w1.F8.cvt_ref("e4m3", np.array([0x3F800000], np.uint32))[0])
    assert f == 0x38
    xs = w1.F8.cvt_ref("e4m3", rows[0, 1] + (np.arange(16, dtype=np.uint32) << 20)).astype(int)
    assert xs.tolist() == list(range(0x40, 0x50))
    g = int(w1.F8.cvt_ref("e4m3", np.array([0x3F800000 + (1 << 21)], np.uint32))[0])
    assert g == 0x3A
    want = [0] * 8
    for k in range(16):
        want[k % 8] += ((f << 8) | xs[k]) | (((xs[k] << 8) | f) << 16)
        want[k % 8] += ((g << 8) | xs[k]) | (((xs[k] << 8) | g) << 16)
    assert h[0].tolist() == [w % 2**32 for w in want] and all(want)


def test_e2m1_ref_grid_ties_saturation_and_sign():
    x = np.array([0, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 5.1, 6, 7, 1e30, -0.0, -0.1, -6, -100], np.float32)
    assert w1.e2m1_ref(x.view(np.uint32)).tolist() == [0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 7, 7, 8, 8, 15, 15]


def test_e2m1_packing_pinned():
    """f = 1 (code 2), g = 1.25 (a tie, to code 2), and 16 constants from 0.25 in quarter-binade steps, which cross every tie of
    the grid below 4; byte i of a packed word is Q(f, x), Q(x, f), Q(g, x), Q(x, g) with Q(hi, lo) = code(hi) << 4 | code(lo)."""
    rows = np.zeros((32, 8), np.uint32)
    rows[:, 0] = np.float32(1.0).view(np.uint32)
    rows[:, 1] = np.float32(0.25).view(np.uint32)
    cx = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 5, 6]
    want = [0] * 8
    for k in range(16):
        want[k % 8] += (2 << 4 | cx[k]) | (cx[k] << 4 | 2) << 8 | (2 << 4 | cx[k]) << 16 | (cx[k] << 4 | 2) << 24
    assert w1.f2fp_e2m1_model(1, rows, 16, packed=True)[0].tolist() == [w % 2**32 for w in want]
    want = [0] * 8
    for k in range(16):
        want[k % 8] += (2 << 4 | cx[k]) + (cx[k] << 4 | 2)
        want[(k + 1) % 8] += (2 << 4 | cx[k]) + (cx[k] << 4 | 2)
    assert w1.f2fp_e2m1_model(1, rows, 16, packed=False)[0].tolist() == want


def test_cast_store_packing_pinned():
    """Values from 1.0 in steps of 2^20 are one E4M3 ulp apart (codes 0x38 + j, the sign bit on odd j); a chk word is four staged
    codes in store order.  The packed loop without per-code input stores constant 2w, f, g, constant 2w + 1."""
    rows = np.zeros((32, 8), np.uint32)
    rows[:, 0] = rows[:, 1] = np.float32(1.0).view(np.uint32)
    fadd_rows = rows.copy()
    fadd_rows[:, 1] = 0  # y = +0: the chains keep their values
    c = [(0x38 + j) | (0x80 if j % 2 else 0) for j in range(64)]
    want = [c[4 * w] | c[4 * w + 1] << 8 | c[4 * w + 2] << 16 | c[4 * w + 3] << 24 for w in range(16)]
    for op in w1.CS_FADD:
        assert w1.cs_model(op, 1, fadd_rows)[0].tolist() == want
    assert w1.cs_model(17, 5, rows)[0].tolist() == want
    rows[:, 1] = np.float32(2.0).view(np.uint32)
    cx = [(0x40 + j) | (0x80 if j % 2 else 0) for j in range(32)]
    for op in (19, 21, 23, 24, 25):
        assert w1.cs_model(op, 1, rows)[0].tolist() == [cx[2 * w] | 0x38 << 8 | 0x3A << 16 | cx[2 * w + 1] << 24 for w in range(16)]


_PORT = ["FADD"] * 64 + [w1.OPS[16][1]] * 64 + ["LOP3.LUT"] * 32 + ["IMAD"] * 32 + ["STS.U16"] * 32 + ["UIADD3", "UISETP.NE.AND"]


@pytest.mark.parametrize("body, ok", [(_PORT, True), (["FADD.FTZ" if op == "FADD" else op for op in _PORT], False),
                                      (_PORT + ["LOP3.LUT"], False), (_PORT[:-3], False)])
def test_sass_gate_pins_the_cast_store_loop(monkeypatch, body, ok):
    """The port's loop is exactly its FADD feed, 64 converts, 32 LOP3, 32 IMAD and 32 STS.U16: an FTZ add, an extra support
    instruction or a missing store fails."""
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body, "cs_port_fadd_rate")})())
    assert w1.sass_gate(Path("/nonexistent.so"), None)["kernels"]["cs_port_fadd"]["ok"] is ok


_W64 = ["FADD"] * 64 + [w1.OPS[20][1]] * 32 + ["STS.64"] * 8 + ["UIADD3", "UISETP.NE.AND"]


@pytest.mark.parametrize("body, ok", [(_W64, True), (["STS" if op == "STS.64" else op for op in _W64], False),
                                      (_W64[:64 + 32] + ["STS.64"] * 7 + _W64[-2:], False),
                                      (_W64[:64 + 32] + ["STS.128"] * 4 + _W64[-2:], False)])
def test_sass_gate_pins_the_wide_store(monkeypatch, body, ok):
    """The 64-bit-store loop is exactly its FADD feed, 32 converts and 8 STS.64: narrower stores, a missing one or STS.128 fail."""
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body, "cs_w64_fadd_rate")})())
    assert w1.sass_gate(Path("/nonexistent.so"), None)["kernels"]["cs_w64_fadd"]["ok"] is ok


def test_smi_sampled_every_few_rounds(monkeypatch):
    calls = []
    monkeypatch.setattr(w1, "smi_sample", lambda uuid: calls.append(uuid) or {"clocks.sm": "2092"})
    rng = np.random.default_rng(1)
    tim = w1.timing(w1.FakeDevice(), {0: w1.timing_rows(0, rng), 6: w1.timing_rows(6, rng)}, 10, 1 << 20, "GPU-x", [0, 6], smi_every=4)
    assert [s.get("before_round", s.get("after_round")) for s in tim["smi"]] == [0, 3, 7, 9] and len(calls) == 4
    assert all("smi" not in r for reps in tim["reps"].values() for r in reps)


def test_lease_needs_a_real_build(tmp_path, capsys):
    assert w1.main(["--out", str(tmp_path), "--fake-silicon", "--lease-max-min", "5"]) == 2
    assert "--lease-max-min" in capsys.readouterr().err
    assert w1.main(["--out", str(tmp_path), "--lease-via-fill", "bc-x"]) == 2
    assert "--lease-via-fill needs --lease-max-min" in capsys.readouterr().err


def test_e4m3_f16_decode_pinned():
    """1.0, the least subnormal 2^-9 (an FP16 normal), the least normal 2^-6, 448, -0 and -448; the high code goes to the high half,
    and only 0x7F / 0xFF halves are masked as NaN."""
    t = w1.E4M3_F16
    assert [int(t[c]) for c in (0x38, 0x01, 0x08, 0x7E, 0x80, 0xFE, 0x00)] == [0x3C00, 0x1800, 0x2400, 0x5F00, 0x8000, 0xDF00, 0]
    words, nan = w1.f16_decode(np.array([0x3801, 0x7F38, 0xAB38FF], np.uint32))
    assert words[0] == 0x3C001800 and nan.tolist() == [0, 0xFFFF0000, 0x0000FFFF]
    assert words[2] & 0xFFFF0000 == 0x3C000000  # bits above 16 are ignored


@pytest.mark.parametrize("seed", range(40))
def test_f16_timing_rows_never_feed_a_nan(seed):
    rng = np.random.default_rng(seed)
    for op in w1.F16_OPS:
        rows = w1.timing_rows(op, rng)
        chains = w1.f16_chains(op, w1.GATE_ITERS, rows)
        if op != 26:  # the fed chains stay alive; the bare chains collapse, and the table words keep that gate honest
            assert chains.all(axis=1).mean() > 0.5


def test_f16_prices_and_steps(tmp_path, no_nvcc):
    code, res = _run(tmp_path, "--ops", "f2fp_f16_e4m3,f2fp_f16_e4m3_half,f2fp_f16_e4m3_xor")
    assert code == 0, res["failures"]
    g = res["numeric_gate"]["ops"]["f2fp_f16_e4m3"]
    assert g["inputs_converted"] == w1.FakeDevice.sms * w1.THREADS * 8 and g["nan_outputs"] == {"0x7f": ["7fff"], "0xff": ["7fff"]}
    p = res["prices"]["price_units"]
    assert p["f2fp_f16_e4m3 per code"] == pytest.approx(1024 / 128) and p["f2fp_f16_e4m3 per instruction (2 codes)"] == pytest.approx(16)
    st = res["prices"]["f16_lop3_steps"]
    assert [st["first_32_lop3"], st["second_32_lop3"]] == pytest.approx([4096 / 96 - 4096 / 128, 4096 / 64 - 4096 / 96])


def test_negative_controls_are_rejected(tmp_path, no_nvcc):
    code, res = _run(tmp_path, "--ops", "fadd,f2fp_f16_e4m3")
    assert code == 0, res["failures"]
    for name in ("fadd", "f2fp_f16_e4m3", w1.UNIT_OP):
        nc = res["numeric_gate"]["ops"][name]["negative_control"]
        assert nc["rejected"] and nc["mismatches"] > 0 and nc["distinct_sms"] == 1


def test_a_gate_that_accepts_no_writes_fails(tmp_path, no_nvcc, monkeypatch):
    """A device whose no-write launch still returns the right words (say, stale buffers never poisoned) must fail the gate."""
    real = w1.FakeDevice.run
    monkeypatch.setattr(w1.FakeDevice, "run", lambda self, op, iters, rows, negative=False: real(self, op, iters, rows))
    code, res = _run(tmp_path, "--ops", "f2fp_f16_e4m3")
    assert code == 1 and not res["numeric_gate"]["ops"]["f2fp_f16_e4m3"]["negative_control"]["rejected"] and "prices" not in res
    assert res["failures"] == [f"numeric gate: {w1.UNIT_OP}, f2fp_f16_e4m3"]


def test_unwritten_words_must_keep_the_poison():
    rows = np.array(w1.GATE_ROW[6], np.uint32)[None].repeat(32, 0)
    want, nan = w1.gate_words(6, w1.GATE_ITERS, rows, 2)
    assert (want[..., w1.NCHAIN:] == w1.POISON).all() and (want[..., : w1.NCHAIN] != w1.POISON).all()
    chk = want.copy()
    chk[..., w1.NCHAIN:] = 0  # a kernel that zeroes words it shouldn't touch
    assert w1.gate_mismatches(chk, want, nan).sum() == 2 * w1.THREADS * (16 - w1.NCHAIN)


def test_nan_halves_need_an_fp16_nan():
    want, nan = w1.f16_decode(np.array([0x7F38], np.uint32))
    ok = (want & ~nan) | np.uint32(0x7E000000)
    assert not w1.gate_mismatches(ok, want, nan).any()
    assert w1.gate_mismatches((want & ~nan) | np.uint32(0x7C000000), want, nan).all()  # +inf is not a NaN


_F16 = ["F2FP.F16.E4M3.UNPACK_B"] * 64 + ["UIADD3", "UISETP.NE.AND"]


@pytest.mark.parametrize("kernel, name, body, ok", [
    ("f2fp_f16_rate", "f2fp_f16_e4m3", _F16, True), ("f2fp_f16_rate", "f2fp_f16_e4m3", _F16 + ["PRMT"], False),
    ("f2fp_f16_rate", "f2fp_f16_e4m3", _F16[1:], False), ("f2fp_f16_half_rate", "f2fp_f16_e4m3_half", _F16 + ["LOP3.LUT"] * 32, True),
    ("f2fp_f16_half_rate", "f2fp_f16_e4m3_half", _F16 + ["LOP3.LUT"] * 33, False),
    ("f2fp_f16_xor_rate", "f2fp_f16_e4m3_xor", _F16 + ["LOP3.LUT"] * 64, True),
    ("f2fp_f16_xor_rate", "f2fp_f16_e4m3_xor", _F16 + ["LOP3.LUT"] * 64 + ["MOV"], False)])
def test_sass_gate_pins_the_f16_loops(monkeypatch, kernel, name, body, ok):
    """The FP16 convert loops are exactly 64 converts and their LOP3 feed: a PRMT, a MOV or one LOP3 too many fails."""
    monkeypatch.setattr(w1.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": _sass(body, kernel)})())
    assert w1.sass_gate(Path("/nonexistent.so"), None)["kernels"][name]["ok"] is ok


@pytest.fixture
def fill_dir(tmp_path, monkeypatch):
    d = tmp_path / "fill"
    for sub in ("queue", "running", "done", "failed", "logs"):
        (d / sub).mkdir(parents=True)
    (d / "events.jsonl").write_text("")
    monkeypatch.setattr(w1, "FILL_DIR", d)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    return d


def _runner(fill: Path, logged_late: bool):
    """A stand-in for fill_runner.py's step, called in place of sleep: start the queued job (bash, as the runner does), move it to
    done/, then append its events, on the same call or (logged_late) on the next one."""
    state = {"calls": 0}

    def event(kind, **kw):
        with open(fill / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": "T", "event": kind, "job": "w1-r-test.sh", **kw}) + "\n")

    def step(_seconds):
        state["calls"] += 1
        job = fill / "queue" / "w1-r-test.sh"
        if job.exists():
            p = subprocess.run(["bash", str(job)], capture_output=True, text=True, check=False)
            (fill / "logs" / "w1-r-test.sh.120000.log").write_text(p.stdout + p.stderr)
            job.rename(fill / "done" / job.name)
            state["rc"] = p.returncode
            event("start", owner="bc-x")
            if not logged_late:
                event("done", rc=p.returncode)
        elif logged_late and "logged" not in state:
            state["logged"] = True
            event("done", rc=state["rc"])
    return step, state


def test_fill_header_parses_as_the_runner_reads_it(fill_dir):
    step, _ = _runner(fill_dir, False)
    seen = {}

    def peek(s):
        text = (fill_dir / "queue" / "w1-r-test.sh").read_text()
        seen.update(dict(re.findall(r"(\w+)=(\S+)", re.search(r"#\s*fill:\s*(.*)", text).group(1))))
        step(s)

    w1.fill_lease([sys.executable, "-c", "pass"], "bc-x", 8, fill_dir.parent / "sub", sleep=peek)
    assert seen == {"owner": "bc-x", "gpus": "1", "max_min": "8", "cpus": "8", "project": "pous", "prio": "10"}


@pytest.mark.parametrize("logged_late", [False, True])
def test_fill_lease_records_the_child_status(fill_dir, logged_late):
    """The child exits 3 (a failed gate); its script still exits 0, so the runner files it under done/ with no retry, and the lease
    returns 3 with the job's events and log, even when the runner logs the exit after the move."""
    step, state = _runner(fill_dir, logged_late)
    sub = fill_dir.parent / "sub"
    rc, info = w1.fill_lease([sys.executable, "-c", "import sys; print('child ran'); sys.exit(3)"], "bc-x", 8, sub, sleep=step)
    assert rc == 3 and state["rc"] == 0 and (fill_dir / "done" / "w1-r-test.sh").exists()
    assert [e["event"] for e in info["events"]] == ["start", "done"]
    assert "child ran" in (sub / "fill-w1-r-test.sh.120000.log").read_text()


def test_fill_lease_withdraws_a_job_that_never_starts(fill_dir):
    rc, info = w1.fill_lease([sys.executable, "-c", "pass"], "bc-x", 8, fill_dir.parent / "sub", wait_min=1, sleep=lambda s: None)
    assert rc == 124 and "withdrawn" in info and not any((fill_dir / "queue").iterdir())


def test_split_recovers_the_costs():
    loops = {"f2fp_e2m1x2": {w1.OPS[12][1]: 64, "IADD3": 10, "UIADD3": 1, "UISETP.NE.AND": 1, "BRA.U": 1},
             "f2fp_e2m1x2_u8": {w1.OPS[13][1]: 64, "LOP3.LUT": 64, "IADD3": 34, "UIADD3": 1, "UISETP.NE.AND": 1, "BRA.U": 1},
             "f2fp_e2m1x2_x2": {w1.OPS[14][1]: 128, "IADD3": 18, "UIADD3": 1, "UISETP.NE.AND": 1, "BRA.U": 1}}
    a, b, c = 0.49, 0.52, 0.02
    rates = {}
    for op in w1.E2M1_OPS:
        name, opcode, per_loop, work, _ = w1.OPS[op]
        n = loops[name]
        s = sum(v for k, v in n.items() if k != opcode and not k.startswith(("U", "BRA")))
        rates[name] = per_loop * work / (a * n[opcode] + b * s + c * 3)
    sp = w1.split({"rates_per_sm_per_clock": rates, "unit_rate_per_sm_per_clock": 1024.0}, loops)
    assert sp["ok"] and [sp["a_per_cast"], sp["b_per_support"], sp["c_per_uniform"]] == pytest.approx([a, b, c])
    assert sp["cast_alone_price_per_code"] == pytest.approx(1024 * a / 64)


def test_tool_declarations_resolve():
    from tool_fp8 import TC_PROBE_FP8, W1_PRICES, parse_w1

    _, k = TC_PROBE_FP8.resolve_params(["python", "tools/tc_probe_fp4/fp8.py", "--row", "chain", "--out", "/x", "--expect-gpu-uuid", "GPU-1"])
    assert k["row"] == "chain" and k["instruction"] is None and k["steps"] == 2048
    argv = ["python", "tools/tc_probe_fp4/w1.py", "--out", "/x", "--reps", "12", "--lease-max-min", "10", "--smi-every", "5",
            "--lease-via-fill", "bc-x"]
    _, k = W1_PRICES.resolve_params(argv)
    assert k["reps"] == 12 and k["clocks"] == "unlocked" and "_unknown" not in parse_w1(argv)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("nvcc") is None or shutil.which("cuobjdump") is None, reason="needs nvcc and cuobjdump (CUDA 12.9)")
def test_sass_loops_on_a_real_build(tmp_path):
    so, _ = w1.P4.build(w1.F8.ARCH, tmp_path, w1.SOURCE)
    gate = w1.sass_gate(so, None)
    assert gate["ok"], {k: r["loop"] for k, r in gate["kernels"].items() if not r["ok"]}
