"""opcode_probe: the census line and its schema, MACs a word, the declared verdicts, and the rows a run writes.

The gate's verdicts read `fixtures/gpu_kit/` (test_gpu_kit.py says what they are). `test_probes_build_and_gate` builds every
probe with nvcc and gates it; it skips without nvcc and nvdisasm.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures" / "gpu_kit"
MACHINE_MODEL = POUW / "machine_model"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load("opcode_probe_probe", POUW / "opcode_probe" / "probe.py")
rows, G = probe.rows, probe.G
RUN = "r20261016-183617-3914"


def fns(name: str) -> list:
    return [G.H.Function(f) for f in G.H.parse_nvdisasm((FIX / f"{name}.nvdisasm.txt").read_text())]


def table(name: str) -> list[dict]:
    return json.loads((FIX / f"{name}.list.json").read_text())


# ---- the census line ---------------------------------------------------------------------------------------------------


def test_census_line_is_machine_models_reading():
    line = rows.census_line("LDSM_X1", "ROW_LDSMR1", 0.4994851, RUN)
    assert tuple(line) == rows.CENSUS_COLUMNS
    assert line["rate"] == "0.499485" and line["noise"] == "0.02" and line["runs"] == RUN
    assert rows.u_of(rows.CYCLE_U, line["rate"], line["noise"], line["cycles"], line["u"]) == pytest.approx(
        300.08 * 4 / (0.499485 * 1.02))
    text = rows.census_csv([line, rows.census_line("STS128", "R6w", 0.249784, RUN, since="VF3k")])
    assert text.splitlines()[0] == ",".join(rows.CENSUS_COLUMNS)
    assert [d["since"] for d in csv.DictReader(io.StringIO(text))] == ["", "VF3k"]


@pytest.mark.parametrize("args, match", [
    (("", "ROW_X", 0.5, RUN), "kind and row"),
    (("SHFL", "ROW_SHFLM", 0.5, ""), "names the run"),
    (("SHFL", "ROW_SHFLM", 0.5, "note:20261016T1640Z-draft"), "names the run"),
    (("SHFL", "ROW_SHFLM", 0.0, RUN), "positive"),
])
def test_census_line_refusals(args, match):
    with pytest.raises(rows.RowError, match=match):
        rows.census_line(*args)


def test_u_of_takes_exactly_one_value():
    assert rows.u_of(300.08, "", "", "2", "") == pytest.approx(600.16)
    assert rows.u_of(300.08, "", "", "", "12.5") == 12.5
    with pytest.raises(rows.RowError, match="exactly one"):
        rows.u_of(300.08, "0.5", "", "2", "")
    with pytest.raises(rows.RowError, match="derates a rate only"):
        rows.u_of(300.08, "", "0.02", "2", "")
    with pytest.raises(rows.RowError, match="columns"):
        rows.census_csv([{"kind": "SHFL"}])


def test_macs_a_word_pins():
    """The vf3j note's published numbers: SHFL at 0.499815 is 471 MACs a word, a 16-B TMA box at 0.027522 is 68,480."""
    assert round(rows.macs_a_word(0.499815, 32)) == 471
    assert round(rows.macs_a_word(0.027522, 4), -1) == 68480


def test_schema_is_machine_models():
    files = sorted((MACHINE_MODEL / "data" / "census").glob("*.csv"))
    assert files
    lines = []
    for p in files:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            assert tuple(r.fieldnames) == rows.CENSUS_COLUMNS, p.name
            lines += r
    sys.path.insert(0, str(MACHINE_MODEL))
    try:
        from verity_machine_model import model
    finally:
        sys.path.remove(str(MACHINE_MODEL))
    for d in lines:
        assert rows.u_of(300.08, d["rate"], d["noise"], d["cycles"], d["u"]) == model.u_of(
            300.08, d["rate"], d["noise"], d["cycles"], d["u"])


# ---- verdicts ----------------------------------------------------------------------------------------------------------


def test_declared_verdicts():
    lm = probe.judge(Path("lm"), table("lm"), fns("lm"))
    got = {v["probe"]: (v["passed"], v["correct"]) for v in lm}
    assert got == {"ldsm_x1": (True, True), "stsm_x1": (True, True), "lds128": (True, True),
                   "bad_ldsm_merged": (False, True), "bad_stsm_conflict": (False, True)}
    lines = probe.verdict_lines(lm)
    assert any(ln.startswith("REFUSED lm:bad_ldsm_merged (known-invalid, refuse by loops: as declared)") for ln in lines)
    assert any(ln.startswith("REFUSED lm:bad_stsm_conflict (known-invalid, refuse by smem: as declared)") for ln in lines)
    atom = {v["probe"]: v for v in probe.judge(Path("atom"), table("atom"), fns("atom"))}
    assert all(v["passed"] and v["correct"] for v in atom.values())
    assert atom["redg_add32_l2"]["sass_sha256"] == atom["redg_add32"]["sass_sha256"]  # one kernel, two probes


def test_own_rank_atome64_verdicts():
    """64-bit EXCH and CAS at the own rank lower to one ATOM.E each; red.add.u64 nests its CAST.SPIN fallback."""
    cl = probe.judge(Path("cl"), table("cl"), fns("cl"))
    assert {v["probe"]: (v["passed"], v["correct"]) for v in cl} == {
        "dsm_atom_exch64_self": (True, True), "dsm_atom_cas64_self": (True, True), "dsm_red_add64_self": (False, True)}
    assert any(ln.startswith("REFUSED cl:dsm_red_add64_self (known-invalid, refuse by loops: as declared)")
               for ln in probe.verdict_lines(cl))


def test_a_known_invalid_probe_must_be_refused_by_its_check():
    t = {p["name"]: p for p in table("lm")}
    passes = dict(t["ldsm_x1"], refuse="loops")
    wrong_check = dict(t["bad_stsm_conflict"], refuse="loops")
    v = {x["probe"]: x for x in probe.judge(Path("lm"), [passes, wrong_check], fns("lm"))}
    assert not v["ldsm_x1"]["correct"] and not v["bad_stsm_conflict"]["correct"]
    assert any("NOT AS DECLARED" in ln for ln in probe.verdict_lines(list(v.values())))


# ---- rates and rows ----------------------------------------------------------------------------------------------------


def times(probe_name: str, cycles: list[list[int]], iters: int, throttle: str = "0x0") -> list[dict]:
    return [{"op": "time", "probe": probe_name, "rep": r, "iters": iters, "cyc": c, "smid": list(range(len(c))),
             "blocks": len(c), "throttle": throttle, "sm_mhz": {"before": 2092, "during": 2092, "after": 2092}}
            for r, c in enumerate(cycles)]


def test_rate_is_the_median_block_then_the_median_rep():
    p = {"expect": "IMMA.16832.S8.S8=8;DADD=2", "warps": "8;8", "counted": "", "tpb": 512}
    t = times("spec", [[1000, 1100, 4000], [2000, 2000, 2000], [1000, 1000, 1000]], iters=10)
    r = probe.rate_of(p, t)
    assert r["counted"] == "IMMA.16832.S8.S8"
    assert r["rate"] == pytest.approx(8 * 10 * 8 / 1100)
    assert r["rates"]["1:DADD"] == pytest.approx(2 * 10 * 8 / 1100)
    assert r["spread"] == pytest.approx((640 / 1000 - 640 / 2000) / (640 / 1100))
    assert not r["throttled"] and r["distinct_sms"] == 3
    assert probe.rate_of(p, times("spec", [[1000]], 10, throttle="0x4"))["throttled"]
    assert not probe.rate_of(p, times("spec", [[1000]], 10, throttle="0x1"))["throttled"]
    assert r["stop_gap"] is None


def test_an_until_loop_counts_only_the_fixed_roles_time():
    """side.cu's hs_* role runs while the honest warps do: its rate is its trips over the stop's cycles, block by block, not
    its iterations over the kernel's; stop_gap is how far before the kernel's end the honest warps stopped."""
    p = {"expect": "IMMA.16832.S8.S8=8;REDG*=1", "warps": "8;2", "counted": "", "tpb": 320, "until": 2}
    t = times("hs_red32", [[1000, 1000]], iters=10)
    t[0]["trips"], t[0]["stop"] = [450, 400], [900, 1000]
    r = probe.rate_of(p, t)
    assert r["rate"] == pytest.approx(8 * 10 * 8 / 1000)
    assert r["rates"]["1:REDG*"] == pytest.approx((450 / 900 + 400 / 1000) / 2)
    assert r["stop_gap"] == pytest.approx((0.1 + 0.0) / 2)


# ---- EPS1's facets -----------------------------------------------------------------------------------------------------

eps1 = _load("opcode_probe_eps1", POUW / "opcode_probe" / "eps1.py")


def test_facets_are_the_upper_hull_with_the_margin():
    pts = [(0.0, 0.5, "alone"), (0.05, 0.25, "mix"), (0.02, 0.2, "inside"), (0.0625, 0.0, "imma")]
    fs = eps1.facets(pts, 0.02)
    assert [f["tight"] for f in fs] == [["alone", "mix"], ["mix", "imma"]]   # each axis is an edge's end: dropped
    assert (fs[0]["MMA"], fs[0]["X"]) == pytest.approx((9.8, 1.96)) and fs[1]["MMA"] == pytest.approx(15.68)
    for f in fs:   # every point at or below 0.98 on every facet, and its tight points at 0.98
        assert all(f["MMA"] * m + f["X"] * x <= 0.98 + 1e-12 for m, x, _ in pts)
        assert all(f["MMA"] * m + f["X"] * x == pytest.approx(0.98) for m, x, n in pts if n in f["tight"])


def test_a_near_duplicate_facet_is_dropped():
    """A facet within 0.2% of one at least as tight on both coefficients adds nothing (the margin is 2%)."""
    fs = eps1.facets([(0.0, 0.5, "alone"), (0.0001, 0.49999, "flat"), (0.0625, 0.0, "imma")], 0.02)
    assert len(fs) == 1 and fs[0]["tight"] == ["flat", "imma"]
    assert (fs[0]["MMA"], fs[0]["X"]) == pytest.approx((15.68, 1.96), rel=0.002)


def test_prior_points_give_back_the_published_facets(tmp_path):
    """The vertices of published facets, fed back with the alone point, rebuild those facets: --prior adds the K rows'
    evidence to a new run's points without moving what they established."""
    rows_csv = tmp_path / "rows.csv"
    rows_csv.write_text("row,status,runs,description\n"
                        f'KX0,joint,{RUN},"MMA 2.000 + X 1.960 SMSP-cycles (test)"\n'
                        f'KX1,joint,{RUN},"MMA 15.000 + X 0.400 SMSP-cycles (test)"\n'
                        f'KY0,joint,{RUN},"MMA 3.000 + Y 1.000 SMSP-cycles (not X)"\n')
    pts, runs = eps1.prior_points(rows_csv, "KX", "X", 0.02)
    assert runs == {RUN} and [n for *_, n in pts] == ["KX0/KX1", "KX1:axis"]
    fs = eps1.facets(pts + [(0.0, 0.98 / 1.96, "alone")], 0.02)
    got = {(round(f["MMA"], 6), round(f["X"], 6)) for f in fs}
    assert {(2.0, 1.96), (15.0, 0.4)} <= got


sm89 = _load("opcode_probe_sm89_census", POUW / "opcode_probe" / "sm89" / "census.py")


def test_sm89_pipe_rate_counts_the_loops_instructions_on_its_pipe():
    """An ALU probe's loop counter runs on the ALU pipe, so the pipe's rate counts it; SHFL's doesn't (SHF isn't SHFL)."""
    loop = lambda pat, n: {"loops": [{"counts": {pat: n}, "top": {pat.rstrip("*"): n, "IADD3": 1, "ISETP.GE.AND": 1,
                                                                    "BRA": 1}}]}
    assert sm89.pipe_rate({"rates": {"0:LOP3.LUT": 1.6}}, loop("LOP3.LUT", 16)) == pytest.approx(1.6 * 18 / 16)
    assert sm89.pipe_rate({"rates": {"0:SHFL.BFLY": 0.5}}, loop("SHFL.BFLY", 16)) == pytest.approx(0.5)
    assert sm89.pipe_rate({"rates": {"0:IMAD": 1.9}}, loop("IMAD", 16)) == pytest.approx(1.9)
    fold = {"loops": [{"counts": {"LOP3.LUT": 32, "IMAD": 32, "LEA.HI": 32},
                       "top": {"LOP3.LUT": 32, "IMAD": 32, "LEA.HI": 32, "IADD3": 1, "ISETP.GE.AND": 1, "BRA": 1}}]}
    rates = {"0:LOP3.LUT": 0.96, "0:IMAD": 0.96, "0:LEA.HI": 0.96}
    assert sm89.pipe_rate({"rates": rates}, fold) == pytest.approx(0.96 * 66 / 32)   # LOP3, LEA.HI and the counter


def test_sm89_frontier_counts_every_non_mma_instruction_as_x():
    """A mix's loop overhead could be X in an attacker's program, so the facet points count it."""
    v = {"loops": [{"counts": {"IMMA.16832.S8.S8": 8, "LOP3.LUT": 4},
                    "top": {"IMMA.16832.S8.S8": 8, "LOP3.LUT": 4, "IADD3": 1, "ISETP.GE.AND": 1, "BRA": 1}}]}
    m, x = sm89.loop_rates({"rates": {"0:IMMA.16832.S8.S8": 0.25, "0:LOP3.LUT": 0.125}}, v)
    assert (m, x) == pytest.approx((0.25, 0.25 * 7 / 8))


def test_sm89_uniform_lines(tmp_path):
    """UNI runs on UP at its fastest alone loop's rate, every uniform instruction counted, and costs issue the least SMSP-
    cycles a uniform instruction adds beside a base loop, less the margin (JI89 that over SCHED)."""
    rows_, verdicts = [], []

    def put(name, top, counted, cycles):   # cycles: SMSP-cycles a warp-iteration
        rows_.append({"probe": name, "run": RUN, "counted": counted, "rates": {f"0:{counted}": 4 / cycles * top[counted]}})
        verdicts.append({"probe": name, "verdict": {"loops": [{"top": {**top, "IADD3": 1, "BRA": 1}}]}})

    for i, p in enumerate(sm89.UNI_ALONE):
        put(p, {"ULOP3.LUT": 64}, "ULOP3.LUT", 128 * (1 + i / 100))          # 2.0 a SM-clock at best
    put("ut_0", {"IMMA.16816.S8.S8": 8}, "IMMA.16816.S8.S8", 64)
    for p in sm89.UNI_TENSOR[1:]:
        put(p, {"ULOP3.LUT": 48, "IMMA.16816.S8.S8": 8}, "IMMA.16816.S8.S8", 100)   # UP's 96, 4% over
    for base, ps in sm89.UNI_BESIDE.items():
        put(base, {"LOP3.LUT": 32, "IMMA.16816.S8.S8": 8}, "IMMA.16816.S8.S8", 150)
        for p in ps:
            put(p, {"ULOP3.LUT": 64, "LOP3.LUT": 32, "IMMA.16816.S8.S8": 8}, "IMMA.16816.S8.S8",
                150 + 64 * (0.8 if p == "uh_64m" else 1.0))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_))
    (tmp_path / "gate.json").write_text(json.dumps({"verdicts": verdicts}))
    rws, cen = [["issue", "measured", "r0", "i"], ["JI89", "joint", "r0", "j"]], []
    sm89.uniform(tmp_path, rws, cen)
    c = 0.8 / 1.02
    assert cen == [["UNI", "UP", "2", "0.02", "", "", "", "", RUN], ["UNI", "issue", "", "", f"{c:.6g}", "", "", "", RUN],
                   ["UNI", "JI89", "", "", f"{c / sm89.TK.SCHED:.9g}", "", "", "", RUN]]
    assert rws[0][2] == f"r0 {RUN}" and "least at uh_64m" in rws[0][3] and rws[1][2] == f"r0 {RUN}"
    assert rws[2][:3] == ["UP", "measured", RUN] and "within 4.2%" in rws[2][3]


def test_one_census_line_a_kind_and_row(tmp_path):
    binary = tmp_path / "atom"
    binary.write_bytes(b"\x7fELF")
    base = {"tpb": 1024, "kind": "ATOMS32", "row": "ROW_ATOMS32", "words": 32, "counted": "", "warps": "", "note": "",
            "blocks": 0}
    probes = [dict(base, name="atoms_add32", expect="ATOMS.ADD=8"), dict(base, name="atoms_cas32", expect="ATOMS.CAS=8"),
              dict(base, name="joint", kind="", row="", expect="IMMA*=8")]
    verdicts = [{"binary": "atom", "probe": p["name"], "sass_sha256": "ab" * 32} for p in probes]
    lines = (times("atoms_add32", [[65536] * 4], 128) + times("atoms_cas32", [[68000] * 4], 128)
             + times("joint", [[9000] * 4], 128))
    side = probe.make_rows(binary, probes, verdicts, lines, RUN, "")
    census = probe.one_a_row(side)
    assert census == [rows.census_line("ATOMS32", "ROW_ATOMS32", 0.5, RUN)]  # 8 x 128 iterations x 32 warps / 65536
    by = {s["probe"]: s for s in side}
    assert by["atoms_cas32"]["census_by"] == by["atoms_add32"]["census_by"] == "atoms_add32"
    assert by["joint"]["census"] is None and by["joint"]["class"] is None
    assert by["atoms_add32"]["macs_a_word"] == pytest.approx(rows.macs_a_word(0.5, 32))
    assert "0.500000" in probe.row_line(by["atoms_add32"]) and "macs/word=" in probe.row_line(by["atoms_add32"])
    unnamed = probe.make_rows(binary, probes[:1], verdicts, lines, "", "")
    assert unnamed[0]["census"] is None and "names the run" in unnamed[0]["census_refused"]


def test_uniform_reads_each_loop_from_its_sass(tmp_path, capsys):
    """`eps1.py uniform`: V and U from the gate's loop in the listing, T = 4 n / rate, and a uniform instruction's cost as
    T's slope in U within a fixed stream (the K = 0 point included, a step variant left out of the fit)."""
    out = tmp_path / "out"
    (out / "sass").mkdir(parents=True)
    body = ["IMMA.16832.S8.S8 R0, R4, R8, R0 ;", "LOP3.LUT R1, R1, R2, R3, 0x96, !PT ;", "UIADD3 UR4, UPT, UPT, UR4, 0x1, URZ ;"]
    loops = {"ux_0": body, "ux_16": body + ["UIMAD UR5, UR5, UR6, URZ ;"] * 16,
             "ux_16s": body + ["@UP0 USHF.L.U32 UR5, UR5, 0x3, URZ ;"] * 16}
    sass, verdicts = [], []
    for name, ops in loops.items():
        ins = [f"        /*{16 * i:04x}*/                   {op}" for i, op in enumerate(ops + ["BRA.U UP0, 0x0 ;"])]
        sass += [f"\t\tFunction : {name}", "        /*fff0*/                   NOP ;", *ins]
        verdicts.append({"probe": name, "passed": True,
                         "verdict": {"loops": [{"head": "0000", "tail": f"{16 * len(ops):04x}"}]}})
    (out / "sass" / "uniform.sass").write_text("\n".join(sass) + "\n")
    (out / "gate.json").write_text(json.dumps({"verdicts": verdicts}))
    rates = {"ux_0": 4 / 10, "ux_16": 4 / 26, "ux_16s": 4 / 18}
    (out / "rows.jsonl").write_text("".join(json.dumps({"probe": p, "rate": r, "counted": "IMMA.16832.S8.S8", "tags": []})
                                            + "\n" for p, r in rates.items()))
    assert eps1.main(["uniform", str(out)]) == 0
    got = capsys.readouterr().out
    assert re.search(r"^ux_16 +3 +17 +26\.000 .* 1\.000 *$", got, re.M)
    assert re.search(r"^ux_16s +3 +17 +18\.000 .* 0\.500 *$", got, re.M)
    assert "  ux: 1->17:1.000  fit 1.000 over U 1..17" in got


def test_ceiling_reads_each_knob_and_names_the_highest_host_clock(tmp_path, capsys):
    """`eps1.py ceiling`: a row a knob with its rc, what NVML then reports and the clock each kernel reached by the host's
    clock, and the highest host-timed clock against NVML's max."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "found.json").write_text(json.dumps({"uuid": "GPU-x", "index": "0", "max_mhz": "2430", "max_mem_mhz": "12481",
                                                "found_pl_w": "600.00", "found_vboost": "0", "max_vboost": "4"}))
    trials = [{"trial": "lgc-max", "cmd": "nvidia-smi -lgc 2430,2430", "rc": 0, "out": "GPU clocks set to \"(gpcClkMin 2430)\""},
              {"trial": "gpc-offset+105", "cmd": "clocks.py offset 105", "rc": 0, "out": "{\"rc\": 2, \"error\": \"Invalid Argument\"}"}]
    (out / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
    states = [{"trial": "found", "gpc_vf_offset_range": {"value": [0, 0]}, "mem_vf_offset_range": {"value": [0, 0]},
               "power_limit_constraints_mw": {"value": [300000, 600000]}, "clock_offsets_p0": {"graphics": {"min": 0, "max": 0}}},
              {"trial": "lgc-max", "clock_now": {"sm": {"value": [2422]}}, "clock_max": {"sm": {"value": [2430]}},
               "applications": {"graphics": {"value": [2430]}}}]
    (out / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    cal = [{"trial": "lgc-max", "probe": "p_light", "clock_mhz_host": h, "clock_mhz_gt_max": h + 0.2,
            "clock_mhz_gt_min": h + 0.2, "nvml_sm_mhz": 2422, "nvml_mw": 150000, "throttle": "0x0"} for h in (2429.4, 2429.9)]
    cal += [{"trial": "gpc-offset+105", "probe": "p_light", "clock_mhz_host": 2101.0, "clock_mhz_gt_max": 2101.1,
             "nvml_sm_mhz": 2092, "nvml_mw": 140000, "throttle": "0x0"}]
    (out / "calib.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cal))
    assert eps1.main(["ceiling", str(out)]) == 0
    got = capsys.readouterr().out
    assert "GPC VF offset range [0, 0], memory [0, 0]" in got
    assert re.search(r"^lgc-max +0 +2422 +2430 +2430 p_light +2429\.9 +2430\.1 +2429\.6 +2422 +150 +0x0 +GPU clocks", got, re.M)
    assert re.search(r"^gpc-offset\+105 +0 .* p_light +2101\.0", got, re.M)
    assert "highest host-timed clock: 2429.9 MHz (p_light at lgc-max), -0.004% against NVML's max 2430" in got


def test_l2gather_reports_bytes_a_clock_and_mma_kept(tmp_path, capsys):
    """`eps1.py l2gather`: bytes a SM a clock (rate * words * 4) and TB/s (* nSM * clock) a gather path, the best alone
    and beside, the MMA kept beside the honest step, and whether any path beats QSTAR's 3.3 TB/s."""
    out = tmp_path / "out"
    out.mkdir()
    imma, ldg = "IMMA.16832.S8.S8", "LDG.E.ENL2.256.STRONG.GPU"
    jl = [
        {"probe": "honest", "rate": 2.0, "words": None, "blocks": 100, "sm_mhz": 2000, "rates": {f"0:{imma}": 2.0}},
        {"probe": "gl_w32", "rate": 0.01, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s128",
         "rates": {f"0:{ldg}": 0.01}},
        {"probe": "gl_w32_fast", "rate": 0.02, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s16",
         "rates": {f"0:{ldg}": 0.02}},
        {"probe": "gh_w32", "rate": 0.008, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s128 beside",
         "rates": {f"0:{imma}": 1.0, f"1:{ldg}": 0.008}},
    ]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jl))
    assert eps1.main(["l2gather", str(out)]) == 0
    got = capsys.readouterr().out
    assert "honest alone: IMMA 2.00000/SM/clk" in got
    assert re.search(r"^gl_w32 +s128 +32 +0\.0100 +10\.240 +2\.048 +nan", got, re.M)
    assert re.search(r"^gh_w32 .* 8\.192 +1\.638 +0\.5000", got, re.M)
    assert "best alone:  gl_w32_fast 20.480 B/SM/clk = 4.096 TB/s" in got
    assert "best beside: gh_w32 8.192 B/SM/clk = 1.638 TB/s, MMA kept 0.5000" in got
    assert "fastest path 4.096 TB/s beats QSTAR's 3.3 TB/s table kernel" in got


pipes_verdict = _load("opcode_probe_pipes_verdict", POUW / "opcode_probe" / "pipes_verdict.py")


def _pipes_rows(**over: dict[str, float]) -> dict[str, dict]:
    rates = {
        "lop3": {"LOP3.LUT": 2.0}, "mufu": {"MUFU.EX2": 0.5},
        "plop3": {"PLOP3.LUT": 2.0}, "mix_plop3_lop3": {"PLOP3.LUT": 0.5, "LOP3.LUT": 1.5},
        "sgxt": {"SGXT": 2.0}, "mix_sgxt_lop3": {"SGXT": 1.0, "LOP3.LUT": 1.0},
        "bmsk": {"BMSK": 2.0}, "mix_bmsk_lop3": {"BMSK": 1.0, "LOP3.LUT": 1.0},
        "i2f_f2i": {"I2F.U32.RP": 0.25, "F2I.FTZ.U32.TRUNC.NTZ": 0.25},
        "mix_i2f_f2i_mufu": {"I2F.U32.RP": 0.125, "F2I.FTZ.U32.TRUNC.NTZ": 0.125, "MUFU.EX2": 0.25},
        "i2f16_f2i16": {"I2F.S16": 0.25, "F2I.TRUNC.NTZ": 0.25},
        "mix_i2f16_f2i16_mufu": {"I2F.S16": 0.125, "F2I.TRUNC.NTZ": 0.125, "MUFU.EX2": 0.25},
        "f2f64": {"F2F.F64.F32": 0.029, "F2F.F32.F64": 0.029},
        "frnd": {"FRND.FLOOR": 0.5, "FADD": 0.5}, "mix_frnd_mufu": {"FRND.FLOOR": 0.25, "FADD": 0.25, "MUFU.EX2": 0.25},
        "bar_or": {"BAR.RED.OR*": 0.05}, "mix_bar_or_lop3": {"BAR.RED.OR*": 0.04, "LOP3.LUT": 1.9},
        "iadd64": {"IADD.64": 4.0}, "mix_iadd64_lop3": {"IADD.64": 2.0, "LOP3.LUT": 2.0},
        "iabs": {"IABS": 2.0}, "mix_iabs_lop3": {"IABS": 1.0, "LOP3.LUT": 1.0},
        "dfma": {"DFMA": 0.0625},
        "mix_f2f64_mufu24": {"F2F.F64.F32": 0.01, "F2F.F32.F64": 0.01, "MUFU.EX2": 0.24},
        "mix_f2f64_dfma": {"F2F.F64.F32": 0.0145, "F2F.F32.F64": 0.0145, "DFMA": 0.0145},
        "i2f64": {"I2F.F64": 0.029, "F2I.F64.TRUNC": 0.029},
        "mix_i2f64_mufu24": {"I2F.F64": 0.01, "F2I.F64.TRUNC": 0.01, "MUFU.EX2": 0.24},
        "mix_i2f64_dfma": {"I2F.F64": 0.02, "F2I.F64.TRUNC": 0.02, "DFMA": 0.02},
        "s64": {"I2F.S64": 0.029, "F2I.S64.TRUNC": 0.029},
        "mix_s64_mufu24": {"I2F.S64": 0.01, "F2I.S64.TRUNC": 0.01, "MUFU.EX2": 0.24},
        "mix_s64_dfma": {"I2F.S64": 0.0145, "F2I.S64.TRUNC": 0.0145, "DFMA": 0.0145},
        "bar16": {"BAR.RED.OR*": 0.1}, "mix_bar16_lop3": {"BAR.RED.OR*": 0.1, "LOP3.LUT": 1.9},
    }
    for n in ("cvt_s32", "cvt_u16", "cvt_s8", "cvt_frnd", "cvt_i2i", "cvt_i2ip"):
        rates |= {n: {"A": 0.25, "B": 0.25}, f"mix_{n}_mufu": {"A": 0.125, "B": 0.125, "MUFU.EX2": 0.25}}
    for n in ("cvt_u64", "cvt_frnd64", "cvt_f64_u32", "cvt_f64_s64"):
        rates |= {n: {"A": 0.029, "B": 0.029}, f"mix_{n}_mufu24": {"A": 0.02, "B": 0.02, "MUFU.EX2": 0.5},
                  f"mix_{n}_dfma": {"A": 0.0145, "B": 0.0145, "DFMA": 0.0145}}
    rates |= over
    return {n: {"probe": n, "binary": "pipes", "rates": {f"0:{k}": v for k, v in r.items()}} for n, r in rates.items()}


def test_pipes_verdict_holds_each_pipe_to_its_rho(tmp_path, capsys):
    """`pipes_verdict.py`: each alu row's opcode within 2 alone and with LOP3, each 32-bit conversion within 0.5 alone and
    with MUFU.EX2 (FRND's FADD off the XU), each 64-bit one within 0.0625 alone and with DFMA, LOP3 + BAR.RED/2 within 2;
    a conversion at the ALU's rate fails the XU, a 64-bit pair at DFMA's rate beside DFMA fails f64."""
    held, lines = pipes_verdict.verdict(_pipes_rows())
    assert held, lines
    assert "FRND.FLOOR 0.5000 alone, 0.2500 + 0.2500 = 0.5000 beside MUFU" in lines
    assert "BAR.RED.OR 0.0500 alone; LOP3 + BAR/2 = 1.9000 + 0.0400/2 = 1.9200" in lines
    assert "IADD.64 4.0000 alone, 4.0000 with LOP3 (beside alu)" in lines
    assert ("I2F.F64+F2I.F64.TRUNC 0.0580 alone, 0.0400 + 0.0200 = 0.0600 beside DFMA, 0.0200 + 0.2400 = 0.2600 beside "
            "24 MUFU (on the XU)") in lines
    assert "A+B 0.0580 alone, 0.0290 + 0.0145 = 0.0435 beside DFMA, 0.0400 + 0.5000 = 0.5400 beside 24 MUFU (beside the XU)" \
        in lines
    assert "BAR.RED.OR (16 warps) 0.1000 alone; beside a LOP3 stream LOP3 + BAR/2 = 1.9000 + 0.1000/2 = 1.9500" in lines
    held, lines = pipes_verdict.verdict(_pipes_rows(i2f_f2i={"I2F.U32.RP": 1.0, "F2I.FTZ.U32.TRUNC.NTZ": 1.0}))
    assert not held and lines[0] == "I2F.U32.RP+F2I.FTZ.U32.TRUNC.NTZ alone 2.0000 > sfu's 0.5"
    held, lines = pipes_verdict.verdict(_pipes_rows(mix_s64_dfma={"I2F.S64": 0.029, "F2I.S64.TRUNC": 0.029,
                                                                  "DFMA": 0.029}))
    assert not held and lines[0] == "I2F.S64+F2I.S64.TRUNC beside DFMA 0.0870 > f64's 0.0625"
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for n, r in _pipes_rows().items() if n != "mix_bar_or_lop3"))
    assert pipes_verdict.main([str(rows)]) == 0
    assert capsys.readouterr().out.startswith("verdict: hypothesis failed: no row for mix_bar_or_lop3; ")


collective_verdict = _load("opcode_probe_collective_verdict", POUW / "opcode_probe" / "collective_verdict.py")


def _collective_rows(**over: dict[str, float]) -> dict[str, dict]:
    match = {"match_any_u32": {"MATCH.ANY": 0.5}, "match_any_u64": {"MATCH.ANY.U64": 0.25},
             "mix_u32_lop3_32_ffma_32": {"MATCH.ANY": 0.379, "LOP3.LUT": 1.516, "FFMA": 1.516},
             "mix_u64_lop3_16": {"MATCH.ANY.U64": 0.2499, "LOP3.LUT": 0.4999}}
    coll = {"redux_add": {"REDUX.SUM": 0.499, "IADD3": 0.56, "MOV": 0.499}, "vote_ballot": {"VOTE.ANY": 1.39},
            "mix_redux_match": {"REDUX.SUM": 0.25, "MATCH.ANY": 0.25},
            "mix_redux_lop3_16": {"REDUX.SUM": 0.4984, "LOP3.LUT": 0.9969},
            "mix_vote_lop3_16": {"VOTE.ANY": 0.55, "LOP3.LUT": 1.65},
            "mix_vote_lop3_32": {"VOTE.ANY": 0.36, "LOP3.LUT": 1.78},
            "mix_vote_lop3_64": {"VOTE.ANY": 0.21, "LOP3.LUT": 1.89}}
    out = {n: {"probe": n, "binary": b, "rates": {f"0:{k}": v for k, v in r.items()}}
           for b, rs in (("match", match), ("collective", coll)) for n, r in rs.items()}
    for n, r in over.items():
        out[n]["rates"] = {f"0:{k}": v for k, v in r.items()}
    return out


def test_collective_verdict_holds_match_and_redux_to_one_unit_and_vote_to_16_alu_lanes(tmp_path, capsys):
    """`collective_verdict.py`: MATCH.ANY and REDUX beside each other within their faster one's rate alone, .U64 as 2
    units, MATCH beside LOP3 and FFMA within it, LOP3 + VOTE/2 within 2; REDUX and MATCH on two units fail it, and so
    does a VOTE that spends a whole LOP3 of alu."""
    held, lines = collective_verdict.verdict(_collective_rows())
    assert held, lines
    assert lines[0] == "the unit 0.5000 (MATCH.ANY 0.5000, REDUX.SUM 0.4990 alone)"
    assert "REDUX.SUM + MATCH.ANY = 0.2500 + 0.2500 = 0.5000 beside each other" in lines
    assert "MATCH beside LOP3 and FFMA at most 0.4998 units (mix_u64_lop3_16, of 2)" in lines
    assert ("VOTE.ANY 1.3900 alone; LOP3 + VOTE/2 = 1.6500 + 0.5500/2 = 1.9250, 1.7800 + 0.3600/2 = 1.9600, "
            "1.8900 + 0.2100/2 = 1.9950 at Q 16, 32, 64") in lines
    assert "REDUX.SUM beside 16 LOP3 0.4984 (LOP3 0.9969)" in lines
    held, lines = collective_verdict.verdict(_collective_rows(mix_redux_match={"REDUX.SUM": 0.45, "MATCH.ANY": 0.45}))
    assert not held and lines[0] == "REDUX.SUM + MATCH.ANY 0.9000 > the unit's 0.5000 beside each other: two units"
    held, lines = collective_verdict.verdict(_collective_rows(mix_vote_lop3_64={"VOTE.ANY": 0.21, "LOP3.LUT": 1.99}))
    assert not held and lines[0] == "LOP3 + VOTE.ANY/2 2.0950 > alu's 2 (mix_vote_lop3_64)"
    match, coll = tmp_path / "match.jsonl", tmp_path / "collective.jsonl"
    rows = _collective_rows()
    match.write_text("".join(json.dumps(r) + "\n" for r in rows.values() if r["binary"] == "match"))
    coll.write_text("".join(json.dumps(r) + "\n" for n, r in rows.items() if r["binary"] == "collective"
                            and n != "mix_redux_match"))
    assert collective_verdict.main([str(match), str(coll)]) == 0
    assert capsys.readouterr().out.startswith("verdict: hypothesis failed: no row for mix_redux_match; the unit 0.5000")


def test_jitter_reports_host_timing_spread_and_the_gap_to_the_chip(tmp_path, capsys):
    """`eps1.py jitter`: each (trial, length) group's host-timed SM clock mean, its random spread (rel u%) and the
    systematic gap below the chip's own timer, and a per-trial u taken at the longest length (the tightest bound)."""
    out = tmp_path / "out"
    out.mkdir()

    def rep(trial, ln, host, gt, rep_i):
        return {"op": "calib", "trial": trial, "ref": "p_mma0", "len_s": ln, "probe": "p_mma0", "rep": rep_i,
                "clock_mhz_host": host, "clock_mhz_gt_median": gt, "nvml_sm_mhz": 2092, "nvml_mw": 150000,
                "throttle": "0x0"}

    jl = [rep("locked", 0.002, 2090.0, 2120.0, 0), rep("locked", 0.002, 2110.0, 2120.0, 1),
          rep("locked", 0.2, 2098.0, 2104.0, 0), rep("locked", 0.2, 2102.0, 2104.0, 1)]
    (out / "jitter.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jl))
    assert eps1.main(["jitter", str(out)]) == 0
    got = capsys.readouterr().out
    assert re.search(r"^locked +0\.002 +2 +2100\.0 +14\.14 +10\.00 +0\.476 +2120\.0 +0\.943 +2092", got, re.M)
    assert re.search(r"^locked +0\.200 +2 +2100\.0 +2\.83 +2\.00 +0\.095 +2104\.0 +0\.190 +2092", got, re.M)
    assert "locked: at 0.200s the host clock pins the SM clock to within 0.095% (random) and sits 0.190% below" in got
    assert "u ~ 0.285%" in got


# ---- the probes themselves -------------------------------------------------------------------------------------------


def _cuda_tools() -> bool:
    try:
        probe.nvcc()
        G.H.find_tools()
    except (SystemExit, G.H.SassGateFailed, OSError):
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_tools(), reason="needs nvcc, cuobjdump and nvdisasm (CUDA 13)")
def test_probes_build_and_gate(tmp_path):
    """What the fixture tests can't: today's nvcc still compiles every probe to SASS whose verdict is the declared one."""
    built = probe.build(tmp_path, sorted((POUW / "opcode_probe" / "probes").glob("*.cu")))
    _, verdicts = probe.gate_all(built)
    wrong = probe.verdict_lines([v for v in verdicts if not v["correct"]])
    assert not wrong, "\n".join(wrong)
    assert {v["probe"] for v in verdicts if not v["passed"]} == {"bad_ldsm_merged", "bad_stsm_conflict",
                                                                 "dsm_red_add64_self"}


ji89 = _load("opcode_probe_sm89_ji", POUW / "opcode_probe" / "sm89" / "ji.py")


def test_sm89_ji_pins_only_the_schedule_its_name_says():
    top = {"FMUL": 112, "IMMA.16816.S8.S8": 8, "MOV": 1, "IADD3": 1, "ISETP.GE.AND": 1, "BRA": 1}
    assert ji89.exact("j16_fmuli_q112_a2u1_w2", top)
    assert not ji89.exact("j16_fmuli_q112_a2u1_w2", {**top, "CALL.REL.NOINC": 1})
    assert not ji89.exact("j16_fmuli_q112_a2u1_w2", {**top, "FMUL": 111})
    assert not ji89.exact("j16_fmuli_q112_a2u1_w2", {**top, "IMMA.16816.S8.S8": 16})
    assert ji89.exact("j16_iadd3_q64_a2u1_w2", {"IMAD.IADD": 32, "IADD3": 33, "IMMA.16816.S8.S8": 8, "ISETP.GE.AND": 1})
    names = [p[0] for p in ji89.grid(32)]
    assert "j32_fmuli_q112_a2u1_w2" in names and len(names) == len(set(names))


def test_sm89_ji_refit_is_the_upper_hull_at_the_margin(tmp_path):
    """JI89 at ADA-89 is 4.0306 x_IMMA16816 + 0.93458 v <= 1: a program under it leaves the line as it is, one above it
    (load 1.083) splits it into facets through that program at the margin, each holding every point."""
    def run(d, progs):
        d.mkdir()
        (d / "rows.jsonl").write_text("".join(json.dumps({"probe": p, "run": RUN, "rates": {
            "0:IMMA.16816.S8.S8": 4 * x, "0:FMUL": 4 * v}}) + "\n" for p, x, v in progs))
        return d

    under = run(tmp_path / "under", [("j16_a", 0.05, 0.5)])
    (old,) = ji89.refit([under], "IMMA16816")
    assert old[:2] == pytest.approx((4.03060164, 0.934579439))
    beat = run(tmp_path / "beat", [("j16_a", 0.05, 0.5), ("j16_b", 0.06, 0.9)])
    assert ji89.worst(ji89.loads([beat]))["JI89"] == (pytest.approx(4.03060164 * 0.06 + 0.934579439 * 0.9), "j16_b")
    fs = ji89.refit([beat], "IMMA16816")
    assert len(fs) == 2 and all("j16_b" in f[2] for f in fs)
    for a, b, _ in fs:
        assert a * 0.06 * 1.02 + b * 0.9 * 1.02 <= 1 + 1e-9
        assert a * 0.05 * 1.02 + b * 0.5 * 1.02 < 1
        assert b * 1.07 <= 1 + 1e-9 and a / 4.03060164 <= 1 + 1e-9


def test_sm89_ji_points_join_only_their_own_family(tmp_path):
    """A ji32 program joins the E families its lane beat, as (IMMA.16832, every non-MMA instruction) a SMSP-cycle; one
    whose adds ptxas split onto the IMAD pipe stays out, and IMMA.16816 programs and IMAD lanes join nothing."""
    rows_, verdicts = [], []
    for name, top in (("j32_fmuli_q112_a2u2_w2", {"FMUL": 224, "IMMA.16832.S8.S8": 16, "MOV": 6, "IADD3": 1}),
                      ("j32_axi_q64_a3u1_w2", {"IADD3": 33, "LOP3.LUT": 32, "IMMA.16832.S8.S8": 8, "IMAD.MOV.U32": 6}),
                      ("j32_iadd3_q112_a8u1_w2", {"IMAD.IADD": 70, "IADD3": 43, "IMMA.16832.S8.S8": 8}),
                      ("j32_imadi_q32_a3u1_w2", {"IMAD": 32, "IMMA.16832.S8.S8": 8, "IADD3": 1}),
                      ("j16_fmuli_q112_a2u1_w2", {"FMUL": 112, "IMMA.16816.S8.S8": 8, "IADD3": 1})):
        mma = next(op for op in top if op.startswith("IMMA"))
        top = {**top, "ISETP.GE.AND": 1, "BRA": 1}
        rows_.append({"probe": name, "run": RUN, "rates": {f"0:{mma}": 0.25 * top[mma] / 8}})
        verdicts.append({"probe": name, "verdict": {"loops": [{"top": top, "counts": {mma: top[mma]}}]}})
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_))
    (tmp_path / "gate.json").write_text(json.dumps({"verdicts": verdicts}))
    pts = sm89.ji_points([tmp_path])
    assert set(pts) == {"ffma", "iadd3", "lop3"}
    (m, x, name), = pts["ffma"]
    assert name == f"j32_fmuli_q112_a2u2_w2@{RUN}" and m == pytest.approx(0.125) and x == pytest.approx(0.125 * 233 / 16)
    assert [n.split("@")[0] for *_, n in pts["iadd3"]] == [n.split("@")[0] for *_, n in pts["lop3"]] == ["j32_axi_q64_a3u1_w2"]
