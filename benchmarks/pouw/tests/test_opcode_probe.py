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


def test_l2gather_reports_bytes_a_clock_and_mma_kept(tmp_path, capsys):
    """`eps1.py l2gather`: bytes a SM a clock (rate * words * 4) and TB/s (* nSM * clock) a gather path, the best alone
    and beside, the MMA kept beside the honest step, and whether any path beats QSTAR's 3.3 TB/s."""
    out = tmp_path / "out"
    out.mkdir()
    imma = "IMMA.16832.S8.S8"
    jl = [
        {"probe": "honest", "rate": 2.0, "words": None, "blocks": 100, "sm_mhz": 2000, "rates": {f"0:{imma}": 2.0}},
        {"probe": "gl_w32", "rate": 0.01, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s128",
         "rates": {"0:LDG.E*": 0.01}},
        {"probe": "gl_w32_fast", "rate": 0.02, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s16",
         "rates": {"0:LDG.E*": 0.02}},
        {"probe": "gh_w32", "rate": 0.008, "words": 256, "blocks": 100, "sm_mhz": 2000, "note": "s128 beside",
         "rates": {f"0:{imma}": 1.0, "1:LDG.E*": 0.008}},
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
