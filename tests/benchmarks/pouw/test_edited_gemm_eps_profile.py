"""eps_profile.py on made-up ncu exports: a mnemonic's census class, the mbarrier retry loop counted as WAIT, the hot
loop, the bound max_r sum_k n_k u_kr with its binding row (with WAIT and without), eps_sched against ncu's elapsed
cycles, and main's table, --json and --toml on a made-up run directory, priced with the ACCEL model's own census."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import eps_profile as E
import pytest
from verity_machine_model import model as mmod

#: the ACCEL model's directory (its package and data/), where the workspace has it: what the job scripts give as --mm
MODEL = Path(mmod.__file__).resolve().parents[1]
# One call of an arm: 10^7 MMAs over 100 SMSPs, so 10^5 MMAs a SMSP. A MMA comes with 2 LOP3, a quarter of an
# IMAD.WIDE, two instructions of the mbarrier's retry loop, and 0.05 of FROB (no class) and EXIT.
SASS = [("IMMA.16832.S8.S8 R0, R1, R2", 6_000_000, 6), ("IMMA.16832.S8.S8 R4, R1, R2", 4_000_000, 0),
        ("@P0 LOP3.LUT R1, R2, R3", 20_000_000, 2), ("IMAD.WIDE R2, R3, R4", 2_500_000, 0), ("YIELD", 5_000_000, 0),
        ("SYNCS.PHASECHK.TRYWAIT R5", 5_000_000, 1), ("@!P0 BRA 0x10", 5_000_000, 1), ("BRA 0x20", 5_000_000, 0),
        ("FROB R1", 300_000, 0), ("EXIT", 200_000, 0)]
PER_MMA = {"MMA": 1.0, "ALU": 2.0, "WAIT": 2.0, "IMADW": 0.25, "NOROW": 0.05}
RAW = {"device__attribute_multiprocessor_count": ("", 25), "sm__cycles_elapsed.avg": ("cycle", 2_400_000),
       "sm__cycles_active.avg": ("cycle", 1_600_000), "gpu__time_duration.sum": ("msecond", 1.2),
       "sm__cycles_elapsed.avg.per_second": ("Ghz", 2.0),
       "smsp__issue_active.avg.pct_of_peak_sustained_active": ("%", 41.5),
       "sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_active": ("%", 62.5),
       "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": ("%", 30.5),
       "smsp__average_warp_latency_issue_stalled_barrier_per_issue_active.ratio": ("", 0.75),
       "smsp__warps_eligible.avg.per_cycle_active": ("", 1.5)}
CENSUS = {"MMA": {"M": 400.0, "issue": 100.0}, "ALU": {"A": 200.0, "issue": 100.0},
          "IMADW": {"A": 400.0, "issue": 100.0}, "NOROW": {"issue": 100.0}}


def write_report(base: Path, sass: list[tuple[str, int, int]] = SASS) -> Path:
    """ncu's two exports of one call, BASE.sass.csv and BASE.raw.csv; the report's path, as analyse takes it."""
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.sass.csv").write_text(
        '==PROF== a banner\n"Source","Instructions Executed","Warp Stall Sampling (All Samples)"\n'
        + "".join(f'"{src}","{n}","{stall}"\n' for src, n, stall in sass))
    Path(f"{base}.raw.csv").write_text("\n".join(",".join(f'"{x}"' for x in row) for row in (
        ["ID", *RAW], ["", *(unit for unit, _ in RAW.values())], ["0", *(value for _, value in RAW.values())])) + "\n")
    return Path(f"{base}.ncu-rep")


def bound_of(per_mma: dict[str, float], census: dict, cycle_u: float) -> tuple[float, str]:
    """max_r sum_k n_k u_kr in SMSP-cycles a MMA, and the row r that binds; a class with no census row is NOROW's."""
    rows = {r for k in per_mma for r in census.get(k, census["NOROW"])}
    return max((sum(n * census.get(k, census["NOROW"]).get(r, 0.0) for k, n in per_mma.items()) / cycle_u, r)
               for r in rows)


@pytest.mark.parametrize(("mnemonic", "kind"), [
    ("IMMA.16832.S8.S8", "MMA"), ("IMAD.WIDE", "IMADW"), ("IMAD", "IMAD"), ("IDP.4A.S8.S8", "IMAD"),
    ("F2FP.SATFINITE.BF16.F32.PACK_AB", "F2FS"), ("F2FP.BF16.F32.PACK_AB", "ALU"), ("LOP3.LUT", "ALU"),
    ("LDSM.16.MT88.4", "LDSM_X4T"), ("LDSM.16.M88.2", "LDSM_X2"), ("LDSM.16.M88", "LDSM_X1"), ("LDS.U.128", "LDS128"),
    ("LDS", "LDS32"), ("STS.128", "STS128"), ("STS", None), ("LDG.E.128", "LDG128"), ("UTMALDG", "UTMA16K"),
    ("RED.E.ADD.64.STRONG.GPU", "RED64"), ("REDG.E.ADD.STRONG.GPU", "RED32"), ("UMOV", "UNI"), ("R2UR", "UNI"),
    ("EXIT", "NOROW"), ("FROB", None)])
def test_a_mnemonic_has_its_census_class(mnemonic, kind):
    assert E.kind_of(mnemonic, "UTMA16K") == kind


def test_the_retry_loop_is_wait_and_the_hot_loop_is_what_runs_half_as_often_as_a_mma(tmp_path):
    rep = write_report(tmp_path / "o-e_l2q")
    allc, hot, per = E.sass_counts(rep.with_suffix(".sass.csv"))
    assert allc == {"IMMA.16832.S8.S8": 1e7, "LOP3.LUT": 2e7, "IMAD.WIDE": 2.5e6, "WAIT.YIELD": 5e6,
                    "WAIT.SYNCS.PHASECHK.TRYWAIT": 5e6, "WAIT.BRA": 1e7, "FROB": 3e5, "EXIT": 2e5}
    assert hot == {k: v for k, v in allc.items() if k not in ("IMAD.WIDE", "FROB", "EXIT")}   # under 3,000,000: not hot
    assert [op for op, _, _ in per][4:8] == ["WAIT.YIELD", "WAIT.SYNCS.PHASECHK.TRYWAIT", "WAIT.BRA", "WAIT.BRA"]
    assert E.by_class(allc, "UTMA16K") == ({"MMA": 1e7, "ALU": 2e7, "IMADW": 2.5e6, "WAIT": 2e7, "NOROW": 5e5},
                                           {"FROB": 3e5})


@pytest.mark.parametrize(("loop", "waits"), [
    (["YIELD", "SYNCS.PHASECHK.TRYWAIT R5", "BRA 0x10"], 3),                    # no out-of-line spin after it
    (["SYNCS.PHASECHK.TRYWAIT R5", "@!P0 NANOSLEEP.SYNCS R6", "@!P0 SYNCS.PHASECHK R5", "@!P0 BRA 0x10", "BRA 0x20"],
     5),                                                                        # the loop with a suspend-time hint
    (["YIELD", "SYNCS.PHASECHK.TRYWAIT R5", "IADD3 R1, R2, R3"], 0)])           # not a loop: nothing branches back
def test_each_form_of_the_retry_loop(tmp_path, loop, waits):
    sass = [("IMMA.16832.S8.S8 R0, R1, R2", 100, 0), *((src, 40, 0) for src in loop), ("LOP3.LUT R1, R2, R3", 100, 0),
            ("EXIT", 1, 0)]
    allc, _, per = E.sass_counts(write_report(tmp_path / "r", sass).with_suffix(".sass.csv"))
    assert sum(op.startswith("WAIT.") for op, _, _ in per) == waits
    assert sum(n for op, n in allc.items() if op.startswith("WAIT.")) == 40 * waits and allc["LOP3.LUT"] == 100


def test_the_bound_is_the_binding_rows_time_with_wait_and_without(tmp_path):
    r = E.analyse(write_report(tmp_path / "o-e_l2q"), CENSUS, 100.0, "UTMA16K")
    k, loop = r["kernel"], r["loop"]
    assert k["per_mma"] == PER_MMA and k["unpriced"] == {"FROB": 0.03}
    assert (k["mma"], k["inst_per_mma"], k["wait_per_mma"]) == (1e7, pytest.approx(5.3), 2.0)
    # issue: every instruction a cycle, 5.3; A: 2 LOP3 at 2 and a quarter IMAD.WIDE at 4, 5.0; M: the MMA at 4.
    assert (k["bound"], k["facet"]) == (pytest.approx(5.3), "issue") == bound_of(PER_MMA, CENSUS, 100.0)
    assert (k["bound_work"], k["facet_work"]) == (pytest.approx(5.0), "A")
    assert [(pytest.approx(v), row) for v, row in k["facets"]] == [(5.3, "issue"), (5.0, "A"), (4.0, "M")]
    assert (loop["inst_per_mma"], loop["bound"], loop["facet"]) == (5.0, pytest.approx(5.0), "issue")
    # 2,400,000 elapsed and 1,600,000 active cycles over 10^5 MMAs a SMSP: 24 and 16 cycles a MMA.
    assert (r["mma_per_smsp"], r["measured"], r["measured_active"]) == (1e5, 24.0, 16.0)
    assert r["eps_sched"] == pytest.approx(1 - 5.3 / 24) and r["eps_sched_work"] == pytest.approx(1 - 5.0 / 24)
    assert r["eps_sched_active"] == pytest.approx(1 - 5.3 / 16)
    assert r["eps_sched_loop_bound"] == pytest.approx(1 - 5.0 / 24)
    assert (r["ms"], r["mhz"], r["sm_count"], r["issue_pct"], r["tensor_pct"]) == (1.2, 2000.0, 25, 41.5, 62.5)
    assert r["pipes_pct"] == {"sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_active": 62.5,
                              "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": 30.5}
    assert r["stalls"] == {"smsp__average_warp_latency_issue_stalled_barrier_per_issue_active.ratio": 0.75}
    assert r["stall_samples_by_opcode"] == {"IMMA": 0.6, "LOP3": 0.2, "WAIT": 0.2, "IMAD": 0.0, "FROB": 0.0,
                                            "EXIT": 0.0}
    assert r["warps_eligible"] == 1.5


def test_a_report_with_no_sass_export_is_no_profile(tmp_path):
    assert E.analyse(tmp_path / "o-plain.ncu-rep", CENSUS, 100.0, "UTMA16K") is None


def write_run(o: Path) -> None:
    """A run directory: e_l2q and e_plain profiled at shape o (its gate's meta gives 256 x 512 x 128) and e_l2q at o2k
    (no gate: the script's own shape), with the hot bench of shape o."""
    o.mkdir(parents=True)
    write_report(o / "ncu" / "o-e_l2q")
    write_report(o / "ncu" / "o-e_plain", [("IMMA.16832.S8.S8 R0, R1, R2", 10_000_000, 0),
                                           ("LOP3.LUT R1, R2, R3", 15_000_000, 0), ("EXIT", 100, 0)])
    write_report(o / "ncu" / "o2k-e_l2q")
    (o / "gate_o_meta.json").write_text(json.dumps({"m": 256, "n": 512, "k": 128}))
    (o / "bench.jsonl").write_text("a line that is not JSON\n" + json.dumps(
        {"m": 256, "n": 512, "k": 128, "arms": {"plain": {"ms": 1.0, "mhz": 2000.0},
                                                "e_plain": {"ms": 1.0, "mhz": 2000.0},
                                                "e_l2q": {"ms": 1.25, "mhz": 2000.0}}}) + "\n")


def test_main_prices_a_run_directory_with_the_models_census(tmp_path, monkeypatch, capsys):
    o = tmp_path / "run7" / "out"
    write_run(o)
    model = mmod.model(MODEL / "data")
    epoch = mmod.vector(model.current, MODEL / "data").epoch
    census = mmod.census(model, epoch, MODEL / "data")
    bound, facet = bound_of(PER_MMA, census, model.cycle_u)
    work, work_facet = bound_of({k: n for k, n in PER_MMA.items() if k != "WAIT"}, census, model.cycle_u)
    monkeypatch.setattr(sys, "path", list(sys.path))    # main puts --mm on the path
    monkeypatch.setattr(sys, "argv", ["eps_profile.py", str(o), "--mm", str(MODEL), "--json", str(tmp_path / "p.json"),
                                      "--toml", str(tmp_path / "p.toml")])
    assert E.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"# EPS1 PROFILE: the edited GEMM against the machine model (out, epoch {epoch})"
    assert [line for line in out if line.startswith("## ")] == ["## o = 256 x 512 x 128", "## o2k = 2048 x 4096 x 4096"]
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in out if line.startswith("| ")]
    head = rows[0]
    at_o = {r[0]: dict(zip(head, r, strict=True)) for r in rows[1:rows.index(head, 1)]}
    assert list(at_o) == ["plain", "e_plain", "e_l2q"]        # the sweep's order; plain has the hot bench alone
    assert at_o["plain"]["hot ms"] == "1.0000" and at_o["plain"]["bound"] == at_o["plain"]["eps_sched"] == ""
    # hot: 1.25 ms at 2,000 MHz is 2,500,000 cycles over 10^5 MMAs a SMSP, 25 cycles a MMA; the fold is the 0.8
    # instructions a MMA over e_plain's 2.5 that are not the wait, a quarter of them a word.
    assert at_o["e_l2q"] == dict(zip(head, [
        "e_l2q", "3.0", "1.2500", "1.250", "1.250", "2000", "5.30", "2.00", "0.20", "41.5", "62.5", f"{bound:.2f}",
        facet, f"{work:.2f}", work_facet, "24.00", "25.00", f"{1 - bound / 25:.3f}", f"{1 - work / 25:.3f}"],
        strict=True))
    under_ncu = "instructions a MMA, 1.2000 ms under ncu at 2000 MHz, issue 41.5%, warps eligible 1.50 a cycle"
    assert [line for line in out if line.startswith("### ")] == [
        f"### e_plain at o: 2.50 {under_ncu}", f"### e_l2q at o: 5.30 {under_ncu}",
        f"### e_l2q at o2k: 5.30 {under_ncu}"]
    assert "unpriced (as NOROW): FROB 0.030" in out

    got = json.loads((tmp_path / "p.json").read_text())
    assert got["epoch"] == epoch and list(got["reports"]) == ["o-e_l2q", "o-e_plain", "o2k-e_l2q"]
    assert list(got["hot"]) == ["256x512x128"]
    kernel = got["reports"]["o-e_l2q"]["kernel"]
    assert (kernel["bound"], kernel["facet"]) == (pytest.approx(bound), facet)
    assert got["reports"]["o-e_l2q"]["measured_hot"] == 25.0 and "measured_hot" not in got["reports"]["o2k-e_l2q"]

    toml = (tmp_path / "p.toml").read_text().split("\n")
    assert toml[0].startswith("# the edited GEMM's executed SASS a MMA")
    assert toml[0].endswith(f"bounds over census {epoch}")
    assert toml[1].startswith("# WAIT: the mbarrier retry loop") and toml[2:4] == ['run_dir = "run7"', ""]
    tables = [i for i, line in enumerate(toml) if line.startswith("[")]
    assert [toml[i] for i in tables] == ['[profile.e_l2q."256x512x128"]', '[profile.e_plain."256x512x128"]',
                                         '[profile.e_l2q."2048x4096x4096"]']
    assert toml[tables[0] + 1:tables[1]] == [
        "inst_per_mma = 5.3000", "wait_per_mma = 2.0000", "measured_cycles_per_mma = 25.0000",
        "ncu_cycles_per_mma = 24.0000", f"bound_cycles_per_mma = {bound:.4f}", f'facet = "{facet}"',
        f"work_bound_cycles_per_mma = {work:.4f}", f'work_facet = "{work_facet}"', f"eps_sched = {1 - bound / 25:.4f}",
        f"eps_sched_work = {1 - work / 25:.4f}", "issue_pct = 41.50", "tensor_pct = 62.50",
        "per_mma = { ALU = 2.0000, WAIT = 2.0000, MMA = 1.0000, IMADW = 0.2500, NOROW = 0.0500 }",
        "loop_per_mma = { ALU = 2.0000, WAIT = 2.0000, MMA = 1.0000 }", ""]
    # with no hot bench of its shape, a profile's measured cycles are ncu's one call
    assert toml[tables[2] + 3:tables[2] + 5] == ["measured_cycles_per_mma = 24.0000", "ncu_cycles_per_mma = 24.0000"]
