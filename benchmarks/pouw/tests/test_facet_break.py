"""facet_break: SASS opcodes to census classes, a program's load on every facet (IMMA.16832 through the generic MMA lines,
every other form only through its own), the R1 and KFFMA0 violation TENSOR's interleaved points make at VF3n-16, the hull
refit and the family refit (`family`, `--forms`), the L2 facet (only loads that skip L1 count), the full count (`--full`:
every instruction of the gate's loop), and the generator's grid (seeded, unique names, exact SASS counts).
`test_grid_slice_builds_and_gates` builds a
slice of the grid with nvcc and gates it; it skips without nvcc and nvdisasm.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
FB = POUW / "facet_break"
EPOCH = "VF3n-16"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


loads = _load("facet_break_loads", FB / "loads.py")
gen = _load("facet_break_generate", FB / "generate.py")

#: TENSOR's tensor_kinds records: IMMA.16832 alone, beside IADD3+FFMA interleaved at 4 lane ops a
#: MMA (mixu R32), and FFMA alone
MIXU_R32 = {"probe": "tensor_kinds", "mode": "mixu", "form": "imma_s8_m16n8k32", "ops_per_mma": 4.0, "mma_warps": 16,
            "op_warps": 0, "cyc_med": 20034030, "mhz": 2090.9, "mmas_per_sm": 4634624, "ops_per_sm": 18538496}
ALONE = {"probe": "tensor_kinds", "mode": "alone", "form": "imma_s8_m16n8k32", "ops_per_mma": 0.0, "mma_warps": 8,
         "op_warps": 0, "cyc_med": 19796214, "mhz": 2093.9, "mmas_per_sm": 4799040, "ops_per_sm": 0}
FFMA = {"probe": "tensor_kinds", "mode": "opalone", "form": "ffma", "ops_per_mma": 0.0, "mma_warps": 0, "op_warps": 32,
        "cyc_med": 74754167, "mhz": 2093.5, "mmas_per_sm": 0, "ops_per_sm": 286119936}


@pytest.fixture(scope="module")
def census():
    m = loads.model()
    return loads.census(m, EPOCH), m.cycle_u


def _records(tmp_path: Path, *recs: dict) -> Path:
    d = tmp_path / "r20261017-041754-7630" / "out"
    d.mkdir(parents=True)
    p = d / "tensor_kinds.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_class_of():
    want = {"IMMA.16832.S8.S8": "IMMA16832", "HMMA.1684.F32.TF32": "HMMA1684T", "IMAD": "IMAD", "IMAD.WIDE.U32": "IMADW",
            "IADD3": "ALU", "LOP3.LUT": "ALU", "SHF.*": "ALU", "IADD.64": "ALU64", "SEL": "SEL", "FFMA": "FFMA",
            "FMUL": "FFMA", "HFMA2*": "HFMA2", "*MNMX*": "VIMNMX", "DMUL": "DMUL", "DFMA": "W2", "MUFU.EX2": "XU",
            "POPC": "XU", "SHFL.BFLY": "SHFL", "MOV": "NOROW", "NOP": None, "LDS": "LDS32", "LDS.128": "LDS128",
            "LDG.E": "LDG32", "LDG.E.CONSTANT": "LDG32", "LDG.E.128": "LDG128", "LDS.64": "LDS64", "LDS.U8": "LDS32",
            "LDG.E.64": "LDG64", "TLD.LZ": None, "LDG.E.ENL2.256": "LDG256", "LDG.E.NA.128": "LDG128NA",
            "LDG.E.128.STRONG.GPU": "LDG128CG", "LDG.E.ENL2.256.STRONG.GPU": "LDG256CG", "ULOP3.LUT": "UNI",
            "USHF.L.U32": "UNI", "UPRMT": "UNI", "USEL": "UNI", "UIADD3": "UNI", "UMOV": "UNI", "LDCU": "UNI",
            "R2UR": "UNI", "UIADD3.64": "UNI64", "UMOV.64": "UNI64", "UIMAD.WIDE.U32": "UNI64", "LDCU.64": "UNI64",
            "REDG.E.ADD.STRONG.GPU": "RED32", "REDG.E.ADD.64.STRONG.GPU": "RED64",
            "REDG.E.ADD.F32x4.FTZ.RN.STRONG.GPU": "REDV", "REDG.E.ADD.F32.FTZ.RN.STRONG.GPU": None,
            "REDG.E.MAX.STRONG.GPU": "REDG32", "REDG.E.XOR.64.STRONG.GPU": "REDG64",
            "ATOMG.E.ADD.STRONG.GPU": "ATOMG32", "ATOMG.E.CAS.64.STRONG.GPU": "ATOMG64", "ATOMS.ADD": "ATOMS32",
            "ATOMS.CAST.SPIN.64": "ATOMS64", "ATOM.E.ADD.STRONG.GPU": "ATOME32", "ATOM.E.EXCH.64.STRONG.GPU": "ATOME64",
            "STS.128": "STS128", "STS": "STS32", "STS.U8": "STS32", "STG.E": "STG32", "STG.E.128": "STG128",
            "STG.E.64": None, "REDUX.XOR": "REDUX", "ISETP.LE.AND": "ISETP", "VIADD": "VIADD", "VIADDMNMX": None,
            "LDGSTS.E.BYPASS.128": "LDGSTS16", "LDSM.16.M88": "LDSM_X1",
            "LDSM.16.MT88.4": "LDSM_X4T", "LDSM.8.MT1616.2": "LDSM8T_X2", "STSM.16.M88.2": "STSM_X2",
            "STSM.8.MT1616.4": "STSM8T_X4", "UBLKRED.G.S.XOR": "TMAred512", "UBLKCP.S.G": None, "UTMALDG.2D": None,
            "UBLKRED*": "TMAred512", "UBLKPF.L2": "UNI"}
    assert {op: loads.class_of(op) for op in want} == want


def test_a_bulk_instruction_counts_its_shared_memory_bytes():
    """A5b: a bulk or TMA instruction counts `S/99 KB` in UTMABIG, `S` its size, which it reads or writes
    in a block's shared memory on at least one side, so DS-120 charges it `S` on DS_smem and at most its one issue slot.
    fold_bulk.cu's fbw trip reduces 32 KB and twice 16 KB into global memory."""
    big = loads.BULK_BYTES
    assert loads.bulk_counts("UBLKRED.G.S.XOR", 32768) == {"UTMABIG": Fraction(32768, big)}
    for op in ("UBLKRED.S.S.XOR", "UBLKCP.S.S", "UBLKCP.S.G", "UTMALDG.2D", "UTMASTG.3D"):
        assert loads.bulk_counts(op, 512) == {"UTMABIG": Fraction(512, big)}, op
    assert loads.bulk_counts("UBLKCP.S.G", big) == {"UTMABIG": Fraction(1)} and big == 99 * 1024
    for op, n in (("UBLKCP.S.G", big + 16), ("UBLKRED.G.S.ADD", 0), ("UBLKPF.L2", 512)):
        with pytest.raises(ValueError):
            loads.bulk_counts(op, n)


def test_a_side_unit_counted_by_wildcard_takes_its_class_from_its_name():
    row = {"probe": "hs_red64", "rates": {"0:IMMA.16832.S8.S8": 0.4, "1:REDG*": 0.08}}
    assert loads.rates_by_class(row) == pytest.approx({"IMMA16832": 0.1, "RED64": 0.02})
    assert loads.rates_by_class({"probe": "sa_blkred512", "rates": {"0:UBLKRED*": 0.04}}) == {"TMAred512": 0.01}
    assert loads.rates_by_class({"probe": "sa_blkred4k", "rates": {"0:UBLKRED*": 0.04}}) == {"TMAred512": 0.01}
    assert loads.rates_by_class({"probe": "sa_blkcp1k", "rates": {"0:UBLKCP*": 0.04}}) == {}


def test_a_tld_takes_its_width_from_the_program_and_unpriced_loads_are_listed(census):
    c, _ = census
    row = {"probe": "lm_i16832_tld4n_line_q8_w16", "rates": {"0:IMMA.16832.S8.S8": 0.2, "0:TLD.LZ": 0.4}}
    assert loads.rates_by_class(row) == pytest.approx({"IMMA16832": 0.05, "TLD4w": 0.1})
    assert loads.unpriced(row, c) == {}
    row = {"probe": "ld_lds32_lds64_r2_line_q16_w32", "rates": {"0:LDS": 0.4, "0:LDS.64": 0.4}}
    assert loads.rates_by_class(row) == pytest.approx({"LDS32": 0.1, "LDS64": 0.1})
    assert loads.unpriced(row) == {}
    assert loads.unpriced(row, c) == pytest.approx({"LDS.64": 0.1})
    assert loads.unpriced({"probe": "side_tld", "rates": {"0:TLD.LZ": 0.4}}) == pytest.approx({"TLD.LZ": 0.1})


def test_fine_grid_sweeps_the_near_frontier_and_the_ji3n_0_form():
    names = [g["name"] for g in gen.grid(0, 0, family="fine")]
    assert len(set(names)) == len(names) == 1346
    assert {"mx_h16816f16_ffma_q128_p0_w16", "mx_i16832_imad_q34_p0_w16", "mx_i16832_ffma_q192_p0_w32",
            "mx_i16832_dfma_q1_p1_w16"} <= set(names)


def test_rt_families_are_unique_and_hold_the_programs_that_broke_ji3s_and_ji3t():
    sizes = {"rt": 726, "rt2": 408, "rt3": 120, "rt4": 95, "rt5": 168, "rt6": 192, "rt7": 144, "rt8": 377, "rt9": 319,
             "rt10": 99, "rt11": 105}
    names = {f: [g["name"] for g in gen.grid(0, 0, family=f)] for f in sizes}
    assert {f: len(set(n)) for f, n in names.items()} == {f: len(n) for f, n in names.items()} == sizes
    assert "jm_i16832_fmuli_q120_p0_a3u20_w8" in names["rt4"]
    assert {"jm_i16832_fmuli_q120_p0_a3u20_w8_k3", "jm_i16832_fmuli_q120_p0_a3u16_sts128x2_w8",
            "jm_i16832_none_q0_p0_a2u8_ldsm4x4_w8", "jm_i16832_fmuli_shfl_q96_p0_a2u8_w8"} <= set(names["rt9"])
    assert {"jm_i16832_fmuli_q120_p0_a3u20_w8_k2", "jm_i16832_axi_q112_p0_a2u16_w8",
            "jm_i16832_fmuli_dp4ai_q144_p0_a3u20_w8"} <= set(names["rt10"])
    assert {"jm_i16832_fmuli_q120_p0_a3u20_w8_k1", "jm_i16832_fmuli_uaxi_q120_p0_a3u16_w8",
            "jm_i16832_fmuli_axi_uaxi_q144_p0_a2u20_w8", "jm_i16832_uaxi_q64_p0_a3u12_w8"} <= set(names["rt11"])
    assert "jm_isp16864_fmuli_q128_p0_a2u16_w8" in names["rt5"]
    assert {"jx_i16832_isp16864_fmuli_q128_a1u20_w8", "jx_i16832_osf16864_none_q0_a8u4_w8", "al_mufu_popc_w16",
            "jm_isp16864_fmuli_q128_p0_a2u20_w8_k3"} <= set(names["rt8"])


def test_two_forms_take_their_accumulators_and_unroll_and_keep_the_ji_names():
    g = gen._jx("i16832", "isp16864", "fmuli", 128, 2, na=1, u=20)
    assert gen.expect_of(g) == ("IMMA.16832.S8.S8=80,IMMA.SP.16864.S8.S8=80,FMUL=2560", "", 0)
    assert "FB_JMIX2(jx_i16832_isp16864_fmuli_q128_a1u20_w8, I16832, ISP16864, 1, 20, L_FMULI, L_NONE, L_NONE, 128, 256)" \
        in gen.emit_cu([g | {"name": gen.name_of(g)}])
    assert gen.name_of(gen._jx("i16832", "h1688t", ("fmuli", "axi"), 96, 4)) == "jx_i16832_h1688t_fmuli_axi_q96_w16"


def test_mem_grid_is_seeded_unique_and_counts_each_slot():
    a = gen.grid(2, 30, family="mem")
    assert a == gen.grid(2, 30, family="mem") and len({g["name"] for g in a}) == len(a)
    g = next(g for g in a if g["name"] == "ld_lds32_lds128_r4_line_q16_w32")
    assert gen.expect_of(g)[0] == "LDS=12,LDS.128=4"
    ops, fn = gen.smem_of(g)
    assert ops.count("LDS.128*1") == 4 and ops.count("LDS*1") == 12 and fn.startswith("fb::soff<fb::D_LDS32, ")
    g = next(g for g in a if g["name"] == "lm_i16832_ldg32_line_q4_w16")
    assert gen.expect_of(g)[0] == "IMMA.16832.S8.S8=8,LDG.E=4" and gen.smem_of(g) == ("", "")
    assert not any("ldg256" in g["name"] for g in gen.grid(2, 120, family="mem"))


def test_the_l2_facet_counts_only_the_bytes_of_loads_that_skip_l1(tmp_path, census, capsys):
    a = gen.grid(4, 10, family="l2")
    assert a == gen.grid(4, 10, family="l2") and len({g["name"] for g in a}) == len(a)
    g = next(g for g in a if g["name"] == "l2_g256cg_scat_q16_w32")
    assert gen.expect_of(g) == ("LDG.E.ENL2.256.STRONG.GPU=16", "", 0) and gen.smem_of(g) == ("", "")
    assert "FB_L2(l2_g256cg_scat_q16_w32, D_L2G256CG, P_SCAT, 16, 1024)" in gen.emit_cu([g])
    row = {"probe": "l2_g128na_line_q8_w16", "run": "r20261017-000000-0000", "blocks": 188, "sm_mhz": 2000.0,
           "rates": {"0:LDG.E.NA.128": 0.03, "1:LDG.E.128": 0.5, "2:LDG.E.ENL2.256.STRONG.GPU": 0.01}}
    bps = (0.03 * 32 * 16 + 0.01 * 32 * 32) * 188 * 2000e6
    assert loads.l2_bytes_per_s(row) == pytest.approx(bps)
    assert loads.program(row, {}, 300.0, l2=bps / 1.1)["load"][loads.L2_ROW] == pytest.approx(1.1)
    assert loads.L2_ROW not in loads.program(row, {}, 300.0)["load"]
    assert set(loads.unpriced(row, census[0])) == {"LDG.E.NA.128", "LDG.E.ENL2.256.STRONG.GPU"}
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(row) + "\n")
    assert loads.main(["summary", str(p), "--epoch", EPOCH, "--l2", str(bps / 1.1)]) == 1
    assert "L2tbl        load 1.1000" in capsys.readouterr().out


def test_only_imma16832_takes_the_generic_mma_lines(census):
    c, cu = census
    imma = loads.facet_loads({"IMMA16832": 1.0}, c, cu)
    hmma = loads.facet_loads({"HMMA16816F16": 1.0}, c, cu)
    assert imma["M"] == pytest.approx(16) and "R1" in imma and "KFFMA0" in imma
    assert imma["JI3n_0"] == pytest.approx(c["IMMA16832"]["JI3n_0"] / cu)
    assert set(hmma) == {"M", "issue", "JI3n_0", "JI3n_1", "JI3n_2"}
    assert "R1" in loads.facet_loads({"HMMA16816F16": 1.0}, c, cu, generic=True)


def test_interleaved_imma_ffma_beats_r1_and_kffma0(tmp_path, census, capsys):
    """The axes hold, the interleaved point beats both rows."""
    c, cu = census
    path = _records(tmp_path, MIXU_R32, ALONE, FFMA)
    by = {p["probe"].split(":")[0]: loads.program(p, c, cu) for p in loads.load_rows([path])}
    mix = by["mixu"]
    assert mix["x"]["IMMA16832"] == pytest.approx(0.0578344, rel=1e-5)
    assert mix["x"]["FFMA"] == mix["x"]["ALU"] == pytest.approx(0.1156688, rel=1e-5)
    assert mix["load"]["R1"] == pytest.approx(1.0240, abs=1e-4)
    assert mix["load"]["KFFMA0"] == pytest.approx(1.0296, abs=1e-4)
    assert by["alone"]["load"]["M"] == pytest.approx(0.970, abs=1e-3)
    assert max(by["opalone"]["load"].values()) < 1
    assert loads.main(["summary", str(path), "--epoch", EPOCH, "--worst"]) == 1
    out = capsys.readouterr().out
    assert "2 rows beaten" in out and "R1 " in out and "[ε]" in out and "KFFMA0" in out


def test_refit_is_the_hull_at_the_margin(tmp_path, capsys):
    path = _records(tmp_path, MIXU_R32, ALONE, FFMA)
    assert loads.main(["refit", str(path), "--row", "R1", "--epoch", EPOCH]) == 0
    out = capsys.readouterr().out
    facets = [ln for ln in out.splitlines() if ln.strip().startswith("MMA ")]
    assert facets and all("max load 0.980" in ln for ln in facets)
    assert any("mixu:imma_s8_m16n8k32" in ln for ln in facets)


def test_family_refits_beaten_rows_as_one_hull_and_closes_them(tmp_path, capsys):
    """R1 and KFFMA0 as one family (VF3q-16's EFFMA): the upper hull of their vertices and the measured programs at the
    margin, each new row's lines at the new epoch, and every old line closed at the base epoch."""
    path = _records(tmp_path, MIXU_R32, ALONE, FFMA)
    assert loads.main(["family", str(path), "--family", "R1", "KFFMA0", "--name", "EFFMA", "--kind", "FFMA",
                       "--epoch", EPOCH, "--emit", "VF3q-16"]) == 0
    out = capsys.readouterr().out.splitlines()
    new = {tuple(ln.split(": ", 1)[1].split(",")[:2]): float(ln.split(",")[4]) for ln in out
           if ln.startswith("census/VF3q-16.csv: ")}
    assert new == pytest.approx({("MMA", "EFFMA0"): 14.8965878, ("FFMA", "EFFMA0"): 1.02417307,
                                 ("MMA", "EFFMA1"): 15.7425815, ("FFMA", "EFFMA1"): 0.601176252})
    mixu = 0.0578344 * new["MMA", "EFFMA0"] + 0.1156688 * new["FFMA", "EFFMA0"]
    assert mixu == pytest.approx(0.98, abs=1e-5)
    assert sum(ln.startswith("rows.csv: EFFMA") for ln in out) == 2
    assert {ln.split()[2] for ln in out if ln.startswith("close: ")} == {"MMA,R1", "FFMA,R1", "MMA,KFFMA0", "FFMA,KFFMA0"}
    assert all(ln.endswith(f"until {EPOCH}") for ln in out if ln.startswith("close: "))


def test_family_forms_lowers_only_the_beaten_form_and_generic_mma_is_its_least(tmp_path, census, capsys):
    """`family --forms` (VF3q-16's JI3q rows): a HMMA.16816.F16 + FFMA point above JI3n_0 lowers that form's coefficient
    on the row at JI3n_0's vector coefficient until the point sits at the margin; every other form keeps its own, and
    the generic MMA lines take the least form's."""
    c, cu = census
    d = tmp_path / "r20261017-060543-cc28" / "out"
    d.mkdir(parents=True)
    p = d / "rows.jsonl"
    p.write_text(json.dumps({"probe": "mx_h16816f16_ffma_q64_p0_w16",
                             "rates": {"0:HMMA.16816.F16": 0.32, "0:FFMA": 2.0}}) + "\n")
    assert loads.main(["family", str(p), "--family", "JI3n_0", "JI3n_1", "JI3n_2", "--name", "JI3t_", "--forms",
                       "--epoch", EPOCH, "--emit", "VF3q-16"]) == 0
    new = {tuple(ln.split(": ", 1)[1].split(",")[:2]): float(ln.split(",")[4])
           for ln in capsys.readouterr().out.splitlines() if ln.startswith("census/VF3q-16.csv: ")}
    assert new["FFMA", "JI3t_0"] == pytest.approx(c["FFMA"]["JI3n_0"] / cu)
    assert 0.08 * new["HMMA16816F16", "JI3t_0"] + 0.5 * new["FFMA", "JI3t_0"] == pytest.approx(0.98)
    assert new["IMMA16832", "JI3t_0"] == pytest.approx(c["IMMA16832"]["JI3n_0"] / cu)
    assert new["MMA", "JI3t_0"] == min(v for (k, r), v in new.items() if r == "JI3t_0" and k in loads.MMA_FORMS)


def test_old_first_tensor_kinds_build_lane_is_iadd3_alone(tmp_path):
    rec = MIXU_R32 | {"mode": "spec", "op_warps": 8}
    path = _records(tmp_path, rec, {"mode": "opalone", "form": "iadd3_32", "cyc_med": 1, "mmas_per_sm": 0,
                                    "ops_per_sm": 1, "mma_warps": 0, "op_warps": 32})
    x = loads.load_rows([path])[0]["x"]
    assert set(x) == {"IMMA16832", "ALU"}


def test_full_counts_every_instruction_of_the_gates_loop_and_unclassed_ones_on_issue_alone(tmp_path, census):
    c, cu = census
    out = tmp_path / "r20261017-214111-18e4" / "out"
    (out / "sass").mkdir(parents=True)
    body = (["IMAD R0, R1, R2, RZ", "MOV R4, RZ"] + ["IMMA.16832.S8.S8 R8, R12, R16, R8"] * 2
            + ["FMUL R20, R20, R21"] * 6
            + ["@!P0 MOV R5, R4", "IADD3 R6, R6, 0x1, RZ", "ISETP.NE.AND P0, PT, R6, R7, PT", "BRA 0x0020", "EXIT"])
    sass = "\n\tcode for sm_120\n\t\tFunction : jm_t\n" + "".join(
        f"        /*{16 * i:04x}*/                   {ins} ;      /* 0x000fe20000000f00 */\n"
        for i, ins in enumerate(body))
    (out / "sass" / "fb_000.sass").write_text(sass + "\t\tFunction : jm_u\n")
    loop = {"head": "0x0020", "tail": "0x00d0", "counts": {"IMMA.16832.S8.S8": 2, "FMUL": 6}}
    (out / "gate-build.json").write_text(json.dumps({"verdicts": [
        {"probe": p, "passed": True, "verdict": {"kernel": p, "loops": [loop]}} for p in ("jm_t", "jm_u")]}))
    one = {"probe": "jm_t", "rates": {"0:IMMA.16832.S8.S8": 0.2, "0:FMUL": 0.6}}
    two = {"probe": "jm_u", "rates": {"0:IMMA.16832.S8.S8": 0.2, "1:FMUL": 0.6}}
    p = out / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (one, two)))
    want = ["IMMA.16832.S8.S8"] * 2 + ["FMUL"] * 6 + ["MOV", "IADD3", "ISETP.NE.AND", "BRA"]
    assert loads.gate_loops(out)["jm_t"] == want
    full, other = loads.load_rows([p], full=True)
    assert full["x"] == pytest.approx({"IMMA16832": 0.05, "FFMA": 0.15, "NOROW": 0.025, "ALU": 0.025, "ISETP": 0.025})
    assert full["x_issue"] == pytest.approx(0.025) and "ISETP" not in c
    assert "x" not in other
    counted = loads.program(one, c, cu)["load"]["issue"]
    assert loads.program(full, c, cu)["load"]["issue"] == pytest.approx(counted + 0.1) and counted == pytest.approx(0.2)


def test_grid_is_seeded_unique_and_pinned():
    a, b = gen.grid(3, 40), gen.grid(3, 40)
    assert a == b and a != gen.grid(4, 40)
    names = [g["name"] for g in a]
    assert len(names) == len(set(names))
    src = gen.emit_cu(a)
    assert all(src.count(f"({n},") == 1 and src.count(f'"{n}"') == 1 for n in names)
    g = {"kind": "mix", "form": "i16832", "l1": "lop3", "l2": "ffma", "q": 7, "p": 0, "tpb": 512}
    assert gen.expect_of(g) == ("IMMA.16832.S8.S8=8,LOP3.LUT=4,FFMA=3", "", 0)
    s = {"kind": "spec", "form": "h1684t", "l1": "sel", "l2": "", "wm": 8, "tpb": 768}
    assert gen.expect_of(s) == ("HMMA.1684.F32.TF32=8;SEL=16", "8;16", 2)


def test_targets_sit_at_the_section_vertices():
    """KSEL0/KSEL1 meet at 64 SEL a 8 MMAs: the targeted part times that ratio and the ones around it."""
    ratios = {round(8 * x / m) for m, x in gen.vertices("SEL", epoch=EPOCH) if m > 0 and x > 0}
    assert 64 in ratios
    qs = {g["q"] for g in gen.targeted() if g["kind"] == "mix" and g["l1"] == "sel" and g["tpb"] == 512}
    assert {38, 51, 64, 80, 102} <= qs


def _cuda_tools() -> bool:
    probe = _load("opcode_probe_probe", POUW / "opcode_probe" / "probe.py")
    try:
        probe.nvcc()
        probe.G.H.find_tools()
    except (SystemExit, probe.G.H.SassGateFailed, OSError):
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_tools(), reason="needs nvcc, cuobjdump and nvdisasm (CUDA 13)")
def test_grid_slice_builds_and_gates(tmp_path):
    """What the tests above can't: today's nvcc compiles each kind of program to SASS the gate passes, exact counts."""
    probe = _load("opcode_probe_probe", POUW / "opcode_probe" / "probe.py")
    picks = {"mx_i16832_ffma_q32_p0_w16", "mx_i16832_sel_q64_p0_w16", "mx_i16832_imadw_q16_p0_w16",
             "mx_i16832_iadd3_ffma_q64_p0_w16", "sp_i16832_ffma_m8_w16", "al_iadd64_w16", "mm_h1684t_w8",
             "mx_h16816f16_lop3_q64_p0_w16"}
    mem = {"ld_lds32_lds128_r2_line_q16_w32", "ld_lds64_quad_q16_w32", "ld_tld4_line_q8_w16", "ld_ldgnc32_bcast_q8_w8",
           "lm_i16832_lds128_bcast_q8_w16", "lm_h16816f16_tld1_line_q16_w16"}
    l2 = {"l2_g256cg_scat_q16_w32", "l2_g128na_line_q8_w16", "ld_ldg256_line_q8_w8"}
    gs = [g for g in gen.grid(0, 0) if g["name"] in picks] + [g for g in gen.grid(0, 0, family="mem") if g["name"] in mem]
    gs += [g for g in gen.grid(0, 0, family="l2") if g["name"] in l2]
    assert {g["name"] for g in gs} == picks | mem | l2
    src = tmp_path / "slice.cu"
    src.write_text(gen.emit_cu(gs))
    _, verdicts = probe.gate_all(probe.build(tmp_path / "b", [src]))
    assert [v["probe"] for v in verdicts if not v["passed"]] == []
