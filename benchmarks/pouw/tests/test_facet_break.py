"""facet_break: SASS opcodes to census classes, a program's load on every facet (IMMA.16832 through the generic MMA lines,
every other form only through its own), the R1 and KFFMA0 violation TENSOR's interleaved points make at VF3n-16, the hull
refit, and the generator's grid (seeded, unique names, exact SASS counts). `test_grid_slice_builds_and_gates` builds a
slice of the grid with nvcc and gates it; it skips without nvcc and nvdisasm.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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

#: TENSOR's tensor_kinds records (r20261017-041754-7630): IMMA.16832 alone, beside IADD3+FFMA interleaved at 4 lane ops a
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
            "POPC": "XU", "SHFL.BFLY": "SHFL", "MOV": "NOROW", "NOP": None}
    assert {op: loads.class_of(op) for op in want} == want


def test_only_imma16832_takes_the_generic_mma_lines(census):
    c, cu = census
    imma = loads.facet_loads({"IMMA16832": 1.0}, c, cu)
    hmma = loads.facet_loads({"HMMA16816F16": 1.0}, c, cu)
    assert imma["M"] == pytest.approx(16) and "R1" in imma and "KFFMA0" in imma
    assert imma["JI3n_0"] == pytest.approx(c["IMMA16832"]["JI3n_0"] / cu)
    assert set(hmma) == {"M", "issue", "JI3n_0", "JI3n_1", "JI3n_2"}
    assert "R1" in loads.facet_loads({"HMMA16816F16": 1.0}, c, cu, generic=True)


def test_interleaved_imma_ffma_beats_r1_and_kffma0(tmp_path, census, capsys):
    """V1 and V2 of note:20261017T0524Z-draft-eps1-facet-break: the axes hold, the interleaved point beats both rows."""
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


def test_old_first_tensor_kinds_build_lane_is_iadd3_alone(tmp_path):
    rec = MIXU_R32 | {"mode": "spec", "op_warps": 8}
    path = _records(tmp_path, rec, {"mode": "opalone", "form": "iadd3_32", "cyc_med": 1, "mmas_per_sm": 0,
                                    "ops_per_sm": 1, "mma_warps": 0, "op_warps": 32})
    x = loads.load_rows([path])[0]["x"]
    assert set(x) == {"IMMA16832", "ALU"}


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
    gs = [g for g in gen.grid(0, 0) if g["name"] in picks]
    assert {g["name"] for g in gs} == picks
    src = tmp_path / "slice.cu"
    src.write_text(gen.emit_cu(gs))
    _, verdicts = probe.gate_all(probe.build(tmp_path / "b", [src]))
    assert [v["probe"] for v in verdicts if not v["passed"]] == []
