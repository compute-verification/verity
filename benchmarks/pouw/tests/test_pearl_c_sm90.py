"""Pearl-C's sm_90a tree off the card (benchmarks/pouw/pearl_c_sm90): the GEMMs within the H100's 227 KB with their tiles the
runner's, the kernel's constants the scheme's, the SASS gate refusing a lowered or spilling build, the check fixture against
the scheme on the H100 record, and run.py's identity refusal and dry run on a CPU stand-in for libcuda (#449's, with
pearl_c_sm120/fake_tma.c's tensor maps; an H200's name, SM count, compute capability, UUID and shared-memory limit), every
launch's pointers and sizes checked against each kernel's footprint as pearl_c_sm90.cu states it, and each GEMM's operands
against the tensor maps encoded for them at its tile's boxes.  With VERITY_POUW_NVCC naming a CUDA >= 12.8 toolkit, the real
sm_90a build passes the gate."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw.audit import weights_root
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening, coin_reason

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm90"
SM120_DIR = HERE.parent / "pearl_c_sm120"


def load(name: str, path: Path):
    """The module at `path` under its own name: other trees in the suite have a fixture.py and a run.py too."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = load("pearl_c_sm90_fixture", HERE / "fixture.py")
S = load("pearl_c_sm90_sass_gate", HERE / "sass_gate.py")
RUN = load("pearl_c_sm90_run", HERE / "run.py")

SOURCE = (HERE / "pearl_c_sm90.cu").read_text()
HASH_SOURCE = (HERE.parent / "pearl_c" / "hash_h3.cuh").read_text() + (SM120_DIR / "hash_fast.cuh").read_text()
#: hash_decode.cuh's variants the cubin builds (pearl_c_sm90.cu's PD_VARIANT_LIST), each its four kernels
PD_BUILT = [int(v) for v in re.findall(r"X\((\d+)\)", re.search(r"#define PD_VARIANT_LIST\(X\)(.*)", SOURCE).group(1))]
#: hash_prefill.cuh's variants the cubin builds (pearl_c_sm90.cu's PL_INSTANCE lines), each its four kernels
PL_BUILT = [int(v) for v in re.findall(r"^PL_INSTANCE\((\d+)\)", SOURCE, re.M)]
PL_KINDS = ("leaves", "levels", "seed_a", "seeds")
H100_MAX_DYN = 232448                    # sm_90's opt-in dynamic shared memory per block (227 KB)
UUID = "GPU-0123abcd-4567-89ef-0123-456789abcdef"
H200 = "NVIDIA H200"


def const(name: str) -> int:
    return int(re.search(rf"\b{name} = (\d+)", SOURCE).group(1))


def gemms(macro: str = "GEMM") -> list[tuple[str, str, int, int, int]]:
    """Every GEMM kernel (name, mode, tile rows, tile columns, G); with ``GEMM_G`` the grouped calls' (grouped.py)."""
    return [(n, m, int(tm), int(bn), int(g))
            for n, m, tm, bn, g in re.findall(rf"^{macro}\((\w+), (M_\w+), (\d+), (\d+), (\d+)\)", SOURCE, re.M)]


def tile_smem(tm: int, bn: int) -> int:
    """Tile<TM, BN>::SMEM: the stages from a 1,024-aligned base, STAGES = min(CAP, SMEM_BUDGET / STAGE) (CAP: the 64-row
    tiles' MAX_STAGES_64, else MAX_STAGES), and a barrier pair of 16 bytes a stage."""
    stage = (tm + bn) * const("BK")
    cap = const("MAX_STAGES_64") if tm == 64 else const("MAX_STAGES")
    stages = min(cap, 200 * 1024 // stage)
    return 1024 + stages * stage + 16 * stages


def test_gemms_fit_h100_shared_memory_and_are_the_runners():
    """Every GEMM fits the H100's 227 KB at its tile, the runner gives each its tile and block (a consumer warpgroup per 64
    rows, then the producer), the gate knows each one at its N, and the peel's tail stage is exactly one stage."""
    assert "SMEM = 1024 + STAGES * STAGE + 16 * STAGES" in SOURCE and "STAGE = (TM + BN) * BK" in SOURCE
    assert "CAP = NWG == 1 ? MAX_STAGES_64 : MAX_STAGES" in SOURCE
    assert "SMEM_BUDGET = 200 * 1024" in SOURCE and const("PEEL_ROW") == const("BK") == RUN.PEEL_ROW == 128
    names = {n for n, *_ in gemms()}
    assert names == {f"{g}{t}" for g in ("gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_pearlc_s")
                     for t in ("", "_n64", "_d")} | {"gemm_words", "gemm_words_d", "gemm_words_win_d"}
    for name, mode, tm, bn, g in gemms():
        assert 48 * 1024 < tile_smem(tm, bn) <= H100_MAX_DYN, name
        assert RUN.gemm_tile(name) == (tm, bn) and RUN.gemm_threads(name) == 128 * (tm // 64 + 1), name
        assert S.WANT[name][0] == bn and S.WANT[name][2] == (4 if mode in ("M_PEARLC", "M_PEARLC_S") else 0), name
        assert g in (0, 4) and (g == 0) == name.startswith("gemm_fast"), name
        if not name.startswith("gemm_words"):
            base = name.removesuffix(RUN.tile_suffix(tm, bn))
            assert base in ("gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_pearlc_s"), name
    assert {tile_smem(*t) for t in ((128, 128), (128, 64), (64, 64), (128, 32), (64, 32))} == {197728, 197760, 197824, 164992, 148672}
    grouped = {n: (m, tm, bn, g) for n, m, tm, bn, g in gemms("GEMM_G")}
    assert set(grouped) == {"gemm_pearlc_g_d", "gemm_words_g_d"}
    for name, spec in grouped.items():                   # each is its ungrouped kernel's mode, tile and gate
        alone = name.replace("_g_d", "_d")
        assert spec == next((m, tm, bn, g) for n, m, tm, bn, g in gemms() if n == alone), name
        assert RUN.gemm_tile(name) == RUN.gemm_tile(alone) and S.WANT[name] == S.WANT[alone], name
    assert set(S.WANT) - {"form_s5", "form_s5_scalar", "form_s5_g", "form_s5_ep", "form_s5_epw"} == names | set(grouped)
    assert S.WANT["form_s5_g"] == S.WANT["form_s5_ep"] == S.WANT["form_s5_epw"] == S.WANT["form_s5"]
    # form_s5_epw's window is gemm_words_win_d's tile: its four k32 atoms at N = 32
    assert S.WINDOW_ATOMS == {"form_s5_epw": S.WANT["gemm_words_win_d"][1]} and S.WANT["gemm_words_win_d"][0] == 32
    assert set(RUN.FORM_ALTS) == {"form_s5_scalar"}
    assert form_kernels() == ["form_s5", "form_s5_scalar", "form_s5_g", "form_s5_ep", "form_s5_epw"]


def test_a_persistent_gemm_leaves_a_launch_beside_it_the_sms_it_holds(monkeypatch):
    """`Pipeline.beside` (a whole capture's deferred A tree: pd_tree's CTA holds its SM under PD_WHOLE) comes off a
    persistent GEMM's grid, so no CTA waits for that launch to end; a grid the caller gives, and a GEMM of fewer tiles than
    SMs, keep theirs.  Under GRID_BALANCE the grid is the fewest CTAs that keep its rounds of tiles: 448 tiles take four
    rounds on 112 CTAs as on 131."""
    grids = []
    dev = SimpleNamespace(sms=132, launch=lambda name, grid, *a, **k: grids.append(grid))
    pipe = SimpleNamespace(GEMM_SMEM={"gemm_pearlc_d": 1}, boxes={("at", "bt"): RUN.gemm_tile("gemm_pearlc_d")}, dev=dev, k=4096,
                           b=dict.fromkeys(("at", "bt", "c", "tma"), 0), m_real=16)
    cases = ((0, 28672, None), (1, 28672, None), (1, 28672, 64), (1, 4096, None), (0, 8448, None), (1, 8448, None))
    for balance in (False, True):
        monkeypatch.setattr(RUN, "GRID_BALANCE", balance)
        for beside, n, grid in cases:
            pipe.beside = beside
            RUN.Pipeline.gemm(pipe, "gemm_pearlc_d", "at", "bt", 64, n, grid=grid)
    assert grids == [132, 131, 64, 64, 132, 131] + [112, 112, 64, 64, 132, 66] and RUN.Pipeline.beside == 0


def test_a_64_by_64_gemm_stores_its_real_rows_alone(monkeypatch):
    """A 64 x 64 GEMM's i4 (live) is m_real where filler rows follow it (LIVE), so the kernel stores no filler row's C~, U
    or y; with no filler rows, at another tile, under PEARLC_LIVE=0 (LIVE false) it is 0, all M; a given live is taken."""
    lives = []
    dev = SimpleNamespace(sms=132, launch=lambda name, grid, *a, i, **k: lives.append(i[4]))
    pipe = SimpleNamespace(GEMM_SMEM={"gemm_pearlc_d": 1, "gemm_words_d": 1}, dev=dev, k=4096, beside=0, m_real=16,
                           boxes={("at", "bt"): RUN.gemm_tile("gemm_pearlc_d"), ("at", "fbT32"): RUN.gemm_tile("gemm_words_d")},
                           b=dict.fromkeys(("at", "bt", "fbT32", "c", "af", "tma"), 0))
    RUN.Pipeline.gemm(pipe, "gemm_pearlc_d", "at", "bt", 64, 4096)
    RUN.Pipeline.gemm(pipe, "gemm_words_d", "at", "fbT32", 64, 32, out=("af",))
    RUN.Pipeline.gemm(pipe, "gemm_pearlc_d", "at", "bt", 64, 4096, live=40)
    pipe.m_real = 64
    RUN.Pipeline.gemm(pipe, "gemm_pearlc_d", "at", "bt", 64, 4096)
    pipe.m_real = 16
    monkeypatch.setattr(RUN, "LIVE", False)
    RUN.Pipeline.gemm(pipe, "gemm_pearlc_d", "at", "bt", 64, 4096)
    assert lives == [16, 0, 40, 0, 0]


def test_operands_are_in_natural_order():
    """No K32 order: the lines' transposes are the codes' plain transpose, and form_s5 writes its codes at their own
    columns; the fixture's layout says so."""
    assert "t[(size_t)j * count + line] = c[j]" in SOURCE and "k32(" not in SOURCE
    assert "out + (size_t)(r0 + r) * k + t0 + 16 * c" in SOURCE
    assert not hasattr(F, "k32")
    x = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert F.rowmajor(x) == x.tobytes()


def test_descriptors_are_the_layouts_staged():
    """form_s5's staging (`core`) is the no-swizzle K-major layout desc_plain describes (core matrices of 8 rows x 16 bytes,
    128 bytes apart along k, 256 between 8-row groups), and every byte of a 64 x 32 operand lands once; the GEMMs' SW128
    descriptor strides 1,024 bytes between 8-row groups."""
    assert "(r >> 3) * 256 + (j >> 4) * 128 + (r & 7) * 16 + (j & 15)" in SOURCE
    assert "((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32)" in SOURCE
    assert "(1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62)" in SOURCE
    core = lambda r, j: (r >> 3) * 256 + (j >> 4) * 128 + (r & 7) * 16 + (j & 15)  # noqa: E731
    for rows in (64, 128):
        assert sorted(core(r, j) for r in range(rows) for j in range(32)) == list(range(rows * 32))


def test_kernel_constants_match_the_reference():
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes import pearl_kw as P
    from verity.core.protocols.pouw.schemes.pearl_c_device import H100

    assert int(re.search(r"S5_LINE = (0x[0-9A-Fa-f]+);", SOURCE).group(1), 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    run = (HERE / "run.py").read_text()
    n256, n16 = re.search(r"NORM_256, NORM_16 = (0x[0-9A-F]+), (0x[0-9A-F]+)", run).groups()
    assert int(n256, 16) == P.f32_to_bf16(P.NOISE_NORM * 32) and int(n16, 16) == P.f32_to_bf16(C.LINE_NORM_V1 * 32)
    block = F.unit_block(b"s" * 32, b"b" * 64, 7)
    assert len(block) == RUN.TIMING_KEYS["unit"] == 104 and block[96:] == (7).to_bytes(8, "little")
    assert "unit[24] | (uint64_t)unit[25] << 32" in HASH_SOURCE and 'keys["unit"] + 96' in run
    assert RUN.VARIANT_G == {H100.name: H100.g} and F.VARIANTS == (H100,)
    assert {g for *_, g in gemms() if g} == {H100.g} and H100.window == const("BK")
    assert F.H100_UNPROMOTED.g is None and F.H100_UNPROMOTED.atom == H100.atom
    assert C.PearlC(forming="v1", device=H100, hashing="h3").name == "pearl-c-h100-v1-h3"


def test_a_split_pd_pipeline_reads_a_calls_bf16_rows_only_from_a_cubin_that_takes_them(monkeypatch):
    """`Pipeline.reads_bf16`: a split pd pipeline, from a cubin with BF16_ROWS_MARK, under BF16_ROWS.  Its ``rows_a`` then
    gives pd_commit the rows set on it (`rows_bf16`) at p15 and their count at i5, and mode 0 and unset rows launch as
    before; rows set on a pipeline that doesn't read them are refused.  `widen_a` with `pdl` False launches plainly."""
    launched = []

    class D:
        def __init__(self, marked):
            self.marked, self.pd_smem = marked, {851: 0}

        def has(self, name):
            return self.marked and name == RUN.BF16_ROWS_MARK

        def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None, **kw):
            launched.append((name, tuple(p), tuple(i), kw))

    pipe = RUN.Pipeline.__new__(RUN.Pipeline)
    pipe.b = dict.fromkeys(("xa", "rowdig_a", "tree_a", "seeds_a", "lkeys_a", "pd_count", "seed_a", "line_a", "ea_codes"), 1)
    pipe.keys = dict.fromkeys(("unit", "dom_a", "salt_key", "hm"), 2)
    pipe.pd, pipe.split_a, pipe.stamps, pipe.k, pipe.m, pipe.m_real = 851, True, 0, 4096, 64, 16
    for marked in (True, False):
        pipe.dev, pipe.rows_bf16 = D(marked), None
        assert pipe.reads_bf16() is marked
        launched.clear()
        pipe.pd_commit(1)
        assert launched[0][0] == "pd_commit_851" and len(launched[0][1]) == 14 and launched[0][2] == (16, 4096, 1, 64)
        pipe.rows_bf16 = (777, 8)
        if not marked:
            with pytest.raises(RuntimeError, match="reads_bf16"):
                pipe.pd_commit(1)
            continue
        launched.clear()
        pipe.pd_commit(1)
        pipe.pd_commit(0)
        assert launched[0][1][14:] == (0, 777) and launched[0][2] == (16, 4096, 1, 64, 0, 8)
        assert len(launched[1][1]) == 14 and launched[1][2] == (16, 4096, 0, 64)
    pipe.dev = D(True)
    pipe.split_a = False
    assert not pipe.reads_bf16()
    pipe.split_a = True
    monkeypatch.setattr(RUN, "BF16_ROWS", False)
    assert not pipe.reads_bf16()
    assert f"void {RUN.BF16_ROWS_MARK}() {{}}" in SOURCE

    calls = []
    dev = RUN.Dev.__new__(RUN.Dev)
    dev.cu = SimpleNamespace(cuLaunchKernelEx=lambda *a: calls.append("programmatic") or 0,
                             cuLaunchKernel=lambda *a: calls.append("plain") or 0)
    dev.fns, dev.stream, dev.pdl_names = {"widen_a": RUN.vp()}, None, RUN.PDL_ENTRY_KERNELS
    RUN.widen_a(dev, 1, 2, 8, 16, 4096)
    RUN.widen_a(dev, 1, 2, 8, 16, 4096, pdl=False)
    assert calls == ["programmatic", "plain"]


def test_a_programmatic_launch_waits_at_its_entry():
    """Each of run.py's PDL_KERNELS waits for the launch before it (pdl_entry) before it reads anything into the CTA: the GEMMs
    once their barriers are set up in shared memory, having sent only B's first boxes toward L2 (tma_prefetch_l2, which can't
    read a stale line), pa_w, widen_a and pd_commit first, and the forming once it has read the weight's F lines (p3) and F_B^T
    bytes (p7), which only the weight's job writes (`lines_fa`, `lines_fb`); and pd_commit then lets the forming launch behind it.  The Pearl-C GEMMs
    (LATE) read only P_A of what the launch before them (pa_w, pa) writes, so only their producer waits, before its first
    P_A box; pa_w waits at its entry, so what they read before that was written by launches that had completed."""
    def body(start):
        at = SOURCE.index(start)
        return SOURCE[at:SOURCE.index("\n}\n", at)]

    gemm, form, pa_w = (body(x) for x in ("DEV void gemm_body", "DEV void form_body", 'void pa_w('))
    head = gemm[:gemm.index("pdl_entry();")]
    assert "__syncthreads();" in head and "if (wg == NWG)" not in head and "b_row(" not in head
    assert set(re.findall(r"\btma_\w+", head)) == {"tma_prefetch_l2"}
    assert "constexpr bool LATE = PEEL && !GR;" in head and head.rstrip().endswith("if constexpr (LATE)\n    pdl_launch();\n  else")
    producer = gemm[gemm.index("if (wg == NWG)"):gemm.index('asm volatile("exit;");')]
    assert producer.count("pdl_wait();") == 1 and gemm.count("pdl_wait();") == 1
    assert producer.index("if constexpr (LATE) pdl_wait();") < producer.index("maps->pa, full") and \
        "maps->pa, full" not in producer[:producer.index("if constexpr (LATE) pdl_wait();")]
    assert re.search(r"pdl_wait\(\) \{ asm volatile\(\"griddepcontrol\.wait;\"", SOURCE)
    assert re.search(r"pdl_launch\(\) \{ asm volatile\(\"griddepcontrol\.launch_dependents;\"", SOURCE)
    assert "cp.async.bulk.prefetch.tensor.2d.L2.global" in body("DEV void tma_prefetch_l2")
    for b in (pa_w, body('void widen_a(')):
        head = b[:b.index("pdl_entry();")]
        assert "reinterpret_cast" not in head and "return" not in head, head
    head = form[:form.index("pdl_entry();")]
    assert set(re.findall(r"a\.p\[(\d+)\]", head)) == {"3", "5", "7"} and "return" not in head
    assert "if constexpr (!GR) stage_f();\n  pdl_entry();\n  if constexpr (GR) stage_f();" in form
    run = (HERE / "run.py").read_text()
    assert 'form_side("a", m, "xa", "ea_codes", "fa_codes", "at"' in run and 'form_side("b", n, "xb", "eb_codes", "fb_codes"' in run
    assert 'b["ea_codes"], b["fa_codes"], b["at"], b["beta_a"],\n' in run and 'b["epa_codes"], b["fbT32"], b["af_win"]' in run
    job = run[run.index("        job = ["):run.index("        mr, tiles = ")]
    assert '"fa_codes", b["faT32"])' in job and '"fb_codes", b["fbT32"], b["fbT128"])' in job
    assert {k for k in RUN.PDL_KERNELS if k.startswith("form")} <= set(form_kernels())
    assert RUN.PDL_KERNELS - set(form_kernels()) == {"gemm_words_win_d", "pa_w", "gemm_pearlc_d"}
    assert RUN.PDL_ENTRY_KERNELS == {"widen_a"} | {f"pd_commit_{v}" for v in (*RUN.PD_SERVED, *RUN.PD_STAMPED)}
    assert f'void {RUN.PDL_ENTRY_MARK}() {{}}' in SOURCE
    assert SOURCE.index("#define PD_PDL") < SOURCE.index('#include "../pearl_c_sm120/hash_decode.cuh"')
    pd = (SM120_DIR / "hash_decode.cuh").read_text()
    entry = pd[pd.index("DEV void pd_commit_t("):]
    entry = entry[:entry.index("#endif")]
    assert "#ifdef PD_PDL" in entry and 'asm volatile("griddepcontrol.wait;\\n\\tgriddepcontrol.launch_dependents;"' in entry
    assert "a.p[" not in entry[:entry.index("#ifdef PD_PDL")]


SASS_OK = "\n".join([
    "Function : gemm_x", "  /*0010*/ QGMMA.64x128x32.F32.E4M3.E4M3 R24, gdesc[UR4], RZ, !UPT, gsb0 ;",
    "  /*0020*/ QGMMA.64x128x32.F32.E4M3.E4M3 R24, gdesc[UR8], R24, gsb0 ;",
    "  /*0030*/ HGMMA.64x128x16.F32.BF16 R24, gdesc[UR4], R24, gsb0 ;", "  /*0040*/ UTMALDG.2D [UR8], [UR4] ;"])
PTXAS_OK = ("ptxas info    : Compiling entry function 'gemm_x' for 'sm_90a'\n"
            "ptxas info    : Function properties for gemm_x\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 168 registers, used 1 barriers\n")


def test_sass_gate_refuses_what_it_should(monkeypatch):
    monkeypatch.setattr(S, "WANT", {"gemm_x": (128, 2, 1)})
    funcs, problems = S.gate(PTXAS_OK, SASS_OK)
    assert not problems and funcs["gemm_x"]["QGMMA"] == 2 and funcs["gemm_x"]["HGMMA"] == 1 and funcs["gemm_x"]["regs"] == 168
    bad = {
        "a lowered FP8 atom": SASS_OK.replace("QGMMA.64x128x32.F32.E4M3.E4M3", "HGMMA.64x128x16.F32.F16"),
        "a missing peel step": SASS_OK.replace("HGMMA.64x128x16.F32.BF16", "FFMA"),
        "local memory": SASS_OK + "\n  /*0050*/ LDL R1, [R1] ;",
        "a different atom": SASS_OK.replace("QGMMA.64x128x32.F32.E4M3.E4M3", "QGMMA.64x128x32.F32.E5M2.E5M2"),
        "another tile's N": SASS_OK.replace("QGMMA.64x128x32", "QGMMA.64x64x32"),
        "a warp MMA": SASS_OK + "\n  /*0050*/ HMMA.16816.F32.BF16 R4, R8, R12, R4 ;",
    }
    for why, sass in bad.items():
        assert S.gate(PTXAS_OK, sass)[1], why
    assert S.gate(PTXAS_OK.replace("0 bytes spill stores", "8 bytes spill stores"), SASS_OK)[1]
    assert S.gate(PTXAS_OK + "ptxas warning : Registers are spilled\n", SASS_OK)[1]
    assert S.gate(PTXAS_OK + "ptxas info    : (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized\n",
                  SASS_OK)[1]
    assert S.gate(PTXAS_OK, SASS_OK.replace("gemm_x", "gemm_y"))[1]
    # a kernel with A'F_B's window has exactly its atoms at N = 32 beside its own; no other kernel has any
    monkeypatch.setattr(S, "WINDOW_ATOMS", {"gemm_x": 2})
    window = "\n  /*0050*/ QGMMA.64x32x32.F32.E4M3.E4M3 R8, gdesc[UR4], RZ, !UPT, gsb0 ;"
    funcs, problems = S.gate(PTXAS_OK, SASS_OK + window * 2)
    assert not problems and funcs["gemm_x"]["QGMMA_window"] == 2 and funcs["gemm_x"]["QGMMA"] == 2
    assert S.gate(PTXAS_OK, SASS_OK + window)[1] and S.gate(PTXAS_OK, SASS_OK + window * 3)[1]
    monkeypatch.setattr(S, "WINDOW_ATOMS", {})
    assert S.gate(PTXAS_OK, SASS_OK + window * 2)[1]


def test_sass_gate_allows_local_memory_only_in_the_reference_row_digest(monkeypatch):
    assert {k: v[0] for k, v in S.LOCAL_OK.items()} == {"s512_rows": 16}
    monkeypatch.setattr(S, "WANT", {"gemm_x": (128, 2, 1)})
    rows = ("ptxas info    : Compiling entry function 's512_rows' for 'sm_90a'\n"
            "ptxas info    : Function properties for s512_rows\n    8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n"
            "ptxas info    : Used 255 registers\n")
    sass = SASS_OK + "\nFunction : s512_rows\n  /*0010*/ STL [R1], R2 ;\n  /*0020*/ LDL R2, [R1] ;"
    assert not S.gate(PTXAS_OK + rows, sass)[1]
    assert S.gate(PTXAS_OK + rows.replace("8 bytes stack", "24 bytes stack"), sass)[1]
    assert S.gate(PTXAS_OK.replace("0 bytes spill stores", "4 bytes spill stores") + rows, sass)[1]


@pytest.mark.skipif(not os.environ.get("VERITY_POUW_NVCC"), reason="VERITY_POUW_NVCC names a CUDA >= 12.8 toolkit (about 30 s)")
def test_the_sm90a_build_passes_the_gate(tmp_path):
    cuda = Path(os.environ["VERITY_POUW_NVCC"])
    cubin, ptxas, sass = tmp_path / "k.cubin", tmp_path / "ptxas.txt", tmp_path / "k.sass"
    with open(ptxas, "w") as err:
        subprocess.run([str(cuda / "bin" / "nvcc"), "-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "-fmad=false",
                        "-Xptxas", "-v", "-cubin", "-o", str(cubin), str(HERE / "pearl_c_sm90.cu")], check=True, stderr=err)
    env = dict(os.environ, PATH=f"{cuda / 'bin'}:{os.environ['PATH']}")
    sass.write_text(subprocess.run([str(cuda / "bin" / "cuobjdump"), "-sass", str(cubin)], check=True, capture_output=True, text=True,
                                   env=env).stdout)
    funcs, problems = S.gate(ptxas.read_text(), sass.read_text())
    assert not problems, problems
    for name in ("gemm_pearlc", "gemm_pearlc_n64", "gemm_pearlc_d"):
        assert funcs[name]["HGMMA"] == 4 and funcs[name]["QGMMA"] % 4 == 0
    assert H3_LAUNCHED | {"s512_pads", "lines"} <= set(funcs)


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory):
    out = tmp_path_factory.mktemp("pearl-c-sm90-fixture")
    F.build(out, 128, 128, 1024)
    return out


@pytest.mark.slow
def test_fixture_agrees_with_the_scheme(small_fixture):
    man = json.loads((small_fixture / "check.json").read_text())
    assert man["shape"] == [128, 128, 1024] and man["layout"] == "rowmajor"
    assert set(man["variants"]) == {"h100"}
    spec = man["variants"]["h100"]
    assert spec["crosscheck"]["agree"] and spec["scheme"] == "pearl-c-h100-v1-h3" and spec["g"] == 4
    c = np.frombuffer((small_fixture / "h100" / "c.bin").read_bytes(), "<u4")
    u = np.frombuffer((small_fixture / "h100" / "u.bin").read_bytes(), "<u4")
    assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99
    y_plain = (small_fixture / man["common"]["y_plain"]["file"]).read_bytes()
    y_fast = (small_fixture / man["common"]["y_fast"]["file"]).read_bytes()
    assert len(y_plain) == len(y_fast) == 2 * 128 * 128
    commitment, opening = CoinCommitment.from_json(man["coins"]), CoinOpening.from_json(man["salt_opening"])
    assert commitment == F.test_challenger().commitment and "beacon" not in man
    assert coin_reason(commitment, opening, "salt", weights_root({F.WEIGHT: bytes.fromhex(man["seeds"]["root_b"])})) is None


@pytest.mark.slow
def test_widening_fixture_is_the_silicon_model(small_fixture):
    """The fixture's widened B: BF16 words as their FP32 words, and each E4M3 code times its block's scale as
    `silicon.fp32.mul` of `silicon.cast.e4m3_to_f32_word` (every element, subnormal products among them), its block the
    row's and column's 128 x 128 one; the decode is the silicon model's on every finite code."""
    from verity.ml.circuits.silicon import fp32
    from verity.ml.circuits.silicon.cast import e4m3_to_f32_word

    codes = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint8)
    assert F.e4m3_values(codes).view(np.uint32).tolist() == [e4m3_to_f32_word(int(c)) for c in codes]
    spec = json.loads((small_fixture / "check.json").read_text())["widen"]
    rows, k = spec["shape"]
    read = lambda key, dt: np.frombuffer((small_fixture / spec["files"][key]["file"]).read_bytes(), dt)  # noqa: E731
    words, xb16 = read("src_bf16", "<u2").reshape(rows, k), read("xb_bf16", "<u4").reshape(rows, k)
    assert (xb16 == words.astype(np.uint32) << 16).all()
    src, scales = read("src_fp8_block", np.uint8).reshape(rows, k), read("scales", "<u4").reshape(-(-rows // 128), -(-k // 128))
    xb8 = read("xb_fp8_block", "<u4").reshape(rows, k)
    want = [[fp32.mul(e4m3_to_f32_word(int(src[r, c])), int(scales[r // 128, c // 128])) for c in range(k)] for r in range(rows)]
    assert xb8.tolist() == want
    exp = (xb8 >> 23) & 0xFF
    assert ((exp == 0) & (xb8 & 0x7FFFFF != 0)).sum() > 100 and rows % 128 and (scales >> 23 & 0xFF == 0).any()


# ---- the dry run ---------------------------------------------------------------------------------------------------

KERNEL = r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(?!__launch_bounds__)(\w+)\('
#: the -h3 kernels (hash_h3.cuh, hash_fast.cuh) the pipeline launches at every shape; the row digests and tile leaves by
#: hash_fast.cuh's kernels (run.py's default, PEARLC_SHA=fast).  A tree whose count is not a power of two also launches
#: s512_pads.
H3_LAUNCHED = {"s512_heads", "s512_row_segs", "s512_tile_segs", "s512_folds", "hm_leaves", "s512_frame_leaves", "s512_nodes",
               "s512_tree_small", "s512_seeds"}
#: hash_decode.cuh's launches at the check (run.py's check_pd): its tables, each served variant's tile leaves, and its
#: commitment (whole, and split into pd_commit's mode 1 and pd_tree) and its stamp twin's on the variant without PD_GROUP
#: (run.py's ONE_COMMIT)
PD_LAUNCHED = {"pd_heads", *(f"pd_{kind}_{v & ~RUN.PD_GROUP}" for v in RUN.PD_SERVED for kind in ("commit", "tree")),
               *(f"pd_tiles_{v}" for v in RUN.PD_SERVED), *(f"pd_commit_{v & ~RUN.PD_GROUP}" for v in RUN.PD_STAMPED)}
#: hash_prefill.cuh's launches (run.py's chain "pl", the default past pd)
PL_LAUNCHED = {f"pl_{kind}_{RUN.PL_VARIANT}" for kind in PL_KINDS}


def form_kernels() -> list[str]:
    return [n for n in re.findall(KERNEL, SOURCE) if n.startswith("form_")]


def grouped_kernels() -> list[str]:
    """The grouped calls' kernels (grouped.py), which run.py never launches: form_s5_g, the GEMM_G GEMMs and their probes,
    and PD_GROUPED's pd_commit_g and pd_tiles_g, PD_GROUPED_TILES' pd_tiles_g and pd_tiles_gw and PD_GROUPED_ROWS'
    pd_commit_gm."""
    names = [n for n, *_ in gemms("GEMM_G")]
    pd = [f"pd_{kind}_g_{v}" for v in re.findall(r"^PD_GROUPED\((\d+)\)", SOURCE, re.M) for kind in ("commit", "tiles")]
    pd += [f"{name}_{v}" for v, name in re.findall(r"^PD_GROUPED_TILES\((\d+), \w+, (\w+)\)", SOURCE, re.MULTILINE)]
    pd += [f"pd_commit_gm_{v}" for v in re.findall(r"^PD_GROUPED_ROWS\((\d+)\)", SOURCE, re.MULTILINE)]
    return [n for n in re.findall(KERNEL, SOURCE) if n.endswith("_g")] + names + [g + "_smem" for g in names] + pd


#: the cubin's empty kernels, which run.py looks for and never launches
MARKS = (RUN.PDL_MARK, RUN.PDL_ENTRY_MARK, RUN.BF16_ROWS_MARK)


def own_kernels() -> list[str]:
    plain = set(re.findall(KERNEL, SOURCE)) - {"NAME", *MARKS} - set(grouped_kernels())
    names = [n for n, *_ in gemms()]
    return sorted(plain) + names + [g + "_smem" for g in names]


def kernel_names() -> list[str]:
    """Every kernel in the cubin: pearl_c_sm90.cu's (the grouped calls' too) and those of the hashing it includes
    (hash_h3.cuh, hash_fast.cuh, hash_decode.cuh's variants and hash_prefill.cuh's)."""
    pd = ["pd_heads"] + [f"pd_{kind}_{v}" for v in PD_BUILT for kind in ("commit", "tree", "tiles", "probe")]
    pl = [f"pl_{kind}_{v}" for v in PL_BUILT for kind in PL_KINDS]
    return own_kernels() + list(MARKS) + grouped_kernels() + sorted(set(re.findall(KERNEL, HASH_SOURCE))) + pd + pl


def needs(name: str, p: list[int], i: list[int]) -> dict[int, int]:
    """Bytes each pointer slot must hold for the launch, from the kernel's indexing in pearl_c_sm90.cu."""
    if name == "lines":
        count = i[2]
        return {0: 32, 1: 32 * count, 2: 32 * count, 3: 128 * count, 4: 32 * i[4]}
    if name == "stats_s5":
        rows, k = i[:2]
        return {0: 4 * rows * k, 1: 4 * rows, 2: 4 * rows, 3: 4 * rows, 4: 8 * rows}
    if name == "eprime":
        return {0: 32 * i[0], 1: 4 * i[0], 2: 32 * i[0]}
    if name.startswith("form_"):
        rows, k = i[:2]
        ep = {5: 4 * rows, 6: 32 * rows} if name in ("form_s5_ep", "form_s5_epw") else {}
        words = {7: 32 * k, 8: 4 * rows * 32 * (k // RUN.WINDOW)} if name == "form_s5_epw" else {}
        return {0: 4 * rows * k, 1: 4 * rows, 2: 32 * rows, 3: 32 * k, 4: rows * k, **ep, **words}
    if name.startswith("gemm_") and not name.endswith("_smem"):
        M, N, K = i[:3]
        need = {0: M * K, 1: N * K, 10: 4 * 128}                 # Maps: A, B, P_A, P_B
        base = name.removesuffix("_n64").removesuffix("_d")
        if base in ("gemm_plain", "gemm_fast"):
            need.update({4: 2 * M * N, 7: 4 * M, 8: 4 * N})
        elif base in ("gemm_ct", "gemm_words"):
            assert base == "gemm_ct" or N == 32, name
            need[2] = 4 * M * N
        elif base == "gemm_words_win":                           # a slab of M x N words a window
            assert N == 32, name
            need[2] = 4 * M * N * (K // RUN.WINDOW)
        else:
            assert base in ("gemm_pearlc", "gemm_pearlc_s"), name
            need.update({4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
            if base == "gemm_pearlc":
                need.update({2: 4 * M * N, 3: 4 * M * N})
        return need
    if name in ("widen_bf16", "widen_fp8_block"):
        rows, k = i[:2]
        src, scales = RUN.widen_sizes(name.removeprefix("widen_"), rows, k)
        return {0: src, 2: 4 * rows * k, **({1: scales} if scales else {})}
    if name == "widen_a":
        rows, k, filled = i[:3]
        return {0: 2 * rows * k, 2: 4 * filled * k}
    if name == "pa":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 128 * i[0]}
    if name == "pa_w":
        return {0: 32 * i[0], 1: 128 * i[0] * i[1], 2: 128 * i[0], 3: 128 * i[0]}
    if name == "pb":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 32 * 32 * 4, 4: 128 * i[0]}
    if name == "s512_rows":
        return {0: 4 * i[0] * i[1], 1: 64 * i[0]}
    if name == "s512_seeds":
        return {0: 64 * i[0], 1: 32 * i[0], 2: 32 * i[0], 3: 104}
    if name == "s512_tile_leaves":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: 64 * -(-i[0] // 64) * (i[1] // 64)}
    if name == "s512_row_segs":
        return {0: 4 * i[0] * i[1], 1: 64 * i[0] * (i[1] // 256)}
    if name == "s512_tile_segs":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: 64 * tile_segs(i)}
    if name == "s512_folds":
        return {0: 64 * (i[2] * i[1] + (i[0] - i[2]) * i[3]), 1: 64 * i[0]}
    if name in ("s512_heads", "pd_heads"):
        return {}
    if name == "hm_leaves":
        return {0: 64 * i[0], 1: 64 * i[0], 2: 32, 3: 128 + 192 * 256 * 64, 5: 8}
    if name == "s512_frame_leaves":
        return {0: 64 * i[1], 1: 64}
    if name == "s512_pads":
        return {0: 64 * i[1], 1: 64}
    if name == "s512_nodes":
        return {0: 128 * i[1], 1: 64 * i[1], 2: 64}
    if name == "s512_tree_small":
        return {0: 64 * (2 * i[0] - 1), 1: 64}
    if name.startswith(("pd_commit_", "pd_tree_")):
        rows, k, _, m = i[:4]
        stamps = {13: 16 * RUN.PD_SLOTS * (rows + 1)} if p[13] else {}
        tree = {2: 64 * (2 * RUN.pow2(rows) - 1), 5: 104, 6: 64, 10: 32, 11: 32, **stamps}
        if name.startswith("pd_tree_"):
            return {**tree, **({12: 32 * m} if p[12] else {})}
        return {**tree, 0: 4 * rows * k, 1: 64 * rows, 3: 32 * rows, 4: 32 * rows, 7: 32, 8: 128 + 192 * 256 * 64, 9: 16,
                12: 32 * m}
    if name.startswith("pd_tiles_"):
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: 64 * -(-i[0] // 64) * (i[1] // 64)}
    if name.startswith("pl_leaves_"):
        return {0: 64 * i[0], 1: 64 * i[0], 2: 32, 3: 128 + 192 * 256 * 64, 4: 104, 5: 64}
    if name.startswith("pl_levels_"):
        width, _, levels = i[:3]
        return {0: 64 * sum(width >> q for q in range(levels + 1)), 1: 64}
    if name.startswith("pl_seed_a_"):
        return {0: 64, 1: 104, 2: 32, 3: 32}
    if name.startswith("pl_seeds_"):
        return {0: 64 * i[0], 1: 32 * i[0], 2: 32 * i[0], 3: 104}
    if name == "quant_s5":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0], 2: i[0] * i[1]}
    if name == "plainq":
        return {0: 4 * i[0] * i[1], 1: i[0] * i[1], 2: 4 * i[0]}
    if name == "gen_f32":
        return {0: 4 * i[0] * i[1]}
    if name == "diff_words":
        return {0: 4 * i[0], 1: 4 * i[0], 2: 8}
    if name.endswith("_smem"):
        return {0: 4}
    raise AssertionError(f"no footprint for {name}")


def tile_segs(i: list[int]) -> int:
    """s512_tile_segs' segments: 32 a tile of the full 64-row blocks, rows / 2 a tile of the last partial one."""
    rows, cols = i[:2]
    return rows // 64 * (cols // 64) * 32 + cols // 64 * (rows % 64 // 2)


def threads(name: str, i: list[int]) -> int:
    """Threads the launch must cover."""
    segs = RUN.RS_MAX_SEGS * (RUN.RS_MAX_SEGS + 1) // 2 + RUN.RS_MAX_SEGS + 1
    return {"lines": lambda: i[2], "stats_s5": lambda: 32 * i[0], "eprime": lambda: i[0], "pa": lambda: 64 * i[0],
            "pb": lambda: 32 * i[0], "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0],
            "s512_rows": lambda: i[0], "s512_seeds": lambda: i[0], "s512_tile_leaves": lambda: -(-i[0] // 64) * (i[1] // 64),
            "s512_row_segs": lambda: i[0] * (i[1] // 256), "s512_tile_segs": lambda: tile_segs(i), "s512_folds": lambda: i[0],
            "s512_heads": lambda: segs,
            "hm_leaves": lambda: i[0], "s512_frame_leaves": lambda: i[1], "s512_pads": lambda: i[1] - i[0],
            "s512_nodes": lambda: i[1], "s512_tree_small": lambda: 1, "quant_s5": lambda: i[0] * i[1] // 16,
            f"pl_leaves_{RUN.PL_VARIANT}": lambda: i[0], f"pl_seeds_{RUN.PL_VARIANT}": lambda: i[0],
            "widen_bf16": lambda: i[0] * i[1] // 8, "widen_fp8_block": lambda: i[0] * i[1] // 8,
            "widen_a": lambda: i[2] * i[1] // 8}.get(name, lambda: 0)()


@pytest.fixture(scope="module")
def ship(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the libcuda stand-in")
    out = tmp_path_factory.mktemp("pearl-c-sm90-ship")
    subprocess.run([cc, "-O1", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(out / "fake_cuda.so"), str(SM120_DIR / "fake_tma.c")],
                   check=True)
    shutil.copy(HERE / "run.py", out)
    (out / "pearl_c_sm90.cubin").write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    return out


def dry_env(ship: Path, log: Path, **kw) -> dict:
    env = dict(os.environ, PEARLC_LIBCUDA=str(ship / "fake_cuda.so"), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), FAKE_NAME=H200, FAKE_SMS="132", FAKE_CC="9.0",
               FAKE_UUID=UUID[4:].replace("-", ""), FAKE_MAXDYN=str(H100_MAX_DYN), FAKE_SMEM=str(tile_smem(128, 64)),
               CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", PEARLC_EXPECT_UUID=UUID, SHAPES="1024", REPS="2")
    env.update(kw)
    return {k: v for k, v in env.items() if v is not None}


@pytest.mark.slow
def test_runner_refuses_another_device(small_fixture, ship, tmp_path):
    """A wrong UUID, a second visible device index, another card, or an ungated run on the card: exit 3 before any launch."""
    for kw in ({"PEARLC_EXPECT_UUID": UUID.replace("0123abcd", "ffffffff")}, {"CUDA_VISIBLE_DEVICES": "1,2"},
               {"FAKE_NAME": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "FAKE_CC": "12.0"},
               {"FAKE_NAME": "NVIDIA GH200 480GB"}, {"PEARLC_EXPECT_UUID": None}, {"PEARLC_UNGATED": "1"}):
        log = tmp_path / "refused.jsonl"
        log.unlink(missing_ok=True)
        r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=dry_env(ship, log, **kw),
                           capture_output=True, text=True)
        assert r.returncode == 3, (kw, r.stderr)
        env = json.loads(r.stdout.splitlines()[0])
        assert env["kind"] == "env" and env["refused"] and len(r.stdout.splitlines()) == 1
        assert not log.exists() or '"launch"' not in log.read_text()


@pytest.mark.slow
def test_runner_dry_run(small_fixture, ship, tmp_path):
    """The identity passes, the check fails (the stand-in computes nothing) and times nothing; ungated on the stand-in, every
    launch of the check and the timing fits its buffers, and every GEMM's operands have maps at its tile's boxes."""
    log = tmp_path / "gated.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=dry_env(ship, log),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["kind"] for x in rows] == ["env", "mem", "check"] and rows[0]["uuid"] == UUID and rows[0]["cc"] == [9, 0]
    assert rows[1]["pdl"] is True and any(json.loads(x).get("pdl") for x in log.read_text().splitlines())
    unmarked = tmp_path / "unmarked.jsonl"
    old = dry_env(ship, unmarked, FAKE_KERNELS=",".join(n for n in kernel_names() if n != RUN.PDL_MARK))
    r2 = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=old, capture_output=True,
                        text=True)
    assert r2.returncode == 4 and json.loads(r2.stdout.splitlines()[1])["pdl"] is False, r2.stderr
    # a cubin without PDL_ENTRY_MARK (its widen_a and pd_commit don't wait) still launches its PDL_KERNELS programmatically
    entryless = tmp_path / "entryless.jsonl"
    r3 = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], capture_output=True, text=True,
                        env=dry_env(ship, entryless, FAKE_KERNELS=",".join(n for n in kernel_names() if n != RUN.PDL_ENTRY_MARK)))
    assert r3.returncode == 4 and any(json.loads(x).get("pdl") for x in entryless.read_text().splitlines()), r3.stderr
    assert not any(json.loads(x).get("pdl") for x in unmarked.read_text().splitlines())   # a cubin whose kernels don't wait
    assert all(not v["pearl_c_ok"] and v["failed"] for v in rows[2]["variants"].values())
    failed = rows[2]["variants"]["h100"]["failed"]
    assert {f"y@gemm_pearlc/{t}/grid3" for t, *_ in RUN.TILES} <= set(failed)
    # each tree's hm96 salt nonce is the fixture's: B's HM96_WEIGHTS + its weight, A's read from the unit block, the tiles'
    # the transcript's
    evs = [json.loads(x) for x in log.read_text().splitlines()]
    unit = {ev["ptr"] for ev in evs if ev["op"] == "alloc" and ev["bytes"] == 104}
    nonces = {(tuple(ev["i"][1:3]), ev["p"][5] - 96 in unit) for ev in evs if ev["op"] == "launch" and ev["name"] == "hm_leaves"}
    assert json.loads((small_fixture / "check.json").read_text())["weight"] == F.WEIGHT != 0
    assert nonces == {(RUN.nonce_args(F.HM96_WEIGHTS + F.WEIGHT), False), ((0, 0), True),
                      (RUN.nonce_args(F.HM96_TRANSCRIPT), False)}, nonces

    log = tmp_path / "launches.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1024x1024x3072 1024x960x2048"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:3] == ["env", "mem", "check"] and kinds.count("timing") == 2
    assert set(rows[2]["variants"]) == {"h100"}
    for t in rows[3:]:
        assert t["gates"]["ungated"] and {"split_ms", "call_e2e_ms", "ratio_vs_own_plain_gemm", "consistency", "job_ms"} <= set(t)
        assert set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"}
        assert {"form_a_scalar", "form_b_scalar"} <= set(t["ms"])
        assert {"c_ct_vs_pearlc_words_differing", "u_rerun_words_differing", "y_s_vs_pearlc_words_differing"} <= set(t["consistency"])

    allocs: dict[int, int] = {}
    seen, maps, timed_gemms = set(), set(), set()
    timing = False
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs, maps)
        if ev["op"] == "launch":
            seen.add(ev["name"])
            timing = timing or ev["name"] == "gen_f32"
            if timing and ev["name"].startswith("gemm_pearlc"):
                timed_gemms.add(ev["name"])
    # n = 960's B tree (960 rows) and tile tree (8 x 15 tiles) pad
    assert seen == set(own_kernels()) | H3_LAUNCHED | {"s512_pads"} | PD_LAUNCHED | PL_LAUNCHED
    # 1024 x 1024 on 128 x 128 tiles, 1024 x 960 (128 does not divide n) on 128 x 64
    assert timed_gemms == {"gemm_pearlc", "gemm_pearlc_s", "gemm_pearlc_n64", "gemm_pearlc_s_n64"}

    log = tmp_path / "launches_pads.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1008x1024x3072 40x1024x2048"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    ts = [x for x in map(json.loads, r.stdout.splitlines()) if x["kind"] == "timing"]
    assert [t["m_real"] for t in ts] == [1008, 40] and all(set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"}
                                                          for t in ts)
    allocs, launched, timed = {}, [], False
    maps = set()
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs, maps)
        if ev["op"] == "launch":
            timed = timed or ev["name"] == "gen_f32"
            if timed:
                launched.append(ev)
    # the tables: once, at the check; A's commitment past pd: chain "pl"
    assert H3_LAUNCHED - {"s512_heads", "s512_seeds"} | {"s512_pads"} | PL_LAUNCHED <= {ev["name"] for ev in launched}
    assert "s512_seeds" not in {ev["name"] for ev in launched}
    assert {ev["i"][0] for ev in launched if ev["name"] == "s512_row_segs"} == {1008, 1024, 40}   # A's real rows, B's
    lines_ea = [ev for ev in launched if ev["name"] == "lines" and ev["i"][4]]
    assert {(ev["i"][2], ev["i"][4]) for ev in lines_ea} == {(1024, 1008), (64, 40)}
    # decode's 40 rows pad to 64, so the chain runs on 64 x 64 tiles and A'F_B on 64 x 32, a CTA a window (WORDS_WIN) that
    # pa_w adds up: in A's forming launch (WORDS_FORM), and timed beside it on its own
    names = {ev["name"] for ev in launched}
    assert {"gemm_pearlc_d", "gemm_ct_d", "gemm_words_win_d", "form_s5_epw", "pa_w", "gemm_words"} <= names
    assert {ev["i"][0] for ev in launched if ev["name"] == "form_s5_epw"} == {64}
    # the main lane's PDL_KERNELS and PDL_ENTRY_KERNELS start behind the launch before them, and no other launch does
    pdl = (RUN.PDL_KERNELS | RUN.PDL_ENTRY_KERNELS) & names
    assert {ev["name"] for ev in launched if ev.get("pdl")} == pdl >= {"gemm_pearlc_d", "pa_w", "form_s5_ep", "form_s5_epw"}
    wins = [ev for ev in launched if ev["name"] == "gemm_words_win_d"]
    assert wins and all(ev["grid"][0] == min(132, ev["i"][2] // RUN.WINDOW) for ev in wins)


def room(allocs: dict[int, int], ptr: int) -> int:
    """Bytes from ptr to the end of the live allocation holding it (0 if none): a tree's root is its buffer's last node."""
    return next((base + n - ptr for base, n in allocs.items() if base <= ptr < base + n), 0)


def track(ev: dict, allocs: dict[int, int], maps: set | None = None) -> None:
    """One stand-in log line against the live allocations: a launch's pointers and grid cover its footprint, a copy's
    ranges are allocated, a tensor map's matrix is and its box is a tile's; with `maps` (the maps encoded so far), every
    GEMM's operands (and its peel's P_A, P_B) have maps of their shapes at its tile's box rows."""
    if ev["op"] == "alloc":
        allocs[ev["ptr"]] = ev["bytes"]
    elif ev["op"] == "free":
        allocs.pop(ev["ptr"])
    elif ev["op"] == "copy":
        for end in ("dst", "src"):
            assert allocs.get(ev[end], 0) >= ev["bytes"], ev
    elif ev["op"] == "tmap":
        assert room(allocs, ev["ptr"]) >= ev["depth"] * ev["rows"] and ev["box"][0] == 128 and ev["box"][1] in (32, 64, 128), ev
        assert ev["rows"] % ev["box"][1] == 0, ev
        if maps is not None:
            maps.add((ev["ptr"], ev["depth"], ev["rows"], ev["box"][1]))
    elif ev["op"] == "launch":
        name, p, i = ev["name"], ev["p"], ev["i"]
        for slot, n in needs(name, p, i).items():
            if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3), ("lines", 4), ("s512_seeds", 2), ("hm_leaves", 5)}:
                continue
            assert room(allocs, p[slot]) >= n, (name, slot, room(allocs, p[slot]), n)
        g = ev["grid"]
        assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
        if name.startswith("form_s5"):
            assert g[:2] == [i[1] // 128, i[0] // 64] and ev["block"] == 128
        if name == "plainq":
            assert g[0] == i[0]
        if name == "lines":
            assert i[3] in (0x4600, 0x4400)
        if name.startswith("gemm_") and not name.endswith("_smem"):
            tm, bn = RUN.gemm_tile(name)
            assert ev["block"] == 128 * (tm // 64 + 1) and 48 * 1024 < ev["smem"] <= H100_MAX_DYN
            wins = i[2] // RUN.WINDOW if name.startswith("gemm_words_win") else 1
            assert g[0] <= min(132, (i[0] // tm) * (i[1] // bn) * wins) and i[2] % 128 == 0 and i[0] % tm == 0 and i[1] % bn == 0
            if maps is not None:
                M, N, K = i[:3]
                assert {(p[0], K, M, tm), (p[1], K, N, bn)} <= maps, name
                assert not p[5] or {(p[5], 128, M, tm), (p[6], 128, N, bn)} <= maps, name
        if name == "s512_rows":
            assert g[0] == i[0] and i[1] % 256 == 0 and i[1] <= 256 * RUN.RS_MAX_SEGS
        if name == "s512_tile_leaves":
            assert g[0] == threads(name, i) and ev["block"] == 32 and i[0] % 16 == 0 and i[1] % 64 == 0
        if name in ("s512_row_segs", "s512_tile_segs"):
            assert ev["block"] == RUN.RS_FAST_NT and i[1] % (256 if name == "s512_row_segs" else 64) == 0
        if name == "s512_folds":
            assert ev["block"] == RUN.RS_FOLD_NT and i[1] <= RUN.RS_MAX_SEGS and i[3] <= RUN.RS_MAX_SEGS
        if name.startswith("pd_commit_"):
            assert g == [i[0], 1, 1] and i[0] <= RUN.PD_MAX_ROWS and i[1] % 256 == 0 and i[1] <= RUN.PD_MAX_K and i[2] in (0, 1)
            assert ev["block"] == 32 * (-(-i[1] // 256 // 32) * 2 + 2) and RUN.PD_PAIR_SMEM <= ev["smem"] <= H100_MAX_DYN
        if name.startswith("pd_tree_"):                     # the padding rows' lines past the real rows, where there are any
            assert g == [1, 1, 1] and ev["block"] == 128 and 0 < i[0] <= i[3] and i[0] <= RUN.PD_MAX_ROWS, ev
            assert bool(p[12]) == (i[0] < i[3]) and RUN.PD_PAIR_SMEM <= ev["smem"] <= H100_MAX_DYN, ev
        if name.startswith(("pd_commit_", "pd_tree_")):
            assert bool(p[13]) == bool(int(name.rsplit("_", 1)[1]) & RUN.PD_STAMP), ev
        if name.startswith("pl_"):
            half = RUN.PD_PAIR_SMEM // 2
            shape = {"pl_leaves": (-(-i[0] // RUN.PL_ROWS), 128, half), "pl_seed_a": (1, 64, half),
                     "pl_seeds": (-(-i[0] // RUN.PL_SEEDS), 64, half),
                     "pl_levels": (i[0] >> i[2], 128, RUN.PD_PAIR_SMEM)}[name.rsplit("_", 1)[0]]
            assert (g[0], ev["block"], ev["smem"]) == shape and g[1:] == [1, 1], ev
            assert name.startswith("pl_seed_a") or 0 < i[0] <= RUN.PL_MAX_ROWS
            if name.startswith("pl_levels"):
                assert i[0] & (i[0] - 1) == 0 and 1 <= i[2] <= RUN.PL_LEVELS and i[0] >> i[2] << i[2] == i[0], ev
        if name.startswith("pd_tiles_"):
            tiles = -(-i[0] // 64) * (i[1] // 64)
            if int(name.rsplit("_", 1)[1]) & RUN.PD_GROUP:
                assert g[0] == -(-i[0] // 64) * -(-(i[1] // 64) // RUN.PD_G) and ev["block"] == 32 * RUN.PD_G
                assert ev["smem"] == RUN.PD_PAIR_SMEM // 2
            else:
                assert g[0] == tiles and ev["block"] == 32 and ev["smem"] == 0
