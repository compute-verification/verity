"""Pearl-C's sm_120 tree off the card (benchmarks/pouw/pearl_c_sm120): the shared-memory swizzles conflict-free for every
ldmatrix phase and the GEMMs within sm_120's 99 KB, the kernel's constants the scheme's, the SASS gate refusing a lowered or
spilling build, the check fixture against the scheme on both device records, and run.py's identity refusal and dry run on
a CPU stand-in for libcuda (#449's, with fake_tma.c's tensor maps; an RTX PRO 6000's name, SM count, compute capability,
UUID and shared-memory limit), every launch's pointers and sizes checked against each kernel's footprint as pearl_c_sm120.cu
states it, and each mainloop GEMM's operands against the tensor maps encoded for them.  With VERITY_POUW_NVCC naming a
CUDA >= 12.8 toolkit, the real sm_120a build passes the gate."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.core.protocols.pouw.audit import weights_root
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening, coin_reason

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm120"


def load(name: str, path: Path):
    """The module at `path` under its own name: other trees in the suite have a fixture.py and a run.py too."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = load("pearl_c_sm120_fixture", HERE / "fixture.py")
S = load("pearl_c_sm120_sass_gate", HERE / "sass_gate.py")
RUN = load("pearl_c_sm120_run", HERE / "run.py")

SOURCE = (HERE / "pearl_c_sm120.cu").read_text()
HASH_SOURCE = (HERE.parent / "pearl_c" / "hash_h3.cuh").read_text() + (HERE / "hash_fast.cuh").read_text()
SM120_MAX_DYN = 101376                   # sm_120's opt-in dynamic shared memory per block (99 KB)
UUID = "GPU-0123abcd-4567-89ef-0123-456789abcdef"
RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def const(name: str) -> int:
    return int(re.search(rf"\b{name} = (\d+)", SOURCE).group(1))


def swz64(r: int, c: int) -> int:
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4)


def swz128(r: int, c: int) -> int:
    return r * 128 + ((c ^ (r & 7)) << 4)


def test_swizzles_are_the_kernels_and_conflict_free():
    """Each 16-byte chunk of a row lands once, and an ldmatrix phase (eight consecutive rows' chunk c, any 8-row base) reads
    eight distinct 16-byte bank groups."""
    assert "r * 64 + ((c ^ ((r >> 1) & 3)) << 4)" in SOURCE and "r * 128 + ((c ^ (r & 7)) << 4)" in SOURCE
    for swz, chunks in ((swz64, 4), (swz128, 8)):
        row = 16 * chunks
        for r in range(64):
            assert sorted(swz(r, c) - r * row for c in range(chunks)) == list(range(0, row, 16))
        for base in range(0, 64, 8):
            for c in range(chunks):
                assert len({(swz(base + j, c) // 16) % 8 for j in range(8)}) == 8


MAINLOOP = (HERE / "mainloop_sm120.cuh").read_text()
#: the mainloop's dynamic shared memory at 128 x 128 (ML_GEMM's Mainloop<128, 128, G, 3, ...>): 3 stages of 128-byte depth
#: and a barrier pair of 16 bytes per stage
ML_SMEM = 3 * (128 + 128) * 128 + 16 * 3


def gemm_smem(mode: str, tm: int, bn: int) -> int:
    """gemm_body's dynamic shared memory (the 64 x 64 and 128 x 32 GEMMs)."""
    bk, stages, peel = const("BK"), const("STAGES"), const("PEEL_ROW")
    return stages * (tm + bn) * bk + ((tm + bn) * peel if mode in ("M_PEARLC", "M_PEARLC_S") else 0)


def gemms() -> list[tuple[str, str, int, int, int]]:
    """Every GEMM kernel (name, mode, tile, G): the 128 x 128 ones on the mainloop (ML_GEMM), the others on gemm_body."""
    return ([(n, m, 128, 128, int(g)) for n, m, g in re.findall(r"^ML_GEMM\((\w+), (M_\w+), (\d+)\)", SOURCE, re.M)]
            + [(n, m, int(tm), int(bn), int(g))
               for n, m, tm, bn, g in re.findall(r"^GEMM\((\w+), (M_\w+), (\d+), (\d+), (\d+)\)", SOURCE, re.M)])


def gemm_tile(name: str) -> tuple[int, int]:
    if name.startswith("gemm_words"):
        return (128, 32)
    return (64, 64) if name.endswith(("_d", "_d_u")) else (128, 128)


def test_gemms_fit_sm120_shared_memory():
    """gemm_body's GEMMs and the mainloop's (ML_GEMM, all at 128 x 128, each with its _smem probe the mainloop's SMEM) fit
    sm_120's 99 KB, and the gate knows every one."""
    assert "STAGES * (TM + BN) * BK + (MODE == M_PEARLC || MODE == M_PEARLC_S ? (TM + BN) * PEEL_ROW : 0)" in SOURCE
    assert "SMEM = STAGES * STAGE + 16 * STAGES" in MAINLOOP and "STAGE = (TM + BN) * BK" in MAINLOOP
    assert "*out = ml::Mainloop<128, 128, G, 3, true, 1, ML_RP<G>>::SMEM;" in SOURCE
    names = {n for n, *_ in gemms()}
    ml = {"gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_ct_u", "gemm_pearlc_u", "gemm_pearlc_s", "gemm_pearlc_s_u"}
    assert names == ml | {"gemm_ct_d", "gemm_pearlc_d", "gemm_ct_d_u", "gemm_pearlc_d_u", "gemm_words", "gemm_words_u"}
    for name, mode, tm, bn, g in gemms():
        assert (ML_SMEM if name in ml else gemm_smem(mode, tm, bn)) <= SM120_MAX_DYN, name
        assert (tm, bn) == gemm_tile(name) and (name in ml) == ((tm, bn) == (128, 128))
        assert g in (0, 4) and (g == 0) == (name.endswith("_u") or name == "gemm_fast")
    assert ML_SMEM == 98352 and gemm_smem("M_PEARLC", 64, 64) == 49152
    assert form_kernels() == ["form_s5", "form_s5_r144", "form_s5_scalar", "form_rate", "form_rate_r144", "form_rate_scalar",
                              "form_rate_nocast", "form_rate_nocast_r144", "form_rate_atoms"]
    assert set(S.WANT) - {"words", "words_u", *form_kernels()} == names
    assert set(RUN.FORM_RATES) == set(form_rates()) and set(RUN.FORM_ALTS) == {"form_s5_scalar", "form_s5_r144"}



def test_k32_is_the_transposed_grid_of_column_pairs_and_leaves_every_product_unchanged():
    """form_s5's codes and the lines' transposed F lines in K32 order (the pous root's 10:29Z ruling): within each aligned
    32, position 8q + 2b + s holds column 8b + 2q + s; the kernel's k32 and the fixture's are the same map, an involution,
    and a product over k of two operands both in K32 order is the unpermuted one's, word for word, on both device records."""
    t = np.arange(96)
    order = F.k32(t)
    assert (F.k32(order) == t).all()
    grid = np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2).reshape(32)
    assert all((order[32 * i:32 * i + 32] == grid + 32 * i).all() for i in range(3))
    assert "(t & ~31) | (t >> 1 & 3) << 3 | (t >> 3 & 3) << 1 | (t & 1)" in SOURCE
    assert "t[(size_t)j * count + k32(line)] = c[j]" in SOURCE
    rng = np.random.default_rng(20260930)
    a, b = rng.integers(0, 256, (5, 1024), dtype=np.uint8), rng.integers(0, 256, (3, 1024), dtype=np.uint8)
    for x in (a, b):
        x[(x & 0x7F) == 0x7F] = 0
    for device in F.VARIANTS:
        assert (F.product(device, F.k32(a), F.k32(b)) == F.product(device, a, b)).all()
        assert (F.product(device, a, F.k32(b)) != F.product(device, a, b)).any()


def test_kernel_constants_match_the_reference():
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes import pearl_kw as P
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

    assert int(re.search(r"S5_LINE = (0x[0-9A-Fa-f]+);", SOURCE).group(1), 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    run = (HERE / "run.py").read_text()
    n256, n16 = re.search(r"NORM_256, NORM_16 = (0x[0-9A-F]+), (0x[0-9A-F]+)", run).groups()
    assert int(n256, 16) == P.f32_to_bf16(P.NOISE_NORM * 32) and int(n16, 16) == P.f32_to_bf16(C.LINE_NORM_V1 * 32)
    block = F.unit_block(b"s" * 32, b"b" * 64, 7)
    assert len(block) == RUN.TIMING_KEYS["unit"] == 104 and block[96:] == (7).to_bytes(8, "little")
    assert "unit[24] | (uint64_t)unit[25] << 32" in HASH_SOURCE and 'keys["unit"] + 96' in (HERE / "run.py").read_text()
    variants = dict(re.findall(r'"([\w-]+)": (\d)', re.search(r"VARIANT_G = \{[^}]*\}", run).group(0)))
    assert {k: int(v) for k, v in variants.items()} == {SM120.name: SM120.g, SM120_UNPROMOTED.name: 0}
    promoted = {g for *_, g in gemms() if g}
    assert promoted == {SM120.g} and SM120.window % const("BK") == 0 and SM120_UNPROMOTED.g is None


SASS_OK = "\n".join([
    "Function : gemm_x", "  /*0010*/ QMMA.16832.F32.E4M3.E4M3 R4, R8, R12, R4 ;",
    "  /*0020*/ HMMA.16816.F32.BF16 R4, R8, R12, R4 ;", "  /*0030*/ LDSM.16.M88.4 R8, [R2] ;"])
PTXAS_OK = ("ptxas info    : Compiling entry function 'gemm_x' for 'sm_120a'\n"
            "ptxas info    : Function properties for gemm_x\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 96 registers, used 1 barriers, 64 bytes smem\n")


def test_sass_gate_refuses_what_it_should(monkeypatch):
    monkeypatch.setattr(S, "WANT", {"gemm_x": (1, 1)})
    funcs, problems = S.gate(PTXAS_OK, SASS_OK)
    assert not problems and funcs["gemm_x"]["QMMA"] == 1 and funcs["gemm_x"]["HMMA"] == 1 and funcs["gemm_x"]["regs"] == 96
    bad = {
        "a lowered FP8 atom": SASS_OK.replace("QMMA.16832.F32.E4M3.E4M3", "HMMA.16816.F32.F16"),
        "a missing peel step": SASS_OK.replace("HMMA.16816.F32.BF16", "FFMA"),
        "local memory": SASS_OK + "\n  /*0040*/ LDL R1, [R1] ;",
        "a different atom": SASS_OK.replace("QMMA.16832.F32.E4M3.E4M3", "QMMA.16832.F32.E5M2.E5M2"),
    }
    for why, sass in bad.items():
        assert S.gate(PTXAS_OK, sass)[1], why
    assert S.gate(PTXAS_OK.replace("0 bytes spill stores", "8 bytes spill stores"), SASS_OK)[1]
    assert S.gate(PTXAS_OK + "ptxas warning : Registers are spilled\n", SASS_OK)[1]
    assert S.gate(PTXAS_OK, SASS_OK.replace("gemm_x", "gemm_y"))[1]


@pytest.mark.skipif(not os.environ.get("VERITY_POUW_NVCC"), reason="VERITY_POUW_NVCC names a CUDA >= 12.8 toolkit (about 15 s)")
def test_the_sm120a_build_passes_the_gate(tmp_path):
    cuda = Path(os.environ["VERITY_POUW_NVCC"])
    cubin, ptxas, sass = tmp_path / "k.cubin", tmp_path / "ptxas.txt", tmp_path / "k.sass"
    with open(ptxas, "w") as err:
        subprocess.run([str(cuda / "bin" / "nvcc"), "-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-fmad=false",
                        "-Xptxas", "-v", "-cubin", "-o", str(cubin), str(HERE / "pearl_c_sm120.cu")], check=True, stderr=err)
    env = dict(os.environ, PATH=f"{cuda / 'bin'}:{os.environ['PATH']}")
    sass.write_text(subprocess.run([str(cuda / "bin" / "cuobjdump"), "-sass", str(cubin)], check=True, capture_output=True, text=True,
                                   env=env).stdout)
    funcs, problems = S.gate(ptxas.read_text(), sass.read_text())
    assert not problems, problems
    for name in ("gemm_pearlc", "gemm_pearlc_u"):
        assert funcs[name]["HMMA"] == 64 and funcs[name]["QMMA"] % 32 == 0
    assert H3_LAUNCHED | {"s512_pads", "lines"} <= set(funcs)


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory):
    out = tmp_path_factory.mktemp("pearl-c-sm120-fixture")
    F.build(out, 128, 128, 1024)
    return out


@pytest.mark.slow
def test_fixture_agrees_with_the_scheme(small_fixture):
    man = json.loads((small_fixture / "check.json").read_text())
    assert man["shape"] == [128, 128, 1024] and man["layout"] == "rowmajor, k32"
    assert set(man["variants"]) == {"sm120", "sm120-unpromoted"}
    cs = {}
    for variant, spec in man["variants"].items():
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-{variant}-v1-h3"
        c = np.frombuffer((small_fixture / variant / "c.bin").read_bytes(), "<u4")
        u = np.frombuffer((small_fixture / variant / "u.bin").read_bytes(), "<u4")
        assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99
        cs[variant] = c
    assert (cs["sm120"] != cs["sm120-unpromoted"]).any()
    y_plain = (small_fixture / man["common"]["y_plain"]["file"]).read_bytes()
    y_fast = (small_fixture / man["common"]["y_fast"]["file"]).read_bytes()
    assert len(y_plain) == len(y_fast) == 2 * 128 * 128
    commitment, opening = CoinCommitment.from_json(man["coins"]), CoinOpening.from_json(man["salt_opening"])
    assert commitment == F.test_challenger().commitment and "beacon" not in man
    assert coin_reason(commitment, opening, "salt", weights_root({F.WEIGHT: bytes.fromhex(man["seeds"]["root_b"])})) is None


# ---- the dry run ---------------------------------------------------------------------------------------------------

KERNEL = r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\('
#: the -h3 kernels (hash_h3.cuh, hash_fast.cuh) the pipeline launches at the dry run's shapes, every count a power of two:
#: no pads; the row digests and tile leaves by hash_fast.cuh's kernels (run.py's default, PEARLC_SHA=fast)
H3_LAUNCHED = {"s512_heads", "s512_row_segs", "s512_tile_segs", "s512_folds", "hm_leaves", "s512_frame_leaves", "s512_nodes",
               "s512_tree_small", "s512_seeds"}


def form_kernels() -> list[str]:
    return re.findall(r"^FORM_KERNEL\((\w+), \d, (?:true|false), \w+\)", SOURCE, re.M)


def form_rates() -> list[str]:
    return [n for n in form_kernels() if n.startswith("form_rate")]


def own_kernels() -> list[str]:
    plain = set(re.findall(KERNEL, SOURCE)) - {"NAME"}
    names = [n for n, *_ in gemms()]
    return sorted(plain) + names + [g + "_smem" for g in names] + form_kernels()


def kernel_names() -> list[str]:
    """Every kernel in the cubin: pearl_c_sm120.cu's and hash_h3.cuh's, which it includes."""
    return own_kernels() + sorted(set(re.findall(KERNEL, HASH_SOURCE)))


def needs(name: str, p: list[int], i: list[int]) -> dict[int, int]:
    """Bytes each pointer slot must hold for the launch, from the kernel's indexing in pearl_c_sm120.cu."""
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
        return {0: 4 * rows * k, 1: 4 * rows, 2: 32 * rows, 3: 32 * k, 4: rows * k}
    if name.startswith("gemm_") and not name.endswith("_smem"):
        M, N, K = i[:3]
        need = {0: M * K, 1: N * K}
        base = name.removesuffix("_u").removesuffix("_d")
        if base in ("gemm_plain", "gemm_fast"):
            need.update({4: 2 * M * N, 7: 4 * M, 8: 4 * N})
        elif base in ("gemm_ct", "gemm_words"):
            assert base == "gemm_ct" or N == 32, name
            need[2] = 4 * M * N
        else:
            assert base in ("gemm_pearlc", "gemm_pearlc_s"), name
            need.update({4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
            if base == "gemm_pearlc":
                need.update({2: 4 * M * N, 3: 4 * M * N})
        if gemm_tile(name) == (128, 128):
            need[10] = 4 * 128                        # ml::Maps: A, B, P_A, P_B
        return need
    if name in ("words", "words_u"):
        M, N, K = i[:3]
        assert N == 32 and M % 16 == 0 and K % 128 == 0
        return {0: M * K, 1: N * K, 2: 4 * M * N}
    if name == "pa":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 128 * i[0]}
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
    if name == "s512_heads":
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
            "words": lambda: i[0] // 16 * (i[1] // 8) * 32, "words_u": lambda: i[0] // 16 * (i[1] // 8) * 32,
            "s512_rows": lambda: i[0], "s512_seeds": lambda: i[0], "s512_tile_leaves": lambda: -(-i[0] // 64) * (i[1] // 64),
            "s512_row_segs": lambda: i[0] * (i[1] // 256), "s512_tile_segs": lambda: tile_segs(i), "s512_folds": lambda: i[0],
            "s512_heads": lambda: segs,
            "hm_leaves": lambda: i[0], "s512_frame_leaves": lambda: i[1], "s512_pads": lambda: i[1] - i[0],
            "s512_nodes": lambda: i[1], "s512_tree_small": lambda: 1, "quant_s5": lambda: i[0] * i[1] // 16}.get(name, lambda: 0)()


@pytest.fixture(scope="module")
def ship(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the libcuda stand-in")
    out = tmp_path_factory.mktemp("pearl-c-sm120-ship")
    subprocess.run([cc, "-O1", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(out / "fake_cuda.so"), str(HERE / "fake_tma.c")],
                   check=True)
    shutil.copy(HERE / "run.py", out)
    (out / "pearl_c_sm120.cubin").write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    return out


def dry_env(ship: Path, log: Path, **kw) -> dict:
    env = dict(os.environ, PEARLC_LIBCUDA=str(ship / "fake_cuda.so"), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), FAKE_NAME=RTX, FAKE_SMS="188", FAKE_CC="12.0",
               FAKE_UUID=UUID[4:].replace("-", ""), FAKE_MAXDYN=str(SM120_MAX_DYN), FAKE_SMEM=str(ML_SMEM),
               CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", PEARLC_EXPECT_UUID=UUID, SHAPES="1024", REPS="2")
    env.update(kw)
    return {k: v for k, v in env.items() if v is not None}


@pytest.mark.slow
def test_runner_refuses_another_device(small_fixture, ship, tmp_path):
    """A wrong UUID, a second visible device index, another card, or an ungated run on the card: exit 3 before any launch."""
    for kw in ({"PEARLC_EXPECT_UUID": UUID.replace("0123abcd", "ffffffff")}, {"CUDA_VISIBLE_DEVICES": "1,2"},
               {"FAKE_NAME": "NVIDIA H100 80GB HBM3", "FAKE_CC": "9.0"}, {"PEARLC_EXPECT_UUID": None}, {"PEARLC_UNGATED": "1"}):
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
    launch of the check and the timing fits its buffers."""
    log = tmp_path / "gated.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=dry_env(ship, log),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["kind"] for x in rows] == ["env", "mem", "check"] and rows[0]["uuid"] == UUID and rows[0]["cc"] == [12, 0]
    assert all(not v["pearl_c_ok"] and v["failed"] for v in rows[2]["variants"].values())
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
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1024x1024x3072"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:3] == ["env", "mem", "check"] and kinds.count("timing") == 2
    assert set(rows[2]["variants"]) == {"sm120", "sm120-unpromoted"}
    for t in rows[3:]:
        assert t["gates"]["ungated"] and {"split_ms", "call_e2e_ms", "ratio_vs_own_plain_gemm", "consistency", "job_ms"} <= set(t)
        assert set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"}
        assert t["form_rate"]["blocks"] == 2 * 188 and set(form_rates()) <= set(t["form_rate"])
        assert {"form_a_scalar", "form_b_scalar"} <= set(t["ms"])

    allocs: dict[int, int] = {}
    seen, maps = set(), set()
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs, maps)
        if ev["op"] == "launch":
            seen.add(ev["name"])
    assert seen == set(own_kernels()) | H3_LAUNCHED
    for t in rows[3:]:
        assert {"c_ct_vs_pearlc_words_differing", "u_rerun_words_differing", "y_s_vs_pearlc_words_differing"} <= set(t["consistency"])

    log = tmp_path / "launches_pads.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1008x1024x3072", VARIANTS="sm120"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    (t,) = [x for x in map(json.loads, r.stdout.splitlines()) if x["kind"] == "timing"]
    assert t["m_real"] == 1008 and set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"}
    allocs, launched, timed = {}, [], False
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs)
        if ev["op"] == "launch":
            timed = timed or ev["name"] == "gen_f32"
            if timed:
                launched.append(ev)
    assert H3_LAUNCHED - {"s512_heads"} | {"s512_pads"} <= {ev["name"] for ev in launched}   # the tables: once, at the check
    assert {ev["i"][0] for ev in launched if ev["name"] == "s512_row_segs"} == {1008, 1024}   # A's real rows, B's
    assert {tuple(ev["i"][:2]) for ev in launched if ev["name"] == "s512_pads"} == {(1008, 1024)}     # A's; 16 x 16 tiles need none
    lines_ea = [ev for ev in launched if ev["name"] == "lines" and ev["i"][4]]
    assert lines_ea and all(ev["i"][2] == 1024 and ev["i"][4] == 1008 for ev in lines_ea)


def room(allocs: dict[int, int], ptr: int) -> int:
    """Bytes from ptr to the end of the live allocation holding it (0 if none): a tree's root is its buffer's last node."""
    return next((base + n - ptr for base, n in allocs.items() if base <= ptr < base + n), 0)


def track(ev: dict, allocs: dict[int, int], maps: set | None = None) -> None:
    """One stand-in log line against the live allocations: a launch's pointers and grid cover its footprint, a copy's
    ranges are allocated, a tensor map's matrix is; with `maps` (the maps encoded so far), a mainloop GEMM's operands (and
    its peel's P_A, P_B) have maps of their shapes."""
    if ev["op"] == "alloc":
        allocs[ev["ptr"]] = ev["bytes"]
    elif ev["op"] == "free":
        allocs.pop(ev["ptr"])
    elif ev["op"] == "copy":
        for end in ("dst", "src"):
            assert allocs.get(ev[end], 0) >= ev["bytes"], ev
    elif ev["op"] == "tmap":
        assert room(allocs, ev["ptr"]) >= ev["depth"] * ev["rows"] and ev["box"] == [128, 128], ev
        if maps is not None:
            maps.add((ev["ptr"], ev["depth"], ev["rows"]))
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
        if name.startswith("form_rate"):
            assert g[1:] == [1, 1] and 0 < g[0] <= (i[0] // 64) * (i[1] // 128) and ev["block"] == 128 and i[2] > 0
        if name == "plainq":
            assert g[0] == i[0]
        if name == "lines":
            assert i[3] in (0x4600, 0x4400)
        if name.startswith("gemm_") and not name.endswith("_smem"):
            tm, bn = gemm_tile(name)
            ml = (tm, bn) == (128, 128)
            assert ev["block"] == (384 if ml else 256) and 48 * 1024 < ev["smem"] <= SM120_MAX_DYN
            assert g[0] <= min(188, (i[0] // tm) * (i[1] // bn)) and i[2] % 128 == 0 and i[0] % tm == 0 and i[1] % bn == 0
            if ml and maps is not None:
                M, N, K = i[:3]
                assert {(p[0], K, M), (p[1], K, N)} <= maps, name
                assert not p[5] or {(p[5], 128, M), (p[6], 128, N)} <= maps, name
        if name in ("words", "words_u"):
            assert g[0] == i[0] // 16 and ev["block"] == 128
        if name == "s512_rows":
            assert g[0] == i[0] and i[1] % 256 == 0 and i[1] <= 256 * RUN.RS_MAX_SEGS
        if name == "s512_tile_leaves":
            assert g[0] == threads(name, i) and ev["block"] == 32 and i[0] % 16 == 0 and i[1] % 64 == 0
        if name in ("s512_row_segs", "s512_tile_segs"):
            assert ev["block"] == RUN.RS_FAST_NT and i[1] % (256 if name == "s512_row_segs" else 64) == 0
        if name == "s512_folds":
            assert ev["block"] == RUN.RS_FOLD_NT and i[1] <= RUN.RS_MAX_SEGS and i[3] <= RUN.RS_MAX_SEGS
