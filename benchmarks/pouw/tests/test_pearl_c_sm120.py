"""Pearl-C's sm_120 tree off the card (benchmarks/pouw/pearl_c_sm120): the shared-memory swizzles conflict-free for every
ldmatrix phase and the GEMMs within sm_120's 99 KB, the kernel's constants the scheme's, the SASS gate refusing a lowered or
spilling build, the check fixture against the scheme on both device records, and run.py's identity refusal and dry run on
a CPU stand-in for libcuda (#449's, with fake_tma.c's tensor maps; an RTX PRO 6000's name, SM count, compute capability,
UUID and shared-memory limit), every launch's pointers and sizes checked against each kernel's footprint as pearl_c_sm120.cu
states it, and each mainloop GEMM's operands against the tensor maps encoded for them.  The FP8 bench arm
(pearlc_arm.py) against the harness's arm interface v0: its accounting the scheme's, its calls on the stand-in as the harness
drives them, and its gate on the fixture.  With VERITY_POUW_NVCC naming a CUDA >= 12.8 toolkit, the real sm_120a build passes
the gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
H100_DIR = HERE.parent / "pearl_c"


def load(name: str, path: Path):
    """The module at `path` under its own name: #449's tree has a fixture.py too, and the suite imports both trees in one
    process."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = load("pearl_c_sm120_fixture", HERE / "fixture.py")
S = load("pearl_c_sm120_sass_gate", HERE / "sass_gate.py")
RUN = load("pearl_c_sm120_run", HERE / "run.py")
W = F.W

SOURCE = (HERE / "pearl_c_sm120.cu").read_text()
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
    ml = {"gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_ct_u", "gemm_pearlc_u", "gemm_pearlc_s", "gemm_pearlc_s_u",
          "gemm_pearlc_b", "gemm_pearlc_b_u"}
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


def test_decode_spreads_its_row_and_tile_leaves_and_prefill_does_not():
    """`run.spread` on the RTX PRO's 188 SMs: decode's 32 rows go one a CTA and its tiles two a CTA; prefill's 8,192 rows and
    16,384 tiles keep hash_rows_w's 8 and hash_leaf_p's 32; the weight side's smaller n spreads only as far as it fills."""
    sp = RUN.spread
    assert sp(32, RUN.ROWS_WARPS, 188) == 1 and sp(64, RUN.LEAF_TILES, 188, 2) == 2 and sp(448, RUN.LEAF_TILES, 188, 2) == 2
    assert sp(8192, RUN.ROWS_WARPS, 188) == 8 and sp(16384, RUN.LEAF_TILES, 188, 2) == 32
    assert sp(1024, RUN.ROWS_WARPS, 188) == 4 and sp(28672, RUN.ROWS_WARPS, 188) == 8


def test_kernel_constants_match_the_reference():
    from verity_pouw.schemes import pearl_c as C
    from verity_pouw.schemes import pearl_kw as P
    from verity_pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

    assert int(re.search(r"S5_LINE = (0x[0-9A-Fa-f]+);", SOURCE).group(1), 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    run = (HERE / "run.py").read_text()
    n256, n16 = re.search(r"NORM_256, NORM_16 = (0x[0-9A-F]+), (0x[0-9A-F]+)", run).groups()
    assert int(n256, 16) == P.f32_to_bf16(P.NOISE_NORM * 32) and int(n16, 16) == P.f32_to_bf16(C.LINE_NORM_V1 * 32)
    label = re.search(r"LINE_LABEL\[(\d+)\] = \{([^}]*)\}", SOURCE)
    assert bytes(ord(c) for c in re.findall(r"'(.)'", label.group(2))) == P.LABELS["noise-line"] and int(label.group(1)) == 23
    from verity.commitments.blake3 import blake3
    assert F.unit_block(b"s" * 32, b"b" * 32, 7)[72:] == blake3(C.LABELS["seed-A"]) == blake3(b"pearl-c/v0/seed-A")
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
    assert {"hash_rows_w", "hash_tree_small", "hash_msg_v", "hash_leaf_p", "seed_line_a", "b3s_level_keys", "hash_rows_b3s",
            "hash_leaves_b3s"} <= set(funcs)


@pytest.mark.slow
@pytest.mark.parametrize("device", ["sm120", "sm120-unpromoted"])
def test_twin_tile_is_the_scheme_on_the_record(device):
    """#449's twin on each sm_120 record, the bench arm's gate: a tile half of filler rows, and one on rows the caller
    passes, through `pearl_c` itself."""
    from verity_pouw.protocol import Context, Matmul, Tile
    from verity_pouw.protocol import Shape as MShape
    from verity_pouw.schemes import pearl_c as C
    from verity_pouw.schemes.pearl_c_device import DEVICES

    dev, small = DEVICES[device], W.Shape("small", 96, 128, 256)
    jb = W.job(small, dev)
    s = C.PearlC(k_min=32, tile=(64, 64), forming="v1", device=dev, hashing="h1")
    noise = C.Noise(jb.seeds["seed_a"], jb.seeds["seed_b"])
    ctx = Context(bytes(32), Matmul(0, MShape(small.m, small.k, small.n), 0), bytes(32), bytes(32))
    own = W.tile(jb, 64, 64)
    given = W.tile(jb, 0, 0, W.gen_f32(np.arange(64), small.k, 7), W.gen_f32(np.arange(64), small.k, 8))
    assert (own.a.words[small.m_real - 64:] == 0).all() and not (given.a.words == W.a_words(small, jb.seeds, np.arange(64))).all()
    for (r0, c0), t in (((64, 64), own), ((0, 0), given)):
        ra, rb = [r.astype("<u4").tobytes() for r in t.a.words], [r.astype("<u4").tobytes() for r in t.b.words]
        ops = s.encode(ctx, noise, ra, rb, Tile(tuple(range(r0, r0 + 64)), tuple(range(c0, c0 + 64))))
        checked = s.checked(ctx, ops)
        assert ops.a.codes == t.a.codes.tolist() and ops.b.codes == t.b.codes.tolist()
        assert ops.a.alpha == t.a.alpha.tolist() and ops.b.alpha == t.b.alpha.tolist()
        assert checked.c == t.c.tolist() and checked.u == t.u.tolist()
        assert s.digest(ctx, noise, checked) == t.leaf and s.tickets(noise, checked) == t.tickets


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
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-{variant}-v1-h1"
        c = np.frombuffer((small_fixture / variant / "c.bin").read_bytes(), "<u4")
        u = np.frombuffer((small_fixture / variant / "u.bin").read_bytes(), "<u4")
        assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99
        cs[variant] = c
    assert (cs["sm120"] != cs["sm120-unpromoted"]).any()
    y_plain = (small_fixture / man["common"]["y_plain"]["file"]).read_bytes()
    y_fast = (small_fixture / man["common"]["y_fast"]["file"]).read_bytes()
    assert len(y_plain) == len(y_fast) == 2 * 128 * 128


# ---- the dry run ---------------------------------------------------------------------------------------------------

KERNEL = r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\('
HASH_SOURCE = "".join((H100_DIR / f).read_text() for f in ("hash.cuh", "hash_sm120.cuh", "hash_h2.cuh"))
#: the -h1 kernels (#449's hash.cuh and hash_sm120.cuh) the pipeline launches at the dry run's shapes: no pads (every count a
#: power of two) and no level wider than hash_tree_small's; and -h2's (hash_h2.cuh) in place of the leaves'
H1_LAUNCHED = {"hash_rows_w", "hash_tree_small", "hash_node_keys", "hash_msg_v", "hash_leaf_p", "hash_tree_leaves"}
H2_LAUNCHED = H1_LAUNCHED - {"hash_rows_w", "hash_tree_leaves"} | {"hash_rows_b3s", "hash_leaves_b3s"}
#: the kernels of the headers pearl_c_sm120.cu includes from its own directory, and those that run under -h2 only (rows form
#: `s`, A's row leaves and stats_s5's outputs in one pass)
OWN_HEADERS = "".join((HERE / f).read_text() for f in ("h2_rows_stats.cuh",))
H2_ONLY = {"hash_rows_b3s_stats"}


def form_kernels() -> list[str]:
    return re.findall(r"^FORM_KERNEL\((\w+), \d, (?:true|false), \w+\)", SOURCE, re.M)


def form_rates() -> list[str]:
    return [n for n in form_kernels() if n.startswith("form_rate")]


def own_kernels() -> list[str]:
    plain = set(re.findall(KERNEL, SOURCE + OWN_HEADERS)) - {"NAME"}
    names = [n for n, *_ in gemms()]
    return sorted(plain) + names + [g + "_smem" for g in names] + form_kernels()


def kernel_names() -> list[str]:
    """Every kernel in the cubin: pearl_c_sm120.cu's and the hash headers' it includes."""
    return own_kernels() + sorted(set(re.findall(KERNEL, HASH_SOURCE)))


def needs(name: str, p: list[int], i: list[int]) -> dict[int, int]:
    """Bytes each pointer slot must hold for the launch, from the kernel's indexing in pearl_c_sm120.cu."""
    if name == "lines":
        count = i[2]
        return {0: 32, 1: 32 * count, 2: 32 * count, 3: 128 * count}
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
            assert base in ("gemm_pearlc", "gemm_pearlc_s", "gemm_pearlc_b"), name
            need.update({4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
            if base == "gemm_pearlc":
                need.update({2: 4 * M * N, 3: 4 * M * N})
            if base == "gemm_pearlc_b":
                need.update({9: M * N, 11: 32})       # a 32-byte digest per row and 32 columns; the message key
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
    if name == "hash_msg_v":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: i[0] * i[1], 3: 32}
    if name == "hash_leaf_p":
        return {0: i[0] * i[1], 1: -(-i[0] // 64) * (i[1] // 64) * 32, 2: 32}
    if name == "hash_rows_w":
        return {0: i[0] * i[1], 1: 32 * i[0], 2: 32}
    if name == "hash_tree_small":
        return {0: 32 * (2 * i[0] - 1), 1: 32 * (i[0].bit_length() - 1)}
    if name == "hash_tree_leaves":
        return {0: 32 * i[0], 1: 32 * i[0], 2: 32}
    if name == "b3s_level_keys":
        return {0: 32 * i[0], 1: 32}
    if name == "hash_rows_b3s":
        return {0: i[0] * i[1], 1: 32 * i[0], 2: 32, 3: 32 * max(RUN.b3s_levels(i[1]), 1), 4: 64}
    if name == "hash_leaves_b3s":
        return {0: i[0] * i[1], 1: 32 * i[0], 2: 32, 3: 64}
    if name == "hash_rows_b3s_stats":
        rows = i[4]
        return {0: rows * i[1], 1: 32 * i[0], 2: 32, 3: 32 * max(RUN.b3s_levels(i[1]), 1), 4: 64, 5: 4 * rows, 6: 4 * rows,
                7: 4 * rows, 8: 8 * rows}
    if name == "hash_node_keys":
        return {0: 32 * i[0], 1: 32}
    if name == "hash_pad":
        return {0: 32 * i[1], 1: 32}
    if name == "hash_nodes":
        return {0: 64 * i[1], 1: 32 * i[1], 2: 32}
    if name == "seed_line_a":
        return {0: 32, 1: 104, 2: 32, 3: 32}
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


def threads(name: str, i: list[int]) -> int:
    """Threads the launch must cover."""
    return {"lines": lambda: i[2], "stats_s5": lambda: 32 * i[0], "eprime": lambda: i[0], "pa": lambda: 64 * i[0],
            "pb": lambda: 32 * i[0], "hash_msg_v": lambda: i[0] * i[1] // 32, "hash_leaf_p": lambda: 4 * -(-i[0] // 64) * (i[1] // 64),
            "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0], "words": lambda: i[0] // 16 * (i[1] // 8) * 32,
            "words_u": lambda: i[0] // 16 * (i[1] // 8) * 32, "hash_rows_w": lambda: 32 * i[0], "hash_tree_leaves": lambda: i[0],
            "hash_tree_small": lambda: min(1024, i[0] // 2), "hash_node_keys": lambda: i[0], "seed_line_a": lambda: 1,
            "b3s_level_keys": lambda: i[0], "hash_leaves_b3s": lambda: i[0],
            "quant_s5": lambda: i[0] * i[1] // 16}.get(name, lambda: 0)()


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
    assert seen == set(own_kernels()) - H2_ONLY | H1_LAUNCHED
    for t in rows[3:]:
        assert t["fused"] and t["hashing"] == "h1"
        assert set(t["consistency"]) >= {"y_s_vs_unfused_words_differing", "y_fused_vs_unfused_words_differing",
                                         "digests_fused_vs_unfused_words_differing"}

    log = tmp_path / "launches_h2.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1000x1024x3072", VARIANTS="sm120",
                                   HASHING="h2"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    (t,) = [x for x in map(json.loads, r.stdout.splitlines()) if x["kind"] == "timing"]
    assert t["hashing"] == "h2" and not t["fused"] and set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"}
    allocs, seen, timed = {}, set(), False
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs)
        if ev["op"] == "launch":
            timed = timed or ev["name"] == "gen_f32"
            if timed:
                seen.add(ev["name"])
    assert H2_LAUNCHED | {"hash_pad"} <= seen and not seen & {"hash_rows_w", "hash_tree_leaves", "b3s_level_keys", *H2_ONLY}

    log = tmp_path / "launches_h2s.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1000x2048x3072", VARIANTS="sm120",
                                   HASHING="h2", PEARLC_A_ROWS="s"),
                       capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    (t,) = [x for x in map(json.loads, r.stdout.splitlines()) if x["kind"] == "timing"]
    assert t["hashing"] == "h2" and t["rows_form"] == "s" and "stats_a" not in t["ms"] and "hash_rows_a" in t["ms"]
    allocs, launched, timed = {}, [], False
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs)
        if ev["op"] == "launch":
            timed = timed or ev["name"] == "gen_f32"
            if timed:
                launched.append(ev)
    names = {ev["name"] for ev in launched}
    assert H2_LAUNCHED - {"hash_rows_b3s"} | H2_ONLY <= names
    stats = [ev for ev in launched if ev["name"] == "hash_rows_b3s_stats"]
    assert stats and all(ev["i"][:2] == [1000, 4 * 3072] and ev["i"][4] == ev["grid"][0] == 1024 for ev in stats)
    assert {ev["i"][0] for ev in launched if ev["name"] == "hash_rows_b3s"} == {2048}     # B's rows alone
    assert {ev["i"][0] for ev in launched if ev["name"] == "stats_s5"} == {2048}          # B's stats alone


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
            if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3)}:
                continue
            if not p[slot] and (name, slot) in {("seed_line_a", 3), ("hash_rows_b3s", 3), ("hash_rows_b3s_stats", 3)}:
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
        if name == "hash_rows_b3s":
            assert g[0] == i[0] and -(-i[1] // 256) <= ev["block"] <= 512 and ev["block"] % 32 == 0 and i[3] <= 64
        if name == "hash_leaves_b3s":
            assert i[1] <= 256 and i[3] <= 64
        if name == "hash_rows_b3s_stats":
            assert g[0] == i[4] >= i[0] and i[1] % 1024 == 0 and ev["block"] == 32 * -(-i[1] // 256 // 32) <= 512 and i[3] <= 64
            assert ev["smem"] == RUN.b3s_stats_smem(i[1]) <= 48 * 1024


# ---- the bench arm (pearlc_arm.py) against the harness's arm interface v0 ---------------------------------------------------

ARM_V0 = '''"""The harness's arm interface v0 (benchmarks/pouw/harness/arm.py), the members pearlc_arm uses."""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Shape:
    label: str
    m: int
    k: int
    n: int

    @property
    def mkn(self) -> int:
        return self.m * self.k * self.n


@dataclass(frozen=True)
class Buffer:
    ptr: int
    nbytes: int


@dataclass
class OperandSet:
    index: int
    weight_id: int
    seed: int
    a: Buffer
    b: Buffer
    a_scale: Buffer | None = None
    b_scale: Buffer | None = None
    host: Callable[[str], bytes] | None = None


@dataclass(frozen=True)
class Work:
    tensor_macs: dict
    w_ref: str
    credited: str
    unit: str
    hashed_bytes: dict
    weight_hashed_bytes: dict = field(default_factory=dict)
    gamma: dict | None = None


@dataclass
class Gate:
    ok: bool
    checked: dict
    detail: dict[str, Any] = field(default_factory=dict)
'''
ARMS = ["PearlCSm120", "PearlCSm120Unpromoted"]


@pytest.fixture(scope="module")
def arm_dir(tmp_path_factory):
    """The harness's arm.py once it is in the tree, the interface's text until then."""
    real = HERE.parent / "harness"
    if (real / "arm.py").exists():
        return real
    out = tmp_path_factory.mktemp("arm-v0")
    (out / "arm.py").write_text(ARM_V0)
    return out


@pytest.fixture(scope="module")
def A(arm_dir):
    sys.path.insert(0, str(arm_dir))
    try:
        return load("pearlc_arm", HERE / "pearlc_arm.py")
    finally:
        sys.path.remove(str(arm_dir))


def test_arm_accounting_is_the_schemes(A):
    """Work, W_ref and the credit are the scheme's on the arm's record, and G = 4's γ is #449's `bench.gamma` (the H100's
    formula) at SM120_PRICES and the record's cap, 1/1,000; the unpromoted record credits no promotion adds, so its γ is its
    own.  The hashed bytes are
    -h1's: A's rows and B's under their domains, the message digests, the tile leaves and both trees' nodes."""
    from verity_pouw.protocol import Shape as MShape
    from verity_pouw.schemes import pearl_c as C

    sys.path.insert(0, str(H100_DIR))
    try:
        B = load("pearl_c_h100_bench", H100_DIR / "bench.py")
    finally:
        sys.path.remove(str(H100_DIR))
    p = C.SM120_PRICES
    B.FADD_PER_MAC, B.PEEL_UNITS = Fraction(p.fadd, C.WINDOW), p.bf16_mac * C.PEEL_K
    B.FORMING_CREDITED, B.FORMING_A_ONLY = p.forming_credited, p.forming_a_only
    B.GAMMA0 = B.RHO_CAP = Fraction(1, 1000)
    Shape = sys.modules["arm"].Shape
    for cls in ARMS:
        arm = getattr(A, cls)()
        assert (arm.stream, arm.precision, arm.inputs) == ("fp8", "fp8-e4m3", "f32")
        assert arm.flag_sets == {"hash": {"hash"}, "nohash": frozenset()} and arm.panel["line"] == "pearl-c-sm120"
        assert arm.scheme.name == arm.name and arm.name.endswith("-v1-h1")
        assert arm.panel["version"] == {"PearlCSm120": "v1-h1", "PearlCSm120Unpromoted": "v2-h1"}[cls]
        for shape in (Shape("m8192-n8192-k8192", 8192, 8192, 8192), Shape("m32-n8192-k8192", 32, 8192, 8192)):
            assert arm.domain(shape) is None
            w, s, m, mr = arm.work(shape), MShape(shape.m, shape.k, shape.n), A.padded(shape.m), shape.m
            tiles = -(-mr // 64) * (shape.n // 64)
            assert (w.w_ref, w.credited) == (str(arm.scheme.wref(s)), str(arm.scheme.credit_of(s)))
            assert w.tensor_macs == {"fp8-e4m3": m * shape.n * shape.k + m * (C.R + arm.device.atom.k) * shape.k,
                                     "bf16": m * shape.n * C.PEEL_K}
            assert w.hashed_bytes == {"blake3/rows": mr * (64 + 4 * shape.k), "blake3/messages": 8 * mr * shape.n,
                                      "blake3/leaves": mr * shape.n, "blake3/tile_leaves": 96 * tiles,
                                      "blake3/tree_nodes": 64 * (A.R.pow2(mr) - 1 + A.R.pow2(tiles) - 1)}
            assert w.weight_hashed_bytes == {"blake3/rows": shape.n * (64 + 4 * shape.k),
                                             "blake3/tree_nodes": 64 * (A.R.pow2(shape.n) - 1)}
            g = {k: Fraction(v) for k, v in w.gamma.items() if k != "basis"}
            if cls == "PearlCSm120":
                assert g == B.gamma(shape.n, shape.k)
            assert g["floor"] < g["cap"] < Fraction(1, 100) and g["cap"] < g["cap_chain_only"]
    arm = A.PearlCSm120()
    assert "multiple of 1,024" in arm.domain(Shape("x", 128, 1000, 128))
    assert "multiple of 128" in arm.domain(Shape("x", 128, 1024, 100))
    assert arm.domain(Shape("x", 24, 1024, 128))       # the scheme's own rule: 16 | m


def driver(A, ship: Path, arm_dir: Path, tmp: Path, cls: str, mode: str, *args: str) -> list[dict]:
    """pearlc_arm_dry.py in `mode`, the run directory tmp/run and the rows under tmp/rows; its JSON lines."""
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "pearlc_arm_dry.py"), str(arm_dir), cls, mode, str(tmp / "run"),
                        *args], env=dry_env(ship, tmp / f"{mode}.jsonl", PEARLC_SHIP=str(ship), PEARLC_ROWS=str(tmp / "rows")),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in r.stdout.splitlines()]


def windows(log: Path) -> tuple[list[tuple[dict, list[dict]]], dict[int, int]]:
    """The stand-in's log cut at the driver's marks: each launch, weight side or transcript window with the launches, copies
    and flips inside."""
    out, allocs, maps, cur = [], {}, set(), None
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs, maps)
        if ev["op"] == "mark" and ev["what"] in ("launch", "weight", "phases", "transcripts"):
            cur = (ev, [])
        elif ev["op"] == "mark" and ev["what"] == "end":
            out.append(cur)
            cur = None
        elif cur is not None and ev["op"] in ("launch", "copy", "mark"):
            cur[1].append(ev)
    return out, allocs


def transcript(run: Path, arm: str, label: str, root: str = "transcripts") -> tuple[dict, Path]:
    path = run / root / arm / label / "manifest.json"
    return json.loads(path.read_text()), path


@pytest.mark.slow
@pytest.mark.parametrize("cls", ARMS)
def test_arm_dry_run(A, ship, arm_dir, tmp_path, cls):
    """As the harness drives it, on the stand-in: every call and weight side on the harness's stream with no host copy, the
    kernels each role's step needs and no others (nohash: no tree of A, no seed_A, no E_A lines, no hashes), E_A's line key
    seed_line_a's from A's root, each operand set's E_A lines and call buffers its own, the phases the call's launches split
    before, at and after the chain, every footprint allocated; the gate fails (the stand-in computes nothing, so seed_A isn't
    the scheme's and no twin tile matches) naming every output in full; freeing set 0's gated hash calls launches each once
    more, synchronously, and writes its transcript with the trees' frame-b3 sizes; nothing leaks."""
    rows = driver(A, ship, arm_dir, tmp_path, cls, "dry")
    arm = getattr(A, cls)()
    sx = "" if arm.device.g else "_u"
    end = rows[-1]
    fused = {"pre": True, "dec": False}             # prefill's 256 rows are 128 x 128 tiles; decode's 32 pad to 64 x 64
    for g in rows[:-1]:
        assert not g["ok"] and set(g["checked"].values()) == {"all"}
        stored = set() if fused[g["shape"]] else {"c", "u"}
        assert set(g["checked"]) == {"y"} | stored | (set(A.HASHED) if g["role"] == "hash" else set())
        assert g["detail"]["scheme"] == arm.name and all(t["failed"] for t in g["detail"]["twin_tiles"])
        if g["role"] == "hash":
            assert g["detail"]["seeds"] == {"seed_a": False, "line_a": False, "seed_b": True}
    assert end["live"] == [] and set(end["transcribed"]) == {"pre", "dec"}

    words = {"pre": "gemm_words", "dec": "words"}

    def chain(shape: str, role: str = "hash") -> str:
        return ("gemm_pearlc_s" if role == "nohash" else "gemm_pearlc_b") + sx if fused[shape] else "gemm_pearlc_d" + sx

    commit = ["hash_rows_w", "hash_tree_small", "seed_line_a", "lines"]
    forming = lambda shape: ["stats_s5", "eprime", "form_s5", words[shape] + sx, "pa"]  # noqa: E731
    hashes = lambda shape: ([] if fused[shape] else ["hash_msg_v"]) + ["hash_leaf_p", "hash_tree_leaves", "hash_tree_small"]  # noqa: E731
    weight = (["hash_node_keys", "hash_rows_w", "hash_tree_small", "hash_node_keys", "hash_node_keys"] + ["lines"] * 3
              + ["stats_s5", "eprime", "form_s5"] + ["gemm_words" + sx] * 2 + ["pb"])
    got, allocs = windows(tmp_path / "dry.jsonl")
    assert not allocs
    ea: dict[tuple, int] = {}
    launched: dict[tuple, list[dict]] = {}
    for mark, evs in got:
        names = [ev.get("name", ev["op"] if ev["op"] == "copy" else ev.get("what")) for ev in evs]
        copy = ["copy"] if mark["shape"] == "dec" else []
        if mark["what"] == "transcripts":
            sh = mark["shape"]
            call = copy + commit + forming(sh) + [chain(sh)] + hashes(sh)
            assert all(ev["stream"] == 0 for ev in evs)
            assert names == weight + call + copy + forming(sh) + [chain(sh, "nohash")] + weight + call
            man, _ = transcript(tmp_path / "run", arm.name, mark["shape"])
            neg, _ = transcript(tmp_path / "run", arm.name, mark["shape"], root="negative")
            m, n = man["shape"]["m"], man["shape"]["n"]
            tiles = -(-m // 64) * (n // 64)
            assert man["scheme"] == arm.name and man["commitment"]["count"] == tiles and man["matmul"] == {"index": 0, "weight": 0}
            assert {f: s["bytes"] for f, s in man["files"].items()} == {
                "tree_a": 32 * (2 * A.R.pow2(m) - 1), "tree_b": 32 * (2 * A.R.pow2(n) - 1), "tree_t": 32 * (2 * A.R.pow2(tiles) - 1),
                "leaves": 32 * tiles, "rows_a": 4 * m * man["shape"]["k"], "rows_b": 4 * n * man["shape"]["k"]}
            assert Path(man["files"]["rows_a"]["path"]).parent == tmp_path / "rows" / arm.name / mark["shape"]
            assert man["poisoned"] == neg["poisoned"] == "0xA5" and "negative_control" not in man
            assert neg["negative_control"] == A.NEGATIVE and {f: neg["files"][f] for f in ("rows_a", "rows_b")} == {
                f: man["files"][f] for f in ("rows_a", "rows_b")}
            continue
        assert all(ev["stream"] == end["stream"] for ev in evs), mark
        if mark["what"] == "weight":
            assert names == weight
            continue
        if mark["what"] == "phases":
            parts: dict[str, list[dict]] = {}
            for ev in evs:
                if ev["op"] == "mark" and ev["what"] == "phase":
                    cur = parts.setdefault(ev["name"], [])
                else:
                    cur.append(ev)
            whole = launched[(mark["shape"], mark["role"], mark["set"])]
            assert [ev for p in ("before", "gemm", "after") for ev in parts[p]] == whole, mark
            assert [ev["name"] for ev in parts["gemm"]] == [chain(mark["shape"], mark["role"])]
            assert [ev["name"] for ev in parts["after"]] == (hashes(mark["shape"]) if mark["role"] == "hash" else [])
            continue
        launched[(mark["shape"], mark["role"], mark["set"])] = evs
        sh, f = mark["shape"], forming(mark["shape"])
        want = {"hash": copy + commit + f + [chain(sh)] + hashes(sh), "nohash": copy + f + [chain(sh, "nohash")],
                "known_bad": copy + commit + f + [chain(sh), "xor_byte"] + hashes(sh)}[mark["role"]]
        assert names == want, mark
        by = {ev.get("name"): ev for ev in evs}
        if mark["role"] != "nohash":
            seed, lines, gemm = by["seed_line_a"], by["lines"], by[chain(sh)]
            assert lines["p"][0] == seed["p"][2] and seed["p"][1] == mark["keys"]["unit"]
            assert lines["i"][2] == A.padded({"pre": 256, "dec": 32}[sh])
            assert by["eprime"]["p"][0] == lines["p"][1]
            if fused[sh]:        # the digests gemm_pearlc_b writes under the message key are those the tile leaves read
                assert gemm["p"][9] == by["hash_leaf_p"]["p"][0] and gemm["p"][11] == mark["keys"]["k_msg"] and not gemm["p"][2]
            else:
                assert by["hash_msg_v"]["p"][:2] == gemm["p"][2:4] and by["hash_msg_v"]["p"][2] == by["hash_leaf_p"]["p"][0]
        e = by["eprime"]["p"][0]
        assert ea.setdefault((mark["shape"], mark["set"]), e) == e
    for shape in ("pre", "dec"):
        assert ea[(shape, 0)] != ea[(shape, 1)]
        calls = [next(ev for ev in launched[(shape, "hash", s)] if ev.get("name") == chain(shape)) for s in (0, 1)]
        assert all(calls[0]["p"][j] != calls[1]["p"][j] for j in ((0, 4, 9, 10) if fused[shape] else (0, 2, 3, 4)))


@pytest.mark.slow
@pytest.mark.parametrize("cls", ARMS)
def test_arm_gate_accepts_the_fixture_and_rejects_a_flip(A, small_fixture, ship, arm_dir, tmp_path, cls):
    """The fixture's bytes (the scheme's, at its epoch and seeds) where the kernels would have written them pass the gate, the
    device's seed_A and E_A's line key the scheme's from its root_A.  At the fixture's 128 rows the call is fused, so it stores
    y and the digests, no C̃ or U: the twin reads the second (unfused) launch's C̃ and U, and one message digest flipped in the
    last sampled tile's row fails the gate against the second launch's hash_msg_v (on the stand-in, which computes no tile
    leaf, there alone).  Restored, it passes again and freeing the call writes the transcript, which the reference verifier
    (verify.py, `verity_pouw.audit.Verifier`) accepts, and its negative control, whose trees and leaves are all poison (the
    stand-in writes nothing) and which the verifier rejects."""
    first, flipped, restored = driver(A, ship, arm_dir, tmp_path, cls, "accept", str(small_fixture / "check.json"))
    arm = getattr(A, cls)()
    assert first["ok"] and first["checked"] == {name: "all" for name in ("y",) + A.HASHED}
    assert not any(first["detail"]["words_differing_from_relaunch"].values())
    assert first["detail"]["seeds"] == {"seed_a": True, "line_a": True, "seed_b": True}
    assert len(first["detail"]["twin_tiles"]) == 3 and not any(t["failed"] for t in first["detail"]["twin_tiles"])
    assert not flipped["ok"] and flipped["flipped"][0] == "digests"
    diff = flipped["detail"]["words_differing_from_relaunch"]
    assert diff["digests"] == 1 and sum(diff.values()) == 1
    assert not any(t["failed"] for t in flipped["detail"]["twin_tiles"])
    assert restored["ok"] and set(restored["transcribed"]) == {"fixture"}

    man, path = transcript(tmp_path / "run", arm.name, "fixture")
    check = json.loads((small_fixture / "check.json").read_text())
    assert man["beacon"] == check["beacon"] and man["weight_root"] == check["seeds"]["root_b"]
    _, neg_path = transcript(tmp_path / "run", arm.name, "fixture", root="negative")
    assert all(set((neg_path.parent / f"{f}.bin").read_bytes()) == {A.POISON} for f in ("tree_a", "tree_t", "tree_b", "leaves"))
    r = subprocess.run([sys.executable, str(HERE / "verify.py"), str(tmp_path / "run"), "--draws", "1", "--jobs", "1",
                        "--commit", "test", "--run", "r-test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    line, control = r.stdout.splitlines()
    sha, neg_sha = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, neg_path))
    assert line.startswith(f"ACCEPT {arm.name} fixture: verity_pouw.audit.Verifier, Sampled(1) drew tiles ")
    assert line.endswith(f"; transcript sha256:{sha}; verifier test; run r-test")
    assert control.startswith(f"REJECT negative control {arm.name} fixture: ")
    assert control.endswith(f"; transcript sha256:{neg_sha}; verifier test; run r-test")
    verdict = json.loads((tmp_path / "run" / "verify.json").read_text())
    assert verdict["ok"] and verdict["transcripts"][0]["accepted"] and verdict["transcripts"][0]["line"] == line
    assert not verdict["negative_controls"][0]["accepted"] and verdict["negative_controls"][0]["line"] == control


@pytest.mark.slow
def test_a_transcript_its_call_did_not_write_is_rejected(A, small_fixture, ship, arm_dir, tmp_path):
    """The stale-output case (bc-b139c29c, 11:33Z): the fixture's bytes pass the gate, but the transcript's call writes
    nothing, as a kernel that skipped its commitment would.  The dump then reads the poison, not the gate's bytes, and the
    verifier rejects it and its negative control, so verify.py fails."""
    *_, restored = driver(A, ship, arm_dir, tmp_path, "PearlCSm120", "stale", str(small_fixture / "check.json"))
    assert restored["ok"] and set(restored["transcribed"]) == {"fixture"}
    _, path = transcript(tmp_path / "run", "pearl-c-sm120-v1-h1", "fixture")
    assert set((path.parent / "tree_a.bin").read_bytes()) == {A.POISON}
    r = subprocess.run([sys.executable, str(HERE / "verify.py"), str(tmp_path / "run"), "--draws", "1", "--jobs", "1",
                        "--commit", "test", "--run", "r-test"], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    line, control = r.stdout.splitlines()
    assert line.startswith("REJECT pearl-c-sm120-v1-h1 fixture: ") and control.startswith("REJECT negative control ")
    assert not json.loads((tmp_path / "run" / "verify.json").read_text())["ok"]


@pytest.mark.slow
def test_verify_rejects_what_the_prover_cannot_open(A, small_fixture, ship, arm_dir, tmp_path):
    """verify.py rejects, before any tile is recomputed: another root_A (every drawn tile's activation openings fail), another
    tile root (the leaf openings), a beacon the salt isn't from, a file that isn't the manifest's (its SHA-256), a scheme the
    arguments don't name; and a run directory with no transcript, or an arm's error in place of one."""
    driver(A, ship, arm_dir, tmp_path, "PearlCSm120", "accept", str(small_fixture / "check.json"))
    V = load("pearl_c_sm120_verify", HERE / "verify.py")
    _, path = transcript(tmp_path / "run", "pearl-c-sm120-v1-h1", "fixture")
    flip = lambda h: ("11" if h[:2] == "00" else "00") + h[2:]  # noqa: E731

    def tampered(edit) -> dict:
        d = tmp_path / "t" / edit.__name__
        shutil.copytree(path.parent, d)
        man = json.loads(path.read_text())
        edit(man, d)
        (d / "manifest.json").write_text(json.dumps(man))
        return V.verify(d / "manifest.json", 4, "test")

    def root_a(man, d):
        man["activation_root"] = flip(man["activation_root"])

    def tile_root(man, d):
        man["commitment"]["root"] = flip(man["commitment"]["root"])

    def beacon(man, d):
        man["beacon"] = flip(man["beacon"])

    def leaves(man, d):
        (d / "leaves.bin").write_bytes(bytes(32) + (d / "leaves.bin").read_bytes()[32:])

    def scheme(man, d):
        man["scheme_args"]["device"] = "sm120-unpromoted"

    for edit, why in ((root_a, "activation opening"), (tile_root, "transcript opening"), (beacon, "salt"),
                      (leaves, "the manifest says"),
                      (scheme, "is not")):
        r = tampered(edit)
        assert not r["accepted"] and r["line"].startswith("REJECT") and why in r["line"], (edit.__name__, r["line"])

    empty, failed = tmp_path / "empty", tmp_path / "failed" / "transcripts" / "pearl-c-sm120-v1-h1" / "pre"
    empty.mkdir()
    failed.mkdir(parents=True)
    (failed / "error.txt").write_text("RuntimeError: no\n")
    for run, missing in ((empty, []), (failed.parents[2], ["transcripts/pearl-c-sm120-v1-h1/pre"])):
        assert V.main([str(run), "--run", "r-test"]) == 1
        assert json.loads((run / "verify.json").read_text()) == {
            "verifier_commit": os.environ.get("RESEARCH_SOURCE_SHA", "unknown"), "draws": 3, "run": "r-test", "ok": False,
            "transcripts": [], "negative_controls": [], "uncontrolled": [], "missing": missing}

    alone = tmp_path / "alone"
    shutil.copytree(path.parent, alone / "transcripts" / "pearl-c-sm120-v1-h1" / "fixture")
    assert V.main([str(alone), "--draws", "1", "--jobs", "1", "--run", "r-test"]) == 1
    verdict = json.loads((alone / "verify.json").read_text())
    assert verdict["transcripts"][0]["accepted"] and verdict["uncontrolled"] == [verdict["transcripts"][0]["manifest"]]
