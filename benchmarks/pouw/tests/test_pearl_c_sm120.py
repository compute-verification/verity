"""Pearl-C's sm_120 tree off the card (benchmarks/pouw/pearl_c_sm120): the shared-memory swizzles conflict-free for every
ldmatrix phase and the GEMMs within sm_120's 99 KB, the kernel's constants the scheme's, the SASS gate refusing a lowered or
spilling build, the check fixture against the scheme on both device records, and run.py's identity refusal and dry run on
a CPU stand-in for libcuda (an RTX PRO 6000's name, SM count, compute capability, UUID and shared-memory limit), every
launch's pointers and sizes checked against each kernel's footprint as pearl_c_sm120.cu states it.  With VERITY_POUW_NVCC
naming a CUDA >= 12.8 toolkit, the real sm_120a build passes the gate."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
H100_DIR = HERE.parent / "pearl_c"
sys.path.insert(0, str(HERE))

import fixture as F  # noqa: E402
import sass_gate as S  # noqa: E402

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


def gemm_smem(mode: str, bn: int) -> int:
    bm, bk, stages, peel = const("BM"), const("BK"), const("STAGES"), const("PEEL_ROW")
    return stages * (bm + bn) * bk + ((bm + bn) * peel if mode == "M_PEARLC" else 0)


def gemms() -> list[tuple[str, str, int, int]]:
    return [(n, m, int(bn), int(g)) for n, m, bn, g in re.findall(r"^GEMM\((\w+), (M_\w+), (\d+), (\d+)\)", SOURCE, re.M)]


def test_gemms_fit_sm120_shared_memory():
    assert "STAGES * (BM + BN) * BK + (MODE == M_PEARLC ? (BM + BN) * PEEL_ROW : 0)" in SOURCE
    names = {n for n, *_ in gemms()}
    assert names == {"gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_words", "gemm_ct_u", "gemm_pearlc_u", "gemm_words_u"}
    for name, mode, bn, g in gemms():
        assert gemm_smem(mode, bn) <= SM120_MAX_DYN, name
        assert g in (0, 4) and (g == 0) == (name.endswith("_u") or name == "gemm_fast")
    assert gemm_smem("M_PEARLC", 128) == 98304
    assert set(S.WANT) - {"form_s5"} == names


def test_kernel_constants_match_the_reference():
    from verity_pouw.schemes import pearl_c as C
    from verity_pouw.schemes import pearl_kw as P
    from verity_pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED

    assert int(re.search(r"S5_LINE = (0x[0-9A-Fa-f]+);", SOURCE).group(1), 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    run = (HERE / "run.py").read_text()
    n256, n16 = re.search(r"NORM_256, NORM_16 = (0x[0-9A-F]+), (0x[0-9A-F]+)", run).groups()
    assert int(n256, 16) == P.f32_to_bf16(P.NOISE_NORM * 32) and int(n16, 16) == P.f32_to_bf16(C.LINE_NORM_V1 * 32)
    for d in (C.D_MESSAGE, C.D_LEAF, 0x23):
        assert f"0x{d:02x}" in SOURCE.lower()
    variants = dict(re.findall(r'"([\w-]+)": (\d)', re.search(r"VARIANT_G = \{[^}]*\}", run).group(0)))
    assert {k: int(v) for k, v in variants.items()} == {SM120.name: SM120.g, SM120_UNPROMOTED.name: 0}
    promoted = {g for name, _, _, g in gemms() if g}
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
    assert funcs["gemm_pearlc"]["HMMA"] == 64 and funcs["gemm_pearlc"]["QMMA"] % 32 == 0


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory):
    out = tmp_path_factory.mktemp("pearl-c-sm120-fixture")
    F.build(out, 128, 128, 1024)
    return out


@pytest.mark.slow
def test_fixture_agrees_with_the_scheme(small_fixture):
    man = json.loads((small_fixture / "check.json").read_text())
    assert man["shape"] == [128, 128, 1024] and man["layout"] == "rowmajor"
    assert set(man["variants"]) == {"sm120", "sm120-unpromoted"}
    cs = {}
    for variant, spec in man["variants"].items():
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-{variant}-v1"
        c = np.frombuffer((small_fixture / variant / "c.bin").read_bytes(), "<u4")
        u = np.frombuffer((small_fixture / variant / "u.bin").read_bytes(), "<u4")
        assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99
        cs[variant] = c
    assert (cs["sm120"] != cs["sm120-unpromoted"]).any()
    y_plain = (small_fixture / man["common"]["y_plain"]["file"]).read_bytes()
    y_fast = (small_fixture / man["common"]["y_fast"]["file"]).read_bytes()
    assert len(y_plain) == len(y_fast) == 2 * 128 * 128


# ---- the dry run ---------------------------------------------------------------------------------------------------

def kernel_names() -> list[str]:
    plain = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', SOURCE)) - {"NAME"}
    names = [n for n, *_ in gemms()]
    return sorted(plain) + names + [g + "_smem" for g in names]


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
    if name == "form_s5":
        rows, k = i[:2]
        return {0: 4 * rows * k, 1: 4 * rows, 2: 32 * rows, 3: 32 * k, 4: rows * k}
    if name.startswith("gemm_") and not name.endswith("_smem"):
        M, N, K = i[:3]
        need = {0: M * K, 1: N * K}
        base = name.removesuffix("_u")
        if base in ("gemm_plain", "gemm_fast"):
            need.update({4: 2 * M * N, 7: 4 * M, 8: 4 * N})
        elif base == "gemm_ct":
            need[2] = 4 * M * N
        elif base == "gemm_pearlc":
            need.update({2: 4 * M * N, 3: 4 * M * N, 4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
        else:
            assert N == 32
            need[2] = 4 * M * N
        return need
    if name == "pa":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 128 * i[0]}
    if name == "pb":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 32 * 32 * 4, 4: 128 * i[0]}
    if name == "hash_msg":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: i[0] * i[1]}
    if name == "hash_leaf":
        return {0: i[0] * i[1], 1: (i[0] // 64) * (i[1] // 64) * 32}
    if name == "tickets":
        return {0: 4 * i[0] * i[1], 1: (i[0] // 16) * (i[1] // 16) * 32, 2: 32}
    if name == "hash_rows_f32":
        assert i[1] % 1024 == 0
        return {0: i[0] * i[1], 1: i[0] * (i[1] // 1024) * 32}
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
            "pb": lambda: 32 * i[0], "hash_msg": lambda: i[0] * i[1] // 32, "hash_leaf": lambda: (i[0] // 64) * (i[1] // 64),
            "tickets": lambda: (i[0] // 16) * (i[1] // 16), "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0],
            "quant_s5": lambda: i[0] * i[1] // 16, "hash_rows_f32": lambda: i[0] * (i[1] // 1024)}.get(name, lambda: 0)()


@pytest.fixture(scope="module")
def ship(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the libcuda stand-in")
    out = tmp_path_factory.mktemp("pearl-c-sm120-ship")
    subprocess.run([cc, "-O1", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(out / "fake_cuda.so"), str(H100_DIR / "fake_cuda.c")],
                   check=True)
    shutil.copy(HERE / "run.py", out)
    (out / "pearl_c_sm120.cubin").write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    return out


def dry_env(ship: Path, log: Path, **kw) -> dict:
    env = dict(os.environ, PEARLC_LIBCUDA=str(ship / "fake_cuda.so"), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), FAKE_NAME=RTX, FAKE_SMS="188", FAKE_CC="12.0",
               FAKE_UUID=UUID[4:].replace("-", ""), FAKE_MAXDYN=str(SM120_MAX_DYN), FAKE_SMEM=str(gemm_smem("M_PEARLC", 128)),
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
                           capture_output=True, text=True, timeout=120)
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
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["kind"] for x in rows] == ["env", "mem", "check"] and rows[0]["uuid"] == UUID and rows[0]["cc"] == [12, 0]
    assert all(not v["pearl_c_ok"] and v["failed"] for v in rows[2]["variants"].values())

    log = tmp_path / "launches.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1"), capture_output=True, text=True, timeout=600)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:3] == ["env", "mem", "check"] and kinds.count("timing") == 2
    assert set(rows[2]["variants"]) == {"sm120", "sm120-unpromoted"}
    for t in rows[3:]:
        assert t["gates"]["ungated"] and {"split_ms", "call_e2e_ms", "ratio_vs_own_plain_gemm", "consistency", "job_ms"} <= set(t)
        assert set(t["split_ms"]) == {"forming", "gemm", "cleanup", "hashing"}

    allocs: dict[int, int] = {}
    seen = set()
    for ev in map(json.loads, log.read_text().splitlines()):
        if ev["op"] == "alloc":
            allocs[ev["ptr"]] = ev["bytes"]
        elif ev["op"] == "free":
            allocs.pop(ev["ptr"])
        else:
            name, p, i = ev["name"], ev["p"], ev["i"]
            seen.add(name)
            for slot, n in needs(name, p, i).items():
                if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3)}:
                    continue
                assert p[slot] in allocs, (name, slot)
                assert allocs[p[slot]] >= n, (name, slot, allocs[p[slot]], n)
            g = ev["grid"]
            assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
            if name == "form_s5":
                assert g[:2] == [i[1] // 128, i[0] // 64] and ev["block"] == 128
            if name == "plainq":
                assert g[0] == i[0]
            if name == "lines":
                assert i[3] in (0x4600, 0x4400)
            if name.startswith("gemm_") and not name.endswith("_smem"):
                bn = 32 if name.startswith("gemm_words") else 128
                assert ev["block"] == 256 and 48 * 1024 < ev["smem"] <= SM120_MAX_DYN
                assert g[0] <= min(188, (i[0] // 128) * (i[1] // bn)) and i[2] % 128 == 0 and i[0] % 128 == 0 and i[1] % bn == 0
    assert seen == set(kernel_names())
