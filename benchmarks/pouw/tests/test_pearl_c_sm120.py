"""Pearl-C's sm_120 tree off the card (benchmarks/pouw/pearl_c_sm120): the shared-memory swizzles conflict-free for every
ldmatrix phase and the GEMMs within sm_120's 99 KB, the kernel's constants the scheme's, the SASS gate refusing a lowered or
spilling build, the check fixture against the scheme on both device records, and run.py's identity refusal and dry run on
a CPU stand-in for libcuda (an RTX PRO 6000's name, SM count, compute capability, UUID and shared-memory limit), every
launch's pointers and sizes checked against each kernel's footprint as pearl_c_sm120.cu states it.  The FP8 bench arm
(pearlc_arm.py) against the harness's arm interface v0: its accounting the scheme's, its calls on the stand-in as the harness
drives them, and its gate on the fixture.  With VERITY_POUW_NVCC naming a CUDA >= 12.8 toolkit, the real sm_120a build passes
the gate."""

from __future__ import annotations

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


def gemm_smem(mode: str, tm: int, bn: int) -> int:
    bk, stages, peel = const("BK"), const("STAGES"), const("PEEL_ROW")
    return stages * (tm + bn) * bk + ((tm + bn) * peel if mode in ("M_PEARLC", "M_PEARLC_S", "M_PEARLC_H", "M_PEARLC_B") else 0)


def gemms() -> list[tuple[str, str, int, int, int]]:
    return [(n, m, int(tm), int(bn), int(g))
            for n, m, tm, bn, g in re.findall(r"^GEMM\((\w+), (M_\w+), (\d+), (\d+), (\d+)\)", SOURCE, re.M)]


def gemm_tile(name: str) -> tuple[int, int]:
    if name.startswith("gemm_words"):
        return (128, 32)
    return (64, 64) if name.endswith(("_d", "_d_u")) else (128, 128)


def test_gemms_fit_sm120_shared_memory():
    assert "STAGES * (TM + BN) * BK + (MODE == M_PEARLC || MODE >= M_PEARLC_S ? (TM + BN) * PEEL_ROW : 0)" in SOURCE
    assert "enum { M_PLAIN, M_CT, M_PEARLC, M_PEARLC_S, M_PEARLC_H, M_PEARLC_B };" in SOURCE
    names = {n for n, *_ in gemms()}
    assert names == {"gemm_plain", "gemm_fast", "gemm_ct", "gemm_pearlc", "gemm_ct_u", "gemm_pearlc_u",
                     "gemm_ct_d", "gemm_pearlc_d", "gemm_ct_d_u", "gemm_pearlc_d_u", "gemm_words", "gemm_words_u",
                     "gemm_pearlc_s", "gemm_pearlc_s_u", "gemm_pearlc_h", "gemm_pearlc_h_u", "gemm_pearlc_b", "gemm_pearlc_b_u"}
    for name, mode, tm, bn, g in gemms():
        assert gemm_smem(mode, tm, bn) <= SM120_MAX_DYN, name
        assert (tm, bn) == gemm_tile(name)
        assert g in (0, 4) and (g == 0) == (name.endswith("_u") or name == "gemm_fast")
    assert gemm_smem("M_PEARLC", 128, 128) == 98304 and gemm_smem("M_PEARLC", 64, 64) == 49152
    assert set(S.WANT) - {"form_s5", "words", "words_u"} == names


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
    keys = dict(re.findall(r'"(k_\w+)": "([0-9a-f]{64})"', re.search(r"H1_KEYS = \{[^}]*\}", run).group(0)))
    assert keys == {"k_msg": C.H1_KEYS["message"].hex(), "k_leaf": C.H1_KEYS["leaf"].hex()}
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
    for name in ("gemm_pearlc", "gemm_pearlc_s", "gemm_pearlc_h", "gemm_pearlc_b"):
        assert funcs[name]["HMMA"] == 64 and funcs[name]["QMMA"] % 32 == 0


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
    s = C.PearlC(k_min=32, tile=(64, 64), forming="v1", device=dev, hashing=W.HASHING)
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
    assert man["shape"] == [128, 128, 1024] and man["layout"] == "rowmajor"
    assert set(man["variants"]) == {"sm120", "sm120-unpromoted"}
    cs = {}
    for variant, spec in man["variants"].items():
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-{variant}-v1"
        assert spec["crosscheck"]["h1"]["agree"] and spec["crosscheck"]["h1"]["scheme"] == f"pearl-c-{variant}-v1-h1"
        assert {"digests_h1", "leaves_h1", "tree_t_h1"} <= set(spec["files"])
        c = np.frombuffer((small_fixture / variant / "c.bin").read_bytes(), "<u4")
        u = np.frombuffer((small_fixture / variant / "u.bin").read_bytes(), "<u4")
        assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99
        cs[variant] = c
    assert (cs["sm120"] != cs["sm120-unpromoted"]).any()
    y_plain = (small_fixture / man["common"]["y_plain"]["file"]).read_bytes()
    y_fast = (small_fixture / man["common"]["y_fast"]["file"]).read_bytes()
    assert len(y_plain) == len(y_fast) == 2 * 128 * 128


# ---- the dry run ---------------------------------------------------------------------------------------------------

HASH_SOURCE = (H100_DIR / "hash.cuh").read_text()


def kernel_names() -> list[str]:
    """Every kernel run.py launches: pearl_c_sm120.cu's, and hash.cuh's but its tickets (dropped from pearl-c-sm120)."""
    entry = r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\('
    plain = set(re.findall(entry, SOURCE)) - {"NAME"} | set(re.findall(entry, HASH_SOURCE)) - {"tickets"}
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
        base = name.removesuffix("_u").removesuffix("_d")
        if base in ("gemm_plain", "gemm_fast"):
            need.update({4: 2 * M * N, 7: 4 * M, 8: 4 * N})
        elif base in ("gemm_ct", "gemm_words"):
            assert base == "gemm_ct" or N == 32, name
            need[2] = 4 * M * N
        else:
            assert base in ("gemm_pearlc", "gemm_pearlc_s", "gemm_pearlc_h", "gemm_pearlc_b"), name
            need.update({4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
            if base == "gemm_pearlc":
                need.update({2: 4 * M * N, 3: 4 * M * N})
            if base in ("gemm_pearlc_h", "gemm_pearlc_b"):
                need[9] = M * N
            if base == "gemm_pearlc_b":
                need[10] = 32
        return need
    if name in ("words", "words_u"):
        M, N, K = i[:3]
        assert N == 32 and M % 16 == 0 and K % 128 == 0
        return {0: M * K, 1: N * K, 2: 4 * M * N}
    if name == "pa":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 128 * i[0]}
    if name == "pb":
        return {0: 32 * i[0], 2: 128 * i[0], 3: 32 * 32 * 4, 4: 128 * i[0]}
    if name == "hash_msg":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: i[0] * i[1], **({3: 32} if p[3] else {})}
    if name == "hash_leaf":
        return {0: i[0] * i[1], 1: -(-i[0] // 64) * (i[1] // 64) * 32, **({2: 32} if p[2] else {})}
    if name == "hash_rows":
        return {0: i[0] * i[1], 1: 32 * i[0], 2: 32}
    if name == "hash_pad":
        return {0: 32 * i[1], 1: 32}
    if name == "hash_node_keys":
        return {0: 32 * i[0], 1: 32}
    if name == "hash_nodes":
        return {0: 64 * i[1], 1: 32 * i[1], 2: 32}
    if name == "hash_tree_leaves":
        return {0: 32 * i[0], 1: 32 * i[0], 2: 32}
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
            "pb": lambda: 32 * i[0], "hash_msg": lambda: i[0] * i[1] // 32, "hash_leaf": lambda: -(-i[0] // 64) * (i[1] // 64),
            "hash_rows": lambda: 256 * i[0], "hash_pad": lambda: i[1] - i[0], "hash_nodes": lambda: i[1],
            "hash_tree_leaves": lambda: i[0],
            "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0], "words": lambda: i[0] // 16 * (i[1] // 8) * 32,
            "words_u": lambda: i[0] // 16 * (i[1] // 8) * 32,
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
    for cubin in ("pearl_c_sm120.cubin", "hash_sm120a.cubin"):
        (out / cubin).write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    return out


def dry_env(ship: Path, log: Path, **kw) -> dict:
    env = dict(os.environ, PEARLC_LIBCUDA=str(ship / "fake_cuda.so"), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), FAKE_NAME=RTX, FAKE_SMS="188", FAKE_CC="12.0",
               FAKE_UUID=UUID[4:].replace("-", ""), FAKE_MAXDYN=str(SM120_MAX_DYN), FAKE_SMEM=str(gemm_smem("M_PEARLC", 128, 128)),
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
    launch of the check and the timing fits its buffers, h0 and -h1 (96 rows pad A's frame-b3 tree)."""
    log = tmp_path / "gated.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=dry_env(ship, log),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["kind"] for x in rows] == ["env", "mem", "check"] and rows[0]["uuid"] == UUID and rows[0]["cc"] == [12, 0]
    assert all(not v["pearl_c_ok"] and v["failed"] for v in rows[2]["variants"].values())

    log = tmp_path / "launches.jsonl"
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")],
                       env=dry_env(ship, log, PEARLC_ANY_GPU="1", PEARLC_UNGATED="1", SHAPES="1024 96x1024x1024"),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:3] == ["env", "mem", "check"] and kinds.count("timing") == 8
    assert set(rows[2]["variants"]) == {"sm120", "sm120-unpromoted", "sm120-h1", "sm120-unpromoted-h1"}
    for t in rows[3:]:
        assert t["gates"]["ungated"] and {"split_ms", "call_e2e_ms", "ratio_vs_own_plain_gemm", "consistency", "job_ms",
                                          "call_e2e_fused_ms", "a_commitment"} <= set(t)
        assert set(t["split_ms"]) == {"forming", "gemm", "cleanup", "hashing"}
        assert t["hashing"] == ("h1" if t["variant"].endswith("-h1") else "h0")

    allocs: dict[int, int] = {}
    seen = set()
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs)
        if ev["op"] == "launch":
            seen.add(ev["name"])
    assert seen == set(kernel_names())


def room(allocs: dict[int, int], ptr: int) -> int:
    """Bytes from `ptr` to the end of the live allocation holding it (a tree level's pointer is inside its tree's), or -1."""
    return next((base + n - ptr for base, n in allocs.items() if base <= ptr < base + n), -1)


def track(ev: dict, allocs: dict[int, int]) -> None:
    """One stand-in log line against the live allocations: a launch's pointers and grid cover its footprint, a copy's
    ranges are allocated."""
    if ev["op"] == "alloc":
        allocs[ev["ptr"]] = ev["bytes"]
    elif ev["op"] == "free":
        allocs.pop(ev["ptr"])
    elif ev["op"] == "copy":
        for end in ("dst", "src"):
            assert allocs.get(ev[end], 0) >= ev["bytes"], ev
    elif ev["op"] == "launch":
        name, p, i = ev["name"], ev["p"], ev["i"]
        for slot, n in needs(name, p, i).items():
            if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3)}:
                continue
            assert room(allocs, p[slot]) >= n, (name, slot, room(allocs, p[slot]), n)
        g = ev["grid"]
        assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
        if name == "form_s5":
            assert g[:2] == [i[1] // 128, i[0] // 64] and ev["block"] == 128
        if name == "plainq":
            assert g[0] == i[0]
        if name == "lines":
            assert i[3] in (0x4600, 0x4400)
        if name.startswith("gemm_") and not name.endswith("_smem"):
            tm, bn = gemm_tile(name)
            assert ev["block"] == 256 and 48 * 1024 < ev["smem"] <= SM120_MAX_DYN
            assert g[0] <= min(188, (i[0] // tm) * (i[1] // bn)) and i[2] % 128 == 0 and i[0] % tm == 0 and i[1] % bn == 0
        if name in ("words", "words_u"):
            assert g[0] == i[0] // 16 and ev["block"] == 128


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
    """Work, W_ref and the credit are the scheme's on the arm's record, and G = 4's γ is #449's `bench.gamma` (the H100's, at
    the same stand-in prices); the unpromoted record credits no promotion adds, so its γ is its own."""
    from verity_pouw.protocol import Shape as MShape
    from verity_pouw.schemes import pearl_c as C

    sys.path.insert(0, str(H100_DIR))
    try:
        B = load("pearl_c_h100_bench", H100_DIR / "bench.py")
    finally:
        sys.path.remove(str(H100_DIR))
    Shape = sys.modules["arm"].Shape
    for cls in ARMS:
        arm = getattr(A, cls)()
        assert (arm.stream, arm.precision, arm.inputs) == ("fp8", "fp8-e4m3", "f32")
        assert arm.flag_sets == {"hash": {"hash", "xof"}, "nohash": frozenset()} and arm.panel["line"] == "pearl-c-sm120"
        for shape in (Shape("m8192-n8192-k8192", 8192, 8192, 8192), Shape("m32-n8192-k8192", 32, 8192, 8192)):
            assert arm.domain(shape) is None
            w, s, m = arm.work(shape), MShape(shape.m, shape.k, shape.n), A.padded(shape.m)
            assert (w.w_ref, w.credited) == (str(arm.scheme.wref(s)), str(arm.scheme.credit_of(s)))
            assert w.tensor_macs == {"fp8-e4m3": m * shape.n * shape.k + m * (C.R + arm.device.atom.k) * shape.k,
                                     "bf16": m * shape.n * C.PEEL_K}
            assert w.hashed_bytes == {"turboshake128/messages": 8 * m * shape.n, "turboshake128/leaves": m * shape.n,
                                      "turboshake128/rows": 4 * shape.m * shape.k}
            g = {k: Fraction(v) for k, v in w.gamma.items() if k != "basis"}
            if cls == "PearlCSm120":
                assert g == B.gamma(shape.n, shape.k)
            assert g["floor"] < g["cap"] < Fraction(1, 100) and g["cap"] < g["cap_chain_only"]
    arm = A.PearlCSm120()
    assert "multiple of 1,024" in arm.domain(Shape("x", 128, 1000, 128))
    assert "multiple of 128" in arm.domain(Shape("x", 128, 1024, 100))


def driver(A, ship: Path, arm_dir: Path, log: Path, cls: str, *args: str) -> list[dict]:
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "pearlc_arm_dry.py"), str(arm_dir), cls, *args],
                       env=dry_env(ship, log, PEARLC_SHIP=str(ship)), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return [json.loads(x) for x in r.stdout.splitlines()]


def windows(log: Path) -> tuple[list[tuple[dict, list[dict]]], dict[int, int]]:
    """The stand-in's log cut at the driver's marks: each launch or weight side with the launches, copies and flips inside."""
    out, allocs, cur = [], {}, None
    for ev in map(json.loads, log.read_text().splitlines()):
        track(ev, allocs)
        if ev["op"] == "mark" and ev["what"] in ("launch", "weight", "phases"):
            cur = (ev, [])
        elif ev["op"] == "mark" and ev["what"] == "end":
            out.append(cur)
            cur = None
        elif cur is not None and ev["op"] in ("launch", "copy", "mark"):
            cur[1].append(ev)
    return out, allocs


@pytest.mark.slow
@pytest.mark.parametrize("cls", ARMS)
def test_arm_dry_run(A, ship, arm_dir, tmp_path, cls):
    """As the harness drives it, on the stand-in: every call and weight side on the harness's stream with no host copy,
    the kernels each role's step needs and no others (nohash: no XOF, no hash), each operand set's E_A lines and call
    buffers its own, the phases the call's launches split before, at and after the chain, every footprint allocated; the
    gate fails (the stand-in computes nothing) naming every output in full, and nothing leaks."""
    log = tmp_path / "arm.jsonl"
    rows = driver(A, ship, arm_dir, log, cls, "dry")
    arm = getattr(A, cls)()
    sx = "" if arm.device.g else "_u"
    for g in rows[:-1]:
        assert not g["ok"] and set(g["checked"].values()) == {"all"}
        hashed = {"leaves", "rows"} if g["role"] == "hash" else set()
        assert set(g["checked"]) == ({"y"} | hashed | ({"digests"} if hashed else set()) if g["shape"] == "pre" else
                                     {"y", "c", "u"} | hashed)
        assert all(t["failed"] for t in g["detail"]["twin_tiles"])
    assert rows[-1]["live"] == []

    # prefill's m = 256 takes 128 x 128 tiles, so its chain is fused (gemm_pearlc_h, or gemm_pearlc_s without hashing)
    chain = lambda shape, role: {"pre": "gemm_pearlc_s" if role == "nohash" else "gemm_pearlc_h",  # noqa: E731
                                 "dec": "gemm_pearlc_d"}[shape] + sx
    forming = lambda shape, role: ["stats_s5", "eprime", "form_s5", {"pre": "gemm_words", "dec": "words"}[shape] + sx,  # noqa: E731
                                   "pa", chain(shape, role)]
    hashes = lambda shape: (["hash_leaf"] if shape == "pre" else ["hash_msg", "hash_leaf"]) + ["hash_rows_f32"]  # noqa: E731
    got, allocs = windows(log)
    assert not allocs
    ea: dict[tuple, int] = {}
    launched: dict[tuple, list[dict]] = {}
    for mark, evs in got:
        assert all(ev["stream"] == 0x1234 for ev in evs), mark
        names = [ev.get("name", ev["op"] if ev["op"] == "copy" else ev.get("what")) for ev in evs]
        copy = ["copy"] if mark["shape"] == "dec" else []
        if mark["what"] == "weight":
            assert names == ["lines"] * 3 + ["stats_s5", "eprime", "form_s5"] + ["gemm_words" + sx] * 2 + ["pb"]
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
        f, h = forming(mark["shape"], mark["role"]), hashes(mark["shape"])
        want = {"hash": copy + ["lines"] + f + h, "nohash": copy + f,
                "known_bad": copy + ["lines"] + f + ["xor_byte"] + h}[mark["role"]]
        assert names == want, mark
        if mark["role"] != "nohash":
            lines = evs[len(copy)]
            assert lines["p"][0] == mark["keys"]["line_a"] and lines["i"][2] == A.padded({"pre": 256, "dec": 32}[mark["shape"]])
            assert next(ev for ev in evs if ev.get("name") == "eprime")["p"][0] == lines["p"][1]
        e = next(ev for ev in evs if ev.get("name") == "eprime")["p"][0]
        assert ea.setdefault((mark["shape"], mark["set"]), e) == e
    for shape in ("pre", "dec"):
        assert ea[(shape, 0)] != ea[(shape, 1)]
        calls = [next(ev for ev in launched[(shape, "hash", s)] if ev.get("name", "").startswith("gemm_pearlc")) for s in (0, 1)]
        assert calls[0]["p"][4] != calls[1]["p"][4] and calls[0]["p"][0] != calls[1]["p"][0]
        if shape == "pre":
            assert calls[0]["p"][9] != calls[1]["p"][9] and calls[0]["p"][2] == 0


@pytest.mark.slow
@pytest.mark.parametrize("cls", ARMS)
def test_arm_gate_accepts_the_fixture_and_rejects_a_flip(A, small_fixture, ship, arm_dir, tmp_path, cls):
    """The fixture's bytes (the scheme's, at its seeds) where the kernels would have written them pass the gate; at its
    128 x 128 tiles the call is fused, and one message digest's byte flipped in the last sampled tile fails it twice over,
    against the second launch (unfused, hash_msg over gemm_pearlc's C̃ and U) and against the twin."""
    first, flipped = driver(A, ship, arm_dir, tmp_path / "accept.jsonl", cls, "accept", str(small_fixture / "check.json"))
    assert first["ok"] and first["checked"] == {name: "all" for name in ("leaves", "rows", "digests", "y")}
    assert not any(first["detail"]["words_differing_from_relaunch"].values())
    assert len(first["detail"]["twin_tiles"]) == 3 and not any(t["failed"] for t in first["detail"]["twin_tiles"])
    assert not flipped["ok"] and flipped["flipped"][0] == "digests"
    diff = flipped["detail"]["words_differing_from_relaunch"]
    assert diff["digests"] == sum(diff.values()) == 1
    assert [t["failed"] for t in flipped["detail"]["twin_tiles"]] == [[], [], ["digests"]]
