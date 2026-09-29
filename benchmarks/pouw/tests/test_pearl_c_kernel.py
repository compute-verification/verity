"""Pearl-C's H100 harness off the pod (benchmarks/pouw/pearl_c): the tiled layout against the kernel's offset formula, the
check fixture against the scheme itself, and run.py's dry run on a CPU stand-in for libcuda, every launch's pointers and
sizes checked against each kernel's footprint as pearl_c.cu states it."""

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

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
sys.path.insert(0, str(HERE))

import fixture as F  # noqa: E402

SOURCE = (HERE / "pearl_c.cu").read_text()


def kernel_offset(row: int, k: int, h: int, depth: int) -> int:
    """pearl_c.cu's tiled()."""
    return ((row // h) * (depth // 32) + k // 32) * (h * 32) + ((row % h) >> 3) * 256 + ((k & 31) >> 4) * 128 + (row & 7) * 16 + (k & 15)


@pytest.mark.parametrize("rows,depth,h", [(128, 64, 128), (256, 1024, 128), (32, 256, 32), (128, 128, 128)])
def test_tiled_layout_is_the_kernels(rows, depth, h):
    rng = np.random.default_rng(rows + depth)
    x = rng.integers(0, 256, (rows, depth), dtype=np.uint8)
    got = F.tiled(x, h)
    want = bytearray(rows * depth)
    for r in range(rows):
        for k in range(depth):
            want[kernel_offset(r, k, h, depth)] = x[r, k]
    assert got == bytes(want)


def test_kernel_constants_match_the_reference():
    from verity_pouw.schemes import pearl_c as C
    from verity_pouw.schemes import pearl_kw as P

    consts = dict(re.findall(r"constexpr uint(?:16|32)_t (\w+) = (0x[0-9A-Fa-f]+);", SOURCE))
    dr, dstd = P.delta_consts(P.H100)
    assert int(consts["BF_DELTA_R"], 16) == dr and int(consts["BF_DELTA_STD"], 16) == dstd
    assert int(consts["BF_QMAX"], 16) == P.f32_to_bf16(P.Q_MAX) and int(consts["BF_FLOOR"], 16) == P.NORM_FLOOR_BF16
    assert int(consts["S5_LINE"], 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    for d in (C.D_MESSAGE, C.D_LEAF, F.D_ROWS):
        assert f"0x{d:02x}" in SOURCE.lower()


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory):
    out = tmp_path_factory.mktemp("pearl-c-fixture")
    F.build(out, 128, 128, 1024)
    return out


@pytest.mark.slow
def test_fixture_agrees_with_the_scheme(small_fixture):
    man = json.loads((small_fixture / "check.json").read_text())
    assert man["shape"] == [128, 128, 1024]
    for forming, spec in man["formings"].items():
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-h100-{forming}"
        c = np.frombuffer((small_fixture / forming / "c.bin").read_bytes(), "<u4")
        u = np.frombuffer((small_fixture / forming / "u.bin").read_bytes(), "<u4")
        assert c.size == u.size == 128 * 128 and (c != u).mean() > 0.99


# ---- the dry run ---------------------------------------------------------------------------------------------------

def kernel_names() -> list[str]:
    plain = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', SOURCE)) - {"NAME"}
    gemms = re.findall(r"^GEMM\((\w+),", SOURCE, re.M)
    return sorted(plain) + gemms + [g + "_smem" for g in gemms]


def needs(name: str, p: list[int], i: list[int]) -> dict[int, int]:
    """Bytes each pointer slot must hold for the launch, from the kernel's indexing in pearl_c.cu."""
    if name == "lines":
        side, factor, count = i[:3]
        return {0: 32, 1: 32 * count, 2: 32 * count, 3: 32 * count, 4: 128 * count}
    if name in ("norms", "stats_s5"):
        rows, k = i[:2]
        v1 = name == "stats_s5"
        return {0: rows * (4 * k if v1 else k + k // 4), 1: (4 if v1 else 2) * rows, 2: (4 if v1 else 2) * rows, 3: 4 * rows,
                4: (8 if v1 else 4) * rows}
    if name == "eprime":
        return {0: 32 * i[0], 1: 4 * i[0], 2: 32 * i[0], 3: 32 * i[0]}
    if name in ("form", "form_s5"):
        rows, k = i[:2]
        v1 = name == "form_s5"
        need = {0: rows * (4 * k if v1 else k + k // 4), 1: (4 if v1 else 2) * rows, 3: 32 * rows, 4: 32 * k, 5: rows * k}
        if not v1:
            need[2] = 2 * rows
        return need
    if name.startswith("gemm_") and not name.endswith("_smem"):
        M, N, K = i[:3]
        need = {0: M * K, 1: N * K}
        if name in ("gemm_plain", "gemm_fast"):
            need.update({4: 2 * M * N, 7: 4 * M, 8: 4 * N})
        elif name == "gemm_ct":
            need[2] = 4 * M * N
        elif name == "gemm_pearlc":
            need.update({2: 4 * M * N, 3: 4 * M * N, 4: 2 * M * N, 5: 128 * M, 6: 128 * N, 7: 4 * M, 8: 4 * N})
        else:
            assert N == 32
            need[2] = 4 * M * N
        return need
    if name == "pa":
        rows, v1 = i[:2]
        return {0: 32 * rows, 2: 128 * rows, 3: 128 * rows, **({} if v1 else {1: 2 * rows})}
    if name == "pb":
        rows, v1 = i[:2]
        return {0: 32 * rows, 2: 128 * rows, 3: 32 * 32 * 4, 4: 128 * rows, **({} if v1 else {1: 2 * rows})}
    if name == "hash_msg":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: i[0] * i[1]}
    if name == "hash_leaf":
        return {0: i[0] * i[1], 1: (i[0] // 64) * (i[1] // 64) * 32}
    if name == "tickets":
        return {0: 4 * i[0] * i[1], 1: (i[0] // 16) * (i[1] // 16) * 32, 2: 32}
    if name.startswith("hash_rows"):
        seg = 1280 if name.endswith("fp10") else 1024
        assert i[1] % seg == 0
        return {0: i[0] * i[1], 1: i[0] * (i[1] // seg) * 32}
    if name == "plainq":
        return {0: i[0] * i[1] * (4 if i[2] else 2), 1: i[0] * i[1], 2: 4 * i[0]}
    if name == "open_rows":
        return {0: i[0] * (i[1] + i[1] // 4), 1: 2 * i[0] * i[1]}
    if name == "gen_fp10":
        return {0: i[0] * (i[1] + i[1] // 4)}
    if name == "gen_f32":
        return {0: 4 * i[0] * i[1]}
    if name == "diff_words":
        return {0: 4 * i[0], 1: 4 * i[0], 2: 8}
    if name.endswith("_smem"):
        return {0: 4}
    raise AssertionError(f"no footprint for {name}")


def threads(name: str, i: list[int]) -> int:
    """Threads the launch must cover."""
    return {"lines": lambda: i[2], "norms": lambda: 32 * i[0], "stats_s5": lambda: 32 * i[0], "eprime": lambda: i[0],
            "pa": lambda: 64 * i[0], "pb": lambda: 32 * i[0], "hash_msg": lambda: i[0] * i[1] // 32,
            "hash_leaf": lambda: (i[0] // 64) * (i[1] // 64), "tickets": lambda: (i[0] // 16) * (i[1] // 16),
            "open_rows": lambda: i[0] * i[1] // 8, "gen_fp10": lambda: i[0] * (i[1] + i[1] // 4) // 4,
            "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0]}.get(name, lambda: 0)()


@pytest.mark.slow
def test_runner_dry_run(small_fixture, tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the libcuda stand-in")
    lib = tmp_path / "fake_cuda.so"
    subprocess.run([cc, "-O1", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(lib), str(HERE / "fake_cuda.c")], check=True)
    ship = tmp_path / "ship"
    ship.mkdir()
    shutil.copy(HERE / "run.py", ship)
    (ship / "pearl_c.cubin").write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    log = tmp_path / "launches.jsonl"
    env = dict(os.environ, PEARLC_LIBCUDA=str(lib), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), SHAPES="1024", REPS="2", PEARLC_UNGATED="1")
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:2] == ["env", "check"] and kinds.count("timing") == 2
    check = rows[1]["formings"]
    assert set(check) == {"v0", "v1"} and all(not v["pearl_c_ok"] and v["failed"] for v in check.values())
    for t in rows[2:]:
        assert t["gates"]["ungated"] and {"split_ms", "unit_e2e_ms", "ratio_vs_plain", "consistency", "job_ms"} <= set(t)
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
                if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3), ("lines", 4)}:
                    continue
                assert p[slot] in allocs, (name, slot)
                assert allocs[p[slot]] >= n, (name, slot, allocs[p[slot]], n)
            g = ev["grid"]
            assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
            if name in ("form", "form_s5"):
                assert g[:2] == [i[1] // 128, i[0] // 64]
            if name == "plainq":
                assert g[0] == i[0]
            if name.startswith("gemm_") and not name.endswith("_smem"):
                bn = 32 if name == "gemm_words" else 128
                assert ev["block"] == 384 and ev["smem"] > 48 * 1024 and g[0] <= (i[0] // 128) * (i[1] // bn)
    assert seen == set(kernel_names())
