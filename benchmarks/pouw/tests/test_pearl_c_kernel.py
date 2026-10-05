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

SOURCE = (HERE / "pearl_c.cu").read_text() + (HERE / "hash.cuh").read_text()


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


def test_form_s5_stores_code_pairs_at_the_tiled_offsets():
    """form_s5 casts codes i and i + 1 of a thread's 64 into one 16-bit word at a per-thread base plus a constant: for every
    thread and pair, that offset is the tiled layout's for element i (row r, column t) and i + 1 sits at t + 1."""
    body = re.search(r'form_s5\(const Args a\) \{(.*?)\n\}', SOURCE, re.S).group(1)
    assert "e4m3x2(d[i], d[i + 1])" in body and "e4m3(d[i])" not in body
    base_src = re.search(r"const int base = ([^;]+);", body).group(1)
    off_src = re.search(r"uint16_t\*>\(sO \+ base \+ (.*)\) =$", body, re.M).group(1)
    for tid in range(128):
        rA, cA = 16 * (tid // 32) + (tid % 32) // 4, 2 * (tid % 4)
        base = eval(base_src, {"rA": rA, "cA": cA})
        for i in range(0, 64, 2):
            q, m = (i // 2) % 2, i // 4
            off = base + eval(off_src.replace("/", "//"), {"q": q, "m": m})
            r, t = rA + 8 * ((i // 2) % 2), 8 * (i // 4) + cA + i % 2
            want = (t // 32) * 2048 + (r >> 3) * 256 + ((t & 31) >> 4) * 128 + (r & 7) * 16 + (t & 15)
            assert off == want and off % 2 == 0 and (t + 1) & 15 == (t & 15) + 1


def test_kernel_constants_match_the_reference():
    from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
    from verity.protocols.accounting.work.pouw.schemes import pearl_kw as P

    consts = dict(re.findall(r"constexpr uint(?:16|32)_t (\w+) = (0x[0-9A-Fa-f]+);", SOURCE))
    dr, dstd = P.delta_consts(P.H100)
    assert int(consts["BF_DELTA_R"], 16) == dr and int(consts["BF_DELTA_STD"], 16) == dstd
    assert int(consts["BF_QMAX"], 16) == P.f32_to_bf16(P.Q_MAX) and int(consts["BF_FLOOR"], 16) == P.NORM_FLOOR_BF16
    assert int(consts["S5_LINE"], 16) == C.S5_LINE
    assert "0x1p-32f" in SOURCE and C.S5_FLOOR == P.f32_bits(2.0 ** -32)
    run = (HERE / "run.py").read_text()
    sys.path.insert(0, str(HERE))
    import run as R

    assert {k: bytes.fromhex(v) for k, v in R.H1_KEYS.items()} == {"k_msg": C.H1_KEYS["message"], "k_leaf": C.H1_KEYS["leaf"]}
    from verity.primitives.crypto.blake3 import blake3

    assert bytes.fromhex(R.SEED_KEY) == blake3(C.LABELS["seed-A"])
    n256, n16 = re.search(r"NORM_256, NORM_16 = (0x[0-9A-F]+), (0x[0-9A-F]+)", run).groups()
    assert int(n256, 16) == P.f32_to_bf16(P.NOISE_NORM * 32) and int(n16, 16) == P.f32_to_bf16(C.LINE_NORM_V1 * 32)
    from verity.protocols.accounting.work.pouw.audit import ROW_SCHEMA, TILE_SCHEMA, rows_domain

    for name, schema in (("ROW_SCHEMA", ROW_SCHEMA), ("TILE_SCHEMA", TILE_SCHEMA)):
        chars = re.search(rf"__constant__ uint8_t {name}\[(\d+)\] = \{{([^}}]*)\}}", SOURCE)
        assert "".join(c.strip()[1] for c in chars.group(2).split(",")) == schema and int(chars.group(1)) == len(schema)
    prefix = sum(int(x) for x in re.search(r"constexpr int ROW_PREFIX = ([\d +]+),", SOURCE).group(1).split("+"))
    assert prefix == len(rows_domain(bytes(32), 0, 1, "blake3").leaf_prefix(0, 0, ROW_SCHEMA, 0))


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
        assert spec["crosscheck"]["agree"] and spec["scheme"] == f"pearl-c-h100-{forming}-h1" and man["hashing"] == "h1"
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
        return {0: 4 * i[0] * i[1], 1: 4 * i[0] * i[1], 2: i[0] * i[1], 3: 32}
    if name == "hash_leaf":
        return {0: i[0] * i[1], 1: -(-i[0] // 64) * (i[1] // 64) * 32, 2: 32}
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
    if name == "b3_seeds":
        return {0: 32 * (i[0] if i[2] else 1), 1: 32 * i[0], 2: 32, 3: 32, 4: 32}
    if name == "seed_subkeys":
        return {0: 32, 1: 64}
    if name == "tickets":
        return {0: 4 * i[0] * i[1], 1: (i[0] // 16) * (i[1] // 16) * 32, 2: 32}
    if name == "quant_s5":
        return {0: 4 * i[0] * i[1], 1: 4 * i[0], 2: i[0] * i[1]}
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
    if name == "ieee_ops":
        return {0: 4 * i[0], 1: 4 * i[0], 2: 24 * i[0]}
    if name == "ieee_q":
        return {0: 4 * i[0], 1: 4 * i[0]}
    if name.endswith("_smem"):
        return {0: 4}
    raise AssertionError(f"no footprint for {name}")


def within(allocs: dict[int, int], ptr: int, size: int) -> bool:
    """Whether [ptr, ptr + size) lies inside one live allocation (a tree's level is a pointer into its buffer)."""
    return any(base <= ptr and ptr + size <= base + n for base, n in allocs.items())


def threads(name: str, i: list[int]) -> int:
    """Threads the launch must cover."""
    return {"lines": lambda: i[2], "norms": lambda: 32 * i[0], "stats_s5": lambda: 32 * i[0], "eprime": lambda: i[0],
            "pa": lambda: 64 * i[0], "pb": lambda: 32 * i[0], "hash_msg": lambda: i[0] * i[1] // 32, "hash_rows": lambda: i[0],
            "hash_pad": lambda: i[1] - i[0], "hash_node_keys": lambda: i[0], "hash_nodes": lambda: i[1],
            "hash_tree_leaves": lambda: i[0], "hash_leaf": lambda: -(-i[0] // 64) * (i[1] // 64),
            "tickets": lambda: (i[0] // 16) * (i[1] // 16),
            "open_rows": lambda: i[0] * i[1] // 8, "gen_fp10": lambda: i[0] * (i[1] + i[1] // 4) // 4,
            "gen_f32": lambda: i[0] * i[1], "diff_words": lambda: i[0], "quant_s5": lambda: i[0] * i[1] // 16,
            "ieee_ops": lambda: i[0], "ieee_q": lambda: i[0], "b3_seeds": lambda: i[0], "seed_subkeys": lambda: 2}.get(name, lambda: 0)()


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
               FAKE_KERNELS=",".join(kernel_names()), SHAPES="1152", REPS="2", PEARLC_UNGATED="1", HASH_STREAM="1")
    r = subprocess.run([sys.executable, str(ship / "run.py"), str(small_fixture / "check.json")], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    kinds = [x["kind"] for x in rows]
    assert kinds[:2] == ["env", "check"] and kinds.count("timing") == 2
    check = rows[1]["formings"]
    assert set(check) == {"v0", "v1"} and all(not v["pearl_c_ok"] and v["failed"] for v in check.values())
    for t in rows[2:]:
        assert t["gates"]["ungated"] and {"split_ms", "unit_e2e_ms", "ratio_vs_plain", "consistency", "job_ms"} <= set(t)
        assert set(t["split_ms"]) == {"commit", "forming", "gemm", "cleanup", "hashing"} and t["hash_stream"]

    allocs: dict[int, int] = {}
    seen, side, streams, waits = set(), set(), {}, set()
    for ev in map(json.loads, log.read_text().splitlines()):
        if ev["op"] == "alloc":
            allocs[ev["ptr"]] = ev["bytes"]
        elif ev["op"] == "free":
            allocs.pop(ev["ptr"])
        elif ev["op"] == "stream":
            side.add(ev["ptr"])
        elif ev["op"] == "wait":
            waits.add(ev["stream"])
        else:
            name, p, i = ev["name"], ev["p"], ev["i"]
            seen.add(name)
            streams.setdefault(name, set()).add(ev["stream"])
            for slot, n in needs(name, p, i).items():
                if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3), ("lines", 4)}:
                    continue
                assert within(allocs, p[slot], n), (name, slot, n)
            g = ev["grid"]
            assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
            if name in ("form", "form_s5"):
                assert g[:2] == [i[1] // 128, i[0] // 64]
            if name in ("plainq", "hash_rows"):
                assert g[0] == i[0]
            if name == "lines":
                assert i[3] in (0x4600, 0x4400)
            if name.startswith("gemm_") and not name.endswith("_smem"):
                bn = 32 if name == "gemm_words" else 128
                assert ev["block"] == 384 and ev["smem"] > 48 * 1024 and g[0] <= (i[0] // 128) * (i[1] // bn)
    assert seen == set(kernel_names())
    # the check runs on the default stream; each timing row's post-GEMM hashing on its own side stream, fenced both ways, and
    # A's commitment and seed_A on the default stream, before the forming that reads them
    hashing = {"hash_pad", "hash_node_keys", "hash_nodes", "hash_msg", "hash_leaf", "hash_tree_leaves", "tickets"}
    assert len(side) == 2 and waits == side | {0}
    for name, ss in streams.items():
        assert side <= ss <= {0} | side if name in hashing else ss == {0}, (name, ss)
