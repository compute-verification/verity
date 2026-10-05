"""Pearl-C's fixed-shape bench off the pod (benchmarks/pouw/pearl_c: twin.py, bench.py): `gen_f32` against its CUDA source,
the twin against `pearl_c`'s scalar reference and the fixture's buffers, the sampled tiles' offsets in the kernels' layout,
γ against the lane's pinned values, bench.json against the twin, and bench.py's dry run on the CPU stand-in for libcuda."""

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

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
sys.path.insert(0, str(HERE))


def _load(name: str, path: Path):
    """The module at `path` under its own name: the harness has a bench.py too, and the suite imports both in one process."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("pearl_c_bench", HERE / "bench.py")
import fixture as F  # noqa: E402
import twin as W  # noqa: E402
from test_pearl_c_kernel import kernel_names, needs, small_fixture, threads, within  # noqa: E402, F401
from verity_pouw.audit import ROW_SCHEMA, activation_binding, commit_rows, rows_domain  # noqa: E402
from verity_pouw.protocol import Context, Matmul, Tile  # noqa: E402
from verity_pouw.protocol import Shape as MShape  # noqa: E402
from verity_pouw.schemes import pearl_c as C  # noqa: E402
from verity_pouw.schemes import pearl_kw as P  # noqa: E402

SOURCE = (HERE / "pearl_c.cu").read_text() + (HERE / "hash.cuh").read_text()
SMALL = W.Shape("small", 96, 128, 256)          # two tile rows, the second half filler


def test_gen_f32_is_the_kernels(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    mix = re.search(r"DEV uint32_t mix32\(uint32_t x\) \{.*?\n\}", SOURCE, re.S).group(0)
    body = re.search(r'extern "C" __global__ void gen_f32\(const Args a\) \{(.*?)\n\}', SOURCE, re.S).group(1)
    body, casts = re.subn(r"reinterpret_cast<([^>]+)>\(([^()]*)\)", r"((\1)(\2))", body)
    assert casts == 1
    (tmp_path / "gen.c").write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define DEV static\n"
        "typedef struct { uint64_t p[16]; int32_t i[16]; } Args;\nstatic struct { unsigned x; } blockIdx, threadIdx, blockDim;\n"
        "static float __fmul_rn(float a, float b) { volatile float r = a * b; return r; }\n"
        "static float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }\n"
        "static uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }\n"
        f"{mix}\nstatic void gen_f32(const Args a) {{{body}\n}}\n"
        "int main(int argc, char** v) {\n  (void)argc;\n  int rows = atoi(v[1]), k = atoi(v[2]);\n"
        "  uint32_t* out = (uint32_t*)calloc((size_t)rows * k, 4);\n  Args a = {{0}, {0}};\n  a.p[0] = (uint64_t)(uintptr_t)out;\n"
        "  a.i[0] = rows; a.i[1] = k; a.i[2] = atoi(v[3]); blockDim.x = 256;\n"
        "  for (unsigned b = 0; b < ((unsigned)rows * k + 255) / 256; b++)\n"
        "    for (unsigned t = 0; t < 256; t++) { blockIdx.x = b; threadIdx.x = t; gen_f32(a); }\n"
        "  fwrite(out, 4, (size_t)rows * k, stdout);\n}\n")
    subprocess.run([cc, "-O1", "-ffp-contract=off", "-o", str(tmp_path / "gen"), str(tmp_path / "gen.c")], check=True)
    for rows, k, seed in ((3, 256, 11), (5, 3584, W.seeds(SMALL)["gen_a"]), (2, 18944, 2**31 - 1)):
        raw = subprocess.run([str(tmp_path / "gen"), str(rows), str(k), str(seed)], check=True, capture_output=True).stdout
        want = np.frombuffer(raw, "<u4").reshape(rows, k)
        assert (W.gen_f32(np.arange(rows), k, seed) == want).all()
        assert (W.gen_f32(np.array([rows - 1]), k, seed)[0] == want[-1]).all()


def test_fp8_cast_is_pearls():
    rng = np.random.default_rng(7)
    edges = [0, 0x80000000, P.f32_bits(448.0), P.f32_bits(464.0), P.f32_bits(-500.0), P.f32_bits(2.0 ** -6),
             P.f32_bits(2.0 ** -9), P.f32_bits(1.5 * 2.0 ** -10), P.f32_bits(2.0 ** -10), P.f32_bits(1.0625), P.f32_bits(1.1875)]
    words = np.concatenate([rng.integers(0, 2**32, 5000, dtype=np.uint64).astype(np.uint32),
                            rng.uniform(-600, 600, 10000).astype(np.float32).view(np.uint32),
                            (rng.uniform(-1, 1, 5000) * 2.0 ** -6).astype(np.float32).view(np.uint32),
                            np.array(edges, np.uint32)])
    words = words[((words >> 23) & 0xFF) != 0xFF]
    assert W.fp8(words).tolist() == [P.f32_to_fp8(P.f32_of_bits(int(w))) for w in words]
    with pytest.raises(P.Reject):
        W.fp8(np.array([0x7F800000], np.uint32))


@pytest.fixture(scope="module")
def small_tiles():
    jb = W.job(SMALL)
    return jb, {(r0, c0): W.tile(jb, r0, c0) for r0 in (0, 64) for c0 in (0, 64)}


def test_twin_tile_is_the_scheme(small_tiles):
    """Two tiles, one all real rows and one half filler, through `pearl_c` itself (hashing -h1); the baseline through the
    fixture's functions; row r0's leaf as `verity_pouw.audit` commits A's rows."""
    jb, tiles = small_tiles
    s = C.PearlC(k_min=32, tile=(64, 64), forming="v1", hashing="h1")
    noise = C.Noise(jb.seeds["seed_a"], jb.seeds["seed_b"])
    ctx = Context(bytes(32), Matmul(0, MShape(SMALL.m, SMALL.k, SMALL.n), 0), bytes(32), bytes(32))
    for (r0, c0) in ((0, 0), (64, 64)):
        t = tiles[(r0, c0)]
        assert (t.a.words[SMALL.m_real - r0:] == 0).all() if r0 else (t.a.words != 0).any(axis=1).all()
        ra, rb = [r.astype("<u4").tobytes() for r in t.a.words], [r.astype("<u4").tobytes() for r in t.b.words]
        ops = s.encode(ctx, noise, ra, rb, Tile(tuple(range(r0, r0 + 64)), tuple(range(c0, c0 + 64))))
        checked = s.checked(ctx, ops)
        assert ops.a.codes == t.a.codes.tolist() and ops.b.codes == t.b.codes.tolist()
        assert ops.a.alpha == t.a.alpha.tolist() and ops.b.alpha == t.b.alpha.tolist()
        assert checked.c == t.c.tolist() and checked.u == t.u.tolist()
        assert s.digest(ctx, noise, checked) == t.leaf and s.tickets(noise, checked) == t.tickets
        vals = [t.a.words.view(np.float32), t.b.words.view(np.float32)]
        qa, qb = F.adopted(vals[0], t.a.alpha.tolist()), F.adopted(vals[1], t.b.alpha.tolist())
        assert (qa == t.a.quant).all() and (qb == t.b.quant).all()
        assert (F.glue_y(F.product(qa, qb), F.recip(t.a.alpha), F.recip(t.b.alpha)) == t.y_adopt).all()
        assert t.row_leaf == rows_domain(activation_binding(jb.seeds["salt"], 0), 0, SMALL.m, "blake3").leaf(r0, r0, ROW_SCHEMA, ra[0])


def test_samples_read_the_kernels_layout(small_tiles):
    """The full buffers laid out as the fixture (and so the kernels) lay them out, assembled from the twin's tiles; bench.py's
    offsets read each tile back, and a changed byte names its buffer."""
    jb, tiles = small_tiles
    m, n, k = SMALL.m, SMALL.n, SMALL.k
    c, u, y = (np.zeros((m, n), dt) for dt in (np.uint32, np.uint32, np.uint16))
    alpha_a, alpha_b = np.zeros(m, np.uint32), np.zeros(n, np.uint32)
    for (r0, c0), t in tiles.items():
        c[r0:r0 + 64, c0:c0 + 64], u[r0:r0 + 64, c0:c0 + 64], y[r0:r0 + 64, c0:c0 + 64] = t.c, t.u, t.y_adopt
        alpha_a[r0:r0 + 64], alpha_b[c0:c0 + 64] = t.a.alpha, t.b.alpha
    leaves = F.leaves(F.digests(c, u), m, n)
    assert leaves == [tiles[(r0, c0)].leaf for r0 in (0, 64) for c0 in (0, 64)]
    tickets = F.tickets(c, jb.seeds["seed_a"])
    rows = [r.astype("<u4").tobytes() for r in W.a_words(SMALL, jb.seeds, np.arange(m))]
    tree_a = F.tree_bytes(W.a_domain(SMALL, jb.seeds), rows, ROW_SCHEMA)
    assert tree_a[-32:] == commit_rows(activation_binding(jb.seeds["salt"], 0), 0, rows, "blake3").commitment.root
    bufs = {"c": c.astype("<u4").tobytes(), "u": u.astype("<u4").tobytes(), "y_adopt": y.astype("<u2").tobytes(),
            "leaves": b"".join(leaves), "tickets": b"".join(tickets), "alpha_a": alpha_a.astype("<u4").tobytes(),
            "alpha_b": alpha_b.astype("<u4").tobytes(), "tree_a": tree_a}

    def read(name, off, size):
        assert 0 <= off and off + size <= len(bufs[name]), (name, off, size)
        return bufs[name][off:off + size]

    for (r0, c0), t in tiles.items():
        assert B.compare_tile(B.tile_bytes(read, n, k, r0, c0), t.expected()) == []
    raw = bytearray(bufs["u"])
    raw[4 * (70 * n + 100)] ^= 1
    bufs["u"] = bytes(raw)
    assert B.compare_tile(B.tile_bytes(read, n, k, 64, 64), tiles[(64, 64)].expected()) == ["u"]
    assert B.compare_tile(B.tile_bytes(read, n, k, 64, 0), tiles[(64, 0)].expected()) == []


def test_gamma_reproduces_the_lanes_pins():
    """rev1's pinned values at the cap, `DeviceRev1Gamma.lean`'s `pearlCGammaH100Rev1_8192` and `_16384`: 140237/26090000
    (0.538%) and 89359/17230000, and 1.149% chain-only at 8,192³; 188/65225 and 116/43075 at a zero debit and 587/65225
    chain-only (`Gamma.lean`'s 377/130850, 697/258850 and 47/5234 before rev1 took the first promotion add out).  The
    constants are pearl_c's W_ref terms."""
    assert B.gamma(8192, 8192)["cap"] == Fraction(140237, 26090000)
    assert B.gamma(16384, 16384)["cap"] == Fraction(89359, 17230000)
    assert B.gamma(8192, 8192)["floor"] == Fraction(188, 65225)
    assert B.gamma(16384, 16384)["floor"] == Fraction(116, 43075)
    assert B.gamma(8192, 8192)["floor_chain_only"] == Fraction(587, 65225)
    g = B.gamma(8192, 8192)
    assert round(float(g["cap"]) * 100, 3) == 0.538 and round(float(g["cap_chain_only"]) * 100, 3) == 1.149
    h100 = C.H100_PRICES
    assert (B.FORMING_CREDITED, B.FORMING_A_ONLY) == (h100.forming_credited, h100.forming_a_only) and B.RANK == C.R
    assert B.PEEL_UNITS == h100.bf16_mac * C.PEEL_K and B.FADD_PER_MAC == Fraction(h100.fadd, C.WINDOW)
    assert B.WINDOW == C.WINDOW
    assert B.T == C.TILE == W.T


def test_run_takes_whole_promotion_groups():
    import run as R

    for m, n, k, forming in ((128, 128, 192, "v0"), (128, 128, 64, "v1"), (96, 128, 256, "v1"), (128, 64, 256, "v1")):
        with pytest.raises(ValueError):
            R.Pipeline(None, m, n, k, forming, {}, 0, 0)


def test_bench_json_is_the_twins():
    """Every shape, key, seed and sampled tile of the committed bench.json is the twin's (the bytes: the slow test)."""
    man = json.loads((HERE / "bench.json").read_text())
    assert man["schema"] == W.SCHEMA and man["scheme"] == C.NAME_V1 + "-h1" and man["tile"] == W.T and man["hashing"] == "h1"
    assert [s["label"] for s in man["shapes"]] == [s.label for s in W.SHAPES]
    for spec, shape in zip(man["shapes"], W.SHAPES):
        want = W.entry(shape, [])
        assert {k: v for k, v in spec.items() if k != "samples"} == {k: v for k, v in want.items() if k != "samples"}
        assert [tuple(s["tile"]) for s in spec["samples"]] == W.samples(shape)
        assert shape.m % 128 == 0 and shape.n % 128 == 0 and shape.k % 256 == 0 and shape.m_real <= shape.m


@pytest.mark.slow
@pytest.mark.parametrize("label", ["w1-2048", "qwen7b-o-decode"])
def test_bench_json_bytes_are_the_twins(label):
    shape = next(s for s in W.SHAPES if s.label == label)
    spec = next(s for s in json.loads((HERE / "bench.json").read_text())["shapes"] if s["label"] == label)
    jb = W.job(shape)
    assert [W.tile(jb, r0, c0).expected() for r0, c0 in W.samples(shape)] == spec["samples"]


@pytest.mark.slow
def test_bench_dry_run(small_fixture, tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the libcuda stand-in")
    lib = tmp_path / "fake_cuda.so"
    subprocess.run([cc, "-O1", "-Wall", "-Werror", "-shared", "-fPIC", "-o", str(lib), str(HERE / "fake_cuda.c")], check=True)
    ship = tmp_path / "ship"
    ship.mkdir()
    for f in ("run.py", "bench.py"):
        shutil.copy(HERE / f, ship)
    (ship / "pearl_c.cubin").write_bytes(b"\x7fELF" + bytes(14) + bytes([190, 0]) + bytes(44))
    small = W.build([SMALL, W.Shape("small-decode", 32, 128, 512)], processes=2)
    (ship / "bench.json").write_text(json.dumps(small))
    log = tmp_path / "launches.jsonl"
    env = dict(os.environ, PEARLC_LIBCUDA=str(lib), PEARLC_LIBNVML="/nonexistent", FAKE_LOG=str(log),
               FAKE_KERNELS=",".join(kernel_names()), REPS="2", PEARLC_UNGATED="1")
    r = subprocess.run([sys.executable, str(ship / "bench.py"), str(ship / "bench.json"), str(small_fixture / "check.json")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 4, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["kind"] for x in rows] == ["env", "check", "bench", "bench"]
    for row, shape in zip(rows[2:], (SMALL, W.Shape("small-decode", 32, 128, 512))):
        assert row["shape"] == [shape.m, shape.n, shape.k] and row["m_real"] == shape.m_real and not row["tiles_ok"]
        assert row["timing"]["gates"]["ungated"] and {"prover_x", "with_hashing_x", "hashing_ms", "per_call_b_x"} <= set(row["summary"])
        assert all(set(t["failed"]) == {"leaf", "c", "u", "tickets", "y_adopt", "alpha_a", "alpha_b", "row_leaf"} for t in row["tiles"])
    allocs: dict[int, int] = {}
    gens = []
    for ev in map(json.loads, log.read_text().splitlines()):
        if ev["op"] == "alloc":
            allocs[ev["ptr"]] = ev["bytes"]
        elif ev["op"] == "free":
            allocs.pop(ev["ptr"])
        elif ev["op"] == "launch":
            name, p, i = ev["name"], ev["p"], ev["i"]
            for slot, size in needs(name, p, i).items():
                if not p[slot] and (name, slot) in {("lines", 2), ("lines", 3), ("lines", 4)}:
                    continue
                assert within(allocs, p[slot], size), (name, slot)
            g = ev["grid"]
            assert g[0] * g[1] * g[2] * ev["block"] >= threads(name, i), name
            if name == "gen_f32":
                gens.append(i[:3])
    assert gens == [[96, 256, W.seeds(SMALL)["gen_a"]], [128, 256, W.seeds(SMALL)["gen_b"]],
                    [32, 512, W.seeds(W.Shape("small-decode", 32, 128, 512))["gen_a"]],
                    [128, 512, W.seeds(W.Shape("small-decode", 32, 128, 512))["gen_b"]]]
