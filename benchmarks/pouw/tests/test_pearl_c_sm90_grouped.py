"""Pearl-C's grouped call on sm_90a off the card (benchmarks/pouw/pearl_c_sm90/grouped.py, and the vLLM call over it,
`pouw_pearl_c_grouped.gemm_grouped`), on a recording stand-in for the device: each expert's rows cut into units of at most
pd_commit's 256 rows; each unit's record in the call's table is the launch `run.Pipeline` makes for that unit's rows called
alone, at the unit's offsets; a call is eight launches whatever its experts; and the vLLM call's matmuls, tiles, keys, screen
and retained records are those of each unit called alone through `PearlCDeviceRun.gemm`, in unit order."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.audit import weights_root  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.pouw.protocol import Matmul, Shape  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PPD  # noqa: E402
from verity_vllm.protocols.pouw.pearl_c import grouped as PG  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm90"
SCHEME = "pearl-c-h100-v1-h3"
SALT_KEY = bytes(range(100, 132))
STEPS = ["pd_commit_g_851", "stats_s5", "eprime", "form_s5_g", "gemm_words_g_d", "pa", "gemm_pearlc_g_d", "pd_tiles_g_851"]


def load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = load("pearl_c_sm90_run", HERE / "run.py")
F = load("pearl_c_sm90_fixture", HERE / "fixture.py")
G = load("pearl_c_sm90_grouped", HERE / "grouped.py")


class RecordingDev:
    """What `run.Pipeline`, `grouped.Grouped` and the run touch of `run.Dev`, recording each launch (name, grid, block, dynamic
    shared memory, stream, pointers, ints); its attribute reads get an H100's opt-in shared memory and no static shared memory."""

    sms, dev = 132, 0

    def __init__(self):
        self.stream, self.log, self.n = None, [], 0

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        self.log.append((name, grid, block, smem, self.stream if stream is None else stream, tuple(p), tuple(i)))

    def ck(self, *_):
        pass

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 232448, "cuFuncGetAttribute": 0}[f]

    def fn(self, name):
        return name

    def alloc(self, n):
        self.n += 1 << 20
        return self.n

    def free(self, p):
        pass

    def sync(self):
        pass

    def read(self, p, n):
        return bytes(n)

    def tmap(self, ptr, depth, rows, box_rows=128):
        return bytes(128)


def launches(dev) -> list:
    return [x for x in dev.log if not x[0].endswith("_smem")]


def test_units_cut_each_experts_rows_into_blocks():
    us = G.units([3, 0, 300, 64, 17], 128, 256)
    assert [(u.pos, u.src, u.m, u.m16, u.row, u.seg, u.tile, u.tiles) for u in us] == [
        (0, 0, 3, 16, 0, 64, 0, 2), (2, 3, 256, 256, 64, 256, 2, 8), (2, 259, 44, 48, 320, 64, 10, 2),
        (3, 303, 64, 64, 384, 64, 12, 2), (4, 367, 17, 32, 448, 64, 14, 2)]
    assert G.units([0, 0], 128, 256) == []


def grouped(n=128, k=1024, experts=4, rows=1024, count=16):
    dev = RecordingDev()
    sizes = {**G.sizes(R, rows, count, n, k), **{x: 1 << 30 for x in G.STACKED}}
    b = {x: (j + 1) << 32 for j, x in enumerate(sorted(sizes))}
    keys = {"salt_key": 7 << 40, "hm": (7 << 40) + 256}
    return G.Grouped(R, dev, n, k, experts, rows, count, b, keys), dev, b, keys


def test_each_units_record_is_its_own_calls_launches():
    """Unit j's pd_commit and pd_tiles Args are what `run.Pipeline` passes for its m₁₆ rows of its segment called alone, its
    buffers at the unit's offsets (A's rows, row digests, seeds and lines at its first row, its tree and counter at its slot,
    C~ and U at its rows, its leaves at its first tile), its keys at its record; the table's expert map, gathered rows and
    root nodes are the units'."""
    n, k = 128, 1024
    g, dev, b, kk = grouped(n, k)
    us = G.units([3, 0, 300, 64], n, R.PD_MAX_ROWS)
    keys = np.random.default_rng(1).bytes(G.KEY_BYTES * len(us))
    leaves = 9 << 40
    meta = g.meta(us, keys, leaves)
    rec = np.frombuffer(meta[:G.REC * len(us)], np.uint8).reshape(len(us), G.REC)
    commit = rec[:, :G.TILES_AT].copy().view(G.ARGS).ravel()
    tiles = rec[:, G.TILES_AT:G.KEYS_AT].copy().view(G.ARGS).ravel()
    for j, u in enumerate(us):
        key = b["meta"] + G.REC * j + G.KEYS_AT
        bufs = {"rowdig_a": b["rowdig_a"] + 64 * u.row, "tree_a": b["tree_a"] + G.tree_bytes(R) * j,
                "seeds_a": b["seeds_a"] + 32 * u.row, "lkeys_a": b["lkeys_a"] + 32 * u.row, "pd_count": b["pd_count"] + 16 * j,
                "seed_a": b["seed_a"] + 32 * j, "line_a": b["line_a"] + 32 * j, "ea_codes": b["ea_codes"] + 32 * u.row,
                "c": b["c"] + 4 * n * u.row, "u": b["u"] + 4 * n * u.row, "leaves": leaves + 64 * u.tile}
        full = {"dom_a": key, "unit": key + 64, "dom_b": 0, "dom_t": 0, "line_b": 0, **kk}
        pipe = R.Pipeline(dev, u.seg, n, k, "h100", full, b["xa"] + 4 * k * u.row, 0, bufs=bufs, m_real=u.m16)
        assert pipe.pd == R.pd_variant(u.m16, n, k)
        dev.log.clear()
        pipe.pd_commit()
        pipe.pd_tiles()
        (_, _, _, _, _, cp, ci), (_, _, _, _, _, tp, ti) = dev.log
        assert tuple(commit["p"][j][:len(cp)]) == cp and tuple(commit["i"][j][:len(ci)]) == ci, j
        assert tuple(tiles["p"][j][:len(tp)]) == tp and tuple(tiles["i"][j][:len(ti)]) == ti, j
        assert not commit["p"][j][len(cp):].any() and not tiles["p"][j][len(tp):].any()
        assert rec[j, G.KEYS_AT:G.KEYS_AT + G.KEY_BYTES].tobytes() == keys[G.KEY_BYTES * j:G.KEY_BYTES * (j + 1)]
    blocks = sum(u.seg for u in us) // G.SEG
    ex = np.frombuffer(meta, "<i4", blocks, G.REC * len(us))
    assert ex.tolist() == [0] + [2] * 4 + [2, 3] and G.Grouped.rows_at(us) == G.REC * len(us) + 4 * blocks + 4
    at = G.Grouped.rows_at(us)
    rows = np.frombuffer(meta, "<i8", 367, at)
    assert rows.tolist() == [*range(3), *range(64, 320), *range(320, 364), *range(384, 448)]
    nodes = np.frombuffer(meta, "<i8", len(us), at + 8 * 367)
    assert nodes.tolist() == [(G.tree_bytes(R) * j) // 64 + 2 * R.pow2(u.m16) - 2 for j, u in enumerate(us)]
    assert len(meta) == at + 8 * (367 + len(us)) <= G.sizes(R, 1024, 16, n, k)["meta"]
    with pytest.raises(ValueError, match="past the call's"):
        g.meta(G.units([1100], n, R.PD_MAX_ROWS), bytes(G.KEY_BYTES * 5), leaves)


@pytest.mark.parametrize("counts", [[40], [3, 0, 300, 64, 17, 1]])
def test_a_call_is_eight_launches_whatever_its_experts(counts):
    """One launch a step, in `grouped.STEPS`' order: pd_commit_g on (max m₁₆, units) CTAs and pd_tiles_g on (max tiles,
    units), each reading its records from the table; stats, E', P_A over all of A's rows; form_s5_g and both GEMMs with the
    expert map, the GEMMs persistent at the H100's 132 SMs on A's rows and their map set."""
    n, k = 128, 1024
    g, dev, b, _ = grouped(n, k, experts=len(counts))
    us = G.units(counts, n, R.PD_MAX_ROWS)
    rows, ex = sum(u.seg for u in us), b["meta"] + G.REC * len(us)
    dev.log.clear()
    for _, f in g.steps(us):
        f()
    got = launches(dev)
    assert [x[0] for x in got] == STEPS
    by = {x[0]: x for x in got}
    assert by["pd_commit_g_851"][1:4] == ((max(u.m16 for u in us), len(us)), 32 * 4, dev.pd_smem_g)
    assert by["pd_commit_g_851"][5:] == ((b["meta"],), (G.REC,))
    assert by["pd_tiles_g_851"][1:3] == ((max(u.tiles for u in us), len(us)), 32)
    assert by["pd_tiles_g_851"][5:] == ((b["meta"] + G.TILES_AT,), (G.REC,))
    assert by["stats_s5"][6] == (rows, k) and by["eprime"][6] == (rows,) and by["pa"][6] == (rows,)
    assert by["form_s5_g"][1] == (k // 128, rows // 64) and by["form_s5_g"][5][3:] == (b["fa_codes"], b["at"], ex)
    for name, slot, cols in (("gemm_words_g_d", 1, R.WORDS_N), ("gemm_pearlc_g_d", 0, n)):
        _, grid, block, _, _, p, i = by[name]
        assert grid == min((rows // 64) * (cols // R.gemm_tile(name)[1]), 132) and block == R.gemm_threads(name)
        assert p[12] == ex and p[10] == g.tma + R.MAP_SET * slot and i == (rows, cols, k, R.RASTER)
    assert by["gemm_pearlc_g_d"][5][:9] == (b["at"], b["bt"], b["c"], b["u"], b["y"], b["pa_t"], b["pb_t"], b["ia_a"], b["ia_b"])


def device_run(experts: int, n: int = 128, k: int = 1024, retain=None):
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy((rng.standard_normal((n, k)).astype(np.float32) * 0.05)).to(torch.bfloat16)
               for w in range(experts)}
    kernel = PPD.Kernel(R, F, RecordingDev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, SCHEME, salt_key=SALT_KEY)
    ch = Challenger(0, np.random.default_rng(7).bytes)
    coins = ch.commitment, ch.open("salt", weights_root(roots))
    run = PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, *coins, capacity=1 << 12,
                              salt_key=SALT_KEY)
    run.begin(retain=retain)
    return run, weights


def test_the_vllm_call_commits_each_unit_as_its_call_alone(tmp_path):
    """`gemm_grouped` over experts (1, 3, 0) with (3, 300, 20) rows, and each unit's rows called alone through
    `PearlCDeviceRun.gemm` in unit order on another run: the same matmuls, tiles, screened rows, records and retained rows,
    and each unit's keys in the table are the ones its call alone uploads.  The call is eight launches, y is x's rows
    by n, and an expert's dense call after grouping reads its stacked B̃."""
    ws, counts, n, k = (1, 3, 0), [3, 300, 20], 128, 1024
    x = torch.randn(sum(counts), k).to(torch.bfloat16)
    run, weights = device_run(4, n, k, tmp_path / "grouped")
    dev = run.kernel.dev
    dev.log.clear()
    y = PG.gemm_grouped(run, ws, x, counts)
    assert y.shape == (sum(counts), n) and y.dtype == torch.bfloat16
    assert [x[0] for x in launches(dev)] == STEPS
    us = G.units(counts, n, R.PD_MAX_ROWS)
    g = run._groups[ws]
    meta = PPD._bytes(g.b["meta"])
    alone, _ = device_run(4, n, k, tmp_path / "alone")
    for j, u in enumerate(us):
        w = ws[u.pos]
        alone.gemm(w, x[u.src:u.src + u.m], weights[w], None)
        sh = alone.shapes[(u.m16, n, k)]
        assert meta[G.REC * j + G.KEYS_AT:G.REC * j + G.KEYS_AT + G.KEY_BYTES] == PPD._bytes(sh.keys[:G.KEY_BYTES]), j
    a, b = run.cur, alone.cur
    assert a.matmuls == b.matmuls == [Matmul(j, Shape(u.m16, k, n), ws[u.pos]) for j, u in enumerate(us)]
    assert (a.tiles, a.flag_rows) == (b.tiles, b.flag_rows) == (sum(u.tiles for u in us), sum(counts))
    assert a.records == b.records and [f["rows"]["bytes"] for f in a.files] == [f["rows"]["bytes"] for f in b.files]
    for f, h in zip(a.files, b.files, strict=True):
        assert (tmp_path / "grouped" / f["rows"]["path"]).read_bytes() == (tmp_path / "alone" / h["rows"]["path"]).read_bytes()
        assert f["tree"]["bytes"] == h["tree"]["bytes"]
    assert torch.equal(run.flag_buf[:a.flag_rows], alone.flag_buf[:b.flag_rows])
    dev.log.clear()
    run.gemm(3, x[:5], weights[3], None)
    bt = next(p[1] for name, *_, p, _ in dev.log if name == "gemm_pearlc_d")
    assert bt == g.stacked["bt"].data_ptr() + ws.index(3) * n * k


def test_the_vllm_call_refuses_what_it_cant_run():
    run, weights = device_run(3)
    x = torch.zeros(4, 1024).to(torch.bfloat16)
    with pytest.raises(ValueError, match="distinct"):
        PG.gemm_grouped(run, (0, 0), x, [2, 2])
    PG.gemm_grouped(run, (0, 1), x, [1, 3])
    with pytest.raises(ValueError, match="another group"):
        PG.gemm_grouped(run, (1, 2), x, [2, 2])
    with pytest.raises(ValueError, match="Σ counts"):
        PG.gemm_grouped(run, (0, 1), x, [1, 2])
