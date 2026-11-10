"""Pearl-C's grouped call on sm_90a off the card (benchmarks/pouw/pearl_c_sm90/grouped.py, and the vLLM call over it,
`pouw_pearl_c_grouped.gemm_grouped`), on a recording stand-in for the device: each expert's rows cut into units of at most
pd_commit's 256 rows; each unit's record in the call's table is the launch `run.Pipeline` makes for that unit's rows called
alone, at the unit's offsets; a call is eight launches whatever its experts; and the vLLM call's matmuls, tiles, keys, screen
and retained records are those of each unit called alone through `PearlCDeviceRun.gemm`, in unit order."""
from __future__ import annotations

import importlib.util
import json
import struct
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
STEPS = ["pd_commit_gm_851", "stats_s5", "eprime", "form_s5_g", "gemm_words_g_d", "pa", "gemm_pearlc_g_d", "pd_tiles_g_979"]


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
    shared memory, stream, pointers, ints) and each kernel's maximum dynamic shared memory set (``max_dyn``); its attribute
    reads get an H100's opt-in shared memory and no static shared memory."""

    sms, dev = 132, 0

    def __init__(self):
        self.stream, self.log, self.n, self.max_dyn = None, [], 0, {}

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        self.log.append((name, grid, block, smem, self.stream if stream is None else stream, tuple(p), tuple(i)))

    def ck(self, f, *a):
        if f == "cuFuncSetAttribute" and a[1] == R.MAX_DYN:
            self.max_dyn[a[0]] = a[2]

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


def test_rows_order_puts_the_units_rows_apart_on_the_sms_and_filler_rows_past_them():
    """pd_commit_gm's CTAs at decode's 8 experts (77 rows, 176 committed with their filler rows) on 132 SMs: every unit's row
    once; the 77 rows on CTAs below 132, and the 44 CTAs past 132 and the 44 they share SMs with (x and x + 132) filler rows.
    While the rows fit the SMs, the rows come first."""
    us = G.units([3, 1, 17, 1, 40, 8, 2, 5], 6144, R.PD_MAX_ROWS)
    order = G.rows_order(us, 132)
    assert sorted(order) == [(j, r) for j, u in enumerate(us) for r in range(u.m16)] and len(order) == 176
    real = [x for x, (j, r) in enumerate(order) if r < us[j].m]
    assert len(real) == 77 and max(real) < 132 and not {*range(44), *range(132, 176)} & set(real)
    small = G.units([3, 1], 6144, R.PD_MAX_ROWS)
    assert G.rows_order(small, 132) == [(0, 0), (0, 1), (0, 2), (1, 0), *((0, r) for r in range(3, 16)), *((1, r) for r in range(1, 16))]

def grouped(n=128, k=1024, experts=4, rows=1024, count=16):
    dev = RecordingDev()
    sizes = {**G.sizes(R, rows, count, n, k), **{x: 1 << 30 for x in G.STACKED}}
    b = {x: (j + 1) << 32 for j, x in enumerate(sorted(sizes))}
    keys = {"salt_key": 7 << 40, "hm": (7 << 40) + 256}
    return G.Grouped(R, dev, n, k, experts, rows, count, b, keys), dev, b, keys


@pytest.mark.parametrize("split", [False, True])
def test_each_units_record_is_its_own_calls_launches(split):
    """Unit j's pd_commit and pd_tiles Args are what `run.Pipeline` passes for its m₁₆ rows of its segment called alone, its
    buffers at the unit's offsets (A's rows, row digests, seeds and lines at its first row, its tree and counter at its slot,
    C~ and U at its rows, its leaves at its first tile), its keys at its record, and pd_commit_g's zero row digest and the
    unit's rows (where its zero filler rows start) at p14 and i4; the table's expert map, gathered rows and root nodes are the
    units'.  With `split` the record holds pd_commit's Args in mode 1 and pd_tree's, as the pipeline
    passes them with ``split_a`` and no ``fillers`` (no padding rows' E_A lines); without, its pd_tree Args are zero."""
    n, k = 128, 1024
    g, dev, b, kk = grouped(n, k)
    us = G.units([3, 0, 300, 64], n, R.PD_MAX_ROWS)
    keys = np.random.default_rng(1).bytes(G.KEY_BYTES * len(us))
    leaves = 9 << 40
    meta = g.meta(us, keys, leaves, split)
    rec = np.frombuffer(meta[:G.REC * len(us)], np.uint8).reshape(len(us), G.REC)
    commit = rec[:, :G.TILES_AT].copy().view(G.ARGS).ravel()
    tiles = rec[:, G.TILES_AT:G.KEYS_AT].copy().view(G.ARGS).ravel()
    tree = rec[:, G.TREE_AT:].copy().view(G.ARGS).ravel()
    for j, u in enumerate(us):
        key = b["meta"] + G.REC * j + G.KEYS_AT
        bufs = {"rowdig_a": b["rowdig_a"] + 64 * u.row, "tree_a": b["tree_a"] + G.tree_bytes(R) * j,
                "seeds_a": b["seeds_a"] + 32 * u.row, "lkeys_a": b["lkeys_a"] + 32 * u.row, "pd_count": b["pd_count"] + 16 * j,
                "seed_a": b["seed_a"] + 32 * j, "line_a": b["line_a"] + 32 * j, "ea_codes": b["ea_codes"] + 32 * u.row,
                "c": b["c"] + 4 * n * u.row, "u": b["u"] + 4 * n * u.row, "leaves": leaves + 64 * u.tile}
        full = {"dom_a": key, "unit": key + 64, "dom_b": 0, "dom_t": 0, "line_b": 0, **kk}
        pipe = R.Pipeline(dev, u.seg, n, k, "h100", full, b["xa"] + 4 * k * u.row, 0, bufs=bufs, m_real=u.m16,
                          split_a=split, fillers=not split)
        assert pipe.pd == R.pd_variant(u.m16, n, k)
        dev.log.clear()
        pipe.pd_commit(int(split))
        if split:
            pipe.pd_tree()
        pipe.pd_tiles()
        (*_, cp, ci), *mid, (*_, tp, ti) = dev.log
        assert tuple(commit["p"][j][:len(cp)]) == cp and tuple(commit["i"][j][:len(ci)]) == ci, j
        assert tuple(tiles["p"][j][:len(tp)]) == tp and tuple(tiles["i"][j][:len(ti)]) == ti, j
        assert not commit["p"][j][len(cp):14].any() and not tiles["p"][j][len(tp):].any()
        assert (commit["p"][j][14], commit["i"][j][len(ci)], len(ci)) == (g.zero, u.m, 4) and not commit["p"][j][15]
        if split:
            ((*_, rp, ri),) = mid
            assert tuple(tree["p"][j][:len(rp)]) == rp and tuple(tree["i"][j][:len(ri)]) == ri and rp[12] == 0, j
            assert not tree["p"][j][len(rp):].any()
        else:
            assert not mid and not tree["p"][j].any() and not tree["i"][j].any()
        assert rec[j, G.KEYS_AT:G.KEYS_AT + G.KEY_BYTES].tobytes() == keys[G.KEY_BYTES * j:G.KEY_BYTES * (j + 1)]
    blocks = sum(u.seg for u in us) // G.SEG
    ex = np.frombuffer(meta, "<i4", blocks, G.REC * len(us))
    assert ex.tolist() == [0] + [2] * 4 + [2, 3] and G.Grouped.rows_at(us) == G.REC * len(us) + 4 * blocks + 4
    at = G.Grouped.rows_at(us)
    rows = np.frombuffer(meta, "<i8", 367, at)
    assert rows.tolist() == [*range(3), *range(64, 320), *range(320, 364), *range(384, 448)]
    nodes = np.frombuffer(meta, "<i8", len(us), at + 8 * 367)
    assert nodes.tolist() == [(G.tree_bytes(R) * j) // 64 + 2 * R.pow2(u.m16) - 2 for j, u in enumerate(us)]
    ctas = np.frombuffer(meta, "<u4", sum(u.m16 for u in us), G.Grouped.map_at(us))
    assert G.Grouped.map_at(us) == at + 8 * (367 + len(us)) and ctas.tolist() == [j << 16 | r for j, r in G.rows_order(us, 132)]
    assert len(meta) == G.Grouped.map_at(us) + 4 * len(ctas) <= G.sizes(R, 1024, 16, n, k)["meta"]
    with pytest.raises(ValueError, match="past the call's"):
        g.meta(G.units([1100], n, R.PD_MAX_ROWS), bytes(G.KEY_BYTES * 5), leaves)


@pytest.mark.parametrize("counts", [[40], [3, 1, 17, 1, 40, 8, 2, 5], [3, 0, 300, 64, 17, 1]])
def test_a_call_is_eight_launches_whatever_its_experts(counts):
    """One launch a step, in `grouped.STEPS`' order: pd_commit_gm on Σ m₁₆ CTAs, each an SM's whole shared memory
    while the units' real rows fit the 132 SMs ([40], and decode's 77 on 176 CTAs) and its least past them
    ([3, 0, 300, 64, 17, 1]), and pd_tiles_g_979 on
    (max row blocks x groups of PD_G tiles a row, units) of PD_G warps (`grouped.tiles_g`: its CTAs fit the SMs), each reading its records from the table; stats, E', P_A over all of A's rows; form_s5_g and both
    GEMMs with the expert map, the GEMMs persistent at the H100's 132 SMs on A's rows and their map set."""
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
    smem = dev.pd_smem_g if sum(counts) <= 132 else R.PD_PAIR_SMEM
    assert (smem == dev.pd_smem_g) == (len(counts) != 6) and dev.pd_smem_g > R.PD_PAIR_SMEM
    assert G.commit_rows(dev, us) == sum(counts) and (sum(u.m16 for u in us) > 132) == (len(counts) != 1)
    assert by["pd_commit_gm_851"][1:4] == (sum(u.m16 for u in us), 32 * 4, smem)
    assert by["pd_commit_gm_851"][5:] == ((b["meta"], b["meta"] + G.Grouped.map_at(us)), (G.REC,))
    groups = max(R.cdiv(u.m16, 64) * R.cdiv(n // 64, R.PD_G) for u in us)
    assert by["pd_tiles_g_979"][1:4] == ((groups, len(us)), 32 * R.PD_G, R.PD_PAIR_SMEM // 2)
    assert dev.max_dyn["pd_tiles_gw_979"] == R.PD_PAIR_SMEM // 2
    assert by["pd_tiles_g_979"][5:] == ((b["meta"] + G.TILES_AT,), (G.REC,))
    assert by["stats_s5"][6] == (rows, k) and by["eprime"][6] == (rows,) and by["pa"][6] == (rows,)
    assert by["form_s5_g"][1] == (k // 128, rows // 64) and by["form_s5_g"][5][3:] == (b["fa_codes"], b["at"], ex)
    for name, slot, cols in (("gemm_words_g_d", 1, R.WORDS_N), ("gemm_pearlc_g_d", 0, n)):
        _, grid, block, _, _, p, i = by[name]
        assert grid == min((rows // 64) * (cols // R.gemm_tile(name)[1]), 132) and block == R.gemm_threads(name)
        assert p[12] == ex and p[10] == g.tma + R.MAP_SET * slot and i == (rows, cols, k, R.RASTER)
    assert by["gemm_pearlc_g_d"][5][:9] == (b["at"], b["bt"], b["c"], b["u"], b["y"], b["pa_t"], b["pb_t"], b["ia_a"], b["ia_b"])


def test_a_split_call_runs_as_tree_from_its_record_after_the_per_row_keying():
    """With `split` a call is `grouped.SPLIT_STEPS`: pd_commit_gm as unsplit (its records then in mode 1), pd_tree_g on one
    CTA of 128 threads a unit, reading its record's pd_tree Args, at PD_PAIR_SMEM (not a whole SM: a GEMM CTA fits beside
    it), then the rest as unsplit."""
    n, k = 128, 1024
    g, dev, b, _ = grouped(n, k, experts=6)
    us = G.units([3, 0, 300, 64, 17, 1], n, R.PD_MAX_ROWS)
    dev.log.clear()
    steps = g.steps(us, split=True)
    assert [x for x, _ in steps] == list(G.SPLIT_STEPS)
    for _, f in steps:
        f()
    got = launches(dev)
    assert [x[0] for x in got] == [STEPS[0], "pd_tree_g_851", *STEPS[1:]]
    assert got[1][1:4] == ((1, len(us)), 128, R.PD_PAIR_SMEM) and dev.pd_tree_smem_g == R.PD_PAIR_SMEM
    assert got[1][5:] == ((b["meta"] + G.TREE_AT,), (G.REC,))
    dev.log.clear()
    for _, f in g.steps(us):
        f()
    assert launches(dev) == got[:1] + got[2:]


def test_the_tile_leaves_take_sixteen_tiles_a_cta_where_four_would_be_past_the_sms(monkeypatch):
    """`grouped.tiles_g`: decode's eight units at 6144 columns are 192 CTAs at PD_G (past the 132 SMs), so TILES_GW's 48 a
    launch of pd_tiles_gw_979 at its 512 threads; at 4096 columns 128, which fit, so PD_G; prefill's units past them too.  A
    cubin without pd_tiles_gw, or the tiles4 probe, takes PD_G."""
    decode, prefill = [3, 1, 17, 1, 40, 8, 2, 5], [256] * 20
    dev = RecordingDev()
    got = {(n, len(c)): G.tiles_g(R, dev, G.units(c, n, R.PD_MAX_ROWS), n) for n in (4096, 6144) for c in (decode, prefill)}
    assert got == {(4096, 8): R.PD_G, (6144, 8): G.TILES_GW, (4096, 20): G.TILES_GW, (6144, 20): G.TILES_GW}
    us = G.units(decode, 6144, R.PD_MAX_ROWS)
    assert G.tiles_g(R, OldCubinDev(), us, 6144) == R.PD_G
    monkeypatch.setattr(G, "PLACE", frozenset({"tiles4"}))
    assert G.tiles_g(R, dev, us, 6144) == R.PD_G
    monkeypatch.setattr(G, "PLACE", frozenset())
    g, dev, _, _ = grouped(6144, 1024, experts=len(decode))
    dev.log.clear()
    for _, f in g.steps(us):
        f()
    tiles = launches(dev)[-1]
    assert tiles[:4] == ("pd_tiles_gw_979", (R.cdiv(96, G.TILES_GW), len(us)), 32 * G.TILES_GW, R.PD_PAIR_SMEM // 2)
    assert dev.max_dyn["pd_tiles_gw_979"] == R.PD_PAIR_SMEM // 2


class OldCubinDev(RecordingDev):
    """A cubin built before pd_tree_g, pd_tiles_g_979, pd_tiles_gw_979 and pd_commit_gm: looking them up fails as `run.Dev.fn`'s does."""

    def fn(self, name):
        if name.startswith("pd_tree_g_") or name in ("pd_tiles_g_979", "pd_tiles_gw_979", "pd_commit_gm_851"):
            raise RuntimeError("cuModuleGetFunction returned CUresult 500 (named symbol not found)")
        return name


def device_run(experts: int, n: int = 128, k: int = 1024, retain=None, schedule="serial", dev=None):
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy((rng.standard_normal((n, k)).astype(np.float32) * 0.05)).to(torch.bfloat16)
               for w in range(experts)}
    kernel = PPD.Kernel(R, F, dev or RecordingDev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, SCHEME, salt_key=SALT_KEY)
    ch = Challenger(0, np.random.default_rng(7).bytes)
    coins = ch.commitment, ch.open("salt", weights_root(roots))
    run = PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, *coins, capacity=1 << 12,
                              schedule=schedule, salt_key=SALT_KEY)
    run.begin(retain=retain)
    return run, weights


def lane_steps(split: bool) -> list[str]:
    """A call's launches in the order a side lane's schedule makes them: the main lane's, then A's tree and the tile leaves."""
    return [*STEPS[:-1], *(["pd_tree_g_851"] if split else []), STEPS[-1]]


@pytest.mark.parametrize("schedule", ["serial", "deferred", "deferred-a"])
def test_the_vllm_call_commits_each_unit_as_its_call_alone(tmp_path, schedule):
    """`gemm_grouped` over experts (1, 3, 0) with (3, 300, 20) rows under each schedule, and each unit's rows called alone
    through `PearlCDeviceRun.gemm` in unit order on another run under the serial one: the same matmuls, tiles, screened
    rows, records and retained rows, and each unit's keys in the table are the ones its call alone uploads.  The call is a
    launch a step, y is x's rows by n, and an expert's dense call after grouping reads its stacked B̃."""
    ws, counts, n, k = (1, 3, 0), [3, 300, 20], 128, 1024
    x = torch.randn(sum(counts), k).to(torch.bfloat16)
    run, weights = device_run(4, n, k, tmp_path / "grouped", schedule)
    dev = run.kernel.dev
    dev.log.clear()
    y = PG.gemm_grouped(run, ws, x, counts)
    assert y.shape == (sum(counts), n) and y.dtype == torch.bfloat16
    assert [x[0] for x in launches(dev)] == lane_steps(schedule == "deferred-a")
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


@pytest.mark.parametrize("schedule", ["deferred", "deferred-caller", "deferred-a"])
def test_the_vllm_call_runs_on_the_schedules_lanes_from_rotating_sets(schedule):
    """A grouped call's main lane (the recording stand-in's handle 2, or the caller's 0 under ``deferred-caller``) runs every
    step but the side ones, which run on the run's side lanes in turn (1, then 3): the tile leaves, and under ``deferred-a``
    first A's tree, pd_commit_gm then in mode 1.  Calls take the ring's sets in turn (C~, U, the table and A's tree), and A's
    roots are gathered into the pass from the call's set."""
    ws, counts, n, k = (1, 3, 0), [3, 300, 20], 128, 1024
    run, _ = device_run(4, n, k, schedule=schedule)
    dev, nb = run.kernel.dev, run.nb
    us = G.units(counts, n, R.PD_MAX_ROWS)
    g = PG.group(run, ws)
    g.fit(run, sum(u.seg for u in us), run.schedule.ring)
    for j, st in enumerate(g.sets):                     # stand-in trees, since the recording stand-in writes nothing
        st["tree_a"].fill_(0x11 * (j + 1))
    split, main = schedule == "deferred-a", 0 if schedule == "deferred-caller" else 2
    dev.log.clear()
    for _ in range(3):
        PG.gemm_grouped(run, ws, torch.randn(sum(counts), k).to(torch.bfloat16), counts)
    log = launches(dev)
    side = ("pd_tree_g_851", "pd_tiles_g_979")
    assert [(x[0], x[4]) for x in log] == [(x, lane if x in side else main) for lane in (1, 3, 1) for x in lane_steps(split)]
    by = {name: [x[5] for x in log if x[0] == name] for name in lane_steps(True)}
    ptr = lambda j, x: g.sets[j][x].data_ptr()
    for c, j in enumerate((0, 1, 0)):
        assert by["pd_commit_gm_851"][c] == (ptr(j, "meta"), ptr(j, "meta") + G.Grouped.map_at(us))
        assert by["pd_tiles_g_979"][c] == (ptr(j, "meta") + G.TILES_AT,)
        assert by["gemm_pearlc_g_d"][c][2:4] == (ptr(j, "c"), ptr(j, "u"))
        assert by["pd_tree_g_851"][c:c + 1] == ([(ptr(j, "meta") + G.TREE_AT,)] if split else [])
    for j in (0, 1):
        rec = np.frombuffer(PPD._bytes(g.sets[j]["meta"])[:G.REC * len(us)], np.uint8).reshape(len(us), G.REC)
        commit, tree = (rec[:, a:z].copy().view(G.ARGS).ravel() for a, z in ((0, G.TILES_AT), (G.TREE_AT, G.REC)))
        assert commit["i"][:, 2].tolist() == [int(split)] * len(us)
        assert tree["p"][:, 2].tolist() == [ptr(j, "tree_a") + G.tree_bytes(R) * i if split else 0 for i in range(len(us))]
    assert len({ptr(j, x) for j in (0, 1) for x in PG.ROTATED}) == 2 * len(PG.ROTATED)
    assert [set(PPD._bytes(run.roots_buf[nb * i:nb * (i + 1)])) for i in range(3 * len(us))] == [
        {0x11 * (j + 1)} for j in (0, 1, 0) for _ in us]


@pytest.mark.parametrize("schedule", ["serial", "deferred-a"])
def test_each_calls_table_is_its_layouts_with_its_keys_leaves_and_pass_offsets(schedule):
    """A call's table, uploaded into its ring set's, is `grouped.Grouped.meta` of its units, its keys and its first tile
    leaf's address, then its byte offsets in the pass for its scatter: its A roots at its first matmul's position, its
    flags at its first row's.  Its units' keys are those of their indices in the epoch, so a call in the next pass (the
    third) continues the first pass's indices while its offsets start from the pass's.  Its layout (the template the table
    is filled from, and the steps) is made once a ring set."""
    ws, counts, n, k = (1, 3, 0), [3, 300, 20], 128, 1024
    run, _ = device_run(4, n, k, schedule=schedule)
    us, split = G.units(counts, n, R.PD_MAX_ROWS), schedule == "deferred-a"
    for call in range(3):
        if call == 2:
            run.end_pass()
        pos, tiles, flag_rows = len(run.cur.matmuls), run.cur.tiles, run.cur.flag_rows
        index = run.cur.base + pos
        assert index == call * len(us)
        PG.gemm_grouped(run, ws, torch.randn(sum(counts), k).to(torch.bfloat16), counts)
        assert [u.index for u in run.cur.matmuls[pos:]] == list(range(index, index + len(us)))
        g = run._groups[ws]
        j = call % len(g.sets)
        keys = b"".join(run.keys(run.roots[ws[u.pos]], index + i, u.m16) for i, u in enumerate(us))
        g.grouped.b.update({x: t.data_ptr() for x, t in g.sets[j].items()})
        want = (g.grouped.meta(us, keys, run.leaves.data_ptr() + run.nb * tiles, split)
                + struct.pack("<3q", run.nb * pos, 2 * flag_rows, 0))
        assert PPD._bytes(g.sets[j]["meta"])[:len(want)] == want, call
    assert len(g.calls) == len(g.sets) == (2 if split else 1)


def test_grouped_checks_pass_diff_counts_what_is_committed_not_the_schedule(tmp_path):
    """grouped_check's `_same` over a grouped call's retained pass under ``deferred-a`` and its units called alone under the
    serial schedule: their manifests differ in the schedule they ran under alone, which it doesn't count, and a changed A
    rows file it does."""
    C = load("pearl_c_sm90_grouped_check", HERE / "grouped_check.py")
    ws, counts, n, k = (1, 3), [3, 20], 128, 1024
    x = torch.randn(sum(counts), k).to(torch.bfloat16)
    run, weights = device_run(4, n, k, tmp_path / "grouped", "deferred-a")
    PG.gemm_grouped(run, ws, x, counts)
    alone, _ = device_run(4, n, k, tmp_path / "alone")
    for u in G.units(counts, n, R.PD_MAX_ROWS):
        alone.gemm(ws[u.pos], x[u.src:u.src + u.m], weights[ws[u.pos]], None)
    run.end_pass()
    alone.end_pass()
    a, b = (json.loads((tmp_path / d / "manifest.json").read_text()) for d in ("alone", "grouped"))
    assert [kk for kk in a if a[kk] != b.get(kk)] == ["schedule"]
    assert C._same(tmp_path / "alone", tmp_path / "grouped") == []
    rows = tmp_path / "grouped" / b["matmuls"][1]["rows"]["path"]
    rows.write_bytes(bytes(c ^ 1 for c in rows.read_bytes()))
    assert C._same(tmp_path / "alone", tmp_path / "grouped") == ["matmul 1 rows"]


def test_the_vllm_call_runs_unsplit_on_a_cubin_without_pd_tree_g():
    """A ship built before pd_tree_g (`grouped.has_tree` false) runs ``deferred-a``'s grouped calls as ``deferred``'s: A's
    whole commitment in pd_commit_g's mode 0 (a unit's rows on its CTA row, since it has no pd_commit_gm) on the main lane,
    the tile leaves on the side lane, a CTA a tile (pd_tiles_g_851, since it has no pd_tiles_g_979)."""
    run, _ = device_run(4, schedule="deferred-a", dev=OldCubinDev())
    dev = run.kernel.dev
    dev.log.clear()
    PG.gemm_grouped(run, (1, 3), torch.randn(40, 1024).to(torch.bfloat16), [30, 10])
    old = ["pd_commit_g_851", *STEPS[1:-1], "pd_tiles_g_851"]
    assert [(x[0], x[4]) for x in launches(dev)] == [(x, 1 if x == "pd_tiles_g_851" else 2) for x in old]
    us = G.units([30, 10], 128, R.PD_MAX_ROWS)
    assert launches(dev)[-1][1:4] == ((max(u.tiles for u in us), 2), 32, 0)
    assert launches(dev)[0][1] == (max(u.m16 for u in us), 2) and launches(dev)[0][5] == (run._groups[(1, 3)].b["meta"].data_ptr(),)
    assert not G.has_tree(dev) and G.has_tree(RecordingDev())
    assert (G.tiles_pd(dev), G.tiles_pd(RecordingDev())) == (G.PD, G.TILES_PD)
    rec = np.frombuffer(PPD._bytes(run._groups[(1, 3)].b["meta"])[:2 * G.REC], np.uint8).reshape(2, G.REC)
    assert rec[:, :G.TILES_AT].copy().view(G.ARGS).ravel()["i"][:, 2].tolist() == [0, 0]


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
