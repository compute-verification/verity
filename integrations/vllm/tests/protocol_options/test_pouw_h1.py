"""`protocol_options.pouw_h1`: each served int8_exact linear call is a Program on bits that `pouw.circuit.compile` takes to
H1 tile units, every unit a full 32 x 32 grid at decode's one row, a partial block and an off-tile column count; a call is one
seed group, the calls take consecutive positions, and the `pouw` option serves the arm's own y beside sampled proofs.

The first compile builds the H1 tile unit (its seeds' and noise blocks' SHA-512 in gates), about 40 s once per process;
every later shape reuses it."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.ml.circuits.refs import runs  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import hooks  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402

K = 64
ROW = K * PH.I8_BITS
SHAPES = [(K, 32, 1), (K, 32, 5), (K, 40, 33)]
SALT = hashlib.sha512(b"salt").digest()


class Int8Arm:
    """A stand-in for `engine.int8_vllm.Int8PlainLinearMethod`: the plain arm's linear, counting its calls."""

    int8_arm = E.ARM

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        return E.plain_linear(x, layer.weight, layer.weight_scale)


class Linear:
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = Int8Arm()


class Model:
    def __init__(self, *layers):
        self.layers = list(layers)

    def modules(self):
        return iter(self.layers)


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _entry():
    return f"pouw:{PH.SCHEME}"


@pytest.mark.parametrize("k,n,m", SHAPES, ids=["decode-1", "partial-5", "rows-33-cols-40"])
def test_every_served_shape_compiles_to_full_warp_tiles(k, n, m):
    """Decode's one row, a partial block and a column count off the tile: every unit is a full 32 x 32 grid, the call one
    site and one seed group covering its units."""
    c = PH.compiled(k, n, m)
    mp, np_ = PH.padded(m, n)
    assert len(c.units) == PH.units(k, n, m) == (mp // 32) * (np_ // 32)
    assert {(u.tile.r, u.tile.c, u.tile.grid) for u in c.units} == {(32, 32, True)}
    assert [s.shape for s in c.sites] == [(mp, k, np_)] and c.groups == ((0,),) and c.cover == (len(c.units),)
    assert {u.defn.base_id for u in c.units} == {"PouwH1Tile_v1"}


def test_the_site_rows_are_the_tap_rows_then_distinct_pad_rows():
    """A's first m rows are the call's s8 rows (parameter x) in order and its B rows the weight's (parameter w); each pad row
    reads a constant of its own, so the 32 rows are distinct and the tile a grid."""
    (site,) = PH.compiled(K, 32, 5).sites
    assert [runs(r) for r in site.a_rows[:5]] == [[("p", 0, i * ROW, (i + 1) * ROW)] for i in range(5)]
    assert [runs(r) for r in site.b_rows] == [[("p", 2, j * ROW, (j + 1) * ROW)] for j in range(32)]
    pads = [tuple(runs(r)) for r in site.a_rows[5:]]
    assert len(set(pads)) == 27 and all(s == "n" for p in pads for s, _i, _lo, _hi in p)


def test_the_served_outputs_are_the_real_cells():
    """The Program serves cell (i, j) for i < m, j < n, each one bf16 of the coordinate batch; pad cells are not served."""
    fn = PH.program(K, 40, 33).fn
    np_ = PH.padded(33, 40)[1]
    (node,) = {i for s, i, _lo, _hi in runs(fn.body.ret)}
    assert fn.body.nodes[node].fn.base_id == "ScaledMmS8Coordinate_v2" and fn.body.nodes[node].n == 64 * np_
    want = [(i * np_ * 16, (i * np_ + 40) * 16) for i in range(33)]
    assert [(lo, hi) for _s, _i, lo, hi in runs(fn.body.ret)] == want


def test_calls_take_consecutive_positions_and_seeds_of_their_own_roots():
    rng = np.random.default_rng(3)
    run = PH.H1Run("t", SALT, {0: b"\x01" * 64, 1: b"\x02" * 64}, {0: "q", 1: "o"})
    calls = [run.record(w, rng.integers(-127, 128, (m, K), dtype=np.int8), np.ones(m, np.float32), n)
             for w, (_k, n, m) in zip((0, 1, 0), SHAPES)]
    assert [c.base for c in calls] == [0, 1, 2] and run.next_base == 6 and run.commits["units"] == 6
    seeds = [s for c in calls for s in c.compiled().seeds(SALT, [c.roots], c.base)]
    assert len(seeds) == len(set(seeds)) == 6
    salt, positions, roots = calls[2].inputs(SALT)
    assert salt == SD.leaves(SALT, True) and len(positions) == 4 * SD.POS_BITS.leaves
    assert roots == SD.leaves(calls[2].roots[0] + b"\x01" * 64, True)
    assert calls[0].roots[0] != calls[2].roots[0]
    assert set(run.check()) == {"1x64x32", "5x64x32", "33x64x40"}


def test_z_is_the_exact_product_on_each_units_cells():
    rng = np.random.default_rng(5)
    xq = rng.integers(-127, 128, (33, K), dtype=np.int8)
    w = rng.integers(-127, 128, (40, K), dtype=np.int8)
    y = xq.astype(np.int64) @ w.astype(np.int64).T
    call = PH.Call(0, 0, "q", 33, K, 40, 0, (b"\x00" * 64, b"\x00" * 64))
    full = np.zeros((64, 64), np.int64)
    full[:33, :40] = y
    for n, u in enumerate(call.compiled().units):
        z = call.z(y.astype(np.int32), n)
        assert np.array_equal(z, [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs])
        pad = [u.rows[i] >= 33 or u.cols[j] >= 40 for i, j in u.tile.pairs]
        assert not z[pad].any()


def test_h1_is_admitted_beside_sampled_proofs_and_executes_nowhere():
    assert PH.SCHEME in pouw.schemes() and PH.SCHEME in pouw.EXECUTED and pouw.executes(PH.SCHEME) == ()
    assert pouw.executes("ncp-v2") == pouw.EXECUTES
    commit = PO.Context(stage="commit", enabled=(_entry(), "sampled-proofs:vllm-v1"), settings={"pouw": {"coins": Challenger(0)}})
    assert PO.refusals(commit) == ()
    eager = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0)}}, enforce_eager=False)
    assert any("graph-capturable" in r for r in PO.refusals(eager))


def test_the_option_serves_the_arms_y_and_records_each_call(tmp_path):
    """Installed on int8_exact layers, each apply serves the arm's own y bit for bit and records the call: its s8 rows' root,
    its weight's root and its positions; the record compiles at every served shape."""
    model = Model(Linear(32, 1), Linear(40, 2))
    ctx = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0), "transcript_dir": str(tmp_path),
                                                                              "retain_steps": 2}})
    comp = PO.install(model, ctx)
    try:
        g = torch.Generator().manual_seed(9)
        for m, layer in ((1, model.layers[0]), (5, model.layers[1]), (33, model.layers[0])):
            x = (torch.randn(m, K, generator=g) * 3).bfloat16()
            y = layer.quant_method.apply(layer, x)
            assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        run = pouw.h1_run(_entry())
        assert [(c.layer, c.m, c.n, c.base) for c in run.calls] == [(0, 1, 32, 0), (1, 5, 40, 1), (0, 33, 32, 3)]
        assert [layer.quant_method.calls for layer in model.layers] == [2, 1]
        assert {c.roots[1] for c in run.calls if c.layer == 0} == {PH.weight_root(0, model.layers[0].weight.numpy())}
        index, xq, _sa, z = run.retained[-1]
        assert index == 2 and len(run.retained) == 2
        assert np.array_equal(z, xq.astype(np.int64) @ model.layers[0].weight.numpy().astype(np.int64).T)
        assert set(run.check()) == {"1x64x32", "5x64x40", "33x64x32"}
    finally:
        comp.close()
    assert hooks.live(_entry()) == ()
    assert (tmp_path / PH.CALLS_FILE).exists() and comp.installed[0].commitment()["calls"] == 3


def test_a_layer_off_the_int8_arm_is_refused_at_install():
    layer = Linear(32, 1)
    layer.quant_method = type("Other", (), {"apply": lambda self, layer, x, bias=None: x})()
    with pytest.raises(ValueError, match="not the int8_exact plain arm"):
        PO.install(Model(layer), PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0)}}))
    assert hooks.live(_entry()) == ()
