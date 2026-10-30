"""`protocol_options.pouw_h1`: each served int8_exact linear call is a Program on bits that `pouw.circuit.compile` takes to
H1 tile proof blocks, every proof block a full 32 x 32 grid at decode's one row, a partial block and an off-tile column count; a call is one
seed group, the calls sit at positions fixed by the public parameters (T rows, S steps), and the `pouw` option serves the
arm's own y beside sampled proofs.

The first compile builds the H1 tile proof block (its seeds' and noise blocks' SHA-512 in gates), about 40 s once per process;
every later shape reuses it."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as LC  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveRecord  # noqa: E402
from verity.ml.circuits.refs import runs  # noqa: E402
from verity_vllm import protocols as PO  # noqa: E402
from verity_vllm.engine import hooks  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.protocols import pouw  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import hooks as PHK  # noqa: E402
from verity_vllm.protocols.pouw.h1 import challenger as PHC  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402

from tests.protocols import challenger_host as CH  # noqa: E402

K = 64
ROW = K * PH.I8_BITS
SHAPES = [(K, 32, 1), (K, 32, 5), (K, 40, 33)]
SALT = hashlib.sha512(b"salt").digest()


class Int8Arm:
    """A stand-in for `engine.int8_vllm.Int8PlainLinearMethod`: the plain arm's linear, counting its calls, handing its values
    to ``observe`` after both ops."""

    int8_arm = E.ARM

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        xq, s_a = E.quantize_rows(x)
        y = E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)
        self.observe(layer, xq, s_a, y)
        return y

    def observe(self, layer, xq, s_a, y):
        pass


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


@pytest.mark.slow
@pytest.mark.parametrize("k,n,m", SHAPES, ids=["decode-1", "partial-5", "rows-33-cols-40"])
def test_every_served_shape_compiles_to_full_warp_tiles(k, n, m):
    """Decode's one row, a partial block and a column count off the tile: every proof block is a full 32 x 32 grid, the call one
    site and one seed group covering its proof blocks."""
    c = PH.compiled(k, n, m)
    mp, np_ = PH.padded(m, n)
    assert len(c.units) == PH.units(k, n, m) == (mp // 32) * (np_ // 32)
    assert {(u.tile.r, u.tile.c, u.tile.grid) for u in c.units} == {(32, 32, True)}
    assert [s.shape for s in c.sites] == [(mp, k, np_)] and c.groups == ((0,),) and c.cover == (len(c.units),)
    assert {u.defn.base_id for u in c.units} == {"PouwH1Tile_v1"}


@pytest.mark.slow
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


def _tap(rng, m, n):
    a = torch.from_numpy(rng.integers(-127, 128, (m, K), dtype=np.int8))
    w = torch.from_numpy(rng.integers(-127, 128, (n, K), dtype=np.int8))
    return a, torch.rand(m) + 0.5, w, torch.ones(n, 1)


def _run(steps=3, rows=32):
    return PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], rows, steps)


@pytest.mark.slow
def test_calls_sit_at_public_positions_and_empty_steps_close_the_window():
    """Call (s, w) is at s U + off(w) whatever its served rows; the steps past the last served one are empty calls, so the
    window is every (step, linear) in order; given each call's (root(A), root(B)), every proof block's seed is its own."""
    rng = np.random.default_rng(3)
    run = _run()
    assert (run.offs, run.u_step) == ([0, 1], 3) and run.commits["units"] == 9
    served = [run.record(w, *_tap(rng, m, n)) for w, n, m in ((0, 32, 1), (1, 40, 5), (0, 32, 32), (1, 40, 3))]
    run.close()
    assert [(c.step, c.layer, c.base, c.m, c.rows) for c in run.calls] == [
        (0, 0, 0, 1, 32), (0, 1, 1, 5, 32), (1, 0, 3, 32, 32), (1, 1, 4, 3, 32), (2, 0, 6, 0, 32), (2, 1, 7, 0, 32)]
    assert served == run.calls[:4] and run.static() and run.details["steps_served"] == 2 and run.details["last_step_complete"]
    roots = [(bytes([c.index + 1]) * 64, bytes([c.layer + 9]) * 64) for c in run.calls]
    seeds = [s for c, r in zip(run.calls, roots) for s in c.compiled().seeds(SALT, [r], c.base)]
    assert len(seeds) == len(set(seeds)) == 9
    salt, positions, leaves = run.calls[3].inputs(SALT, roots[3])
    assert salt == SD.leaves(SALT, True) and len(positions) == 2 * SD.POS_BITS.leaves
    assert leaves == SD.leaves(roots[3][0] + roots[3][1], True)
    assert set(run.check()) == {"32x64x32", "32x64x40"}


def test_a_call_past_the_window_or_its_rows_is_refused_and_the_window_is_not_static():
    rng = np.random.default_rng(6)
    run = _run(steps=1)
    assert run.record(0, *_tap(rng, 33, 32)) is None
    assert run.record(1, *_tap(rng, 2, 40)) is not None
    assert run.record(0, *_tap(rng, 1, 32)) is None
    run.close()
    assert run.refused == [{"step": 0, "linear": 0, "m": 33}, {"step": 1, "linear": 0, "m": 1}] and not run.static()
    partial = _run(steps=1)
    partial.record(1, *_tap(rng, 1, 40))
    partial.close()
    assert partial.order_ok and not partial.details["last_step_complete"] and not partial.static()


def test_a_forward_out_of_module_order_sits_each_call_at_its_step_and_linear():
    """GLM-5.3's DSA attention runs fused_qkv_a_proj and q_b_proj before its indexer's wq_b, which module order puts
    first; counting steps at linear 0 put each forward's first two calls in the step before (step -1, refused, for the
    first) and past their round's root. A step starts at the window's first call and at a linear
    the step already holds: each call sits at (s, w)'s index and position, and round s closes once all of step s's calls
    are in. A step whose order departs from the first step's isn't static."""
    rng = np.random.default_rng(5)
    shapes, forward = [(32, K), (40, K), (32, K)], (1, 2, 0)

    def run_of(orders):
        run = PH.H1Run("t", SALT, {0: "wq_b", 1: "qkv_a", 2: "q_b"}, shapes, 32, 3)
        closed = []
        for m, order in zip((3, 2), orders):
            for w in order:
                assert run.record(w, *_tap(rng, m, shapes[w][0])) is not None
                closed.append(len(run.round_log))
        run.close()
        return run, closed

    run, closed = run_of([forward, forward])
    assert closed == [0, 0, 1, 1, 1, 2] and run.refused == [] and run.order_ok and run.static()
    assert [(c.index, c.step, c.layer, c.base) for c in run.calls] == [
        (3 * s + w, s, w, s * run.u_step + run.offs[w]) for s in range(3) for w in range(3)]
    assert run.step_rows() == [3, 2, 0] and run.details["last_step_complete"]
    assert run.call(4).module == "qkv_a" and run.call(4).step == 1
    swapped, _ = run_of([forward, (2, 1, 0)])
    assert not swapped.order_ok and not swapped.static()


def test_credit_is_the_served_rows_and_the_padding_is_shown_beside_it():
    """Calls recorded outside a traced apply are refused by the credit gate (test_credit_gate has the credited case): their
    served rows are shown as refused, and the padding beside them."""
    rng = np.random.default_rng(7)
    run = _run(steps=2)
    run.record(0, *_tap(rng, 3, 32))
    run.record(1, *_tap(rng, 2, 40))
    run.close()
    c = run.credit()
    served = (3 * 32 + 2 * 40) * K
    assert c["gate_passed"] == {"rows": 0, "macs": 0}
    assert (c["refused"]["calls"], c["refused"]["rows"], c["refused"]["macs"]) == (2, 5, served)
    assert c["padded"]["rows"] == 4 * 32 - 5 and c["padded"]["macs"] == (2 * 32 * 32 + 2 * 32 * 64) * K - served
    assert c["units"] == 6 == run.commits["units"]


def test_fill_declares_every_served_step_full_and_serve_h1_is_119_of_1152():
    """`FILL`: 64 one-token prompts at cap 16 in T = 64, S = 18 prefill at step 0 and decode together, so each of
    the 16 served steps carries 64 rows and u = 16/18; the engine runs all 64 in flight with no prefix
    cache. The 5 batched prompts (`tests/engine/int8_exact_gpu.PROMPTS`, 44 tokens with BOS under the int8 Llama-3.1-8B's
    tokenizer) at the same cap are 44 + 15 x 5 rows: 119/1152."""
    assert PH.FILL.schedule() == [64] * 16 + [0, 0] and PH.FILL.u == Fraction(16, 18)
    assert PH.FILL.engine_args() == {"max_num_seqs": 64, "max_num_batched_tokens": 64, "enable_chunked_prefill": True,
                                     "enable_prefix_caching": False}
    serve_h1 = PH.Static((6, 14, 6, 7, 11), 16, 64, 18)
    assert serve_h1.schedule()[:2] == [44, 5] and serve_h1.u == Fraction(119, 1152)
    with pytest.raises(ValueError, match="don't prefill in one step of 64 rows"):
        PH.Static((2,) * 64, 16, 64, 18)
    with pytest.raises(ValueError, match="doesn't fit a window of 18 steps"):
        PH.Static((1,) * 64, 19, 64, 18)


def _fill_run(rows_per_step):
    """`FILL`'s window over two small linears, step s serving rows_per_step[s] rows."""
    rng = np.random.default_rng(11)
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], PH.FILL.rows, PH.FILL.steps)
    for m in rows_per_step:
        for w, n in ((0, 32), (1, 40)):
            assert run.record(w, *_tap(rng, m, n)) is not None
    run.close()
    return run


def test_a_filled_window_measures_its_declared_u():
    run = _fill_run(PH.FILL.schedule()[:PH.FILL.max_tokens])
    assert run.step_rows() == PH.FILL.schedule() and run.utilization() == PH.FILL.u == Fraction(16, 18)
    assert run.static() and PH.FILL.check(run.step_rows()) == [] and run.details["u"] == "8/9"


def test_a_step_short_of_its_declared_rows_is_padded_in_the_layout_and_refused_by_the_declaration():
    """One sequence missing from step 5 (a preemption, say) and the last served step gone (an early stop): the layout pads
    them (every call keeps its public position, the empty slots and step are padding, u falls), and `Static.check` names
    both steps."""
    short = [64] * 5 + [63] + [64] * 9
    run, full = _fill_run(short), _fill_run([64] * 16)
    assert run.static() and [(c.step, c.layer, c.base) for c in run.calls] == [(c.step, c.layer, c.base) for c in full.calls]
    assert run.step_rows() == short + [0, 0, 0] and run.utilization() == Fraction(16 * 64 - 65, 18 * 64)
    assert run.credit()["padded"]["rows"] == full.credit()["padded"]["rows"] + 2 * 65
    assert [p.split(":")[0] for p in PH.FILL.check(run.step_rows())] == ["step 5", "step 15"]
    assert "63 rows served, 64 declared (fewer" in PH.FILL.check(run.step_rows())[0]


def test_slots_put_the_served_rows_first_then_empty_rows_under_scale_0():
    a, s_a, _w, _ws = _tap(np.random.default_rng(8), 3, 32)
    t, sc = PH.slots(a, s_a, 32)
    assert t.dtype == torch.int8 and t.shape == (32, K) and torch.equal(t[:3], a) and not t[3:].any()
    assert torch.equal(sc[:3], s_a) and not sc[3:].any()
    with pytest.raises(ValueError, match="past the call's 32 slots"):
        PH.slots(*_tap(np.random.default_rng(8), 33, 32)[:2], 32)


def test_listeners_get_the_served_tensors_and_bind_the_calls_roots():
    """A listener gets each call with the very tensors the arm served (slot i is AROW leaf i of the call's (step, layer)
    table), the empty steps' calls with no rows, and what it returns is in the manifest and the calls root; `bind_all`
    reads the roots back."""
    rng = np.random.default_rng(4)
    run = _run(steps=2)
    seen = []

    def listener(call, a, s_a, weight, weight_scale):
        seen.append((call, a, s_a, weight, weight_scale))
        return {"root_a": bytes([call.index]).hex() * 64, "root_b": "ab" * 64} if call.index else None

    run.listen(listener)
    taps = [_tap(rng, 5, 32), _tap(rng, 1, 40)]
    calls = [run.record(w, *t) for w, t in enumerate(taps)]
    run.close()
    assert [s[0] for s in seen] == run.calls and all(s[1] is t[0] and s[3] is t[2] for s, t in zip(seen, taps))
    assert [tuple(s[1].shape) for s in seen[2:]] == [(0, K), (0, K)] and seen[2][3] is taps[0][2]
    manifest = run.manifest()
    assert "bound" not in manifest[0] and manifest[1]["bound"]["root_b"] == "ab" * 64
    assert (manifest[0]["served"], manifest[0]["rows"], manifest[0]["leaves"]) == (5, 32, [0, 32])
    assert PH.bind_all(calls, run.bound) == [None, (b"\x01" * 64, b"\xab" * 64)]
    before = run.calls_root()
    run.bound[1] = {**run.bound[1], "root_b": "cd" * 64}
    assert run.calls_root() != before


@pytest.mark.slow
def test_units_are_in_row_block_then_column_block_order():
    """Proof block u of a call covers row block u // (N' / 32) and column block u % (N' / 32), so position base + u names a tile."""
    c = PH.compiled(K, 40, 33)
    blocks = [(u.rows[0] // 32, u.cols[0] // 32) for u in c.units]
    assert blocks == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(list(u.rows) == list(range(u.rows[0], u.rows[0] + 32)) for u in c.units)


@pytest.mark.slow
def test_z_is_the_exact_product_on_each_units_cells():
    rng = np.random.default_rng(5)
    xq = rng.integers(-127, 128, (33, K), dtype=np.int8)
    w = rng.integers(-127, 128, (40, K), dtype=np.int8)
    y = xq.astype(np.int64) @ w.astype(np.int64).T
    call = PH.Call(0, 0, 0, "q", 33, K, 40, 0, 64)
    full = np.zeros((64, 64), np.int64)
    full[:33, :40] = y
    for n, u in enumerate(call.compiled().units):
        z = call.z(y.astype(np.int32), n)
        assert np.array_equal(z, [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs])
        pad = [u.rows[i] >= 33 or u.cols[j] >= 40 for i, j in u.tile.pairs]
        assert not z[pad].any()


#: two-step windows that admit b and c in round 1, so they admit in every round
H1 = {"coins": None, "rows": 64, "steps": 3, "admission": "per-round"}


class Runner:
    """A stand-in for vLLM's Model Runner V2: ``execute_model`` gets the step's scheduler output; one row a decode."""

    decode_query_len = 1

    def __init__(self):
        self.steps = 0

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.steps += 1


def _so(*new, running=None, tokens=None):
    """A scheduler output admitting requests ``new``, each (req_id, prompt token ids, cap) scheduled whole, after the
    ``running`` requests' {req_id: rows}, in the scheduler's order (running, then new)."""
    reqs = [SimpleNamespace(req_id=i, prompt_token_ids=p, sampling_params=SimpleNamespace(max_tokens=c)) for i, p, c in new]
    scheduled = dict(running or {}) | {i: len(p or ()) for i, p, _c in new}
    total = sum(scheduled.values()) if tokens is None else tokens
    return SimpleNamespace(scheduled_new_reqs=reqs, num_scheduled_tokens=scheduled, total_num_scheduled_tokens=total)


def _rounds(n=3, seed=0):
    clock = itertools.count()
    return LiveRecord(Wd.live_challenger(0, n, random.Random(seed).randbytes), Wd.ROUND, clock=lambda: float(next(clock)))


def _settings(**kw):
    return {"pouw": {**H1, "coins": Challenger(0, random.Random(0).randbytes), "runner": Runner(), "rounds": _rounds(), **kw}}


def test_h1_is_admitted_beside_sampled_proofs_and_executes_nowhere():
    assert PH.SCHEME in pouw.schemes() and PH.SCHEME in pouw.EXECUTED and pouw.executes(PH.SCHEME) == ()
    assert pouw.executes("ncp-v2") == pouw.EXECUTES
    commit = PO.Context(stage="commit", enabled=(_entry(), "sampled-proofs:rows-sha512"), settings=_settings())
    assert PO.refusals(commit) == ()
    eager = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(), enforce_eager=False)
    assert any("graph-capturable" in r for r in PO.refusals(eager))
    bare = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0, random.Random(0).randbytes)}})
    assert any("'rows' (T" in r for r in PO.refusals(bare))
    with pytest.raises(ValueError, match="positive multiple of 32"):
        pouw.Settings.of({"rows": 40})
    unadmitted = {"pouw": {k: v for k, v in _settings()["pouw"].items() if k != "runner"}}
    assert any("the pouw settings 'runner'" in r for r in PO.refusals(PO.Context(stage="serve", enabled=(_entry(),),
                                                                                 settings=unadmitted)))
    capture = PO.Context(stage="capture", enabled=(_entry(),), settings=unadmitted)
    assert PO.refusals(capture) == ()
    short = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(rounds=_rounds(2)))
    assert any("2 unopened coins, fewer than the window's 3 rounds" in r for r in PO.refusals(short))
    other = PO.Context(stage="serve", enabled=("pouw:ncp-v1",), settings={"pouw": {"coins": Challenger(0, CH.entropy(0)), "runner": Runner()}})
    assert any("'runner', 'rounds', 'admission' and 'served' are H1's" in r for r in PO.refusals(other))
    with pytest.raises(ValueError, match="'rounds' is the window's round coins"):
        pouw.Settings.of({"rounds": object()})
    with pytest.raises(ValueError, match="loaded Model Runner V2"):
        pouw.Settings.of({"runner": SimpleNamespace(execute_model=lambda so: None)})


@pytest.mark.slow
def test_the_option_serves_the_arms_y_and_records_each_call(tmp_path):
    """Installed on int8_exact layers, each apply serves the arm's own y bit for bit and the option's observe records the call
    at its public position, its exact Y retained; closing the window adds the empty step, and the record compiles at T."""
    model = Model(Linear(32, 1), Linear(40, 2))
    settings = _settings(transcript_dir=str(tmp_path), retain_steps=2)
    runner = settings["pouw"]["runner"]
    ctx = PO.Context(stage="serve", enabled=(_entry(),), settings=settings)
    comp = PO.install(model, ctx)
    try:
        g = torch.Generator().manual_seed(9)
        steps = [(_so(("a", [5], 4)), ((1, model.layers[0]), (5, model.layers[1]))),
                 (_so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}),
                  ((33, model.layers[0]), (2, model.layers[1])))]
        for so, step in steps:
            runner.execute_model(so)
            for m, layer in step:
                x = (torch.randn(m, K, generator=g) * 3).bfloat16()
                y = layer.quant_method.apply(layer, x)
                assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        run = pouw.h1_run(_entry())
        assert runner.steps == 2 and [(e.index, e.round, e.slot) for e in run.requests] == [(0, 0, 0), (1, 1, 1), (2, 1, 2)]
        assert run.owners[1] == [0, 1] + [2] * 31
        assert [(c.step, c.layer, c.m, c.base) for c in run.calls] == [(0, 0, 1, 0), (0, 1, 5, 2), (1, 0, 33, 6), (1, 1, 2, 8)]
        assert [layer.quant_method.calls for layer in model.layers] == [2, 2]
        index, xq, _sa, z = run.retained[-1]
        assert index == 3 and len(run.retained) == 2
        assert np.array_equal(z, xq.astype(np.int64) @ model.layers[1].weight.numpy().astype(np.int64).T)
        assert set(run.check()) == {"64x64x32", "64x64x40"}
    finally:
        comp.close()
    assert hooks.live(_entry()) == () and runner.execute_model.__func__ is Runner.execute_model
    commitment = comp.installed[0].commitment()
    assert commitment["calls"] == 6 and commitment["units"] == 18 and commitment["static"]
    assert [r["round"] for r in commitment["rounds"]] == [0, 1, 2]
    coins = settings["pouw"]["rounds"].challenger.commitment
    assert RQ.rounds_reason(commitment["rounds"], coins) is None
    assert all(RQ.request_reason(commitment["rounds"], coins, run.salt, 64, e, run.open_request(e.index),
                                 run.open_first(e.index), admission=RQ.PER_ROUND) is None for e in run.requests)
    assert _slot_reasons(run, coins) == []
    record = json.loads((tmp_path / PH.CALLS_FILE).read_text())
    assert [c["base"] for c in record["calls"]] == [0, 2, 6, 8, 12, 14]
    assert record["commitment"]["rounds"] == commitment["rounds"] and record["details"]["requests"]["per_round"] == [1, 2, 0]
    assert record["details"]["requests"]["gaps"] == [] and record["details"]["requests"]["a_roots"].startswith("bound")


def test_the_served_setting_reaches_the_run_and_its_entries_the_commitment():
    """The pouw setting ``served`` is the window's `pouw_h1_served.Served`: the install hands it to the run, which commits
    its trees at its close and carries their entries and refusals in the commitment (here refused: n = 40 is no whole
    block of the tile's 32 B rows)."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    with pytest.raises(ValueError, match="a fresh pouw_h1_served.Served"):
        pouw.Settings.of({"served": object()})
    served = SV.Served(window=bytes(64))
    commitment, run = _serve_window(_settings(served=served))
    assert run.served is served and any("whole blocks" in r for r in served.refused)
    assert commitment["served"] == {"entries": [], "refused": served.refused}


def test_the_window_connects_on_the_port_the_gate_answers():
    from verity_vllm.protocols.pouw.h1 import window as PHW
    assert PHW.at_port("127.0.0.1:7811", 7812) == "127.0.0.1:7812" and PHW.at_port("h:1", None) == "h:1"
    with pytest.raises(ValueError, match="not a TCP port"):
        PHW.at_port("h:1", "7811")


def _vllm_with(runner):
    """A stand-in for the built ``vllm.LLM``: its in-process engine core down to the model runner
    (`engine.handle.model_runner`)."""
    worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    core = SimpleNamespace(engine_core=SimpleNamespace(model_executor=SimpleNamespace(driver_worker=worker)))
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core))


@pytest.fixture
def engine(monkeypatch):
    """`verity_vllm.LLM`'s engine build replaced by a toy int8_exact model and a stand-in vLLM whose model runner is
    ``engine["runner"]``; the in-process coins' entropy is seeded."""
    from verity.core.protocols.pouw import verifier
    from verity_vllm.engine import vllm_adapter as va
    rng = random.Random(1)
    monkeypatch.setattr(verifier, "coins", rng.randbytes)
    got = {"runner": Runner(), "model": Model(Linear(32, 1), Linear(40, 2)), "builds": 0}

    def build_engine(checkpoints, **kw):
        got["builds"] += 1
        return SimpleNamespace(llm=got.get("vllm") or _vllm_with(got["runner"]), model=got["model"], kwargs={},
                               env_flags={}, execution=kw["execution"], effective={}, build_s=0.0)

    monkeypatch.setattr(va, "build_engine", build_engine)
    return got


LLAMA = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
PUBLIC = {"rows": 64, "steps": 3, "admission": "per-round"}


def test_llm_installs_h1_from_its_public_parameters_with_its_engines_runner(engine):
    """`verity_vllm.LLM` with H1 and only its public parameters (T, S) installs it after the build with the engine's own
    model runner and in-process coins: each step's admissions reach the window through that runner, the window's coins
    are recorded as in-process (they credit nothing), `reinstall` opens the next window with the next epoch's coins, and
    `close` restores the runner."""
    from verity_vllm import llm as L
    llm = L.LLM(LLAMA[0], revision=LLAMA[1], protocols=_entry(), protocols_config={"pouw": dict(PUBLIC)})
    runner, model = engine["runner"], engine["model"]
    assert getattr(runner.execute_model, "__func__", None) is not Runner.execute_model and hooks.live(_entry())
    run = pouw.h1_run(_entry())
    assert (run.rows, run.steps, run.coin_source) == (64, 3, "in-process")
    runner.execute_model(_so(("a", [5], 4)))
    x = (torch.randn(1, K, generator=torch.Generator().manual_seed(3)) * 3).bfloat16()
    model.layers[0].quant_method.apply(model.layers[0], x)
    assert runner.steps == 1 and [(e.index, e.round) for e in run.requests] == [(0, 0)] and len(run.calls) == 1
    first = llm.engine_record["protocols"]["protocols"][0]["commitment"]
    assert first["coin_source"] == "in-process" and first["round_coins"] and first["coins"] != {}

    again = llm.reinstall({"pouw": {**PUBLIC, "steps": 2}})
    assert again is llm.protocols and llm.protocols.entries == (_entry(),)
    second = pouw.h1_run(_entry())
    assert second is not run and second.steps == 2 and second.calls == [] and "draw" in run.commits
    assert llm.protocols.installed[0].commitment()["coins"] != first["coins"]
    runner.execute_model(_so(("b", [7], 2)))
    assert [(e.index, e.round) for e in second.requests] == [(0, 0)]
    llm.close()
    assert hooks.live(_entry()) == () and runner.execute_model.__func__ is Runner.execute_model


def test_llm_refuses_h1_by_name_when_its_engine_cant_supply_the_runner(engine, monkeypatch):
    """Fail-closed: H1 without its public parameters is refused before any engine is built; an engine without an
    in-process model runner is refused naming the entry and the settings it couldn't supply; and an adapter that names a
    setting it then doesn't fill is refused by the install's own refusals.  None leaves a hook behind."""
    from verity_vllm import protocols as PO
    from verity_vllm import llm as L
    with pytest.raises(PO.Refused, match=r"'rows' \(T"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols=_entry(), protocols_config={"pouw": {"admission": "per-round"}})
    assert engine["builds"] == 0
    engine["vllm"] = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=None))
    with pytest.raises(PO.Refused, match=rf"{_entry()}: its settings \['runner', 'coins', 'rounds'\] come from the built "
                                         r"engine.*in-process model runner"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols=_entry(), protocols_config={"pouw": dict(PUBLIC)})
    assert engine["builds"] == 1 and hooks.live(_entry()) == ()
    engine.pop("vllm")
    monkeypatch.setattr(pouw, "from_engine", lambda scheme, settings, llm, epoch=0: {})
    with pytest.raises(PO.Refused, match="the pouw settings 'runner'"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols=_entry(), protocols_config={"pouw": dict(PUBLIC)})
    assert hooks.live(_entry()) == ()


def test_h1s_engine_settings_are_named_before_the_build_and_never_override_the_callers():
    """`pouw.fills`: H1's runner, and without a challenger its coins and (once S is set) its rounds; a setting the caller
    gives is never filled, nor anything for another scheme; the refusals before the build read the named ones as set."""
    from verity_vllm import protocols as PO
    assert pouw.fills(PH.SCHEME, PUBLIC) == ("runner", "coins", "rounds")
    assert pouw.fills(PH.SCHEME, {"rows": 64}) == ("runner", "coins")
    assert pouw.fills(PH.SCHEME, {**PUBLIC, "challenger": "h:1"}) == ("runner",)
    assert pouw.fills(PH.SCHEME, _settings()["pouw"]) == ()
    assert pouw.fills("ncp-v2", {}) == ()
    ctx = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": dict(PUBLIC)})
    assert any("the pouw settings 'runner'" in r for r in PO.refusals(ctx))
    assert PO.engine_fills(ctx) == {"pouw": ("runner", "coins", "rounds")}
    assert PO.refusals(dataclasses.replace(ctx, filled=PO.engine_fills(ctx))) == ()


def test_a_layer_off_the_int8_arm_is_refused_at_install():
    layer = Linear(32, 1)
    layer.quant_method = type("Other", (), {"apply": lambda self, layer, x, bias=None: x})()
    with pytest.raises(ValueError, match="not the int8_exact plain arm"):
        PO.install(Model(layer), PO.Context(stage="serve", enabled=(_entry(),), settings=_settings()))
    assert hooks.live(_entry()) == ()


class LinearMethodBase:
    """Stands in for vLLM's base class by name (`kernel_readback.routers`)."""


class Bf16Router(LinearMethodBase):
    def apply(self, layer, x, bias=None):
        return (x.float() @ layer.weight.float().T).bfloat16()


class FusedMoEMethodBase:
    """Stands in for vLLM's base class by name (`pouw_h1_hooks.layers`)."""


class Experts(FusedMoEMethodBase):
    """A stand-in for `engine.int8_vllm.Int8ExpertsMethod`: its sites, as `int8_experts.sites` makes them."""

    int8_arm = E.ARM

    def __init__(self, sites):
        self.sites = sites


class MoE:
    """A MoE layer of `experts` experts (hidden K, intermediate 16): its bf16 router beside its runner, whose routed
    experts are int8 (`Experts`), or `method` if given."""

    def __init__(self, experts, seed, method=None):
        from verity_vllm.engine import int8_experts as X
        g = torch.Generator().manual_seed(seed)
        w13, w2 = (torch.randint(-127, 128, s, generator=g, dtype=torch.int8) for s in ((experts, 32, K), (experts, K, 16)))
        s13, s2 = ((torch.rand(*s, generator=g) * 0.01 + 1e-4).float() for s in ((experts, 32, 1), (experts, K, 1)))
        self.router = Linear(experts, seed + 1)
        self.router.weight = torch.randn(experts, K, generator=g).bfloat16()
        self.router.quant_config, self.router.output_size, self.router.quant_method = None, experts, Bf16Router()
        self.runner = SimpleNamespace(routed_experts=SimpleNamespace(global_num_experts=experts),
                                      quant_method=method or Experts(X.sites("moe.experts", w13, s13, w2, s2)))

    def children(self):
        return iter([self.router, self.runner])


def test_a_moe_layer_records_every_expert_gemm_in_static_order_and_not_its_router():
    """A MoE layer's routed experts record as their sites, (e, gate_up) then (e, down), every forward: an expert no token
    routed to is a call of zero rows, so each step is the same calls at the same public positions; its bf16 router is no
    call. The served y is the arm's, and only served calls count."""
    from verity_vllm.engine import int8_experts as X
    moe = MoE(2, 5)
    dense = Linear(32, 1)
    model = Model(dense, moe, moe.router, moe.runner)
    sites = moe.runner.quant_method.sites
    assert PHK.layers(model, (Linear,)) == [dense, *sites]
    settings = _settings()
    runner = settings["pouw"]["runner"]
    comp = PO.install(model, PO.Context(stage="serve", enabled=(_entry(),), settings=settings))
    try:
        g = torch.Generator().manual_seed(9)
        for so, m, ids in ((_so(("a", [5, 6, 7], 4)), 3, [0, 0, 0]), (_so(running={"a": 1}), 1, [1])):
            runner.execute_model(so)
            dense.quant_method.apply(dense, (torch.randn(m, K, generator=g) * 3).bfloat16())
            x = (torch.randn(m, K, generator=g) * 3).bfloat16()
            topk = torch.tensor(ids).reshape(m, 1)
            w = torch.ones(m, 1)
            y = X.forward(sites, x, w, topk)
            fresh = [X.ExpertSite(s.prefix, s.expert, s.projection, s.weight, s.weight_scale) for s in sites]
            assert torch.equal(y.view(torch.int16), X.forward(fresh, x, w, topk).view(torch.int16))
        run = pouw.h1_run(_entry())
        assert [(c.step, c.layer, c.module, c.m) for c in run.calls] == [
            (0, 0, "linear 0", 3), (0, 1, "moe.experts.0.gate_up", 3), (0, 2, "moe.experts.0.down", 3),
            (0, 3, "moe.experts.1.gate_up", 0), (0, 4, "moe.experts.1.down", 0),
            (1, 0, "linear 0", 1), (1, 1, "moe.experts.0.gate_up", 0), (1, 2, "moe.experts.0.down", 0),
            (1, 3, "moe.experts.1.gate_up", 1), (1, 4, "moe.experts.1.down", 1)]
        assert [s.quant_method.calls for s in sites] == [2, 2, 2, 2]
        assert run.tables[3]["z"] is not None and set(run.gated()) == {0, 1, 2, 5, 8, 9}
    finally:
        comp.close()
    commitment = comp.installed[0].commitment()
    assert commitment["static"] and commitment["calls"] == 15
    assert run.shapes_ == [(32, K), (32, K), (K, 16), (32, K), (K, 16)]


def test_a_linear_the_checkpoint_keeps_bf16_is_not_recorded():
    """A linear ``modules_to_not_convert`` names (`engine.int8_vllm.Int8KeptLinearMethod`, bf16) is no call, as a router;
    an unmarked bf16 linear is still refused."""
    dense, kv_b = Linear(32, 1), Linear(32, 2)
    kv_b.quant_method = type("Int8KeptLinearMethod", (LinearMethodBase,), {"int8_kept": True})()
    assert PHK.layers(Model(dense, kv_b), (Linear,)) == [dense]
    kv_b.quant_method = LinearMethodBase()
    with pytest.raises(ValueError, match="not the int8_exact plain arm"):
        PHK.layers(Model(dense, kv_b), (Linear,))


def test_a_moe_layer_without_int8_sites_is_refused_at_install():
    moe = MoE(2, 5, method=type("UnquantizedFusedMoEMethod", (FusedMoEMethodBase,), {})())
    with pytest.raises(ValueError, match="routed experts record as their sites"):
        PO.install(Model(Linear(32, 1), moe, moe.router, moe.runner),
                   PO.Context(stage="serve", enabled=(_entry(),), settings=_settings()))
    assert hooks.live(_entry()) == ()


KEY = bytes(range(32))


def _opening(table, r, p):
    o = table.open(p)
    return RQ.RoundOpening(r, o.position, o.value, o.path)


def _check_slots(log, coins, rows, manifest, open_slot, entries, open_entry, open_first, salt=SALT):
    """`slot_reason` on every slot of every round in `log`, each named request's entry opened beside it: the refusals."""
    out = []
    for s in range(len(log)):
        calls = [m for m in manifest if m["step"] == s]
        for p in range(rows):
            slot = open_slot(s, p)
            owner = RQ.slot_owner(slot.value[:-RQ.HM96_SALT])
            e = entries[owner] if owner is not None else None
            why = RQ.slot_reason(log, coins, salt, rows, calls, slot, e, e and open_entry(e), e and open_first(e),
                                 admission=RQ.PER_ROUND)
            if why is not None:
                out.append(why)
    return out


def _slot_reasons(run, coins):
    return _check_slots(run.round_log, coins, run.rows, run.manifest(), run.open_slot, run.requests,
                        lambda e: run.open_request(e.index), lambda e: run.open_first(e.index), salt=run.salt)


def _admitted_run():
    """A live window of 3 rounds through `admitting`: request a admitted in round 0 (3 prompt rows), b and c in round 1
    beside a's decode, and all three decoding in round 2."""
    rng = np.random.default_rng(12)
    rounds, runner = _rounds(), Runner()
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], 32, 3, rounds=rounds, key=KEY, admission=RQ.PER_ROUND)
    step = PHK.admitting(run, runner.execute_model, runner)
    for so in (_so(("a", [1, 2, 3], 4)), _so(("b", [4], 2), ("c", [5, 6], 3), running={"a": 1}),
               _so(running={"a": 1, "b": 1, "c": 1})):
        step(so)
        assert run.round_log == [] or run.round_log[-1]["round"] == run.step
        for w, n in ((0, 32), (1, 40)):
            run.record(w, *_tap(rng, so.total_num_scheduled_tokens, n))
    run.close()
    return run, rounds


def test_each_admitted_request_is_a_leaf_under_its_rounds_root_recorded_before_its_coin():
    """Each request is admitted into the round of the step about to run, at its first row in that step's batch order
    (Model Runner V2's: decodes, then prefills by rows); the round's root, over the step's calls, its T request leaves and
    its T slot leaves, is recorded before its coin opens; each entry opens under its own round's root at a slot naming
    it, and every slot of every step names the request its row is."""
    run, rounds = _admitted_run()
    coins = rounds.challenger.commitment
    assert [(e.index, e.round, e.slot, e.max_tokens, e.prompt) for e in run.requests] == [
        (0, 0, 0, 4, (1, 2, 3)), (1, 1, 1, 2, (4,)), (2, 1, 2, 3, (5, 6))]
    assert run.owners == {0: [0, 0, 0], 1: [0, 1, 2, 2], 2: [0, 1, 2]} and run.request_gaps() == []
    assert [r["round"] for r in run.round_log] == [0, 1, 2] and run.details["requests"]["per_round"] == [1, 2, 0]
    assert [ev["event"] for ev in rounds.log] == ["root", "release"] * 3
    roots = [r["root"] for r in run.round_log]
    assert [ev["root"] for ev in rounds.log[::2]] == [ev["after"] for ev in rounds.log[1::2]] == roots
    for row, r in zip(run.round_log, range(3)):
        table = RQ.request_table(SALT, r, 32, run.admitted.get(r, []), KEY)
        slots = RQ.slot_table(SALT, r, 32, run.owners[r], KEY)
        calls = RQ.step_digest(SALT, r, [m for m in run.manifest() if m["step"] == r])
        assert (row["requests"], row["slots"], row["calls"]) == (table.commitment.root.hex(), slots.commitment.root.hex(),
                                                                 calls.hex())
        assert row["receipts"] == (run.receipts_of(r - 1) if r else RQ.NO_RECEIPTS).hex()
        assert row["inner"] == RQ.round_root(SALT, r, calls, table.commitment.root, slots.commitment.root,
                                             bytes.fromhex(row["receipts"])).hex()
        assert row["a"] == {} and row["root"] == CR.round_root(r, bytes.fromhex(row["inner"]), {}).hex()
    for e in run.requests:
        assert RQ.request_reason(run.round_log, coins, SALT, 32, e, run.open_request(e.index), run.open_first(e.index),
                                 admission=RQ.PER_ROUND) is None
    assert _slot_reasons(run, coins) == []
    assert RQ.rounds_reason(run.round_log, coins) is None


def _deferred(run, name=True):
    """A prover that serves each request where the engine schedules it and commits its entry one round late, at its slot
    there, consistently: every table, root and live coin of its log in order. With `name`, each slot names the request
    its row is; without, a row of a request whose entry isn't committed yet names none."""
    late = [dataclasses.replace(e, round=e.round + 1, slot=run.owners[e.round + 1].index(e.index)) for e in run.requests]
    owners = {s: [o if name or late[o].round <= s else None for o in run.owners[s]] for s in run.owners}
    rounds, manifest, log, tables, slots = _rounds(seed=1), run.manifest(), [], {}, {}
    for r in range(run.steps):
        tables[r] = RQ.request_table(SALT, r, 32, [e for e in late if e.round == r], KEY)
        slots[r] = RQ.slot_table(SALT, r, 32, owners[r], KEY)
        calls = RQ.step_digest(SALT, r, [m for m in manifest if m["step"] == r])
        receipts = bytes.fromhex(run.round_log[r]["receipts"])
        row = RQ.round_row(SALT, r, calls, tables[r].commitment.root, slots[r].commitment.root, receipts, {})
        log.append({**row, "coin": rounds.record(bytes.fromhex(row["root"])).to_json()})
    return (log, rounds.challenger.commitment, late,
            lambda e: _opening(tables[e.round], e.round, [x for x in late if x.round == e.round].index(e)),
            lambda s, p: _opening(slots[s], s, p), lambda e: _opening(slots[e.round], e.round, e.slot))


def test_a_prover_that_defers_every_entry_a_round_is_refused_at_the_rows_it_served_first():
    """An entry's round was only the prover's claim, so a prover committing every entry a round
    late, its roots and coins in order, passed. Each deferred entry still opens under its round's root at a slot naming
    it, but the slot table binds each served row of step s to its request: a row served before the entry's round is
    refused, and so is a served row that names no request."""
    run, _rounds_ = _admitted_run()
    log, coins, late, open_entry, open_slot, open_first = _deferred(run)
    assert RQ.rounds_reason(log, coins) is None
    assert all(RQ.request_reason(log, coins, SALT, 32, e, open_entry(e), open_first(e), RQ.PER_ROUND) is None for e in late)
    why = _check_slots(log, coins, 32, run.manifest(), open_slot, late, open_entry, open_first)
    assert why == [f"request {i} is served at slot {p} of step {s}, before round {s + 1}, the round its entry is under"
                   for i, s, p in ((0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 1, 1), (2, 1, 2), (2, 1, 3))]
    log, coins, late, open_entry, open_slot, open_first = _deferred(run, name=False)
    assert all(RQ.request_reason(log, coins, SALT, 32, e, open_entry(e), open_first(e), RQ.PER_ROUND) is None for e in late)
    why = _check_slots(log, coins, 32, run.manifest(), open_slot, late, open_entry, open_first)
    unnamed = "is a served row and names no admitted request: a request served without an entry"
    assert why == [f"slot {p} of step 0 {unnamed}" for p in range(3)] + [f"slot {p} of step 1 {unnamed}" for p in (1, 2, 3)]


def test_a_request_entry_under_the_wrong_round_or_under_none_is_refused():
    """The check refuses, by name: round 0's entry opened under round 1's root, or with round 1's leaf; an entry the prover
    commits into a later round's table than the one that admitted it; an entry opened under a round the log doesn't have,
    or under no table at all; a first slot other than the entry's; a round root that isn't over its parts; and a round
    whose coin isn't the challenger's, or opened before its root."""
    run, rounds = _admitted_run()
    coins = rounds.challenger.commitment
    log, e0, e1 = run.round_log, run.requests[0], run.requests[1]
    o0, o1 = run.open_request(0), run.open_request(1)

    def why(entry, opening, log=log, first=None):
        return RQ.request_reason(log, coins, SALT, 32, entry, opening, first or run.open_first(entry.index),
                                 RQ.PER_ROUND)

    wrong = "request 0 was admitted in round 0, and its entry is opened under round 1's root"
    assert why(e0, dataclasses.replace(o0, round=1)) == wrong and why(e0, o1) == wrong
    assert why(e0, dataclasses.replace(o0, round=5)) == \
        "request 0: the round log has no round 5, so its entry is under no round root"
    assert why(e0, dataclasses.replace(o0, round=1), log=log[:1]) == \
        "request 0: the round log has no round 1, so its entry is under no round root"
    assert why(e0, dataclasses.replace(o0, value=o1.value)) == "request 0: the leaf opened under round 0 is not its entry"
    late = dataclasses.replace(e0, round=1)
    table = RQ.request_table(SALT, 1, 32, [late, *run.admitted[1]], KEY)
    moved = [log[0], {**log[1], **RQ.round_row(SALT, 1, bytes.fromhex(log[1]["calls"]), table.commitment.root,
                                               bytes.fromhex(log[1]["slots"]), bytes.fromhex(log[1]["receipts"]), {})}, log[2]]
    assert why(late, _opening(table, 1, 0), log=moved, first=run.open_slot(1, 0)) == \
        "request 0: round 1's coin: coin was not opened after what the verifier recorded"
    assert why(e0, _opening(table, 1, 0), log=moved) == wrong
    forged = RQ.RequestEntry(3, 2, 0, 1, (9,))
    stray = RQ.request_table(SALT, 2, 32, [forged], b"\1" * 32).open(0)
    assert why(forged, RQ.RoundOpening(2, 0, stray.value, stray.path), first=run.open_slot(2, 0)) == \
        "request 3's entry doesn't open under round 2's request table"
    assert why(e1, o1, first=run.open_slot(1, 0)) == "request 1's first slot is slot 1 of step 1, and slot 0 of step 1 is opened"
    bent = [{**log[0], "calls": log[1]["calls"]}, *log[1:]]
    assert why(e0, o0, log=bent) == "request 0: round 0's root is not over its calls, its request table, its slot table and the receipts before it"
    coinless = [{**log[0], "coin": None}, *log[1:]]
    assert why(e0, o0, log=coinless) == "request 0: round 0 has no live coin, so nothing shows its root committed before one"
    early = [{**log[0], "coin": {**log[0]["coin"], "after": "00" * 64}}, *log[1:]]
    assert why(e0, o0, log=early) == "request 0: round 0's coin: coin was not opened after what the verifier recorded"
    foreign = [{**log[0], "coin": _rounds(seed=7).record(bytes.fromhex(log[0]["root"])).to_json()}, *log[1:]]
    assert why(e0, o0, log=foreign) == "request 0: round 0's coin: coin does not open against the epoch's coin commitment"
    assert RQ.rounds_reason(early, coins) == f"{Wd.ROUND} coin 0: coin was not opened after what the verifier recorded"
    assert RQ.rounds_reason(log[1:], coins) == "the round log is not rounds 0, 1, ... in order"
    assert why(e1, o1) is None


def test_a_request_resumed_after_a_preemption_keeps_its_one_entry():
    """Model Runner V2 puts a request resumed after a preemption back in ``scheduled_new_reqs``
    (vLLM d9105ea8, scheduler.py:1277-1279). Keyed by req_id, it keeps its one entry and is counted readmitted, and its
    recomputed rows name that entry."""
    rng = np.random.default_rng(14)
    rounds, runner = _rounds(), Runner()
    run = PH.H1Run("t", SALT, {0: "q"}, [(32, K)], 32, 3, rounds=rounds, key=KEY)
    step = PHK.admitting(run, runner.execute_model, runner)
    for so in (_so(("a", [1, 2], 4), ("b", [3], 4)), _so(running={"a": 1}), _so(("b", [3, 9], 4), running={"a": 1})):
        step(so)
        run.record(0, *_tap(rng, so.total_num_scheduled_tokens, 32))
    run.close()
    assert [(e.index, e.round, e.slot, e.prompt) for e in run.requests] == [(0, 0, 1, (1, 2)), (1, 0, 0, (3,))]
    assert run.readmitted == [{"round": 2, "index": 1}] and run.owners == {0: [1, 0, 0], 1: [0], 2: [0, 1, 1]}
    assert run.details["requests"]["readmitted"] == 1 and run.request_gaps() == [] and run.refused_requests == []
    assert _slot_reasons(run, rounds.challenger.commitment) == []


def test_a_request_served_without_an_entry_is_a_gap_its_rows_are_refused_and_the_window_credits_nothing():
    """Every request the step schedules must have an entry. Prompt embeddings, a pooling request
    and a request scheduled without being admitted (one running before the install) get none: each one's rows are a gap,
    the window credits nothing (test_credit_gate has a credited call refused for it), and a slot of theirs is refused. A
    step whose served rows aren't its batch's is a gap too."""
    rng = np.random.default_rng(15)
    rounds, runner = _rounds(2), Runner()
    run = PH.H1Run("t", SALT, {0: "q"}, [(32, K)], 32, 2, rounds=rounds, key=KEY)
    step = PHK.admitting(run, runner.execute_model, runner)
    so = _so(("a", [1], 2), running={"e": 3, "p": 1, "x": 1})
    so.scheduled_new_reqs += [SimpleNamespace(req_id="e", prompt_token_ids=None, prompt_embeds=object(),
                                              sampling_params=SimpleNamespace(max_tokens=2)),
                              SimpleNamespace(req_id="p", prompt_token_ids=[4], sampling_params=None, pooling_params=object())]
    step(so)
    run.record(0, *_tap(rng, 6, 32))
    step(_so(running={"a": 1}))
    run.record(0, *_tap(rng, 2, 32))
    run.close()
    assert [(e.index, e.slot) for e in run.requests] == [(0, 2)] and run.owners[0] == [None, None, 0, None, None, None]
    assert run.refused_requests == [{"round": 0, "reason": "no prompt token ids or no token cap"}] * 2
    gaps = ["step 0: 1 row of a request with no entry", "step 0: 1 row of a request with no entry",
            "step 0: 3 rows of a request with no entry",
            "step 1: 2 rows served and 1 in the step's batch, so the slot table can't name each served row's request"]
    assert run.request_gaps() == gaps and run.details["requests"]["gaps"] == gaps and run.credit()["requests"] == gaps
    why = _slot_reasons(run, rounds.challenger.commitment)
    assert why[:5] == [f"slot {p} of step 0 is a served row and names no admitted request: a request served without an "
                       "entry" for p in (0, 1, 3, 4, 5)]
    assert why[5:] == ["slot 1 of step 1 is a served row and names no admitted request: a request served without an entry"]


def test_the_engines_admissions_go_to_the_next_steps_round_and_past_the_window_are_refused():
    """`admitting` hands the scheduler's new requests and batch order to the run before the step's forward, a dummy run or
    an empty step none; a request admitted past the window's S steps, or with no prompt, gets no entry and is named; an
    admission after the window closes is refused."""
    rng = np.random.default_rng(13)
    run = PH.H1Run("t", SALT, {0: "q"}, [(32, K)], 32, 2, rounds=_rounds(2), key=bytes(32), admission=RQ.PER_ROUND)
    seen, runner = [], Runner()
    step = PHK.admitting(run, lambda so, *a, **kw: seen.append(so), runner)
    step(_so(("p", [1], 1)), None, True)
    step(_so(("q", [2], 1)), dummy_run=True)
    step(_so(("r", [3], 1), tokens=0))
    assert run.requests == [] and len(seen) == 3
    step(_so(("s", [4, 5], 2), ("t", [], 2)))
    run.record(0, *_tap(rng, 2, 32))
    step(_so(("u", [6], 1), running={"s": 1}))
    run.record(0, *_tap(rng, 2, 32))
    step(_so(("v", [7], 1), running={"s": 1}))
    run.close()
    assert [(e.prompt, e.round, e.slot) for e in run.requests] == [((4, 5), 0, 0), ((6,), 1, 1)]
    assert run.refused_requests == [{"round": 0, "reason": "no prompt token ids or no token cap"},
                                    {"round": 2, "reason": "admitted for step 2, past the window's 2 steps"}]
    assert {k: v for k, v in run.details["requests"].items() if k != "a_roots"} == {
        "admitted": 2, "refused": run.refused_requests, "per_round": [1, 1], "readmitted": 0,
        "gaps": ["step 2: 1 row of a request with no entry"], "coins": "live", "coin_source": "in-process",
        "admission": RQ.PER_ROUND}
    with pytest.raises(RuntimeError, match="an admission after the window closed"):
        step(_so(("w", [8], 1)))


def test_the_batch_order_is_model_runner_v2s():
    """`batch_order` lays a step's requests out as Model Runner V2's `sort_batch_req_ids` does at d9105ea8: verification
    rows first, then decodes, then the rest by rows, ties in scheduler order; a runner module's own sort is used when it
    has one."""
    so = SimpleNamespace(num_scheduled_tokens={"p": 5, "d": 1, "v": 3, "x": 2, "e": 1},
                         scheduled_spec_decode_tokens={"v": [7, 8]})
    assert RQ.batch_order(Runner(), so) == [("v", 3), ("d", 1), ("e", 1), ("x", 2), ("p", 5)]
    own = type("ModelRunner", (), {"__module__": __name__, "decode_query_len": 1})()
    globals()["sort_batch_req_ids"] = lambda tokens, drafts, q: sorted(tokens)
    try:
        assert [r for r, _n in RQ.batch_order(own, so)] == ["d", "e", "p", "v", "x"]
    finally:
        del globals()["sort_batch_req_ids"]


AUDITOR = "vy-challenger-1:7811"


def _host(tmp_path, monkeypatch, name="auditor", seed=1, rounds=3, challenger=None, session=CH.session_id(1),
          leaves=None, ledger=None, gated=True, reserved=True):
    """A challenger host serving one challenge phase over a live epoch of `rounds` round coins, reached at `AUDITOR`, whose coin
    commitment `SESSION` (ncp-v2's gate answer) names; `RESERVED` (H1's reserve answer) names its challenge phase, which the
    ledger `ledger` (its own, `<name>.jsonl`, unless given) reserves (unless not `reserved`), and `_gate` opens it, the
    ledger opening it for the H1 window's open root (unless not `gated`, for `CH.WINDOW` at once)."""
    host = CH.Host(tmp_path / name, challenger or Wd.live_challenger(0, rounds, CH.entropy(seed)), seed, session, leaves,
                   ledger, gated=gated, reserved=reserved)
    monkeypatch.setattr(R.RemoteChallenger, "connect",
                        classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
    monkeypatch.setitem(SESSION, "commitment", host.challenger.commitment.root.hex())
    monkeypatch.setitem(RESERVED, "session", session)
    monkeypatch.setitem(_GATES, "host", host)
    return host


_GATES: dict = {}


def _gate(session, root):
    """The gate (`pouw_h1_window.Gate`) of the host `_host` made last."""
    return _GATES["host"].open(session, root)


def _serve_window(settings):
    """`test_the_option_serves_the_arms_y_and_records_each_call`'s two served steps of a 3-step window, under `settings`:
    the installed option's commitment and its run."""
    model, runner = Model(Linear(32, 1), Linear(40, 2)), settings["pouw"]["runner"]
    comp = PO.install(model, PO.Context(stage="serve", enabled=(_entry(),), settings=settings))
    try:
        g = torch.Generator().manual_seed(9)
        for so, rows in ((_so(("a", [5], 4)), (1, 5)), (_so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}),
                                                        (33, 2))):
            runner.execute_model(so)
            for m, layer in zip(rows, model.layers):
                layer.quant_method.apply(layer, (torch.randn(m, K, generator=g) * 3).bfloat16())
        run = pouw.h1_run(_entry())
    finally:
        comp.close()
    return comp.installed[0].commitment(), run


#: the challenger gate's open answer for the window: its Commit root and coin commitment (the host's, `_host`)
SESSION = {"session": "s20261020T000000Z-000a", "seq": 0, "root": "ca" * 64, "commitment": "c0" * 64}
#: the gate's reserve answer for an H1 window (its challenge phase `_host`'s), which the window opens itself (`_gate`)
RESERVED = {"session": CH.session_id(1), "seq": 0, "peer": "vy-nebius-1", "by": "vllm", "expires_utc": "2026-10-20T11:18:00Z"}


def _remote(**kw):
    return {"pouw": {"rows": 64, "steps": 3, "runner": Runner(), "challenger": AUDITOR, "session": RESERVED,
                     "gate": _gate, "admission": "per-round", **kw}}


def test_h1_takes_its_salt_and_round_coins_from_the_auditor_hosts_challenger(tmp_path, monkeypatch):
    """With the setting ``challenger`` the window's one challenge phase on the auditor host, reserved on the gate, is opened on the
    window's open root once round 0 is admitted (over the weights' root, round 0's request table and the challenge phase), its
    salt coin after that root and round s's coin after round s's root, and closes once the window has finished; the
    verifier, reading the challenger's own directory under its key, trusts the window's coins (`trusted_rounds_reason`)
    only with the host's pins log, and every request entry and slot still opens under its round's root."""
    host = _host(tmp_path, monkeypatch)
    assert PO.refusals(PO.Context(stage="serve", enabled=(_entry(),), settings=_remote())) == ()
    commitment, run = _serve_window(_remote())
    assert host.ended() is None and run.details["challenger"] == {"closed": True}
    coins = host.challenger.commitment
    assert commitment["coin_source"] == "challenger" and commitment["coins"] == commitment["round_coins"] == coins.to_json()
    root = PS.window_open_root(bytes.fromhex(commitment["weights_root"]), bytes.fromhex(commitment["requests_root"]),
                                 CH.session_id(1))
    assert host.window == {"root": root.hex()} and commitment["rounds"][0]["requests"] == commitment["requests_root"]
    assert PHC.trusted_rounds_reason(commitment, host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=CH.MAIN) is None
    untrusted = "the challenger's session is not trusted: "
    assert PHC.trusted_rounds_reason(commitment, host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=None) == untrusted + LC.NO_MAIN
    assert PHC.trusted_rounds_reason(commitment, host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=CH.MAIN[:1]) == (
        untrusted + LC.first_parent_reason(host.session_id, CH.PINNED, list(CH.MAIN[:1])))
    received = R.released_roots(host.directory / R.RELEASES, coins)
    assert received[Wd.SALT] == (root,)
    assert commitment["session"] == {"session": CH.session_id(1), "seq": 1, "root": root.hex(),
                                     "commitment": coins.root.hex()}
    assert received[Wd.DRAW] == (bytes.fromhex(commitment["draw"]["root"]),)
    assert received[Wd.ROUND] == tuple(bytes.fromhex(r["root"]) for r in commitment["rounds"]) and len(received[Wd.ROUND]) == 3
    assert all(RQ.request_reason(commitment["rounds"], coins, run.salt, 64, e, run.open_request(e.index),
                                 run.open_first(e.index), admission=RQ.PER_ROUND) is None for e in run.requests)
    assert len(run.requests) == 3 and _slot_reasons(run, coins) == [] and run.request_gaps() == []
    pin = hashlib.sha512(host.key.public).hexdigest()
    assert PHC.pinned_key(host.directory / R.KEY, pin) == host.key
    with pytest.raises(CoinError, match="not the auditor's pinned"):
        PHC.pinned_key(host.directory / R.KEY, "00" * 64)


def test_the_record_is_one_manifest_under_its_coins(tmp_path, monkeypatch):
    """`pouw_h1_challenger.manifest_reason` holds a served window's record to its coins: the honest record passes, and one with a
    call's table root, a served tree's entry or a round's root changed is refused."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    _host(tmp_path, monkeypatch)
    _serve_window(_remote(transcript_dir=str(tmp_path / "w"), served=SV.Served(window=bytes(64))))
    record = json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text())
    assert PHC.manifest_reason(record) is None and record["commitment"]["served"]["refused"]

    def edited(edit):
        doc = json.loads(json.dumps(record))
        edit(doc)
        return PHC.manifest_reason(doc)
    first = next(i for i, c in enumerate(record["calls"]) if c.get("tables"))
    step = record["calls"][first]["step"]
    assert f"step {step}'s calls in the record" in edited(lambda d: d["calls"][first]["tables"].update(a="00" * 64))
    assert f"of step {step} has no A table root" in edited(lambda d: d["calls"][first].pop("tables"))
    assert "which no round of the log closes" in edited(lambda d: d["commitment"]["rounds"].pop())
    assert "calls_root is not over" in edited(lambda d: d["commitment"]["served"].update(refused=[]))
    assert "round 1's root is not over" in edited(lambda d: d["commitment"]["rounds"][1].update(root="00" * 64))
    assert "salt can't be derived" in edited(lambda d: d["commitment"].pop("salt_opening"))


def test_a_slot_table_naming_leaves_of_an_a_table_no_round_root_commits_is_refused(tmp_path, monkeypatch):
    """A window whose served calls commit no A table has round roots over calls without one, each opened in order before
    its coin, and a calls_root over the same rows, so its slot tables name A-table leaves that no round root commits: the
    run says so (``a_roots``) and `manifest_reason` refuses its record."""
    _host(tmp_path, monkeypatch)
    real = PH.H1Run._commit_tables

    def bare(self, call, *args):
        real(self, call, *args)
        if call.m:
            self.tables[call.index] = {**self.tables[call.index], "a": None}
    monkeypatch.setattr(PH.H1Run, "_commit_tables", bare)
    _commitment, run = _serve_window(_remote(transcript_dir=str(tmp_path / "w")))
    record = json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text())
    first = next(c for c in record["calls"] if c["served"])
    want = f"call {first['index']} of step {first['step']} has no A table root"
    assert run.details["requests"]["a_roots"].startswith(f"unbound: {want}")
    assert PHC.manifest_reason(record).startswith(want)


def _salted(tmp_path, monkeypatch, name, seed, honest, opened_on, salt_after, reserved=True):
    """`honest` (a served window's commitment) with its coins swapped for those of challenge phase `CH.session_id(seed)`, which
    the auditor's ledger reserved (unless not `reserved`) and opened on `opened_on` and whose salt coin opened after
    `salt_after`, then its round and flip coins after the window's own roots; and that challenge phase's host (`name`)."""
    sid = CH.session_id(seed)
    host = _host(tmp_path, monkeypatch, name=name, seed=seed, session=sid, reserved=reserved)
    answer = host.open(sid, opened_on.hex())
    salt = host.client.open(Wd.SALT, 0, salt_after)
    log = [{**r, "coin": host.client.open(Wd.ROUND, s, bytes.fromhex(r["root"])).to_json()}
           for s, r in enumerate(honest["rounds"])]
    root = RQ.draw_root(bytes.fromhex(honest["calls_root"]), bytes.fromhex(honest["draw"]["receipts"]), log)
    draw = host.client.open(Wd.DRAW, 0, root)
    host.client.close()
    assert host.ended() is None
    coins = host.challenger.commitment.to_json()
    return {**honest, "coins": coins, "round_coins": coins, "salt_opening": salt.to_json(), "rounds": log,
            "session": PHC.carried(answer), "draw": {**honest["draw"], "root": root.hex(), "coin": draw.to_json()}}, host


def test_a_windows_salt_opens_only_after_its_own_window_open_root(tmp_path, monkeypatch):
    """A served window's salt coin opens after its own open root, `salt.window_open_root` of its weights' root,
    its round-0 request table's root (committed before the salt, round 0's ``requests``) and its challenge phase, the root the
    auditor's ledger opened that challenge phase on (`core_check.salt`). Refused: a challenge phase the ledger opened on another window's
    root, on the bare weights' root or on another challenge phase's open root, whatever its salt followed; a challenge phase opened on its
    own root whose salt followed
    another window's root or the bare weights' root; a window whose record names another round-0 table or another
    challenge phase; and a round 0 whose request table isn't the one the open root is over. Each forged challenge phase's other coins
    follow the window's own roots, so only the salt is off."""
    a = _host(tmp_path, monkeypatch, name="a", seed=1, session=CH.session_id(0xa))
    honest, _ = _serve_window(_remote())
    assert a.ended() is None
    assert PHC.trusted_rounds_reason(honest, a.directory, [a.ledger.path], a.window, a.key,
                                     pins=[a.ledger.pins_path], main=CH.MAIN) is None
    weights, requests = bytes.fromhex(honest["weights_root"]), bytes.fromhex(honest["requests_root"])
    other = bytes.fromhex(a.window["root"])

    def why(name, seed, opened_on, salt_after):
        commits, host = _salted(tmp_path, monkeypatch, name, seed, honest, opened_on, salt_after)
        return PHC.trusted_rounds_reason(commits, host.directory, [host.ledger.path], host.window, host.key,
                                         pins=[host.ledger.pins_path], main=CH.MAIN)
    def opens(seed):
        return PS.window_open_root(weights, requests, CH.session_id(seed))
    assert why("own", 2, opens(2), opens(2)) is None
    assert why("other", 3, other, other) == f"the window's salt: {CC.OTHER_WINDOW}"
    assert why("bare", 4, weights, weights) == f"the window's salt: {CC.OTHER_WINDOW}"
    assert why("late", 5, opens(5), other) == f"the window's salt: {CC.NOT_RECEIVED}"
    assert why("early", 6, opens(6), weights) == f"the window's salt: {CC.NOT_RECEIVED}"
    assert why("seventh", 7, opens(2), opens(2)) == f"the window's salt: {CC.OTHER_WINDOW}"

    def relabeled(**kw):
        return PHC.trusted_rounds_reason({**honest, **kw}, a.directory, [a.ledger.path], a.window, a.key,
                                         pins=[a.ledger.pins_path], main=CH.MAIN)
    assert relabeled(requests_root="ee" * 64) == f"the window's salt: {CC.OTHER_WINDOW}"
    assert relabeled(session={**honest["session"], "session": CH.session_id(2)}) == (
        f"the window records session {CH.session_id(2)!r}, not {CH.session_id(0xa)}, the one the auditor's ledger "
        "opened on its root")
    assert relabeled(rounds=[{**honest["rounds"][0], "requests": "ee" * 64}, *honest["rounds"][1:]]) == (
        "round 0's request table is not the one the window's open root is over (its requests_root)")


def test_a_reserved_session_opens_on_its_epochs_root_and_its_salt_coin_after_it(tmp_path, monkeypatch):
    """`pouw_h1_window.open_session`, Pearl-C's device run's live coins: the gate opens the reserved
    challenge phase on salt epoch k's root R over the weights' root W (`salt.epoch_open_root`), its commitment read from
    ``status`` when the open answers none, the challenger is reached on the answer's port once it accepts connections
    (`wait` between refusals), its coins are the salt coin alone (`window.SALT_FAMILIES`), and that coin opens after R,
    the salt over W; the run carries the answer without its secret. An
    answer naming another root or another challenge phase is refused before any coin, and so is a challenge phase of H1's layout (salt,
    draw and round coins: one a Pearl-C pass would leave incomplete), which is closed before its first coin."""
    from verity.core.protocols.pouw.audit import salt_of
    from verity_vllm.protocols.pouw.h1 import window as PHW

    salt_only = LC.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, CH.entropy(1))
    host = _host(tmp_path, monkeypatch, challenger=salt_only)
    sid, root, calls, waits = CH.session_id(1), bytes(range(64)), [], []

    def gate(*words):
        calls.append(words)
        if words[0] == "open":
            return {**host.open(words[2], words[4]), "commitment": None, "port": 7811, "secret": "ab" * 32}
        return {"commitment": host.challenger.commitment.root.hex()}
    reached, refusals = R.RemoteChallenger.connect, iter((True, True))

    def connect(cls, at, secret=None):
        if next(refusals, False):
            raise ConnectionRefusedError(at)
        return reached(at, secret)
    monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(connect))
    now = [0.0]

    def wait(s):
        waits.append(s)
        now[0] += s
    client, c, opening, salt, carried = PHW.open_session("pouw:test", gate, sid, root, 2, at="vy-challenger-1:1",
                                                         wait=wait, clock=lambda: now[0])
    client.close()
    r = PS.epoch_open_root(root, 2)
    assert host.ended() is None and c == host.challenger.commitment and Wd.salt_epoch_reason(c) is None
    assert calls == [("open", "--session", sid, "--root", r.hex()), ("status", sid)] and waits == [0.5, 0.5]
    assert LC.live_reason(c, opening, Wd.SALT, 0, r) is None and salt == salt_of(c, opening.coin, root)
    assert R.released_roots(host.directory / R.RELEASES, c) == {Wd.SALT: (r,)}
    assert "secret" not in carried and (carried["session"], carried["root"], carried["port"]) == (sid, r.hex(), 7811)
    for answer in ({"session": sid, "seq": 0, "root": root.hex(), "commitment": "c0" * 64},
                   {"session": CH.session_id(2), "seq": 0, "root": r.hex(), "commitment": "c0" * 64}):
        with pytest.raises(ValueError, match=f"not session {sid} opened on salt epoch 2's root"):
            PHW.open_session("pouw:test", lambda *w, a=answer: a, sid, root, 2, wait=waits.append)

    h1 = _host(tmp_path, monkeypatch, name="h1-layout", rounds=1)
    with pytest.raises(ValueError, match="is not a live window's: the coin commitment's coins are not a Pearl-C salt "
                                         "session's: its salt coin alone"):
        PHW.open_session("pouw:test", lambda *w: {**h1.open(sid, PS.epoch_open_root(root, 0).hex()), "port": None}, sid,
                         root, at=AUDITOR, wait=waits.append)
    assert h1.ended() is None and R.released_roots(h1.directory / R.RELEASES, h1.challenger.commitment) == {
        Wd.SALT: (), Wd.DRAW: (), Wd.ROUND: ()}


def test_a_session_whose_tunnel_never_listens_is_refused_by_name_once_its_wait_is_over(monkeypatch):
    """`pouw_h1_window.opened` on the injected clock: a session the gate opened whose port refuses every connection is
    tried every half second until `remote.TUNNEL_WAIT_S` has passed, then refused (`CoinError`, naming the address)
    with no hello sent and no coin opened."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    sid, root, tries, waits, now = CH.session_id(1), PS.epoch_open_root(bytes(range(64)), 0), [], [], [0.0]

    def gate(session, r):
        return {"session": session, "seq": 0, "root": r, "commitment": "c0" * 64, "port": 7811, "secret": "ab" * 32}

    def refused(cls, at, secret=None):
        tries.append((at, secret))
        raise ConnectionRefusedError(at)

    def wait(s):
        waits.append(s)
        now[0] += s
    monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(refused))
    with pytest.raises(CoinError, match=f"pouw:test: nothing listened at vy-challenger-1:7811 within {R.TUNNEL_WAIT_S} s"):
        PHW.opened("pouw:test", "vy-challenger-1:1", 0, gate, sid, root, "salt epoch 0's root", wait,
                   Wd.salt_epoch_reason, lambda: now[0])
    assert waits == [0.5] * (2 * R.TUNNEL_WAIT_S)
    assert tries == [("vy-challenger-1:7811", bytes.fromhex("ab" * 32))] * (2 * R.TUNNEL_WAIT_S + 1)


def test_a_session_the_ledger_opened_without_a_reserve_salts_no_window(tmp_path, monkeypatch):
    """A window's open root binds the challenge phase the auditor's signed reserve entry names (S, `core_check.salt`), so
    a challenge phase the ledger opened on the window's own open root without reserving it first is refused, its salt and every
    other coin following the window's own roots."""
    a = _host(tmp_path, monkeypatch, name="a", seed=1, session=CH.session_id(0xa))
    honest, _ = _serve_window(_remote())
    assert a.ended() is None
    own = PS.window_open_root(bytes.fromhex(honest["weights_root"]), bytes.fromhex(honest["requests_root"]),
                              CH.session_id(2))
    commits, host = _salted(tmp_path, monkeypatch, "unreserved", 2, honest, own, own, reserved=False)
    assert [json.loads(line)["kind"] for line in host.ledger.lines] == ["open", "end"]
    assert PHC.trusted_rounds_reason(commits, host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=CH.MAIN) == (
        f"the window's salt: {CC.NO_RESERVE}")


def test_coins_from_a_prover_side_challenger_are_refused(tmp_path, monkeypatch):
    """The verifier trusts only the challenge phase the auditor's signed ledger opens and ends for the window, under the key the
    deployment pins, its record and log signed by its own challenger, and the roots that log shows: a window whose coins
    are held in the serving process, or come from a challenger host of the prover's under its own key, is refused; so is a
    missing ledger or an unsigned log, a ledger cut before the challenge phase's end, and a window whose round roots (each coin's
    ``after`` rewritten to match) or salt root aren't the ones the challenger received, or whose log is short."""
    auditor = _host(tmp_path, monkeypatch, gated=False)
    auditor.client.close()
    assert auditor.ended() is None
    local, _ = _serve_window(_settings())
    assert local["coin_source"] == "in-process"
    why = PHC.trusted_rounds_reason(local, auditor.directory, [auditor.ledger.path], CH.WINDOW, auditor.key,
                                    pins=[auditor.ledger.pins_path], main=CH.MAIN)
    assert why.startswith("the window's round_coins ") and "coins from a challenger the verifier doesn't hold" in why
    prover = _host(tmp_path, monkeypatch, name="prover", seed=2, session=CH.session_id(2))
    honest, _ = _serve_window(_remote())
    assert prover.ended() is None
    ours, w = prover.ledger.path, prover.window
    assert PHC.trusted_rounds_reason(honest, prover.directory, [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) is None
    assert PHC.trusted_rounds_reason(honest, prover.directory, [ours], w, auditor.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN).startswith(
        "the challenger's session is not trusted: the auditor's ledger entry 1 is signed under challenger key ")
    forged = CH.Ledger(tmp_path / "forged.jsonl", 2, named=auditor.key)
    forged.open(CH.session_id(2), w)
    forged.end(CH.session_id(2), prover.directory, True)
    assert PHC.trusted_rounds_reason(honest, prover.directory, [forged.path], w, auditor.key,
                                     pins=[forged.pins_path], main=CH.MAIN) == (
        "the challenger's session is not trusted: the auditor's ledger is refused: entry 0: its signature doesn't verify "
        "under the challenger key")
    assert PHC.trusted_rounds_reason(honest, prover.directory, [tmp_path / "none.jsonl"], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN).startswith(
        f"the challenger's session is not trusted: {LC.NO_AUDIT}")
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(ours.read_text().splitlines(keepends=True)[:2]))
    assert PHC.trusted_rounds_reason(honest, prover.directory, [cut], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == (
        f"the challenger's session is not trusted: the auditor's ledger has no end for session {CH.session_id(2)}: the "
        "session never ended, or the ledger is older than its end")
    (tmp_path / "empty").mkdir()
    assert PHC.trusted_rounds_reason(honest, tmp_path / "empty", [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == (
        "the challenger's session is not trusted: the challenger's record.json is not the one the auditor's ledger "
        "sealed at its session's end")
    close = prover.directory / R.CLOSE
    signed = close.read_bytes()
    close.unlink()
    assert PHC.trusted_rounds_reason(honest, prover.directory, [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == (
        "the challenger's session is not trusted: the challenger's challenger.json is not the one the auditor's ledger "
        "sealed at its session's end")
    close.write_bytes(signed)
    log = honest["rounds"]
    forged_rounds = [log[0], {**log[1], "root": "ee" * 64, "coin": {**log[1]["coin"], "after": "ee" * 64}}, log[2]]
    coins = prover.challenger.commitment
    assert RQ.rounds_reason(forged_rounds, coins) is None
    assert PHC.trusted_rounds_reason({**honest, "rounds": forged_rounds}, prover.directory, [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == \
        "the window's rounds: the round roots are not the roots the challenger's log shows it received, in order"
    assert PHC.trusted_rounds_reason({**honest, "rounds": log[:2]}, prover.directory, [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == \
        "the window's round log has 2 rounds, not its 3 steps"
    assert PHC.trusted_rounds_reason({**honest, "weights_root": "ee" * 64}, prover.directory, [ours], w, prover.key,
                                     pins=[prover.ledger.pins_path], main=CH.MAIN) == \
        f"the window's salt: {CC.OTHER_WINDOW}"


def test_a_second_session_under_the_same_key_for_one_window_is_refused(tmp_path, monkeypatch):
    """Two challenge phases under one challenger key, each honest, can't both pass for one window: the auditor's ledger opens one
    challenge phase a window, so the other's coins are refused for it, and a ledger opening a second challenge phase for a window
    trusts no window."""
    leaves = itertools.count()
    ledger = CH.Ledger(tmp_path / "windows.jsonl", 2, leaves)
    first = _host(tmp_path, monkeypatch, name="first", seed=2, session=CH.session_id(1), leaves=leaves, ledger=ledger)
    one, _ = _serve_window(_remote())
    assert first.ended() is None
    second = _host(tmp_path, monkeypatch, name="second", seed=2, session=CH.session_id(2), leaves=leaves, ledger=ledger)
    two, _ = _serve_window(_remote())
    assert second.ended() is None and first.key == second.key
    w, later = first.window, second.window
    assert w["root"] != later["root"]
    assert PHC.trusted_rounds_reason(one, first.directory, [ledger.path], w, first.key, pins=[ledger.pins_path],
                                     main=CH.MAIN) is None
    assert PHC.trusted_rounds_reason(two, second.directory, [ledger.path], later, first.key,
                                     pins=[ledger.pins_path], main=CH.MAIN) is None
    assert PHC.trusted_rounds_reason(two, second.directory, [ledger.path], w, first.key, pins=[ledger.pins_path],
                                     main=CH.MAIN) == (
        "the challenger's session is not trusted: the challenger's record.json is not the one the auditor's ledger sealed "
        "at its session's end")
    ledger.open("s-3", w)
    for coins, host, window in ((one, first, w), (two, second, later)):
        assert PHC.trusted_rounds_reason(coins, host.directory, [ledger.path], window, first.key,
                                         pins=[ledger.pins_path], main=CH.MAIN) == (
            f"the challenger's session is not trusted: the auditor's ledger is refused: entry 6: a second open for root "
            f"{w['root'][:16]}… or session s-3")


@pytest.mark.slow
def test_the_challenger_setting_is_h1s_one_coin_source(tmp_path, monkeypatch):
    """``challenger`` is HOST:PORT, the challenged schemes' only, never beside in-process ``coins`` or ``rounds`` and never at capture; a pipeline
    that is its own auditor adds no in-process coins beside it; a challenger whose layout isn't a live window of at least S
    rounds is refused at install, with its challenge phase closed."""
    with pytest.raises(ValueError, match="HOST:PORT"):
        pouw.Settings.of({"challenger": "7811"})

    def refused(settings, stage="serve", entry=None):
        return PO.refusals(PO.Context(stage=stage, enabled=(entry or _entry(),), settings=settings))
    assert any("from one source" in r for r in refused(_remote(coins=Challenger(0, CH.entropy(0)))))
    assert any("from one source" in r for r in refused(_remote(rounds=_rounds())))
    assert any("opens no session" in r for r in refused(_remote(), stage="capture"))
    assert any("'challenger' is for eps1-sm120-h1fx1, ncp-v2, pearl-c-sm120-v1-h3" in r
               for r in refused({"pouw": {"challenger": AUDITOR, "session": SESSION}}, entry="pouw:ncp-v1"))
    assert any("needs 'session'" in r for r in refused(_remote(session=None)))
    assert any("'session' is the session of the challenger" in r for r in refused(_settings(session=SESSION)))
    itself = "the window opens its session itself, on its window-open root"
    assert any(itself in r for r in refused(_remote(gate=None))) and any(itself in r for r in refused(_remote(session=SESSION)))
    assert any("'session' is the gate's open answer" in r for r in
               refused({"pouw": {"challenger": AUDITOR, "session": RESERVED}}, entry="pouw:ncp-v2"))
    assert any("'gate' opens a reserved session" in r for r in
               refused({"pouw": {"challenger": AUDITOR, "session": SESSION, "gate": _gate}}, entry="pouw:ncp-v2"))
    with pytest.raises(ValueError, match="pouw setting 'gate'"):
        pouw.Settings.of({"gate": "ssh"})
    with pytest.raises(ValueError, match="H1's pouw setting 'session' is the challenger gate's reserve answer"):
        pouw.Settings.of({"challenger": AUDITOR, "session": {"session": "s/1"}})
    with pytest.raises(ValueError, match="pouw setting 'session'"):
        pouw.Settings.of({"challenger": AUDITOR, "session": {"id": "s"}})
    assert "coins" not in PO.own_auditor({"pouw": {"challenger": AUDITOR}}, SimpleNamespace(protocols=(_entry(),)))["pouw"]
    assert "coins" in PO.own_auditor({"pouw": {}}, SimpleNamespace(protocols=(_entry(),)))["pouw"]
    short = _host(tmp_path, monkeypatch, rounds=2)
    with pytest.raises(ValueError, match="its 2 round coins are fewer than the window's 3 rounds"):
        _serve_window(_remote())
    assert short.ended() is None
    other = LC.LiveChallenger("vllm/commit", 0, [("replay", 2)], CH.entropy(3))
    vllm = _host(tmp_path, monkeypatch, name="vllm", seed=3, challenger=other)
    with pytest.raises(ValueError, match="not a live window's: the coin commitment's scheme is vllm/commit"):
        _serve_window(_remote())
    assert vllm.ended() is None


@pytest.mark.parametrize("n,width,nonce", [(1, 7, 0), (64, 68, RQ.REQUESTS_NONCE + 3), (37, 6144, RQ.TABLES_NONCE + 3 * 2175 + 2)])
def test_hiding_rows_is_core_commit_rows_with_its_salts_drawn_as_one_stream(n, width, nonce):
    """`pouw_h1_requests.hiding_rows`, which H1's tables commit through, is `audit.commit_rows` under hm96 at the same salt
    key and nonce, leaf for leaf: its one RFC 8439 stream is `audit.hm96_salt`'s chunks."""
    from verity.core.protocols.pouw.audit import HM96, commit_rows
    rng, key = random.Random(n), hashlib.sha256(b"key").digest()
    values = [rng.randbytes(width) for _ in range(n)]
    want = commit_rows(SALT, 5, values, hash=HM96, salt_key=key, nonce=nonce)
    got = RQ.hiding_rows(SALT, 5, values, key, nonce)
    assert got.commitment == want.commitment and all(got.open(i) == want.open(i) for i in range(n))
