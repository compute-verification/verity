"""The accounting layer (`benchmarks/datacenter`) reads the MLP: the same values, the same crossings, a sound account."""
import pytest

D = pytest.importorskip("datacenter", reason="benchmarks/datacenter isn't importable (it lands with PR #2475)")

from workload import accounting as A  # noqa: E402
from workload import model as M  # noqa: E402
from workload.block import Context  # noqa: E402

SPEC = {"d": 2, "layers": 3, "requests": 2}
CLOCK = D.Clock(1e-3)


def adapted(**config):
    ctx = Context({"spec": SPEC, "blocks": 3, "devices": 2, "frame_bytes": 16, "seed": 1, **config})
    c = A.circuit(ctx.model)
    dc = A.datacenter(ctx.hw, CLOCK)
    return ctx, c, A.placement(c, ctx.hw, ctx.placement, dc)


def test_the_circuit_has_the_programs_values():
    ctx, c, _ = adapted()
    x, ws = M.requests(ctx.spec, 1), M.weights(ctx.spec, 1)
    t = ctx.model.bind_inputs(x, ws)
    ctx.model.evaluate(t)
    assert c.evaluate(A.inputs(ctx.model, x, ws)) == t
    assert c.boolean_size == sum(k == "computed" for k in ctx.model.kind)


@pytest.mark.parametrize("config", [{}, {"plan": "tensor", "blocks": 2, "spec": dict(SPEC, groups=2)}])
def test_both_layers_derive_the_same_crossings(config):
    ctx, _, dp = adapted(**config)
    assert dp.datacenter.validate() == [] and dp.paced()
    assert A.crossing_pairs(dp) == {(e.value, e.dst) for e in ctx.edges}


def test_an_honest_run_is_audited_and_credited_to_the_blocks_it_ran_on():
    ctx, c, dp = adapted()
    x, ws = M.requests(ctx.spec, 1), M.weights(ctx.spec, 1)
    trace = D.run(dp, A.inputs(ctx.model, x, ws), frame_bits=32, bucket=1, window=4, clock=CLOCK)
    claims = D.audit(trace, D.occupancy(dp, 32, 1, 4), syncs=128)
    assert not [k for k in claims if isinstance(k, D.Refused)]
    acc = D.account(claims, dp, CLOCK, D.Interval(0, 8))
    assert acc.sound
    per_block = {u: sum(ctx.model.kind[v] == "computed" for v in ctx.held(u)) for u in ctx.hw.units}
    assert {b.block: b.credited for b in acc.blocks} == per_block
    assert sum(acc.input_bits.values()) == ctx.spec.requests * ctx.spec.d * ctx.spec.act_bits


def test_a_covert_frame_is_refused():
    ctx, _, dp = adapted()
    x, ws = M.requests(ctx.spec, 1), M.weights(ctx.spec, 1)
    link = ("b0.gpu0", "b0.cert")
    trace = D.run(dp, A.inputs(ctx.model, x, ws), frame_bits=32, bucket=1, window=4, clock=CLOCK,
                  deviation=D.Deviation(extra={(link, 2): 1}))
    refused = [k for k in D.audit(trace, D.occupancy(dp, 32, 1, 4)) if isinstance(k, D.Refused)]
    assert refused and {k.what for k in refused} == {f"link {link} window 0"}
