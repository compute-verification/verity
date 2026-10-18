"""The accounting layer (`benchmarks/datacenter`) reads the MLP: the same values, the same crossings, a sound account."""
import pytest

D = pytest.importorskip("datacenter", reason="benchmarks/datacenter isn't importable (it lands with PR #2475)")

from workload import accounting as A  # noqa: E402
from workload import model as M  # noqa: E402
from workload.block import Context  # noqa: E402

SPEC = {"d": 2, "layers": 3, "requests": 2}
GRID = D.Grid(frame_bits=32, bucket=1, window=4, syncs=128, tick=D.Assumption("tick", 1e-3, "s", "declared"))


def adapted(**config):
    ctx = Context({"spec": SPEC, "blocks": 3, "devices": 2, "frame_bytes": 16, "seed": 1, **config})
    c = A.circuit(ctx.model)
    dc = A.datacenter(ctx.hw, GRID)
    return ctx, c, A.placement(c, ctx.hw, ctx.placement, dc)


def honest(ctx, dp, deviation=None):
    x, ws = M.requests(ctx.spec, 1), M.weights(ctx.spec, 1)
    plan = D.Plan(A.epoch(dp))
    profile = dp.public(plan)
    return D.audit(D.run(dp, A.inputs(ctx.model, x, ws), plan, deviation), profile), profile


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
    claims, profile = honest(ctx, dp)
    assert not [k for k in claims if isinstance(k, D.Refused)]
    acc = D.account(claims, profile)
    assert acc.sound
    per_block = {u: sum(ctx.model.kind[v] == "computed" for v in ctx.held(u)) for u in ctx.hw.units}
    assert {b.block: b.credited for b in acc.blocks} == per_block
    assert sum(acc.input_bits.values()) == ctx.spec.requests * ctx.spec.d * ctx.spec.act_bits
    closed = D.account(claims, profile, assume=(D.FLEET_CLOSED,))
    assert all(b.local_at_least == b.credited for b in closed.blocks)


def test_a_covert_frame_is_refused():
    ctx, _, dp = adapted()
    port = D.Port("b0.cert", "egress")
    claims, _ = honest(ctx, dp, D.Deviation(extra={(port, 2): 1}))
    refused = [k for k in claims if isinstance(k, D.Refused)]
    assert refused and {repr(k.on) for k in refused} == {"port b0.cert:out [0, 4)"}
