"""The accounting layer (`benchmarks/datacenter`) reads the MLP's Program and verity_vllm's serving Program themselves:
the same values, the same crossings, a sound account, and a covert frame refused."""
from collections import Counter

import pytest

D = pytest.importorskip("datacenter", reason="benchmarks/datacenter isn't importable")

from workload import accounting as A  # noqa: E402
from workload.block import Context  # noqa: E402

SPEC = {"d": 2, "layers": 3, "requests": 2}
CADENCE = D.Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                    tick=D.Assumption("tick", 1e-3, "s", "declared"))


def adapted(**config):
    ctx = Context({"spec": SPEC, "blocks": 3, "devices": 2, "frame_bytes": 16, "seed": 1, **config})
    run_inputs, registered = A.split(ctx.model, 1)
    dc = A.datacenter(ctx.hw, CADENCE, A.kinds(ctx.model))
    return ctx, run_inputs, A.placement(ctx.model, ctx.hw, ctx.placement, dc, D.register(ctx.model.program, registered))


def audited(dp, run_inputs, deviation=None):
    plan = D.Plan(A.epoch(dp))
    profile = dp.public(plan)
    trace = D.run(dp, run_inputs, plan, deviation)
    return trace, D.audit(trace, profile), profile


def test_the_accounting_layer_evaluates_the_programs_own_values_and_counts_its_gates():
    ctx, run_inputs, dp = adapted()
    t = ctx.model.evaluate(dict(ctx.model.inputs(1)))
    got, program = D.evaluate(ctx.model.program, A.parameters(ctx.model, 1)), ctx.model.program
    assert all(t[g] == x for g, x in got.items())
    assert {g for v in D.values(program) for g in A.gates(program, v)} <= set(got)
    total = Counter()
    for w in D.work(ctx.model.program).values():
        total.update(w)
    assert dict(total) == {"cuda:gate": sum(k == "computed" for k in ctx.model.kind)}


@pytest.mark.parametrize("config", [{}, {"plan": "tensor", "blocks": 2, "spec": dict(SPEC, groups=2)}])
def test_both_layers_derive_the_same_crossings(config):
    ctx, _, dp = adapted(**config)
    assert dp.datacenter.validate() == [] and dp.paced()
    assert A.crossing_pairs(dp) == {(e.value, e.dst) for e in ctx.edges}


def test_an_honest_run_is_audited_and_credited_to_the_blocks_it_ran_on():
    ctx, run_inputs, dp = adapted()
    _, claims, profile = audited(dp, run_inputs)
    assert not [k for k in claims if isinstance(k, D.Refused)]
    acc = D.account(claims, profile)
    assert acc.sound
    ands = D.ands(ctx.model.program)
    per_block = {u: sum(n for k, n in ands.items() if dp.block(dp.device[k]) == u) for u in ctx.hw.units}
    assert {b.block: b.credited for b in acc.blocks} == per_block and min(per_block.values()) > 0
    assert sum(acc.input_bits.values()) == ctx.spec.requests * ctx.spec.d * ctx.spec.act_bits
    closed = D.account(claims, profile, assume=(D.FLEET_CLOSED,))
    assert all(b.local_at_least == b.credited for b in closed.blocks)


def test_a_covert_frame_is_refused():
    _, run_inputs, dp = adapted()
    _, claims, _ = audited(dp, run_inputs, D.Deviation(extra={(D.Port("b0.cert", "egress"), 2): 1}))
    refused = [k for k in claims if isinstance(k, D.Refused)]
    assert refused and {repr(k.on) for k in refused} == {"port b0.cert:out [0, 4)"}


@pytest.mark.slow
def test_the_served_program_is_audited_and_its_macs_reported_uncredited():
    """verity_vllm's `Serve_v1` for a truncated SmolLM2, a pipeline stage a block: the accounting layer re-executes its
    word Definitions and gets the tokens the Program does. Its layer Definitions have no Boolean version, so no stage
    is credited ANDs and the account names why, and each stage's BF16 tensor-core products are reported beside it."""
    pytest.importorskip("torch")
    pytest.importorskip("verity_vllm")
    from workload import served as S
    cfg = S.Config()
    ctx, run_inputs, dp = adapted(workload="served", spec=cfg.to_json(), devices=1, frame_bytes=1024)
    assert dp.datacenter.validate() == [] and dp.paced()
    assert A.crossing_pairs(dp) == {(e.value, e.dst) for e in ctx.edges}
    trace, claims, profile = audited(dp, run_inputs)
    assert ctx.model.output_words(trace.values) == ctx.model.reference(1)
    acc = D.account(claims, profile)
    assert {c.part: c.work for c in claims if isinstance(c, D.Computed)} == {b.block: 0 for b in acc.blocks}
    assert acc.sound and all(b.macs.get("mac:bf16", 0) > 0 for b in acc.blocks)
    assert "no Boolean gate count" in profile.uncounted and f"nothing credited: {profile.uncounted}" in repr(acc)
    assert acc.input_bits == {"prompt": 32 * cfg.prompt}
