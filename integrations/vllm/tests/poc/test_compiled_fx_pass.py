"""CPU tests for the W1d record-insertion pass on toy aten graphs (no GPU, no vLLM).

The pass (``verity_vllm_adapter.compiled_fx.fx_pass.RecordInsertionPass``) must
* insert ``verity::record`` only after buffers Inductor materialises anyway: extern-kernel outputs,
  extern-kernel inputs, graph outputs, activation placeholders (symbolic leading dim);
* never record weights (static placeholders) or values internal to a pointwise/reduction chain;
* leave the graph executable (side-effecting op, no outputs) with one call per id per run;
* be deterministic (same graph -> same ids) and a no-op in dry-run mode.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402

from verity_vllm_adapter.compiled_fx import fx_pass, record_op  # noqa: E402


def toy(x, w, r):
    h = torch.mm(x, w)  # extern kernel (aten.mm): output materialised
    y = h + r  # fused pointwise: y is also a graph output -> materialised
    v = y * y
    m = v.mean(dim=-1, keepdim=True)
    z = torch.rsqrt(m + 1e-5) * y  # z is a graph output
    return z, y


def trace_toy(symbolic: bool):
    """Like vLLM's pieces: activations ``x``/``r`` have a symbolic leading dim (num_tokens), the weight
    ``w`` is static."""
    x = torch.randn(6, 8)
    w = torch.randn(8, 8)
    r = torch.randn(6, 8)
    if not symbolic:
        gm = make_fx(toy, tracing_mode="fake")(x, w, r)
        return gm, (x, w, r)
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext

    fm = FakeTensorMode(shape_env=ShapeEnv())
    ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC])
    fx_, fw, fr = fm.from_tensor(x, symbolic_context=ctx), fm.from_tensor(w), fm.from_tensor(r, symbolic_context=ctx)
    with fm:
        gm = make_fx(toy, tracing_mode="real")(fx_, fw, fr)
    return gm, (x, w, r)


def run_pass(gm, **kw):
    table = fx_pass.FxIdTable()
    p = fx_pass.RecordInsertionPass(table, **kw)
    fx_pass.CURRENT_PIECE.update({"index": 3, "submod_name": "submod_3"})
    p(gm.graph)
    gm.recompile()
    return table, p


def record_nodes(gm):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is record_op.overload()]


def test_inserts_after_materialised_buffers_only():
    gm, args = trace_toy(symbolic=True)
    table, _ = run_pass(gm)
    recs = record_nodes(gm)
    roles = {e.node: e.role for e in table.entries.values()}
    targets = {e.node: e.target for e in table.entries.values()}
    # extern output (mm), graph outputs (add -> y, mul -> z); x is an activation placeholder
    assert set(targets.values()) >= {"aten.mm", "aten.add", "aten.mul"}
    assert "aten.mean" not in targets.values() and "aten.rsqrt" not in targets.values()
    assert any(r == "extern_out" for r in roles.values())
    assert sum(1 for r in roles.values() if r == "graph_out") == 2
    inputs = [e for e in table.entries.values() if e.role == "input"]
    assert len(inputs) == 2  # x and r have a symbolic leading dim; w (8x8, static) is a weight -> never recorded
    assert all(e.target == "placeholder" for e in inputs)
    assert len(recs) == len(table.entries) == 5
    ids = sorted(e.id for e in table.entries.values())
    assert ids == [3 * fx_pass.ID_STRIDE + k for k in range(5)]
    # every record sits right after the node it observes (or after the last placeholder for inputs)
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    for n in recs:
        base = n.args[0]
        assert order[n] > order[base]
        assert n.args[1] in table.entries and table.entries[n.args[1]].node == base.name


def test_graph_runs_and_records_once_per_id():
    gm, args = trace_toy(symbolic=True)
    table, _ = run_pass(gm)
    seen = []
    record_op.SINK.reset()
    record_op.SINK.enabled = True
    record_op.SINK.snapshot = lambda t: t.detach().clone()
    try:
        z, y = gm(*args)
        seen = list(record_op.SINK.records)
    finally:
        record_op.SINK.enabled = False
        record_op.SINK.snapshot = None
        record_op.SINK.reset()
    zr, yr = toy(*args)
    assert torch.equal(z, zr) and torch.equal(y, yr)  # observing does not change the computation
    assert sorted(r.id for r in seen) == sorted(table.entries)
    by_id = {r.id: r for r in seen}
    for e in table.entries.values():
        rec = by_id[e.id]
        assert list(rec.shape) == [6 if d == "s0" or isinstance(d, str) else d for d in e.shape] or len(rec.shape) == len(e.shape)
    y_entry = next(e for e in table.entries.values() if e.target == "aten.add")
    assert torch.equal(by_id[y_entry.id].host, yr)


def test_static_graph_has_no_input_records():
    gm, _ = trace_toy(symbolic=False)
    table, _ = run_pass(gm)
    assert all(e.role != "input" for e in table.entries.values())
    assert len(table.entries) == 3  # mm out, y, z


def test_dry_run_and_determinism():
    gm1, _ = trace_toy(symbolic=True)
    before = [n.name for n in gm1.graph.nodes]
    table_dry, _ = run_pass(gm1, dry_run=True)
    assert [n.name for n in gm1.graph.nodes] == before
    assert len(table_dry.entries) == 5
    gm2, _ = trace_toy(symbolic=True)
    table2, _ = run_pass(gm2)
    assert [(e.id, e.node, e.role) for e in table_dry.entries.values()] == [(e.id, e.node, e.role) for e in table2.entries.values()]


def test_roles_filter():
    gm, _ = trace_toy(symbolic=True)
    table, _ = run_pass(gm, roles=("extern_out",))
    assert {e.role for e in table.entries.values()} == {"extern_out"}
    assert len(table.entries) == 1


def test_mutating_custom_op_output_is_recorded_after_the_op():
    lib = torch.library.Library("verity_test", "FRAGMENT")  # noqa: F841
    if not hasattr(torch.ops, "verity_test") or not hasattr(torch.ops.verity_test, "scale_out"):
        @torch.library.custom_op("verity_test::scale_out", mutates_args=("out",))
        def scale_out(x: torch.Tensor, out: torch.Tensor) -> None:
            out.copy_(x * 2)

        @scale_out.register_fake
        def _(x, out):
            return None

    def f(x):
        out = torch.empty_like(x)
        torch.ops.verity_test.scale_out(x, out)
        return out + 1

    gm = make_fx(f, tracing_mode="symbolic")(torch.randn(4, 3))
    table, _ = run_pass(gm, placement="producer")
    outs = [e for e in table.entries.values() if e.role == "extern_out"]
    assert len(outs) == 1 and outs[0].target in ("aten.empty_like", "aten.empty", "aten.empty_strided", "aten.empty_permuted")
    assert outs[0].inserted_after.startswith("scale_out")
    recs = record_nodes(gm)
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    op_node = next(n for n in gm.graph.nodes if "scale_out" in n.name)
    assert any(order[r] > order[op_node] and r.args[0].name == outs[0].node for r in recs)


def test_late_placement_puts_records_after_the_last_reader():
    """placement='late' (default): a record follows the buffer's last in-graph reader through views, so
    no reader is ordered behind the side-effecting call (Qwen2's fused bias-add + rope case)."""
    gm, _ = trace_toy(symbolic=True)
    table, _ = run_pass(gm)  # default placement
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    recs = {n.args[1]: n for n in record_nodes(gm)}
    out_node = gm.graph.output_node()
    for e in table.entries.values():
        rec = recs[e.id]
        base = rec.args[0]
        readers = [u for a in gm.graph.nodes if fx_pass.base_of(a) is base for u in a.users
                   if u is not out_node and u.target is not record_op.overload() and fx_pass.base_of(u) is not base]
        for u in readers:
            assert order[rec] > order[u], f"{e.node}: record before reader {u.name}"
    mm = next(e for e in table.entries.values() if e.target == "aten.mm")
    assert mm.extra.get("producer_point") is not None  # the mm output is read later by the rmsnorm chain: moved
    # the graph still computes the same thing
    gm2, args = trace_toy(symbolic=True)
    run_pass(gm2)
    record_op.SINK.enabled = False
    z, y = gm2(*args)
    zr, yr = toy(*args)
    assert torch.equal(z, zr) and torch.equal(y, yr)


def test_late_placement_not_moved_past_inplace_writer():
    if not hasattr(torch.ops, "verity_test") or not hasattr(torch.ops.verity_test, "scale_out"):
        @torch.library.custom_op("verity_test::scale_out", mutates_args=("out",))
        def scale_out(x: torch.Tensor, out: torch.Tensor) -> None:
            out.copy_(x * 2)

        @scale_out.register_fake
        def _(x, out):
            return None

    def f(x):
        out = torch.empty_like(x)
        torch.ops.verity_test.scale_out(x, out)
        y = out + 1
        torch.ops.verity_test.scale_out(y, out)  # writes `out` again after its read
        return y, out

    gm = make_fx(f, tracing_mode="symbolic")(torch.randn(4, 3))
    table, _ = run_pass(gm)
    outs = [e for e in table.entries.values() if e.node.startswith("empty")]
    assert len(outs) == 1
    # the record of `out` sits after the LAST writer (second scale_out), not after the intermediate reader
    writers = [n.name for n in gm.graph.nodes if "scale_out" in n.name]
    assert len(writers) == 2 and outs[0].inserted_after == writers[1]


def test_fake_impl_under_fake_mode():
    from torch._subclasses.fake_tensor import FakeTensorMode

    record_op.install()
    with FakeTensorMode():
        x = torch.empty(4, 4)
        assert torch.ops.verity.record.default(x, 7) is None


def test_explicit_inductor_lowering_is_invisible_to_other_nodes():
    """The record op must not join Inductor's ``needs_realized_inputs``/``fallbacks`` sets: membership
    changes how *other* nodes are lowered (``GraphLowering.run_node`` realises a pointwise producer early
    via ``realize_hint`` when one of its users needs realised inputs; ``fx_utils.is_node_realized`` feeds
    the reinplace pass). Registering the lowering explicitly keeps the op a plain FallbackKernel."""
    L = pytest.importorskip("torch._inductor.lowering")
    kind = record_op.install_inductor_lowering()
    op = record_op.overload()
    assert kind in ("explicit", "preexisting")
    assert op in L.lowerings
    assert getattr(L.lowerings[op], "_is_fallback_handler", False)
    if kind == "explicit":
        assert op not in L.needs_realized_inputs
        assert op not in L.fallbacks
        assert L.maybe_layout_constraints(op) is None
    assert record_op.lowering_kind() == kind
    # idempotent
    assert record_op.install_inductor_lowering() == kind


def test_explicit_lowering_compiles_and_records_on_cpu():
    """End to end on CPU: a compiled function containing verity::record runs, the record fires once per
    call with the producer's materialised value, and the op is not DCE'd."""
    torch._dynamo.reset()
    record_op.install_inductor_lowering()
    seen = []
    orig = record_op.SINK.on_record

    def spy(x, i):
        seen.append((int(i), x.detach().clone()))

    record_op.SINK.on_record = spy
    try:
        def f(x, w):
            h = torch.mm(x, w)
            y = h + 1.0
            torch.ops.verity.record.default(y, 3)
            return y * 2

        x, w = torch.randn(4, 5), torch.randn(5, 3)
        out = torch.compile(f, backend="inductor", dynamic=False)(x, w)
        ref = (torch.mm(x, w) + 1.0) * 2
        assert torch.allclose(out, ref)
        assert len(seen) == 1 and seen[0][0] == 3
        assert torch.allclose(seen[0][1], torch.mm(x, w) + 1.0)
    finally:
        record_op.SINK.on_record = orig


def test_explicit_lowering_reuses_the_extern_consumers_clone_on_cpu():
    """A pointwise value consumed by an extern op with a fixed-stride constraint is never a buffer of
    its own: Inductor materialises a *clone* for the extern (fused into the producer kernel). Recording
    the FX value after that extern must record the clone, not add a second materialisation (which
    changes the producer kernel's source: an extra store). The graph is lowered as given (make_fx +
    compile_fx_inner) so the record really sits after the extern, as the pass's late placement puts it."""
    C = pytest.importorskip("torch._inductor.config")
    from torch._inductor.compile_fx import compile_fx_inner

    record_op.install_inductor_lowering()
    if not hasattr(torch.ops, "verity_test") or not hasattr(torch.ops.verity_test, "consume"):
        @torch.library.custom_op("verity_test::consume", mutates_args=())
        def consume(x: torch.Tensor) -> torch.Tensor:
            return x.sum(dim=-1, keepdim=True) * 0 + x[:, :1]

        @consume.register_fake
        def _(x):
            return x.new_empty(x.shape[0], 1)

    def f(x, w):
        y = (x * w) + 1.0  # pointwise; its only in-graph reader is the extern below
        z = torch.ops.verity_test.consume(y)  # custom op: default needs_fixed_stride_order -> clone of y
        torch.ops.verity.record.default(y, 11)  # late placement: after the extern
        return (z,)

    x, w = torch.randn(4, 6), torch.randn(4, 6)
    gm = make_fx(f, tracing_mode="fake")(x, w)
    names = [n.name for n in gm.graph.nodes]
    assert names.index("consume") < names.index("record")
    seen = []
    orig = record_op.SINK.on_record
    record_op.SINK.on_record = lambda t, i: seen.append((int(i), t.detach().clone()))
    before = dict(record_op.LOWERING_STATS)
    old = (C.fx_graph_cache, C.force_disable_caches)
    C.fx_graph_cache, C.force_disable_caches = False, True
    try:
        compiled = compile_fx_inner(gm, [x, w])
        (out,) = compiled([x, w])
        assert torch.allclose(out, torch.ops.verity_test.consume(x * w + 1.0))
        assert len(seen) == 1 and seen[0][0] == 11 and torch.allclose(seen[0][1], x * w + 1.0)
        assert record_op.LOWERING_STATS["reused_clone"] == before["reused_clone"] + 1
        assert record_op.LOWERING_STATS["realised_here"] == before["realised_here"]
    finally:
        record_op.SINK.on_record = orig
        C.fx_graph_cache, C.force_disable_caches = old
