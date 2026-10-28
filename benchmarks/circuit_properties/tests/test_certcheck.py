"""`certcheck` on the Definitions the training step's consumers are read from: what a counted product's lane reads, a row
GEMM's coordinates and classes on each route the training step takes, an output's required gates against the explicit
checker, the interval overlap weights against `upstream.weights_by_multiplicity`, and (opt-in) the scaled step."""

from __future__ import annotations

import os
import random

import pytest
from circuit_properties import certcheck as CC
from circuit_properties import regions as RG
from circuit_properties import upstream as U
from circuit_properties.ops import flatten, op_shape, param_bases

from verity.ml.circuits.defs import bind
from verity.ml.circuits.layout import Circuit
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.definitions.registry import cublas_bf16 as CB


def _lane_params(fn) -> list[dict[int, set[int]]]:
    """Per lane, the parameter leaves its gates read, by parameter; and that they read no gate of the op outside the lane."""
    shape, flat, base = op_shape(fn, CC.SITE), flatten(fn), param_bases(fn)
    out = []
    for lane in range(shape.lanes):
        mine = set(shape.members[lane])
        assert not {o for g in mine for o in flat.operands[g] if o >= 0 and o not in mine}
        by: dict[int, set[int]] = {}
        for g in mine:
            for o in flat.operands[g]:
                if o < 0:
                    j = -1 - o
                    pi = max(i for i in range(len(base) - 1) if base[i] <= j)
                    by.setdefault(pi, set()).add(j - base[pi])
        out.append(by)
    return out


def _params(fn) -> list[str]:
    return [n for n, _t in fn.signature()[0]]


def test_a_whole_dot_lane_reads_exactly_its_x_and_w_elements():
    fn = bind(BG.HopperBF16WgmmaDot16)
    names = _params(fn)
    lanes = _lane_params(fn)
    assert len(lanes) == 16
    for k, by in enumerate(lanes):
        element = set(range(16 * k, 16 * k + 16))
        assert by == {names.index("a"): element, names.index("b"): element}


def _serial(K: int = 64, N: int = 3):
    return CB.route_gemm({"kernel": "serialk", "SPLIT_K": 2}, K, N)


#: the row GEMM's routes in the training step: unsplit, serial split-K (the M = 4096 LM head), split-K with bf16 partials
#: (the M = 256 MLP input gradients and LM head) and its aligned form (the M = 4096 weight gradients)
ROUTES = {"nosplit": lambda K, N: bind(BG.Gemm, K=K, N=N, DOT=BG.HopperBF16WgmmaDot16), "serialk": _serial,
          "splitk": lambda K, N: CB.route_gemm({"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"}, K, N),
          "aligned": lambda K, N: CB.route_gemm({"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16", "align": 8,
                                                 "tile_k": 32}, K, N)}


def test_a_dot_step_lane_reads_its_x_sign_and_its_w_element():
    gemm = _serial().body.nodes[0].fn
    steps = {node.fn.id: node.fn for node in gemm.body.nodes if CC.OPS.fullmatch(node.fn.id)}
    assert len(steps) == 2, steps
    for fn in steps.values():
        names = _params(fn)
        for k, by in enumerate(_lane_params(fn)):
            assert by == {names.index("a"): {16 * k + 15}, names.index("b"): set(range(16 * k, 16 * k + 16))}


def _row(fn):
    """The route as a Program, whose root is its body after its inputs: the row, and the gates of its x."""
    program = Program(fn, registered=("w",), theta=("w",))
    C = Circuit(program.root)
    x = next(j for j, node in enumerate(program.root.body.nodes) if node.name == "x")
    xs = C.root_scope.child(x, 0).offset
    return C, C.root_scope, range(xs, xs + 16 * fn.signature()[0][0][1].n)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_rows_coordinates_are_disjoint_and_read_x_through_their_products(route):
    K, N = 64, 3
    fn = ROUTES[route](K, N)
    C, row, x = _row(fn)
    res = RG.Resolver()
    spans = []
    for k in range(N):
        acts = CC.coordinate(row, k)
        assert all(not s.spec.id.startswith("HopperBf16Dot16StepX") for s in acts)
        assert sum(op_shape(op.spec, CC.SITE).lanes for s in acts for op in CC.ops(s)) == K
        spans += [(s.offset, s.offset + s.spec.gates) for s in acts]
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    first, last, n = CC.classes(row, res)
    assert n == N and len(first) == K
    xs = [sorted(g for g in c if g in x) for c in first]
    assert xs == [sorted(g for g in c if g in x) for c in last]
    want = [list(x[16 * k:16 * k + 16]) if route == "nosplit" else [x[16 * k + 15]] for k in range(K)]
    assert xs == want


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_an_output_bits_required_gates_agree_with_the_explicit_checker(route):
    fn = ROUTES[route](32, 2)
    C, row, _x = _row(fn)
    acts = CC.coordinate(row, 1)
    A = frozenset(g for s in acts for g in range(s.offset, s.offset + s.spec.gates))
    reads = {g: tuple(C.gate(g).operands) for g in A}
    rets = RG.returns(row)
    for bit in (0, 7, 15):
        out = rets[16 + bit]
        assert out in A
        got = RG.required_size(acts, [frozenset({out})]).required - 1
        G = U.required(reads, A, [U.into(reads, frozenset({out}))])
        assert got == len(G) > 0
        assert U.holds(reads, U.Consumers(A=(A,), G=(G,), theta=frozenset(), fins=((frozenset({out}),),)))


def test_interval_overlap_weights_are_the_multiplicity_weights():
    rng = random.Random(5)
    for _ in range(200):
        regions = {}
        for c in range(rng.randint(1, 6)):
            cuts = sorted(rng.sample(range(40), 2 * rng.randint(1, 3)))
            regions[c] = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
        sets = [frozenset(g for lo, hi in regions[c] for g in range(lo, hi)) for c in regions]
        got = CC.overlap_weights(regions)
        assert [got[c] for c in regions] == U.weights_by_multiplicity(sets)
        assert U.overlap_weights(sets, [got[c] for c in regions])[0]


def test_a_regions_intervals_merge_where_adjacent_and_refuse_an_overlap():
    assert CC.intervals([(5, 9), (0, 3), (3, 5), (12, 12), (10, 11)]) == [(0, 9), (10, 11)]
    with pytest.raises(ValueError):
        CC.intervals([(0, 4), (3, 6)])


@pytest.mark.skipif(not os.environ.get("VERITY_CIRCUIT_PROPERTIES_STEP"),
                    reason="builds the scaled TrainStep_v1 (about 3 min and 6 GB): set VERITY_CIRCUIT_PROPERTIES_STEP=1")
def test_the_scaled_steps_layer_uses_are_certified_and_disjoint_and_cover_its_linear_products():
    from circuit_properties import training as TR

    with TR.scaled():
        program = TR.step()
        S = CC.Step(program, TR.STEP["L"], TR.STEP["B"], TR.STEP["S"])
        att = S.whole("l0.attn", 0)
        q = [(r.offset, r.offset + r.spec.gates) for r in (S.row("l0.q", t) for t in range(S.M))]
        for m in (0, 3, 4, 17):
            g = S.res.param(att.child(0, m), 0, 0)
            assert [t for t, (lo, hi) in enumerate(q) if lo <= g < hi] == [m // TR.SCALED["NH"]]
        assert set(CC.overlap_weights(CC.spans(S, "layer", range(S.M))).values()) == {1}
        verdicts = CC.uses(S, [5], design="layer")
        assert all(v.required for v in verdicts), verdicts
        per_token = {v.kind: v.finishing_sets for v in verdicts if v.layer in (None, 0)}
        p = CC.coverage(S)
        assert S.M * (S.L * (per_token["fwd"] + per_token["bwd"]) + per_token["yn"] + per_token["dlogits"]) == \
            p["forward"] + p["input_gradient"]
        gemm = {v.kind: v for v in CC.uses(S, [5], [1], design="gemm", kinds={"y2", "mul"})}
        assert gemm["y2"].required and not gemm["mul"].required
        assert gemm["mul"].why == "no gate of the region reaches every class"
