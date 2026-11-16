"""verity_vllm.program.predict.dsa: what the sparse-attention decoder refuses, each by name.  A target whose device has
no tuned fused-MoE tile (`engine.profiles.devices.FUSED_MOE_CONFIGS`) is `Unexpressible` where the MoE block is written, and
nowhere before it; a node of such ranks (`node.build_node_program`) is refused in the same words.  A block's all-reduce on a
route that no capture covers under the engine's switches is `Unexpressible` naming the route and the switches; under the
row's own it fuses with its norm up to the rows vLLM hands FlashInfer.  A history opens once, before a layer's first call,
and an edge is every layer's rows at one length: a caller that breaks either gets the decoder's own `AssertionError`, which
`build_program`'s order of calls never does.

GLM-5.3's checkpoint is cut to its first 4 layers: three dense MLPs and then one MoE block, three full indexers and then one
that shares the third's top-k."""

from __future__ import annotations

import re
from collections import Counter

import pytest
from verity.circuits.defs import Builder
from verity.circuits.refs import Affine, Coll, runs
from verity.circuits.types import Array, RecordType, Value

from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_predict import GLM53, _row_phi
from verity_vllm.program.predict.build import build_program, decoder
from verity_vllm.program.predict.common import Unexpressible, _Peers
from verity_vllm.program.predict.node import build_node_program
from verity_vllm.program.predict.phi import Phi

LAYERS, H200, H100 = 4, "NVIDIA H200", "NVIDIA H100 80GB HBM3"
GROUPS = ("prefix", "prefix_ik", "prefix_iks")
NO_TILE = (f"vocabulary cannot express: no tuned fused-MoE config for E=256, N=256 on {H100!r} "
           f"(engine.profiles.devices.FUSED_MOE_CONFIGS)")
SWITCHES = {"custom_all_reduce": True, "symmetric_memory": True, "flashinfer": True, "batch_invariant": False, "nccl_transport": None}
NO_CAPTURE = " (no capture covers it: tp_allreduce.covered)"


@pytest.fixture
def cut(tmp_path):
    """GLM-5.3's φ on one rank at TP8, its checkpoint cut to its first `layers`, on a target that reports `device`."""

    def make(layers: int = LAYERS, device: str = H200) -> Phi:
        p = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), layers)
        assert p.hf_config["mlp_layer_types"] == (["dense"] * 3 + ["sparse"])[:layers] and p.target["device_name"] == H200
        return Phi.from_json({**p.to_json(), "target": {**p.target, "device_name": device}})

    return make


def _families(r) -> Counter:
    return Counter(n.fn.id.split("{", 1)[0] for n in r.program.fn.body.nodes)


def _model(phi: Phi, C: int):
    """The decoder's forward as `build_program` makes it, over a Program of one hidden row, a history of C positions
    (`Net.prefix_params`) and the weights, with no launch batch (a call is its own launch) -> the forward, the row, the history's
    three Inputs and the Builder."""
    net = decoder(phi)
    params = [("x", Array(int(net.C["H"]), Value(16))), *net.prefix_params(C), ("weights", RecordType(*net.leaves()))]
    B = Builder(params, Value(1))
    inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    return net.model(B, inputs[-1], _Peers([], net.world, net.rank)), inputs[0].refs, inputs[1:4], B


def _engine(phi: Phi, env: dict | None = None, **options) -> Phi:
    """φ with these engine options, and `env` set in the environment its engines run under."""
    return Phi.from_json({**phi.to_json(), "engine": {**phi.engine, **options, "env": {**phi.engine["env"], **(env or {})}}})


def _reduced(phi: Phi, M: int) -> Counter:
    """`reduce_norm` over M rows as a Program of its own (the rows' partials on this rank, their residuals, the partials on
    each of the 7 other ranks and the weights), with no launch batch -> its calls by family.  Each row's norm is a call of
    its own, the last M: its normed row is the call's first output and its residual row the second."""
    net = decoder(phi)
    H = int(net.C["H"])
    rows = Array(M, Array(H, Value(16)))
    params = [("parts", rows), ("res", rows), *((f"peers_{k}", rows) for k in range(net.world - 1)), ("weights", RecordType(*net.leaves()))]
    B = Builder(params, Value(1))
    inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    model = net.model(B, inputs[-1], _Peers(inputs[2:-1], net.world, net.rank))
    parts, res = ([c.refs.slice(i * H, (i + 1) * H) for i in range(M)] for c in inputs[:2])
    normed, residual = model.reduce_norm(parts, res, model.w[(0, "ln2")])
    at = len(B.nodes) - M
    assert [(runs(n), runs(r)) for n, r in zip(normed, residual)] == [([("n", at + i, 0, H)], [("n", at + i, H, 2 * H)]) for i in range(M)]
    assert model.peers.n == net.world - 1
    return Counter(n.fn.id.split("{", 1)[0] for n in B.nodes)


def _rows(model) -> list[tuple[int, ...]]:
    """Per layer, in layer order, the rows its latent, index-K and scale histories hold."""
    return [tuple(len(model.logs[L][h]) for h in ("latent", "ik", "iks")) for L in sorted(model.logs)]


def test_a_device_without_a_tuned_fused_moe_tile_is_refused_where_the_moe_block_is_written(cut):
    """vLLM keys its tuned fused-MoE tiles by the name the device reports, and the decoder holds the H200's file alone.  On
    another device the fourth layer, the MoE block, is refused by name.  The first three layers read no tile: cut to them,
    the other device's Program is the H200's."""
    assert _families(build_program(cut(), "step", M=1, pos0=0, with_desc=False))["MoeExpertFp8Gemm_v1"] == 8
    with pytest.raises(Unexpressible, match=re.escape(NO_TILE)):
        build_program(cut(device=H100), "step", M=1, pos0=0, with_desc=False)
    dense, same = (build_program(cut(3, device), "step", M=1, pos0=0) for device in (H100, H200))
    assert not any(f.startswith("Moe") for f in _families(dense)) and dense.digest == same.digest


def test_a_node_of_refused_ranks_is_refused_in_the_ranks_words(cut):
    """A node's ranks are written in lockstep, each on a thread of its own, a collective at a time.  The first rank to reach
    the MoE block is refused there; every other rank stops where it waits, and the caller gets that rank's refusal, whichever
    rank φ names.  On the H200 the eight ranks join."""
    assert build_node_program(cut(), "step", M=1, pos0=0, with_desc=False).world == 8
    for rank in (0, 5):
        phi = cut(device=H100)
        with pytest.raises(Unexpressible, match=re.escape(NO_TILE)):
            build_node_program(Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}}), "step", M=1, pos0=0, with_desc=False)


def test_a_blocks_all_reduce_fuses_with_its_norm_up_to_the_rows_vllm_hands_flashinfer(cut):
    """After a block the rows' partials are all-reduced, added to the residual and normed.  vLLM hands FlashInfer's fused
    kernel at most 512 KiB of bf16, 42 rows of 6144: up to there each row is one fused call, and past it the rows are one
    all-reduce and a fused norm each.  With FlashInfer's all-reduce off nothing fuses, and the all-reduce vLLM then takes
    has no capture: it is refused as that all-reduce, by its route and the switches."""
    assert _reduced(cut(), 1) == {"AllReduceRmsNormMnnvl_v1": 1}
    assert _reduced(cut(), 42) == {"AllReduceRmsNormMnnvl_v1": 42}
    assert _reduced(cut(), 43) == {"AllReduceMultimem_v1": 1, "RMSNormFusedCudaVector8_v3": 43}
    off = {**SWITCHES, "flashinfer": False}
    with pytest.raises(Unexpressible, match=re.escape(f"vocabulary cannot express: an all-reduce of 1 x 6144 on route 'ca_comm' under {off}" + NO_CAPTURE)):
        _reduced(_engine(cut(), {"VLLM_ALLREDUCE_USE_FLASHINFER": "0"}), 1)


def test_the_all_reduces_switches_are_the_engines(cut):
    """The decoder reads vLLM's all-reduce switches from φ's engine: its `disable_custom_all_reduce` and its environment's.
    Without symmetric memory 43 rows take the custom all-reduce; with that off too vLLM takes NCCL's, and one row
    FlashInfer's fused kernel, neither of which a capture covers under these switches: each is refused as what it is,
    naming the switches as the engine states them."""
    assert decoder(cut()).settings == SWITCHES
    symm_off = _engine(cut(), {"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"})
    assert _reduced(symm_off, 43) == {"AllReduceCustom_v1": 1, "RMSNormFusedCudaVector8_v3": 43}
    both_off = {**SWITCHES, "custom_all_reduce": False, "symmetric_memory": False}
    with pytest.raises(Unexpressible, match=re.escape(f"vocabulary cannot express: an all-reduce of 43 x 6144 on route 'pynccl_comm' under {both_off}" + NO_CAPTURE)):
        _reduced(_engine(symm_off, disable_custom_all_reduce=True), 43)
    with pytest.raises(Unexpressible, match=re.escape(f"vocabulary cannot express: a fused all-reduce + norm of 1 x 6144 under {both_off}" + NO_CAPTURE)):
        _reduced(_engine(symm_off, disable_custom_all_reduce=True), 1)


def test_a_history_opens_once_and_before_a_layers_first_call(cut):
    """`open_with` after layer 0 has written a row, its own or a first opening's, is refused, and leaves every history as it was."""
    model, x, prefix, _ = _model(cut(), 2)
    model.attention(0, [x], 0)
    assert _rows(model) == [(1, 1, 1)]
    with pytest.raises(AssertionError, match=re.escape("layer 0's history opens once, before its first call")):
        model.open_with(prefix, 2)
    assert _rows(model) == [(1, 1, 1)]
    model, _, prefix, _ = _model(cut(), 2)
    model.open_with(prefix, 2)
    opened = _rows(model)
    assert opened == [(2, 2, 2)] * 3 + [(2, 0, 0)], "the layer that shares a top-k keeps no index-K history"
    with pytest.raises(AssertionError, match=re.escape("layer 0's history opens once, before its first call")):
        model.open_with(prefix, 2)
    assert _rows(model) == opened


def test_an_edge_is_every_layers_rows_at_one_length(cut):
    """A history opened and handed on at once is its three Inputs, wires and no gates.  After layer 0 alone has written a
    row, the layers disagree and no edge is returned."""
    model, x, prefix, B = _model(cut(), 2)
    model.open_with(prefix, 2)
    edge = model.edge_rows(GROUPS)
    assert [e.type for e in edge] == [p.type for p in prefix] and [runs(e.refs) for e in edge] == [runs(p.refs) for p in prefix]
    assert not B.nodes
    model.attention(0, [x], 2)
    assert _rows(model) == [(3, 3, 3)] + [(2, 2, 2)] * 2 + [(2, 0, 0)]
    for groups in (GROUPS[:1], GROUPS):
        with pytest.raises(AssertionError, match=re.escape("every layer's history holds the 3 rows of layer 0's")):
            model.edge_rows(groups)
