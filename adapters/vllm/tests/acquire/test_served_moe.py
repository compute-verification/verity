"""The served Build's reading of vLLM's fused-MoE op (`acquire.served`, `acquire.served_moe`): a wrapper line `moe_forward(x, x, None,
None, layer, 0)`, or with a router-logits slot of x's shape that the op never reads, parses into one opaque call; `served_moe.expand` turns it into the Calls the op runs, the eager MoE block's Call
boundaries, with the planes the serving process commits from inside the op, and refuses a block of any other form; each expert
GEMM is Hopper's k16 chain at every step whose rows the process launched the kernel at."""
from __future__ import annotations

import copy

import pytest

from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire import served_moe as SM
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity.ml.definitions.registry import act
from verity.ml.definitions.registry import moe as M

H, E, TOPK, VPT, I = 32, 8, 2, 4, 16
OP = "torch.ops.vllm.moe_forward.default"
LAYER = "model.layers.0.mlp.experts"

WRAPPER = f"""
class Runner:
    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s7 = arg0_1
        assert_size_stride(arg1_1, (s7, {H}), ({H}, 1), 'input')
        assert_size_stride(arg2_1, ({H}, {H}), ({H}, 1), 'input')
        buf0 = empty_strided_cuda((s7, {H}), ({H}, 1), torch.bfloat16)
        extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, ({H}, {H}), (1, {H}), 0), out=buf0)
        run_intermediate_hooks('verity|mm|buf0', buf0)
        buf1 = {OP}(buf0, buf0, None, None, arg3_1, 0)
        assert_tensor_metadata(buf1, (s7, {H}), ({H}, 1), torch.bfloat16, '{OP}')
        assert_alignment(buf1, 16, '{OP}')
        run_intermediate_hooks('verity|moe_forward|buf1', buf1)
        del buf0
        return (buf1, )


runner = Runner(partitions=[])
"""


def _piece(src: str = WRAPPER) -> SV._Piece:
    return SV._Piece(src, None, "moe.py", (9, 0))


def test_the_op_is_one_call_on_its_hidden_states() -> None:
    p = _piece()
    prog = p.program()
    assert [v["name"] for v in prog["values"]] == ["mm/buf0", "moe_forward/buf1"]
    assert prog["calls"][1] == {"op": "moe_forward", "layer": {"arg": 3}, "reads": [{"value": 0, "width": H}], "writes": [1]}
    assert prog["returns"] == [{"value": 1, "width": H}]
    assert "empty" in p.cur["buf0"], "the op mutates its hidden states: no later line reads them as the value"


def test_a_layer_named_in_the_wrapper_is_a_literal() -> None:
    prog = _piece(WRAPPER.replace("None, arg3_1, 0)", f"None, '{LAYER}', 0)")).program()
    assert prog["calls"][1]["layer"] == {"name": LAYER}


#: the op beside the hidden states it mutates, its router-logits slot another buffer of their rows (which the runner never reads),
#: and its return renamed, as Inductor emits it when another kernel writes both inputs
TWO_INPUTS = (WRAPPER.replace("        buf1 = ", f"        buf9 = empty_strided_cuda((s7, {H}), ({H}, 1), torch.bfloat16)\n"
                                                 f"        extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, ({H}, {H}), (1, {H}), 0), out=buf9)\n"
                                                 "        run_intermediate_hooks('verity|mm|buf9', buf9)\n        buf1 = ")
              .replace("(buf0, buf0, None", "(buf0, buf9, None")
              .replace(f"        assert_tensor_metadata(buf1, (s7, {H}), ({H}, 1), torch.bfloat16, '{OP}')\n        assert_alignment(buf1, 16, '{OP}')\n"
                       "        run_intermediate_hooks('verity|moe_forward|buf1', buf1)\n",
                       f"        buf6 = buf1\n        assert_size_stride(buf6, (s7, {H}), ({H}, 1), '{OP}')\n        assert_alignment(buf6, 16, '{OP}')\n"
                       "        del buf1\n        run_intermediate_hooks('verity|extern|buf6', buf6)\n")
              .replace("return (buf1, )", "return (buf6, buf9, )"))


def test_the_router_logits_slot_may_be_another_buffer_never_read() -> None:
    prog = _piece(TWO_INPUTS).program()
    assert [v["name"] for v in prog["values"]] == ["mm/buf0", "mm/buf9", "extern/buf6"]
    assert prog["calls"][2] == {"op": "moe_forward", "layer": {"arg": 3}, "reads": [{"value": 0, "width": H}], "writes": [2]}
    assert prog["returns"] == [{"value": 2, "width": H}, {"value": 1, "width": H}]
    with pytest.raises(Unmodelled, match="'buf1' is not a buffer"):
        _piece(TWO_INPUTS.replace("return (buf6, buf9, )", "return (buf1, buf9, )"))


@pytest.mark.parametrize("old,new,why", [
    ("(buf0, buf0, None", "(buf0, arg2_1, None", "a buffer of its shape"),
    ("None, arg3_1, 0)", "None, arg3_1, 1)", "a buffer of its shape"),
    ("None, None, arg3_1", "arg1_1, None, arg3_1", "a buffer of its shape"),
    (f"({H}, 1), torch.bfloat16, '{OP}'", f"({H}, 1), torch.float32, '{OP}'", "bf16 buffer"),
    (f"assert_tensor_metadata(buf1, (s7, {H})", f"assert_tensor_metadata(buf1, (s7, {2 * H})", "bf16 buffer"),
    (f"torch.bfloat16, '{OP}')", "torch.bfloat16, 'torch.ops.vllm.other.default')", "bf16 buffer"),
    ("(buf0, buf0, None, None, arg3_1", "(arg2_1, arg2_1, None, None, arg3_1", "whole token rows"),
])
def test_another_form_of_the_op_is_refused(old: str, new: str, why: str) -> None:
    assert old in WRAPPER
    with pytest.raises(Unmodelled, match=why):
        _piece(WRAPPER.replace(old, new))


# ------------------------------------------------------------------------------------------------------------- the expansion
X, OUT = "c01.add_rms_norm/buf3", "c01.moe_forward/buf4"
FACTS = {"gate": {"weight": "model.layers.0.mlp.gate.weight", "shape": [E, H]},
         "w13": {"weight": f"{LAYER}.w13_weight", "shape": [E, 2 * I, H]}, "w2": {"weight": f"{LAYER}.w2_weight", "shape": [E, H, I]},
         "E": E, "top_k": TOPK, "renormalize": True, "VPT": VPT, **SM.REQUIRED}
#: launch configs Triton lowers to MMA v2 (BLOCK_SIZE_M 16) and to wgmma (64, 64, 4 warps)
SMALL = {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "GROUP_SIZE_M": 1, "num_stages": 3, "num_warps": 4}
WIDE = {"BLOCK_SIZE_K": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_stages": 3, "num_warps": 4}
RECORD = {"layers": {LAYER: FACTS}, "pieces": {"1": LAYER}, "cc": [9, 0],
          "configs": {"8": {"False": SMALL, "True": SMALL}, "256": {"False": WIDE, "True": SMALL}}}
UP, DOWN = f"{E},{H},{2 * I},0", f"{E},{I},{H},1"


def _wired(layer: dict | None = None) -> dict:
    return {"calls": [{"op": "mm", "gemm": [H, H], "piece": 1, "reads": [{"value": "c00.mm/buf0", "width": H},
                                                                        {"weight": "w", "shape": [H, H], "t": True}], "writes": [X]},
                      {"op": "moe_forward", "piece": 1, "layer": layer or {"layer_name": True}, "reads": [{"value": X, "width": H}],
                       "writes": [OUT]}],
            "values": {"c00.mm/buf0": {"width": H, "piece": 0}, X: {"width": H, "piece": 1}, OUT: {"width": H, "piece": 1}},
            "attention": []}


def _planes() -> list[str]:
    return [f"c01.moe/{n}" for n in SM.PLANES]


def test_the_op_expands_into_the_eager_blocks_calls() -> None:
    w, plan = SM.expand(_wired(), RECORD)
    lg, rw, ri, up, ac, down = _planes()
    assert [c["op"] for c in w["calls"]] == ["mm", "mm", "moe_router", "moe_up", "moe_act", "moe_down", "moe_sum"]
    assert all(c["piece"] == 1 for c in w["calls"])
    gate, router, ups, acts, downs, total = w["calls"][1:]
    assert gate == {"op": "mm", "gemm": [H, E], "piece": 1, "writes": [lg],
                    "reads": [{"value": X, "width": H}, {**FACTS["gate"], "t": True}]}
    assert (router["base"], router["statics"]) == (M.MoeRouterTopKNorm.id, {"E": E, "TOPK": TOPK, "VPT": VPT})
    assert router["reads"] == [{"value": lg, "width": E}] and router["writes"] == [rw, ri]
    assert ups["moe"] == UP and ups["reads"] == [{"value": X, "width": H}, {"value": ri, "width": TOPK}, FACTS["w13"]]
    assert (acts["base"], acts["statics"], acts["writes"]) == (act.SiluMul.id, {"I": I}, [ac])
    assert downs["moe"] == DOWN and downs["reads"] == [{"value": ac, "width": TOPK * I}, {"value": ri, "width": TOPK},
                                                     {"value": rw, "width": TOPK}, FACTS["w2"]]
    assert (total["base"], total["statics"], total["writes"]) == (M.MoeSum.id, {"TOPK": TOPK, "H": H}, [OUT])
    assert list(w["values"]) == ["c00.mm/buf0", X, *_planes(), OUT]
    assert [w["values"][n]["width"] for n in _planes()] == [E, TOPK, TOPK, TOPK * 2 * I, TOPK * I, TOPK * H]
    assert plan is not None and w["attention"] == []


def test_without_renormalize_the_router_is_plain_top_k() -> None:
    rec = copy.deepcopy(RECORD)
    rec["layers"][LAYER]["renormalize"] = False
    assert SM.expand(_wired(), rec)[0]["calls"][2]["base"] == M.MoeRouterTopK.id


def test_a_program_without_the_op_is_unchanged() -> None:
    w = {"calls": [], "values": {}, "attention": []}
    assert SM.expand(w, None) == (w, None)
    assert SM.expand(w, {"layers": {}}) == (w, None)


def test_the_expert_gemms_are_hoppers_k16_chain_at_every_launched_config() -> None:
    plan = SM.expand(_wired(), RECORD)[1]
    small, wide = plan(8), plan(256)
    assert small == wide and set(small) == {UP, DOWN}
    assert (small[UP]["base"], small[UP]["statics"]) == (M.MoeExpertGemmV2.id, {"E": E, "K": H, "N": 2 * I, "DOT": HopperBF16WgmmaDot16.id})
    assert (small[DOWN]["base"], small[DOWN]["statics"]) == (M.MoeExpertGemmWV2.id, {"E": E, "K": I, "N": H, "DOT": HopperBF16WgmmaDot16.id})
    with pytest.raises(Unmodelled, match="no fused_moe_kernel launch at 16 rows"):
        plan(16)


@pytest.mark.parametrize("cc", [[10, 0], [12, 0]])
def test_cc_10_and_12_bind_the_same_expert_gemms(cc: list[int]) -> None:
    assert SM.expand(_wired(), {**RECORD, "cc": cc})[1](256) == SM.expand(_wired(), RECORD)[1](256)


def test_step_plans_carry_the_expert_gemms_only_with_moe() -> None:
    sched = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 2}]
    kw = dict(capture_sizes=[1, 2, 4, 8], max_num_seqs=8, split_cap=SV.SPLIT_CAP, device=SD.device_of((9, 0), 132))
    dense, _ = SV.step_plans(sched, 2, [(H, H)], (H, 64), (2, 1, 64), {}, **kw)
    assert all("moe" not in p for p in dense)
    plan = SM.expand(_wired(), {**RECORD, "configs": {"4": {"False": WIDE, "True": WIDE}, "1": {"False": SMALL, "True": SMALL}}})[1]
    moe, _ = SV.step_plans(sched, 2, [(H, H)], (H, 64), (2, 1, 64), {}, moe=plan, **kw)
    assert [p["gemm_m"] for p in moe] == [4, 1]
    assert [p["moe"][UP]["base"] for p in moe] == [M.MoeExpertGemmV2.id, M.MoeExpertGemmV2.id]


def _flag(k: str, v):
    def f(rec):
        rec["layers"][LAYER][k] = v
    return f


@pytest.mark.parametrize("edit,why", [
    *[(_flag(k, not v), k) for k, v in SM.REQUIRED.items()],
    (_flag("silu", None), "silu"),
    (_flag("renormalize", None), "renormalize"),
    (_flag("w2", {"weight": f"{LAYER}.w2_weight", "shape": [E, H, 2 * I]}), "w2"),
    (_flag("gate", {"weight": None, "shape": [E, H]}), "gate"),
    (_flag("E", 2 * E), "gate"),
    (lambda rec: rec.update(cc=[8, 9]), "compute capability"),
    (lambda rec: rec.update(pieces={"2": LAYER}), "the serving process ran layer None"),
    (lambda rec: rec.update(layers={"model.layers.1.mlp.experts": FACTS}), "no facts"),
])
def test_a_block_of_another_form_is_refused(edit, why: str) -> None:
    rec = copy.deepcopy(RECORD)
    edit(rec)
    with pytest.raises(Unmodelled, match=why):
        SM.expand(_wired(), rec)


def test_the_op_needs_the_serving_processs_record() -> None:
    with pytest.raises(Unmodelled, match="no fused-MoE record"):
        SM.expand(_wired(), None)
    with pytest.raises(Unmodelled, match="no piece calls the fused-MoE op"):
        SM.expand({"calls": [], "values": {}}, RECORD)


def test_a_literal_layer_must_be_the_one_the_process_ran() -> None:
    assert SM.expand(_wired({"name": LAYER}), RECORD)[0]["calls"][2]["op"] == "moe_router"
    with pytest.raises(Unmodelled, match="the serving process ran layer"):
        SM.expand(_wired({"name": "model.layers.1.mlp.experts"}), RECORD)


def test_two_ops_in_a_piece_or_a_plane_named_twice_are_refused() -> None:
    w = _wired()
    w["calls"].append({**w["calls"][1], "writes": ["c01.moe_forward/buf9"]})
    w["values"]["c01.moe_forward/buf9"] = {"width": H, "piece": 1}
    with pytest.raises(Unmodelled, match="calls the fused-MoE op 2 times"):
        SM.expand(w, RECORD)
    w = _wired()
    w["values"] = {"c01.moe/up": {"width": 1, "piece": 1}, **w["values"]}
    with pytest.raises(Unmodelled, match="c01.moe/up"):
        SM.expand(w, RECORD)
