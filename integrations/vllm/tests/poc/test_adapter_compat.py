"""What W1's adapter (`verity_vllm_adapter/capture.py`, branch cursor/vllm-poc-adapter-7e60) emits
must load and check here.  The description below is shaped exactly like `Capture.description()`
builds it (extra node keys, `cols`/`unbound` arg keys, `t: all_le`, `"unresolved"` relations,
`sampler.select.greedy.v1`, int64 sampler output, named layers)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from verity_vllm import commit, hashing, relations, synthetic, verify
from verity_vllm.capture import Boundary, CaptureReader, Value
from verity_vllm.description import Description, Node, NodeInstance
from verity_vllm.negatives import PatchedReader

W1_DESCRIPTION = {
    "schema": "verity-vllm/description/v1",
    "profile_id": "0123456789ab",
    "model": "Qwen/Qwen3-0.6B",
    "symbols": {"H": 1024, "V": 151936},
    "interception": "dispatch-mode",
    "patches": ["bi_mm", "bi_rms_norm"],
    "nodes": [
        {
            "op": "embed_tokens.embedding",
            "layer": "embed",
            "op_type": "embedding",
            "family": "embedding",
            "dtype": "bfloat16",
            "shape": [1024],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "all",
            "args": [
                {"name": "w", "source": "weight", "param": "embed_tokens.weight"},
                {"name": "ids", "source": "token", "which": "input"},
            ],
            "consts": {},
            "relation": "embedding.same_runtime.aten.v1",
            "origin": "captured",
            "decompositions": [
                {"id": "row", "interface": {"input_bytes": 8, "output_bits": 16384, "range": "row"}}
            ],
            "runtime_op": "aten.embedding",
            "runtime_impl": "aten:embedding (CUDA)",
            "module_path": "embed_tokens",
            "instances_observed": 12,
            "arg_violations": [],
        },
        {
            "op": "layers.0.input_layernorm.rmsnorm",
            "layer": "0",
            "op_type": "rmsnorm",
            "family": "rmsnorm",
            "dtype": "bfloat16",
            "shape": [1024],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "all",
            "args": [
                {"name": "x", "source": "node", "op": "embed_tokens.embedding", "port": 0},
                {"name": "w", "source": "weight", "param": "layers.0.input_layernorm.weight"},
            ],
            "consts": {"eps": 1e-06},
            "relation": "rmsnorm.same_runtime.bi_triton.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 2048, "output_bits": 16384, "range": "row"},
                }
            ],
            "runtime_op": "verity.bi_rms_norm",
            "runtime_impl": "triton:..._rms_norm_kernel",
            "module_path": "layers.0.input_layernorm",
            "instances_observed": 12,
            "arg_violations": [],
        },
        {
            "op": "layers.0.self_attn.qkv_proj.gemm",
            "layer": "0",
            "op_type": "gemm",
            "family": "gemm",
            "dtype": "bfloat16",
            "shape": [4096],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "all",
            "args": [
                {
                    "name": "x",
                    "source": "node",
                    "op": "layers.0.input_layernorm.rmsnorm",
                    "port": 0,
                },
                {"name": "w_t", "source": "weight", "param": "layers.0.self_attn.qkv_proj.weight"},
            ],
            "consts": {},
            "relation": "gemm.same_runtime.bi_triton.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 2048, "output_bits": 65536, "range": "row"},
                }
            ],
            "runtime_op": "verity.bi_mm",
            "runtime_impl": "triton:...matmul_persistent",
            "module_path": "layers.0.self_attn.qkv_proj",
            "instances_observed": 12,
            "arg_violations": [],
        },
        {
            "op": "layers.0.self_attn.rope",
            "layer": "0",
            "op_type": "rope",
            "family": "rope",
            "dtype": "bfloat16",
            "shape": [2048],
            "ports": 2,
            "outputs": [{"port": 1, "dtype": "bfloat16", "shape": [1024]}],
            "phases": ["decode", "prefill"],
            "positions": "all",
            "args": [
                {"name": "positions", "source": "const", "value": "positions"},
                {
                    "name": "q",
                    "source": "node",
                    "op": "layers.0.self_attn.qkv_proj.gemm",
                    "port": 0,
                    "cols": [0, 2048],
                },
                {
                    "name": "k",
                    "source": "node",
                    "op": "layers.0.self_attn.qkv_proj.gemm",
                    "port": 0,
                    "cols": [2048, 3072],
                },
                {
                    "name": "cos_sin_cache",
                    "source": "const",
                    "value": "buffer:cos_sin_cache",
                    "unbound": True,
                },
            ],
            "consts": {"head_size": 128, "is_neox": True},
            "relation": "rope.same_runtime.cuda.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 6144, "output_bits": 49152, "range": "row"},
                }
            ],
            "runtime_op": "_C.rotary_embedding",
            "runtime_impl": "cuda:torch.ops._C.rotary_embedding",
            "module_path": "layers.0.self_attn",
            "instances_observed": 12,
            "arg_violations": [{"arg": "cos_sin_cache", "detail": "unbound buffer"}],
        },
        {
            "op": "layers.0.self_attn.attn",
            "layer": "0",
            "op_type": "attention",
            "family": "attn",
            "dtype": "bfloat16",
            "shape": [2048],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "all",
            "args": [
                {"name": "q", "source": "node", "op": "layers.0.self_attn.rope", "port": 0},
                {
                    "name": "k",
                    "source": "node",
                    "op": "layers.0.self_attn.rope",
                    "port": 1,
                    "t": "all_le",
                },
                {
                    "name": "v",
                    "source": "node",
                    "op": "layers.0.self_attn.qkv_proj.gemm",
                    "port": 0,
                    "cols": [3072, 4096],
                    "t": "all_le",
                },
            ],
            "consts": {"layer_name": "model.layers.0.self_attn.attn"},
            "relation": "attention.same_runtime.paged.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 4096, "output_bits": 32768, "range": "row"},
                }
            ],
            "runtime_op": "vllm.unified_attention_with_output",
            "runtime_impl": "FLASH_ATTN",
            "module_path": "layers.0.self_attn.attn",
            "instances_observed": 12,
            "arg_violations": [],
        },
        {
            "op": "layers.0.mlp.gather",
            "layer": "0",
            "op_type": "other",
            "family": "gather",
            "dtype": "bfloat16",
            "shape": [1024],
            "ports": 1,
            "outputs": [],
            "phases": ["prefill"],
            "positions": "all",
            "args": [{"name": "x", "source": "node", "op": "layers.0.self_attn.attn", "port": 0}],
            "consts": {},
            "relation": "unresolved",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 4096, "output_bits": 16384, "range": "row"},
                }
            ],
            "runtime_op": "aten.index",
            "runtime_impl": "unresolved",
            "module_path": "layers.0.mlp",
            "instances_observed": 3,
            "arg_violations": [],
        },
        {
            "op": "lm_head.gemm",
            "layer": "lm_head",
            "op_type": "gemm",
            "family": "gemm",
            "dtype": "bfloat16",
            "shape": ["V"],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "sampled",
            "args": [
                {"name": "x", "source": "node", "op": "layers.0.self_attn.attn", "port": 0},
                {"name": "w_t", "source": "weight", "param": "lm_head.weight"},
            ],
            "consts": {},
            "relation": "gemm.same_runtime.bi_triton.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 2048, "output_bits": 2430976, "range": "row"},
                }
            ],
            "runtime_op": "verity.bi_mm",
            "runtime_impl": "triton:...matmul_persistent",
            "module_path": "lm_head",
            "instances_observed": 6,
            "arg_violations": [],
        },
        {
            "op": "sampler.select",
            "layer": "sampler",
            "op_type": "sampler",
            "family": "sampler",
            "dtype": "int64",
            "shape": [],
            "ports": 1,
            "outputs": [],
            "phases": ["decode", "prefill"],
            "positions": "sampled",
            "args": [{"name": "logits", "source": "node", "op": "lm_head.gemm", "port": 0}],
            "consts": {},
            "relation": "sampler.select.greedy.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {"input_bytes": 303872, "output_bits": 64, "range": "row"},
                }
            ],
            "runtime_op": "sampler",
            "runtime_impl": "vllm sampler (greedy)",
            "module_path": "sample",
            "instances_observed": 6,
            "arg_violations": [],
        },
    ],
}


def test_w1_description_loads_and_round_trips():
    d = Description(W1_DESCRIPTION)
    assert [n.op for n in d.nodes] == [n["op"] for n in W1_DESCRIPTION["nodes"]]
    rope = d.node("layers.0.self_attn.rope")
    assert rope.extra["runtime_op"] == "_C.rotary_embedding"
    assert rope.extra["consts"] == {"head_size": 128, "is_neox": True}
    assert rope.arg("q").extra == {"cols": [0, 2048]}
    assert rope.arg("cos_sin_cache").extra == {"unbound": True}
    assert rope.port_shape(1) == (1024,) and rope.port_dtype(1) == "bfloat16"
    attn = d.node("layers.0.self_attn.attn")
    assert attn.arg("k").t == "all_le" and attn.arg("v").extra == {"cols": [3072, 4096]}
    # everything W1 wrote survives normalisation: extra keys verbatim, known keys semantically
    # (defaults such as an empty ``outputs`` list are omitted), and the digest is idempotent
    out = d.to_json()
    assert out["patches"] == ["bi_mm", "bi_rms_norm"] and out["interception"] == "dispatch-mode"
    for original, emitted in zip(W1_DESCRIPTION["nodes"], out["nodes"], strict=True):
        for key in (
            "family",
            "consts",
            "runtime_op",
            "module_path",
            "instances_observed",
            "arg_violations",
            "runtime_impl",
            "relation",
            "positions",
            "phases",
        ):
            assert emitted[key] == original[key], (original["op"], key)
    again = Description(out)
    assert again.nodes == d.nodes and again.digest() == d.digest()
    # the sampler's greedy relation is executable here; W1's same_runtime ids are for W1/W4 to fill
    assert relations.get("sampler.select.greedy.v1").implemented
    d.validate(known_relations=list(relations.ids()) + [
        "embedding.same_runtime.aten.v1", "rmsnorm.same_runtime.bi_triton.v1",
        "gemm.same_runtime.bi_triton.v1", "rope.same_runtime.cuda.v1",
        "attention.same_runtime.paged.v1", "unresolved",
    ])  # fmt: skip


def test_w1_relation_ids_are_not_silently_exact(tmp_path):
    """A toy bundle whose description says what W1's does for a node nobody implemented:
    the verifier must come out PARTIAL (never PASS) and label the op unresolved."""
    toy = synthetic.build_toy_run(tmp_path / "toy", seed=3)
    bundle = tmp_path / "w1ish"
    shutil.copytree(toy.run_dir, bundle)
    doc = json.loads((bundle / "description.json").read_text())
    for node in doc["nodes"]:
        node.update({"family": node["op_type"], "consts": {}, "module_path": node["op"]})
        if node["op"] == "layers.1.mlp.gemm":
            node["relation"] = "unresolved"
        if node["op"] == "layers.0.mlp.gemm":
            node["relation"] = "gemm.same_runtime.bi_triton.v1"
        if node["op"] == "sampler.select":
            node["relation"] = "sampler.select.greedy.v1"
    doc["interception"] = "dispatch-mode"
    (bundle / "description.json").write_bytes(hashing.canonical_json_bytes(doc))
    result = verify.verify(
        bundle,
        seed=1,
        exhaustive=True,
        out_dir=tmp_path / "out",
        case="W1ISH",
        plugins=[],
        torch_mode="never",
        negatives=None,
    )
    assert result.verdict == "PARTIAL"
    t2 = {r["op_id"]: r for r in result.table2}
    assert t2["layers.1.mlp.gemm"]["status"] == "unresolved"
    assert t2["layers.1.mlp.gemm"]["checker"] == "none"
    assert t2["layers.0.mlp.gemm"]["status"] == "unknown_relation"
    assert (
        t2["sampler.select"]["status"] == "exact"
        and t2["sampler.select"]["checker"] == "independent"
    )
    assert (
        result.table1["token_selection"]["exact"]
        == result.table1["token_selection"]["attempted"]
        > 0
    )
    statuses = result.report.statuses()
    assert statuses["unresolved"] > 0 and statuses["unknown_relation"] > 0
    assert "mismatch" not in statuses and "auth_failed" not in statuses


def test_sampler_select_greedy_refuses_non_greedy_params(tmp_path):
    toy = synthetic.build_toy_run(tmp_path / "toy", seed=3)
    reader = CaptureReader(toy.run_dir)
    sampler_ids = [
        c for c in toy.description.expected_ids(reader.boundary("req-0"), reader.profile_id)
        if "sampler.select" in c
    ]  # fmt: skip
    # wire sampling_params into the toy sampler node and switch to the greedy relation
    doc = toy.description.to_json()
    for node in doc["nodes"]:
        if node["op"] == "sampler.select":
            node["relation"] = "sampler.select.greedy.v1"
            node["args"].append(
                {"name": "sampling_params", "source": "boundary", "field": "sampling_params"}
            )
    greedy = Description(doc)
    report = relations.check(greedy, reader, toy.commitments, sampler_ids[:1])
    assert report.results[0].status == "exact", report.results[0].detail
    # a non-greedy request must not be accepted by the greedy relation
    b = reader.boundary("req-0")
    hot = json.loads(json.dumps(b.to_json()))
    hot["sampling_params"] = {**(b.sampling_params or {}), "temperature": 0.7}
    patched = PatchedReader(reader, boundaries={"req-0": Boundary.from_json(hot)})
    commitments = commit.build_commitments(patched, recompute_leaves=True, write=False)
    report = relations.check(greedy, patched, commitments, sampler_ids[:1])
    assert report.results[0].status == "error"
    assert "non-greedy" in report.results[0].detail


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_greedy_output_dtype(dtype):
    node = Node.from_json({
        "op": "sampler.select", "layer": "sampler", "op_type": "sampler", "dtype": dtype,
        "shape": [], "positions": "sampled", "relation": "sampler.select.greedy.v1",
        "args": [{"name": "logits", "source": "node", "op": "lm_head.gemm", "port": 0}],
    })  # fmt: skip
    inst = NodeInstance(node, "r", 0, "prefill")
    logits = Value("x", "float32", (4,), np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32).tobytes())
    out = relations.get("sampler.select.greedy.v1").checker(inst, {"logits": logits})
    assert int(np.frombuffer(out, dtype=dtype)[0]) == 1
