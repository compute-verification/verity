"""The acquisition plan on SYNTHETIC manifests + runtime trees: 0 / 1 / many sites, unsupported fail-closed, residuals, a stable digest.
No model, no vLLM class, no module path of a real case: the shapes are the tests' own."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys


from verity_vllm.correspondence.reader_for_acquire import (
    AcquireCorrespondence,
    ArgSlot,
    ReturnSlot,
)
from verity_vllm.query.manifest import format as RM
from verity_vllm.records import capture as CP
from verity_vllm.records.runtime_tree import runtime_tree_from_paths

PD = hashlib.sha256(b"program").hexdigest()

#: the lifetime tables of the synthetic runtime tree (the plan never reads them to choose a site; the lifetimes and residuals do)
TABLES = {"HAZARD_LEAVES": {"proj_a", "embed"}, "GATHER_FLUSH_LEAVES": {"proj_a"}, "PRE_FLUSH_LEAVES": {"norm_in"}}


def tree(**extra):
    paths = {"": ["Wrap", "Module"], "body": ["Body", "Module"], "body.embed": ["Embed", "Module"], "body.blocks.0.norm_in": ["Norm", "Module"],
             "body.blocks.0.proj_a": ["Lin", "Base", "Module"], "body.blocks.0.act": ["Act", "Module"], "body.blocks.0.norm_out": ["Norm", "Module"],
             "body.norm": ["Norm", "Module"], "head": ["Head", "Module"], "logits": ["Logits", "Module"], "body.blocks.0.attn": ["Attn", "Module"]}
    paths.update(extra)
    return runtime_tree_from_paths(paths, buffers=["body.blocks.0.attn.rotary_emb.cos_sin_cache", "body.scale"], runner={"sample": True, "sampler_module": True, "sampler_class": ["Sampler"]},
                                   source="synthetic")


def row(family, op, member, *, step=0, **extra):
    return {"family": family, "program_digest": PD, "request_id": "r0", "step": step, "invocation": 0, "op_path": op, "output_member": member,
            "element_range": [0, 8], "rank": 0, **extra}


def manifest(rows, **hdr):
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    d = {"schema": RM.SCHEMA, "program_digest": PD, "identities": rows, "populations": {"identities": len(rows), "by_family": fams}, "wrapper_prefix": "", "op_path_aliases": {}}
    d.update(hdr)
    d["manifest_digest"] = RM.manifest_digest(rows)
    return d


def base_rows():
    rows = []
    for step in (0, 1):
        rows += [row("input_token_ids", "body.embed", "in", step=step, rows=3, width=1, dtype="i32"),
                 row("instance_outputs", "body.embed", "out", step=step, spec="Embedding_v1"),
                 row("instance_outputs", "body.blocks.0.norm_in", "out" if step == 0 else "0", step=step, spec="Norm_v1"),
                 row("instance_outputs", "body.blocks.0.proj_a", "0", step=step, spec="Gemm_v1"),
                 row("instance_outputs", "body.blocks.0.act", "out", step=step, spec="Act_v1"),
                 row("instance_outputs", "body.blocks.0.norm_out", "0", step=step, spec="Norm_v1"),
                 row("instance_outputs", "body.blocks.0.norm_out", "1", step=step, spec="Norm_v1"),
                 row("instance_outputs", "logits", "out", step=step, spec="Gemm_v1"),
                 row("sampled_token_ids", "runner.sampler", "sampled_token_ids", step=step, dtype="i64"),
                 row("fa2_hidden_m1_stream", "fa2.m1/L0", f"g4x{8 + step}x2", step=step, dtype="u32", attention_module="body.blocks.0.attn")]
        if step == 1:
            rows.append(row("instance_outputs", "body.blocks.0.norm_in", "1", step=step, spec="Norm_v1"))
    rows += [row("weights", "body.blocks.0.proj_a.weight", "weight", step="*", dtype="bf16"),
             row("weights", "body.blocks.0.attn.rotary_emb.cos_sin_cache", "weight", step="*", dtype="bf16")]
    return rows




# ------------------------------------------------------------------------------------------------------------------ correspondence

def test_adapter_three_functions():
    corr = AcquireCorrespondence(tree(), wrapper_prefix="model.")
    assert corr.owning_module({"op_path": "body.embed", "output_member": "out"}) == "body.embed"
    assert corr.owning_module({"name": "model.body.blocks.0.proj_a/linear_3/[7]", "spec": "Gemm_v1{K=8,N=8}"}) == "body.blocks.0.proj_a"
    assert corr.return_slot({"op_path": "x", "output_member": "out"}) == ReturnSlot()
    assert corr.return_slot({"op_path": "x", "output_member": "1"}) == ReturnSlot(index=1)
    assert corr.return_slot({"op_path": "x", "output_member": "in"}) is None
    assert corr.return_slot({"op_path": "runner.sampler", "output_member": "sampled_token_ids"}) == ReturnSlot(attr="sampled_token_ids")
    assert corr.arg_slot({"op_path": "x"}, 0) == ArgSlot(0) and ArgSlot(0).label() == "in[0]"
    corr2 = AcquireCorrespondence(tree(), aliases={"body.blocks.1.rope": ["body.blocks.0.rope", 1]})
    assert corr2.owning_module({"op_path": "body.blocks.1.rope"}) == "body.blocks.0.rope"


# ------------------------------------------------------------------------------------------------------------------ sites















ACT, PROJ = "body.blocks.0.act", "body.blocks.0.proj_a"
















# ------------------------------------------------------------------------------------------------------------------ residuals



_PLAN_OF_FILES = """
import json, sys
if sys.argv[3] == "blocked":
    sys.modules["torch"] = None
from verity_vllm.acquire import plan as P
from verity_vllm.correspondence.reader_for_acquire import AcquireCorrespondence, RuntimeTree
plan = P.plan_of_capture(json.load(open(sys.argv[1])), AcquireCorrespondence(RuntimeTree.load(sys.argv[2])))
print(json.dumps({"digest": plan.digest, "residuals": {r.table: r.values for r in plan.residuals},
                  "tables": {k: sorted(v) for k, v in P._lifetime_tables().items()}, "torch": sys.modules.get("torch") is not None}))
"""


def _plan_in_a_fresh_process(tmp_path, mode: str) -> dict:
    out = subprocess.run([sys.executable, "-c", _PLAN_OF_FILES, str(tmp_path / "manifest.json"), str(tmp_path / "runtime_tree.json"), mode],
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_tables_do_not_depend_on_torch(tmp_path):
    """The plan with its default lifetime tables (no `tables=`) where torch imports and where it does not: the same tables, residuals
    and digest, so a CPU gate without torch and the GPU phase evaluate one plan."""
    from verity_vllm.commit.committer import flush_points

    classes = [{"op_path": op, "member": m, "capture": CP.MODULE_OUTPUT, "width": 8, "dtype": "bf16", "definitions": [d]}
               for op, m, d in (("body.blocks.0.qkv_proj", "0", "Gemm_v1"), ("body.norm", "out", "Norm_v1"))]
    (tmp_path / "manifest.json").write_text(json.dumps(classes))
    tree(**{"body.blocks.0.qkv_proj": ["Lin", "Base", "Module"]}).save(tmp_path / "runtime_tree.json")
    free, blocked = _plan_in_a_fresh_process(tmp_path, "free"), _plan_in_a_fresh_process(tmp_path, "blocked")
    assert not blocked["torch"]
    assert blocked["tables"]["GATHER_FLUSH_LEAVES"] == sorted(flush_points.GATHER_FLUSH_LEAVES)
    assert blocked["tables"]["PRE_FLUSH_LEAVES"] == sorted(flush_points.PRE_FLUSH_LEAVES)
    assert blocked["residuals"]["GATHER_FLUSH_LEAVES"] == ["body.blocks.0.qkv_proj/0"] and blocked["residuals"]["PRE_FLUSH_LEAVES"] == ["body.norm/out"]
    assert (blocked["tables"], blocked["residuals"], blocked["digest"]) == (free["tables"], free["residuals"], free["digest"])


# ------------------------------------------------------------------------------------------------------------------ serialisation





def _body_by_site_scan(p):
    """The plan body as `values_at` gives it: one scan of every Value per site (the reference the one-pass body must equal)."""
    b = p.body()
    b["hooks"] = [{**h, "values": p.values_at(s)} for h, s in zip(b["hooks"], p.hooks())]
    b["sites"] = [{**h, "values": p.values_at(s)} for h, s in zip(b["sites"], p.chosen_sites())]
    b["counts"]["sites_by_kind"] = {k: len(v) for k, v in sorted(p.sites_by_kind().items())}
    return b


