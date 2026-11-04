"""The acquisition plan on SYNTHETIC manifests + runtime trees: 0 / 1 / many sites, unsupported fail-closed, residuals, a stable digest.
No model, no vLLM class, no module path of a real case: the shapes are the tests' own."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from verity_vllm.query.manifest import format as RM
from verity_vllm.acquire import plan as P
from verity_vllm.correspondence.reader_for_acquire import AnnotationCorrespondence, ArgSlot, ReturnSlot, runtime_tree_from_paths

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
    fams = {}
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


def make_plan(rows, rt=None, **hdr):
    rt = rt or tree()
    man = manifest(rows, **hdr)
    corr = AnnotationCorrespondence(rt, wrapper_prefix=man["wrapper_prefix"], aliases=man["op_path_aliases"])
    return P.acquisition_plan(man, corr, tables=TABLES), man, rt


# ------------------------------------------------------------------------------------------------------------------ correspondence

def test_adapter_three_functions():
    corr = AnnotationCorrespondence(tree(), wrapper_prefix="model.")
    assert corr.owning_module({"op_path": "body.embed", "output_member": "out"}) == "body.embed"
    assert corr.owning_module({"name": "model.body.blocks.0.proj_a/linear_3/[7]", "spec": "Gemm_v1{K=8,N=8}"}) == "body.blocks.0.proj_a"
    assert corr.return_slot({"op_path": "x", "output_member": "out"}) == ReturnSlot()
    assert corr.return_slot({"op_path": "x", "output_member": "1"}) == ReturnSlot(index=1)
    assert corr.return_slot({"op_path": "x", "output_member": "in"}) is None
    assert corr.return_slot({"op_path": "runner.sampler", "output_member": "sampled_token_ids"}) == ReturnSlot(attr="sampled_token_ids")
    assert corr.arg_slot({"op_path": "x"}, 0) == ArgSlot(0) and ArgSlot(0).label() == "in[0]"
    corr2 = AnnotationCorrespondence(tree(), aliases={"body.blocks.1.rope": ["body.blocks.0.rope", 1]})
    assert corr2.owning_module({"op_path": "body.blocks.1.rope"}) == "body.blocks.0.rope"


# ------------------------------------------------------------------------------------------------------------------ sites

def test_one_site_producer_output_and_consumer_input_and_tap():
    plan, man, rt = make_plan(base_rows())
    assert plan.complete, plan.unsupported
    by = {v.value_id: v for v in plan.values}
    v = by["body.blocks.0.proj_a/0"]
    assert [s.kind for s in v.sites] == ["producer_output"] and v.chosen.slot == "0" and v.status == "chosen" and v.identities == 2
    assert v.chosen.lifetime == "mutated-after-hook (HAZARD_LEAVES)"                      # the table's in-place fact, named as such
    assert by["body.blocks.0.act/out"].chosen.lifetime == "unknown"                        # the annotations carry no in-place fact
    e = by["body.embed/in"]
    assert e.chosen.kind == "consumer_input" and e.chosen.module == "body.embed" and e.chosen.slot == "in[0]" and e.chosen.lifetime == "unknown"
    h = by["fa2.m1/L0/*"]
    assert h.chosen.kind == "native_tap" and h.identities == 2 and h.anchor == "body.blocks.0.attn"
    s = by["runner.sampler/sampled_token_ids"]
    assert s.chosen.kind == "runner_return" and s.chosen.slot == ".sampled_token_ids"
    w = by["body.blocks.0.proj_a.weight/weight"]
    assert w.chosen.kind == "registered_input" and len(w.sites) == 1
    # the hooks are exactly the module sites, one per (module, kind, slot)
    keys = [h.key() for h in plan.hooks()]
    assert ("body.blocks.0.norm_out", "producer_output", "0") in keys and ("body.blocks.0.norm_out", "producer_output", "1") in keys
    assert ("body.embed", "consumer_input", "in[0]") in keys and ("runner.sampler", "runner_return", ".sampled_token_ids") in keys
    assert all(k[1] in P.HOOK_KINDS for k in keys) and keys == sorted(keys) and len(set(keys)) == len(keys)


def test_many_sites_policy_producer_output_first_and_analytic_beside():
    rows = base_rows()
    # a non-promoted Value whose manifest row also names consumers: producer output AND consumer input are legal -> producer output first
    for r in rows:
        if r["op_path"] == "body.blocks.0.act":
            r["consumers"] = ["body.blocks.0.norm_out"]
    plan, _, _ = make_plan(rows)
    by = {v.value_id: v for v in plan.values}
    v = by["body.blocks.0.act/out"]
    assert sorted(s.kind for s in v.sites) == ["consumer_input", "producer_output"]
    assert v.chosen.kind == "producer_output" and v.why.startswith("several legal sites")
    # the analytic-table weight: registered input first, the analytic check beside it
    w = by["body.blocks.0.attn.rotary_emb.cos_sin_cache/weight"]
    assert sorted(s.kind for s in w.sites) == ["analytic_table", "registered_input"] and w.chosen.kind == "registered_input"


def test_promoted_value_consumer_input_only():
    rows = base_rows() + [row("instance_outputs", "body", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 3, "width": 8, "dtype": "bf16", "spec": "Scale_v1", "consumers": ["body.blocks.0.norm_in"]}))]
    plan, _, _ = make_plan(rows)
    v = {v.value_id: v for v in plan.values}["body/out"]
    assert v.promoted and [s.kind for s in v.sites] == ["consumer_input"] and v.chosen.module == "body.blocks.0.norm_in"
    assert any(r["kind"] == "producer_output" and "promoted" in r["why"] for r in v.rejected)     # the producer's output is refused, by name
    assert v.chosen.lifetime == "unknown"
    # in-place facts, when the annotations carry them, decide the lifetime
    man = manifest(rows)
    corr = AnnotationCorrespondence(tree(), inplace_facts={"body.blocks.0.norm_in": {"mutates_input": False}})
    p2 = P.acquisition_plan(man, corr, tables=TABLES)
    assert {v.value_id: v for v in p2.values}["body/out"].chosen.lifetime == "unmutated"
    # one promoted-consumer pre-hook, whatever the consumer's class
    assert ("body.blocks.0.norm_in", "consumer_input", "in[0]") in [h.key() for h in plan.hooks()]


def test_zero_sites_is_unsupported_and_fails_closed():
    rows = base_rows() + [row("instance_outputs", "body.blocks.0.missing", "out", spec="X_v1"),
                          row("instance_outputs", "body", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 3, "width": 8, "dtype": "bf16", "spec": "S_v1", "consumers": ["nowhere"]})),
                          row("mystery_family", "body.norm", "out")]
    plan, _, _ = make_plan(rows)
    assert not plan.complete
    un = {v.value_id: v for v in plan.unsupported}
    assert set(un) == {"body.blocks.0.missing/out", "body/out", "body.norm/out"}
    assert "not a module of the runtime tree" in un["body.blocks.0.missing/out"].why
    assert "consumer 0 is not a module" in un["body/out"].why
    assert "no site rule" in un["body.norm/out"].why
    doc = plan.to_json()
    assert doc["complete"] is False and {u["value"] for u in doc["unsupported"]} == set(un)
    # install refuses it by name
    from verity_vllm.acquire.install import IncompletePlan, install_from_plan
    torch = pytest.importorskip("torch")
    with pytest.raises(IncompletePlan):
        install_from_plan(plan, torch.nn.Module())


def test_call_boundaries_are_a_native_tap_of_their_module_body():
    rows = base_rows() + [row("call_boundaries", "body.blocks.0.norm_in/mul_1", "out", rows=3, width=8, dtype="f32"),
                          row("call_boundaries", "body.blocks.0.missing/add", "out", rows=3, width=8, dtype="f32")]
    plan, _, _ = make_plan(rows)
    v = {v.value_id: v for v in plan.values}["body.blocks.0.norm_in/mul_1/out"]
    assert v.chosen.kind == "native_tap" and v.chosen.module == "body.blocks.0.norm_in/mul_1" and "call-boundary source" in v.chosen.derived_from
    assert {u.value_id: u for u in plan.unsupported}["body.blocks.0.missing/add/out"].rejected[0]["why"].endswith("is not a module of the runtime tree")
    assert "body.blocks.0.norm_in/mul_1" not in {h.module for h in plan.hooks()}              # the source installs its own pre-hook


def test_sampler_without_runner_is_unsupported():
    rt = tree()
    rt.runner = {"sample": False, "sampler_module": False}
    plan, _, _ = make_plan(base_rows(), rt=rt)
    un = {v.value_id for v in plan.unsupported}
    assert un == {"runner.sampler/sampled_token_ids"}


def test_prescribed_padding_identities():
    rows = base_rows() + [row("instance_outputs", "body.blocks.0.act", "out", step=5, spec="Act_v1", boundary_class="padding")]
    plan, _, _ = make_plan(rows)
    # the padded row joins its Value class: the executed identities keep the runtime site, the prescribed site is listed beside
    v = {v.value_id: v for v in plan.values}["body.blocks.0.act/out"]
    assert not v.prescribed and v.prescribed_identities == 1 and v.identities == 3 and v.chosen.kind == "producer_output"
    assert [s.kind for s in v.sites] == ["prescribed", "producer_output"]
    # padding identities as the finalize phase knows them (`padding_steps_of_record`): every identity of a Value -> prescribed as a whole
    pad = [r for r in rows if r["op_path"] == "logits"]
    plan2, _, _ = make_plan(rows) if False else (P.acquisition_plan(manifest(rows), AnnotationCorrespondence(tree()), tables=TABLES, padding_identities=pad), None, None)
    lg = {v.value_id: v for v in plan2.values}["logits/out"]
    assert lg.prescribed and lg.chosen.kind == "prescribed" and lg.prescribed_identities == 2
    assert ("logits", "producer_output", "out") not in [h.key() for h in plan2.hooks()]
    assert plan2.complete


# ------------------------------------------------------------------------------------------------------------------ residuals

def test_residuals_name_the_table():
    plan, _, rt = make_plan(base_rows())
    res = {(r.table, r.module): r for r in plan.residuals}
    # no module-class table is a residual: the plan's sites are the manifest's Values
    assert {t for t, _m in res} <= {"HAZARD_LEAVES", "GATHER_FLUSH_LEAVES", "PRE_FLUSH_LEAVES", "ANALYTIC_TABLES"}
    # lifetime tables: the leaf table + the committer's two name rules (`embed` in the leaf; tuple slot 1 of a `norm` leaf = the residual slot)
    assert set(res[("HAZARD_LEAVES", "*")].values) == {"body.blocks.0.proj_a/0", "body.embed/out", "body.blocks.0.norm_in/1", "body.blocks.0.norm_out/1"}
    assert res[("GATHER_FLUSH_LEAVES", "*")].values == ["body.blocks.0.proj_a/0"]
    assert set(res[("PRE_FLUSH_LEAVES", "*")].values) == {"body.blocks.0.norm_in/0", "body.blocks.0.norm_in/1", "body.blocks.0.norm_in/out"}
    # the analytic buffer that IS a weights Value is not a residual; a buffer of that leaf without one is not either (only cos_sin / normalizer leaves)
    assert ("ANALYTIC_TABLES", "body.blocks.0.attn.rotary_emb.cos_sin_cache") not in res
    rows = [r for r in base_rows() if r["op_path"] != "body.blocks.0.attn.rotary_emb.cos_sin_cache"]
    plan2, _, _ = make_plan(rows)
    assert any(r.table == "ANALYTIC_TABLES" and r.module == "body.blocks.0.attn.rotary_emb.cos_sin_cache" for r in plan2.residuals)
    doc = plan.to_json()
    assert all(set(r) == {"table", "module", "cls", "reason", "values"} for r in doc["residuals"])


_PLAN_OF_FILES = """
import json, sys
if sys.argv[3] == "blocked":
    sys.modules["torch"] = None
from verity_vllm.acquire import plan as P
from verity_vllm.correspondence.reader_for_acquire import AnnotationCorrespondence, RuntimeTree
plan = P.acquisition_plan(json.load(open(sys.argv[1])), AnnotationCorrespondence(RuntimeTree.load(sys.argv[2])))
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

    rows = base_rows() + [row("instance_outputs", "body.blocks.0.qkv_proj", "0", spec="Gemm_v1"), row("instance_outputs", "body.norm", "out", spec="Norm_v1")]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest(rows)))
    tree(**{"body.blocks.0.qkv_proj": ["Lin", "Base", "Module"]}).save(tmp_path / "runtime_tree.json")
    free, blocked = _plan_in_a_fresh_process(tmp_path, "free"), _plan_in_a_fresh_process(tmp_path, "blocked")
    assert not blocked["torch"]
    assert blocked["tables"]["GATHER_FLUSH_LEAVES"] == sorted(flush_points.GATHER_FLUSH_LEAVES)
    assert blocked["tables"]["PRE_FLUSH_LEAVES"] == sorted(flush_points.PRE_FLUSH_LEAVES)
    assert blocked["residuals"]["GATHER_FLUSH_LEAVES"] == ["body.blocks.0.qkv_proj/0"] and blocked["residuals"]["PRE_FLUSH_LEAVES"] == ["body.norm/out"]
    assert (blocked["tables"], blocked["residuals"], blocked["digest"]) == (free["tables"], free["residuals"], free["digest"])


# ------------------------------------------------------------------------------------------------------------------ serialisation

def test_plan_digest_stable_and_order_independent(tmp_path):
    rows = base_rows()
    p1, _, _ = make_plan(rows)
    p2, _, _ = make_plan(list(reversed(rows)))
    assert p1.digest == p2.digest and p1.to_json()["plan_digest"] == p1.digest
    path = tmp_path / "plan.json"
    dig = p1.save(path)
    doc = P.load_plan(path)
    assert doc["plan_digest"] == dig and doc["schema"] == P.SCHEMA and P.hooks_of_doc(doc) == [h.key() for h in p1.hooks()]
    # any change to the body changes the digest; a tampered file is refused
    rows2 = copy.deepcopy(rows)
    rows2.append(row("instance_outputs", "body.norm", "out", spec="Norm_v1"))
    p3, _, _ = make_plan(rows2)
    assert p3.digest != p1.digest
    bad = json.loads(path.read_text())
    bad["hooks"] = bad["hooks"][1:]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        P.load_plan(path)
    # the digest is over the plan's CONTENT: the runtime tree's provenance string is not part of it -- the meta-instantiated model and the
    # live engine of the same row give the same sites (they did, Value for Value, on the reference row) and so the same digest
    rt = tree(); rt.source = "other"
    p4, _, _ = make_plan(rows, rt=rt)
    assert p4.digest == p1.digest and p4.body()["runtime_source"] == "other" != p1.body()["runtime_source"]
    assert P.plan_digest(p1.to_json()) == p1.digest                     # the saved document re-digests to itself (load_plan's check)


def _body_by_site_scan(p):
    """The plan body as `values_at` gives it: one scan of every Value per site (the reference the one-pass body must equal)."""
    b = p.body()
    b["hooks"] = [{**h, "values": p.values_at(s)} for h, s in zip(b["hooks"], p.hooks())]
    b["sites"] = [{**h, "values": p.values_at(s)} for h, s in zip(b["sites"], p.chosen_sites())]
    b["counts"]["sites_by_kind"] = {k: len(v) for k, v in sorted(p.sites_by_kind().items())}
    return b


def test_a_call_boundary_sized_plan_serialises_in_one_pass(monkeypatch):
    """Gemma-2-2B B1 1024/128 (m007): 75,112 call-boundary Values, one site each.  `body()` read `values_at` per site -- a scan of every
    Value per site -- and never finished writing the plan (47 min in prep.committer_setup; 468 s at B16 256/32's 19,432).  One index
    pass gives the same document, so the same digest."""
    p, _, _ = make_plan(base_rows())
    many = [P.ValuePlan(f"body.blocks.0.norm_in/mul_{i}", "out", "call_boundaries", 1, status="chosen",
                        chosen=P.AcquisitionSite("registered_input", f"body.blocks.0.norm_in/mul_{i}", "out")) for i in range(3000)]
    shared = [P.ValuePlan(f"body.blocks.0.act/v{i}", "out", "instance_outputs", 2, status="chosen", chosen=p.values[0].chosen) for i in range(5)]
    p.values = p.values + many + shared + [P.ValuePlan("body.gone", "out", "instance_outputs", 1, why="no legal site")]
    want = _body_by_site_scan(p)
    assert p.body() == want and P.plan_digest(want) == p.digest and p.body()["counts"]["unsupported"] == 1

    def scan(self, site):
        raise AssertionError("body() scans the Values per site")

    monkeypatch.setattr(P.AcquisitionPlan, "values_at", scan)
    assert p.body()["counts"]["sites"] == len(p.chosen_sites())
