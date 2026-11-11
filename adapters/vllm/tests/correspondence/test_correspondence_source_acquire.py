"""The plan reads its owning modules off ONE reader -- the `runtime-correspondence/v1` record (`verity_vllm.correspondence.runtime`) -- and
falls back to the manifest's v1-annotation reading only when a Build carries no record; either way the plan says which (`correspondence`
block), and the plan's CONTENT (sites, digest) is the same under both sources on the same manifest + tree."""
from __future__ import annotations

import gzip
import json

from verity_vllm.correspondence import runtime as RC
from verity_vllm.acquire import gate as G
from verity_vllm.acquire import plan as P
from verity_vllm.correspondence.reader_for_acquire import AcquireCorrespondence, AnnotationCorrespondence, read_program_correspondence

from tests.acquire.test_plan import TABLES, base_rows, manifest, tree


def _record(modules_of_calls: list[str], *, prefix: str = "model.") -> dict:
    """A `runtime-correspondence/v1` document: Call k lowered from an occurrence whose owning module (Program vocabulary = prefix + collector path)."""
    occs, calls, seen = [], [], {}
    for k, m in enumerate(modules_of_calls):
        pm = (prefix + m) if m else ""
        if pm not in seen:
            seen[pm] = len(occs)
            occs.append({"node": f"n{k}", "module": pm, "module_class": None, "op": "aten.op", "rule": None, "outputs": [{"shape": [8], "dtype": "bf16"}],
                         "written": [], "inputs": [], "lifetime": {}, "collective": None})
        calls.append({"occ": seen[pm], "member": None, "row": k, "hint": None, "returns": [{"port": 0, "output": 0, "leaves": [0, 8]}], "args": []})
    return {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": None, "rank": None, "occurrences": occs, "calls": calls}


def _program_dir(tmp_path, name: str, doc: dict | None, *, recorded_digest: str | None | bool = True) -> str:
    d = tmp_path / name
    d.mkdir()
    desc = {"schema": "x", "program_fn": "f", "annotations": ({"node_hints": {"0": "a/b[0]"}} | ({"correspondence": doc} if doc is not None else {}))}
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    prog: dict = {"digest": "p" * 64}
    if doc is not None and recorded_digest is True:
        prog["correspondence_digest"] = RC.correspondence_digest(doc)
    elif isinstance(recorded_digest, str):
        prog["correspondence_digest"] = recorded_digest
    (d / "result.json").write_text(json.dumps({"ok": True, "program": prog}))
    return str(d)


MODULES = ["body.embed", "body.blocks.0.norm_in", "body.blocks.0.proj_a", "body.blocks.0.act", "body.blocks.0.norm_out", "logits", ""]


def test_same_plan_under_both_sources_and_the_source_is_recorded():
    man, rt = manifest(base_rows(), wrapper_prefix="model."), tree()
    fallback = AnnotationCorrespondence(rt, wrapper_prefix="model.")
    canonical = AcquireCorrespondence(rt, wrapper_prefix="model.", tables={"build_request": RC.RuntimeCorrespondence.from_json(_record(MODULES))},
                                      digests={"build_request": "d" * 64})
    assert fallback.source == "v1_annotations" and canonical.source == "correspondence"
    a, b = P.acquisition_plan(man, fallback, tables=TABLES), P.acquisition_plan(man, canonical, tables=TABLES)
    assert a.digest == b.digest                                                   # the content is the same: the source is provenance
    assert a.correspondence["source"] == "v1_annotations" and "fallback" in a.correspondence and a.correspondence["modules"] == {"applicable": False, "confirmed": 0, "unconfirmed": []}
    assert b.correspondence["source"] == "correspondence" and b.correspondence["schema"] == RC.SCHEMA and b.correspondence["programs"] == {"build_request": "d" * 64}
    assert b.correspondence["modules"]["applicable"] and b.correspondence["modules"]["unconfirmed"] == []
    assert b.correspondence["modules"]["confirmed"] == sum(1 for v in b.values if P.ROUTE_OF_FAMILY.get(v.family) in P.MODULE_ROUTES) + sum(len(v.consumers) for v in b.values)
    da, db = a.to_json(), b.to_json()
    assert da["correspondence"] != db["correspondence"] and da["plan_digest"] == db["plan_digest"]
    assert P.load_plan.__name__ and P.plan_digest(db) == db["plan_digest"]
    cmp = G.compare_plans(da, db)
    assert cmp["identical"] and cmp["plan_digest_equal"] and cmp["differing"] == [] and cmp["a"]["correspondence_source"] == "v1_annotations" and cmp["b"]["correspondence_source"] == "correspondence"


def test_unconfirmed_module_is_a_listed_finding_not_a_changed_site():
    man, rt = manifest(base_rows(), wrapper_prefix="model."), tree()
    partial = [m for m in MODULES if m != "body.blocks.0.act"]                     # the record owns no Call under body.blocks.0.act
    corr = AcquireCorrespondence(rt, wrapper_prefix="model.", tables={"build_request": RC.RuntimeCorrespondence.from_json(_record(partial))})
    plan = P.acquisition_plan(man, corr, tables=TABLES)
    assert plan.digest == P.acquisition_plan(man, AnnotationCorrespondence(rt, wrapper_prefix="model."), tables=TABLES).digest
    unc = plan.correspondence["modules"]["unconfirmed"]
    assert [u["value"] for u in unc] == ["body.blocks.0.act/out"] and unc[0]["module"] == "body.blocks.0.act" and unc[0]["role"] == "owning module"
    act = next(v for v in plan.values if v.value_id == "body.blocks.0.act/out")
    assert act.chosen is not None and act.chosen.kind == "producer_output"      # the manifest's op_path stands


def test_global_manifest_without_wrapper_prefix_determines_it_from_the_runtime_tree():
    """A GLOBAL manifest (`build-global`) carries no `wrapper_prefix`: the prefix under which the record's `model.body...` paths are the
    tree's `body...` modules is determined from the tree, so the confirmation is real (not 0 confirmed / all unconfirmed), and the plan's
    provenance says where the prefix came from.  The plan's content is unchanged either way."""
    man, rt = manifest(base_rows(), wrapper_prefix="model."), tree()
    man = {k: v for k, v in man.items() if k != "wrapper_prefix"}                  # as compose_global writes it
    table = RC.RuntimeCorrespondence.from_json(_record(MODULES))
    given = AcquireCorrespondence(rt, wrapper_prefix="model.", tables={"p": table})
    inferred = AcquireCorrespondence(rt, wrapper_prefix="", tables={"p": table})
    assert given.wrapper_prefix_source == "manifest" and inferred.wrapper_prefix_source == "runtime_tree" and inferred.wrapper_prefix == "model."
    assert inferred.owned_modules() == given.owned_modules() and inferred.confirmed("body.blocks.0.act") is True
    pa, pb = P.acquisition_plan(man, given, tables=TABLES), P.acquisition_plan(man, inferred, tables=TABLES)
    assert pa.digest == pb.digest and pb.correspondence["modules"]["unconfirmed"] == [] and pb.correspondence["modules"]["confirmed"] == pa.correspondence["modules"]["confirmed"] > 0
    assert pb.correspondence["wrapper_prefix"] == "model." and pb.correspondence["wrapper_prefix_source"] == "runtime_tree"
    # the fallback has no record to confirm against: no prefix is determined
    fb = AnnotationCorrespondence(rt, wrapper_prefix="")
    assert fb.wrapper_prefix == "" and fb.wrapper_prefix_source is None
    # any wrapper attribute chain is determined the same way; a record whose paths meet the tree under NO prefix gets "" and every module
    # reads unconfirmed (shown in the provenance, not hidden)
    deep = AcquireCorrespondence(rt, wrapper_prefix="", tables={"p": RC.RuntimeCorrespondence.from_json(_record(MODULES, prefix="serve.wrapped.model."))})
    assert deep.wrapper_prefix == "serve.wrapped.model." and deep.confirmed("body.blocks.0.act") is True
    alien = AcquireCorrespondence(rt, wrapper_prefix="", tables={"p": RC.RuntimeCorrespondence.from_json(_record(["zz.a", "zz.b.c", ""], prefix="q."))})
    assert alien.wrapper_prefix == "" and alien.confirmed("body.blocks.0.act") is False and alien.owned_modules() == ["", "q.zz.a", "q.zz.b.c"]


def test_owning_module_by_call_id_reads_the_record():
    corr = AcquireCorrespondence(tree(), wrapper_prefix="model.", tables={"p": RC.RuntimeCorrespondence.from_json(_record(MODULES))},
                                 aliases={"body.blocks.0.act": ["body.blocks.0.norm_in", 1]})
    assert corr.owning_module({"call": 2}) == "body.blocks.0.proj_a"
    assert corr.owning_module({"call": 3, "program": "p"}) == "body.blocks.0.norm_in"     # op_path_aliases apply to the record's path too
    assert corr.owning_module({"i": 6, "name": "argmax[0]"}) == "" and corr.owning_module({"name": "model.body.embed/e[0]"}) == "body.embed"
    assert corr.confirmed("body.embed") is True and corr.confirmed("runner.sampler") is False
    assert AnnotationCorrespondence(tree()).confirmed("body.embed") is None


def test_from_programs_presence_test_digest_check_and_uniform_fallback(tmp_path):
    doc = _record(MODULES)
    p_ok = _program_dir(tmp_path, "build_request", doc)
    p_v1 = _program_dir(tmp_path, "build_request_LP1_T1", None)
    p_bad = _program_dir(tmp_path, "build_request_LP2_T2", doc, recorded_digest="0" * 64)
    t, dg, prob = read_program_correspondence(p_ok)
    assert t is not None and dg == RC.correspondence_digest(doc) and prob is None
    assert read_program_correspondence(p_v1) == (None, None, None)              # v1 Build: result.json records no digest, the descriptor is not read
    t2, _dg2, prob2 = read_program_correspondence(p_bad)
    assert t2 is not None and prob2 and "!=" in prob2
    c = AcquireCorrespondence.from_programs(tree(), [p_ok], wrapper_prefix="model.")
    assert c.source == "correspondence" and list(c.digests) == ["build_request"] and c.absent == [] and c.problems == []
    mixed = AcquireCorrespondence.from_programs(tree(), [p_ok, p_v1], wrapper_prefix="model.")
    assert mixed.source == "v1_annotations" and mixed.absent == ["build_request_LP1_T1"] and mixed.tables == {} and any("falls back" in p for p in mixed.problems)
    assert G.request_program_dirs(tmp_path) == sorted([p_ok, p_v1, p_bad])
    assert AcquireCorrespondence.from_programs(tree(), [], wrapper_prefix="model.").source == "v1_annotations"


def test_compare_plans_lists_differing_values_by_value():
    man, rt = manifest(base_rows(), wrapper_prefix="model."), tree()
    a = P.acquisition_plan(man, AnnotationCorrespondence(rt, wrapper_prefix="model."), tables=TABLES).to_json()
    b = json.loads(json.dumps(a))
    v = next(x for x in b["values"] if x["value"] == "logits/out")
    v["chosen"] = dict(v["chosen"], module="head")
    b["plan_digest"] = P.plan_digest(b)
    cmp = G.compare_plans(a, b)
    assert not cmp["identical"] and not cmp["plan_digest_equal"] and [d["value"] for d in cmp["differing"]] == ["logits/out"]
    assert cmp["differing"][0]["a"]["chosen"][0] == "logits" and cmp["differing"][0]["b"]["chosen"][0] == "head"
