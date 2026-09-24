"""The required-value manifest FORMAT (`manifest_format`, MAN-01): name / spec parsing, the identity digest, the coverage check's negatives
the manifest makes detectable BY CONSTRUCTION (omit a whole producer family; omit one step; drop one member and bind another twice; a member
committed short) and the binding loader's two engine-step shapes -- over manifests of record built by the v2 query
(`test_sampled_replay.v2_manifest`) on a synthetic wired Program (no torch, no pod)."""
from __future__ import annotations

import copy
import json

import pytest

from verity_capture.commit import binding as B
from verity_capture.commit.manifest_format import canonical_identity, coverage_check, manifest_digest, spec_statics, split_name
from verity_capture.commit.tests.test_sampled_replay import _program, v2_manifest

EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def _layouts_from(man: dict) -> dict[int, list[dict]]:
    """A collector's per-step layouts that cover the manifest exactly (the positive case), in the committed-name vocabulary."""
    lay: dict[int, list[dict]] = {}
    for r in man["identities"]:
        if r["family"] == "weights":
            continue
        lay.setdefault(int(r["step"]), []).append({"name": f"{r['op_path']}/{r['output_member']}", "numel": r["element_range"][1], "inv": r["invocation"]})
    for step in lay:                     # committed order: a module's k-th call is the k-th occurrence in the step
        lay[step].sort(key=lambda m: (m["name"], m["inv"]))
    return lay


def test_parsing():
    assert spec_statics("Gemm_v1{K=576,N=960}") == {"K": 576, "N": 960}
    assert spec_statics(f"RMSNormFusedCuda_v2{{N=576,{EPS}}}")["N"] == 576
    assert split_name("model.model.layers.3.self_attn.rotary_emb/rotary_embedding_3/key[7]") == ("model.model.layers.3.self_attn.rotary_emb", "rotary_embedding_3", "key", 7)
    assert split_name("argmax_2[0]") == ("", "argmax_2", "", 0)


def test_digest_determinism():
    inst, _woff = _program()
    man = v2_manifest(inst)
    assert man["complete"] is True and man["steps"] == [0, 1, 2] and man["engine_steps"] == [0, 1, 2]
    assert v2_manifest(inst)["manifest_digest"] == man["manifest_digest"]                     # rebuild -> same
    ids = copy.deepcopy(man["identities"])
    for r in ids:                                                                             # annotations do not enter
        r.pop("geometry", None)
        r.pop("rows", None)
    assert manifest_digest(ids) == man["manifest_digest"]
    assert v2_manifest(inst, digest="q" * 64)["manifest_digest"] != man["manifest_digest"]   # the Program digest does
    assert len(canonical_identity(man["identities"][0])) == 8


def test_coverage_positive_and_negatives():
    inst, _woff = _program()
    man = v2_manifest(inst)
    lay = _layouts_from(man)
    ok = coverage_check(man, lay)
    assert ok["ok"], ok["missing"][:3]
    checked = [r for r in man["identities"] if r["family"] != "weights"]
    assert ok["committed_unmatched_n"] == 0 and ok["checked"] == len(checked)

    # (1) omit a WHOLE producer family: the sampler never committed -> coverage failure naming the family
    sampled = {f"{r['op_path']}/{r['output_member']}" for r in man["identities"] if r["family"] == "sampled_token_ids"}
    lay1 = {s: [m for m in ms if m["name"] not in sampled] for s, ms in lay.items()}
    r1 = coverage_check(man, lay1)
    assert not r1["ok"] and r1["families_missing_entirely"] == ["sampled_token_ids"] and r1["missing_by_family"] == {"sampled_token_ids": len(man["steps"])}

    # (2) omit one step (the last decode step never committed) -> coverage failure on exactly that step
    last = man["steps"][-1]
    r2 = coverage_check(man, {s: ms for s, ms in lay.items() if s != last})
    assert not r2["ok"] and r2["steps_with_missing"] == [last] and r2["missing_n"] == sum(1 for r in checked if r["step"] == last)

    # (3) drop one output member of a two-member module and bind the other twice: same count, same module set -> coverage failure on it
    two = next((s, op, sorted(ms)) for (s, op), ms in _members_by_step_op(checked).items() if len(ms) >= 2)
    s3, op3, (keep, drop) = two[0], two[1], two[2][:2]
    lay3 = copy.deepcopy(lay)
    k = next(i for i, m in enumerate(lay3[s3]) if m["name"] == f"{op3}/{drop}")
    lay3[s3][k] = dict(lay3[s3][k], name=f"{op3}/{keep}")
    r3 = coverage_check(man, lay3)
    assert not r3["ok"] and r3["missing_n"] == 1 and r3["missing"][0]["output_member"] == drop and r3["missing"][0]["step"] == s3
    assert r3["committed_unmatched_n"] == 1                               # the duplicate binds nothing required (invocation 1 is not expected)

    # (4) a member committed SHORT (wrong element range: one row fewer) -> coverage failure (containment)
    tall = next(r for r in checked if r["family"] == "instance_outputs" and int(r.get("rows") or 1) > 1)
    lay4 = copy.deepcopy(lay)
    m = next(m for m in lay4[int(tall["step"])] if m["name"] == f"{tall['op_path']}/{tall['output_member']}")
    m["numel"] -= m["numel"] // int(tall["rows"])
    r4 = coverage_check(man, lay4)
    assert not r4["ok"] and r4["missing"][0]["op_path"] == tall["op_path"]


def _members_by_step_op(ids: list[dict]) -> dict[tuple[int, str], set[str]]:
    out: dict[tuple[int, str], set[str]] = {}
    for r in ids:
        if r["family"] == "instance_outputs" and int(r["invocation"]) == 0:
            out.setdefault((int(r["step"]), r["op_path"]), set()).add(r["output_member"])
    return out


def test_fresh_b1_manifest_loads_in_binding_and_composition_skips(tmp_path):
    """rel M-0213/M-0245: a B=1 manifest writes engine_steps as the request's engine-step NUMBERS (ints); binding.load_required_manifest
    keeps them as numbers and declares no composition (composition_check skips, named).  A B>=2 GLOBAL manifest declares the dict
    composition and the loader keeps it; a manifest mixing the two shapes is refused by name."""
    inst, _woff = _program()
    man = v2_manifest(inst)
    assert man["engine_steps"] and all(isinstance(e, int) for e in man["engine_steps"]), man["engine_steps"]
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(man))
    pop, hdr = B.load_required_manifest(str(p), man["program_digest"], requests=["r0"], steps=len(man["steps"]), ranks=[0], arrivals={"r0": 0})
    assert len(pop) > 0 and hdr["engine_steps"] == [] and hdr["engine_step_numbers"] == man["engine_steps"] and hdr["global"] is False
    comp = B.composition_check(hdr, {"entries": []})
    assert comp["ok"] and comp["checked_steps"] == 0 and "no engine-step composition" in comp["skipped"]
    g = v2_manifest(inst, ("r0", "r1"))
    pg = tmp_path / "global.json"
    pg.write_text(json.dumps(g))
    _popg, hdrg = B.load_required_manifest(str(pg), g["program_digest"], requests=["r0", "r1"], steps=len(man["steps"]), ranks=[0], arrivals={"r0": 0, "r1": 0})
    assert hdrg["global"] and len(hdrg["engine_steps"]) == len(g["engine_steps"]) and hdrg["engine_step_numbers"] == [e["engine_step"] for e in g["engine_steps"]]
    assert all(isinstance(e["members"], list) and e["rows"] is not None for e in hdrg["engine_steps"])
    bad = dict(man, engine_steps=[0, {"engine_step": 1, "members": []}])
    pb = tmp_path / "bad.json"
    pb.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="engine_steps mixes"):
        B.load_required_manifest(str(pb), man["program_digest"], requests=["r0"], steps=2, ranks=[0], arrivals={"r0": 0})
