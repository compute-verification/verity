"""The attention family's split into Definitions and vocabulary changed no circuit and no task.

`fixtures/attention_split_facts.json` holds the facts of the tree before the split: each family Definition's id and canonical
descriptor digest at every circuit-check binding, pin and served binding (`targets.attention_spec` at served shapes, which reads the
moved `MASKED_FROM` rules), and circuit-check's task list.  The tree has the same facts, and keeps every task.  A change that means
to change one edits its entry in the fixture.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from test_circuit_check import PINS, REPO, SUITE, TEMPLATES

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "attention_split_facts.json").read_text())
PKG = "verity_vllm.program.registry"
REGISTRY = REPO / "integrations" / "vllm" / "verity_vllm" / "program" / "registry"
#: the family's Definition modules (verity#1130's DSA and sparse-MLA ones included), and `hopper`, which the split left as it was
MODULES = ("fa2_check_inf", "fa3_check_inf", "fa2_softcap", "boolean_attention", "boolean_softcap", "mla", "dsa", "boolean_dsa",
           "sparse_mla", "boolean_sparse_mla", "hopper")


def _top_level(tree: ast.Module) -> list[str]:
    """Top-level names bound by a def, a class or a plain assignment."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.Assign):
            out += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
    return out


def _family() -> dict[str, str]:
    """``id -> module`` for every Definition a family module binds at top level itself (not one it imports)."""
    from verity.ml.circuits.defs import FunctionDefinition, SpecializedDefinition

    out = {}
    for m in MODULES:
        mod = importlib.import_module(f"{PKG}.{m}")
        for name in _top_level(ast.parse((REGISTRY / f"{m}.py").read_text(encoding="utf-8"))):
            obj = getattr(mod, name, None)
            if isinstance(obj, FunctionDefinition):
                out[obj.base_id if isinstance(obj, SpecializedDefinition) else obj.id] = m
    return out


def _digest(fn) -> str:
    """The canonical descriptor digest of ``fn``: its Program's (`verity.ml.circuits.codec.program_digest`), a primitive's entry's."""
    from verity.ml.circuits.codec import canonical_json, definition_entry, program_digest
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.program import Program

    if isinstance(fn, PrimitiveDefinition):
        return hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest()
    return program_digest(Program(fn))


def _served_bindings() -> dict[str, object]:
    """`targets.attention_spec` on the registered targets at served shapes: the bindings the vocabulary computes (`MASKED_FROM`)."""
    from verity_vllm.program.frontend import targets as TGT
    from verity_vllm.program.frontend.target_profile import TargetProfile

    profiles = {"ada": TargetProfile(), "h100-fa3": TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=3),
                "h100-guard": TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=3,
                                            fa3_construction="guard-every-block"),
                "sm120-fa2": TargetProfile(compute_capability=(12, 0), num_sms=188)}
    #: (T, max_seqlen_q, row, own_rows): a first decode, a long decode, a prefill's last and first rows, a chunked prefill's row
    rows = [(1, 1, 0, 1), (300, 1, 0, 1), (130, 130, 129, 130), (130, 130, 0, 130), (300, 17, 16, 17)]
    out = {}
    for pname, prof in profiles.items():
        for NH, KVH, D in ((12, 2, 128), (8, 1, 64)):
            for T, sq, row, own in rows:
                for cap in (None, 50.0):
                    for scale_input in (False, True):
                        label = f"{pname}/NH={NH},KVH={KVH},D={D}/T={T},sq={sq},row={row},own={own}/cap={cap}/scale_input={scale_input}"
                        try:
                            out[label] = TGT.attention_spec(prof, T, NH, KVH, D, sq, row, own, cap=cap, scale_input=scale_input)
                        except (TGT.UnregisteredTarget, KeyError, ValueError) as e:
                            out[label] = f"{type(e).__name__}: {e}"
    return out


@pytest.fixture(scope="module")
def now():
    """The family's Definitions with each one's id and digest at every circuit-check binding, pin and served binding."""
    from circuit_check import targets as TG
    from verity.ml.circuits.codec import descriptor_id

    TG.load_registries()
    fam = _family()
    bound = {fid: fn for fid, fn in SUITE.items() if (getattr(fn, "base_id", None) or fn.id) in fam}
    pinned = {}
    for key, pin in PINS.get("definitions", {}).items():
        if key.split("{")[0] in fam:
            pinned[key] = pin
            if key not in bound:
                bound[key] = TG.definition(key)
    served = {}
    for label, fn in _served_bindings().items():
        if isinstance(fn, str):
            served[label] = fn
        else:
            served[label] = fn.id
            bound.setdefault(fn.id, fn)
    definitions = {fid: {"descriptor_id": descriptor_id(fn), "digest": _digest(fn)} for fid, fn in bound.items()}
    return {"family": fam, "definitions": definitions, "served": served, "pins": pinned}


@pytest.mark.slow
def test_every_family_definition_keeps_its_id_and_digest(now):
    """Every id and digest at every binding the snapshot holds, and every family Definition the snapshot names."""
    assert sorted(set(SNAPSHOT["family"]) - set(now["family"])) == []
    changed = {k: (v, now["definitions"].get(k)) for k, v in SNAPSHOT["definitions"].items() if now["definitions"].get(k) != v}
    assert changed == {}
    assert {k: now["pins"].get(k) for k in SNAPSHOT["pins"]} == SNAPSHOT["pins"]


@pytest.mark.slow
def test_the_served_bindings_are_unchanged(now):
    """`targets.attention_spec` binds the same Definition (`MASKED_FROM` included), or refuses the same way, at every served shape."""
    assert {k: now["served"].get(k) for k in SNAPSHOT["served"]} == SNAPSHOT["served"]


def test_circuit_check_keeps_every_task():
    assert sorted(set(SNAPSHOT["tasks"]["definitions"]) - set(SUITE)) == []
    templates = {label: json.loads(json.dumps(sts)) for label, (_sub, sts) in TEMPLATES.items()}
    assert {label: templates.get(label) for label, _ in SNAPSHOT["tasks"]["templates"]} == dict(SNAPSHOT["tasks"]["templates"])
