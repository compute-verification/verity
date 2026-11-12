"""What the `test_attention_split*.py` files share: the facts of the tree before the attention family's split into Definitions
and vocabulary, and this tree's.

`fixtures/attention_split_facts.json` holds the facts of the tree before the split: each family Definition's id and canonical
descriptor digest at every circuit-check binding, pin and served binding (`targets.attention_spec` at served shapes, which reads the
moved `MASKED_FROM` rules), and circuit-check's task list.  The tree has the same facts, and keeps every task.  A change that means
to change one edits its entry in the fixture.

Each digest is taken of the Definition its id makes (`made`, as circuit-check makes a target from its id, without the catalog), in
`PARTS` files (`test_attention_split_digests_K.py`), a slice each, since each digest builds its Definition's body: in one file
they would be the suite's longest, which a split by file can't divide.  `test_attention_split.py` holds that the Definition at each
binding is that very object.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

from circuit_check import units as UN
from test_circuit_check import PINS, REPO

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "attention_split_facts.json").read_text())
PKG = "verity.definitions.registry"
REGISTRY = REPO / "verity" / "definitions" / "registry"
#: the family's Definition modules (the DSA and sparse-MLA ones included), and `hopper`, which the split left as it was
MODULES = ("fa2_check_inf", "fa3_check_inf", "fa2_softcap", "boolean_attention", "boolean_softcap", "mla", "dsa", "boolean_dsa",
           "sparse_mla", "boolean_sparse_mla", "hopper")
#: the digest files, `test_attention_split_digests_1.py` to `_{PARTS}.py`, of which file K holds slice K - 1
PARTS = 8
DIGESTS = "test_attention_split_digests_"


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


def family() -> dict[str, str]:
    """``id -> module`` for every Definition a family module binds at top level itself (not one it imports)."""
    from circuit_check import targets as TG
    from verity.circuits.defs import FunctionDefinition, SpecializedDefinition

    TG.load_registries()
    out = {}
    for m in MODULES:
        mod = importlib.import_module(f"{PKG}.{m}")
        for name in _top_level(ast.parse((REGISTRY / f"{m}.py").read_text(encoding="utf-8"))):
            obj = getattr(mod, name, None)
            if isinstance(obj, FunctionDefinition):
                out[obj.base_id if isinstance(obj, SpecializedDefinition) else obj.id] = m
    return out


def pinned(fam: dict[str, str]) -> dict[str, dict]:
    """circuit-check's ``definitions`` pins of the family's Definitions, by key."""
    return {key: pin for key, pin in PINS.get("definitions", {}).items() if key.split("{")[0] in fam}


def served_bindings() -> dict[str, object]:
    """`targets.attention_spec` on the registered targets at served shapes: the bindings the vocabulary computes (`MASKED_FROM`)."""
    from circuit_check import targets as TG
    from verity_vllm.program.frontend import targets as TGT
    from verity_vllm import TargetProfile

    TG.load_registries()
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


def made(fid: str):
    """The Definition id ``fid`` makes without the catalog (`units._from_id`: `targets.definition`, whose `defs.bind` shares a
    specialization per binding record, so a binding of that record holds this very object), or None."""
    return UN._from_id(fid)


def digest(fn) -> str:
    """The canonical descriptor digest of ``fn``: its Program's (`verity.circuits.codec.program_digest`), a primitive's entry's."""
    from verity.circuits.codec import canonical_json, definition_entry, program_digest
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.program import Program

    if isinstance(fn, PrimitiveDefinition):
        return hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest()
    return program_digest(Program(fn))


def changed_ids_and_digests(digests_file: str) -> dict[str, tuple]:
    """Of the slice `DIGESTS` file ``digests_file`` holds, the snapshot's Definitions whose id or digest, made from their id
    (`made`), is not the snapshot's, with both (None where the id makes none)."""
    from verity.circuits.codec import descriptor_id

    part = int(Path(digests_file).stem.removeprefix(DIGESTS)) - 1
    keys = sorted(SNAPSHOT["definitions"])[part::PARTS]
    got = {k: {"descriptor_id": descriptor_id(fn), "digest": digest(fn)} if (fn := made(k)) is not None else None for k in keys}
    return {k: (SNAPSHOT["definitions"][k], got[k]) for k in keys if got[k] != SNAPSHOT["definitions"][k]}
