"""The attention family's Definition modules import only `verity` and Definition modules: what the catalog move needs of them.

A Definition module of the family (`split_attention.DEFINITION_MODULES`, and verity#1130's DSA and sparse-MLA modules when present)
imports the standard library, numpy, `verity`, the family's other Definition modules, and from another family's registry module
only the Definitions and the word constants of a Definition body that `OTHER_FAMILIES` names.  Its vocabulary (`split_attention.MOVES`)
is `attention_vocab`'s, which no Definition module imports.  An import is any import statement, at module level or in a function.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

VLLM = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("split_attention", VLLM / "tools" / "migrate" / "split_attention.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
REGISTRY = VLLM / "verity_vllm" / "program" / "registry"
PKG = S.PKG

#: another family's registry module -> the names a family Definition module may take from it (None: the whole module), and why
OTHER_FAMILIES = {
    "prims": (None, "the word primitives (F32MaxFtz, MufuEx2Ftz, F2fpBf16, ...) and their types (A16, BF16, F32, I32, bits_of)"),
    "b1": ({"ZERO16", "_lane", "_scale_log2_bits", "AttnBlockV2", "AttnBlockV3", "AttentionHeadV2", "AttentionHeadV3", "AttentionV2",
            "AttentionV3"},
           "Const16's zero, the mma accumulator's lane map and the softmax scale's word constant the bodies read, and the "
           "accepted chains the Boolean versions lower"),
    "dense": ({"softcap_mul_bits", "softcap_scale_log2_bits"}, "the softcap stage's word constants (`set_params_fprop`)"),
    "hopper": ({"Fa3InvSum", "HopperBF16WgmmaDot16"}, "FA3's reciprocal primitive and the Hopper BF16 dot (the B1 Hopper profile)"),
    "fp8": ({"E4M3"}, "the e4m3 type of DSA's quantized indexer keys"),
    "topp_word_gates": ({"I32ToF32"}, "the int-to-float conversion primitive DSA's top-k reads"),
}
THIRD_PARTY = {"numpy"}
#: the catalog, where the layout move put core's Definitions and silicon models
CATALOG = "verity.ml.circuits"


def _core(name: str) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, C-Flock, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.ml.commitments.multiproof", "verity.ml.private_circuits", "verity.ml.circuits.ops",
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock")
    return under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.core.claims"))))


def _catalog(name: str) -> bool:
    return name == CATALOG or name.startswith(CATALOG + ".")


def _definition_modules() -> list[str]:
    return [*S.DEFINITION_MODULES, *(m for m in S.OPTIONAL_DEFINITION_MODULES if (REGISTRY / f"{m}.py").exists())]


DEFINITION_MODULES = _definition_modules()


def _violations(mod: str) -> list[str]:
    tree = ast.parse((REGISTRY / f"{mod}.py").read_text(encoding="utf-8"))
    package = ["verity_vllm", "program", "registry"]
    out, aliases = [], {}

    def take(other: str, names: list[str], line: int) -> None:
        if other in DEFINITION_MODULES:
            return
        if other not in OTHER_FAMILIES:
            out.append(f"{mod}:{line}: imports {PKG}.{other}, neither a family Definition module nor an allowed one")
            return
        allowed = OTHER_FAMILIES[other][0]
        out.extend(f"{mod}:{line}: takes {other}.{n}" for n in names if allowed is not None and n not in allowed)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if a.name.startswith(PKG + "."):
                    out.append(f"{mod}:{node.lineno}: `import {a.name}` (name what it takes: from ... import)")
                elif not _core(a.name) and top not in THIRD_PARTY and top not in sys.stdlib_module_names and not _catalog(a.name):
                    out.append(f"{mod}:{node.lineno}: imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            base = (node.module if node.level == 0
                    else ".".join(package[:len(package) - node.level + 1] + ([node.module] if node.module else [])))
            top = base.split(".")[0]
            if base == PKG:
                for a in node.names:
                    aliases[a.asname or a.name] = a.name
                    if a.name not in DEFINITION_MODULES and a.name not in OTHER_FAMILIES:
                        out.append(f"{mod}:{node.lineno}: imports {PKG}.{a.name}")
            elif base.startswith(PKG + "."):
                take(base[len(PKG) + 1:], [a.name for a in node.names], node.lineno)
            elif (not _core(base) and top not in ("__future__", *THIRD_PARTY) and top not in sys.stdlib_module_names
                  and not _catalog(base)):
                out.append(f"{mod}:{node.lineno}: imports {base}")
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) in ("import_module", "__import__"):
            out.append(f"{mod}:{node.lineno}: imports by name")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            take(aliases[node.value.id], [node.attr], node.lineno)
    return out


@pytest.mark.parametrize("mod", DEFINITION_MODULES)
def test_a_definition_module_imports_only_verity_and_definition_modules(mod):
    assert _violations(mod) == []


def test_the_vocabulary_is_out_of_the_definition_modules():
    vocab = ast.parse((REGISTRY / f"{S.VOCAB}.py").read_text(encoding="utf-8"))
    assert {n for names in S.MOVES.values() for n in names} <= set(S._top_level(vocab))
    for mod, names in S.MOVES.items():
        assert not set(names) & set(S._top_level(ast.parse((REGISTRY / f"{mod}.py").read_text(encoding="utf-8")))), mod


def test_the_checker_catches_a_frontend_import(monkeypatch, tmp_path):
    (tmp_path / "fa2_check_inf.py").write_text(
        "from verity.ml.circuits.defs import composite\n"
        "from verity_vllm.program.registry import prims as P, targets\n"
        "from verity_vllm.program.registry.b1 import ZERO16, fa2_kblock_n\n"
        "from verity_vllm.config import X\n\n\n"
        "def f():\n    from verity_vllm.program.registry import hopper as H\n    return P.F32, H.SERVE_HOPPER, H.Fa3InvSum\n")
    monkeypatch.setattr(sys.modules[__name__], "REGISTRY", tmp_path)
    assert sorted(v.split(": ", 1)[1] for v in _violations("fa2_check_inf")) == [
        "imports verity_vllm.config", "imports verity_vllm.program.registry.targets", "takes b1.fa2_kblock_n", "takes hopper.SERVE_HOPPER"]
