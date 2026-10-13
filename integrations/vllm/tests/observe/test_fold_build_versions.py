"""The fold binds each launch to the Definition version the Build binds it to.

A Match compares the fold's Program with the Build's spec for spec, so a fold pattern that binds another version of a Definition than
the Build's vocabulary (`program/frontend/rules/vocab.py`) fails the Match on every instance of it.  Both sides are read off the code:

* the fold: every registered composite Definition a fold pattern module names (a module defining a class with `match(self, node, ...)`;
  each name resolved in the module, its local imports included).  A launch bound through a shared selector (`frontend.targets`) or
  through the vocabulary itself (`vocabulary_for(...).kind`) is the Build's choice by construction and names nothing here;
* the Build: the Definitions each role of the vocabulary binds, over representative statics and every registered value of the statics
  that select among Definitions.

Checked against each vocabulary a current Build derives under (the `TargetProfile` default and the profiles binding the same kinds):

1. a family the vocabulary binds is named by the fold only at versions the vocabulary binds;
2. when the fold names what a role binds and the role's registered-values twin (`<role>_value`, `Profile.registered_values`) binds
   another Definition, the fold names that one too.

`ALLOWED` holds the exceptions, one reason each.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import itertools
from functools import lru_cache
from pathlib import Path

import pytest

import verity_vllm
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition
from verity_vllm.program.frontend.rules.vocab import PROFILES
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.registry.hopper import Fa3InvSum

PACKAGE = Path(verity_vllm.__file__).parent
DEFAULT = TargetProfile().numerics_profile
CURRENT = sorted(n for n, v in PROFILES.items() if n and v.kinds == PROFILES[DEFAULT].kinds)

#: a representative value of every static a vocabulary role takes; an optional static is also tried at its default
STATICS = dict(K=64, N=64, I=8, H=64, E=8, TOPK=2, VPT=8, T=4, NH=2, KVH=1, D=64, BN=128, EPS=1e-6, V=16, C=1.0, G=128, S=1, WORLD=4,
               VS=8, START=0, NHEADS=2, W=32, BITS=0, CAP=50.0, MASKED_FROM=1, FA2_MASKED_FROM=1, RSF=2.5, KERNEL="block",
               ROPE=32, DR=16, DV=16, SM_SCALE=0.0625)
#: the statics that select among Definitions, at every registered value
SELECTING = {"DOT": tuple(TG.DOTS.values()), "INV": (P.Fa2InvSum, Fa3InvSum), "SCALE": TargetProfile.FP8_SCALE_CONSTRUCTIONS,
             "CONSTRUCTION": TargetProfile.SAMPLER_CONSTRUCTIONS,
             "TARGET": (TargetProfile.default(), TargetProfile.default().with_overrides(batch_invariant=False))}

_GEMMA2_RV = ("Gemma-2 only: its fold refuses GeluTanhMul_v3 under registered_values by name (gen_dense_gemma2_patterns), so no "
              "registered-values Gemma-2 fold completes")
#: Definition id -> why the fold may name it at another version than the Build (rule 1), or not name it (rule 2)
ALLOWED = {
    "AddScalarBf16_v3": _GEMMA2_RV,
    "AddScalarF32_v3": _GEMMA2_RV,
    "Bf16DivScalar_v3": _GEMMA2_RV,
    "Bf16MulScalar_v3": _GEMMA2_RV,
    "LayerNormAten_v2": "GPT-NeoX only, which has no Build: its GeluErf_v1 has no vocabulary kind",
}


def _chain(e: ast.AST) -> list[str] | None:
    if isinstance(e, ast.Name):
        return [e.id]
    if isinstance(e, ast.Attribute):
        c = _chain(e.value)
        return None if c is None else c + [e.attr]
    return None


def _is_pattern_module(tree: ast.Module) -> bool:
    return any(isinstance(m, ast.FunctionDef) and m.name == "match" and [a.arg for a in m.args.args[:2]] == ["self", "node"]
               for c in ast.walk(tree) if isinstance(c, ast.ClassDef) for m in c.body)


def _imported(module: str, node: ast.Import | ast.ImportFrom) -> dict:
    """The names an import statement binds (module-level or local), resolved."""
    out = {}
    if isinstance(node, ast.Import):
        for a in node.names:
            out[a.asname or a.name.split(".")[0]] = importlib.import_module(a.name if a.asname else a.name.split(".")[0])
        return out
    base = node.module or ""
    if node.level:
        pkg = module.rsplit(".", node.level)[0]
        base = f"{pkg}.{base}" if base else pkg
    m = importlib.import_module(base)
    for a in node.names:
        out[a.asname or a.name] = getattr(m, a.name) if hasattr(m, a.name) else importlib.import_module(f"{base}.{a.name}")
    return out


@lru_cache(maxsize=None)
def fold_references() -> dict[str, tuple[CompositeDefinition, str]]:
    """Definition id -> (the Definition, `module:line` of its first mention) over every fold pattern module."""
    out: dict[str, tuple[CompositeDefinition, str]] = {}
    for f in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(f.read_text())
        if not _is_pattern_module(tree):
            continue
        module = ".".join(("verity_vllm",) + f.relative_to(PACKAGE).with_suffix("").parts)
        ns: dict = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                ns.update(_imported(module, n))
        ns.update(vars(importlib.import_module(module)))
        for n in ast.walk(tree):
            c = _chain(n) if isinstance(n, (ast.Name, ast.Attribute)) else None
            if not c or c[0] not in ns:
                continue
            obj = ns[c[0]]
            try:
                for a in c[1:]:
                    obj = getattr(obj, a)
            except AttributeError:
                continue
            if isinstance(obj, CompositeDefinition):
                out.setdefault(obj.id, (obj, f"{module}:{n.lineno}"))
    return out


def _tries(p: inspect.Parameter) -> tuple:
    if p.name in SELECTING:
        return SELECTING[p.name]
    v = STATICS.get(p.name, p.default)
    return (v,) if p.default is inspect.Parameter.empty else (p.default, v)


@lru_cache(maxsize=None)
def build_bindings(profile: str) -> dict[str, frozenset]:
    """Role -> the Definitions (unspecialised) the profile's vocabulary binds it to."""
    out = {}
    for role, binder in PROFILES[profile].kinds.items():
        params = list(inspect.signature(binder).parameters.values())
        defs = set()
        for combo in itertools.product(*(_tries(p) for p in params)):
            try:
                d = binder(**{p.name: v for p, v in zip(params, combo)})
            except (TypeError, KeyError, ValueError):
                continue
            defs.add(d.defn if isinstance(d, SpecializedDefinition) else d)
        out[role] = frozenset(defs)
    return out


def violations(refs: dict[str, tuple[CompositeDefinition, str]], profile: str, allowed=ALLOWED) -> list[str]:
    bound = build_bindings(profile)
    versions: dict[str, set[int]] = {}
    for defs in bound.values():
        for d in defs:
            versions.setdefault(d.name, set()).add(d.version)
    named = {d for d, _ in refs.values()}
    out = []
    for i, (d, where) in sorted(refs.items()):
        if d.name in versions and d.version not in versions[d.name] and i not in allowed:
            out.append(f"rule 1: {where} names {i}; the {profile} vocabulary binds {d.name} at "
                       f"{sorted(f'v{v}' for v in versions[d.name])}")
    for role, defs in sorted(bound.items()):
        twin = bound.get(f"{role}_value", frozenset())
        for d in sorted(defs & named, key=lambda d: d.id):
            for t in sorted(twin - named, key=lambda t: t.id):
                if t.id not in allowed:
                    out.append(f"rule 2: {refs[d.id][1]} names {d.id} (role {role!r}) but no fold pattern names {t.id}, which "
                               f"{role}_value binds under registered_values")
    return out


@pytest.mark.parametrize("profile", CURRENT)
def test_the_fold_names_each_definition_at_the_version_the_build_binds(profile):
    assert not violations(fold_references(), profile), "\n".join(violations(fold_references(), profile))


@pytest.mark.parametrize("profile", CURRENT)
def test_every_role_of_the_vocabulary_binds_a_definition_over_the_statics(profile):
    """A role that binds nothing over `STATICS` / `SELECTING` takes a static they lack: the check above would not see its Definitions."""
    assert not [r for r, defs in build_bindings(profile).items() if not defs]


def test_every_allowed_definition_is_reported_without_its_entry():
    reported = "\n".join(v for profile in CURRENT for v in violations(fold_references(), profile, allowed={}))
    assert not [i for i in ALLOWED if f" {i}" not in reported], reported


def test_the_scan_reads_the_fold_pattern_modules():
    refs = fold_references()
    where = {w.rsplit(":", 1)[0] for _, w in refs.values()}
    assert {"verity_vllm.engine.profiles.gen_ov_moe_patterns", "verity_vllm.observe.fold.patterns.norms"} <= where, sorted(where)
    assert {"SiluMul_v2", "MoeSum_v1", "RMSNormTriton_v1", "RMSNormTriton_v3"} <= set(refs), sorted(refs)
    assert "b1-eager-v3" in CURRENT and "dense2-eager-v1" in CURRENT


def test_a_fold_pattern_naming_the_older_silu_is_a_violation():
    """The dense `SiluAndMul` named `SiluMul_v1` while the b1-eager-v3 Build binds `SiluMul_v2` (#820): exactly what rule 1 reports."""
    refs = {k: v for k, v in fold_references().items() if not k.startswith("SiluMul")}
    refs["SiluMul_v1"] = (b1.SiluMul, "verity_vllm.observe.fold.patterns.activations:37")
    assert violations(refs, DEFAULT) == ["rule 1: verity_vllm.observe.fold.patterns.activations:37 names SiluMul_v1; the b1-eager-v3 "
                                         "vocabulary binds SiluMul at ['v2']"]
