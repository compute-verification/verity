"""The constant text a passing run writes into every record of its kind, pinned by name.

A word change alters no recorded text: a rule, a why, a claim, a convention or a scope a record carries keeps its bytes.
Each site is read from the producing module's source (its string parts in order, a formatted field as `{}`), so a branch
no small run reaches is held too.  A fault or refusal message is no site: it reaches a record only when a check fails.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

#: name -> (module, function or "" for the module's top level, what): what is `=NAME` (an assignment to NAME, or to a
#: subscript whose key is NAME), `:KEY` (every dict entry under KEY) or `~REGEX` (every assignment to a name matching REGEX)
SITES = {
    "population.EXECUTED_PREFIX_RULE": ("verity_vllm.check.replay.population", "", "=EXECUTED_PREFIX_RULE"),
    "population.NOT_EXECUTED_SCOPE": ("verity_vllm.check.replay.population", "", "=NOT_EXECUTED_SCOPE"),
    "population.MECH_RECURRENCE": ("verity_vllm.check.replay.population", "", "=MECH_RECURRENCE"),
    "population.MECH_ARITH_NO_SERVED": ("verity_vllm.check.replay.population", "", "=MECH_ARITH_NO_SERVED"),
    "population.phase_anchor.why": ("verity_vllm.check.replay.population", "phase_anchor", ":why"),
    "driver.strata.phase": ("verity_vllm.check.replay.driver", "", ":phase"),
    "driver.sample.order": ("verity_vllm.check.replay.driver", "", ":order"),
    "checks.RULE": ("verity_vllm.check.replay.checks", "", "=RULE"),
    "served_check.scope": ("verity_vllm.check.replay.served_check", "", "=scope"),
    "stoch.convention": ("verity_vllm.check.replay.stoch_recompute", "", ":convention"),
    "stoch.lag_slot": ("verity_vllm.check.replay.stoch_recompute", "", "=lag_slot"),
    "stoch.measured": ("verity_vllm.check.replay.stoch_recompute", "", "=measured"),
    "stoch.rng.state": ("verity_vllm.check.replay.stoch_recompute", "", ":state"),
    "stoch.rng.advances_iff_executed": ("verity_vllm.check.replay.stoch_recompute", "", ":advances_iff_executed"),
    "stoch.rng.checked_here": ("verity_vllm.check.replay.stoch_recompute", "", ":checked_here"),
    "commit_rules.claims": ("verity_vllm.check.commit_rules", "", "=claim"),
    "c2_rules.rules": ("verity_vllm.check.c2_rules", "", "~^C_[A-Z0-9_]+$"),
    "static_schedule.RULE": ("verity_vllm.engine.static_schedule", "", "=RULE"),
    "run_facts.profile_id_method": ("verity_vllm.engine.run_facts", "profile_manifest", "=profile_id_method"),
    "admission_bound.assumptions": ("verity_vllm.pipeline.admission_bound", "", ":assumptions"),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "admission_bound.assumptions": "6b144ea0d151798ce01343f6b66c60816522bb7fc069a9829ba548a28f94ded2",
    "c2_rules.rules": "69886fcd2109e1394a416e64425291fe47e492f8d7ecbc68eefda6e07439f96b",
    "checks.RULE": "7292bb22289d4388ff52cd1accb0971e219216b4fbeca124a9025bba12818cc9",
    "commit_rules.claims": "07bb035a8edc959bb14bbf48367e4f9afe926d89b7b1208d3fb5a0800815065e",
    "driver.sample.order": "341937c2945c316c9dd1822d8ee026b1c43a1e23f3cc62b84ed174ba97c2e135",
    "driver.strata.phase": "14c4334ec780a489790124eb8ee935619dba934eba73cefaa6cc1eebcd7b92a6",
    "population.EXECUTED_PREFIX_RULE": "9197ad66b361885b07b9ff23873c47e72e8843cf0c68812be14b0be8fa839612",
    "population.MECH_ARITH_NO_SERVED": "b469a93979d0cc8dcea0da15e29dc7bc1a712b3e815c8c1c27ad84ac62e8a0be",
    "population.MECH_RECURRENCE": "eded1cc9002d7e2f0f4c879ccd38eb141b6325a13258b57a124db9c5587499b2",
    "population.NOT_EXECUTED_SCOPE": "9fb6f110d7787962e1a84762ea722d517b962fa29ab2d79524de4f577a1a65a2",
    "population.phase_anchor.why": "1128c1a565c37eda828cbf9309ffcab05a429d35b8bcd072b03ff1663c979d19",
    "run_facts.profile_id_method": "6d251dc2d14943b52df29ea91d6ef0b0fadfe91708586fa95e5c847db5e15c56",
    "served_check.scope": "88da540c9adbb3b7ca0799a09a89ea36db877a88ef47b950cf068de4ecf2d2cd",
    "static_schedule.RULE": "618ec92b9680744475481c46b75a86ad7b465e7a02924d0126756b1a8a73d109",
    "stoch.convention": "4cdcd247b3949f030efdf4596af5cb03e6dcff494d1543fafc327a4bd488f469",
    "stoch.lag_slot": "be711e6640a7804238b0e4e12fd580da6255226dba1ede1e5c26d605e22b4136",
    "stoch.measured": "c3d0dae3e525cf10d130de25f651522c52dc0030a6c7d3839f892709a096039d",
    "stoch.rng.advances_iff_executed": "38200cfdcbc4c117187948d0b4f685928612190be9694ba2869e2c3260b44674",
    "stoch.rng.checked_here": "161537cbdaaede423123c1c64a4915c3d4d684bd725fd995716816ff0502a838",
    "stoch.rng.state": "71a253d886032a2ac33948669895039be4d4437342659fb77327538292f36bbc",
}


def _parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                        for v in node.values)]
    return [p for c in ast.iter_child_nodes(node) for p in _parts(c)]


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
        return [t.slice.value]
    if isinstance(t, ast.Tuple):
        return [n for e in t.elts for n in _target_names(e)]
    return []


def text(module: str, function: str, what: str) -> str:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, f"{module}: not importable"
    tree = ast.parse(Path(spec.origin).read_text())
    scopes = [tree] if not function else [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                          and n.name == function]
    assert scopes, f"{module}: no function {function}"
    kind, key = what[0], what[1:]
    parts: list[str] = []
    for scope in scopes:
        for n in ast.walk(scope):
            if kind == ":" and isinstance(n, ast.Dict):
                parts += [p for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == key for p in _parts(v)]
            elif kind in "=~" and isinstance(n, (ast.Assign, ast.AnnAssign)):
                names = [m for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) for m in _target_names(t)]
                if n.value is not None and any(m == key if kind == "=" else re.match(key, m) for m in names):
                    parts += _parts(n.value)
    assert parts, f"{module}{'.' + function if function else ''}: no {what} site; a moved site moves its entry here"
    return "\x00".join(parts)


def digest(name: str) -> str:
    return hashlib.sha256(text(*SITES[name]).encode()).hexdigest()


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(SITES))
def test_a_recorded_text_keeps_its_bytes(name):
    assert digest(name) == PINS[name], (
        f"{name}: the text {SITES[name][0]} writes into its records changed. A word change alters no recorded text: put "
        f"main's bytes back (`git show origin/main:<file>`). Only a new record format re-pins, with its schema id bumped.")
