"""The attention family's split (`integrations/vllm/tools/migrate/split_attention.py`) changed no circuit and no task.

`fixtures/attention_split_facts.json` is the script's `facts` of the tree before the split: each family Definition's id and canonical
descriptor digest at every circuit-check binding, pin and served binding (`targets.attention_spec` at served shapes, which reads the
moved `MASKED_FROM` rules), and circuit-check's task list.  The tree has the same facts, and keeps every task.  Regenerate the fixture
only with a change that means to change one, from its parent: ``split_attention.py facts --out F``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_circuit_check import PINS, REPO, SUITE, TEMPLATES

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "attention_split_facts.json").read_text())


def _split():
    spec = importlib.util.spec_from_file_location("split_attention", REPO / "integrations/vllm/tools/migrate/split_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def now():
    return _split().facts(REPO, SUITE, PINS)


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
