"""The sampling family's catalog split (`integrations/vllm/tools/migrate/split_sampling.py`) left circuit-check's view of the family as
it was: every circuit-check task of a family Definition, every subcircuit template instance of one (with its template and
statics) and every pin of one, recorded before the split in `data/split_sampling_tasks.json`, is still there."""
from __future__ import annotations

import json
from pathlib import Path

from circuit_check import checks as CK
from test_circuit_check import SUITE, TEMPLATES

DATA = json.loads((Path(__file__).parent / "data" / "split_sampling_tasks.json").read_text())


def test_the_family_definitions_are_still_checked():
    assert not [label for label in DATA["definitions"] if label not in SUITE]


def test_the_family_template_instances_are_unchanged():
    now = {label: {"label": label, "template": sub.template.name, "instance_statics": json.loads(json.dumps(sts))}
           for label, (sub, sts) in TEMPLATES.items()}
    assert not [t for t in DATA["templates"] if now.get(t["label"]) != t]


def test_the_family_pins_are_unchanged():
    pins = CK.load_pins()
    for kind, recorded in DATA["pins"].items():
        assert {i: pins[kind].get(i) for i in recorded} == recorded, kind
