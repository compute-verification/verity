"""GP-01 composes components that share one registration (`Workload_v1`'s `shared.values` and tables): each records the same
registered fields with the same bits and the same tables (result.json `program.supplied`), or none registers any; parameter positions,
which shift with the request-local parameters, are not compared."""
from __future__ import annotations

import json

from verity_vllm.pipeline.components import recorded_registration, registration_refusal

FIELDS = [{"name": "rms_eps", "width": 32, "bits": "0x358637bd", "offset": 0}]


def component(tmp_path, name: str, supplied: dict | None):
    d = tmp_path / name
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "0" * 64, **({"supplied": supplied} if supplied else {})}}))
    return str(d)


def supplied(param: int, fields=FIELDS, tables=()):
    return {"registration": {"param": param, "name": "values", "fields": fields}, "run": [],
            **({"tables": [{"param": param + 1 + i, "name": n} for i, n in enumerate(tables)]} if tables else {})}


def test_components_that_register_the_same_values_compose_whatever_their_positions(tmp_path):
    a = component(tmp_path, "a", supplied(1, tables=("gelu_tanh_bf16",)))
    b = component(tmp_path, "b", supplied(4, tables=("gelu_tanh_bf16",)))
    assert recorded_registration(a) == {"values": FIELDS, "tables": ["gelu_tanh_bf16"]}
    assert registration_refusal([a, b]) is None
    assert registration_refusal([component(tmp_path, "c", None), component(tmp_path, "d", None)]) is None


def test_components_that_register_differently_are_refused(tmp_path):
    a = component(tmp_path, "a", supplied(1))
    other_bits = component(tmp_path, "b", supplied(1, fields=[{**FIELDS[0], "bits": "0x3727c5ac"}]))
    held = component(tmp_path, "c", None)
    tables = component(tmp_path, "d", supplied(1, tables=("gelu_tanh_bf16",)))
    for other in (other_bits, held, tables):
        assert "register different values" in registration_refusal([a, other])
