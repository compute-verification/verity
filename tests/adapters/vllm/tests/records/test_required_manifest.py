"""`records.required_manifest`: the required-value manifest as its writers write it, refused by name when a field they all write is
missing, and its alias table the same through the record and the parsed JSON (`check.replay.opening.alias_table`).

`data/required_manifest.json` is what `query.required.request_manifest` writes for a one-request Program, frozen with
`json.dump(..., indent=1, sort_keys=True)`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.check.replay import opening as OP
from verity_vllm.query.manifest import format as MF
from verity_vllm.records.fields import MissingField, RecordError
from verity_vllm.records.required_manifest import SCHEMA, RequiredManifest

DATA = Path(__file__).parent / "data"
MANIFEST = DATA / "required_manifest.json"
REQUIRED = ("schema", "program_digest", "manifest_digest", "requests", "identities")


def _doc() -> dict:
    return json.loads(MANIFEST.read_text())


def test_a_writers_manifest_loads():
    doc = _doc()
    m = RequiredManifest.load(MANIFEST)
    assert m.where == str(MANIFEST) and doc["schema"] == SCHEMA == MF.SCHEMA
    assert (m.program_digest, m.manifest_digest, m.requests, m.identities) == (doc["program_digest"], doc["manifest_digest"], ["r0"],
                                                                               doc["identities"])
    assert m.query == doc["query"] and m.query["query_id"] == "Q_module_body_v1"
    assert m.op_path_aliases == {}
    assert (m.engine, m.guard, m.cross_check) == (None, None, None)                # no writer in this tree writes them
    assert RequiredManifest.of(m, "elsewhere") is m


@pytest.mark.parametrize("how", ("absent", "null"))
@pytest.mark.parametrize("field", REQUIRED)
def test_a_manifest_missing_a_required_field_is_refused_naming_the_file_and_the_field(tmp_path, field, how):
    doc = _doc()
    if how == "absent":
        del doc[field]
    else:
        doc[field] = None
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(MissingField) as e:
        RequiredManifest.load(p)
    assert (e.value.where, e.value.field) == (str(p), field)
    assert str(e.value) == f"{p}: field {field!r} is missing: every writer of the record writes it"


@pytest.mark.parametrize("field, value", [("schema", "verity-sweep/required-manifest/v0"), ("program_digest", 7), ("manifest_digest", ["ab"]),
                                          ("requests", "r0"), ("identities", {}), ("query", []), ("op_path_aliases", []), ("engine", 3),
                                          ("guard", "armed"), ("cross_check", True)])
def test_a_wrong_schema_or_a_mistyped_field_is_refused_by_name(field, value):
    with pytest.raises(RecordError) as e:
        RequiredManifest.of(dict(_doc(), **{field: value}), "the test's manifest")
    assert not isinstance(e.value, MissingField) and e.value.field == field
    assert str(e.value).startswith(f"the test's manifest: field {field!r} is ")


def test_a_manifest_that_is_not_a_json_object_is_refused():
    with pytest.raises(RecordError, match="the test's manifest: field 'schema' cannot be read: the manifest is a list"):
        RequiredManifest.of([_doc()], "the test's manifest")


def test_the_query_header_and_the_alias_table_are_none_where_a_writer_writes_none():
    doc = {k: v for k, v in _doc().items() if k not in ("query", "op_path_aliases")}
    m = RequiredManifest.of(doc, "the served-compiled manifest")
    assert (m.query, m.op_path_aliases) == (None, None)
    assert OP.alias_table(m) == {}


def test_the_alias_table_is_the_same_through_the_record_and_the_parsed_json():
    aliases = {"model.layers.1.self_attn.rotary_emb": ["model.layers.0.self_attn.rotary_emb", 1]}
    doc = dict(_doc(), op_path_aliases=aliases)
    table = OP.alias_table(doc)
    assert table == {"model.layers.1.self_attn.rotary_emb": ("model.layers.0.self_attn.rotary_emb", 1)}
    assert OP.alias_table(RequiredManifest.of(doc, "the test's manifest")) == table
    assert OP.alias_table({"op_path_aliases": aliases}) == table                     # a served window hands the store its aliases alone


