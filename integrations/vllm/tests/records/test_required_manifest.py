"""`records.required_manifest`: the required-value manifest as its writers write it, refused by name when a field they all write is
missing, and read by the replay's coverage (`check.replay.coverage`) to the same bytes as the parsed JSON.

`data/required_manifest.json` is what `query.required.request_manifest` writes for the replay tests' one-request Program
(`tests.check.test_sampled_replay._program`, frozen with `json.dump(..., indent=1, sort_keys=True)`); `data/replay_addresses.json` is
that Program's row addresses as the replay's population enumerates them for request r0 (`check.replay.population.population`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.check.c2_rules import _query_population_scope
from verity_vllm.check.replay import coverage as COV, opening as OP
from verity_vllm.query.manifest import format as MF
from verity_vllm.records.fields import MissingField, RecordError
from verity_vllm.records.required_manifest import SCHEMA, RequiredManifest

DATA = Path(__file__).parent / "data"
MANIFEST = DATA / "required_manifest.json"
REQUIRED = ("schema", "program_digest", "manifest_digest", "requests", "identities")
#: sha256 of the population record the replay writes over the fixture (`json.dumps(..., sort_keys=True, default=str)`), with the
#: population's row addresses and with none
POPULATION_SHA256 = {"rows": "dcc58359c703ebe43b029a9f420bd21d4105698c48296b778cb18b43682a1d53",
                     "none": "0303122052a443079b633608120c2dc484f1c2d2c95f4a34dad196386491fbb4"}


def _doc() -> dict:
    return json.loads(MANIFEST.read_text())


def _addresses() -> dict[tuple, dict]:
    return {tuple(k): v for *k, v in json.loads((DATA / "replay_addresses.json").read_text())}


def _sha256(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


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


@pytest.mark.parametrize("rows", ("rows", "none"))
def test_the_replay_writes_the_same_population_bytes_through_the_record(rows):
    addresses = _addresses() if rows == "rows" else {}
    doc = _doc()
    by_json = COV.query_population(doc, addresses, set(), ranks=[0])
    assert _sha256(by_json) == POPULATION_SHA256[rows]
    assert _sha256(COV.query_population(RequiredManifest.load(MANIFEST), addresses, set(), ranks=[0])) == POPULATION_SHA256[rows]
    assert (by_json["query_checks"]["ok"], by_json["query_total"]["identities"]) == (rows == "rows", 35)
    assert COV.form_b_families(doc) == COV.form_b_families(RequiredManifest.of(doc, "the test's manifest")) == {}


@pytest.mark.parametrize("field", REQUIRED)
def test_a_refused_manifest_fails_the_reconciliation_by_name_never_a_crash_or_a_pass(field):
    bad, addresses = {k: v for k, v in _doc().items() if k != field}, _addresses()
    qp = COV.query_population(bad, addresses, set(), ranks=[0])
    chk = qp["query_checks"]
    assert chk["ok"] is False and chk["manifest_refused"]["field"] == field
    assert chk["manifest_refused"]["why"] == f"{COV.HANDED}: field {field!r} is missing: every writer of the record writes it"
    # the reach is the replay's own row addresses, as with no manifest, and says so
    assert qp["query_total"]["source"].startswith("required manifest REFUSED") and qp["query_total"]["identities"] == len(addresses)
    note, fail = _query_population_scope({"population": {"query_total": qp["query_total"], "query_checks": chk, "scope": qp["scope"]}})
    assert note is None and f"required manifest refused, nothing reconciled ({chk['manifest_refused']['why']})" in fail
    with pytest.raises(MissingField, match=f"field {field!r} is missing"):
        COV.form_b_families(bad)
