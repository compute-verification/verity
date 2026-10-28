"""`records.workload`: every workload file and what each writer writes load as the typed workload and dump back to the same JSON; the
older v1 layout a pod's runs still hold loads too; a missing or unknown field is refused by name."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from verity_vllm.config import WORKLOAD_SCHEMA, RowConfig, workload_files
from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.workload import SCHEMA, V1, Workload

ROWS = workload_files()


def _v2() -> dict:
    return {"schema": SCHEMA, "model": "unsloth/Llama-3.2-1B", "case": "LLAMA32_1B",
            "engine": {"revision": "0" * 40, "vllm": "d9105ea80", "args": {}}, "seed": 0,
            "variants": {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}}, "seeds": [],
            "sampling": {"variant": "greedy", "seed": None, "n": 1, "ignore_eos": True}, "note": "n",
            "prompt_lengths": [3], "max_tokens": [4],
            "requests": [{"request_id": "a", "index": 0, "prompt_len": 3, "max_tokens": 4, "prompt_token_ids": [1, 2, 3],
                          "arrive_step": 0, "origin": "corpus", "prompt_text": "x"}]}


def _v1() -> dict:
    doc = {k: v for k, v in _v2().items() if k != "engine"}
    return doc | {"schema": V1, "tokenizer_revision": "0" * 40, "engine_args_required": {"max_model_len": 64}}


def _sha256(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_every_row_file_loads_and_dumps_back_to_itself():
    assert len(ROWS) > 70 and SCHEMA == WORKLOAD_SCHEMA
    for p in ROWS:
        doc = json.loads(p.read_text())
        w = Workload.load(p)
        assert w.dump() == doc and Workload.of(w.dump(), str(p)) == w, p.name
        assert w.doc_sha256 == _sha256(doc)
        cfg = RowConfig.of(doc)
        assert (w.model, w.case, w.engine) == (cfg.model, cfg.case, doc["engine"]), p.name


def test_the_writers_records_load():
    from verity_vllm.pipeline import workload as PW
    from verity_vllm.properties.holdout import make_workload

    held = make_workload(case="B0", model="m/x", vocab=1000, seed=7, n_requests=2, tokenizer_revision="0" * 40)
    w = Workload.of(held, "holdout")
    assert (w.sampling, w.sampling_variant, [r.prompt_len for r in w.requests]) == (None, None, [None, None])
    assert [r.prompt_length for r in w.requests] == [len(r["prompt_token_ids"]) for r in held["requests"]]
    plan = PW.plan("m__bf16__x__tp1__b2__i64__o8__mixed__greedy__eager", 2, 64, 8, sampling="greedy", arrivals="staggered")
    lengths_only = PW.workload_doc(plan, hf_repo="m/x", revision="0" * 40, case=None, tok=None)
    w = Workload.of(lengths_only, "plan")
    assert [r.prompt_token_ids for r in w.requests] == [None, None] and w.sweep["row_id"] == plan["row_id"]
    assert [r.prompt_length for r in w.requests] == plan["input_lengths"]
    assert w.dump() == {k: v for k, v in lengths_only.items() if v is not None}


def test_a_v1_workload_loads_with_its_flat_engine_keys():
    w = Workload.of(_v1(), "old run")
    assert (w.schema, w.engine, w.info["tokenizer_revision"]) == (V1, None, "0" * 40)
    assert w.dump() == _v1()
    with pytest.raises(UnknownField, match="'engine_args_required' is not a field"):
        Workload.of(_v2() | {"engine_args_required": {}}, "w")
    with pytest.raises(UnknownField, match="'engine' is not a field"):
        Workload.of(_v1() | {"engine": {}}, "w")


@pytest.mark.parametrize("path", ("schema", "model", "variants", "requests", "engine", "requests.0.request_id",
                                  "requests.0.max_tokens", "sampling.variant"))
@pytest.mark.parametrize("how", ("absent", "null"))
def test_a_missing_field_is_refused_by_name(path, how):
    doc = _v2()
    *up, last = path.split(".")
    d = doc
    for k in up:
        d = d[int(k)] if k.isdigit() else d[k]
    if how == "absent":
        del d[last]
    else:
        d[last] = None
    with pytest.raises(MissingField) as e:
        Workload.of(doc, "w.json")
    assert (e.value.where, e.value.field) == ("w.json", path.replace(".0.", "[0]."))


@pytest.mark.parametrize("path", ("extra", "requests.0.extra", "sampling.extra"))
def test_an_unknown_field_is_refused_by_name(path):
    doc = _v2()
    *up, last = path.split(".")
    d = doc
    for k in up:
        d = d[int(k)] if k.isdigit() else d[k]
    d[last] = 1
    with pytest.raises(UnknownField) as e:
        Workload.of(doc, "w.json")
    assert (e.value.where, e.value.field) == ("w.json", path.replace(".0.", "[0]."))


def test_a_wrong_schema_or_type_is_refused():
    with pytest.raises(RecordError, match="'schema' is 'verity-sweep/workload/v1'"):
        Workload.of(_v2() | {"schema": "verity-sweep/workload/v1"}, "w")
    doc = _v2()
    doc["requests"][0]["prompt_token_ids"] = [1, "2"]
    with pytest.raises(RecordError, match=r"'requests\[0\].prompt_token_ids' holds a value that is not an integer"):
        Workload.of(doc, "w")
    with pytest.raises(RecordError, match="'variants.greedy' is a float"):
        Workload.of(_v2() | {"variants": {"greedy": 0.0}}, "w")
    with pytest.raises(RecordError, match="not a JSON object"):
        Workload.of([], "w")


def test_a_record_passes_through_and_the_digest_is_of_the_document_read():
    doc = _v2()
    w = Workload.of(doc, "w")
    assert Workload.of(w, "elsewhere") is w and w.doc_sha256 == _sha256(doc)
    edited = copy.deepcopy(doc)
    edited["requests"][0]["seed"] = 5
    assert Workload.of(edited, "w").doc_sha256 != w.doc_sha256


def test_a_requests_arrival_is_its_own_else_the_sweeps_else_zero():
    doc = _v2()
    r = doc["requests"][0]
    doc["requests"] = [r, *({k: v for k, v in r.items() if k != "arrive_step"} | {"request_id": x} for x in "bc")]
    assert [Workload.of(doc, "w").arrive_step(i) for i in range(3)] == [0, 0, 0]
    doc["requests"][0]["arrive_step"] = 4
    doc["sweep"] = {"arrive_steps": [9, 5]}
    w = Workload.of(doc, "w")
    assert [w.arrive_step(i) for i in range(3)] == [4, 5, 0]
