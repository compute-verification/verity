"""`records.served_tokens`: the driver's served tokens as its three writers write them, refused by name when a field they all
write is missing, and read by the global match's chronology check (`check.match.chronology`, G6) to the same document as the parsed JSON.

`data/served_tokens.json` is a real record: the one-process capture's `match/capture/tokens.json` of the regression workload
`llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager` (`tests/regression/files.toml`, sha256
c83936017d07...), byte for byte.  The G6 runs are the global match's toy (`tests.check.test_global_match`), with the driver's record rewritten
to each writer's field set and to the claims G6 holds it to.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.check import test_global_match as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401 - fixture
from verity_vllm.check.match import global_match as gm
from verity_vllm.records.fields import MissingField, RecordError
from verity_vllm.records.served_tokens import ServedRequest, ServedTokens

DATA = Path(__file__).parent / "data"
REAL = DATA / "served_tokens.json"
#: sha256 of the global match's outcomes over `_corpus` (`json.dumps(..., sort_keys=True, default=str)`, the run directory replaced), computed
#: with G6 on the parsed JSON before it read the record, then once more when G6 began refusing a declared request the record has no
#: entry for: that turned `a request unserved` and `no requests` from PASS to FAIL, and changed no other case
OUTCOMES_SHA256 = "e1c4765cc877c03197d6e5546636ead5397ab51157445b575fcb4b590e513d8d"
REQUIRED = ("requests", "requests[0].request_id", "requests[0].generated_token_ids", "lag.value")
OPTIONAL = ("arrive_step_realized", "first_step", "finish_step", "finish_reason", "served_output_len", "executed_prefix_len")
#: the TP capture's request fields (`pipeline.tp.capture`), and the TP Match arms' (`pipeline.tp.match`)
TP_CAPTURE = ("request_id", "prompt_token_ids", "max_tokens", "generated_token_ids", "accepted_record_token_ids", "equals_accepted")
TP_MATCH = ("request_id", "prompt_token_ids", "max_tokens", "generated_token_ids")


def _real() -> dict:
    return json.loads(REAL.read_text())


def _drop(doc: dict, path: str) -> None:
    """Delete the field at a dotted path (`requests[0].request_id`)."""
    *head, last = path.replace("[0]", ".0").split(".")
    for k in head:
        doc = doc[int(k)] if k.isdigit() else doc[k]
    del doc[last]


def _set(doc: dict, path: str, value) -> None:
    *head, last = path.replace("[0]", ".0").split(".")
    for k in head:
        doc = doc[int(k)] if k.isdigit() else doc[k]
    doc[int(last) if last.isdigit() else last] = value


def _tp(doc: dict, fields: tuple[str, ...]) -> dict:
    return {"requests": [{k: r[k] for k in fields if k in r} for r in doc["requests"]]}


def test_the_real_record_loads():
    t = ServedTokens.load(REAL)
    assert t.where == str(REAL) and t.lag == 1 and len(t.requests) == 1
    r = t.requests[0]
    assert r == ServedRequest(request_id="d73b90fd-3f5e-5f82-a4c7-16a7321ad556", generated_token_ids=_real()["requests"][0]["generated_token_ids"],
                              arrive_step_realized=0, first_step=0, finish_step=31, finish_reason="length", served_output_len=32,
                              executed_prefix_len=32)
    assert len(r.generated_token_ids) == 32 and t.by_request == {r.request_id: r}
    assert ServedTokens.of(t, "elsewhere") is t


@pytest.mark.parametrize("fields", [TP_CAPTURE, TP_MATCH], ids=["tp-capture", "tp-match"])
def test_a_tp_record_loads_with_no_driver_claims(fields):
    t = ServedTokens.of(_tp(_real(), fields), "tokens.json")
    assert t.lag is None and [r.request_id for r in t.requests] == ["d73b90fd-3f5e-5f82-a4c7-16a7321ad556"]
    assert all(getattr(t.requests[0], k) is None for k in OPTIONAL)


@pytest.mark.parametrize("path", REQUIRED)
def test_a_missing_required_field_is_refused(path):
    doc = _real()
    _drop(doc, path)
    with pytest.raises(MissingField) as e:
        ServedTokens.of(doc, "capture/tokens.json")
    assert str(e.value) == f"capture/tokens.json: field {path!r} is missing: every writer of the record writes it"
    assert e.value.field == path


@pytest.mark.parametrize("path", ["requests", "requests[0].request_id", "requests[0].generated_token_ids"])
def test_a_null_required_field_is_refused(path):
    doc = _real()
    _set(doc, path, None)
    with pytest.raises(MissingField) as e:
        ServedTokens.of(doc, "capture/tokens.json")
    assert str(e.value) == f"capture/tokens.json: field {path!r} is missing: every writer of the record writes it"


def test_a_null_lag_value_is_an_unreported_lag():
    doc = _real()
    doc["lag"]["value"] = None
    assert ServedTokens.of(doc, "capture/tokens.json").lag is None


@pytest.mark.parametrize("path,value,kind", [("requests", {}, "dict, not a list or tuple"), ("requests[0]", 5, "int, not a Mapping"),
                                             ("requests[0].request_id", 7, "int, not a str"),
                                             ("requests[0].generated_token_ids", "7 8", "str, not a list or tuple"),
                                             ("requests[0].first_step", "0", "str, not a int"), ("requests[0].finish_reason", 1, "int, not a str"),
                                             ("requests[0].executed_prefix_len", 3.0, "float, not a int"), ("lag", 1, "int, not a Mapping"),
                                             ("lag.value", "1", "str, not a int")])
def test_a_mistyped_field_is_refused(path, value, kind):
    doc = _real()
    _set(doc, path, value)
    with pytest.raises(RecordError) as e:
        ServedTokens.of(doc, "capture/tokens.json")
    assert str(e.value) == f"capture/tokens.json: field {path!r} is a {kind}"


def test_a_non_object_is_refused():
    with pytest.raises(RecordError, match="field 'requests' cannot be read: the record is a list, not a JSON object"):
        ServedTokens.of([], "capture/tokens.json")


def _tokens(oracle: str) -> dict:
    return json.loads(Path(oracle, "capture", "tokens.json").read_text())


def _write(oracle: str, doc: dict | None) -> None:
    p = Path(oracle, "capture", "tokens.json")
    if doc is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(json.dumps(doc))


def _req(doc: dict, rid: str) -> dict:
    return next(r for r in doc["requests"] if r["request_id"] == rid)


def _outcome(gp: dict, oracle: str, tokens_optional: bool = False) -> dict:
    res = gm.check(gp, oracle, run_x09=False, record_alternate=False, tokens_optional=tokens_optional)
    return {"verdict": res["verdict"], "failed": res["failed_checks"], "G6": T._check(res, "G6"), "requests": res["requests"],
            "lag": res.get("lag"), "concurrency": res.get("achieved_concurrency")}


def _corpus(tmp_path: Path) -> dict[str, dict]:
    """The global match over the driver's record as each writer writes it and as each G6 path reads it: served to the cap, an EOS stop under a
    synchronous Program, an EOS stop under a LAG-1 Program with the one-process capture's claims; agreeing and not."""
    out: dict[str, dict] = {}

    def run(name, gp, oracle, base, edit=None, tokens_optional=False):
        doc = copy.deepcopy(base)
        if edit is not None:
            doc = edit(doc)
        _write(oracle, doc)
        out[name] = _outcome(gp, oracle, tokens_optional)

    gp, oracle = T.assemble([T.R0, T.R1], tmp_path / "cap")
    base = _tokens(oracle)

    def swapped(d):
        a, b = d["requests"]
        a["generated_token_ids"], b["generated_token_ids"] = b["generated_token_ids"], a["generated_token_ids"]
        return d

    def edit(rid, **kw):
        def f(d):
            r = _req(d, rid)
            for k, v in kw.items():
                if v is _POP:
                    r.pop(k)
                else:
                    r[k] = v
            return d
        return f

    run("as written", gp, oracle, base)
    run("swapped outputs", gp, oracle, base, swapped)
    run("realized arrival differs", gp, oracle, base, edit("r1", arrive_step_realized=3))
    run("first step stands for an unwritten arrival", gp, oracle, base, edit("r1", arrive_step_realized=_POP, first_step=2))
    run("finish step differs", gp, oracle, base, edit("r0", finish_step=9))
    run("tp capture", gp, oracle, base, lambda d: _tp(d, TP_CAPTURE))
    run("tp match arm", gp, oracle, base, lambda d: _tp(d, TP_MATCH))
    run("a request unserved", gp, oracle, base, lambda d: {"requests": [_req(d, "r0")]})
    run("no requests", gp, oracle, base, lambda d: {"requests": []})
    run("lag 0 written", gp, oracle, base, lambda d: d | {"lag": {"value": 0, "async_scheduling": False, "source": "test"}})
    run("lag unreported", gp, oracle, base, lambda d: d | {"lag": {"value": None, "async_scheduling": None, "source": "test"}})
    run("lag 1 against a synchronous record", gp, oracle, base, lambda d: d | {"lag": {"value": 1, "async_scheduling": True, "source": "test"}})
    run("no tokens.json", gp, oracle, None)
    run("no tokens.json, optional", gp, oracle, None, tokens_optional=True)

    gp, oracle = T.assemble([T.R0, T.R1_EOS], tmp_path / "eos")
    base = _tokens(oracle)
    run("eos stop", gp, oracle, base)
    run("eos stop called a length stop", gp, oracle, base, edit("r1", finish_reason="length"))
    run("eos stop, executed length unattested", gp, oracle, base, edit("r1", served_output_len=2, executed_prefix_len=None))

    gp, oracle = T.assemble([T.R0, T.R1_LAG], tmp_path / "lag", lag=1)
    T._set_async_scheduling(oracle, True)
    T._driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
    T._driver_claims(oracle, "r1", lag=1, served_output_len=2, executed_prefix_len=3, cap_exhausted=False)
    T._driver_claims(oracle, "r0", served_output_len=3, executed_prefix_len=3, cap_exhausted=True)
    base = _tokens(oracle)
    run("lag 1 claims agree", gp, oracle, base)
    run("lag 1, the driver's lag disagrees", gp, oracle, base, lambda d: d | {"lag": dict(d["lag"], value=0)})
    run("lag 1, executed claim differs", gp, oracle, base, edit("r1", executed_prefix_len=2))
    run("lag 1, served claim differs", gp, oracle, base, edit("r0", served_output_len=2))
    run("lag 1, finish step at the lag row", gp, oracle, base, edit("r1", finish_step=2))
    return out


_POP = object()


def test_g6_reads_the_record_to_the_same_document(memory, tmp_path):
    """Every outcome of the global match over `_corpus` (verdict, G6's check, the per-request account) is what it was when G6 read the parsed
    JSON."""
    got = _corpus(tmp_path)
    assert {k: v["verdict"] for k, v in got.items()}["as written"] == "PASS"
    blob = json.dumps(got, sort_keys=True, default=str).replace(str(tmp_path), "<run>")
    assert hashlib.sha256(blob.encode()).hexdigest() == OUTCOMES_SHA256, {k: (v["verdict"], v["G6"]["problems"][:2]) for k, v in got.items()}


@pytest.mark.parametrize("path,why", [("requests[1].generated_token_ids", "is missing: every writer of the record writes it"),
                                      ("requests[1].request_id", "is missing: every writer of the record writes it"),
                                      ("requests", "is missing: every writer of the record writes it"),
                                      ("lag", "is a list, not a Mapping")])
@pytest.mark.parametrize("tokens_optional", [False, True])
def test_g6_fails_a_refused_record_by_name(memory, tmp_path, path, why, tokens_optional):
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    doc = _tokens(oracle)
    if path == "lag":
        doc["lag"] = [0]
    else:
        *head, last = path.replace("[1]", ".1").split(".")
        d = doc
        for k in head:
            d = d[int(k)] if k.isdigit() else d[k]
        del d[last]
    _write(oracle, doc)
    res = gm.check(gp, oracle, run_x09=False, record_alternate=False, tokens_optional=tokens_optional)
    g6 = T._check(res, "G6")
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"]
    want = (f"the driver's served tokens are refused, so the output-association legs cannot run: capture/tokens.json: field "
            f"{path!r} {why}")
    assert want in g6["problems"], g6["problems"]
    assert g6["tokens_json"] is False
    assert os.path.exists(os.path.join(oracle, "capture", "tokens.json"))


def _omitted(rid: str) -> str:
    return (f"{rid}: the Program declares it but the driver's capture/tokens.json has no entry for it: its output association (the "
            f"driver's served tokens vs the recorder's TokenSelects) and the driver's length claims cannot run, so it is refused by name")


@pytest.mark.parametrize("requests,shortfall", [((T.R0, T.R1), None),
                                               ((T.R0, T.R1_EOS), "engine steps [2, 3]: live requests observed [1, 0] != programmed [2, 1] -- "
                                                                  "the shortfall is not after a verified EOS (['r1'] stopped early without one): "
                                                                  "declared schedule not exercised")],
                         ids=["served to the cap", "eos stop"])
@pytest.mark.parametrize("tokens_optional", [False, True])
def test_g6_refuses_a_declared_request_the_driver_omits(memory, tmp_path, requests, shortfall, tokens_optional):
    """A passing record with one request dropped from its tokens.json FAILS G6 naming that request, never a PASS with the request's
    output association skipped.  An EOS stop is then not verified, so the steps it left empty are named as a shortfall too."""
    gp, oracle = T.assemble(list(requests), tmp_path)
    assert _outcome(gp, oracle)["verdict"] == "PASS"
    doc = _tokens(oracle)
    _write(oracle, {**doc, "requests": [_req(doc, "r0")]})
    got = _outcome(gp, oracle, tokens_optional)
    assert got["verdict"] == "FAIL" and got["failed"] == ["G6"]
    assert got["G6"]["problems"] == [_omitted("r1")] + ([shortfall] if shortfall else []), got["G6"]["problems"]
    by = {e["request_id"]: e for e in got["requests"]}
    assert by["r1"]["g6_problems"] == 1 and by["r0"]["g6_problems"] == 0
    assert by["r1"].get("eos_verified") is (False if shortfall else None)


def test_g6_refuses_a_request_the_program_does_not_declare(memory, tmp_path):
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    doc = _tokens(oracle)
    _write(oracle, {**doc, "requests": doc["requests"] + [dict(_req(doc, "r1"), request_id="r9")]})
    got = _outcome(gp, oracle)
    assert got["verdict"] == "FAIL" and got["failed"] == ["G6"]
    assert got["G6"]["problems"] == ["the driver's capture/tokens.json carries request(s) ['r9'] the Program does not declare: served "
                                     "tokens of no declared request are not an execution of the Program"]


def test_g6_refuses_a_request_the_driver_names_twice(memory, tmp_path):
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    doc = _tokens(oracle)
    _write(oracle, {**doc, "requests": doc["requests"] + [copy.deepcopy(_req(doc, "r1"))]})
    got = _outcome(gp, oracle)
    assert got["verdict"] == "FAIL" and got["failed"] == ["G6"]
    assert got["G6"]["problems"] == ["the driver's capture/tokens.json has more than one entry for request(s) ['r1']: which one the driver "
                                     "served is not stated"]
