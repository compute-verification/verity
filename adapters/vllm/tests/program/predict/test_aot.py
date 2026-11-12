"""verity_vllm.program.predict.aot: a switched row's derive dir is the predicted Program under its trace's annotations, byte for byte, its
instance sequence under the trace's Call and weight names, and refused where the trace is of another Program; the comparison names
each Program equal or not."""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from verity.circuits.types import RecordType

from tests.program.predict.test_predict import TINY, _phi
from verity_vllm.config import UsageError
from verity_vllm.correspondence.runtime import SCHEMA as CORRESPONDENCE
from verity_vllm.correspondence.runtime import correspondence_digest
from verity_vllm.program.predict import aot as A
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.build import build_program

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

AOT = {"comparison": "r20261018-001500-aaaa", "trace": "art:" + "5" * 64}


def _weights(program) -> tuple[str, RecordType]:
    return next((n, t) for n, t in program.fn.body.params if isinstance(t, RecordType))


def _trace(d: Path) -> Path:
    """A traced derive dir of TINY's step: its descriptor under a runtime correspondence and naming hints, result.json naming its
    digests, its weight leaves under the trace's own names (`inner.` + the predictor's), and the sidecars a switched row's carries."""
    r = build_program(_phi(TINY), "step", M=1, pos0=8)
    desc = {**r.desc, "annotations": {"correspondence": {"schema": CORRESPONDENCE, "calls": [{"occ": {"fx": "mm_0", "module": "lm_head"}}]},
                                      "node_hints": {str(k): f"m{k}/fx_{k}" for k in range(len(r.program.fn.body.nodes))}}}
    raw = json.dumps(desc, sort_keys=True).encode()
    d.mkdir(parents=True)
    (d / "descriptor.json.gz").write_bytes(gzip.compress(raw))
    (d / "result.json").write_text(json.dumps({"ok": True, "M": 1, "pos0": 8, "program": {
        "digest": r.digest, "descriptor_sha256": hashlib.sha256(raw).hexdigest(), "correspondence_digest": correspondence_digest(desc)}}))
    for name in A.CARRIED[1:]:
        (d / name).write_text(json.dumps({"name": name}))
    leaves = [{"name": f"inner.{n}", "shape": [t.leaves], "field": i} for i, (n, t) in enumerate(_weights(r.program)[1].fields)]
    (d / "derive-report.json").write_text(json.dumps({"provenance": {"weights": leaves[::-1]}}))
    return d


def test_emit_writes_the_predicted_program_under_the_traces_annotations_and_names(tmp_path):
    trace, out = _trace(tmp_path / "trace"), tmp_path / "out"
    block = A.emit(_phi(TINY), "step", str(trace), str(out), aot=AOT)
    assert gzip.decompress((out / "descriptor.json.gz").read_bytes()) == gzip.decompress((trace / "descriptor.json.gz").read_bytes())
    res = json.loads((trace / "result.json").read_text())
    assert block["predicted_digest"] == res["program"]["digest"] and {k: block[k] for k in AOT} == AOT
    assert json.loads((out / "result.json").read_text()) == {**res, "aot": block}
    assert json.loads((out / "construction_manifest.json").read_text()) == {"name": "construction_manifest.json", "aot": block}
    for name in ("artifact.json", "derive-report.json", "tp_links.json", "inputs.json"):
        assert (out / name).read_bytes() == (trace / name).read_bytes()
    with gzip.open(out / "instances.json.gz", "rt") as f:
        doc = json.load(f)
    assert (doc["digest"], doc["correspondence_digest"]) == (res["program"]["digest"], res["program"]["correspondence_digest"])
    assert [r["name"] for r in doc["rows"]] == [f"m{k}/fx_{k}" for k in range(len(doc["rows"]))]
    name, rec = _weights(build_program(_phi(TINY), "step", M=1, pos0=8).program)
    assert dict(doc["params"])[name] == str(RecordType(*((f"inner.{n}", t) for n, t in rec.fields)))


def test_emit_refuses_weight_leaves_that_are_not_the_programs(tmp_path):
    trace = _trace(tmp_path / "trace")
    rep = json.loads((trace / "derive-report.json").read_text())
    (trace / "derive-report.json").write_text(json.dumps({"provenance": {"weights": rep["provenance"]["weights"][1:]}}))
    with pytest.raises(UsageError, match="fields are not the trace's .* weight leaves in size"):
        A.emit(_phi(TINY), "step", str(trace), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,match", [("digest", "is not the predicted"), ("descriptor_sha256", "the predicted descriptor under the trace's")])
def test_emit_refuses_a_trace_of_another_program(tmp_path, field, match):
    trace = _trace(tmp_path / "trace")
    res = json.loads((trace / "result.json").read_text())
    (trace / "result.json").write_text(json.dumps({**res, "program": {**res["program"], field: "0" * 64}}))
    with pytest.raises(UsageError, match=match):
        A.emit(_phi(TINY), "step", str(trace), str(tmp_path / "out"))
    assert not (tmp_path / "out" / "descriptor.json.gz").exists()


def test_emit_cannot_say_the_descriptor_it_predicted_under_annotations_that_hold_no_correspondence(tmp_path):
    """A traced descriptor whose annotations hold no runtime correspondence, under a `result.json` that names another descriptor:
    the refusal's message cuts the predicted correspondence's digest, which is None, so it is a TypeError that names nothing."""
    trace = _trace(tmp_path / "trace")
    with gzip.open(trace / "descriptor.json.gz", "rt") as f:
        desc = json.load(f)
    del desc["annotations"]["correspondence"]
    (trace / "descriptor.json.gz").write_bytes(gzip.compress(json.dumps(desc, sort_keys=True).encode()))
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        A.emit(_phi(TINY), "step", str(trace), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("shape", [{}, {"LP": 4}, {"T": 2}])
def test_emit_of_a_request_asked_for_without_its_shape_is_ints_typeerror(tmp_path, shape):
    """`predict emit --wrapper request` without `--LP` or `--T` asks for a request's Program of no shape, which is `int`'s
    TypeError on None before anything is written."""
    trace = _trace(tmp_path / "trace")
    with pytest.raises(TypeError, match=r"^int\(\) argument must be a string, a bytes-like object or a real number, not 'NoneType'$"):
        A.emit(_phi(TINY), "request", str(trace), str(tmp_path / "out"), **shape)
    assert not (tmp_path / "out").exists()


def test_emit_refuses_a_trace_whose_result_names_no_program(tmp_path):
    """A traced derive dir whose `result.json` has no `program` block names no digest, so it is a trace of another Program."""
    trace = _trace(tmp_path / "trace")
    res = json.loads((trace / "result.json").read_text())
    (trace / "result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "program"}))
    with pytest.raises(UsageError, match="the trace's Program None… is not the predicted"):
        A.emit(_phi(TINY), "step", str(trace), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_emit_takes_a_traced_descriptor_without_annotations_as_one_that_names_nothing(tmp_path):
    """A traced descriptor without `annotations` gives the predicted Program under empty ones, the descriptor its `result.json`
    must then name, and no row of the instance sequence has a name."""
    trace, out = _trace(tmp_path / "trace"), tmp_path / "out"
    r = build_program(_phi(TINY), "step", M=1, pos0=8)
    bare = {k: v for k, v in r.desc.items() if k != "annotations"}
    (trace / "descriptor.json.gz").write_bytes(gzip.compress(json.dumps(bare, sort_keys=True).encode()))
    desc = {**bare, "annotations": {}}
    raw = json.dumps(desc, sort_keys=True).encode()
    res = json.loads((trace / "result.json").read_text())
    named = {"descriptor_sha256": hashlib.sha256(raw).hexdigest(), "correspondence_digest": correspondence_digest(desc)}
    (trace / "result.json").write_text(json.dumps({**res, "program": {**res["program"], **named}}))
    A.emit(_phi(TINY), "step", str(trace), str(out))
    assert gzip.decompress((out / "descriptor.json.gz").read_bytes()) == raw
    with gzip.open(out / "instances.json.gz", "rt") as f:
        assert {row["name"] for row in json.load(f)["rows"]} == {None}


def test_a_descriptor_without_node_hints_names_no_call(tmp_path):
    """`traced_names` on a descriptor without `annotations`, or whose annotations hold no `node_hints`: no Call has a hint, and
    the weights record's fields are still the trace's."""
    trace = _trace(tmp_path / "trace")
    program = build_program(_phi(TINY), "step", M=1, pos0=8).program
    name, rec = _weights(program)
    for desc in ({}, {"annotations": {"correspondence": {"schema": CORRESPONDENCE}}}):
        hints, params = A.traced_names(program, desc, str(trace))
        assert hints == {} and dict(params)[name] == str(RecordType(*((f"inner.{n}", t) for n, t in rec.fields)))


def test_the_emit_command_prints_the_dir_its_digest_and_its_timings(tmp_path, monkeypatch, capsys):
    """`predict emit`'s stdout is one JSON line: the derive dir it wrote, the predicted Program's digest and `emit`'s timings."""
    trace, out = _trace(tmp_path / "trace"), tmp_path / "out"
    monkeypatch.setattr(A, "phi_at", lambda row, rank, config: _phi(TINY))
    assert A.main(A.EmitOptions(row="row", config="config.json", trace=str(trace), out=str(out), wrapper="step", aot=json.dumps(AOT))) == 0
    block = json.loads((out / "result.json").read_text())["aot"]
    assert json.loads(capsys.readouterr().out) == {"out": str(out), "digest": block["predicted_digest"], "timings": block["timings"]}
    assert block["predicted_digest"] == json.loads((trace / "result.json").read_text())["program"]["digest"]


@pytest.mark.slow
def test_compare_names_each_program_and_is_equal_only_when_every_one_is(tmp_path, monkeypatch):
    d = tmp_path / "row"
    for r in range(2):
        (d / "build" / f"rank{r}" / "build_step").mkdir(parents=True)
        (d / "build" / f"rank{r}" / "build_request_LP9_T2").mkdir()
        (d / "build" / f"rank{r}" / "build_request_LP9_T2" / "result.json").write_text(json.dumps({"LP": 9, "T": 2}))
    (d / "build_summary.json").write_text(json.dumps({"tp": {"world": 2}, "ranks": {
        f"rank{r}": {"build_step": {"program_digest": f"s{r}"}, "build_request_LP9_T2": {"program_digest": f"q{r}"},
                     "workload": {"workload_digest": f"w{r}"}} for r in range(2)}}))
    got: dict[int, dict] = {0: {"step": "s0", "requests": {"build_request_LP9_T2": "q0"}, "workload": "w0", "errors": {}, "secs": {"step": 1.0}},
                            1: {"step": "s1", "requests": {}, "workload": None, "errors": {"build_request_LP9_T2": "Unexpressible: x"}, "secs": {}}}
    monkeypatch.setattr(A, "ProcessPoolExecutor", lambda max_workers, **_: concurrent.futures.ThreadPoolExecutor(max_workers))
    monkeypatch.setattr(A, "_predict_rank", lambda task: got[task[1]])
    doc = A.compare(str(d), "row", "config.json")
    assert doc["equal"] is False and doc["schema"] == A.SCHEMA
    assert doc["ranks"]["0"] == {"step": {"traced": "s0", "predicted": "s0", "equal": True},
                                 "workload": {"traced": "w0", "predicted": "w0", "equal": True},
                                 "build_request_LP9_T2": {"traced": "q0", "predicted": "q0", "equal": True}, "secs": {"step": 1.0}}
    assert [n for n, v in doc["ranks"]["1"].items() if n not in ("errors", "secs") and not v["equal"]] == ["workload", "build_request_LP9_T2"]
    assert doc["ranks"]["1"]["errors"] == {"build_request_LP9_T2": "Unexpressible: x"}
    got[1] = {**got[0], "step": "s1", "requests": {"build_request_LP9_T2": "q1"}, "workload": "w1"}
    assert A.compare(str(d), "row", "config.json")["equal"] is True


@pytest.mark.parametrize("equal", [True, False])
def test_the_compare_command_prints_a_line_a_rank_and_its_verdict_last(tmp_path, monkeypatch, capsys, equal):
    """`predict compare` writes `compare`'s comparison to `--out` and prints a line a rank, the predicted Build's line and then the
    verdict it exits by (0 equal, 1 not): on stdout, where a caller reads them by their heads (`rank`, `predicted`, `[predict]`)
    and takes the last line as the verdict."""
    part = {"traced": "s0", "predicted": "s0", "equal": True}
    doc = {"schema": A.SCHEMA, "row": "row", "equal": equal,
           "ranks": {"0": {"step": part, "workload": part, "secs": {"step": 1.0}},
                     "1": {"step": part, "workload": {**part, "equal": equal}, "errors": {} if equal else {"workload": "Unexpressible: x"}}},
           "predicted": {"dir": "pred", "parts": {"workload": part, "manifest": part}, "equal": True}}
    monkeypatch.setattr(A, "compare", lambda *_a: doc)
    out = tmp_path / "comparison.json"
    a = A.CompareOptions(row_dir="traced", row="row", config="config.json", out=str(out), predicted="pred")
    assert A.main(a) == (0 if equal else 1) and json.loads(out.read_text()) == doc
    assert capsys.readouterr().out.splitlines() == [
        "rank 0: step =  workload =", f"rank 1: step =  workload {'=' if equal else '≠'}", "predicted pred: workload =  manifest =",
        f"[predict] row: {'EQUAL on every rank' if equal else 'NOT EQUAL'} -> {out}"]


def test_a_rank_composes_its_workload_from_each_requests_component_as_predicted(tmp_path, monkeypatch):
    phi = _phi(TINY, [{"request_id": f"u{i}", "index": i, "prompt_len": lp, "max_tokens": cap, "arrive_step": i}
                      for i, (lp, cap) in enumerate([(5, 3), (7, 2), (5, 3)])])
    monkeypatch.setattr(A, "phi_at", lambda row, rank, config: phi)
    (tmp_path / "result.json").write_text(json.dumps({"M": 1, "pos0": 8}))
    names = [f"build_request_LP{lp}_T{t}" for lp, t in phi.request_shapes()]
    got = A._predict_rank(("row", 0, "config.json", str(tmp_path), [(n, *s) for n, s in zip(names, phi.request_shapes())]))
    assert got["errors"] == {} and set(got["secs"]) == {"step", "workload", *names}
    assert got["requests"] == {n: build_program(phi, "request", LP=lp, T=t).digest for n, (lp, t) in zip(names, phi.request_shapes())}
    assert got["workload"] == S.predict_workload(phi)["digest"]


def test_a_predicted_rows_workload_is_one_gm01_verifies_by_its_descriptor(tmp_path):
    """`write_workload` writes predict's workload Program as GP-01 writes its own: GM-01's `from_gprog` recomputes its digest from
    the descriptor beside it and holds the `lag` block to the Program's `lag` static, and refuses either edited."""
    from verity_vllm.check.match.declaration import from_gprog
    from verity_vllm.pipeline.global_program import request_id_of
    phi = _phi(TINY, [{"request_id": f"u{i}", "index": i, "prompt_len": lp, "max_tokens": cap, "arrive_step": i}
                      for i, (lp, cap) in enumerate([(5, 3), (7, 2)])])
    wl, p = A.write_workload(phi, {}, str(tmp_path)), tmp_path / "workload_program.json"
    g = from_gprog(str(p))
    assert g["workload_digest"] == wl.digest == S.predict_workload(phi)["digest"]
    dv = g["gprog_artifact"]["digest_verification"]
    assert dv["verified"] is True and dv["lag_static"] == wl.lag == g["lag"]["value"], dv
    assert [r["request_id"] for r in g["requests"]] == [request_id_of(0), request_id_of(1)]
    art = json.loads(p.read_text())
    p.write_text(json.dumps({**art, "lag": {**art["lag"], "value": 1 - wl.lag}}))
    with pytest.raises(ValueError, match="lag declaration is not the Program's"):
        from_gprog(str(p))
    p.write_text(json.dumps({**art, "workload_digest": "0" * 128}))
    with pytest.raises(ValueError, match="is not the Program it names"):
        from_gprog(str(p))
