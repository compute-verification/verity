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

from verity.ml.circuits.types import Record
from verity_vllm.config import UsageError
from verity_vllm.correspondence.runtime import SCHEMA as CORRESPONDENCE, correspondence_digest
from verity_vllm.program.predict import aot as A
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.build import build_program

from tests.program.predict.test_predict import TINY, _phi

AOT = {"comparison": "r20261018-001500-aaaa", "trace": "art:" + "5" * 64}


def _weights(program) -> tuple[str, Record]:
    return next((n, t) for n, t in program.fn.body.params if isinstance(t, Record))


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
    assert dict(doc["params"])[name] == str(Record(*((f"inner.{n}", t) for n, t in rec.fields)))


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
    got = {0: {"step": "s0", "requests": {"build_request_LP9_T2": "q0"}, "workload": "w0", "errors": {}, "secs": {"step": 1.0}},
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
    wl, p = A.write_workload(phi, {}, {}, str(tmp_path)), tmp_path / "workload_program.json"
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
