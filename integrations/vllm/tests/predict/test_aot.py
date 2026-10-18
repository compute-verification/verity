"""verity_vllm.predict.aot: a switched row's derive dir is the predicted Program under its trace's annotations, byte for byte, and
refused where the trace is of another Program; the comparison names each Program equal or not."""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.config import UsageError
from verity_vllm.correspondence.runtime import SCHEMA as CORRESPONDENCE, correspondence_digest
from verity_vllm.predict import aot as A
from verity_vllm.predict.build import build_program

from tests.predict.test_predict import TINY, _phi

AOT = {"comparison": "r20261018-001500-aaaa", "trace": "art:" + "5" * 64}


def _trace(d: Path) -> Path:
    """A traced derive dir of TINY's step: its descriptor under a runtime correspondence, result.json naming its digests, and the
    sidecars a switched row's carries."""
    r = build_program(_phi(TINY), "step", M=1, pos0=8)
    desc = {**r.desc, "annotations": {"correspondence": {"schema": CORRESPONDENCE, "calls": [{"occ": {"fx": "mm_0", "module": "lm_head"}}]},
                                      "derive": {"wrapper": "step"}}}
    raw = json.dumps(desc, sort_keys=True).encode()
    d.mkdir(parents=True)
    (d / "descriptor.json.gz").write_bytes(gzip.compress(raw))
    (d / "result.json").write_text(json.dumps({"ok": True, "M": 1, "pos0": 8, "program": {
        "digest": r.digest, "descriptor_sha256": hashlib.sha256(raw).hexdigest(), "correspondence_digest": correspondence_digest(desc)}}))
    for name in A.CARRIED[1:]:
        (d / name).write_text(json.dumps({"name": name}))
    return d


def test_emit_writes_the_predicted_program_under_the_traces_annotations(tmp_path):
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
        head = json.loads(f.readline())
    assert (head["digest"], head["correspondence_digest"]) == (res["program"]["digest"], res["program"]["correspondence_digest"])


@pytest.mark.parametrize("field,match", [("digest", "is not the predicted"), ("descriptor_sha256", "the predicted descriptor under the trace's")])
def test_emit_refuses_a_trace_of_another_program(tmp_path, field, match):
    trace = _trace(tmp_path / "trace")
    res = json.loads((trace / "result.json").read_text())
    (trace / "result.json").write_text(json.dumps({**res, "program": {**res["program"], field: "0" * 64}}))
    with pytest.raises(UsageError, match=match):
        A.emit(_phi(TINY), "step", str(trace), str(tmp_path / "out"))
    assert not (tmp_path / "out" / "descriptor.json.gz").exists()


def test_compare_names_each_program_and_is_equal_only_when_every_one_is(tmp_path, monkeypatch):
    d = tmp_path / "row"
    (d / "build" / "rank0" / "build_step").mkdir(parents=True)
    (d / "build_summary.json").write_text(json.dumps({"tp": {"world": 2}, "ranks": {
        f"rank{r}": {"build_step": {"program_digest": f"s{r}"}, "build_request_LP9_T2": {"program_digest": f"q{r}"},
                     "workload": {"workload_digest": f"w{r}"}} for r in range(2)}}))
    got = {0: {"step": "s0", "requests": {"build_request_LP9_T2": "q0"}, "workload": "w0", "errors": {}, "secs": {"step": 1.0}},
           1: {"step": "s1", "requests": {}, "workload": None, "errors": {"build_request_LP9_T2": "Unexpressible: x"}, "secs": {}}}
    monkeypatch.setattr(A, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
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
