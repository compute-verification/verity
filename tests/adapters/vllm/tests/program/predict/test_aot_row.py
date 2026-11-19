"""`aot.write_row` (``verity-vllm predict row``) end to end on TINY: each request's named Build, the workload Program and
`predicted.json` naming the registered compiler and those Programs, no manifest; then `aot.compare_row`, which holds the row to a
traced Build of it. CPU only, seconds."""
from __future__ import annotations

import json

import pytest

from tests.program.predict.test_steps import _registered
from verity_vllm.config import UsageError
from verity_vllm.program.predict import aot as A
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.phi import Phi
from verity_vllm.records.predicted_row import FILE, PredictedRow

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")


def _tiny(tmp_path, monkeypatch, n: int) -> None:
    """The row `tiny` is TINY serving `n` greedy requests, its workload file under `tmp_path`."""
    reqs = [{"request_id": f"u{i}", "index": i, "prompt_len": 5 + 2 * i, "max_tokens": 3, "arrive_step": i} for i in range(n)]
    phi = Phi.from_json({**_registered().phi.to_json(), "sampling": {"kind": "greedy"}, "workload": {"requests": reqs}})
    (wl := tmp_path / "workload.json").write_text(json.dumps({"model": "org/m", "requests": reqs}))
    monkeypatch.setattr(A, "row_file", lambda row: row)
    monkeypatch.setattr(A, "workload_file", lambda _row: wl)
    monkeypatch.setattr(S, "phi_of", lambda *_a, **_k: phi)
    monkeypatch.setattr(A, "_compiler", lambda: COMPILER)


COMPILER = {"name": "verity/prove-path/capped-compiler/v0", "digest": "c" * 128}


def test_the_records_compiler_digest_moves_with_an_adapter_file_of_the_tree_it_is_written_from(tmp_path, monkeypatch):
    """`_compiler` reads the work tree: two trees that differ only in a file under ``adapters/vllm`` give two digests, so a
    record written from one is refused by the other."""
    import subprocess

    from verity_vllm.pipeline import source_identity as SI

    (f := tmp_path / "adapters" / "vllm" / "a.py").parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(SI, "TREE", tmp_path)
    before = A._compiler()
    f.write_text("x = 2\n")
    assert A._compiler()["name"] == before["name"] and A._compiler()["digest"] != before["digest"]


@pytest.mark.parametrize("n", [1, 2])
def test_predict_row_binds_the_compiler_and_its_programs_and_writes_no_manifest(tmp_path, monkeypatch, n):
    """`predicted.json` names the registered compiler and each request Program's and the workload's digest, which the Build dirs
    beside it hold; no manifest is written."""
    _tiny(tmp_path, monkeypatch, n)
    rec = A.write_row("tiny", str(out := tmp_path / "row"), config="config.json")
    assert rec.compiler == COMPILER and not (out / "manifest.json").exists()
    assert len(rec.requests) == n and all((out / name / "instances.json.gz").exists() for name in rec.requests)
    assert json.loads((out / "build_workload" / "workload_program.json").read_text())["built_by"] == A.ROW_BUILT_BY
    assert PredictedRow.of(json.loads((out / FILE).read_text()), str(out / FILE)) == rec


def test_the_row_command_prints_four_fields_of_its_record_and_says_on_stderr_what_it_wrote(tmp_path, monkeypatch, capsys):
    """`predict row`'s stdout is one JSON line, four fields of the record it wrote beside the Build (a caller keeps the line as a
    file); what it wrote on the way is on stderr, a line a request Build."""
    _tiny(tmp_path, monkeypatch, 1)
    assert A.main(A.RowOptions(row="tiny", out=str(out := tmp_path / "row"), config="config.json")) == 0
    rec = PredictedRow.of(json.loads((out / FILE).read_text()), str(out / FILE))
    got = capsys.readouterr()
    assert json.loads(got.out) == {"row": "tiny", "workload_digest": rec.workload_digest, "compiler": COMPILER, "secs": rec.secs}
    ((name, digest),) = rec.requests.items()
    assert any(line.startswith(f"[predict] {name}: {digest[:16]}… (") for line in got.err.splitlines())


def test_a_predicted_row_is_held_to_the_traced_build_of_the_same_row(tmp_path, monkeypatch):
    """`compare_row` reads a single-GPU row's traced Build as its dir holds it (the digests in `build_summary.json`, the request's
    shape in its derive's `result.json`) and holds the predicted row's request Program and workload to it: equal only when each part is, and refused where the traced Build has more than one rank.  A request the record
    names otherwise than `write_row` does has no shape to look its traced Program up by: None's AttributeError."""
    _tiny(tmp_path, monkeypatch, 1)
    rec = A.write_row("tiny", str(pred := tmp_path / "row"), config="config.json")
    ((name, digest),) = rec.requests.items()
    lp, t = A._request_shape(name)
    traced = tmp_path / "traced"
    (traced / "build_step").mkdir(parents=True)
    (traced / "build_request").mkdir()
    (traced / "build_request" / "result.json").write_text(json.dumps({"LP": lp, "T": t}))
    summary = {"build_step": {"program_digest": "s"}, "request": {"program_digest": digest}, "workload": {"workload_digest": rec.workload_digest}}
    (traced / "build_summary.json").write_text(json.dumps(summary))
    assert A.compare_row(str(traced), str(pred)) == {"dir": str(pred.resolve()), "equal": True, "parts": {
        "workload": {"traced": rec.workload_digest, "predicted": rec.workload_digest, "equal": True},
        name: {"traced": digest, "predicted": digest, "equal": True}}}
    other = "0" * len(digest)
    (traced / "build_summary.json").write_text(json.dumps({**summary, "request": {"program_digest": other}}))
    got = A.compare_row(str(traced), str(pred))
    assert got["equal"] is False and {n: v["equal"] for n, v in got["parts"].items()} == {"workload": True, name: False}
    assert got["parts"][name] == {"traced": other, "predicted": digest, "equal": False}
    doc = json.loads((pred / FILE).read_text())
    (pred / FILE).write_text(json.dumps({**doc, "requests": {"build_request": digest}}))
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'groups'"):
        A.compare_row(str(traced), str(pred))
    (pred / FILE).write_text(json.dumps(doc))
    (traced / "build_summary.json").write_text(json.dumps({"ranks": {"rank0": summary, "rank1": summary}}))
    with pytest.raises(UsageError, match="a predicted row is one rank's, and the traced Build has ranks"):
        A.compare_row(str(traced), str(pred))
