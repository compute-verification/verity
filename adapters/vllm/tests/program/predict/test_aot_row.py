"""`aot.write_row` (``verity-vllm predict row``) end to end on TINY: each request's named Build, the workload Program, the row's
manifest under the row's own policies (`manifest.build` for one request, `manifest.build_global` for more) and `predicted.json`
naming that manifest; then `aot.compare_row`, which holds the row to a traced Build of it. CPU only, seconds."""
from __future__ import annotations

import hashlib
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


@pytest.mark.parametrize("n", [1, 2])
def test_predict_row_writes_the_rows_manifest_by_the_query_path(tmp_path, monkeypatch, n):
    """One request takes `manifest.build` over its Build, two take `manifest.build_global` over the row: each accepts the
    policies `write_row` passes, and the manifest is complete, with the word check run and its sha256 in `predicted.json`."""
    _tiny(tmp_path, monkeypatch, n)
    rec = A.write_row("tiny", str(out := tmp_path / "row"), config="config.json")
    raw = (out / "manifest.json").read_bytes()
    man = json.loads(raw)
    assert man["complete"] is True and "word_check" in man["query"]
    assert rec.manifest_sha256 == hashlib.sha256(raw).hexdigest() and rec.manifest_digest == man["manifest_digest"]
    assert len(rec.requests) == n and all((out / name / "instances.json.gz").exists() for name in rec.requests)
    assert man["program_digest"] == (next(iter(rec.requests.values())) if n == 1 else rec.workload_digest)
    assert ("global" in man) is (n > 1)
    assert PredictedRow.of(json.loads((out / FILE).read_text()), str(out / FILE)).manifest_digest == man["manifest_digest"]


def test_the_row_command_prints_six_fields_of_its_record_and_says_on_stderr_what_it_wrote(tmp_path, monkeypatch, capsys):
    """`predict row`'s stdout is one JSON line, six fields of the record it wrote beside the Build (a caller keeps the line as a
    file); what it wrote on the way is on stderr, the request's Build first and the manifest's lines after it."""
    _tiny(tmp_path, monkeypatch, 1)
    assert A.main(A.RowOptions(row="tiny", out=str(out := tmp_path / "row"), config="config.json", jobs=1)) == 0
    rec = PredictedRow.of(json.loads((out / FILE).read_text()), str(out / FILE))
    got = capsys.readouterr()
    assert json.loads(got.out) == {"row": "tiny", "workload_digest": rec.workload_digest, "manifest_digest": rec.manifest_digest,
                                   "program_digest": rec.program_digest, "identities": rec.identities, "secs": rec.secs}
    ((name, digest),) = rec.requests.items()
    said = got.err.splitlines()
    first = next(i for i, line in enumerate(said) if line.startswith(f"[predict] {name}: {digest[:16]}… ("))
    assert said[first + 1].startswith("  ")


def test_a_predicted_row_is_held_to_the_traced_build_of_the_same_row(tmp_path, monkeypatch):
    """`compare_row` reads a single-GPU row's traced Build as its dir holds it (the digests in `build_summary.json`, the request's
    shape in its derive's `result.json`, the manifest beside them) and holds the predicted row's request Program, workload and
    manifest to it: equal only when each part is, and refused where the traced Build has more than one rank."""
    _tiny(tmp_path, monkeypatch, 1)
    rec = A.write_row("tiny", str(pred := tmp_path / "row"), config="config.json")
    ((name, digest),) = rec.requests.items()
    lp, t = A._request_shape(name, "the predicted row")
    traced = tmp_path / "traced"
    (traced / "build_step").mkdir(parents=True)
    (traced / "build_request").mkdir()
    (traced / "build_request" / "result.json").write_text(json.dumps({"LP": lp, "T": t}))
    (traced / "manifest.json").write_bytes((pred / "manifest.json").read_bytes())
    summary = {"build_step": {"program_digest": "s"}, "request": {"program_digest": digest}, "workload": {"workload_digest": rec.workload_digest}}
    (traced / "build_summary.json").write_text(json.dumps(summary))
    facts = {"manifest_digest": rec.manifest_digest, "program_digest": rec.program_digest, "identities": rec.identities}
    assert A.compare_row(str(traced), str(pred)) == {"dir": str(pred.resolve()), "equal": True, "parts": {
        "workload": {"traced": rec.workload_digest, "predicted": rec.workload_digest, "equal": True},
        name: {"traced": digest, "predicted": digest, "equal": True},
        "manifest": {"traced": facts, "predicted": facts, "equal": True}}}
    other = "0" * len(digest)
    (traced / "build_summary.json").write_text(json.dumps({**summary, "request": {"program_digest": other}}))
    got = A.compare_row(str(traced), str(pred))
    assert got["equal"] is False and {n: v["equal"] for n, v in got["parts"].items()} == {"workload": True, name: False, "manifest": True}
    assert got["parts"][name] == {"traced": other, "predicted": digest, "equal": False}
    (traced / "build_summary.json").write_text(json.dumps({"ranks": {"rank0": summary, "rank1": summary}}))
    with pytest.raises(UsageError, match="a predicted row is one rank's, and the traced Build has ranks"):
        A.compare_row(str(traced), str(pred))
