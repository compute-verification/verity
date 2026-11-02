"""`aot.write_row` (``verity-vllm predict row``) end to end on TINY: each request's named Build, the workload Program, the row's
manifest under the row's own policies (`manifest.build` for one request, `manifest.build_global` for more) and `predicted.json`
naming that manifest. CPU only, seconds."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity_vllm.program.predict import aot as A
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.phi import Phi
from verity_vllm.records.predicted_row import FILE, PredictedRow

from tests.program.predict.test_steps import _registered


@pytest.mark.parametrize("n", [1, 2])
def test_predict_row_writes_the_rows_manifest_by_the_query_path(tmp_path, monkeypatch, n):
    """One request takes `manifest.build` over its Build, two take `manifest.build_global` over the row: each accepts the
    policies `write_row` passes, and the manifest is complete, with the word check run and its sha256 in `predicted.json`."""
    reqs = [{"request_id": f"u{i}", "index": i, "prompt_len": 5 + 2 * i, "max_tokens": 3, "arrive_step": i} for i in range(n)]
    phi = Phi.from_json({**_registered().phi.to_json(), "sampling": {"kind": "greedy"}, "workload": {"requests": reqs}})
    (wl := tmp_path / "workload.json").write_text(json.dumps({"model": "org/m", "requests": reqs}))
    monkeypatch.setattr(A, "row_file", lambda row: row)
    monkeypatch.setattr(A, "workload_file", lambda _row: wl)
    monkeypatch.setattr(S, "phi_of", lambda *_a, **_k: phi)
    rec = A.write_row("tiny", str(out := tmp_path / "row"), config="config.json")
    raw = (out / "manifest.json").read_bytes()
    man = json.loads(raw)
    assert man["complete"] is True and "word_check" in man["query"]
    assert rec.manifest_sha256 == hashlib.sha256(raw).hexdigest() and rec.manifest_digest == man["manifest_digest"]
    assert len(rec.requests) == n and all((out / name / "instances.json.gz").exists() for name in rec.requests)
    assert man["program_digest"] == (next(iter(rec.requests.values())) if n == 1 else rec.workload_digest)
    assert ("global" in man) is (n > 1)
    assert PredictedRow.of(json.loads((out / FILE).read_text()), str(out / FILE)).manifest_digest == man["manifest_digest"]
