"""vLLM serving under PoUB (`serve.py --local`): the vLLM tests' tiny Llama on a GPU, `Binder` on its engine
and `verity_vllm.protocols.remote_bandwidth.remote.StoreBackend` proving its store to an in-process verifier while it serves a trace of
requests, one session a process.  Skipped without vLLM, a GPU, or a `StoreBackend` to import (the vLLM integration's
remote session)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "serve.py"
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parents[2] / "integrations" / "vllm")]


def remote_or_skip():
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("no vLLM")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    remote = pytest.importorskip("verity_vllm.protocols.remote_bandwidth.remote")
    if not hasattr(remote, "StoreBackend"):
        pytest.skip("no StoreBackend in verity_vllm.protocols.remote_bandwidth.remote")


def serve(out: Path, plan: str) -> dict:
    r = subprocess.run([sys.executable, str(SERVE), "--local", "--plan", plan, "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return json.loads((out / "serve.json").read_text())


@pytest.mark.slow
def test_a_real_engines_serving_is_accepted_and_a_prover_that_drops_a_kv_block_is_rejected(tmp_path):
    """The one test that runs a real engine's steps under the audit (every other session test scripts its serving):
    the engine's allocs, KV writes and frees reach the verifier as appends, releases and plans, a pass a step.  Honest
    serving is accepted, with the memory line's weights the layout's and its KV read the record's post-seed reads; a
    prover that withholds the first block appended is rejected at a receipt on it (every receipt is drawn)."""
    remote_or_skip()
    d = serve(tmp_path / "serve", "serve")
    v, p = d["verifier"], d["prover"]
    assert v["accepted"] and p["accepted"] and v["failure"] is None
    assert d["trace"]["steps"] >= 12 and sum(r["finished"] is not None for r in d["trace"]["requests"]) >= 2
    pub = v["record"]["public"]
    assert pub["appended"] > 0 and pub["released"] > 0 and sum(pub["post_seed"]) > 0
    B = pub["block_bytes"]
    weights = sum(r["blocks"] for r in d["layout"] if r["kind"] == "weights") * B
    assert weights == d["store"]["bytes_by_kind"]["weights"] > 0
    mem = v["ledger"]["memory"]
    assert mem["labels"] and not mem["kv_credited"] and mem["read"]["kv"] == sum(pub["post_seed"]) * B
    assert all(w["held"]["weights"] == weights for w in mem["windows"])

    d = serve(tmp_path / "drop", "servedrop:1")
    v, p = d["verifier"], d["prover"]
    assert not v["accepted"] and not p["accepted"]
    assert v["failure"]["failure"] == "receipt" and p["failure_on_dropped"] is True
