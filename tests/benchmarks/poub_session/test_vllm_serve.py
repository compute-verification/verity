"""vLLM serving under PoUB (`serve.py --local`): the vLLM tests' tiny Llama on a GPU, `Binder` on its engine
and PoUB's prover's `serving.StoreBackend` proving its store to an in-process verifier while it serves a trace of
requests, one session a process.  The engine's test is skipped without vLLM or a CUDA card, saying which; the
prover `serve.py` serves by is tested without either, and without the adapter."""
from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SERVE = HERE.parents[2] / "benchmarks" / "poub_session" / "serve.py"
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parents[2] / "adapters" / "vllm")]


def engine_or_skip():
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("needs the library vLLM, which is not installed")
    torch = pytest.importorskip("torch", reason="needs the library torch, which is not installed")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card, and torch sees none")


def test_the_prover_serve_proves_by_loads_without_a_card_or_the_adapter_and_its_prove_fits_the_call():
    """`serve.py`'s prover (`provers`) and the backend it builds on it, which the engine's test reaches only on a
    card: they load none of the adapter, the backend is a `StoreBackend`, and `prove` takes the arguments `serve.py`
    passes it."""
    spec = importlib.util.spec_from_file_location("poub_session_serve", SERVE)
    serve_py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve_py)
    adapter = {m for m in sys.modules if m.partition(".")[0] == "verity_vllm"}
    serving, memory = serve_py.provers()
    assert issubclass(serve_py.backend_class(serving), serving.StoreBackend) and callable(memory)
    assert {m for m in sys.modules if m.partition(".")[0] == "verity_vllm"} == adapter
    inspect.signature(serving.prove).bind(None, "127.0.0.1:0", "serve", node=b"n", codec=None, log=print)


def serve(out: Path, plan: str, code: int) -> dict:
    """`serve.py --local` under ``plan``, which must exit ``code``: 0 only for an accepted session."""
    r = subprocess.run([sys.executable, str(SERVE), "--local", "--plan", plan, "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == code, r.stdout[-4000:] + r.stderr[-4000:]
    return json.loads((out / "serve.json").read_text())


@pytest.mark.slow
def test_a_real_engines_serving_is_accepted_and_a_prover_that_drops_a_kv_block_is_rejected(tmp_path):
    """The one test that runs a real engine's steps under the audit (every other session test scripts its serving):
    the engine's allocs, KV writes and frees reach the verifier as appends, releases and plans, a pass a step.  Honest
    serving is accepted, with the memory line's weights the layout's and its KV read the record's post-seed reads; a
    prover that withholds the first block appended is rejected at a receipt on it (every receipt is sampled)."""
    engine_or_skip()
    d = serve(tmp_path / "serve", "serve", 0)
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

    d = serve(tmp_path / "drop", "servedrop:1", 6)
    v, p = d["verifier"], d["prover"]
    assert not v["accepted"] and not p["accepted"]
    assert v["failure"]["failure"] == "receipt" and p["failure_on_dropped"] is True
