"""`pouw_h1_gpu.py`'s engine flags reach the engine it builds: ``--engine-arg KEY=VALUE`` over the workload's ``LLM()``
keywords, and ``--no-batch-invariant`` as the target's ``batch_invariant``. Its ``--check-challenger``, the verifier's side
on no GPU, fails closed, and its `producer` is the module at its path."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.protocols.spot.device import remote as R

from tests.protocols import challenger_host as CH
from tests.protocols import pouw_h1_gpu as G
from verity_vllm.engine import build as B
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw.h1 import served as SV


class _Built(Exception):
    pass


@pytest.fixture
def built(monkeypatch):
    got: dict = {}

    def build_engine(path, *, case, target, max_num_seqs, engine_args):
        got.update(target=target, max_num_seqs=max_num_seqs, engine_args=engine_args)
        raise _Built

    monkeypatch.setattr(TargetProfile, "from_device", classmethod(lambda cls, device=0: cls.default()))
    monkeypatch.setattr(G.torch.cuda, "get_device_name", lambda i: "stand-in")
    monkeypatch.setattr(G, "pin", lambda a: ({"repo": "r", "revision": "v"}, "stand-in"))
    monkeypatch.setattr(G, "scratch_checkpoints", lambda c, entry, scratch: scratch)
    monkeypatch.setattr(B, "build_engine", build_engine)
    return got


@pytest.mark.parametrize("workload", ["batch", "fill"])
def test_engine_args_and_no_batch_invariant_reach_the_engine(tmp_path, built, workload):
    base = ["--out", str(tmp_path), "--at", "t", "--by", "b", "--workload", workload]
    with pytest.raises(_Built):
        G.main([*base, "--engine-arg", 'hf_overrides={"num_hidden_layers": 2}', "--engine-arg",
                "kv_cache_dtype=fp8_ds_mla", "--no-batch-invariant"])
    assert built["target"].batch_invariant is False
    args = built["engine_args"]
    assert args["hf_overrides"] == {"num_hidden_layers": 2} and args["kv_cache_dtype"] == "fp8_ds_mla"
    assert args["enable_prefix_caching"] is False
    with pytest.raises(_Built):
        G.main(base)
    assert built["target"].batch_invariant is True and "hf_overrides" not in built["engine_args"]


def test_an_engine_arg_overrides_the_workloads_keyword(tmp_path, built):
    with pytest.raises(_Built):
        G.main(["--out", str(tmp_path), "--at", "t", "--by", "b", "--engine-arg", "enable_prefix_caching=true"])
    assert built["engine_args"]["enable_prefix_caching"] is True


def test_ks_is_the_served_fill_workloads(tmp_path, built, capsys):
    base = ["--out", str(tmp_path), "--at", "t", "--by", "b", "--workload", "fill"]
    with pytest.raises(SystemExit):
        G.main([*base, "--ks", "2048"])
    assert "--ks is the fill workload's, with --served" in capsys.readouterr().err
    with pytest.raises(_Built):
        G.main([*base, "--served", "--j10r", "--ks", "8192,2048,2048"])


def test_the_period_program_under_ks_declares_the_linears_at_those_ks_alone(monkeypatch):
    """`period_program` with ``ks`` (``--ks``) declares the linears at those K's alone, named as `pouw._install_h1` names
    its run's, over J10-R's tile unit with ``--j10r``, in the digest's child as in this process: serve-openai's
    ``{"pouw": {"unit": "j10r", "ks": [2048]}}`` over Llama-3.2-1B's linears (K 2,048 and 8,192). The declaration and the
    digest's child are stand-ins that record what they were given (a real one takes minutes)."""
    class Linear(torch.nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.weight = torch.empty(n, k, dtype=torch.int8, device="meta")
            self.weight_scale = torch.empty(n, 1, device="meta")
            self.quant_method = SimpleNamespace(int8_arm="plain", observe=lambda *a: None)

    llama = torch.nn.Module()
    llama.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.qkv_proj, layer.o_proj = Linear(3072, 2048), Linear(2048, 2048)
        layer.gate_up_proj, layer.down_proj = Linear(16384, 2048), Linear(2048, 8192)
        llama.layers.append(layer)
    child, declared = [], []

    class Popen:
        returncode = 0

        def __init__(self, argv, **kw):
            child.append(json.loads(argv[-1]))

        def communicate(self):
            return json.dumps({"digest": "d" * 128, "s": 0.0, "peak_gib": 0.0}), None

    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(G.subprocess, "Popen", Popen)

    def declare(shapes, modules, rows, steps, unit=None, arch=None):
        declared.append((shapes, modules, unit))
        return "program"

    monkeypatch.setattr(SV, "declared", declare)
    program, digest, got = G.period_program(llama, 32, 4, j10r=True, arch="sm_90", ks=(2048,))
    assert (program, digest) == ("program", "d" * 128)
    shapes, modules, unit = declared[0]
    assert shapes == got["shapes"] == [(3072, 2048), (2048, 2048), (16384, 2048)] * 2
    assert [modules[w] for w in range(6)] == [f"layers.{i}.{n}" for i in (0, 1) for n in ("qkv_proj", "o_proj", "gate_up_proj")]
    assert unit is not None and child[0][0] == [list(s) for s in shapes] and child[0][4:] == [True, "sm_90"]
    G.period_program(llama, 32, 4, j10r=True, arch="sm_90")
    assert len(declared[1][0]) == 8


def test_check_challenger_fails_closed_and_its_record_says_why(tmp_path, capsys):
    """``pouw_h1_gpu.py --check-challenger CALLS DIR`` over a record that carries no coins and a ledger head with no entry
    exits 1, and its record (``pouw_h1_challenger.json``) says why at each step. Without ``--window`` the key is read from
    DIR's ``out/``: with none there nothing is read under the pin, so there is no credit, and a period is refused for the
    unread ledger. With ``--window`` the key is read from DIR itself: one under another pin is no key; the pinned one
    leaves the session untrusted (the ledger opens none for the window), a credit of none, and no such period."""
    _private, key = CH.challenger_key(3)
    pin = hashlib.sha512(key.public).hexdigest()
    calls, ledger, window, tree = (tmp_path / n for n in ("calls.json", "windows.jsonl", "window.json", "tree"))
    calls.write_text(json.dumps({"commitment": {}, "calls": []}))
    ledger.write_text("")
    window.write_text(json.dumps({"root": "00" * 64}))
    tree.mkdir()

    def check(pinned: str, *argv: str) -> dict:
        rc = G.main(["--out", str(tmp_path / "out"), "--check-challenger", str(calls), str(tree), "--key-sha512", pinned,
                     "--ledger", str(ledger), *argv])
        assert "POUW-H1-CHALLENGER FAIL rounds=0 host=None" in capsys.readouterr().out
        doc = json.loads((tmp_path / "out" / "pouw_h1_challenger.json").read_text())
        assert rc == 1 and doc["ok"] is False and doc["rounds"] == 0 and doc["pinned_public_sha512"] == pinned
        assert doc["manifest"].startswith("the window's salt can't be derived from its record")
        assert "error" in doc[R.RECORD] and "error" in doc[R.CLOSE]
        return doc

    doc = check(pin, "--period", "p")
    assert doc["reason"].startswith(f"CoinError: no challenger key at {tree / 'out' / R.KEY} (")
    assert "key" not in doc and "credit" not in doc and doc["window_file"] is None
    assert doc["period"] == {"name": "p", "reasons": ["ValueError: no key the pin names, so the auditor's ledger is unread"]}
    (tree / R.KEY).write_text(json.dumps(key.to_json()))
    doc = check("ab" * 64, "--window", str(window))
    assert doc["reason"].startswith(f"CoinError: the challenger key at {tree / R.KEY} is {pin[:16]}..., not the auditor's")
    assert "key" not in doc and "credit" not in doc and "period" not in doc
    doc = check(pin, "--window", str(window), "--period", "p")
    assert doc["key"] == {"W": key.W, "H": key.H, "public_sha512": pin} and doc["window"] == {"root": "00" * 64}
    assert doc["reason"].startswith("the challenger's session is not trusted: ")
    assert doc["credit"]["credit"] is None and doc["credit"]["manifest"] == doc["manifest"]
    assert doc["period"] == {"name": "p", "reasons": ["the auditor's ledger has no period p"]}
    (tree / "out").mkdir()
    (tree / "out" / R.KEY).write_text(json.dumps(key.to_json()))
    doc = check(pin)
    assert doc["key"]["public_sha512"] == pin and "window" not in doc and doc["credit"]["credit"] is None
    assert doc["reason"].startswith("the window's coins were held in the serving process")
    with pytest.raises(SystemExit):
        G.main(["--out", str(tmp_path / "out"), "--check-challenger", str(calls), str(tree), "--key-sha512", pin])
    assert "--check-challenger needs --key-sha512" in capsys.readouterr().err


def test_the_producer_is_the_module_at_its_path_loaded_once(tmp_path, monkeypatch):
    """`pouw_h1_gpu.producer` runs the file at `PRODUCER` as a module of its own, once a process, and registers it under
    no name: a second call hands the same module back. The file here is a stand-in for the benchmark's."""
    path = tmp_path / "producer.py"
    path.write_text("LOADS = [__name__]\n")
    monkeypatch.setattr(G, "PRODUCER", path)
    G.producer.cache_clear()
    try:
        mod = G.producer()
        assert mod.LOADS == ["served_h1_producer"] and mod.__file__ == str(path)
        mod.LOADS.append("kept")
        assert G.producer() is mod and G.producer().LOADS == ["served_h1_producer", "kept"]
        assert "served_h1_producer" not in sys.modules
    finally:
        G.producer.cache_clear()
