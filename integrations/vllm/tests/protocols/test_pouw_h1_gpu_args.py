"""`pouw_h1_gpu.py`'s engine flags reach the engine it builds: ``--engine-arg KEY=VALUE`` over the workload's ``LLM()``
keywords, and ``--no-batch-invariant`` as the target's ``batch_invariant``."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verity_vllm.engine import build as B
from verity_vllm.program.frontend.target_profile import TargetProfile

from tests.protocols import pouw_h1_gpu as G


class _Built(Exception):
    pass


@pytest.fixture
def built(monkeypatch):
    got = {}

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
