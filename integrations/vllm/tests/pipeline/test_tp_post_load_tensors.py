"""A DeepSeek-V4 rank serves layer 0's `hc_attn_fn_broadcast` as a plain attribute its loader's post-load hook made, which its Build
registered as a Parameter.  The TP drivers (`tp-capture`, `tp-commit`) read those names from each rank's Build and the rank registers them
before its weights are read, so the weights it registers, pins and replays, and the Match capture's roots the fold binds, hold the tensor
its forward reads.  CPU, a toy module."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.observe.capture import Capture, ObserverConfig
from verity_vllm.pipeline.tp.rank_worker import register_post_load_tensors
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.check.component_programs import post_load_tensors

NAME = "model.layers.0.hc_attn_fn_broadcast"


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hc_attn_fn = torch.nn.Parameter(torch.arange(8.0).view(2, 4), requires_grad=False)

    def forward(self, x):
        return x @ self.hc_attn_fn_broadcast.T


def _model():
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])
    m.model.layers[0].hc_attn_fn_broadcast = m.model.layers[0].hc_attn_fn.view(2, 2, 2).sum(dim=1)
    return m


def test_a_rank_registers_its_builds_post_load_tensors_as_parameters(tmp_path):
    for r, made in ((0, {"applies": True, "parameters": {NAME: {"shape": [2, 2], "dtype": "float32"}}}), (1, {"applies": False})):
        d = tmp_path / f"rank{r}" / "build_request_LP4_T2"
        d.mkdir(parents=True)
        (d / "instances.json.gz").write_bytes(b"")
        (d / "result.json").write_text(json.dumps({"post_load_tensors": made}))
    names = post_load_tensors({0: str(tmp_path / "rank0"), 1: str(tmp_path / "rank1")})
    assert names == {"0": [NAME], "1": []}
    m = _model()
    served, x = m.model.layers[0].hc_attn_fn_broadcast, torch.ones(1, 2)
    y = m.model.layers[0](x)
    assert NAME not in dict(m.named_parameters())
    assert register_post_load_tensors(m, names["0"]) == [NAME] and register_post_load_tensors(m, names["0"]) == [NAME]
    p = dict(m.named_parameters())[NAME]
    assert p.data_ptr() == served.data_ptr() and torch.equal(m.model.layers[0](x), y), "the forward reads the same tensor"
    with pytest.raises(TypeError, match="holds no tensor"):
        register_post_load_tensors(_model(), ["model.layers.1.hc_attn_fn_broadcast"])


def _roots(m) -> dict[str, tuple]:
    cap = Capture(SimpleNamespace(runner=None, model=m), header=None, log_path=None, config=ObserverConfig(write_log=False))
    cap._install_roots()
    return {r.name: r.view.shape for r in cap.roots}


def test_the_match_capture_roots_the_post_load_tensors_of_each_ranks_build(tmp_path, monkeypatch):
    from tests.pipeline.test_row import TP_DENSE, _tp_row
    m = _model()
    assert NAME not in _roots(m)
    register_post_load_tensors(m, [NAME])
    assert _roots(m)[NAME] == (2, 2), "the observer roots every named parameter, the registered post-load tensor among them"
    t = _tp_row(tmp_path, TP_DENSE)
    for r in range(t.W):
        (Path(t.rank(r)) / "build_step").mkdir(parents=True)
        (Path(t.rank(r)) / "build_step" / "result.json").write_text('{"ok": true}')
    legs = []

    def run_gpu(stage, to, log, cmd, **kw):
        legs.append(cmd)
        raise Stop(99)

    monkeypatch.setattr(t, "run_gpu", run_gpu)
    with pytest.raises(Stop):
        t.match()
    cap = legs[0]
    assert [cap[i + 1] for i, a in enumerate(cap) if a == "--build-rank"] == [t.rank(r) for r in range(t.W)]
    commit = t.commit_cmd([])
    assert [commit[i + 1] for i, a in enumerate(commit) if a == "--build-rank"] == [t.rank(r) for r in range(t.W)]
