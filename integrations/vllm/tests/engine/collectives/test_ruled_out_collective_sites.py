"""The mid-module collective calls an instance's flags rule out (`collective_sites.mid_module_site_census` "ruled_out") are listed
where a reviewer reads the verdicts, and stay visible to the observers.

vLLM's `Qwen3MoeSparseMoeBlock.forward` all-gathers the experts' output only `if self.is_sequence_parallel:` (sequence-parallel MoE, off
by default); the census drops that call.  Each dropped call is listed with the module path, the flag, the truth the branch wants and the
instance's value, in the Match record (`coll_meta.rank<r>.json`, `tp2_match.json`) and the Commit's identity report
(`tp_sites_ruled_out`); its python module stays patched, so a dropped call that runs anyway is an unattributed, named error.  The rule
itself is tested in `tests/acquire/test_tp_partial_source_mid_module.py`; here, the site-level cases and the listing.  Torch-free."""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("verity_vllm.acquire.sources.partial_source")
from verity_vllm.acquire.sources import partial_source as tps
from tests.acquire.test_tp_partial_source_moe import _Committer, _Mod, _T, _make, _model
from verity_vllm.engine.collectives import collective_sites as CS

THIS = __name__
MLP = "model.layers.0.mlp"


def tensor_model_parallel_all_gather(x, dim=-1):                 # the names a model module binds by `from vllm.distributed import ...`
    return x


def tensor_model_parallel_all_reduce(x):
    return x


def sequence_parallel_chunk(x):
    return x


class Qwen3MoeSparseMoeBlockLike(_Mod):
    """vLLM qwen3_moe.py's forward: the experts' output is gathered across ranks only under sequence-parallel MoE."""

    def __init__(self, is_sequence_parallel, tp_size=2):
        super().__init__()
        self.tp_size = tp_size
        self.is_sequence_parallel = is_sequence_parallel

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        final_hidden_states = self.experts(hidden_states)
        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        return final_hidden_states


class EitherFlag(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel, self.gather_out = 2, False, False

    def forward(self, x):
        if self.is_sequence_parallel or self.gather_out:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class CalledCondition(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size = 2

    def use_sequence_parallel(self):
        return False

    def forward(self, x):
        if self.use_sequence_parallel():
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class PartlyGuarded(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def forward(self, x):
        x = tensor_model_parallel_all_reduce(x)
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


def _census(mods):
    return CS.mid_module_site_census(_model(mods), 2, model_prefix=THIS)


def _site_calls(census):
    return {s["name"]: s["calls"] for s in census["sites"]}


def test_qwen3_block_without_sequence_parallel_is_no_site_and_is_listed():
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(False)})
    assert c["sites"] == []
    assert c["ruled_out"] == [{"name": MLP, "cls": "Qwen3MoeSparseMoeBlockLike", "py_module": THIS, "site_dropped": True,
                               "calls": [{"kind": "all_gather", "flag": "is_sequence_parallel", "want": True, "value": False}]}]
    assert CS.mid_module_sites(_model({MLP: Qwen3MoeSparseMoeBlockLike(False)}), 2, model_prefix=THIS) == []
    # the ruled-out module is still a patch target: if the gather ran anyway, the observers would see it
    assert (THIS, "tensor_model_parallel_all_gather") in [(m.__name__, a) for m, a, _k in CS.patch_targets(c["sites"] + c["ruled_out"])]
    assert CS.mid_module_site_census(_model({MLP: Qwen3MoeSparseMoeBlockLike(False)}), 1, model_prefix=THIS) == {"sites": [], "ruled_out": []}


def test_qwen3_block_with_sequence_parallel_is_a_site():
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(True)})
    assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["ruled_out"] == []


def test_a_missing_flag_keeps_the_site():
    m = Qwen3MoeSparseMoeBlockLike(False)
    del m.is_sequence_parallel
    c = _census({MLP: m})
    assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["ruled_out"] == []


def test_a_condition_other_than_a_flag_or_a_flag_conjunct_keeps_the_site():
    for m in (EitherFlag(), CalledCondition()):
        c = _census({MLP: m})
        assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["ruled_out"] == [], type(m).__name__


def test_a_site_keeps_the_calls_its_instance_allows():
    c = _census({MLP: PartlyGuarded()})
    assert c["sites"] == [{"name": MLP, "cls": "PartlyGuarded", "py_module": THIS, "calls": {"all_reduce": 1}, "per_step": 1}]
    assert [(r["site_dropped"], [x["kind"] for x in r["calls"]]) for r in c["ruled_out"]] == [(False, ["all_gather"])]


def test_commit_source_lists_the_ruled_out_site_and_still_names_a_call_that_runs(monkeypatch):
    monkeypatch.setattr(CS, "MODEL_MODULE_PREFIX", THIS)
    c = _Committer()
    mods = {MLP: Qwen3MoeSparseMoeBlockLike(False), MLP + ".experts": _make("MoERunner")}
    src = tps.TPPartialSource(c, _model(mods), rank=0, world=2)
    assert src.site_names() == [MLP + ".experts"] and src.site_counts() == {MLP + ".experts": 1}
    assert [(r["name"], r["calls"][0]["flag"], r["calls"][0]["value"]) for r in src.mid_sites_ruled_out] == [(MLP, "is_sequence_parallel", False)]
    assert src.declaration_classes()[0]["mid_module_sites_ruled_out"] == src.mid_sites_ruled_out
    nh = pytest.importorskip("verity_vllm.commit.committer.native_host")
    man = nh.NativeHostCommitter.expected_manifest(types.SimpleNamespace(_modules=[], runner=None, sources=[src]))
    assert MLP not in man["modules"] and man["tp_sites_ruled_out"] == src.mid_sites_ruled_out
    # the module's own collective name stays wrapped: a gather issued outside every hooked module is an unattributed, named error
    src.install()
    try:
        assert any(p.owner.__name__ == THIS and p.name == "tensor_model_parallel_all_gather" for p in src._hooks.entries)
        sys.modules[THIS].tensor_model_parallel_all_gather(_T(8), 0)
    finally:
        src.uninstall()
    assert src.stats["unattributed"] == 1 and any("no site attribution" in e for e in c.errors)


def test_match_verdict_lists_the_recorders_ruled_out_sites(tmp_path):
    """`tp2_match.json` carries the list the recorder wrote to each rank's `coll_meta`: once when the ranks agree, per rank when not."""
    match = pytest.importorskip("verity_vllm.pipeline.tp.match")
    ruled = _census({MLP: Qwen3MoeSparseMoeBlockLike(False)})["ruled_out"]
    for r in (0, 1):
        (tmp_path / f"coll_meta.rank{r}.json").write_text(json.dumps({"mid_module_sites_ruled_out": ruled}))
    assert match._ruled_out_sites(tmp_path, [0, 1]) == {"n": 1, "same_on_all_ranks": True, "sites": ruled}
    (tmp_path / "coll_meta.rank1.json").write_text(json.dumps({"mid_module_sites_ruled_out": []}))
    got = match._ruled_out_sites(tmp_path, [0, 1])
    assert got["same_on_all_ranks"] is False and got["per_rank"] == {"0": ruled, "1": []}
    assert match._ruled_out_sites(tmp_path / "none", [0, 1]) is None


def _calls_in(path: Path, fn: str, cls: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text())
    if cls is not None:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    tree = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
    return {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}


def test_match_recorder_and_commit_source_read_the_one_census():
    """Both observers take their expected mid-module sites from `mid_module_site_census`, so the Match and the Commit cannot disagree on
    which calls are expected."""
    pkg = Path(tps.__file__).resolve().parents[2]
    for calls in (_calls_in(pkg / "pipeline" / "tp" / "rank_worker.py", "tp2_coll_install"), _calls_in(Path(tps.__file__), "__init__", "TPPartialSource")):
        assert "mid_module_site_census" in calls and "mid_module_sites" not in calls and "class_collective_calls" not in calls
