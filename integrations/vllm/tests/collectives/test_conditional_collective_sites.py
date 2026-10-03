"""A mid-module collective call the INSTANCE can never execute is no expected site (`collective_sites.mid_module_site_census`).

vLLM's `Qwen3MoeSparseMoeBlock.forward` all-gathers the experts' output only `if self.is_sequence_parallel:` (sequence-parallel MoE, off
by default).  Counted statically, that call made every `model.layers.<L>.mlp` an expected TP site no step executes, and the TP8 Commit's
identity_check failed `missing_modules model.layers.*.mlp`.  The rule skips a call only under a bare `if self.<attr>:` whose attribute is
on the live instance and is False / None / 0; every other shape keeps the call (fail closed), and a skipped call is listed and its module
stays patched, so a call that runs anyway is named.  Torch-free (fake tensors / modules / committer)."""
from __future__ import annotations

import ast
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("verity_vllm.acquire.sources.partial_source")
from verity_vllm.acquire.sources import partial_source as tps  # noqa: E402
from tests.acquire.test_tp_partial_source_moe import _Committer, _Mod, _T, _make, _model  # noqa: E402
from verity_vllm.collectives import collective_sites as CS  # noqa: E402

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


class CompoundGuard(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel, self.gather_out = 2, False, False

    def forward(self, x):
        if self.is_sequence_parallel and self.gather_out:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class NegatedGuard(_Mod):
    def __init__(self, flag):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, flag

    def forward(self, x):
        if not self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class ElseBranch(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def forward(self, x):
        if self.is_sequence_parallel:
            return x
        else:
            return tensor_model_parallel_all_reduce(x)


class PartlyGuarded(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def forward(self, x):
        x = tensor_model_parallel_all_reduce(x)
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class RebindsGuard(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def enable(self):
        self.is_sequence_parallel = True

    def forward(self, x):
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class ReflectiveWrite(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size = 2
        setattr(self, "is_sequence_parallel", False)

    def forward(self, x):
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class NestedScope(_Mod):
    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def forward(self, x):
        if self.is_sequence_parallel:
            def gather(t):
                return tensor_model_parallel_all_gather(t, 0)
            return gather(x)
        return x


class MentionsTheCall(_Mod):
    """Gathers with tensor_model_parallel_all_gather(x, 0) under sequence parallelism (the regex counts this line, the parse does not)."""

    def __init__(self):
        super().__init__()
        self.tp_size, self.is_sequence_parallel = 2, False

    def forward(self, x):
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


class ClassLevelGuard(_Mod):
    is_sequence_parallel = False

    def __init__(self):
        super().__init__()
        self.tp_size = 2

    def forward(self, x):
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
        return x


def _census(mods):
    return CS.mid_module_site_census(_model(mods), 2, model_prefix=THIS)


def _site_calls(census):
    return {s["name"]: s["calls"] for s in census["sites"]}


def test_qwen3_block_without_sequence_parallel_is_no_site_and_the_skip_is_listed():
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(False)})
    assert c["sites"] == []
    lines, first = inspect.getsourcelines(Qwen3MoeSparseMoeBlockLike)
    line = first + next(i for i, ln in enumerate(lines) if "tensor_model_parallel_all_gather(" in ln)
    assert c["skipped"] == [{"name": MLP, "cls": "Qwen3MoeSparseMoeBlockLike", "py_module": THIS, "kind": "all_gather",
                             "attr": "is_sequence_parallel", "value": "False", "line": line, "site_dropped": True}]
    assert CS.mid_module_sites(_model({MLP: Qwen3MoeSparseMoeBlockLike(False)}), 2, model_prefix=THIS) == []
    # the skipped call's module is still a patch target: if the gather ran anyway, the observers would see it
    assert (THIS, "tensor_model_parallel_all_gather") in [(m.__name__, a) for m, a, _k in CS.patch_targets(c["sites"] + c["skipped"])]


def test_qwen3_block_with_sequence_parallel_is_a_site():
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(True)})
    assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["skipped"] == []
    assert CS.mid_module_site_census(_model({MLP: Qwen3MoeSparseMoeBlockLike(False)}), 1, model_prefix=THIS) == {"sites": [], "skipped": []}


@pytest.mark.parametrize("value", [None, 0])
def test_none_and_zero_are_falsy_guards(value):
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(value)})
    assert c["sites"] == [] and [(s["attr"], s["value"]) for s in c["skipped"]] == [("is_sequence_parallel", repr(value))]


@pytest.mark.parametrize("value", [0.0, "", [], 1, "yes"])
def test_any_other_value_keeps_the_site(value):
    c = _census({MLP: Qwen3MoeSparseMoeBlockLike(value)})
    assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["skipped"] == []


def test_missing_attribute_keeps_the_site():
    m = Qwen3MoeSparseMoeBlockLike(False)
    del m.is_sequence_parallel
    assert _site_calls(_census({MLP: m})) == {MLP: {"all_gather": 1}}
    assert _site_calls(_census({MLP: ClassLevelGuard()})) == {MLP: {"all_gather": 1}}       # on the class, not the instance: kept


def test_compound_negated_and_else_conditions_keep_the_site():
    assert _site_calls(_census({MLP: CompoundGuard()})) == {MLP: {"all_gather": 1}}
    assert _site_calls(_census({MLP: NegatedGuard(False)})) == {MLP: {"all_gather": 1}}
    assert _site_calls(_census({MLP: NegatedGuard(True)})) == {MLP: {"all_gather": 1}}
    assert _site_calls(_census({MLP: ElseBranch()})) == {MLP: {"all_reduce": 1}}


def test_unreadable_guards_keep_the_site():
    """A guard written outside `__init__`, a reflective write, a call in a nested scope, and a class whose parsed calls are not its static
    count: the instance's value proves nothing about the call, so the call stays expected."""
    for cls in (RebindsGuard, ReflectiveWrite, NestedScope):
        c = _census({MLP: cls()})
        assert _site_calls(c) == {MLP: {"all_gather": 1}} and c["skipped"] == [], cls.__name__
    c = _census({MLP: MentionsTheCall()})
    assert _site_calls(c) == {MLP: {"all_gather": 2}} and c["skipped"] == []                 # the static count, as before the rule


def test_a_site_keeps_the_calls_its_instance_executes():
    c = _census({MLP: PartlyGuarded()})
    assert c["sites"] == [{"name": MLP, "cls": "PartlyGuarded", "py_module": THIS, "calls": {"all_reduce": 1}, "per_step": 1}]
    assert [(s["kind"], s["attr"], s["site_dropped"]) for s in c["skipped"]] == [("all_gather", "is_sequence_parallel", False)]


def test_commit_source_expects_no_skipped_site_lists_it_and_still_names_a_call_that_runs(monkeypatch):
    monkeypatch.setattr(CS, "MODEL_MODULE_PREFIX", THIS)
    c = _Committer()
    mods = {MLP: Qwen3MoeSparseMoeBlockLike(False), MLP + ".experts": _make("MoERunner")}
    src = tps.TPPartialSource(c, _model(mods), rank=0, world=2)
    assert src.site_names() == [MLP + ".experts"] and src.received_site_names() == [MLP + ".experts"]
    assert src.site_counts() == {MLP + ".experts": 1}
    assert [(s["name"], s["attr"], s["value"]) for s in src.mid_sites_skipped] == [(MLP, "is_sequence_parallel", "False")]
    assert src.declaration_classes()[0]["mid_module_sites_skipped"] == src.mid_sites_skipped
    nh = pytest.importorskip("verity_vllm.commit.committer.native_host")
    host = types.SimpleNamespace(_modules=[], runner=None, sources=[src])
    man = nh.NativeHostCommitter.expected_manifest(host)
    assert MLP not in man["modules"] and man["tp_sites_skipped"] == src.mid_sites_skipped
    # the module's own collective name stays wrapped: a gather issued outside every hooked module is an unattributed, named error
    src.install()
    try:
        assert any(p.owner.__name__ == THIS and p.name == "tensor_model_parallel_all_gather" for p in src._hooks.entries)
        sys.modules[THIS].tensor_model_parallel_all_gather(_T(8), 0)
    finally:
        src.uninstall()
    assert src.stats["unattributed"] == 1 and any("no site attribution" in e for e in c.errors)


def test_match_verdict_lists_the_recorders_skipped_sites(tmp_path):
    """`tp2_match.json` carries the list the recorder wrote to each rank's `coll_meta`, once when the ranks agree, per rank when not."""
    match = pytest.importorskip("verity_vllm.pipeline.tp.match")
    skip = _census({MLP: Qwen3MoeSparseMoeBlockLike(False)})["skipped"]
    for r in (0, 1):
        (tmp_path / f"coll_meta.rank{r}.json").write_text(json.dumps({"mid_module_sites_skipped": skip}))
    assert match._skipped_sites(tmp_path, [0, 1]) == {"n": 1, "same_on_all_ranks": True, "sites": skip}
    (tmp_path / "coll_meta.rank1.json").write_text(json.dumps({"mid_module_sites_skipped": []}))
    got = match._skipped_sites(tmp_path, [0, 1])
    assert got["same_on_all_ranks"] is False and got["per_rank"] == {"0": skip, "1": []}
    assert match._skipped_sites(tmp_path / "none", [0, 1]) is None


def _calls_in(path: Path, fn: str, cls: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text())
    if cls is not None:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    tree = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
    return {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}


def test_match_recorder_and_commit_source_read_the_one_census():
    """Both observers derive their expected mid-module sites from `mid_module_site_census`, so the Match and the Commit cannot disagree on
    which calls are expected."""
    pkg = Path(tps.__file__).resolve().parents[2]
    recorder = _calls_in(pkg / "engine" / "rank_worker.py", "tp2_coll_install")
    commit = _calls_in(Path(tps.__file__), "__init__", "TPPartialSource")
    for calls in (recorder, commit):
        assert "mid_module_site_census" in calls and "mid_module_sites" not in calls and "class_collective_calls" not in calls
