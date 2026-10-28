"""The vocabulary-range source (`acquire.sources.vocab_range_source`) on CPU tensors with a recording committer: its values are the circuit's
arithmetic on int32 and int64 ids, it commits `<site>/vocab_range` only inside the manifest site during a committed step, refuses a launch of
another range or a padded shard, and `attach` refuses a manifest without the family."""
from __future__ import annotations

import types

import pytest

from tests.query.test_vocab_range import VS, V, _rank_manifest
from verity_vllm.query import vocab_range as VR

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import vocab_range_source as S  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self._cur = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, object]] = []
        self.sources: list = []
        self.name = "rec"
        self.model = None

    def _acquire(self, name, t, job):
        self.acquired.append((name, t.clone()))

    def attach_source(self, src):
        self.sources.append(src)


def _model():
    m = torch.nn.Module()
    m.embed_tokens = torch.nn.Identity()
    return m


def test_range_values_are_the_references_on_int32_and_int64_ids():
    ids = [0, 1, VS - 1, VS, VS + 1, 2 * VS - 1, 2 * VS, V - 1, 2**31 - 1]
    for dt in (torch.int32, torch.int64):
        got = S.range_values(torch, torch.tensor(ids, dtype=dt), VS, VS)
        assert got.dtype == torch.int32 and got.tolist() == [list(VR.values(t, VS, VS)) for t in ids]


def test_the_source_commits_only_inside_the_site_during_a_step():
    com = Recorder()
    src = S.VocabRangeSource(com, _model(), {"embed_tokens": {"start": VS, "shard_rows": VS}})
    launches = []
    fn = src._wrap(lambda *a: launches.append(a) or None)
    ids = torch.tensor([3, VS, 2 * VS - 1, 2 * VS], dtype=torch.int64)
    args = (None, ids, None, VS, 2 * VS, 0, 2 * VS, 2 * VS)
    fn(*args)                                                   # outside a step
    enter, leave = src._enter("embed_tokens"), src._leave("embed_tokens")
    enter(None, ())
    fn(*args)                                                   # inside the site, no step
    com._cur = types.SimpleNamespace(step=0)
    fn(*args)
    leave(None, (), None)
    fn(*args)                                                   # a step, outside the site
    assert len(launches) == 4 and len(com.acquired) == 1
    name, t = com.acquired[0]
    assert name == f"embed_tokens/{VR.MEMBER}" and t.tolist() == [list(VR.values(int(i), VS, VS)) for i in ids]
    enter(None, ())
    fn(None, ids, None, 0, VS, 0, VS, VS)                       # not the manifest's range
    fn(None, ids, None, VS, 2 * VS, 64, 2 * VS, 2 * VS)         # a padded shard
    assert len(com.acquired) == 1 and len(com.errors) == 2 and len(launches) == 6


def test_attach_refuses_a_manifest_without_vocab_range_and_adds_the_required_class():
    com = Recorder()
    assert S.attach(com, False, {}, ("a",)) == (("a",), None)
    with pytest.raises(SystemExit):
        S.attach(com, True, {"identities": []}, ("a",))
    man = _rank_manifest(1, True)
    classes, src = S.attach(com, True, man, ("a",), model=_model().__class__())
    assert classes == ("a", VR.FAMILY) and src.sites == {"model.embed_tokens": {"start": VS, "shard_rows": VS}} and com.sources == [src]


def test_a_wrapper_prefixed_site_names_the_models_module():
    """[GLM-5.3] the global TP manifest spells `inner.model.model.embed_tokens`; the served model's module is `model.embed_tokens`."""
    from verity_vllm.acquire.sources.vocab_range_source import module_of
    mods = {"": 0, "model": 1, "model.embed_tokens": 2, "model.layers.0": 3, "lm_head": 4}
    assert module_of("model.embed_tokens", mods) == "model.embed_tokens"
    assert module_of("inner.model.model.embed_tokens", mods) == "model.embed_tokens"
    assert module_of("inner.model.model.vision.embed_tokens", mods) is None
    assert module_of("x.a.b", {"a.b": 0, "b": 1}) is None          # two candidate suffixes: ambiguous
