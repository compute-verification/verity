"""`verity_vllm.observe.capture`'s `make_header`: the run header is the profile manifest and its id as
`run_facts.profile_manifest` gives them for the workload's sampling, the model's checkpoint and configuration, the workload's
requests, and the notes: the manifest's, the observer's own, then the caller's.  A request with `n` over one is listed with
the children vLLM makes of it.  The manifest, the configuration, the re-entered implementations' names and the clock are
stood in."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.observe import capture as C
from verity_vllm.observe import events as E

OPS = ("vllm.unified_attention_with_output", "vllm.unified_kv_cache_update")


@pytest.fixture
def asked(monkeypatch) -> list:
    """What `make_header` asked of `profile_manifest`, which is stood in beside the model's configuration, the re-entered
    implementations' names and the clock."""
    calls: list = []

    def profile_manifest(engine: Any, **kw: Any) -> tuple[dict, str, dict]:
        calls.append((engine, kw))
        return {"the": "manifest"}, "the profile id", {"case": "of the manifest", "phases": "the manifest's"}

    monkeypatch.setattr(C, "profile_manifest", profile_manifest)
    monkeypatch.setattr(C, "model_config_subset", lambda engine: {"hidden_size": 2, "of": engine.case})
    monkeypatch.setattr(C, "reenter_impl_name", lambda op: None if op == OPS[1] else f"the implementation of {op}")
    monkeypatch.setattr(C, "time", SimpleNamespace(gmtime=lambda: "now", strftime=lambda form, at: f"{form} of {at}"))
    return calls


def _engine(**checkpoint: Any) -> SimpleNamespace:
    return SimpleNamespace(case="a case", cp={"repo": "a repo", "revision": "a revision", **checkpoint})


def test_the_header_is_the_manifest_the_model_the_workload_and_the_notes(asked):
    engine = _engine(local_path="/a/path", more="than the header keeps")
    workload = {"requests": [{"request_id": "r0", "n": 1, "seed": 5}, {"request_id": "r1"}], "sampling": {"greedy": True},
                "source": "a source", "sha256": "a digest", "more": "than the header keeps"}
    header = C.make_header(engine, workload, adapter_revision="a revision of the adapter", wheel_sha256="a wheel", vllm_commit="a commit")
    assert asked == [(engine, {"adapter_revision": "a revision of the adapter", "wheel_sha256": "a wheel", "vllm_commit": "a commit",
                               "sampling": {"greedy": True}})]
    assert isinstance(header, E.RunHeader) and header.schema == E.SCHEMA and header.created_utc == "%Y-%m-%dT%H:%M:%SZ of now"
    assert (header.profile_id, header.profile) == ("the profile id", {"the": "manifest"})
    assert header.model == {"repo": "a repo", "revision": "a revision", "local_path": "/a/path", "case": "a case",
                            "config": {"hidden_size": 2, "of": "a case"}}
    assert header.workload == {"requests": workload["requests"], "sampling": {"greedy": True}, "source": "a source", "sha256": "a digest"}
    assert sorted(header.notes) == ["case", "observer_sync", "phases", "reenter_impls"] and header.notes["case"] == "of the manifest"
    assert header.notes["reenter_impls"] == {OPS[0]: f"the implementation of {OPS[0]}", OPS[1]: None}
    assert header.notes["phases"].startswith("forward = inside the model's root forward; other = ")
    assert header.notes["observer_sync"].startswith("no observer-issued synchronisation; ")


def test_a_header_with_no_wheel_commit_path_or_source_has_none_and_the_callers_notes_are_the_last_word(asked):
    header = C.make_header(_engine(), {"requests": [], "sampling": {}}, adapter_revision="a revision of the adapter",
                           extra_notes={"phases": "the caller's", "argv": ["a", "b"]})
    assert asked[0][1] == {"adapter_revision": "a revision of the adapter", "wheel_sha256": None, "vllm_commit": None, "sampling": {}}
    assert header.model["local_path"] is None and header.workload == {"requests": [], "sampling": {}, "source": None, "sha256": None}
    assert header.notes["phases"] == "the caller's" and header.notes["argv"] == ["a", "b"] and len(header.notes) == 5
    assert C.make_header(_engine(), {"requests": [], "sampling": {}}, adapter_revision="r", extra_notes={}).notes["phases"] != "the caller's"


def test_a_request_with_n_over_one_is_listed_with_the_children_vllm_makes_of_it(asked):
    requests = [{"request_id": "one", "n": 1, "seed": 3}, {"request_id": "none"}, {"request_id": "zero", "n": 0},
                {"request_id": "seeded", "n": 3, "seed": "7"}, {"request_id": "unseeded", "n": 2, "seed": None},
                {"request_id": "no seed", "n": "2"}]
    header = C.make_header(_engine(), {"requests": requests, "sampling": {}}, adapter_revision="r")
    assert header.workload["requests"] == requests and header.workload["children"] == [
        {"request_id": "0_seeded", "parent": "seeded", "index": 0, "seed": 7},
        {"request_id": "1_seeded", "parent": "seeded", "index": 1, "seed": 8},
        {"request_id": "2_seeded", "parent": "seeded", "index": 2, "seed": 9},
        {"request_id": "0_unseeded", "parent": "unseeded", "index": 0, "seed": None},
        {"request_id": "1_unseeded", "parent": "unseeded", "index": 1, "seed": None},
        {"request_id": "0_no seed", "parent": "no seed", "index": 0, "seed": None},
        {"request_id": "1_no seed", "parent": "no seed", "index": 1, "seed": None}]
    assert "children" not in C.make_header(_engine(), {"requests": requests[:3], "sampling": {}}, adapter_revision="r").workload
