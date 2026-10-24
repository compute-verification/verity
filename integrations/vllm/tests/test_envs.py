"""`verity_vllm.envs`: each access reads the environment then, and each parse is the expression it replaced at its call
site (unset, set, and the edge values each parse treats specially)."""
from __future__ import annotations

import pytest

from verity_vllm import envs


def test_each_access_reads_the_environment_then(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_TIMELINE", raising=False)
    assert envs.VERITY_TIMELINE is None
    monkeypatch.setenv("VERITY_TIMELINE", "/x/timeline.jsonl")
    assert envs.VERITY_TIMELINE == "/x/timeline.jsonl"


@pytest.mark.parametrize("name", ["VERITY_FA2_COMPACT", "VERITY_FA2_ZERO"])
@pytest.mark.parametrize("value,on", [(None, True), ("1", True), ("", True), ("yes", True), ("0", False), (" false ", False),
                                      ("no", False), ("False", True)])
def test_on_unless_off(monkeypatch: pytest.MonkeyPatch, name: str, value: str | None, on: bool) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert getattr(envs, name) is on


@pytest.mark.parametrize("value,want", [(None, None), ("512", 512), (" 64 ", 64), ("", 0), ("0", 0)])
def test_fa2_tap_cap(monkeypatch: pytest.MonkeyPatch, value: str | None, want: int | None) -> None:
    if value is None:
        monkeypatch.delenv("VERITY_FA2_TAP_CAP_MB", raising=False)
    else:
        monkeypatch.setenv("VERITY_FA2_TAP_CAP_MB", value)
    assert envs.VERITY_FA2_TAP_CAP_MB == want


def test_defaults_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VERITY_PINNED_EXACT", "VERITY_REPLAY_DUMP_FAMILIES", "VERITY_REPLAY_DUMP_MAX", "CXX"):
        monkeypatch.delenv(name, raising=False)
    assert (envs.VERITY_PINNED_EXACT, envs.VERITY_REPLAY_DUMP_FAMILIES, envs.VERITY_REPLAY_DUMP_MAX, envs.CXX) == (True, (), 1024, "g++")
    monkeypatch.setenv("VERITY_PINNED_EXACT", "false")
    assert envs.VERITY_PINNED_EXACT is True
    monkeypatch.setenv("VERITY_PINNED_EXACT", "0")
    assert envs.VERITY_PINNED_EXACT is False
    monkeypatch.setenv("VERITY_REPLAY_DUMP_FAMILIES", "Rope,,Gemm")
    monkeypatch.setenv("VERITY_REPLAY_DUMP_MAX", "3")
    monkeypatch.setenv("CXX", "clang++")
    assert (envs.VERITY_REPLAY_DUMP_FAMILIES, envs.VERITY_REPLAY_DUMP_MAX, envs.CXX) == (("Rope", "Gemm"), 3, "clang++")


def test_an_undeclared_variable_is_refused_by_name() -> None:
    with pytest.raises(AttributeError, match="declare the variable in verity_vllm/envs.py"):
        envs.VERITY_NOT_A_SETTING  # noqa: B018
    assert "VERITY_TIMELINE" in dir(envs)
