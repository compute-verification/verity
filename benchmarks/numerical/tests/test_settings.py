"""The package's settings registry: what an unset variable takes, that a read parses the environment at that read, and that
the probe, which reads the environment itself, reads only declared names."""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_numerical import settings
from verity_numerical.bench import cell, placement


def test_an_unset_variable_takes_its_declared_default(monkeypatch):
    for name in settings.REGISTRY.own:
        monkeypatch.delenv(name, raising=False)
    assert settings.RESEARCH_RUNS == "~/.research/runs" and settings.VERITY_BENCH_FULL is False
    unset = ("RESEARCH_RUN_ID", "RESEARCH_RUN_DIR", "RESEARCH_SOURCE_COMMIT", "RESEARCH_CAMPAIGN", "RESEARCH_MACHINES_D",
             "VERITY_CENSUS")
    assert [settings.get(n) for n in unset] == [None] * len(unset)
    assert set(settings.REGISTRY.own) == {"RESEARCH_RUNS", "VERITY_BENCH_FULL", *unset}


def test_a_read_takes_the_environment_as_it_is_then(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setenv("VERITY_BENCH_FULL", "0")
    assert cell.runs_root() == tmp_path and settings.RESEARCH_RUN_ID == "r1" and settings.VERITY_BENCH_FULL is True
    monkeypatch.setenv("RESEARCH_RUN_ID", "")
    assert settings.RESEARCH_RUN_ID is None


def test_an_undeclared_name_is_refused():
    with pytest.raises(AttributeError, match="NOT_DECLARED"):
        _ = settings.NOT_DECLARED
    with pytest.raises(AttributeError, match="not a declared setting"):
        settings.get("NOT_DECLARED")


def test_the_probe_reads_only_declared_names():
    assert [Path(settings.__file__).parent / f for f in settings.STANDALONE] == [Path(placement.__file__)]
    assert settings.REGISTRY.standalone_problems() == []
    assert settings.REGISTRY.role_of("RUNPOD_POD_ID") == settings.REGISTRY.role_of("RESEARCH_MACHINES_D") == "where"
