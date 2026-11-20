"""The package's settings registry: what an unset variable takes, that a read parses the environment at that read, and that
the probe, which reads the environment itself, reads only declared names."""
from __future__ import annotations

from pathlib import Path

from verity_numerical import settings
from verity_numerical.bench import placement


def test_the_probe_reads_only_declared_names():
    assert [Path(settings.__file__).parent / f for f in settings.STANDALONE] == [Path(placement.__file__)]
    assert settings.REGISTRY.standalone_problems() == []
    assert settings.REGISTRY.role_of("RUNPOD_POD_ID") == settings.REGISTRY.role_of("RESEARCH_MACHINES_D") == "where"
