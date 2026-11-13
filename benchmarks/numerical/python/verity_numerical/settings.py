"""Every environment variable ``verity_numerical`` reads, declared once in `REGISTRY` and read only through it: as an
attribute of this module (``settings.RESEARCH_RUN_ID``) or by name (`get`), its value parsed from the environment at that
read, or its default when the variable is unset or empty.  ``verity/settings.py`` says what a `Setting` and its roles are.

The registry is built on ``verity/settings.py`` alone: a pod's job runs this package with the repository's root and
``benchmarks/numerical/python`` on its path and no other distribution of the workspace, so the variables research's runner
sets are declared here rather than read from ``verity_config.settings.SHARED``.  It owns no prefix.

``bench/placement.py`` (`STANDALONE`) probes a pod under the pod's own Python, standard library only, so it reads the
environment itself; each name it reads is declared here, and ``tests/test_settings.py`` holds its reads to these declarations.
"""
from __future__ import annotations

from typing import Any

from verity.settings import VALUE, WHERE, EnvSettings, Setting, present

__all__ = ["REGISTRY", "STANDALONE", "get"]

RECORD_SCHEMA = "verity-numerical/settings/v1"
#: the files that run outside the tree, relative to this one
STANDALONE = ("bench/placement.py",)

S = Setting
REGISTRY = EnvSettings(
    __file__, (),
    # ---- research's runner sets these for a run
    S("RESEARCH_RUN_ID", str, None, WHERE, "the research run's id: a result and a generated set's provenance name it; unset, "
                                           "no result.json is written"),
    S("RESEARCH_RUN_DIR", str, None, WHERE, "the research run's directory, where its result.json and reports go"),
    S("RESEARCH_SOURCE_COMMIT", str, None, WHERE, "the commit a caller names for a generated set's provenance; unset, the "
                                                  "checkout's HEAD"),
    # ---- a cell's registration (bench/cell.py, bench/placement.py)
    S("RESEARCH_RUNS", str, "~/.research/runs", WHERE, "the directory research keeps its runs' directories in, where a cell's "
                                                       "prover and verifier runs are read from"),
    S("RESEARCH_CAMPAIGN", str, None, WHERE, "the campaign a cell is registered under: `cell`'s --campaign default"),
    S("RESEARCH_MACHINES_D", str, None, WHERE, "research's machines registry, a directory, where a pod's name resolves to "
                                               "its pod id"),
    # ---- data
    S("VERITY_CENSUS", str, None, VALUE, "the census directory (hardware.json, workloads.json, ...) when it is not the "
                                         "repository's benchmarks/census/"),
    # ---- which tests run
    S("VERITY_BENCH_FULL", present, False, WHERE, "set, tests/bench/test_instances.py rechecks every instance of the built "
                                                  "dataset (minutes): which tests run, never what one computes"),
    schema=RECORD_SCHEMA,
    # a RunPod container's own variables (RUNPOD_POD_ID, RUNPOD_PUBLIC_IP, RUNPOD_DC_ID), which a probe reads by name
    families={"RUNPOD_": WHERE},
    standalone=STANDALONE)
del S

get = REGISTRY.get


def __getattr__(name: str) -> Any:
    if name in REGISTRY.settings:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *REGISTRY.settings])
