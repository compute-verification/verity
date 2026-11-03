"""Every distribution's settings registry but ``verity``'s and the vLLM integration's: each declares every environment
variable it reads once, in its own registry, and reads them only through it.  A registry is a `Registry` of
``verity/settings.py``, the machinery every registry in the tree is built on, which says what a `Setting` is, what its roles
(`VALUE`, `WHERE`, `DEBUG`) mean and how a read parses; this module adds what the other distributions share: `SHARED`, the
record schema they write (`RECORD_SCHEMA`), and `of`, which finds a distribution's registry by path.

A distribution's registry is the ``settings.py`` beside its ``pyproject.toml``, which declares its variables and the prefixes
it owns::

    from verity_config.settings import VALUE, WHERE, Registry, Setting as S
    REGISTRY = Registry(__file__, ("PEARLC_",),
        S("PEARLC_SCHEDULE", str, "deferred", VALUE, "when the Pearl-C arm's hashes run: deferred (served) or serial"),
        ...)

and its code reads them by path, so two distributions' registries never shadow each other on ``sys.path`` (a ``research run``
puts every flat workspace member's directory there, so each ``settings.py`` would be a top-level ``settings``)::

    from verity_config.settings import of
    S = of(__file__)        # the registry of the distribution this file is in
    S.PEARLC_SCHEDULE       # parsed from the process environment at this read

`SHARED` declares, once for every registry, the variables the tools around a run set: research's runner, gpu-lease, CUDA,
Hugging Face and Python.  Every `Registry` reads them too, and a distribution with no variables of its own reads them from
`SHARED` directly, with no ``settings.py``.  A run of record refuses a debug setting that is set (`Registry.of_record`), and
`of` checks a registry's prefixes (`Registry.validate_environ`) when it first loads it, so a misspelt or retired knob fails
the run instead of being ignored.

A ship (files a build copies out of the tree to run on another machine) takes its distribution's ``settings.py`` with it,
and `of` finds it beside them.  A file that can't import the registry at all (stdlib only, or run under another tree's
PYTHONPATH) reads the environment itself; the registry still declares each variable it reads and names the file
(``standalone``), and `Registry.standalone_problems`, which the distribution's suite runs, holds each read to its
declaration.

Stdlib and ``verity.settings`` only.  Outside the vLLM integration the environment is read in the registries (a module named
``settings.py``) and in the files ``tests/lint/allowlists/repo_env_reads.json`` still lists, which only shrinks.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from verity import settings
from verity.settings import WHERE, Setting, paths
from verity.settings import (DEBUG as DEBUG, REDACTED as REDACTED, ROLES as ROLES, SECRET_WORDS as SECRET_WORDS, VALUE as VALUE,
                             DebugOnRecord as DebugOnRecord, UnknownSetting as UnknownSetting, choice as choice, csv as csv,
                             direct_reads as direct_reads, flag as flag, ints as ints, not_zero as not_zero, one as one,
                             present as present, stripped as stripped, words as words)

RECORD_SCHEMA = "verity-config/settings/v1"
#: a distribution's registry, beside its pyproject.toml
REGISTRY_FILE = "settings.py"


class Registry(settings.Registry):
    """A distribution's registry: `verity.settings.Registry`, writing `RECORD_SCHEMA`, that reads `SHARED`'s settings too
    (with ``shared`` false, its own alone)."""

    def __init__(self, path: str | os.PathLike, prefixes: Iterable[str], *declared: Setting, shared: bool = True,
                 **kw: Any) -> None:
        super().__init__(path, prefixes, *declared, schema=RECORD_SCHEMA, shared=SHARED if shared else None, **kw)


SHARED = Registry(__file__, (), *(Setting(*a) for a in (
    ("RESEARCH_RUN_ID", str, None, WHERE, "the research run's id (research's runner sets it): it names a run's outputs"),
    ("RESEARCH_RUN_DIR", str, None, WHERE, "the research run's directory (research's runner sets it), where its result.json "
                                           "goes"),
    ("RESEARCH_SOURCE_SHA", str, None, WHERE, "the commit a research run's source tree is (research's runner sets it): "
                                              "provenance, which the run records"),
    ("RESEARCH_SOURCE_COMMIT", str, None, WHERE, "the commit a caller names for a run's provenance (research's runner sets "
                                                 "RESEARCH_SOURCE_SHA instead)"),
    ("RESEARCH_LANE", str, None, WHERE, "the lane a run is for (research's runner sets it)"),
    ("RESEARCH", str, "research", WHERE, "the research command a script runs"),
    ("OUT", str, None, WHERE, "the run's output directory (research's runner sets it: the writing-runs skill)"),
    ("GPU_LEASE_UUID", str, None, WHERE, "the UUID of the GPU gpu-lease leased the process (gpu-lease sets it)"),
    ("GPU_LEASE_WHO", str, None, WHERE, "who holds the process's GPU lease (gpu-lease sets it)"),
    ("CUDA_VISIBLE_DEVICES", str, None, WHERE, "the GPUs CUDA shows the process (gpu-lease sets it)", True),
    ("CUDA_DEVICE_ORDER", str, None, WHERE, "how CUDA numbers the GPUs (PCI_BUS_ID: as nvidia-smi does)"),
    ("CUDA_HOME", str, None, WHERE, "the CUDA toolkit's directory"),
    ("NVCC", str, None, WHERE, "the nvcc binary a build runs"),
    ("HF_HOME", str, None, WHERE, "Hugging Face's cache directory, where weights are read from"),
    ("PYTHONPATH", paths, (), WHERE, "Python's import roots (research's runner sets them from the workspace)"),
    ("OMP_NUM_THREADS", int, None, WHERE, "the threads of a CPU reference's BLAS: how fast it runs, not what it computes"),
    ("FILL_JOB", str, None, WHERE, "the fill job a process runs as (research's fill runner sets it): it names a run's outputs"),
    ("DITHER_NUMA_RECORD", json.loads, None, WHERE, "the NUMA binding benchmarks/hash/dither/numabind.py started the process "
                                                    "under, as JSON (it sets it): where the process ran, which a run records"),
)), shared=False)


def _loaded(root: Path) -> Registry:
    name = f"verity_config.settings:{root}"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, root / REGISTRY_FILE)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
            reg = getattr(mod, "REGISTRY", None)
            if not isinstance(reg, Registry):
                raise TypeError(f"{root / REGISTRY_FILE}: REGISTRY is the distribution's Registry")
            reg.validate_environ()
        except BaseException:
            del sys.modules[name]
            raise
    return mod.REGISTRY


def of(path: str | os.PathLike) -> Registry:
    """The registry of the distribution ``path`` is in: the ``settings.py`` beside the nearest ``pyproject.toml`` above it,
    or, in a ship (files copied out of the tree with their registry, no ``pyproject.toml`` above them), the ``settings.py``
    beside ``path``.  Loaded once a process (by its path, never by its module name) and checked then
    (`Registry.validate_environ`)."""
    here = Path(path).resolve()
    for root in here.parents:
        if (root / "pyproject.toml").is_file():
            if not (root / REGISTRY_FILE).is_file():
                raise LookupError(f"{here}: its distribution ({root}) has no settings registry: declare its variables in "
                                  f"{root / REGISTRY_FILE} (verity_config.settings), or read the shared ones from SHARED")
            return _loaded(root)
    if (here.parent / REGISTRY_FILE).is_file():
        return _loaded(here.parent)
    raise LookupError(f"{here}: no pyproject.toml above it and no {REGISTRY_FILE} beside it, so no settings registry: ship "
                      f"its distribution's {REGISTRY_FILE} with it")
