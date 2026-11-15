"""Every environment variable verity-bench reads, declared once in a `verity_config.settings.Registry`, the settings
machinery every distribution's registry is built on: what a `Setting` and its roles (`VALUE`, `WHERE`, `DEBUG`) are, how a
read parses and what a record holds are said there.

The harness declares its own run variables (``RESEARCH_RUN_ID`` defaults to ``local`` here, and ``RESEARCH_SOURCE_SHA`` is a
value, which a point's record carries) rather than reading `verity_config.settings.SHARED`'s.  A point writes the record of the
value settings in force (`record`) beside its result (``out/settings.json``).

Read a setting as an attribute, ``settings.VERITY_BENCH_VENV`` (or `get` by name).  `put`, `setdefault` and `unset` write a
declared variable; `environ` is a copy of the whole environment, for code that passes it on (a spec's ``[env]`` and
``"$NAME"`` values name variables the arms read).  A variable the vLLM integration declares (``verity_vllm/settings.py``) is
never declared here too: the harness writes one through the integration's writer, ``verity_vllm.engine.env.apply_target``.

`validate` (the registry's `Registry.check`) refuses a ``VERITY_BENCH_*`` variable that nothing reads (the registry's
``elsewhere`` names the ones other parts of the tree read), a declared value that doesn't parse, and on a run of record a
debug setting that is set, each by name: ``verity-bench`` runs it at startup, so a misspelt or retired setting fails the run
instead of being ignored.

The harness reads and writes ``os.environ`` here and nowhere else, but for the uses ``tests/test_settings.py``'s shrink-only
``BASELINE`` holds (`verity_config.settings.direct_uses` counts them).
"""
from __future__ import annotations

from typing import Any

from verity_config.settings import VALUE, WHERE, Registry, Setting

S = Setting
REGISTRY = Registry(__file__, ("VERITY_BENCH_",),
    # ---- the dispatcher and a point's job (jobs.py, run_point.sh, env.sh, setup/)
    S("VERITY_BENCH_VENV", str, None, WHERE, "the node's pinned-vLLM Python a point runs under (env.sh), which the dispatcher "
      "forwards when set: an interpreter, whose vLLM and torch versions the result's identity records"),
    S("VERITY_BENCH_SETUP", str, None, VALUE, "the spec's setup script, which run_point.sh sources before the point's Python "
      "(the dispatcher sets it from the spec; a point refuses any other)"),
    S("VERITY_BENCH_LIB_CACHE", str, None, WHERE, "the node's cache of the libraries setup/vllm_encoded.sh builds, each under its "
      "recipe's and sources' digest; unset, /workspace/research/cache/verity-bench-libs"),
    # ---- a steady-state point's node side (steady_point.sh, steady_node.py)
    S("VERITY_BENCH_STEADY_PY", str, None, WHERE, "the Python (3.12 with numpy and blake3) the spec's setup made on the "
      "node, which runs a steady-state point's node side, the stager and the script that times the provers"),
    S("VERITY_BENCH_STEADY_BIN", str, None, WHERE, "the prover the spec's setup built on the node, which a steady-state "
      "point's provers and verifiers run; its SHA-256 is in the point's record"),
    S("VERITY_BENCH_STEADY_SCRATCH", str, "/dev/shm", WHERE, "where a steady-state point stages its statement and keeps "
      "the provers' session files, in a directory of its own that it removes at its end; a spec's setup exports a "
      "directory on the node's disk where a row's statement is fetched from the store"),
    S("CUDA_VISIBLE_DEVICES", str, None, WHERE, "the card gpu-lease gave a steady-state point's timed step"),
    # ---- research run's (the point's run directory and record)
    S("RESEARCH_RUN_DIR", str, ".", WHERE, "the run directory a point writes result.json in"),
    S("OUT", str, None, WHERE, "the directory a point writes its outputs in; unset, RESEARCH_RUN_DIR/out"),
    S("RESEARCH_RUN_ID", str, "local", WHERE, "the research run a point is, which its result names"),
    S("RESEARCH_SOURCE_SHA", str, None, VALUE, "the commit research run shipped, which a point's identity hashes; unset, the "
      "checkout's HEAD"),
    # ---- gpu-lease's (the kit's host stamp)
    S("GPU_LEASE_UUID", str, None, WHERE, "the UUIDs of the GPUs gpu-lease leased the job (it sets them), which the host stamp "
      "reads the clocks of; unset, every GPU of the host"),
    elsewhere=(
        "VERITY_BENCH_FULL",       # benchmarks/numerical/tests/bench/test_instances.py's opt-in
        "VERITY_BENCH_CUBIN",      # `verity bench overhead run --cubin`'s file, which overhead_node sets and a scheme's arm loads
    ), shared=False)
del S

SETTINGS = REGISTRY.settings
setting, raw, is_set, get, put, unset, setdefault = (REGISTRY.setting, REGISTRY.raw, REGISTRY.is_set, REGISTRY.get,
                                                     REGISTRY.put, REGISTRY.unset, REGISTRY.setdefault)
environ, record, unknown, validate = REGISTRY.environ, REGISTRY.record, REGISTRY.unknown, REGISTRY.check


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])
