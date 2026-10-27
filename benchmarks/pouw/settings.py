"""Every environment variable the PoUW benchmarks read (``benchmarks/pouw`` and its filler), declared once: the shape of the
vLLM integration's registry (``integrations/vllm/verity_vllm/settings.py``), for this distribution.

A `Setting` has a name, a type (the parser of the variable's string), a default, a role and a doc.  The role says what the
setting can change:

* ``value`` (`VALUE`): what is computed or measured.  `record` carries every value setting that is set, with its digest, and
  `run_result.write` puts it in each run's ``result.json``.
* ``where`` (`WHERE`): only where and how a run happens: paths, devices, run ids, who leased the GPU.
* ``debug`` (`DEBUG`): diagnostics that change no number a run reports.  `record` refuses one that is set, so a run of record
  never carries one.

Read a setting as an attribute, ``settings.REPS`` (or `get` by name): its value parsed from the process environment at that
read, or its default when the variable is unset or empty.  An undeclared name is an AttributeError, and a string that doesn't
parse is a ValueError naming the variable.  `put` and `setdefault` write a declared variable for the processes a script starts;
`environ` is a copy of the whole environment, for code that passes it on.

Importing this module refuses a ``VERITY_POUW_*`` variable that neither this registry nor the rest of the tree reads
(`ELSEWHERE`), naming it.  A script imports it as ``import settings`` with ``benchmarks/pouw`` on its path, as it imports
``run_result``.  Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

VALUE, WHERE, DEBUG = "value", "where", "debug"
ROLES = (VALUE, WHERE, DEBUG)
RECORD_SCHEMA = "verity-pouw-benchmarks/settings/v1"
PREFIX = "VERITY_POUW_"


@dataclass(frozen=True)
class Setting:
    name: str
    type: Callable[[str], Any]
    default: Any
    role: str
    doc: str


def _declare(*settings: Setting) -> dict[str, Setting]:
    out: dict[str, Setting] = {}
    for s in settings:
        if s.name in out or s.role not in ROLES:
            raise ValueError(f"{s.name}: declared twice, or role {s.role!r} is not one of {ROLES}")
        out[s.name] = s
    return out


S = Setting
SETTINGS: dict[str, Setting] = _declare(
    # ---- research run's (tools/research/src/research/result.py, the runner, the GPU lease)
    S("RESEARCH_RUN_ID", str, None, WHERE, "the research run this process is part of"),
    S("RESEARCH_RUN_DIR", str, None, WHERE, "the run's directory, where result.json goes"),
    S("RESEARCH_SOURCE_SHA", str, None, WHERE, "the commit of the tree the run ships (research run --source)"),
    S("RESEARCH_LANE", str, None, WHERE, "the lane the run is recorded under"),
    S("OUT", str, None, WHERE, "the run's out/ directory, which the run's custody keeps"),
    S("GPU_LEASE_UUID", str, None, WHERE, "the UUID of the GPU the run leased"),
    S("GPU_LEASE_WHO", str, None, WHERE, "who holds the GPU lease"),
    S("CUDA_VISIBLE_DEVICES", str, None, WHERE, "the devices this process may use"),
    S("PATH", str, "", WHERE, "the executable search path (filler/score.py puts nvcc's directory first)"),
    # ---- the toolchain
    S("NVCC", str, None, WHERE, "the nvcc a build runs; unset, the script's own"),
    S("CUDA_HOME", str, None, WHERE, "the CUDA toolkit whose bin/nvcc opcode_probe builds with, after NVCC"),
    S("NVCC_APPEND_FLAGS", str, None, VALUE, "flags nvcc appends to every compile (a filler arm sets its own)"),
    S("VERITY_POUW_CUDA_BUILD", str, None, WHERE, "vllm_bench's NCP-2 committer's build directory"),
    S("VERITY_POUW_CUDA_ARCH", str, None, VALUE, "the architecture vllm_bench's NCP-2 committer compiles for"),
    S("VLLM_ENABLE_V1_MULTIPROCESSING", str, None, VALUE, "vLLM's engine-core process; vllm_bench sets 0, one process"),
    # ---- nvfp4_sm120/ (nvf4_bench.py, fp8_bench.py)
    S("NVF4_EXPECT_UUID", str, None, WHERE, "the GPU UUID nvf4_bench refuses any other than"),
    S("FP8_EXPECT_UUID", str, None, WHERE, "the GPU UUID fp8_bench refuses any other than"),
    S("HARNESS_LIB", str, None, WHERE, "the CUTLASS harness library the bench times against (its sha256 is emitted)"),
    S("LT2_LIB", str, None, WHERE, "the second cuBLASLt harness library (its sha256 is emitted)"),
    S("PLAIN_LIB", str, "", WHERE, "plain baseline libraries, space-separated (each sha256 is emitted)"),
    S("MODEL_CACHE", str, None, WHERE, "the directory of the host model's cached outputs (`fp8_bench --model-cache` sets it)"),
    S("SHAPE", str, "8192", VALUE, "the timed cube sizes, space-separated (fp8_bench sets 8192 16384)"),
    S("SMALL", str, "256x128x128 512x256x1024 768x384x640", VALUE, "the gated small shapes MxNxK, space-separated"),
    S("SCALES", str, "0 1", VALUE, "the operands' scale ranges, space-separated"),
    S("RASTER", int, 8, VALUE, "the kernels' raster swizzle"),
    S("CUTLASS", str, None, VALUE, "the harness's CUTLASS kernels the registry times, space-separated; unset, the bench's"),
    S("PLAIN_ENTRIES", str, "", VALUE, "the plain libraries' entries timed, space-separated; empty, all"),
    S("KERNELS", str, None, VALUE, "the bench's own kernels, space-separated; unset, the bench's"),
    S("LT", str, "1", VALUE, "1: time cuBLASLt through HARNESS_LIB's plan"),
    S("TIME_SCALES", int, 1, VALUE, "the scale range the timing runs at"),
    S("GATE_REPEATS", int, 3, VALUE, "the gate's repeats at a timed size"),
    S("MODEL_SAMPLES", int, 4096, VALUE, "the words the host model checks a shape at"),
    S("FULL_MODEL", int, 131072, VALUE, "the largest shape, in words, the host model checks whole"),
    S("ONLY_CHECK", str, None, VALUE, "set: gate the kernels and time nothing"),
    S("STEADY", int, None, VALUE, "set: seconds of steady-clock timing a size gets"),
    S("STEADY_ROUNDS", int, 4, VALUE, "the steady timing's rounds"),
    S("STAMPS", str, None, DEBUG, "set: each gated kernel's clock64 stamps, one launch each"),
    S("PROFILE", str, None, DEBUG, "set: one launch of each kernel between cuProfilerStart and Stop (ncu)"),
    S("REPS", int, None, VALUE, "the timed repetitions; unset, nvf4_bench's 20 and served_zk/rows_bench's 30"),
    # ---- served_zk/, hidden_zk/, h1_prove/, h1_lanes/
    S("PEARLC_EXPECT_UUID", str, None, WHERE, "the GPU UUID served_zk/rows_bench refuses any other than"),
    S("CALLS", int, 32, VALUE, "served_zk/rows_bench's timed calls"),
    S("PREFILL", int, 8192, VALUE, "served_zk/rows_bench's prefill tokens"),
    S("HZ_STAGE", str, "fast", VALUE, "how a hidden-ZK unit's witness is staged: fast, or the reference"),
    S("HZ_LEAN", str, None, VALUE, "the Lean checks hidden_zk/summarize runs; unset, every one (*)"),
    S("HZ_STAGE_ONLY", str, "0", VALUE, "1: the summary is of the staging alone, with no verdict"),
    S("H1_TILE", str, "v1", VALUE, "the H1 tile under h1_prove's unit: v1 or v3"),
    S("H1_TMPL", str, None, VALUE, "h1_prove's template mode; unset, flat"),
    S("H1L_RC", str, "16x16", VALUE, "h1_lanes/j10r's tile, RxC"),
    # ---- opt-in tests (tests/)
    S("PARTIALS_LIB", str, None, WHERE, "the built partials library test_partials and test_lane_refs load; unset, they skip"),
    S("VERITY_POUW_H1_PIN", str, None, WHERE, "the H1 pin file test_h1_pin checks; unset, it skips"),
    S("SZ_PARITY_E2E", str, None, VALUE, "1: test_served_zk runs the served parity end to end"),
    S("SERVED_RUN", str, None, WHERE, "the served run test_served_zk's end-to-end parity reads"),
    S("SZ_CALL_LIST", str, None, VALUE, "the calls test_served_zk's end-to-end parity proves"),
    S("HZ_MEM_GB", str, None, WHERE, "the memory bound test_served_zk's end-to-end parity holds the prover to"),
)
del S

#: The ``VERITY_POUW_*`` variables the rest of the tree reads: the vLLM integration's PoUW device, the kernels' tests, the
#: window's lanes test and all-protocols' PoUW leg.
ELSEWHERE = frozenset("""
VERITY_POUW_CHECK VERITY_POUW_H1U_SO VERITY_POUW_NOISE_CUBIN VERITY_POUW_NVCC VERITY_POUW_WINDOW_LOWERED_LANES
""".split())


def setting(name: str) -> Setting:
    s = SETTINGS.get(name)
    if s is None:
        raise AttributeError(f"{name} is not a declared setting (benchmarks/pouw/settings.py)")
    return s


def get(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """A declared setting's value in ``environ`` (default the process environment)."""
    s = setting(name)
    v = (os.environ if environ is None else environ).get(name)
    if not v:
        return s.default
    try:
        return s.type(v)
    except ValueError as e:
        raise ValueError(f"{name}={v!r}: {e}") from None


def put(name: str, value: str) -> None:
    """Set a declared variable in this process's environment, which the processes it starts inherit."""
    setting(name)
    os.environ[name] = value


def setdefault(name: str, value: str) -> str:
    setting(name)
    return os.environ.setdefault(name, value)


def environ() -> dict[str, str]:
    """A copy of the whole process environment, to pass on to a child process."""
    return dict(os.environ)


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def record(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The value settings in force, as the environment holds them, with their digest (SHA-512 of the canonical JSON of
    ``values``).  Refuses a debug setting that is set: a run of record carries none."""
    env = os.environ if environ is None else environ
    if debug := sorted(k for k, s in SETTINGS.items() if s.role == DEBUG and env.get(k)):
        raise ValueError(f"debug setting{'s' * (len(debug) > 1)} {', '.join(debug)} set on a run of record: unset "
                         f"{'them' if len(debug) > 1 else 'it'}")
    values = {k: env[k] for k in sorted(SETTINGS) if SETTINGS[k].role == VALUE and env.get(k)}
    blob = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"schema": RECORD_SCHEMA, "values": values, "digest": hashlib.sha512(blob).hexdigest()}


def unknown(environ: Iterable[str] | None = None) -> list[str]:
    return sorted(k for k in (os.environ if environ is None else environ)
                  if k.startswith(PREFIX) and k not in SETTINGS and k not in ELSEWHERE)


def validate_environ(environ: Iterable[str] | None = None) -> None:
    """Refuse the ``VERITY_POUW_*`` variables nothing in the tree reads, naming each."""
    if bad := unknown(environ):
        raise ValueError(f"unknown setting(s) {', '.join(bad)}: nothing in the tree reads them; unset them, fix the name, "
                         f"or declare them in benchmarks/pouw/settings.py (SETTINGS, or ELSEWHERE when another part of "
                         f"the tree reads them)")


validate_environ()
