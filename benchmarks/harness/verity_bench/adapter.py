"""What the benchmarks take from the vLLM adapter (`verity_vllm`) that its public interface doesn't offer yet.

The adapter's public interface is the modules ``tests/test_vllm_public_api.py`` lists in ``PUBLIC`` and, from each, the names
in its ``__all__``. Benchmarks reach the adapter through that interface only (ARCHITECTURE.md), so in ``benchmarks/harness``,
``overhead``, ``prove``, ``exfiltration``, ``recursion_e2e`` and ``recursion_figures`` this module is the one file that imports
past it, each name once, and every other file imports the name from here (``tests/test_adapter.py``). A name gains a public home
when the adapter lists it; then its importers take it from there and its line here goes.

The modules imported at the top here import only the standard library. A name whose module imports torch, vLLM or PoUW's
serving code is a function that imports it when called, so importing this module stays cheap and the callers' imports stay as
lazy as they were.

| Here | Non-public home | Proposed public name | Used by (ours) |
| --- | --- | --- | --- |
| `Hooks`, `live` | `verity_vllm.engine.hooks` | `verity_vllm.engine.hooks.Hooks`, `.live` (I2: the patches a component installs on an engine, and those installed) | vllm/arms.py, vllm/engine.py |
| `apply_engine`, `apply_target`, `apply_compile_fresh`, `own_inductor_dir`, `INDUCTOR_DIR_MARK` | `verity_vllm.engine.env` | `verity_vllm.engine.env.apply_engine`, `.apply_target`, `.apply_compile_fresh`, `.own_inductor_dir`, `.INDUCTOR_DIR_MARK` (I1: the engine's environment, written before torch and vLLM are imported) | vllm/engine.py, vllm/commit_arm.py, tests/test_bench.py |
| `run_requests_arrivals` | `verity_vllm.engine.arrivals` | `verity_vllm.engine.arrivals.run_requests_arrivals` (I1: an engine driven over a step-indexed arrival schedule) | vllm/engine.py |
| `register_int8_parts` | `verity_vllm.engine.int8_ranks.register` | `verity_vllm.engine.int8_ranks.register` (I1: the exact-int8 parts a process's engines build with) | vllm/engine.py |
| `int8_exact_of` | `verity_vllm.program.frontend.kernel_readback.int8_exact_of` | the same, its module public (I7: the int8 scope a built model's layers run) | vllm/engine.py |
| `native_collect_committer` | `verity_vllm.commit.committer.native_collect.NativeCollectCommitter` | the same, its module public beside `native_host` (I7: the windowed native collector) | vllm/commit_arm.py |
| `challenged` | `verity_vllm.protocols.pouw.challenged` | `verity_vllm.protocols.pouw.challenged` (I1: a protocols config whose PoUW coins are an in-process challenger's) | vllm/arms.py |
| `h1_run` | `verity_vllm.protocols.pouw.h1_run` | `verity_vllm.protocols.pouw.h1_run` (I1: the H1 run an installed H1 option serves) | vllm/engine.py |
| `pouw_refusals` | `verity_vllm.protocols.pouw.refusals` | `verity_vllm.protocols.pouw.refusals` (I1: why PoUW's option refuses a scheme in a context) | tests/test_bench.py |
| `pearl_c_device_run` | `verity.protocols.accounting.pouw.prover.serving.pearl_c.PearlCDeviceRun` | the same, its module public (I1: an installed Pearl-C option's device run) | vllm/engine.py, tests/test_bench.py |
| `pearl_c_schemes` | `verity.protocols.accounting.pouw.prover.serving.pearl_c.SCHEMES` | the same, its module public (I1: the Pearl-C schemes the device path runs) | tests/test_bench.py |

Each proposal makes the module public as it stands, with ``__all__`` listing at least these names. The last five are PoUW's
serving code, which ARCHITECTURE.md moves under ``verity/protocols/accounting/pouw/``; until then they are the adapter's PoUW
option.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from verity_vllm.engine.arrivals import run_requests_arrivals
from verity_vllm.engine.env import INDUCTOR_DIR_MARK, apply_compile_fresh, apply_engine, apply_target, own_inductor_dir
from verity_vllm.engine.hooks import Hooks, live

__all__ = ["INDUCTOR_DIR_MARK", "Hooks", "apply_compile_fresh", "apply_engine", "apply_target", "challenged", "h1_run",
           "int8_exact_of", "live", "native_collect_committer", "own_inductor_dir", "pearl_c_device_run", "pearl_c_schemes",
           "pouw_refusals", "register_int8_parts", "run_requests_arrivals"]


def register_int8_parts(parts: Iterable[str]) -> list[str]:
    from verity_vllm.engine import int8_ranks

    return int8_ranks.register(parts)


def int8_exact_of(model: Any) -> str | None:
    """Imports torch."""
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of

    return int8_exact_of(model)


def native_collect_committer() -> type:
    """The class; imports torch."""
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

    return NativeCollectCommitter


def challenged(settings: Mapping[str, Mapping[str, Any]], enabled: Any, epoch: int = 0) -> dict[str, dict[str, Any]]:
    from verity_vllm.protocols.pouw import challenged

    return challenged(settings, enabled, epoch)


def h1_run(entry: str) -> Any:
    from verity_vllm.protocols.pouw import h1_run

    return h1_run(entry)


def pouw_refusals(scheme: str, ctx: Any) -> tuple[str, ...]:
    from verity_vllm.protocols.pouw import refusals

    return refusals(scheme, ctx)


def pearl_c_device_run() -> type:
    """The class."""
    from verity.protocols.accounting.pouw.prover.serving.pearl_c import PearlCDeviceRun

    return PearlCDeviceRun


def pearl_c_schemes() -> tuple[str, ...]:
    from verity.protocols.accounting.pouw.prover.serving.pearl_c import SCHEMES

    return SCHEMES
