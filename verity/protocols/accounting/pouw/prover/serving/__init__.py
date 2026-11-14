"""PoUW's serving side: what runs beside a served model to commit its matrix multiplies, one directory a scheme (`h1`,
`lead`, `pearl_c`, and `ncp` for ``ncp-v1`` and ``ncp-v2``) and the calls' operands, devices and circuits beside them.

A workload reaches it only through the names this module exports. Per call there are two:

* `run_matmul(run, site, x, weight, bias)`: run this matrix multiply under PoUW, y = x weightᵀ (+ bias), served and
  committed by ``run`` (a `MatmulRun`): ``ncp-v1`` (`NcpRun`), ``ncp-v2`` (`NcpCircuitRun`), Pearl-C (`PearlCDeviceRun`)
  and the lead (`LeadRun`, whose ``x`` is the arm's `QuantizedRows` and ``weight`` the linear its backend loaded).
* `observe_matmul(run, site, x, weight, y, ...)`: a matrix multiply the workload served itself, recorded by ``run`` (a
  `MatmulObserver`), which computes nothing: H1 (`H1Run`), whose ``x`` is the int8 arm's s8 rows. The workload hands
  H1 its arm's GEMM and dequantization first (`use_int8_arm`, `int8_arm.Int8Arm`).

``site`` is the linear's ordinal in the forward pass (its weight id), the same at every forward. The buffers are the
caller's own (torch tensors on its device), so this module loads no torch; every other name loads its module when first
read. A run closes with ``finish()``; its ``commits`` and ``details`` are what it publishes.

To manage a window: `in_process_coins` (coins held in this process, which credit nothing), `reserve_salt` and `gate_of`
(a salt session at infra's gate), `PearlCKernel` (Pearl-C's kernel, loaded), `ready_verifier` (the Lean verifier's
executable, built), and `served_passes`, `coins_of` and `SERVED_SCHEMA` (what a served Pearl-C window records). The rest
of `__all__` names the runs, their schemes and settings."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

_PROVER = "verity.protocols.accounting.pouw.prover"
_SERVING = f"{_PROVER}.serving"
#: each exported name's module and attribute, loaded when first read
_LAZY = {
    "NcpRun": (f"{_SERVING}.ncp", "NcpRun"),
    "NcpCircuitRun": (f"{_SERVING}.ncp", "NcpCircuitRun"),
    "NcpSettings": (f"{_SERVING}.ncp", "NcpSettings"),
    "LayerRows": (f"{_SERVING}.ncp", "LayerRows"),
    "weight_operands": (f"{_SERVING}.ncp", "weight_operands"),
    "QUANTIZER": (f"{_SERVING}.ncp", "QUANTIZER"),
    "quantize": (f"{_SERVING}.operands", "quantize"),
    "host_array": (f"{_SERVING}.ncp", "host"),
    "use_int8_arm": (f"{_SERVING}.int8_arm", "use"),
    "H1Run": (f"{_SERVING}.h1", "H1Run"),
    "LeadRun": (f"{_SERVING}.lead", "LeadRun"),
    "PearlCDeviceRun": (f"{_SERVING}.pearl_c", "PearlCDeviceRun"),
    "PearlCKernel": (f"{_SERVING}.pearl_c.kernel", "Kernel"),
    "PearlC": (f"{_PROVER}.schemes.pearl_c", "PearlC"),
    "H100": (f"{_PROVER}.schemes.pearl_c_device", "H100"),
    "reserve_salt": (f"{_SERVING}.h1.window", "reserve_salt"),
    "gate_of": (f"{_SERVING}.h1.window", "gate_of"),
    "served_passes": (f"{_SERVING}.pearl_c.window", "served_passes"),
    "coins_of": (f"{_SERVING}.pearl_c.window", "coins_of"),
    "SERVED_SCHEMA": (f"{_SERVING}.pearl_c.window", "SERVED_SCHEMA"),
}


class MatmulRun(Protocol):
    """One install's PoUW run of a scheme that computes y: ``gemm(site, x, weight, bias)`` serves y = x weightᵀ (+ bias)
    of the linear at ``site`` and commits the call as its scheme does."""

    def gemm(self, site: int, x: Any, weight: Any, bias: Any, /) -> Any: ...


class MatmulObserver(Protocol):
    """One install's PoUW run of a scheme that computes nothing: ``observe(site, x, weight, y, ...)`` records the call the
    workload served."""

    def observe(self, site: int, x: Any, weight: Any, y: Any, /, *, x_scale: Any = None, weight_scale: Any = None,
                z: Any = None) -> None: ...


@dataclass(frozen=True)
class QuantizedRows:
    """A call's rows as its arm quantized them: ``rows`` [m, K] s8 and their f32 ``scales`` [m]."""

    rows: Any
    scales: Any


def run_matmul(run: MatmulRun, site: int, x: Any, weight: Any, bias: Any = None) -> Any:
    """Run this matrix multiply under PoUW: y = x weightᵀ (+ ``bias``) of the linear at ``site`` (its ordinal in the
    forward pass), on the caller's GPU buffers, served and committed by ``run``; y is a new buffer on x's device."""
    return run.gemm(site, x, weight, bias)


def observe_matmul(run: MatmulObserver, site: int, x: Any, weight: Any, y: Any, *, x_scale: Any = None,
                   weight_scale: Any = None, z: Any = None) -> None:
    """Record a matrix multiply the workload served itself, y = x weightᵀ of the linear at ``site``, computing nothing:
    under H1, ``x`` is the arm's s8 rows with their ``x_scale``, ``weight`` its int8 weight with its ``weight_scale``, and
    ``z`` the z/y records its kernel wrote, if it wrote any."""
    run.observe(site, x, weight, y, x_scale=x_scale, weight_scale=weight_scale, z=z)


def in_process_coins(window: int) -> Any:
    """Window ``window``'s challenger held in this process, its coins the Lean verifier's entropy: a run on them credits
    nothing."""
    from verity.protocols.accounting.pouw.verifier import verifier
    from verity.protocols.accounting.pouw.verifier.coins import Challenger

    return Challenger(window, verifier.coins)


def ready_verifier() -> Any:
    """The Lean verifier's executable, built if it isn't (`verifier.executable`)."""
    from verity.protocols.accounting.pouw.verifier import verifier

    return verifier.executable()


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    value = globals()[name] = getattr(importlib.import_module(module), attr)
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["MatmulObserver", "MatmulRun", "QuantizedRows", "in_process_coins", "observe_matmul", "ready_verifier",
           "run_matmul", *_LAZY]
