"""Compute accounting's credit rule for a served linear: a served call earns credit only if the y
the linear hands back to the serving engine is the credited kernel's own output, the same GEMM invocation that wrote the credited z, and
no other GEMM ran for that linear.  Pearl-C's e2e gate holds served y to its own ``run.gemm``
(`benchmarks/pouw/pearl_c_vllm/e2e.py`); this is the same rule for H1, read from the call's trace.  H1 adds the value
check of the z it commits (`pouw_h1_records.reason`, refused as ``credit: z-records: ...``): its z/y records'
y is dq of their z and the served y is theirs.

* **The trace** (`trace`, the workload's: `tensors.H1Device.trace`; the device adapter's is a dispatch mode) around
  one linear call (its ``apply``).  Every op the device library's dispatcher sees is
  an invocation of the call's trace, in order; a custom op registered with a Python implementation runs that
  implementation inside the mode, so the ops it runs are seen too, attributed to it.  Each invocation records the tensors it
  wrote (`_written`: its fresh results and the arguments its schema marks written; a view writes nothing) by storage, and
  the GEMMs inside it (`GEMMS`: ``aten::_int_mm``, ``aten::mm``, ``aten::_scaled_mm`` and the rest of the matmul family; an
  opaque op, one not in aten and with no Python implementation, counts as one GEMM unless `NOT_GEMMS` names it).
* **The verdict** (`verdict`): a tensor's producer is the last invocation that wrote its storage.  Credit iff y's
  producer is z's and the call's trace holds exactly one GEMM, inside that invocation.  Otherwise the verdict names why:

      credit: served-output-not-credited-kernel: <linear> y from <op>, z from <op>
      credit: gemm-count: <linear> <n> GEMMs (<op>, <op> in <op>, ...); only the credited kernel's is allowed

  ``<untraced>`` is a tensor no traced op wrote, ``<none>`` a z the path never computed.
* **What it sees.** Only ops through the device library's dispatcher.  A kernel launched outside it (a cubin through the driver) writes
  nothing the trace sees, so its output is ``<untraced>`` and earns nothing: a fused kernel is credited once it is a
  dispatcher op (a registered custom op) returning its z and its y.  Ops run `outside` the trace are not seen either; that is
  for work beside the call that writes neither its y nor its z and runs no GEMM, so its verdict is the same without it.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import h1_device

#: the matmul family: a call of any of these is a GEMM
GEMMS = frozenset({"aten::_int_mm", "aten::mm", "aten::addmm", "aten::addmm_", "aten::_addmm_activation", "aten::bmm",
                   "aten::baddbmm", "aten::addbmm", "aten::_scaled_mm", "aten::_scaled_grouped_mm", "aten::_grouped_mm",
                   "aten::matmul", "aten::linear", "aten::mv", "aten::addmv", "aten::dot", "aten::vdot", "aten::einsum",
                   "aten::tensordot", "aten::_weight_int8pack_mm", "aten::_weight_int4pack_mm"})
#: the namespaces whose ops are known: one of these is a GEMM iff `GEMMS` lists it
KNOWN = ("aten", "prims")
#: opaque ops (not in `KNOWN`, no Python implementation) that are not GEMMs; any other opaque op counts as one
NOT_GEMMS: frozenset[str] = frozenset()
UNTRACED, NONE = "<untraced>", "<none>"
NOT_CREDITED = "served-output-not-credited-kernel"
GEMM_COUNT = "gemm-count"
#: H1's refusal of a call whose z isn't its committed z/y records or whose records aren't its served product's
RECORDS = "z-records"


@dataclass
class Invocation:
    """One op the linear's call ran at the top of its trace: its index, its name, and the GEMMs inside it (itself included)."""

    index: int
    name: str
    gemms: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        return [g if g == self.name else f"{g} in {self.name}" for g in self.gemms]


def trace(linear: str) -> AbstractContextManager[Any]:
    """One linear call's ops, as `Invocation`s, and which invocation last wrote each buffer: ``with trace(linear) as t:``
    around the call, ``t.producer(y)`` the invocation that wrote y last (`tensors.H1Device.trace`)."""
    return h1_device().trace(linear)


def outside() -> AbstractContextManager[None]:
    """The ops inside run past the innermost trace, which doesn't see them (`tensors.H1Device.outside`). A trace is only ever
    the workload's, so where none handed its H1 device in no trace is open and they run as they would."""
    try:
        device = h1_device()
    except NotImplementedError:
        return nullcontext()
    return device.outside()


def verdict(trace: Any, y: Any, z: Any) -> dict[str, Any]:
    """The credit rule on one call: y, the tensor the linear handed back to the serving engine, and z, the credited kernel's output (None if
    the path computed none)."""
    yp, zp = trace.producer(y), trace.producer(z) if z is not None else None
    y_from = yp.name if yp else UNTRACED
    z_from = zp.name if zp else (UNTRACED if z is not None else NONE)
    gemms = [g for inv in trace.invocations for g in inv.describe()]
    out = {"linear": trace.linear, "y_from": y_from, "z_from": z_from, "gemms": gemms, "credit": False}
    if yp is None or yp is not zp:
        out["reason"] = f"credit: {NOT_CREDITED}: {trace.linear} y from {y_from}, z from {z_from}"
    elif len(gemms) != 1 or len(yp.gemms) != 1:
        out["reason"] = (f"credit: {GEMM_COUNT}: {trace.linear} {len(gemms)} GEMMs ({', '.join(gemms) or 'none'}); only the "
                         "credited kernel's is allowed")
    else:
        out.update(credit=True, kernel=yp.name)
    return out


def reasons(verdicts: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Each refusal's reason and how many calls it refused."""
    return dict(Counter(v["reason"] for v in verdicts if not v["credit"]))


__all__ = ["GEMMS", "GEMM_COUNT", "Invocation", "KNOWN", "NONE", "NOT_CREDITED", "NOT_GEMMS", "RECORDS", "UNTRACED", "outside",
           "reasons", "trace", "verdict"]
