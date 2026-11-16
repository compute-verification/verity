"""PoUW's serving side: what runs beside a served model to commit its matrix multiplies, one directory a scheme (`h1`,
`lead`, `pearl_c`, and `ncp` for ``ncp-v1`` and ``ncp-v2``) and the calls' operands, devices and circuits beside them.

A workload reaches it only through the names this module exports. Per call there are two:

* `run_matmul(run, site, x, weight, bias)`: run this matrix multiply under PoUW, y = x weightᵀ (+ bias), served and
  committed by ``run`` (a `MatmulRun`): ``ncp-v1`` (`NcpRun`), ``ncp-v2`` (`NcpCircuitRun`), the lead (`LeadRun`, whose
  ``x`` is the arm's `QuantizedRows` and ``weight`` the linear its backend loaded), or a workload's own (Pearl-C's run on
  the device is the workload's).
* `observe_matmul(run, site, x, weight, y, ...)`: a matrix multiply the workload served itself, recorded by ``run`` (a
  `MatmulObserver`), which computes nothing: H1 (`H1Run`), whose ``x`` is the int8 arm's s8 rows. The workload hands
  H1 its arm's GEMM and dequantization first (`use_int8_arm`, `int8_arm.Int8Arm`).

``site`` is the linear's ordinal in the forward pass (its weight id), the same at every forward. The buffers are the
caller's own, on its device, and this package names no tensor library: the workload hands in how its buffers are read,
written and allocated (`use_tensors`, `tensors.Tensors`) and H1's device half (`use_h1_device`, `tensors.H1Device`);
without them the host paths run. Every other name loads its module when first read. A run closes with ``finish()``; its ``commits`` and ``details`` are what it publishes.

To manage a window: `in_process_coins` (coins held in this process, which credit nothing), `reserve_salt` and `gate_of`
(a salt session at infra's gate), `PearlCKernel` (Pearl-C's kernel, loaded), `ready_verifier` (the Lean verifier's
executable, built), and `served_passes`, `coins_of` and `SERVED_SCHEMA` (what a served Pearl-C window records). A
workload's own GPU commitments build on the row-table kernels: `row_table_host_lib`, `row_table_nvcc`, `hm_tables`,
`leaf_prefixes` and the other ``row_table``/``ROW_TABLE`` names. The rest of `__all__` names the runs, their schemes and
settings."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

_POUW = "verity.protocols.resource_accounting.compute.pouw"
_PROVER = f"{_POUW}.prover"
_SERVING = f"{_PROVER}.serving"
#: each exported name's module and attribute, read from it at every lookup (never cached here), so an import inside a
#: function sees the module's current attribute
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
    "use_tensors": (f"{_SERVING}.tensors", "use"),
    "use_h1_device": (f"{_SERVING}.tensors", "use_h1"),
    "H1Run": (f"{_SERVING}.h1", "H1Run"),
    "LeadRun": (f"{_SERVING}.lead", "LeadRun"),
    "PearlCKernel": (f"{_SERVING}.pearl_c.kernel", "Kernel"),
    "PearlC": (f"{_PROVER}.schemes.pearl_c", "PearlC"),
    "H100": (f"{_PROVER}.schemes.pearl_c_device", "H100"),
    "reserve_salt": (f"{_SERVING}.h1.window", "reserve_salt"),
    "gate_of": (f"{_SERVING}.h1.window", "gate_of"),
    "served_passes": (f"{_SERVING}.pearl_c.window", "served_passes"),
    "coins_of": (f"{_SERVING}.pearl_c.window", "coins_of"),
    "SERVED_SCHEMA": (f"{_SERVING}.pearl_c.window", "SERVED_SCHEMA"),
    "ADMISSIONS": (f"{_SERVING}.h1.requests", "ADMISSIONS"),
    "COIN_SAMPLE": (f"{_POUW}.verifier.window", "SAMPLE"),
    "COIN_ROUND": (f"{_POUW}.verifier.window", "ROUND"),
    "Challenger": (f"{_POUW}.verifier.coins", "Challenger"),
    "CoinCommitment": (f"{_POUW}.verifier.coins", "CoinCommitment"),
    "CoinWindow": (f"{_SERVING}.h1.window", "Window"),
    "CommitTap": (f"{_SERVING}.h1.commit", "CommitTap"),
    "CreditTrace": (f"{_SERVING}.credit_gate", "trace"),
    "DomainError": (f"{_POUW}.verifier.protocol", "DomainError"),
    "Epoch": (f"{_POUW}.verifier.audit", "Epoch"),
    "H1Call": (f"{_SERVING}.h1", "Call"),
    "H1_CHALLENGER": (f"{_SERVING}.h1", "CHALLENGER"),
    "H1_SCHEME": (f"{_SERVING}.h1", "SCHEME"),
    "H1_TILE": (f"{_SERVING}.h1", "TILE"),
    "H1_WEIGHT_KEY": (f"{_SERVING}.h1", "WEIGHT_KEY"),
    "HM96": (f"{_POUW}.verifier.audit", "HM96"),
    "HM96_WEIGHTS": (f"{_POUW}.verifier.audit", "HM96_WEIGHTS"),
    "LEAD_BACKEND": (f"{_SERVING}.lead", "BACKEND"),
    "LEAD_SCHEME": (f"{_SERVING}.lead", "SCHEME"),
    "Matmul": (f"{_POUW}.verifier.protocol", "Matmul"),
    "NCP_CIRCUIT_SCHEMES": (f"{_PROVER}.circuit", "SCHEMES"),
    "NCP_SCHEMES": (f"{_PROVER}.schemes", "SCHEMES"),
    "NO_SAMPLE_SESSION": (f"{_SERVING}.h1.window", "NO_SAMPLE_SESSION"),
    "PEARL_C_CALL_CAPACITY": (f"{_SERVING}.pearl_c", "CALL_CAPACITY"),
    "PEARL_C_SCHEMES": (f"{_SERVING}.pearl_c", "SCHEMES"),
    "ROUND0": (f"{_SERVING}.h1.requests", "ROUND0"),
    "SCHEDULES": (f"{_POUW}.verifier.serving", "SCHEDULES"),
    "Served": (f"{_SERVING}.h1.served", "Served"),
    "Shape": (f"{_POUW}.verifier.protocol", "Shape"),
    "Workload": (f"{_POUW}.verifier.audit", "Workload"),
    "challenger_address": (f"{_SERVING}.h1.challenger", "address"),
    "closed_on_error": (f"{_SERVING}.h1.window", "closed_on_error"),
    "commit_committed": (f"{_SERVING}.h1.commit", "committed"),
    "commit_rows": (f"{_POUW}.verifier.audit", "commit_rows"),
    "commit_site": (f"{_SERVING}.h1.commit", "site"),
    "sample_challenger": (f"{_POUW}.verifier.window", "sample_challenger"),
    "finish_window": (f"{_SERVING}.h1.window", "finish"),
    "h1_weight_root": (f"{_SERVING}.h1", "weight_root"),
    "in_process_sample": (f"{_SERVING}.h1.window", "in_process_sample"),
    "lead_refusals": (f"{_SERVING}.lead", "refusals"),
    "live_challenger": (f"{_POUW}.verifier.window", "live_challenger"),
    "live_epoch_reason": (f"{_POUW}.verifier.window", "live_epoch_reason"),
    "ncp_scheme": (f"{_PROVER}.schemes", "get"),
    "numpy_of": (f"{_SERVING}.h1", "numpy_of"),
    "pearl_c_refusals": (f"{_SERVING}.pearl_c", "refusals"),
    "pearl_c_salt_key": (f"{_SERVING}.pearl_c", "new_salt_key"),
    "pearl_c_schedule": (f"{_SERVING}.pearl_c", "schedule_of"),
    "pearl_c_scheme": (f"{_SERVING}.pearl_c", "scheme_of"),
    "period_reason": (f"{_SERVING}.h1.window", "period_reason"),
    "period_setting": (f"{_SERVING}.h1.window", "period_setting"),
    "program_salt": (f"{_SERVING}.h1", "program_salt"),
    "reserved_of": (f"{_SERVING}.h1.window", "reserved_of"),
    "salt_session": (f"{_SERVING}.h1.window", "salt_session"),
    "session_setting": (f"{_SERVING}.h1.window", "session_setting"),
    "traced_as": (f"{_PROVER}.circuit", "traced_as"),
    "weight_binding": (f"{_POUW}.verifier.audit", "weight_binding"),
    "weights_root": (f"{_POUW}.verifier.audit", "weights_root"),
    "window_refusals": (f"{_SERVING}.h1.window", "refusals"),
    "zy_host": (f"{_SERVING}.h1.records", "host"),
    "CommitRecord": (f"{_SERVING}.h1.commit", "CommitRecord"),
    "GateSessions": (f"{_SERVING}.h1.serve_gate", "GateSessions"),
    "H1_COMMIT_SCHEME": (f"{_SERVING}.h1.commit", "SCHEME"),
    "H1_PROGRAM": (f"{_SERVING}.h1", "PROGRAM"),
    "H1_PROGRAM_META": (f"{_SERVING}.h1", "PROGRAM_META"),
    "LISTEN_S": (f"{_SERVING}.h1.serve_gate", "LISTEN_S"),
    "MARGIN": (f"{_SERVING}.h1.serve_gate", "MARGIN"),
    "PEARL_C_ROW_ALIGN": (f"{_SERVING}.pearl_c", "ROW_ALIGN"),
    "POLLS": (f"{_SERVING}.h1.serve_gate", "POLLS"),
    "REPLY_FIELDS": (f"{_SERVING}.h1.challenger", "REPLY_FIELDS"),
    "SESSION_FIELDS": (f"{_SERVING}.h1.challenger", "SESSION_FIELDS"),
    "Sites": (f"{_SERVING}.h1.commit", "Sites"),
    "Writer": (f"{_SERVING}.h1.prover", "Writer"),
    "a_value": (f"{_SERVING}.pearl_c.window", "a_value"),
    "b_value": (f"{_SERVING}.pearl_c.window", "b_value"),
    "challenger_carried": (f"{_SERVING}.h1.challenger", "carried"),
    "commit_registration": (f"{_SERVING}.h1.commit", "registration"),
    "committed_k": (f"{_POUW}.verifier.credit", "committed_k"),
    "connect_window": (f"{_SERVING}.h1.window", "connect"),
    "declared_layout": (f"{_SERVING}.h1.served", "declared_layout"),
    "descriptor": (f"{_SERVING}.h1.served", "descriptor"),
    "sample_k": (f"{_SERVING}.h1.challenger", "sample_k"),
    "sampled_reads": (f"{_SERVING}.h1.serve_sample", "sampled_reads"),
    "sampled_blocks": (f"{_SERVING}.h1.serve_sample", "sampled_blocks"),
    "dump_live": (f"{_SERVING}.h1.live", "dump"),
    "h1_units": (f"{_SERVING}.h1", "units"),
    "kept_bytes": (f"{_SERVING}.h1.prover", "kept_bytes"),
    "layout_of": (f"{_POUW}.verifier.credit", "layout_of"),
    "listening": (f"{_SERVING}.h1.serve_gate", "listening"),
    "lms_verify": (f"{_SERVING}.h1.challenger", "lms_verify"),
    "pins_head": (f"{_SERVING}.h1.challenger", "pins_head"),
    "prover_private": (f"{_SERVING}.h1.prover", "private"),
    "register_weights": (f"{_SERVING}.h1.served", "register_weights"),
    "registered_commit": (f"{_SERVING}.h1.registered", "commit"),
    "registered_salts": (f"{_SERVING}.h1.registered", "salts"),
    "served_declared": (f"{_SERVING}.h1.served", "declared"),
    "session_of": (f"{_SERVING}.h1.challenger", "session_of"),
    "unit_of": (f"{_SERVING}.h1.served", "unit_of"),
    "weights_of": (f"{_SERVING}.h1.served", "weights_of"),
    "window_units": (f"{_SERVING}.h1.serve_sample", "units"),
    "write_program": (f"{_SERVING}.h1.served", "write_program"),
    "y_value": (f"{_SERVING}.pearl_c.window", "y_value"),
    "RowTableArgs": (f"{_SERVING}.row_table_kernels", "Args"),
    "ROW_TABLE_CUDA_DIR": (f"{_SERVING}.row_table_kernels", "CUDA_DIR"),
    "ROW_TABLE_SEG_BYTES": (f"{_SERVING}.row_table_kernels", "SEG_BYTES"),
    "ROW_TABLE_TAB_WORDS": (f"{_SERVING}.row_table_kernels", "TAB_WORDS"),
    "row_table_host_lib": (f"{_SERVING}.row_table_kernels", "host_lib"),
    "row_table_nvcc": (f"{_SERVING}.row_table_kernels", "nvcc"),
    "hm_tables": (f"{_SERVING}.row_table_kernels", "hm_tables"),
    "leaf_prefixes": (f"{_SERVING}.row_table_kernels", "leaf_prefixes"),
    "table_width": (f"{_SERVING}.row_table_kernels", "table_width"),
    "u64_words": (f"{_SERVING}.row_table_kernels", "u64_words"),
    "to_host": (f"{_SERVING}.h1.commit", "to_host"),
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
    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier
    from verity.protocols.resource_accounting.compute.pouw.verifier.coins import Challenger

    return Challenger(window, verifier.coins)


def verifier_coins(n: int) -> bytes:
    """``n`` bytes of the Lean verifier's entropy (`verifier.coins`), the in-process challengers' coins."""
    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier

    return verifier.coins(n)


def lead_in_process_certifier() -> Any:
    """The lead's coin link held in this process (`lead.feed.InProcessCertifier`), whose coins credit nothing."""
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.lead import feed

    return feed.InProcessCertifier()


def ready_verifier() -> Any:
    """The Lean verifier's executable, built if it isn't (`verifier.executable`)."""
    from verity.protocols.resource_accounting.compute.pouw.verifier import verifier

    return verifier.executable()


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    return getattr(importlib.import_module(module), attr)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["MatmulObserver", "MatmulRun", "QuantizedRows", "in_process_coins", "observe_matmul", "ready_verifier",
           "run_matmul", "verifier_coins", "lead_in_process_certifier", *_LAZY]
