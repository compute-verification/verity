"""`pouw_h1.H1Run`'s call tables: a call's A table and z table, each `pouw_h1_requests.hiding_rows` of its leaves (the
reference, `commit_python`), committed by `commit_tables` from the host's arrays here and from the workload's buffers by its
H1 device (`tensors.H1Device.commit_tables`, the row-table kernels on the GPU), whose roots `settle` records as they finish."""
from __future__ import annotations

from typing import Any

import numpy as np
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import tagged

from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import prover as PV
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import records as ZR
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import requests as RQ
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1.requests import TABLE_NONCES, TABLE_TAGS, table_leaves
from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import dtype_name, h1_device


def _host(x: Any) -> bool:
    return x is None or isinstance(x, np.ndarray)


def row_words(t: Any, m: int, dtype: str) -> np.ndarray:
    """`t`'s m rows as little-endian `dtype` words: an array's here, a buffer's by the workload's H1 device (a bf16 one by
    its bits, `tensors.H1Device.row_words`)."""
    if not (_host(t) or isinstance(t, (list, tuple))):
        return h1_device().row_words(t, m, dtype)
    return np.ascontiguousarray(np.asarray(t).reshape(m, -1), dtype=dtype)


def flush(run: Any) -> None:
    """Hash the z table still waiting for its call's registration on the device (`tensors.H1Device.flush`), if one is."""
    if getattr(run, "_zy", None) is not None:
        h1_device().flush(run)


def take_z(run: Any, call: Any, words: Any) -> tuple[bytes, int] | None:
    """The call's z table for its registered ``zy@<c>`` to commit beside it, when the device holds one back for it
    (`tensors.H1Device.take_z`); else None."""
    if getattr(run, "_zy", None) is None:
        return None
    return h1_device().take_z(run, call, words)


def settle(run: Any, wait: bool) -> None:
    """Record the roots of ``run``'s calls whose tables the GPU has hashed, in call order: all of them with ``wait`` (a z
    table still waiting for its registration, `flush`ed first), else the leading ones already done."""
    if wait:
        flush(run)
    while run._hashing and (wait or run._hashing[0][1].ready()):
        index, pending = run._hashing.popleft()
        run._roots[index].update(zip(pending.names, (r.hex() for r in pending.roots())))


def _binding(run: Any, call: Any, name: str) -> tuple[bytes, int]:
    """The call's `name` table's domain binding and salt nonce."""
    return tagged(TABLE_TAGS[name], run.salt, call.index), RQ.TABLES_NONCE + 3 * call.index + TABLE_NONCES[name]


def _table(run: Any, call: Any, name: str, leaves: list[bytes]) -> str:
    binding, nonce = _binding(run, call, name)
    return RQ.hiding_rows(binding, call.index, leaves, run._key, nonce).commitment.root.hex()


def _keep(run: Any, call: Any, a: Any, s_a: Any, records: Any) -> None:
    """The served call's prover state (`pouw_h1_prover.keep_call`): by the `pouw_h1_prover.Writer` its directory is open
    on, off this thread, when its rows, scales and records are the int8, f32 and uint8 the file keeps; else here."""
    writer = PV.writer_of(run.prover)
    if writer is not None and dtype_name(a) == "int8" and dtype_name(s_a) == "float32" and \
            (records is None or dtype_name(records) == "uint8"):
        writer.keep(run.prover, call.index, a.reshape(call.m, -1), s_a, records)
        return
    PV.keep_call(run.prover, call.index, row_words(a, call.m, "i1"), row_words(s_a, call.m, "<f4"), ZR.host(records))


def commit_python(run: Any, call: Any, a: Any, s_a: Any, records: Any) -> None:
    """`commit_tables` by `hiding_rows`, the reference, on the host."""
    m, t = call.m, run.rows
    if a is not None and m:
        av, sv = row_words(a, m, "i1"), row_words(s_a, m, "<f4")
    else:
        av = sv = np.zeros((0, 1), "i1")
    out: dict[str, str | None] = {"a": _table(run, call, "a", table_leaves(t, call.k + 4, m, av, sv))}
    held = ZR.host(records)
    leaves = [bytes(ZR.ZY_BYTES)] * call.units if not m else None if held is None else [r.tobytes() for r in held]
    out["z"] = None if leaves is None else _table(run, call, "z", leaves)
    run._roots[call.index] = out
    if run.prover is not None and m:
        _keep(run, call, av, sv, held)


def commit_tables(run: Any, call: Any, a: Any, s_a: Any, records: np.ndarray | None, zy: Any = None) -> None:
    """``run``'s (a `pouw_h1.H1Run`'s) commit of the call's A table (`table_leaves`) and its z table, leaf u proof block u's z/y
    record (`pouw_h1_records`; an empty call's all zero), none for a served call whose arm handed over no records: by
    `commit_python` when they are the host's arrays, else by the workload's H1 device (`tensors.H1Device.commit_tables`),
    each root `hiding_rows`'."""
    if all(_host(x) for x in (a, s_a, zy)):
        commit_python(run, call, a, s_a, records)
    else:
        h1_device().commit_tables(run, call, a, s_a, records, zy)


__all__ = ["commit_python", "commit_tables", "flush", "row_words", "settle", "take_z"]
