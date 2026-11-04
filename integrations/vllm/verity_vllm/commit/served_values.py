"""A window's served rows committed by the ZK server's value service (`values`, under ``FC_FIREWALL_VALUES``): the rows
`served_rows.register` commits under the window's run root, a row each window of P steps, in the same domain and schema, in one
session of the window's, registered at its close. The salts are the server's alone: `ServedValues` is `served_rows.Served`'s
record and ``bc`` beside the service's table and the session's registration."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import served_rows as SRW
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.rowleaf import bit_row
from verity_vllm.commit import values as V


@dataclass(frozen=True)
class ServedValues:
    """The window's one table, as the service committed it (its root the record's), and the session's registration."""

    table: V.Committed
    registration: V.Registration
    binding: bytes
    n_bits: int
    G: int

    def record(self) -> dict[str, Any]:
        return {"schema": SRW.SCHEMA, "binding": self.binding.hex(), "root": self.table.root.hex(), "count": self.table.n,
                "n_bits": self.n_bits, "G": self.G,
                "values": {"table": self.table.ref().hex(), "registration": self.registration.entry()}}

    def bc(self, i: int) -> bytes:
        return self.table.bcs[i]

    def held(self) -> dict[str, Any]:
        """What the prover keeps in place of salts: the table's ref and every row's ``b ‖ c``."""
        return {"values": {"table": self.table.ref().hex(), "bcs": [bc.hex() for bc in self.table.bcs]}}


def commit(S: Any, session: Sequence, run_root: bytes, service: str) -> ServedValues:
    """The session's windows (`served_rows.windows`' form) committed by the value service at ``service``, then registered."""
    rows = [bit_row((CS.words(SRW.block_bits(S, w, S.P, 0)), CS.WORD)) for w in session]
    n_bits = rows[0][1]
    binding, digest = SRW.binding(run_root), BitRowLeaf(n_bits).layout().digest
    table = V.Table(ADV.domain(binding, len(rows)).domain_id, BitRowLeaf(n_bits).hiding_schema, b"".join(digest(r) for r, _n in rows))
    s = V.Session(service)
    (got,) = s.commit([table])
    return ServedValues(got, s.register(), binding, n_bits, S.P)
