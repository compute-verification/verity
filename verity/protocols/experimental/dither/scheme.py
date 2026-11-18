"""Dither as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``dither``.

Dither has no verifier yet, so its protocol id is a placeholder (``verity/pouw/dither/v0``). `PROTOCOL.md` credits an
output cell K − L MACs, its noised segments s < S; a block's cost is the multiply-adds it checks of a credited segment:
L for `PouwDitherBlock_v1`, and none for `PouwDitherLast_v1`, the last segment, which is checked through O and earns
nothing.
"""

from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.scheme import Block, Charge, Lacks, Order, Scheme, register

#: why dither fills no field a configuration, a launcher or a record reads
_NOT_RUN = Lacks("experimental: no application lists dither and it has no verifier")
#: `Pouw.Dither.Prover.epsWorkSound`: the error of `Pouw.Guarantees.Dither.WorkSound`, over every term of ε_skip
_EPS = "Pouw.Dither.Prover.epsWorkSound"


def attach(exchange: Any, ctx: Any) -> None:
    raise NotImplementedError("dither: its kernels (the noised segments' and the last segment's, `PouwDitherBlock_v1` "
                              "and `PouwDitherLast_v1`) are not built; kernels builds them")

DITHER = register(Scheme(
    name="dither",
    protocol="verity/pouw/dither/v0",
    resource="compute",
    blocks=(Block("PouwDitherBlock_v1", lambda shape: shape["L"]), Block("PouwDitherLast_v1", lambda shape: 0)),
    assumptions=tuple(f"Pouw.Assumptions.{a}" for a in (
        "SAKMatmul", "SAKSha", "OutsourceWithinEgress", "ProofSystemSound", "CertifierSeesImports",
        "ImportWithinCertifier", "OwnDither", "CoinsConfidential")),
    log=(Order("operand roots", "seeds"), Order("value roots", "check")),
    charges=(Charge("sample", _EPS), Charge("egress", _EPS), Charge("answers", _EPS)),
    attach=attach,
    **dict.fromkeys(("parties", "channels", "colocated", "needs", "depends", "record", "symbol", "published",
                     "channel_schemas", "record_owner", "tables", "output_format", "product_family", "y_committed",
                     "salt_open_root", "kernels"), _NOT_RUN),
))
