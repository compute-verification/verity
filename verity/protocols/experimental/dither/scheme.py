"""Dither as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``dither``.

Dither has no verifier yet, so its protocol id is a placeholder (``verity/pouw/dither/v0``). `PROTOCOL.md` credits an
output cell K − L MACs, its noised segments s < S; a block's cost is the multiply-adds it checks of a credited segment:
L for `PouwDitherBlock_v1`, and none for `PouwDitherLast_v1`, the last segment, which is checked through O and earns
nothing.
"""

from __future__ import annotations

from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Scheme, register

#: `Pouw.Dither.Prover.epsWorkSound`: the error of `Pouw.Guarantees.Dither.WorkSound`, over every term of ε_skip
_EPS = "Pouw.Dither.Prover.epsWorkSound"

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
))
