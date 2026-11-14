"""What a window's span hands the verifier in place of its openings: the proof system's accepted runs of PoUB's opening unit
(`BlockOpeningCheck_v2`, one unit a draw), which the Lean decides (`RemoteAudit.Span.checkSpan`, through
`Session.verifySpan`). This module decides nothing: it computes no verdict and holds no row to a commitment.

A span is one or more statements of the sampled proof system's one-stage audit (`verity/sampled-proofs/one-stage/v2`),
every unit proved: each its public file's SHA-512, its registration record and evidence, and per instance, in the
public file's order, its verdict word and the salts of its ``pub``, ``signs`` and ``o0`` rows (`statement`); with the
blobs its evidence names (`span`). The statements' instances in order are the window's draws in order, one each. The
auditor's own side is the audit's parameters and the registered opening circuit's ports in order, each with its row's
bits (`auditor`, `layout`).

`public_rows` is the unit's two public rows for a draw as the verifier derives them (`block_check.opening_public_v2`'s,
for a registered block of a pass whose plan is the identity), which the prover commits and the Lean derives again for
itself (`RemoteAudit.Span.publicWords`, `signWords`); the difftest holds the two to each other."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.commitments.merkle import merkle_depth
from verity.protocols.accounting.poub.verifier.commitments.gates import (
    block_check as BC,
)

from . import audit, receipts

ROWS = receipts.SKETCH_ROWS
#: the opening unit's rows the verifier opens: its public rows and its verdict
OPENED = ("pub", "signs", "o0")


def public_rows(d: audit.PositionSample, codec_seed: bytes, S) -> dict[str, list[int]]:
    """Draw ``d``'s ``pub`` and ``signs`` rows as u16 words at the unit's statics ``S``: block j = p of a pass whose plan
    is the identity, registered (``coin`` None)."""
    if d.coin is not None or d.block != d.position:
        raise ValueError("a span's unit opens a registered block at its own position: the pass's plan is the identity")
    rdom = receipts.domain(d.seed, d.plan)
    first = d.domain.positions.start
    a = BC.opening_public_v2(S, codec_seed, d.block, None, receipts.sketch_prefix(d.seed), receipts.sign_bits(d.seed, ROWS),
                             (d.root, rdom.domain_id, merkle_depth(rdom.count)),
                             (d.commitment.root, d.domain.domain_id, d.block - first, merkle_depth(d.domain.count)), None)
    pub = BC.opening_rows_v2(S)[0][2]
    return {"pub": [w for p in pub for w in a[p]], "signs": a["signs"]}


def layout(S, verdict_bits: int) -> list[list]:
    """The registered opening circuit's ports in order, ``[name, bits]``: the unit's input rows (`BC.opening_rows_v2`),
    then its one output row, ``o0``, of ``verdict_bits`` bits as the circuit commits it."""
    return [[name, bits] for name, bits, _ in BC.opening_rows_v2(S)] + [["o0", verdict_bits]]


def auditor(params: dict, ports: list[list]) -> dict:
    """The verifier's own side: the one-stage audit's parameters (``proofs.public`` is each statement's) and the layout."""
    return {"layout": ports, "params": params}


def instance(verdict: int, salts: dict[str, bytes]) -> dict:
    """One unit: its verdict word and its opened rows' salts."""
    return {"pub_salt": salts["pub"].hex(), "signs_salt": salts["signs"].hex(), "verdict": verdict,
            "verdict_salt": salts["o0"].hex()}


def statement(public: bytes, record: bytes, evidence: bytes, instances: Sequence[dict]) -> dict:
    """One statement: its public file's SHA-512, its registration record's and evidence's bytes, its instances in order."""
    return {"evidence": evidence.hex(), "instances": list(instances), "pub": hashlib.sha512(public).hexdigest(),
            "record": record.hex()}


def span(statements: Sequence[dict], blobs: Sequence[bytes]) -> dict:
    """The span: its statements in draw order and the blobs their evidence names (their public files among them)."""
    return {"blobs": [b.hex() for b in blobs], "statements": list(statements)}


__all__ = ["OPENED", "auditor", "instance", "layout", "public_rows", "span", "statement"]
