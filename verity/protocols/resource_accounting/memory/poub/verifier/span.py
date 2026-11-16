"""What a window's span hands the verifier in place of its openings: the proof system's accepted runs of PoUB's opening unit
(`BlockOpeningUnit_v2`, one unit a sample), which the Lean decides (`Poub.Span.checkSpan`, through
`Session.verifySpan`). This module decides nothing: it computes no verdict and holds no row to a commitment.

A span is one or more statements of the sampled proof system's one-stage audit (`verity/sampled-proofs/one-stage/v2`),
every unit proved: each its public file's SHA-512, its docket and evidence, and per instance, in the
public file's order, its verdict word and the salt of its ``o0`` row (`statement`); with the blobs its evidence names
(`span`). The statements' units in order are the population's 0, 1, …, the window's samples in order, one each. The
auditor's own side is the audit's parameters and the registered opening circuit's ports in order, each with its row's
bits (`auditor`, `layout`).

`public_rows` is the unit's two public rows for a sample as the verifier derives them (`block_check.opening_public_v2`'s,
for a registered block of a pass whose plan is the identity), and `supplied` the window's, as the verifier gives them
the audit as its supplied values ``pub`` and ``signs`` (R7, `Spot.Supplied`) and the prover's window commits them
(`Window.give`). The Lean derives both again for itself (`Poub.Span.publicWords`, `signWords`, `supplied`); the difftest
holds the two to each other."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.commitments.merkle import merkle_depth
from verity.protocols.resource_accounting.memory.poub.verifier.commitments.gates import (
    block_check as BC,
)

from . import audit, receipts

#: the opening unit's rows the verifier opens under the prover's salt: its verdict
OPENED = ("o0",)
#: the registered parameters the verifier supplies, a row a unit: the unit's public rows
SUPPLIED = ("pub", "signs")


def public_v2(d: audit.PositionSample, codec_seed: bytes, S) -> dict[str, list[int]]:
    """Sample ``d``'s public parameters as u16 words at the unit's statics ``S`` (`BC.opening_public_v2`, its sign rows
    among them): block j = p of a pass whose plan is the identity, registered (``coin`` None)."""
    if d.coin is not None or d.block != d.position:
        raise ValueError("a span's unit opens a registered block at its own position: the pass's plan is the identity")
    rdom = receipts.domain(d.seed, d.plan)
    first = d.domain.positions.start
    return BC.opening_public_v2(S, codec_seed, d.block, None, receipts.sketch_prefix(d.seed),
                                receipts.sign_bits(d.seed, S.ROWS), (d.root, rdom.domain_id, merkle_depth(rdom.count)),
                                (d.commitment.root, d.domain.domain_id, d.block - first, merkle_depth(d.domain.count)), None)


def public_rows(d: audit.PositionSample, codec_seed: bytes, S) -> dict[str, list[int]]:
    """Sample ``d``'s ``pub`` and ``signs`` rows as u16 words at the unit's statics ``S`` (`public_v2`)."""
    a = public_v2(d, codec_seed, S)
    pub = BC.opening_rows_v2(S)[0][2]
    return {"pub": [w for p in pub for w in a[p]], "signs": a["signs"]}


def supplied(ds: Sequence[audit.PositionSample], codec_seed: bytes, S) -> dict[str, list[bytes]]:
    """Each supplied parameter's rows, sample k's row k (`public_rows`): u16 words w0, w1, ... as big-endian 32-bit halves
    w1·2^16 + w0, ..., the bytes `one_stage.registered.bytes_leaves` reads back as 16-bit leaves w0, w1, ...."""
    rows = [public_rows(d, codec_seed, S) for d in ds]
    return {r: [_halves(x[r]) for x in rows] for r in SUPPLIED}


def _halves(words: Sequence[int]) -> bytes:
    if len(words) % 2:
        raise ValueError(f"a supplied row is whole 32-bit halves, not {len(words)} u16 words")
    return b"".join((words[k + 1] << 16 | words[k]).to_bytes(4, "big") for k in range(0, len(words), 2))


def layout(S, verdict_bits: int) -> list[list]:
    """The registered opening circuit's ports in order, ``[name, bits]``: the unit's input rows (`BC.opening_rows_v2`),
    then its one output row, ``o0``, of ``verdict_bits`` bits as the circuit commits it."""
    return [[name, bits] for name, bits, _ in BC.opening_rows_v2(S)] + [["o0", verdict_bits]]


def auditor(params: dict, ports: list[list]) -> dict:
    """The verifier's own side: the one-stage audit's parameters (``proofs.public`` is each statement's) and the layout."""
    return {"layout": ports, "params": params}


def instance(verdict: int, salt: bytes) -> dict:
    """One unit: its verdict word and its ``o0`` row's salt."""
    return {"verdict": verdict, "verdict_salt": salt.hex()}


def statement(public: bytes, record: bytes, evidence: bytes, instances: Sequence[dict]) -> dict:
    """One statement: its public file's SHA-512, its docket's and evidence's bytes, its instances in order."""
    return {"evidence": evidence.hex(), "instances": list(instances), "pub": hashlib.sha512(public).hexdigest(),
            "record": record.hex()}


def span(statements: Sequence[dict], blobs: Sequence[bytes]) -> dict:
    """The span: its statements in sample order and the blobs their evidence names (their public files among them)."""
    return {"blobs": [b.hex() for b in blobs], "statements": list(statements)}


__all__ = ["OPENED", "SUPPLIED", "auditor", "instance", "layout", "public_rows", "public_v2", "span", "statement",
           "supplied"]
