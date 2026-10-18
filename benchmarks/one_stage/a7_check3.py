"""A7's check 3 under zero knowledge: each receipt the remote session draws goes through the check-3 proof units
(`verity.ml.commitments.gates.block_check`) in place of its opening, so the verifier holds the drawn block, its payload,
its receipt leaf and the step's plan only as hiding commitments, and each unit's verdict.

For a draw (pass i of window t at step s, position p, block j = plan_i[p]) the prover stages one `BlockReceiptCheck`
(the block's sketch receipt is entry p mod 16 of its receipt leaf, and the step's certified plan row holds j at p) and
the `slices` of `BlockDecodeCheck` (the block decodes to its payload under b_j(c_a), or under the seed's key for a
registered block). Each row is an hm96-sha512/row/v2 row: the block under one salt in every unit of the draw, the payload
under its payload salt, the leaf under its salt in the pass's receipt tree, the plan under the canonical plan's salt of
step s. The verifier opens only the public rows, which it derives itself (`public`: the codec seed, j, q and the
append's coin; the pass's chaining value, sign rows, j, k and p), and the ok rows, and holds every other row by
``b || c`` (`links`): the units' block rows to each other, the leaf's (as an hm96 tree leaf) to the pass's receipt
root, the payload's to the registration's payload root or its append's and, for an appended block, to the canonical
payload row the memory-map unit reads, and the plan's to the canonical plan row of step s.

`Session` is `audit.Session` with check 3's outcome in place of `verify`'s openings (`drawn`, `decide`), and `Store` is
`audit.Store` keeping the payloads it appends (the prover's witness). `reference` decides a session by the units'
references on the prover's witness and the same links: the stand-in for their proofs in the sessions of A7's negatives.
A7's run proves them (`a7.check3_audit`)."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from verity.core.protocols.remote_bandwidth import audit, codec, receipts
from verity.core.protocols.remote_bandwidth.receipts import PER_LEAF, Plan
from verity.ml.commitments import Commitment, CommitmentDomain, Opening as MerkleOpening, verify_opening
from verity.ml.commitments.gates import block_check as BC
from verity_flock import circuit as FC

ROWS = receipts.SKETCH_ROWS
WORDS = 16
WITNESS_ROWS = ("block", "payload", "leaf", "plan")


def decode_statics(c: codec.Codec, words: int = WORDS) -> SimpleNamespace:
    return SimpleNamespace(N=c.field.n, C=c.field.c, K=c.rounds, MASKED=int(c.masked), BYTES=codec.BLOCK_BYTES, WORDS=words)


def receipt_statics(leaf: int, P: int) -> SimpleNamespace:
    return SimpleNamespace(ROWS=ROWS, LEAF=leaf, P=P, BYTES=codec.BLOCK_BYTES)


def slices(D) -> int:
    return D.BYTES * 8 // D.N // D.WORDS


@dataclass(frozen=True)
class Draw:
    """What the verifier knows of one draw: window t, pass i at step s, position p, block j = plan_i[p], the pass's seed,
    plan and root, and the commitment block j's payload is a leaf of: the registration's payload tree (``coin`` None) or
    its append's, under the append's coin."""

    window: int
    index: int
    position: int
    block: int
    step: int
    seed: bytes
    plan: Plan
    root: bytes
    domain: CommitmentDomain
    commitment: Commitment
    coin: bytes | None

    @property
    def leaf(self) -> int:
        return self.position // PER_LEAF

    @property
    def leaf_size(self) -> int:
        return min(PER_LEAF, self.plan.length - PER_LEAF * self.leaf)


class Session(audit.Session):
    """`audit.Session` with check 3 decided by proofs: `drawn` is what the verifier knows of window t's draws, and
    `decide` takes check 3's outcome on each in place of `verify`'s openings."""

    def drawn(self, t: int) -> tuple[Draw, ...]:
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None:
            raise audit.SessionError(f"window {t} has no draws")
        N, out = self.params.blocks, []
        for i, p in w.draws:
            plan = w.plans[i]
            j = plan.block(p)
            if j < N:
                domain, commitment, coin = self._payload_domain, Commitment(self.registration.payload_root, N), None
            else:
                _f, domain, commitment, coin = self._appends[bisect.bisect_right(self._appends, j, key=lambda a: a[0]) - 1]
            out.append(Draw(t, i, p, j, w.steps[i], receipts.seed(w.challenge, i, self._previous(t, i)), plan, w.roots[i],
                            domain, commitment, coin))
        return tuple(out)

    def decide(self, t: int, failures: Sequence[str | None]) -> audit.Verdict:
        """Window t's verdict from check 3's outcome on each of its draws, in the order drawn: the first failure (one of
        `audit.FAILURES`) rejects it."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None or w.verdict is not None:
            raise audit.SessionError(f"window {t} has no draws awaiting check 3")
        if len(failures) != len(w.draws):
            return self._fail(t, "malformed")
        for at, f in zip(w.draws, failures):
            if f is not None:
                return self._fail(t, f if f in audit.FAILURES else "malformed", at)
        w.verdict = audit.Verdict(t, True)
        return w.verdict


class Store(audit.Store):
    """`audit.Store` keeping each appended block's payload, which check 3's decode unit takes as its witness."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.kept: dict[int, bytes] = {}

    def append(self, payloads: Sequence[bytes], salts: Sequence[bytes]) -> audit.Append:
        a = super().append(payloads, salts)
        self.kept.update((a.first + u, m) for u, m in enumerate(payloads))
        return a

    def payload(self, j: int) -> bytes:
        return self.payloads[j] if j < self.registration.blocks else self.kept[j]


@dataclass(frozen=True)
class Witness:
    """The prover's side of one draw: the receipt unit's arguments, each decode slice's, every witness row's salt, and the
    paths of the receipt leaf and the payload to their roots."""

    receipt: dict[str, list[int]]
    decode: tuple[dict[str, list[int]], ...]
    salts: dict[str, bytes]
    leaf_path: tuple[bytes, ...]
    payload_path: tuple[bytes, ...]


def witness(d: Draw, ps: audit.Pass, store: Store, plan_row: Sequence[int], plan_salt: bytes, block_salt: bytes, P: int,
            D) -> Witness:
    """The honest prover's units for draw ``d`` of pass ``ps``: its block, payload and receipt leaf from its store and the
    pass, the step's certified plan row under the canonical plan's salt, the block under ``block_salt``."""
    j, c = d.block, d.leaf
    block, payload = store.blocks[j], store.payload(j)
    pay_salt, pay_path, _d = store.committed(j)
    leaf = b"".join(ps.receipts[PER_LEAF * c:PER_LEAF * (c + 1)])
    r = BC.receipt_args(receipt_statics(len(leaf) // 64, P), receipts.sketch_prefix(d.seed), receipts.sign_bits(d.seed, ROWS),
                        j, d.position - PER_LEAF * c, d.position, block, leaf, list(plan_row))
    seed = store.registration.codec_seed
    dec = tuple(BC.decode_args(D, seed, j, block, payload, d.coin, q) for q in range(slices(D)))
    return Witness(r, dec, {"block": block_salt, "payload": pay_salt, "leaf": ps.salts[c], "plan": plan_salt},
                   tuple(ps.tree.open(c).path), tuple(pay_path))


def public(d: Draw, codec_seed: bytes, P: int, D) -> SimpleNamespace:
    """The verifier's own copy of draw ``d``'s public rows: the receipt unit's (its chaining value, j, k, p, and the pass's
    sign rows) and each decode slice's."""
    r = BC.receipt_public(receipt_statics(d.leaf_size, P), receipts.sketch_prefix(d.seed), receipts.sign_bits(d.seed, ROWS),
                          d.block, d.position - PER_LEAF * d.leaf, d.position)
    return SimpleNamespace(receipt=r, decode=[BC.decode_public(D, codec_seed, d.block, d.coin, q) for q in range(slices(D))])


def commit(name: str, bits: int, words: Sequence[int], salt: bytes) -> bytes:
    """A row's ``b || c`` under ``salt`` (C-Flock's hm96-sha512/row/v2 commit string)."""
    port = FC._port(name, -(-bits // 16), bits, False)
    return FC._commit_strings([port], {name: np.asarray([list(words)], dtype=np.uint16)}, {name: [salt]})[name][0]


def witness_bcs(rows, args: dict, salts: dict[str, bytes]) -> dict[str, bytes]:
    """Each witness row's ``b || c`` from an instance's arguments under the draw's salts."""
    return {r: commit(r, bits, [w for p in ps for w in args[p]], salts[r]) for r, bits, ps in rows if r in WITNESS_ROWS}


def links(d: Draw, receipt: dict[str, bytes], decodes: Sequence[dict[str, bytes]], leaf_path: Sequence[bytes],
          payload_path: Sequence[bytes], plan_bc: bytes, kv_bc: bytes | None) -> str | None:
    """Check 3's links by ``b || c`` alone: the block row of every decode slice is the receipt unit's; the payload row is
    one row across the slices, its hm96 tree leaf leaf j of the registration's payload tree or its append's, and, for an
    appended block, the canonical payload row ``kv_bc`` the memory-map unit reads; the receipt leaf's tree leaf is leaf
    p / 16 of the pass's receipt tree; the plan row is the canonical plan row ``plan_bc`` of the pass's step. Returns
    the failure, one of `audit.FAILURES`, or None."""
    if any(x["block"] != receipt["block"] for x in decodes):
        return "codec"
    pay = decodes[0]["payload"]
    if any(x["payload"] != pay for x in decodes):
        return "payload-path"
    dom = receipts.domain(d.seed, d.plan)
    leaf = MerkleOpening(d.leaf, receipts.HM96.tree_leaf(receipts.KEY, receipt["leaf"]), tuple(leaf_path))
    if not verify_opening(dom, Commitment(d.root, dom.count), leaf, receipts.SCHEMA, receipts.LIMITS):
        return "receipt-path"
    opening = MerkleOpening(d.block, receipts.HM96.tree_leaf(receipts.KEY, pay), tuple(payload_path))
    if not verify_opening(d.domain, d.commitment, opening, audit.PAYLOAD_SCHEMA, receipts.LIMITS):
        return "payload-path"
    if kv_bc is not None and pay != kv_bc:
        return "payload-path"
    if receipt["plan"] != plan_bc:
        return "plan"
    return None


def check(d: Draw, w: Witness, codec_seed: bytes, P: int, D, plan_bc: bytes, kv_bc: bytes | None) -> str | None:
    """Check 3 on one draw by the units' references (`BC.receipt_check`, `BC.decode_check`) on the prover's witness, its
    public rows the verifier's own, then `links`."""
    pub, RS = public(d, codec_seed, P, D), receipt_statics(d.leaf_size, P)
    if len(w.receipt["leaf"]) != 32 * RS.LEAF or any(w.receipt[k] != v for k, v in pub.receipt.items()) or any(
            a[k] != v for a, p in zip(w.decode, pub.decode, strict=True) for k, v in p.items()):
        return "malformed"
    if not BC.receipt_check(RS, w.receipt):
        return "receipt"
    if not all(BC.decode_check(D, a) for a in w.decode):
        return "codec"
    return links(d, witness_bcs(BC.receipt_rows(RS), w.receipt, w.salts),
                 [witness_bcs(BC.decode_rows(D), a, w.salts) for a in w.decode], w.leaf_path, w.payload_path, plan_bc, kv_bc)


def kv(canon_payload_bcs: Sequence[bytes], N: int, j: int) -> bytes | None:
    """The canonical payload row block j's payload is to be (None for a registered block; empty, matching no row, for an
    index the layout never wrote)."""
    if j < N:
        return None
    return canon_payload_bcs[j - N] if j - N < len(canon_payload_bcs) else b""


def reference(session: Session, store: Store, passes: dict, plan_rows: Sequence[Sequence[int]], plan_salts: Sequence[bytes],
              plan_bcs: Sequence[bytes], payload_bcs: Sequence[bytes], P: int, D, entropy: Callable[[int], bytes]
              ) -> list[audit.Verdict]:
    """Every window of a session whose draws are drawn, decided by `check` on the honest prover's witness: the stand-in
    for the units' proofs. Stops at the first rejected window."""
    out, N = [], store.registration.blocks
    for t in range(len(session.windows)):
        fails = []
        for d in session.drawn(t):
            w = witness(d, passes[d.window, d.index], store, plan_rows[d.step], plan_salts[d.step], entropy(receipts.SALT_BYTES),
                        P, D)
            fails.append(check(d, w, store.registration.codec_seed, P, D, plan_bcs[d.step], kv(payload_bcs, N, d.block)))
        out.append(session.decide(t, fails))
        if not out[-1].accepted:
            break
    return out
