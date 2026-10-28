"""The remote bandwidth audit over PoUS's P2 store (`p2-unit4/v0`): the node's registered store is PoUS's `C*`, the
weights `W` encoded under P2 (`pous.schemes.p2`, `p2-16448/v4`) at PoUS's setup, and each receipt is of a unit of four
of its 2,056-byte blocks, which a GPU sketches as it decodes them while serving.

~~~text
block j       C*'s block j: 2,056 bytes, big-endian, at byte 2056·j of the store; j = blocks_per_segment·s + v
unit g        blocks 4g … 4g + 3, 8,224 bytes at byte 8224·g: 257 whole 32-byte sectors.  U = ⌊N/4⌋ units; the last
              N − 4U blocks (none in a store of whole units, as PoUS's of 2^16-block segments) are in no unit, and no
              receipt certifies them
registration  node, P2's parameters, PoUS's commitment to W, PoUS's salt, the vk root, k_s: PoUS's setup's, the salt
              drawn after the commitment (`setup.draw_salt`) and the root the verifier built from the C* it decoded to
              the committed W (`setup.verifier_key_from_encoding`); the node the one whose store the deployment's
              registration log registers (`stored`)
digest        SHA512(REGISTRATION_DOMAIN ‖ frame(node) ‖ u32be w ‖ u32be c0 ‖ u32be blocks_per_segment ‖
              u64be segments ‖ frame(expander) ‖ commitment ‖ salt ‖ root ‖ frame(layout) ‖ frame(receipt kind) ‖
              u32be k_s),  REGISTRATION_DOMAIN = "verity/remote-bandwidth/p2-registration/v0\\0"
S_i           k_s × 8224 signs (`receipts.sign_bits(seed_i, k_s, 8224)`): K = derive(seed_i, SIGN_DOMAIN, {"tag":
              "sketch-sha512", "k": k_s, "n": 8224}), row c the first 1,028 bytes of K.stream(("row", c)), bit x
              (byte x // 8, LSB first) 1 → −1, 0 → +1
π_(i,j)       block j's partial: Σ_x S_i[c][2056·(j mod 4) + x] · int8(block_j[x]) over x < 2056, the block's bytes
              as stored; its signs are bytes 257·(j mod 4) … 257·(j mod 4) + 256 of each row.  k_s int32
              little-endian, |π[c]| ≤ 128·2056
σ_(i,g)       Σ π_(i,j) over unit g's blocks, row by row: S_i · int8(unit g's 8,224 bytes), k_s int32 little-endian,
              |σ[c]| ≤ 128·8224 < 2^21
y_(i,p)       SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be g ‖ σ_(i,g)),  g = plan_i[p]
~~~

Everything else is the audit's (`audit.py`): seeds, plans (over units), leaves of sixteen receipts, their trees,
windows, deadlines, coins and draws.  A unit is the receipt's vector so that it is whole sectors: block j starts at
byte 8j mod 32 of the store, so a receipt of one block would leave a sector half read, and the sketch's bound on a
guessed receipt (`SketchBoundSession`, over sectors) would lose a factor 2^32 a receipt.

An opening (`Opening`) is the unit's four blocks, each with its path to the registration's root, and the receipt
leaf with its salt and path to the pass's root.  The verifier (`Session.verify`) checks the leaf on the pass's root,
each block on the vk root under PoUS's leaf (`pous.verifier.leaf_hash`; `payload-path` if one fails), and the receipt
recomputed from the four blocks.  It decodes nothing: vk's leaves are of the C* the verifier decoded to the committed W
at PoUS's setup, so a block on its path is C*'s block j, by SHA-512's collision resistance.  A P2 store is fixed at
setup: a session appends nothing to it (`append` rejects with `append`).  `audit.Rebuild` refuses a P2 codec, so
`audit.ledger` refuses a P2 record until μ_rebuild at P2's encode is stated (`codec.P2Codec`).

The decode kernel's side of the receipts is `PROTOCOL.md` §11: per block it decodes, π from the pass's sign bits, and
per unit σ and y.  `Sketcher` is its CPU reference.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity.core.protocols import registration as log
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.params import SALT_BITS
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, leaf_hash
from verity.ml.randomness import frame

from . import audit, coins, receipts
from .codec import P2_LAYOUT, P2Codec
from .receipts import DIGEST_BYTES, PER_LEAF, SALT_BYTES, Plan

PROTOCOL_ID = "verity/remote-bandwidth/p2/v0"
REGISTRATION_DOMAIN = b"verity/remote-bandwidth/p2-registration/v0\0"
BLOCK_BYTES = 2056
UNIT_BLOCKS = 4
UNIT_BYTES = UNIT_BLOCKS * BLOCK_BYTES
SECTOR_BYTES = 32
UNIT_SECTORS = UNIT_BYTES // SECTOR_BYTES
BLOCK_SIGN_BYTES = BLOCK_BYTES // 8
"""A block's share of each sign row, in bytes: 257, so block j's signs are whole bytes of the row."""


def codec_of(params: P2Params) -> P2Codec:
    """The codec a registration names for PoUS's parameters `params` (ValueError off the registered point)."""
    if type(params) is not P2Params:
        raise ValueError("a P2 store's parameters are PoUS's P2Params")
    return P2Codec(params.blocks_per_segment, params.segments, params.w, params.c0, params.expander)


def pous_params(codec: P2Codec) -> P2Params:
    return P2Params(codec.w, codec.c0, codec.blocks_per_segment, codec.segments, codec.expander)


def _bytes(value: object, n: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != n:
        raise ValueError(f"{name} is {n} bytes")
    return value


@dataclass(frozen=True, slots=True)
class Registration:
    """What the store is: the node, P2's parameters (`codec`), PoUS's setup's commitment to W, salt and vk root, and
    sketch-sha512's rows k_s.  Its `blocks` are the store's units, what its sessions' plans range over."""

    node: bytes
    codec: P2Codec
    commitment: bytes
    salt: bytes
    root: bytes
    rows: int = receipts.SKETCH_ROWS

    receipt = receipts.SKETCH_SHA512
    bind = False
    layout = P2_LAYOUT

    def __post_init__(self) -> None:
        audit._node(self.node)
        if type(self.codec) is not P2Codec:
            raise ValueError("a P2 registration names a P2Codec")
        if self.codec.blocks < UNIT_BLOCKS:
            raise ValueError(f"a P2 store is at least one unit of {UNIT_BLOCKS} blocks")
        _bytes(self.commitment, DIGEST_BYTES, "PoUS's commitment to W")
        _bytes(self.salt, SALT_BITS // 8, "PoUS's salt")
        _bytes(self.root, DIGEST_BYTES, "the vk root")
        receipts.sketch_rows(self.receipt, self.rows)

    @property
    def blocks(self) -> int:
        """U, the store's units."""
        return self.codec.blocks // UNIT_BLOCKS

    @property
    def params(self) -> P2Params:
        return pous_params(self.codec)

    @property
    def key(self) -> VerifierKey:
        """PoUS's root-only `vk`: what an opened block is checked on (`VerifierKey.check_path`)."""
        return VerifierKey(self.params, self.salt, self.root, ())

    @property
    def digest(self) -> bytes:
        c = self.codec
        return hashlib.sha512(
            REGISTRATION_DOMAIN + frame(self.node) + c.w.to_bytes(4, "big") + c.c0.to_bytes(4, "big")
            + c.blocks_per_segment.to_bytes(4, "big") + c.segments.to_bytes(8, "big") + frame(c.expander)
            + self.commitment + self.salt + self.root + frame(self.layout) + frame(self.receipt)
            + self.rows.to_bytes(4, "big")).digest()


def register(source: bytes, node: bytes, commitment: bytes, vk: VerifierKey, *,
             rows: int | None = None) -> Registration:
    """The verifier's registration of a node's P2 store from its PoUS setup: the salt it draws from its own secret
    `source` after the commitment (`setup.draw_salt`), which `vk`, the key it built from the C* it decoded at setup
    (`setup.verifier_key_from_encoding`), must be under."""
    salt = setup.draw_salt(source, _bytes(commitment, DIGEST_BYTES, "PoUS's commitment to W"))
    if type(vk) is not VerifierKey or type(vk.params) is not P2Params or vk.salt != salt:
        raise ValueError("vk is the verifier's own, built at PoUS's setup under the salt it drew after the commitment")
    rows = receipts.sketch_rows(receipts.SKETCH_SHA512, rows)
    return Registration(node, codec_of(vk.params), commitment, salt, vk.root, rows)


def stored(view: log.View, store: str, blocks: int) -> bytes:
    """The node whose P2 store of `blocks` blocks the registration log registers as `store`, read from an accepted
    log's view: the store's hardware, the memory leaf of that id, in a host, whose id is the node, holding the blocks'
    bytes."""
    x = view.site.loc(store)
    if x is None or x.kind != "memory":
        raise ValueError(f"the registration log registers no memory leaf {store}")
    host = view.site.host(x)
    if host is None:
        raise ValueError(f"the store {store} is in no host")
    if x.bytes != blocks * BLOCK_BYTES:
        raise ValueError(f"the store {store} is not {blocks} blocks of {BLOCK_BYTES} bytes")
    return audit._node(host.encode())


def register_stored(source: bytes, view: log.View, store: str, commitment: bytes, vk: VerifierKey, *,
                    rows: int | None = None) -> Registration:
    """`register` of the node whose P2 store, of vk's blocks, the registration log registers as `store` (`stored`)."""
    if type(vk) is not VerifierKey:
        raise ValueError("vk is the verifier's own, built at PoUS's setup")
    return register(source, stored(view, store, codec_of(vk.params).blocks), commitment, vk, rows=rows)


class Sketcher(receipts.SketchReceipter):
    """sketch-sha512 receipts of P2 units under one pass's seed at `rows` = k_s: `sketcher(g, unit)` is `y` of unit g
    from its 8,224 bytes, `block(j, b)` block j's partial π, `total` the partials of a unit summed, and
    `receipt(g, σ)` y from them, the decode's side (`PROTOCOL.md` §11)."""

    def __init__(self, seed: bytes, rows: int = receipts.SKETCH_ROWS) -> None:
        super().__init__(seed, rows, UNIT_BYTES)
        self.seed = seed

    def block(self, j: int, block: bytes) -> bytes:
        """π: block j's partial, its 2,056 bytes at their place in unit j // 4."""
        if type(j) is not int or j < 0:
            raise ValueError("a block index is a nonnegative integer")
        _bytes(block, BLOCK_BYTES, "a P2 block")
        return self.partial(block, BLOCK_BYTES * (j % UNIT_BLOCKS))

    def total(self, partials: Sequence[bytes]) -> bytes:
        """σ: `partials` summed row by row, as int32 (exact: a unit's |σ[c]| < 2^21)."""
        sums = [0] * self.rows
        for x in partials:
            for c, v in enumerate(struct.unpack(f"<{self.rows}i", _bytes(x, 4 * self.rows, "a partial"))):
                sums[c] += v
        return struct.pack(f"<{self.rows}i", *sums)

    def signs(self) -> tuple[bytes, ...]:
        """S_i's sign bits, k_s rows of 1,028 bytes, what the kernel reads."""
        return receipts.sign_bits(self.seed, self.rows, UNIT_BYTES)


class Store:
    """The prover's C* under a P2 registration: its N blocks and PoUS's vk tree over them, whose root must be the
    registration's."""

    def __init__(self, registration: Registration, blocks: Sequence[bytes]) -> None:
        params = registration.params
        if len(blocks) != params.blocks or any(type(b) is not bytes or len(b) != BLOCK_BYTES for b in blocks):
            raise ValueError(f"the registration's C* is {params.blocks} blocks of {BLOCK_BYTES} bytes")
        self.registration, self.blocks = registration, tuple(blocks)
        self.tree = MerkleTree([leaf_hash(params, registration.salt, j, b) for j, b in enumerate(self.blocks)])
        if self.tree.root != registration.root:
            raise ValueError("these blocks are not the registered C*: their vk root is not the registration's")

    def unit(self, g: int) -> bytes:
        return b"".join(self.blocks[UNIT_BLOCKS * g:UNIT_BLOCKS * (g + 1)])


@dataclass(frozen=True, slots=True)
class Opening:
    """Receipt (i, p) of a window opened, position p of pass i, unit g = plan_i[p]: its four blocks, each block's path
    to the registration's root, and its receipt leaf's receipts, salt and path to pass i's root."""

    pass_index: int
    position: int
    blocks: tuple[bytes, ...]
    paths: tuple[tuple[bytes, ...], ...]
    leaf: bytes
    leaf_salt: bytes
    receipt_path: tuple[bytes, ...]

    @property
    def size(self) -> int:
        """Its bytes on the wire, with u32 and u64 headers for the indices and a u32 for each path's length."""
        return (4 + 8 + UNIT_BYTES + len(self.leaf) + len(self.leaf_salt)
                + DIGEST_BYTES * (len(self.receipt_path) + sum(len(x) for x in self.paths)) + 4 * (2 + UNIT_BLOCKS))


class Prover(audit.Prover):
    """The honest prover over a P2 store: each pass's receipts from its plan's units (`pass_under`), or from receipts
    the decode computed (`commit`), committed in the audit's tree."""

    def __init__(self, store: Store, *, entropy: Callable[[int], bytes] = os.urandom) -> None:
        super().__init__(store, None, entropy=entropy)

    def pass_under(self, window: int, i: int, seed: bytes, plan: Plan | None = None) -> audit.Pass:
        plan = plan or Plan.whole(self.store.registration.blocks)
        sketcher = Sketcher(seed, self.store.registration.rows)
        return self.commit(window, i, seed, plan, [sketcher(g, self.store.unit(g)) for g in plan])

    def commit(self, window: int, i: int, seed: bytes, plan: Plan, ys: Sequence[bytes]) -> audit.Pass:
        """Pass i of `window` under `seed` from its receipts `ys` in plan order, however they were computed."""
        salts = tuple(self.entropy(SALT_BYTES) for _ in range(receipts.leaves(plan.length)))
        return audit.Pass(window, i, seed, plan, tuple(ys), salts, receipts.tree(seed, plan, ys, salts))

    def open(self, ps: audit.Pass, p: int) -> Opening:
        g, c = ps.plan.block(p), p // PER_LEAF
        js = range(UNIT_BLOCKS * g, UNIT_BLOCKS * (g + 1))
        return Opening(ps.index, p, tuple(self.store.blocks[j] for j in js), tuple(self.store.tree.path(j) for j in js),
                       b"".join(ps.receipts[PER_LEAF * c:PER_LEAF * (c + 1)]), ps.salts[c], ps.tree.open(c).path)


def _paths(paths: object) -> bool:
    return type(paths) is tuple and len(paths) == UNIT_BLOCKS and all(
        type(x) is tuple and all(type(d) is bytes and len(d) == DIGEST_BYTES for d in x) for x in paths)


def check(registration: Registration, seed: bytes, plan: Plan, root: bytes, i: int, p: int, o: Opening,
          sketcher: Sketcher | None = None) -> str | None:
    """Opening `o` of receipt (i, p), position p of pass i under `seed` and `plan`, whose receipt tree's root is
    `root`: None if it checks, else the first of `malformed` (not (i, p), not four 2,056-byte blocks with four paths of
    64-byte siblings, a leaf salt not 192 bytes, or receipts not the size of the position's leaf), `receipt-path`,
    `payload-path` (a block not unit g's on the vk root, under PoUS's leaf) and `receipt` (y recomputed from the
    four blocks not the opened one).  `sketcher` is the pass's, if the caller holds one (`Sketcher(seed, rows)`)."""
    if (type(o) is not Opening or o.pass_index != i or o.position != p or type(o.blocks) is not tuple
            or len(o.blocks) != UNIT_BLOCKS or any(type(b) is not bytes or len(b) != BLOCK_BYTES for b in o.blocks)
            or not _paths(o.paths) or type(o.leaf) is not bytes or type(o.leaf_salt) is not bytes
            or len(o.leaf_salt) != SALT_BYTES or type(o.receipt_path) is not tuple
            or any(type(x) is not bytes or len(x) != DIGEST_BYTES for x in o.receipt_path)):
        return "malformed"
    g = plan.block(p)
    y = receipts.leaf_receipt(o.leaf, p, plan.length)
    if y is None:
        return "malformed"
    if not receipts.verify_leaf(seed, plan, root, p, o.leaf, o.leaf_salt, o.receipt_path):
        return "receipt-path"
    key = registration.key
    if not all(key.check_path(UNIT_BLOCKS * g + b, x, path) for b, (x, path) in enumerate(zip(o.blocks, o.paths))):
        return "payload-path"
    if (sketcher or Sketcher(seed, registration.rows))(g, b"".join(o.blocks)) != y:
        return "receipt"
    return None


class Session(audit.AuditRun):
    """The verifier's side of one session over a P2 store: the audit's session (`audit.AuditRun`) with units for blocks,
    its openings checked on PoUS's vk (`_check`), and no appends."""

    codecs = (P2Codec,)
    block_bytes = UNIT_BYTES

    def __init__(self, params: audit.Params, registration: Registration, coins_: coins.SessionCoins, *,
                 not_before_ns: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        if type(registration) is not Registration:
            raise ValueError("a P2 session is over a P2 Registration")
        super().__init__(params, registration, None, coins_, not_before_ns=not_before_ns, clock=clock)

    def append(self, root: bytes, count: int) -> tuple[int, bytes] | audit.Verdict:
        """Refused (`append`): a P2 store is fixed at PoUS's setup."""
        self._live()
        return self._fail(self.filling, "append")

    def _check(self, t: int, i: int, p: int, o: Opening, key: VerifierKey, receipters: dict) -> str | None:
        w = self.windows[t]
        seed = receipts.seed(w.challenge, i, self._previous(t, i))
        if i not in receipters:
            receipters[i] = Sketcher(seed, self.registration.rows)
        return check(self.registration, seed, w.plans[i], w.roots[i], i, p, o, receipters[i])
