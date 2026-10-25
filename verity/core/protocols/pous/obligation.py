"""PoUS as an obligation at test scale: KV pages re-encoded on write under coin-keyed tags, the memory challenger's coin
per page write, spot reads against a window's snapshot, and the verdict on the challenger's log, which
`PousCheck.Obligation` (`check/PousCheck/Obligation.lean`) computes in Lean.

The store is the weights' `C*` (`period`'s setup, unchanged) and the KV region: `pages` pages of `blocks` P2 blocks each
(`params`: P2 at the ladder's 64-bit width, one segment a page). Page `p`'s version `v` is written in three steps, which
the memory challenger logs in the order it sees them:

1. `commit`: the prover sends the commitment to the page's plaintext (`commitment`);
2. `coin`: the memory challenger draws the page's coin, one per `(p, v)` and only after its commit (`PageChallenger.coin`
   refuses any other);
3. `store`: the prover encodes the page under its tag (`page_salt`, a function of `salt ‖ p ‖ v ‖ coin`) and sends the
   encoded page; the verifier decodes it under that tag and checks it against the commitment (setup's public-encoder
   check, per page), and the page's root is the root over its blocks' leaves (`page_root`).

The page table is public: page `p`'s versions are `0, 1, 2, …` in order, so the set of tags a snapshot reads is fixed
and injective, and no rewrite selects a key. A window opens with `snapshot`, the root over every page's latest stored
root (`snapshot_root`), and each of its `read`s names a page and a block, carries the prover's block and audit path,
and the memory challenger's measured latency. Reads are answered against the snapshot, so a page rewritten during the
window is answered from its snapshot version.

`verdict(log)` judges a log, in order, and rejects at the first event that breaks a rule, with one of `REASONS`:

* `page`: a page or block index outside the store;
* `version`: a commit that skips the page's next version;
* `recommit`: a second commit of a `(p, v)`: content chosen after its coin, the only way to a page whose content is its
  own keystream (`keystream`);
* `coin`: a coin for a `(p, v)` with no commit, or a second one;
* `early`: a store of a `(p, v)` before its coin: an encoding that no fresh coin keys;
* `restore`: a store of a version no newer than the page's stored one;
* `decode`: a store that does not decode under its tag to the committed plaintext (a page encoded under another coin,
  before its own was drawn, among them);
* `snapshot`: a snapshot before every page is stored, or whose root is not the pages' latest roots';
* `order`: a read before any snapshot;
* `late`: a read answered past `limit_ns`;
* `stale`: a read answered with an earlier version of the page's block (it verifies against that version's root);
* `wrong`: any other wrong answer;
* `reads`: a log with no read.

An accepted log answers `{"verdict": "accept", "reads", "windows", "writes"}`, a rejected one
`{"verdict": "reject", "reason", "event"}` (the event's index; the log's length for `reads`). A log that isn't one
(a field missing or mistyped, hex that isn't lowercase and even, a salt or coin of the wrong length, an unknown event,
a store of more than 2^16 pages or blocks) raises `ValueError`.

The commitment here is SHA-512 of the plaintext under a domain, and the verifier decodes each page in the clear, as
setup's public-encoder check does with `C*` (#4417's tie); deployed, both are proved in zero knowledge over hiding
commitments. The coins come from `coins` in the memory challenger's process, as `period`'s setup salt comes from the
node's: `trusted-nebius-host`'s premise for them holds only where that process runs off the node.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import verifier as V
from .schemes import p2 as P2

W, C0 = 64, 189
"""The page codec's width: P2's smallest ladder width (`2^64 − 189` prime, ≡ 3 mod 4), 6 payload bytes an 8-byte block."""
SALT_BYTES = 24
"""P2's package salt, which a page's tag is cut to (`page_salt`), and the store's salt."""
COIN_BYTES = 32
MAX = 1 << 16
"""At most this many pages, and blocks a page (P2's block index within a segment is 16 bits)."""
TAG_DOMAIN = b"verity/pous/kv-page/tag/v0"
COMMIT_DOMAIN = b"verity/pous/kv-page/commit/v0"
REASONS = ("page", "version", "recommit", "coin", "early", "restore", "decode", "snapshot", "order", "late", "stale",
           "wrong", "reads")
_HEX = re.compile(r"(?:[0-9a-f]{2})*")


def params(blocks: int) -> P2.P2Params:
    """A page's codec: one P2 segment of `blocks` blocks at the page width."""
    return P2.P2Params(W, C0, blocks, 1)


def page_salt(salt: bytes, page: int, version: int, coin: bytes) -> bytes:
    """Page `page`'s version `version`'s tag, as the P2 salt its blocks' keys expand from:
    `SHA512(TAG_DOMAIN ‖ salt ‖ u64be(page) ‖ u64be(version) ‖ coin)` cut to `SALT_BYTES`."""
    return hashlib.sha512(TAG_DOMAIN + salt + page.to_bytes(8, "big") + version.to_bytes(8, "big") + coin).digest()[
        :SALT_BYTES]


def commitment(page: int, version: int, plaintext: bytes) -> bytes:
    """The commitment to a page's plaintext: `SHA512(COMMIT_DOMAIN ‖ u64be(page) ‖ u64be(version) ‖ plaintext)`."""
    return hashlib.sha512(COMMIT_DOMAIN + page.to_bytes(8, "big") + version.to_bytes(8, "big") + plaintext).digest()


def encode_page(q: P2.P2Params, tag_salt: bytes, plaintext: bytes) -> list[bytes]:
    """The page's blocks under its tag (`page_salt`): P2's two roots a block."""
    return P2.encode_segment(q, None, tag_salt, 0, plaintext)


def decode_page(q: P2.P2Params, tag_salt: bytes, blocks: Sequence[bytes]) -> bytes:
    """The page's plaintext from its blocks (decode on use: two squarings a block); `ValueError` on a non-codeword."""
    return P2.decode_segment(q, None, tag_salt, 0, blocks)


def keystream(q: P2.P2Params, tag_salt: bytes) -> bytes:
    """The page's keystream as content: each block's mask `K_j` cut to the payload. Content a prover can choose only
    after the coin, which the commit order excludes."""
    prefix, nb, pb = P2.expand_prefix(q, tag_salt), q.block_bytes, q.payload_bytes
    return b"".join(P2.sha512_keys(q, prefix, 0, j)[0].to_bytes(nb, "big")[:pb] for j in range(q.blocks_per_segment))


def page_leaves(q: P2.P2Params, tag_salt: bytes, blocks: Sequence[bytes]) -> list[bytes]:
    return [V.leaf_hash(q, tag_salt, j, b) for j, b in enumerate(blocks)]


def page_root(q: P2.P2Params, tag_salt: bytes, blocks: Sequence[bytes]) -> bytes:
    """The root over the page's blocks' leaves (the package's leaf under the page's tag, RFC 6962's tree)."""
    return V.MerkleTree(page_leaves(q, tag_salt, blocks)).root


def snapshot_root(roots: Sequence[bytes]) -> bytes:
    """A window's snapshot: the root over every page's latest root, in page order."""
    return V.MerkleTree(roots).root


@dataclass
class PageChallenger:
    """The memory challenger's side of the page writes and of a window's reads: it draws one coin per `(page, version)`
    after that version's commit, refusing any other (`ValueError`), and logs every message it sees, in order. Its log is
    what `verdict` judges. `coins(n)` is its live secret, `clock()` its clock in nanoseconds."""

    salt: bytes
    pages: int
    blocks: int
    limit_ns: int
    coins: Callable[[int], bytes] = secrets.token_bytes
    clock: Callable[[], int] = time.perf_counter_ns
    events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _committed: dict[tuple[int, int], bool] = field(default_factory=dict, init=False)

    def commit(self, page: int, version: int, c: bytes) -> None:
        self.events.append({"op": "commit", "page": page, "version": version, "c": c.hex()})
        self._committed.setdefault((page, version), False)

    def coin(self, page: int, version: int) -> bytes:
        if self._committed.get((page, version)) is not False:
            raise ValueError(f"no coin for page {page} version {version}: it is uncommitted or has its coin")
        coin = self.coins(COIN_BYTES)
        self._committed[page, version] = True
        self.events.append({"op": "coin", "page": page, "version": version, "coin": coin.hex()})
        return coin

    def store(self, page: int, version: int, blocks: Sequence[bytes]) -> None:
        self.events.append({"op": "store", "page": page, "version": version, "blocks": [b.hex() for b in blocks]})

    def snapshot(self, root: bytes) -> None:
        self.events.append({"op": "snapshot", "root": root.hex()})

    def read(self, page: int, block: int, answer: Callable[[int, int], tuple[bytes, Sequence[bytes]]]) -> None:
        """One timed spot read: the clock before the challenge goes out and after the whole answer is in."""
        t0 = self.clock()
        got, path = answer(page, block)
        ns = self.clock() - t0
        self.events.append({"op": "read", "page": page, "block": block, "answer": got.hex(),
                            "path": [p.hex() for p in path], "ns": ns})

    def log(self) -> dict[str, Any]:
        return {"salt": self.salt.hex(), "pages": self.pages, "blocks": self.blocks, "limit_ns": self.limit_ns,
                "events": list(self.events)}


@dataclass
class PageStore:
    """The honest prover's KV region: each page's latest encoded version, its tag and its tree, kept; a page is decoded
    on use (`plaintext`). `snapshot` keeps the window's versions, so reads during the window answer from them."""

    salt: bytes
    pages: int
    blocks: int
    versions: dict[int, int] = field(default_factory=dict)
    held: dict[int, tuple[bytes, tuple[bytes, ...], V.MerkleTree]] = field(default_factory=dict)
    window: dict[int, tuple[bytes, tuple[bytes, ...], V.MerkleTree]] = field(default_factory=dict)

    @property
    def q(self) -> P2.P2Params:
        return params(self.blocks)

    def write(self, mc: PageChallenger, page: int, plaintext: bytes) -> None:
        """Commit, take the coin, encode under the tag, store: the page's next version."""
        version = self.versions.get(page, -1) + 1
        mc.commit(page, version, commitment(page, version, plaintext))
        ts = page_salt(self.salt, page, version, mc.coin(page, version))
        blocks = tuple(encode_page(self.q, ts, plaintext))
        mc.store(page, version, blocks)
        self.versions[page] = version
        self.held[page] = (ts, blocks, V.MerkleTree(page_leaves(self.q, ts, blocks)))

    def snapshot(self, mc: PageChallenger) -> None:
        self.window = dict(self.held)
        mc.snapshot(snapshot_root([self.window[p][2].root for p in range(self.pages)]))

    def answer(self, page: int, block: int) -> tuple[bytes, tuple[bytes, ...]]:
        _, blocks, tree = self.window[page]
        return blocks[block], tree.path(block)

    def plaintext(self, page: int) -> bytes:
        ts, blocks, _ = self.held[page]
        return decode_page(self.q, ts, blocks)


def _nat(j: Mapping[str, Any], k: str) -> int:
    v = j.get(k)
    if type(v) is not int or v < 0:
        raise ValueError(f"{k} is missing or not a natural number")
    return v


def _hex(j: Mapping[str, Any], k: str) -> bytes:
    v = j.get(k)
    if not isinstance(v, str) or not _HEX.fullmatch(v):
        raise ValueError(f"{k} is missing or not lowercase hex of whole bytes")
    return bytes.fromhex(v)


def _hexes(j: Mapping[str, Any], k: str) -> list[bytes]:
    v = j.get(k)
    if not isinstance(v, list):
        raise ValueError(f"{k} is missing or not a list")
    return [_hex({"x": x}, "x") for x in v]


@dataclass
class _Version:
    c: bytes
    coin: bytes | None = None
    tag_salt: bytes | None = None
    root: bytes | None = None


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def verdict(log: Mapping[str, Any]) -> dict[str, Any]:
    """The verdict on a memory challenger's log (module docstring); `ValueError` on a log that isn't one."""
    if not isinstance(log, Mapping):
        raise ValueError("a log is an object")
    salt, pages, blocks, limit = _hex(log, "salt"), _nat(log, "pages"), _nat(log, "blocks"), _nat(log, "limit_ns")
    if len(salt) != SALT_BYTES:
        raise ValueError(f"the salt is {SALT_BYTES} bytes")
    if not 1 <= pages <= MAX or not 1 <= blocks <= MAX:
        raise ValueError(f"pages and blocks are 1 to {MAX}")
    events = log.get("events")
    if not isinstance(events, list):
        raise ValueError("events is missing or not a list")
    q = params(blocks)
    vers: dict[int, list[_Version]] = {p: [] for p in range(pages)}
    stored: dict[int, int] = {}
    window: list[int] | None = None
    reads = windows = writes = 0
    for i, e in enumerate(events):
        if not isinstance(e, Mapping):
            raise ValueError("an event is an object")
        op = e.get("op")
        try:
            if op == "commit":
                page, version, c = _nat(e, "page"), _nat(e, "version"), _hex(e, "c")
                if page >= pages:
                    raise _Rejected("page")
                vs = vers[page]
                if version < len(vs):
                    raise _Rejected("recommit")
                if version > len(vs):
                    raise _Rejected("version")
                vs.append(_Version(c))
            elif op == "coin":
                page, version, coin = _nat(e, "page"), _nat(e, "version"), _hex(e, "coin")
                if len(coin) != COIN_BYTES:
                    raise ValueError(f"a coin is {COIN_BYTES} bytes")
                if page >= pages:
                    raise _Rejected("page")
                vs = vers[page]
                if version >= len(vs) or vs[version].coin is not None:
                    raise _Rejected("coin")
                vs[version].coin = coin
            elif op == "store":
                page, version, bs = _nat(e, "page"), _nat(e, "version"), _hexes(e, "blocks")
                if page >= pages:
                    raise _Rejected("page")
                vs = vers[page]
                if version >= len(vs) or vs[version].coin is None:
                    raise _Rejected("early")
                if page in stored and version <= stored[page]:
                    raise _Rejected("restore")
                v = vs[version]
                ts = page_salt(salt, page, version, v.coin)
                try:
                    plain = decode_page(q, ts, bs)
                except ValueError:
                    raise _Rejected("decode") from None
                if commitment(page, version, plain) != v.c:
                    raise _Rejected("decode")
                v.tag_salt, v.root = ts, page_root(q, ts, bs)
                stored[page] = version
                writes += 1
            elif op == "snapshot":
                root = _hex(e, "root")
                if len(stored) < pages or root != snapshot_root([vers[p][stored[p]].root for p in range(pages)]):
                    raise _Rejected("snapshot")
                window = [stored[p] for p in range(pages)]
                windows += 1
            elif op == "read":
                page, j, got, path, ns = (_nat(e, "page"), _nat(e, "block"), _hex(e, "answer"), _hexes(e, "path"),
                                          _nat(e, "ns"))
                if window is None:
                    raise _Rejected("order")
                if page >= pages or j >= blocks:
                    raise _Rejected("page")
                if ns > limit:
                    raise _Rejected("late")
                if len(got) != q.block_bytes:
                    raise _Rejected("wrong")
                v = vers[page][window[page]]
                if not V.verify_path(v.root, blocks, j, V.leaf_hash(q, v.tag_salt, j, got), path):
                    old = vers[page][:window[page]]
                    if any(u.root is not None and V.verify_path(u.root, blocks, j, V.leaf_hash(q, u.tag_salt, j, got),
                                                                 path) for u in old):
                        raise _Rejected("stale")
                    raise _Rejected("wrong")
                reads += 1
            else:
                raise ValueError(f"event {i}'s op is not one of commit, coin, store, snapshot, read")
        except _Rejected as r:
            return {"verdict": "reject", "reason": r.reason, "event": i}
    if reads == 0:
        return {"verdict": "reject", "reason": "reads", "event": len(events)}
    return {"verdict": "accept", "reads": reads, "windows": windows, "writes": writes}


__all__ = ["C0", "COIN_BYTES", "MAX", "REASONS", "SALT_BYTES", "W", "PageChallenger", "PageStore", "commitment",
           "decode_page", "encode_page", "keystream", "page_leaves", "page_root", "page_salt", "params", "snapshot_root",
           "verdict"]
