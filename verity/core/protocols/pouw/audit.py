"""The protocol's commitments and epoch, the same for every `PoUWScheme` (``PROTOCOL.md``, "Lifecycle"): what the served
path's prover commits and its verifier reads.

0. **Coins.**  Before the epoch the challenger publishes its coin commitment over the epoch's coins (`coins`).
1. **Weights.**  Each weight is committed as its rows along k; `weights_root` binds every weight root.
2. **Salt.**  The challenger opens the epoch's salt coin after it recorded the weights' root, and the salt is
   ``derive(coin, "verity/pouw/epoch-salt/v1", {"epoch": e, "coins": root, "weights": weights_root})`` (`Epoch.start`).
3. **Units.**  For each matmul the prover commits the activation rows, bound to the salt and the index
   (`activation_binding`, `commit_rows`).
4. **Transcript.**  A Merkle root over every tile's digest (`transcript_domain`), in matmul order and then tile order
   (`tile_refs`).

The clear audit that drew, opened and recomputed tiles in the clear (`Prover`, `Verifier`, `Sampled`, `WorkProfile`) is a
reference tool, not a verifier: ``benchmarks/pouw/clear/clear_audit.py``.  Served windows are checked by PoUW's verifier of
record, the Lean (`lean/check`, `core_check`).

Commitments are `verity.ml.commitments.merkle` trees, framed SHA-256 unless a scheme names another frame hash:
``"hm96-sha512"`` (`HM96`) is ``frame-v3-sha512`` with hm96-sha512 leaves (`Hm96Domain`): each committed value is its preimage
followed by a 192-byte salt, a row's leaf ``hm96-sha512/row-seg/v1`` over the row's ``sha512/row-seg/v1`` digest and a tile's
over its 64-byte digest itself; the prover's salts are a ChaCha20 stream under its own key (`hm96_salt`), and an opening's
value carries its salt, so `opened_value` strips it before a scheme reads the value.
"""

from __future__ import annotations

import functools
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.core.protocols.pouw.salt import pouw_salt, pouw_window
from verity.errors import ProtocolError
from verity.ml.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.ml.commitments import domains, hm96
from verity.ml.commitments.gates import sha512 as sha512_gates
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, sha512_row_seg_digest
from verity.ml.commitments.scheme import DIGEST_BYTES
from verity.ml.randomness import derive

from .coins import CoinCommitment, CoinOpening, coin_reason
from .protocol import DomainError, Matmul, PoUWScheme, Tile, tagged

SALT_DOMAIN = domains.POUW_UNVERIFIED_SALT
EPOCH_SALT_DOMAIN = domains.POUW_EPOCH_SALT
ROW_SCHEMA = domains.POUW_AUDIT_ROW
TILE_SCHEMA = domains.POUW_AUDIT_TILE
#: ``frame-v3-sha512`` with hm96-sha512 leaves (module docstring)
HM96 = "hm96-sha512"
HM96_KEY = hm96.SHA512.default_key
HM96_SALT = hm96.SHA512.salt_bytes
#: the hm96 salt stream's nonce of each tree: an activation tree's is its unit index, below 2^62
HM96_WEIGHTS, HM96_TRANSCRIPT = 1 << 62, 1 << 63
_M32 = 0xFFFFFFFF


def chacha20_block(key: bytes, counter: int, nonce: int) -> bytes:
    """ChaCha20's block function, 20 rounds, with a 64-bit counter (words 12, 13) and a 64-bit nonce (words 14, 15): C-Flock's
    ``chacha20_block`` (``verity/ml/flock/cuda/sha512.cuh``), the stream its hm96 leaf salts are drawn from."""
    s = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *struct.unpack("<8I", key),
         counter & _M32, counter >> 32, nonce & _M32, nonce >> 32]
    x = list(s)

    def qr(a: int, b: int, c: int, d: int) -> None:
        for p, q, r, n in ((a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)):
            x[p] = (x[p] + x[q]) & _M32
            x[r] ^= x[p]
            x[r] = ((x[r] << n) | (x[r] >> (32 - n))) & _M32

    for _ in range(10):
        qr(0, 4, 8, 12), qr(1, 5, 9, 13), qr(2, 6, 10, 14), qr(3, 7, 11, 15)
        qr(0, 5, 10, 15), qr(1, 6, 11, 12), qr(2, 7, 8, 13), qr(3, 4, 9, 14)
    return struct.pack("<16I", *[(x[i] + s[i]) & _M32 for i in range(16)])


def hm96_salt(key: bytes, nonce: int, i: int) -> bytes:
    """Leaf i's 192-byte salt in the tree with salt nonce `nonce`: ChaCha20 blocks 3i, 3i + 1 and 3i + 2 under the prover's
    32-byte `key` (C-Flock's ``hm96_salt``)."""
    return b"".join(chacha20_block(key, 3 * i + j, nonce) for j in range(3))


@functools.lru_cache(maxsize=1)
def hm96_kernel_constants() -> bytes:
    """`HM96`'s constants as a kernel reads them (C-Flock's ``hm96_finish_leaves``, pearl_c's ``hm_leaves``): the midstates
    after the salt prefix and after the leaf prefix (eight u64 little-endian each), then the byte tables, per salt byte position
    and byte value the 64-byte column XOR little-endian (3 MiB)."""
    mids = [sha512_gates.midstate(p) for p in (hm96.SHA512.salt_prefix, hm96.SHA512.leaf_prefix(HM96_KEY))]
    tables = b"".join(v.to_bytes(DIGEST_BYTES["sha512"], "little")
                      for table in hm96.SHA512.byte_tables(HM96_KEY) for v in table)
    return b"".join(struct.pack("<8Q", *m) for m in mids) + tables


@dataclass(frozen=True, slots=True)
class Hm96Domain(CommitmentDomain):
    """A ``frame-v3-sha512`` domain whose leaves are hm96-sha512 leaves under the pinned key: a value is its preimage followed
    by its 192-byte salt y, x the preimage's ``sha512/row-seg/v1`` digest (a row) or, with ``digests``, the 64-byte preimage
    itself (a tile's digest), and the frame leaf is ``rowleaf.row_leaf``'s over ``hm96.SHA512.tree_leaf(b ‖ c)`` under
    ``hm96-sha512/row-seg/v1``.  ``schema`` is not read: the domain's binding separates its trees."""

    digests: bool = False

    def leaf(self, rank: int, position: int, schema: str, value: bytes) -> bytes:
        if len(value) < HM96_SALT or self.digests and len(value) != DIGEST_BYTES["sha512"] + HM96_SALT:
            raise ProtocolError(f"an {HM96} value is its preimage and a {HM96_SALT}-byte salt")
        pre, y = value[:-HM96_SALT], value[-HM96_SALT:]
        x = pre if self.digests else sha512_row_seg_digest(pre, 8 * len(pre))
        hl = hm96.SHA512.tree_leaf(HM96_KEY, hm96.SHA512.commit_string(HM96_KEY, x, y))
        return CommitmentDomain.leaf(self, rank, position, SCHEMA_HM96_SHA512_ROW_SEG, hl)


def opened_value(hash: str, value: bytes) -> bytes:
    """What a scheme reads of a committed value: the value, less its salt under `HM96`."""
    return value[:-HM96_SALT] if hash == HM96 else value


def rows_domain(binding: bytes, owner: int, count: int, hash: str = "sha256", digests: bool = False) -> CommitmentDomain:
    if hash == HM96:
        return Hm96Domain(binding, owner, RangeIndexedDomain(count), hash="sha512", digests=digests)
    return CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash=hash)


def weight_binding(weight: int) -> bytes:
    return tagged(domains.POUW_WEIGHT_ROWS, weight)


def activation_binding(salt: bytes, index: int) -> bytes:
    return tagged(domains.POUW_ACTIVATION_ROWS, salt, index)


def commit_rows(binding: bytes, owner: int, rows: Sequence[bytes], hash: str = "sha256", salt_key: bytes | None = None,
                nonce: int = 0, digests: bool = False) -> MerkleTree:
    """`rows` in one tree; under `HM96` each value is followed by its salt, `hm96_salt(salt_key, nonce, i)`."""
    if hash == HM96:
        if salt_key is None:
            raise ProtocolError(f"an {HM96} tree takes the prover's salt key")
        rows = [r + hm96_salt(salt_key, nonce, i) for i, r in enumerate(rows)]
    return MerkleTree(rows_domain(binding, owner, len(rows), hash, digests), dict(enumerate(rows)), lambda _: ROW_SCHEMA)


def weights_root(weight_roots: Mapping[int, bytes]) -> bytes:
    return tagged(domains.POUW_WEIGHTS_ROOT, *(p for w in sorted(weight_roots) for p in (w, weight_roots[w])))


@dataclass(frozen=True, slots=True)
class Workload:
    matmuls: tuple[Matmul, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matmuls", tuple(self.matmuls))
        if len({u.index for u in self.matmuls}) != len(self.matmuls):
            raise DomainError("matmul indices are unique")
        dims: dict[int, tuple[int, int]] = {}
        for u in self.matmuls:
            if dims.setdefault(u.weight, (u.shape.n, u.shape.k)) != (u.shape.n, u.shape.k):
                raise DomainError(f"every matmul on weight {u.weight} has the weight's (n, k)")

    def weights(self) -> dict[int, tuple[int, int]]:
        """Weight id -> (n, k)."""
        return {u.weight: (u.shape.n, u.shape.k) for u in self.matmuls}


@dataclass(frozen=True, slots=True)
class Epoch:
    """The committed weights, the salt drawn after them, and the challenger's coin commitment the epoch's coins open
    against (None for an `unverified` epoch's salt with no draw coin)."""

    workload: Workload
    weight_roots: Mapping[int, bytes]
    salt: bytes
    coins: CoinCommitment | None = None

    @property
    def weights_root(self) -> bytes:
        return weights_root(self.weight_roots)

    @classmethod
    def start(cls, workload: Workload, weight_roots: Mapping[int, bytes], coins: CoinCommitment, opening: CoinOpening
              ) -> Epoch:
        """The epoch whose salt is the challenger's salt coin: `opening` must be epoch `coins.epoch`'s salt coin, opened
        against `coins` (the commitment the challenger published before the epoch) after it recorded the weights' root."""
        _check_weights(workload, weight_roots)
        root = weights_root(weight_roots)
        reason = coin_reason(coins, opening, "salt", root)
        if reason is not None:
            raise DomainError(f"the salt's {reason}")
        return cls(workload, dict(weight_roots), salt_of(coins, opening.coin, root), coins)

    @classmethod
    def unverified(cls, workload: Workload, weight_roots: Mapping[int, bytes], seed: bytes,
                   coins: CoinCommitment | None = None) -> Epoch:
        """The epoch whose salt is keyed by bare bytes under ``verity/pouw/salt/v1``, with no coin behind it: test
        instances, replays of recorded runs, and integrations whose salt claims nothing.  `coins`, when given, is the
        commitment the epoch's draw coin opens against."""
        _check_weights(workload, weight_roots)
        salt = derive(seed, SALT_DOMAIN, {"weights": weights_root(weight_roots)}).stream("salt", 32)
        return cls(workload, dict(weight_roots), salt, coins)

    @classmethod
    def certified(cls, workload: Workload, weight_roots: Mapping[int, bytes], windows: Sequence[int], k: int,
                  reveal: bytes, stream: bytes, prover: bytes, coins: CoinCommitment | None = None) -> Epoch:
        """The epoch of PoUW window p, the network certifier's windows `windows` with k per PoUW window (p·k … p·k + k − 1,
        refused unless whole and aligned at 0: `salt.pouw_window`), registered under `stream` and `prover`, whose salt is
        `salt.pouw_salt(reveal, p, stream, prover)`, 64 bytes: `reveal` is p's certifier salt as `salt.revealed` read it.
        `coins`, when given, is the commitment the epoch's draw coin opens against."""
        _check_weights(workload, weight_roots)
        return cls(workload, dict(weight_roots), pouw_salt(reveal, pouw_window(windows, k), stream, prover), coins)


def _check_weights(workload: Workload, weight_roots: Mapping[int, bytes]) -> None:
    if set(weight_roots) != set(workload.weights()):
        raise DomainError("every weight of the workload is committed, and only those")


def salt_of(coins: CoinCommitment, coin: bytes, weights_root: bytes) -> bytes:
    """The epoch's 32-byte salt from its opened salt coin.  A live window's salt coin opens after the window's open root
    (`salt.window_open_root`), and its salt is still keyed by `weights_root`, the weights' root: the coin and `coins` are
    its own session's."""
    return derive(coin, EPOCH_SALT_DOMAIN, {"epoch": coins.epoch, "coins": coins.root, "weights": weights_root}
                  ).stream("salt", 32)


@dataclass(frozen=True, slots=True)
class TileRef:
    """Global tile number `id`: tile `tile` of the matmul at workload position `matmul`."""

    id: int
    matmul: int
    tile: Tile


def tile_refs(scheme: PoUWScheme, workload: Workload) -> list[TileRef]:
    out: list[TileRef] = []
    for pos, u in enumerate(workload.matmuls):
        for t in scheme.tiles(u.shape):
            out.append(TileRef(len(out), pos, t))
    return out


def transcript_binding(epoch: Epoch) -> bytes:
    return tagged(domains.POUW_TRANSCRIPT, epoch.salt, epoch.weights_root)


def transcript_domain(epoch: Epoch, count: int, hash: str = "sha256") -> CommitmentDomain:
    return rows_domain(transcript_binding(epoch), -1, count, hash, digests=True)


__all__ = ["EPOCH_SALT_DOMAIN", "HM96", "HM96_KEY", "HM96_SALT", "HM96_TRANSCRIPT", "HM96_WEIGHTS", "ROW_SCHEMA", "SALT_DOMAIN",
           "TILE_SCHEMA", "Epoch", "Hm96Domain", "TileRef", "Workload", "activation_binding", "chacha20_block", "commit_rows",
           "hm96_kernel_constants", "hm96_salt", "opened_value", "rows_domain", "salt_of", "tile_refs", "transcript_binding",
           "transcript_domain", "weight_binding", "weights_root"]
