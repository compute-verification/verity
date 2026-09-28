"""The protocol's lifecycle, the same for every `PoUWScheme` (``PROTOCOL.md``, "Lifecycle").

0. **Coins.**  Before the epoch the challenger publishes its coin commitment over the epoch's coins (`coins`).
1. **Weights.**  Each weight is committed as its rows along k; `weights_root` binds every weight root.
2. **Salt.**  The challenger opens the epoch's salt coin after it recorded the weights' root, and the salt is
   ``derive(coin, "verity/pouw/epoch-salt/v1", {"epoch": e, "coins": root, "weights": weights_root})`` (`Epoch.start`).
3. **Units.**  For each matmul the prover commits the activation rows (bound to the salt and the index); the scheme
   derives its noise from (salt, matmul, roots), and every tile yields checked values and a digest.  The useful output is
   assembled from the tiles.
4. **Transcript.**  A Merkle root over every tile's digest, in matmul order and then tile order.
5. **Selection.**  `Sampled`: ``derive(source, "verity/pouw/audit/v1", {"transcript": root, "salt": salt})`` draws t
   tiles independently, each with probability ``W_ref(tile)/W`` (Lean `WorkWeightedSampling`); on an epoch with coins the
   source is the epoch's draw coin, which the challenger opens after the verifier recorded the transcript.  Served
   accounting uses only this sampled audit.  `Lottery` is Pearl's mining rule, not one of the audit's laws, and stays
   unwired: the tiles whose digest, read little-endian, is at most ``target·|rows|·|cols|·k``.  The prover opens
   each selected tile's strips and leaf; the verifier recomputes the tile through the scheme.
6. **Profile.**  When every selected tile passes, `WorkProfile` states what that establishes.

The protocol owns every outer coin (the salt and the draw), and both are the challenger's live coins.  A scheme's noise is
its own derivation from the salt and the prover's roots, Fiat-Shamir inside the scheme.  Commitments are `verity.ml.commitments.merkle` trees, framed SHA-256 unless the
scheme names another frame hash in ``commitment_hash`` (`commitment_hash`): ``"hm96-sha512"`` (`HM96`) is ``frame-v3-sha512`` with hm96-sha512 leaves (`Hm96Domain`): each committed value is its preimage
followed by a 192-byte salt, a row's leaf ``hm96-sha512/row-seg/v1`` over the row's ``sha512/row-seg/v1`` digest and a tile's
over its 64-byte digest itself; the prover's salts are a ChaCha20 stream under its own key (`hm96_salt`), and an opening's
value carries its salt, so `opened_value` strips it before the scheme reads the value.
"""

from __future__ import annotations

import functools
import os
import struct
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate
from math import lcm

from verity.errors import ProtocolError
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits, verify_opening
from verity.ml.commitments import hm96
from verity.ml.commitments.gates import sha512 as sha512_gates
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, sha512_row_seg_digest
from verity.ml.randomness import derive

from .coins import Challenger, CoinCommitment, CoinOpening, coin_reason
from .protocol import Certificate, Context, DomainError, Matmul, PoUWScheme, Tile, tagged

SALT_DOMAIN = "verity/pouw/salt/v1"
EPOCH_SALT_DOMAIN = "verity/pouw/epoch-salt/v1"
AUDIT_DOMAIN = "verity/pouw/audit/v1"
ROW_SCHEMA = "verity/pouw/row/v1"
TILE_SCHEMA = "verity/pouw/tile/v1"
#: ``frame-v3-sha512`` with hm96-sha512 leaves (module docstring)
HM96 = "hm96-sha512"
HM96_KEY = hm96.SHA512.default_key
HM96_SALT = hm96.SHA512.salt_bytes
#: the hm96 salt stream's nonce of each tree: an activation tree's is its unit index, below 2^62
HM96_WEIGHTS, HM96_TRANSCRIPT = 1 << 62, 1 << 63
_M32 = 0xFFFFFFFF


def commitment_hash(scheme: PoUWScheme) -> str:
    """The frame hash of a scheme's row and tile trees: its ``commitment_hash``, SHA-256 by default."""
    return getattr(scheme, "commitment_hash", "sha256")


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
    tables = b"".join(v.to_bytes(64, "little") for table in hm96.SHA512.byte_tables(HM96_KEY) for v in table)
    return b"".join(struct.pack("<8Q", *m) for m in mids) + tables


@dataclass(frozen=True, slots=True)
class Hm96Domain(CommitmentDomain):
    """A ``frame-v3-sha512`` domain whose leaves are hm96-sha512 leaves under the pinned key: a value is its preimage followed
    by its 192-byte salt y, x the preimage's ``sha512/row-seg/v1`` digest (a row) or, with ``digests``, the 64-byte preimage
    itself (a tile's digest), and the frame leaf is ``rowleaf.row_leaf``'s over ``hm96.SHA512.tree_leaf(b ‖ c)`` under
    ``hm96-sha512/row-seg/v1``.  ``schema`` is not read: the domain's binding separates its trees."""

    digests: bool = False

    def leaf(self, rank: int, position: int, schema: str, value: bytes) -> bytes:
        if len(value) < HM96_SALT or self.digests and len(value) != 64 + HM96_SALT:
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
    return tagged("verity/pouw/weight-rows/v1", weight)


def activation_binding(salt: bytes, index: int) -> bytes:
    return tagged("verity/pouw/activation-rows/v1", salt, index)


def commit_rows(binding: bytes, owner: int, rows: Sequence[bytes], hash: str = "sha256", salt_key: bytes | None = None,
                nonce: int = 0, digests: bool = False) -> MerkleTree:
    """`rows` in one tree; under `HM96` each value is followed by its salt, `hm96_salt(salt_key, nonce, i)`."""
    if hash == HM96:
        if salt_key is None:
            raise ProtocolError(f"an {HM96} tree takes the prover's salt key")
        rows = [r + hm96_salt(salt_key, nonce, i) for i, r in enumerate(rows)]
    return MerkleTree(rows_domain(binding, owner, len(rows), hash, digests), dict(enumerate(rows)), lambda _: ROW_SCHEMA)


def weights_root(weight_roots: Mapping[int, bytes]) -> bytes:
    return tagged("verity/pouw/weights/v1", *(p for w in sorted(weight_roots) for p in (w, weight_roots[w])))


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


def _check_weights(workload: Workload, weight_roots: Mapping[int, bytes]) -> None:
    if set(weight_roots) != set(workload.weights()):
        raise DomainError("every weight of the workload is committed, and only those")


def salt_of(coins: CoinCommitment, coin: bytes, weights_root: bytes) -> bytes:
    """The epoch's 32-byte salt from its opened salt coin."""
    return derive(coin, EPOCH_SALT_DOMAIN, {"epoch": coins.epoch, "coins": coins.root, "weights": weights_root}
                  ).stream("salt", 32)


def draw_coin(epoch: Epoch, challenger: Challenger, after: bytes) -> tuple[bytes | None, str | None]:
    """The epoch's draw coin, asked of the challenger once the verifier recorded `after`, and checked against the
    epoch's coin commitment: (coin, None), or (None, why it is refused)."""
    if epoch.coins is None:
        return None, "the epoch has no coin commitment for its draw coin to open against"
    opening = challenger.open("draw", after)
    reason = coin_reason(epoch.coins, opening, "draw", after)
    return (None, f"the draw's {reason}") if reason is not None else (opening.coin, None)


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
    return tagged("verity/pouw/transcript/v1", epoch.salt, epoch.weights_root)


def transcript_domain(epoch: Epoch, count: int, hash: str = "sha256") -> CommitmentDomain:
    return rows_domain(transcript_binding(epoch), -1, count, hash, digests=True)


@dataclass(frozen=True, slots=True)
class Transcript:
    """What the prover publishes: every activation root and the transcript commitment; the useful outputs go to the
    workload."""

    activation_roots: tuple[bytes, ...]
    commitment: Commitment
    useful: tuple[list[list], ...] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class TileOpening:
    ref: TileRef
    a: tuple[Opening, ...]
    b: tuple[Opening, ...]
    leaf: Opening


def reference_kernel(scheme: PoUWScheme, ctx: Context, noise, a: Sequence[bytes], b: Sequence[bytes],
                     tiles: Sequence[Tile]) -> list[tuple[bytes, list[list]]]:
    """One matmul through the scheme's reference, tile by tile: ``[(digest, useful tile)]``."""
    out = []
    for t in tiles:
        ops = scheme.encode(ctx, noise, [a[i] for i in t.rows], [b[j] for j in t.cols], t)
        if ops is None:
            raise DomainError(f"matmul {ctx.matmul.index}: a row of tile {t.rows[0]}, {t.cols[0]} is not a well-formed operand row")
        c = scheme.checked(ctx, ops)
        out.append((scheme.digest(ctx, noise, c), scheme.useful(ctx, noise, ops, c)))
    return out


class Prover:
    """The honest prover.  `kernel` computes one matmul's tiles (default: the scheme's reference, `reference_kernel`); a
    fast implementation replaces it and must agree with the reference on the scheme's vectors."""

    def __init__(self, scheme: PoUWScheme, weights: Mapping[int, Sequence[bytes]],
                 kernel: Callable[..., list[tuple[bytes, list[list]]]] = reference_kernel, salt_key: bytes | None = None):
        """`salt_key` is the prover's key of its hm96 salts under `HM96` (fresh by default)."""
        self.scheme, self.kernel = scheme, kernel
        self.weights = {w: list(rows) for w, rows in weights.items()}
        self.hash = commitment_hash(scheme)
        self.salt_key = (salt_key or os.urandom(32)) if self.hash == HM96 else None
        self.weight_trees = {w: commit_rows(weight_binding(w), -2, rows, self.hash, self.salt_key, HM96_WEIGHTS + w)
                             for w, rows in self.weights.items()}
        self._act_trees: list[MerkleTree] = []
        self._refs: list[TileRef] = []
        self._tree: MerkleTree | None = None

    @property
    def weight_roots(self) -> dict[int, bytes]:
        return {w: t.commitment.root for w, t in self.weight_trees.items()}

    def prove(self, epoch: Epoch, activations: Mapping[int, Sequence[bytes]]) -> Transcript:
        scheme, work = self.scheme, epoch.workload
        for u in work.matmuls:
            scheme.check(u)
        self._refs, self._act_trees = tile_refs(scheme, work), []
        leaves: list[bytes] = []
        outputs: list[list[list]] = []
        for u in work.matmuls:
            rows = list(activations[u.index])
            if len(rows) != u.shape.m:
                raise DomainError(f"matmul {u.index} has {u.shape.m} activation rows")
            tree = commit_rows(activation_binding(epoch.salt, u.index), u.index, rows, self.hash, self.salt_key, u.index)
            self._act_trees.append(tree)
            ctx = Context(epoch.salt, u, tree.commitment.root, epoch.weight_roots[u.weight])
            tiles = scheme.tiles(u.shape)
            results = self.kernel(scheme, ctx, scheme.noise(ctx), rows, self.weights[u.weight], tiles)
            out = [[None] * u.shape.n for _ in range(u.shape.m)]
            for t, (d, block) in zip(tiles, results, strict=True):
                leaves.append(d)
                for bi, i in enumerate(t.rows):
                    for bj, j in enumerate(t.cols):
                        out[i][j] = block[bi][bj]
            outputs.append(out)
        if self.hash == HM96:
            leaves = [d + hm96_salt(self.salt_key, HM96_TRANSCRIPT, i) for i, d in enumerate(leaves)]
        self._tree = MerkleTree(transcript_domain(epoch, len(leaves), self.hash), dict(enumerate(leaves)), lambda _: TILE_SCHEMA)
        return Transcript(tuple(t.commitment.root for t in self._act_trees), self._tree.commitment, tuple(outputs))

    def digest(self, tile_id: int) -> bytes:
        return opened_value(self.hash, self._tree.open(tile_id).value)

    def open(self, epoch: Epoch, tile_id: int) -> TileOpening:
        ref = self._refs[tile_id]
        u = epoch.workload.matmuls[ref.matmul]
        a, b = self._act_trees[ref.matmul], self.weight_trees[u.weight]
        return TileOpening(ref, tuple(a.open(i) for i in ref.tile.rows), tuple(b.open(j) for j in ref.tile.cols),
                           self._tree.open(tile_id))


def tickets(scheme: PoUWScheme, workload: Workload, refs: Sequence[TileRef]) -> list[int]:
    """Integer tickets proportional to each tile's share of its matmul's W_ref (its cells over m·n)."""
    den = lcm(*(u.shape.m * u.shape.n for u in workload.matmuls)) if workload.matmuls else 1
    wref = [scheme.wref(u.shape) for u in workload.matmuls]
    return [wref[r.matmul] * r.tile.cells * (den // (workload.matmuls[r.matmul].shape.m * workload.matmuls[r.matmul].shape.n))
            for r in refs]


@dataclass(frozen=True, slots=True)
class Sampled:
    """t independent work-weighted draws with replacement (Lean `WorkWeightedSampling`: δ_s = (1 − ε_s)^t)."""

    t: int

    def select(self, scheme: PoUWScheme, epoch: Epoch, transcript: Transcript, source: bytes, refs: Sequence[TileRef],
               digest: Callable[[int], bytes]) -> tuple[int, ...]:
        key = derive(source, AUDIT_DOMAIN, {"transcript": transcript.commitment.root, "salt": epoch.salt})
        cumulative = list(accumulate(tickets(scheme, epoch.workload, refs)))
        return tuple(bisect_right(cumulative, key.uniform(cumulative[-1], j)) for j in range(self.t))


@dataclass(frozen=True, slots=True)
class Lottery:
    """Pearl's rule: a tile wins when its digest, read little-endian, is at most ``target·|rows|·|cols|·k``."""

    target: int

    def wins(self, digest: bytes, cost: int) -> bool:
        return int.from_bytes(digest, "little") <= self.target * cost

    def select(self, scheme: PoUWScheme, epoch: Epoch, transcript: Transcript, source: bytes, refs: Sequence[TileRef],
               digest: Callable[[int], bytes]) -> tuple[int, ...]:
        mm = epoch.workload.matmuls
        return tuple(r.id for r in refs if self.wins(digest(r.id), r.tile.cells * mm[r.matmul].shape.k))


@dataclass(frozen=True, slots=True)
class WorkProfile:
    """What an accepted audit establishes, by Lean `EndToEnd`: for every ε_s in (0, 1] and every operator whose online
    cost is at most T, ``Pr[accept ∧ T < (1 − γ)(1 − ε_s)·W] ≤ (1 − ε_s)^t + η``, under the certificate's conjecture and
    hypotheses, with W the workload's total W_ref.  Without a γ only the sampling half holds: the incorrect tiles carry at
    most ε_s·W, except with probability (1 − ε_s)^t.  A lottery selection gives neither bound."""

    scheme: str
    work: int
    drawn: int
    sampled: bool
    certificate: Certificate

    def delta_s(self, eps_s: Fraction) -> Fraction | None:
        if not 0 < eps_s <= 1:
            raise ValueError("0 < eps_s <= 1")
        return (1 - Fraction(eps_s)) ** self.drawn if self.sampled else None

    def certified(self, eps_s: Fraction) -> Fraction | None:
        """The work certified, (1 − γ)(1 − ε_s)·W, or None without a γ or a sampled audit."""
        g = self.certificate.gamma
        return None if g is None or not self.sampled else (1 - g) * (1 - Fraction(eps_s)) * self.work

    def to_json(self) -> dict:
        c = self.certificate
        return {"scheme": self.scheme, "work": self.work, "drawn": self.drawn, "sampled": self.sampled,
                "conjecture": c.conjecture, "gamma0": None if c.gamma0 is None else str(c.gamma0), "omega": str(c.omega),
                "gamma": None if c.gamma is None else str(c.gamma), "claims": list(c.claims), "theorems": list(c.theorems),
                "hypotheses": list(c.hypotheses), "allowances": list(c.allowances), "parameters": dict(c.parameters)}


@dataclass(frozen=True, slots=True)
class Verdict:
    tile: int
    ok: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Audit:
    accepted: bool
    selected: tuple[int, ...]
    verdicts: tuple[Verdict, ...]
    profile: WorkProfile | None


class Verifier:
    def __init__(self, scheme: PoUWScheme, epoch: Epoch, limits: VerificationLimits | None = None):
        self.scheme, self.epoch = scheme, epoch
        self.limits = limits or VerificationLimits()
        for u in epoch.workload.matmuls:
            scheme.check(u)
        self.refs = tile_refs(scheme, epoch.workload)

    @property
    def work(self) -> int:
        return sum(self.scheme.wref(u.shape) for u in self.epoch.workload.matmuls)

    def check_tile(self, transcript: Transcript, opening: TileOpening) -> Verdict:
        scheme, ep, ref = self.scheme, self.epoch, opening.ref
        if not 0 <= ref.id < len(self.refs) or ref != self.refs[ref.id]:
            return Verdict(ref.id, False, "tile does not match the verifier's own enumeration")
        u = ep.workload.matmuls[ref.matmul]
        root_a, root_b = transcript.activation_roots[ref.matmul], ep.weight_roots[u.weight]
        h = commitment_hash(scheme)
        da = rows_domain(activation_binding(ep.salt, u.index), u.index, u.shape.m, h)
        db = rows_domain(weight_binding(u.weight), -2, u.shape.n, h)
        if [o.position for o in opening.a] != list(ref.tile.rows) or [o.position for o in opening.b] != list(ref.tile.cols):
            return Verdict(ref.id, False, "openings do not cover the tile")

        def opens(dom: CommitmentDomain, c: Commitment, o: Opening, schema: str) -> bool:
            try:
                return verify_opening(dom, c, o, schema, self.limits)
            except ProtocolError:
                return False

        if not all(opens(da, Commitment(root_a, u.shape.m), o, ROW_SCHEMA) for o in opening.a):
            return Verdict(ref.id, False, "activation opening")
        if not all(opens(db, Commitment(root_b, u.shape.n), o, ROW_SCHEMA) for o in opening.b):
            return Verdict(ref.id, False, "weight opening")
        if opening.leaf.position != ref.id or not opens(transcript_domain(ep, len(self.refs), h), transcript.commitment,
                                                        opening.leaf, TILE_SCHEMA):
            return Verdict(ref.id, False, "transcript opening")
        ctx = Context(ep.salt, u, root_a, root_b)
        noise = scheme.noise(ctx)
        ops = scheme.encode(ctx, noise, [opened_value(h, o.value) for o in opening.a], [opened_value(h, o.value) for o in opening.b],
                            ref.tile)
        if ops is None or not scheme.admissible(ctx, ops):
            return Verdict(ref.id, False, "inadmissible inputs")
        if scheme.digest(ctx, noise, scheme.checked(ctx, ops)) != opened_value(h, opening.leaf.value):
            return Verdict(ref.id, False, "checked values")
        return Verdict(ref.id, True)

    def audit(self, prover: Prover, transcript: Transcript, selection: Sampled | Lottery, source: bytes | Challenger
              ) -> Audit:
        """Draw, open and check the selected tiles.  On an epoch with a coin commitment `source` is the challenger, asked
        for the epoch's draw coin after the transcript is recorded; on one without, the draw's bare source bytes."""
        if len(transcript.activation_roots) != len(self.epoch.workload.matmuls):
            return Audit(False, (), (), None)
        if self.epoch.coins is not None or not isinstance(source, bytes):
            source, reason = ((None, "an epoch with a coin commitment draws from its draw coin, not from bare bytes")
                              if isinstance(source, bytes) else draw_coin(self.epoch, source, transcript.commitment.root))
            if reason is not None:
                return Audit(False, (), (Verdict(-1, False, reason),), None)
        selected = selection.select(self.scheme, self.epoch, transcript, source, self.refs, prover.digest)
        verdicts = tuple(self.check_tile(transcript, prover.open(self.epoch, tid)) for tid in selected)
        ok = all(v.ok for v in verdicts)
        profile = (WorkProfile(self.scheme.name, self.work, len(selected), isinstance(selection, Sampled), self.scheme.certificate)
                   if ok else None)
        return Audit(ok, selected, verdicts, profile)


__all__ = ["AUDIT_DOMAIN", "EPOCH_SALT_DOMAIN", "HM96", "SALT_DOMAIN", "Audit", "Epoch", "Hm96Domain", "Lottery", "Prover", "Sampled",
           "TileOpening", "TileRef", "Transcript", "Verdict", "Verifier", "WorkProfile", "Workload", "commit_rows", "commitment_hash",
           "draw_coin", "hm96_kernel_constants", "hm96_salt", "opened_value", "reference_kernel", "salt_of", "tickets", "tile_refs"]
