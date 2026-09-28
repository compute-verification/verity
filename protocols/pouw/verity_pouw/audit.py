"""The protocol's lifecycle, the same for every `PoUWScheme` (``PROTOCOL.md``, "Lifecycle").

1. **Weights.**  Each weight is committed as its rows along k; `weights_root` binds every weight root.
2. **Salt.**  ``derive(beacon, "verity/pouw/salt/v1", {"weights": weights_root})``, drawn after the weights.
3. **Units.**  For each matmul the prover commits the activation rows (bound to the salt and the index); the scheme
   derives its noise from (salt, matmul, roots), and every tile yields checked values and a digest.  The useful output is
   assembled from the tiles.
4. **Transcript.**  A Merkle root over every tile's digest, in matmul order and then tile order.
5. **Selection.**  `Sampled`: ``derive(source, "verity/pouw/audit/v1", {"transcript": root, "salt": salt})`` draws t
   tiles independently, each with probability ``W_ref(tile)/W`` (Lean `WorkWeightedSampling`).  `Lottery`: the tiles whose
   digest, read little-endian, is at most ``target·|rows|·|cols|·k`` (Pearl's mining rule).  The prover opens each
   selected tile's strips and leaf; the verifier recomputes the tile through the scheme.
6. **Profile.**  When every selected tile passes, `WorkProfile` states what that establishes.

The protocol owns every outer coin (the salt and the draw).  A scheme's noise is its own derivation from the salt and the
prover's roots, Fiat-Shamir inside the scheme.  Commitments are `verity.commitments.merkle` trees.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate
from math import lcm

from verity.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits, verify_opening
from verity.randomness import derive

from .protocol import Certificate, Context, DomainError, Matmul, PoUWScheme, Tile, tagged

SALT_DOMAIN = "verity/pouw/salt/v1"
AUDIT_DOMAIN = "verity/pouw/audit/v1"
ROW_SCHEMA = "verity/pouw/row/v1"
TILE_SCHEMA = "verity/pouw/tile/v1"


def rows_domain(binding: bytes, owner: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(binding, owner, RangeIndexedDomain(count))


def weight_binding(weight: int) -> bytes:
    return tagged("verity/pouw/weight-rows/v1", weight)


def activation_binding(salt: bytes, index: int) -> bytes:
    return tagged("verity/pouw/activation-rows/v1", salt, index)


def commit_rows(binding: bytes, owner: int, rows: Sequence[bytes]) -> MerkleTree:
    return MerkleTree(rows_domain(binding, owner, len(rows)), dict(enumerate(rows)), lambda _: ROW_SCHEMA)


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
    """The committed weights and the salt drawn after them."""

    workload: Workload
    weight_roots: Mapping[int, bytes]
    salt: bytes

    @property
    def weights_root(self) -> bytes:
        return weights_root(self.weight_roots)

    @classmethod
    def start(cls, workload: Workload, weight_roots: Mapping[int, bytes], beacon: bytes) -> Epoch:
        if set(weight_roots) != set(workload.weights()):
            raise DomainError("every weight of the workload is committed, and only those")
        salt = derive(beacon, SALT_DOMAIN, {"weights": weights_root(weight_roots)}).stream("salt", 32)
        return cls(workload, dict(weight_roots), salt)


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


def transcript_domain(epoch: Epoch, count: int) -> CommitmentDomain:
    return rows_domain(tagged("verity/pouw/transcript/v1", epoch.salt, epoch.weights_root), -1, count)


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
                 kernel: Callable[..., list[tuple[bytes, list[list]]]] = reference_kernel):
        self.scheme, self.kernel = scheme, kernel
        self.weights = {w: list(rows) for w, rows in weights.items()}
        self.weight_trees = {w: commit_rows(weight_binding(w), -2, rows) for w, rows in self.weights.items()}
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
            tree = commit_rows(activation_binding(epoch.salt, u.index), u.index, rows)
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
        self._tree = MerkleTree(transcript_domain(epoch, len(leaves)), dict(enumerate(leaves)), lambda _: TILE_SCHEMA)
        return Transcript(tuple(t.commitment.root for t in self._act_trees), self._tree.commitment, tuple(outputs))

    def digest(self, tile_id: int) -> bytes:
        return self._tree.open(tile_id).value

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
                "hypotheses": list(c.hypotheses), "allowances": list(c.allowances)}


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
        da = rows_domain(activation_binding(ep.salt, u.index), u.index, u.shape.m)
        db = rows_domain(weight_binding(u.weight), -2, u.shape.n)
        if [o.position for o in opening.a] != list(ref.tile.rows) or [o.position for o in opening.b] != list(ref.tile.cols):
            return Verdict(ref.id, False, "openings do not cover the tile")
        if not all(verify_opening(da, Commitment(root_a, u.shape.m), o, ROW_SCHEMA, self.limits) for o in opening.a):
            return Verdict(ref.id, False, "activation opening")
        if not all(verify_opening(db, Commitment(root_b, u.shape.n), o, ROW_SCHEMA, self.limits) for o in opening.b):
            return Verdict(ref.id, False, "weight opening")
        if opening.leaf.position != ref.id or not verify_opening(transcript_domain(ep, len(self.refs)), transcript.commitment,
                                                                 opening.leaf, TILE_SCHEMA, self.limits):
            return Verdict(ref.id, False, "transcript opening")
        ctx = Context(ep.salt, u, root_a, root_b)
        noise = scheme.noise(ctx)
        ops = scheme.encode(ctx, noise, [o.value for o in opening.a], [o.value for o in opening.b], ref.tile)
        if ops is None or not scheme.admissible(ctx, ops):
            return Verdict(ref.id, False, "inadmissible inputs")
        if scheme.digest(ctx, noise, scheme.checked(ctx, ops)) != opening.leaf.value:
            return Verdict(ref.id, False, "checked values")
        return Verdict(ref.id, True)

    def audit(self, prover: Prover, transcript: Transcript, selection: Sampled | Lottery, source: bytes) -> Audit:
        if len(transcript.activation_roots) != len(self.epoch.workload.matmuls):
            return Audit(False, (), (), None)
        selected = selection.select(self.scheme, self.epoch, transcript, source, self.refs, prover.digest)
        verdicts = tuple(self.check_tile(transcript, prover.open(self.epoch, tid)) for tid in selected)
        ok = all(v.ok for v in verdicts)
        profile = (WorkProfile(self.scheme.name, self.work, len(selected), isinstance(selection, Sampled), self.scheme.certificate)
                   if ok else None)
        return Audit(ok, selected, verdicts, profile)


__all__ = ["AUDIT_DOMAIN", "SALT_DOMAIN", "Audit", "Epoch", "Lottery", "Prover", "Sampled", "TileOpening", "TileRef", "Transcript",
           "Verdict", "Verifier", "WorkProfile", "Workload", "commit_rows", "reference_kernel", "tickets", "tile_refs"]
