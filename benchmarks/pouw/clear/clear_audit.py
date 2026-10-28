"""PoUW's clear audit: the protocol's lifecycle run in the clear, the same for every `PoUWScheme` (``PROTOCOL.md``,
"Lifecycle").  It is a reference tool, not a verifier: every protocol is zero-knowledge, PoUW's served path is the served-ZK
window, and its verifier of record is the Lean (`verity/core/protocols/pouw/lean/check`, `pouw.verifier`).  Benchmarks
replay recorded clear runs through it (`pearl_c_vllm/verify_run.py`, `pearl_c4/replay.py`), the schemes' tests run their
end-to-end vectors through it, and the Lean's draws (`PouwCommit.Draws`) are difftested against `Sampled.select` and
`draw_coin`.  This directory is not a package: callers put it on ``sys.path``.

0.-2. **Coins, weights and the salt** are core's (`verity.core.protocols.pouw.audit`: `weights_root`, `Epoch.start`).
3. **Units.**  For each matmul the prover commits the activation rows (bound to the salt and the index); the scheme
   derives its noise from (salt, matmul, roots), and every tile yields checked values and a digest.  The useful output is
   assembled from the tiles.
4. **Transcript.**  A Merkle root over every tile's digest, in matmul order and then tile order (core's `tile_refs`).
5. **Selection.**  `Sampled`: ``derive(source, "verity/pouw/audit/v1", {"transcript": root, "salt": salt})`` draws t
   tiles independently, each with probability ``W_ref(tile)/W`` (Lean `WorkWeightedSampling`); on an epoch with coins the
   source is the epoch's draw coin, which the challenger opens after the verifier recorded the transcript.  `Lottery` is
   Pearl's mining rule, not one of the audit's laws, and stays unwired: the tiles whose digest, read little-endian, is at
   most ``target·|rows|·|cols|·k``.  The prover opens each selected tile's strips and leaf; the verifier recomputes the
   tile through the scheme.
6. **Profile.**  When every selected tile passes, `WorkProfile` states what that establishes.

The protocol owns every outer coin (the salt and the draw), and both are the challenger's live coins.  A scheme's noise is
its own derivation from the salt and the prover's roots, Fiat-Shamir inside the scheme.  The row and tile trees are core's
(`audit.commit_rows`, `rows_domain`, `transcript_domain`), framed SHA-256 unless the scheme names another frame hash in
``commitment_hash`` (`commitment_hash`), and an opening's value under ``"hm96-sha512"`` carries its salt, which
`audit.opened_value` strips before the scheme reads the value.
"""

from __future__ import annotations

import os
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate
from math import lcm

from verity.core.protocols.pouw.audit import (HM96, HM96_TRANSCRIPT, HM96_WEIGHTS, ROW_SCHEMA, TILE_SCHEMA, Epoch, TileRef,
                                              Workload, activation_binding, commit_rows, hm96_salt, opened_value, rows_domain,
                                              tile_refs, transcript_domain, weight_binding)
from verity.core.protocols.pouw.coins import Challenger, coin_reason
from verity.core.protocols.pouw.protocol import Certificate, Context, DomainError, PoUWScheme, Tile
from verity.errors import ProtocolError
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, VerificationLimits, verify_opening
from verity.ml.randomness import derive

AUDIT_DOMAIN = "verity/pouw/audit/v1"


def commitment_hash(scheme: PoUWScheme) -> str:
    """The frame hash of a scheme's row and tile trees: its ``commitment_hash``, SHA-256 by default."""
    return getattr(scheme, "commitment_hash", "sha256")


def draw_coin(epoch: Epoch, challenger: Challenger, after: bytes) -> tuple[bytes | None, str | None]:
    """The epoch's draw coin, asked of the challenger once the verifier recorded `after`, and checked against the
    epoch's coin commitment: (coin, None), or (None, why it is refused)."""
    if epoch.coins is None:
        return None, "the epoch has no coin commitment for its draw coin to open against"
    opening = challenger.open("draw", after)
    reason = coin_reason(epoch.coins, opening, "draw", after)
    return (None, f"the draw's {reason}") if reason is not None else (opening.coin, None)


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
        """`salt_key` is the prover's key of its hm96 salts under `HM96` (fresh by default).  A scheme of a plain format (h0, or
        no ``commitment_hash``) has trees that are not hiding, which this reference tool replays as `audit.commit_rows`'
        ``public`` trees; a served commitment is never one."""
        self.scheme, self.kernel = scheme, kernel
        self.weights = {w: list(rows) for w, rows in weights.items()}
        self.hash = commitment_hash(scheme)
        self.salt_key = (salt_key or os.urandom(32)) if self.hash == HM96 else None
        self.plain = plain = self.hash != HM96
        self.weight_trees = {w: commit_rows(weight_binding(w), -2, rows, self.hash, self.salt_key, HM96_WEIGHTS + w, public=plain)
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
            tree = commit_rows(activation_binding(epoch.salt, u.index), u.index, rows, self.hash, self.salt_key, u.index,
                               public=self.plain)
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


__all__ = ["AUDIT_DOMAIN", "Audit", "Lottery", "Prover", "Sampled", "TileOpening", "Transcript", "Verdict", "Verifier",
           "WorkProfile", "commitment_hash", "draw_coin", "reference_kernel", "tickets"]
