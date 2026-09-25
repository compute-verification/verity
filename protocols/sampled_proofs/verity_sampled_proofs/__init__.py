"""Sampled proofs: the integrity protocol as one protocol among several (prototype).

    prover    commit_boundary -> log TRANSCRIPT -> ru_sample(beacon after it) -> commit_interiors -> log INTERIORS
              -> vu_sample(beacon after it) -> open
    verifier  verify(record, beacon=, log=, prescribed=) -> IntegrityAllowance, or raises Rejected

Replay units (RUs) and verification units (VUs) are IR queries over the Program.  Every VU lies inside one RU and
every RU has the same number of VUs (the two-stage law's n_v).  Commitment positions are gate indices, the Program's
value identities, so another protocol can read a committed boundary value through `reader(record)` without knowing
the layout.  The backend is exact replay (`verity.evaluation.evaluate_call`); a proof backend replaces `_replay`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from verity.commitments.indexed import ExplicitIndexedDomain
from verity.commitments.limits import VerificationLimits
from verity.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, verify_opening
from verity.evaluation import evaluate_call
from verity.ir import FunctionCall, Program, program_digest, query
from verity.randomness import derive
from verity.resource_accounting import IntegrityAllowance

TRANSCRIPT = "sampled-proofs/transcript"
INTERIORS = "sampled-proofs/interiors"
_LIMITS = VerificationLimits()


class Rejected(Exception):
    """The audit rejected the transcript; no allowance follows."""


@dataclass(frozen=True)
class Record:
    boundary: Commitment
    interiors: Mapping[int, Commitment]
    openings: Mapping[int, Opening]


def interiors_digest(interiors: Mapping[int, Commitment]) -> bytes:
    h = hashlib.sha256(b"verity/sampled-proofs/interiors/v1")
    for i in sorted(interiors):
        h.update(i.to_bytes(8, "big") + interiors[i].root)
    return h.digest()


class SampledProofs:
    def __init__(self, program: Program, *, ru: str, vu: str, p: Fraction, k: int, delta: float) -> None:
        self.program, self.p, self.k, self.delta = program, p, k, delta
        C = self.circuit = program.circuit
        self.rus: tuple[FunctionCall, ...] = tuple(query(C, ru))
        vus = query(C, vu)
        self.vus = tuple(tuple(v for v in vus if r.interval[0] <= v.interval[0] and v.interval[1] <= r.interval[1])
                         for r in self.rus)
        sizes = {len(v) for v in self.vus}
        if len(sizes) != 1 or sum(map(len, self.vus)) != len(vus):
            raise ValueError("every VU must lie in exactly one RU, and every RU must hold the same number of VUs")
        (self.n_v,) = sizes
        self.boundary = tuple(sorted({g for r in self.rus for g in (*r.args, *r.returns)}))
        self._in_boundary = frozenset(self.boundary)
        self._binding = hashlib.sha256(bytes.fromhex(program_digest(program))).digest()

    # -- layout ------------------------------------------------------------------------------------------------------
    def _schema(self, gate: int) -> str:
        return f"verity/value/w{self.circuit[gate].prim.ret.w}"

    def _encode(self, gate: int, value: int) -> bytes:
        return value.to_bytes((self.circuit[gate].prim.ret.w + 7) // 8, "big")

    def _boundary_domain(self) -> CommitmentDomain:
        return CommitmentDomain(self._binding, -1, ExplicitIndexedDomain(self.boundary))

    def _interior_domain(self, i: int, boundary_root: bytes) -> CommitmentDomain:
        lo, hi = self.rus[i].interval
        return CommitmentDomain(boundary_root, i, ExplicitIndexedDomain(
            g for g in range(lo, hi) if g not in self._in_boundary))

    def _tree(self, domain: CommitmentDomain, values: Mapping[int, int]) -> MerkleTree:
        return MerkleTree(domain, {g: self._encode(g, values[g]) for g in domain.positions}, self._schema)

    def _read(self, domain: CommitmentDomain, commitment: Commitment, gate: int, opening: Opening) -> int | None:
        if opening.position != gate or not verify_opening(domain, commitment, opening, self._schema(gate), _LIMITS):
            return None
        return int.from_bytes(opening.value, "big")

    # -- challenges --------------------------------------------------------------------------------------------------
    def ru_sample(self, round_: bytes, root: bytes) -> tuple[int, ...]:
        key = derive(round_, "verity/sampled-proofs/ru-sample/v1", {"program": self._binding, "root": root})
        return tuple(i for i in range(len(self.rus)) if key.bernoulli(self.p, i))

    def vu_sample(self, round_: bytes, interiors: bytes, rus: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
        key = derive(round_, "verity/sampled-proofs/vu-sample/v1", {"program": self._binding, "interiors": interiors})
        return {i: key.subset(self.k, self.n_v, i) for i in rus}

    # -- prover ------------------------------------------------------------------------------------------------------
    def commit_boundary(self, values: Mapping[int, int]) -> MerkleTree:
        return self._tree(self._boundary_domain(), values)

    def commit_interiors(self, values: Mapping[int, int], boundary: MerkleTree,
                         rus: tuple[int, ...]) -> dict[int, MerkleTree]:
        return {i: self._tree(self._interior_domain(i, boundary.commitment.root), values) for i in rus}

    def open(self, boundary: MerkleTree, interiors: Mapping[int, MerkleTree],
             vus: Mapping[int, tuple[int, ...]]) -> Record:
        openings = {g: boundary.open(g) if g in self._in_boundary else interiors[i].open(g)
                    for i, js in vus.items() for j in js for g in (*self.vus[i][j].args, *self.vus[i][j].returns)}
        return Record(boundary.commitment, {i: t.commitment for i, t in interiors.items()}, openings)

    # -- verifier ----------------------------------------------------------------------------------------------------
    def verify(self, record: Record, *, beacon: Callable[[int], bytes], log: Callable[[str], tuple[int, bytes]],
               prescribed: Mapping[int, int]) -> IntegrityAllowance:
        """`log(scope)` is the (time, digest) of the one root registered under `scope`; `beacon(t)` is the first
        public round published after time t.  `prescribed` gives the value of every `Input` gate: request inputs,
        weights of record, and whatever other protocols prescribe (PoUW's noise)."""
        t, root = log(TRANSCRIPT)
        if root != record.boundary.root:
            raise Rejected("transcript root is not the registered one")
        rus = self.ru_sample(beacon(t), root)
        if set(record.interiors) != set(rus):
            raise Rejected("interiors committed for a different RU set than the sample")
        t2, digest = log(INTERIORS)
        if digest != interiors_digest(record.interiors):
            raise Rejected("interior roots are not the registered ones")
        boundary = self._boundary_domain()
        for i, js in self.vu_sample(beacon(t2), digest, rus).items():
            interior = self._interior_domain(i, root)
            for j in js:
                self._replay(self.vus[i][j], record, boundary, interior, i, prescribed)
        return IntegrityAllowance(n_r=len(self.rus), n_v=self.n_v, p=self.p, k=self.k, delta=self.delta)

    def _replay(self, vu: FunctionCall, record: Record, boundary: CommitmentDomain, interior: CommitmentDomain,
                i: int, prescribed: Mapping[int, int]) -> None:
        values: dict[int, int] = {}
        for g in {*vu.args, *vu.returns}:
            opening = record.openings.get(g)
            domain, commitment = ((boundary, record.boundary) if g in self._in_boundary
                                  else (interior, record.interiors[i]))
            value = None if opening is None else self._read(domain, commitment, g, opening)
            if value is None:
                raise Rejected(f"gate {g}: missing or unauthenticated opening")
            values[g] = value
        for g in set(vu.args):
            if getattr(self.circuit[g].prim, "family", None) == "Input" and prescribed.get(g) != values[g]:
                raise Rejected(f"gate {g}: Input differs from its prescribed value")
        if evaluate_call(self.circuit, vu, [values[a] for a in vu.args]) != tuple(values[r] for r in vu.returns):
            raise Rejected(f"VU at {vu.interval}: replay disagrees with the committed outputs")

    def reader(self, record: Record) -> Callable[[int, Opening], int | None]:
        """An authenticated read of committed boundary values, for other protocols.  Use after `verify` accepted."""
        domain = self._boundary_domain()
        return lambda gate, opening: (self._read(domain, record.boundary, gate, opening)
                                      if gate in self._in_boundary else None)


__all__ = ["INTERIORS", "TRANSCRIPT", "Record", "Rejected", "SampledProofs", "interiors_digest"]
