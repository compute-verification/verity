"""Sampled proofs across a serving epoch (prototype).

    serving  Prover.commit(step, program, values, now)   the step's boundary root is registered in the log at `now`
             Prover.poll(now)                            "serving" mode: draw a step's samples once the beacon round
                                                         after its registration has been published
    audit    Prover.close(now) -> Evidence               "audit" mode: one round after the epoch root samples every step
             SampledProofs.verify(evidence, ...) -> IntegrityProfile

Boundary values (every RU's arguments and returns) ship with each commit and are checked exhaustively: every `Input`
gate is either prescribed or linked to a committed output of an earlier step.  Interiors are committed only for
selected RUs and opened only for sampled VUs.  Sampling during serving or at audit changes how long the prover must
retain interiors, not soundness: either way the sample comes from a round published after the root it follows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from verity.commitments.indexed import ExplicitIndexedDomain
from verity.commitments.limits import VerificationLimits
from verity.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, verify_opening
from verity.evaluation import evaluate_call
from verity.ir import FunctionCall, Program, program_digest, query
from verity.proofs import IntegrityProfile
from verity.randomness import derive

_LIMITS = VerificationLimits()
Beacon = Callable[[float], tuple[float, bytes]]  # beacon(t) -> (publication time, value) of the first round after t
Log = Callable[[str], tuple[float, bytes]]  # log(scope) -> (registration time, digest)


class Rejected(Exception):
    """The audit rejected the transcript; no profile follows."""


def _digest(tag: bytes, items: Mapping) -> bytes:
    h = hashlib.sha256(tag)
    for k, v in items.items():
        h.update(str(k).encode() + b"\0" + v.root)
    return h.digest()


class Layout:
    """RUs, VUs and commitment domains of one Program at one step."""

    def __init__(self, program: Program, ru: str, vu: Sequence[str]) -> None:
        C = self.circuit = program.circuit
        self.program, self.inputs = program, program.input_gates
        self.rus: tuple[FunctionCall, ...] = tuple(query(C, ru))
        vus = sorted((v for q in vu for v in query(C, q)), key=lambda v: v.interval)
        self.vus = tuple(tuple(v for v in vus if r.interval[0] <= v.interval[0] and v.interval[1] <= r.interval[1])
                         for r in self.rus)
        sizes = {len(v) for v in self.vus}
        if len(sizes) != 1 or sum(map(len, self.vus)) != len(vus):
            raise ValueError("every VU must lie in exactly one RU, and every RU must hold the same number of VUs")
        (self.n_v,) = sizes
        self.boundary = tuple(sorted({g for r in self.rus for g in (*r.args, *r.returns)}))
        self.in_boundary = frozenset(self.boundary)
        self.digest = program_digest(program)

    def schema(self, gate: int) -> str:
        return f"verity/value/w{self.circuit[gate].prim.ret.w}"

    def _tree(self, domain: CommitmentDomain, values: Mapping[int, int]) -> MerkleTree:
        width = lambda g: (self.circuit[g].prim.ret.w + 7) // 8  # noqa: E731
        return MerkleTree(domain, {g: values[g].to_bytes(width(g), "big") for g in domain.positions}, self.schema)

    def boundary_tree(self, step: str, values: Mapping[int, int]) -> MerkleTree:
        binding = hashlib.sha256(f"{self.digest}/{step}".encode()).digest()
        return self._tree(CommitmentDomain(binding, -1, ExplicitIndexedDomain(self.boundary)), values)

    def interior_domain(self, i: int, root: bytes) -> CommitmentDomain:
        lo, hi = self.rus[i].interval
        return CommitmentDomain(root, i, ExplicitIndexedDomain(g for g in range(lo, hi) if g not in self.in_boundary))

    def interior_tree(self, i: int, root: bytes, values: Mapping[int, int]) -> MerkleTree:
        return self._tree(self.interior_domain(i, root), values)


@dataclass
class StepEvidence:
    boundary: dict[int, int]
    interiors: dict[int, Commitment]
    openings: dict[int, Opening]


@dataclass
class _Step:
    layout: Layout
    boundary: dict[int, int]
    tree: MerkleTree
    time: float
    values: dict[int, int] | None
    trees: dict[int, MerkleTree] = field(default_factory=dict)
    rus: tuple[int, ...] | None = None
    interiors_time: float | None = None
    openings: dict[int, Opening] | None = None


class SampledProofs:
    def __init__(self, *, ru: str, vu: Sequence[str], p: Fraction, k: int, delta: float, sample: str = "audit") -> None:
        if sample not in ("audit", "serving"):
            raise ValueError("sample is 'audit' or 'serving'")
        self.ru, self.vu, self.p, self.k, self.delta, self.sample = ru, tuple(vu), p, k, delta, sample
        self._layouts: dict[int, Layout] = {}

    def layout(self, program: Program) -> Layout:
        if id(program) not in self._layouts:
            self._layouts[id(program)] = Layout(program, self.ru, self.vu)
        return self._layouts[id(program)]

    def ru_sample(self, round_: bytes, step: str, root: bytes, n_r: int) -> tuple[int, ...]:
        key = derive(round_, "verity/sampled-proofs/ru-sample/v1", {"step": step, "root": root})
        return tuple(i for i in range(n_r) if key.bernoulli(self.p, i))

    def vu_sample(self, round_: bytes, step: str, interiors: bytes, rus: tuple[int, ...], n_v: int):
        key = derive(round_, "verity/sampled-proofs/vu-sample/v1", {"step": step, "interiors": interiors})
        return {i: key.subset(self.k, n_v, i) for i in rus}

    # -- verifier ----------------------------------------------------------------------------------------------------
    def verify(self, evidence: Mapping[str, StepEvidence], *, beacon: Beacon, log: Log,
               programs: Mapping[str, Program], prescribed: Mapping[str, Mapping[str, Mapping[int, int]]],
               links: Mapping[str, Mapping[int, tuple[str, int]]]) -> IntegrityProfile:
        """`prescribed[family][step][gate]` gives external values (prompt, weights, noise, ...); `links[step][gate]` names
        the (earlier step, gate) whose committed value an `Input` must equal.  Every `Input` must be one or the other."""
        epoch = log("sp/epoch")[0] if self.sample == "audit" else None
        seen, n_r, replayed, digests = {}, 0, [0, 0], []
        for step, ev in evidence.items():
            lay = self.layout(programs[step])
            t, root = log(f"sp/step/{step}")
            if lay.boundary_tree(step, ev.boundary).commitment.root != root:
                raise Rejected(f"step {step}: boundary values are not the registered root")
            for g in range(lay.inputs):
                want = next((m[step][g] for m in prescribed.values() if g in m.get(step, {})), None)
                if want is None and g in links.get(step, {}):
                    src_step, src_gate = links[step][g]
                    want = seen.get(src_step, {}).get(src_gate)
                if want is None or ev.boundary.get(g) != want:
                    raise Rejected(f"step {step}: Input gate {g} is neither prescribed nor linked to its value")
            rus = self.ru_sample(beacon(epoch if epoch is not None else t)[1], step, root, len(lay.rus))
            if set(ev.interiors) != set(rus):
                raise Rejected(f"step {step}: interiors committed for a different RU set than the sample")
            t2, digest = log(f"sp/interiors/{step}")
            if digest != _digest(b"verity/sampled-proofs/interiors/v1", ev.interiors):
                raise Rejected(f"step {step}: interior roots are not the registered ones")
            for i, js in self.vu_sample(beacon(t2)[1], step, digest, rus, lay.n_v).items():
                domain = lay.interior_domain(i, root)
                for j in js:
                    self._replay(lay, lay.vus[i][j], ev, domain, ev.interiors[i])
                    replayed[1] += 1
            replayed[0] += len(rus)
            n_r += len(lay.rus)
            seen[step] = ev.boundary
            digests.append(lay.digest)
        n_v = {self.layout(programs[s]).n_v for s in evidence}
        return IntegrityProfile(programs=tuple(digests), prescribed=frozenset(f for f, m in prescribed.items() if m),
                                linked=True, n_r=n_r, n_v=n_v.pop() if len(n_v) == 1 else 0, p=self.p, k=self.k,
                                delta=self.delta, replayed=tuple(replayed))

    def _replay(self, lay: Layout, vu: FunctionCall, ev: StepEvidence, domain: CommitmentDomain,
                commitment: Commitment) -> None:
        values = {}
        for g in {*vu.args, *vu.returns}:
            if g in lay.in_boundary:
                values[g] = ev.boundary[g]
                continue
            op = ev.openings.get(g)
            if op is None or op.position != g or not verify_opening(domain, commitment, op, lay.schema(g), _LIMITS):
                raise Rejected(f"gate {g}: missing or unauthenticated interior opening")
            values[g] = int.from_bytes(op.value, "big")
        if evaluate_call(lay.circuit, vu, [values[a] for a in vu.args]) != tuple(values[r] for r in vu.returns):
            raise Rejected(f"VU at {vu.interval}: replay disagrees with the committed values")

    @staticmethod
    def reader(evidence: Mapping[str, StepEvidence]) -> Callable[[str, int], int | None]:
        """Committed boundary values, for other protocols.  Authenticated once `verify` accepted the evidence."""
        return lambda step, gate: evidence[step].boundary.get(gate) if step in evidence else None


class Prover:
    """The serving-side half: commits each step, draws samples when their rounds publish, retains what later
    openings need.  `retained_words` is what the application must find memory for."""

    def __init__(self, sp: SampledProofs, *, beacon: Beacon, register: Callable[[str, bytes, float], None]) -> None:
        self.sp, self.beacon, self.register = sp, beacon, register
        self.steps: dict[str, _Step] = {}

    @property
    def retained_words(self) -> int:
        return sum(len(s.values) for s in self.steps.values() if s.values is not None)

    def commit(self, step: str, program: Program, values: Mapping[int, int], now: float) -> None:
        lay = self.sp.layout(program)
        boundary = {g: values[g] for g in lay.boundary}
        tree = lay.boundary_tree(step, boundary)
        interior = {g: values[g] for r in lay.rus for g in range(*r.interval) if g not in lay.in_boundary}
        self.register(f"sp/step/{step}", tree.commitment.root, now)
        self.steps[step] = _Step(lay, boundary, tree, now, interior)

    def poll(self, now: float) -> None:
        if self.sp.sample == "serving":
            for step, st in self.steps.items():
                self._advance(step, st, now, st.time)

    def close(self, now: float) -> dict[str, StepEvidence]:
        if self.sp.sample == "audit":
            self.register("sp/epoch", _digest(b"verity/sampled-proofs/epoch/v1",
                                              {s: st.tree.commitment for s, st in self.steps.items()}), now)
        for step, st in self.steps.items():
            self._advance(step, st, now, now if self.sp.sample == "audit" else st.time, wait=True)
        return {s: StepEvidence(st.boundary, {i: t.commitment for i, t in st.trees.items()}, st.openings)
                for s, st in self.steps.items()}

    def _advance(self, step: str, st: _Step, now: float, source: float, wait: bool = False) -> None:
        """Take whichever sampling stages have their rounds published by `now`; with `wait`, wait for them."""
        lay, root = st.layout, st.tree.commitment.root
        if st.rus is None:
            published, round_ = self.beacon(source)
            if published > now and not wait:
                return
            now = max(now, published)
            st.rus = self.sp.ru_sample(round_, step, root, len(lay.rus))
            st.trees = {i: lay.interior_tree(i, root, st.values) for i in st.rus}
            keep = {g for i in st.rus for g in st.trees[i].domain.positions}
            st.values = {g: v for g, v in st.values.items() if g in keep}
            st.interiors_time = now
            self.register(f"sp/interiors/{step}", _digest(b"verity/sampled-proofs/interiors/v1",
                                                          {i: t.commitment for i, t in st.trees.items()}), now)
        if st.openings is None:
            published, round_ = self.beacon(st.interiors_time)
            if published > now and not wait:
                return
            digest = _digest(b"verity/sampled-proofs/interiors/v1", {i: t.commitment for i, t in st.trees.items()})
            vus = self.sp.vu_sample(round_, step, digest, st.rus, lay.n_v)
            st.openings = {g: st.trees[i].open(g) for i, js in vus.items() for j in js
                           for g in {*lay.vus[i][j].args, *lay.vus[i][j].returns} if g not in lay.in_boundary}
            st.values = None


__all__ = ["Layout", "Prover", "Rejected", "SampledProofs", "StepEvidence"]
