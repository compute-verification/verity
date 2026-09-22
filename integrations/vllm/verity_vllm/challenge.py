"""Staged, verifier-owned challenge randomness for the vLLM replay verifier (GUIDANCE C1/C1a).

What the protocol requires is a **fresh challenge source**, not a particular party: after the
prover is irrevocably bound to phase transcript ``T_i``, the source yields randomness ``r_i``
that was unpredictable to the prover before ``T_i`` became binding and that the prover cannot
bias except with negligible probability.  :class:`ChallengeSource` is that contract as an
interface; the sampling that consumes ``r_i`` is the core protocol's exact Bernoulli machinery
(:mod:`veritor.protocol.challenge`: ``K ~ Binomial(N, p)`` by CDF inversion, then a uniform
``K``-subset by Floyd's algorithm, all HMAC-SHA256 keyed by ``r_i`` over the phase digest), so
the adapter and the core draw the same distribution from the same bytes.

Two instantiations exist here:

``InteractiveVerifierSource`` (**security mode**, the canonical instantiation)
    The verifier holds a secret 32-byte seed (``check_seed`` from the core session module) drawn
    from the OS at session start; the transcript records ``H(seed)`` before any phase closes and
    the seed itself only in the final transcript, exactly as the core session releases a stage
    seed only inside the challenge message that follows the phase it keys.  ``r_i =
    HMAC-SHA256(seed, frame(domain, phase_digest, purpose))``.  Unpredictability rests on the
    verifier keeping the seed from the prover until ``T_i`` is fixed; a third party re-verifying
    the transcript checks the derivation and the commitment but must trust that ordering.

``DiagnosticSeedSource`` (**diagnostic mode**, explicitly not a security instantiation)
    The same derivation keyed by ``H("verity-vllm/diagnostic-seed/v1", be64(--seed))`` with a
    public integer seed, so a run can be reproduced exactly from the command line.  A prover
    that knows the seed can predict every challenge from its own commitments and grind them; every
    artifact this source touches carries ``challenge: {"mode": "diagnostic"}`` and the verdict is
    rendered qualified (``PASS (diagnostic challenge)``) -- it demonstrates the checks, not the
    security claim.

Extension point (documented, **not implemented**): ``BeaconSource``.  A public randomness beacon
can replace the interactive verifier under the same interface -- ``randomness(phase_digest,
purpose)`` returns the first beacon output after the epoch at which ``phase_digest`` was fixed,
mixed with the digest and purpose exactly as above -- but it needs one extra ingredient the
interactive source gets for free: a credible proof that the commitment was fixed *before* the
beacon output (a signed verifier receipt over the phase digest, an append-only log, a
timestamped transparency service or a ledger entry).  Without it the prover waits for the beacon
output and then chooses what it claims to have committed.  Any implementation must carry that
receipt in the transcript and the re-verifier must check it; see ``docs/vllm-poc/binding.md``.

What is **not** an acceptable source, and why (the negative argument): ``r = H(commitment)`` and
``r = H(public seed, commitment)`` let the prover grind -- it recomputes the commitment with
witness-dependent freedom (padding, ordering, which faulty values to include) until the derived
sample avoids what it wants hidden; ordinary Fiat-Shamir fails here for the same reason, because
the prover controls witness-dependent commitment material and the statement is not fixed before
the commitments.  Replacing interaction needs an external unpredictability/timing assumption,
not a hash.

Staging for the vLLM flow (each digest covers the previous one, as :mod:`veritor.protocol.phases`
does)::

    T_0 = statement: registered profile, production record (sha256), boundary roots, weights root,
          description/required digests, the C3 binding results, the policy (q, s), the source's
          seed commitment
    r_q = source.randomness(T_0, "q/replay-unit")        -> J: requests, each with probability q
    T_1 = H(T_0, replay challenge {r_q, J})
    interiors: the committed replay-unit roots (every step) of the requests in J
    T_2 = H(T_1, interiors)
    r_s = source.randomness(T_2, "s/verification-unit")  -> T: leaves of J, each with probability s
          and, from the same r_s, the seed of the stratified *coverage* sample
    T_3 = H(T_2, sample challenge {r_s, T, coverage sample})
    final: the source's seed reveal (security mode), so that r_q and r_s can be re-derived

The verifier generates ``r_q`` only after accepting the boundary commitments (and the bindings
they must satisfy) and ``r_s`` only after accepting the selected interiors; :func:`reverify`
re-derives both selections from a recorded transcript and rejects any transcript whose
selections are not what its randomness yields.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from secrets import token_bytes
from typing import Any

from . import hashing

SCHEMA = "verity-vllm/challenge-transcript/v1"
DOMAIN = b"verity-vllm/challenge-source/v1"
SEED_COMMITMENT_TAG = "verity-vllm/challenge-seed-commitment/v1"
DIAGNOSTIC_SEED_TAG = "verity-vllm/diagnostic-seed/v1"
COVERAGE_SEED_TAG = b"coverage/stratified-seed"
PURPOSE_Q = b"q/replay-unit"
PURPOSE_S = b"s/verification-unit"
# the core protocol's stage tags (veritor.protocol.challenge._Q_STAGE / _S_STAGE), passed through
CORE_STAGE_Q = b"q/replay-unit"
CORE_STAGE_S = b"s/verification-unit"
PHASE_STATEMENT = "verity-vllm/phase/statement/v1"
PHASE_REPLAY = "verity-vllm/phase/replay/v1"
PHASE_INTERIOR = "verity-vllm/phase/interior/v1"
PHASE_SAMPLE = "verity-vllm/phase/sample/v1"

MODES = ("security", "diagnostic")


class ChallengeError(ValueError):
    """A transcript that does not re-derive, or a source used out of contract."""


def _frame(*parts: bytes) -> bytes:
    out = bytearray()
    for part in parts:
        out.extend(len(part).to_bytes(8, "big"))
        out.extend(part)
    return bytes(out)


def check_seed(name: str, value: object) -> bytes:
    """``value`` as a 32-byte seed (the core session's ``check_seed`` rule)."""
    if type(value) is not bytes or len(value) != 32:
        raise ChallengeError(f"expected {name} of 32 bytes")
    return value


def seed_commitment(seed: bytes) -> str:
    return hashing.H(SEED_COMMITMENT_TAG, seed).hex()


# --- the interface -------------------------------------------------------------------------------


class ChallengeSource(ABC):
    """A source of staged challenge randomness.

    Contract (theory side, GUIDANCE C1a): *after the prover is irrevocably bound to phase
    transcript* ``T_i``, *the source yields randomness* ``r_i`` *that was unpredictable to the
    prover before* ``T_i`` *became binding and that the prover cannot bias except with negligible
    probability.*

    The one operation, :meth:`randomness`, takes the phase digest (the binding transcript of
    everything committed so far, computed as :mod:`veritor.protocol.phases` computes its phase
    digests: each digest covers the previous one and the message just accepted) and a purpose
    tag, and returns the 32 challenge bytes that key :mod:`veritor.protocol.challenge`'s exact
    Bernoulli sampling for that stage.  Implementations must never return bytes for a phase
    digest the caller has not fixed -- the interface cannot enforce this; the transcript makes it
    checkable after the fact (the digest a challenge is bound to must equal the digest of the
    documents that precede it).

    Admissibility (GUIDANCE C1b): the quantitative-integrity theorem is stated over a source with
    (1) *delayed unpredictability* -- conditioned on everything the prover knows before ``T_i`` is
    binding, ``r_i`` is computationally unpredictable; (2) *distributional correctness / non-bias*
    -- ``r_i`` induces the prescribed sampling distribution up to negligible error (here: the core
    sampler's exact Bernoulli law keyed by 32 uniform bytes); (3) *transcript binding and
    uniqueness* -- ``r_i`` is bound to ``(phase digest, purpose)`` and the prover cannot obtain
    several valid challenges for the same phase and keep the favourable one.  (3) is enforced and
    recorded: a source yields **one** draw per ``(phase digest, purpose)`` per session
    (:meth:`randomness` refuses a second), the transcript records exactly one ``r_q`` and one ``r_s``,
    and :func:`reverify` recomputes both from the revealed seed, so any other value is rejected.
    ``InteractiveVerifierSource`` satisfies (1)-(3); ``DiagnosticSeedSource`` satisfies (2)-(3) and
    fails (1), which is exactly why its rows are conformance tests outside the theorem.
    """

    mode: str  # "security" | "diagnostic"

    @abstractmethod
    def randomness(self, phase_digest: bytes, purpose: bytes) -> bytes:
        """32 bytes of challenge randomness for ``(phase_digest, purpose)``; one draw per pair."""

    @abstractmethod
    def opening(self) -> dict[str, Any]:
        """What the *initial* transcript records about the source (a commitment, never the key)."""

    @abstractmethod
    def reveal(self) -> dict[str, Any]:
        """What the *final* transcript records so a third party can re-derive every ``r_i``."""


class _HmacSource(ChallengeSource):
    """HMAC-SHA256 keyed by a 32-byte seed over ``frame(DOMAIN, phase_digest, purpose)``."""

    def __init__(self, seed: bytes) -> None:
        self._seed = check_seed("challenge seed", seed)
        self.commitment = seed_commitment(self._seed)
        self.draws: list[dict[str, str]] = []  # every (phase digest, purpose) this source answered

    def randomness(self, phase_digest: bytes, purpose: bytes) -> bytes:
        if not isinstance(phase_digest, bytes) or len(phase_digest) != 32:
            raise ChallengeError("phase digest must be 32 bytes")
        # admissibility (3): one draw per purpose per session.  The derivation is a function of the
        # seed, so a repeat could only return the same bytes -- refusing it keeps "one canonical r_i
        # per phase" a fact of the session and not merely of the arithmetic.
        if any(d["purpose"] == purpose.decode() for d in self.draws):
            raise ChallengeError(f"a challenge for purpose {purpose!r} was already drawn in this session")
        r = hmac.new(self._seed, _frame(DOMAIN, phase_digest, purpose), hashlib.sha256).digest()
        self.draws.append({"phase_digest": phase_digest.hex(), "purpose": purpose.decode(), "r": r.hex()})
        return r

    @staticmethod
    def derive(seed: bytes, phase_digest: bytes, purpose: bytes) -> bytes:
        return hmac.new(seed, _frame(DOMAIN, phase_digest, purpose), hashlib.sha256).digest()


class InteractiveVerifierSource(_HmacSource):
    """Security mode: the verifier's own secret seed, drawn from the OS, released at the end."""

    mode = "security"

    def __init__(self, seed: bytes | None = None) -> None:
        super().__init__(token_bytes(32) if seed is None else seed)

    def opening(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": "InteractiveVerifierSource",
            "seed_commitment": self.commitment,
            "seed_commitment_derivation": f'H("{SEED_COMMITMENT_TAG}", seed)',
            "randomness_derivation": "HMAC-SHA256(seed, frame(domain, phase_digest, purpose))",
            "domain": DOMAIN.decode(),
            # WB2 (audit-security-baseline finding 3): the actual order of operations -- the seed exists before
            # the statement (its commitment is a field of T_0), never depends on it, and is revealed after T_3.
            "note": "seed drawn from the OS (secrets.token_bytes) by the verifier before the statement, committed into the "
                    "statement (seed_commitment under T_0), held by the verifier only until T_3, revealed in the final transcript",
        }

    def reveal(self) -> dict[str, Any]:
        return {"seed": self._seed.hex(), "seed_commitment": self.commitment, "draws": list(self.draws)}


class DiagnosticSeedSource(_HmacSource):
    """Diagnostic mode: a public integer ``--seed``; reproducible, predictable, not a security
    instantiation.  Everything derived from it must be labelled ``challenge.mode = diagnostic``."""

    mode = "diagnostic"

    def __init__(self, seed: int) -> None:
        self.public_seed = int(seed)
        super().__init__(hashing.H(DIAGNOSTIC_SEED_TAG, self.public_seed.to_bytes(8, "big", signed=True)))

    def opening(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": "DiagnosticSeedSource",
            "public_seed": self.public_seed,
            "seed_derivation": f'H("{DIAGNOSTIC_SEED_TAG}", be64(public_seed))',
            "seed_commitment": self.commitment,
            "randomness_derivation": "HMAC-SHA256(seed, frame(domain, phase_digest, purpose))",
            "domain": DOMAIN.decode(),
            "warning": "NOT a security instantiation: the seed is public, so a prover can predict every "
                       "challenge from its own commitments and grind them; verdicts are qualified",
        }

    def reveal(self) -> dict[str, Any]:
        return {"seed": self._seed.hex(), "public_seed": self.public_seed, "seed_commitment": self.commitment, "draws": list(self.draws)}


def make_source(mode: str, *, seed: int | None = None) -> ChallengeSource:
    if mode == "security":
        if seed is not None:
            raise ChallengeError("security mode takes no --seed: the verifier draws its own")
        return InteractiveVerifierSource()
    if mode == "diagnostic":
        if seed is None:
            raise ChallengeError("diagnostic mode needs a public --seed")
        return DiagnosticSeedSource(seed)
    raise ChallengeError(f"unknown challenge mode {mode!r}; one of {MODES}")


# --- the core sampler ----------------------------------------------------------------------------


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


_CORE_SCRIPT = r"""
import json, sys
from fractions import Fraction
from veritor.core import VerificationLimits
from veritor.protocol.challenge import bernoulli_subset
req = json.loads(sys.stdin.read())
out = bernoulli_subset(bytes.fromhex(req["seed"]), req["stage"].encode(), bytes.fromhex(req["phase_digest"]),
                       int(req["count"]), Fraction(req["p"]), VerificationLimits())
sys.stdout.write(json.dumps(list(out)))
"""


def _core_python() -> str | None:
    env = os.environ.get("VERITY_CORE_PYTHON")
    if env:
        return env
    for cand in ("/workspace/venv312/bin/python", "python3.12", "python3.13"):
        if cand.startswith("/") and Path(cand).exists():
            return cand
        if not cand.startswith("/") and shutil.which(cand):
            return cand
    return None


def core_bernoulli_subset(seed: bytes, stage: bytes, phase_digest: bytes, count: int, p: Fraction) -> tuple[int, ...]:
    """``veritor.protocol.challenge.bernoulli_subset`` -- in-process when this interpreter can import the
    core package (Python >= 3.12), otherwise through a core-capable interpreter (``VERITY_CORE_PYTHON``
    or a known venv), with the same arguments and result.  The arguments are bytes and integers, so the
    hop is exact.  The sampler itself is stdlib (``hashlib`` / ``hmac`` / ``fractions``), but importing it
    imports the ``veritor`` package, whose ``__init__`` needs **numpy** (WA's audit, finding 5): the
    core-capable interpreter is "Python >= 3.12 with numpy", not "stdlib only"."""
    if p == 0 or count == 0:
        return ()
    if sys.version_info >= (3, 12):
        try:
            sys.path.insert(0, str(_repo_src()))
            from veritor.core import VerificationLimits  # type: ignore
            from veritor.protocol.challenge import bernoulli_subset  # type: ignore

            return tuple(bernoulli_subset(seed, stage, phase_digest, count, p, VerificationLimits()))
        except ImportError:
            pass
    py = _core_python()
    if py is None:
        raise ChallengeError("no interpreter can import veritor.protocol.challenge (needs Python >= 3.12 *with numpy*: "
                             "`import veritor` pulls veritor.core.silicon); set VERITY_CORE_PYTHON")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_src()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    req = {"seed": seed.hex(), "stage": stage.decode(), "phase_digest": phase_digest.hex(), "count": int(count), "p": f"{p.numerator}/{p.denominator}"}
    proc = subprocess.run([py, "-c", _CORE_SCRIPT], input=json.dumps(req), capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise ChallengeError(f"core sampler failed: {proc.stderr.strip()[-500:]}")
    return tuple(int(x) for x in json.loads(proc.stdout))


def core_sampler_provenance() -> dict[str, Any]:
    py = sys.executable if sys.version_info >= (3, 12) else _core_python()
    return {
        "module": "veritor.protocol.challenge.bernoulli_subset",
        "interpreter": py,
        "in_process": sys.version_info >= (3, 12),
        "distribution": "K ~ Binomial(N, p) by 256-bit CDF inversion, then a uniform K-subset (Floyd); HMAC-SHA256 keyed by r_i",
    }


# --- phase digests -------------------------------------------------------------------------------


def phase_digest(tag: str, manifest: Mapping[str, Any]) -> bytes:
    return hashing.H(tag, hashing.canonical_json_bytes(dict(manifest)))


def statement_digest(statement: Mapping[str, Any]) -> bytes:
    return phase_digest(PHASE_STATEMENT, statement)


def replay_phase(previous: bytes, replay_challenge: Mapping[str, Any]) -> bytes:
    return phase_digest(PHASE_REPLAY, {"previous": previous.hex(), "challenge": dict(replay_challenge)})


def interior_phase(previous: bytes, interiors: Mapping[str, Any]) -> bytes:
    return phase_digest(PHASE_INTERIOR, {"previous": previous.hex(), "interiors": dict(interiors)})


def sample_phase(previous: bytes, sample_challenge: Mapping[str, Any]) -> bytes:
    return phase_digest(PHASE_SAMPLE, {"previous": previous.hex(), "challenge": dict(sample_challenge)})


def coverage_seed(r_s: bytes) -> int:
    """The stratified coverage sample's numpy seed, a function of ``r_s`` (so it is fixed after T_2)."""
    return int.from_bytes(hmac.new(r_s, _frame(DOMAIN, COVERAGE_SEED_TAG), hashlib.sha256).digest()[:8], "big")


# --- the staged session --------------------------------------------------------------------------


@dataclass
class Policy:
    q: Fraction
    s: Fraction

    def to_json(self) -> dict[str, str]:
        return {"q": f"{self.q.numerator}/{self.q.denominator}", "s": f"{self.s.numerator}/{self.s.denominator}"}

    @classmethod
    def parse(cls, q: str, s: str) -> Policy:
        pq, ps = Fraction(q), Fraction(s)
        if not (0 <= pq <= 1 and 0 <= ps <= 1):
            raise ChallengeError("q and s must be probabilities")
        return cls(pq, ps)


@dataclass
class RankedUnits:
    """The verification-unit population of the selected replay units, block by block in J order.

    ``blocks`` are ``(tree_id, leaf_count)`` in order; ``resolve(rank)`` maps a rank into
    ``(tree_id, index)``.  Only the block sizes are needed to sample; the ids of a block are read
    only for blocks that hold a selected rank."""

    blocks: list[tuple[str, int]]
    ends: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        total = 0
        self.ends = []
        for _, n in self.blocks:
            total += n
            self.ends.append(total)

    @property
    def count(self) -> int:
        return self.ends[-1] if self.ends else 0

    def resolve(self, rank: int) -> tuple[str, int]:
        from bisect import bisect_right

        b = bisect_right(self.ends, rank)
        start = self.ends[b] - self.blocks[b][1]
        return self.blocks[b][0], rank - start


class StagedSession:
    """The verifier's side of the staged challenge, one phase at a time; writes the transcript."""

    def __init__(self, source: ChallengeSource, policy: Policy) -> None:
        self.source = source
        self.policy = policy
        self.transcript: dict[str, Any] = {
            "schema": SCHEMA,
            "source": source.opening(),
            "policy": policy.to_json(),
            "core_sampler": core_sampler_provenance(),
            "phases": [],
        }
        self._phase = "statement"
        self.t0 = b""
        self.t1 = b""
        self.t2 = b""
        self.t3 = b""
        self.r_q = b""
        self.r_s = b""
        self.selected_requests: list[str] = []
        self.certified_sample: list[str] = []
        self.coverage_sample: list[str] = []

    def _expect(self, phase: str) -> None:
        if self._phase != phase:
            raise ChallengeError(f"session is in phase {self._phase!r}, not {phase!r}")

    # phase 0: the statement, then r_q
    def accept_statement(self, statement: Mapping[str, Any]) -> bytes:
        self._expect("statement")
        self.t0 = statement_digest(statement)
        self.transcript["statement"] = dict(statement)
        self.transcript["phases"].append({"phase": "statement", "digest": self.t0.hex()})
        self._phase = "replay"
        return self.t0

    def challenge_replay(self, request_ids: Sequence[str]) -> list[str]:
        """``J``: each request (replay unit) independently with probability ``q``, in canonical order."""
        self._expect("replay")
        candidates = sorted(request_ids)
        self.r_q = self.source.randomness(self.t0, PURPOSE_Q)
        picked = core_bernoulli_subset(self.r_q, CORE_STAGE_Q, self.t0, len(candidates), self.policy.q)
        self.selected_requests = [candidates[i] for i in picked]
        msg = {"r_q": self.r_q.hex(), "purpose": PURPOSE_Q.decode(), "candidates": len(candidates),
               "selected_indices": list(picked), "selected": list(self.selected_requests)}
        self.t1 = replay_phase(self.t0, msg)
        self.transcript["replay_challenge"] = msg
        self.transcript["phases"].append({"phase": "replay", "digest": self.t1.hex()})
        self._phase = "interiors"
        return list(self.selected_requests)

    # phase 2: the selected interiors, then r_s
    def accept_interiors(self, interiors: Mapping[str, Any]) -> bytes:
        """``interiors``: the committed replay-unit tree records (root, leaf_count, ids_digest) of every
        step of every selected request, keyed by tree id."""
        self._expect("interiors")
        self.t2 = interior_phase(self.t1, interiors)
        self.transcript["interiors"] = {"trees": len(interiors), "digest": hashing.json_digest(dict(interiors)).hex()}
        self.transcript["phases"].append({"phase": "interior", "digest": self.t2.hex()})
        self._phase = "sample"
        return self.t2

    def challenge_sample(self, units: RankedUnits, resolve_ids: Any, *, coverage: Any) -> tuple[list[str], list[str]]:
        """``T``: each leaf of the selected replay units independently with probability ``s``;
        ``resolve_ids(tree_id, indices) -> ids`` reads the ids of one block at the given tree-order
        indices; ``coverage(seed) -> list[str]`` draws the labelled stratified coverage sample."""
        self._expect("sample")
        self.r_s = self.source.randomness(self.t2, PURPOSE_S)
        ranks = core_bernoulli_subset(self.r_s, CORE_STAGE_S, self.t2, units.count, self.policy.s)
        by_tree: dict[str, list[int]] = {}
        for rank in ranks:
            tid, idx = units.resolve(rank)
            by_tree.setdefault(tid, []).append(idx)
        certified: list[str] = []
        for tid, idxs in by_tree.items():
            certified.extend(resolve_ids(tid, idxs))
        self.certified_sample = certified
        cov_seed = coverage_seed(self.r_s)
        self.coverage_sample = list(coverage(cov_seed))
        msg = {
            "r_s": self.r_s.hex(), "purpose": PURPOSE_S.decode(), "population": units.count,
            "blocks": [[tid, n] for tid, n in units.blocks],
            "selected_ranks": list(ranks), "certified_sample": list(certified),
            "coverage_seed": cov_seed, "coverage_seed_derivation": "be64(HMAC-SHA256(r_s, frame(domain, 'coverage/stratified-seed'))[:8])",
            "coverage_sample": list(self.coverage_sample),
        }
        self.t3 = sample_phase(self.t2, msg)
        self.transcript["sample_challenge"] = msg
        self.transcript["phases"].append({"phase": "sample", "digest": self.t3.hex()})
        self._phase = "done"
        return list(certified), list(self.coverage_sample)

    def finish(self) -> dict[str, Any]:
        self._expect("done")
        self.transcript["reveal"] = self.source.reveal()
        return self.transcript


# --- re-verification of a recorded transcript ----------------------------------------------------


def reverify(transcript: Mapping[str, Any], *, statement: Mapping[str, Any], interiors: Mapping[str, Any],
             request_ids: Sequence[str], units: RankedUnits, resolve_ids: Any, coverage: Any) -> dict[str, Any]:
    """Re-derive every phase digest and both selections of a recorded transcript from the documents
    it claims to be bound to; a third party runs this to check that the recorded sample is what the
    recorded randomness yields over the recorded commitments.  Returns ``{"ok": bool, "problems": [...],
    "mode": ...}``; never trusts a recorded selection."""
    problems: list[str] = []
    src = transcript.get("source") or {}
    mode = src.get("mode")
    reveal = transcript.get("reveal") or {}
    seed_hex = reveal.get("seed")
    if mode not in MODES:
        problems.append(f"unknown source mode {mode!r}")
    if not seed_hex:
        problems.append("transcript reveals no seed: r_q / r_s cannot be re-derived")
        return {"ok": False, "problems": problems, "mode": mode}
    seed = bytes.fromhex(seed_hex)
    if seed_commitment(seed) != src.get("seed_commitment"):
        problems.append("revealed seed does not open the recorded seed commitment")
    if mode == "diagnostic":
        expect = hashing.H(DIAGNOSTIC_SEED_TAG, int(src.get("public_seed", 0)).to_bytes(8, "big", signed=True))
        if expect != seed:
            problems.append("diagnostic seed is not H(tag, be64(public_seed))")
    policy = Policy.parse(transcript["policy"]["q"], transcript["policy"]["s"])
    t0 = statement_digest(statement)
    phases = {p["phase"]: p["digest"] for p in transcript.get("phases", [])}
    if phases.get("statement") != t0.hex():
        problems.append("statement digest differs from the documents (T_0)")
    rc = transcript.get("replay_challenge") or {}
    r_q = _HmacSource.derive(seed, t0, PURPOSE_Q)
    if rc.get("r_q") != r_q.hex():
        problems.append("r_q is not HMAC(seed, T_0, q purpose)")
    candidates = sorted(request_ids)
    picked = core_bernoulli_subset(r_q, CORE_STAGE_Q, t0, len(candidates), policy.q)
    if [candidates[i] for i in picked] != list(rc.get("selected", [])):
        problems.append("selected replay units are not what r_q yields over the candidates")
    t1 = replay_phase(t0, rc)
    if phases.get("replay") != t1.hex():
        problems.append("replay phase digest differs (T_1)")
    t2 = interior_phase(t1, interiors)
    if phases.get("interior") != t2.hex():
        problems.append("interior phase digest differs from the selected replay units' committed roots (T_2)")
    sc = transcript.get("sample_challenge") or {}
    r_s = _HmacSource.derive(seed, t2, PURPOSE_S)
    if sc.get("r_s") != r_s.hex():
        problems.append("r_s is not HMAC(seed, T_2, s purpose)")
    if [[tid, n] for tid, n in units.blocks] != [list(b) for b in sc.get("blocks", [])]:
        problems.append("VU population (blocks) differs from the committed replay units")
    ranks = core_bernoulli_subset(r_s, CORE_STAGE_S, t2, units.count, policy.s)
    if list(ranks) != list(sc.get("selected_ranks", [])):
        problems.append("selected ranks are not what r_s yields over the population")
    by_tree: dict[str, list[int]] = {}
    for rank in ranks:
        tid, idx = units.resolve(rank)
        by_tree.setdefault(tid, []).append(idx)
    certified: list[str] = []
    for tid, idxs in by_tree.items():
        certified.extend(resolve_ids(tid, idxs))
    if certified != list(sc.get("certified_sample", [])):
        problems.append("certified sample ids differ from the selected ranks' ids")
    cov_seed = coverage_seed(r_s)
    if cov_seed != sc.get("coverage_seed"):
        problems.append("coverage seed is not derived from r_s")
    elif list(coverage(cov_seed)) != list(sc.get("coverage_sample", [])):
        problems.append("coverage sample is not what the coverage seed yields")
    t3 = sample_phase(t2, sc)
    if phases.get("sample") != t3.hex():
        problems.append("sample phase digest differs (T_3)")
    # admissibility (3), uniqueness: the source answered exactly one draw per purpose, and each is
    # the draw the transcript used, bound to the phase digest that precedes it
    draws = reveal.get("draws")
    if draws is not None:
        want = {PURPOSE_Q.decode(): (t0.hex(), rc.get("r_q")), PURPOSE_S.decode(): (t2.hex(), sc.get("r_s"))}
        by_purpose: dict[str, list[dict[str, str]]] = {}
        for d in draws:
            by_purpose.setdefault(str(d.get("purpose")), []).append(d)
        if sorted(by_purpose) != sorted(want):
            problems.append(f"the source answered purposes {sorted(by_purpose)}, expected exactly {sorted(want)}")
        for purpose, ds in by_purpose.items():
            if len(ds) != 1:
                problems.append(f"{len(ds)} draws for purpose {purpose!r}: a challenge must be drawn once")
            elif purpose in want and (ds[0].get("phase_digest"), ds[0].get("r")) != want[purpose]:
                problems.append(f"the recorded draw for {purpose!r} is not bound to the phase digest / randomness the transcript used")
    return {"ok": not problems, "problems": problems, "mode": mode, "digests": {"T_0": t0.hex(), "T_1": t1.hex(), "T_2": t2.hex(), "T_3": t3.hex()}}


# --- the population, from the committed documents alone (no GPU, no weights) --------------------
#
# These read only commitments.json and the ru/ indices of a capture directory, so a third party
# holding the bundle can rebuild the VU population, the ids of the selected ranks and the coverage
# draw and run :func:`reverify` on the recorded transcript without the adapter or a model.


def interiors_of(commitments: Mapping[str, Any], reader: Any, selected: Sequence[str]) -> tuple[dict[str, Any], RankedUnits]:
    """The committed replay-unit records of the selected requests (every step, boundary order) and
    the ranked VU population they form.  ``reader`` is a :class:`verity_vllm.capture.CaptureReader`."""
    from .capture import ru_tree_id

    trees = commitments.get("trees", {})
    interiors: dict[str, Any] = {}
    blocks: list[tuple[str, int]] = []
    for rid in selected:
        for step in reader.boundary(rid).steps:
            tid = ru_tree_id(rid, step.step)
            rec = trees[tid]
            interiors[tid] = {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest"),
                              "phase": rec.get("phase"), "positions": rec.get("positions")}
            blocks.append((tid, int(rec["leaf_count"])))
    return interiors, RankedUnits(blocks)


def make_resolver(reader: Any):
    """``resolve_ids(tree_id, indices) -> ids`` over the ru index in tree order.  Pure in its arguments, so
    the result is memoised: :func:`reverify` (and the N18 variants) resolve the same ranks again."""
    memo: dict[tuple[str, tuple[int, ...]], list[str]] = {}

    def resolve_ids(tid: str, idxs: Sequence[int]) -> list[str]:
        key = (tid, tuple(int(i) for i in idxs))
        if key not in memo:
            _, rid, step = tid.split("/")
            ids = reader.ru(rid, int(step)).index.ids()
            memo[key] = [ids[i] for i in key[1]]
        return list(memo[key])

    return resolve_ids


def make_coverage(description: Any, reader: Any, adjacent: Sequence[str], n: int, request_ids: Sequence[str]):
    """``coverage(seed) -> ids``: the adjacent chains first, then the stratified draw of ``n`` over the
    selected requests (:func:`verity_vllm.relations.draw_sample`), duplicates removed.  The draw is a pure
    function of the seed (about two minutes on a 4M-leaf capture) and is memoised per seed."""
    from . import relations

    memo: dict[int, list[str]] = {}

    def coverage(seed: int) -> list[str]:
        if seed not in memo:
            drawn = relations.draw_sample(description, reader, n, seed=seed, request_ids=list(request_ids))
            memo[seed] = list(dict.fromkeys(list(adjacent) + drawn))
        return list(memo[seed])

    return coverage


def statement_checks(statement: Mapping[str, Any], *, reader: Any, description: Any, required_doc: Mapping[str, Any] | None,
                     commitments: Mapping[str, Any], production_sha256: str | None) -> list[str]:
    """The fields of a recorded statement a third party *can* re-derive from the bundle alone (profile,
    description/required digests, boundary and weights roots, the declared replay units, the production
    record's sha256).  The C3 binding outcomes need the pinned checkpoint and are not re-derived here."""
    from . import required as vrequired
    from .capture import WEIGHTS_TREE_ID

    problems: list[str] = []
    trees = commitments.get("trees", {})
    if statement.get("profile_id") != reader.profile_id:
        problems.append("statement.profile_id is not the bundle's profile")
    if statement.get("manifest_digest") != hashlib.sha256((reader.run_dir / "manifest.json").read_bytes()).hexdigest():
        problems.append("statement.manifest_digest is not sha256(manifest.json)")
    if statement.get("description_digest") != description.digest().hex():
        problems.append("statement.description_digest is not the bundle's description")
    want_req = vrequired.required_digest(required_doc).hex() if required_doc else None
    if statement.get("required_digest") != want_req:
        problems.append("statement.required_digest is not the bundle's required set")
    boundary = {tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
                for tid, rec in sorted(trees.items()) if rec.get("kind") == "boundary"}
    if statement.get("boundary_roots") != boundary:
        problems.append("statement.boundary_roots differ from commitments.json")
    weights = {k: trees[WEIGHTS_TREE_ID].get(k) for k in ("root", "leaf_count", "ids_digest", "weights_sha256")} if WEIGHTS_TREE_ID in trees else None
    if statement.get("weights_root") != weights:
        problems.append("statement.weights_root differs from commitments.json")
    if statement.get("replay_units_declared") != sorted(tid for tid, rec in trees.items() if rec.get("kind") == "ru"):
        problems.append("statement.replay_units_declared differ from commitments.json")
    rec_sha = ((statement.get("bindings") or {}).get("output") or {}).get("production_record_sha256")
    if production_sha256 is not None and rec_sha is not None and rec_sha != production_sha256:
        problems.append("statement binds a different production record (sha256)")
    return problems


def verdict_label(result: str, mode: str | None) -> str:
    """How a verdict is rendered: unqualified in security mode, qualified in diagnostic mode (a ``FAIL``
    is a fail under any challenge)."""
    if mode == "diagnostic" and result != "FAIL":
        return f"{result} (diagnostic challenge)"
    return result
