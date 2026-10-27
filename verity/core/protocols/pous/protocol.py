"""The generic POUS protocol: one lifecycle for every scheme, which implements only what differs between schemes.

A scheme (`Scheme`) is a keyed, rate-1 encoding of `W`, one segment at a time, together with the certificate its audit
carries at its parameters (`Certificate`).  It implements `encode_segment` and `decode_segment`, and `respond` when an
honest answer is more than the stored label.  Everything else is written once, here:

1. **Commit.** The server commits to `W` before any salt exists (`commit`).
2. **Salt.** The verifier draws the salt after the commitment (`draw_salt`).
3. **Encode.** The server computes `C = Enc(W, salt)` (`encode`).  The verifier builds `vk` (`verifier_key`) either by
   encoding `W` itself (`PRIVATE`) or by decoding the server's `C*` and checking `Dec(C*)` against the commitment
   (`PUBLIC`).  The two give the same guarantee (Lean `pubAuditSecure_exact_iff`).
4. **Serve.** `decode` of a segment, and `decode_tensor` of one tensor of a `layout.Layout`.
5. **Audit.** `TimedAudit` reveals `k` indices one at a time.  Each answer must arrive within `Δ + RTT` and match `vk`.
   The memory challenger refuses any `k` or deadline that none of the scheme's certificates covers.  An accepted audit
   gives a `StorageProfile`: what the certificate says about the server, and what that rests on.

Every coin is drawn here through `verity.ml.randomness` (`setup.SALT_DOMAIN`, `audit.CHALLENGE_DOMAIN`), and a scheme draws
none.  The claims a certificate rests on are ids in `verity.core.claims`' format (`claim`).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar

from verity.core.claims import Entry
from verity.core.claims import get as core_claim
from verity.ml.randomness import Key

from . import audit, setup
from .codec import Codec, Geometry, split
from .layout import Layout
from .layout import decode_tensor as _decode_tensor
from .primitives import Primitives
from .verifier import VerifierKey

PRIVATE, PUBLIC = "private", "public"
"""How the verifier gets `vk`: it encodes `W` itself, or it decodes the server's `C*` and checks it against the
commitment."""

LOCAL_CLAIMS: dict[str, Entry] = {e.id: e for e in (
    Entry("ideal-permutation", "model",
          "Ideal-permutation model: the scheme's permutation is uniformly random and queried in both directions",
          "Ideal permutation"),
    # The id is the earlier model's, kept because certificates and their pinned vectors carry it.
    Entry("perfect-isolation", "model",
          "Help accounted: on PoUS's challenge schedule some challenge tick t_j has mem_X(t_j) + 2^e·h >= S for the "
          "audited location X, where h is the help X received (console's actual-circuit model; proved at e = 0 for a "
          "location that does no arithmetic modulo p). Nothing attests the prover's machine",
          "Help accounted"),
    Entry("sequential-step-floor", "assumption",
          "No call of the scheme's permutation completes faster than the certificate's call time on a CPU or GPU core "
          "(ASICs out of scope); it bounds an adversary only in the ideal-permutation model, since at the reference "
          "Feistel-SHAKE256 permutation a prover answers from kept round states without completing any call",
          "Step floor", "measured"),
    Entry("feistel-indifferentiability", "assumption",
          "REFUTED: a 10-round Feistel network over SHAKE256 in place of the ideal permutation keeps the storage bound. "
          "It does not: a prover keeps five 200-byte post-absorb round states per 64 KiB block (1.53% of C) and "
          "rebuilds any label in 481 sequential Keccak-f",
          "Feistel-SHAKE256 as ideal (refuted)", "conjectured"),
    Entry("m1-ideal-permutations", "model",
          "Model M1: independent forward-only ideal permutations, one per block, each block the inverse image of its "
          "data", "M1"),
    Entry("sms-m1-instantiation", "assumption",
          "Square-mask-square with fresh per-block keys instantiates M1's slow-inverse permutation (novel and "
          "algebraic; not accepted)", "SMS as M1", "conjectured"),
    Entry("seq-root", "assumption",
          "One modular square root at the scheme's width takes longer than twice the deadline on the fastest core, "
          "with up to 8 cores of one chiplet cooperating", "SeqRoot", "measured"),
    # Lean's `Pous.P2.Assumptions.P2RingRoundSecureWide` (Security/Definitions/Pous/Assumptions/P2.lean)
    Entry("p2-ring-round-secure-wide", "assumption",
          "P2 in the ring model, per round, at every block count N from 2^19 to 2^27: outside a bad event of mass k·ε₀, "
          "a state of at most ⌊ρ·|C|⌋ bits answers each fresh uniform challenge with probability at most 1 − p, with "
          "(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130; nobody has checked a proof's ε₀ against N above 2^25",
          "P2 ring rounds to 2^27", "conjectured"),
    # Lean's `Pous.Period.RingDeviceAt` (Security/Properties/Pous/PeriodAt.lean)
    Entry("pous-ring-device-at", "model",
          "PoUS ring device at the store's block count: pous-ring-device, pous-ring-instantiation and prf/sha-512 per "
          "run, read over the ring game at N, the blocks the point's session challenges",
          "PoUS ring device at N"),
)}
"""Claims a certificate cites that `verity.core.claims` does not hold yet, in its id format.  Each moves to core when a
render cites it; the tests fail once core defines an id this table still holds."""


def claim(id: str) -> Entry:
    """The claim with this id: `verity.core.claims`' own, else `LOCAL_CLAIMS`'; `KeyError` when neither has it."""
    try:
        return core_claim(id)
    except KeyError:
        return LOCAL_CLAIMS[id]


@dataclass(frozen=True)
class Certificate:
    """What an accepted audit of a scheme at its parameters establishes: Lean's `Meets S .sequential D Q k ε`.

    A server that holds fewer than `⌊rho · |C|⌋` bits when the challenges arrive, and answers each within `rounds`
    sequential calls of at least `call_ns` each using at most `work` calls, passes `k` challenges with probability at
    most `delta + 2^eps_log2`.  This is `theorem`, given the open named `hypotheses`, under `claims`; where a claim is
    refuted (`feistel-indifferentiability` at the reference `Π₂`), it holds only in the model the others name.

    `cliff` is the most rounds the structure tolerates at any `k`, or None when security degrades gracefully past
    `rounds`.  The band has its cliff at its in-degree `d`.  `segments` is the number of segments of `W` the theorem is
    about, or None for any number."""

    k: int
    rounds: int
    work: int
    call_ns: int
    theorem: str
    rho: Fraction = Fraction(18, 19)
    delta: Fraction = Fraction(1, 100)
    eps_log2: int = -128
    cliff: int | None = None
    segments: int | None = 1
    hypotheses: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.k, self.work, self.call_ns) < 1 or self.rounds < 0:
            raise ValueError("a certificate's k, work and call time are positive, and its rounds nonnegative")
        if not (0 < self.rho < 1 and 0 < self.delta < 1):
            raise ValueError("a certificate needs 0 < rho < 1 and 0 < delta < 1")
        if self.cliff is not None and self.rounds > self.cliff:
            raise ValueError(f"a certificate for {self.rounds} rounds lies past its cliff at {self.cliff}")
        if self.segments is not None and self.segments < 1:
            raise ValueError("a certificate is about at least one segment")
        for c in self.claims:
            claim(c)

    def effective_rounds(self, limit_ns: int) -> int:
        """The sequential calls of at least `call_ns` that fit in `Δ + RTT`: the effective `D`."""
        return limit_ns // self.call_ns

    def refusals(self, k: int, limit_ns: int) -> tuple[str, ...]:
        """Why an audit of `k` challenges within `limit_ns` is outside this certificate (empty: it is inside)."""
        out = []
        if k < self.k:
            out.append(f"k = {k} is below the certificate's {self.k}")
        d = self.effective_rounds(limit_ns)
        if self.cliff is not None and d > self.cliff:
            out.append(f"Δ + RTT = {limit_ns} ns fits {d} calls of {self.call_ns} ns, past the cliff at {self.cliff}")
        elif d > self.rounds:
            out.append(f"Δ + RTT = {limit_ns} ns fits {d} calls of {self.call_ns} ns; the certificate covers "
                       f"{self.rounds}, so Δ + RTT must stay under {(self.rounds + 1) * self.call_ns} ns")
        return tuple(out)

    def to_json(self) -> dict[str, Any]:
        return {"k": self.k, "rounds": self.rounds, "work": self.work, "call_ns": self.call_ns, "theorem": self.theorem,
                "rho": str(self.rho), "delta": str(self.delta), "eps_log2": self.eps_log2, "cliff": self.cliff,
                "segments": self.segments, "hypotheses": list(self.hypotheses), "claims": list(self.claims)}


class Scheme(ABC):
    """A POUS scheme over `params` (its `codec.Geometry`, with the number of segments of `W`) and primitives `prims`
    (the reference backend by default).

    Like `verity.ml.commitments.scheme.CommitmentScheme`, it is a named, versioned format behind a small interface, looked
    up by name (`schemes.get`), with its conformance vectors pinned under the name (`tests/fixtures/schemes/`).  A
    subclass sets `name` (its byte-format version string), `backend` (the reference primitives for its parameters) and
    `certificate` (and `certificates`, when it has several).  It implements `encode_segment` and `decode_segment`, and
    `respond` only when an honest answer is more than the stored label.

    **Segment `s` must be keyed injectively in `s`**, across every segment of `W`: `tag_s = salt ‖ u64 s`
    (`labels.tag`), or one salt with global indices.  With a shared domain, identical segments get identical labels,
    and an adversary storing one of them passes every audit (Lean `shared_domain_breaks`; `segDom_inj` is the
    certificates' hypothesis).  The conformance suite checks it for every registered scheme."""

    name: ClassVar[str] = ""
    experimental: ClassVar[str | None] = None
    """Why the scheme is experimental, or None.  An experimental scheme is never the default; the memory challenger
    audits it only when asked to (`TimedAudit(..., experimental=True)`), and its profile states the reason."""
    backend: ClassVar[Callable[[Any], Primitives]]

    def __init__(self, params: Geometry, prims: Primitives | None = None) -> None:
        self.params = params
        self.prims = type(self).backend(params) if prims is None else prims

    @property
    def certificate(self) -> Certificate | None:
        """The certificate at these parameters; None when no theorem covers them."""
        return None

    @property
    def certificates(self) -> tuple[Certificate, ...]:
        """Every certificate at these parameters, the default (`certificate`, when there is one) first.  A scheme with
        guarantees at several round budgets lists them all; the memory challenger audits under the first that covers its
        `k` and deadline."""
        return () if self.certificate is None else (self.certificate,)

    def deadline_refusals(self, deadline_ns: int, rtt_allowance_ns: int) -> tuple[str, ...]:
        """Why the scheme's spec refuses this `Δ` and RTT allowance apart, beyond what its certificates read from their
        sum (empty: it does not)."""
        return ()

    @abstractmethod
    def encode_segment(self, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
        """Segment `s` of `C`: `blocks_per_segment` labels of `block_bytes` bytes, from segment `s` of `W`."""

    @abstractmethod
    def decode_segment(self, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
        """Segment `s` of `W`, bit-exact, from segment `s` of `C`."""

    def respond(self, block: Callable[[int], bytes], i: int) -> bytes:
        """The honest answer to challenge `i`, from the stored labels of `C`: the label itself."""
        return block(i)

    def codec(self) -> Codec:
        """This scheme as a `codec.Codec`, for the modules that take one (`setup`, `layout`)."""
        def encode(params: Geometry, prims: Primitives, salt: bytes, s: int, w_seg: bytes) -> list[bytes]:
            return self._at(params, prims).encode_segment(salt, s, w_seg)

        def decode(params: Geometry, prims: Primitives, salt: bytes, s: int, blocks: Sequence[bytes]) -> bytes:
            return self._at(params, prims).decode_segment(salt, s, blocks)

        return Codec(self.name, encode, decode)

    def _at(self, params: Geometry, prims: Primitives) -> Scheme:
        if params == self.params and prims is self.prims:
            return self
        if params.with_segments(1) != self.params.with_segments(1):
            raise ValueError(f"{self.name} runs at {self.params.manifest()}, not {params.manifest()}")
        return type(self)(params, prims)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params!r})"


def commit(scheme: Scheme, w: bytes | Iterable[bytes]) -> bytes:
    """The server's commitment to `W` under the scheme's parameters (`setup.commit`)."""
    return setup.commit(scheme.params, w)


def draw_salt(source: bytes, commitment: bytes) -> bytes:
    """The verifier's salt, drawn after the commitment from its own secret or a later beacon round (`setup.draw_salt`)."""
    return setup.draw_salt(source, commitment)


def encode(scheme: Scheme, w: bytes | Iterable[bytes], salt: bytes) -> list[list[bytes]]:
    """`C = Enc(W, salt)`, segment by segment."""
    return [scheme.encode_segment(salt, s, seg) for s, seg in enumerate(split(scheme.params, w))]


def decode(scheme: Scheme, salt: bytes, encoded: Iterable[Sequence[bytes]]) -> bytes:
    """`W = Dec(C, salt)`, segment by segment."""
    return b"".join(scheme.decode_segment(salt, s, blocks) for s, blocks in enumerate(encoded))


def decode_tensor(scheme: Scheme, layout: Layout, salt: bytes, name: str, block: Callable[[int], bytes]) -> bytes:
    """One tensor of `layout`, decoded from the segments of `C` it overlaps (`block(i)` is label `i` of `C`)."""
    if layout.params != scheme.params:
        raise ValueError("the layout's parameters are not the scheme's")
    return _decode_tensor(layout, scheme.prims, salt, name, block, codec=scheme.codec())


def verifier_key(scheme: Scheme, commitment: bytes, salt: bytes, *, mode: str = PRIVATE,
                 w: bytes | Sequence[bytes] | None = None,
                 encoded: Iterable[Sequence[bytes]] | None = None) -> VerifierKey:
    """The verifier's `vk`.  `PRIVATE`: it checks `W` against the commitment and encodes it itself; `encoded` may pass
    its own per-segment encoding (for example from a process pool).  `PUBLIC`: it decodes the server's `encoded`
    segments and accepts only if `Dec(C*)` matches the commitment, never running the encoder.  Raises `ValueError` on
    any mismatch."""
    if mode == PRIVATE:
        if w is None:
            raise ValueError("the private setup needs W")
        return setup.verifier_key(scheme.params, scheme.prims, w, commitment, salt, encoded=encoded, codec=scheme.codec())
    if mode == PUBLIC:
        if encoded is None:
            raise ValueError("the public setup needs the server's C*")
        return setup.verifier_key_from_encoding(scheme.params, scheme.prims, commitment, salt, encoded,
                                                codec=scheme.codec())
    raise ValueError(f"the setup mode is {PRIVATE!r} or {PUBLIC!r}, not {mode!r}")


def challenge_key(source: bytes, vk: VerifierKey) -> Key:
    """The audit's key, bound to `vk`'s root and salt; `source` is the memory challenger's own secret, kept secret until
    the audit starts, never a public beacon or the prover's (`audit.challenge_key`). A public source lets a prover
    compute the challenged indices before the audit and rebuild only those labels within the deadline."""
    return audit.challenge_key(source, vk)


def responder(scheme: Scheme, block: Callable[[int], bytes]) -> Callable[[int], bytes]:
    """The honest responder: challenge `i` answered by `scheme.respond` from the stored labels."""
    return lambda i: scheme.respond(block, i)


@dataclass(frozen=True)
class StorageProfile:
    """What an accepted audit establishes about the server, and what that rests on.

    With a certificate and no `gaps`: a server whose state held fewer than `state_bound_bits` when the challenges
    arrived passes this audit with probability at most `delta + 2^eps_log2`, by the certificate's theorem, given its
    hypotheses, under its claims.  `gaps` lists where the certificate does not cover this audit."""

    scheme: str
    manifest: dict[str, Any]
    blocks: int
    block_bits: int
    k: int
    limit_ns: int
    certificate: Certificate | None
    gaps: tuple[str, ...] = field(default=())
    experimental: str | None = None

    @property
    def certified(self) -> bool:
        """The certificate covers this audit (its open hypotheses aside)."""
        return self.certificate is not None and not self.gaps

    @property
    def state_bound_bits(self) -> int | None:
        """`⌊rho · |C|⌋`, the state below which the certificate bounds acceptance."""
        if self.certificate is None:
            return None
        rho = self.certificate.rho
        return self.blocks * self.block_bits * rho.numerator // rho.denominator

    @property
    def effective_rounds(self) -> int | None:
        return None if self.certificate is None else self.certificate.effective_rounds(self.limit_ns)

    def statement(self) -> str:
        cert = self.certificate
        head = f"EXPERIMENTAL ({self.experimental}). " if self.experimental else ""
        if not self.certified:
            return f"{head}{self.scheme}, {self.k} challenges within {self.limit_ns:,} ns: not certified ({'; '.join(self.gaps)})."
        given = f", given {', '.join(cert.hypotheses)}" if cert.hypotheses else ""
        return (f"{head}A server holding fewer than {self.state_bound_bits:,} bits of state passes {self.k} challenges "
                f"within {self.limit_ns:,} ns with probability at most {cert.delta} + 2^{cert.eps_log2} "
                f"({cert.theorem}{given}; under {', '.join(cert.claims)}).")

    def to_json(self) -> dict[str, Any]:
        return {"schema": "verity-pous/storage-profile/v1", "scheme": self.scheme, "params": self.manifest,
                "blocks": self.blocks, "block_bits": self.block_bits, "k": self.k, "limit_ns": self.limit_ns,
                "effective_rounds": self.effective_rounds, "state_bound_bits": self.state_bound_bits,
                "certificate": None if self.certificate is None else self.certificate.to_json(),
                "gaps": list(self.gaps), "certified": self.certified, "experimental": self.experimental}


class TimedAudit:
    """The timed audit of a scheme: `audit.UnsignedRounds` at the certificate's `k`, with sequential reveal and a strict
    deadline per answer.

    It audits under the first of the scheme's `certificates` that covers `k` and the deadline (with `k` unset, each at
    its own `k`), so the default `k` is the first certificate's: the band's `k = 111` at `D = 5`, and `k = 97` at
    `D = 2` or `k = 93` at `D = 1` only when asked for.  It refuses an audit that none covers, one whose `Δ` or RTT
    allowance the scheme refuses (`Scheme.deadline_refusals`), and a scheme with no certificate at its parameters.  `research=True` runs such an audit anyway, and its profile lists the first
    certificate's refusals as gaps.  An experimental scheme is refused unless `experimental=True`.  The band's cliff is
    refused regardless (`band.check_deployment`); the attack past it is replayed with `audit.simulate`."""

    def __init__(self, scheme: Scheme, vk: VerifierKey, key: Key, *, k: int | None = None,
                 deadline_ns: int = audit.DEADLINE_NS, rtt_allowance_ns: int = audit.RTT_ALLOWANCE_NS,
                 clock: Callable[[], int] = time.perf_counter_ns, root_only: bool = False,
                 research: bool = False, experimental: bool = False) -> None:
        if vk.params != scheme.params:
            raise ValueError("vk was built for other parameters than the scheme's")
        if scheme.experimental and not experimental:
            raise ValueError(f"{scheme.name} is experimental ({scheme.experimental}): pass experimental=True to audit it")
        certs, limit = scheme.certificates, deadline_ns + rtt_allowance_ns
        cert = next((c for c in certs if not c.refusals(c.k if k is None else k, limit)), certs[0] if certs else None)
        uncovered: tuple[str, ...] = ()
        if cert is None:
            if k is None:
                raise ValueError(f"{scheme.name} has no certificate at these parameters: pass k")
            refused = (f"{scheme.name} has no certificate at {scheme.params.manifest()}",)
        else:
            k = cert.k if k is None else k
            refused = cert.refusals(k, limit)
            if cert.segments is not None and scheme.params.segments != cert.segments:
                uncovered = (f"W has {scheme.params.segments} segments; {cert.theorem} is about {cert.segments}",)
        refused += scheme.deadline_refusals(deadline_ns, rtt_allowance_ns)
        if refused and not research:
            others = "".join(f"; under {c.theorem}: {'; '.join(c.refusals(k, limit))}" for c in certs if c is not cert)
            raise ValueError(f"the audit is outside the certificate: {'; '.join(refused)}{others} "
                             "(research=True runs it anyway)")
        self.scheme, self.certificate, self.gaps = scheme, cert, refused + uncovered
        self.timed = audit.UnsignedRounds(vk, key, k, deadline_ns, rtt_allowance_ns, clock, root_only=root_only)

    @property
    def k(self) -> int:
        return self.timed.k

    @property
    def limit_ns(self) -> int:
        return self.timed.limit_ns

    @property
    def done(self) -> bool:
        return self.timed.done

    @property
    def accepted(self) -> bool:
        return self.timed.accepted

    def challenge(self) -> int:
        return self.timed.challenge()

    def respond(self, block: bytes, path: tuple[bytes, ...] | None = None, report: object = None) -> audit.Answer:
        return self.timed.respond(block, path, report)

    def result(self) -> audit.AuditResult:
        return self.timed.result()

    def run(self, respond: Callable[[int], bytes | tuple[bytes, object]],
            path: Callable[[int], tuple[bytes, ...]] | None = None) -> audit.AuditResult:
        """Drive the audit against `respond(index)` until all `k` are answered."""
        return audit.run_timed(self.timed, respond, path)

    def profile(self) -> StorageProfile:
        """The storage profile of the accepted audit."""
        if not self.accepted:
            raise ValueError("only an accepted audit has a storage profile")
        p = self.scheme.params
        return StorageProfile(self.scheme.name, p.manifest(), p.blocks, p.m, self.k, self.limit_ns, self.certificate,
                              self.gaps, self.scheme.experimental)


__all__ = ["LOCAL_CLAIMS", "PRIVATE", "PUBLIC", "Certificate", "Scheme", "StorageProfile", "TimedAudit", "challenge_key",
           "claim", "commit", "decode", "decode_tensor", "draw_salt", "encode", "responder", "verifier_key"]
