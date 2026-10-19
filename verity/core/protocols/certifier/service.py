"""One link-window's network certificates through the proof service (N2): the network `Spec`, the firewall's salted
commitment to the link-window's certified records, and the check's verifier of record.

**What is committed.** One `Value`, `certificates`: four leaves under one salted root, the link-window's two certified
records (`CertifiedRecord`), egress leaf 0 and ingress leaf 1, and their heads, egress leaf 2 and ingress leaf 3
(`rows`).
Each leaf is `hm96-sha512/row/v2` over its bytes, salted with its own fresh 192-byte salt, in a `frame-v3-sha512` tree
under one-stage's served domain, whose run is the link and the window. A record leaf's bytes are `encode`'s (the record
head, then the certificate), and a head leaf's are the head's alone (`RecordHead.encode`). A proof that reads only the
head, such as N3's, which holds the window's frames to the head's root, opens a head leaf in the head's two or three
SHA-512 compressions instead of the certificate's 272. The certificates and heads are witness: the challenger sees the
root, never a certificate or a head.

**Selection** is `All`: every link-window is checked, so an accepted outcome is the exhaustive fact. A registration is
one link-window, one proof unit. Its `window` provenance is `{"kind": "served", "window": w}`, and the challenger holds w
to the registration's order in the stream (`LinkWindows.register`), so the receipt's order is the link-window.

**Public items**, each with its reason (`PUBLIC`): the population (one), the window (its index, the order), the
committed value's leaf count and domain, its salted root, the parameters digests, the charge and the occupancy profile
(fixed before the run, bound by the spec's context), and the check's two public outputs, `verdict` and `deadline`. A
rejected link-window shows only the service's `reject`: the violation's kind stays in the verifier.

**The occupancy profile** (`LinkWindows.occupancy`) is the placement's public occupancy for each end of the link: each
window's T counts, the frames that end's record holds in each bucket, as the Program and the placement fix them. An end
whose rows the Program reads rather than predicts, such as requests from outside, has none.

**The check** is `UNIT`, `NetworkLinkWindow` v1: one opaque primitive over the four leaves' committed digests, returning
the verdict bit. It has no gates. Its Program's digest names the check, so the spec, the stream and each registration
bind to it, but nothing proves it in a circuit yet. Its verifier of record is `LinkWindows.verify`, which runs on the
host and reads the opening: the certified records and their four salts. It accepts when
- the opening reproduces the registered root under the link-window's domain, so each head leaf holds its record's
  head;
- the audit (`audit.Audit`) accepts the egress record against the Program's grid under the declared syncs, and the
  ingress record on its status and parameters, both under the network certifier's public key in its certifier session
  (`certifier_session`); so each certificate verifies, names the registered link-window and signs its head's digest,
  and each head's deadline bit is set;
- each head's counts are the occupancy profile's for its end and window, on every end that has a profile (`declares`).
  A window past the profile is rejected.

The count check is the part of the egress grid check that needs no frame's bytes, and the only grid check on ingress.
It runs on every link-window, as a structural check must, while N3 runs on drawn windows. N3 recomputes the head's
frames root over its N frames, which holds N to the head's Σ n_t up to a SHA-512 collision (`certificate`: leaf k is
frame k for k < Σ n_t and E_0 after), so together they hold N to the profile's window total. In a proof of `UNIT` it is
an equality of each head's 32·T count bits with the profile's row, 32·T − 1 ANDs a head (19,199 at T = 600) beside the
head's opening.

When the audit is the Lean check's (`lean_audit.Audit`), `audit.Audit` runs beside it as the reference: a link-window
where the two differ rejects, and the verdict says `"by": "lean"` and carries the reference's accept bit (`reference`),
never a violation's kind.

What that establishes, and what it doesn't: the outcome is accept exactly when the host audit accepts the committed
records and their counts are the profile's, and the outcome record binds the registered root, the draw and the delivered
session. The stand-in reads the witness, so it is not zero-knowledge: until the check is a Program with a proof, the
verifier of record must run where the certificates may go. One clock per spatial unit across links (`Audit.syncs`) is not checked here: each registration
is one link-window, checked alone. Content consistency (that the frames are the served computation's) is N3's.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Value as Bits
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.certificate import CertifiedRecord, PublicKey, parameters_digest
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

SPEC = "verity-network-certifier/service-spec/v0"
SESSION = "verity-network-certifier/session/v0"
STATEMENT = "verity-network-certifier/statement/v0"
VERDICT = "verity-network-certifier/verdict/v0"
VALUE = "certificates"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "network-audit"
_INST = hm96.SHA512


def _host_only(egress: int, ingress: int) -> int:
    raise NotImplementedError("the link-window check has no gates: its verifier of record is LinkWindows.verify")


_LEAVES = (("egress", Bits(512)), ("ingress", Bits(512)), ("egress_head", Bits(512)), ("ingress_head", Bits(512)))
LEAVES = len(_LEAVES)
#: the leaf of the head of the record at leaf i (0 egress, 1 ingress) is HEADS + i
HEADS = 2
_CHECK = PrimitiveDefinition(
    "NetworkLinkWindowCheck", 1, _LEAVES, Bits(1), _host_only,
    doc="one link-window's verdict from its two certified records' and their heads' committed digests; no gates, "
        "checked on the host",
    register=False)
UNIT = bind(CompositeDefinition(
    "NetworkLinkWindow", 1, (), lambda S_: (_LEAVES, Bits(1)),
    lambda B, S_, *leaves: B.call(_CHECK, *leaves),
    doc="the proof unit of network accounting: one link-window, both directions", register=False))
QUERY = P.template_instance_query(UNIT)
_PROGRAM = P.population_program(UNIT, 1)
_PARTITION = P.partition(_PROGRAM, QUERY)

PUBLIC = (
    S.Public("population", "structure", "one proof unit per registration: the link-window"),
    S.Public("window", "structure", "provenance: the link-window's index, held to the registration's order"),
    S.Public(f"leaves:{VALUE}", "structure", "four leaves, the egress and ingress certified records and their heads"),
    S.Public(f"domain:{VALUE}", "structure", "the served domain, which binds the check, the link and the window"),
    S.Public(VALUE, "commitment", "the link-window's certified records under one salted root"),
    S.Public("parameters", "structure", "the parameters digests (W1) the certificates sign, fixed before the run"),
    S.Public("charge", "structure", "log2 #Σ_sync bits per window, the clock-sync advice the accepted-runs bound charges"),
    S.Public("occupancy", "structure", "each end's frames per bucket in each window, the placement's occupancy profile, "
             "fixed before the run; none for an end the Program reads rather than predicts"),
    S.Public("verdict", "output", "the link-window's verdict, accept or reject; a violation's kind stays hidden"),
    S.Public("deadline", "output", "the certifier settled both records by the verdict deadline"),
)
OUTPUTS = ("verdict", "deadline")


def encode(record: CertifiedRecord) -> bytes:
    """The leaf's bytes: the record head (`RecordHead.encode`), then the certificate, every field fixed-width or
    length-prefixed: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖ digest (64) ‖
    u32be len(signature) ‖ signature."""
    c = record.certificate
    link = c.link.encode("utf-8")
    return b"".join([record.head.encode(), struct.pack(">H", len(c.key_id)), c.key_id, struct.pack(">H", len(link)), link,
                     struct.pack(">BQ", c.direction.code, c.window), c.digest, struct.pack(">I", len(c.signature)),
                     c.signature])


def rows(records: Sequence[CertifiedRecord]) -> list[bytes]:
    """The four leaves' bytes, in leaf order: each record's `encode`, then each record's head alone."""
    return [encode(r) for r in records] + [r.head.encode() for r in records]


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


def _accepts(egress: Sequence[Any], ingress: Sequence[Any], window: int) -> bool:
    return len(egress) > window and egress[window].accepted and ingress[window].accepted


@dataclass(frozen=True)
class Opening:
    """The firewall's side of one link-window's commitment, all witness: the egress and the ingress certified records,
    in leaf order, and one salt per leaf (`rows`'s order: the two records', then the two heads')."""

    window: int
    records: tuple[CertifiedRecord, CertifiedRecord]
    salts: tuple[bytes, bytes, bytes, bytes]


@dataclass(frozen=True)
class LinkWindows:
    """Network accounting's protocol on one link, every field public: the link, its egress and ingress grids, Σ_sync, the
    network certifier's public key, the run that names the served domains, the certifier session's id (the 32 bytes the
    challenger drew for the certifier's session, which every certificate signs), the proof deadline in seconds, and the
    occupancy profile: for each end (egress, ingress), each window's counts, or None for an end without one."""

    link: str
    params: GridParams
    sync_set: SyncSet
    key: PublicKey
    run: str
    certifier_session: bytes
    ingress: GridParams | None = None
    proof_deadline: float | None = None
    occupancy: tuple[Sequence[Sequence[int]] | None, Sequence[Sequence[int]] | None] = (None, None)

    def __post_init__(self) -> None:
        if len(self.occupancy) != 2:
            raise ValueError("the occupancy profile has one entry per end: (egress, ingress)")
        for end, profile, g in zip(("egress", "ingress"), self.occupancy, (self.params, self.ingress_params)):
            for w, row in enumerate(profile or ()):
                if len(row) != g.buckets or any(type(n) is not int or not 0 <= n <= g.allocation for n in row):
                    raise ValueError(f"{end} window {w}: an occupancy row is {g.buckets} counts of 0 … {g.allocation}")

    @property
    def ingress_params(self) -> GridParams:
        return self.ingress or self.params

    def declares(self, direction: int, window: int, counts: Sequence[int]) -> bool:
        """Whether `counts` are the occupancy profile's for the end (0 egress, 1 ingress) in the window: always for an
        end with no profile, never for a window past its profile."""
        profile = self.occupancy[direction]
        return profile is None or (window < len(profile) and tuple(counts) == tuple(profile[window]))

    @property
    def charge(self) -> float:
        """log2 #Σ_sync: 7 bits per window at 16 × 8."""
        return self.sync_set.bits

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({
            "format": SPEC, "link": self.link, "run": self.run, "key": self.key.key_id.hex(),
            "certifier_session": self.certifier_session.hex(), "charge": self.charge,
            "parameters": {"egress": parameters_digest(self.params, self.sync_set).hex(),
                           "ingress": parameters_digest(self.ingress_params).hex()},
            "occupancy": [None if p is None else _sha512([list(row) for row in p]) for p in self.occupancy]})
        check = S.Check(P.program_sha512(Program(UNIT)), QUERY, public_outputs=OUTPUTS, proof_deadline=self.proof_deadline)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, PUBLIC, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        """The challenger's stream for this link (`S.Stream`'s budget, delta, challenger, coins and clock)."""
        return S.Stream(self.spec, UNIT, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """One-stage's served domain (`registration.served_domain`) for the link-window: run `<run>/<link>/<window>`."""
        binding = identity_digest_sha512(R.SERVED_DOMAIN, {
            "program": _PARTITION["program"], "partition": P.digest(_PARTITION), "port": VALUE, "schema": SCHEMA,
            "leaves": LEAVES, "run": f"{self.run}/{self.link}/{window}"})
        return CommitmentDomain(binding, -1, RangeIndexedDomain(0, LEAVES), hash="sha512")

    def ports(self, window: int) -> dict[str, dict[str, Any]]:
        """The challenger's own copy of the link-window's port: four leaves, under its domain."""
        return {VALUE: {"leaves": LEAVES, "domain": self.domain(window).domain_id.hex()}}

    @staticmethod
    def provenance(window: int) -> dict[str, Any]:
        return {"kind": "served", "window": window}

    # the firewall

    def commit(self, window: int, egress: CertifiedRecord, ingress: CertifiedRecord,
               salts: Sequence[bytes] | None = None) -> Opening:
        """The link-window's opening. `salts`: four distinct salts, one per leaf, drawn once (default: fresh from the
        operating system)."""
        salts = tuple(salts) if salts is not None else tuple(_INST.fresh_salt() for _ in range(LEAVES))
        if len(salts) != LEAVES or len(set(salts)) != LEAVES or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"{LEAVES} distinct {_INST.salt_bytes}-byte salts")
        return Opening(window, (egress, ingress), tuple(bytes(s) for s in salts))  # type: ignore[arg-type]

    def leaves(self, opening: Opening) -> list[bytes]:
        """The salted leaves, which hide the records and their heads: `hm96-sha512/row/v2` over each of `rows`."""
        dom, key = self.domain(opening.window), _INST.default_key
        out = []
        for i, (row, salt) in enumerate(zip(rows(opening.records), opening.salts, strict=True)):
            bc = _INST.commit_string(key, sha512_row_v2_digest(row, 8 * len(row)), salt)
            out.append(row_leaf(dom, i, _INST.tree_leaf(key, bc), SCHEMA))
        return out

    def root(self, opening: Opening) -> str:
        return tree_levels(self.domain(opening.window), self.leaves(opening))[-1][0].hex()

    def registration(self, opening: Opening, previous: str | None = None) -> dict[str, Any]:
        """One-stage's record of the link-window: its salted root and nothing of the certificates. `previous`: the
        digest of the stream's previous registration."""
        w = opening.window
        roots = [{"port": VALUE, "root": self.root(opening), **self.ports(w)[VALUE]}]
        return R.record(_PROGRAM, QUERY, {"law": "subset", "k": 1}, S.SCHEME, roots, self.provenance(w), previous=previous)

    def session(self, opening: Opening, draw: S.Draw) -> str:
        """The delivered session's SHA-512: the registration, the draw and the salted leaves the opening reproduces."""
        return _sha512({"format": SESSION, "registration": draw.registration.digest, "draw": _sha512(dict(draw.draw)),
                        "leaves": [x.hex() for x in self.leaves(opening)]})

    # the challenger

    def register(self, stream: S.Stream, record: Mapping[str, Any]) -> S.Registration:
        """`stream.register` with the link-window's own ports, after holding the record's `window` provenance to exactly
        `{"kind": "served", "window": order}`: the service declares the provenance one public item and doesn't read it,
        and anything else there would reach the auditor."""
        window = len(stream.records)
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        return stream.register(record, self.ports(window))

    # the verifier of record, a host-side stand-in for a proof of `UNIT`

    def verify(self, opening: Opening, draw: S.Draw, frames: Sequence[LogicalFrame],
               syncs: Sequence[ClockSync]) -> dict[str, Any]:
        """The verdict of record on the delivered opening: accepted when it reproduces the registered root under the
        link-window's domain and the audit accepts both records (the module docstring). `frames` are the egress
        Program's logical frames and `syncs` the declared clock syncs of windows 0 … w. The verdict reports the public
        outputs and never the violation's kind."""
        reg = draw.registration.record
        w = reg["window"]["window"]
        if opening.window != w or len(opening.records) != 2:
            raise ValueError("the opening is not the registered link-window's")
        egress, ingress = opening.records
        sealed = self.root(opening) == reg["roots"][0]["root"]
        audit = Audit()
        out = audit.check_link(self.link, [egress], frames, list(syncs)[:w + 1], self.sync_set, self.params, key=self.key,
                               session=self.certifier_session)
        inn = audit.check_ingress_link(self.link, [ingress], w + 1, key=self.key, params=self.ingress_params,
                                       session=self.certifier_session)
        counted = all(self.declares(d, w, r.head.counts) for d, r in enumerate(opening.records))
        accepted = sealed and counted and _accepts(out, inn, w)
        public = {"verdict": int(accepted), "deadline": int(egress.head.deadline and ingress.head.deadline)}
        sessions = [self.session(opening, draw)]
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                          "accepted": accepted, "public": public})
        verdict = {"of_record": True, "accepted": accepted, "digest": digest, "sessions": sessions, "public": public}
        reference = getattr(audit, "reference", None)
        if reference is not None:
            verdict |= {"by": "lean",
                        "reference": {"accepted": sealed and counted
                                      and _accepts(reference[:len(out)], reference[len(out):], w)}}
        return verdict

    def statement(self, opening: Opening) -> dict[str, Any]:
        return {"format": STATEMENT, "check": P.program_sha512(Program(UNIT)), "link": self.link,
                "window": opening.window, "root": self.root(opening)}

    def outcome(self, stream: S.Stream, draw: S.Draw, opening: Opening, frames: Sequence[LogicalFrame],
                syncs: Sequence[ClockSync]) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict, with the root the opening reproduces as the session's,
        which one-stage's audit holds to the registered root."""
        return stream.outcome(draw, {VERIFIER: self.verify(opening, draw, frames, syncs)}, self.statement(opening),
                              session_roots={VALUE: self.root(opening)})


__all__ = ["HEADS", "LEAVES", "LinkWindows", "OUTPUTS", "Opening", "PUBLIC", "QUERY", "SCHEMA", "UNIT", "VALUE",
           "encode", "rows"]
