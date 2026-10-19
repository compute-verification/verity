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

**The check** is `LinkWindows.unit`, `NetworkLinkWindow` v2{EGRESS, INGRESS}, the AND of three parts:
- the certificates' check, one opaque primitive over the two record leaves' committed digests with no gates, checked on
  the host: each certificate verifies under the network certifier's public key in its certifier session
  (`certifier_session`), names the registered link-window and signs its head's digest, and each head's deadline bit is
  set;
- each end's head check, `head_check.NetworkHeadCheck_v1{L, T, COUNTS}` (EGRESS and INGRESS are those statics), a
  Boolean Definition: the head opens at leaf 2 + direction under the registered root, the hiding read N3 makes of the
  same leaf; its link, direction and window are the link-window's; its status is complete; its T, r, B, τ_b and
  parameters digest are the end's declared grid's; and on an end with an occupancy profile its T counts are the
  profile's row for the window. A window past the profile has no row to give, so it is rejected.

In the unit's Program each head check is one gate that names the Definition (`head_check.gate`), because the commitment
gadgets' Boolean bodies encode to about 140 MB and the proof service encodes the Program on every registration. The
Program's digest names the check, so the spec, the stream and each registration bind to it.

The count check is the part of the egress grid check that needs no frame's bytes, and the only grid check on ingress.
It runs on every link-window, as a structural check must, while N3 runs on drawn windows. N3 recomputes the head's
frames root over its N frames, which holds N to the head's Σ n_t up to a SHA-512 collision (`certificate`: leaf k is
frame k for k < Σ n_t and E_0 after), so together they hold N to the profile's window total. In a proof it is an
equality of each head's 32·T count bits with the profile's row, 32·T − 1 ANDs a head (19,199 at T = 600), beside the
head's opening (`head_check.size`).

Its verifier of record is `LinkWindows.verify`, which runs on the host and reads the opening: the certified records and
their four salts. It accepts when
- the opening reproduces the registered root under the link-window's domain;
- each end's head check accepts (`head_check.reference`, which the tests hold to the circuit);
- the audit (`audit.Audit`) accepts both records under the certifier's key and session: the egress record under the
  declared syncs, and the ingress record on its status and parameters. On an egress end with a profile the audit
  compares no grid (`check_link(grid=False)`): the head check holds the head's status, grid and counts, and the frames'
  content is N3's. Without an egress profile it compares the certified digest with the Program's grid in the clear.

`LinkWindows.stand_in` is the verdict as it was before the head check: the in-the-clear audit with the egress grid
compared, and each head's counts the profile's (`declares`). The tests hold `verify` to it. The two differ only where
an egress window's frames differ from the Program's grid but its counts are the profile's: the stand-in rejects, and N2
accepts, leaving the content to N3.

When the audit is the Lean check's (`lean_audit.Audit`), `audit.Audit` runs beside it as the reference: a link-window
where the two differ rejects, and the verdict says `"by": "lean"` and carries the reference's accept bit (`reference`),
never a violation's kind.

What that establishes, and what it doesn't: the outcome is accept exactly when the opening seals, both head checks
accept and the host audit accepts the committed records, and the outcome record binds the registered root, the draw and
the delivered session. The verifier of record reads the witness, so it is not zero-knowledge: until the unit is proved,
the certificates' check and the head checks run where the certificates may go. One clock per spatial unit across links
(`Audit.syncs`) is not checked here: each registration is one link-window, checked alone. Content consistency (that the
frames are the served computation's) is N3's.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Value as Bits
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.core.protocols.certifier import audit as host_audit, head_check as HC
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
    raise NotImplementedError("the certificates' check has no gates: its verifier of record is LinkWindows.verify")


_RECORDS = (("egress", Bits(512)), ("ingress", Bits(512)))
LEAVES = 4
#: the leaf of the head of the record at leaf i (0 egress, 1 ingress) is HEADS + i
HEADS = HC.HEADS
_CERTIFICATES = PrimitiveDefinition(
    "NetworkCertificatesCheck", 1, _RECORDS, Bits(1), _host_only,
    doc="the link-window's two certified records from their committed digests: each certificate verifies in the "
        "certifier session, names the link-window and signs its head, which is settled by the deadline; no gates, "
        "checked on the host",
    register=False)
_ENDS = ("egress", "ingress")


def _heads(S_) -> tuple[tuple[str, Any], ...]:
    return tuple((f"{end}_{name}", t) for end, key in zip(_ENDS, (S_.EGRESS, S_.INGRESS))
                 for name, t in HC.gate(*key).params)


def _body(B, S_, *args):
    n = len(HC.gate(*S_.EGRESS).params)
    oks = [B.call(_CERTIFICATES, *args[:2]), B.call(HC.gate(*S_.EGRESS), *args[2:2 + n]),
           B.call(HC.gate(*S_.INGRESS), *args[2 + n:])]
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BL.BIT), bits))


_UNIT = CompositeDefinition(
    "NetworkLinkWindow", 2, ("EGRESS", "INGRESS"), lambda S_: (_RECORDS + _heads(S_), Bits(1)), _body,
    doc="the proof unit of network accounting: one link-window, both directions; EGRESS and INGRESS are each end's head "
        "check's (L, T, COUNTS)", register=False)


def unit(egress: tuple[int, int, int], ingress: tuple[int, int, int]):
    """N2's proof unit for heads of the size classes `egress` and `ingress`, each (L, T, COUNTS) (`head_check.unit`)."""
    return bind(_UNIT, EGRESS=tuple(egress), INGRESS=tuple(ingress))

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

    def head_key(self, direction: int) -> tuple[int, int, int]:
        """The end's head check's size class (L, T, COUNTS): the link name's UTF-8 length, the end's buckets, and 1 when
        the end has an occupancy profile."""
        g = (self.params, self.ingress_params)[direction]
        return len(self.link.encode("utf-8")), g.buckets, int(self.occupancy[direction] is not None)

    def grid(self, direction: int) -> bytes:
        """The end's declared grid as its head encodes it (`head_check.grid_bytes`): the egress grid's parameters digest
        binds Σ_sync, the ingress grid's none."""
        if direction == 0:
            return HC.grid_bytes(self.params, parameters_digest(self.params, self.sync_set))
        return HC.grid_bytes(self.ingress_params, parameters_digest(self.ingress_params))

    @cached_property
    def unit(self):
        """N2's proof unit for this link (`unit`): its heads' size classes are the link's."""
        return unit(self.head_key(0), self.head_key(1))

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    @cached_property
    def _program(self) -> Program:
        return P.population_program(self.unit, 1)

    @cached_property
    def _partition(self) -> dict[str, Any]:
        return P.partition(self._program, self.query)

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({
            "format": SPEC, "link": self.link, "run": self.run, "key": self.key.key_id.hex(),
            "certifier_session": self.certifier_session.hex(), "charge": self.charge,
            "parameters": {"egress": parameters_digest(self.params, self.sync_set).hex(),
                           "ingress": parameters_digest(self.ingress_params).hex()},
            "occupancy": [None if p is None else _sha512([list(row) for row in p]) for p in self.occupancy]})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query, public_outputs=OUTPUTS,
                        proof_deadline=self.proof_deadline)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, PUBLIC, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        """The challenger's stream for this link (`S.Stream`'s budget, delta, challenger, coins and clock)."""
        return S.Stream(self.spec, self.unit, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """One-stage's served domain (`registration.served_domain`) for the link-window: run `<run>/<link>/<window>`."""
        binding = identity_digest_sha512(R.SERVED_DOMAIN, {
            "program": self._partition["program"], "partition": P.digest(self._partition), "port": VALUE,
            "schema": SCHEMA, "leaves": LEAVES, "run": f"{self.run}/{self.link}/{window}"})
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
        return R.record(self._program, self.query, {"law": "subset", "k": 1}, S.SCHEME, roots, self.provenance(w),
                        previous=previous)

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

    # the verifier of record, on the host until `unit` is proved

    def head_checks(self, opening: Opening, root: bytes) -> list[HC.Instance | None]:
        """Each end's head check on the opening under `root`, the registered root: the link-window's public inputs with
        the end's grid and its profile's row, and as witness the head, its salt and its two siblings. None where the
        unit takes no instance: a window past the end's profile, which has no row to give, or a head whose length isn't
        the unit's."""
        w, dom = opening.window, self.domain(opening.window)
        levels = tree_levels(dom, self.leaves(opening))
        out: list[HC.Instance | None] = []
        for d, record in enumerate(opening.records):
            L, T, _ = self.head_key(d)
            profile, head, i = self.occupancy[d], record.head.encode(), HEADS + d
            if (profile is not None and w >= len(profile)) or len(head) != HC.Shape(L, T).HL:
                out.append(None)
                continue
            out.append(HC.Instance(root, dom.domain_id, self.link.encode("utf-8"), d, w, self.grid(d),
                                   None if profile is None else tuple(profile[w]), head, opening.salts[i],
                                   tuple(level[(i >> k) ^ 1] for k, level in enumerate(levels[:-1])), T))
        return out

    def _audit(self, audit: Any, opening: Opening, frames: Sequence[LogicalFrame], syncs: Sequence[ClockSync],
               grid: bool) -> tuple[list[Any], list[Any]]:
        w, (egress, ingress) = opening.window, opening.records
        out = audit.check_link(self.link, [egress], frames, list(syncs)[:w + 1], self.sync_set, self.params, key=self.key,
                               session=self.certifier_session, grid=grid)
        inn = audit.check_ingress_link(self.link, [ingress], w + 1, key=self.key, params=self.ingress_params,
                                       session=self.certifier_session)
        return out, inn

    def _registered(self, opening: Opening, draw: S.Draw) -> str:
        reg = draw.registration.record
        if opening.window != reg["window"]["window"] or len(opening.records) != 2:
            raise ValueError("the opening is not the registered link-window's")
        return reg["roots"][0]["root"]

    def verify(self, opening: Opening, draw: S.Draw, frames: Sequence[LogicalFrame],
               syncs: Sequence[ClockSync]) -> dict[str, Any]:
        """The verdict of record on the delivered opening: accepted when it reproduces the registered root under the
        link-window's domain, both head checks accept, and the audit accepts both records, comparing the egress grid
        only on an egress end without a profile (the module docstring). `frames` are the egress Program's logical
        frames and `syncs` the declared clock syncs of windows 0 … w. The verdict reports the public outputs and never
        the violation's kind."""
        root = self._registered(opening, draw)
        w, (egress, ingress) = opening.window, opening.records
        sealed = self.root(opening) == root
        headed = all(i is not None and HC.reference(i) for i in self.head_checks(opening, bytes.fromhex(root)))
        audit = Audit()
        out, inn = self._audit(audit, opening, frames, syncs, grid=self.occupancy[0] is None)
        accepted = sealed and headed and _accepts(out, inn, w)
        public = {"verdict": int(accepted), "deadline": int(egress.head.deadline and ingress.head.deadline)}
        sessions = [self.session(opening, draw)]
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                          "accepted": accepted, "public": public})
        verdict = {"of_record": True, "accepted": accepted, "digest": digest, "sessions": sessions, "public": public}
        reference = getattr(audit, "reference", None)
        if reference is not None:
            verdict |= {"by": "lean",
                        "reference": {"accepted": sealed and headed
                                      and _accepts(reference[:len(out)], reference[len(out):], w)}}
        return verdict

    def stand_in(self, opening: Opening, draw: S.Draw, frames: Sequence[LogicalFrame],
                 syncs: Sequence[ClockSync]) -> bool:
        """`verify`'s accept bit as it was before the head check, the reference its tests hold `verify` to: the opening
        reproduces the registered root, the host audit (`audit.Audit`) accepts both records with the egress grid
        compared in the clear, and each head's counts are the profile's (`declares`)."""
        root = self._registered(opening, draw)
        out, inn = self._audit(host_audit.Audit(), opening, frames, syncs, grid=True)
        counted = all(self.declares(d, opening.window, r.head.counts) for d, r in enumerate(opening.records))
        return self.root(opening) == root and counted and _accepts(out, inn, opening.window)

    def statement(self, opening: Opening) -> dict[str, Any]:
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link, "window": opening.window,
                "root": self.root(opening)}

    def outcome(self, stream: S.Stream, draw: S.Draw, opening: Opening, frames: Sequence[LogicalFrame],
                syncs: Sequence[ClockSync]) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict, with the root the opening reproduces as the session's,
        which one-stage's audit holds to the registered root."""
        return stream.outcome(draw, {VERIFIER: self.verify(opening, draw, frames, syncs)}, self.statement(opening),
                              session_roots={VALUE: self.root(opening)})


__all__ = ["HEADS", "LEAVES", "LinkWindows", "OUTPUTS", "Opening", "PUBLIC", "SCHEMA", "VALUE", "encode", "rows",
           "unit"]
