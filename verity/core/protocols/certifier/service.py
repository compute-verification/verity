"""One link-window's network certificates through the proof service (N2): the network `Spec`, the firewall's salted
commitment to the link-window's certified records, and the check's verifier of record.

**What is committed.** One `Value`, `certificates`: four leaves under one salted root, the link-window's two certified
records (`CertifiedRecord`), egress leaf 0 and ingress leaf 1, and their heads, egress leaf 2 and ingress leaf 3
(`rows`).
Each leaf is `hm96-sha512/row/v2` over its bytes, salted with its own fresh 192-byte salt, in a `frame-v3-sha512` tree
under one-stage's served domain, whose run is the link and the window. A record leaf's bytes are `encode`'s (the record
head, the certifier's salt for it, then the certificate), and a head leaf's are the head's alone (`RecordHead.encode`).
A proof that reads only the head, such as N3's, which is designed to hold the window's frames to the head's root, opens
a head leaf in the head's two or three SHA-512 compressions instead of the certificate's 272. The certificates, salts
and heads are witness: the challenger sees the root, never a certificate, a salt or a head.

**Selection** is `All`: every link-window is checked, so an accepted outcome is the exhaustive fact. A registration is
one link-window, one proof unit. Its `window` provenance is `{"kind": "served", "window": w}`, and the challenger holds w
to the registration's order in the stream (`LinkWindows.register`), so the receipt's order is the link-window. The
network verdict of record is `LinkWindows.network_verdict`, the AND over the link's link-windows with no ε or share
credit: it accepts only when every window of the egress profile (a served or advised link's profile) is registered in
order and its outcome of record accepted, so a rejected or late window, one with no outcome, and a gap in the
registrations each reject the link. A
run's is the AND over its links'. The datacenter model (`benchmarks/datacenter`, a Python model) builds no
`LinkWindows` outcomes, so its account doesn't read it, though `Account.sound` is the same AND over its own port
records.

**Public items**, each with its reason (`PUBLIC`): the population (one), the window (its index, the order), the
committed value's leaf count and domain, its salted root, the parameters digests, the charge and the occupancy profile
(fixed before the run, bound by the spec's context), and the check's two public outputs, `verdict` and `deadline`. A
rejected link-window shows only the service's `reject`: the violation's kind stays in the verifier.

**The occupancy profile** (`LinkWindows.occupancy`) is the placement's public occupancy for each end of the link: each
window's T counts, the frames that end's record holds in each bucket, as the Program and the placement fix them. An end
whose rows the Program reads rather than predicts, such as requests from outside, has none. The profile is fixed before
the run and bound by the spec's context, and nothing declared at run time sets it. So a port-window it leaves empty can
pass N3's design from public occupancy, with no N3 unit: a head with frames where the profile says 0 fails N2's count
check. An end without a profile has no such backing, since its occupancy is its record's own.

**The check** is `LinkWindows.unit`, `NetworkLinkWindow` v4{EGRESS, INGRESS}, over the two certificates' signed digests
(`certificate_digest`: each end's record digest hidden under the certifier's salt), the AND of three parts:
- the certificates' check, one opaque primitive over the two signed digests with no gates, checked on the host: each
  certificate verifies under the network certifier's public key in its certifier session (`certifier_session`) over
  its public fields and its signed digest, names the registered link-window, and its head's deadline bit is set;
- each end's head check, `head_check.NetworkHeadCheck_v2{L, T, COUNTS}` (EGRESS and INGRESS are those statics), a
  Boolean Definition over its end's signed digest: the head opens at leaf 2 + direction under the registered root, the
  hiding read N3 makes of the same leaf; its link, direction and window are the link-window's; its status is complete;
  its T, r, B, τ_b and parameters digest are the end's declared grid's; on an end with an occupancy profile its T
  counts are the profile's row for the window; and its record digest under the certifier's salt is the signed digest,
  so the certificate signs the head the check opened. A window past the profile has no row to give, so it is rejected.

In the unit's Program each head check is one gate that names the Definition and its circuit digest (`head_check.gate`,
`Gate[NetworkHeadCheck_v2{L,T,COUNTS}@<digest>]_v2`), because the commitment gadgets' Boolean bodies encode to about
140 MB and the proof service encodes the Program on every registration. The digest is core's `program_digest` of the
Definition's Program (`head_check.circuit_digest`), the one a C-Flock statement names a template by, pinned in
`head_check.circuit-check.json` and held to the Definition by circuit-check. So the Program's digest binds the head
check's circuit, and the spec, the stream and each registration bind to it: a proof of another circuit with the same
interface names another Program, which the verifier of record refuses (`LinkWindows.proved`), and a proof's circuit
puts the Definition where its gate stands only once the digest is the gate's (`head_check.expand`). The digest is read
from the pin, never computed, since tracing the gadgets takes minutes: a size class no sidecar pins has a gate that
names none (`@unpinned`), and no proof of its unit is accepted, while its verdict of record in the clear is unchanged.

The count check is the part of the egress grid check that needs no frame's bytes, and the only grid check on ingress.
It runs on every link-window, as a structural check must. In a proof it is an equality of each head's 32·T count bits
with the profile's row, 32·T − 1 ANDs a head (19,199 at T = 600), beside the head's opening (`head_check.size`).

N3, the content check, is a design and the simulator's behaviour, not part of this protocol: a prototype unit evaluated
in the clear (`benchmarks/network_sim`, "N3 in zero knowledge"), in no proof-service Spec and not proved. Its design
runs it on every port-window that recorded frames, recomputing the head's frames root over its N frames, which would
hold N to the head's Σ n_t up to a SHA-512 collision (`certificate`: leaf k is frame k for k < Σ n_t and E_0 after).

Its verifier of record is `LinkWindows.verify`, which runs on the host and reads the opening: the certified records, the
certifier's salts, and the four leaves' salts. It accepts when
- the opening reproduces the registered root under the link-window's domain;
- each end's head check accepts (`head_check.reference`, which the tests hold to the circuit);
- the audit (`audit.Audit`) accepts both records under the certifier's key and session, comparing no grid
  (`check_link(grid=False)`): the egress record on its status, its signature and the declared syncs, and the ingress
  record on its status and parameters;
- and, given the findings of a proof of the unit (`proved`, the verifier of record's run over the delivered proof,
  `proofs.verify`), every gate of the unit names a pinned circuit digest and every session accepts and names this
  link's registered Program (`LinkWindows.proved`).

N2 compares no grid in the clear and takes no frames, so its claim is counts only: on every link-window the egress
head's counts are the profile's (and the ingress head's, where that end declares one), which puts the link's timing at
the profile. The egress grid is the head's counts filled with the head's frames in order (`grid.fill`), and the head's
digest binds both, so the frames are the rest of the grid, and N2 says nothing of them. Until N3 is a proof-service
unit with Selection `All`, nothing here holds the egress grid to the Program's.

**A served link** (`LinkWindows(served=served.ServedProfile(…))`) has egress counts that follow its requests' arrivals
and the run-time syncs, so no profile fixes them before the run. Its unit is `NetworkServedLinkWindow` v3{EGRESS,
INGRESS, SERVED}: the checks above, and one gate naming `served.NetworkServedCounts_v1` and its pinned circuit digest
(`served.gate`, pinned in `served.circuit-check.json`), which holds the egress head's
counts to the ones the ingress head's counts derive under the window's declared sync (`served`'s docstring has the
derivation, its premises and the carry-over). Its public items are `PUBLIC_SERVED`: no egress profile, the served
sizes, and the window's declared sync, the advice `charge` already counts, which the served counts read. Its verifier
of record runs `served_check` too.

**An advised link** (`LinkWindows(advised=advised.AdvisedProfile(…))`) is a served link under arbitrary traffic:
responses of any length, batched and timed as serving chose, carried across windows. The prover's advice (one per
certifier session, after serving, charged by `advice.bits`) states what serving chose, and its unit is
`NetworkAdvisedLinkWindow` v1{EGRESS, INGRESS, ADVISED}: the checks above, and one gate naming
`advised.NetworkAdvisedCounts_v1` and its pinned circuit digest (`advised.gate`, pinned in
`advised.circuit-check.json`), which holds the egress head's counts to the release of the advice's frames and the
ingress head's counts to the advice (`advised`'s docstring has the fields it reads, the soundness argument and the
premises). Its public inputs are the window's view of the advice (`advised.view`), which the verifier computes from the
public advice and the declared syncs; the heads' counts stay witness. Its public items are `PUBLIC_ADVISED`. Its
verifier of record runs `advised_check` on the delivered advice (`verify(advice=…)`), refusing a window without one,
and reports the advice it read as a third public output, `advice` (`advised.digest`); the network verdict refuses a
window that names another advice than the first accepted window's, so every window's counts are one advice's.

A link whose egress end has no occupancy profile and is neither served nor advised has no verdict (`ValueError`,
failing closed): nothing would hold its egress timing.

`LinkWindows.stand_in` is the in-the-clear verdict, the Program's egress grid compared as it was before the head check,
and each head's counts the profile's (`declares`). It reads the frames, so it is the tests' oracle, never a verdict:
`verify` accepts every link-window it accepts, and differs from it only where the record's grid, at the profile's
counts, isn't the Program's: in its frames, which N2 doesn't hold (N3's design would), or because the declared syncs
move the Program's counts off the profile, which N2 holds the counts to instead.

When the audit is the Lean check's (`lean_audit.Audit`), `audit.Audit` runs beside it as the reference: a link-window
where the two differ rejects, and the verdict says `"by": "lean"` and carries the reference's accept bit (`reference`),
never a violation's kind.

What that establishes, and what it doesn't: the outcome is accept exactly when the opening seals, both head checks
accept and the host audit accepts the committed records, and the outcome record binds the registered root, the draw and
the delivered session. The verifier of record reads the witness, so it is not zero-knowledge: until the unit is proved,
the head checks run where the heads and the certifier's salts may go, and the audit checks each signed digest against
its head and salt there too (`CertifiedRecord.verify`). Once it is proved, the certificates' check reads each
certificate's public fields and its signed digest, which hides the head under the certifier's salt, and each head's
deadline bit, which is the public `deadline` output. One clock per spatial unit across links
(`Audit.syncs`) is not checked here: each registration is one link-window, checked alone. The frames, their content
and their order within the counts, are left to N3: an accepted outcome says nothing of them, so the claim is counts
only (timing at the profile) until N3 is a proof-service unit with Selection `All`.
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
from verity.core.protocols.certifier import advised as AD, audit as host_audit, head_check as HC, served as SV
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.certificate import CertifiedRecord, PublicKey, parameters_digest
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

SPEC = "verity-network-certifier/service-spec/v0"
SESSION = "verity-network-certifier/session/v0"
STATEMENT = "verity-network-certifier/statement/v0"
VERDICT = "verity-network-certifier/verdict/v0"
NETWORK_VERDICT = "verity-network-certifier/network-verdict/v0"
VALUE = "certificates"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "network-audit"
_INST = hm96.SHA512


def _host_only(egress: int, ingress: int) -> int:
    raise NotImplementedError("the certificates' check has no gates: its verifier of record is LinkWindows.verify")


_SIGNED = (("egress", HC.DIGEST), ("ingress", HC.DIGEST))
LEAVES = 4
#: the leaf of the head of the record at leaf i (0 egress, 1 ingress) is HEADS + i
HEADS = HC.HEADS
_CERTIFICATES = PrimitiveDefinition(
    "NetworkCertificatesCheck", 2, _SIGNED, Bits(1), _host_only,
    doc="the link-window's two network certificates over their signed digests, the ones each end's head check reads: "
        "each certificate verifies in the certifier session and names the link-window, and its head is settled by the "
        "deadline; no gates, checked on the host",
    register=False)
_ENDS = ("egress", "ingress")


def _heads(S_) -> tuple[tuple[str, Any], ...]:
    """Each end's head check's parameters but its signed digest, which is the unit's `egress` or `ingress`."""
    return tuple((f"{end}_{name}", t) for end, key in zip(_ENDS, (S_.EGRESS, S_.INGRESS))
                 for name, t in HC.gate(*key).params if name != "signed")


def _checks(B, S_, args) -> tuple[list[Coll], list[Coll]]:
    """The certificates' check and each end's head check, each over its end's signed digest (the unit's `egress` or
    `ingress`), and each end's head counts as the head check reads them."""
    oks, counts, rest = [B.call(_CERTIFICATES, *args[:2])], [], list(args[2:])
    for d, key in enumerate((S_.EGRESS, S_.INGRESS)):
        g = HC.gate(*key)
        names = [name for name, _ in g.params]
        ends, rest = rest[:len(names) - 1], rest[len(names) - 1:]
        ends.insert(names.index("signed"), args[d])
        oks.append(B.call(g, *ends))
        sh = HC.Shape(*key[:2])
        head = HC._refs(ends[names.index("head")])
        counts.append(HC._coll(Array(sh.T, Array(4, HC.BYTE)), head[8 * sh.CNT:8 * sh.ROOT]))
    return oks, counts


def _all(B, oks: list[Coll]) -> Coll:
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BL.BIT), bits))


def _body(B, S_, *args):
    return _all(B, _checks(B, S_, args)[0])


def _body_served(B, S_, *args):
    oks, (egress, ingress) = _checks(B, S_, args[:-1])
    return _all(B, oks + [B.call(SV.gate(*S_.SERVED), ingress, egress, args[-1])])


_UNIT = CompositeDefinition(
    "NetworkLinkWindow", 4, ("EGRESS", "INGRESS"), lambda S_: (_SIGNED + _heads(S_), Bits(1)), _body,
    doc="the proof unit of network accounting: one link-window, both directions, over the two certificates' signed "
        "digests; EGRESS and INGRESS are each end's head check's (L, T, COUNTS)", register=False)
_SYNC = ("sync", Array(3, Array(SV.SYNC_BITS, BL.BIT)))
_SERVED_UNIT = CompositeDefinition(
    "NetworkServedLinkWindow", 3, ("EGRESS", "INGRESS", "SERVED"),
    lambda S_: (_SIGNED + _heads(S_) + (_SYNC,), Bits(1)), _body_served,
    doc="the proof unit of network accounting on a served link: `NetworkLinkWindow`'s checks, and the egress head's "
        "counts derived from the ingress head's under the declared sync (`served.NetworkServedCounts`, SERVED its "
        "(T, FOUT, FIN, R, RIN))", register=False)
#: the window's view of the advice (`advised.view`), the advised counts' public inputs
ADVICE = ("handed", "floor", "queued", "received")


def _advice(S_) -> tuple[tuple[str, Any], ...]:
    counts = Array(S_.ADVISED[0], AD.COUNT)
    return (("handed", counts), ("floor", counts), ("queued", AD.COUNT), ("received", AD.COUNT))


def _body_advised(B, S_, *args):
    n = len(ADVICE)
    oks, (egress, ingress) = _checks(B, S_, args[:-n])
    return _all(B, oks + [B.call(AD.gate(*S_.ADVISED), ingress, egress, *args[-n:])])


_ADVISED_UNIT = CompositeDefinition(
    "NetworkAdvisedLinkWindow", 1, ("EGRESS", "INGRESS", "ADVISED"),
    lambda S_: (_SIGNED + _heads(S_) + _advice(S_), Bits(1)), _body_advised,
    doc="the proof unit of network accounting on an advised link: `NetworkLinkWindow`'s checks, and the egress head's "
        "counts held to the release of the advice's frames, and the advice to the ingress head's counts "
        "(`advised.NetworkAdvisedCounts`, ADVISED its (T, R, RIN, OWN))", register=False)


def unit(egress: tuple[int, int, int], ingress: tuple[int, int, int], served: tuple[int, ...] | None = None,
         advised: tuple[int, int, int, int] | None = None):
    """N2's proof unit for heads of the size classes `egress` and `ingress`, each (L, T, COUNTS) (`head_check.unit`),
    and on a served link the served counts' size class `served`, (T, FOUT, FIN, R, RIN, OWN) (`served.unit`), or on an
    advised link the advised counts' `advised`, (T, R, RIN, OWN) (`advised.unit`)."""
    if served is not None and advised is not None:
        raise ValueError("a link is served or advised, not both")
    if advised is not None:
        return bind(_ADVISED_UNIT, EGRESS=tuple(egress), INGRESS=tuple(ingress), ADVISED=tuple(advised))
    if served is None:
        return bind(_UNIT, EGRESS=tuple(egress), INGRESS=tuple(ingress))
    return bind(_SERVED_UNIT, EGRESS=tuple(egress), INGRESS=tuple(ingress), SERVED=tuple(served))

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
#: a served link's: no egress occupancy profile, its sizes instead, and the window's declared sync, which its unit reads
PUBLIC_SERVED = tuple(p for p in PUBLIC if p.item != "occupancy") + (
    S.Public("occupancy", "structure", "the ingress end's frames per bucket in each window where the placement fixes "
             "them, fixed before the run; none on egress, whose counts the unit derives"),
    S.Public("served", "structure", "frames a response (FIXLEN), frames a request, windows and the certifier's own "
             "frames a window (its salt's), fixed before the run"),
    S.Public("sync", "output", "the window's declared clock sync, one of Σ_sync (fixed before the run): the advice "
             "`charge` already counts, which the served counts read"),
)
#: an advised link's: no egress occupancy profile, its sizes, each window's declared sync, and the advice, whose window
#: view (`advised.view`) its unit reads
PUBLIC_ADVISED = tuple(p for p in PUBLIC_SERVED if p.item not in ("served", "sync")) + (
    S.Public("advised", "structure",
             "windows and the certifier's own frames a window (its salt's), fixed before the run"),
    S.Public("sync", "output", "each window's declared clock sync, one of Σ_sync (fixed before the run): the advice "
             "`charge` already counts, whose δ and ρ the unit's ingress floor reads"),
    S.Public("advice", "output", "the advice (one per certifier session, after serving), charged by `advice.bits`: the "
             "unit reads its window view, `advised.view` of its requests' prompt and output lengths and ingress "
             "frames, its steps' rows, its release buckets and its windows' received request frames; each window "
             "reports the SHA-512 of those fields (`advised.digest`), which the network verdict holds to one advice"),
)
OUTPUTS = ("verdict", "deadline")
#: an advised link's: and the advice its window read (`advised.digest`)
OUTPUTS_ADVISED = OUTPUTS + ("advice",)


def encode(record: CertifiedRecord) -> bytes:
    """The leaf's bytes: the record head (`RecordHead.encode`), the certifier's salt (64 bytes), then the certificate,
    every field fixed-width or length-prefixed: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖
    u64be window ‖ signed digest (64) ‖ u32be len(signature) ‖ signature."""
    c = record.certificate
    link = c.link.encode("utf-8")
    return b"".join([record.head.encode(), record.salt, struct.pack(">H", len(c.key_id)), c.key_id,
                     struct.pack(">H", len(link)), link, struct.pack(">BQ", c.direction.code, c.window), c.digest,
                     struct.pack(">I", len(c.signature)), c.signature])


def rows(records: Sequence[CertifiedRecord]) -> list[bytes]:
    """The four leaves' bytes, in leaf order: each record's `encode`, then each record's head alone."""
    return [encode(r) for r in records] + [r.head.encode() for r in records]


def _sha512(obj: Any) -> str:
    return hashlib.sha512(canonical_json(obj)).hexdigest()


_NO_PROFILE = ("N2 compares no egress grid, so a link whose egress end has no occupancy profile has no verdict: "
               "nothing would hold its egress counts (declare LinkWindows.occupancy, LinkWindows.served on a served "
               "link, or LinkWindows.advised on an advised one)")


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
    occupancy profile: for each end (egress, ingress), each window's counts, or None for an end without one. A served
    link (`served`, FIXLEN) or an advised one (`advised`, arbitrary traffic under the advice) has no egress profile."""

    link: str
    params: GridParams
    sync_set: SyncSet
    key: PublicKey
    run: str
    certifier_session: bytes
    ingress: GridParams | None = None
    proof_deadline: float | None = None
    occupancy: tuple[Sequence[Sequence[int]] | None, Sequence[Sequence[int]] | None] = (None, None)
    served: SV.ServedProfile | None = None
    advised: AD.AdvisedProfile | None = None

    def __post_init__(self) -> None:
        if len(self.occupancy) != 2:
            raise ValueError("the occupancy profile has one entry per end: (egress, ingress)")
        if self.served is not None and self.advised is not None:
            raise ValueError("a link is served (FIXLEN) or advised, not both")
        if self.advised is not None:
            g, i = self.params, self.ingress_params
            if self.occupancy[0] is not None:
                raise ValueError("an advised link's egress counts are the advice's: it has no egress occupancy profile")
            if (g.buckets, g.bucket_ms) != (i.buckets, i.bucket_ms):
                raise ValueError("an advised link's ends share T and τ_b: its anchors are ingress buckets")
        if self.served is not None:
            g, i = self.params, self.ingress_params
            if self.occupancy[0] is not None:
                raise ValueError("a served link's egress counts are derived: it has no egress occupancy profile")
            if (g.buckets, g.bucket_ms) != (i.buckets, i.bucket_ms):
                raise ValueError("a served link's ends share T and τ_b: its anchors are ingress buckets")
            for sync in self.sync_set:
                SV.sync_words(sync)
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

    def served_key(self) -> tuple[int, int, int, int, int, int] | None:
        """The served counts' size class (T, FOUT, FIN, R, RIN, OWN) on a served link, else None."""
        if self.served is None:
            return None
        return (self.params.buckets, self.served.response_frames, self.served.request_frames, self.params.allocation,
                self.ingress_params.allocation, self.served.own)

    def advised_key(self) -> tuple[int, int, int, int] | None:
        """The advised counts' size class (T, R, RIN, OWN) on an advised link, else None."""
        if self.advised is None:
            return None
        return self.params.buckets, self.params.allocation, self.ingress_params.allocation, self.advised.own

    @property
    def windows(self) -> int:
        """The link's windows, which `network_verdict` takes by default: the served or advised profile's, else the
        egress occupancy profile's."""
        if self.occupancy[0] is None and self.served is None and self.advised is None:
            raise ValueError(_NO_PROFILE)
        derived = self.served or self.advised
        return derived.windows if derived is not None else len(self.occupancy[0])

    @cached_property
    def unit(self):
        """N2's proof unit for this link (`unit`): its heads' size classes are the link's, and on a served link its
        served counts' (`served_key`), on an advised link its advised counts' (`advised_key`)."""
        return unit(self.head_key(0), self.head_key(1), self.served_key(), self.advised_key())

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
            "occupancy": [None if p is None else _sha512([list(row) for row in p]) for p in self.occupancy],
            **({} if self.served is None else {"served": [self.served.response_frames, self.served.request_frames,
                                                          self.served.windows, self.served.own]}),
            **({} if self.advised is None else {"advised": [self.advised.windows, self.advised.own]})})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query,
                        public_outputs=OUTPUTS if self.advised is None else OUTPUTS_ADVISED,
                        proof_deadline=self.proof_deadline)
        public = PUBLIC_SERVED if self.served is not None else PUBLIC_ADVISED if self.advised is not None else PUBLIC
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, public, context)

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
        the end's grid, its profile's row and its certificate's signed digest, and as witness the head, its salt, its
        two siblings and the certifier's salt for the record. None where the unit takes no instance: a window past the
        end's profile, which has no row to give, or a head whose length isn't the unit's."""
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
                                   None if profile is None else tuple(profile[w]), record.certificate.digest, head,
                                   opening.salts[i], tuple(level[(i >> k) ^ 1] for k, level in enumerate(levels[:-1])),
                                   record.salt, T))
        return out

    def _audit(self, audit: Any, opening: Opening, syncs: Sequence[ClockSync],
               frames: Sequence[LogicalFrame] | None = None) -> tuple[list[Any], list[Any]]:
        """The audit of both records; the egress grid compared with the Program's only when `frames` are given."""
        w, (egress, ingress) = opening.window, opening.records
        out = audit.check_link(self.link, [egress], frames or (), list(syncs)[:w + 1], self.sync_set, self.params,
                               key=self.key, session=self.certifier_session, grid=frames is not None)
        inn = audit.check_ingress_link(self.link, [ingress], w + 1, key=self.key, params=self.ingress_params,
                                       session=self.certifier_session)
        return out, inn

    def _registered(self, opening: Opening, draw: S.Draw) -> str:
        reg = draw.registration.record
        if opening.window != reg["window"]["window"] or len(opening.records) != 2:
            raise ValueError("the opening is not the registered link-window's")
        return reg["roots"][0]["root"]

    def gates(self) -> list[PrimitiveDefinition]:
        """The unit's gates: each end's head check (`head_check.gate`), and on a served link the served counts'
        (`served.gate`), on an advised link the advised counts' (`advised.gate`)."""
        out = [HC.gate(*self.head_key(d)) for d in (0, 1)]
        if self.served is not None:
            out.append(SV.gate(*self.served_key()))
        if self.advised is not None:
            out.append(AD.gate(*self.advised_key()))
        return out

    def proved(self, findings: Sequence[Finding]) -> bool:
        """Whether `findings`, the verifier of record's run over a delivered proof of the unit (`proofs.verify`), prove
        this link's unit: every gate of it names its Definition's pinned circuit digest (`head_check.pinned`), and one or
        more sessions each accepted and named the link's registered Program (the partition object's `program`), which
        binds those digests. A proof of another circuit with the same interface names another Program, and a unit with
        a size class no sidecar pins has no proof."""
        want = self._partition["program"]
        return all(HC.pinned(g) for g in self.gates()) and bool(findings) \
            and all(f.accepted and f.program == want for f in findings)

    def verify(self, opening: Opening, draw: S.Draw, syncs: Sequence[ClockSync],
               proved: Sequence[Finding] | None = None, advice: Any = None) -> dict[str, Any]:
        """The verdict of record on the delivered opening: accepted when it reproduces the registered root under the
        link-window's domain, both head checks accept, and the audit accepts both records, comparing no grid (the module
        docstring). `syncs` are the declared clock syncs of windows 0 … w (on an advised link, of every window the
        advice hands a frame in). The verdict reports the public outputs and never the violation's kind. On a served
        link it also runs the served counts' check (`served_check`), and on an advised link the advised counts' on
        `advice`, the link's advice (`advised_check`), and reports the advice it read as the `advice` output. Given
        `proved`, the findings of a proof of the unit, it accepts only when they prove this link's unit (`proved`).
        Raises `ValueError` on a link with no egress occupancy, served or advised profile."""
        if self.occupancy[0] is None and self.served is None and self.advised is None:
            raise ValueError(_NO_PROFILE)
        root = self._registered(opening, draw)
        w, (egress, ingress) = opening.window, opening.records
        sealed = self.root(opening) == root
        headed = all(i is not None and HC.reference(i) for i in self.head_checks(opening, bytes.fromhex(root)))
        headed = headed and (self.served is None or self.served_check(opening, syncs))
        headed = headed and (self.advised is None or self.advised_check(opening, syncs, advice))
        headed = headed and (proved is None or self.proved(proved))
        audit = Audit()
        out, inn = self._audit(audit, opening, syncs)
        accepted = sealed and headed and _accepts(out, inn, w)
        public = {"verdict": int(accepted), "deadline": int(egress.head.deadline and ingress.head.deadline)}
        if self.advised is not None:
            public["advice"] = None if advice is None else AD.digest(advice)
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

    def served_check(self, opening: Opening, syncs: Sequence[ClockSync]) -> bool:
        """The served counts' check on the link-window (`served.reference`, which the tests hold to the circuit): the
        egress head's counts are those its ingress head's derive under the window's declared sync. False for a window
        with no declared sync, or one the unit's input can't hold."""
        w, (egress, ingress) = opening.window, opening.records
        if w >= len(syncs):
            return False
        try:
            sync = SV.sync_words(syncs[w])
        except ValueError:
            return False
        return SV.reference(*self.served_key(), ingress.head.counts, egress.head.counts, sync)

    def advised_check(self, opening: Opening, syncs: Sequence[ClockSync], advice: Any) -> bool:
        """The advised counts' check on the link-window (`advised.reference`, which the tests hold to the circuit): the
        egress head's counts are the release of the advice's frames in the window's view of it (`advised.view`, under
        the declared syncs), and the ingress head's counts release the request frames that view needs by each bucket
        and the window's `received`. False with no advice, or an advice `view` refuses: malformed, not covering the
        window, or handing a frame in a window with no declared sync or before any request frame."""
        if advice is None:
            return False
        w, (egress, ingress) = opening.window, opening.records
        try:
            v = AD.view(advice, w, self.params.buckets, self.params.allocation, syncs)
        except ValueError:
            return False
        return AD.reference(*self.advised_key(), ingress.head.counts, egress.head.counts, v.handed, v.floor, v.queued,
                            v.received)

    def stand_in(self, opening: Opening, draw: S.Draw, frames: Sequence[LogicalFrame],
                 syncs: Sequence[ClockSync]) -> bool:
        """The in-the-clear accept bit, the tests' oracle: the opening reproduces the registered root, the host audit
        (`audit.Audit`) accepts both records with the egress grid compared with the Program's, from its logical
        `frames`, and each head's counts are the profile's (`declares`)."""
        root = self._registered(opening, draw)
        out, inn = self._audit(host_audit.Audit(), opening, syncs, frames)
        counted = all(self.declares(d, opening.window, r.head.counts) for d, r in enumerate(opening.records))
        return self.root(opening) == root and counted and _accepts(out, inn, opening.window)

    def statement(self, opening: Opening) -> dict[str, Any]:
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link, "window": opening.window,
                "root": self.root(opening)}

    def outcome(self, stream: S.Stream, draw: S.Draw, opening: Opening, syncs: Sequence[ClockSync],
                proved: Sequence[Finding] | None = None, advice: Any = None) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict (`verify`, with `proved` when a proof of the unit was
        delivered, and an advised link's `advice`), with the root the opening reproduces as the session's, which
        one-stage's audit holds to the registered root."""
        return stream.outcome(draw, {VERIFIER: self.verify(opening, draw, syncs, proved, advice)},
                              self.statement(opening), session_roots={VALUE: self.root(opening)})

    def network_verdict(self, stream: S.Stream, windows: int | None = None) -> dict[str, Any]:
        """The link's network verdict of record: the AND over its link-windows 0 … `windows` − 1 (by default the egress
        profile's windows, fixed before the run), with no ε and no share credit. It accepts only when, for every w,
        the stream's w-th registration is window w and its outcome of record accepted. A window registered out of
        order or not at all (a gap, or a run that stops short), a registration past the last window, a registration
        with no outcome yet, and a rejected or late outcome each reject the link; `refused` lists their windows.
        `records` is each window's outcome record SHA-512 (`S.Outcome.record`, None without one), and `digest` binds
        the verdict to the stream and those records. On a served or advised link the windows are by default its
        profile's (`windows`), and on an advised link every accepted window must name the first accepted window's advice
        (its `advice` output, `advised.digest`), which the verdict reports as `advice`, or it is refused. Raises
        `ValueError` on a link with no egress occupancy, served or advised profile, as `verify` does, and on a stream
        that isn't this link's."""
        default = self.windows
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this link's: its spec isn't LinkWindows.spec")
        n = default if windows is None else windows
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a link has one or more link-windows, not {n!r}")
        outcomes = {r["registration"]: r for r in stream.records}
        records: list[str | None] = []
        refused: list[int] = []
        advice = None
        for w in range(max(n, len(stream.log))):
            reg = stream.log[w] if w < len(stream.log) else None
            body = outcomes.get(R.digest(reg)) if reg is not None and reg.get("window") == self.provenance(w) else None
            records.append(None if body is None else _sha512(body))
            ok = w < n and body is not None and body["symbol"] == S.ACCEPT
            if ok and self.advised is not None:
                named = body["public"].get("advice")
                advice = named if advice is None else advice
                ok = named is not None and named == advice
            if not ok:
                refused.append(w)
        accepted = not refused
        digest = _sha512({"format": NETWORK_VERDICT, "stream": stream.digest, "windows": n, "records": records,
                          "accepted": accepted})
        out = {"accepted": accepted, "windows": n, "refused": refused, "records": records, "digest": digest}
        return out if self.advised is None else out | {"advice": advice}


__all__ = ["ADVICE", "HEADS", "LEAVES", "LinkWindows", "OUTPUTS", "OUTPUTS_ADVISED", "Opening", "PUBLIC",
           "PUBLIC_ADVISED", "PUBLIC_SERVED", "SCHEMA", "VALUE", "encode", "rows", "unit"]
