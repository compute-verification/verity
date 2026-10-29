"""One link-window's network certificates through the proof system (N2): the network `Spec`, the firewall's salted
commitment to the link-window's certified records, and the check's verifier of record.

**What is committed.** One `Value`, `certificates`: four leaves under one salted root, the link-window's two certified
records (`CertifiedRecord`), egress leaf 0 and ingress leaf 1, and their heads, egress leaf 2 and ingress leaf 3
(`rows`).
Each leaf is `hm96-sha512/row/v2` over its bytes, salted with its own fresh 192-byte salt, in a `frame-v3-sha512` tree
under one-stage's served domain, whose run is the link and the window. A record leaf's bytes are `encode`'s (the record
head, the certifier's salt for it, then the certificate), and a head leaf's are the head's alone (`RecordHead.encode`).
A proof that reads only the head, such as the head check's, opens a head leaf in the head's two or three SHA-512
compressions instead of the certificate's 272. The certificates, salts
and heads are witness: the challenger sees the root, never a certificate, a salt or a head.

**Selection** is `All`: every link-window is checked, so an accepted outcome is the exhaustive fact. A registration is
one link-window, one proof block. Its `window` provenance is `{"kind": "served", "window": w}`, and the challenger holds w
to the registration's order in the stream (`LinkWindows.register`), so the receipt's order is the link-window. The
network verdict of record is `LinkWindows.network_verdict`, the AND over the link's link-windows with no ε or share
credit: it accepts only when every window of the egress profile (a served or advised link's profile) is registered in
order and its outcome of record accepted, so a rejected or late window, one with no outcome, and a gap in the
registrations each reject the link. A
run's is the AND over its links'.

**The period draw** (`LinkWindows.period_draw`) samples the link-windows instead. The prover first commits the
period: every window's salted root (`root`) and, on an advised link, the advice's digest (`advised.digest`), which the
challenger records before it reads a coin (a coin source with a `commit` is first committed to the period's digest, as
a remote challenger's is). It then draws `subset:K` of the n windows, and only the drawn windows are registered, in
window order, each held to its committed root (`register(drawn=…)`), selected and proved. `network_verdict(drawn=…)`
accepts when exactly the drawn windows are registered, in order, each under its committed root, with an accepted
outcome of record (on an advised link naming the committed advice), and it states what the draw established: except
with probability δ, at most `wrong_units["bound"]` of the n committed windows are wrong (the uniform K-subset's exact
bound, one-stage's `wrong_units` over windows), so ε = bound / n at δ. The challenger's draw is Python's
(`service.select_draw`), which records the coin bytes it read (`read`); the draw of record is the Lean verifier of
record's on those bytes (`lean_draw`: C-Flock's `flock-verify draw --population n --k K --stream-file`, the Lean
`draw` the one-stage drivers run, `Flock.Draw.drawOn`), which the network verdict runs itself, as a registration's own
draw is the Lean's `Select.select` on the bytes it read. The verdict accepts only when the windows registered and
proved are exactly that draw's, and otherwise refuses by name (`PERIOD_DRAWN`), whatever units the draw object it is
handed lists. Selection `All` (K None) draws every window and reads no coin, and K = n leaves nothing to draw.
The datacenter model (`benchmarks/datacenter`, a Python model) builds no
`LinkWindows` outcomes, so its account doesn't read it, though `Account.sound` is the same AND over its own port
records.

**Public items**, each with its reason (`PUBLIC`): the population (one), the window (its index, the order), the
committed value's leaf count and domain, its salted root, the parameters digests, the charge and the occupancy profile
(fixed before the run, bound by the spec's context), and the check's two public outputs, `verdict` and `deadline`. A
rejected link-window shows only the service's `reject`: the violation's kind stays in the verifier.

**The occupancy profile** (`LinkWindows.occupancy`) is the placement's public occupancy for each end of the link: each
window's T counts, the frames that end's record holds in each bucket, as the Program and the placement fix them. An end
whose rows the Program reads rather than predicts, such as requests from outside, has none. The profile is fixed before
the run and bound by the spec's context, and nothing declared at run time sets it. So a port-window it leaves empty
needs no content check: a head with frames where the profile says 0 fails N2's count check. An end without a profile
has no such backing, since its occupancy is its record's own.

**The check** is `LinkWindows.unit`, `NetworkLinkWindow` v5{EGRESS, INGRESS}: each end's head check,
`head_check.NetworkHeadCheck_v2{L, T, COUNTS}` (EGRESS and INGRESS are those statics), a Boolean Definition over its
end's signed digest (`certificate_digest`: the end's record digest hidden under the certifier's salt), its direction a
constant of the unit (egress 0, ingress 1). The head opens at leaf 2 + direction under the registered root, a hiding
read; its link, direction and window are the link-window's; its status is complete; its T, r,
B, τ_b and parameters digest are the end's declared grid's; on an end with an occupancy profile its T counts are the
profile's row for the window; and its record digest under the certifier's salt is the signed digest, so the certificate
signs the head the check opened. A window past the profile has no row to give, so it is rejected. The unit's parameters
are the two signed digests (``egress``, ``ingress``) and each end's head check's others, ``<end>_<name>``; its output is
two bits, `OUTPUTS`: ``verdict``, the AND of the head checks (and on a served link the served counts' check), and
``deadline``, the AND of the two heads' deadline bits. The certificates are not in the unit: the verifier checks them on
their public fields (`LinkWindows.certified`).

In the unit's Program each head check is one gate that names the Definition and its circuit digest (`head_check.gate`,
`Gate[NetworkHeadCheck_v2{L,T,COUNTS}@<digest>]_v2`), because the commitment gadgets' Boolean bodies encode to about
140 MB and the proof system encodes the Program on every registration. The digest is core's `program_digest` of the
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

N3, the content check, a prototype in `benchmarks/network_sim` evaluated in the clear and in no proof-system Spec, is
retired, superseded by the step-indexed scan (`ingress_scan`, `egress`), which `benchmarks/network_traces/scan_zk.py`
proves in zero knowledge, outside this protocol.

**The proof.** The prover proves the drawn unit as one C-Flock statement of one instance: rows ``pub`` (the unit's
public parameters) and ``wit`` (each end's head, its leaf's salt, its two siblings and the certifier's salt), each the
little-endian concatenation of its parameters' bits in whole u16 words (`LinkWindows.layout`, `LinkWindows.row`), and
the output row ``o0``, one u16 word holding the two output bits (`OUT_BITS`); each row committed under a hiding salt. Its
circuit is `LinkWindows.proof_unit`, the registered unit with each gate the Definition its pinned digest names, bound to
the registered partition and Program (`benchmarks/network_traces/n2_zk.py` proves it behind the firewall).

Its verifier of record is `LinkWindows.verify`, which decides on the delivered proof (`Delivered`) and public inputs
alone. It accepts when
- the verifier's own run of C-Flock's Lean verifier (`proofs.verify`, `verify --zk`) accepts every session on the circuit
  file the auditor committed, the sessions prove this link's registered unit (`LinkWindows.proved`: every gate names a
  pinned circuit digest and every session names the registered Program) and the drawn unit;
- the proof's ``pub`` row opens, under the salt delivered, to the verifier's own copy (`public_arguments`: the registered
  root, the link-window's public fields and both certificates' signed digests), and its ``o0`` row to the output word
  delivered, whose verdict and deadline bits are both 1;
- the certificates' check accepts (`certified`: each verifies under the network certifier's public key in its certifier
  session over its public fields and its signed digest, and names the registered link-window), and the window's
  declared sync is one of Σ_sync.

What the verifier receives: the registration (the salted root), the draw, the two certificates' public fields and
signatures, from the certifier, which is the auditor's device; the declared syncs; and from the prover the statement's
public file (each row's hiding commitment), its session files, the circuit file (held to the committed digest), the
salts of the ``pub`` and ``o0`` rows and the output word. Never the opening nor the ``wit`` row's salt. The certificates
are not a public item of the Spec: the service's public kinds (structure, outputs, coins, hiding commitments) have none
for a trusted device's signature over a hiding commitment, so they reach the verifier from the certifier beside the
service, in the clear. That is a provisional exception to the four public kinds, not a fifth kind.

`LinkWindows.clear` is the in-the-clear check on the opening, for a debug comparison only (`verify(opening=…)`, off by
default, which adds its accept bit and decides nothing) and the tests' reference: the opening reproduces the registered
root, each end's head check accepts (`head_check.reference`), and the audit (`audit.Audit`) accepts both records under
the certifier's key and session, comparing no grid (`check_link(grid=False)`). `LinkWindows.unproved_outcome` decides a
run that proves nothing (a rehearsal, the simulator) on it, so its verdict says ``"decided_by": "clear"`` and is never
zero-knowledge.

N2 compares no grid in the clear and takes no frames, so its claim is counts only: on every link-window the egress
head's counts are the profile's (and the ingress head's, where that end declares one), which puts the link's timing at
the profile. The egress grid is the head's counts filled with the head's frames in order (`grid.fill`), and the head's
digest binds both, so the frames are the rest of the grid, and N2 says nothing of them. Nothing here holds the egress
grid to the Program's.

**A served link** (`LinkWindows(served=served.ServedProfile(…))`) has egress counts that follow its requests' arrivals
and the run-time syncs, so no profile fixes them before the run. Its unit is `NetworkServedLinkWindow` v4{EGRESS,
INGRESS, SERVED}: the checks above, and one gate naming `served.NetworkServedCounts_v1` and its pinned circuit digest
(`served.gate`, pinned in `served.circuit-check.json`), which holds the egress head's
counts to the ones the ingress head's counts derive under the window's declared sync (`served`'s docstring has the
derivation, its premises and the carry-over). Its public items are `PUBLIC_SERVED`: no egress profile, the served
sizes, and the window's declared sync, the advice `charge` already counts, which the served counts read and the unit
takes as its ``sync`` parameter. The in-the-clear check runs `served_check` too.

**An advised link** (`LinkWindows(advised=advised.AdvisedProfile(…))`) is a served link under arbitrary traffic:
responses of any length, batched and timed as serving chose, carried across windows. The prover's advice
(`advised.LinkAdvice`: I1's `advice.Advice`, one per certifier session, after serving, charged by `advice.bits`, and
each window's request frames) states what serving chose, the profile's public framing says where each request ends on
the ingress link (`advised.ends`, which refuses a link without the tie of its requests to the certified ingress,
`advised.IoTieVerdict`, which the verifier takes only from its own tie source, `ties`: `tie_service.Ties`, from the
IoTieRequest proofs it verified, refusing every window of a size class with no pinned IoTieRequest digest), and its
unit is `NetworkAdvisedLinkWindow` v2{EGRESS, INGRESS, ADVISED}: the checks above, and one gate naming
`advised.NetworkAdvisedCounts_v1` and its pinned circuit digest (`advised.gate`, pinned in
`advised.circuit-check.json`), in its verdict bit, which holds the egress head's counts to the advice's releases and
the ingress head's counts to the advice (`advised`'s docstring has the fields it reads, the soundness argument and the
premises). Its public inputs, in the ``pub`` row, are the window's view of the advice (`advised.view`: ``released``,
``floor`` and ``received``, `ADVICE`), which the verifier computes from the public advice, the framing and the
syncs the certifier ran; the heads' counts stay witness. Its public items are `PUBLIC_ADVISED`. Its
verifier of record takes the delivered advice (`verify(advice=…)`) and the tie only from its own tie source
(`LinkWindows.ties`, `advised.TieSource`: `advised.IoTieVerdict`, IoTieRequest's verdict from proofs it verified,
naming the link and its ingress registrations), never beside the advice. That source is `tie_service.Ties`, on the
ingress tie's stream of record (IoTieRequest in the proof system), and without one (`ties` None) every advised link
is refused. Every floor is under the sync the certifier ran, which the verifier takes only from its own clock source
(`LinkWindows.clock_source`: the certifier's salted clock certificates, each opened under the link's key and session,
`advised.SignedClocks`; the openings are witness, the verifier's own read, never a public answer), since a δ below the
traffic's loosens the floor, and a window whose declared sync is another, or outside Σ_sync, is refused alone;
without a clock source, every advised link is refused too. It refuses a window without an advice or one whose view
`advised.view` refuses, and reports the advice it read as a third public output, `advice` (`advised.digest`); the
network verdict refuses a window that names another advice than the first accepted window's, so every window's
counts are one advice's. The in-the-clear check runs `advised_check` too.

**A warden link** (`LinkWindows(warden=WardenProfile(…))`) is the active warden's (`warden.Warden`) own two records
of each link-window: its responses on egress and its requests on ingress, on its one grid, certified under its
schedule's parameters digest (`warden.WardenParams.digest`, which its deployment publishes) rather than a certifier's
(`certificate.parameters_digest`). The warden is the auditor's device, and its code puts every bucket's cells, r for
each live connection and the rest padding (`warden.audit` reads them in the clear, the scan's units in zero
knowledge), so its counts are its schedule's: no occupancy profile holds them, the period's schedule terms charge them
(log2 #S_s and Σ_c log2(cap_c + 1)), and the warden runs no shaper clock, so a window declares no sync and the
charge is 0. Its unit is the plain one, `NetworkLinkWindow` v5{EGRESS, INGRESS}, each head check holding its end's
head to the warden's grid and digest and to the warden's signed digest. Its public items are `PUBLIC_WARDEN`. The
in-the-clear check is the opening's root and the head checks: the records' cells are the scan's, and the certifier's
host audit doesn't read a warden's records.

A link whose egress end has no occupancy profile and is neither served, advised nor a warden's has no verdict
(`ValueError`, failing closed): nothing would hold its egress timing.

`LinkWindows.stand_in` is the in-the-clear verdict, the Program's egress grid compared as it was before the head check,
and each head's counts the profile's (`declares`). It reads the frames, so it is the tests' oracle, never a verdict:
`clear` accepts every link-window it accepts, and differs from it only where the record's grid, at the profile's
counts, isn't the Program's: in its frames, which N2 doesn't hold, or because the declared syncs
move the Program's counts off the profile, which N2 holds the counts to instead.

When the in-the-clear check's audit is the Lean check's (`lean_audit.Audit`), `audit.Audit` runs beside it as the
reference: a link-window where the two differ rejects, and the verdict says `"by": "lean"` and carries the reference's
accept bit (`reference`), never a violation's kind.

**The verdict of record is the Lean's** on plain, served and warden links: `NetworkCheck.LinkWindow.verdict`
(`verity/core/protocols/network/lean`, `network-records verdicts`' ``link-window``, `lean_verdict`), switched by the one
line `OF_RECORD`. `verify` runs it and this module's `python_verdict` on every proof, records
whose verdict it is (`by`) and the other's accept bit (`reference`), and rejects a link-window where they differ or the
Lean refuses the case. On a warden link the Lean reads the warden's parameters digest (`lean_case`'s ``warden``) for
both ends' grids and no sync, as `grid` and `python_verdict` do. The gap list, what `python_verdict` reads that the
Lean takes as an input rather than checks:
- an advised link's window view (`advised.view`, under the tie source's IoTieRequest verdict and the clock source's
  syncs), which is why an advised link's verdict of record stays this module's;
- on a warden link, the warden's parameters digest (`WardenProfile.digest`, which the warden's deployment publishes
  as its schedule's, `warden.WardenParams.digest`), which neither derives;
- what both take from the proof system and the lowering: the findings of the verifier's own run of C-Flock's Lean
  verifier (`proofs.verify`), the delivered sessions' SHA-512s, the registration and the draw, and the registered
  Program's and partition's digests and the gate ids (`P.partition`, `head_check.gate`), which the Lean holds the
  findings and pins to but doesn't derive.
The Lean reads an `LmsKey` only, and refuses (so `verify` rejects) any other key. `tie_service.IngressTie.verify`, the
ingress tie's verifier, is not ported and stays Python. Every proving run of N2 recorded before the switch replayed
through both to the verdict it recorded, and the properties of the Lean's verdict are
`Security/Properties/NetworkCertifier/LinkWindow.lean`.

What that establishes, and what it doesn't: the outcome is accept exactly when the Lean verifier accepts a proof of the
registered unit whose public row is the verifier's own and whose output word is 0b11, the certificates verify and the
sync is declared; the outcome record binds the registered root, the draw and the delivered sessions. The verifier of
record reads no witness but an advised link's clock openings (`certified_clocks`), on the host until N2's advised
unit opens them (PROTOCOL.md's named gap, "The clock's opening"). The proof opens the head leaves (2, 3) only, so the
record leaves (0, 1), which hold each certified record whole, are committed but read by nothing in the verdict, and
the certificates' signatures are checked
on the host from their public fields, not in the proof. One clock per spatial unit across links (`Audit.syncs`) is
not checked here: each registration is one link-window, checked alone. The frames, their content and their order
within the counts, are not held: an accepted outcome says nothing of them, so the claim is counts only (timing at the
profile).
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.core.protocols.certifier import advised as AD, audit as host_audit, head_check as HC, lean_audit as LA, \
    served as SV
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.certificate import Certificate, CertifiedClock, CertifiedRecord, Direction, LmsKey, \
    PublicKey, parameters_digest
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import audit as OA, partition as P, proofs as PF, registration as R
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

#: Whose verdict on a link-window's proof is the record's, by kind of link (`LinkWindows.kind`): the Lean's
#: (``"lean"``, `NetworkCheck.LinkWindow`) or this module's (``"python"``). Both run on every proof either way. Setting a
#: kind to ``"python"`` is the one line that switches it back. An advised link stays Python's: the Lean takes its
#: window view (`advised.view`) as an input.
OF_RECORD = {"plain": "lean", "served": "lean", "advised": "python", "warden": "lean"}

SPEC = "verity-network-certifier/service-spec/v0"
SESSION = "verity-network-certifier/session/v0"
STATEMENT = "verity-network-certifier/statement/v0"
VERDICT = "verity-network-certifier/verdict/v0"
NETWORK_VERDICT = "verity-network-certifier/network-verdict/v0"
PERIOD_DRAW = "verity-network-certifier/period-draw/v0"
#: the sampled network verdict's refusal by name: the windows registered and proved are not the period's draw of record
#: (`lean_draw` on the coins the period draw read), or that draw can't be made
PERIOD_DRAWN = "period-draw"
VALUE = "certificates"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
VERIFIER = "network-audit"
_INST = hm96.SHA512


_SIGNED = (("egress", HC.DIGEST), ("ingress", HC.DIGEST))
LEAVES = 4
#: the leaf of the head of the record at leaf i (0 egress, 1 ingress) is HEADS + i
HEADS = HC.HEADS
_ENDS = ("egress", "ingress")
#: the unit's output, two bits in this order (`Check.public_outputs`)
OUTPUTS = ("verdict", "deadline")
#: the statement's rows: ``pub`` the unit's public parameters, ``wit`` its witness (`LinkWindows.layout`), then its one
#: output row ``o0``, the two output bits; each instance's commitments are in this order in a statement's public file
ROWS = ("pub", "wit")
PORTS = ROWS + ("o0",)
#: the output row's length: C-Flock commits an output port as whole u16 words, so ``o0`` is one word, its bit 0 the
#: verdict and bit 1 the deadline, its other bits 0
OUT_BITS = 16
#: a head check's witness parameters; the rest of the unit's parameters are public
_WITNESS = ("head", "salt", "path", "salt_c")
#: a head check's parameters the unit fixes rather than takes: each end's direction is its own (egress 0, ingress 1)
_FIXED = ("signed", "direction")


def _heads(S_) -> tuple[tuple[str, Any], ...]:
    """Each end's head check's parameters but its signed digest, which is the unit's `egress` or `ingress`, and its
    direction, which the unit fixes."""
    return tuple((f"{end}_{name}", t) for end, key in zip(_ENDS, (S_.EGRESS, S_.INGRESS))
                 for name, t in HC.gate(*key).params if name not in _FIXED)


@lru_cache(maxsize=None)
def _head_expanded(L: int, T: int, counts: bool):
    return HC.expand(HC.gate(L, T, counts), HC.unit(L, T, counts))


@lru_cache(maxsize=None)
def _served_expanded(*key: int):
    return HC.expand(SV.gate(*key), SV.unit(*key))


@lru_cache(maxsize=None)
def _advised_expanded(*key: int):
    return HC.expand(AD.gate(*key), AD.unit(*key))


def _checks(B, S_, args, check=HC.gate) -> tuple[list[Coll], list[Coll], list[Any]]:
    """Each end's head check (`check`'s for its size class) over its end's signed digest (the unit's `egress` or
    `ingress`) and its own direction, a constant; each end's head counts and its deadline byte's low bit, as the head
    check reads the head."""
    K = HC._Consts(B)
    oks, counts, deadline, rest = [], [], [], list(args[2:])
    for d, key in enumerate((S_.EGRESS, S_.INGRESS)):
        g = check(*key)
        names = [name for name, _ in g.signature()[0]]
        ends, rest = rest[:len(names) - len(_FIXED)], rest[len(names) - len(_FIXED):]
        ends.insert(names.index("direction"), HC._coll(Array(1, BL.BIT), K.bits(d, 1)))
        ends.insert(names.index("signed"), args[d])
        oks.append(B.call(g, *ends))
        sh = HC.Shape(*key[:2])
        head = HC._refs(ends[names.index("head")])
        counts.append(HC._coll(Array(sh.T, Array(4, HC.BYTE)), head[8 * sh.CNT:8 * sh.ROOT]))
        deadline.append(head[8 * sh.DL])
    return oks, counts, deadline


def _all(B, oks: list[Coll]) -> Coll:
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BL.BIT), bits))


def _outputs(B, oks: list[Coll], deadline: list[Any]) -> Coll:
    """The unit's two output bits: the AND of its checks, then the AND of both heads' deadline bits."""
    settled = B.call(BL.and_gate(2), HC._coll(Array(2, BL.BIT), deadline))
    return Coll(_OUT, concat([_all(B, oks).refs, settled.refs]))


_OUT = Array(len(OUTPUTS), BL.BIT)
_SYNC = ("sync", Array(3, Array(SV.SYNC_BITS, BL.BIT)))
#: the window's view of the advice (`advised.view`), the advised counts' public inputs
ADVICE = ("released", "floor", "received")


def _advice(S_) -> tuple[tuple[str, Any], ...]:
    counts = Array(S_.ADVISED[0], AD.COUNT)
    return (("released", counts), ("floor", counts), ("received", AD.COUNT))


def _units(check, served, advised=AD.gate) -> tuple[CompositeDefinition, CompositeDefinition, CompositeDefinition]:
    """N2's unit and its served and advised links', each head check called as `check`, and the served and advised
    counts as `served` and `advised` give them, for its size class."""

    def body(B, S_, *args):
        oks, _, deadline = _checks(B, S_, args, check)
        return _outputs(B, oks, deadline)

    def body_served(B, S_, *args):
        oks, (egress, ingress), deadline = _checks(B, S_, args[:-1], check)
        return _outputs(B, oks + [B.call(served(*S_.SERVED), ingress, egress, args[-1])], deadline)

    def body_advised(B, S_, *args):
        n = len(ADVICE)
        oks, (egress, ingress), deadline = _checks(B, S_, args[:-n], check)
        return _outputs(B, oks + [B.call(advised(*S_.ADVISED), ingress, egress, *args[-n:])], deadline)

    return (CompositeDefinition(
        "NetworkLinkWindow", 5, ("EGRESS", "INGRESS"), lambda S_: (_SIGNED + _heads(S_), _OUT), body,
        doc="the proof block of network accounting: one link-window, both directions, over the two certificates' signed "
            "digests; EGRESS and INGRESS are each end's head check's (L, T, COUNTS). Its two output bits are the "
            "verdict, both head checks, and the deadline, both heads' deadline bits", register=False),
        CompositeDefinition(
        "NetworkServedLinkWindow", 4, ("EGRESS", "INGRESS", "SERVED"),
        lambda S_: (_SIGNED + _heads(S_) + (_SYNC,), _OUT), body_served,
        doc="the proof block of network accounting on a served link: `NetworkLinkWindow`'s checks, and the egress "
            "head's counts derived from the ingress head's under the declared sync (`served.NetworkServedCounts`, "
            "SERVED its (T, FOUT, FIN, R, RIN)), in its verdict bit", register=False),
        CompositeDefinition(
        "NetworkAdvisedLinkWindow", 2, ("EGRESS", "INGRESS", "ADVISED"),
        lambda S_: (_SIGNED + _heads(S_) + _advice(S_), _OUT), body_advised,
        doc="the proof block of network accounting on an advised link: `NetworkLinkWindow`'s checks, and the egress "
            "head's counts held to the advice's releases, and the advice to the ingress head's counts "
            "(`advised.NetworkAdvisedCounts`, ADVISED its (T, RIN, OWN)), in its verdict bit", register=False))


_UNIT, _SERVED_UNIT, _ADVISED_UNIT = _units(HC.gate, SV.gate, AD.gate)
#: the same units with each gate replaced by the Definition it names, as a proof's circuit lowers them
_PROOF_UNIT, _PROOF_SERVED_UNIT, _PROOF_ADVISED_UNIT = _units(_head_expanded, _served_expanded, _advised_expanded)


def unit(egress: tuple[int, int, int], ingress: tuple[int, int, int], served: tuple[int, ...] | None = None,
         advised: tuple[int, int, int] | None = None, *, proof: bool = False):
    """N2's proof block for heads of the size classes `egress` and `ingress`, each (L, T, COUNTS) (`head_check.unit`),
    and on a served link the served counts' size class `served`, (T, FOUT, FIN, R, RIN, OWN) (`served.unit`), or on an
    advised link the advised counts' `advised`, (T, RIN, OWN) (`advised.unit`). With `proof`, the unit a proof's
    circuit lowers: the same id and signature, so the template the registered query names, with each gate the
    Definition it names once that Definition's circuit digest is the gate's pin (`head_check.expand`, which traces it:
    minutes a head check, and a `ValueError` for an unpinned size class)."""
    if served is not None and advised is not None:
        raise ValueError("a link is served or advised, not both")
    plain, served_, advised_ = ((_PROOF_UNIT, _PROOF_SERVED_UNIT, _PROOF_ADVISED_UNIT) if proof
                                else (_UNIT, _SERVED_UNIT, _ADVISED_UNIT))
    if advised is not None:
        return bind(advised_, EGRESS=tuple(egress), INGRESS=tuple(ingress), ADVISED=tuple(advised))
    if served is None:
        return bind(plain, EGRESS=tuple(egress), INGRESS=tuple(ingress))
    return bind(served_, EGRESS=tuple(egress), INGRESS=tuple(ingress), SERVED=tuple(served))

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
             "windows, the link's framing (the ingress frames a request of each prompt length takes) and the "
             "certifier's own frames a window (its salt's), fixed before the run"),
    S.Public("sync", "output", "each window's declared clock sync, one of Σ_sync (fixed before the run): the advice "
             "`charge` already counts, whose δ and ρ the unit's ingress floor reads, refused outside Σ_sync"),
    S.Public("advice", "output", "the advice (`advised.LinkAdvice`, one per certifier session, after serving): I1's, "
             "charged by `advice.bits`, and its windows' received request frames, which the unit holds to the "
             "ingress heads. The unit reads its window view, `advised.view` of its kept outputs' release buckets, its "
             "requests' ingress ends (`advised.ends`, from its prompt lengths, which the host holds to the certified "
             "ingress's by the tie, and the framing) and its windows' received request frames, in the proof's public "
             "row; each window reports the SHA-512 of the advice's encoding and those frames (`advised.digest`), "
             "which the network verdict holds to one advice"),
)
#: a warden link's: no occupancy profile and no sync, the warden's schedule instead, which puts its counts
PUBLIC_WARDEN = tuple(p for p in PUBLIC if p.item not in ("charge", "occupancy")) + (
    S.Public("warden", "structure", "the active warden's schedule (`warden.WardenParams`, the digest both ends' heads "
             "bind) and the link's windows, fixed before the run: the device's code puts every bucket's cells (r for "
             "each live connection), so its counts are the schedule's, which the period's schedule terms charge"),
)
#: an advised link's: and the advice its window read (`advised.digest`), which the verifier reports beside the unit's
#: two output bits
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


def _root_of(record: Any) -> Any:
    """A link-window record's one salted root, None when it holds no such root."""
    roots = record.get("roots") if isinstance(record, Mapping) else None
    if isinstance(roots, list) and len(roots) == 1 and isinstance(roots[0], Mapping):
        return roots[0].get("root")
    return None


def lean_draw(k: int, n: int, read: bytes) -> dict[str, Any]:
    """The period's draw of record: C-Flock's Lean verifier of record (`proofs.executable`, which verifies the
    windows' proofs) drawing `subset:k` of n on `read`, the coin bytes the period draw read, in order
    (``flock-verify draw --population n --k k --stream-file``, `Flock.Draw.drawOn`, which `select_draw`'s sampler is
    byte for byte: `test_lean_verifier.test_unit_draw_samplers_are_the_exact_reduction` and `test_coins`); its ``DRAW``
    object, or ``{"refused": why}`` where the bytes run out or the Lean refuses or can't run."""
    try:
        with tempfile.TemporaryDirectory(prefix="verity-period-draw-") as d:
            (Path(d) / "coins.bin").write_bytes(read)
            p = subprocess.run([str(PF.executable()), "draw", "--population", str(n), "--k", str(k), "--stream-file",
                                str(Path(d) / "coins.bin")], capture_output=True, text=True)
    except (OSError, PF.LeanError) as e:
        return {"refused": f"flock-verify draw can't run: {e}"}
    lines = [x[len("DRAW "):] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"refused": f"flock-verify draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}"}
    try:
        return json.loads(lines[0])
    except ValueError:
        return {"refused": f"flock-verify draw answered a DRAW line that is not JSON: {lines[0][:300]}"}


_NO_PROFILE = ("N2 compares no egress grid, so a link whose egress end has no occupancy profile has no verdict: "
               "nothing would hold its egress counts (declare LinkWindows.occupancy, LinkWindows.served on a served "
               "link, LinkWindows.advised on an advised one, or LinkWindows.warden on a warden's)")


def _accepts(egress: Sequence[Any], ingress: Sequence[Any], window: int) -> bool:
    return len(egress) > window and egress[window].accepted and ingress[window].accepted


def _profile(rows: Sequence[Sequence[int]] | None) -> list[list[int]] | None:
    return None if rows is None else [list(r) for r in rows]


@dataclass(frozen=True)
class Delivered:
    """A link-window's proof as the verifier receives it: the statement's proof files (`proofs.Proof`, its circuit file
    the auditor's own lowering of the unit), and for the drawn unit the salts of its public row and its output row
    (``pub``, ``o0``) and its output word, which the verifier opens against the public file's commitments to those rows.
    Never the witness row's salt."""

    proof: PF.Proof
    salts: Mapping[str, bytes]
    output: int

    def sessions(self) -> list[str]:
        """The delivered sessions' SHA-512s, their `session.json`s', in order."""
        return [PF.file_sha512(Path(s) / "session.json") for s in self.proof.sessions]


#: a row commitment's bytes in a statement's public file, ``b || c`` (`hm96.SHA512.commit_string`)
BC_BYTES = 128


def _commitments(public: Path, unit: int) -> dict[str, bytes] | None:
    """The row commitments a statement's public file holds for its one instance whose unit index is `unit`, by port
    (`PORTS`, in their order after the header line): instance-major, or in a shared-row file (header ``shared_rows``,
    `verity_flock.circuit.write`) each port's table port-major, then every instance's refs into them (u32 LE,
    instance-major, the ports in order); None when no one instance is, or the file is short."""
    try:
        with open(public, "rb") as f:
            head = json.loads(f.readline())
            indices, shared = list(head["units"]["indices"]), head.get("shared_rows")
            if indices.count(unit) != 1:
                return None
            i, k = indices.index(unit), len(PORTS)
            if shared is None:
                f.seek(f.tell() + i * k * BC_BYTES)
                data = f.read(k * BC_BYTES)
            else:
                counts = [int(shared[p]) for p in PORTS]
                if set(shared) != set(PORTS) or min(counts) < 1:
                    return None
                body = f.read(sum(counts) * BC_BYTES + len(indices) * k * 4)
                if len(body) != sum(counts) * BC_BYTES + len(indices) * k * 4:
                    return None
                refs = body[sum(counts) * BC_BYTES + i * k * 4:][:k * 4]
                data, base = b"", 0
                for j, n in enumerate(counts):
                    r = int.from_bytes(refs[4 * j:4 * j + 4], "little")
                    if r >= n:
                        return None
                    data += body[(base + r) * BC_BYTES:(base + r + 1) * BC_BYTES]
                    base += n
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if len(data) != k * BC_BYTES:
        return None
    return {p: data[j * BC_BYTES:(j + 1) * BC_BYTES] for j, p in enumerate(PORTS)}


def _opens(data: bytes, bits: int, salt: Any, bc: bytes) -> bool:
    """Whether `bc` commits to the `bits`-bit row `data` under `salt`: hm96 over its sha512/row/v2 digest."""
    if not isinstance(salt, bytes) or len(salt) != _INST.salt_bytes:
        return False
    return _INST.commit_string(_INST.default_key, sha512_row_v2_digest(data, bits), salt) == bc


def output_row(word: int) -> bytes:
    """The output row's bytes for the output word `word` (`OUT_BITS`)."""
    return word.to_bytes(OUT_BITS // 8, "little")


def _opened(delivered: Delivered, unit: int, public: bytes | None, bits: int) -> int | None:
    """The unit's output word, once the proof's public file holds the unit's public row as `public` (the verifier's own
    copy, `bits` long) and its output row as the word delivered, each under the salt delivered; else None."""
    bcs, word, salts = _commitments(delivered.proof.public, unit), delivered.output, delivered.salts
    if public is None or bcs is None or type(word) is not int or not 0 <= word < 1 << len(OUTPUTS):
        return None
    if not (_opens(public, bits, salts.get("pub"), bcs["pub"])
            and _opens(output_row(word), OUT_BITS, salts.get("o0"), bcs["o0"])):
        return None
    return word


@dataclass(frozen=True)
class Opening:
    """The firewall's side of one link-window's commitment, all witness: the egress and the ingress certified records,
    in leaf order, and one salt per leaf (`rows`'s order: the two records', then the two heads')."""

    window: int
    records: tuple[CertifiedRecord, CertifiedRecord]
    salts: tuple[bytes, bytes, bytes, bytes]


@dataclass(frozen=True)
class WardenProfile:
    """A warden link's profile (`LinkWindows.warden`, the module docstring's "A warden link"): the link's windows and the
    warden's parameters digest (`warden.WardenParams.digest`), which both ends' heads bind, both fixed before the run."""

    windows: int
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError(f"a warden link has one or more windows, not {self.windows!r}")
        if type(self.digest) is not bytes or len(self.digest) != HC.DIGEST_BYTES:
            raise ValueError(f"a warden's parameters digest is {HC.DIGEST_BYTES} bytes")


@dataclass(frozen=True)
class LinkWindows:
    """Network accounting's protocol on one link, every field public: the link, its egress and ingress grids, Σ_sync, the
    network certifier's public key, the run that names the served domains, the certifier session's id (the 32 bytes the
    challenger drew for the certifier's session, which every certificate signs), the proof deadline in seconds, and the
    occupancy profile: for each end (egress, ingress), each window's counts, or None for an end without one. A served
    link (`served`, FIXLEN) or an advised one (`advised`, arbitrary traffic under the advice) has no egress profile, nor
    has a warden's (`warden`: the active warden's records, its key and session the certifier's here, its counts its
    schedule's, and no sync, so `sync_set` is unread).
    `ties` is no public field but the verifier's own source of an advised link's tie (`advised.TieSource`), the only
    place `_view` takes IoTieRequest's verdict from (`tie_service.Ties`, on the ingress tie's stream of record): None
    refuses every advised link. `certified_clocks` is no public field either: the certifier's salted clock
    certificates of the link's egress windows with their openings (`certificate.CertifiedClock`: the sync and salt
    are witness, the certificate public), the verifier's own read of the certifier's output, never a public answer.
    `clock_source` opens each against the salted digest its certificate signs, under the link's key and session
    (`advised.SignedClocks`), and gives the sync the certifier signed for each window, which every floor reads and an
    advised window's declared sync must be. `clocks` is a stand-in clock source in their place (a test's, or a run's
    log, labelled as one), never both; with neither, every advised link is refused."""

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
    warden: WardenProfile | None = None
    ties: AD.TieSource | None = field(default=None, compare=False, repr=False)
    clocks: AD.ClockSource | None = field(default=None, compare=False, repr=False)
    certified_clocks: Sequence[CertifiedClock] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.occupancy) != 2:
            raise ValueError("the occupancy profile has one entry per end: (egress, ingress)")
        if self.clocks is not None and self.certified_clocks is not None:
            raise ValueError("a link's syncs come from the certifier's clock certificates or a stand-in clock source "
                             "(clocks), not both")
        if self.served is not None and self.advised is not None:
            raise ValueError("a link is served (FIXLEN) or advised, not both")
        if self.warden is not None:
            if self.served is not None or self.advised is not None or tuple(self.occupancy) != (None, None):
                raise ValueError("a warden link's counts are its schedule's: it has no occupancy, served or advised "
                                 "profile")
            if self.ingress_params != self.params:
                raise ValueError("a warden link's two ends are on the warden's one grid")
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

    @property
    def kind(self) -> str:
        """``"served"``, ``"advised"``, ``"warden"``, or ``"plain"`` (none of them: its egress counts are its occupancy
        profile's)."""
        return ("served" if self.served is not None else "advised" if self.advised is not None
                else "warden" if self.warden is not None else "plain")

    @cached_property
    def clock_source(self) -> AD.ClockSource | None:
        """The syncs `_view` holds an advised window's declared sync to: the certifier's salted clock certificates,
        each opened under this link's key and session (`advised.SignedClocks`), the verifier of record's; else the
        stand-in `clocks`; else None, which refuses every advised link."""
        if self.certified_clocks is not None:
            return AD.SignedClocks(self.key, self.certifier_session, tuple(self.certified_clocks))
        return self.clocks

    def declares(self, direction: int, window: int, counts: Sequence[int]) -> bool:
        """Whether `counts` are the occupancy profile's for the end (0 egress, 1 ingress) in the window: always for an
        end with no profile, never for a window past its profile."""
        profile = self.occupancy[direction]
        return profile is None or (window < len(profile) and tuple(counts) == tuple(profile[window]))

    @property
    def charge(self) -> float:
        """log2 #Σ_sync: 7 bits per window at 16 × 8; 0 on a warden link, which declares no sync."""
        return 0.0 if self.warden is not None else self.sync_set.bits

    def head_key(self, direction: int) -> tuple[int, int, int]:
        """The end's head check's size class (L, T, COUNTS): the link name's UTF-8 length, the end's buckets, and 1 when
        the end has an occupancy profile."""
        g = (self.params, self.ingress_params)[direction]
        return len(self.link.encode("utf-8")), g.buckets, int(self.occupancy[direction] is not None)

    def grid(self, direction: int) -> bytes:
        """The end's declared grid as its head encodes it (`head_check.grid_bytes`): the egress grid's parameters digest
        binds Σ_sync, the ingress grid's none; on a warden link both are the warden's (`WardenProfile.digest`)."""
        if self.warden is not None:
            return HC.grid_bytes((self.params, self.ingress_params)[direction], self.warden.digest)
        if direction == 0:
            return HC.grid_bytes(self.params, parameters_digest(self.params, self.sync_set))
        return HC.grid_bytes(self.ingress_params, parameters_digest(self.ingress_params))

    def served_key(self) -> tuple[int, int, int, int, int, int] | None:
        """The served counts' size class (T, FOUT, FIN, R, RIN, OWN) on a served link, else None."""
        if self.served is None:
            return None
        return (self.params.buckets, self.served.response_frames, self.served.request_frames, self.params.allocation,
                self.ingress_params.allocation, self.served.own)

    def advised_key(self) -> tuple[int, int, int] | None:
        """The advised counts' size class (T, RIN, OWN) on an advised link, else None."""
        if self.advised is None:
            return None
        return self.params.buckets, self.ingress_params.allocation, self.advised.own

    @property
    def windows(self) -> int:
        """The link's windows, which `network_verdict` takes by default: the served, advised or warden profile's, else
        the egress occupancy profile's."""
        self._profiled()
        derived = self.served or self.advised or self.warden
        return derived.windows if derived is not None else len(self.occupancy[0])

    @cached_property
    def unit(self):
        """N2's proof block for this link (`unit`): its heads' size classes are the link's, and on a served link its
        served counts' (`served_key`), on an advised link its advised counts' (`advised_key`)."""
        return unit(self.head_key(0), self.head_key(1), self.served_key(), self.advised_key())

    @cached_property
    def proof_unit(self):
        """The unit a proof's circuit lowers: `unit` with each gate the Definition it names (`unit(proof=True)`), the
        template of the registered query under the same id. The prover's and the auditor's lowering, never the
        registered Program, which holds the gates."""
        return unit(self.head_key(0), self.head_key(1), self.served_key(), self.advised_key(), proof=True)

    @cached_property
    def query(self) -> dict[str, Any]:
        return P.template_instance_query(self.unit)

    @cached_property
    def _program(self) -> Program:
        return P.population_program(self.unit, 1)

    @cached_property
    def _partition(self) -> dict[str, Any]:
        return P.partition(self._program, self.query)

    @property
    def partition(self) -> dict[str, Any]:
        """The registered partition object, which a statement proving the unit binds (its digest and its ``program``)."""
        return self._partition

    @cached_property
    def spec(self) -> S.Spec:
        context = canonical_json({
            "format": SPEC, "link": self.link, "run": self.run, "key": self.key.key_id.hex(),
            "certifier_session": self.certifier_session.hex(), "charge": self.charge,
            "parameters": {"egress": self.grid(0)[-HC.DIGEST_BYTES:].hex(),
                           "ingress": self.grid(1)[-HC.DIGEST_BYTES:].hex()},
            "occupancy": [None if p is None else _sha512([list(row) for row in p]) for p in self.occupancy],
            **({} if self.served is None else {"served": [self.served.response_frames, self.served.request_frames,
                                                          self.served.windows, self.served.own]}),
            **({} if self.advised is None else {"advised": [self.advised.windows, self.advised.own,
                                                            _sha512(list(self.advised.request_frames))]}),
            **({} if self.warden is None else {"warden": [self.warden.windows, self.warden.digest.hex()]})})
        check = S.Check(P.program_sha512(Program(self.unit)), self.query,
                        public_outputs=OUTPUTS if self.advised is None else OUTPUTS_ADVISED,
                        proof_deadline=self.proof_deadline)
        public = (PUBLIC_SERVED if self.served is not None else PUBLIC_ADVISED if self.advised is not None
                  else PUBLIC_WARDEN if self.warden is not None else PUBLIC)
        return S.Spec((S.Value(VALUE, SCHEMA, "serve"),), S.All(), check, public, context)

    def stream(self, **kwargs: Any) -> S.Stream:
        """The challenger's stream for this link (`S.Stream`'s budget, delta, challenger, coins and clock)."""
        return S.Stream(self.spec, self.unit, **kwargs)

    def domain(self, window: int) -> CommitmentDomain:
        """One-stage's window domain (`registration.window_domain`) for the link-window: run `<run>/<link>/<window>`."""
        binding = identity_digest_sha512(R.WINDOW_DOMAIN, {
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

    def register(self, stream: S.Stream, record: Mapping[str, Any],
                 drawn: Mapping[str, Any] | None = None) -> S.Registration:
        """`stream.register` with the link-window's own ports, after holding the record's `window` provenance to exactly
        `{"kind": "served", "window": w}`: the service declares the provenance one public item and doesn't read it,
        and anything else there would reach the auditor. w is the registration's order in the stream, or with `drawn`,
        a period draw (`period_draw`) over this stream, the order-th drawn window, whose record's root must be the one
        the period committed. Raises `Refused` (`window`, `period`) otherwise."""
        order = len(stream.log)
        if drawn is None:
            window = order
        else:
            units = self._drawn(stream, len(drawn.get("roots") or ()) if isinstance(drawn, Mapping) else 0, drawn)
            if order >= len(units):
                raise S.Refused(Verdict(False, ("period",), {"period": {"drawn": units, "registered": order}}))
            window = units[order]
        if not isinstance(record, Mapping) or record.get("window") != self.provenance(window):
            raise S.Refused(Verdict(False, ("window",), {"window": {"expected": self.provenance(window)}}))
        if drawn is not None and _root_of(record) != drawn["roots"][window]:
            raise S.Refused(Verdict(False, ("period",), {"period": {"window": window, "root": "not the committed one"}}))
        return stream.register(record, self.ports(window))

    # the prover's side: the unit's arguments and outputs on the opening

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

    @cached_property
    def layout(self) -> dict[str, tuple[str, ...]]:
        """The statement's two input rows (`ROWS`), each its parameters in the unit's signature's order: ``pub`` the
        public ones, ``wit`` each end's head, its leaf's salt, its two siblings and the certifier's salt. A row is the
        little-endian concatenation of its parameters' bits, in whole u16 words, which each row fills exactly: every
        parameter is a multiple of 16 bits but each end's head, of 157 + L + 4T bytes, and the two heads together are.
        Raises `ValueError` for a unit whose rows aren't whole words."""
        params, _ = self.unit.signature()
        out = {r: tuple(n for n, _ in params if ("_" in n and n.split("_", 1)[1] in _WITNESS) == (r == "wit"))
               for r in ROWS}
        for r, names in out.items():
            bits = sum(t.leaves for n, t in params if n in names)
            if bits % 16:
                raise ValueError(f"the unit's {r} row is {bits} bits, not whole u16 words")
        return out

    def bits(self, row: str) -> int:
        """A row's length in bits: `layout`'s, or the output row's (``o0``, `OUT_BITS`)."""
        if row == "o0":
            return OUT_BITS
        params, _ = self.unit.signature()
        return sum(t.leaves for n, t in params if n in self.layout[row])

    def _arguments(self, insts: Sequence[HC.Instance], sync: tuple[int, int, int] | None,
                   view: AD.WindowView | None = None) -> dict[str, list[int]]:
        out = {end: HC._digest_bits(inst.signed) for end, inst in zip(_ENDS, insts)}
        for end, inst in zip(_ENDS, insts):
            for (name, _), bits in zip(inst.unit.signature()[0], HC.arguments(inst)):
                if name not in _FIXED:
                    out[f"{end}_{name}"] = bits
        if self.served is not None:
            out["sync"] = SV.arguments((), (), sync)[2]
        if self.advised is not None:
            out |= dict(zip(ADVICE, AD.arguments((), (), view)[2:]))
        return out

    def _sync(self, window: int, syncs: Sequence[ClockSync]) -> tuple[int, int, int] | None:
        if self.served is None:
            return None
        try:
            return SV.sync_words(syncs[window]) if window < len(syncs) else None
        except ValueError:
            return None

    def _view(self, window: int, syncs: Sequence[ClockSync], advice: Any) -> AD.WindowView | None:
        """On an advised link, the window's view of `advice` (`advised.view`) under the tie from the verifier's own
        tie source (`ties`), held to this link and the ingress registrations that source records for it, and the
        syncs its own clock source (`clock_source`) says the certifier ran the link's windows under; None without
        either source or an `advised.LinkAdvice`, and where `view` refuses."""
        clocks = self.clock_source
        if self.advised is None or self.ties is None or clocks is None or type(advice) is not AD.LinkAdvice:
            return None
        try:
            tie = self.ties.verdict(self.link, advice, self.advised)
            return AD.view(advice, window, self.params.buckets, self.params.allocation, syncs, self.sync_set,
                           self.advised.request_frames, tie=tie, link_name=self.link,
                           registrations=self.ties.registrations(self.link), clocks=clocks.syncs(self.link))
        except ValueError:
            return None

    def _takes(self, sync: tuple[int, int, int] | None, view: AD.WindowView | None) -> bool:
        """Whether the unit's input holds the window's declared sync on a served link and its view on an advised one."""
        return (self.served is None or sync is not None) and (self.advised is None or view is not None)

    def arguments(self, opening: Opening, root: bytes, syncs: Sequence[ClockSync] = (),
                  advice: Any = None) -> dict[str, list[int]] | None:
        """The unit's arguments on the opening under `root`, the registered root, by parameter, each its leaves' bits
        (each end's `head_check.arguments`, on a served link the window's declared sync, and on an advised link the
        window's view of `advice`): the prover's side. None where the unit takes no instance (`head_checks`), on a
        served link no sync its input can hold, or on an advised link a view `advised.view` refuses."""
        insts = self.head_checks(opening, root)
        sync, view = self._sync(opening.window, syncs), self._view(opening.window, syncs, advice)
        if None in insts or not self._takes(sync, view):
            return None
        return self._arguments(insts, sync, view)

    def public_arguments(self, window: int, root: bytes, certificates: Sequence[Certificate],
                         syncs: Sequence[ClockSync] = (), advice: Any = None) -> dict[str, list[int]] | None:
        """The unit's public arguments (`layout`'s ``pub``) as the verifier derives them from public fields alone: the
        registered `root`, the link-window's domain, the link, the window, each end's grid and its profile's row, each
        end's certificate's signed digest, on a served link the window's declared sync, and on an advised link the
        window's view of the advice (`advised.view`). None where the unit takes no instance: a window past an end's
        profile, a signed digest of another length, no sync, or a view `advised.view` refuses."""
        sync, view = self._sync(window, syncs), self._view(window, syncs, advice)
        if len(certificates) != 2 or not self._takes(sync, view):
            return None
        insts = []
        for d, c in enumerate(certificates):
            L, T, _ = self.head_key(d)
            profile = self.occupancy[d]
            if (profile is not None and window >= len(profile)) or len(getattr(c, "digest", b"")) != HC.DIGEST_BYTES:
                return None
            insts.append(HC.Instance(root, self.domain(window).domain_id, self.link.encode("utf-8"), d, window,
                                     self.grid(d), None if profile is None else tuple(profile[window]), c.digest,
                                     bytes(HC.Shape(L, T).HL), bytes(_INST.salt_bytes),
                                     (bytes(HC.DIGEST_BYTES), bytes(HC.DIGEST_BYTES)),
                                     bytes(64), T))
        args = self._arguments(insts, sync, view)
        return {n: args[n] for n in self.layout["pub"]}

    def row(self, args: Mapping[str, Sequence[int]], row: str) -> bytes:
        """A row's bytes (`layout`): its parameters' bits in order, least significant first."""
        bits = [b for n in self.layout[row] for b in args[n]]
        return bytes(sum(b << j for j, b in enumerate(bits[k:k + 8])) for k in range(0, len(bits), 8))

    def outputs(self, opening: Opening, root: bytes, syncs: Sequence[ClockSync] = (), advice: Any = None) -> int:
        """The unit's output word on the opening under `root` by each gate's reference (`head_check.reference`, on a
        served link `served_check`, and on an advised link `advised_check` on `advice`), which the tests hold to the
        circuits: bit 0 the verdict, bit 1 the deadline. What the prover claims, and the debug comparison's."""
        verdict = all(i is not None and HC.reference(i) for i in self.head_checks(opening, root))
        verdict = verdict and (self.served is None or self.served_check(opening, syncs))
        verdict = verdict and (self.advised is None or self.advised_check(opening, syncs, advice))
        return int(verdict) | int(all(r.head.deadline for r in opening.records)) << 1

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

    @cached_property
    def program(self) -> bytes:
        """The registered Program's descriptor, the verifier's own copy, whose SHA-512 the partition object names."""
        return canonical_json(encode_program(self._program))

    def certified(self, window: int, certificates: Sequence[Certificate]) -> bool:
        """The certificates' check, on the host from their public fields alone: two certificates, each the certifier's
        and verifying in its session over its fields and its signed digest (`Certificate.verify`), and each naming this
        link, its end (egress first) and the window. Each signed digest is the one its end's head check reads
        (`public_arguments`), which a proof holds to the head it opens."""
        return len(certificates) == 2 and all(
            isinstance(c, Certificate) and (c.link, c.direction.code, c.window) == (self.link, d, window)
            and c.verify(self.key, self.certifier_session) for d, c in enumerate(certificates))

    def _profiled(self) -> None:
        if self.occupancy[0] is None and self.served is None and self.advised is None and self.warden is None:
            raise ValueError(_NO_PROFILE)

    def _advice_output(self, advice: Any) -> dict[str, str | None]:
        """On an advised link, the `advice` output: the advice the window read (`advised.digest`), None without one."""
        if self.advised is None:
            return {}
        return {"advice": AD.digest(advice) if type(advice) is AD.LinkAdvice else None}

    @staticmethod
    def _decided(draw: S.Draw, sessions: Sequence[str], accepted: bool, public: dict[str, Any]) -> dict[str, Any]:
        """A verdict's accept bit and public outputs, with the digest binding them to the registration and sessions."""
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": list(sessions),
                          "accepted": accepted, "public": public})
        return {"accepted": accepted, "digest": digest, "public": public}

    def python_verdict(self, draw: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                       delivered: Delivered, findings: Sequence[Finding] | None, sessions: Sequence[str],
                       advice: Any = None) -> dict[str, Any]:
        """This module's verdict on the link-window (`verify`'s conditions), on the findings of the verifier's own run
        of C-Flock's Lean verifier (None where `proofs.verify` refused): its accept bit, digest and public outputs."""
        reg = draw.registration.record
        w, root = reg["window"]["window"], bytes.fromhex(reg["roots"][0]["root"])
        proved = findings is not None and self.proved(findings) \
            and all(list(f.units) == list(draw.units) for f in findings)
        args = self.public_arguments(w, root, certificates, syncs, advice)
        opened = _opened(delivered, draw.units[0], self.row(args, "pub") if args is not None else None,
                         self.bits("pub")) if proved else None
        word = opened if opened is not None else 0
        accepted = (proved and opened is not None and word == 0b11 and self.certified(w, certificates)
                    and (self.warden is not None or (w < len(syncs) and syncs[w] in self.sync_set)))
        return self._decided(draw, sessions, accepted,
                             {"verdict": int(accepted), "deadline": word >> 1 & 1} | self._advice_output(advice))

    def lean_case(self, draw: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                  delivered: Delivered, findings: Sequence[Finding] | None, sessions: Sequence[str],
                  advice: Any = None) -> dict[str, Any] | None:
        """`verify`'s inputs as `NetworkCheck.LinkWindow.caseOf` reads them: the verifier's own structure (the Program's
        and partition's digests, the gate ids), the registration, the draw, the findings of its own run of C-Flock's
        Lean verifier (None where `proofs.verify` refused), the delivered public file, salts, output word and session
        SHA-512s, the certificates (each null unless a `Certificate` of well-typed fields) and each one's signed digest
        as `public_arguments` reads it, the declared syncs, Σ_sync,
        the key and session, each end's grid and profile, on an advised link the window's view (`advised.view`) and
        the advice output, and on a warden link the warden's parameters digest. None for a key the Lean doesn't read
        (not an `LmsKey`)."""
        if not isinstance(self.key, LmsKey):
            return None
        reg, p = draw.registration.record, self.key.params
        w = reg["window"]["window"]

        def cert(c: Any) -> dict[str, Any] | None:
            if not (isinstance(c, Certificate) and all(type(x) is bytes for x in (c.key_id, c.digest, c.signature))
                    and type(c.link) is str and isinstance(c.direction, Direction) and type(c.window) is int
                    and c.window >= 0):
                return None
            return {"key_id": c.key_id.hex(), "link": c.link, "direction": c.direction.code, "window": c.window,
                    "digest": c.digest.hex(), "signature": c.signature.hex()}

        def hexed(b: Any) -> str | None:
            return b.hex() if type(b) is bytes else None

        def digest(c: Any) -> str:
            d = getattr(c, "digest", b"")
            return bytes(d).hex() if isinstance(d, (bytes, bytearray)) else ""

        try:
            public = Path(delivered.proof.public).read_bytes().hex()
        except OSError:
            public = None
        word = delivered.output
        case: dict[str, Any] = {
            "kind": self.kind, "link": self.link, "run": self.run, "window": w, "root": reg["roots"][0]["root"],
            "registration": draw.registration.digest, "units": list(draw.units), "program": self._partition["program"],
            "partition": P.digest(self._partition), "gates": [g.id for g in self.gates()],
            "findings": None if findings is None else [
                {"accepted": f.accepted, "program": f.program, "units": list(f.units)} for f in findings],
            "public": public, "salts": {k: hexed(delivered.salts.get(k)) for k in ("pub", "o0")},
            "output": word if type(word) is int and word >= 0 else None, "sessions": list(sessions),
            "certificates": [cert(c) for c in certificates], "digests": [digest(c) for c in certificates],
            "syncs": [[s.delta, s.rho.numerator, s.rho.denominator] for s in syncs],
            "sync_set": {"deltas": list(self.sync_set.deltas),
                         "rhos": [[r.numerator, r.denominator] for r in self.sync_set.rhos]},
            "lms": {"N": p.N, "W": p.W, "H": p.H, "toy": p.toy}, "key": self.key.public.hex(),
            "session": bytes(self.certifier_session).hex(),
            "egress": {"grid": LA._grid_json(self.params), "profile": _profile(self.occupancy[0])},
            "ingress": {"grid": LA._grid_json(self.ingress_params), "profile": _profile(self.occupancy[1])}}
        if self.advised is not None:
            v = self._view(w, syncs, advice)
            case["view"] = None if v is None else {"released": list(v.released), "floor": list(v.floor),
                                                   "received": v.received}
            case["advice"] = self._advice_output(advice)["advice"]
        if self.warden is not None:
            case["warden"] = self.warden.digest.hex()
        return case

    def lean_verdict(self, draw: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                     delivered: Delivered, findings: Sequence[Finding] | None, sessions: Sequence[str],
                     advice: Any = None, server: LA.Server | None = None) -> dict[str, Any] | None:
        """The Lean's verdict on the link-window (`NetworkCheck.LinkWindow.verdict`, `network-records verdicts`'
        ``link-window``): its accept bit, digest and public outputs; None where `lean_case` has no case or the Lean
        refuses it. The Lean not built or not running raises."""
        case = self.lean_case(draw, certificates, syncs, delivered, findings, sessions, advice)
        if case is None:
            return None
        try:
            out = (server or LA.SERVER).call("link-window", case)
        except ValueError:
            return None
        return {"accepted": out["accepted"], "digest": out["digest"], "public": out["public"]}

    def verify(self, draw: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
               delivered: Delivered, *, circuit_sha512: str, opening: Opening | None = None,
               advice: Any = None) -> dict[str, Any]:
        """The verdict of record on the link-window's delivered proof, from public inputs and the proof alone: accepted
        when
        - the verifier's own run of C-Flock's Lean verifier (`proofs.verify`, on the circuit file the auditor built,
          `circuit_sha512`) accepts every session, and they prove this link's registered unit (`proved`) and the drawn
          unit;
        - the proof's public row opens, under the salt delivered, to the verifier's own copy (`public_arguments`: the
          registered root, the public fields and both certificates' signed digests, and on an advised link the window's
          view of `advice`, the link's advice, which `advised.view` takes under the tie from `ties` alone), and its
          output row to the word delivered (`_opened`), whose verdict bit and deadline bit are both 1;
        - the certificates' check accepts (`certified`), and the window's declared sync is one of Σ_sync (on a warden
          link, which declares none, nothing is read of `syncs`).
        The verdict reports the public outputs, the proved deadline bit among them, on an advised link the advice it
        read (`advised.digest`), and never a violation's kind. `opening` (the witness) is for a debug comparison only,
        off by default: given, the verdict also carries the in-the-clear check's accept bit (`clear`), which decides
        nothing. Raises `ValueError` on a link with no egress occupancy, served, advised or warden profile.

        The Lean (`NetworkCheck.LinkWindow`, `lean_verdict`) decides the same on every call beside this Python, and
        `OF_RECORD` names whose verdict is the record's (`by`); the other's accept bit is `reference`'s. Where the two
        differ in the accept bit, the public outputs or the digest, or the Lean refuses the case, the link-window is
        rejected, with deadline 0."""
        self._profiled()
        findings, refused = PF.verify(delivered.proof, circuit_sha512=circuit_sha512, program=self.program,
                                      partition=self._partition)
        findings = None if refused is not None else findings
        sessions = delivered.sessions()
        python = self.python_verdict(draw, certificates, syncs, delivered, findings, sessions, advice)
        lean = self.lean_verdict(draw, certificates, syncs, delivered, findings, sessions, advice)
        by = OF_RECORD[self.kind]
        mine, other = (lean, python) if by == "lean" else (python, lean)
        if lean is None or any(lean[k] != python[k] for k in python):
            mine = self._decided(draw, sessions, False, {"verdict": 0, "deadline": 0} | self._advice_output(advice))
        verdict = {"of_record": True, "decided_by": "proof", "by": by, **mine, "sessions": sessions,
                   "reference": {"accepted": None if other is None else other["accepted"]}}
        if opening is not None:
            verdict["clear"] = {"accepted": self.clear(opening, draw, syncs, advice)["accepted"]}
        return verdict

    def clear(self, opening: Opening, draw: S.Draw, syncs: Sequence[ClockSync], advice: Any = None) -> dict[str, Any]:
        """The in-the-clear check on the opening, the witness: the debug comparison `verify` makes on request and the
        tests' reference, never a verdict of record. Accepted when the opening reproduces the registered root under the
        link-window's domain, both head checks accept, and the audit accepts both records, comparing no grid (the module
        docstring). `syncs` are the declared clock syncs of windows 0 … w (on an advised link, of every window the
        advice releases a frame in). On a served link it also runs the served counts' check (`served_check`), and on an
        advised link the advised counts' on `advice` (`advised_check`), reporting the advice it read as `advice`. On a
        warden link it is the root and the head checks alone (the module docstring's "A warden link")."""
        self._profiled()
        root = self._registered(opening, draw)
        w, (egress, ingress) = opening.window, opening.records
        sealed = self.root(opening) == root
        headed = all(i is not None and HC.reference(i) for i in self.head_checks(opening, bytes.fromhex(root)))
        headed = headed and (self.served is None or self.served_check(opening, syncs))
        headed = headed and (self.advised is None or self.advised_check(opening, syncs, advice))
        if self.warden is not None:
            public = {"verdict": int(sealed and headed), "deadline": int(egress.head.deadline and ingress.head.deadline)}
            return {"of_record": False, "accepted": sealed and headed, "public": public}
        audit = Audit()
        out, inn = self._audit(audit, opening, syncs)
        accepted = sealed and headed and _accepts(out, inn, w)
        public = {"verdict": int(accepted), "deadline": int(egress.head.deadline and ingress.head.deadline)}
        public |= self._advice_output(advice)
        verdict = {"of_record": False, "accepted": accepted, "public": public}
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
        egress head's counts are the advice's releases in the window's buckets (`advised.view`), and the ingress head's
        counts release the request frames that view needs by each bucket, under the syncs the clock source says the
        certifier ran, and the window's `received`. False without an `advised.LinkAdvice`, a tie source (`ties`) or a
        clock source (`clock_source`), or on an advice `view` refuses: not covering the window, without the tie from
        that source or not under it, with requests whose ends under the profile's framing aren't the windows' total,
        releasing more than R frames in a bucket, or a frame past the link's windows, in one the clock source has no
        sync for, or before its request; or where this window declares no sync, one outside Σ_sync or other than the
        clock source's."""
        v = self._view(opening.window, syncs, advice)
        if v is None:
            return False
        egress, ingress = opening.records
        return AD.reference(*self.advised_key(), ingress.head.counts, egress.head.counts, v.released, v.floor,
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

    def statement(self, draw: S.Draw) -> dict[str, Any]:
        reg = draw.registration.record
        return {"format": STATEMENT, "check": self.spec.check.program, "link": self.link,
                "window": reg["window"]["window"], "root": reg["roots"][0]["root"]}

    def outcome(self, stream: S.Stream, draw: S.Draw, certificates: Sequence[Certificate], syncs: Sequence[ClockSync],
                delivered: Delivered, *, circuit_sha512: str, opening: Opening | None = None,
                advice: Any = None) -> S.Outcome:
        """`stream.outcome` on the verifier of record's verdict (`verify`, with an advised link's `advice`), with the
        registered root as the session's: the root the proof's public row names, which it opens only to the verifier's
        own copy."""
        verdict = self.verify(draw, certificates, syncs, delivered, circuit_sha512=circuit_sha512, opening=opening,
                              advice=advice)
        return stream.outcome(draw, {VERIFIER: verdict}, self.statement(draw),
                              session_roots={VALUE: draw.registration.record["roots"][0]["root"]})

    def unproved_outcome(self, stream: S.Stream, draw: S.Draw, opening: Opening,
                         syncs: Sequence[ClockSync], advice: Any = None) -> S.Outcome:
        """`stream.outcome` for a run that proves nothing (a rehearsal, the simulator): on the in-the-clear check
        (`clear`, with an advised link's `advice`), which reads the opening, so it is never zero-knowledge, and its
        verdict says ``"decided_by": "clear"``. The delivered session is the opening's (`session`), and its root the
        one the opening reproduces, which one-stage's audit holds to the registered root."""
        v = self.clear(opening, draw, syncs, advice)
        sessions = [self.session(opening, draw)]
        digest = _sha512({"format": VERDICT, "registration": draw.registration.digest, "sessions": sessions,
                          "accepted": v["accepted"], "public": v["public"], "decided_by": "clear"})
        verdict = {**v, "of_record": True, "decided_by": "clear", "digest": digest, "sessions": sessions}
        return stream.outcome(draw, {VERIFIER: verdict}, self.statement(draw),
                              session_roots={VALUE: self.root(opening)})

    def _period(self, stream: S.Stream, windows: int | None) -> int:
        default = self.windows
        if stream.spec.digest() != self.spec.digest():
            raise ValueError("the stream is not this link's: its spec isn't LinkWindows.spec")
        n = default if windows is None else windows
        if not (type(n) is int and n >= 1):
            raise ValueError(f"a link has one or more link-windows, not {n!r}")
        return n

    def _committed(self, stream: S.Stream, roots: Sequence[str], advice: str | None) -> str:
        """The period's digest, what the coins come after: this stream, and the committed roots and advice."""
        return _sha512({"format": PERIOD_DRAW, "stream": stream.digest, "roots": list(roots), "advice": advice})

    def period_draw(self, stream: S.Stream, roots: Sequence[str], k: int | None, coins: Any, *,
                    advice: str | None = None) -> dict[str, Any]:
        """The period's draw over its link-windows 0 … n − 1, after the prover committed them: `roots`, window w's
        salted root (`root`, hex) at w, n of them (by default the link's windows, which `network_verdict` takes), and on
        an advised link `advice`, the advice's digest (`advised.digest`), which every accepted window must name. Every
        window for `k` None (`All`, reading no coins), else `subset:k`, 1 <= k <= n, on `coins` (a callable of a byte
        count, as a stream's); a coin source with a `commit` method is first committed to the period's digest
        (`commitment`), so the coins come after every commitment of the period. Call it before the stream's first
        registration. The object is `{format, stream, law, k, population, units, roots, advice, commitment, read}`,
        `read` the coin bytes the draw read, in order (hex), from which the network verdict's Lean draws its own
        (`lean_draw`); `register` and `network_verdict` take it as `drawn`. Raises `ValueError` for a stream that isn't
        this link's or holds a registration, roots that aren't n SHA-512 hex digests, an advice on a link that isn't
        advised or none on one that is, and k outside [1, n]."""
        n = self._period(stream, len(roots) if isinstance(roots, Sequence) and not isinstance(roots, str) else None)
        if stream.log:
            raise ValueError("the period is committed and drawn before the stream's first registration")
        roots = list(roots)
        if not all(isinstance(r, str) and len(r) == 128 and all(c in "0123456789abcdef" for c in r) for r in roots):
            raise ValueError("a period commits each window's salted root, a SHA-512 in lowercase hex")
        if (self.advised is not None) != (advice is not None) or not (advice is None or (
                isinstance(advice, str) and len(advice) == 128)):
            raise ValueError("an advised link's period commits its advice's digest (advised.digest), and only an advised "
                             "link's does")
        if k is not None and not (type(k) is int and 1 <= k <= n):
            raise ValueError(f"a period draw takes 1 <= K <= {n} of the link-windows, not {k!r}")
        commitment = self._committed(stream, roots, advice)
        rule = S.All() if k is None else S.Law(f"subset:{k}")
        if k is not None and callable(getattr(coins, "commit", None)):
            coins.commit(bytes.fromhex(commitment))
        read = bytearray()

        def reading(m: int) -> bytes:
            b = coins(m)
            read.extend(b)
            return b
        draw = S.select_draw(rule, S.selection_law(rule, n), n, reading)
        return {"format": PERIOD_DRAW, "stream": stream.digest, "law": "subset", "k": draw["k"], "population": n,
                "units": list(draw["units"]), "roots": roots, "advice": advice, "commitment": commitment,
                "read": read.hex()}

    def _drawn(self, stream: S.Stream, n: int, drawn: Mapping[str, Any]) -> list[int]:
        """The drawn windows, once `drawn` is a period draw over this stream's n windows; else `ValueError`."""
        get = drawn.get if isinstance(drawn, Mapping) else (lambda _: None)
        units, k, roots, read = get("units"), get("k"), get("roots"), get("read")
        if not (isinstance(drawn, Mapping) and get("format") == PERIOD_DRAW and get("law") == "subset"
                and get("population") == n and type(k) is int and 1 <= k <= n and isinstance(units, list)
                and all(type(u) is int and 0 <= u < n for u in units) and units == sorted(set(units))
                and len(units) == k and isinstance(roots, list) and len(roots) == n and isinstance(read, str)
                and len(read) % 2 == 0 and all(c in "0123456789abcdef" for c in read)):
            raise ValueError(f"not a period draw of subset:K over this link's {n} windows")
        if get("stream") != stream.digest or get("commitment") != self._committed(stream, roots, get("advice")):
            raise ValueError("the period draw is not this stream's, or not over the period it committed")
        return list(units)

    @staticmethod
    def _of_record(n: int, drawn: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
        """The period's draw of record over its n windows, ascending, and None; or None and why there is none: every
        window at K = n, else `lean_draw` on the coins `drawn` read, which must be `subset:K` over n."""
        k = drawn["k"]
        if k == n:
            return list(range(n)), None
        got = lean_draw(k, n, bytes.fromhex(drawn["read"]))
        if not isinstance(got, Mapping) or "refused" in got:
            return None, str(got.get("refused") if isinstance(got, Mapping) else got)
        units = got.get("units")
        if not (got.get("law") == "subset" and got.get("k") == k and got.get("population") == n
                and isinstance(units, list) and all(type(u) is int and 0 <= u < n for u in units)
                and len(set(units)) == len(units) == k):
            return None, f"the Lean's draw is not subset:{k} over the {n} windows: {json.dumps(got)[:300]}"
        return sorted(units), None

    def network_verdict(self, stream: S.Stream, windows: int | None = None,
                        drawn: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The link's network verdict of record: the AND over its link-windows 0 … `windows` − 1 (by default the egress
        profile's windows, fixed before the run), with no ε and no share credit. It accepts only when, for every w,
        the stream's w-th registration is window w and its outcome of record accepted. A window registered out of
        order or not at all (a gap, or a run that stops short), a registration past the last window, a registration
        with no outcome yet, and a rejected or late outcome each reject the link; `refused` lists their windows.
        `records` is each window's outcome record SHA-512 (`S.Outcome.record`, None without one), and `digest` binds
        the verdict to the stream and those records. On a served or advised link the windows are by default its
        profile's (`windows`), and on an advised link every accepted window must name the first accepted window's advice
        (its `advice` output, `advised.digest`), which the verdict reports as `advice`, or it is refused.

        With `drawn`, the period draw (`period_draw`) over this stream's n windows, the stream's registrations must be
        exactly the drawn windows, in order, each under the root the period committed for it, and each needs an
        accepted outcome; on an advised link each names the committed advice. An undrawn window has no registration and
        its record is None; a registration past the drawn windows is listed in `refused` as n, n + 1, … The verdict
        then carries `selection` (the draw) and `wrong_units`, what it established at the stream's δ: except with
        probability δ, an accepted link has at most `bound` wrong windows of the `of` committed, and `epsilon` is
        bound / of (the uniform K-subset's exact bound, one-stage's `wrong_units` over windows); `digest` binds the
        draw too.

        The drawn windows are the draw of record's, never the units `drawn` lists: the verdict runs `lean_draw` on
        `drawn`'s coins (`read`), and every window at K = n, which leaves nothing to draw. It refuses the link by name,
        `why[PERIOD_DRAWN]` with the Lean's draw (or why there is none), `drawn`'s units and the windows registered,
        when the Lean can't draw, when `drawn`'s units aren't the Lean's, and when a registered window isn't one the
        Lean drew; a drawn window not registered is in `refused`. `selection` is the draw of record, ``by`` ``lean``
        (or ``all`` at K = n), and its units are the Lean's. Raises `ValueError` on a link with no egress occupancy,
        served or advised profile, as `verify` does, on a stream that isn't this link's, and on a draw that isn't this
        stream's period draw."""
        n = self._period(stream, windows)
        why: dict[str, Any] = {}
        if drawn is None:
            expected = list(range(n))
        else:
            claimed = self._drawn(stream, n, drawn)
            lean, no = self._of_record(n, drawn)
            expected = claimed if lean is None else lean
            by = None if lean is None else "all" if drawn["k"] == n else "lean"
            registered = [p.get("window") if isinstance(p := r.get("window"), Mapping) else None for r in stream.log]
            if lean is None or lean != claimed or any(w not in lean for w in registered):
                why[PERIOD_DRAWN] = {"lean": lean if lean is not None else {"refused": no}, "drawn": claimed,
                                     "registered": registered,
                                     "why": no or "the windows registered and proved are not the Lean verifier of "
                                                  "record's draw on the period's coins"}
        outcomes = {r["registration"]: r for r in stream.records}
        records: list[str | None] = [None] * n
        refused: list[int] = []
        advice = None if drawn is None else drawn.get("advice")
        for i in range(max(len(expected), len(stream.log))):
            reg = stream.log[i] if i < len(stream.log) else None
            w = expected[i] if i < len(expected) else n + i - len(expected)
            in_order = reg is not None and reg.get("window") == self.provenance(w)
            body = outcomes.get(R.digest(reg)) if in_order else None
            record = None if body is None else _sha512(body)
            if w < n:
                records[w] = record
            else:
                records.append(record)
            ok = (w < n and in_order and body is not None and body["symbol"] == S.ACCEPT
                  and (drawn is None or _root_of(reg) == drawn["roots"][w]))
            if ok and self.advised is not None:
                named = body["public"].get("advice")
                advice = named if advice is None else advice
                ok = named is not None and named == advice
            if not ok:
                refused.append(w)
        accepted = not refused and not why
        sampled = {} if drawn is None else {"drawn": dict(drawn), "of_record": expected, "why": why}
        digest = _sha512({"format": NETWORK_VERDICT, "stream": stream.digest, "windows": n, "records": records,
                          "accepted": accepted, **sampled})
        out = {"accepted": accepted, "windows": n, "refused": refused, "records": records, "digest": digest}
        if drawn is not None:
            bound = OA.core_law({"law": "subset", "k": drawn["k"]}, n).count_bound(stream.delta)
            out |= {"why": why,
                    "selection": {**{k: drawn[k] for k in ("law", "k", "population", "commitment")}, "units": expected,
                                  "by": by},
                    "wrong_units": {"delta": stream.delta, "of": n, "bound": bound, "epsilon": bound / n,
                                    "rule": "except with probability delta, an accepted link has at most `bound` wrong "
                                            "link-windows of the `of` its period committed"}}
        return out if self.advised is None else out | {"advice": advice}


__all__ = ["ADVICE", "BC_BYTES", "Delivered", "HEADS", "LEAVES", "LinkWindows", "OUTPUTS", "OUTPUTS_ADVISED",
           "OUT_BITS", "Opening", "PORTS", "PUBLIC", "PUBLIC_ADVISED", "PUBLIC_SERVED", "ROWS", "SCHEMA", "VALUE",
           "encode", "output_row", "rows", "unit"]
