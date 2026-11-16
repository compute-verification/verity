module

public import MemoryChallenger.WipeProtocol
public import MemoryChallenger.WipeCount
public import NetworkCheck.Span
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Definitions.Certifier.Premises
public import Definitions.Certifier.Guarantees.Marks

/-!
# A counted wipe's egress is the network certifier's count over the wipe's span

`MemoryChallenger.WipeProtocol.checkCounted` (`verity/wipe/counted/v0`) publishes the egress its evidence gives
(`egress_mib`), and the survivor bound it certifies is of the foreign content plus that egress. `WipeCountTied` says
what that egress is when the network certifier's count over a span (`NetworkCheck.Span.check`,
`verity/certifier/span/v0`) is accepted for the same marks: at least the bits the device sent in the execution phases
from the start mark's phase to the end mark's phase plus the span's tail, phases that hold every cell the device sends
from the start mark on out of what it read by the end mark. The statement only; `Proofs.Properties.Pous.WipeCount`
proves it from its antecedents.

## The marks

The network certifier on the wipe's link signs two marks for the wipe, each over 64 bytes it is handed, and stamps each
with the execution phase it handles it in (`Definitions.Certifier.Marks`): the start mark over the start message
(`StreamedWipe.startMessage`, of the wipe's commitment), before the fill; and the end mark over the end message
(`StreamedWipe.endMessage`, of the signed session), after the wipe closes. The memory challenger signs the SHA-512 of
the start mark's certificate, the bytes the device handed it, in the session (`StreamedWipe.Signed.startMark`). The
span's commitment holds the two marks (`NetworkCheck.Span.Commitment`), and `NetworkCheck.Span.named` reads off it the
start certificate's SHA-512, the two messages and the bits that can have left over the span and its tail.

## The tie

Each protocol reads its own half: `WipeProtocol.named` of the wipe's case and `NetworkCheck.Span.named` of the span's.
Nothing on `main` holds one half to the other: no composition is in the tree, and `checkCounted` is served only by its
difftest (`tests/test_counted_callers.py`). `SpanTied` is that tie, a named premise stated as narrowly as the proof
needs it: the two messages agree and the wipe's count, in its unit, is at least the span's. A composition's tie would
hold the start mark too, which the proof doesn't read.

## What it rests on

* `SpanTied`, open: nothing on `main` proves it, since nothing composes the two checks.
* `NetTiming.DeviceIntact`, at the phase the device stamps on a mark: the device runs its code (`trusted-device`), so the
  stamp is `NetTiming.markPhase` of the bucket it handles the mark in.
* The memory challenger's side: the session's `start_mark` is the SHA-512 of the certificate of a mark the device made
  over the start message, since the challenger hands the device exactly the start message and hashes the bytes the
  device hands back.
* `CountRan`, open: what an accepted span means, that its marks are two the device made at the phases the count
  names, or its run carries the break. It is to become a theorem about `NetworkCheck.Span.check` whose break is an LMS
  forgery: a certificate that verifies under the device's key in its session on a message the device never signed.
* `MarksOnce`, open until it is stated in the device's model: the device refuses to mark a message twice, so the
  challenger's mark and the span's start mark, both over the start message, are one mark at one phase.
* `NetTiming.Guarantees.SpanCovers`, open until it is proved: the device model's lines for a span.
* `SentWithinCapacity`, open until it is stated in the device's model: the device sends at most a phase's capacity a
  phase, the capacity the span's check counts (`NetworkCheck.Span.capacityBits` of the schedule, which `CountRan` holds
  the device's to).

## What it does not say

* Nothing serves `checkCounted` until the tie exists: the statement is about a composition the tree doesn't have.
* The deployed protocol, `WipeProtocol.protocol` (`verity/wipe/v0`), publishes `WipeProtocol.uncounted`, the saturated
  egress (`StreamedWipe.EGRESS_MAX_UNITS`), at which its check certifies nothing; this statement doesn't read it.
* It stops at the marks. That the fill began after the start mark was handled and that the end mark was handled after
  the wipe closed is the memory challenger's run and the end message's being of the signature.
* A link no network certifier relays has no span, and its wipe no counted case.
-/

@[expose] public section

namespace MemoryChallenger.Properties

open Flock

/-- A mark the network certifier made: the message it signed, the bucket it handled it in (the ticks it had done), and
the certificate it returned, the bytes of the certificate it signs under its key in its session on its link's marks,
egress, at the phase it stamps, over the SHA-512 of a tag, the mark's own fresh salt and the message. -/
structure Mark where
  message : ByteArray
  bucket : ℕ
  certificate : ByteArray

/-- **The network certifier on the wipe's link**: its key, session and link; its capacity a phase, in bits; its
schedule, the buckets of an execution phase (`T`) and the most egress cells a connection has (`E`); the phase the
physical device stamps on a mark handled in a bucket; the marks it made; and the bits it sent on the link in each
phase. -/
structure MarkDevice (κ : Type) where
  key : ByteArray
  session : ByteArray
  link : String
  capacityBits : ℕ
  schedule : NetTiming.CertifierSchedule κ
  phaseBuckets : ℕ
  cellCap : ℕ
  stamp : ℕ → ℕ
  marks : List Mark
  sentBits : ℕ → ℕ

/-- The span count `n`, accepted for the commitment `cs`, is of the marks `a` and `b` of the device `d`: both are marks
it made, `a` over the start mark's 64 bytes and `b` over the end mark's; the phases are the ones the device stamped on
them; the tail is the device's (`NetTiming.spanTail` of its schedule); and the capacity a phase is the device's. -/
def SpanOf {κ : Type} (n : NetworkCheck.Span.Count) (cs : NetworkCheck.Span.Commitment) (d : MarkDevice κ)
    (a b : Mark) : Prop :=
  a ∈ d.marks ∧ b ∈ d.marks ∧ a.message = cs.startMark.mark ∧ b.message = cs.endMark.mark ∧
    n.startPhase = d.stamp a.bucket ∧ n.endPhase = d.stamp b.bucket ∧
    n.tail = NetTiming.spanTail d.schedule.D d.cellCap d.schedule.r d.phaseBuckets ∧ n.capacityBits = d.capacityBits

/-- **The count's egress**, in bits, as `NetworkCheck.Span.publicJson` publishes it: the phases counted times the bits
one phase can carry. -/
def countBits (n : NetworkCheck.Span.Count) : ℕ := n.phases * n.capacityBits

/-- The bits the device `d` sent in the phases the count `n` counts, from its start phase on. -/
def sentOver {κ : Type} (n : NetworkCheck.Span.Count) (d : MarkDevice κ) : ℕ :=
  ∑ i ∈ Finset.range n.phases, d.sentBits (n.startPhase + i)

/-- **What an accepted span means**, of the count's check run under the parameters `p`, the break `broken` its run
carries and the device `d`: every span `NetworkCheck.Span.check` accepts under `p` is of two marks the device made
(`SpanOf`), or the run carries the break. An open premise: it is to become a theorem about `NetworkCheck.Span.check`,
for `p` the device's own key, session and link, whose break is an LMS forgery, a certificate that verifies under the
device's key in its session on a message the device never signed. -/
def CountRan {κ : Type} (p : NetworkCheck.Span.Params)
    (broken : NetworkCheck.Span.Commitment → NetworkCheck.Span.Evidence → Prop) (d : MarkDevice κ) : Prop :=
  ∀ (cs : NetworkCheck.Span.Commitment) (ev : NetworkCheck.Span.Evidence) (n : NetworkCheck.Span.Count)
    (pr : NetworkCheck.Span.Profile),
    NetworkCheck.Span.check p cs ev = .accept n pr → (∃ a b, SpanOf n cs d a b) ∨ broken cs ev

/-- **The tie between the wipe's case and the span's**: the wipe's case names a count (`WipeProtocol.named`), its start
and end messages are the 64 bytes the span's two marks open to (`NetworkCheck.Span.named`), and its count, in whole MiB,
is at least the span's egress bits in that unit (`StreamedWipe.egressUnits`). An open premise: nothing on `main` proves
it, since no composition of the two checks is in the tree. -/
def SpanTied (p : NetworkCheck.Span.Params) (cs : NetworkCheck.Span.Commitment) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) : Prop :=
  ∃ w : WipeProtocol.Named, WipeProtocol.named c e = .ok (some w) ∧
    w.startMessage = (NetworkCheck.Span.named p cs).startMessage ∧
    w.endMessage = (NetworkCheck.Span.named p cs).endMessage ∧
    ∀ b : ℕ, (NetworkCheck.Span.named p cs).egressBits = some b → StreamedWipe.egressUnits b ≤ w.egressMib

/-- **The device sends at most its capacity a phase**, a line of the network certifier's model: in every execution
phase the bits it sends on the link, every cell of every connection, are at most its capacity a phase, so a count's bits
over a span bound the bits it sent in the span's phases. -/
def SentWithinCapacity {κ : Type} (d : MarkDevice κ) : Prop :=
  ∀ φ : ℕ, d.sentBits φ ≤ d.capacityBits

/-- **The device marks a message at most once**, a line of the network certifier's model: it refuses to mark a message
it has marked, so any two of its marks over one message are one mark, with one certificate and one phase. -/
def MarksOnce {κ : Type} (d : MarkDevice κ) : Prop :=
  ∀ a ∈ d.marks, ∀ b ∈ d.marks, a.message = b.message →
    a.certificate = b.certificate ∧ d.stamp a.bucket = d.stamp b.bucket

/-- **The wipe's egress is the count over its span**: the device `d` made a start mark `a` over the start message of the
wipe's commitment `c`, with the certificate whose SHA-512 the session `s` signs, and an end mark `b` over the session's
end message; every egress cell of a connection admitted before `b`'s bucket that is sent at or after `a`'s is sent in a
phase from the count's start phase to its end phase plus its tail; the bits the device sent in the phases the count
counts are at most the count's egress bits; and the egress the wipe publishes, `egressMib` in its unit, when below the
saturated one, is at least the count's egress bits. -/
def SpanCounted {κ : Type} (d : MarkDevice κ) (c : StreamedWipe.Commitment) (s : StreamedWipe.Signed)
    (n : NetworkCheck.Span.Count) (egressMib : Int) : Prop :=
  ∃ a ∈ d.marks, ∃ b ∈ d.marks,
    (a.message = StreamedWipe.startMessage c ∧ Sha512.hash a.certificate = s.startMark ∧
      StreamedWipe.endMessage s = .ok b.message) ∧
    (∀ (k : κ) (j : ℕ), d.schedule.cells k ≤ d.cellCap → j < d.schedule.cells k → d.schedule.adm k < b.bucket →
      a.bucket ≤ d.schedule.bucket k j →
        n.startPhase ≤ d.schedule.bucket k j / d.phaseBuckets ∧
          d.schedule.bucket k j / d.phaseBuckets ≤ n.endPhase + n.tail) ∧
    sentOver n d ≤ countBits n ∧
    (egressMib < StreamedWipe.EGRESS_MAX_UNITS → (countBits n : Int) ≤ egressMib * StreamedWipe.EGRESS_UNIT_BITS)

/-- **A counted wipe's egress is the network certifier's count over the wipe's span.** For the network certifier `d` on
the wipe's link, whose stamp is its code (`NetTiming.DeviceIntact`): when the counted wipe's compiled check
(`WipeProtocol.checkCounted` at the deployed schemes) accepts and publishes `out`, the count's compiled check
(`NetworkCheck.Span.check`) accepts the span `cs` under the parameters `sp` and counts `n`, the session's signed
`start_mark` is the SHA-512 of the certificate of a mark the device made over the start message, and the two cases are
tied (`SpanTied`), then the wipe's egress is the count over its span (`SpanCounted`), or the count's run carries its
break. Open: the tie (`SpanTied`), what an accepted span means (`CountRan`), the device model's lines for a span
(`NetTiming.Guarantees.SpanCovers`), its capacity a phase (`SentWithinCapacity`) and its marking a message once
(`MarksOnce`). Nothing serves `checkCounted` until the tie exists, and the deployed `WipeProtocol.protocol` publishes
the saturated egress, which this doesn't read. -/
def WipeCountTied : Prop :=
  ∀ {κ : Type} (d : MarkDevice κ) (p : WipeProtocol.Params) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
    WipeProtocol.checkCounted SCHEMES p c e = .accept out prof →
    ∀ (sp : NetworkCheck.Span.Params) (cs : NetworkCheck.Span.Commitment) (ev : NetworkCheck.Span.Evidence)
      (n : NetworkCheck.Span.Count) (pr : NetworkCheck.Span.Profile),
      NetworkCheck.Span.check sp cs ev = .accept n pr →
      ∀ (s : StreamedWipe.Signed), e.signed = some s →
      (∃ a ∈ d.marks, a.message = StreamedWipe.startMessage c ∧ Sha512.hash a.certificate = s.startMark) →
      ∀ (broken : NetworkCheck.Span.Commitment → NetworkCheck.Span.Evidence → Prop),
      SpanTied sp cs c e → NetTiming.DeviceIntact d.stamp (NetTiming.markPhase d.phaseBuckets) →
      CountRan sp broken d → NetTiming.Guarantees.SpanCovers → SentWithinCapacity d → MarksOnce d →
        SpanCounted d c s n out.egressMib ∨ broken cs ev

end MemoryChallenger.Properties
