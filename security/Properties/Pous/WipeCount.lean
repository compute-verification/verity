module

public import MemoryChallenger.WipeProtocol
public import Definitions.NetworkCertifier.Premises
public import Definitions.NetworkCertifier.Guarantees.Marks

/-!
# An accepted wipe's egress is the network certifier's count over the wipe's span

`MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`) publishes an egress `E` (`egress_mib`), and the survivor
bound it certifies is of the foreign content plus `E` (`MemoryChallenger.Properties.WipeSurvivorBound`). `WipeCountTied`
says what `E` is when a count of the network certifier over the wipe's span is accepted beside the wipe: at least the
count over the execution phases from the start mark's phase to the end mark's phase plus the device's tail, phases
that hold every cell the device sends from the start mark on out of what it held by the end mark. The statement only;
`Proofs.Properties.Pous.WipeCount` proves it from its antecedents.

## The marks

The network certifier on the wipe's link signs two marks for the wipe, each over a message it is handed, and stamps
each with the execution phase it handles it in (`Definitions.NetworkCertifier.Marks`): the start mark over the start
message, which is of the wipe's commitment, before the fill; and the end mark over the end message, which is of the
signed session, after the wipe closes. The memory challenger signs the SHA-512 of the start mark's certificate in the
session (`start_mark`), so the session names one start mark, and the end message is of the session's signature, so the
end mark is made after it. The count over the span names the two messages and the start certificate's SHA-512, and
holds the two phases in its commitment; it publishes its egress in bits and its tail.

## What is a parameter

The statement is over what the tree has, and takes the rest as parameters, each to be replaced by the code it names:

* `WipeMarks`: the signed session's `start_mark` and the two messages, functions of the session and its commitment.
* `count` and `broken`: the count's check, its accepted outcome read as the span it names (`SpanCount`, from its
  commitment and its public output), and the break its run carries.
* `MarkDevice`: the network certifier on the wipe's link, its schedule, the marks it made and the phase it stamped on
  each.

While `WipeMarks` is a parameter, the wipe's check reads none of it: the wipe's acceptance holds nothing of `M`, and the
statement follows from the verifier's other checks and the two antecedents alone. It says something of the wipe's own
run at the `M` the session's code fixes, where `start_mark` is under the memory challenger's signature.

## What the verifier checks

All of it is public or opened to it: the wipe's check accepts; the count's check accepts; the start certificate's
SHA-512 the count names is the session's signed `start_mark`; the start message it names is the start message of the
wipe's commitment; the end message it names is the end message of the signed session; and the egress the wipe
publishes is at least the count's, in the wipe's unit (`StreamedWipe.egressUnits`). That the end phase is not before
the start phase is the count's own check and is not needed here: a span that ends before it starts holds no phase, and
then no cell.

## What it rests on

* `NetTiming.DeviceIntact`, at the phase the device stamps on a mark: the device runs its code (`trusted-host`), so the
  stamp is `NetTiming.markPhase` of the bucket it handles the mark in.
* `CountRan`, an antecedent until the count's check is in the tree: what an accepted count means, that the span it names
  is of two marks the device made, or its run carries the break.
* `NetTiming.Guarantees.SpanCovers`, an antecedent until it is proved: the device model's lines for a span.

It takes no premise on a clock and none on where a byte came from: the marks' phases are the device's own, and the count
is of every cell the device sends in them.

## What it does not say

* It stops at the marks. That the fill began after the start mark was handled and that the end mark was handled after
  the wipe closed is the memory challenger's run (`MemoryChallenger.Properties.WipeSigned`, once the session signs
  `start_mark`) and the end message's being of the signature.
* That the count's egress bounds the bytes the device sends in its phases is the count's own guarantee, the schedule's
  capacity a phase times the phases.
* A link no network certifier relays has no count. The wipe's egress is then the saturated one
  (`StreamedWipe.EGRESS_MAX_UNITS`), at which its check certifies nothing (`WipeProtocol.gaps`): `certified` is false
  and `foreign_bits` is the whole memory. The statement's bound in bits is for an egress below the saturated one.
-/

@[expose] public section

namespace MemoryChallenger.Properties

open Flock

/-- **The wipe's marks**, the functions of the signed session and its commitment that are not in the tree yet: the
session's signed `start_mark` (the SHA-512 of the start mark's certificate), the start message (of the commitment)
and the end message (of the signed session, its signature included; none for a session with no message). -/
structure WipeMarks where
  startMark : StreamedWipe.Signed → ByteArray
  startMessage : StreamedWipe.Commitment → ByteArray
  endMessage : StreamedWipe.Signed → Except String ByteArray

/-- **What an accepted count over a span names**, read from its commitment and its public output: the SHA-512 of the
start mark's certificate, the start message and the end message; the start mark's phase and the end mark's; the tail,
in phases; and the schedule's egress capacity a phase, in bits. -/
structure SpanCount where
  startMark : ByteArray
  startMessage : ByteArray
  endMessage : ByteArray
  startPhase : ℕ
  endPhase : ℕ
  tail : ℕ
  capacityBits : ℕ

/-- **The count's egress**, in bits: the schedule's capacity a phase times the phases from the start phase to the end
phase plus the tail. -/
def SpanCount.egressBits (n : SpanCount) : ℕ := (n.endPhase + n.tail + 1 - n.startPhase) * n.capacityBits

/-- A mark the network certifier made: the message it signed, the bucket it handled it in, and the certificate it
returned. -/
structure Mark where
  message : ByteArray
  bucket : ℕ
  certificate : ByteArray

/-- **The network certifier on the wipe's link**: its schedule, the buckets of an execution phase (`T`), the most
egress cells a connection has (`E`), the phase the physical device stamps on a mark handled in a bucket, and the marks
it made. -/
structure MarkDevice (κ : Type) where
  schedule : NetTiming.CertifierSchedule κ
  phaseBuckets : ℕ
  cellCap : ℕ
  stamp : ℕ → ℕ
  marks : List Mark

/-- The span `n` is of the marks `a` and `b` of the device `d`: both are marks it made, `a` over the start message with
the certificate whose SHA-512 `n` names, `b` over the end message; the phases are the ones the device stamped on them;
and the tail is the device's (`NetTiming.spanTail` of its schedule). -/
def SpanCount.Of {κ : Type} (n : SpanCount) (d : MarkDevice κ) (a b : Mark) : Prop :=
  a ∈ d.marks ∧ b ∈ d.marks ∧ a.message = n.startMessage ∧ Sha512.hash a.certificate = n.startMark ∧
    b.message = n.endMessage ∧ n.startPhase = d.stamp a.bucket ∧ n.endPhase = d.stamp b.bucket ∧
    n.tail = NetTiming.spanTail d.schedule.D d.cellCap d.schedule.r d.phaseBuckets

/-- **What an accepted count means**, of the count's check `count`, the break `broken` its run carries and the device
`d`: every span it accepts is of two marks the device made (`SpanCount.Of`), or the run carries the break. It stands
for the count's own property until its check is in the tree. -/
def CountRan {κ P C Ev Pr : Type} (count : P → C → Ev → Core.Outcome SpanCount Pr) (broken : P → C → Ev → Prop)
    (d : MarkDevice κ) : Prop :=
  ∀ (pn : P) (cn : C) (en : Ev) (n : SpanCount) (profn : Pr),
    count pn cn en = .accept n profn → (∃ a b, n.Of d a b) ∨ broken pn cn en

/-- **The wipe's egress is the count over its span**: the device `d` made a start mark `a` over the start message of the
wipe's commitment `c`, with the certificate whose SHA-512 the session `s` signs, and an end mark `b` over the session's
end message; every egress cell of a connection admitted before `b`'s bucket that is sent at or after `a`'s is sent in a
phase from the span's start phase to its end phase plus its tail; and the egress the wipe publishes, `egressMib` in its
unit, when below the saturated count, is at least the count's egress in bits. -/
def SpanCounted {κ : Type} (M : WipeMarks) (d : MarkDevice κ) (c : StreamedWipe.Commitment) (s : StreamedWipe.Signed)
    (n : SpanCount) (egressMib : Int) : Prop :=
  ∃ a ∈ d.marks, ∃ b ∈ d.marks,
    (a.message = M.startMessage c ∧ Sha512.hash a.certificate = M.startMark s ∧ M.endMessage s = .ok b.message) ∧
    (∀ (k : κ) (j : ℕ), d.schedule.cells k ≤ d.cellCap → j < d.schedule.cells k → d.schedule.adm k < b.bucket →
      a.bucket ≤ d.schedule.bucket k j →
        n.startPhase ≤ d.schedule.bucket k j / d.phaseBuckets ∧
          d.schedule.bucket k j / d.phaseBuckets ≤ n.endPhase + n.tail) ∧
    (egressMib < StreamedWipe.EGRESS_MAX_UNITS → (n.egressBits : Int) ≤ egressMib * StreamedWipe.EGRESS_UNIT_BITS)

/-- **An accepted wipe's egress is the network certifier's count over the wipe's span.** For the wipe's marks `M`, a
count's check `count` with its break `broken`, and the network certifier `d` on the wipe's link, whose stamp is its code
(`NetTiming.DeviceIntact`): when the wipe's compiled check accepts and publishes `out`, then for every span `n` the
count accepts that names the session's signed `start_mark`, the start message of the wipe's commitment and the end
message of the signed session, with the published egress at least the count's in the wipe's unit, the wipe's egress is
the count over its span (`SpanCounted`), or the count's run carries its break. Two antecedents stand until their own
pieces are in the tree: what an accepted count means (`CountRan`) and the device model's lines for a span
(`NetTiming.Guarantees.SpanCovers`). -/
def WipeCountTied : Prop :=
  ∀ {κ P C Ev Pr : Type} (M : WipeMarks) (count : P → C → Ev → Core.Outcome SpanCount Pr)
    (broken : P → C → Ev → Prop) (d : MarkDevice κ) (p : WipeProtocol.Params) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
    WipeProtocol.protocol.check p c e = .accept out prof →
    NetTiming.DeviceIntact d.stamp (NetTiming.markPhase d.phaseBuckets) →
    ∀ (s : StreamedWipe.Signed) (pn : P) (cn : C) (en : Ev) (n : SpanCount) (profn : Pr),
      e.signed = some s → count pn cn en = .accept n profn →
      n.startMark = M.startMark s → n.startMessage = M.startMessage c → M.endMessage s = .ok n.endMessage →
      StreamedWipe.egressUnits n.egressBits ≤ out.egressMib →
      CountRan count broken d → NetTiming.Guarantees.SpanCovers →
        SpanCounted M d c s n out.egressMib ∨ broken pn cn en

/-! ## The witnesses' instances -/

/-- Marks that say nothing: an empty `start_mark` and empty messages. -/
def emptyMarks : WipeMarks := ⟨fun _ => .empty, fun _ => .empty, fun _ => .ok .empty⟩

/-- A mark over the empty message, handled in bucket 0, with an empty certificate. -/
def emptyMark : Mark := ⟨.empty, 0, .empty⟩

/-- A network certifier that made one mark (`emptyMark`), on a schedule of one cell a bucket with no latency and no
connection's cells, in phases of one bucket, its stamp its code's. -/
def markedDevice : MarkDevice Unit :=
  ⟨⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, [emptyMark]⟩

/-- The span from `markedDevice`'s mark to itself: one phase, no tail and no capacity. -/
def markedSpan : SpanCount := ⟨Sha512.hash emptyMark.certificate, .empty, .empty, 0, 0, 0, 0⟩

/-- A count that accepts `markedSpan` of everything. -/
def markedCount : Unit → Unit → Unit → Core.Outcome SpanCount Unit := fun _ _ _ => .accept markedSpan ()

/-- A network certifier that made no mark. -/
def idleDevice : MarkDevice Unit := ⟨⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, []⟩

end MemoryChallenger.Properties
