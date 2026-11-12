module

public import MemoryChallenger.WipeProtocol
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Definitions.NetworkCertifier.Premises
public import Definitions.NetworkCertifier.Guarantees.Marks

/-!
# An accepted wipe's egress is the network certifier's count over the wipe's span

`MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`) publishes an egress `E` (`egress_mib`), and the survivor
bound it certifies is of the foreign content plus `E` (`MemoryChallenger.Properties.WipeSurvivorBound`). `WipeCountTied`
says what `E` is when a count of the network certifier over the wipe's span is accepted beside the wipe, under the
device's parameters: at least the bits the device sent in the execution phases from the start mark's phase to the end
mark's phase plus the device's tail, phases that hold every cell the device sends from the start mark on out of what it
read by the end mark. The statement only; `Proofs.Properties.Pous.WipeCount` proves it from its antecedents.

## The marks

The network certifier on the wipe's link signs two marks for the wipe, each over a message it is handed, and stamps
each with the execution phase it handles it in (`Definitions.NetworkCertifier.Marks`): the start mark over the start
message, which is of the wipe's commitment, before the fill; and the end mark over the end message, which is of the
signed session, after the wipe closes. The memory challenger signs the SHA-512 of the start mark's certificate, the
bytes the device handed it, in the session (`start_mark`), and the end message is of the session's signature, so the
end mark is made after it. The count over the span names the two messages and the start certificate's SHA-512, and
holds the two phases in its commitment; it publishes its egress in bits and its tail.

## What is a parameter

The statement is over what the tree has, and takes the rest as parameters, each to be replaced by the code it names:

* `WipeMarks`: the signed session's `start_mark` and the two messages, functions of the session and its commitment.
* `count`, `params` and `broken`: the count's check, its accepted outcome read as the span it names (`SpanCount`, from
  its commitment and its public output); the device's parameters it runs under, all public (the registered hardware's
  key and link, the run's certifier session, and the schedule each phase's signed head names by digest); and the
  break its run carries.
* `MarkDevice`: the network certifier on the wipe's link, its key, session and link, the link's registered capacity,
  its schedule, the marks it made, the phase it stamped on each and the bits it sent in each phase.

While `WipeMarks` is a parameter, the wipe's check reads none of it: the wipe's acceptance holds nothing of `M`, and the
statement follows from the verifier's other checks and the antecedents alone. It says something of the wipe's own run
at the `M` the session's code fixes, where `start_mark` is under the memory challenger's signature.

## What the verifier checks

All of it is public or opened to it: the wipe's check accepts; the count's check accepts under the device's parameters;
the start certificate's SHA-512 the count names is the session's signed `start_mark`; the start message it names is the
start message of the wipe's commitment; the end message it names is the end message of the signed session; and the
egress the wipe publishes is at least the count's, in the wipe's unit (`StreamedWipe.egressUnits`). That the end phase
is not before the start phase is the count's own check and is not needed here: a span that ends before it starts holds
no phase, and then no cell.

## What it rests on

* `NetTiming.DeviceIntact`, at the phase the device stamps on a mark: the device runs its code (`trusted-host`), so the
  stamp is `NetTiming.markPhase` of the bucket it handles the mark in.
* The memory challenger's side: the session's `start_mark` is the SHA-512 of a mark's certificate the device made, since
  the challenger hashes the bytes the device handed it.
* `CountRan`, an antecedent until the count's check is in the tree: what an accepted count under the device's parameters
  means, that the span it names is of two marks the device made, or its run carries the break.
* `NetTiming.Guarantees.SpanCovers`, an antecedent until it is proved: the device model's lines for a span.
* `SentWithinCapacity`, an antecedent until it is stated in the device's model: the device sends at most its link's
  registered capacity a phase.

It takes no premise on a clock and none on where a byte came from: the marks' phases are the device's own, and the count
is of every cell the device sends in them.

## What it does not say

* It stops at the marks. That the fill began after the start mark was handled and that the end mark was handled after
  the wipe closed is the memory challenger's run (`MemoryChallenger.Properties.WipeSigned`, once the session signs
  `start_mark`) and the end message's being of the signature.
* That the span's start mark is the mark whose certificate the session signs. An accepted count gives a mark the device
  made over the start message at the span's start phase, not that the bytes the count names are that mark's
  certificate; the two are one mark when the device marks the start message once.
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
in phases; and the link's egress capacity a phase, in bits, from its parameters. -/
structure SpanCount where
  startMark : ByteArray
  startMessage : ByteArray
  endMessage : ByteArray
  startPhase : ℕ
  endPhase : ℕ
  tail : ℕ
  capacityBits : ℕ

/-- **The count's egress**, in bits: the link's capacity a phase times the phases from the start phase to the end phase
plus the tail. -/
def SpanCount.egressBits (n : SpanCount) : ℕ := (n.endPhase + n.tail + 1 - n.startPhase) * n.capacityBits

/-- A mark the network certifier made: the message it signed, the bucket it handled it in (the ticks it had done), and
the certificate it returned, the bytes of the certificate it signs under its key in its session on its link's marks,
egress, at the phase it stamps, over the SHA-512 of a tag, the mark's own fresh salt and the message. -/
structure Mark where
  message : ByteArray
  bucket : ℕ
  certificate : ByteArray

/-- **The network certifier on the wipe's link**: its key, session and link; the link's egress capacity a phase, in
bits, a public parameter fixed when the link is registered; its schedule, the buckets of an execution phase (`T`) and
the most egress cells a connection has (`E`); the phase the physical device stamps on a mark handled in a bucket; the
marks it made; and the bits it sent on the link in each phase. -/
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

/-- The span `n` is of the marks `a` and `b` of the device `d`: both are marks it made, `a` over the start message and
`b` over the end message; the phases are the ones the device stamped on them; the tail is the device's
(`NetTiming.spanTail` of its schedule); and the capacity a phase is its link's registered one. -/
def SpanCount.Of {κ : Type} (n : SpanCount) (d : MarkDevice κ) (a b : Mark) : Prop :=
  a ∈ d.marks ∧ b ∈ d.marks ∧ a.message = n.startMessage ∧ b.message = n.endMessage ∧
    n.startPhase = d.stamp a.bucket ∧ n.endPhase = d.stamp b.bucket ∧
    n.tail = NetTiming.spanTail d.schedule.D d.cellCap d.schedule.r d.phaseBuckets ∧ n.capacityBits = d.capacityBits

/-- The bits the device `d` sent in the span `n`'s phases, from its start phase to its end phase plus its tail. -/
def SpanCount.sentBits {κ : Type} (n : SpanCount) (d : MarkDevice κ) : ℕ :=
  ∑ i ∈ Finset.range (n.endPhase + n.tail + 1 - n.startPhase), d.sentBits (n.startPhase + i)

/-- **What an accepted count means**, of the count's check `count` run under the device's own parameters `params` (its
key, its certifier session, its link and its schedule), the break `broken` its run carries and the device `d`: every
span it accepts there is of two marks the device made (`SpanCount.Of`), or the run carries the break. It stands for the
count's own property until its check is in the tree. -/
def CountRan {κ P C Ev Pr : Type} (count : P → C → Ev → Core.Outcome SpanCount Pr) (params : P)
    (broken : C → Ev → Prop) (d : MarkDevice κ) : Prop :=
  ∀ (cn : C) (en : Ev) (n : SpanCount) (profn : Pr),
    count params cn en = .accept n profn → (∃ a b, n.Of d a b) ∨ broken cn en

/-- **The device sends at most its link's capacity a phase**, a line of the network certifier's model: in every
execution phase the bits it sends on the link, every cell of every connection, are at most the link's registered
capacity, so a count's bits over a span bound the bits it sent in the span's phases. -/
def SentWithinCapacity {κ : Type} (d : MarkDevice κ) : Prop :=
  ∀ φ : ℕ, d.sentBits φ ≤ d.capacityBits

/-- **The wipe's egress is the count over its span**: the session `s` signs the SHA-512 of the certificate of a mark the
device `d` made; the device made a start mark `a` over the start message of the wipe's commitment `c` and an end mark
`b` over the session's end message; every egress cell of a connection admitted before `b`'s bucket that is sent at or
after `a`'s is sent in a phase from the span's start phase to its end phase plus its tail; the bits the device sent in
those phases are at most the count's egress in bits; and the egress the wipe publishes, `egressMib` in its unit, when
below the saturated count, is at least the count's egress in bits. -/
def SpanCounted {κ : Type} (M : WipeMarks) (d : MarkDevice κ) (c : StreamedWipe.Commitment) (s : StreamedWipe.Signed)
    (n : SpanCount) (egressMib : Int) : Prop :=
  (∃ a₀ ∈ d.marks, Sha512.hash a₀.certificate = M.startMark s) ∧
  ∃ a ∈ d.marks, ∃ b ∈ d.marks,
    (a.message = M.startMessage c ∧ M.endMessage s = .ok b.message) ∧
    (∀ (k : κ) (j : ℕ), d.schedule.cells k ≤ d.cellCap → j < d.schedule.cells k → d.schedule.adm k < b.bucket →
      a.bucket ≤ d.schedule.bucket k j →
        n.startPhase ≤ d.schedule.bucket k j / d.phaseBuckets ∧
          d.schedule.bucket k j / d.phaseBuckets ≤ n.endPhase + n.tail) ∧
    n.sentBits d ≤ n.egressBits ∧
    (egressMib < StreamedWipe.EGRESS_MAX_UNITS → (n.egressBits : Int) ≤ egressMib * StreamedWipe.EGRESS_UNIT_BITS)

/-- **An accepted wipe's egress is the network certifier's count over the wipe's span.** For the wipe's marks `M`, a
count's check `count` with its break `broken`, the network certifier `d` on the wipe's link, whose stamp is its code
(`NetTiming.DeviceIntact`), and `d`'s parameters `params` (the registered hardware's key and link, the run's certifier
session, the schedule each phase's signed head names): when the wipe's compiled check accepts and publishes `out`, and
the session's signed `start_mark` is the SHA-512 of a mark's certificate the device made (the memory challenger hashes
the bytes the device handed it), then for every span `n` the count accepts under the device's parameters that names
the session's signed `start_mark`, the start message of the wipe's commitment and the end message of the signed
session, with the published egress at least the count's in the wipe's unit, the wipe's egress is the count over its
span (`SpanCounted`), or the count's run carries its break. Three antecedents stand until their own pieces are in the
tree: what an accepted count means (`CountRan`), the device model's lines for a span
(`NetTiming.Guarantees.SpanCovers`) and its capacity a phase (`SentWithinCapacity`). -/
def WipeCountTied : Prop :=
  ∀ {κ P C Ev Pr : Type} (M : WipeMarks) (count : P → C → Ev → Core.Outcome SpanCount Pr) (params : P)
    (broken : C → Ev → Prop) (d : MarkDevice κ) (p : WipeProtocol.Params) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
    WipeProtocol.protocol.check p c e = .accept out prof →
    NetTiming.DeviceIntact d.stamp (NetTiming.markPhase d.phaseBuckets) →
    ∀ (s : StreamedWipe.Signed) (cn : C) (en : Ev) (n : SpanCount) (profn : Pr),
      e.signed = some s → (∃ a ∈ d.marks, Sha512.hash a.certificate = M.startMark s) →
      count params cn en = .accept n profn →
      n.startMark = M.startMark s → n.startMessage = M.startMessage c → M.endMessage s = .ok n.endMessage →
      StreamedWipe.egressUnits n.egressBits ≤ out.egressMib →
      CountRan count params broken d → NetTiming.Guarantees.SpanCovers → SentWithinCapacity d →
        SpanCounted M d c s n out.egressMib ∨ broken cn en

/-! ## The witnesses' instances -/

/-- Marks that say nothing: an empty `start_mark` and empty messages. -/
def emptyMarks : WipeMarks := ⟨fun _ => .empty, fun _ => .empty, fun _ => .ok .empty⟩

/-- A mark over the empty message, handled in bucket 0, with an empty certificate. -/
def emptyMark : Mark := ⟨.empty, 0, .empty⟩

/-- A network certifier with an empty key and session on a link of no capacity that made one mark (`emptyMark`) and sent
nothing, on a schedule of one cell a bucket with no latency and no connection's cells, in phases of one bucket, its
stamp its code's. -/
def markedDevice : MarkDevice Unit :=
  ⟨.empty, .empty, "", 0, ⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, [emptyMark], fun _ => 0⟩

/-- The span from `markedDevice`'s mark to itself: one phase, no tail and no capacity. -/
def markedSpan : SpanCount := ⟨Sha512.hash emptyMark.certificate, .empty, .empty, 0, 0, 0, 0⟩

/-- A count that accepts `markedSpan` of everything. -/
def markedCount : Unit → Unit → Unit → Core.Outcome SpanCount Unit := fun _ _ _ => .accept markedSpan ()

/-- A network certifier that made no mark. -/
def idleDevice : MarkDevice Unit :=
  ⟨.empty, .empty, "", 0, ⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, [], fun _ => 0⟩

end MemoryChallenger.Properties
