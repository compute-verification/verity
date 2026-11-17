module

public import MemoryChallenger.WipeProtocol
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Definitions.Certifier.Premises
public import Definitions.Certifier.Guarantees.Marks

/-! An accepted wipe's egress is the network attester's count over the wipe's span (`WipeCountTied`), over parameters
for the marks, the count and its device. -/

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

/-- A mark the network attester made: the message it signed, the bucket it handled it in (the ticks it had done), and
the certificate it returned, the bytes of the certificate it signs under its key in its session on its link's marks,
egress, at the phase it stamps, over the SHA-512 of a tag, the mark's own fresh salt and the message. -/
structure Mark where
  message : ByteArray
  bucket : ℕ
  certificate : ByteArray

/-- **The network attester on the wipe's link**: its key, session and link; the link's egress capacity a phase, in
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

/-- **The device sends at most its link's capacity a phase**, a line of the network attester's model: in every
execution phase the bits it sends on the link, every cell of every connection, are at most the link's registered
capacity, so a count's bits over a span bound the bits it sent in the span's phases. -/
def SentWithinCapacity {κ : Type} (d : MarkDevice κ) : Prop :=
  ∀ φ : ℕ, d.sentBits φ ≤ d.capacityBits

/-- **The device marks a message at most once**, a line of the network attester's model: it refuses to mark a message
it has marked, so any two of its marks over one message are one mark, with one certificate and one phase. -/
def MarksOnce {κ : Type} (d : MarkDevice κ) : Prop :=
  ∀ a ∈ d.marks, ∀ b ∈ d.marks, a.message = b.message →
    a.certificate = b.certificate ∧ d.stamp a.bucket = d.stamp b.bucket

/-- **The wipe's egress is the count over its span**: the device `d` made a start mark `a` over the start message of the
wipe's commitment `c`, with the certificate whose SHA-512 the session `s` signs, and an end mark `b` over the session's
end message; every egress cell of a connection admitted before `b`'s bucket that is sent at or after `a`'s is sent in a
phase from the span's start phase to its end phase plus its tail; the bits the device sent in those phases are at most
the count's egress in bits; and the egress the wipe publishes, `egressMib` in its unit, when below the saturated count,
is at least the count's egress in bits. -/
def SpanCounted {κ : Type} (M : WipeMarks) (d : MarkDevice κ) (c : StreamedWipe.Commitment) (s : StreamedWipe.Signed)
    (n : SpanCount) (egressMib : Int) : Prop :=
  ∃ a ∈ d.marks, ∃ b ∈ d.marks,
    (a.message = M.startMessage c ∧ Sha512.hash a.certificate = M.startMark s ∧ M.endMessage s = .ok b.message) ∧
    (∀ (k : κ) (j : ℕ), d.schedule.cells k ≤ d.cellCap → j < d.schedule.cells k → d.schedule.adm k < b.bucket →
      a.bucket ≤ d.schedule.bucket k j →
        n.startPhase ≤ d.schedule.bucket k j / d.phaseBuckets ∧
          d.schedule.bucket k j / d.phaseBuckets ≤ n.endPhase + n.tail) ∧
    n.sentBits d ≤ n.egressBits ∧
    (egressMib < StreamedWipe.EGRESS_MAX_UNITS → (n.egressBits : Int) ≤ egressMib * StreamedWipe.EGRESS_UNIT_BITS)

/-- **An accepted wipe's egress is the network attester's count over the wipe's span.** For the wipe's marks `M`, a
count's check `count` with its break `broken`, the network attester `d` on the wipe's link, whose stamp is its code
(`NetTiming.DeviceIntact`), and `d`'s parameters `params` (the registered hardware's key and link, the run's certifier
session, the schedule each phase's signed head names): when the wipe's compiled check accepts and publishes `out`, and
the session's signed `start_mark` is the SHA-512 of the certificate of a mark the device made over the start message
(the memory challenger hands the device the start message and hashes the bytes it hands back), then for every span `n`
the count accepts under the device's parameters that names the session's signed `start_mark`, the start message of the
wipe's commitment and the end message of the signed session, with the published egress at least the count's in the
wipe's unit, the wipe's egress is the count over its span (`SpanCounted`), or the count's run carries its break. Four
antecedents stand until their own pieces are in the tree: what an accepted count means (`CountRan`), the device model's
lines for a span (`NetTiming.Guarantees.SpanCovers`), its capacity a phase (`SentWithinCapacity`) and its marking a
message once (`MarksOnce`). -/
def WipeCountTied : Prop :=
  ∀ {κ P C Ev Pr : Type} (M : WipeMarks) (count : P → C → Ev → Core.Outcome SpanCount Pr) (params : P)
    (broken : C → Ev → Prop) (d : MarkDevice κ) (p : WipeProtocol.Params) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
    WipeProtocol.protocol.check p c e = .accept out prof →
    NetTiming.DeviceIntact d.stamp (NetTiming.markPhase d.phaseBuckets) →
    ∀ (s : StreamedWipe.Signed) (cn : C) (en : Ev) (n : SpanCount) (profn : Pr),
      e.signed = some s → (∃ a ∈ d.marks, a.message = M.startMessage c ∧ Sha512.hash a.certificate = M.startMark s) →
      count params cn en = .accept n profn →
      n.startMark = M.startMark s → n.startMessage = M.startMessage c → M.endMessage s = .ok n.endMessage →
      StreamedWipe.egressUnits n.egressBits ≤ out.egressMib →
      CountRan count params broken d → NetTiming.Guarantees.SpanCovers → SentWithinCapacity d → MarksOnce d →
        SpanCounted M d c s n out.egressMib ∨ broken cn en

/-- Marks that say nothing: an empty `start_mark` and empty messages. -/
def emptyMarks : WipeMarks := ⟨fun _ => .empty, fun _ => .empty, fun _ => .ok .empty⟩

/-- A mark over the empty message, handled in bucket 0, with an empty certificate. -/
def emptyMark : Mark := ⟨.empty, 0, .empty⟩

/-- A network attester with an empty key and session on a link of no capacity that made one mark (`emptyMark`) and sent
nothing, on a schedule of one cell a bucket with no latency and no connection's cells, in phases of one bucket, its
stamp its code's. -/
def markedDevice : MarkDevice Unit :=
  ⟨.empty, .empty, "", 0, ⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, [emptyMark], fun _ => 0⟩

/-- The span from `markedDevice`'s mark to itself: one phase, no tail and no capacity. -/
def markedSpan : SpanCount := ⟨Sha512.hash emptyMark.certificate, .empty, .empty, 0, 0, 0, 0⟩

/-- A count that accepts `markedSpan` of everything. -/
def markedCount : Unit → Unit → Unit → Core.Outcome SpanCount Unit := fun _ _ _ => .accept markedSpan ()

/-- A network attester that made no mark. -/
def idleDevice : MarkDevice Unit :=
  ⟨.empty, .empty, "", 0, ⟨1, 0, fun _ => 0, fun _ => 0⟩, 1, 0, NetTiming.markPhase 1, [], fun _ => 0⟩

end MemoryChallenger.Properties
