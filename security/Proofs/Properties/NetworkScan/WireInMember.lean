module

public import Proofs.Flock.SpotFlock.WireMapRewires

/-!
# A wire inside one member points to a later gate

A scratch module. In a key program every source gate comes before every row (`Spot.Bits.keyProg`: the constant, the
zero and every unit's source gates, then the rows), and lowering keeps that order. So a map that sends a member's
source gate to that member's gate on a returned output's bit sends it to a later gate, and is no `Prog.Rewires` map,
whose `le` asks `ω i ≤ i`.

* `inputs_le_sum`, `source_lt_row`: a source gate's position is below every row's.
* `not_le_of_wire_in_member`: no map that sends a source gate to a row of the same key program, both moved by one
  base, is at most the identity.
* `not_rewires_of_wire_in_member`: so over the members' programs side by side, a map that sends a member's source gate
  to that member's gate on a returned output's bit does not rewire them.
-/

@[expose] public section

namespace ScanWireInMember

open Claims.Sampling Properties.Spot Flock.Bridge.Members

theorem inputs_le_sum (kp : Spot.Bits.KeyProg) {a : ℕ} (h : kp.inputs ≤ a) :
    kp.inputs ≤ ((List.range a).map (Spot.Bits.gateSize kp)).sum := by
  induction a, h using Nat.le_induction with
  | base => exact (sum_gateSize_inputs kp kp.inputs le_rfl).ge
  | succ n _ ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    exact le_trans ih (Nat.le_add_right _ _)

/-- **A source gate's position is below every row's.** -/
theorem source_lt_row (kp : Spot.Bits.KeyProg) {g a : ℕ} (hg : g < kp.inputs) (ha : (kp.row? a).isSome) :
    Spot.Bits.posOf kp g < Spot.Bits.posOf kp a := by
  have hsz : a < kp.size := lt_size_of_isSome_row? ha
  have hge : kp.inputs ≤ a := by
    by_contra hlt
    rw [row?_of_lt_inputs (Nat.lt_of_not_le hlt)] at ha
    simp at ha
  have hoff : kp.inputs ≤ Spot.Bits.off kp a := by
    rw [Properties.Spot.SecurityProofs.off_of_le kp hsz.le]
    exact inputs_le_sum kp hge
  have h3 : Spot.Bits.gateSize kp a = 3 := by
    unfold Spot.Bits.gateSize
    exact if_pos ha
  rw [posOf_of_lt_inputs kp hg]
  unfold Spot.Bits.posOf
  omega

/-- **No map that sends a source gate to a row of the same key program is at most the identity**, both gates moved by
one base `b`. -/
theorem not_le_of_wire_in_member (kp : Spot.Bits.KeyProg) {g a b : ℕ} (hg : g < kp.inputs)
    (ha : (kp.row? a).isSome) {ω : ℕ → ℕ} (hω : ω (b + g) = b + Spot.Bits.posOf kp a) : ¬ ∀ i, ω i ≤ i := by
  intro h
  have h1 := h (b + g)
  have h2 := source_lt_row kp hg ha
  rw [posOf_of_lt_inputs kp hg] at h2
  omega

/-- **A map that wires a member's source gate to that member's own returned output does not rewire the members'
programs**: `Prog.Rewires.le` fails at the source gate. -/
theorem not_rewires_of_wire_in_member {ms : Spot.Bits.Members} {m g a : ℕ} (hm : m < ms.length)
    (hg : g < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs)
    (ha : a ∈ (Spot.Bits.keyProg ms[m].1 ms[m].2.size).outs) {P : Claims.Sampling.Partition} {ω : ℕ → ℕ}
    {keep : ℕ → Bool}
    (hω : ω (Spot.Bits.base ms m + g) =
      Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) a) :
    ¬ (bitsProgs ms).Rewires P ω keep := fun hW =>
  not_le_of_wire_in_member _ hg (Flock.Bridge.isSome_row?_of_mem_outs ha) hω hW.le

end ScanWireInMember
