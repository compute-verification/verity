module

public import Proofs.Flock.SpotFlock.WireMapRewires

/-!
# A wire inside one member points to a later gate

A scratch module. In a key program every source gate comes before every row (`Spot.Bits.keyProg`: the constant, the
zero and every unit's source gates, then the rows), and lowering keeps that order. So a map that sends a member's
source gate to that member's gate on a returned output's bit sends it to a later gate, and is no `Prog.Rewires` map,
whose `le` asks `ω i ≤ i`.

* `inputs_le_sum_add`, `inputs_le_sum`, `source_lt_row`: a source gate's position is below every row's.
* `not_le_of_wire_in_member`: no map that sends a source gate to a row of the same key program, both moved by one
  base, is at most the identity.
* `not_rewires_of_wire_in_member`: so over the members' programs side by side, a map that sends a member's source gate
  to that member's gate on a returned output's bit does not rewire them.
-/

@[expose] public section

namespace ScanWireInMember

open Claims.Sampling Properties.Spot Flock.Bridge.Members

theorem inputs_le_sum_add (kp : Spot.Bits.KeyProg) :
    ∀ d : ℕ, kp.inputs ≤ ((List.range (kp.inputs + d)).map (Spot.Bits.gateSize kp)).sum
  | 0 => Nat.le_of_eq (sum_gateSize_inputs kp kp.inputs (Nat.le_refl _)).symm
  | d + 1 => by
    rw [← Nat.add_assoc, List.range_succ, List.map_append, List.sum_append]
    exact Nat.le_trans (inputs_le_sum_add kp d) (Nat.le_add_right _ _)

theorem inputs_le_sum (kp : Spot.Bits.KeyProg) {a : ℕ} (h : kp.inputs ≤ a) :
    kp.inputs ≤ ((List.range a).map (Spot.Bits.gateSize kp)).sum := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  exact inputs_le_sum_add kp d

/-- **A source gate's position is below every row's.** -/
theorem source_lt_row (kp : Spot.Bits.KeyProg) {g a : ℕ} (hg : g < kp.inputs) (ha : (kp.row? a).isSome) :
    Spot.Bits.posOf kp g < Spot.Bits.posOf kp a := by
  have hsz : a < kp.size := lt_size_of_isSome_row? ha
  have hge : kp.inputs ≤ a := by
    rcases Nat.lt_or_ge a kp.inputs with hlt | hge
    · rw [row?_of_lt_inputs hlt] at ha
      simp at ha
    · exact hge
  have hoff : kp.inputs ≤ Spot.Bits.off kp a := by
    rw [Properties.Spot.SecurityProofs.off_of_le kp (Nat.le_of_lt hsz)]
    exact inputs_le_sum kp hge
  have h3 : Spot.Bits.gateSize kp a = 3 := by
    unfold Spot.Bits.gateSize
    first
      | exact ite_eq_left ha
      | simp [ha]
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
