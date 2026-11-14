module

public import Proofs.Flock.SpotFlock.NeededBitsOf

/-!
# A unit's proof block's size, from its circuit file's descriptor

A unit's proof block in a member's program (`Spot.Bits.blockOf` at the member's key program) holds the unit's lowered
gates and the position of each input it owns. `length_blockOf_le` bounds its length by the stacked unit rows' inputs
plus three gates a computed row, `(bitsRows c).nIn + 3 * (bitsRows c).nComp`, whatever the member's unit count `n`:
the bound a member's size condition can ask of one proof block.

* `length_unitGates_le`: a unit's rows are the `nComp` gates `2 + n·B + u·R + i` (`unitOf_some`), three lowered gates
  each.
* `length_insOf_le`: an input a unit owns is one its rows read (`exists_of_owner`), so the constant or the zero, which
  the constants fix, one of its own rows, which is no input, or one of its `B` source gates `2 + u·B + b`.
-/

@[expose] public section

namespace Flock.Bridge.BlockSize

open Claims.Sampling Properties.Spot Flock.Bridge.NeededBitsOf

/-- A list of naturals without repeats, all in `[a, a + w)`, has at most `w` entries. -/
theorem length_le_of_window {l : List ℕ} {a w : ℕ} (hnd : l.Nodup) (h : ∀ x ∈ l, a ≤ x ∧ x < a + w) :
    l.length ≤ w := by
  have hs : l.toFinset ⊆ Finset.Ico a (a + w) := fun x hx => Finset.mem_Ico.2 (h x (List.mem_toFinset.1 hx))
  have hc := Finset.card_le_card hs
  rw [List.toFinset_card_of_nodup hnd, Nat.card_Ico] at hc
  omega

/-- A `flatMap` whose pieces have at most `k` entries where `p` holds, and none elsewhere, has at most `k` entries for
each element `p` holds at. -/
theorem length_flatMap_le {α β : Type} (f : α → List β) (p : α → Bool) (k : ℕ) :
    ∀ l : List α, (∀ a ∈ l, (f a).length ≤ if p a = true then k else 0) →
      (l.flatMap f).length ≤ k * (l.filter p).length
  | [], _ => by simp
  | a :: l, h => by
    have ha := h a (List.mem_cons.2 (Or.inl rfl))
    have ih := length_flatMap_le f p k l fun b hb => h b (List.mem_cons.2 (Or.inr hb))
    rw [List.flatMap_cons, List.length_append]
    by_cases hp : p a = true
    · rw [List.filter_cons_of_pos hp, List.length_cons, Nat.mul_add, Nat.mul_one]
      rw [if_pos hp] at ha
      omega
    · rw [List.filter_cons_of_neg hp]
      rw [if_neg hp] at ha
      omega

variable (c : Flock.Circuit) (n : ℕ)

/-- **A unit's lowered gates**: three for each of its `nComp` rows. -/
theorem length_unitGates_le (u : ℕ) :
    (Spot.Bits.unitGates (Spot.Bits.keyProg c n) u).length ≤ 3 * (bitsRows c).nComp := by
  unfold Spot.Bits.unitGates
  refine (length_flatMap_le _ (fun g => Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u) 3 _
    fun g _ => ?_).trans (Nat.mul_le_mul_left 3 (length_le_of_window
      (a := 2 + n * (bitsRows c).nIn + u * (bitsRows c).nComp) (List.nodup_range.filter _) fun g hg => ?_))
  · by_cases hg : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u
    · obtain ⟨a, b, hab⟩ := row?_of_unitOf hg
      simp [hg, gateSize_of_some hab]
    · simp [hg]
  · obtain ⟨-, i, hi, rfl⟩ := unitOf_some c n (eq_of_beq (List.mem_filter.1 hg).2)
    exact ⟨by omega, by omega⟩

/-- **The inputs a unit owns** are among its `nIn` source gates `2 + u·B + b`. -/
theorem length_insOf_le (u : ℕ) :
    (Spot.Bits.insOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u).length ≤ (bitsRows c).nIn := by
  refine length_le_of_window (a := 2 + u * (bitsRows c).nIn) ?_ fun g hg => ?_
  · unfold Spot.Bits.insOf
    exact List.nodup_range.filter _
  · obtain ⟨hs, hr, hκ, ho⟩ := NeededBitsOf.mem_insOf.1 hg
    obtain ⟨g', hg', hw⟩ := exists_of_owner ho
    have hin := lt_inputs_of_row_none hs hr
    rw [keyProg_inputs] at hin
    rcases (args_of_unit c n hg' hw).2 with h0 | h0 | ⟨col, b, hb, h0⟩ | ⟨i, hi, h0⟩
    · rw [h0, consts_zero] at hκ
      cases hκ
    · rw [h0, consts_one] at hκ
      cases hκ
    · have hlt := keyIdx_lt hb
      exact ⟨by omega, by omega⟩
    · exact absurd hin (by omega)

/-- **A unit's proof block has at most its stacked rows' inputs and three gates a computed row**, whatever the member's
unit count. -/
theorem length_blockOf_le (u : ℕ) :
    (Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u).length ≤
      (bitsRows c).nIn + 3 * (bitsRows c).nComp := by
  unfold Spot.Bits.blockOf
  rw [List.length_append, List.length_map]
  have h1 := length_unitGates_le c n u
  have h2 := length_insOf_le c n u
  omega

end Flock.Bridge.BlockSize
