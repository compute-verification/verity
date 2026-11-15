module

public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordGrid
public import Proofs.Properties.Nci.RecordTally
public import Mathlib.Tactic

/-!
# A block's gates

A block is a slot's units, and its size is the gates those units compute.

* `inSlot_whole`, `slot_units`: where every grid of a template is whole periods of `q · S` units and the grids
  together are the template's units, a slot holds the template's units over `S`; with the grids apart, those are
  the units `slotOf` puts in the slot.
* `block_size`: where the computed gates of a circuit lie in ranges that are apart, range `a` the gates
  `[off a, off a + size a)`, all of them computed and all of block `slot a`, a block's size is its ranges' sizes
  together: `at_` carries the block's gates one to one onto its ranges.
* `slot_sum`: over units indexed by a template and a unit of it, the gates of the units in a slot are, template by
  template, the template's units in the slot times its gates a unit.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Grid fieldsOf outerOf)
open NciCheck.Record (inSlot)
open NciCheck.Record.SecurityProofs (slotOf OneGrid)
open Spot.Bits (bitsOf)

/-! ## A slot's units -/

/-- A grid of whole periods, all of it among the first `units` units, has its units over `S` in a slot. -/
private theorem term_whole (S : ℕ) (g : Grid) (units s : ℕ) (hs : s < S) (hq : 0 < g.q) (hper : g.q * S ∣ g.units)
    (hin : g.start + g.units ≤ units) :
    NciCheck.Record.below S g.q (min units (g.start + g.units) - g.start) s -
        NciCheck.Record.below S g.q (max 0 g.start - g.start) s = g.units / S := by
  obtain ⟨m, hm⟩ := hper
  have hS : 0 < S := Nat.lt_of_le_of_lt (Nat.zero_le s) hs
  have hP : 0 < g.q * S := Nat.mul_pos hq hS
  have h1 : min units (g.start + g.units) - g.start = g.q * S * m := by omega
  have h2 : max 0 g.start - g.start = 0 := by omega
  have h3 : g.q * S * m / S = m * g.q := by
    rw [Nat.mul_comm g.q S, Nat.mul_assoc, Nat.mul_div_cancel_left _ hS, Nat.mul_comm]
  rw [h1, h2, hm, h3]
  unfold NciCheck.Record.below
  rw [Nat.mul_div_cancel_left _ hP, Nat.mul_mod_right, Nat.zero_div, Nat.zero_mod, Nat.zero_mul, Nat.zero_sub,
    Nat.min_zero, Nat.add_zero, Nat.add_zero, Nat.sub_zero]

/-- Grids of whole periods, all among the first `units` units, have their units over `S` in a slot. -/
private theorem inSlot_sum (S : ℕ) (units s : ℕ) (hs : s < S) : ∀ gs : List Grid, (∀ g ∈ gs, 0 < g.q) →
    (∀ g ∈ gs, g.q * S ∣ g.units) → (∀ g ∈ gs, g.start + g.units ≤ units) →
    inSlot S gs 0 units s = (gs.map (·.units)).sum / S := by
  intro gs
  induction gs with
  | nil => intro _ _ _; exact (Nat.zero_div S).symm
  | cons g gs ih =>
    intro hq hper hin
    have hg := term_whole S g units s hs (hq g List.mem_cons_self) (hper g List.mem_cons_self)
      (hin g List.mem_cons_self)
    have ih' := ih (fun x hx => hq x (List.mem_cons_of_mem g hx)) (fun x hx => hper x (List.mem_cons_of_mem g hx))
      (fun x hx => hin x (List.mem_cons_of_mem g hx))
    obtain ⟨m, hm⟩ := hper g List.mem_cons_self
    have hS : 0 < S := Nat.lt_of_le_of_lt (Nat.zero_le s) hs
    have hu : g.units = S * (g.q * m) := by rw [hm, Nat.mul_comm g.q S, Nat.mul_assoc]
    have e : inSlot S (g :: gs) 0 units s =
        (NciCheck.Record.below S g.q (min units (g.start + g.units) - g.start) s -
            NciCheck.Record.below S g.q (max 0 g.start - g.start) s) + inSlot S gs 0 units s := by
      unfold inSlot
      rw [List.map_cons, List.sum_cons]
    rw [e, hg, ih', List.map_cons, List.sum_cons]
    show g.units / S + (gs.map (·.units)).sum / S = (g.units + (gs.map (·.units)).sum) / S
    rw [hu, Nat.mul_add_div hS, Nat.mul_div_cancel_left _ hS]

/-- Where a template's grids lie in its units, hold them all and are whole periods, each slot has the template's
units over the slots. -/
theorem inSlot_whole (S : ℕ) (gs : List Grid) (units s : ℕ) (hs : s < S) (hq : ∀ g ∈ gs, 0 < g.q)
    (hper : ∀ g ∈ gs, g.q * S ∣ g.units) (hin : ∀ g ∈ gs, g.start + g.units ≤ units)
    (hsum : (gs.map (·.units)).sum = units) : inSlot S gs 0 units s = units / S := by
  rw [inSlot_sum S units s hs gs hq hper hin, hsum]

/-- The same, as a count of the units whose slot is `s`. -/
theorem slot_units (S : ℕ) (gs : List Grid) (units s : ℕ) (hs : s < S) (hg : OneGrid gs)
    (hq : ∀ g ∈ gs, 0 < g.q) (hper : ∀ g ∈ gs, g.q * S ∣ g.units) (hin : ∀ g ∈ gs, g.start + g.units ≤ units)
    (hsum : (gs.map (·.units)).sum = units) :
    ((List.range units).filter fun u => slotOf S gs u == some s).length = units / S := by
  have h := NciCheck.Record.SecurityProofs.inSlot_count S gs 0 units s hg hq hs
  rw [Nat.sub_zero, ← List.range_eq_range'] at h
  rw [← h]
  exact inSlot_whole S gs units s hs hq hper hin hsum

/-! ## A block's size -/

/-- A block's gates, counted by its units: where every computed gate lies in one unit of a family whose gate ranges
are apart, its block that unit's slot, and every gate of a unit is a computed gate, a block has the gates of the
units of its slot. -/
theorem block_size (D : Verity.Circuit) {n : ℕ} (U : Verity.Partition D n) (at_ : Fin D.N → ℕ)
    (hinj : Function.Injective at_) {ι : Type} [Fintype ι] (off size slot : ι → ℕ)
    (hapart : ∀ a b, a ≠ b → off a + size a ≤ off b ∨ off b + size b ≤ off a)
    (hin : ∀ x, D.isInput x = false →
      ∃ a, off a ≤ at_ x ∧ at_ x < off a + size a ∧ (U.unit x).map Fin.val = some (slot a))
    (hall : ∀ a g, off a ≤ g → g < off a + size a → ∃ x, at_ x = g ∧ D.isInput x = false) (u : Fin n) :
    D.size (U.gates u) = ∑ a ∈ Finset.univ.filter (fun a => slot a = u.val), size a := by
  classical
  have himg : (U.gates u).image at_ =
      (Finset.univ.filter (fun a => slot a = u.val)).biUnion fun a => Finset.Ico (off a) (off a + size a) := by
    ext g
    simp only [Finset.mem_image, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Ico,
      Verity.Partition.gates]
    constructor
    · rintro ⟨x, hx, rfl⟩
      have hni : D.isInput x = false := by
        cases hb : D.isInput x with
        | false => rfl
        | true =>
          have := (U.input_iff x).2 hb
          rw [hx] at this
          exact absurd this (by simp)
      obtain ⟨a, h1, h2, h3⟩ := hin x hni
      rw [hx] at h3
      exact ⟨a, (Option.some.inj h3).symm, h1, h2⟩
    · rintro ⟨a, ha, h1, h2⟩
      obtain ⟨x, rfl, hx⟩ := hall a g h1 h2
      obtain ⟨b, h1', h2', h3⟩ := hin x hx
      have hab : a = b := by
        by_contra hne
        rcases hapart a b hne with h | h <;> omega
      subst hab
      refine ⟨x, ?_, rfl⟩
      cases hu : U.unit x with
      | none => rw [hu] at h3; exact absurd h3 (by simp)
      | some v =>
        rw [hu] at h3
        have hv : v.val = u.val := (Option.some.inj h3).trans ha
        rw [Fin.ext hv]
  have hdisj : ((Finset.univ.filter (fun a => slot a = u.val) : Finset ι) : Set ι).PairwiseDisjoint
      fun a => Finset.Ico (off a) (off a + size a) := by
    intro a _ b _ hne
    refine Finset.disjoint_left.2 fun g hga hgb => ?_
    rw [Finset.mem_Ico] at hga hgb
    rcases hapart a b hne with h | h <;> omega
  calc D.size (U.gates u) = (U.gates u).card := rfl
    _ = ((U.gates u).image at_).card := (Finset.card_image_of_injective _ hinj).symm
    _ = ∑ a ∈ Finset.univ.filter (fun a => slot a = u.val), (Finset.Ico (off a) (off a + size a)).card := by
      rw [himg, Finset.card_biUnion hdisj]
    _ = ∑ a ∈ Finset.univ.filter (fun a => slot a = u.val), size a := by
      refine Finset.sum_congr rfl fun a _ => ?_
      rw [Nat.card_Ico, Nat.add_sub_cancel_left]

/-- The units of a slot, template by template: where template `i` has `c i` units in slot `u`, the gates of the
slot's units are each template's `c i` times its Definition's gates. -/
theorem slot_sum {m : ℕ} (units G c : Fin m → ℕ) (slot : (i : Fin m) → Fin (units i) → ℕ) (u : ℕ)
    (hc : ∀ i, (Finset.univ.filter fun v : Fin (units i) => slot i v = u).card = c i) :
    ∑ a ∈ (Finset.univ : Finset ((i : Fin m) × Fin (units i))).filter (fun a => slot a.1 a.2 = u), G a.1 =
      ∑ i, c i * G i := by
  rw [Finset.sum_filter, ← Finset.univ_sigma_univ, Finset.sum_sigma]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_filter]
  show ∑ _v ∈ Finset.univ.filter (fun v : Fin (units i) => slot i v = u), G i = c i * G i
  rw [Finset.sum_const, smul_eq_mul, hc i]

end NciCheck.Properties.Record.SecurityProofs
