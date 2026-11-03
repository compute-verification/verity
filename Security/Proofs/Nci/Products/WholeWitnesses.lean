import Proofs.Nci.Products.Whole
import Proofs.Nci.Products.NoCellWitnesses
import Definitions.Nci.Guarantees.Products.WholeWitnesses
import Mathlib.Tactic.NormNum

/-!
# Products made whole through chains: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Products/WholeWitnesses.lean`, at `Pair`'s product alone as a
whole-making system (`Security/Definitions/Nci/Products/WholeInstances.lean`). The circuit's facts are decided over
`Pair`'s blocks (`gates_cases`): P-share at `1/4` as `4 Σ |had A| |had B| ≤ |had values|²` in `ℕ`, P-width extended
with every had value paid at its width, and its failure at gates 4 and 5 with the operands registered. P-match holds
with no value made from an output entry. The whole-making factor is 0 at `A = 0`, so the nontrivial bounds are the
no-cell ones (`Pair.not_noCellCap`, `Pair.not_noCellBound_le`). The `example`s at the end apply each guarantee to its
lemmas, so the lemmas are the premises as the guarantee states them.
-/

namespace Nci.Products.SecurityProofs

open Finset
open Nci.Formalism Nci.Products
open scoped ENNReal

namespace Pair

theorem values_fresh : Pair.freshSystem.Values := by unfold System.Values; decide

theorem values_registered : Pair.registeredSystem.Values := by unfold System.Values; decide

/-- **P-share at `1/4`** with the operands fresh, at `(c, k) = (1, 3)`. -/
theorem shared_fresh : Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) := by
  rintro I hI j F -
  have h : 4 * (∑ p ∈ F, ((Pair.freshSystem.prod p).hadA I).card * ((Pair.freshSystem.prod p).hadB I).card) ≤
      (Pair.freshSystem.hadValues F I).card ^ 2 := by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> revert F <;> decide
  have h' : (4 : ℝ) * ((∑ p ∈ F, ((Pair.freshSystem.prod p).hadA I).card *
      ((Pair.freshSystem.prod p).hadB I).card : ℕ) : ℝ) ≤ ((Pair.freshSystem.hadValues F I).card : ℝ) ^ 2 := by
    exact_mod_cast h
  dsimp only
  linarith

/-- **P-share at `1/4`** with the operands registered, at `(c, k) = (2, 1)`. -/
theorem shared_registered : Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) := by
  rintro I hI j F -
  have h : 4 * (∑ p ∈ F, ((Pair.registeredSystem.prod p).hadA I).card *
      ((Pair.registeredSystem.prod p).hadB I).card) ≤ (Pair.registeredSystem.hadValues F I).card ^ 2 := by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> revert F <;> decide
  have h' : (4 : ℝ) * ((∑ p ∈ F, ((Pair.registeredSystem.prod p).hadA I).card *
      ((Pair.registeredSystem.prod p).hadB I).card : ℕ) : ℝ) ≤
      ((Pair.registeredSystem.hadValues F I).card : ℝ) ^ 2 := by
    exact_mod_cast h
  dsimp only
  linarith

/-- **P-width extended** with the operands fresh: a block reads every operand bit it has, beyond `θ = ∅`. -/
theorem widthExtended_fresh : Pair.freshSystem.WidthExtended 1 3 1 := fun I hI => by
  rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide

/-- **P-width extended fails** with the operands registered: the block of gates 4 and 5 has `a₀` free. -/
theorem not_widthExtended_registered : ¬ Pair.registeredSystem.WidthExtended 2 1 1 := fun h =>
  absurd (h {4, 5} (by decide)) (by decide)

/-- **P-match** with no value made from an output entry. -/
theorem matched (θ : Finset (Fin Pair.C.N)) (hθ : ∀ g ∈ θ, Pair.C.isInput g = true) :
    (Pair.system θ hθ).Matched (fun _ _ => 0) :=
  ⟨fun _ _ _ _ hx => absurd hx (notMem_empty _), fun x _ => by
    rw [filter_false_of_mem fun v _ => notMem_empty x, card_empty, Nat.cast_zero]⟩

theorem wholeFactor_zero (μ : Unit → ℝ) : wholeFactor (fun _ _ => (0 : ℝ)) μ () = 0 := by simp [wholeFactor]

/-- `μ = 3/16` is a post-fixed point at `(k, b) = (3, 1)`: `(1/4) 3 / 2² = 3/16`. -/
theorem postFixed_fresh :
    Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16) := fun _ => by
  show (1 / 4 : ℝ) * ((3 : ℕ) : ℝ) / (((2 : ℕ) : ℝ) ^ 2 * ((1 : ℕ) : ℝ)) *
    (1 + wholeFactor (fun _ _ => (0 : ℝ)) (fun _ => 3 / 16) ()) ^ 2 ≤ 3 / 16
  rw [wholeFactor_zero]
  norm_num

/-- `μ = 1/16` is a post-fixed point at `(k, b) = (1, 1)`: `(1/4) 1 / 2² = 1/16`. -/
theorem postFixed_registered :
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) := fun _ => by
  show (1 / 4 : ℝ) * ((1 : ℕ) : ℝ) / (((2 : ℕ) : ℝ) ^ 2 * ((1 : ℕ) : ℝ)) *
    (1 + wholeFactor (fun _ _ => (0 : ℝ)) (fun _ => 1 / 16) ()) ^ 2 ≤ 1 / 16
  rw [wholeFactor_zero]
  norm_num

theorem not_classBound :
    ¬ ((∑ p ∈ univ.filter (fun p => Pair.registeredSystem.cls p = ()),
        ((Pair.registeredSystem.prod p).wholeOut {4, 5}).card : ℕ) : ℝ) ≤ 1 / 16 * (1 : ℕ) / (1 : ℕ) := by
  rw [show (∑ p ∈ univ.filter (fun p => Pair.registeredSystem.cls p = ()),
    ((Pair.registeredSystem.prod p).wholeOut {4, 5}).card) = 1 by decide]
  norm_num

theorem not_wholeCap :
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (1 : ℕ) * (((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 3 *
          √(((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 3) /
          √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (1 : ℕ)) +
        ((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) *
          √(((1 : ℕ) : ℝ) * (1 * 1 : ℕ)) := by
  simp only [wholeFactor_zero, add_zero, mul_one]
  exact not_noCellCap

theorem not_wholeBound_le :
    ¬ (univ : Finset Unit).sup (fun _ => wholeBound Pair.registered 2 1 2 1 1 1
        (wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ())) ≤ partitionSlowdown Pair.registered 1 2 := by
  rw [univ_unique, sup_singleton, wholeFactor_zero]
  have h : wholeBound Pair.registered 2 1 2 1 1 1 0 = noCellBound Pair.registered 1 2 1 2 1 1 1 1 := by
    unfold wholeBound noCellBound
    rw [add_zero, mul_one]
  rw [h]
  exact not_noCellBound_le

end Pair

theorem ClassBound_honest : Guarantees.ClassBound_honest :=
  ⟨fun _ => Pair.algorithm_fresh, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh, Pair.matched _ _,
    Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl, Pair.postFixed_fresh, by decide⟩

theorem ClassBound_nontrivial : Guarantees.ClassBound_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered, Pair.matched _ _, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_registered, by decide, Pair.not_classBound⟩

theorem WholeCap_honest : Guarantees.WholeCap_honest :=
  ⟨fun _ => Pair.algorithm_fresh, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh, Pair.matched _ _,
    Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh⟩

theorem WholeCap_nontrivial : Guarantees.WholeCap_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered, Pair.matched _ _, Pair.received_registered 2 1, Pair.isCK_P1, Nat.one_pos,
    fun _ => by norm_num, fun _ _ => le_rfl, Pair.postFixed_registered, Pair.not_wholeCap⟩

theorem WholeBound_honest : Guarantees.WholeBound_honest :=
  ⟨fun _ => Pair.algorithm_fresh, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh, Pair.matched _ _,
    Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh⟩

theorem WholeBound_nontrivial : Guarantees.WholeBound_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered, Pair.matched _ _, Pair.received_registered 2 1, Nat.one_pos,
    fun _ => by norm_num, fun _ _ => le_rfl, Pair.postFixed_registered,
    fun h => Pair.not_wholeBound_le (h 1 Pair.P1 Pair.isCK_P1)⟩

example : ((∑ p ∈ univ.filter (fun p => Pair.freshSystem.cls p = ()),
    ((Pair.freshSystem.prod p).wholeOut {4}).card : ℕ) : ℝ) ≤ 3 / 16 * (3 : ℕ) / (1 : ℕ) := by
  obtain ⟨hS, hV, hsh, hW, hM, hb, hcs, hA, hμ, hI⟩ := ClassBound_honest
  exact ClassBound _ _ _ Pair.freshSystem 1 3 1 _ _ _ {4} () hS hV hsh hW hM hb hcs hA hμ hI

example (h : Pair.registeredSystem.WidthExtended 2 1 1) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hb, hcs, hA, hμ, hI, hn⟩ := ClassBound_nontrivial
  exact hn (ClassBound _ _ _ Pair.registeredSystem 2 1 1 _ _ _ {4, 5} () hS hV hsh h hM hb hcs hA hμ hI)

example : ((1 * 2 * 1 : ℕ) : ℝ) ≤
    (2 : ℕ) * (((3 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) ()) / 3 *
        √(((3 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) ()) / 3) /
        √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (3 : ℕ)) +
      ((3 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) ()) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) *
        √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ)) := by
  obtain ⟨hS, hV, hsh, hW, hM, hR, hP, hb, hcs, hA, hμ⟩ := WholeCap_honest
  exact WholeCap _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ () 2 Pair.P hS hV hsh hW hM hR hP hb hb hcs hA hμ

example (h : Pair.registeredSystem.WidthExtended 2 1 1) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hR, hP, hb, hcs, hA, hμ, hn⟩ := WholeCap_nontrivial
  exact hn (WholeCap _ _ _ Pair.registeredSystem 2 1 1 1 _ _ _ () 1 Pair.P1 hS hV hsh h hM hR hP hb hb hcs hA hμ)

example : (univ : Finset Unit).sup (fun _ => wholeBound Pair.fresh 2 1 1 3 1 1
    (wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) ())) ≤ partitionSlowdown Pair.fresh 2 1 := by
  obtain ⟨hS, hV, hsh, hW, hM, hR, hP, hb, hcs, hA, hμ⟩ := WholeBound_honest
  exact WholeBound _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ univ hS hV hsh hW hM (fun _ _ => hR) hb hb hcs hA hμ 2 Pair.P hP

example (h : Pair.registeredSystem.WidthExtended 2 1 1) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hR, hb, hcs, hA, hμ, hn⟩ := WholeBound_nontrivial
  exact hn (WholeBound _ _ _ Pair.registeredSystem 2 1 1 1 _ _ _ univ hS hV hsh h hM (fun _ _ => hR) hb hb hcs hA hμ)

end Nci.Products.SecurityProofs
