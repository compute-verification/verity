import Proofs.Nci.Products.Sum
import Proofs.Nci.Products.WholeWitnesses
import Definitions.Nci.Guarantees.Products.SumWitnesses
import Mathlib.Tactic.NormNum

/-!
# Products made whole through chains, several in a block: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Products/SumWitnesses.lean`, at `Pair`'s product alone as a
whole-making system (`Security/Definitions/Nci/Products/WholeInstances.lean`). The class bound's premises, the cells and
the received partial sums are the whole-making lemmas' (`Security/Proofs/Nci/Products/WholeWitnesses.lean`); P-share on
the one product is P-share at its class (`System.sharedOn_of_shared`), and a product is apart from itself. At `b = 2`,
`μ = 1/32` is a post-fixed point, and P-width extended fails at gates 4 and 5 as at `b = 1`. Each bound is compared with
its price in `ℝ` or `ℝ≥0∞`, with `√(1/4) = 1/2`. The `example`s at the end apply each guarantee to its lemmas, so the
lemmas are the premises as the guarantee states them.
-/

namespace Nci.Products.SecurityProofs

open Finset
open Nci.Formalism Nci.Products
open scoped ENNReal

namespace Pair

/-- **Apart**: one product is apart from itself. -/
theorem apart (θ : Finset (Fin Pair.C.N)) (hθ : ∀ g ∈ θ, Pair.C.isInput g = true) :
    (Pair.system θ hθ).Apart univ :=
  fun p _ q _ h => absurd (Subsingleton.elim p q) h

/-- **P-width extended fails** with the operands registered at `b = 2`, as at `b = 1`. -/
theorem not_widthExtended_registered_two : ¬ Pair.registeredSystem.WidthExtended 2 1 2 := fun h =>
  absurd (h {4, 5} (by decide)) (by decide)

/-- `μ = 1/32` is a post-fixed point at `(k, b) = (1, 2)`: `(1/4) 1 / (2² 2) = 1/32`. -/
theorem postFixed_registered_two :
    Pair.registeredSystem.PostFixed 1 2 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 32) := fun _ => by
  show (1 / 4 : ℝ) * ((1 : ℕ) : ℝ) / (((2 : ℕ) : ℝ) ^ 2 * ((2 : ℕ) : ℝ)) *
    (1 + wholeFactor (fun _ _ => (0 : ℝ)) (fun _ => 1 / 32) ()) ^ 2 ≤ 1 / 32
  rw [wholeFactor_zero]
  norm_num

theorem sqrt_quarter : √(1 / 4 : ℝ) = 1 / 2 := by
  rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

theorem sum_macs : (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
    Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q) = 2 := by decide

theorem sum_outputs :
    (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q) = 1 := by decide

theorem sum_wholeFactor (μ : Unit → ℝ) :
    ∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls, wholeFactor (fun _ _ => (0 : ℝ)) μ j = 0 := by
  simp [wholeFactor]

theorem not_sumCap :
    ¬ ((∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
          Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q : ℕ) : ℝ) ≤
      (1 : ℕ) * (√(1 / 4) * (((1 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls,
          wholeFactor (fun _ _ => 0) (fun _ => 1 / 32) j) / (2 : ℕ)) * √(((1 : ℕ) : ℝ) / (1 : ℕ)) +
          (1 : ℕ) * (1 : ℕ)) +
        √(1 / 4) * (((1 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls,
          wholeFactor (fun _ _ => 0) (fun _ => 1 / 32) j) / (2 : ℕ)) *
          √(((1 : ℕ) : ℝ) * (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q : ℕ)) := by
  rw [sum_macs, sum_outputs, sum_wholeFactor, sqrt_quarter]
  norm_num

/-- Two MACs against a block of `Γ` and `B > 0` with `Γ + B < 2` need more than one block. -/
theorem one_lt_chainBlocks_two {Γ B : ℝ} (hΓ : 0 ≤ Γ) (hB : 0 < B) (h : Γ + B < 2) : 1 < chainBlocks 2 Γ B := by
  unfold chainBlocks
  have hR0 : 0 ≤ √(B ^ 2 + 4 * Γ * 2) := Real.sqrt_nonneg _
  have hR : √(B ^ 2 + 4 * Γ * 2) < 4 - B := by
    rw [Real.sqrt_lt' (by linarith)]
    nlinarith
  have h1 : 1 < 2 * 2 / (√(B ^ 2 + 4 * Γ * 2) + B) := by
    rw [lt_div_iff₀ (by linarith)]
    linarith
  exact one_lt_pow₀ h1 two_ne_zero

theorem not_sumBound_le :
    ¬ sumBound Pair.registered
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
          Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q)
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q) 1 2 1 2 1 (1 / 4)
        (∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 1 / 32) j) ≤
      partitionSlowdown Pair.registered 1 2 := by
  rw [sum_macs, sum_outputs, sum_wholeFactor]
  have hcb : 1 < chainBlocks ((2 : ℕ) : ℝ)
      (√(1 / 4) * (((1 : ℕ) : ℝ) * (1 + 0) / (2 : ℕ)) * √(((1 : ℕ) : ℝ) / (1 : ℕ)) + (1 : ℕ) * (1 : ℕ))
      (√(1 / 4) * (((1 : ℕ) : ℝ) * (1 + 0) / (2 : ℕ)) * √((1 : ℕ) : ℝ)) := by
    rw [sqrt_quarter]
    norm_num
    exact one_lt_chainBlocks_two (by norm_num) (by norm_num) (by norm_num)
  unfold sumBound partitionSlowdown
  rw [computed_card, ENNReal.mul_div_cancel_right (by norm_num) (by norm_num),
    ENNReal.div_self (by norm_num) (by norm_num), not_le, ENNReal.one_lt_ofReal]
  exact hcb

end Pair

theorem SumCap_honest : Guarantees.SumCap_honest :=
  ⟨fun _ => Pair.algorithm_fresh.toWiring, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh,
    Pair.matched _ _, Pair.freshSystem.sharedOn_of_shared Pair.shared_fresh (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh⟩

theorem SumCap_nontrivial : Guarantees.SumCap_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered.toWiring, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered_two, Pair.matched _ _,
    Pair.registeredSystem.sharedOn_of_shared Pair.shared_registered (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_registered, Pair.received_registered 2 1, Pair.isCK_P1, Nat.two_pos, Nat.one_pos, fun _ => by norm_num,
    fun _ _ => le_rfl, Pair.postFixed_registered_two, Pair.not_sumCap⟩

theorem SumBound_honest : Guarantees.SumBound_honest :=
  ⟨fun _ => Pair.algorithm_fresh.toWiring, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh,
    Pair.matched _ _, Pair.freshSystem.sharedOn_of_shared Pair.shared_fresh (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh⟩

theorem SumBound_nontrivial : Guarantees.SumBound_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered.toWiring, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered_two, Pair.matched _ _,
    Pair.registeredSystem.sharedOn_of_shared Pair.shared_registered (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_registered, Pair.received_registered 2 1, Nat.two_pos, Nat.one_pos, fun _ => by norm_num,
    fun _ _ => le_rfl, Pair.postFixed_registered_two, fun h => Pair.not_sumBound_le (h 1 Pair.P1 Pair.isCK_P1)⟩

example : ((∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.K (Pair.freshSystem.cls q) *
      Pair.freshSystem.nj q : ℕ) : ℝ) ≤
    (2 : ℕ) * (√(1 / 4) * (((3 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image Pair.freshSystem.cls,
        wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) j) / (1 : ℕ)) * √(((3 : ℕ) : ℝ) / (1 : ℕ)) +
        (1 : ℕ) * (3 : ℕ)) +
      √(1 / 4) * (((3 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image Pair.freshSystem.cls,
        wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) j) / (1 : ℕ)) *
        √(((2 : ℕ) : ℝ) * (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.nj q : ℕ)) := by
  obtain ⟨hS, hV, hsh, hW, hM, hsO, hap, hc, hR, hP, hb, hcs, hA, hμ⟩ := SumCap_honest
  exact SumCap _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 2 Pair.P hS hV hsh hW hM hsO
    hap (fun _ _ => hc) (fun _ _ => hR) hP hb hb hcs hA hμ

example (h : Pair.registeredSystem.WidthExtended 2 1 2) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hsO, hap, hc, hR, hP, hb, hC, hcs, hA, hμ, hn⟩ := SumCap_nontrivial
  exact hn (SumCap _ _ _ Pair.registeredSystem 2 1 2 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 1 Pair.P1 hS hV hsh h
    hM hsO hap (fun _ _ => hc) (fun _ _ => hR) hP hb hC hcs hA hμ)

example : sumBound Pair.fresh (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q *
      Pair.freshSystem.K (Pair.freshSystem.cls q) * Pair.freshSystem.nj q)
    (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.nj q) 1 1 3 1 1 (1 / 4)
    (∑ j ∈ (univ : Finset Unit).image Pair.freshSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) j) ≤
    partitionSlowdown Pair.fresh 2 1 := by
  obtain ⟨hS, hV, hsh, hW, hM, hsO, hap, hc, hR, hP, hb, hcs, hA, hμ⟩ := SumBound_honest
  exact SumBound _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 hS hV hsh hW hM hsO hap
    (fun _ _ => hc) (fun _ _ => hR) hb hb hcs hA hμ 2 Pair.P hP

example (h : Pair.registeredSystem.WidthExtended 2 1 2) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hsO, hap, hc, hR, hb, hC, hcs, hA, hμ, hn⟩ := SumBound_nontrivial
  exact hn (SumBound _ _ _ Pair.registeredSystem 2 1 2 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 hS hV hsh h hM hsO
    hap (fun _ _ => hc) (fun _ _ => hR) hb hC hcs hA hμ)

end Nci.Products.SecurityProofs
