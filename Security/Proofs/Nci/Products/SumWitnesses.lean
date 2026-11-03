import Proofs.Nci.Products.Sum
import Proofs.Nci.Products.WholeWitnesses
import Definitions.Nci.Guarantees.Products.SumWitnesses
import Mathlib.Tactic.NormNum

/-!
# Products made whole through chains, several in a block: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Products/SumWitnesses.lean`, at `Pair`'s product alone as a
whole-making system (`Security/Definitions/Nci/Products/WholeInstances.lean`). The class bound's premises, the cells and
the received partial sums are the whole-making lemmas' (`Security/Proofs/Nci/Products/WholeWitnesses.lean`); P-share on
the one product is P-share at its class (`System.sharedOn_of_shared`), and a product is apart from itself. Each bound is
compared with its price in `ℝ` or `ℝ≥0∞`, with `√(1/4) = 1/2` and `√2 < 2`. The `example`s at the end apply each
guarantee to its lemmas, so the lemmas are the premises as the guarantee states them.
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
      (1 : ℕ) * ((1 : ℕ) * (1 : ℕ)) + √(1 / 4) * (((1 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image
          Pair.registeredSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) j) / (1 : ℕ)) *
        √(((1 : ℕ) : ℝ) * ((1 : ℕ) * (((1 : ℕ) : ℝ) / (1 : ℕ)) +
          (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q : ℕ))) := by
  rw [sum_macs, sum_outputs, sum_wholeFactor, sqrt_quarter]
  have h2 : √(2 : ℝ) < 2 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  norm_num
  linarith

theorem not_sumBound_le :
    ¬ sumBound Pair.registered
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
          Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q)
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q) 1 2 1 1 1 (1 / 4)
        (∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) j)
        1 ≤ partitionSlowdown Pair.registered 1 2 := by
  rw [sum_macs, sum_outputs, sum_wholeFactor]
  unfold sumBound partitionSlowdown
  rw [computed_card, ENNReal.mul_div_cancel_right (by norm_num) (by norm_num),
    ENNReal.div_self (by norm_num) (by norm_num), not_le, ENNReal.one_lt_ofReal, sqrt_quarter]
  norm_num

end Pair

theorem SumCap_honest : Guarantees.SumCap_honest :=
  ⟨fun _ => Pair.algorithm_fresh.toWiring, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh,
    Pair.matched _ _, Pair.freshSystem.sharedOn_of_shared Pair.shared_fresh (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh⟩

theorem SumCap_nontrivial : Guarantees.SumCap_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered.toWiring, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered, Pair.matched _ _,
    Pair.registeredSystem.sharedOn_of_shared Pair.shared_registered (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_registered, Pair.received_registered 2 1, Pair.isCK_P1, Nat.one_pos, fun _ => by norm_num,
    fun _ _ => le_rfl, Pair.postFixed_registered, Pair.not_sumCap⟩

theorem SumBound_honest : Guarantees.SumBound_honest :=
  ⟨fun _ => Pair.algorithm_fresh.toWiring, Pair.values_fresh, Pair.shared_fresh, Pair.widthExtended_fresh,
    Pair.matched _ _, Pair.freshSystem.sharedOn_of_shared Pair.shared_fresh (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_fresh, one_pos⟩

theorem SumBound_nontrivial : Guarantees.SumBound_nontrivial :=
  ⟨fun _ => Pair.algorithm_registered.toWiring, Pair.values_registered, Pair.shared_registered,
    Pair.not_widthExtended_registered, Pair.matched _ _,
    Pair.registeredSystem.sharedOn_of_shared Pair.shared_registered (j := ()) fun _ _ => rfl, Pair.apart _ _,
    Pair.cells_registered, Pair.received_registered 2 1, Nat.one_pos, fun _ => by norm_num, fun _ _ => le_rfl,
    Pair.postFixed_registered, one_pos, fun h => Pair.not_sumBound_le (h 1 Pair.P1 Pair.isCK_P1)⟩

example : ((∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.K (Pair.freshSystem.cls q) *
      Pair.freshSystem.nj q : ℕ) : ℝ) ≤
    (2 : ℕ) * ((1 : ℕ) * (3 : ℕ)) + √(1 / 4) * (((3 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image
        Pair.freshSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) j) / (1 : ℕ)) *
      √(((2 : ℕ) : ℝ) * ((2 : ℕ) * (((3 : ℕ) : ℝ) / (1 : ℕ)) +
        (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.nj q : ℕ))) := by
  obtain ⟨hS, hV, hsh, hW, hM, hsO, hap, hc, hR, hP, hb, hcs, hA, hμ⟩ := SumCap_honest
  exact SumCap _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 2 Pair.P hS hV hsh hW hM hsO
    hap (fun _ _ => hc) (fun _ _ => hR) hP hb hb hcs hA hμ

example (h : Pair.registeredSystem.WidthExtended 2 1 1) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hsO, hap, hc, hR, hP, hb, hcs, hA, hμ, hn⟩ := SumCap_nontrivial
  exact hn (SumCap _ _ _ Pair.registeredSystem 2 1 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 1 Pair.P1 hS hV hsh h
    hM hsO hap (fun _ _ => hc) (fun _ _ => hR) hP hb hb hcs hA hμ)

example : sumBound Pair.fresh (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q *
      Pair.freshSystem.K (Pair.freshSystem.cls q) * Pair.freshSystem.nj q)
    (∑ q ∈ (univ : Finset Unit), Pair.freshSystem.ni q * Pair.freshSystem.nj q) 1 1 3 1 1 (1 / 4)
    (∑ j ∈ (univ : Finset Unit).image Pair.freshSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 3 / 16) j) 1 ≤
    partitionSlowdown Pair.fresh 2 1 := by
  obtain ⟨hS, hV, hsh, hW, hM, hsO, hap, hc, hR, hP, hb, hcs, hA, hμ, ht⟩ := SumBound_honest
  exact SumBound _ _ _ Pair.freshSystem 1 3 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 1 hS hV hsh hW hM hsO hap
    (fun _ _ => hc) (fun _ _ => hR) hb hb hcs hA hμ ht 2 Pair.P hP

example (h : Pair.registeredSystem.WidthExtended 2 1 1) : False := by
  obtain ⟨hS, hV, hsh, -, hM, hsO, hap, hc, hR, hb, hcs, hA, hμ, ht, hn⟩ := SumBound_nontrivial
  exact hn (SumBound _ _ _ Pair.registeredSystem 2 1 1 1 _ _ _ univ (1 / 4) (fun _ => Pair.cell) 1 1 hS hV hsh h hM
    hsO hap (fun _ _ => hc) (fun _ _ => hR) hb hb hcs hA hμ ht)

end Nci.Products.SecurityProofs
