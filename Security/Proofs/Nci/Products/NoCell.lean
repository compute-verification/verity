import Definitions.Nci.Guarantees.Products.NoCell
import Proofs.Nci.Products.Cap

/-!
# Products of two fresh operands without the word level: the cap and its bound on `ρ`

For a partition `P` into `m` inference blocks, block `u` with gates `I_u`, reads `U_u = in(I_u) \ θ` and whole MACs
`T_u`, and the sets `start_ij` that partial sums received at their width (`Standard.Received`) give:

1. **Splits** (`card_le_whole_add`): `N ≤ Σ_u |T_u| + Σ_u |U_u ∩ macs|`, and each block reads at most `k` of the MACs'
   gates (`card_add_card_le`), so `Σ_u |U_u ∩ macs| ≤ m k`.
2. **Partial sums received** (`received_sums`): `b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + Σ_ij |I_u ∩ start_ij|`, each entry a
   block touches paying `b_C` bits of reads of its gates less the start's gates the block holds.
3. **Each start held once** (`sum_sum_card_inter_le`): `Σ_u Σ_ij |I_u ∩ start_ij| ≤ ni nj b_C`, the blocks being
   disjoint and each start at most `b_C` gates.

With `s_u = Σ_ij |I_u ∩ start_ij| / b_C`, facts 2 and 3 are the cap along chains' (`Verity.Combinatorics.sum_card_le_chain`)
per-block partial sums and its `Σ_u s_u ≤ ni nj`, so `Σ_u |T_u| ≤ m α + β √(m ni nj)`, and fact 1 adds `m k`
(`NoCellCap`). `NoCellBound` inverts the cap: `m Γ + B √m ≥ N` gives `m ≥ chainBlocks N Γ B` (`chainBlocks_le`).
-/

namespace Nci.Products

open Finset
open Verity (Declaration Partition)
open Nci.Formalism
open scoped ENNReal

variable {d : Declaration}

/-- A partition's blocks hold at most all of a set's gates. -/
theorem sum_card_inter_gates_le {m : ℕ} (P : Partition d.D m) (X : Finset (Fin d.D.N)) :
    ∑ u, (P.gates u ∩ X).card ≤ X.card :=
  calc ∑ u, (P.gates u ∩ X).card = ∑ u, (P.gates u ∩ (d.D.computed ∩ X)).card :=
        sum_congr rfl fun u _ => by rw [← inter_assoc, inter_eq_left.2 (P.gates_subset_computed u)]
    _ = (d.D.computed ∩ X).card := sum_card_inter_gates P inter_subset_left
    _ ≤ X.card := card_le_card inter_subset_right

/-- **Each start held once**: sets of at most `b_C` gates, one an entry, have at most `ni nj b_C` of their gates in a
partition's blocks. -/
theorem sum_sum_card_inter_le {ni nj bC : ℕ} {start : Fin ni → Fin nj → Finset (Fin d.D.N)}
    (hcard : ∀ i j, (start i j).card ≤ bC) {m : ℕ} (P : Partition d.D m) :
    ∑ u, ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card ≤ ni * nj * bC := by
  rw [sum_comm]
  calc ∑ e : Fin ni × Fin nj, ∑ u, (P.gates u ∩ start e.1 e.2).card
      ≤ ∑ _e : Fin ni × Fin nj, bC := sum_le_sum fun e _ => (sum_card_inter_gates_le P _).trans (hcard e.1 e.2)
    _ = ni * nj * bC := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, smul_eq_mul]

namespace Standard

variable {ni nl nj : ℕ} (S : Standard d ni nl nj)

/-- **Partial sums received**: `b_C |π₁₃ T_u| ≤ |U_u ∩ macs| + Σ_ij |I_u ∩ start_ij|`, each entry a block touches
paying `b_C` bits of reads of its gates, less its start's gates the block holds. -/
theorem received_sums (hS : S.Algorithm) {c k bC : ℕ} {start : Fin ni → Fin nj → Finset (Fin d.D.N)}
    (hread : ∀ I, InferenceBlock d c k I → ∀ i l j, S.mac i l j ⊆ I →
      bC ≤ (d.unregistered I ∩ S.entry i j).card + (I ∩ start i j).card)
    {m : ℕ} {P : Partition d.D m} (hP : P.IsCK d c k) (u : Fin m) :
    bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card ≤
      (d.unregistered (P.gates u) ∩ S.macs).card + ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card :=
  calc bC * ((S.whole (P.gates u)).image fun x => (x.1, x.2.2)).card
      = ∑ _e ∈ (S.whole (P.gates u)).image fun x => (x.1, x.2.2), bC := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ e ∈ (S.whole (P.gates u)).image fun x => (x.1, x.2.2),
          ((d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card + (P.gates u ∩ start e.1 e.2).card) := by
        refine sum_le_sum fun e he => ?_
        obtain ⟨⟨i, l, j⟩, hx, rfl⟩ := mem_image.1 he
        exact hread _ (inferenceBlock_gates hP u) i l j (mem_filter.1 hx).2
    _ ≤ ∑ e : Fin ni × Fin nj,
          ((d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card + (P.gates u ∩ start e.1 e.2).card) :=
        sum_le_sum_of_subset (subset_univ _)
    _ = ∑ e : Fin ni × Fin nj, (d.unregistered (P.gates u) ∩ S.entry e.1 e.2).card +
          ∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card := sum_add_distrib
    _ ≤ _ := add_le_add (S.sum_card_entry_le hS _) le_rfl

end Standard

/-- **The fewest blocks**: `N ≤ m Γ + B √m` with `N, Γ, B, m ≥ 0` gives `chainBlocks N Γ B ≤ m`. -/
theorem chainBlocks_le {N Γ B m : ℝ} (hN : 0 ≤ N) (hΓ : 0 ≤ Γ) (hB : 0 ≤ B) (hm : 0 ≤ m)
    (h : N ≤ m * Γ + B * √m) : chainBlocks N Γ B ≤ m := by
  unfold chainBlocks
  have hR0 : 0 ≤ √(B ^ 2 + 4 * Γ * N) := Real.sqrt_nonneg _
  have hs0 : 0 ≤ √m := Real.sqrt_nonneg _
  have hs2 : √m ^ 2 = m := Real.sq_sqrt hm
  rcases eq_or_lt_of_le (add_nonneg hR0 hB) with hD | hD
  · rw [← hD, div_zero, zero_pow two_ne_zero]
    exact hm
  have hs : 2 * N / (√(B ^ 2 + 4 * Γ * N) + B) ≤ √m := by
    rw [div_le_iff₀ hD]
    have hN' : N ≤ Γ * √m ^ 2 + B * √m := by rw [hs2]; linarith
    have hR : √(B ^ 2 + 4 * Γ * N) ≤ 2 * Γ * √m + B := by
      rw [← Real.sqrt_sq (by positivity : 0 ≤ 2 * Γ * √m + B)]
      exact Real.sqrt_le_sqrt (by nlinarith)
    have hRB : B ≤ √(B ^ 2 + 4 * Γ * N) :=
      calc B = √(B ^ 2) := (Real.sqrt_sq hB).symm
        _ ≤ _ := Real.sqrt_le_sqrt (by nlinarith)
    rcases eq_or_lt_of_le hΓ with hΓ0 | hΓ0
    · subst hΓ0
      nlinarith [mul_le_mul_of_nonneg_left hRB hs0]
    · nlinarith [mul_le_mul_of_nonneg_right hR hD.le, Real.sq_sqrt (by positivity : 0 ≤ B ^ 2 + 4 * Γ * N)]
  calc (2 * N / (√(B ^ 2 + 4 * Γ * N) + B)) ^ 2 ≤ √m ^ 2 :=
        pow_le_pow_left₀ (div_nonneg (by linarith) hD.le) hs 2
    _ = m := hs2

namespace SecurityProofs

open Standard

theorem NoCellCap : Guarantees.NoCellCap := by
  intro d ni nl nj S c k bA bB bC m P hS hW hR hP hA hB hC
  obtain ⟨start, hcard, hread⟩ := hR
  have hCpos : (0 : ℝ) < bC := by exact_mod_cast hC
  have key := Verity.Combinatorics.sum_card_le_chain (fun u => S.whole (P.gates u))
    (fun u => ((d.unregistered (P.gates u) ∩ S.cones).card : ℝ))
    (fun u => ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ))
    (fun u => ((∑ e : Fin ni × Fin nj, (P.gates u ∩ start e.1 e.2).card : ℕ) : ℝ) / bC)
    (k := k) (bA := bA) (bB := bB) (bC := bC) (O := ((ni * nj : ℕ) : ℝ)) (Nat.cast_nonneg _)
    (by exact_mod_cast hA) (by exact_mod_cast hB) hCpos (fun _ => Nat.cast_nonneg _)
    (fun _ => div_nonneg (Nat.cast_nonneg _) hCpos.le)
    (fun u => by exact_mod_cast S.mul_card_le_cones hS hW (inferenceBlock_gates hP u))
    (fun u => by exact_mod_cast S.card_add_card_le hS (inferenceBlock_gates hP u))
    (fun u => by
      have h := S.received_sums hS hread hP u
      rw [mul_div_cancel₀ _ hCpos.ne']
      exact_mod_cast h)
    (by
      rw [← sum_div, div_le_iff₀ hCpos]
      exact_mod_cast sum_sum_card_inter_le hcard P)
  rw [Fintype.card_fin] at key
  have hsplit : ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) ≤ m * k := by
    calc ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) ≤ ∑ _u : Fin m, (k : ℝ) :=
          sum_le_sum fun u _ => by
            exact_mod_cast (Nat.le_add_left _ _).trans (S.card_add_card_le hS (inferenceBlock_gates hP u))
      _ = m * k := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hN : ((ni * nl * nj : ℕ) : ℝ) ≤ ∑ u, ((S.whole (P.gates u)).card : ℝ) +
      ∑ u, ((d.unregistered (P.gates u) ∩ S.macs).card : ℝ) := by
    exact_mod_cast S.card_le_whole_add hS P
  rw [mul_add]
  linarith

theorem NoCellBound : Guarantees.NoCellBound := by
  intro d ni nl nj S c k bA bB bC hS hW hR hA hB hC m P hP
  have hcap := NoCellCap d ni nl nj S c k bA bB bC m P hS hW hR hP hA hB hC
  have hα : 0 ≤ (k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + k := by positivity
  have hβ : 0 ≤ (k : ℝ) / 2 / √((bA : ℝ) * bB) * √((ni * nj : ℕ) : ℝ) := by positivity
  have hm : chainBlocks ((ni * nl * nj : ℕ) : ℝ) ((k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + k)
      ((k : ℝ) / 2 / √((bA : ℝ) * bB) * √((ni * nj : ℕ) : ℝ)) ≤ m := by
    refine chainBlocks_le (Nat.cast_nonneg _) hα hβ (Nat.cast_nonneg _) ?_
    rw [Real.sqrt_mul (Nat.cast_nonneg (α := ℝ) m) ((ni * nj : ℕ) : ℝ)] at hcap
    linarith
  unfold noCellBound partitionSlowdown
  refine ENNReal.div_le_div_right ?_ _
  rw [show ((m * c : ℕ) : ℝ≥0∞) = (m : ℝ≥0∞) * c from Nat.cast_mul m c, ← ENNReal.ofReal_natCast m]
  gcongr

end SecurityProofs

end Nci.Products
