import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Option
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Fintype.Sum
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Audit.SampledAudit
import Proofs.Pous.B5.ExPostFacto
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq

/-!
# The dense single-layer scheme meets the requirement: `PousDense.dense_meets`

The first end-to-end `Pous.Meets` with no named assumption, in the random-oracle model (`docs/p3-scheme.md` §4,
S1). The scheme is the trusted `Pous.Dense.denseScheme`, and the statement is `Pous.Guarantees.DenseMeets`'s, verbatim.
The proof composes five pieces:

1. **Reach** (`card_reach_le`, `complete_hard`). Greedy pebbling of the complete DAG adds at most the least
   unpebbled node per round, so `B − D − 1` pebbles reach at most `B − 1` nodes in `D` rounds.
2. **B5 gives `TimedINC`** (`dense_timedINC`). Rebuilding every block is "all `B` challenges right", so
   `Pous.Guarantees.B5ExPostFactoBounded` (`PousB5`) bounds it by
   `π = B·2^β·((B(Q + 1) + B)²/2^ℓ)^(B − D)`.
3. **`Pous.Guarantees.Theorem1iiSeq`** (the `thm1-family` proof, `PousProofs.theorem1_ii_seq`) gives
   `Pr[pass] ≤ |Hist|·π + p₀^k`.
4. **Rate 1** (`denseScheme_correct`, `CodeHoldsW`, `SpaceBound`).
5. **The certificate at `β = 531499840`.** `|Hist|·π ≤ 2^-128` (`dense_union_term`, in exponents of 2), and
   `p₀^107 ≤ 1%` (`dense_sampling_term`, exact rationals), with `S = ⌊(18/19)·2^29⌋ = 508614548`
   (`stateBound_dense`).

It imports the Theorem 1 library (`PousThm1`, with `theorem1_ii_seq`) and the B5 proof (`PousB5`); the scheme and
the proof are `PousDense`.
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-! ### The threshold `p₀` -/

/-! ### Compression -/

/-! ### More uniform probability -/

/-! ### Programs -/

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

/-! ### Compression against `TimedINC` -/

/-! ### Sequential responders -/

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

end PousProofs

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

/-! ### Transcripts of a program, and what they depend on -/

section Trace

variable {M : IdealModel} {α : Type}

end Trace

/-! ### Labels -/

section Labels

variable {N w : ℕ}

end Labels

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

variable {N w : ℕ}

namespace Replay

variable (R : Replay N w)

section Stable

variable {H H' : LabelQry N w → Bits w} {κ : ℕ} (hag : ∀ c ∈ R.VC, AgreeBefore H H' (R.P c) κ)
include hag

end Stable

end Replay

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

variable {N w : ℕ}

/-! ### Pebbling -/

namespace Replay

variable (R : Replay N w)

/-! ### A prediction costs `w` bits -/

/-! ### Fresh labels and the ex post facto pebbling -/

/-! ### The prediction of a fresh label -/

end Replay

end PousB5

namespace PousB5

open Pous Pous.Labelling
open scoped ENNReal

end PousB5

/-! ## The dense single-layer scheme (S1) and its `Meets` -/

namespace PousDense

open Pous Pous.Labelling Pous.Dense
open scoped ENNReal

/-! ### Reach on the complete DAG: `s + D` -/

/-- Greedy pebbling of the complete DAG adds at most one node per round (the least unpebbled one), so from `P₀` it
covers at most `|P₀| + r` nodes in `r` rounds. -/
theorem card_reach_le (B : ℕ) (P₀ : Finset (Fin B)) : ∀ r, (reach (completeDAG B) P₀ r).card ≤ P₀.card + r
  | 0 => by rw [reach]; omega
  | r + 1 => by
    have ih := card_reach_le B P₀ r
    have hnew : ((Finset.univ.filter fun v => (completeDAG B).par v ⊆ reach (completeDAG B) P₀ r) \
        reach (completeDAG B) P₀ r).card ≤ 1 := by
      rw [Finset.card_le_one]
      intro a ha b hb
      rw [Finset.mem_sdiff, Finset.mem_filter] at ha hb
      by_contra hab
      rcases lt_or_gt_of_ne hab with h | h
      · exact ha.2 (hb.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
      · exact hb.2 (ha.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    rw [reach, ← Finset.union_sdiff_self_eq_union]
    calc _ ≤ (reach (completeDAG B) P₀ r).card + ((Finset.univ.filter fun v =>
          (completeDAG B).par v ⊆ reach (completeDAG B) P₀ r) \ reach (completeDAG B) P₀ r).card :=
          Finset.card_union_le _ _
      _ ≤ P₀.card + (r + 1) := by omega

/-- The complete DAG with every node a challenge is `(B − D − 1, D, B)`-hard: `B − D − 1` pebbles reach at most
`B − 1` nodes in `D` rounds. -/
theorem complete_hard (B D : ℕ) (hD : D + 1 ≤ B) : PebblingHard (completeDAG B) Finset.univ (B - D - 1) D B := by
  intro P₀ hP₀
  calc (Finset.univ.filter fun c => c ∈ reach (completeDAG B) P₀ D).card
      ≤ (reach (completeDAG B) P₀ D).card := Finset.card_le_card fun c hc => (Finset.mem_filter.1 hc).2
    _ ≤ P₀.card + D := card_reach_le B P₀ D
    _ < B := by omega

/-! ### Correctness and rate 1 -/

theorem bxor_bxor_self {w : ℕ} (a b : Bits w) : bxor (bxor a b) a = b := by
  funext j
  simp only [bxor]
  cases a j <;> cases b j <;> rfl

/-- Bit-exact public decoding: `C_i ⊕ H(i, C_{<i}) = W_i`. -/
theorem denseScheme_correct (B ℓ : ℕ) [NeZero B] : (denseScheme B ℓ).Correct := by
  intro (W : Bits (B * ℓ)) (H : LabelQry B ℓ → Bits ℓ) r
  show codeEquiv B ℓ (fun i => bxor (label (completeDAG B) (blocks B ℓ W) H i)
    (H (i, fun j => if j ∈ (completeDAG B).par i then some (label (completeDAG B) (blocks B ℓ W) H j) else none)))
    = W
  have h : ∀ i, bxor (label (completeDAG B) (blocks B ℓ W) H i)
      (H (i, fun j => if j ∈ (completeDAG B).par i then some (label (completeDAG B) (blocks B ℓ W) H j) else none))
      = blocks B ℓ W i := by
    intro i
    rw [PousB5.label_eq]
    exact bxor_bxor_self _ _
  rw [funext h]
  exact (codeEquiv B ℓ).apply_symm_apply W

/-! ### B5 gives `TimedINC` -/

/-- **Glue.** Rebuilding every block within `(D, Q)` from a `β`-bit compression answers all `B` challenges of the
complete DAG, so B5 bounds `TimedINC` by `π = B·2^β·((B(Q + 1) + B)²/2^ℓ)^(B − D)`. -/
theorem dense_timedINC (B ℓ D Q β : ℕ) [NeZero B] (hD : D + 1 ≤ B) :
    TimedINC (denseScheme B ℓ) D Q β
      (B * 2 ^ β * ((((B * (Q + 1) + B) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ ℓ) ^ (B - D - 1 + 1)) := by
  classical
  intro W cmp exp hexp
  have hB5 := PousB5.b5_ex_post_facto_bounded B ℓ β (B - D - 1) D Q B (completeDAG B) Finset.univ (blocks B ℓ W)
    (complete_hard B D hD)
    (fun H => cmp W (fun i => i.elim0) (fun i => label (completeDAG B) (blocks B ℓ W) H i) H)
    (fun σ i => exp W (fun i => i.elim0) σ i) (fun σ i => hexp W _ σ i)
  rw [Finset.card_univ, Fintype.card_fin] at hB5
  refine le_trans (le_of_eq_of_le (PousThm1.pr_fst (β := Unit) (fun H : LabelQry B ℓ → Bits ℓ =>
    ∀ i, (exp W (fun i => i.elim0) (cmp W (fun i => i.elim0) (fun i => label (completeDAG B) (blocks B ℓ W) H i) H)
      i).run H = label (completeDAG B) (blocks B ℓ W) H i)) (pr_mono fun H hH => ?_)) hB5
  refine le_trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin]))
    (Finset.card_le_card (s := Finset.univ) fun c _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, hH c⟩)

/-! ### The certificate -/

theorem sum_pow_le (n t : ℕ) (hn : 2 ≤ n) : ∑ j ∈ Finset.range t, (n : ℝ≥0∞) ^ j ≤ (n : ℝ≥0∞) ^ t := by
  have h : ∑ j ∈ Finset.range t, n ^ j ≤ n ^ t := by
    induction t with
    | zero => simp
    | succ t ih =>
      rw [Finset.sum_range_succ, pow_succ]
      nlinarith [pow_pos (by omega : 0 < n) t]
  exact_mod_cast h

theorem two_pow_mul_inv_pow (a b : ℕ) (h : a ≤ b) :
    (2 : ℝ≥0∞) ^ a * (2⁻¹ : ℝ≥0∞) ^ b = (2⁻¹ : ℝ≥0∞) ^ (b - a) := by
  obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [pow_add, ← mul_assoc, ← mul_pow, ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_pow, one_mul,
    Nat.add_sub_cancel_left]

theorem two_pow_combine (a b c d : ℕ) (h : a + b + c ≤ d) :
    (2 : ℝ≥0∞) ^ a * ((2 : ℝ≥0∞) ^ b * 2 ^ c * (2⁻¹ : ℝ≥0∞) ^ d) = (2⁻¹ : ℝ≥0∞) ^ (d - (a + b + c)) := by
  rw [← two_pow_mul_inv_pow _ _ h, pow_add, pow_add]
  ring

theorem sq_div_two_pow_le (x c w : ℕ) (hx : x ^ 2 ≤ 2 ^ c) (hc : c ≤ w) :
    ((x ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w ≤ (2⁻¹ : ℝ≥0∞) ^ (w - c) := by
  calc ((x ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ w ≤ (2 : ℝ≥0∞) ^ c / 2 ^ w := by
        gcongr
        exact_mod_cast hx
    _ = (2⁻¹ : ℝ≥0∞) ^ (w - c) := by
        rw [div_eq_mul_inv, ENNReal.inv_pow, two_pow_mul_inv_pow c w hc]

/-- `|Hist|·π ≤ 2^-128` at `B = 2^16`, `ℓ = 8192`, `D = 64`, `Q = 2^20`, `k = 107`, `β = 531499840`. -/
theorem dense_union_term :
    (∑ j ∈ Finset.range 107, ((2 ^ 16 : ℕ) : ℝ≥0∞) ^ j) *
      (((2 ^ 16 : ℕ) : ℝ≥0∞) * 2 ^ 531499840 *
        (((((2 ^ 16) * (2 ^ 20 + 1) + 2 ^ 16) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 8192) ^ (2 ^ 16 - 64 - 1 + 1)) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have he : 2 ^ 16 - 64 - 1 + 1 = 65472 := by norm_num
  have hsq : ((2 ^ 16) * (2 ^ 20 + 1) + 2 ^ 16) ^ 2 ≤ 2 ^ 74 := by norm_num
  have h16 : ((2 ^ 16 : ℕ) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 16 := by norm_num
  rw [he]
  calc _ ≤ ((2 ^ 16 : ℕ) : ℝ≥0∞) ^ 107 *
        (((2 ^ 16 : ℕ) : ℝ≥0∞) * 2 ^ 531499840 * ((2⁻¹ : ℝ≥0∞) ^ (8192 - 74)) ^ 65472) := by
        gcongr
        · exact sum_pow_le _ _ (by norm_num)
        · exact sq_div_two_pow_le _ 74 8192 hsq (by norm_num)
    _ = (2 : ℝ≥0∞) ^ (16 * 107) * ((2 : ℝ≥0∞) ^ 16 * 2 ^ 531499840 * (2⁻¹ : ℝ≥0∞) ^ ((8192 - 74) * 65472)) := by
        rw [h16, ← pow_mul, ← pow_mul]
    _ = (2⁻¹ : ℝ≥0∞) ^ ((8192 - 74) * 65472 - (16 * 107 + 16 + 531499840)) :=
        two_pow_combine _ _ _ _ (by norm_num)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

/-- `p₀^107 ≤ 1%` at `β = 531499840`, `S = 508614548`, `B = 2^16`, `ℓ = 8192`. -/
theorem dense_sampling_term :
    ENNReal.ofReal (p₀ 531499840 508614548 (2 ^ 16) 8192) ^ 107 ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  have h0 : 0 ≤ p₀ 531499840 508614548 (2 ^ 16) 8192 := by unfold p₀; norm_num
  rw [← ENNReal.ofReal_pow h0]
  refine ENNReal.ofReal_le_ofReal ?_
  unfold p₀
  norm_num [Params.δ]

/-! ### The requirement -/

theorem stateBound_dense : (denseScheme (2 ^ 16) 8192).stateBound Params.ρ = 508614548 := by
  show ⌊Params.ρ * ((2 ^ 16 * 8192 : ℕ) : ℚ)⌋₊ = 508614548
  rw [Nat.floor_eq_iff (by norm_num [Params.ρ])]
  norm_num [Params.ρ]

end PousDense

/-- **The first end-to-end `Meets` with no named assumption** (`docs/p3-scheme.md` §4). The dense scheme on
`B = 2^16` blocks of 1 KB meets the requirement with sequential reveal, `D = 64` rounds and `Q = 2^20` queries per
answer, `k = 107` challenges and `ε_crypto = 2^-128`: it is correct, at rate 1, and every `⌊ρ|C|⌋`-bit adversary is
accepted with probability at most `1% + 2^-128`. -/
theorem PousDense.dense_meets :
    Pous.Meets (Pous.Dense.denseScheme (2 ^ 16) 8192) .sequential 64 (2 ^ 20) 107 ((2⁻¹ : ENNReal) ^ 128) := by
  refine ⟨le_rfl, PousDense.denseScheme_correct _ _, le_rfl, ?_, ?_⟩
  · show (((2 ^ 16 * 8192 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * ((2 ^ 16 * 8192 : ℕ) : ℚ)
    norm_num [Pous.Params.expansion]
  · intro W A₁ A₂
    refine (PousProofs.theorem1_ii_seq (Pous.Dense.denseScheme (2 ^ 16) 8192) 64 (2 ^ 20) 531499840
      ((Pous.Dense.denseScheme (2 ^ 16) 8192).stateBound Pous.Params.ρ) 107 _
      (PousDense.dense_timedINC (2 ^ 16) 8192 64 (2 ^ 20) 531499840 (by norm_num)) W A₁ A₂).trans ?_
    rw [PousDense.stateBound_dense, add_comm]
    exact add_le_add PousDense.dense_sampling_term PousDense.dense_union_term
