module

public import Definitions.NetworkCertifier.Assumptions
public import Definitions.NetworkCertifier.Schedule
public import Mathlib.Data.Set.Card
import Mathlib.Data.Sym.Sym2.Init
import Proofs.NetworkCertifier.Advice
import Proofs.NetworkCertifier.Counting

/-!
# Theorem 3 for the reference schedule: the constant-rate Program

`docs/network-transparency/timing-channel.md` §2.4, for `constantRate` (`Definitions.NetworkCertifier.Schedule`). Its
advice is one value per execution block, so `encard_accSeqs_le_advice` applies as it stands. Causality
(`constantRate_rows_causal`: a step's rows read only its block's advice) sharpens it to the blocks the counted steps
reach. `oneFrame` is a schedule of one frame of one execution block.
-/

@[expose] public section

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

theorem encard_accSeqs_le_constantRate {Adv : Type} [Fintype Adv] (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤
      ((Fintype.card Adv ^ E * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_advice (constantRate (S := S) T g E frames) op n D
  rwa [Fintype.card_fun, Fintype.card_fin] at this

/-! ### Causality: a step's rows read only its execution block's advice -/

theorem constantRate_rows_causal {Adv : Type} (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (t : ℕ) (ins : List In) (a a' : Fin E → Adv)
    (ha : ∀ e : Fin E, (e : ℕ) = t / (g * T) → a e = a' e) :
    (constantRate (S := S) T g E frames).rows t ins a = (constantRate (S := S) T g E frames).rows t ins a' := by
  funext ℓ
  simp only [constantRate]
  split_ifs with h
  · rw [ha ⟨t / (g * T), h⟩ rfl]
  · rfl

/-! ### `|a|` bits per execution block opened -/

theorem encard_accSeqs_le_constantRate_executionPhases {Adv : Type} [Fintype Adv] (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hm : n ≤ m * (g * T)) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤
      ((Fintype.card Adv ^ min E m * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Fin E → Adv)
  swap
  · have : AccSeqs (constantRate (S := S) T g E frames) op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Fin E → Adv := Classical.choice hne
  let adv : List (Obs L K S) → Fin E → Adv := fun o =>
    if h : AcceptedUpTo (constantRate (S := S) T g E frames) op n o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs (constantRate (S := S) T g E frames) op n D,
      Accepted (constantRate (S := S) T g E frames) op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let cut : (Fin E → Adv) → Fin (min E m) → Adv := fun a i =>
    a ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩
  let code : List (Obs L K S) → (Fin (min E m) → Adv) × (Fin n → ℕ) := fun o => (cut (adv o), jitterCode o)
  have hinj : Set.InjOn code (AccSeqs (constantRate (S := S) T g E frames) op n D) := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨hσ, hj⟩ := h
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · rw [hr, hr', ingressUpTo_congr op hpre]
      apply constantRate_rows_causal
      intro e he
      have htm : t < m * (g * T) := lt_of_lt_of_le ht hm
      have hem : (e : ℕ) < m := by
        rw [he]; exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm (g * T) m]; exact htm)
      have hσe := congrFun hσ ⟨e, lt_min e.2 hem⟩
      simpa only [cut, Fin.eta] using hσe
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (AccSeqs (constantRate (S := S) T g E frames) op n D).encard
      ≤ (((univ : Finset (Fin (min E m) → Adv)) ×ˢ
          Fintype.piFinset fun _ : Fin n => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, mem_product, mem_univ, true_and]
        exact jitterCode_mem o ho.1 ho.2.1
    _ = ((Fintype.card Adv ^ min E m * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, Fintype.card_piFinset]
        simp

/-- One frame of one execution block, sent at bucket 1 if the block's one advice bit is set and at bucket 0
otherwise. -/
def oneFrame : List Unit → Fin 1 → (Fin 1 → Bool) → Fin 1 → List LogicalFrame :=
  fun _ _ v _ => [⟨[true], if v 0 then 1 else 0⟩]

end NetTiming
