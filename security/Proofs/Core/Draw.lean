module

public import Definitions.Core.Draw
public import Proofs.Core.Game.Prob
public import Mathlib.Data.Fintype.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Logic.Equiv.Sum
import Proofs.Core.Game.Basic

/-!
# The product of designs by kind, and the uniform subset: proof

A set of units of the product (`SamplingDesign.prod`) is its share of each kind, and the product misses it exactly when
each kind's design misses that kind's share, on coins of its own. So its miss probability is the product over kinds
(`SamplingDesign.missProb_prod`), and with the kinds the calls of a PoUW window it is the calls' product distribution.

The uniform `k`-subset misses a set `B` of units with probability `C(n − |B|, k)/C(n, k)` (`subset_escape`): its coins
that miss `B` are the `k`-subsets of `B`'s complement (`card_disjoint_subsets`). A uniform permutation's `k`-prefix
samples as the uniform `k`-subset does, for every property of its sample (`prCoin_permPrefix`), so with the same miss
probability (`missProb_permPrefix`): a permutation carries any `k`-subset onto any other (`exists_perm_image_eq`), so
composing with it maps the permutations with one prefix one to one onto those with another, and every `k`-subset is the
prefix of equally many permutations (`Game.avg_comp_of_fiber`).
-/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace Game

/-- **An average read through a map whose fibers have one size is the average over its codomain.** -/
theorem avg_comp_of_fiber {Ω Ω' : Type} [Fintype Ω] [Fintype Ω'] [Nonempty Ω] [DecidableEq Ω'] (g : Ω → Ω')
    (hg : ∀ a b, (Finset.univ.filter fun x => g x = a).card = (Finset.univ.filter fun x => g x = b).card)
    (f : Ω' → ℝ≥0∞) : avg (fun x => f (g x)) = avg f := by
  obtain ⟨x₀⟩ := ‹Nonempty Ω›
  set c := (Finset.univ.filter fun x => g x = g x₀).card with hc
  have hfib : ∀ a, (Finset.univ.filter fun x => g x = a).card = c := fun a => hg a (g x₀)
  have hcard : Fintype.card Ω = c * Fintype.card Ω' := by
    rw [← Finset.card_univ,
      Finset.card_eq_sum_card_fiberwise (f := g) (t := Finset.univ) (fun _ _ => Finset.mem_univ _)]
    simp [hfib, mul_comm]
  have hc0 : c ≠ 0 := by
    intro h0
    have := hfib (g x₀)
    rw [h0, Finset.card_eq_zero] at this
    exact (Finset.eq_empty_iff_forall_notMem.1 this) x₀ (by simp)
  have hsum : ∑ x : Ω, f (g x) = ∑ a : Ω', (c : ℝ≥0∞) * f a := by
    rw [← Finset.sum_fiberwise_of_maps_to (g := g) (t := Finset.univ) (fun _ _ => Finset.mem_univ _)]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.sum_congr rfl fun x hx => by rw [(Finset.mem_filter.1 hx).2], Finset.sum_const, hfib, nsmul_eq_mul]
  simp only [avg, ← Finset.mul_sum, hsum, hcard, Nat.cast_mul]
  rw [ENNReal.mul_inv (Or.inl (by exact_mod_cast hc0)) (Or.inl (ENNReal.natCast_ne_top _)), mul_assoc,
    mul_left_comm, ENNReal.inv_mul_cancel_left (by exact_mod_cast hc0) (ENNReal.natCast_ne_top _)]

end Game

namespace SamplingDesign

open Game

/-- **The product misses a set exactly as often as every kind's design misses that kind's share, independently**:
the miss probability of `⋃ₖ {k} × T k` under the product of designs by kind is `∏ₖ missProb (L k) (T k)`. -/
theorem missProb_prod {K : Type} [Fintype K] [DecidableEq K] {V : K → Type} (L : (k : K) → SamplingDesign (V k))
    (T : (k : K) → Finset (V k)) :
    (prod L).missProb (Finset.univ.sigma T) = ∏ k, (L k).missProb (T k) := by
  have hd : ∀ ω : (prod L).Ω, Disjoint ((prod L).draw ω) (Finset.univ.sigma T) ↔
      ∀ k ∈ (Finset.univ : Finset K), Disjoint ((L k).draw (ω k)) (T k) := by
    intro ω
    simp only [prod, Finset.disjoint_left, Finset.mem_sigma, Finset.mem_univ, true_and, true_implies, Sigma.forall]
  simp only [missProb]
  rw [show (fun ω : (prod L).Ω => Disjoint ((prod L).draw ω) (Finset.univ.sigma T)) =
      fun ω : (k : K) → (L k).Ω => ∀ k ∈ (Finset.univ : Finset K), Disjoint ((L k).draw (ω k)) (T k) from
    funext fun ω => propext (hd ω)]
  exact prCoin_pi_forall Finset.univ fun k (c : (L k).Ω) => Disjoint ((L k).draw c) (T k)

variable {n : ℕ}

/-- The `k`-subsets of `n` units that miss `B` are the `k`-subsets of its complement. -/
theorem card_disjoint_subsets (k : ℕ) (B : Finset (Fin n)) :
    (Finset.univ.filter fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B).card =
      (n - B.card).choose k := by
  classical
  rw [← Finset.card_map ⟨Subtype.val, Subtype.val_injective⟩]
  have : (Finset.univ.filter fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B).map
      ⟨Subtype.val, Subtype.val_injective⟩ = Bᶜ.powersetCard k := by
    ext S
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and,
      Function.Embedding.coeFn_mk, Finset.mem_powersetCard, Finset.subset_compl_iff_disjoint_right]
    constructor
    · rintro ⟨S', hS', rfl⟩
      exact ⟨hS', S'.2⟩
    · rintro ⟨hS, hk⟩
      exact ⟨⟨S, hk⟩, hS, rfl⟩
  rw [this, Finset.card_powersetCard, Finset.card_compl, Fintype.card_fin]

/-- **The hypergeometric miss probability.** A uniform `k`-subset misses `B` with probability
`C(n - |B|, k) / C(n, k)`. -/
theorem subset_escape {k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    (subset n k hk).missProb B = ((n - B.card).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := by
  classical
  unfold missProb
  rw [prCoin_eq_card]
  congr 1
  · exact congrArg _ (card_disjoint_subsets k B)
  · exact congrArg _ ((Fintype.card_finset_len k).trans (by rw [Fintype.card_fin]))

/-- A permutation carries any `k`-subset onto any other. -/
theorem exists_perm_image_eq (S T : Finset (Fin n)) (h : S.card = T.card) :
    ∃ σ : Equiv.Perm (Fin n), S.image σ = T := by
  classical
  have e : {x // x ∈ S} ≃ {x // x ∈ T} := Fintype.equivOfCardEq (by simp [h])
  have e' : {x // ¬ x ∈ S} ≃ {x // ¬ x ∈ T} :=
    Fintype.equivOfCardEq (by rw [Fintype.card_subtype_compl, Fintype.card_subtype_compl]; simp [h])
  let σ : Equiv.Perm (Fin n) :=
    (Equiv.sumCompl (· ∈ S)).symm.trans ((Equiv.sumCongr e e').trans (Equiv.sumCompl (· ∈ T)))
  have hσ : ∀ x, σ x ∈ T ↔ x ∈ S := by
    intro x
    by_cases hx : x ∈ S
    · simp [σ, Equiv.sumCompl_symm_apply_of_pos hx, hx, (e ⟨x, hx⟩).2]
    · simp [σ, Equiv.sumCompl_symm_apply_of_neg hx, hx, (e' ⟨x, hx⟩).2]
  refine ⟨σ, Finset.ext fun y => ?_⟩
  simp only [Finset.mem_image]
  constructor
  · rintro ⟨x, hx, rfl⟩
    exact (hσ x).2 hx
  · intro hy
    exact ⟨σ.symm y, (hσ _).1 (by simpa using hy), by simp⟩

/-- A permutation's `k`-prefix, as a coin of the uniform `k`-subset. -/
def prefixOf {k : ℕ} (hk : k ≤ n) (π : Equiv.Perm (Fin n)) : Draw.Subset n k :=
  ⟨(permPrefix n k).draw π, by
    simp [permPrefix, Finset.card_image_of_injective _ π.injective, Fin.card_filter_val_lt, hk]⟩

/-- Every `k`-subset is the prefix of equally many permutations. -/
theorem card_prefixOf_eq {k : ℕ} (hk : k ≤ n) (S T : Draw.Subset n k) :
    (Finset.univ.filter fun π => prefixOf hk π = S).card =
      (Finset.univ.filter fun π => prefixOf hk π = T).card := by
  obtain ⟨σ, hσ⟩ := exists_perm_image_eq S.1 T.1 (S.2.trans T.2.symm)
  refine Finset.card_bij (fun π _ => σ * π) (fun π hπ => ?_) (fun _ _ _ _ h => mul_left_cancel h) (fun π hπ => ?_)
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hπ ⊢
    apply Subtype.ext
    rw [← hσ, ← congrArg Subtype.val hπ]
    simp [prefixOf, permPrefix, Finset.image_image, Equiv.Perm.coe_mul]
  · refine ⟨σ⁻¹ * π, ?_, by simp⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hπ ⊢
    apply Subtype.ext
    have h1 := congrArg Subtype.val hπ
    simp only [prefixOf, permPrefix] at h1 ⊢
    rw [Equiv.Perm.coe_mul, ← Finset.image_image, h1, ← hσ, Finset.image_image]
    simp

/-- **A uniform permutation's `k`-prefix samples as the uniform `k`-subset does**: every property of the sample holds
with the same probability. -/
theorem prCoin_permPrefix {k : ℕ} (hk : k ≤ n) (p : Finset (Fin n) → Prop) :
    prCoin (fun π : (permPrefix n k).Ω => p ((permPrefix n k).draw π)) =
      prCoin (fun S : (subset n k hk).Ω => p ((subset n k hk).draw S)) := by
  classical
  unfold prCoin
  exact avg_comp_of_fiber (prefixOf hk) (card_prefixOf_eq hk) fun S : Draw.Subset n k => if p S.1 then 1 else 0

/-- **A uniform permutation's `k`-prefix misses a set as the uniform `k`-subset does.** -/
theorem missProb_permPrefix {k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    (permPrefix n k).missProb B = (subset n k hk).missProb B :=
  prCoin_permPrefix hk (Disjoint · B)

/-- **A uniform permutation's `k`-prefix misses a set of `j` units with probability `C(n − j, k)/C(n, k)`.** -/
theorem missProb_permPrefix_eq {k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    (permPrefix n k).missProb B = ((n - B.card).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := by
  rw [missProb_permPrefix hk, subset_escape]

end SamplingDesign

end Verity
