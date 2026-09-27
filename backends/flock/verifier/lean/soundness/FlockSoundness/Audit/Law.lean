import FlockSoundness.Game.Basic
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Nat.Choose.Basic

/-!
# Draw laws

A draw law is the verifier's coin space `Ω`, uniform, and the set of units each coin draws. What a
law gives a consumer is its escape function: the probability that a draw misses a set `B` of units.

* `escape L B`: the draw is disjoint from `B`;
* `miss L K`: the largest escape over sets of at least `K` units, the sampling term of the count
  curve `δ(K) = miss K + ε`;
* `incl L u`: the probability that `u` is drawn; a single unit escapes with `1 - incl L u`
  (`escape_singleton`), and a set escapes with at most that for each of its members
  (`escape_le_of_mem`).

`subset n k` is the uniform `k`-subset of `n` units, whose escape is hypergeometric:
`C(n - |B|, k) / C(n, k)` (`subset_escape`), so its count curve is `C(n - K, k) / C(n, k)`
(`subset_miss`).
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

/-- A draw law over `n` units: the verifier's uniform coins, and the unit set each draws. -/
structure Law (n : ℕ) where
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  draw : Ω → Finset (Fin n)

attribute [instance] Law.fintype Law.nonempty

namespace Law

variable {n : ℕ}

/-- The probability that a draw misses every unit of `B`. -/
noncomputable def escape (L : Law n) (B : Finset (Fin n)) : ℝ≥0∞ :=
  prCoin fun ω => Disjoint (L.draw ω) B

/-- The count curve's sampling term: the largest escape over sets of at least `K` units. -/
noncomputable def miss (L : Law n) (K : ℕ) : ℝ≥0∞ :=
  ⨆ (B : Finset (Fin n)) (_ : K ≤ B.card), L.escape B

/-- The inclusion probability of a unit. -/
noncomputable def incl (L : Law n) (u : Fin n) : ℝ≥0∞ :=
  prCoin fun ω => u ∈ L.draw ω

theorem prCoin_not {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) :
    prCoin (fun c => ¬ p c) = 1 - prCoin p := by
  classical
  have h : prCoin p + prCoin (fun c => ¬ p c) = 1 := by
    unfold prCoin
    rw [← avg_add]
    refine (congrArg avg (funext fun c => ?_)).trans (avg_const 1)
    by_cases hc : p c <;> simp [hc]
  have hne : prCoin p ≠ ∞ := ne_top_of_le_ne_top ENNReal.one_ne_top (h ▸ le_self_add)
  exact ENNReal.eq_sub_of_add_eq hne (by rw [add_comm]; exact h)

theorem escape_le_one (L : Law n) (B : Finset (Fin n)) : L.escape B ≤ 1 := by
  classical
  unfold escape prCoin
  refine (avg_mono fun ω => ?_).trans (avg_const (C := L.Ω) 1).le
  split_ifs <;> simp

/-- Missing more units is less likely. -/
theorem escape_anti (L : Law n) {B B' : Finset (Fin n)} (h : B ⊆ B') : L.escape B' ≤ L.escape B :=
  prCoin_mono fun _ hd => hd.mono_right h

theorem escape_empty (L : Law n) : L.escape ∅ = 1 := by
  unfold escape prCoin
  simp only [Finset.disjoint_empty_right, ite_true]
  exact avg_const 1

/-- A single unit escapes exactly when it is not drawn. -/
theorem escape_singleton (L : Law n) (u : Fin n) : L.escape {u} = 1 - L.incl u := by
  unfold escape incl
  rw [← prCoin_not]
  exact congrArg prCoin (funext fun ω => propext Finset.disjoint_singleton_right)

/-- A set escapes with at most the escape of any one of its units. -/
theorem escape_le_of_mem (L : Law n) {B : Finset (Fin n)} {u : Fin n} (hu : u ∈ B) :
    L.escape B ≤ 1 - L.incl u :=
  (L.escape_anti (Finset.singleton_subset_iff.2 hu)).trans (L.escape_singleton u).le

theorem escape_le_miss (L : Law n) {K : ℕ} {B : Finset (Fin n)} (h : K ≤ B.card) :
    L.escape B ≤ L.miss K :=
  le_iSup₂_of_le (f := fun (B : Finset (Fin n)) (_ : K ≤ B.card) => L.escape B) B h le_rfl

theorem miss_le_one (L : Law n) (K : ℕ) : L.miss K ≤ 1 :=
  iSup₂_le fun B _ => L.escape_le_one B

/-- The count curve falls as `K` grows. -/
theorem miss_anti (L : Law n) {K K' : ℕ} (h : K ≤ K') : L.miss K' ≤ L.miss K :=
  iSup₂_le fun _ hB => L.escape_le_miss (h.trans hB)

/-! ## The uniform `k`-subset -/

/-- The uniform `k`-subset of `n` units. -/
def subset (n k : ℕ) (hk : k ≤ n) : Law n where
  Ω := {S : Finset (Fin n) // S.card = k}
  nonempty := by
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hk)
    exact ⟨⟨S, hS⟩⟩
  draw := Subtype.val

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

/-- **The hypergeometric escape.** A uniform `k`-subset misses `B` with probability
`C(n - |B|, k) / C(n, k)`. -/
theorem subset_escape {k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    (subset n k hk).escape B = ((n - B.card).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := by
  classical
  unfold escape
  rw [prCoin_eq_card]
  congr 1
  · exact congrArg _ (card_disjoint_subsets k B)
  · exact congrArg _ ((Fintype.card_finset_len k).trans (by rw [Fintype.card_fin]))

/-- **The uniform `k`-subset's count curve**, exactly: `miss K = C(n - K, k) / C(n, k)`. -/
theorem subset_miss {k : ℕ} (hk : k ≤ n) {K : ℕ} (hK : K ≤ n) :
    (subset n k hk).miss K = ((n - K).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := by
  refine le_antisymm (iSup₂_le fun B hB => ?_) ?_
  · rw [subset_escape]
    gcongr
  · obtain ⟨B, -, hB⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hK)
    have := (subset n k hk).escape_le_miss (K := K) hB.ge
    rwa [subset_escape, hB] at this

end Law

end FlockSoundness.Audit
