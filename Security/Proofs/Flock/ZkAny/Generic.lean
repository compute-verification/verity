module

public import Definitions.Flock.Guarantees.ZkAny
public import Proofs.Core.Game.Basic
public import Proofs.Flock.Soundness.ZK.Hiding

/-!
# Zero knowledge at any values: the generic forms

* `prCoin_or_le`: the union bound for two events.
* `card_mul_avg`: `|C|` times the average over `C` is the sum.
* `Flock.SecurityProofs.ZkAnyOutcome`: each outcome's branch bounds the release jointly with that outcome
  (`prCoin_mono`); the union bound over outcomes (`prCoin_exists_le`) gives the sum, which is `|St|` times the average
  over a uniform outcome (`prCoin_prod_split`).
* `Flock.SecurityProofs.ZkAnyGate`: two outcomes, the check passing or not.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **The union bound for two events.** -/
theorem prCoin_or_le {C : Type} [Fintype C] (p q : C → Prop) :
    prCoin (fun c => p c ∨ q c) ≤ prCoin p + prCoin q := by
  classical
  unfold prCoin
  rw [← avg_add]
  refine avg_mono fun c => ?_
  by_cases hp : p c
  · rw [ite_eq_left (Or.inl hp), ite_eq_left hp]
    exact le_self_add
  · by_cases hq : q c
    · rw [ite_eq_left (Or.inr hq), ite_eq_right hp, ite_eq_left hq, zero_add]
    · rw [ite_eq_right (not_or.2 ⟨hp, hq⟩)]
      exact zero_le

/-- **`|C|` times the average over `C` is the sum**, `C` empty or not. -/
theorem card_mul_avg {C : Type} [Fintype C] (g : C → ℝ≥0∞) : (Fintype.card C : ℝ≥0∞) * avg g = ∑ c, g c := by
  unfold avg
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  have h : (Fintype.card C : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.2 (Fintype.card_pos_iff.2 ⟨c⟩).ne'
  rw [← mul_assoc, ENNReal.mul_inv_cancel h (ENNReal.natCast_ne_top _), one_mul]

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Verity.Game FlockSoundness.ZK
open scoped ENNReal

/-- **Zero knowledge with a public outcome, generic** (`Flock.Guarantees.ZkAnyOutcome`). -/
theorem ZkAnyOutcome : Flock.Guarantees.ZkAnyOutcome := by
  intro Ω St Ωs V _ _ _ rel out B S ε hrel hB E
  have hper : ∀ s, prCoin (fun ω => out ω = s ∧ E (rel ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s := fun s =>
    (prCoin_mono fun ω h => by
      obtain ⟨hs, hE⟩ := h
      rw [hrel ω, hs] at hE
      exact hE).trans (hB s E)
  have hsum : prCoin (fun ω => E (rel ω)) ≤ ∑ s, (prCoin (fun σ => E (S s σ)) + ε s) :=
    (prCoin_mono fun ω h => ⟨out ω, Finset.mem_univ _, rfl, h⟩).trans
      ((prCoin_exists_le Finset.univ fun s ω => out ω = s ∧ E (rel ω)).trans (Finset.sum_le_sum fun s _ => hper s))
  refine ⟨hper, hsum, hsum.trans (le_of_eq ?_)⟩
  rw [Finset.sum_add_distrib, prCoin_prod_split (fun s σ => E (S s σ)), card_mul_avg]

/-- **Check-then-release, generic** (`Flock.Guarantees.ZkAnyGate`). -/
theorem ZkAnyGate : Flock.Guarantees.ZkAnyGate := by
  intro Ω Ωf Ωc V P _ _ _ view d cut go simFull simCut εf εc hcut hd hgo E
  have h1 : prCoin (fun ω => go ω = true ∧ E (.inl (view ω))) ≤ prCoin (fun σ => E (.inl (simFull σ))) + εf :=
    (prCoin_mono fun ω h => by
      obtain ⟨hg, hE⟩ := h
      rw [hgo ω hg] at hE
      exact hE).trans (hd fun v => E (.inl v))
  have h2 : prCoin (fun ω => go ω = false ∧ E (.inr (cut ω))) ≤ prCoin (fun σ => E (.inr (simCut σ))) + εc :=
    (prCoin_mono fun ω h => h.2).trans (hcut fun v => E (.inr v))
  have hor : prCoin (fun ω => E (if go ω then .inl (view ω) else .inr (cut ω))) ≤
      prCoin (fun ω => (go ω = true ∧ E (.inl (view ω))) ∨ (go ω = false ∧ E (.inr (cut ω)))) := by
    refine prCoin_mono fun ω (h : E (if go ω then .inl (view ω) else .inr (cut ω))) => ?_
    show (go ω = true ∧ E (.inl (view ω))) ∨ (go ω = false ∧ E (.inr (cut ω)))
    by_cases hg : go ω = true
    · exact Or.inl ⟨hg, by rwa [ite_eq_left hg] at h⟩
    · exact Or.inr ⟨by simpa using hg, by rwa [ite_eq_right hg] at h⟩
  exact ⟨h1, h2, hor.trans ((prCoin_or_le _ _).trans ((add_le_add h1 h2).trans_eq (add_add_add_comm _ _ _ _)))⟩

end Flock.SecurityProofs
