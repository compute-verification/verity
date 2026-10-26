import Proofs.Flock.Soundness.Audit.HarmBound

/-!
# The stratified law, one stratum at a time

`Law.stratified σ k hk` draws an independent uniform `k s`-subset inside each stratum `s` of `σ`; its coin `ω` holds
them, stratum `s`'s at `ω s`. With `n_s = |stratum σ s|` and `m_s = |B.filter (σ · = s)|`:

* `stratified_proj_escape`: **the per-stratum projection**. Stratum `s`'s coordinate of the coin, `(ω s).1`, misses a
  set `B` with stratum `s`'s hypergeometric factor `C(n_s − m_s, k_s) / C(n_s, k_s)`, the factor of `s` in
  `stratified_escape`. A consumer that draws one stratum by itself (PoUW's tile stratum) cites this for its coin.
* `stratified_proj_escape_eq_exp`: the same factor is `exp(−stratumRisk σ k s m_s)` when `m_s + k_s ≤ n_s`: stratum
  `s`'s exact risk is minus the log of its coordinate's escape.
* `stratified_escape_eq_exp`: so `B`'s escape is `exp(−∑ s stratumRisk σ k s m_s)`. The detection budget a set spends,
  exactly, is the sum of its strata's exact risks, each at least its stratum's term `k_s m_s / n_s` of the closed form
  (`le_stratumRisk`).
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- **The per-stratum projection**: stratum `s`'s coordinate of the stratified law's coin misses `B` with stratum `s`'s
hypergeometric factor, `C(n_s − m_s, k_s) / C(n_s, k_s)` with `n_s = |stratum σ s|` and `m_s = |B.filter (σ · = s)|`. -/
theorem stratified_proj_escape (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (s : Fin m) (B : Finset (Fin n)) :
    prCoin (fun ω : (stratified σ k hk).Ω => Disjoint (ω s).1 B) =
      (((stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ≥0∞) /
        ((stratum σ s).card.choose (k s) : ℝ≥0∞) := by
  classical
  have : ∀ s, Nonempty ((stratum σ s).powersetCard (k s)) :=
    fun s => ⟨⟨_, (Finset.powersetCard_nonempty.2 (hk s)).choose_spec⟩⟩
  have hd : (fun ω : (stratified σ k hk).Ω => Disjoint (ω s).1 B) =
      fun ω => ∀ t ∈ ({s} : Finset (Fin m)), Disjoint (ω t).1 B := by
    funext ω
    refine propext ⟨fun h t ht => ?_, fun h => h s (Finset.mem_singleton_self s)⟩
    obtain rfl := Finset.mem_singleton.1 ht
    exact h
  rw [hd]
  refine (prCoin_pi_forall (Ω := fun s => ((stratum σ s).powersetCard (k s))) {s}
    (fun s S => Disjoint S.1 B)).trans ?_
  rw [Finset.prod_singleton]
  refine (prCoin_coe ((stratum σ s).powersetCard (k s)) (fun S => Disjoint S B)).trans ?_
  rw [card_powersetCard_disjoint, card_powersetCard, filter_eq_inter_stratum]

/-- `C(N − c, k) / C(N, k) = exp(−ln(C(N, k) / C(N − c, k)))` when `c + k ≤ N`. -/
theorem choose_div_eq_exp {N k c : ℕ} (h : c + k ≤ N) :
    ((N - c).choose k : ℝ≥0∞) / (N.choose k : ℝ≥0∞) =
      ENNReal.ofReal (Real.exp (-Real.log ((N.choose k : ℝ) / ((N - c).choose k : ℝ)))) := by
  have ha : (0 : ℝ) < (N.choose k : ℝ) := by exact_mod_cast Nat.choose_pos (by omega)
  have hb : (0 : ℝ) < ((N - c).choose k : ℝ) := by exact_mod_cast Nat.choose_pos (by omega)
  rw [Real.exp_neg, Real.exp_log (div_pos ha hb), inv_div, ENNReal.ofReal_div_of_pos ha, ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

/-- **Stratum `s`'s exact risk is minus the log of its coordinate's escape**: with `m_s + k_s ≤ n_s`, stratum `s`'s
coordinate misses `B` with `exp(−stratumRisk σ k s m_s)`. -/
theorem stratified_proj_escape_eq_exp (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (s : Fin m) (B : Finset (Fin n)) (hc : (B.filter (σ · = s)).card + k s ≤ (stratum σ s).card) :
    prCoin (fun ω : (stratified σ k hk).Ω => Disjoint (ω s).1 B) =
      ENNReal.ofReal (Real.exp (-stratumRisk σ k s (B.filter (σ · = s)).card)) := by
  rw [stratified_proj_escape]
  unfold stratumRisk
  exact choose_div_eq_exp hc

/-- **The stratified escape in exact risks**: with `m_s + k_s ≤ n_s` in every stratum, `B` escapes with
`exp(−∑ s stratumRisk σ k s m_s)`, the product of its strata's coordinates' escapes. -/
theorem stratified_escape_eq_exp (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (B : Finset (Fin n)) (hc : ∀ s, (B.filter (σ · = s)).card + k s ≤ (stratum σ s).card) :
    (stratified σ k hk).escape B =
      ENNReal.ofReal (Real.exp (-∑ s, stratumRisk σ k s (B.filter (σ · = s)).card)) := by
  rw [stratified_escape, ← Finset.sum_neg_distrib, Real.exp_sum,
    ENNReal.ofReal_prod_of_nonneg (f := fun s => Real.exp (-stratumRisk σ k s (B.filter (σ · = s)).card))
      (fun s _ => (Real.exp_pos _).le)]
  refine Finset.prod_congr rfl fun s _ => ?_
  unfold stratumRisk
  exact choose_div_eq_exp (hc s)

end Law

end FlockSoundness.Audit
