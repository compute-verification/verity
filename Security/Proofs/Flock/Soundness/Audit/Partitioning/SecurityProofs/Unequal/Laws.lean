import Proofs.Flock.Soundness.Audit.Partitioning.Model

/-!
# Two laws for the free-compute proofs

* `bern num den` draws unit `u` independently with probability `num u / den`. A set escapes with
  `∏_{u ∈ B} (1 − num u / den)` (`bern_escape_toReal`), and unit `u` is drawn with `num u / den` (`bern_incl_toReal`).
  `Law.bernoulli` is its case of one `num` for every unit.
* `restrict L S` is `L` seen on the units of `S` alone, numbered in increasing order. Its escape of `B` is `L`'s escape
  of `B`'s image (`restrict_escape`), so its expected number of draws is `∑_{u ∈ S} incl(u)` (`meanDraws_restrict`).
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal

open Finset Game
open scoped ENNReal

variable {n : ℕ}

theorem incl_le_one (L : Law n) (u : Fin n) : L.incl u ≤ 1 := by
  classical
  unfold Law.incl prCoin
  refine (avg_mono fun ω => ?_).trans (avg_const (C := L.Ω) 1).le
  split_ifs <;> simp

theorem incl_toReal_le_one (L : Law n) (u : Fin n) : (L.incl u).toReal ≤ 1 := by
  simpa using ENNReal.toReal_mono ENNReal.one_ne_top (incl_le_one L u)

theorem meanDraws_nonneg (L : Law n) : 0 ≤ meanDraws L := sum_nonneg fun _ _ => ENNReal.toReal_nonneg

theorem cost_nonneg (L : Law n) {c : Fin n → ℝ} (hc : ∀ u, 0 ≤ c u) : 0 ≤ cost L c :=
  sum_nonneg fun u _ => mul_nonneg ENNReal.toReal_nonneg (hc u)

/-! ## Independent draws with per-unit probabilities -/

/-- Each unit `u` drawn independently with probability `num u / den`. -/
abbrev bern (num : Fin n → ℕ) (den : ℕ) [NeZero den] : Law n where
  Ω := Fin n → Fin den
  draw ω := univ.filter fun u => (ω u).val < num u

theorem prCoin_fin_lt (den k : ℕ) [NeZero den] (hk : k ≤ den) :
    (prCoin fun x : Fin den => x.val < k).toReal = (k : ℝ) / den := by
  classical
  rw [prCoin_eq_card, Fin.card_filter_val_lt, Fintype.card_fin, min_eq_right hk, ENNReal.toReal_div,
    ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem prCoin_fin_not_lt (den k : ℕ) [NeZero den] (hk : k ≤ den) :
    (prCoin fun x : Fin den => ¬ x.val < k).toReal = 1 - (k : ℝ) / den := by
  rw [Law.prCoin_not, ENNReal.toReal_sub_of_le _ ENNReal.one_ne_top, ENNReal.toReal_one, prCoin_fin_lt den k hk]
  calc prCoin (fun x : Fin den => x.val < k) ≤ prCoin (fun _ : Fin den => True) := prCoin_mono fun _ _ => trivial
    _ = 1 := by
      unfold prCoin
      simp only [ite_true]
      exact avg_const 1

/-- **The per-unit Bernoulli escape**: `∏_{u ∈ B} (1 − num u / den)`. -/
theorem bern_escape_toReal {num : Fin n → ℕ} {den : ℕ} [NeZero den] (hnum : ∀ u, num u ≤ den)
    (B : Finset (Fin n)) : ((bern num den).escape B).toReal = ∏ u ∈ B, (1 - (num u : ℝ) / den) := by
  classical
  have h : (bern num den).escape B = ∏ u ∈ B, prCoin (fun x : Fin den => ¬ x.val < num u) := by
    rw [← prCoin_pi_forall (Ω := fun _ : Fin n => Fin den) B (fun u x => ¬ x.val < num u)]
    unfold Law.escape
    refine congrArg prCoin (funext fun ω => propext ?_)
    simp only [Finset.disjoint_left, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h u hu hlt => h hlt hu, fun h u hlt hu => h u hu hlt⟩
  rw [h, ENNReal.toReal_prod]
  exact Finset.prod_congr rfl fun u _ => prCoin_fin_not_lt den (num u) (hnum u)

theorem bern_incl_toReal {num : Fin n → ℕ} {den : ℕ} [NeZero den] (hnum : ∀ u, num u ≤ den) (u : Fin n) :
    ((bern num den).incl u).toReal = (num u : ℝ) / den := by
  classical
  have h : (bern num den).incl u = ∏ i ∈ {u}, prCoin (fun x : Fin den => x.val < num i) := by
    rw [← prCoin_pi_forall (Ω := fun _ : Fin n => Fin den) {u} (fun i x => x.val < num i)]
    unfold Law.incl
    refine congrArg prCoin (funext fun ω => propext ?_)
    simp
  rw [h, Finset.prod_singleton]
  exact prCoin_fin_lt den (num u) (hnum u)

/-! ## A law restricted to some of its units -/

/-- `L` on the units of `S` alone, numbered in increasing order. -/
def restrict (L : Law n) (S : Finset (Fin n)) : Law S.card where
  Ω := L.Ω
  draw ω := univ.filter fun i => S.orderEmbOfFin rfl i ∈ L.draw ω

theorem restrict_escape (L : Law n) (S : Finset (Fin n)) (B : Finset (Fin S.card)) :
    (restrict L S).escape B = L.escape (B.map (S.orderEmbOfFin rfl).toEmbedding) := by
  unfold Law.escape
  refine congrArg prCoin (funext fun ω => propext ?_)
  simp only [restrict, Finset.disjoint_left, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map,
    RelEmbedding.coe_toEmbedding, not_exists, not_and]
  exact ⟨fun h u hu i hi hiu => by subst hiu; exact h hu hi, fun h i hi hiB => h hi i hiB rfl⟩

theorem restrict_incl (L : Law n) (S : Finset (Fin n)) (i : Fin S.card) :
    (restrict L S).incl i = L.incl (S.orderEmbOfFin rfl i) := by
  unfold Law.incl
  refine congrArg prCoin (funext fun ω => propext ?_)
  simp [restrict]

theorem meanDraws_restrict (L : Law n) (S : Finset (Fin n)) :
    meanDraws (restrict L S) = ∑ u ∈ S, (L.incl u).toReal := by
  unfold meanDraws
  simp only [restrict_incl]
  conv_rhs => rw [← Finset.map_orderEmbOfFin_univ S rfl]
  rw [Finset.sum_map]
  rfl

/-- A set of the restricted law is a set of `L`'s units of `S`, of the same size, doing the same work. -/
theorem sum_map_restrict (S : Finset (Fin n)) (B : Finset (Fin S.card)) (w : Fin n → ℝ) :
    ∑ u ∈ B.map (S.orderEmbOfFin rfl).toEmbedding, w u = ∑ i ∈ B, w (S.orderEmbOfFin rfl i) := by
  rw [Finset.sum_map]
  rfl

theorem orderEmbOfFin_mem' (S : Finset (Fin n)) (i : Fin S.card) : S.orderEmbOfFin rfl i ∈ S :=
  Finset.orderEmbOfFin_mem S rfl i

end FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal
