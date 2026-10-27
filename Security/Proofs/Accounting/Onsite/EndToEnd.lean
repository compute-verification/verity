import Definitions.Accounting.Guarantees.Onsite.EndToEnd
import Proofs.Accounting.Actual.Dampener
import Proofs.Accounting.JointLive

/-!
# The onsite end-to-end overhead guarantee

`OnsiteTimeLowerBound` is `DampenerWindowsCap` at the onsite class's hypotheses, with `M_d = 0` and `M_r = M`,
inverted through `⌈t/ω⌉₊ ω ≤ t + ω − 1` (`ceil_div_eq_pred_div`); its size form sums the compute rate over the servers.
`OnsiteEndToEnd` is the spine's step 1 (`endToEnd_local`) with each of its three terms bounded: the claims' failure by
`JointPeriodLive`, the class's by the bridge, and a cheap success in the class by coverage, through
`OnsiteTimeLowerBound`.
-/

namespace Accounting.SecurityProofs

open Verity Verity.Game Finset

/-- **The spine's step 1, the event decomposition** (note:training-20261027T0134Z-draft-e2e-overhead-proof): at a
strategy `s`, `Pr[acc ∧ succ ∧ cheap] ≤ Pr[acc ∧ ¬claims] + Pr[acc ∧ claims ∧ ¬cls] + Pr[cls ∧ succ ∧ cheap]`, with
no independence. A local proof: W-core's generic `EndToEnd` (branch `cursor/overhead-core-4207`) replaces it once it
lands. -/
theorem endToEnd_local {O : Type} (G : Game O) (s : Strategy G) (acc claims cls succ cheap : O → Prop) :
    prob (fun o => acc o ∧ succ o ∧ cheap o) G s ≤
      prob (fun o => acc o ∧ ¬ claims o) G s + prob (fun o => acc o ∧ claims o ∧ ¬ cls o) G s +
        prob (fun o => cls o ∧ succ o ∧ cheap o) G s := by
  calc prob (fun o => acc o ∧ succ o ∧ cheap o) G s
      ≤ prob (fun o => (acc o ∧ ¬ claims o) ∨ (acc o ∧ claims o ∧ ¬ cls o) ∨ (cls o ∧ succ o ∧ cheap o)) G s :=
        prob_mono (fun o ⟨ha, hs, hc⟩ => by
          by_cases hcl : claims o
          · by_cases hk : cls o
            · exact Or.inr (Or.inr ⟨hk, hs, hc⟩)
            · exact Or.inr (Or.inl ⟨ha, hcl, hk⟩)
          · exact Or.inl ⟨ha, hcl⟩) G s
    _ ≤ prob (fun o => acc o ∧ ¬ claims o) G s +
          (prob (fun o => acc o ∧ claims o ∧ ¬ cls o) G s + prob (fun o => cls o ∧ succ o ∧ cheap o) G s) :=
        (prob_or_le _ _ G s).trans (add_le_add le_rfl (prob_or_le _ _ G s))
    _ = _ := (add_assoc _ _ _).symm

namespace Onsite

variable {L E : Type*}

/-- `[0, t)` as an interval from 0. -/
theorem Ico_zero_eq_Iio (t : ℕ) : Set.Ico 0 t = Set.Iio t := by
  ext x
  simp

/-- **The servers' gates over `[0, t)` number at most `Φ_total t`**: the region of the servers' union is in the union of
their regions, each within its compute rate at unit price. -/
theorem region_card_le [DecidableEq L] (P : Onsite L E) (A : Placed L) (t : ℕ)
    (hcr : A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) :
    (A.region (univ.biUnion P.X) (Set.Iio t)).card ≤ P.computeTotal * t := by
  classical
  have hsub : A.region (univ.biUnion P.X) (Set.Iio t) ⊆ univ.biUnion fun i => A.region (P.X i) (Set.Iio t) := by
    intro g hg
    simp only [Placed.region, mem_filter, mem_biUnion, mem_univ, true_and] at hg ⊢
    obtain ⟨hgc, ⟨i, hi⟩, ht⟩ := hg
    exact ⟨i, hgc, hi, ht⟩
  have hi : ∀ i, (A.region (P.X i) (Set.Iio t)).card ≤ P.hw.compute (P.X i) * t := by
    intro i
    have h := hcr (P.X i) ⟨i, rfl⟩ 0 t
    rw [Ico_zero_eq_Iio, Nat.sub_zero] at h
    simpa [Placed.work] using h
  calc (A.region (univ.biUnion P.X) (Set.Iio t)).card
      ≤ (univ.biUnion fun i => A.region (P.X i) (Set.Iio t)).card := card_le_card hsub
    _ ≤ ∑ i, (A.region (P.X i) (Set.Iio t)).card := card_biUnion_le
    _ ≤ ∑ i, P.hw.compute (P.X i) * t := sum_le_sum fun i _ => hi i
    _ = P.computeTotal * t := by rw [Onsite.computeTotal, sum_mul]

/-- **The windows' bound inverted**: from `b_a M ≤ n ⌈t/ω⌉₊ K Λ(K)` with `ω > 0`, `b_a > 0` and `N ≤ M`, the horizon is at
least `t_min = ω (b_a N/(n K Λ(K)) − 1) + 1`. When `N > 0` the bound forces `n K Λ(K) > 0` and
`⌈t/ω⌉₊ ≥ b_a N/(n K Λ(K))`, and `⌈t/ω⌉₊ ω ≤ t + ω − 1`; when `N = 0`, `t_min = 1 − ω ≤ 0`. -/
theorem tMin_le (P : Onsite L E) {t M N : ℕ} (hω : 0 < P.ω) (hba : 0 < P.b_a) (hN : N ≤ M)
    (h : P.b_a * (M : ℝ) ≤ P.n * ⌈(t : ℝ) / P.ω⌉₊ * (P.K * P.Λ)) : P.tMin N ≤ t := by
  set c := ⌈(t : ℝ) / P.ω⌉₊ with hc
  have hcω : c * P.ω ≤ t + P.ω - 1 := by
    rw [hc, Actual.ceil_div_eq_pred_div hω]
    exact Nat.div_mul_le_self _ _
  have hω1 : (1 : ℝ) ≤ P.ω := by exact_mod_cast hω
  have hcωR : (c : ℝ) * P.ω ≤ t + P.ω - 1 := by
    have h1 : ((t + P.ω - 1 : ℕ) : ℝ) = t + P.ω - 1 := by
      rw [Nat.cast_sub (by omega : 1 ≤ t + P.ω)]
      push_cast
      ring
    rw [← h1]
    exact_mod_cast hcω
  unfold Onsite.tMin
  rcases Nat.eq_zero_or_pos N with rfl | hNpos
  · simp only [Nat.cast_zero, mul_zero, zero_div, zero_sub, mul_neg, mul_one]
    have : (0 : ℝ) ≤ t := Nat.cast_nonneg t
    linarith
  · set D := (P.n : ℝ) * P.K * P.Λ with hD
    have hbN : 0 < P.b_a * (N : ℝ) := mul_pos hba (by exact_mod_cast hNpos)
    have hle : P.b_a * (N : ℝ) ≤ c * D := by
      have hNM : P.b_a * (N : ℝ) ≤ P.b_a * M := mul_le_mul_of_nonneg_left (by exact_mod_cast hN) hba.le
      calc P.b_a * (N : ℝ) ≤ P.b_a * M := hNM
        _ ≤ P.n * c * (P.K * P.Λ) := h
        _ = c * D := by rw [hD]; ring
    have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg c
    have hDpos : 0 < D := by
      by_contra hD0
      have : (c : ℝ) * D ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hc0 (not_lt.1 hD0)
      linarith
    have hx : P.b_a * N / D ≤ c := (div_le_iff₀ hDpos).2 hle
    have hω0 : (0 : ℝ) ≤ P.ω := by linarith
    calc (P.ω : ℝ) * (P.b_a * N / D - 1) + 1 ≤ P.ω * ((c : ℝ) - 1) + 1 := by
          have := mul_le_mul_of_nonneg_left (sub_le_sub_right hx 1) hω0
          linarith
      _ = (c : ℝ) * P.ω - P.ω + 1 := by ring
      _ ≤ t := by linarith

end Onsite

/-- **The onsite time floor** (`Accounting.Guarantees.OnsiteTimeLowerBound`): `DampenerWindowsCap` on the class with
`M_r = M`, `M_d = 0`, inverted (`Onsite.tMin_le`), and the compute rate summed over the servers
(`Onsite.region_card_le`). -/
theorem OnsiteTimeLowerBound : Accounting.Guarantees.OnsiteTimeLowerBound := by
  intro L E _ _ P A t M N hC hW hN
  obtain ⟨hω, hX, hca, hcr, ⟨link, hL⟩, hbw, hba, hd, hΓ, hcap, hR, ⟨θ, hθ, hcont⟩, hT, hM⟩ := hC
  have h := Actual.DampenerWindowsCap A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X θ hθ t P.ω P.Γ P.R P.F
    P.α P.b_w P.b_a P.c P.T P.d P.e M 0 hω hX hca hcr hL hbw hba hd hΓ hcap hR hcont hT hW (by simpa using hM)
  refine ⟨Onsite.tMin_le P hω hba hN ?_, Onsite.region_card_le P A t hcr⟩
  simpa [Onsite.K, Onsite.Λ] using h

/-- **The onsite end-to-end guarantee** (`Accounting.Guarantees.OnsiteEndToEnd`): step 1's decomposition
(`endToEnd_local`), at `acc` every subprotocol accepting with no break, `claims` every claim holding and `cls` the
onsite class, whose three terms are at most `∑ δ` (`JointPeriodLive`), `δ_b` (`OnsiteBridge`) and `η`
(`OnsiteCoverage`, through `OnsiteTimeLowerBound`: in the class, `M ≥ N` puts the horizon at `t_min` or above). -/
theorem OnsiteEndToEnd : Accounting.Guarantees.OnsiteEndToEnd := by
  intro ι α _ d C Cm O r G c p δ L E _ _ P A t M success N δb η hδ hB hCov hW s
  have h1 := endToEnd_local G s (fun o => (∀ a, r.accepts a o) ∧ ¬ r.Break G s o)
    (fun o => ∀ a, r.Holds c p G s a o) (fun o => P.Class (A s o) (t s o) (M s o)) (success s)
    (fun o => (t s o : ℝ) < P.tMin N)
  have hJ : prob (fun o => ((∀ a, r.accepts a o) ∧ ¬ r.Break G s o) ∧ ¬ ∀ a, r.Holds c p G s a o) G s ≤
      ∑ a, ENNReal.ofReal (δ a) :=
    (prob_mono (fun o ⟨⟨ha, hb⟩, hh⟩ => by
      obtain ⟨a, hn⟩ := not_forall.1 hh
      exact ⟨⟨a, ha a, hn⟩, hb⟩) G s).trans (JointPeriodLive ι α d C Cm O r G c p δ hδ s)
  have hBr : prob (fun o => ((∀ a, r.accepts a o) ∧ ¬ r.Break G s o) ∧ (∀ a, r.Holds c p G s a o) ∧
      ¬ P.Class (A s o) (t s o) (M s o)) G s ≤ ENNReal.ofReal δb :=
    (prob_mono (fun o ⟨⟨ha, hb⟩, hh, hk⟩ => ⟨ha, hh, hb, hk⟩) G s).trans (hB s)
  have hCv : prob (fun o => P.Class (A s o) (t s o) (M s o) ∧ success s o ∧ (t s o : ℝ) < P.tMin N) G s ≤
      ENNReal.ofReal η :=
    (prob_mono (fun o ⟨hk, hs, ht⟩ => ⟨hk, hs, by
      by_contra hlt
      exact absurd (OnsiteTimeLowerBound P (A s o) (t s o) (M s o) N hk hW (not_lt.1 hlt)).1 (not_le.2 ht)⟩) G s).trans
      (hCov s)
  calc prob (fun o => (∀ a, r.accepts a o) ∧ success s o ∧ (t s o : ℝ) < P.tMin N ∧ ¬ r.Break G s o) G s
      ≤ prob (fun o => ((∀ a, r.accepts a o) ∧ ¬ r.Break G s o) ∧ success s o ∧ (t s o : ℝ) < P.tMin N) G s :=
        prob_mono (fun o ⟨ha, hs, ht, hb⟩ => ⟨⟨ha, hb⟩, hs, ht⟩) G s
    _ ≤ _ := h1
    _ ≤ ∑ a, ENNReal.ofReal (δ a) + ENNReal.ofReal δb + ENNReal.ofReal η := add_le_add (add_le_add hJ hBr) hCv

end Accounting.SecurityProofs
