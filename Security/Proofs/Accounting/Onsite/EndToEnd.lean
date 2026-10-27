import Definitions.Accounting.Guarantees.Onsite.EndToEnd
import Proofs.Accounting.Actual.Dampener
import Proofs.Accounting.JointLive
import Proofs.Accounting.Onsite
import Proofs.Accounting.Overhead

/-!
# The onsite end-to-end overhead guarantee

`OnsiteTimeLowerBound` is `DampenerWindowsCap` at the onsite class and valid parameters, with `M_r = M` and `M_d = 0`,
inverted through `⌈t/ω⌉₊ ω ≤ t + ω − 1` (`ceil_div_eq_pred_div`); its size form sums the compute rate over the servers.

`OnsiteEndToEnd` is the spine's step 1 (`Accounting.Guarantees.Overhead.EndToEnd`) with accepting every subprotocol's
and every wipe's acceptance with no break, the claims every subprotocol's claim and every wipe's survivor bound, the
class `OnsiteClass` and the cost the horizon `t` against `t_min`. Its four hypotheses are `JointPeriodLive` with the
wipes' errors, the bridges (`Accounting.Guarantees.Onsite.ClaimsImplyOnsiteClass`, at the strategies where
`IntegrityContains` holds, and the trivial bound `⊤` at the others), `OnsiteTimeLowerBound` and coverage.
-/

namespace Accounting.SecurityProofs

open Verity Verity.Game Finset

namespace OnsiteTime

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

end OnsiteTime

/-- **The onsite time floor** (`Accounting.Guarantees.OnsiteTimeLowerBound`): `DampenerWindowsCap` on the class with
`M_r = M`, `M_d = 0`, inverted (`OnsiteTime.tMin_le`), and the compute rate summed over the servers
(`OnsiteTime.region_card_le`). -/
theorem OnsiteTimeLowerBound : Accounting.Guarantees.OnsiteTimeLowerBound := by
  intro L E _ _ P A link θ hθ t M N hV hC hW hM hN
  obtain ⟨hω, hX, hbw, hba, hd, hΓ, hT⟩ := hV
  obtain ⟨hca, hcr, hL, hcap, hR, hcont⟩ := hC
  have h := Actual.DampenerWindowsCap A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X θ hθ t P.ω P.Γ P.R P.F
    P.α P.b_w P.b_a P.c P.T P.d P.e M 0 hω hX hca hcr hL hbw hba hd hΓ hcap hR hcont hT hW (by simpa using hM)
  refine ⟨OnsiteTime.tMin_le P hω hba hN ?_, OnsiteTime.region_card_le P A t hcr⟩
  simpa [Onsite.K, Onsite.Λ] using h

/-- **The onsite end-to-end guarantee** (`Accounting.Guarantees.OnsiteEndToEnd`): step 1's decomposition
(`Overhead.EndToEnd`) at every subprotocol and every wipe accepting with no break, every claim and survivor bound holding
and the onsite class. Its terms are at most `∑ δ + ∑ ε_wipe` (`JointPeriodLive` and the wipes' bounds, by the union
bound), `0` (`Onsite.ClaimsImplyOnsiteClass` at `s`, where `IntegrityContains` holds) and `η` (coverage, with
`OnsiteTimeLowerBound` putting the horizon at `t_min` or above in the class once `M ≥ N`). -/
theorem OnsiteEndToEnd : Accounting.Guarantees.OnsiteEndToEnd := by
  classical
  intro ι Acc _ d C Cm O r G c p δ m w f εWipe L E _ _ P S link θ hθ t M success N η hV hWI hM hδ hε hC hΦ hL hWS
    hCov s hI
  have h := Overhead.EndToEnd O G
    (fun s o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ ¬ r.Break G s o)
    (fun s o => (∀ a, r.Holds c p G s a o) ∧ ∀ k, (w k).Survives f o)
    (fun _ o => OnsiteClass (S o).A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β (link o) P.n P.X (θ o) (hθ o) t P.ω P.R P.F
      P.α P.c)
    (fun _ o => success o) (fun _ o => (M o : ℝ)) (fun _ _ => (t : ℝ)) (N : ℝ) (P.tMin N)
    (fun _ => ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k))
    (fun s' => if Assumptions.IntegrityContains r G s' c p S P.X t θ hθ P.c then 0 else ⊤)
    (fun _ => ENNReal.ofReal η) ?_ ?_ ?_ ?_ s
  · beta_reduce at h
    simp only [hI, ↓reduceIte, add_zero] at h
    exact (prob_mono (fun o ⟨ha, hs, ht, hb⟩ => ⟨⟨ha, hb⟩, hs, ht⟩) G s).trans h
  · intro s'
    calc prob (fun o => (((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ ¬ r.Break G s' o) ∧
          ¬ ((∀ a, r.Holds c p G s' a o) ∧ ∀ k, (w k).Survives f o)) G s'
        ≤ prob (fun o => ((∃ a, r.accepts a o ∧ ¬ r.Holds c p G s' a o) ∧ ¬ r.Break G s' o) ∨
            ∃ k, (w k).accepts o ∧ ¬ (w k).Survives f o) G s' :=
          prob_mono (fun o ⟨⟨⟨ha, hw⟩, hb⟩, hn⟩ => by
            by_cases hh : ∀ a, r.Holds c p G s' a o
            · obtain ⟨k, hk⟩ := not_forall.1 fun hs => hn ⟨hh, hs⟩
              exact Or.inr ⟨k, hw k, hk⟩
            · obtain ⟨a, ha'⟩ := not_forall.1 hh
              exact Or.inl ⟨⟨a, ha a, ha'⟩, hb⟩) G s'
      _ ≤ ∑ a, ENNReal.ofReal (δ a) + ∑ k, prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s' :=
          (prob_or_le _ _ G s').trans (add_le_add (JointPeriodLive ι Acc d C Cm O r G c p δ hδ s')
            (Live.prob_exists_univ_le (fun k o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s'))
      _ ≤ _ := add_le_add le_rfl (sum_le_sum fun k _ => hε k s')
  · intro s'
    split_ifs with hI'
    · exact (Onsite.ClaimsImplyOnsiteClass d C Cm O r G c p m w f S P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X
        θ hθ t P.ω P.R P.F P.α P.c s' hC hΦ hL hWS hI').le
    · exact le_top
  · intro _ o hk hNM
    exact (OnsiteTimeLowerBound P (S o).A (link o) (θ o) (hθ o) t (M o) N hV hk hWI (hM o)
      (by exact_mod_cast hNM)).1
  · exact fun s' => (prob_mono (fun o ⟨hk, hs, hlt⟩ => ⟨hk, hs, by exact_mod_cast hlt⟩) G s').trans (hCov s')

end Accounting.SecurityProofs
