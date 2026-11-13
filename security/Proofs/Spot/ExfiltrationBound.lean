module

public import Definitions.Spot.Dataflow
public import Definitions.Spot.Exfiltration
public import Proofs.Spot.Exfiltration
public import Proofs.Spot.Leakage
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv
import Mathlib.Logic.Equiv.PartialEquiv
import Proofs.Spot.Determinacy
meta import Lean.Meta.Tactic.NormCast

/-!
# Exfiltration before detection: from the sample and the population to the probability of theft

* `indepSample_acceptCoins`: the sample that catches unit `u` independently, when its coin `p u` comes up `true`, lets the
  incorrect set `B` escape with probability exactly `Π_{u ∈ B} P(p u = false)`.
* `pressure_of_sample`, `pressure_indepSample`: **detection pressure from the sample and the cone.** A window whose leak is at
  most its incorrect proof blocks' cone egress `Σ_{u ∈ B} E(u)`, under a sample that catches `u` with probability
  at least `κ E(u)`, meets `DetectionPressure κ`.
* `Dataflow.card_changed_le`: **a checked population is cone-local.** The egress slots where an assignment differs from
  the honest one, with the same inputs, number at most `Σ_{u ∈ incorrect} E(u)` (POP1's `eq_off_cone`, `card_coneEgress_le`).
* `guess_le`: **bits to probability.** A view that is a deviation `d ∈ 𝓓` applied to the honest view gives up the target
  with probability at most `Pr[d ∉ 𝓓] + #𝓓 · g`, when the honest view alone gives it with probability at most `g`.
* `card_devBall_le`: the deviations that change at most `K` of `m` slots number at most `Σ_{j ≤ K} C(m, j) · |V| ^ K`.
* `exfiltration_bound`: **the end-to-end bound.** If every deployment's egress is its honest egress with at most
  `cum + Δ` slots changed, `cum` what leaves through the first detection under an adaptive strategy with detection
  pressure `κ` and per-window leak `ℓ`, then any decoder of the egress and the side channel `Y` outputs the target
  with probability at most `exp (-κ (K - Δ - ℓ)) + Σ_{j ≤ K} C(m, j) · |V| ^ K · |Y| · g`, for every `K`.
* `card_changedThrough_le`, `exfiltration_bound_windows`: **over a deployment's windows.** The slots the windows change
  through the first detection number at most `cum`, so the deviation-size premise holds whenever each window's egress
  differs from honest only on the slots it changes (`Dataflow.card_changed_le` bounds those by its incorrect proof
  blocks' cone egress), plus at most `Δ` slots after the detection. The count is over the deployment's `n · |O|` slots.
-/

@[expose] public section

namespace Population.Exfil

open Finset
open Verity.Game (iid piLaw_toReal prodLaw prodLaw_toReal)

/-! ## Detection pressure from the sample and the cone -/

/-- A catch coin comes up `false` with the probability it doesn't come up `true`. -/
theorem toReal_false_eq (q : PMF Bool) : (q false).toReal = 1 - (q true).toReal := by
  have h := Verity.Game.sum_toReal_pmf q
  rw [Fintype.sum_bool] at h
  linarith

theorem toReal_true_le_one (q : PMF Bool) : (q true).toReal ≤ 1 := by
  have h := Verity.Game.sum_toReal_pmf q
  rw [Fintype.sum_bool] at h
  linarith [ENNReal.toReal_nonneg (a := q false)]

/-- **The independent sample's acceptance probability**: the incorrect set `B` escapes with probability
`Π_{u ∈ B} P(p u = false)`. -/
theorem indepSample_acceptCoins {U : Type} [Fintype U] [DecidableEq U] (p : U → PMF Bool) (B : Finset U) :
    Verity.Game.prPMFReal (indepSample p) (· ∈ acceptCoins B) = ∏ u ∈ B, (p u false).toReal := by
  rw [Verity.Game.prPMFReal_mem]
  let f : U → Bool → ℝ := fun u b => if u ∈ B ∧ b = true then 0 else (p u b).toReal
  have hpt : ∀ c : U → Bool, (if ∀ u ∈ B, c u = false then (indepSample p c).toReal else 0) = ∏ u, f u (c u) := by
    intro c
    by_cases hc : ∀ u ∈ B, c u = false
    · rw [ite_eq_left hc, indepSample, piLaw_toReal]
      refine prod_congr rfl fun u _ => ?_
      have : ¬ (u ∈ B ∧ c u = true) := fun ⟨hu, hcu⟩ => by simp [hc u hu] at hcu
      simp only [f, this, ite_false]
    · rw [ite_eq_right hc]
      simp only [not_forall, Bool.not_eq_false] at hc
      obtain ⟨u, hu, hcu⟩ := hc
      exact (prod_eq_zero (mem_univ u) (by simp [f, hu, hcu])).symm
  have hsum : ∀ u, ∑ b, f u b = if u ∈ B then (p u false).toReal else 1 := by
    intro u
    by_cases hu : u ∈ B
    · rw [if_pos hu, Fintype.sum_bool]
      simp [f, hu]
    · have hf : ∀ b, f u b = (p u b).toReal := fun b => if_neg fun h => hu h.1
      rw [sum_congr rfl fun b _ => hf b, if_neg hu, Verity.Game.sum_toReal_pmf]
  rw [acceptCoins, sum_filter, sum_congr rfl fun c _ => hpt c, ← Fintype.prod_sum, prod_congr rfl fun u _ => hsum u,
    prod_ite_mem, univ_inter]

/-- **Detection pressure from the sample and the cone**: if each window's leak is at most its incorrect proof blocks' cone
egress `Σ_{u ∈ B} E(u)`, and the window escapes with probability at most `Π_{u ∈ B} (1 - a u)` for catch probabilities
`a u ≥ κ E(u)`, the strategy meets detection pressure `κ`. -/
theorem pressure_of_sample {Ω U : Type} {μ : PMF Ω} {κ : ℝ} (hκ : 0 ≤ κ) (A : Strategy Ω)
    (B : List Ω → Finset U) (E a : U → ℝ) (ha : ∀ u, 0 ≤ a u ∧ a u ≤ 1) (hκE : ∀ u, κ * E u ≤ a u)
    (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, E u)
    (hesc : ∀ h, Verity.Game.prPMFReal μ (· ∈ A.esc h) ≤ ∏ u ∈ B h, (1 - a u)) :
    DetectionPressure μ κ A := by
  intro h
  refine (hesc h).trans ((acceptProb_le_exp (B h) a fun u _ => ha u).trans (Real.exp_le_exp.2 ?_))
  have h1 : κ * ∑ u ∈ B h, E u ≤ ∑ u ∈ B h, a u := by
    rw [mul_sum]
    exact sum_le_sum fun u _ => hκE u
  have h2 : κ * A.leak h ≤ κ * ∑ u ∈ B h, E u := mul_le_mul_of_nonneg_left (hleak h) hκ
  linarith

/-- **The independent sample gives detection pressure**: a strategy whose window escapes exactly when its incorrect set `B h`
does, and leaks at most `Σ_{u ∈ B h} E(u)`, meets detection pressure `κ` under the sample catching `u` with probability
`P(p u = true) ≥ κ E(u)`. -/
theorem pressure_indepSample {U : Type} [Fintype U] [DecidableEq U] {κ : ℝ} (hκ : 0 ≤ κ) (E : U → ℝ)
    (p : U → PMF Bool) (hκE : ∀ u, κ * E u ≤ (p u true).toReal) (A : Strategy (U → Bool))
    (B : List (U → Bool) → Finset U) (hA : ∀ h, A.esc h = acceptCoins (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, E u) :
    DetectionPressure (indepSample p) κ A :=
  pressure_of_sample hκ A B E (fun u => (p u true).toReal)
    (fun u => ⟨ENNReal.toReal_nonneg, toReal_true_le_one (p u)⟩) hκE hleak fun h => by
    rw [hA h, indepSample_acceptCoins]
    exact (prod_congr rfl fun u _ => toReal_false_eq (p u)).le

/-! ## Bits to probability -/

/-- **Bits to probability**: if the view is `F d` of the honest view `hon d` for a deviation `d = dv z`, and the honest
view alone gives the target up with probability at most `g` for every `d ∈ 𝓓`, then any decoder of the view gets the
target with probability at most `Pr[dv ∉ 𝓓] + #𝓓 · g`. The honest view may itself depend on the deviation (users who
adapt to what they saw). -/
theorem guess_le {Z D H X T : Type} [Fintype Z] (ρ : PMF Z) (S : Z → T) (𝓓 : Finset D)
    (hon : D → Z → H) (F : D → H → X) (dv : Z → D) {g : ℝ} (hT : ∀ d ∈ 𝓓, TrustedModel ρ S (hon d) g)
    (dec : X → T) :
    Verity.Game.prPMFReal ρ (fun z => dec (F (dv z) (hon (dv z) z)) = S z) ≤
      Verity.Game.prPMFReal ρ (fun z => dv z ∉ 𝓓) + 𝓓.card * g := by
  classical
  rw [Verity.Game.prPMFReal_eq_sum_mul, Verity.Game.prPMFReal_eq_sum_mul]
  have hT' : ∀ d ∈ 𝓓, ∑ z, (ρ z).toReal * (if dec (F d (hon d z)) = S z then (1 : ℝ) else 0) ≤ g := by
    intro d hd
    have := hT d hd fun h => dec (F d h)
    rwa [Verity.Game.prPMFReal_eq_sum_mul] at this
  have hpt : ∀ z, (if dec (F (dv z) (hon (dv z) z)) = S z then (1 : ℝ) else 0) ≤
      (if dv z ∉ 𝓓 then 1 else 0) + ∑ d ∈ 𝓓, (if dec (F d (hon d z)) = S z then (1 : ℝ) else 0) := by
    intro z
    have hnn : ∀ d ∈ 𝓓, (0 : ℝ) ≤ if dec (F d (hon d z)) = S z then 1 else 0 := fun d _ => by
      split_ifs <;> norm_num
    by_cases hd : dv z ∈ 𝓓
    · rw [if_neg (not_not.2 hd), zero_add]
      exact single_le_sum (f := fun d => if dec (F d (hon d z)) = S z then (1 : ℝ) else 0) hnn hd
    · rw [if_pos hd]
      have := sum_nonneg hnn
      split_ifs <;> linarith
  calc _ ≤ ∑ z, (ρ z).toReal * ((if dv z ∉ 𝓓 then 1 else 0) +
          ∑ d ∈ 𝓓, (if dec (F d (hon d z)) = S z then (1 : ℝ) else 0)) :=
        sum_le_sum fun z _ => mul_le_mul_of_nonneg_left (hpt z) ENNReal.toReal_nonneg
    _ = ∑ z, (ρ z).toReal * (if dv z ∉ 𝓓 then 1 else 0) +
          ∑ d ∈ 𝓓, ∑ z, (ρ z).toReal * (if dec (F d (hon d z)) = S z then (1 : ℝ) else 0) := by
        simp only [mul_add, sum_add_distrib, mul_sum]
        rw [sum_comm]
    _ ≤ _ := by
        refine add_le_add le_rfl ((sum_le_sum fun d hd => hT' d hd).trans ?_)
        rw [sum_const, nsmul_eq_mul]

/-- The deviations that change at most `K` slots. -/
noncomputable def devBall (Slot V : Type) [Fintype Slot] [DecidableEq Slot] [Fintype V] (K : ℕ) :
    Finset (Dev Slot V) :=
  (univ.filter fun C : Finset Slot => C.card ≤ K).sigma fun _ => univ

/-- **The Hamming-ball count**: at most `Σ_{j ≤ K} C(m, j) · |V| ^ K` deviations change at most `K` of `m` slots. -/
theorem card_devBall_le (Slot V : Type) [Fintype Slot] [DecidableEq Slot] [Fintype V] [Nonempty V] (K : ℕ) :
    (devBall Slot V K).card ≤ (∑ j ∈ range (K + 1), (Fintype.card Slot).choose j) * Fintype.card V ^ K := by
  have hV : 1 ≤ Fintype.card V := Fintype.card_pos
  have hfilter : (univ.filter fun C : Finset Slot => C.card ≤ K).card ≤
      ∑ j ∈ range (K + 1), (Fintype.card Slot).choose j := by
    calc _ ≤ ((range (K + 1)).biUnion fun j => powersetCard j (univ : Finset Slot)).card := by
          refine card_le_card fun C hC => mem_biUnion.2 ⟨C.card, ?_, mem_powersetCard.2 ⟨subset_univ _, rfl⟩⟩
          exact mem_range.2 (Nat.lt_succ_of_le (mem_filter.1 hC).2)
      _ ≤ ∑ j ∈ range (K + 1), (powersetCard j (univ : Finset Slot)).card := card_biUnion_le
      _ = _ := by simp only [card_powersetCard, card_univ]
  rw [devBall, card_sigma]
  refine (sum_le_sum (g := fun _ => Fintype.card V ^ K) fun C hC => ?_).trans ?_
  · rw [card_univ, Fintype.card_fun, Fintype.card_coe]
    exact Nat.pow_le_pow_right hV (mem_filter.1 hC).2
  · rw [sum_const, smul_eq_mul]
    exact Nat.mul_le_mul_right _ hfilter

/-! ## The end-to-end bound -/

/-- **The end-to-end bound.** The weights `w` are sampled from `ρ` and the deployment's `n` windows' coins from `μ`. The
prover plays the adaptive strategy `A w`, with detection pressure `κ` and per-window leak at most `ℓ`. The outsider sees
the egress, which is the honest egress `hon (d, y)` with the deviation `d = dv` applied, and the side channel
`y = side ∈ Y`. Suppose the deviation changes at most `cum + Δ` slots, where `cum` is what leaves through the first
detection and `Δ` covers the windows between that detection and the halt. Suppose also that the honest egress alone
gives the target up with probability at most `g`. Then for every `K`, any decoder outputs the target with probability
at most `exp (-κ (K - Δ - ℓ)) + Σ_{j ≤ K} C(m, j) · |V| ^ K · |Y| · g`, where `m` is the number of egress slots. -/
theorem exfiltration_bound {Ω W T Slot V Y : Type} [Fintype Ω] [DecidableEq Ω] [Fintype W] [Fintype Slot]
    [DecidableEq Slot] [Fintype V] [Nonempty V] [Fintype Y] {μ : PMF Ω} {ρ : PMF W}
    {κ ℓ Δ g : ℝ} (hκ : 0 ≤ κ) (hg : 0 ≤ g) (n K : ℕ) (A : W → Strategy Ω)
    (hP : ∀ w, DetectionPressure μ κ (A w)) (hL : ∀ w, LeakBounded ℓ (A w)) (S : W → T)
    (hon : Dev Slot V × Y → W × (Fin n → Ω) → Slot → V) (dv : W × (Fin n → Ω) → Dev Slot V)
    (side : W × (Fin n → Ω) → Y) (hdv : ∀ w cs, ((dv (w, cs)).1.card : ℝ) ≤ (A w).cum n [] cs + Δ)
    (hT : ∀ d, TrustedModel (prodLaw ρ (iid μ n)) (fun z => S z.1) (hon d) g)
    (dec : (Slot → V) × Y → T) :
    Verity.Game.prPMFReal (prodLaw ρ (iid μ n))
        (fun z => dec ((dv z).apply (hon (dv z, side z) z), side z) = S z.1) ≤
      Real.exp (-(κ * (K - Δ - ℓ))) +
        (∑ j ∈ range (K + 1), (Fintype.card Slot).choose j) * Fintype.card V ^ K * Fintype.card Y * g := by
  classical
  let 𝓓 : Finset (Dev Slot V × Y) := devBall Slot V K ×ˢ univ
  have hmem : ∀ z, (dv z, side z) ∈ 𝓓 ↔ (dv z).1.card ≤ K := fun z => by
    simp [𝓓, devBall, mem_product, mem_sigma, mem_filter]
  have key := guess_le (prodLaw ρ (iid μ n)) (fun z => S z.1) 𝓓 hon (fun d e => (d.1.apply e, d.2))
    (fun z => (dv z, side z)) (fun d _ => hT d) dec
  have hcard : (𝓓.card : ℝ) ≤
      (∑ j ∈ range (K + 1), (Fintype.card Slot).choose j) * Fintype.card V ^ K * Fintype.card Y := by
    rw [card_product, card_univ]
    exact_mod_cast Nat.mul_le_mul_right _ (card_devBall_le Slot V K)
  refine key.trans (add_le_add ?_ (mul_le_mul_of_nonneg_right hcard hg))
  calc _ ≤ Verity.Game.prPMFReal (prodLaw ρ (iid μ n)) (fun z => (K : ℝ) - Δ ≤ (A z.1).cum n [] z.2) := by
        refine Verity.Game.prPMFReal_mono _ fun z hz => ?_
        have hK : (K : ℝ) < (dv z).1.card := by exact_mod_cast not_le.1 fun h => hz ((hmem z).2 h)
        have := hdv z.1 z.2
        linarith
    _ = ∑ w, (ρ w).toReal * prob μ n (fun cs => (K : ℝ) - Δ ≤ (A w).cum n [] cs) := by
        rw [Verity.Game.prPMFReal_eq_sum_mul, Fintype.sum_prod_type]
        refine sum_congr rfl fun w _ => ?_
        rw [prob_eq_ex, ex, mul_sum]
        refine sum_congr rfl fun cs _ => ?_
        have hpr := prodLaw_toReal ρ (iid μ n) w cs
        by_cases hc : (K : ℝ) - Δ ≤ (A w).cum n [] cs <;> simp [hc, hpr]
    _ ≤ ∑ w, (ρ w).toReal * Real.exp (-(κ * ((K : ℝ) - Δ - ℓ))) :=
        sum_le_sum fun w _ => mul_le_mul_of_nonneg_left (tail_cum hκ (hP w) (hL w) n [] ((K : ℝ) - Δ))
          ENNReal.toReal_nonneg
    _ = _ := by rw [← sum_mul, Verity.Game.sum_toReal_pmf, one_mul]

/-! ## A deployment's windows -/

/-- **The windows' changed slots add up to `cum`**: when each window leaks at least the number of egress slots it
changes, the slots changed through the first detection number at most what leaves through it. -/
theorem card_changedThrough_le {Ω O : Type} [DecidableEq Ω] [DecidableEq O] (A : Strategy Ω) (chg : List Ω → Finset O)
    (hchg : ∀ h, ((chg h).card : ℝ) ≤ A.leak h) :
    ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω), ((changedThrough A chg n h cs).card : ℝ) ≤ A.cum n h cs
  | 0, _, _ => by simp [changedThrough, Strategy.cum]
  | n + 1, h, cs => by
    rw [changedThrough, Strategy.cum]
    refine (Nat.cast_le.2 (card_union_le _ _)).trans ?_
    rw [Nat.cast_add, card_map]
    refine add_le_add (hchg h) ?_
    split_ifs
    · rw [card_map]
      exact card_changedThrough_le A chg hchg n (cs 0 :: h) (Fin.tail cs)
    · simp

/-- **The end-to-end bound over a deployment's windows.** Window `i` of `n` has egress slots `O`, so the deployment
has `n · |O|`. The window after history `h` changes the slots `chg w h`, and its leak is their number. The egress
deviates from honest only on the slots changed through the first detection and on at most `Δ` slots after it, before
the halt. Then any decoder of the egress and the side channel outputs the target with probability at most
`exp (-κ (K - Δ - ℓ)) + Σ_{j ≤ K} C(n |O|, j) · |V| ^ K · |Y| · g`. -/
theorem exfiltration_bound_windows {Ω W T O V Y : Type} [Fintype Ω] [DecidableEq Ω] [Fintype W] [Fintype O]
    [DecidableEq O] [Fintype V] [Nonempty V] [Fintype Y] {μ : PMF Ω} {ρ : PMF W} {κ ℓ g : ℝ} (hκ : 0 ≤ κ)
    (hg : 0 ≤ g) (n K Δ : ℕ) (A : W → Strategy Ω)
    (chg : W → List Ω → Finset O) (hchg : ∀ w h, (A w).leak h = (chg w h).card)
    (hP : ∀ w, DetectionPressure μ κ (A w)) (hL : ∀ w, LeakBounded ℓ (A w)) (S : W → T)
    (hon : Dev (Fin n × O) V × Y → W × (Fin n → Ω) → Fin n × O → V)
    (dv : W × (Fin n → Ω) → Dev (Fin n × O) V) (side : W × (Fin n → Ω) → Y)
    (lag : W × (Fin n → Ω) → Finset (Fin n × O)) (hlag : ∀ z, (lag z).card ≤ Δ)
    (hdv : ∀ w cs, (dv (w, cs)).1 ⊆ changedThrough (A w) (chg w) n [] cs ∪ lag (w, cs))
    (hT : ∀ d, TrustedModel (prodLaw ρ (iid μ n)) (fun z => S z.1) (hon d) g)
    (dec : (Fin n × O → V) × Y → T) :
    Verity.Game.prPMFReal (prodLaw ρ (iid μ n))
        (fun z => dec ((dv z).apply (hon (dv z, side z) z), side z) = S z.1) ≤
      Real.exp (-(κ * (K - Δ - ℓ))) +
        (∑ j ∈ range (K + 1), (n * Fintype.card O).choose j) * Fintype.card V ^ K * Fintype.card Y * g := by
  have hdv' : ∀ w cs, ((dv (w, cs)).1.card : ℝ) ≤ (A w).cum n [] cs + Δ := by
    intro w cs
    have h1 := card_le_card (hdv w cs)
    have h2 := card_union_le (changedThrough (A w) (chg w) n [] cs) (lag (w, cs))
    have h3 := card_changedThrough_le (A w) (chg w) (fun h => (hchg w h).ge) n [] cs
    have h4 : ((lag (w, cs)).card : ℝ) ≤ Δ := by exact_mod_cast hlag (w, cs)
    have h5 : ((dv (w, cs)).1.card : ℝ) ≤ (changedThrough (A w) (chg w) n [] cs).card + (lag (w, cs)).card := by
      exact_mod_cast h1.trans h2
    linarith
  have := exfiltration_bound hκ hg n K A hP hL S hon dv side hdv' hT dec
  rwa [Fintype.card_prod, Fintype.card_fin] at this

end Population.Exfil

/-! ## A checked population is cone-local -/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

/-- **A checked population is cone-local**: an assignment that holds the honest run's inputs differs from it on at most
`Σ_{u ∈ incorrect} E(u)` egress slots, `E(u) = |coneEgress O {u}|`. -/
theorem card_changed_le [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [DecidableEq V] (f : P.Fns V)
    (O : Finset S) {σ honest : S → V} (hin : ∀ t ∈ P.input, σ t = honest t) (hh : ∀ u, P.Correct f honest u) :
    (O.filter fun s => σ s ≠ honest s).card ≤ ∑ u ∈ P.incorrect f σ, (P.coneEgress O {u}).card := by
  refine (card_le_card fun s hs => ?_).trans (P.card_coneEgress_le O (P.incorrect f σ))
  obtain ⟨hsO, hne⟩ := mem_filter.1 hs
  refine P.mem_coneEgress.2 ⟨hsO, ?_⟩
  by_contra hw
  exact hne (P.eq_off_cone hc ha f hin le_rfl (by simp [P.incorrect_eq_empty hh]) hw)

end Population.Dataflow
