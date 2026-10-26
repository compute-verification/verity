import Definitions.Pouw.Assumptions.PearlC.HiddenTile
import Proofs.Pouw.PearlC.DeviceCapRev1Gamma
import Proofs.Pouw.PearlC.TTOutW1
import Proofs.Core.Game.Prob

/-!
# The hidden audit keeps the sampled audit's γ (compute accounting's file)

* **One prover** (`gammaHidden_of_sampled`): on each draw, when the hidden audit accepts while some drawn tile is
  neither good nor blank, that tile's verdict was accept (`hidden_pointwise`). So the hidden audit's win probability is
  at most `GγSampled`'s plus, over the `t` draws, `TileProofSound`'s per-draw error.
* **Every prover in the class** (`gammaHidden_of_sampled_all`): `GγSampled` and the open obligation
  `TileProofSoundAll` give `GγHidden`, at `t·ηzk(q)` more error, for any hidden audit and every opening.
* **The headline, conditional** (`pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`): v1 under rev1 per tile at 1/1,000,
  8192³, 0.36949% under the hidden audit at W1, for a free hidden audit and `ηzk`; nothing about C-Flock until its
  instance lands.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions

variable {Q R S : Type}

theorem ex_mono {α : Type} [Fintype α] {f g : α → ℝ} (h : ∀ x, f x ≤ g x) : ex f ≤ ex g :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun x _ => h x) (Nat.cast_nonneg _)

theorem ex_add {α : Type} [Fintype α] (f g : α → ℝ) : ex (fun x => f x + g x) = ex f + ex g := by
  simp only [ex, Finset.sum_add_distrib, add_div]

theorem ex_sum {α ι : Type} [Fintype α] [Fintype ι] (b : ι → α → ℝ) :
    ex (fun x => ∑ i, b i x) = ∑ i, ex (b i) := by
  simp only [ex, div_eq_mul_inv, Finset.sum_mul]
  exact Finset.sum_comm

theorem ex_ite_one {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    ex (fun x => if p x then (1 : ℝ) else 0) = pr p := by
  classical
  simp only [ex, pr, Finset.sum_boole]
  congr 3
  ext x
  simp

open Classical in
/-- One draw: when the audit fails on the opened tiles, some drawn tile is neither good nor blank, and the joint verdict is
at most that tile's verdict. -/
theorem hidden_pointwise {w : ℕ → ℕ} {NT t : ℕ} (g : Verity.Game (List Bool)) (s : Verity.Game.Strategy g)
    (K : Finset ℕ) (c : Fin t → Ticket w NT) (E : Prop) :
    (if E then (Verity.Game.prob (fun v => ∀ i : Fin t, v.getD i false = true) g s).toReal else 0) ≤
      (if AuditAccepts K c ∧ E then (1 : ℝ) else 0) +
        ∑ i : Fin t,
          (if ((c i).1 : ℕ) ∈ K then 0 else (Verity.Game.prob (fun v => v.getD i false = true) g s).toReal) := by
  have hnn : 0 ≤ ∑ i : Fin t,
      (if ((c i).1 : ℕ) ∈ K then (0 : ℝ) else (Verity.Game.prob (fun v => v.getD i false = true) g s).toReal) :=
    Finset.sum_nonneg fun i _ => by split_ifs <;> simp
  by_cases hE : E
  · by_cases hA : AuditAccepts K c
    · simp only [hE, hA, and_self, ite_true]
      have h1 : (Verity.Game.prob (fun v => ∀ i : Fin t, v.getD i false = true) g s).toReal ≤ 1 :=
        ENNReal.toReal_le_of_le_ofReal zero_le_one (by simpa using Verity.Game.prob_le_one _ g s)
      linarith
    · obtain ⟨i, hi⟩ : ∃ i, ((c i).1 : ℕ) ∉ K := by simpa [AuditAccepts] using hA
      simp only [hE, hA, false_and, ite_true, ite_false, zero_add]
      refine le_trans ?_ (Finset.single_le_sum (f := fun i : Fin t =>
        if ((c i).1 : ℕ) ∈ K then (0 : ℝ) else (Verity.Game.prob (fun v => v.getD i false = true) g s).toReal)
        (fun j _ => by split_ifs <;> simp) (Finset.mem_univ i))
      simp only [hi, ite_false]
      exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (Verity.Game.prob_le_one _ g s))
        (Verity.Game.prob_mono (fun v hv => hv i) g s)
  · simp only [hE, and_false, ite_false, zero_add]
    exact hnn

/-- **The hidden audit keeps the sampled audit's γ, one prover at a time**: under `GγSampled`, a prover of the proof
phase sound per tile up to `ηzk` (`TileProofSound`) wins the hidden audit with probability at most
`(1 − ε_s)^t + η(q, N) + t·ηzk`, for any hidden audit. -/
theorem gammaHidden_of_sampled [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (HA : HiddenAudit Q R S) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) (h : GγSampled CM P TR D γ η) (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ)
    (hq : q ≤ 2 ^ 64) (A : Adversary CM) (hA : A.Bounded U T q) (w : ℕ → ℕ) (κ : ℝ) (hκ : 0 < κ)
    (hw : ∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) (t : ℕ) (εs : ℝ) (hεs : εs ≤ 1)
    (op : (Q → R) → S → HA.Opening) (σ : HA.Prover TR U (A.transcript U) op w t) (ηzk : ℝ)
    (hσ : TileProofSound P TR HA U (A.transcript U) op σ ηzk) :
    HA.winProb TR U (A.transcript U) op σ γ T εs ≤ (1 - εs) ^ t + η q U.layout.N + t * ηzk := by
  classical
  have hs := h U hU T hT q hq A hA w κ hκ hw t εs hεs
  refine le_trans (ex_mono fun x => hidden_pointwise _ (σ x)
    (goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪ blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2
    (T < (1 - γ) * (claimedWref TR U (A.transcript U x.1 x.2.1) - εs * tileWrefTotal TR U.layout))) ?_
  rw [ex_add, ex_sum, ex_ite_one]
  have hz : ∑ i : Fin t, ex (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
      if ((x.2.2 i).1 : ℕ) ∈ goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
          blankTiles TR U (A.transcript U x.1 x.2.1) then 0 else
        (Verity.Game.prob (fun v => v.getD i false = true)
          (HA.batch x.1 x.2.1 U (HA.root U (A.transcript U x.1 x.2.1) (op x.1 x.2.1)) x.2.2) (σ x)).toReal) ≤
        t * ηzk := by
    refine le_trans (Finset.sum_le_sum fun i _ => hσ i) ?_
    simp
  linarith

/-- **L2: the hidden audit keeps the sampled audit's γ**, for every opening and every prover in the proof system's
class: `GγSampled` and the open obligation `TileProofSoundAll` give `GγHidden`, at `t·ηzk(q)` more error, for any
hidden audit. -/
theorem gammaHidden_of_sampled_all [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (HA : HiddenAudit Q R S) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) (ηzk : ℕ → ℝ) (h : GγSampled CM P TR D γ η) (hzk : TileProofSoundAll P TR D HA ηzk) :
    GγHidden CM P TR D HA γ η ηzk :=
  fun U hU T hT q hq A hA w κ hκ hw t εs hεs op σ hσ =>
    gammaHidden_of_sampled CM P TR D HA γ η h U hU T hT q hq A hA w κ hκ hw t εs hεs op σ (ηzk q)
      (hzk U hU (A.transcript U) op w t q σ hσ)

variable [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³, under the hidden audit, conditional**, FP32 at 8.376, the
honest cast at 8.72: `γ = 1648180439/446072750000` (0.36949%), for any hidden audit sound per tile up to `ηzk(q)`, at
the cost model `W1.w1Priced` on `W1Device.sm120Loop sha fetch`, from `TTOutTilePearlCDevRev1W1` there.

Conditional: `HA` and `ηzk` are free, and the open obligation `TileProofSoundAll` holds at a degenerate hidden audit
(`Bounded := fun _ _ _ _ _ => False`, an empty class) for any `ηzk ≥ 0`, so this says nothing about C-Flock until
C-Flock's `HiddenAudit` instance and the obligation's discharge at it land. Cite it only as conditional. -/
theorem pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192 (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S)
    (HA : HiddenAudit Q R S) (ηzk : ℕ → ℝ)
    (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop)
      sem (1 / 1000))
    (hzk : TileProofSoundAll (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000)
      (218 / 25 - 8953 / 1000)) (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000)
      (218 / 25 - 8953 / 1000)) (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) HA ηzk) :
    GγHidden (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) HA
      ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC ηzk :=
  gammaHidden_of_sampled_all _ _ _ _ HA _ _ ηzk
    (pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192 _ sem (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)) hzk

end Pouw.PearlC
