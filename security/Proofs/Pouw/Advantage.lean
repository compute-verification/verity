module

public import Definitions.Pouw.Guarantees.Advantage
public import Proofs.Pouw.Game
public import Proofs.Pouw.NCP
public import Proofs.Pouw.PearlC

/-!
# The advantage form

`G_ε` and the gap property `G_g` are one property: `G_g` at `g < 1` is `G_ε` at `ε = g/(1 − g)`, and `G_ε` at `ε ≥ 0` is
`G_g` at `g = ε/(1 + ε)`, in the game (`gGap_iff_gε`, `gε_iff_gGap`) and in Pearl-C's (`Pouw.PearlC.gGap_iff_gε`,
`Pouw.PearlC.gε_iff_gGap`). Theorem 1 in ε form (`theorem1Eps`), and each guarantee of
`Definitions.Pouw.Guarantees.Advantage` from its gap form: `Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`.
-/

@[expose] public section

namespace Pouw.Proofs

open Finset

/-- `(1 + g/(1 − g))·T = T/(1 − g)` for `g < 1`. -/
theorem one_add_div_one_sub_mul {g : ℝ} (hg : g < 1) (T : ℝ) : (1 + g / (1 - g)) * T = T / (1 - g) := by
  have h : (1 : ℝ) - g ≠ 0 := (sub_pos.mpr hg).ne'
  rw [one_add_div h, sub_add_cancel, div_mul_eq_mul_div, one_mul]

/-- `T/(1 − ε/(1 + ε)) = (1 + ε)·T` for `ε > −1`. -/
theorem div_one_sub_div_one_add {ε : ℝ} (hε : -1 < ε) (T : ℝ) : T / (1 - ε / (1 + ε)) = (1 + ε) * T := by
  have h : (1 : ℝ) + ε ≠ 0 := (by linarith : (0 : ℝ) < 1 + ε).ne'
  rw [one_sub_div h, add_sub_cancel_right, div_div_eq_mul_div, div_one]
  ring

/-- The reduction's advantage: `g/(1 − g) = Ω_max/(1 − g₀) − 1` at the reduction's gap `g = 1 − (1 − g₀)/Ω_max`. -/
theorem reductionGap_div {g₀ Ωmax : ℝ} (hg₀ : g₀ < 1) (hΩ : 0 < Ωmax) :
    reductionGap g₀ Ωmax / (1 - reductionGap g₀ Ωmax) = Ωmax / (1 - g₀) - 1 := by
  have h1 : (1 : ℝ) - g₀ ≠ 0 := (sub_pos.mpr hg₀).ne'
  have h2 : Ωmax ≠ 0 := hΩ.ne'
  rw [one_sub_reductionGap]
  unfold reductionGap
  field_simp

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **`G_g` at `g < 1` is `G_ε` at `ε = g/(1 − g)`.** -/
theorem gGap_iff_gε (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) {g : ℝ} (hg : g < 1)
    (δ : ℕ → ℕ → ℝ) : GGap CM P D g δ ↔ Gε CM P D (g / (1 - g)) δ := by
  unfold GGap Gε Wins
  simp only [one_add_div_one_sub_mul hg, iff_self]

/-- **`G_ε` at `ε ≥ 0` is `G_g` at `g = ε/(1 + ε)`.** -/
theorem gε_iff_gGap (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (δ : ℕ → ℕ → ℝ) : Gε CM P D ε δ ↔ GGap CM P D (ε / (1 + ε)) δ := by
  unfold GGap Gε Wins
  simp only [div_one_sub_div_one_add (by linarith : (-1 : ℝ) < ε), iff_self]

/-- **Theorem 1 in ε form** (§3.5): if `G_ε` holds at the advantage `ε_a` with error `δ` and, for an adversary within
`(T, q)`, the audit (any coins `C`, any acceptance predicate) satisfies `Pr[accept ∧ ∑_{u ∉ K} W_ref(u) > ε_s·W] ≤ δ_s`
with `ε_s < 1`, then `Pr[accept ∧ (1 + ε_a)/(1 − ε_s)·T < W ∧ the run doesn't break SHA-512] ≤ δ_s + δ`. -/
theorem theorem1Eps (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (εa : ℝ) (δ : ℕ → ℕ → ℝ)
    (hG : Gε CM P D εa δ) (U : Workload) (hU : U.InDomain D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary CM) (hA : A.Bounded U T q) (C : Type) [Fintype C] (accept : (Q → R) → S → C → Prop)
    (εs δs : ℝ) (hεs : εs < 1)
    (hδ : pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
        εs * totalWref P U.layout <
          ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ))
      ≤ δs) :
    pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧ (1 + εa) / (1 - εs) * T < totalWref P U.layout ∧
        ¬ A.Breaks U x.1 x.2.1)
      ≤ δs + δ q U.layout.N := by
  let E1 := fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
    εs * totalWref P U.layout <
      ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ)
  let E2 := fun y : (Q → R) × S =>
    (1 + εa) * T < ∑ u ∈ correctSet P U y.1 y.2 (A.transcript U y.1 y.2), (P.Wref U.layout u : ℝ) ∧
      ¬ A.Breaks U y.1 y.2
  have h1s : 0 < 1 - εs := by linarith
  calc _ ≤ pr (fun x : (Q → R) × S × C => E1 x ∨ E2 (x.1, x.2.1)) := pr_mono fun x hx => by
        by_cases h1 : E1 x
        · exact Or.inl h1
        · refine Or.inr ⟨?_, hx.2.2⟩
          have hbad : ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1),
              (P.Wref U.layout u : ℝ) ≤ εs * totalWref P U.layout := by
            by_contra h2
            exact h1 ⟨hx.1, lt_of_not_ge h2⟩
          have hsplit := Finset.sum_sdiff (f := fun u => (P.Wref U.layout u : ℝ))
            (correctSet_subset_range P U x.1 x.2.1 (A.transcript U x.1 x.2.1))
          have hW : (1 + εa) * T < (1 - εs) * totalWref P U.layout := by
            have h := hx.2.1
            rw [div_mul_eq_mul_div, div_lt_iff₀ h1s] at h
            linarith
          show (1 + εa) * T < ∑ u ∈ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ)
          unfold totalWref at hbad hW
          linarith
    _ ≤ pr E1 + pr (fun x : (Q → R) × S × C => E2 (x.1, x.2.1)) := pr_or_le _ _
    _ ≤ δs + δ q U.layout.N := add_le_add hδ ((pr_fst2_le E2).trans (hG U hU T hT q hq A hA))

end Pouw.Proofs

namespace Pouw.PearlC

open Finset Pouw.Proofs

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **Pearl-C's `G_g` at `g < 1` is its `G_ε` at `ε = g/(1 − g)`.** -/
theorem gGap_iff_gε (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) {g : ℝ} (hg : g < 1)
    (δ : ℕ → ℕ → ℝ) : GGap CM P D g δ ↔ Gε CM P D (g / (1 - g)) δ := by
  unfold GGap Gε Wins
  simp only [one_add_div_one_sub_mul hg, iff_self]

/-- **Pearl-C's `G_ε` at `ε ≥ 0` is its `G_g` at `g = ε/(1 + ε)`.** -/
theorem gε_iff_gGap (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (δ : ℕ → ℕ → ℝ) : Gε CM P D ε δ ↔ GGap CM P D (ε / (1 + ε)) δ := by
  unfold GGap Gε Wins
  simp only [div_one_sub_div_one_add (by linarith : (-1 : ℝ) < ε), iff_self]

end Pouw.PearlC

namespace Pouw.SecurityProofs

/-- **(d) End to end in ε form**: `gapFromTT` gives `G_g` at the reduction's gap, `gGap_iff_gε` turns it into `G_ε` at
`Ω_max/(1 − g₀) − 1`, and `theorem1Eps` composes it with work-weighted sampling. -/
theorem EndToEndEps : Pouw.Guarantees.EndToEndEps := by
  intro Q R S _ _ _ _ CM P D g₀ Ωmax δ hg₀ hΩ hW hTT U hU T hT q hq A hA t εs hεs
  have hG := (Pouw.Proofs.gGap_iff_gε CM P D (Pouw.Proofs.reductionGap_lt_one hg₀ hΩ) δ).1
    (Pouw.Proofs.gapFromTT Q R S CM P D g₀ Ωmax δ hg₀ hΩ hW hTT)
  rw [Pouw.Proofs.reductionGap_div hg₀ hΩ] at hG
  refine Pouw.Proofs.theorem1Eps CM P D _ δ hG U hU T hT q hq A hA (Fin t → Ticket (P.Wref U.layout) U.layout.N)
    (fun H s c => AuditAccepts (correctSet P U H s (A.transcript U H s)) c) εs ((1 - εs) ^ t) hεs ?_
  let G := fun y : ((Q → R) × S) × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
    AuditAccepts (correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2)) y.2 ∧
      εs * ∑ u ∈ Finset.range U.layout.N, (P.Wref U.layout u : ℝ) <
        ∑ u ∈ Finset.range U.layout.N \ correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2),
          (P.Wref U.layout u : ℝ)
  calc _ = Pouw.pr G := Pouw.pr_equiv (Equiv.prodAssoc _ _ _).symm G
    _ ≤ _ := Pouw.Proofs.workWeightedSampling ((Q → R) × S) (P.Wref U.layout) U.layout.N t
        (fun y => correctSet P U y.1 y.2 (A.transcript U y.1 y.2)) εs hεs.le

end Pouw.SecurityProofs

namespace Pouw.SecurityProofs.NCP

/-- **`G_ε` from TT_NCP_U at ncp-v1**: `GapFromTTNCP_U_v1` and `gGap_iff_gε`. -/
theorem EpsFromTTNCP_U_v1 : Pouw.Guarantees.NCP.EpsFromTTNCP_U_v1 := by
  intro Q R S _ _ _ _ CM P D g₀ n₀ δ hg₀ hn₀ hP hTT
  obtain ⟨hW, hn, hside⟩ := hP
  have hΩ : (0 : ℝ) < 1 + 8 / n₀ := by positivity
  have hG := (Pouw.Proofs.gGap_iff_gε CM (P.toProtocolU 128) D (Pouw.Proofs.reductionGap_lt_one hg₀ hΩ) δ).1
    (GapFromTTNCP_U_v1 Q R S CM P D g₀ n₀ δ hg₀ hn₀ hW hn hside hTT)
  rwa [Pouw.Proofs.reductionGap_div hg₀ hΩ] at hG

end Pouw.SecurityProofs.NCP

namespace Pouw.SecurityProofs.PearlC

/-- **Pearl-C's design point at 8192³ on the RTX PRO 6000, in ε form** (not the served point, which is Llama 3.2 1B's
linears under the H100 record): `GapSm120v1LoopCast8p72Rev1Cap1000_8192` and `Pouw.PearlC.gGap_iff_gε`, at
`ε = g/(1 − g) = 1648180439/444424569561`. -/
theorem EpsSm120v1LoopCast8p72Rev1Cap1000_8192 :
    Pouw.Guarantees.PearlC.EpsSm120v1LoopCast8p72Rev1Cap1000_8192 := by
  intro Q R S _ _ _ _ sha fetch salt sem hTT
  have e : ((1648180439 / 444424569561 : ℚ) : ℝ) =
      ((1648180439 / 446072750000 : ℚ) : ℝ) / (1 - ((1648180439 / 446072750000 : ℚ) : ℝ)) := by
    push_cast
    norm_num
  rw [e]
  exact (Pouw.PearlC.gGap_iff_gε _ _ _ (by push_cast; norm_num) _).1
    (GapSm120v1LoopCast8p72Rev1Cap1000_8192 sha fetch salt sem hTT)

/-- **Pearl-C4's headline at 8192³ in ε form**: `GapFp4Sm120_8192` and `Pouw.PearlC.gGap_iff_gε`, at
`ε = g/(1 − g) = 9875013901/1366788186099`. -/
theorem EpsFp4Sm120_8192 : Pouw.Guarantees.PearlC.EpsFp4Sm120_8192 := by
  intro Q R S _ _ _ _ sha fetch salt sem hTT
  have e : ((9875013901 / 1366788186099 : ℚ) : ℝ) =
      ((9875013901 / 1376663200000 : ℚ) : ℝ) / (1 - ((9875013901 / 1376663200000 : ℚ) : ℝ)) := by
    push_cast
    norm_num
  rw [e]
  exact (Pouw.PearlC.gGap_iff_gε _ _ _ (by push_cast; norm_num) _).1 (GapFp4Sm120_8192 sha fetch salt sem hTT)

end Pouw.SecurityProofs.PearlC
