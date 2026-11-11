module

public import Proofs.WeightSecurity.TwoReg
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Tape
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.SamplingGeneric.TwoRegMiss
import Proofs.WeightSecurity.Stop

/-!
# A period of two-stage windows at the committed traces, proved

`PeriodTwoReg` is `period_undetected` at `twoRegRec`. Each record reads only the strategy its window is played with, so
both of a window's bounds hold at every strategy: `hacc` is `CompiledTwo.KSRegTwo`'s first conjunct at the statement
and prover it sends, and `hstage` is `CompiledTwo.MissTwo` at the count it fixes (`regCount`).

* `twoRegRec_accepted_le`: a window is accepted with a sampled proof block in the committed incorrect set with
  probability at most `ε_reg,2` at the statement and prover it sends, and never out of scope.
* `twoRegRec_missed_le`: its sample misses the committed incorrect set with probability at most `(1 − r₁·r₂)^n`, `n`
  the count its strategy fixes.
* `periodTwoReg_of`: `PeriodTwoReg` from each window's `KSRegTwo` and `MissTwo`, its refusal of what is out of scope,
  and the two stages' rates in scope.
* `compilerPeriodTwoReg_of`: `CompilerPeriodTwoReg` from `CompilerKSRegTwo`, through `compilerMissTwo`. So
  **`CompilerPeriodTwoReg` is proved** (`compilerPeriodTwoReg`), at `compilerKSRegTwo`.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

open Classical in
/-- **The count a window fixes** is `0`, or its statement is in the two-stage scope and the count is the replay blocks
wrong at the committed boundary trace. -/
theorem regCount_cases (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (P : Prover Unit) :
    regCount H L x P = 0 ∨ ∃ s Pt d, L.parse x = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧
      InScopeTwo Pt d s ∧ regCount H L x P = (wrongAtBoundaryAt s Pt d.part (L.regVal H P x ())).card := by
  rcases hs : L.parse x with _ | s
  · exact Or.inl (by simp only [regCount, hs])
  rcases hPt : L.partition s with _ | Pt
  · exact Or.inl (by simp only [regCount, hs, hPt])
  rcases hd : L.two s with _ | d
  · exact Or.inl (by simp only [regCount, hs, hPt, hd])
  by_cases hin : InScopeTwo Pt d s
  · exact Or.inr ⟨s, Pt, d, rfl, hPt, hd, hin, by simp only [regCount, hs, hPt, hd, ite_eq_left hin]⟩
  · exact Or.inl (by simp only [regCount, hs, hPt, hd, ite_eq_right hin])

/-- **A window is accepted with a sampled proof block in the committed incorrect set** at most as often as `ε_reg,2` at
the statement and prover it sends, at a statement in scope by `CompiledTwo.KSRegTwo` at a prover with no tape, and out
of scope never, since the verifier refuses it. -/
theorem twoRegRec_accepted_le (H : Msg → Msg) (L : CompiledTwo Coin A) (hks : L.KSRegTwo H)
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x) (n : ℕ)
    (τ : Strategy (twoGame L ((O → V) × Tick × Adv))) :
    prob (fun b => (twoRegRec H L n b).accepted ∧ (twoRegRec H L n b).sampledIncorrect) (twoGame L _) τ ≤
      ENNReal.ofReal (L.εregTwo H τ.2.1.1 τ.2.1.2) := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  show Verity.Game.prCoin (fun r => (run H P L.V x () r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.FregTwo H P x () r) ≤ ENNReal.ofReal (L.εregTwo H x P)
  by_cases hx : L.Scope x
  · rw [Verity.Game.prCoin_eq_ofReal_pr]
    refine ENNReal.ofReal_le_ofReal (le_trans (le_of_eq ?_) (hks x hx Unit P).1)
    exact (Pr_unit fun r => (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.FregTwo H P x () r).symm
  · rw [show (fun r => (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.FregTwo H P x () r) =
        fun _ => False from funext fun r => propext (iff_false_intro fun h => hx (href x P r h.1)),
      Verity.Game.prCoin_false]
    exact zero_le

/-- **A window's sample misses the committed incorrect set** with probability at most `(1 − r₁·r₂)^n`, `n` the count its
strategy fixes: `CompiledTwo.MissTwo` in scope, where the two stages' rates hold, and `1` out of it, where the count is
`0`. -/
theorem twoRegRec_missed_le (H : Msg → Msg) (L : CompiledTwo Coin A) {r₁ r₂ : ℝ} (hx0 : 0 ≤ 1 - r₁ * r₂)
    (hmiss : L.MissTwo H r₁ r₂)
    (hrate : ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
      d.FirstRate r₁ ∧ d.SecondRate Pt r₂)
    (τ : Strategy (twoGame L ((O → V) × Tick × Adv))) :
    prob (fun b => ¬ (twoRegRec H L (regCount H L τ.2.1.1 τ.2.1.2) b).sampledIncorrect) (twoGame L _) τ ≤
      ENNReal.ofReal (1 - r₁ * r₂) ^ regCount H L τ.2.1.1 τ.2.1.2 := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  show prob (fun b => ¬ (twoRegRec H L (regCount H L x P) b).sampledIncorrect) (twoGame L _) ⟨m, ⟨x, P⟩, k⟩ ≤
    ENNReal.ofReal (1 - r₁ * r₂) ^ regCount H L x P
  rcases regCount_cases H L x P with h0 | ⟨s, Pt, d, hs, hPt, hd, hin, hc⟩
  · rw [h0, pow_zero]
    exact Verity.Game.prob_le_one _ _ _
  · have hrt := hrate x s Pt d hs hPt hd hin
    rw [hc]
    show Verity.Game.prCoin (fun r => ¬ ∃ u ∈ L.checked x r, u ∈ L.FregTwo H P x () r) ≤ _
    rw [Verity.Game.prCoin_eq_ofReal_pr, ← ENNReal.ofReal_pow hx0]
    refine ENNReal.ofReal_le_ofReal (le_trans (Pr_mono ?_)
      (hmiss x s Pt d ⟨s, Pt, d, hs, hPt, hd, hin⟩ hs hPt hd hrt.1 hrt.2 Unit P ()))
    exact fun r hr u hu hF => hr ⟨u, hu, hF⟩

/-- **The period at the committed traces, from each window's committed form and miss bound**: when every window's
verifier meets `CompiledTwo.KSRegTwo` and `CompiledTwo.MissTwo` at rates `r₁` and `r₂`, refuses what is out of its
scope, and has the two stages' rates at every statement in scope, `PeriodTwoReg` holds. -/
theorem periodTwoReg_of (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) {r₁ r₂ : ℝ} (hr₁ : 0 ≤ r₁) (hr₁' : r₁ ≤ 1)
    (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) (hks : ∀ t, (Lv t).KSRegTwo H) (hmiss : ∀ t, (Lv t).MissTwo H r₁ r₂)
    (href : ∀ t x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hrate : ∀ t x s Pt d, (Lv t).parse x = some s → (Lv t).partition s = some Pt → (Lv t).two s = some d →
      InScopeTwo Pt d s → d.FirstRate r₁ ∧ d.SecondRate Pt r₂) :
    PeriodTwoReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv r₁ r₂ := by
  intro T ε s herr M
  have hx0 : 0 ≤ 1 - r₁ * r₂ := by nlinarith
  have h := period_undetected (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoRegRec H (Lv t) (regCount H (Lv t) τ.2.1.1 τ.2.1.2))
    (fun t τ => (Lv t).εregTwo H τ.2.1.1 τ.2.1.2 ≤ ε t) (fun t => ENNReal.ofReal (ε t))
    (ENNReal.ofReal (1 - r₁ * r₂)) (ENNReal.ofReal_le_one.2 (sub_le_self _ (mul_nonneg hr₁ hr₂)))
    (fun t τ hτ => (twoRegRec_accepted_le H (Lv t) (hks t) (href t) _ τ).trans (ENNReal.ofReal_le_ofReal hτ))
    (fun t τ _ => ⟨regCount H (Lv t) τ.2.1.1 τ.2.1.2, fun _ => rfl,
      twoRegRec_missed_le H (Lv t) hx0 (hmiss t) (hrate t) τ⟩)
    T 0 s M (periodGood_of_plays _ _ T 0 s herr)
  simp only [Nat.zero_add] at h
  rw [← ENNReal.ofReal_pow hx0] at h
  exact h

/-- **The period bound from the committed form**: `CompilerKSRegTwo` gives `CompilerPeriodTwoReg`, with each window's
miss bound from its draw laws (`compilerMissTwo`) and its refusal of what is out of scope from its contract
(`CompiledTwo.Sessions.refuses`). -/
theorem compilerPeriodTwoReg_of (hks : CompilerKSRegTwo) : CompilerPeriodTwoReg := by
  intro Coin _ _ _ A O V Tick Adv H Lv r₁ r₂ hA hS hr₁ hr₁' hr₂ hr₂' hrate
  exact periodTwoReg_of H Lv hr₁ hr₁' hr₂ hr₂' (fun t => hks Coin A (Lv t) H hA (hS t).1 (hS t).2)
    (fun t => compilerMissTwo Coin A (Lv t) H (hS t).2 r₁ r₂ hr₁ hr₁' hr₂ hr₂')
    (fun t x P r h => (hS t).2.refuses x Unit P () r h) hrate

/-- **The period bound at the committed traces**: `compilerPeriodTwoReg_of` at `compilerKSRegTwo`. -/
theorem compilerPeriodTwoReg : CompilerPeriodTwoReg :=
  compilerPeriodTwoReg_of compilerKSRegTwo

end WeightSecurity.SecurityProofs
