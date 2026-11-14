module

public import Proofs.WeightSecurity.Reg
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Tape
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Session
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.SamplingGeneric.Reg

/-!
# A period of one-stage windows at the committed traces, proved

`PeriodReg` is `period_undetected` at `oneRegRec`. Each record reads only the strategy its window is played with, so
both of a window's bounds hold at every strategy: `hacc` is `Compiled.KSReg`'s first conjunct at the statement and
prover it sends, and `hstage` is the contract's sample (`Compiled.Sessions.draws`) at the committed incorrect set,
which the statement and prover fix before the run's coins.

* `oneRegRec_accepted_le`: a window is accepted with a sampled proof block in the committed incorrect set with
  probability at most `ε_reg` at the statement and prover it sends, and never out of scope.
* `oneRegRec_missed_le`: its sample misses the committed incorrect set with probability at most `x^n`, `n` the count
  its strategy fixes.
* `periodReg_of`: `PeriodReg` from each window's `KSReg` and contract, at designs that miss each set `B` with
  probability at most `x^{|B|}`.
* `compilerPeriodReg_of`: `CompilerPeriodReg` from `CompilerKSReg`. So **`CompilerPeriodReg` is proved**
  (`compilerPeriodReg`), at `compilerKSReg`.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

open Classical in
/-- **The count a window fixes** is `0`, or its statement is in scope and the count is the committed incorrect set's
size. -/
theorem oneRegCount_cases (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (P : Prover Unit) :
    oneRegCount H L x P = 0 ∨ L.Scope x ∧ oneRegCount H L x P = (L.Freg H P x ()).card := by
  by_cases hx : L.Scope x
  · exact Or.inr ⟨hx, by rw [oneRegCount, ite_eq_left hx]⟩
  · exact Or.inl (by rw [oneRegCount, ite_eq_right hx])

/-- **A window is accepted with a sampled proof block in the committed incorrect set** at most as often as `ε_reg` at
the statement and prover it sends, at a statement in scope by `Compiled.KSReg` at a prover with no tape, and out of
scope never, since the verifier refuses it. -/
theorem oneRegRec_accepted_le (H : Msg → Msg) (L : Compiled Coin A) (hks : L.KSReg H)
    (href : ∀ x (P : Prover Unit) r, (run H P L.V x () r).out = .acc → L.Scope x) (n : ℕ)
    (τ : Strategy (oneGame L ((O → V) × Tick × Adv))) :
    prob (fun b => (oneRegRec H L n b).accepted ∧ (oneRegRec H L n b).sampledIncorrect) (oneGame L _) τ ≤
      ENNReal.ofReal (L.εreg H τ.2.1.1 τ.2.1.2) := by
  obtain ⟨m, ⟨x, P⟩, k⟩ := τ
  show Verity.Game.prCoin (fun r => (run H P L.V x () r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ()) ≤ ENNReal.ofReal (L.εreg H x P)
  by_cases hx : L.Scope x
  · rw [Verity.Game.prCoin_eq_ofReal_pr]
    refine ENNReal.ofReal_le_ofReal (le_trans (le_of_eq ?_) (hks x hx Unit P).1)
    exact (Pr_unit fun r => (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ()).symm
  · rw [show (fun r => (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ()) =
        fun _ => False from funext fun r => propext (iff_false_intro fun h => hx (href x P r h.1)),
      Verity.Game.prCoin_false]
    exact zero_le

/-- **A window's sample misses the committed incorrect set** with probability at most `x^n`, `n` the count its strategy
fixes: in scope by the contract's sample at that set (`Sessions.draws`), which the statement and prover fix before the
coins, and the design's `P_a(B) ≤ x^{|B|}`; out of scope the count is `0`. -/
theorem oneRegRec_missed_le (H : Msg → Msg) (L : Compiled Coin A) (hS : L.Sessions H) {x : ℝ} (hx0 : 0 ≤ x)
    (hpa : ∀ y s P, L.parse y = some s → L.partition s = some P → InScope P s →
      ∀ B ⊆ Finset.range P.blocks.length, s.design.Pa P B ≤ x ^ B.card)
    (τ : Strategy (oneGame L ((O → V) × Tick × Adv))) :
    prob (fun b => ¬ (oneRegRec H L (oneRegCount H L τ.2.1.1 τ.2.1.2) b).sampledIncorrect) (oneGame L _) τ ≤
      ENNReal.ofReal x ^ oneRegCount H L τ.2.1.1 τ.2.1.2 := by
  obtain ⟨m, ⟨y, P⟩, k⟩ := τ
  show prob (fun b => ¬ (oneRegRec H L (oneRegCount H L y P) b).sampledIncorrect) (oneGame L _) ⟨m, ⟨y, P⟩, k⟩ ≤
    ENNReal.ofReal x ^ oneRegCount H L y P
  rcases oneRegCount_cases H L y P with h0 | ⟨⟨s, Pt, hs, hPt, hin⟩, hc⟩
  · rw [h0, pow_zero]
    exact Verity.Game.prob_le_one _ _ _
  · rw [hc]
    show Verity.Game.prCoin (fun r => ¬ ∃ u ∈ L.checked y r, u ∈ L.Freg H P y ()) ≤ _
    rw [Verity.Game.prCoin_eq_ofReal_pr, ← ENNReal.ofReal_pow hx0]
    have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
    have hsub : L.Freg H P y () ⊆ Finset.range Pt.blocks.length := by
      intro v hv
      simp only [Compiled.Freg, hs, Finset.mem_filter] at hv
      rw [← hpart]
      exact hv.1
    obtain ⟨esc, -, hesc, hEesc⟩ := hS.draws y s Pt hs hPt hin _ hsub
    refine ENNReal.ofReal_le_ofReal ?_
    calc Pr (fun r : Fin (L.V.k y) → Coin => ¬ ∃ u ∈ L.checked y r, u ∈ L.Freg H P y ())
        ≤ Pr (fun r : Fin (L.V.k y) → Coin => ∀ u ∈ L.checked y r, u ∉ L.Freg H P y ()) :=
          Pr_mono fun _ hr u hu hF => hr ⟨u, hu, hF⟩
      _ = Ex fun r => Pr fun d : Fin (L.V.k y) → Coin =>
            ∀ u ∈ L.checked y (forkOn (L.stage2 y) r d), u ∉ L.Freg H P y () := Pr_forkOn (L.stage2 y) _
      _ ≤ Ex esc := Ex_mono hesc
      _ ≤ s.design.Pa Pt (L.Freg H P y ()) := hEesc
      _ ≤ x ^ (L.Freg H P y ()).card := hpa y s Pt hs hPt hin _ hsub

/-- **The period at the committed traces, from each window's committed form and contract**: when every window's
verifier meets `Compiled.KSReg` and the contract, and every statement in its scope has a sampling design that misses
each set `B` of proof blocks with probability at most `x^{|B|}`, `PeriodReg` holds. -/
theorem periodReg_of (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1)
    (hks : ∀ t, (Lv t).KSReg H) (hS : ∀ t, (Lv t).Sessions H)
    (hpa : ∀ t y s P, (Lv t).parse y = some s → (Lv t).partition s = some P → InScope P s →
      ∀ B ⊆ Finset.range P.blocks.length, s.design.Pa P B ≤ x ^ B.card) :
    PeriodReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv x := by
  intro T ε s herr M
  have h := period_undetected (fun t => oneGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => oneRegRec H (Lv t) (oneRegCount H (Lv t) τ.2.1.1 τ.2.1.2))
    (fun t τ => (Lv t).εreg H τ.2.1.1 τ.2.1.2 ≤ ε t) (fun t => ENNReal.ofReal (ε t))
    (ENNReal.ofReal x) (ENNReal.ofReal_le_one.2 hx1)
    (fun t τ hτ => (oneRegRec_accepted_le H (Lv t) (hks t) (fun y P r h => (hS t).refuses y Unit P () r h) _
      τ).trans (ENNReal.ofReal_le_ofReal hτ))
    (fun t τ _ => ⟨oneRegCount H (Lv t) τ.2.1.1 τ.2.1.2, fun _ => rfl,
      oneRegRec_missed_le H (Lv t) (hS t) hx0 (hpa t) τ⟩)
    T 0 s M (monitoringPeriodGood_of_plays _ _ T 0 s herr)
  simp only [Nat.zero_add] at h
  rw [← ENNReal.ofReal_pow hx0] at h
  exact h

/-- **The period bound from the committed form**: `CompilerKSReg` gives `CompilerPeriodReg`, with each window's miss
from its sampling design and its refusal of what is out of scope, both from its contract (`Compiled.Sessions`). -/
theorem compilerPeriodReg_of (hks : CompilerKSReg) : CompilerPeriodReg := by
  intro Coin _ _ _ A O V Tick Adv H Lv x hA hS hx0 hx1 hpa
  exact periodReg_of H Lv hx0 hx1 (fun t => hks Coin A (Lv t) H hA (hS t).1 (hS t).2) (fun t => (hS t).2) hpa

/-- **The period bound at the committed traces**: `compilerPeriodReg_of` at `compilerKSReg`. -/
theorem compilerPeriodReg : CompilerPeriodReg :=
  compilerPeriodReg_of compilerKSReg

end WeightSecurity.SecurityProofs
