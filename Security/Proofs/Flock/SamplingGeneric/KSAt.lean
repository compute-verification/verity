module

public import Definitions.Claims.SamplingGeneric
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Sampling.Bext

/-!
# The sampled compiler's KS read at a time budget

`KSAt` reads `KS`'s two hash terms at a budget `T` on one run's whole SHA-512 work instead of at the calls the oracle
logs: the birthday term at `B` declared at `2T` (`sbB`), the binding term at `B_ext` declared at `T` a rerun
(`BextAt`). Each reading bounds the probability it replaces, and is `⊤` wherever its finder fails the assumption
(`ofReal_sqrt_pr_le_sbBound`, `ofReal_pr_le_ecrBound`), and `B_ext` at `T` has `B_ext`'s coins and outcome
(`BextAt_pr`). So `KSLinOn` at `β = 1` and `B_ext` gives `KSAt` at every budget, whatever `κ`, `γ` and `work`
(`Sampling.KSAt_of_KSLinOn`).

`B_ext` at `T` costs at most `2·(own + T·runs)` in expectation, the extractor's own calls and `T` a rerun
(`Sampling.BextAt_cost_le`, as `Bext_cost_le` counts the reruns' logged calls), so at most `2·T·work` at `T ≥ 1`
wherever `runs + own ≤ work` (`Sampling.BextAtCostLe_of`). At a compiled sampled verifier those are `KS` and
`Efficient`'s expected reruns and own calls (`Compiled.ksAt_of_ks`), so `CompilerKS` and `CompilerEfficient` give
`CompilerKSAt` (`compilerKSAt_of`), beside `CompilerBits`, its reading at the calls the oracle logs.
-/

@[expose] public section

namespace Claims

/-- **`sbBound` bounds the root of a finder's probability**: `√Pr ≤ q/2^256.5` where the finder meets `StrictBirthday`
with at most `q` calls on every coin, and `sbBound` is `⊤` elsewhere. -/
theorem ofReal_sqrt_pr_le_sbBound (H : Msg → Msg) (F : Finder) (q : ℝ) :
    ENNReal.ofReal (Real.sqrt (F.Pr H)) ≤ sbBound H F q := by
  unfold sbBound
  split_ifs with h
  · have hq : 0 ≤ q := (Nat.cast_nonneg _).trans (h.2 (Classical.arbitrary F.S))
    have h513 : Real.sqrt ((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
      rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity), ← Real.rpow_add (by norm_num)]
      norm_num
    refine ENNReal.ofReal_le_ofReal ?_
    calc Real.sqrt (F.Pr H) ≤ Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ)) := Real.sqrt_le_sqrt (h.1 q h.2)
      _ = q / 2 ^ (256.5 : ℝ) := by rw [Real.sqrt_div (by positivity), Real.sqrt_sq hq, h513]
  · exact le_top

/-- **`ecrBound` bounds a finder's probability**: `Pr ≤ q/2^256` where the finder meets `ExpectedCR` within `q`
expected calls, and `ecrBound` is `⊤` elsewhere. -/
theorem ofReal_pr_le_ecrBound (H : Msg → Msg) (F : Finder) (q : ℝ) :
    ENNReal.ofReal (F.Pr H) ≤ ecrBound H F q := by
  unfold ecrBound
  split_ifs with h
  · exact ENNReal.ofReal_le_ofReal (h.1.trans (div_le_div_of_nonneg_right h.2 (by positivity)))
  · exact le_top

/-- **The birthday term at a budget**: `√Pr[B] ≤ sbB` at every budget `q`, since `B` declared at `q` has `B`'s coins and
outcome. -/
theorem ofReal_sqrt_prB_le_sbB {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin]
    [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (q : ℕ) :
    ENNReal.ofReal (Real.sqrt (PrB H V P x)) ≤ sbB H V P x q := by
  unfold PrB sbB
  by_cases hk : 0 < V.k x
  · simp only [dite_eq_left hk]
    exact ofReal_sqrt_pr_le_sbBound H (BAt H V P x hk q) q
  · simp only [dite_eq_right hk, Real.sqrt_zero, ENNReal.ofReal_zero, le_refl]

end Claims

namespace Claims.Sampling

open Claims

section KSAt

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Nonempty Coin] in
/-- `B_ext` at a budget has `B_ext`'s coins and outcome, so its probability. -/
theorem BextAt_pr (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) (T : ℕ) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) : (BextAt H V E T Ω P x).Pr H = (Bext H V E Ω P x).Pr H := by
  unfold BextAt Bext
  by_cases hk : 0 < V.k x
  · simp only [dite_eq_left hk]
    rfl
  · simp only [dite_eq_right hk]

/-- **`KSAt` from `KSLinOn` at `β = 1` and `B_ext`**, at every budget and whatever `κ`, `γ` and `work`: the birthday
term's reading at `2T` bounds `√(γ·Pr[B])`, and the binding term's at any cost bound bounds `Pr[B_ext]`. -/
theorem KSAt_of_KSLinOn {H : Msg → Msg} {V : Verifier Coin} {Scope : Msg → Prop} {R : Rel W} {E : Extractor Coin W V}
    {κ γ : Msg → ℝ} (work : Msg → ℝ) (h : KSLinOn H V Scope R E κ γ (fun _ => 1) (Bext H V E)) :
    KSAt H V Scope R E κ γ work := by
  intro x hx Ω _ _ P T
  have hks := h x hx Ω P
  rw [one_mul] at hks
  have hB : 0 ≤ PrB H V P x := by
    unfold PrB
    split_ifs
    · exact Pr_nonneg _
    · exact le_rfl
  calc ENNReal.ofReal (PrBad H V R E x P)
      ≤ ENNReal.ofReal (κ x) + ENNReal.ofReal (Real.sqrt (γ x * PrB H V P x)) +
          ENNReal.ofReal ((Bext H V E Ω P x).Pr H) :=
        (ENNReal.ofReal_le_ofReal hks).trans
          (ENNReal.ofReal_add_le.trans (add_le_add ENNReal.ofReal_add_le le_rfl))
    _ ≤ _ := by
        rw [Real.sqrt_mul' _ hB, ENNReal.ofReal_mul (Real.sqrt_nonneg _), ← BextAt_pr H V E T Ω P x]
        gcongr
        · exact ofReal_sqrt_prB_le_sbB H V P x (2 * T)
        · exact ofReal_pr_le_ecrBound H _ _

omit [Nonempty Coin] in
/-- A bound on an extractor's expected reruns is never negative. -/
theorem runs_nonneg {H : Msg → Msg} {V : Verifier Coin} {E : Extractor Coin W V} {runs : Msg → ℝ}
    (hruns : ExtractRunsExpLe H E runs) (x : Msg) : 0 ≤ runs x := by
  refine le_trans ?_ (hruns x Unit ⟨fun m _ _ => .ret m⟩ ())
  exact Ex_nonneg fun _ => Nat.cast_nonneg _

/-- **`B_ext` at a budget, its expected cost**: an extractor with at most `own(x)` own calls and `runs(x)` reruns in
expectation makes `B_ext` at the budget `T` cost at most `2·(own(x) + T·runs(x))` in expectation, as `Bext_cost_le`
with each rerun at `T`. -/
theorem BextAt_cost_le (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) (T : ℕ) {runs : Msg → ℝ} {own : Msg → ℕ}
    (hruns : ExtractRunsExpLe H E runs) (hown : ExtractOwnLe H E own) :
    Ex (fun s => ((BextAt H V E T Ω P x).cost s : ℝ)) ≤ 2 * (own x + T * runs x) := by
  have : Nonempty (Fin (V.k x)) := ⟨⟨0, hk⟩⟩
  have hb1 : ∀ f : Ω × (Fin (V.k x) → Coin) × E.C x → ℝ,
      Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x => f (s.1.2.1, s.1.2.2.1, s.2.1)) = Ex f := fun f => by
    rw [← Ex_fst (β := Fin (V.k x) × (Fin (V.k x) → Coin) × E.C x) f,
      ← Ex_equiv (bextSplit (Fin (V.k x)) Ω (Fin (V.k x) → Coin) (E.C x))]
    rfl
  have hb2 : ∀ f : Ω × (Fin (V.k x) → Coin) × E.C x → ℝ, Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
      f (s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)) = Ex f := fun f => by
    rw [← hb1 f]
    exact Ex_equiv (bextSwap Ω Coin (E.C x) (V.k x)) fun s => f (s.1.2.1, s.1.2.2.1, s.2.1)
  set o : Ω × (Fin (V.k x) → Coin) × E.C x → ℝ :=
    fun q => ((ownLog H (run H P V x q.1) (E.ext x q.2.1 q.2.2)).length : ℝ) with ho
  set g : Ω × (Fin (V.k x) → Coin) × E.C x → ℝ := fun q => ((E.eval H P x q.1 q.2.1 q.2.2).2.2 : ℝ) with hg
  have ho1 : Ex o ≤ own x := by
    calc Ex o ≤ Ex (fun _ : Ω × (Fin (V.k x) → Coin) × E.C x => (own x : ℝ)) :=
          Ex_mono fun q => Nat.cast_le.mpr (hown x Ω P q.1 q.2.1 q.2.2)
      _ = own x := Ex_const _
  have hg1 : Ex g ≤ runs x := by
    calc Ex g = Ex (fun ω : Ω => Ex (fun q : (Fin (V.k x) → Coin) × E.C x =>
          ((E.eval H P x ω q.1 q.2).2.2 : ℝ))) := Ex_prod g
      _ ≤ Ex (fun _ : Ω => runs x) := Ex_mono fun ω => hruns x Ω P ω
      _ = runs x := Ex_const _
  unfold BextAt
  rw [dite_eq_left hk]
  show Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
    (((ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1)).length +
      (ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2)).length +
      T * ((E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.2 +
        (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.2) : ℕ) : ℝ)) ≤ _
  simp only [Nat.cast_add, Nat.cast_mul]
  show Ex (fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
    o (s.1.2.1, s.1.2.2.1, s.2.1) + o (s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2) +
      (T : ℝ) * (g (s.1.2.1, s.1.2.2.1, s.2.1) + g (s.1.2.1, forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2))) ≤ _
  rw [Ex_add, Ex_add, Ex_mul_left, Ex_add, hb1 o, hb2 o, hb1 g, hb2 g]
  have hT : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  nlinarith

/-- **`B_ext` at a budget costs at most `2·T·work`** at every budget `T ≥ 1`, at an extractor whose expected reruns
`runs` and own calls `own` come to at most `work`. -/
theorem BextAtCostLe_of {H : Msg → Msg} {V : Verifier Coin} {E : Extractor Coin W V} {runs work : Msg → ℝ}
    {own : Msg → ℕ} (hruns : ExtractRunsExpLe H E runs) (hown : ExtractOwnLe H E own)
    (hw : ∀ x, runs x + own x ≤ work x) : BextAtCostLe H V E work := by
  intro x Ω _ _ P T hT
  have hT1 : (1 : ℝ) ≤ T := by exact_mod_cast hT
  have ho : (0 : ℝ) ≤ own x := Nat.cast_nonneg _
  have hr := runs_nonneg hruns x
  have hw' := hw x
  by_cases hk : 0 < V.k x
  · refine (BextAt_cost_le H V E P x hk T hruns hown).trans ?_
    nlinarith
  · have h0 : Ex (fun s => ((BextAt H V E T Ω P x).cost s : ℝ)) = 0 := by
      unfold BextAt
      rw [dite_eq_right hk]
      show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) = 0
      rw [Nat.cast_zero, Ex_const]
    have : 0 ≤ work x := by linarith
    rw [h0]
    positivity

end KSAt

end Claims.Sampling

namespace Claims.SamplingGeneric

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`KSAt` from `KS`**, at an extractor whose expected reruns `runs` and own SHA-512 calls `own` come to at most
`work`. -/
theorem ksAt_of_ks {H : Msg → Msg} {L : Compiled Coin A} (hks : L.KS H) {runs : Msg → ℝ} {own : Msg → ℕ}
    (hruns : ExtractRunsExpLe H L.E runs) (hown : Sampling.ExtractOwnLe H L.E own)
    (hw : ∀ x, runs x + own x ≤ L.work x) : L.KSAt H :=
  ⟨fun δ hδ => Sampling.KSAt_of_KSLinOn L.work (hks δ hδ), Sampling.BextAtCostLe_of hruns hown hw⟩

end Compiled

/-- **KS read at a time budget**, from `CompilerKS` and `CompilerEfficient`'s expected reruns and own calls, which
come to `work`. -/
theorem compilerKSAt_of (hks : CompilerKS) (heff : CompilerEfficient) : CompilerKSAt := by
  intro Coin _ _ _ A L H hA hB hS hopen
  obtain ⟨-, hr, ho⟩ := heff Coin A L H hS hopen
  exact Compiled.ksAt_of_ks (hks Coin A L H hA hB hS) hr ho
    fun x => le_of_eq (by unfold Compiled.work; push_cast; ring)

end Claims.SamplingGeneric
