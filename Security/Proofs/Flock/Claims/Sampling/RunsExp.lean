module

public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Cost
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Monotonicity.Lemmas
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
import Mathlib.Topology.Sheaves.Presheaf
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Runs

/-!
# A wait's expected reruns

The compiled sampled extractor's expected reruns (`SamplingGeneric.Compiled`, `Proofs.Flock.SamplingGeneric.Efficient`)
amortize each wait against the acceptance probability it waits on (`waitRuns_exp`, `accExp_le`): a wait at runs
accepting with probability `ε` makes `k/ε` reruns in expectation, and its accepted runs weigh as runs given acceptance
(`waitAccC_sum`). This file holds the steps it reads:

* `sum_zip_finRange`, `sum_zip_snd_le`, `accSum_map`, `accSum_nonneg`, `waitExp_nonneg`, `accExp_nonneg`: sums along
  a wait's kept runs.
* `wait_flat_runs`, `outer_runs_le_sum`: a wait and the programs at its kept runs rerun as the wait does and then as
  each program.
* `level_ex`, `level_amortized`: **one wait and a body at each kept run**, in expectation over the wait's samples and
  the bodies' coins, and against the wait's acceptance probability.
-/

@[expose] public section

namespace Claims.Sampling

open Claims

/-! ## Sums along a wait's kept runs -/

theorem sum_zip_finRange {α : Type} : ∀ (n : ℕ) (F : Fin n × α → ℝ) (l : List α),
    (((List.finRange n).zip l).map F).sum = ∑ i : Fin n, (l[i.val]?.map fun a => F (i, a)).getD 0
  | 0, _, _ => by simp
  | _ + 1, _, [] => by simp
  | n + 1, F, a :: l => by
    rw [List.finRange_succ, List.zip_cons_cons, List.map_cons, List.sum_cons, List.zip_map_left, List.map_map,
      sum_zip_finRange n (F ∘ Prod.map Fin.succ id) l, Fin.sum_univ_succ]
    simp

theorem sum_zip_snd_le {α β : Type} (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) :
    ∀ (l₁ : List β) (l₂ : List α), ((l₁.zip l₂).map fun q => f q.2).sum ≤ (l₂.map f).sum
  | [], l₂ => by
    simpa using List.sum_nonneg fun y hy => by
      obtain ⟨a, -, rfl⟩ := List.mem_map.1 hy
      exact hf a
  | _ :: _, [] => by simp
  | _ :: l₁, a :: l₂ => by
    simp only [List.zip_cons_cons, List.map_cons, List.sum_cons]
    exact add_le_add le_rfl (sum_zip_snd_le f hf l₁ l₂)

theorem accSum_map {R R' : Type} (run : R → Transcript) (k : ℕ) (g : R → ℝ) (f : R' → R) :
    ∀ (l : List R') (a : ℕ), accSum run k g (l.map f) a = accSum (run ∘ f) k (g ∘ f) l a
  | [], _ => rfl
  | r :: l, a => by simp only [List.map_cons, accSum, accSum_map run k g f l, Function.comp]

theorem accSum_nonneg {R : Type} (run : R → Transcript) (k : ℕ) {g : R → ℝ} (hg : ∀ r, 0 ≤ g r) :
    ∀ (l : List R) (a : ℕ), 0 ≤ accSum run k g l a
  | [], _ => le_rfl
  | r :: l, a => by
    simp only [accSum]
    split_ifs
    · exact le_rfl
    · exact add_nonneg (hg r) (accSum_nonneg run k hg l _)
    · exact accSum_nonneg run k hg l _

theorem waitExp_nonneg {D : Type} [Fintype D] (run : D → Transcript) (k K a : ℕ) : 0 ≤ waitExp run k K a :=
  Ex_nonneg fun _ => Nat.cast_nonneg _

theorem accExp_nonneg {D : Type} [Fintype D] (run : D → Transcript) (k : ℕ) {g : D → ℝ} (hg : ∀ d, 0 ≤ g d) (K a : ℕ) :
    0 ≤ accExp run k g K a :=
  Ex_nonneg fun _ => accSum_nonneg run k hg _ _

/-- A wait and the programs at its kept runs, indexed, rerun as the wait does and then as each program. Stated at
every `k`, so that the kernel never meets the deployed `k`'s `List.finRange`. -/
theorem wait_flat_runs {R γ : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (rs : List R)
    (f : Fin k × (R × Seen) → RP R (List γ)) :
    (((waitAccC k rs []).bind fun l => flatMapRP f ((List.finRange k).zip l)).eval H run).2.2 =
      waitRuns run k rs 0 + (((List.finRange k).zip ((waitAccC k rs []).eval H run).1).map fun q =>
        ((f q).eval H run).2.2).sum := by
  rw [(RP.eval_bind _ _ _ _).2, waitAccC_runs_eq, flatMapRP_runs, List.length_nil]

/-- **The outer wait and what follows it**, as reals: the wait's reruns and each kept run's body bound `B`. Stated at
every `k`, so that the kernel never meets the deployed `k`'s `List.finRange`. -/
theorem outer_runs_le_sum {R γ β : Type} (H : Msg → Msg) (run : R → Transcript) (k : ℕ) (rs : List R)
    (body : Fin k × (R × Seen) → RP R (List γ)) (fin : List γ → β) (B : Fin k × (R × Seen) → ℝ)
    (hB : ∀ jo, (((body jo).eval H run).2.2 : ℝ) ≤ B jo) :
    ((((waitAccC k rs []).bind fun outer =>
        (flatMapRP body ((List.finRange k).zip outer)).bind fun found => RP.ret (fin found)).eval H run).2.2 : ℝ) ≤
      waitRuns run k rs 0 + (((List.finRange k).zip ((waitAccC k rs []).eval H run).1).map B).sum := by
  rw [(RP.eval_bind _ _ _ _).2, (RP.eval_bind _ _ _ _).2, waitAccC_runs_eq, flatMapRP_runs]
  simp only [RP.eval, List.length_nil, Nat.add_zero, Nat.cast_add, Nat.cast_list_sum, List.map_map]
  exact add_le_add le_rfl (List.sum_le_sum fun jo _ => hB jo)

/-- **One wait and a body at each kept run, in expectation**: over the wait's `K` samples (each `f` of a uniform `D'`)
and the bodies' coins `X`, the reruns are at most the wait's expected reruns and its kept runs' weights `G`, when each
body averages at most `G` of its run, at every position. -/
theorem level_ex {D' R X : Type} [Fintype D'] [Nonempty D'] [Fintype X] [Nonempty X] (H : Msg → Msg)
    (run : R → Transcript) (f : D' → R) (k K : ℕ) (Bf : Fin k → R → X → ℝ) (G : R → ℝ) (hB : ∀ j r, Ex (Bf j r) ≤ G r)
    (hG : ∀ r, 0 ≤ G r) :
    Ex (fun c : (Fin K → D') × X => (waitRuns run k ((List.finRange K).map fun l => f (c.1 l)) 0 : ℝ) +
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (c.1 l)) []).eval H run).1).map
          fun jo => Bf jo.1 jo.2.1 c.2).sum) ≤
      waitExp (run ∘ f) k K 0 + accExp (run ∘ f) k (G ∘ f) K 0 := by
  have hstep : ∀ σ : Fin K → D', (Ex fun ξ : X =>
      (waitRuns run k ((List.finRange K).map fun l => f (σ l)) 0 : ℝ) +
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => Bf jo.1 jo.2.1 ξ).sum) ≤
      (waitRuns (run ∘ f) k ((List.finRange K).map σ) 0 : ℝ) +
        accSum (run ∘ f) k (G ∘ f) ((List.finRange K).map σ) 0 := by
    intro σ
    have hw : waitRuns run k ((List.finRange K).map fun l => f (σ l)) 0 =
        waitRuns (run ∘ f) k ((List.finRange K).map σ) 0 := by
      rw [← waitRuns_map, List.map_map]
      rfl
    rw [Ex_add, Ex_const, Ex_listSum, hw]
    refine add_le_add le_rfl ?_
    calc (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => Ex (Bf jo.1 jo.2.1)).sum
        ≤ (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => G jo.2.1).sum := List.sum_le_sum fun jo _ => hB jo.1 jo.2.1
      _ ≤ (((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1.map fun p => G p.1).sum :=
          sum_zip_snd_le (fun p : R × Seen => G p.1) (fun p => hG p.1) _ _
      _ = accSum run k G ((List.finRange K).map fun l => f (σ l)) 0 := by
          rw [waitAccC_sum]
          simp
      _ = accSum (run ∘ f) k (G ∘ f) ((List.finRange K).map σ) 0 := by
          rw [← accSum_map, List.map_map]
          rfl
  rw [Ex_prod]
  refine (Ex_mono hstep).trans_eq ?_
  rw [Ex_add]
  rfl

/-- `level_ex`'s bound against the wait's acceptance probability: at most `k` reruns for the wait, and `k` times
the kept runs' weights given acceptance. -/
theorem level_amortized {D R : Type} [Fintype D] [Nonempty D] (run : R → Transcript) (f : D → R) (k K : ℕ) (G : R → ℝ)
    (hG : ∀ r, 0 ≤ G r) :
    Pr (fun d => (run (f d)).out = .acc) * (waitExp (run ∘ f) k K 0 + accExp (run ∘ f) k (G ∘ f) K 0) ≤
      k + k * Ex (fun d => (if (run (f d)).out = .acc then (1 : ℝ) else 0) * G (f d)) := by
  have h1 := waitRuns_exp (run ∘ f) k K 0 (Nat.zero_le _)
  have h2 := accExp_le (run ∘ f) k (G ∘ f) (fun d => hG (f d)) K 0 (Nat.zero_le _)
  simp only [Function.comp_apply, Nat.cast_zero, sub_zero] at h1 h2
  rw [mul_add]
  exact add_le_add h1 h2

end Claims.Sampling
