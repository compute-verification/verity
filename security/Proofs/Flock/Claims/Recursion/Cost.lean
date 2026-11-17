module

public import Definitions.Claims.Recursion
public import Proofs.Flock.Claims.Recursion.Compose
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Proofs.Flock.Claims.ERep

/-! The recursion's cost `Efficient` from the outer extractor `E_O`'s reruns (`efficient_of_EO`, `efficient_le`). -/

@[expose] public section

namespace Claims.Recursion

open Claims

theorem serve_eq {R₁ R₂ α : Type} (q : R₁ → RP R₂ Seen) : ∀ p : RP R₁ α, serve q p = RP.serve q p
  | .ret _ => rfl
  | .hash m k => by
      simp only [serve, RP.serve]
      exact congrArg (RP.hash m) (funext fun a => serve_eq q (k a))
  | .rerun r k => by
      simp only [serve, RP.serve]
      exact congrArg (RP.bind (q r)) (funext fun v => serve_eq q (k v))

theorem served_eq {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Verifier Coin)
    (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) :
    served V E₂ dec m k K opens assemble check = servedERep V E₂ dec m k K opens assemble check := by
  unfold served servedERep
  congr
  funext x r₀ c
  exact serve_eq _ _

theorem efficient_of_EO {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T)
    (hle : ExtractRunsLe H (EO D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap)
    (hexp : ExtractRunsExpLe H (EO D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ)) :
    Efficient H D I DO decO := by
  unfold Efficient E
  rw [served_eq]
  refine ⟨servedERep_runs_le H _ _ _ _ _ _ _ _ _ hle, fun x Ω P ω => ?_⟩
  refine (servedERep_runs_exp H _ _ _ _ _ _ _ _ _ hexp x Ω P ω).trans (le_of_eq ?_)
  simp only [NI]
  push_cast
  ring

theorem collect_runs_le {R α : Type} (H : Msg → Msg) (run : R → Transcript) :
    ∀ ps : List (RP R (Option α)), ((collect ps).eval H run).2.2 ≤ (ps.map fun p => (p.eval H run).2.2).sum
  | [] => by simp [collect, RP.eval]
  | p :: ps => by
      simp only [collect, List.map_cons, List.sum_cons]
      rw [(RP.eval_bind H run p _).2]
      refine add_le_add le_rfl ?_
      rcases (p.eval H run).1 with _ | a
      · simp [RP.eval]
      · simp only
        rw [(RP.eval_bind H run _ _).2]
        simp only [RP.eval, add_zero]
        exact collect_runs_le H run ps

theorem viewing_runs {R : Type} (H : Msg → Msg) (run : R → Transcript) (f : Seen → Seen) (r : R) :
    ((viewing f r).eval H run).2.2 = 1 := by
  simp [viewing, RP.eval]

theorem EO_runs_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) :
    ExtractRunsLe H (EO D I DO decO) fun x => 1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap := by
  intro x Ω P ω r₀ c
  simp only [Extractor.eval, EO, RP.eval]
  split_ifs
  · rw [add_comm 1]
    refine Nat.add_le_add_right ((collect_runs_le H _ _).trans ?_) 1
    rw [List.map_map, Fin.sum_univ_def]
    refine List.sum_le_sum fun s _ => ?_
    simp only [Function.comp, serve_eq]
    refine (RP.serve_runs_le H _ _ _ ((firstWait_runs_le H _ _ _ _ _ _).trans (le_of_eq ?_))
      fun r => (viewing_runs H _ _ r).le).trans (le_of_eq (mul_one _))
    simp only [List.map_map, Function.comp_def, List.length_map, List.length_finRange, List.map_const',
      List.sum_replicate, smul_eq_mul]
  · simp [RP.eval]

theorem efficient_le {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) :
    ExtractRunsLe H (E H D I DO decO) fun x => (1 + I.waits (xI D x) * I.cap (xI D x)) *
      (1 + ∑ s : Fin (D.nS x), (D.nums x s).waits * (D.nums x s).cap) := by
  unfold E
  rw [served_eq]
  exact servedERep_runs_le H _ _ _ _ _ _ _ _ _ (EO_runs_le H D I DO decO)

end Claims.Recursion
