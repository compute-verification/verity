module

public import Definitions.Core.Interaction
public import Proofs.Flock.Claims.ERepCost
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
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

/-! Serving one rerun program's reruns by another's (`RP.serve`), whose worst cases multiply. -/

@[expose] public section

namespace Claims

def RP.serve {R₁ R₂ α : Type} (q : R₁ → RP R₂ Seen) : RP R₁ α → RP R₂ α
  | .ret a => .ret a
  | .hash m k => .hash m fun a => RP.serve q (k a)
  | .rerun r k => (q r).bind fun v => RP.serve q (k v)

def servedRun {R₁ R₂ : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript) (r : R₁) : Transcript :=
  ⟨((q r).eval H run).1.msgs, [], [], ((q r).eval H run).1.out⟩

theorem RP.eval_runs_eq {R α : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) :
    (p.eval H run).2.2 = (p.rerunsAt H run).length := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [RP.eval, RP.rerunsAt]; exact ih _
  | rerun r k ih => simp only [RP.eval, RP.rerunsAt, List.length_cons]; rw [ih]

theorem RP.eval_bind_hashes {R α β : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).2.1 = (p.eval H run).2.1 + ((f (p.eval H run).1).eval H run).2.1 := by
  induction p with
  | ret a => simp [RP.bind, RP.eval]
  | hash q k ih => simp only [RP.bind, RP.eval]; rw [ih]; ring
  | rerun r k ih => simp only [RP.bind, RP.eval]; rw [ih]; ring

theorem RP.rerunsAt_bind {R α β : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) (f : α → RP R β) :
    (p.bind f).rerunsAt H run = p.rerunsAt H run ++ (f (p.eval H run).1).rerunsAt H run := by
  induction p with
  | ret a => simp [RP.bind, RP.eval, RP.rerunsAt]
  | hash q k ih => simp only [RP.bind, RP.eval, RP.rerunsAt]; exact ih _
  | rerun r k ih => simp only [RP.bind, RP.eval, RP.rerunsAt, List.cons_append]; rw [ih]

theorem servedRun_seen {R₁ R₂ : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript) (r : R₁) :
    (servedRun H q run r).seen = ((q r).eval H run).1 := rfl

theorem RP.serve_eval {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript) (p : RP R₁ α) :
    ((p.serve q).eval H run).1 = (p.eval H (servedRun H q run)).1 ∧
      ((p.serve q).eval H run).2.1 = (p.eval H (servedRun H q run)).2.1 +
        ((p.rerunsAt H (servedRun H q run)).map fun r => ((q r).eval H run).2.1).sum ∧
      ((p.serve q).eval H run).2.2 =
        ((p.rerunsAt H (servedRun H q run)).map fun r => ((q r).eval H run).2.2).sum := by
  induction p with
  | ret a => simp [RP.serve, RP.eval, RP.rerunsAt]
  | hash m k ih =>
    obtain ⟨h1, h2, h3⟩ := ih (H m)
    simp only [RP.serve, RP.eval, RP.rerunsAt]
    refine ⟨h1, ?_, h3⟩
    rw [h2]; ring
  | rerun r k ih =>
    obtain ⟨h1, h2, h3⟩ := ih ((q r).eval H run).1
    simp only [RP.serve, RP.eval, RP.rerunsAt, servedRun_seen, List.map_cons, List.sum_cons]
    refine ⟨(RP.eval_bind H run _ _).1.trans h1, ?_, ?_⟩
    · rw [RP.eval_bind_hashes, h2]
      simp only [servedRun, List.length_nil]
      ring
    · rw [(RP.eval_bind H run _ _).2, h3]

theorem RP.serve_runs_cast {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript)
    (p : RP R₁ α) :
    (((p.serve q).eval H run).2.2 : ℝ) =
      ((p.rerunsAt H (servedRun H q run)).map fun r => (((q r).eval H run).2.2 : ℝ)).sum := by
  rw [(RP.serve_eval H q run p).2.2, Nat.cast_list_sum, List.map_map]
  rfl

theorem RP.serve_hashes_cast {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript)
    (p : RP R₁ α) :
    (((p.serve q).eval H run).2.1 : ℝ) = ((p.eval H (servedRun H q run)).2.1 : ℝ) +
      ((p.rerunsAt H (servedRun H q run)).map fun r => (((q r).eval H run).2.1 : ℝ)).sum := by
  rw [(RP.serve_eval H q run p).2.1, Nat.cast_add, Nat.cast_list_sum, List.map_map]
  rfl

theorem list_sum_le_mul {R : Type} (l : List R) (f : R → ℕ) (N : ℕ) (h : ∀ r ∈ l, f r ≤ N) :
    (l.map f).sum ≤ l.length * N := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have := ih fun r hr => h r (List.mem_cons_of_mem a hr)
    have := h a List.mem_cons_self
    nlinarith

theorem RP.serve_runs_le {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript)
    (p : RP R₁ α) {N₁ N₂ : ℕ} (hp : (p.eval H (servedRun H q run)).2.2 ≤ N₁)
    (hq : ∀ r, ((q r).eval H run).2.2 ≤ N₂) : ((p.serve q).eval H run).2.2 ≤ N₁ * N₂ := by
  rw [(RP.serve_eval H q run p).2.2]
  refine (list_sum_le_mul _ _ N₂ fun r _ => hq r).trans ?_
  rw [← RP.eval_runs_eq]
  exact Nat.mul_le_mul_right N₂ hp

theorem RP.serve_hashes_le {R₁ R₂ α : Type} (H : Msg → Msg) (q : R₁ → RP R₂ Seen) (run : R₂ → Transcript)
    (p : RP R₁ α) {N₁ Nh : ℕ} (hp : (p.eval H (servedRun H q run)).2.2 ≤ N₁)
    (hq : ∀ r, ((q r).eval H run).2.1 ≤ Nh) :
    ((p.serve q).eval H run).2.1 ≤ (p.eval H (servedRun H q run)).2.1 + N₁ * Nh := by
  rw [(RP.serve_eval H q run p).2.1]
  refine Nat.add_le_add_left ((list_sum_le_mul _ _ Nh fun r _ => hq r).trans ?_) _
  rw [← RP.eval_runs_eq]
  exact Nat.mul_le_mul_right Nh hp

theorem RP.eval_bind_ret {R α β : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) (f : α → β) :
    ((p.bind fun a => .ret (f a)).eval H run) = (f (p.eval H run).1, (p.eval H run).2.1, (p.eval H run).2.2) := by
  induction p with
  | ret a => rfl
  | hash q k ih => simp only [RP.bind, RP.eval]; rw [ih]
  | rerun r k ih => simp only [RP.bind, RP.eval]; rw [ih]

end Claims
