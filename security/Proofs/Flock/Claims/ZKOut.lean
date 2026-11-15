module

public import Definitions.Claims.ZKOut
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# Zero knowledge with a public outcome: the leak form

Zero knowledge per outcome gives the leak form (`zkOutLeak`): an event's probability is the sum over the outcomes of the
outcome's and the event's (`pr_eq_sum_fiber`), each bounded by its simulator's, and the sum of the simulators' is the
number of outcomes times the simulator's that samples the outcome uniformly (`pr_prod_eq_sum`).

`ZKOut` is not true of everything (`ZKOut_fails`): the quiet prover against the verifier that accepts its first
message, with a simulator that outputs the empty transcript, fails it at every error below `1`.
-/

@[expose] public section

namespace Claims

open Classical in
theorem pr_eq_sum_fiber {α St : Type} [Fintype α] [Fintype St] (f : α → St) (p : α → Prop) :
    Pr p = ∑ o, Pr (fun a => f a = o ∧ p a) := by
  have h : (Finset.univ.filter p).card = ∑ o, (Finset.univ.filter fun a => f a = o ∧ p a).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := f) (t := Finset.univ) fun _ _ => Finset.mem_univ _]
    refine Finset.sum_congr rfl fun o _ => ?_
    congr 1
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact and_comm
  unfold Claims.Pr Claims.prUniform
  rw [h, Nat.cast_sum, Finset.sum_div]
  exact Finset.sum_congr rfl fun o _ => by congr

open Classical in
theorem pr_prod_eq_sum {St C : Type} [Fintype St] [Nonempty St] [Fintype C] (q : St → C → Prop) :
    (Fintype.card St : ℝ) * Pr (fun p : St × C => q p.1 p.2) = ∑ o, Pr (fun c => q o c) := by
  have h : (Finset.univ.filter fun p : St × C => q p.1 p.2).card =
      ∑ o, (Finset.univ.filter fun c => q o c).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    exact Finset.sum_congr rfl fun o _ => (Finset.card_filter _ _).symm
  have ha : (Fintype.card St : ℝ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  unfold Claims.Pr Claims.prUniform
  rw [h, Fintype.card_prod, Nat.cast_sum, Nat.cast_mul, ← Finset.sum_div, ← mul_div_assoc,
    mul_div_mul_left _ _ ha]

theorem zkOutLeak : ZKOutLeak := by
  intro W St _ _ N Ω _ _ _ _ H k P R out S ε h x w hR Θ Vs θ hk Ev
  rw [pr_eq_sum_fiber (fun ω => out x (runStar H (P w x) Vs x ω θ).msgs),
    pr_prod_eq_sum (fun o c => Ev ((S.given o).out H Vs θ x c)), ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun o _ => h x w hR Θ Vs θ hk o Ev

/-- **`ZKOut` fails**: at every `ε < 1`, with one outcome, the quiet prover and a simulator that outputs the empty
transcript fail it, against the verifier that accepts the first message. -/
theorem ZKOut_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (ε : Msg → ℝ)
    (x : Msg) (hε : ε x < 1) :
    ∃ (P : Unit → (x : Msg) → Prover (N × Ω)) (S : SimOut Unit),
      ¬ Claims.ZKOut (fun _ => N) (fun _ => Ω) H (fun _ => 1) P (fun _ _ => True) (fun _ _ => ()) S
        (fun x _ _ _ _ _ => ε x) := by
  refine ⟨fun _ _ => ZKMal.quietP N Ω [], { C := fun _ => Unit, prog := fun _ _ _ => .ret [] }, fun h => ?_⟩
  have a : Pr (fun ω : N × Ω => () = () ∧
        (runStar H (ZKMal.quietP N Ω []) (ZKMal.accNow fun _ => 1) x ω ()).msgs = [([], .acc)]) ≤
      Pr (fun _ : Unit => ([] : List (Msg × Out)) = [([], .acc)]) + ε x :=
    h x () trivial Unit (ZKMal.accNow fun _ => 1) () rfl () fun τ => τ = [([], .acc)]
  rw [Pr_congr (q := fun _ => True) fun ω =>
      iff_true_intro (⟨rfl, ZKMal.runStar_quietP_accNow H [] (fun _ => 1) x one_pos ω⟩ : () = () ∧ _),
    ZKMal.Pr_true, Pr_congr (q := fun _ => False) fun _ => iff_false_intro (List.cons_ne_nil _ _).symm,
    Pr_false] at a
  linarith

end Claims
