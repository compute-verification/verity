module

public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
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

/-!
# `ZKOutLeak` says something

`ZKOutLeak` is a guarantee (`Claims.zkOutLeak`), held to two witnesses (the lock's `witnesses`), at the hash SHA-512
(`FlockDeploy.sha512`):

* Honest (`zkOutLeak_honest`): its antecedent holds where the outcome takes two values. The talking prover, with every
  witness in its relation and the verdict public (`verdictOut`), is zero knowledge per outcome at error `0`, at the
  simulator that echoes it whatever the outcome (`echoSimAt`, `echo_zkOut`).
* Nontrivial (`zkOutLeak_nontrivial`): its conclusion fails at the talking prover with the silent simulator, against the
  verifier that accepts everything (`talk_leaks`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

/-- **The verdict as the public outcome**: whether the transcript's last round is accepted. -/
def verdictOut (_ : Msg) (τ : List (Msg × Out)) : Bool := decide (τ.getLast?.map Prod.snd = some .acc)

/-- **The simulator that echoes the talking prover, given an outcome it ignores**: `echoSimOut` at the outcomes `St`. -/
def echoSimAt (St : Type) : SimOut St where
  C _ := Unit
  prog _ _ _ := .ask [[0]] fun b => .ret [([0], b)]

/-- **The talking prover is zero knowledge per outcome** at error `0`, at every witness, all in its relation, and every
public outcome: the echoing simulator's transcript is the talking prover's. -/
theorem echo_zkOut {St : Type} (H : Msg → Msg) (out : Msg → List (Msg × Out) → St) :
    ZKOut (fun _ => Unit) (fun _ => Unit) H (fun _ => 1) (fun _ _ => talkP) (fun _ (_ : Unit) => True) out
      (echoSimAt St) (fun _ _ _ _ _ _ => 0) := by
  intro x w _ Θ Vs θ hk o Ev
  change Pr (fun ω : Unit × Unit => out x (runStar H talkP Vs x ω θ).msgs = o ∧ Ev (runStar H talkP Vs x ω θ).msgs) ≤
    Pr (fun c : Unit => Ev (((echoSimAt St).given o).out H Vs θ x c)) + 0
  rw [add_zero]
  by_cases hE : Ev [([0], ((Vs.next x [[0]] θ).eval H).1)]
  · refine (Pr_le_one _).trans (le_of_eq ?_)
    exact ((Pr_congr fun c : Unit => iff_true_intro (show Ev (((echoSimAt St).given o).out H Vs θ x c) from hE)).trans
      Claims.ZKMal.Pr_true).symm
  · refine (le_of_eq ?_).trans (Pr_nonneg _)
    exact (Pr_congr fun ω : Unit × Unit => iff_false_intro fun h : out x (runStar H talkP Vs x ω θ).msgs = o ∧
      Ev (runStar H talkP Vs x ω θ).msgs => hE (talk_msgs H Vs x θ hk ω ▸ h.2)).trans Pr_false

/-- **`ZKOutLeak`'s antecedent holds where the outcome takes two values**: the talking prover is zero knowledge per
outcome with the verdict public, and the verdict tells an accepted round from a rejected one. -/
theorem zkOutLeak_honest :
    ZKOut (fun _ => Unit) (fun _ => Unit) FlockDeploy.sha512 (fun _ => 1) (fun _ _ => talkP) (fun _ (_ : Unit) => True)
      verdictOut (echoSimAt Bool) (fun _ _ _ _ _ _ => 0) ∧
      verdictOut [] [([0], .acc)] ≠ verdictOut [] [([0], .rej)] :=
  ⟨echo_zkOut _ _, by decide⟩

/-- **`ZKOutLeak`'s conclusion fails** at the talking prover with the silent simulator, against the verifier that
accepts everything, with one outcome and error `0`. -/
theorem zkOutLeak_nontrivial :
    ¬ Pr (fun ω : Unit × Unit => (runStar FlockDeploy.sha512 talkP acceptAllVs [] ω ()).msgs ≠ []) ≤
      (Fintype.card Unit : ℝ) * Pr (fun p : Unit × silentSimOut.C [] =>
        (silentSimOut.given p.1).out FlockDeploy.sha512 acceptAllVs () [] p.2 ≠ []) + ∑ _o : Unit, (0 : ℝ) :=
  talk_leaks _

end Claims.Witness.SamplingGeneric
