module

public import Definitions.NetworkCertifier.Checked
public import Proofs.NetworkCertifier.Witnesses
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
import Proofs.NetworkCertifier.Checked

/-!
# The nontrivial lemma of the leak bound with checked inputs

`SampledLeakChecked`'s acceptance conclusion fails at the accepting compiled verifier `leakL`, as `SampledLeakAccepted`'s
does, at the check every run passes, where the check misses with probability `0` on any runs (`checkedWitness_true`),
the trace's among them: a witness in `RLeakChecked` is in `RLeak` (`rLeak_of_rLeakChecked`), and none is
(`leakL_noRLeak`).

Its honest lemma is `SampledAuditAccepted_honest`, at its one stated premise `Compiled.KS`. `CheckedUnique` and
`CheckedWitness` are open premises, discharged by a deployed check that pins its checked inputs and bounds its miss.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (leakL)

/-- **A check every run passes misses with probability `0`**, at every compiled sampled verifier and statement, on any
runs. -/
theorem checkedWitness_true {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Msg → Msg) (D : Compiled Coin A)
    (runs : Claims.Sampling.AuditStatement → Claims.Sampling.Partition → Claims.Sampling.Wit → Set (ℕ → Bool))
    (x : Msg) : CheckedWitness H D runs (fun _ _ => True) (fun _ => 0) x := by
  intro Ω _ _ P
  unfold PrCheckMiss
  refine le_of_eq ((Pr_congr fun r => iff_false_intro ?_).trans Pr_false)
  simp

/-- **`SampledLeakChecked`'s acceptance conclusion fails at the accepting verifier** `leakL`, at one step at `δ = 0`,
with no input registered or checked, at the check every run passes: at the statement `[]`, in scope
(`leakL_scopeAcc`) with the check's miss `0` (`checkedWitness_true`), its silent prover's bound fails, since no witness
is in `RLeakChecked` (`rLeak_of_rLeakChecked`, `leakL_noRLeak`). -/
theorem SampledLeakChecked_nontrivial :
    ¬ KSLinOn FlockDeploy.sha512 (leakL Bool).V
      (fun x => ScopeAcc (leakL Bool) (fun _ => notLay) (fun _ => 1) x ∧
        CheckedWitness FlockDeploy.sha512 (leakL Bool) (traceRun FlockDeploy.sha512 (leakL Bool).commit)
          (fun _ _ => True) (fun _ => 0) x)
      (RLeakChecked FlockDeploy.sha512 (leakL Bool).commit (leakL Bool).parse (leakL Bool).partition 1 1 1
        (fun _ => notLay) (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) (fun _ _ => True) 1)
      (leakL Bool).E (fun x => (leakL Bool).κ 0 x + 0) (leakL Bool).γ (fun _ => 1)
      (Bext FlockDeploy.sha512 (leakL Bool).V (leakL Bool).E) := by
  intro h
  have hb := h [] ⟨leakL_scopeAcc [], checkedWitness_true _ _ _ []⟩ Unit silentP
  refine leakL_bound_fails FlockDeploy.sha512 _ ((leakL Bool).κ 0 [] + 0) [] ?_
    (by linarith [leakL_κ (Coin := Bool) []]) hb
  exact fun w hw => leakL_noRLeak FlockDeploy.sha512 _ [] w (rLeak_of_rLeakChecked hw)

end NetTiming.Witness
