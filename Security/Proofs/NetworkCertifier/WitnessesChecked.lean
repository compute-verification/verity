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
# The nontrivial lemmas of the leak bounds with checked inputs

* `SampledLeakChecked`. Its acceptance conclusion fails at the accepting verifier `leakD`, as `SampledLeakAccepted`'s does,
  at the check every run passes, where the check misses with probability `0` (`checkedWitness_true`): a witness in
  `RLeakChecked` is in `RLeak` (`rLeak_of_rLeakChecked`), and none is (`leakD_noRLeak`).
* `PooledLeakChecked`. At one window whose audit always accepts, `q = 1`, `Fmax = 1` and no error, its observations empty
  where the horizon is one step, no window meets its leak relation, and the escape bound reads `1 ≤ 0`.

Their honest lemmas are open: each needs a deployed check that pins its checked inputs (`CheckedUnique`) and its miss
bounded (`CheckedWitness`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

/-- **A check every run passes misses with probability `0`**, at every deployment and statement. -/
theorem checkedWitness_true {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) :
    CheckedWitness H D decode (fun _ _ => True) (fun _ => 0) x := by
  intro Ω _ _ P
  unfold PrCheckMiss
  refine le_of_eq ((Pr_congr fun r => iff_false_intro ?_).trans Pr_false)
  simp

/-- **`SampledLeakChecked`'s acceptance conclusion fails at the accepting verifier** `leakD`, at one step at `δ = 0`,
with no input registered or checked, at the check every run passes: at the statement `[]`, in scope (`leakD_scopeAcc`)
with the check's miss `0` (`checkedWitness_true`), its silent prover's bound fails, since no witness is in
`RLeakChecked` (`rLeak_of_rLeakChecked`, `leakD_noRLeak`). -/
theorem SampledLeakChecked_nontrivial :
    ¬ KSLinOn FlockDeploy.sha512 (leakD Bool).V
      (fun x => ScopeAcc (leakD Bool) (fun _ => notLay) (fun _ => 1) x ∧
        CheckedWitness FlockDeploy.sha512 (leakD Bool) (fun _ => [()]) (fun _ _ => True) (fun _ => 0) x)
      (RLeakChecked FlockDeploy.sha512 (leakD Bool).commit (leakD Bool).parse (leakD Bool).partition 1 1 1
        (fun _ => notLay) (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) (fun _ _ => True) 1)
      (Claims.Sampling.E (leakD Bool) fun _ => [()]) (fun x => Claims.Sampling.κ (leakD Bool) 0 x + 0)
      (Claims.Sampling.γ (leakD Bool)) (fun _ => 1)
      (Bext FlockDeploy.sha512 (leakD Bool).V (Claims.Sampling.E (leakD Bool) fun _ => [()])) := by
  intro h
  have hb := h [] ⟨leakD_scopeAcc [], checkedWitness_true _ _ _ []⟩ Unit silentP
  exact leakD_bound_fails FlockDeploy.sha512 _ _ (Claims.Sampling.κ (leakD Bool) 0 [] + 0) []
    (fun w hw => leakD_noRLeak FlockDeploy.sha512 _ [] w (rLeak_of_rLeakChecked hw))
    (by linarith [leakD_κ (Coin := Bool) []]) hb

/-- **`PooledLeakChecked`'s escape bound fails at an audit that always accepts**: one window, `q = 1`, `Fmax = 1`, no
error, and the window's observations empty where its horizon is one step, so its good event fails at every sample. -/
theorem PooledLeakChecked_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬PooledGoodChecked (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) 1
          (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ => [false]) 1 (fun _ => ∅)
          (fun _ => [])) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  have hng : ¬PooledGoodChecked (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) 1
      (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ => [false]) 1 (fun _ => ∅)
      (fun _ => []) := fun h => absurd (h.2 0).2.2.1 (by simp)
  rw [Pr_congr fun _ => iff_true_intro ⟨fun _ => trivial, hng⟩]
  simp [Pr]

end NetTiming.Witness
