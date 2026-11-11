module

public import Definitions.NetworkCertifier.Carried
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
import Proofs.NetworkCertifier.Carried
import Proofs.NetworkCertifier.WitnessesChecked

/-!
# The witnesses of the leak bounds over a carried pool

`SampledLeakCarried`'s honest lemma (`SampledLeakCarried_honest`) is its premises at the toy code: `Compiled.KS`, as
`SampledLeakChecked`'s (`SampledAuditAccepted_honest`), and opening gates that are inputs of each circuit, at a run that
opens its row at every one of its input gates. `CheckedUnique` and `CheckedWitness` are open premises, as they are
`SampledLeakChecked`'s.

Their nontrivial lemmas:

* `SampledLeakCarried`. Its acceptance conclusion fails at the accepting compiled verifier `leakL`, as
  `SampledLeakChecked`'s does, at the check every run passes and no opening or closing gate: a witness in `RLeakCarried`
  is in `RLeak` (`rLeak_of_rLeakCarried`), and none is (`leakL_noRLeak`).
* `PooledLeakCarried`, at its one pattern of no restarts. At one run whose audit always accepts, `q = 1`, `Fmax = 1`, no
  error, every row the empty pool's, and its observations empty where the horizon is one step: no run meets its leak
  relation, no opening collides (the two digests of the empty row hash one message, `poolColl_empty`), and the escape
  bound reads `1 ≤ 0`.

`PooledLeakCarried`'s honest lemma is open: it needs a deployed check that pins its closing row (`CheckedClosing`) and the
roots it opens and leaves (`CarriedRoots`), and its period's inputs (`PeriodChannel`, `PeriodAdaptive`, `PeriodEscapes`,
`PeriodLeaksCarried`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling
open Claims.Witness.SamplingGeneric (notStmt notPart leakL)

set_option synthInstance.maxSize 1024 in
/-- **`SampledLeakCarried`'s premises at the toy code**: `Compiled.KS` (`SampledAuditAccepted_honest`), and a run that
opens its row at every input gate of its circuit opens it at inputs (`OpensAtInputs`). -/
theorem SampledLeakCarried_honest :
    (SamplingDeploy.compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) SamplingDeploy.toyCode
        (fun _ _ _ => [])).KS FlockDeploy.sha512 ∧
      OpensAtInputs fun s => (List.range s.prog.gates.length).filter fun g => s.prog.gates[g]? = some .input :=
  ⟨SampledAuditAccepted_honest, fun _ _ hg => by simpa using (List.mem_filter.1 hg).2⟩

open Claims.Witness Claims.Witness.Sampling in
/-- **`SampledLeakCarried`'s acceptance conclusion fails at the accepting verifier** `leakL`, at one step at `δ = 0`,
with no input registered and no opening or closing gate, at the check every run passes: at the statement `[]`, in scope
(`leakL_scopeAcc`) with the check's miss `0` (`checkedWitness_true`), its silent prover's bound fails, since no witness
is in `RLeakCarried` (`rLeak_of_rLeakCarried`, `leakL_noRLeak`). -/
theorem SampledLeakCarried_nontrivial :
    ¬ KSLinOn FlockDeploy.sha512 (leakL Bool).V
      (fun x => ScopeAcc (leakL Bool) (fun _ => notLay) (fun _ => 1) x ∧
        CheckedWitness FlockDeploy.sha512 (leakL Bool) (fun _ _ => True) (fun _ => 0) x)
      (RLeakCarried FlockDeploy.sha512 (leakL Bool).commit (leakL Bool).parse (leakL Bool).partition 1 1 1
        (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅) (fun _ _ => True) (fun _ => []) (fun _ => []) 1)
      (leakL Bool).E (fun x => (leakL Bool).κ 0 x + 0) (leakL Bool).γ (fun _ => 1)
      (Bext FlockDeploy.sha512 (leakL Bool).V (leakL Bool).E) := by
  intro h
  have hb := h [] ⟨leakL_scopeAcc [], checkedWitness_true _ _ []⟩ Unit silentP
  refine leakL_bound_fails FlockDeploy.sha512 _ ((leakL Bool).κ 0 [] + 0) [] ?_
    (by linarith [leakL_κ (Coin := Bool) []]) hb
  exact fun w hw => leakL_noRLeak FlockDeploy.sha512 _ [] w (rLeak_of_rLeakCarried hw)

/-- **`PooledLeakCarried`'s escape bound fails at an audit that always accepts**, at its one pattern of no restarts: one
run, `q = 1`, `Fmax = 1`, no error, every row the empty pool's, and the run's observations empty where its horizon is one
step, so its good event fails at every sample and no opening collides. -/
theorem PooledLeakCarried_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬PooledGoodCarried (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) (fun _ => [])
          (fun _ => []) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false)
          (fun _ _ _ _ => [false]) (fun _ => false) (fun _ => []) 1 (fun _ => ∅) (fun _ => [])) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) +
        ∑ i : Fin 1, Pr (fun _ : Fin 1 → Unit =>
          PoolColl (fun _ _ => 0) (fun _ => false) (fun _ => []) (fun _ => []) (fun _ => []) i) := by
  have hp : (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
      ¬PooledGoodCarried (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) (fun _ _ => True) (fun _ => [])
        (fun _ => []) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart) (fun _ _ => false)
        (fun _ _ _ _ => [false]) (fun _ => false) (fun _ => []) 1 (fun _ => ∅) (fun _ => [])) = fun _ => True :=
    funext fun _ => propext (iff_true_intro ⟨fun _ => trivial, fun h => by
      obtain ⟨-, rin, rout, -, hl⟩ := h
      exact absurd (hl 0).2.2.1 (by simp)⟩)
  have hc : ∀ i : Fin 1, (fun _ : Fin 1 → Unit =>
      PoolColl (fun _ _ => 0) (fun _ => false) (fun _ => []) (fun _ => []) (fun _ => []) i) = fun _ => False :=
    fun i => funext fun _ => propext (iff_false_intro (poolColl_empty _ _ i))
  rw [hp]
  simp only [hc]
  simp [Pr, Claims.prUniform]

end NetTiming.Witness
