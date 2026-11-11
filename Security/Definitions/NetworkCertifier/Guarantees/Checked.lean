module

public import Definitions.NetworkCertifier.Checked
public import Definitions.NetworkCertifier.Guarantees.SampledAudit
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

/-!
# The leak bound with checked inputs

`SampledLeakAccepted` with each statement's checked inputs `chk` left out of the count: the free inputs a check on the
whole run pins (`Definitions.NetworkCertifier.Checked`). Each statement's check pins its checked inputs
(`CheckedUnique`), and the trace's run is valid except with probability `εchk` (`CheckedWitness`), so a run whose
observations the bound counts is a valid run, whose checked inputs the rest of the run fixes: they cost no advice bits.
The incorrect proof blocks' committed bits are charged at the most any possible incorrect set commits, rather than at a
bound the statement takes.

A monitoring period's bound with checked inputs is `PooledLeakCarried` (`Guarantees.Carried`), over runs that carry the
pool; at the empty pool it bounds windows that each hold a pool of their own.
-/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

section Sampled

open Claims.Sampling
open Claims.SamplingGeneric (Arg Compiled)

/-- **An accepted sampled audit bounds the leak with the checked inputs left out.** At every compiled sampled verifier
`D` where `Claims.SamplingGeneric.Compiled.KS` holds, a public layout `lay` and horizon `k` of each statement, `δ ≥ 0`,
registered inputs `reg`, checked inputs `chk` and each statement's check `valid`:

* on the statements in scope whose trace run the check holds on except with probability `εchk` (`CheckedWitness`),
  `Pr[acc ∧ ¬RLeakChecked δ] ≤ δ + ε_ks + ε_bind + εchk`: except with that probability, every observation sequence of
  length `k` with jitter below `Dj` that the trace's transfer gates carry with every status complete is among the
  sequences `AccSeqsChecked` counts at the registration's values;
* on a statement in scope whose check pins `chk` (`CheckedUnique`), at every `ρ`, those sequences number at most
  `2^#(freeInputs \ (reg ∪ chk)) · #faultySets(δ) · 2^β · Dj^k`, `β` the most committed bits of a possible incorrect
  set. -/
def SampledLeakChecked : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (εchk : Claims.Msg → ℝ)
    (Dj : ℕ),
    KSLinOn H D.V (fun x => ScopeAcc D lay k x ∧ CheckedWitness H D valid εchk x)
        (RLeakChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) valid
          Dj)
        D.E (fun x => D.κ δ x + εchk x) D.γ (fun _ => 1) (Bext H D.V D.E) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        CheckedUnique s.prog P s.design (lay s) (k s) (chk s) (valid s) →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
          (AccSeqsChecked s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) δ op
              Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ (reg s ∪ chk s)).card * (faultySets s.design P δ).card *
              2 ^ ((faultySets s.design P δ).sup (faultyBits s.prog P)) * Dj ^ k s : ℕ) : ℕ∞)

end Sampled

end NetTiming.Guarantees
