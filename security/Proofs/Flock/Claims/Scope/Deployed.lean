module

public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Claims.Scope.Tables
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope

/-!
# The deployed relation at the deployed parser

The deployed record's parser is `FlockDeploy.parse` (`parseOf` at the verifier's reading of the bytes,
`FlockDeploy.readOf`), and `ScopeMatches` at both deployed records is `FlockDeploy.scopeMatches_zk` and
`FlockDeploy.scopeMatches_plain` (`Matches`' `scopeMatches`). At the deployed relation (`relation`):

* `relS_deployed`: `RelS` of an in-scope statement the deployed parser accepts is every table's `Layout.model` satisfied
  (`Fit.relS_layout`; `InScope` gives the untyped form without public input ports it needs).
-/

@[expose] public section

namespace Flock.Readable

/-- **The deployed relation at the deployed parser**: a statement it parses is the readable form of a statement the
verifier parses, and when that statement is in scope, `RelS` is every table's `Layout.model` at `regionsOf` satisfied by
its share of the hidden values. -/
theorem relS_deployed {zk : Bool} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : FlockDeploy.parse zk relation x = some s) (hin : Claims.Flock.InScope zk s) (w : Claims.Flock.Wit) :
    ∃ p : Parsed, parseBytes (FlockDeploy.readOf zk) x = some p ∧ s = toReadable relation p ∧
      (Claims.Flock.RelS s w ↔ ∀ c (hc : c < p.sts.size), ∃ L : FlockSoundness.Layout p.sts[c],
        (L.model (FlockSoundness.Refine.regionsOf p.sts[c])).Satisfies (zOf p.sts.size w c p.sts[c].m)) := by
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
  obtain ⟨-, -, -, -, -, ht, -, -, -, hpi, -⟩ := hin
  simp only [toReadable, Read.form, Bool.not_eq_true] at ht hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  exact ⟨p, hp, rfl, relS_layout (parses_of_parse hrp) ht hpi' w⟩

end Flock.Readable
