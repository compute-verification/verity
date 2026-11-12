module

public import Proofs.Flock.Claims.KS.Land2TableAtProbe
public import Definitions.Claims.Land2OneTableProbe
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A reader's probe: the table statement at a masked statement of one table

Not for `main`: the two other probes put together at the masked verifier. At a statement of one table, the probability
`Flock.KSAt` bounds at table `0` is the one `Flock.KS` bounds (`prBad_zero_eq_zk`), and the table statement the tree
proves there (`zk_KSAt_untrim`) gives `Flock.KS`'s inequality with the session's own `κ` and `γ` (`ks_one_table_zk`).
-/

@[expose] public section

namespace FlockDeploy.KS.Land2Probe

open Flock

/-- At a statement of one table, table `0`'s error and word are the session's. -/
theorem one_table_numbers {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) :
    tableErrAt true x 0 = εtable true x ∧ (wordLenJ true x 0 : ℝ) = N₀ true x := by
  have h0 : (tables true x)[0]? = some (spec, st) := by simp [h]
  refine ⟨?_, ?_⟩
  · rw [tableErrAt_eq h0]
    unfold εtable
    rw [h]
    simp
  · rw [wordLenJ_cast h0]
    unfold N₀
    rw [h]
    simp

/-- A statement that parses to one table and sets up has one table in the records. -/
theorem tables_one {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true rel x = some s) (h1 : s.tables = 1) (hx : (setupOf true x).isSome) :
    ∃ spec st, tables true x = [(spec, st)] := by
  have hl : (tables true x).length = 1 := (tables_length hs hx).trans h1
  have he : ∃ p, tables true x = [p] := by
    first
      | exact List.length_eq_one_iff.1 hl
      | exact List.length_eq_one.1 hl
  obtain ⟨⟨a, b⟩, hp⟩ := he
  exact ⟨a, b, hp⟩

/-- **The probability the table statement bounds at table `0` of a masked statement of one table is the one the
session's statement bounds.** -/
theorem prBad_zero_eq_zk (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Claims.Flock.Stmt}
    {spec : SessionSpec} {st : Stmt} (hs : parse true rel x = some s) (h1 : s.tables = 1)
    (h : tables true x = [(spec, st)]) (decode : _) {Ω : Type} [Fintype Ω] (P : Claims.Prover Ω) :
    Claims.PrBad sha512 (deployZk rel).V (Claims.Flock.RAt (deployZk rel).table 0)
        (Claims.Flock.EAt (deployZk rel) (wordLenJ true) decode 0) x P =
      Claims.PrBad sha512 (deployZk rel).V (Claims.Flock.R (deployZk rel).parse)
        (Claims.Flock.E (deployZk rel) decode) x P := by
  have ht : (deployZk rel).table x 0 = some s := table_zero_of_tables_eq_one hs h1
  have hp : (deployZk rel).parse x = some s := hs
  have hN : (wordLenJ true x 0 : ℝ) = (deployZk rel).N₀ x := (one_table_numbers h).2
  exact Claims.Flock.Land2Probe.prBad_zero_eq sha512 (deployZk rel) (wordLenJ true) decode x hp h1 ht hN P

/-- **At a masked statement of one table the table statement gives the session's**: `Flock.KS`'s inequality there, with
the session's `κ` and `γ`, from `zk_KSAt_untrim`. -/
theorem ks_one_table_zk {x : Claims.Msg} {s : Claims.Flock.Stmt} {spec : SessionSpec} {st : Stmt}
    (hs : parse true Readable.relation x = some s) (h1 : s.tables = 1) (h : tables true x = [(spec, st)])
    (decode : _)
    (hlw : Claims.Flock.ListWithin (deployZk Readable.relation).len (deployZk Readable.relation).enc
      (deployZk Readable.relation).θ decode)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) :
    Claims.PrBad sha512 (deployZk Readable.relation).V (Claims.Flock.R (deployZk Readable.relation).parse)
        (Claims.Flock.E (deployZk Readable.relation) decode) x P ≤
      Claims.Flock.κ (deployZk Readable.relation) x +
        Real.sqrt (Claims.Flock.γ (deployZk Readable.relation) x *
          Claims.PrB sha512 (deployZk Readable.relation).V P x) := by
  have ht : (deployZk Readable.relation).table x 0 = some s := table_zero_of_tables_eq_one hs h1
  have hp : (deployZk Readable.relation).parse x = some s := hs
  have hN : (wordLenJ true x 0 : ℝ) = (deployZk Readable.relation).N₀ x := (one_table_numbers h).2
  have hε : tableErrAt true x 0 = (deployZk Readable.relation).εtable x := (one_table_numbers h).1
  exact Claims.Flock.Land2Probe.ks_at_one_table sha512 (deployZk Readable.relation) (wordLenJ true)
    (tableErrAt true) zk_KSAt_untrim decode hlw x hp h1 ht hN hε P

end FlockDeploy.KS.Land2Probe
