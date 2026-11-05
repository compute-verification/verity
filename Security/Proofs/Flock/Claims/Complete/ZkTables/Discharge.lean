module

public import Proofs.Flock.Claims.Complete.ZkTables.Skeleton
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
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-!
# The masked skeleton's goals discharged, at several tables

`Zk/Discharge.lean`'s goals at every honest masked run of any number of tables, from the scope (`Case.tables`):

* `tables_goal` (`TablesZk`, §3): one statement per table.
* `schedules_goal` (`SchedulesZk`): `Zk.schedule` has a schedule at every table's `m` (`zkSchedule_ok`).
* `proofCount_goal` (`ProofCountZk`): the masked session returns one proof file per rep stream (`sessionPZk_proofs`).
-/

@[expose] public section

open Flock FlockDeploy

namespace FlockComplete.ZkTables

/-- **One statement per table** (§3): the goal `TablesZk`. -/
theorem tables_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : TablesZk H rel := by
  intro o
  obtain ⟨hpos, -, hn, -⟩ := o.tables
  unfold Run.tabs tabsOf
  simp only [Array.size_map, hn, bne_self_eq_false, Bool.false_or, beq_eq_false_iff_ne]
  omega

/-- **Each table has its `--zk` schedule**: the goal `SchedulesZk`. -/
theorem schedules_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SchedulesZk H rel := by
  intro o st hst
  obtain ⟨-, -, -, hall⟩ := o.tables
  unfold Run.tabs tabsOf at hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  simp only [Array.size_map] at hj
  obtain ⟨h1, h2, -⟩ := hall j hj
  simp only [Array.getElem_map, Setup.ofCircuit]
  exact zkSchedule_ok h1 h2

/-- **One proof file per rep stream**: the goal `ProofCountZk`. -/
theorem proofCount_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    ProofCountZk H rel := by
  intro o
  have := sessionPZk_proofs (Claims.hOf H) (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (MaskRank.masksOf o.sts) (traces o.sts o.w) o.ω #[] _ _ o.ran
  simp [this]

end FlockComplete.ZkTables
