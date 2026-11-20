module

public import Proofs.Flock.Claims.DeployReads
public import Spot.OneStageZk
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
import Std.Tactic.BVDecide.Normalize.Bool
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-!
# A step of `acc_of_accepts`: a V\* part's base setup, compiled and model

A V\* part's verdict (`VStar.Recursive.partVerdict`) sets its session up with `Spot.OneStageZk.buildSessionOf`
at the statement `verity/flock-circuit`'s tags `tags0`, on one table's keyed coin tree, with `--zk`'s hooks at the
circuit's `mask_log`, and no lookup tables. The model's recursive setup (`RecDeploy.RecIns.setup`) starts with the same
calls at its statement's reading `r` (`Readable.Read`): the hm96-sha512 rows check, `Read.setup`, each table's
`checkRegistered` and `noIdentifier`, and `Registered.checkFull` at the reading's sample.

`baseSetup_of_build`: where `r` reads the part's inputs as `FlockDeploy.Raw.ins` reads a `--zk` statement (the tags
`tags0`, the coin tree, one table, no tables, no `--work`, `--stratified`, `--closure` or
`--public-inputs`), the compiled setup's success gives each of those model calls, with the same `(spec, sts)`:

* the tags agree: `r.tags` (`ZkExec.zkTags`) is the compiled `tags`, and `r.coins` (`ZkExec.zkCoins`) the compiled `coins`;
* the session agrees: under hidden outputs the compiled setup runs `Stmt.setupTablesOf` on `ProvedScope.circuitOf`'s
  parse, which is `Stmt.setupTables` (`Stmt.setupTables_eq`);
* the compiled `checkIdentifier` refuses every header naming an identifier, which is `noIdentifier`'s refusal.

The compiled `checkProgram`, `checkComplete`, `checkCover` and `KnownRead.check` have no counterpart in the model's
setup, so this needs none of them.
-/

@[expose] public section

open Lean (Json)
open Flock (V)

namespace Verity.Recursive

set_option maxHeartbeats 1000000 in
open FlockDeploy.BaseSetup in
/-- **A V\* part's base setup.** Let `tags0` be the part's statement tags, and `coins` and `tags` the coin tree
and tags `partVerdict` sets its session up on. Suppose the compiled `buildSessionOf` sets the part's circuit `circuitB`
and staged public file `pubB` up at the sample `draw`, with `partition`, `program` and the verifier's registered reads
`registered`, to `(spec, sts)`. Let `r` read the same inputs as `Raw.ins` reads a `--zk` statement. Then the model's
calls at the start of `RecIns.setup` all succeed at `r` with the same `(spec, sts)`: the hm96-sha512 rows check,
`Read.setup`, each table's `checkRegistered` at its reads derived by `Registered.withPositions` and `noIdentifier` (pointwise, and as `RecIns.setup`'s loop), and
`Registered.checkFull` at `r`'s sample. -/
theorem baseSetup_of_build {tags0 : Flock.Tags} {circuitB pubB : ByteArray} {draw partition : Option Json}
    {program : Option ByteArray} {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt}
    (hb : Spot.OneStageZk.buildSessionOf
      (Flock.ProvedScope.circuitOf
        (Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1)
        circuitB [])
      (Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1)
      circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) [] 1 draw partition program registered =
        .ok (spec, sts))
    (r : Flock.Readable.Read)
    (hI : r.I =
      { tags := tags0
        circuitFile := circuitB
        publicFile := pubB
        coins := .tree (Flock.Zk.coinSpec (tags0.tableNames 1))
        count := 1
        tables := []
        partition := partition
        program := program
        work := none
        stratified := none
        closure := none })
    (hzk : r.zk = true) (hdraw : r.draw = draw) (hpi : r.publicInputs = none) :
    r.tags.hm96Rows = true ∧ r.setup = .ok (spec, sts) ∧
      (∀ st ∈ sts, (Flock.Registered.withPositions st.c st.pub program registered >>= Flock.checkRegistered st) = .ok () ∧
        FlockDeploy.noIdentifier st = .ok ()) ∧
      (forIn sts PUnit.unit fun st _ => do
          Flock.checkRegistered st (← Flock.Registered.withPositions st.c st.pub program registered)
          FlockDeploy.noIdentifier st
          pure (ForInStep.yield PUnit.unit) : V PUnit) = .ok PUnit.unit ∧
      Flock.Registered.checkFull registered r.draw = .ok () := by
  have hcoins : r.coins = .tree (Flock.Zk.coinSpec (tags0.tableNames 1)) := by
    simp [Flock.Readable.Read.coins, hzk, FlockSoundness.Discharge.ZkExec.zkCoins, hI]
  have htags : r.tags =
      Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1 := by
    simp [Flock.Readable.Read.tags, hzk, FlockSoundness.Discharge.ZkExec.zkTags,
      FlockSoundness.Discharge.ZkExec.zkCoins, hI]
  have hsetup : r.setup = Flock.Stmt.setupTables
      (Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1)
      circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) 1 [] draw partition program := by
    simp only [Flock.Readable.Read.setup, htags, hcoins, hdraw, hpi, hI]
  unfold Spot.OneStageZk.buildSessionOf at hb
  have hh : (Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1).hm96Rows = true := by
    refine Classical.byContradiction fun hne => ?_
    simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hb
  simp only [hh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind] at hb
  have hstep : ∃ x : Flock.SessionSpec × Array Flock.Stmt,
      Flock.Stmt.setupTables
        (Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1)
        circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) 1 [] draw partition program = .ok x ∧
      (∀ st ∈ x.2, (Flock.Registered.withPositions st.c st.pub program registered >>= Flock.checkRegistered st) = .ok () ∧
        Spot.OneStageZk.checkIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull registered draw = .ok () ∧ (x.1, x.2) = (spec, sts) := by
    obtain ⟨x, hs, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, hloop, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, hfull, hb⟩ := bind_ok hb
    refine ⟨x, ?_, fun st hst => (body_bind_ok (W := fun st => Flock.Registered.withPositions st.c st.pub program registered)
      (C := Flock.checkRegistered) (B := Spot.OneStageZk.checkIdentifier)).1 (forIn_mem body_bind_yield hloop st hst), hfull,
      Except.ok.inj hb⟩
    split at hs
    · next hhid =>
      rw [Flock.Stmt.setupTables_eq]
      simpa only [Flock.ProvedScope.circuitOf, hhid, ↓reduceIte] using hs
    · exact hs
  obtain ⟨⟨spec', sts'⟩, hs, hpt, hfull, hx⟩ := hstep
  simp only [Prod.mk.injEq] at hx
  obtain ⟨rfl, rfl⟩ := hx
  have hpt' : ∀ st ∈ sts', (Flock.Registered.withPositions st.c st.pub program registered >>= Flock.checkRegistered st) =
      .ok () ∧ FlockDeploy.noIdentifier st = .ok () :=
    fun st hst => ⟨(hpt st hst).1, noIdentifier_of_check (hpt st hst).2⟩
  refine ⟨by rw [htags]; exact hh, hsetup.trans hs, hpt', forIn_ok fun st hst =>
    (body_bind_ok (W := fun st => Flock.Registered.withPositions st.c st.pub program registered)
      (C := Flock.checkRegistered) (B := FlockDeploy.noIdentifier)).2 (hpt' st hst), ?_⟩
  rw [hdraw]
  exact hfull

end Verity.Recursive
