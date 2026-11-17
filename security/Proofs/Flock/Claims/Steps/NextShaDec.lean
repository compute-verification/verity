module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
import Proofs.Flock.Claims.Complete.Json.Decode
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.DeployK
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Steps.NextCs
public import Proofs.Flock.Claims.Steps.NextJ
public import Proofs.Flock.Claims.Steps.NextP
public import Proofs.Flock.Claims.Steps.NextSha
import Flock.With.SimVerify
import all Init.Data.Array.Basic
import all Lean.Data.Json.Basic
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

/-! # What the plain KS proofs read of `Claims.Recursion.NextShaDec`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

open FlockComplete.JsonRT (f128s_map array_mapM_ok_of dk_cases)

theorem bind_congr_ok {α β : Type} {x : Flock.V α} {f g : α → Flock.V β} (h : ∀ a, x = .ok a → f a = g a) :
    x >>= f = x >>= g := by
  cases x with
  | error e => rfl
  | ok a => exact h a rfl

theorem sample_ne_proofKey (spec : SessionSpec) : "unit_draw" ≠ s!"proof_{spec.dk}" := by
  rcases dk_cases spec with e | e <;> simp [e, toString]

theorem liftV_run_eval {α : Type} (H : Claims.Msg → Claims.Msg) (v : Flock.V α) :
    (((monadLift v : Claims.OCV α).run).eval H).1 = v := by
  first
  | rfl
  | (cases v <;> rfl)

theorem evRun_bind {α β : Type} (H : Claims.Msg → Claims.Msg) (x : Claims.OCV α) (f : α → Claims.OCV β) :
    (((x >>= f).run).eval H).1 = (((x.run).eval H).1 >>= fun a => (((f a).run).eval H).1) := by
  simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  first
  | (cases hx : ((x.run).eval H).1 <;> rfl)
  | (split <;> rfl)

/-- **The verifier at a parsed value over `OC`, against `H`, is `verifyJWithM` at `hOf H`.** -/
theorem verifyJOC_eval (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (tabs : Array Flock.Setup) (j : Json)
    (proofs : Array ByteArray) :
    ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval H).1 =
      verifyJWithM (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec tabs j proofs := by
  rw [← (Flock.With.verifyJWithM_sim (Claims.ocLog H) (Claims.ocLog_hashes H) spec tabs j proofs #[]).1]
  exact Flock.With.verifyJWithM_sim (Flock.With.Sim.forget _) (Flock.With.Sim.forget_logged _) spec tabs j proofs #[]

theorem sampleOfJ_shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) :
    sampleOfJ (shaSubJ spec ds j) = sampleOfJ j := by
  unfold sampleOfJ shaSubJ
  rw [getObjVal?_mapKey_ne _ _ (sample_ne_proofKey spec)]

/-- The checks after the verdict read the statement and the sample, which `shaSubJ` keeps, and no proof file. -/
theorem checksAtJ_shaSubJ (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (stmt : Claims.Msg) (ds : Array ByteArray) (j : Json)
    (bs : Array ByteArray) (j' : Json) :
    checksAtJ s spec x link c₁ mids ⟨stmt, shaSubJ spec ds j, bs⟩ j' =
      checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩ j' := by
  first
  | (simp only [checksAtJ, sampleOfJ_shaSubJ]; done)
  | (unfold checksAtJ; rw [sampleOfJ_shaSubJ])
  | (unfold checksAtJ; dsimp only; rw [sampleOfJ_shaSubJ])

/-- **`decisionAtS` run against `H`**: the plain verdict at `hOf H` on the proofs' files at the substituted value,
then the checks after it. -/
theorem decisionAtS_evalJ (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof} :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 =
      (verifyJWithM (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec (tabsOf sts)
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j))
          (filesAt (tabsOf sts) (pointsOf spec c₁) ps) >>= fun _ =>
        checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j))) := by
  dsimp only [decisionAtS]
  rw [evOk_bind_fst (evOk_mapM_array (evOk_dg H spec) (filesAt (tabsOf sts) (pointsOf spec c₁) ps))]
  dsimp only [decisionAtP, decisionAtJ]
  rw [evRun_bind, verifyJOC_eval]
  refine bind_congr_ok fun _ _ => ?_
  exact (liftV_run_eval H _).trans (checksAtJ_shaSubJ s spec x link c₁ mids stmt _ j _ _)

end FlockDeploy
