module

public import Proofs.Flock.Claims.Recursion.RestVerdictBases
import Batteries.Tactic.OpenPrivate
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
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.Recursion.RestVerdictCustody
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Claims.Recursion.RestVerdictCustody
public import Proofs.Flock.Claims.Complete.ZkTables.Stmt
public import Proofs.Flock.Claims.JsonNorm

/-! `hrest`'s base checks at every reader (`basesHold_spec`), with no JSON premise, adversarial files included. -/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock
open Lean (Json)
open FlockDeploy (setupOf sampleEq)

open private setupOne_stmt unitSample_of from Proofs.Flock.Claims.Complete.ZkTables.StmtOne

theorem unitSample_of_setup {x : Claims.Msg} {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array Flock.Stmt} (hx : setupOf false x = some (s, spec, sts))
    (hq : s.read.check = .ok ()) {d : Json} (hdr : s.read.draw = some d) :
    spec.unitSample = some d ∧ d ≠ Json.null := by
  have hs := (FlockDeploy.setupOf_some hx).2
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_check hq
  have h := FlockDeploy.Ins.setup_read hs
  unfold Readable.Read.setup at h
  rw [hpi] at h
  have hsz := (setupTables_size h).1
  rcases Nat.lt_or_ge 1 sts.size with hgt | hle
  · obtain ⟨hud, -, hd, -⟩ := setupTables_spec h hgt
    obtain ⟨D, hD, -⟩ := hd d hdr
    exact ⟨hud.trans hdr, ofJson_obj hD⟩
  · have h1 : sts.size = 1 := by omega
    rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny h with ⟨-, st, hst, hspec, hsts⟩ | ⟨hc, hsz', -⟩
    · obtain ⟨⟨c, pop, hl, hd⟩, hinst, -, -⟩ := setupOne_stmt hst
      have hu := unitSample_of hl (hd d hdr)
      obtain ⟨u, hu', -⟩ := hinst d hu
      refine ⟨?_, ofJson_obj hu'⟩
      rw [hspec]
      simp [Flock.Stmt.spec, hu, Except.toOption]
    · omega

end FlockComplete.ZkTables

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP VerdictR holdsR decR DecR actR checksAtJ sampleOfJ sampleEq setupOf)
open Claims.Recursion.VStarB

noncomputable section

/-- `SampleTied` at the statements with a sample; at a statement without one, a one-table header may carry any
`unit_draw`. -/
def SampleTiedS : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    s.read.check = .ok () → s.read.draw.isSome = true → sampleEq (sampleOfU spec.unitSample) s.read.draw = true

theorem sampleTiedS : SampleTiedS := by
  intro x s spec sts hx hq hn
  obtain ⟨d, hdr⟩ := Option.isSome_iff_exists.1 hn
  obtain ⟨hU, hnull⟩ := FlockComplete.ZkTables.unitSample_of_setup hx hq hdr
  have hnn := (FlockDeploy.insOf_sample_norm (FlockDeploy.setupOf_some hx).1 hdr).1
  have e : sampleOfU (some d) = some d := by
    cases d with
    | null => exact absurd rfl hnull
    | _ => exact congrArg some hnn
  rw [hU, hdr, e]
  exact FlockDeploy.sampleEq_refl _

theorem holdsR_afterS (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = cs.length)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok () ∧ s.read.draw.isSome = true)
    (hud : SampleTiedS) (hcu : CustodyRecJ rd) :
    holdsR FlockDeploy.sha512 rd .after x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    obtain ⟨hq, hn⟩ := hck _ _ _ hx
    obtain ⟨lv, hL, hC⟩ := hcu x as cs _ hlen hd hq
    have hD : sampleEq (sampleOfJ f.j) s.read.draw = true := by
      obtain ⟨v, c₁', r₀, rss, fl, link', pf, -, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
      obtain ⟨-, hspec, -, -, -, -, hf⟩ := DecR.mk.inj he
      rw [hf]
      dsimp only
      rw [FlockDeploy.recJ, sampleOfJ_recordObj, ← hspec]
      exact sampleEq_map_norm (fun e he => (FlockDeploy.insOf_sample_norm (FlockDeploy.setupOf_some hx).1 he).1)
        (hud _ _ _ _ hx hq hn)
    exact checksAtJ_ok_of (f := ⟨f.stmt, f.j, #[]⟩) (FlockDeploy.decR_stmt hd) hD hL hC hq hn
  · trivial

theorem decStmt (rd : Msg → Option FinalP) : DecStmt rd := fun _ _ _ _ h => FlockDeploy.decR_stmt h

theorem decShaped (rd : Msg → Option FinalP) : DecShaped rd := fun _ _ _ _ h => FlockDeploy.decR_shaped h

theorem basesHold_spec (rd : Msg → Option FinalP) :
    BasesHold rd := by
  intro x st hx b hb as cs hlen
  have hck : ∀ s spec sts, setupOf false (xIOf x) = some (s, spec, sts) →
      s.read.check = .ok () ∧ s.read.draw.isSome = true := by
    intro s spec sts h
    obtain ⟨s', spec', sts', h', hq, hn⟩ := parse_innerOk hx
    rw [h] at h'
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h')
    exact ⟨hq, hn⟩
  simp only [VerdictP.bases, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl | rfl
  · exact holdsR_setup rd _ as cs
  · exact holdsR_shaped rd _ as cs (decShaped rd _ as cs)
  · exact holdsR_names rd _ as cs fun s spec sts h => (hck s spec sts h).1
  · exact holdsR_afterS rd _ as cs hlen hck sampleTiedS (custodyRecJ rd)

end

end RecDeploy
