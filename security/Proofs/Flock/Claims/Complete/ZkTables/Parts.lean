module

public import Proofs.Flock.Claims.Complete.ZkTables.Level0
public import Proofs.Flock.Claims.Complete.ZkTables.Table
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
import Proofs.Flock.Claims.Complete.Json.Goals
import Proofs.Flock.Claims.Complete.ZkTables.View

/-! Masked completeness from level 0's four premises and the rest (`complete_jsonT_of_parts`). -/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT

section Rest

variable (H : Claims.Msg → Claims.Msg)

def ShapeRest : Prop := Level0.ProverShape → RepShapedT H

def RunsRest : Prop := Level0.ProverRuns → Level0.Lanes → Level0.Verifier → ProverRunsT H

def LigeritoRest : Prop := Level0.Verifier → LigeritoT H

def InnerRest : Prop := Level0.Verifier → InnerT H

end Rest

open FlockComplete.Replay in
theorem ligeritoRest (H : Claims.Msg → Claims.Msg) : LigeritoRest H := by
  intro hV h64 st tr hT ω i d k r v cm pts e rep ic hin hrep prior ws l p t' l' _ _ hgo R hAg q hq sw hsw
  obtain ⟨_, _, _, _, _, h, -⟩ :=
    view hV h64 st tr hT ω i d r v cm pts e rep ic hin hrep prior ws l p t' l' hgo R hAg q hq sw hsw
  exact ⟨_, h⟩

open FlockComplete.Replay in
theorem innerRest (H : Claims.Msg → Claims.Msg) : InnerRest H := by
  intro hV h64 st tr hT ω i d k r v cm pts e rep ic hin hrep prior ws l p t' l' _ _ hgo R hAg q hq sw hsw lg hlg lanes
    hlanes
  obtain ⟨rec, tI, lI, rp, lgc, hLg, htk, hlgc, hq1, hinn⟩ :=
    view hV h64 st tr hT ω i d r v cm pts e rep ic hin hrep prior ws l p t' l' hgo R hAg q hq sw hsw
  obtain rfl : lg = (rec, ⟨R, tI.k, tI.pending⟩) := Except.ok.inj (hlg.symm.trans hLg)
  obtain rfl : lanes = lgc := Except.ok.inj (hlanes.symm.trans hlgc)
  rw [hq1]
  obtain ⟨hz, hb, hpi, ht'⟩ := innerP_ret _ (Claims.hOf H) _ _ _ _ _ _ _ hinn
  rw [hpi]
  subst ht'
  exact ⟨_, verifyInnerWith_of' (Claims.hOf H) (JsonRT.hOf_size h64) _ ω i rep (ω.innerSalt i rep) hin.innerCommit _ _
    hz hb tI htk (hAg.mono (Nat.zero_le _))⟩

theorem complete_jsonT_of_parts (H : Claims.Msg → Claims.Msg) 
    (h0 : Level0.ProverRuns) (h1 : Level0.ProverShape) (h2 : Level0.Lanes) (h3 : Level0.Verifier)
    (gS : ShapeRest H) (gR : RunsRest H) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT H (gS h1) (gR h0 h2 h3) (ligeritoRest H h3) (innerRest H h3) hMR

end FlockComplete.ZkTables
