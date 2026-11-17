module

public import Proofs.Flock.Claims.Complete.Zk.Consumed
public import Proofs.Flock.Claims.Complete.Zk.Json.Runs
public import Proofs.Flock.Claims.Complete.Zk.Json.Stmt
public import Proofs.Flock.Claims.Complete.Zk.Lanes
public import Proofs.Flock.Claims.Complete.Zk.RepRounds
public import Proofs.Flock.Claims.Complete.Zk.RingSwitch
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
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.Discharge
import Proofs.Flock.Claims.Complete.Zk.Extras
import Proofs.Flock.Claims.Complete.Zk.Replay

/-! Flock's masked completeness at one table (`complete_zk`) and at the deployed relation, from named premises, among
them the goals not yet proved, `LigeritoZk` and `Inner`. -/

@[expose] public section

namespace FlockComplete

open Flock FlockDeploy

theorem complete_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h64 : Claims.Out64 H) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hX : RsExtrasZk H rel) (hLg : LigeritoZk H rel) (hIn : Inner H rel)
    (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk rel) honestPZk (δcZk H) :=
  have hS : JsonRT.StmtOkZk H rel := JsonRT.stmtOkZk
  have hP : JsonRT.ProverOkZk H rel := JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H rel)
  have hD : RepDecodesZk H rel := JsonRT.repDecodesZk_goal h64 hS hP hR
  have hRp : Replay H rel := replay_goal h64 hS hP hR
  have hRS : RingSwitchZk H rel := ringSwitchZk_goal h64 hS hP hR hX
  complete_zk_of_goals H rel hMR
    { session := JsonRT.sessionRunsZk_goal hS hPR
      sends := sendCountZk_goal H rel
      linkFirst := linkFirstZk_goal H h64 rel
      finalFits := JsonRT.finalFitsZk_goal h64 hS hP hR
      draw := JsonRT.sampleMatchesZk_goal h64 hS hP
      liveReads := JsonRT.liveReadsZk_goal h64 hS hP
      canonical := JsonRT.canonicalZk_goal h64 hS hP
      rootB := JsonRT.custodyRootBZk_goal h64 hS hP
      publicsLink := JsonRT.custodyPublicsZk_goal h64 hS hP
      points := JsonRT.custodyPointsZk_goal h64 hS hP
      count := JsonRT.custodyCountZk_goal h64 hS hP
      rounds := JsonRT.custodyRoundsZk_goal h64 hS hP
      tables := tablesZk_goal H rel
      decodes := JsonRT.decodesZk_goal h64 hS hP
      publics := JsonRT.publicsZk_goal h64 hS hP
      schedules := schedulesZk_goal H rel
      proofCount := proofCountZk_goal H rel
      proofList := JsonRT.proofListZk_goal h64 hS hP
      proofDigests := JsonRT.proofDigestsZk_goal h64 hS hP
      repStream := JsonRT.repStreamZk_goal h64 hS hP
      repExtras := JsonRT.repExtrasZk_goal h64 hS hP
      repDecodes := hD
      repParams := JsonRT.repParamsZk_goal h64 hS hP hR
      padsRoot := JsonRT.padsRoot_goal h64 hS hP hR
      replay := hRp
      ringSwitch := hRS
      ligerito := hLg
      lanes := laneConstraints_goal hLg hD
      inner := hIn
      consumed := consumedZk_goal h64 hS hP hRp hRS hLg hIn
      capCommitted := JsonRT.capCommittedZk_goal h64 hS hP hR }

theorem complete_zk_relation (H : Claims.Msg → Claims.Msg) 
    (hR : JsonRT.RepShapedZk H Readable.relation) (hPR : JsonRT.ProverRunsZk H Readable.relation)
    (hLg : LigeritoZk H Readable.relation) (hIn : Inner H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk Readable.relation) honestPZk (δcZk H) := fun h64 =>
  complete_zk H Readable.relation h64 hR hPR (rsExtrasZk H) hLg hIn hMR h64

end FlockComplete
