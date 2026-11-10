module

public import Proofs.Flock.Claims.Recursion.RestReadR
public import Proofs.Flock.Claims.Recursion.RestVerdictCustody
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
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.Recursion.CallsAgree

/-!
# `DecLive`: the reader's record reads as custody's

At every decision of `nextR` with a coin a message whose statement passes its scope check, `liveOfJ` reads the record
value `decR` builds (`recJ`, its coins the live coins). It is `custodyRecJ`'s reading (`liveOfJ_sub`) at the record
itself, before its proof digests are substituted: the record's link and streams are the honest record's objects
(`recordObj_link`, `recordObj_streams`).
-/

@[expose] public section

namespace RecDeploy.RestRead

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP decR liveOfJ coinSubJ recStreams)
open FlockComplete.JsonRT (recordObj recordList recordList_distinct linkObj streamObj)

theorem recordObj_link (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (pts : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Record.fld (recordObj spec h rb pts streams proofs) "link" = .ok (linkObj spec h rb pts) := by
  unfold recordObj
  exact fld_mkObj (recordList_distinct ..) (by simp [recordList])

theorem recordObj_streams (spec : SessionSpec) (h : ByteArray → ByteArray) (rb : Array ByteArray)
    (pts : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Record.arr (recordObj spec h rb pts streams proofs) "streams" =
      .ok ((spec.repStreams.zip streams).map (streamObj spec h rb)) := by
  unfold recordObj
  simp only [Record.arr, fld_mkObj (recordList_distinct ..)
    (show ("streams", Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rb))) ∈
      recordList spec h rb pts streams proofs by simp [recordList])] <;> rfl

/-- **`DecLive` holds at every reader.** -/
theorem decLive (rd : Msg → Option FinalP) : DecLive rd := by
  intro x as cs d hlen hd hq
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hx, -, -, hrss, hlink, hpf, rfl⟩ := FlockDeploy.decR_some hd
  dsimp only at hq hlink hpf ⊢
  have hs := (FlockDeploy.setupOf_some hx).2
  have hpi := Claims.Bridge.Custody.publicInputs_none_of_check hq
  have htn := FlockDeploy.KS.tables_nodup hs hpi
  have hnd := FlockDeploy.KS.repNames_nodup htn
  have hts : spec.tables.size = sts.size := by
    have h := FlockDeploy.Ins.setup_read hs
    unfold Readable.Read.setup at h
    rw [hpi] at h
    exact (FlockComplete.ZkTables.setupTables_size h).2.2
  obtain ⟨hl1, -⟩ := FlockDeploy.linkR_rootB hlink
  obtain ⟨hsum, hpf2⟩ := readPs_len hpf
  have hrl : rss.length = as.tail.dropLast.length := (FlockDeploy.mapM_some_get hrss).1
  have hMc : pf.2.length ≤ cs.tail.dropLast.length := by
    simp only [List.length_dropLast, List.length_tail] at hrl ⊢
    omega
  have hrb : link.1.toArray.size = spec.tables.size := by
    rw [List.size_toArray, hl1, List.length_map, cutLens_length, hts]
    simp [FlockDeploy.level0Lens]
  have hss : (recStreams spec sts pf.2 cs.tail.dropLast).size = spec.repStreams.size := by
    simp [recStreams, FlockDeploy.repStreams_size, hts]
  have hbytes := recStreams_bytes spec sts pf.2 cs.tail.dropLast (hpf2.trans hsum.symm).le
  have hfl : ((recStreams spec sts pf.2 cs.tail.dropLast).toList.flatMap fun s : Array Round => s.toList).length =
      pf.2.length := by
    have := congrArg List.length hbytes
    simpa [List.length_flatten, List.length_flatMap, Function.comp_def] using this
  obtain ⟨lv, hL, -⟩ := liveOfJ_sub spec c₁ (pf.2.zip cs.tail.dropLast) _
    (Claims.hOf FlockDeploy.sha512) link.1.toArray (FlockDeploy.pointsOf spec c₁)
    (recStreams spec sts pf.2 cs.tail.dropLast) htn hrb hss hnd
    (recordObj_link spec _ _ _ _ (FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) pf.1))
    (recordObj_streams spec _ _ _ _
      (FlockDeploy.filesAt (FlockDeploy.tabsOf sts) (FlockDeploy.pointsOf spec c₁) pf.1))
    (by rw [hfl, List.length_zip]; omega)
  exact ⟨lv, hL⟩

end RecDeploy.RestRead
