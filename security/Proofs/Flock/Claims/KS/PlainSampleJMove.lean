module

public import Proofs.Flock.Claims.KS.ConflictTied
public import Proofs.Flock.Claims.KS.PlainSampleJ
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
import Proofs.Flock.Claims.Bridge.CustodyPlain
import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ

/-! The reader's premise moves between the text and parsed-value writers, as the deployed plain reader reads only a round's coins. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock

theorem plainRepColsOf_erase (sh : Shape) {a b : Array Round} (bytes : ByteArray)
    (h : a.toList.map Round.erase = b.toList.map Round.erase) :
    Claims.Flock.plainRepColsOf sh a bytes = Claims.Flock.plainRepColsOf sh b bytes := by
  refine plainRepColsOf_congr sh bytes fun j => ?_
  have e := congrArg (fun l : List Round => l[j]?.map (fun rd : Round => rd.coins)) h
  simpa only [List.getElem?_map, Option.map_map, Function.comp_def, Round.erase_coins, Array.getElem?_toList]
    using e

end FlockDeploy.KS

namespace Claims.Bridge.PlainSample

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Discharge.Exec (verifyJWithMK_tables_cap)
open FlockDeploy.KS (plainRepColsOf_erase)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem txNoOffAtJK_of_records {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {o : Tables.LiveOutJ 1 mPts} {t t' : TranscriptJ} (hA : AcceptsAtDJK keep I (some d) t)
    (hR : RecordsLiveAtDJK keep I d o t) (hA' : AcceptsAtDJK keep I (some d) t')
    (hR' : RecordsLiveAtDJK keep I d o t') (h : TxNoOffAtJK keep I d C t) : TxNoOffAtJK keep I d C t' := by
  intro spec st sess' sch s₀' s₁' hset hdec' hsch hs₀' hs₁'
  obtain ⟨hpr, hall⟩ := hR hA
  obtain ⟨hpr', hall'⟩ := hR' hA'
  obtain ⟨spec₀, sts₀, hset₀, hv⟩ := hA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hset.symm.trans hset₀))
  obtain ⟨sess, -, hdec, -⟩ := verifyJWithMK_tables_cap keep With.std hv
  obtain ⟨-, hrs⟩ := hall spec #[st] sess hset hdec
  obtain ⟨-, hrs'⟩ := hall' spec #[st] sess' hset hdec'
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain ⟨y₀', hy₀', hr₀'⟩ := (hrs' ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁', hy₁', hr₁'⟩ := (hrs' ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀' = s₀' := Option.some.inj (hy₀'.symm.trans hs₀')
  obtain rfl : y₁' = s₁' := Option.some.inj (hy₁'.symm.trans hs₁')
  have hno := h spec st sess sch y₀ y₁ hset hdec hsch hy₀ hy₁
  rw [plainRepColsOf_erase _ _ (hr₀'.trans hr₀.symm), plainRepColsOf_erase _ _ (hr₁'.trans hr₁.symm), hpr', ← hpr]
  exact hno

theorem txNoOffAtK_iff_JK {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {o : Tables.LiveOutJ 1 mPts} {t : Exec.Transcript} {t' : TranscriptJ} (hA : AcceptsAtDK keep I (some d) t)
    (hR : RecordsLiveAtDK keep I d o t) (hA' : AcceptsAtDJK keep I (some d) t')
    (hR' : RecordsLiveAtDJK keep I d o t') : TxNoOffAtK keep I d C t ↔ TxNoOffAtJK keep I d C t' := by
  obtain ⟨j, hj, hAj⟩ := (acceptsAtDK_iff_exists_J t).1 hA
  have hRj := (recordsLiveAtDK_iff_J t.proofs hj).1 hR
  exact (txNoOffAtK_iff_J t.proofs hj).trans
    ⟨txNoOffAtJK_of_records hAj hRj hA' hR', txNoOffAtJK_of_records hA' hR' hAj hRj⟩

theorem recordsLiveAtDK_wrPDK_of {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} (hA : AcceptsAtDK keep I (some d) (wrPDK keep I d o)) :
    RecordsLiveAtDK keep I d o (wrPDK keep I d o) := by
  revert hA
  unfold wrPDK
  split
  · rename_i h
    exact fun _ => (Classical.choose_spec h).2
  · intro hA
    obtain ⟨j, -, hAj⟩ := (acceptsAtDK_iff_exists_J _).1 hA
    exact absurd hAj (not_acceptsAtDJK_nil _ _ _ _)

theorem txNoOffAtK_wrPDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {o : Tables.LiveOutJ 1 mPts} (hA : AcceptsAtDK keep I (some d) (wrPDK keep I d o)) :
    TxNoOffAtK keep I d C (wrPDK keep I d o) ↔ TxNoOffAtJK keep I d C (wrPDJK keep I d o) :=
  txNoOffAtK_iff_JK hA (recordsLiveAtDK_wrPDK_of hA) (acceptsAtDJK_wrPDJK_of_K hA (recordsLiveAtDK_wrPDK_of hA))
    (recordsLiveAtDJK_wrPDJK keep I d o)

end Claims.Bridge.PlainSample

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge Claims.Bridge.PlainSample
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem plainVerdictAtJK_of_K {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {rootB₀ : Array ByteArray} {o : Tables.LiveOutJ 1 mPts} (h : plainVerdictAtK keep I d C rootB₀ o) :
    plainVerdictAtJK keep I d C rootB₀ o := by
  obtain ⟨hA, hroot, hno⟩ := h
  exact ⟨acceptsAtDJK_wrPDJK_of_K hA (recordsLiveAtDK_wrPDK_of hA), hroot, (txNoOffAtK_wrPDK_iff_J hA).1 hno⟩

end FlockDeploy.KS
