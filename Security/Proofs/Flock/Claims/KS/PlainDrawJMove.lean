import Proofs.Flock.Claims.KS.PlainDrawJ
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# The reader's premise between the text and parsed-value writers

`TxNoOffAtK` reads the session `wrPDK` writes and `TxNoOffAtJK` the one `wrPDJK` writes. Each writer picks an accepted
record of the same live outcome, and the two records' rep streams agree with the outcome's only up to their rounds'
digests (`Round.erase`). The deployed plain reader reads a round's coins only (`plainRepColsOf`), so it reads both
alike:

* `plainRepColsOf_erase`: rounds equal up to `Round.erase` give one rep's columns.
* `txNoOffAtJK_of_records`: two accepted parsed sessions that record one outcome have the same `TxNoOffAtJK`.
* `txNoOffAtK_iff_JK`, `txNoOffAtK_wrPDK_iff_J`: an accepted record text of an outcome and an accepted parsed session
  of it have the same reader's premise (through `acceptsAtDK_iff_exists_J`, `txNoOffAtK_iff_J`), so where flock-verify
  accepts what `wrPDK` writes, which then records the outcome (`recordsLiveAtDK_wrPDK_of`), `TxNoOffAtK` there is
  `TxNoOffAtJK` at what `wrPDJK` writes.
* `plainVerdictAtJK_of_K`: `plainVerdictAtK` gives `plainVerdictAtJK` at the same outcome and root.
-/

namespace FlockDeploy.KS

open Flock

/-- **The plain reader's columns ignore the rounds' digests**: rounds equal up to `Round.erase` give one rep's columns
(`plainRepColsOf` reads the coins). -/
theorem plainRepColsOf_erase (sh : Shape) {a b : Array Round} (bytes : ByteArray)
    (h : a.toList.map Round.erase = b.toList.map Round.erase) :
    Claims.Flock.plainRepColsOf sh a bytes = Claims.Flock.plainRepColsOf sh b bytes := by
  refine plainRepColsOf_congr sh bytes fun j => ?_
  have e := congrArg (fun l : List Round => l[j]?.map (fun rd : Round => rd.coins)) h
  simpa only [List.getElem?_map, Option.map_map, Function.comp_def, Round.erase_coins, Array.getElem?_toList]
    using e

end FlockDeploy.KS

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Discharge.Exec (verifyJWithMK_tables_cap)
open FlockDeploy.KS (plainRepColsOf_erase)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **Two accepted parsed sessions that record one outcome have the same reader's premise**: their proof files are the
outcome's, and at the setup at `d` table 0's rep streams are the outcome's up to the rounds' digests, which the reader
does not read (`plainRepColsOf_erase`). -/
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

/-- **An accepted record text of an outcome and an accepted parsed session of it have the same reader's premise**:
the text parses (`acceptsAtDK_iff_exists_J`), its premise is the one at its parsed value (`txNoOffAtK_iff_J`), and two
parsed sessions of one outcome agree (`txNoOffAtJK_of_records`). -/
theorem txNoOffAtK_iff_JK {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {o : Tables.LiveOutJ 1 mPts} {t : Exec.Transcript} {t' : TranscriptJ} (hA : AcceptsAtDK keep I (some d) t)
    (hR : RecordsLiveAtDK keep I d o t) (hA' : AcceptsAtDJK keep I (some d) t')
    (hR' : RecordsLiveAtDJK keep I d o t') : TxNoOffAtK keep I d C t ↔ TxNoOffAtJK keep I d C t' := by
  obtain ⟨j, hj, hAj⟩ := (acceptsAtDK_iff_exists_J t).1 hA
  have hRj := (recordsLiveAtDK_iff_J t.proofs hj).1 hR
  exact (txNoOffAtK_iff_J t.proofs hj).trans
    ⟨txNoOffAtJK_of_records hAj hRj hA' hR', txNoOffAtJK_of_records hA' hR' hAj hRj⟩

/-- **What `wrPDK` writes records the outcome wherever flock-verify accepts it**: otherwise it is a record without
proof files, whose parsed value no verdict accepts (`not_acceptsAtDJK_nil`). -/
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

/-- **The reader's premise moves from `wrPDK` to `wrPDJK`**: where flock-verify accepts what `wrPDK` writes for the
outcome at the setup at `d`, that record records the outcome (`recordsLiveAtDK_wrPDK_of`), so `wrPDJK` accepts a parsed
session of it (`acceptsAtDJK_wrPDJK_of_K`), and the two writers' sessions have the same `TxNoOff`
(`txNoOffAtK_iff_JK`). -/
theorem txNoOffAtK_wrPDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {o : Tables.LiveOutJ 1 mPts} (hA : AcceptsAtDK keep I (some d) (wrPDK keep I d o)) :
    TxNoOffAtK keep I d C (wrPDK keep I d o) ↔ TxNoOffAtJK keep I d C (wrPDJK keep I d o) :=
  txNoOffAtK_iff_JK hA (recordsLiveAtDK_wrPDK_of hA) (acceptsAtDJK_wrPDJK_of_K hA (recordsLiveAtDK_wrPDK_of hA))
    (recordsLiveAtDJK_wrPDJK keep I d o)

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The verdict at the text writer gives the verdict at the parsed-value writer**, at the same outcome and root:
`wrPDJK` accepts where `wrPDK` does (`acceptsAtDJK_wrPDJK_of_K`), and the reader's premise moves with it
(`txNoOffAtK_wrPDK_iff_J`). -/
theorem plainVerdictAtJK_of_K {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N mPts : ℕ} {C : Fin N → List 𝔽}
    {rootB₀ : Array ByteArray} {o : Tables.LiveOutJ 1 mPts} (h : plainVerdictAtK keep I d C rootB₀ o) :
    plainVerdictAtJK keep I d C rootB₀ o := by
  obtain ⟨hA, hroot, hno⟩ := h
  exact ⟨acceptsAtDJK_wrPDJK_of_K hA (recordsLiveAtDK_wrPDK_of hA), hroot, (txNoOffAtK_wrPDK_iff_J hA).1 hno⟩

end FlockDeploy.KS
