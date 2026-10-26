import Proofs.Flock.Claims.Recursion.HonestStreams
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.Complete.WalkPacked

/-!
# The verdict passes on the honest proofs at the reader's record

`HonestVerdict sha512`: at every plain completeness run (`FlockComplete.Run`) and the deployed verifier's coins, the
verdict on the honest proofs at the reader's record (`verifyJPWith` at `jAt` of `recJ`) accepts. The reader's record is
the honest record (`recStreams_eq`, `recLink_eq`, `recordObj_liveRound`), its proof files are the session's
(`filesAt_eq`), and with the digests and coins substituted it is alike the record (`same_coinSub_recordObj`), so it
decodes as the printed record does (`decodeJ_recordObj`). The verdict at it is the plain verdict on the printed record,
which accepts (`Run.verdict_ok`).

* **The goals** (`goals_plain`): `complete_plain`'s goals, at the record's JSON (`JsonSpec`) and 64-byte outputs.
* **SHA-512's outputs are 64 bytes** (`out64_sha512`).
* **The verdict at a value that decodes as the record** (`verifyJWithM_eq_verifyWithM`).
* **The honest record's reads** (`run_recordJ`): `Run.json`'s, with the decode of every value alike the record.
* **The verdict** (`honestVerdict`).
-/

namespace FlockComplete.Replay

open Flock Transcript FlockDeploy

/-- **`complete_plain`'s goals**, at the record's JSON and the oracle's 64-byte outputs. -/
theorem goals_plain (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) :
    Goals H Readable.relation := by
  have hP : JsonRT.ProverOk H Readable.relation := proverOk h64
  have hW : SchemeWidths H Readable.relation := schemeWidths h64
  have hfin := lgFinal_json H S h64
  have w : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2,
      ZcAt o i r ∧ LcAt o i r ∧ RsAt o i r ∧ LgAt o i r ∧ LgCount o i r :=
    fun o _ hi _ hr => o.walkAt S h64 hP hi hr
  have hzc := fun o i hi r hr => (w o i hi r hr).1
  have hlc := fun o i hi r hr => (w o i hi r hr).2.1
  have hrs := fun o i hi r hr => (w o i hi r hr).2.2.1
  have hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r :=
    fun o _ hi _ hr => o.repWalk S h64 hP hi hr
  have hD := JsonRT.repDecodes_goal S h64 hP
  exact
    { session := JsonRT.sessionRuns_goal
      sends := sendCount_goal H _
      linkFirst := linkFirst_goal H h64 _
      finalFits := JsonRT.finalFits_goal S h64 hP
      draw := JsonRT.drawMatches_goal S h64 hP
      liveReads := JsonRT.liveReads_goal S h64 hP
      canonical := JsonRT.canonical_goal S h64 hP
      rootB := JsonRT.custodyRootB_goal S h64 hP
      publicsLink := JsonRT.custodyPublics_goal S h64 hP
      points := JsonRT.custodyPoints_goal S h64 hP
      count := JsonRT.custodyCount_goal S h64 hP
      rounds := JsonRT.custodyRounds_goal S h64 hP
      tables := tables_goal H _
      decodes := JsonRT.decodes_goal S h64 hP
      publics := JsonRT.publics_goal S h64 hP
      schedules := schedules_goal H _
      proofCount := proofCount_goal H _
      proofList := JsonRT.proofList_goal S h64 hP
      proofDigests := JsonRT.proofDigests_goal S h64 hP
      repStream := JsonRT.repStream_goal S h64 hP
      repExtras := JsonRT.repExtras_goal S h64 hP
      repDecodes := hD
      repParams := JsonRT.repParams_goal S h64 hP
      zerocheck := zerocheck_goal' H hzc
      lincheck := lincheck_goal' H hzc hlc (lcAlgebra H)
      ringSwitch := ringSwitch_goal' H hzc hlc (lcAlgebra H) hrs (rsAlgebra H)
      ligerito := ligerito_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW hfin
      consumed := consumed_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW hfin
      capCommitted := JsonRT.capCommitted_goal S h64 hP }

end FlockComplete.Replay

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest run's record, read as a value**: `Run.json`'s single table, streams, rounds and session, the printed
record printable, and every value alike it decoding as `sessOf`. -/
theorem run_recordJ (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel)
    (o : Run H rel) :
    ∃ st rb streams, o.sts = #[st] ∧ o.spec.tables = #[st.tags.table] ∧
      streams.size = o.spec.repStreams.size ∧
      (∃ bs, Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      Record.decodeWith o.hs o.spec o.record =
        .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) ∧
      o.sess = sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs ∧
      Good (recordObj o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) ∧
      ∀ j, Same (recordObj o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) j →
        Record.decodeJWith o.hs o.spec j =
          .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
  obtain ⟨st, hst, hsp, -, -⟩ := o.table
  obtain ⟨hct, hsig, hdraw, -⟩ := stmtOk o.toCase
  obtain ⟨hC, hR⟩ := hP o.toCase st (by rw [hst]; simp)
  have ht : o.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun := o.ran
  unfold sessionS at hrun
  rw [hst, htabs] at hrun
  obtain ⟨rb, streams, hrec, hrb, hrb64, hss, hrs, hex, bs, hbs, hrecs⟩ :=
    sessionP_facts H o.resp ht hct (Setup.ofCircuit st) _ _ hC hR hrun
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at hrec hex
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  have hd : Good (o.spec.unitDraw.getD Json.null) := by
    unfold DrawOk at hdraw
    revert hdraw
    cases o.spec.unitDraw with
    | none => exact fun _ => .null
    | some d => exact fun hd => hd.2.1
  dsimp only at hrec
  have hdec : Record.decodeWith o.hs o.spec o.record =
      .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
    rw [hrec]
    exact decode_recordObj S (hOf_size h64) ht hct hdraw hsig hrb hpts hpm hss hrs o.proofs
  refine ⟨st, rb, streams, hst, ht, hss, ⟨bs, hrecs⟩, hdec, by unfold Run.sess; rw [hdec]; rfl,
    good_recordObj ht _ rb _ streams o.proofs hd, fun j hj => ?_⟩
  exact decodeJ_recordObj S (hOf_size h64) ht hct hdraw hsig hrb hpts hpm hss hrs o.proofs hj

end FlockComplete.JsonRT

namespace Flock

/-- **The verdict at a parsed value is the verdict at a record** whose decode is the value's. -/
theorem verifyJWithM_eq_verifyWithM (hs : Hashes V) (spec : SessionSpec) (tabs : Array Setup) (proofs : Array ByteArray)
    {j : Lean.Json} {record : String} (h : Record.decodeJWith hs spec j = Record.decodeWith hs spec record) :
    verifyJWithM hs spec tabs j proofs = verifyWithM hs spec tabs record proofs := by
  simp only [verifyWithM, verifyJWithM, h]

end Flock

namespace FlockDeploy

open Flock FlockComplete

/-- **SHA-512's outputs are 64 bytes.** -/
theorem out64_sha512 : Claims.Out64 sha512 := fun m => by
  show (Claims.toMsg (Sha512.hash (Claims.ofMsg m))).length = 64
  have h := Sha512.hash_size (Claims.ofMsg m)
  generalize Sha512.hash (Claims.ofMsg m) = b at h ⊢
  cases b
  unfold Claims.toMsg
  rw [Array.length_toList]
  exact h

/-- **The verdict passes on the honest proofs at the reader's record**, at the record's JSON (`JsonSpec`): at every plain
completeness run and the deployed verifier's coins, `verifyJPWith` at `jAt` of `recJ` accepts. -/
theorem honestVerdict (S : Claims.JsonSpec FlockComplete.printJ Flock.canon) : HonestVerdict sha512 := by
  intro o cs hcs spec' sts' hs
  have h64 := out64_sha512
  have g := FlockComplete.Replay.goals_plain sha512 S h64
  change setupWith (Claims.hOf sha512) o.s.read = _ at hs
  rw [o.recSetup] at hs
  cases hs
  show verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf sha512) : Flock.Hashes Flock.V) o.spec (tabsOf o.sts)
    (coinSubJ o.spec o.recC₁ ((o.recFrames cs).zip cs.tail.dropLast)
      (shaSubJ o.spec ((filesAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) o.recPs).map
        (FlockComplete.dgOf o.spec (Claims.hOf sha512)))
        (recJ o.spec o.sts o.recLink o.recC₁ (o.recFrames cs) cs.tail.dropLast o.recPs))) o.recPs = .ok ()
  obtain ⟨st, rb, HS, hsts, htab, hss, ⟨bs, hrecs⟩, hdec, hsess, hgood, hdecJ⟩ :=
    FlockComplete.JsonRT.run_recordJ S h64 (FlockComplete.Replay.proverOk h64) o
  have hsh : ∀ (k : ℕ) (hk : k < o.recPs.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) k) o.recPs[k] :=
    honestShaped h64 o cs hcs o.spec o.sts o.recSetup
  have hnd : NamesDistinct o.spec := namesDistinct_of_setup (setupOf_some o.setup).2 o.recPublicInputs
  rw [← verifyJWithM_sub (Claims.hOf sha512) o.spec (tabsOf o.sts) o.recC₁ ((o.recFrames cs).zip cs.tail.dropLast)
    (recJ o.spec o.sts o.recLink o.recC₁ (o.recFrames cs) cs.tail.dropLast o.recPs) hsh hnd]
  have hHS : HS.size = 2 := by rw [hss, JsonRT.repStreams_one htab]; rfl
  have hsz : o.sts.size = 1 := by rw [hsts]; rfl
  obtain ⟨-, hts⟩ := setup_tables o.setup
  obtain ⟨hpsz, hall⟩ := sessionS_reps h64 (rfl : o.resp 0 = o.recC₁.bytes) hts.symm o.recTabsOk o.ran [] []
  have hfiles : filesAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) o.recPs = o.proofs := by
    refine filesAt_eq (by unfold Run.recPs; rw [filePs_size]) fun k hk => ?_
    have hk' : k < 2 * o.sts.size := by
      have hk1 : k < (filePs o.spec o.sts o.recC₁ o.proofs).size := hk
      rw [filePs_size, hpsz] at hk1
      exact hk1
    have hd : decodeProof (shapeAt (tabsOf o.sts) (pointsOf o.spec o.recC₁) k) o.proofs[k]! = .ok o.recPs[k]! :=
      (hall k hk').1
    rw [getElem!_pos o.proofs k (by omega), getElem!_pos o.recPs k hk] at hd
    exact hd
  have hJ : recJ o.spec o.sts o.recLink o.recC₁ (o.recFrames cs) cs.tail.dropLast o.recPs =
      JsonRT.recordObj o.spec (Claims.hOf sha512) #[rb] (pointsOf o.spec o.recC₁) HS o.proofs := by
    unfold recJ
    rw [recStreams_eq h64 g o hcs hsts htab hss hsess hrecs, hfiles, recLink_eq h64 g o hsts hsess,
      recordObj_liveRound]
  rw [hfiles, hJ]
  have F := fun k (hk : k < HS.size) => rep_facts h64 g o cs hsts htab hss hsess hk
  have hrl : repLens o.spec o.sts = HS.toList.map Array.size := by
    apply List.ext_getElem (by simp [repLens, hsz, hHS])
    intro k h1 h2
    simp only [repLens, List.getElem_map, List.getElem_range, Array.getElem_toList]
    exact (F k (by simpa [repLens, hsz, hHS] using h1)).1
  have hps : o.recPs.size = 2 := by
    unfold Run.recPs
    rw [filePs_size, hpsz, hsz]
  have hlenF : (o.recFrames cs).length = (HS.toList.map Array.size).sum := by
    unfold Run.recFrames
    rw [List.length_flatten, hps]
    congr 1
    apply List.ext_getElem (by simp [hHS])
    intro k h1 h2
    have hk : k < HS.size := by simpa [hHS] using h1
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    exact (F k hk).2.2.1
  have hmid : ∀ (i : ℕ) (hi : i < HS.size) (k : ℕ) (hk : k < HS[i].size), ∃ m c,
      ((o.recFrames cs).zip cs.tail.dropLast)[((HS.toList.take i).map Array.size).sum + k]? = some (m, c) ∧
        c.vals HS[i][k].n = HS[i][k].coins := by
    intro i hi k hk
    have hoff := arrays_offset HS.toList i (by simpa using hi) k (by simpa using hk)
    have hfl : (HS.toList.flatMap Array.toList).length = (HS.toList.map Array.size).sum := by
      simp [List.length_flatMap]
    obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 hoff
    have h2 := hcs.2
    rw [hrl] at h2
    have hp1 : ((HS.toList.take i).map Array.size).sum + k < (o.recFrames cs).length := by omega
    have hp2 : ((HS.toList.take i).map Array.size).sum + k < cs.tail.dropLast.length := by
      simp only [List.length_dropLast, List.length_tail]
      omega
    refine ⟨(o.recFrames cs)[((HS.toList.take i).map Array.size).sum + k]'hp1,
      cs.tail.dropLast[((HS.toList.take i).map Array.size).sum + k]'hp2, ?_, ?_⟩
    · have hz : ((HS.toList.take i).map Array.size).sum + k < ((o.recFrames cs).zip cs.tail.dropLast).length := by
        rw [List.length_zip]
        omega
      rw [List.getElem?_eq_getElem hz, List.getElem_zip]
    · have hc := hs_coins o hcs hrl hrecs hi hk
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp2, Option.getD_some] at hc
      exact hc
  have hsame := HonestJson.same_coinSub_recordObj o.spec (Claims.hOf sha512) #[rb] o.recC₁ HS o.proofs
    ((o.recFrames cs).zip cs.tail.dropLast) hgood hss (nodup_names hnd) hmid
  have hJ' := hdecJ _ hsame
  exact (verifyJWithM_eq_verifyWithM (Flock.Hashes.ofSha512 (Claims.hOf sha512)) o.spec (tabsOf o.sts) o.proofs
    (hJ'.trans hdec.symm)).trans (o.verdict_ok g)

end FlockDeploy
