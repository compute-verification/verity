import Proofs.Flock.Claims.Recursion.FlockKSDeployE
import Proofs.Flock.Claims.KS.ConflictTied

/-!
# Conflicts tied to collisions at the element-level inner verifier (`FlockDeploy.VE`)

`ConflictTied` at the plain record with `VE` as its verifier (`plainE_conflictTied`), at SHA-512, by
`plain_conflictTied`'s proof at the element-level verdict. An accepted run of `VE` decides within its own verifier log
(`accE_decide`), so the element-level verdict accepts with SHA-512 answered only within that log (`accE_within`). Its
query round places the reader's columns among level 0's Merkle openings, as the shipped verdict's does
(`plainQueryRound_verifyWithMK`): the element-level verdict drops only digests of encodings, and keeps every rep's
checks and every hash call. The decoded session's `root_B` and rep streams are `liveOf`'s (`decodeWithK_rootB_live`,
`liveOf_decodeWithK`), whatever `keep`, so each opening climbs to the link's `root_B` (`plainE_opening`), and two
accepted runs at one tape that open different columns at one position collide in their joined logs.

That discharges (P1)'s first premise at `VE`: (P1) at the bound record at `VE`'s steps
(`RecDeploy.plainKSAbort_bdE`) rests on the table bound counted per round at `deployPlainE rel` alone, proved at
`deployPlain rel` (`plain_tableBoundθOnAccR`'s) and not yet at `VE`.
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128

/-! ## An accepted run of `VE`, within its own log -/

/-- **An accepted run of `VE` decides, within its own verifier log** (`acc_decide`'s proof, at `nextE`). -/
theorem accE_decide {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} (h : (Claims.run H P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P VE x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decisionE s x link c₁ mids f) () ((Claims.run H P VE x ω r).vLog.map Prod.fst) := by
  have ha := FlockDeploy.KS.run_acc_next H P VE x ω r h
  have hl := run_acc_log H P VE x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((nextE x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P VE x ω r).vLog := hl
  cases hact : act false x ((Claims.run H P VE x ω r).msgs.map Prod.fst) cs with
  | rej => rw [nextE_rej H hact] at ha; cases ha
  | coin c => rw [nextE_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · simp only [nextE, hact] at ha
      exact outOf_acc H _ ha
    · simp only [nextE, hact, outOf_log]
      exact hp

/-- **An element-level decision within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the
element-level verdict within `L`, and the checks (`decision_runs`'s proof). -/
theorem decisionE_runs {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts)) (h : Runs sha512 (decisionE s x link c₁ mids f) () L) :
    Runs sha512 (verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks false s spec x link c₁ mids f = .ok () := by
  unfold decisionE at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAtE at h
  obtain ⟨⟨⟩, h1, h2⟩ := Runs.bind h
  exact ⟨h1, Runs.lift h2⟩

/-- **The element-level verdict within `L` is the element-level verdict with SHA-512 answered only within `L`.** -/
theorem verdictE_within {spec : SessionSpec} {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    {L : List Claims.Msg} (h : Runs sha512 (verifyWithME Claims.Hashes.oc spec tabs record proofs) () L) :
    verifyWithME (hashesIn L) spec tabs record proofs = .ok () :=
  (With.verifyWithME_sim (within L) (within_hashes L) spec tabs record proofs) () h

/-- **An accepted run of `VE` at `sha512`** (`acc_within`'s proof): the statement sets up, the run decides on its last
message, read as a final message on the statement, and on the link of the prover's first message; the custody checks
pass; and the element-level verdict accepts with SHA-512 answered only within the run's verifier log. -/
theorem accE_within {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    (h : (Claims.run sha512 P VE x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P VE x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks false s spec x link c₁ mids f = .ok () ∧
      verifyWithME (hashesIn ((Claims.run sha512 P VE x ω r).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accE_decide h
  obtain ⟨hv, hc⟩ := decisionE_runs (setupOf_some (act_decide hact)).2 hrun
  obtain ⟨rest, hhead⟩ := run_head sha512 P VE x ω r h
  obtain ⟨rest', has, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, ?_, hlink, hc, verdictE_within hv⟩
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  exact hl

/-! ## An accepted run's openings, and `ConflictTied` -/

/-- **An accepted run of `VE` opens a column only as a Merkle opening** of table 0's level 0 in its own log
(`plain_opening_raw`'s proof, at the element-level verdict): a column `u` a rep opens at `q` is an opened row whose
opening under table 0's scheme against `root_B[0]` of the tape's first message's link passed, every SHA-512 input of it
in the run's verifier log. -/
theorem plainE_opening_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin (VE.k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P VE x ω r).seen) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := accE_within h
  obtain ⟨st, sch, row, hst, hsch, hop, hu⟩ :=
    plain_opening_raw_of_verdictK (fun _ => true) (L := (Claims.run sha512 P VE x ω r).vLog) hs hf hc hv hu
  exact ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, hop, hu⟩

/-- **An accepted run of `VE` opens a column only as a Merkle opening** of table 0's level 0 in its own log: a column
`u` the run's reader opens at `q` is one a rep opens (`plainE_opening_raw`). -/
theorem plainE_opening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin (VE.k x) → Coin}
    {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P VE x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols false (Claims.run sha512 P VE x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Flock.Stmt) (st : Flock.Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P VE x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of :=
  plainE_opening_raw h (mem_plainRawCols (lookCol_mem hu))

/-- **`ConflictTied` at the plain record at `VE`**: two accepted runs of the element-level verifier at one tape that
open different columns at one position collide in their joined verifier logs, at SHA-512. -/
theorem plainE_conflictTied (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTied sha512 (deployPlainE rel) (wordLen false) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    plainE_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    plainE_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

/-- **(P1) at `VE`, from the table bound alone**: `plainE_KSAbortR` with its conflicts tied to collisions
(`plainE_conflictTied`). -/
theorem plainE_KSAbort_of_table (rel : Readable.Parsed → Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR sha512 (deployPlainE rel) (wordLen false) (wide false))
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK ((deployPlainE rel).mapCols trimCol) decode) :
    (Claims.Recursion.Inner.ofFlock ((deployPlainE rel).mapCols trimCol) decode).KSAbort sha512 :=
  plainE_KSAbortR rel (plainE_conflictTied rel) htab decode hdec

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **(P1) at the bound record at `VE`'s steps** (option (b)), from the table bound counted per round at the plain
record at `VE` (`FlockDeploy.KS.plainE_KSAbort_of_table`), whatever the bound record's other choices. -/
theorem plainKSAbort_bdE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAccR FlockDeploy.sha512 (FlockDeploy.deployPlainE rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :
    (bdDeployed rel FlockDeploy.VE.next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight join rowIn final w).PlainKSAbort FlockDeploy.sha512 :=
  fun decI hdec => FlockDeploy.KS.plainE_KSAbort_of_table rel htab decI hdec

end RecDeploy
