import Proofs.Flock.Soundness.Discharge.PlainHJ.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted

/-!
# The decoder is right at a plain session of `J` hidden-output tables

`decodesOnPHJ_exec`: `PlainJ.decodesOnPJ_exec` under hidden outputs, the plain counterpart of the decoder
`ZkHidden.zk_session_soundHJ` reads (without its masks). At an outcome flock-verify accepts, at table `j`:

* `RecordsLiveJ` gives the record's draw `dj S`, the proof files, the link points, and table `j`'s root and streams.
* `setupOf_tablePHJ` sets table `j` up with setupHidden under `tagsAt` at `drawJ` (`HiddenExec.setupTables_okH` at one
  table, `ZkHidden.setupTables_partsH` at several); `DrawOkPHJ` gives its proved units and link points, so a
  `DrawSetupH` exists (`HiddenExec.drawSetupH_of_setupHiddenI`), and `planPHJ`'s table `j` is its table
  (`tablePHJ_of`), `stmtOf` of setupHidden's statement.
* `accepts_reps_atH` (`DecodeZLJ.accepts_reps_at` under hidden outputs, from the acceptance lemmas without `hh`) gives
  each rep's run and rounds and the salted verdict; `repRun_at` makes each run a `RepRun` for `decJ`, and
  `playsT_of_repRun` a run on its tape.
-/

namespace FlockSoundness.Discharge.PlainHJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.Integrate
open FlockSoundness.Discharge.Composed (nTab nTab_pos drawJ partS)
open FlockSoundness.Discharge.DecodeZLJ (partJson decJ nXT RecordsLiveJ)
open FlockSoundness.Discharge.PlainJ (inputsPJ)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ)

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

/-- **flock-verify sets table `j` up under its tags at its draw, under hidden outputs**, without `--zk`
(`PlainJ.setupOf_tablePJ` with `setupHidden`, `ZkHidden.setupOf_tableJH` at the plain inputs): at the record's draw `d`,
`setupTables` at `I` has `nTab I` tables, table `j` setupHidden's statement under `tagsAt` at `drawJ`. -/
theorem setupOf_tablePHJ {I : Inputs} {record : String} {d : Lean.Json} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = true) (hd : recordDraw record = some d) (hs : setupOf I record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupHidden (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  by_cases hc : I.count ≤ 1
  · rcases HiddenExec.setupTables_okH hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt I.tags I.count 0 = I.tags := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := ZkHidden.setupTables_partsH hh hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partDraw population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : tagsAt I.tags I.count j = { I.tags with identity := I.tags.tableIdentity I.count j } := by
      simp [tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

/-- **flock-verify's acceptance under hidden outputs, table by table and rep by rep** (`DecodeZLJ.accepts_reps_at`
under hidden outputs): the acceptance lemmas without `hh` (`accepts_verdict_leafAny`, `accepts_rep_roundsAny`,
`setupTables_pointsAny`) at every table `j`, the verdict at `leaf512` under `hm96-sha512/v1` leaves (`leafOfId_hm96`). -/
theorem accepts_reps_atH {I : Inputs} {t : Exec.Transcript}
    (hrr : I.tags.retainRounds = true) (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session),
      setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (_hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m), ∃ pend : Fin 2 → PendL 𝔽 Dig,
          (∀ r : Fin 2, ∃ (s : StreamRec) (p : Proof) (Tv : F128),
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r.val) =
              some s ∧
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r.val]! = .ok p ∧
            Plays fOf (Model.repC execArith r.val (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)))
              (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
                PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
              (s.rounds.toList.map (·.coins)) (pend r) ∧
            (chunks (repFrames (Setup.ofCircuit sts[j]) sch (spec.domain spec.tables[j]! r.val).toUTF8
              (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)).length) p Tv)).1.length = s.rounds.size ∧
            ∀ k < s.rounds.size, Retains (s.rounds[k]!) (encs ((chunks (repFrames (Setup.ofCircuit sts[j]) sch
              (spec.domain spec.tables[j]! r.val).toUTF8 (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
                PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p Tv)).1.getD k []))) ∧
          (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[j]!) (pend 0, pend 1)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := HiddenExec.accepts_verdict_leafAny (Or.inl hl) h
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := HiddenExec.accepts_rep_roundsAny hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := HiddenExec.setupTables_pointsAny hsu
  have hpts := decode_points hdec
  refine ⟨spec, sts, sess, hsu, hdec, fun j hj => ?_⟩
  obtain ⟨sch, hf, hk⟩ := hall j hj
  obtain ⟨sch', hf', hR0, hR1⟩ := hallR j hj
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  refine ⟨sch, hf, fun hwf hr hm => ?_⟩
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv⟩ := hk hwf hr hm
  rw [hl, leafOfId_hm96] at hv
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp j hj).symm)
  obtain ⟨e₀, f₀, xs₀, T₀, hf₀, hx₀, hq₀, hl₀, hr₀⟩ := hR0
  obtain ⟨e₁, f₁, xs₁, T₁, hf₁, hx₁, hq₁, hl₁, hr₁⟩ := hR1
  rw [(hext xs₀ hx₀).1] at hq₀ hl₀ hr₀
  rw [(hext xs₁ hx₁).1] at hq₁ hl₁ hr₁
  obtain rfl : f₀ = p₀ := Except.ok.inj (hq₀.symm.trans hp₀)
  obtain rfl : f₁ = p₁ := Except.ok.inj (hq₁.symm.trans hp₁)
  obtain rfl : e₀ = s₀ := Option.some.inj (hf₀.symm.trans hs₀)
  obtain rfl : e₁ = s₁ := Option.some.inj (hf₁.symm.trans hs₁)
  refine ⟨![pend₀, pend₁], fun r => ?_, hv⟩
  rcases r with ⟨_ | _ | k, hk2⟩
  · exact ⟨e₀, f₀, T₀, hf₀, hq₀, hpl₀, hl₀, hr₀⟩
  · exact ⟨e₁, f₁, T₁, hf₁, hq₁, hpl₁, hl₁, hr₁⟩
  · omega

private theorem stmtOf_congr {a b : Flock.Stmt} (h : a = b) (ha : StmtWF a) (hb : StmtWF b) :
    stmtOf a ha (regionsOf a) = stmtOf b hb (regionsOf b) := by
  subst h
  rfl

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (x₀ : HiddenExec.DrawSetupH (inputsPJ I 0) mPts S₀ d₀)

set_option maxHeartbeats 4000000 in
/-- **The decoder is right at a plain session of `J` hidden-output tables** (R11b at `--plain-tables --session-tables J`
under hidden outputs): given the tags, the session's circuit, `wr` the challenger's record and the challenger's draws
setting each table up at `mPts` link points, `DecodesOnJ` holds at `planPHJ` with the decoder `decJ` on each table's two
streams, single coins as field elements, no final messages, table `j`'s cap from its root `root_B[j]` (`capOfRoot`) and
its openings `opnPHJ`. -/
theorem decodesOnPHJ_exec (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (L : Law n)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planPHJ (Reg := Reg) dj x₀ S R).length mPts → Exec.Transcript)
    (hT : TagsOkZHJ I) (hwr : RecordsLiveJ I dj (fun S R => (planPHJ (Reg := Reg) dj x₀ S R).length) wr)
    (hdraw : ∀ ω, DrawOkPHJ I dj mPts (L.draw ω)) :
    DecodesOnJ H512 enc512 leaf512 (planPHJ dj x₀) I wr (opnPHJ I (planPHJ dj x₀) dj) (decJ (planPHJ dj x₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R j o hacc
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hwr _ R o hacc
  obtain ⟨spec, sts, sess, hsu, hdec, hk⟩ := accepts_reps_atH hrr hl hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tablePHJ hh hd hsu
  have hjn : j.val < nTab I := planPHJ_length dj x₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hsu hdec
  obtain ⟨sch, hf, hk⟩ := hk j.val hj
  have hmP : sts[j.val].mPts = mPts := (hdraw ω ⟨j.val, hjn⟩).2 _ hst
  obtain ⟨hwf, hr⟩ := HiddenExec.setupHidden_wf hst
  have hm : 13 ≤ sts[j.val].m := by have := exec_fast100_ge hf; omega
  have hdl : sts[j.val].merkle.digestLen = 64 := by
    rw [HiddenExec.setupHidden_hm96 ((tagsAt_merkleLeaf _ _ _).trans hl) hst]; rfl
  obtain ⟨pop, -, hdr⟩ := HiddenExec.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.drawSetupH_of_setupHiddenI (I := inputsPJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) (PlainJ.typedPJ ht j.val)
    ((ZkHidden.parseH_tagsAt ht I.count j.val I.circuitFile I.tables).trans hc) hst hf hD ((hdraw ω ⟨j.val, hjn⟩).1 D hD) hmP
  obtain ⟨y, hyx, e⟩ := tablePHJ_of dj x₀ (S := L.draw ω) (j := ⟨j.val, hjn⟩) x
  have hget : (planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j = (tablePHJ dj x₀ (L.draw ω) ⟨j.val, hjn⟩).2.T :=
    List.get_ofFn _ j
  have hS : ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j).S = stmtOf sts[j.val] hwf (regionsOf sts[j.val]) := by
    rw [hget, e]
    show stmtOf y.st y.wf.1 (regionsOf y.st) = _
    exact stmtOf_congr (hyx.trans hx) _ _
  have hpl : ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j).ptLocal = PT_LOCAL := by rw [hget, e]; rfl
  have hsch : ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j).sch = schedOf sch := by
    have h1 : Accounting.fast100 sts[j.val].m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j).hsch
    rw [hS] at h2
    exact Option.some.inj (h2.symm.trans h1)
  rw [hmP, hpts] at hk
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hrep : ∀ r : Fin 2,
      PlaysCoin.RepRun (repsOf (D := Dig) execArith ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j) o.pts r)
        (PlaysCoin.repSpine Dig ((planPHJ (Reg := Reg) dj x₀ (L.draw ω) R).get j) o.pts)
        (decJ (planPHJ dj x₀) (L.draw ω) R j) (o.reps j r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := (hrj j).2 r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ sts[j.val] hwf sch hf hdl hS hsch hpl
      o.pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s' (Array.mem_toList_iff.2
        (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig _ o.pts 0) (hrep 0),
    PlaysCoin.playsT_of_repRun (PlaysCoin.repsOf_spine Dig _ o.pts 1) (hrep 1), ?_⟩
  have hopn : opnPHJ I (planPHJ dj x₀) dj (L.draw ω) R o.proofs j =
      opnOf (Setup.ofCircuit sts[j.val]) sch
        (Model.extraClaims (stmtOf sts[j.val] hwf (regionsOf sts[j.val])) PT_LOCAL mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnPHJ
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nXT
    rw [extraClaims_length, extraClaims_length, hS]
  show (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[j.val]!) (pend 0, pend 1) _).accepted = true
  rw [hopn, ← (hrj j).1]
  exact hv

end FlockSoundness.Discharge.PlainHJ
