import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# The decoder is right at the headline's draws

`decodesOn_exec`: `DecodesOn` at the plain headline's tables, from `decodesOnOfRuns`. At an outcome flock-verify
accepts:

* `accepts_reps`: the one table's statement `st` (setupH at the record's draw), and for each rep its stream, its decoded
  proof, its run on the stream's coins with the proof's messages (`accepts_verdict_hm96`), the stream's rounds retaining
  `repFrames`' chunks (`accepts_rep_rounds`), and the salted verdict on the two runs.
* `tableAt_of_accepts` puts `stmtOf st` at the headline's table, `tableAt_ptLocal` its link layout at
  `PT_LOCAL`, and `fast100_schedOf` its schedule at `fast100`'s.
* `repRun_at` (`Runs.lean`) makes each rep's run a `RepRun` for `decAt`; `RecordsLive` says the outcome's rounds, link
  points, root and proof files are the record's.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Integrate

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify's acceptance of a one-table session, rep by rep**: setupH's statement `st` at the record's draw,
the record's spec and session, the table's `fast100` schedule, and at `st`'s link claims: each rep's stream and decoded
proof, its run on the stream's coins with the proof's messages, its stream's rounds retaining `repFrames`' chunks, and
the salted verdict on the two runs at `capOfRoot` of the table's root and `opnOf` of the two proof files. -/
theorem accepts_reps {I : Inputs} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = false)
    (hrr : I.tags.retainRounds = true) (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (hc1 : I.count ≤ 1)
    (h : Accepts I t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : RunRecord) (sch : Flock.Schedule),
      Flock.Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (recordDraw t.record) I.partition I.program =
        .ok st ∧ setupOf I t.record = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
      Flock.fast100 st.m = some sch ∧
      ∀ (hwf : StmtWF st) (_hr : RegionsWF st) (hm : 13 ≤ st.m), ∃ pend : Fin 2 → PendL 𝔽 Dig,
        (∀ r : Fin 2, ∃ (s : StreamRec) (p : Proof) (Tv : F128),
          sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
            some s ∧
          decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
            st.mPts (ptsOf st.mPts sess.points)).length) t.proofs[r.val]! = .ok p ∧
          Plays fOf (Model.repC execArith r.val (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
              (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
            (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) (pend r) ∧
          (chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r.val).toUTF8
            (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
              (ptsOf st.mPts sess.points)).length) p Tv)).1.length = s.rounds.size ∧
          ∀ k < s.rounds.size, Retains (s.rounds[k]!) (encs ((chunks (repFrames (Setup.ofCircuit st) sch
            (spec.domain spec.tables[0]! r.val).toUTF8 (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p Tv)).1.getD k []))) ∧
        (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true := by
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := accepts_verdict_hm96 hl hh h
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := accepts_rep_rounds hh hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := setupTables_points hh hsu
  have hpts := decode_points hdec
  rcases setupTables_ok hh hsu with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  swap
  · omega
  have h0 : 0 < (#[st] : Array Flock.Stmt).size := by simp
  obtain ⟨sch, hf, hk⟩ := hall 0 h0
  obtain ⟨sch', hf', hR0, hR1⟩ := hallR 0 h0
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  refine ⟨st, st.spec, sess, sch, hst, hsu, hdec, hf, fun hwf hr hm => ?_⟩
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv⟩ := hk hwf hr hm
  have hext := ofCircuit_extra st hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp 0 h0).symm)
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

/-- The headline's tables put the link claims at `PT_LOCAL`, at a draw that sets up and at a refused one. -/
theorem tableAt_ptLocal {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) (S : Finset (Fin n)) :
    (tableAt ht dj x₀ S).T.ptLocal = PT_LOCAL := by
  rcases tableAt_cases ht dj x₀ S with ⟨x, e⟩ | e <;> rw [e] <;> rfl

theorem decodesOn_exec : DecodesOnExec := by
  intro c n I ht hc ho mPts dj S₀ x₀ Reg L wr wOP wLIG hOP hLIG hT hwr hdraw
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  refine PlaysCoin.decodesOnOfRuns H512 enc512 leaf512 (tablesAt ht dj x₀) I wr (opnAt (tablesAt ht dj x₀) I dj)
    (decAt (tablesAt ht dj x₀) wOP wLIG) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L ?_
  intro ω R o hacc
  obtain ⟨hd, hprf, hrec⟩ := hwr _ R o hacc
  obtain ⟨hSD, hmPts⟩ := hdraw ω
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAt_of_accepts ht hc ho hh hc1 dj x₀ hacc hd hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_reps hh hrr hl hc1 hacc
  rw [hd] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupH_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.hm96Sha512 := setupH_hm96 hl hst
  rw [hmP, hpts] at hk
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hS : (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R).ptLocal = PT_LOCAL := tableAt_ptLocal ht dj x₀ _
  have hsch : (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R).hsch
    rw [hS] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R)
      o.pts r) (PlaysCoin.repSpine Dig (tablesAt (Reg := Reg) ht dj x₀ (L.draw ω) R) o.pts)
      (decAt (tablesAt ht dj x₀) wOP wLIG (L.draw ω) R) (pick (o.rep0, o.rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at hOP hLIG decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS hsch hpl o.pts r hm rfl _ p hp
      s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s' (Array.mem_toList_iff.2
        (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAt (tablesAt (Reg := Reg) ht dj x₀) I dj (L.draw ω) R o.proofs =
      opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[0]!, (wr (L.draw ω) R o).proofs[1]!] := by
    rw [← hprf]
    unfold opnAt
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, extraClaims_length, hS]
  show (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB) (pend 0, pend 1) _).accepted = true
  rw [hopn, ← hroot]
  exact hv

end FlockSoundness.Discharge.DecodeZL
