import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# The decoder is right at the hidden-output headline's draws

`decodesOn_execH`: DecodeZL's `decodesOn_exec` at the hidden tables, from plays-coin's `decodesOnOfRuns`. At an
outcome flock-verify accepts under hidden outputs:

* `accepts_repsH`: DecodeZL's `accepts_reps` under hidden outputs. The one table's statement `st` is setupHidden's at
  the record's draw, and the per-rep facts come from the acceptance lemmas without `hh` (`accepts_verdict_leafAny`,
  `accepts_rep_roundsAny`, `setupTables_okAny`, `setupTables_pointsAny`, `Discharge/HiddenExec`), the verdict at the
  model's leaf of the tags' leaf scheme (`leafOfId`: `leaf512` under `hm96-sha512/v1`, `leafPlain` under plain SHA-512
  leaves).
* `tableAtH_of_acceptsU` puts `stmtOf st` at the hidden table, from the session alone (so `decodesOn_execHSt` holds at
  the statement circuit, `HmOut.parseStmt`), `tableAtH_ptLocal` its link layout at `PT_LOCAL`, and
  `fast100_schedOf` its schedule at `fast100`'s.
* DecodeZL's `repRun_at`, at its wires' chunk lemmas `op_carries` and `lig_carries` (and `zl_carries`, which `repRun_at`
  reads), makes each rep's run a `RepRun` for `decAt`; `RecordsLive` says the outcome's rounds, link points, root and
  proof files are the record's.
-/

namespace FlockSoundness.Discharge.HiddenDec

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.HiddenExec

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify's acceptance of a one-table session under hidden outputs, rep by rep**: DecodeZL's `accepts_reps`
with setupHidden's statement `st` at the record's draw, the verdict at the model's leaf of the tags' leaf scheme. -/
theorem accepts_repsH {I : Inputs} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = true)
    (hrr : I.tags.retainRounds = true)
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1" ∨ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") (hc1 : I.count ≤ 1)
    (h : Accepts I t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : RunRecord) (sch : Flock.Schedule),
      Flock.Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ setupOf I t.record = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
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
        (tableOutL H512 enc512 (leafOfId I.tags.merkleLeaf) (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true := by
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := accepts_verdict_leafAny (hl.imp id fun hl => ⟨hh, hl⟩) h
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := accepts_rep_roundsAny hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hsu
  have hpts := decode_points hdec
  rcases setupTables_okAny hsu with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  swap
  · omega
  rw [setupOne_hidden hh] at hst
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

/-- The hidden tables put the link claims at `PT_LOCAL`, at a draw that sets up and at a refused one. -/
theorem tableAtH_ptLocal {n mPts : ℕ} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (S : Finset (Fin n)) : (tableAtH dj x₀ S).T.ptLocal = PT_LOCAL := by
  rcases tableAtH_cases dj x₀ S with ⟨x, e⟩ | e <;> rw [e] <;> rfl

/-- **The decoder is right at the hidden-output headline's draws**, from the session alone (`DecodesOnExecH` with no
hypothesis on the circuit: `tableAtH_of_acceptsU`). -/
theorem decodesOn_execHU {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (hT : TagsOkHL I) (hwr : RecordsLive I dj wr) (hdraw : ∀ ω, DrawOkH I dj mPts (L.draw ω)) :
    DecodesOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH (Reg := Reg) dj x₀) I wr
      (opnAtH (tablesAtH dj x₀) I dj)
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L := by
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  refine PlaysCoin.decodesOnOfRuns H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr
    (opnAtH (tablesAtH dj x₀) I dj)
    (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L ?_
  intro ω R o hacc
  obtain ⟨hd, hprf, hrec⟩ := hwr _ R o hacc
  obtain ⟨hSD, hmPts⟩ := hdraw ω
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsU ht hh hc1 dj x₀ hacc hd hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsH hh hrr hl hc1 hacc
  rw [hd] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hdl : st.merkle.digestLen = 64 := by
    rcases hl with hl | hl
    · rw [setupHidden_hm96 hl hst]; rfl
    · rw [setupHidden_plain hl hst]; rfl
  rw [hmP, hpts] at hk
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hS : (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R).ptLocal = PT_LOCAL := tableAtH_ptLocal dj x₀ _
  have hsch : (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R).hsch
    rw [hS] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R)
      o.pts r) (PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ (L.draw ω) R) o.pts)
      (decAt (tablesAtH dj x₀) wOP wLIG (L.draw ω) R) (pick (o.rep0, o.rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf hdl hS hsch hpl o.pts r
      hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s' (Array.mem_toList_iff.2
        (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj (L.draw ω) R o.proofs =
      opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[0]!, (wr (L.draw ω) R o).proofs[1]!] := by
    rw [← hprf]
    unfold opnAtH
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [DecodeZL.extraClaims_length, DecodeZL.extraClaims_length, hS]
  show (tableOutL H512 enc512 (leafOfId I.tags.merkleLeaf) (capOfRoot o.rootB) (pend 0, pend 1) _).accepted = true
  rw [hopn, ← hroot]
  exact hv

set_option linter.unusedVariables false in
/-- **The decoder is right at the hidden-output headline's draws**, wherever the outputs are returned
(`DecodesOnExecH` without `c.outNet = c.unitNet`). -/
theorem decodesOn_execHI {c : Flock.Circuit} {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (hT : TagsOkHL I) (hwr : RecordsLive I dj wr) (hdraw : ∀ ω, DrawOkH I dj mPts (L.draw ω)) :
    DecodesOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH (Reg := Reg) dj x₀) I wr
      (opnAtH (tablesAtH dj x₀) I dj)
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L :=
  decodesOn_execHU ht dj x₀ L wr hT hwr hdraw

set_option linter.unusedVariables false in
/-- `decodesOn_execHI` at the statement circuit (`HmOut.parseStmt`, a templated file's template unrolled). -/
theorem decodesOn_execHISt {c : Flock.Circuit} {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (L : Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (hT : TagsOkHL I) (hwr : RecordsLive I dj wr) (hdraw : ∀ ω, DrawOkH I dj mPts (L.draw ω)) :
    DecodesOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH (Reg := Reg) dj x₀) I wr
      (opnAtH (tablesAtH dj x₀) I dj)
      (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) L :=
  decodesOn_execHU ht dj x₀ L wr hT hwr hdraw

theorem decodesOn_execH : DecodesOnExecH := by
  intro c n I ht _ _ mPts dj S₀ x₀ Reg L wr hT hwr hdraw
  exact decodesOn_execHU ht dj x₀ L wr hT hwr hdraw

theorem decodesOn_execHSt : DecodesOnExecHSt := by
  intro c n I ht _ _ mPts dj S₀ x₀ Reg L wr hT hwr hdraw
  exact decodesOn_execHU ht dj x₀ L wr hT hwr hdraw

end FlockSoundness.Discharge.HiddenDec
