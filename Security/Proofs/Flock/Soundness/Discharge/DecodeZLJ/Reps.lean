import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Statements

/-!
# flock-verify's acceptance at table `j`, rep by rep

`accepts_reps_at`: `DecodeZL.accepts_reps` at any table `j` of the session, at any `count`. e2e-exec's
`accepts_verdict_hm96` and `accepts_rep_rounds` already speak of every table `j < sts.size`: table `j`'s streams
`spec.tables[j]/rep r`, its proof files `2j + r`, its root `root_B[j]` and the session's shared link points.

`planJ_get`, `tableJ_of`: table `j` of `planJ` is `tableJ` at part `j`, and wherever a `SetupJ` exists it is the
`SetupJ.spec` of one with the same statement.
-/

namespace FlockSoundness.Discharge.DecodeZLJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify's acceptance, table by table and rep by rep**: the record's spec, tables and session, and at each
table `j` its `fast100` schedule and, at the table's link claims, each rep's stream and decoded proof (file `2j + r`),
its run on the stream's coins with the proof's messages, its stream's rounds retaining `repFrames`' chunks, and the
salted verdict on the two runs at `capOfRoot` of `root_B[j]` and `opnOf` of proof files `2j`, `2j + 1`. -/
theorem accepts_reps_at {I : Inputs} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = false)
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
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := accepts_verdict_hm96 hl hh h
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := accepts_rep_rounds hh hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := setupTables_points hh hsu
  have hpts := decode_points hdec
  refine ⟨spec, sts, sess, hsu, hdec, fun j hj => ?_⟩
  obtain ⟨sch, hf, hk⟩ := hall j hj
  obtain ⟨sch', hf', hR0, hR1⟩ := hallR j hj
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  refine ⟨sch, hf, fun hwf hr hm => ?_⟩
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv⟩ := hk hwf hr hm
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

variable {n mPts : ℕ} {I : Inputs} (ht : I.tags.typed = false) {Reg : Type}

theorem planJ_get (T₀ : TabSpec mPts) (dj : Finset (Fin n) → Json) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planJ ht T₀ dj S R).length) : (planJ ht T₀ dj S R).get j = tableJ ht T₀ (dj S) j.val := by
  exact List.get_ofFn (fun j : Fin I.count => tableJ ht T₀ (dj S) j.val) j

theorem planJ_length (T₀ : TabSpec mPts) (dj : Finset (Fin n) → Json) (S : Finset (Fin n)) (R : Reg) :
    (planJ ht T₀ dj S R).length = I.count :=
  List.length_ofFn

/-- **Table `j` where flock-verify sets it up**: `tableJ` at a draw with a `SetupJ` is the `SetupJ.spec` of one with
the same statement. -/
theorem tableJ_of {T₀ : TabSpec mPts} {d : Json} {j : ℕ} (x : SetupJ I mPts j (partJson d I.count j)) :
    ∃ y : SetupJ I mPts j (partJson d I.count j), y.st = x.st ∧ tableJ ht T₀ d j = y.spec ht := by
  refine ⟨Classical.choice ⟨x⟩, Except.ok.inj ((Classical.choice ⟨x⟩).h.symm.trans x.h), ?_⟩
  unfold tableJ
  rw [dite_eq_left_of_eq_true (eq_true ⟨x⟩)]

end FlockSoundness.Discharge.DecodeZLJ
