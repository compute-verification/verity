import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Protocol.Discharge.Exec.Tables

/-!
# flock-verify refines each table of a session, under `hm96-sha512/v1` (PROTOCOL.md §5, §9–§14, §16.12)

`verify_refines_hm96` peels a one-table `Flock.verify`. A session of `J` tables runs the same checks on each:

* `verify_ok_tables`: when `Flock.verify spec tabs record proofs` accepts, the record decodes, there is one statement
  per table, and each table `i` has its `fast100` schedule and both reps' checks accept with one cap, on the proof
  files `proofs[2i]` and `proofs[2i+1]`.
* `table_refines_hm96`: one table's accepted reps refine the compiled reps as Lean defines them, with salted openings
  (`verify_refines_hm96` at any table index).
* `table_tableAfterL`: the same as one run of the salted session's table game `tableAfterL` (`SessionL.lean`), played
  on the two streams' coins, which accepts.
-/

namespace FlockSoundness.Discharge.Exec

open Flock FlockLevel3 Refine Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **One table's accepted reps are one accepting run of the salted table game** (§5): `table_refines_hm96`, with the
statement's `extraClaims` at link points `pts`, as one run of `tableAfterL` at `Refine.leaf512`, played on the two streams'
coins, rep 0's then rep 1's. Its openings and salts are the proofs'. -/
theorem table_tableAfterL (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) (sess : Session)
    (hext : ExtraRealizes st S sess.points (Model.extraClaims S ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c) :
    ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutCL 𝔽 Dig (List UInt8)),
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts pts).length) b₀ = .ok p₀ ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts pts).length) b₁ = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (tableAfterL execArith H512 enc512 leaf512 S (schedOf sch) hm ptLocal mPts (capDig p₀.cap) pts) ms
        (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
      out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch ∧ out.salts = saltsOf p₀ p₁ sch := by
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    table_refines_hm96 spec st S hm hmS hkS hfold hms (Model.extraClaims S ptLocal mPts pts) sess hext hwf sch hf i
      b₀ b₁ c h0 h1
  obtain ⟨ms, hint⟩ := Plays.interleave_first hpl₀ hpl₁
  refine ⟨p₀, p₁, s₀, s₁, ms ++ [⟨Opens 𝔽 Dig × Salts (List UInt8), (opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch)⟩],
    ⟨pend₀.verdict (opensOf p₀ p₁ sch) && pend₁.verdict (opensOf p₀ p₁ sch) &&
      decide (OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
        OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch)),
      capDig p₀.cap, (pend₀, pend₁), opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch⟩, hp₀, hp₁, hs₀, hs₁, ?_, ?_, rfl, rfl⟩
  · unfold tableAfterL
    exact Plays.bind' (f := fun pend : PendL 𝔽 Dig × PendL 𝔽 Dig =>
        Game.send (Opens 𝔽 Dig × Salts (List UInt8)) fun os =>
          Game.ret (⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
            decide (OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend.1 os.1 os.2 ∧
              OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend.2 os.1 os.2),
            capDig p₀.cap, pend, os.1, os.2⟩ : OutCL 𝔽 Dig (List UInt8))) hint (Plays.send _ (Plays.ret _)) rfl
      (by simp)
  · simp only [hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
    exact ⟨hok₀, hok₁⟩

/-- **flock-verify refines each table of the session it accepts, under `hm96-sha512/v1`** (§5, §9–§14, §16.12). When
`Accepts I t`, the record decodes against the session's spec, and each table `j` is `setupH`'s statement. For a table
whose leaves are `hm96-sha512/v1`, its schedule is `fast100`'s and, at the statement `stmtOf sts[j]` read off its
layout and the record's link points, one run of the salted table game `tableAfterL` on the table's two streams' coins,
with the proof files' openings and salts, accepts. -/
theorem accepts_tables_hm96 {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false) (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size),
        (∃ d, Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)) ∧
        ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧ 22 ≤ sts[j].m ∧
          ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m)
            (hms : sts[j].merkle = MerkleScheme.hm96Sha512),
          ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutCL 𝔽 Dig (List UInt8)),
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
            Plays fOf (tableAfterL execArith H512 enc512 leaf512 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
                PT_LOCAL sts[j].mPts (capDig p₀.cap) (ptsOf sts[j].mPts sess.points)) ms
              (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
            out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch ∧ out.salts = saltsOf p₀ p₁ sch := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, hpos, hschsz, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupH hh hs
  obtain ⟨hp2, hmp⟩ := setupTables_points hh hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz hpos hschsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, hpos, fun j hj => ⟨hsetup j hj, ?_⟩⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  refine ⟨schs[j], hf, exec_fast100_ge hf, fun hwf hr hm hms => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_tableAfterL spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) hms PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points) sess hext
    (decode_roundsWF hsess) schs[j] hf j _ _ c h0 h1

end FlockSoundness.Discharge.Exec
