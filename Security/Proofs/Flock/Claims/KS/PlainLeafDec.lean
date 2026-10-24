import Proofs.Flock.Claims.KS.PlainLive

/-!
# The deployed plain decoders at plain SHA-512 leaves

The scope admits hidden outputs under plain SHA-512 leaves (`flock-leaf/sha512-unsalted`) without `--zk`
(`ProvedScope.leafOk`). Their model leaf is `Refine.leafPlain` (`Refine.leafOfId`), not the salted `leaf512` that
`PlainDrawAcc` and `PlainDrawDec` read, and the decoders there are right at `leafPlain` by the same proofs, with
`table_verdict_plain` and `setupHidden_plain` for `table_verdict_hm96` and `setupHidden_hm96`:

* `accepts_verdict_plainAnyD`, `accepts_opens₀LDU`, `accepts_repsOneDU`: `accepts_verdict_hm96AnyD`,
  `accepts_opens₀LD`, `accepts_repsOneD` at `leafPlain`.
* `TagsOkHU`, `decodesTable_wrPD_scHU`: `TagsOkH` and `decodesTable_wrPD_scH` under plain SHA-512 leaves.
* `plain_setupHiddenU`, `plain_hdecHUAt`: `plain_setupHidden` and `plain_hdecHAt` there.
-/

namespace FlockSoundness.Discharge.HiddenDec

open FlockSoundness.Discharge.Exec (Inputs)

/-- **The verify command's tags under hidden outputs and plain SHA-512 leaves, without `--zk`**: `TagsOkH` with
`flock-leaf/sha512-unsalted` for `hm96-sha512/v1`. -/
def TagsOkHU (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted" ∧
    I.count ≤ 1

end FlockSoundness.Discharge.HiddenDec

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtD)

local notation "𝔽" => FlockLevel3.GF128

section Any

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` under hidden outputs and plain SHA-512 leaves makes each table's verdict accept
at the plain leaf** (`accepts_verdict_hm96AnyD` with `table_verdict_plain`). -/
theorem accepts_verdict_plainAnyD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript}
    (hh : I.tags.hiddenOutputs = true) (hl : I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
        ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ ∧
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ ∧
          (tableOutL H512 enc512 leafPlain (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  have hms : sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases setupOne_tagsAt_leaf (Or.inr ⟨hh, hl⟩) hset with ⟨e, -⟩ | ⟨-, e⟩
    · exact absurd (hl.symm.trans e) (by decide)
    · exact e
  exact table_verdict_plain spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) hms _ sess hext (decode_roundsWF hsess) schs[j] hf j _ _
    c h0 h1

end Any

section Req0

open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **At flock-verify's acceptance at `dr`, each table's level-0 openings at the plain leaf are its reps' deployed
entries** (`accepts_opens₀LD` at `leafPlain`). -/
theorem accepts_opens₀LDU {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leafPlain cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          out.accepted = true →
          (∀ x ∈ opens₀L (schedOf sch) out, VerifiedL H512 enc512 leafPlain cap
            (lvOf (schedOf sch) 0).lanes (Model.pathLen (lvOf (schedOf sch) 0)) x ∧
              x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
          (opens₀L (schedOf sch) out).map (fun x => (x.1, x.2.1)) =
            Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
              Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
  obtain ⟨spec, sts, sess, hs, hsess, hall⟩ := accepts_req0D h
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨sch, hf, hk⟩ := hall j hj
  refine ⟨sch, hf, fun hwf hr hm p₀ p₁ s₀ s₁ pend₀ pend₁ cap out hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ hout hacc => ?_⟩
  subst hout
  obtain ⟨⟨R₀, rest₀, hR₀, hl₀, hpos₀, hrw₀⟩, hsh₀⟩ := hk hwf hr hm 0 (by norm_num) s₀ p₀ pend₀ hs₀ hp₀ hpl₀
  obtain ⟨⟨R₁, rest₁, hR₁, hl₁, hpos₁, hrw₁⟩, hsh₁⟩ := hk hwf hr hm 1 (by norm_num) s₁ p₁ pend₁ hs₁ hp₁ hpl₁
  have h2 := (fast100_dims _ sch hf).1.2.1
  have hseam : LevelSeam (sch.levels.getD 0 default) :=
    fast100_level _ sch hf _ (by unfold Array.getD; rw [dite_eq_left (by omega)]; exact Array.getElem_mem _)
  have hlv0 := lvOf_schedOf (s := sch) (i := 0) (by omega)
  have hQ : (lvOf (schedOf sch) 0).queries = (sch.levels.getD 0 default).q := by rw [hlv0]; rfl
  have hq : ∀ (R : Req) (coins : Array F128),
      R.positions = (Flock.positions (sch.levels.getD 0 default) coins).toList →
      R.positions.length = (sch.levels.getD 0 default).q := fun R coins h => by
    rw [h, Array.length_toList, positions_size _ hseam]
  have hent : (opens₀L (schedOf sch) (tableOutL H512 enc512 leafPlain cap (pend₀, pend₁)
      (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
        sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]))).map
        (fun x => (x.1, x.2.1)) =
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
          (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
            (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
          (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
            (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
    unfold opens₀L
    rw [List.map_append, hQ, Claims.Flock.plainRepColsOf_ok hp₀, Claims.Flock.plainRepColsOf_ok hp₁,
      opnOf_pair hp₀ hp₁]
    congr 1
    · exact Claims.Flock.slotOpensL_entries _ 0 _ _ _ R₀ rest₀ hR₀ hl₀ hpos₀ (hq _ _ hpos₀) hrw₀ fun _ => rfl
    · exact Claims.Flock.slotOpensL_entries _ 1 _ _ _ R₁ rest₁ hR₁ hl₁ hpos₁ (hq _ _ hpos₁) hrw₁ fun _ => rfl
  refine ⟨fun x hx => ⟨?_, ?_⟩, hent⟩
  · refine Claims.Flock.verified_opens₀L H512 enc512 leafPlain (schedOf sch) _ (pend₀, pend₁) _ hacc
      (fun r hr R hR hlR => ?_) x hx
    rcases r with _ | _ | r
    · exact hlR ▸ hsh₀ R hR
    · exact hlR ▸ hsh₁ R hR
    · omega
  · have hm := List.mem_map_of_mem (f := fun x => (x.1, x.2.1)) hx
    rw [hent, Claims.Flock.plainRepColsOf_ok hp₀, Claims.Flock.plainRepColsOf_ok hp₁] at hm
    rcases List.mem_append.1 hm with hm | hm
    · exact Claims.Flock.lt_of_mem_positions hseam (Claims.Flock.fst_mem_positions_of_mem_zkRepCols hm)
    · exact Claims.Flock.lt_of_mem_positions hseam (Claims.Flock.fst_mem_positions_of_mem_zkRepCols hm)

end Req0

section Reps

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` of a one-table session under hidden outputs and plain SHA-512 leaves, rep by
rep**: `accepts_repsOneD` at `accepts_verdict_plainAnyD`. -/
theorem accepts_repsOneDU {I : Inputs} {dr : Option Json} {t : Exec.Transcript} (hrr : I.tags.retainRounds = true)
    (hh : I.tags.hiddenOutputs = true) (hl : I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") (hc1 : I.count ≤ 1)
    (h : AcceptsAtD I dr t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : Session) (sch : Flock.Schedule),
      setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      setupAtD I dr = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
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
        (tableOutL H512 enc512 leafPlain (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true := by
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := accepts_verdict_plainAnyD hh hl h
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := accepts_rep_roundsAnyD hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hsu
  have hpts := decode_points hdec
  rcases setupTables_okAny hsu with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
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

end Reps

end FlockSoundness.Discharge.AtDraw

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at `wrPD` at hidden outputs under plain SHA-512 leaves**:
`decodesTable_wrPD_scH` at `leafPlain`, from `accepts_repsOneDU`, `accepts_opens₀LDU` and `setupHidden_plain`. -/
theorem decodesTable_wrPD_scHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtD I (some (dj S)) (wrPD I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAt I (dj S) C (wrPD I (dj S) o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j := by
  have hj : j.val = 0 := by
    have h := j.isLt
    simp only [List.length_cons, List.length_nil] at h
    omega
  obtain rfl : j = ⟨0, Nat.one_pos⟩ := Fin.ext hj
  rintro o ⟨hacc, hr₀, hno⟩
  change AcceptsAtD I (some (dj S)) (wrPD (J := 1) I (dj S) o) at hacc
  change TxNoOffAt I (dj S) C (wrPD (J := 1) I (dj S) o) at hno
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  have h := PlaysCoin.decodesOfRuns (fun cap pend os => tableOutL H512 enc512 leafPlain cap pend os)
    (fun o' => o' = oneOf o)
    (fun out => out.accepted = true ∧ ¬ (Off₀L H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) out ∧
        ¬ SelfClashL H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch out))
    (opnAtH (tablesAtH dj x₀) I dj S R) (decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => []) capOfRoot
    (fun pts r => repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) pts r)
    (fun pts _ => PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) pts)
    (PlaysCoin.repsOf_spine Dig (tablesAtH (Reg := Reg) dj x₀ S R)) ?_ (oneOf o) rfl
  · obtain ⟨pend, h0, h1, hok⟩ := h
    exact ⟨pend, h0, h1, hok⟩
  rintro o' rfl
  obtain ⟨hprf, hrec⟩ := recordsLive_wrPD1 I (dj S) o hacc
  obtain ⟨hSD, hmPts⟩ := hS
  obtain ⟨st₁, hst₁, hwf, hTS⟩ := tableAtH_of_acceptsD ht hc hh hc1 dj x₀ hacc hSD hmPts
  obtain ⟨st, spec, sess, sch, hst, hsu, hdec, hf, hk⟩ := accepts_repsOneDU hrr hh hl hc1 hacc
  rw [setupOne_hidden hh] at hst
  obtain rfl : st = st₁ := Except.ok.inj (hst.symm.trans hst₁)
  obtain ⟨hroot, hpts, hstreams⟩ := hrec spec #[st] sess hsu hdec
  have hmP : st.mPts = mPts := hmPts st hst
  obtain ⟨-, hr⟩ := setupHidden_wf hst
  have hm : 13 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := setupHidden_plain hl hst
  obtain ⟨pend, hreps, hv⟩ := hk hwf hr hm
  have hn : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL q pts).length = 2 * st.regions.size := by
    intro q pts
    rw [extraClaims_length]
    simp [stmtOf, regionsOf]
  obtain ⟨s₀, p₀, -, hs₀, hp₀, hpl₀, -⟩ := hreps 0
  obtain ⟨s₁, p₁, -, hs₁, hp₁, hpl₁, -⟩ := hreps 1
  have hcap : sess.rootB[0]! = rootB₀[0]! := hroot.trans hr₀
  have hv' := hv
  rw [hn, hcap] at hv'
  obtain ⟨spec', sts', sess', hsu', hdec', hall⟩ := accepts_opens₀LDU hacc
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨sch', hf', hk'⟩ := hall 0 (by simp)
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  have hn' : ∀ (q : ℕ) (pts : Fin 2 → Fin q → 𝔽),
      (Model.extraClaims (stmtOf #[st][0] hwf (regionsOf #[st][0])) PT_LOCAL q pts).length = 2 * st.regions.size :=
    hn
  obtain ⟨hver, hent⟩ := hk' hwf hr hm p₀ p₁ s₀ s₁ (pend 0) (pend 1) (capOfRoot rootB₀[0]!)
    (tableOutL H512 enc512 leafPlain (capOfRoot rootB₀[0]!) (pend 0, pend 1)
      (opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!]))
    hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ (by rw [hn']; rfl) hv'
  rw [hn'] at hent
  rw [hmP, hpts] at hreps
  have hS' : (tablesAtH (Reg := Reg) dj x₀ S R).S = stmtOf st hwf (regionsOf st) := hTS
  have hpl : (tablesAtH (Reg := Reg) dj x₀ S R).ptLocal = PT_LOCAL := tableAtH_ptLocal dj x₀ _
  have hsch : (tablesAtH (Reg := Reg) dj x₀ S R).sch = schedOf sch := by
    have h1 : Accounting.fast100 st.m = some (schedOf sch) := by rw [← fast100_schedOf, hf]; rfl
    have h2 := (tablesAtH (Reg := Reg) dj x₀ S R).hsch
    rw [hS'] at h2
    exact Option.some.inj (h2.symm.trans h1)
  have hrep : ∀ r : Fin 2, PlaysCoin.RepRun (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R)
      (oneOf o).pts r) (PlaysCoin.repSpine Dig (tablesAtH (Reg := Reg) dj x₀ S R) (oneOf o).pts)
      (decAt (tablesAtH dj x₀) wOP wLIG S R) (pick ((oneOf o).rep0, (oneOf o).rep1) r) [] (pend r) := by
    intro r
    obtain ⟨s, p, Tv, hs, hp, hplay, hlen, hret⟩ := hreps r
    obtain ⟨s', hs', hrs⟩ := hstreams r
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    rw [← hrs]
    exact repRun_at op_carries lig_carries decodeProof_sizes _ st hwf sch hf (by rw [hms]; rfl) hS' hsch hpl
      (oneOf o).pts r hm rfl _ p hp s'.rounds (PlaysCoin.roundsWF_mem (decode_roundsWF hdec s'
        (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs')))) _ hplay _ Tv hlen hret
  refine ⟨(pend 0, pend 1), hrep 0, hrep 1, ?_⟩
  have hopn : opnAtH (tablesAtH (Reg := Reg) dj x₀) I dj S R (oneOf o).proofs =
      opnOf (Setup.ofCircuit st) sch (2 * st.regions.size)
        #[(wrPD (J := 1) I (dj S) o).proofs[0]!, (wrPD (J := 1) I (dj S) o).proofs[1]!] := by
    rw [← hprf]
    unfold opnAtH
    rw [hst]
    dsimp only
    rw [hf]
    dsimp only
    congr 1
    unfold nX
    rw [extraClaims_length, hS']
    simp [stmtOf, regionsOf]
  have e₀ : (oneOf o).rootB = rootB₀[0]! := hr₀
  rw [hopn, e₀, hsch]
  refine ⟨hv', fun ⟨hoff, hsc⟩ => hno spec st sess sch s₀ s₁ hsu hdec hf hs₀ hs₁ ?_⟩
  rw [hsch] at hN
  exact Claims.Flock.exists_ne_of_entries H512 enc512 leafPlain (schedOf sch) (capOfRoot rootB₀[0]!) _ hN C _ _ rfl
    hver hent hoff hsc

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A plain statement in scope with hidden outputs and plain SHA-512 leaves sets up its one table at its draw**:
`plain_setupHidden` with `TagsOkHU` for `TagsOkH`. -/
theorem plain_setupHiddenU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) :
    ins.read.I.tags.typed = false ∧ HmOut.parse ins.read.I.tags ins.read.I.circuitFile ins.read.I.tables = .ok st.c ∧
      HiddenDec.TagsOkHU ins.read.I ∧
      Stmt.setupHidden ins.read.I.tags ins.read.I.circuitFile ins.read.I.publicFile ins.read.I.coins
        ins.read.I.tables (some d) ins.read.I.partition ins.read.I.program = .ok st ∧ spec = st.spec ∧ 22 ≤ st.m ∧
      st.m ≤ 35 := by
  have hm₁ := hin.1
  have hm₂ := hin.2.1
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨-, hpr, hsr⟩ := parseBytes_setup hp hset
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by unfold readOf; rw [(setupOf_some hset).1]; rfl)
  have htags : ins.read.tags = ins.read.I.tags := by simp [Readable.Read.tags, hzk]
  have hform : (Readable.toReadable Readable.relation p).form = ins.read.form := by rw [← hpr]; rfl
  obtain ⟨-, -, -, -, -, ht, hrr, -, hc1, hpi, -⟩ := hin
  rw [hform] at ht hrr hc1 hpi
  simp only [Readable.Read.form, htags] at ht hrr hc1 hpi
  simp only [Bool.not_eq_true] at ht
  simp only [Bool.false_eq_true, ↓reduceIte] at hc1
  have hpi' : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 hpi
  have hcoins : ins.read.coins = ins.read.I.coins := by simp [Readable.Read.coins, hzk]
  unfold Readable.Read.setup at hsr
  rw [htags, hcoins, hpi', hd] at hsr
  rcases HiddenExec.setupTables_okH hh hsr with ⟨-, st', hst, hsp, rfl⟩ | ⟨hc, -⟩
  swap
  · omega
  have e : st' = st := by
    unfold tables at h
    rw [hset] at h
    simpa using h
  subst e
  have hc := HiddenExec.setupHidden_parseC hst
  have hm := parse_m hs hset (by simp)
  simp only [Bool.false_eq_true, ↓reduceIte] at hm₁
  refine ⟨ht, hc, ⟨hh, hrr, hl, hc1⟩, hst, hsp, ?_, ?_⟩
  · simpa [hm] using hm₁
  · simpa [hm] using hm₂

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and plain
SHA-512 leaves**: `plain_hdecHAt` at `leafPlain`, from `decodesTable_wrPD_scHU`. -/
theorem plain_hdecHUAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAt ins.read.I d C rootB₀)
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  obtain ⟨ht, hc, hT, hst, rfl, h22, h35⟩ := plain_setupHiddenU hs hin hset h hh hl hd
  obtain ⟨sch, hf⟩ : ∃ sch, Flock.fast100 st.m = some sch := by
    have e := Refine.fast100_schedOf st.m
    rw [hsch] at e
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at e; cases e
    | some sch => exact ⟨sch, rfl⟩
  obtain ⟨pop, -, hdr⟩ := HiddenExec.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  obtain ⟨x₀, hx₀⟩ := HiddenExec.drawSetupH_of_setupHiddenI (S := (∅ : Finset (Fin 0))) (mPts := st.spec.mPts) ht hc
    hst hf hD (fun u hu => absurd hu (Finset.notMem_empty u)) rfl
  have e : HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ =
      fun _ _ => plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay := by
    funext S R
    obtain rfl : S = ∅ := Finset.eq_empty_of_isEmpty S
    exact tableAtH_plainTab ht (fun _ : Finset (Fin 0) => d) (spec := st.spec) (S := ∅) x₀ x₀ hx₀ L m k0 l₀ l₁
      rest ext hsch hm hlay
  have hS : HiddenDec.DrawOkH ins.read.I (fun _ : Finset (Fin 0) => d) st.spec.mPts ∅ :=
    ⟨fun _ _ u hu => absurd hu (Finset.notMem_empty u), fun st' hst' => by
      obtain rfl := Except.ok.inj (hst.symm.trans hst')
      rfl⟩
  have hN : wordLen false x =
      2 ^ (lvOf (HiddenExec.tablesAtH (Reg := Unit) (fun _ : Finset (Fin 0) => d) x₀ ∅ ()).sch 0).logLen := by
    rw [e]
    exact plain_wordLen h h22 h35 hsch
  have hdec := decodesTable_wrPD_scHU ht hc (fun _ : Finset (Fin 0) => d) x₀ (Reg := Unit) hT hS () hN C rootB₀
  rw [e] at hdec
  exact hdec

end FlockDeploy.KS
