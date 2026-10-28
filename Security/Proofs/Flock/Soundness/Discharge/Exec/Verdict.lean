import Proofs.Flock.Soundness.Discharge.Exec.Decode
import Proofs.Flock.Soundness.Discharge.Exec.Live
import Proofs.Flock.Soundness.Discharge.Exec.Merkle

/-!
# The salted table's verdict at flock-verify's acceptance (the verdict part of `Decodes`)

`Decodes` asks two things of an accepted live outcome: each rep's game runs on the tape decoded from its stream and
proof file, and the salted table's verdict `tableOutL`, at the cap `capOf` reads off `root_B` and the openings and salts
`opn` reads off the proof files, accepts on those runs. This file fixes `opn` and proves the verdict part on the reps'
runs that `verify_refines_hm96` gives, which take the proof's messages and the stream's coins as two lists:

* `opnOf`: the openings and salts the two proof files decode to (`opensOf`, `saltsOf`); `opnOf_pair`.
* `table_verdict_hm96`: when both reps of table `i` verify with one cap under `hm96-sha512/v1`, their runs end in pends
  on which `tableOutL` at `capOfRoot root_B[i]` and `opnOf` accepts.
* `accepts_verdict_hm96`: the same at flock-verify's acceptance, for every table of the session, under tags whose leaf
  scheme is `hm96-sha512/v1`.

What remains of `Decodes` at an accepted outcome is that those runs are runs on the decoded tapes (`repTape`).
`Plays.toT` turns each into a run on one tape with the proof's messages and the stream's coins; the tape alignment is
that the messages between coins `j − 1` and `j` are the ones `dec j` reads off round `j`'s retained bytes, which are the
encodings of chunk `j` of the rep's frames (`accepts_rep_rounds`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model Game

local notation "𝔽" => FlockLevel3.GF128

/-- **The openings and salts the proof files decode to**: rep `r`'s proof file `bs[r]`, decoded at the table's shape
(the default proof when it does not decode), read as the game's openings (`opensOf`) and salts (`saltsOf`). -/
def opnOf (st : Setup) (sch : Flock.Schedule) (n : ℕ) (bs : Array ByteArray) : Opens 𝔽 Dig × Salts (List UInt8) :=
  let p : ℕ → Proof := fun r => ((decodeProof (shapeOf st sch n) bs[r]!).toOption).getD default
  (opensOf (p 0) (p 1) sch, saltsOf (p 0) (p 1) sch)

theorem opnOf_pair {st : Setup} {sch : Flock.Schedule} {n : ℕ} {b₀ b₁ : ByteArray} {p₀ p₁ : Proof}
    (h₀ : decodeProof (shapeOf st sch n) b₀ = .ok p₀) (h₁ : decodeProof (shapeOf st sch n) b₁ = .ok p₁) :
    opnOf st sch n #[b₀, b₁] = (opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch) := by
  simp [opnOf, h₀, h₁, Except.toOption]

/-- **One table's salted verdict accepts, under `hm96-sha512/v1`** (§5, §9–§14, S16). When both reps of table `i`
verify with one cap, each rep's compiled game, played with its proof's messages and its stream's coins, ends in a pend,
and the salted table's verdict on those pends, at the cap `capOfRoot` reads off the table's `root_B` and the openings
and salts `opnOf` reads off the two proof files, accepts. -/
theorem table_verdict_hm96 (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (extra : List (Model.Claim 𝔽 S.m)) (sess : RunRecord)
    (hext : ExtraRealizes st S sess.points extra) (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c) :
    ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      decodeProof (shapeOf st sch extra.length) b₀ = .ok p₀ ∧
      decodeProof (shapeOf st sch extra.length) b₁ = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[i]!) (pend₀, pend₁)
        (opnOf st sch extra.length #[b₀, b₁])).accepted = true := by
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    table_refines_hm96 spec st S hm hmS hkS hfold hms extra sess hext hwf sch hf i b₀ b₁ c h0 h1
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨-, p₀', -, hp₀', hc₀, hroot, -, hw₀, -⟩ := rep_refines (D := Dig) spec st S hm h15 sch hmS hkS hfold
    extra sess hext hdim hlv hα1 (hsm.trans hmS) hwf i 0 _ _ h0
  obtain rfl : p₀' = p₀ := Except.ok.inj (hp₀'.symm.trans hp₀)
  rw [hms] at hw₀
  have hcap : capDig p₀'.cap = capOfRoot sess.rootB[i]! :=
    (capOfRoot_eq hw₀ (hc₀.symm.trans (bytes_eq_of_beq hroot))).symm
  refine ⟨p₀', p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, ?_⟩
  rw [opnOf_pair hp₀ hp₁, ← hcap]
  simp only [tableOutL, hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
  exact ⟨hok₀, hok₁⟩

/-- **One table's verdict accepts, under plain SHA-512 leaves** (`flock-leaf/sha512-unsalted`): `table_verdict_hm96`
with `table_refines_plain`, the verdict at `leafPlain`. -/
theorem table_verdict_plain (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (extra : List (Model.Claim 𝔽 S.m))
    (sess : RunRecord) (hext : ExtraRealizes st S sess.points extra) (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c) :
    ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      decodeProof (shapeOf st sch extra.length) b₀ = .ok p₀ ∧
      decodeProof (shapeOf st sch extra.length) b₁ = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      (tableOutL H512 enc512 leafPlain (capOfRoot sess.rootB[i]!) (pend₀, pend₁)
        (opnOf st sch extra.length #[b₀, b₁])).accepted = true := by
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    table_refines_plain spec st S hm hmS hkS hfold hms extra sess hext hwf sch hf i b₀ b₁ c h0 h1
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨-, p₀', -, hp₀', hc₀, hroot, -, hw₀, -⟩ := rep_refines (D := Dig) spec st S hm h15 sch hmS hkS hfold
    extra sess hext hdim hlv hα1 (hsm.trans hmS) hwf i 0 _ _ h0
  obtain rfl : p₀' = p₀ := Except.ok.inj (hp₀'.symm.trans hp₀)
  rw [hms] at hw₀
  have hcap : capDig p₀'.cap = capOfRoot sess.rootB[i]! :=
    (capOfRoot_eq hw₀ (hc₀.symm.trans (bytes_eq_of_beq hroot))).symm
  refine ⟨p₀', p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, ?_⟩
  rw [opnOf_pair hp₀ hp₁, ← hcap]
  simp only [tableOutL, hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
  exact ⟨hok₀, hok₁⟩

/-- **One table's verdict accepts at the model's leaf of its scheme** (`leafOfId`), under `hm96-sha512/v1` or plain
SHA-512 leaves: `table_verdict_hm96` or `table_verdict_plain`. -/
theorem table_verdict_leaf (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) {id : String}
    (hms : (id = "hm96-sha512/v1" ∧ st.merkle = MerkleScheme.hm96Sha512) ∨
      (id = "flock-leaf/sha512-unsalted" ∧ st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted"))
    (extra : List (Model.Claim 𝔽 S.m))
    (sess : RunRecord) (hext : ExtraRealizes st S sess.points extra) (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c) :
    ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      decodeProof (shapeOf st sch extra.length) b₀ = .ok p₀ ∧
      decodeProof (shapeOf st sch extra.length) b₁ = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      (tableOutL H512 enc512 (leafOfId id) (capOfRoot sess.rootB[i]!) (pend₀, pend₁)
        (opnOf st sch extra.length #[b₀, b₁])).accepted = true := by
  rcases hms with ⟨rfl, hms⟩ | ⟨rfl, hms⟩
  · rw [leafOfId_hm96]
    exact table_verdict_hm96 spec st S hm hmS hkS hfold hms extra sess hext hwf sch hf i b₀ b₁ c h0 h1
  · rw [leafOfId_plain]
    exact table_verdict_plain spec st S hm hmS hkS hfold hms extra sess hext hwf sch hf i b₀ b₁ c h0 h1

/-- **flock-verify's acceptance makes each table's salted verdict accept** (§5, §9–§14, S16). When `Accepts I t` under
tags whose leaf scheme is `hm96-sha512/v1`, the record decodes against the session's spec, and for every table `j`, at
its `fast100` schedule and at the statement `stmtOf sts[j]` read off its layout and the record's link points, each
rep's compiled game, played with its proof file's messages and its stream's coins, ends in a pend, and the salted
table's verdict on those pends, at `capOfRoot` of the table's `root_B` and at `opnOf` of its two proof files, accepts. -/
theorem accepts_verdict_hm96 {I : Inputs} {t : Transcript} (hl : I.tags.merkleLeaf = "hm96-sha512/v1")
    (hh : I.tags.hiddenOutputs = false) (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupOf I t.record = .ok (spec, sts) ∧
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
          (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verify_ok_tables hv
  obtain ⟨-, hsetup⟩ := setupTables_setupH hh hs
  obtain ⟨hp2, hmp⟩ := setupTables_points hh hs
  have hpts := decode_points hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_verdict_hm96 spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) (setupH_tagsAt_hm96 hl hset) _ sess hext (decode_roundsWF hsess) schs[j] hf j _ _ c
    h0 h1

end FlockSoundness.Discharge.Exec
