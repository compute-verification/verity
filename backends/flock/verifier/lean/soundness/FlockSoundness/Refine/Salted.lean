import FlockSoundness.Refine.Regions
import FlockSoundness.Protocol.Refine.Salted

/-!
# The salted scheme `hm96-sha512/v1` refines the model's salted leaves (R7 and R8 for hm96; PROTOCOL.md §14)

`MerkleScheme.hm96Sha512` hashes an opened row's leaf as `SHA-512(leaf_prefix ‖ b ‖ c)`, with `x = SHA-512(row)`,
`b = x ⊕ M·y` and `c = SHA-512(salt_prefix ‖ y)` for the opening's 192-byte salt `y` (`Hm96.Default512`). Its nodes are
SHA-512 of the pair, as the unsalted scheme's are. The compiled model's `Merkle.Leaf.hm96 K` is that leaf for a key `K`.
`K512` is the executable's own key: its prefixes and its mask are `Hm96.Default512`'s, so the two leaves are equal
(`hm96_leaf`).

* **R7:** a passed hm96 Merkle check is the model's salted path check (`merkleCheck_verifiesL`), and a rep whose requests
  passed it is `OpensOKL` at the proof's rows and salts (`opensOKL_of`).
* **R8:** `verify_refines_hm96` is `verify_refines` for the salted scheme, with `OpensOKL` at the proofs' salts.
* **R9:** `verify_refines_ofCircuit_hm96`, for the verifier's own statements. For `Stmt.setupH`'s, `setupH_wf` gives its
  layout and region hypotheses.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

/-! ## The executable's key -/

/-! ## A passed hm96 check is the model's salted path check -/

/-! ## The proof's openings, with their salts, pass the model's salted path checks -/

/-! ## The verifier refines the model's salted table -/

/-- **The verifier refines the model's table, for `hm96-sha512/v1`** (§3, §5, §9–§14). `verify_refines` for the salted
scheme: when `Flock.verify` accepts a one-table session record and its two proof files under `hm96Sha512`, on a statement
whose `m`, `k_log`, circuit fold and extra claims are the model's `S` and `extra`, the record and both proofs decode.
Then at the proofs' openings, each rep's compiled game, played with its proof's messages and every coin of its stream,
accepts, and both reps' paths verify from their rows' salted leaves, under the proofs' salts, against the one level-0
cap. -/
theorem verify_refines_hm96 (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (extra : Session → List (Model.Claim 𝔽 S.m))
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess → ExtraRealizes st S sess.points (extra sess))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) := by
  obtain ⟨sess, sch, c, hsess, hf, h0, h1⟩ := verify_one_ok h
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  have hwf := decode_roundsWF hsess
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hschm : sch.m = S.m := hsm.trans hmS
  obtain ⟨s₀, p₀, hs₀, hp₀, hc₀, -, hsz₀, hw₀, hrest₀⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 0 _ _ h0
  obtain ⟨s₁, p₁, hs₁, hp₁, hc₁, -, hsz₁, hw₁, hrest₁⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 1 _ _ h1
  obtain ⟨pend₀, hpl₀, hcaps₀, hreqs₀, hver₀⟩ := hrest₀ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  obtain ⟨pend₁, hpl₁, hcaps₁, hreqs₁, hver₁⟩ := hrest₁ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  rw [hms] at hreqs₀ hreqs₁ hw₀ hw₁
  have hok₀ := opensOKL_of sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) hcaps₀ hreqs₀
    (fun ℓ j => rfl) (fun ℓ j => rfl)
  have hok₁ := opensOKL_of sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) hcaps₁ hreqs₁
    (fun ℓ j => rfl) (fun ℓ j => rfl)
  have hcapeq : p₁.cap = p₀.cap := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans hc₀)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  rw [hcapeq] at hok₁
  exact ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀,
    hok₁⟩

open Game Model in
/-- **The verifier refines the model's reps, for the statements it is set up with, under `hm96-sha512/v1`** (§3, §5, §10,
§14, §16). `verify_refines_ofCircuit` for the salted scheme: when `Flock.verify` accepts a one-table session under
`hm96Sha512`, with the setup `Setup.ofCircuit st` of a statement whose layout and regions are well formed, the conclusion
of `verify_refines_hm96` holds for the model statement `stmtOf st` read off the layout, with the extra claims the model's
`extraClaims` at the record's link points. -/
theorem verify_refines_ofCircuit_hm96 (st : Flock.Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (hm : 13 ≤ st.m)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (record : String) (proofs : Array ByteArray)
    (h : Flock.verify st.spec #[Setup.ofCircuit st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) :=
  verify_refines_hm96 st.spec (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) hm rfl rfl (ofCircuit_fold st hwf _)
    hms (fun sess => Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points))
    record proofs
    (fun sess hsess => ofCircuit_extra st hwf hr sess.points (decode_points hsess).1 (decode_points hsess).2) h

end FlockSoundness.Refine
