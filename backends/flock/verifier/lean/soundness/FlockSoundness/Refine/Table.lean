import FlockSoundness.Refine.Paths
import FlockSoundness.Refine.Seam
import FlockSoundness.Session
import FlockSoundness.Protocol.Refine.Table

/-!
# The verifier refines the model's table (PROTOCOL.md §3, §5)

`verify_refines`: when the executable's `Flock.verify` accepts a session record and its two proof files, under the
unsalted SHA-512 scheme, the record and both proofs decode. Then at the proofs' openings (`opensOf`), each rep's
compiled game `repC`, played with its proof's messages and its stream's coins, accepts. Both reps' paths also verify
against the one level-0 cap. That is `tableAfter`'s verdict. `verify_tableAfter` states it as an accepting run of
`tableAfter`, with existential messages and rep 0 first.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

/-- **The verifier refines the model's table** (§3, §5, §9–§14). When `Flock.verify` accepts a one-table session record
and its two proof files under the unsalted SHA-512 scheme, on a statement whose `m`, `k_log`, circuit fold and extra claims are
the model's `S` and `extra`, the record and both proofs decode. Then at the proofs' openings, each rep's compiled game,
played with its proof's messages and every coin of its stream, accepts, and both reps' paths verify against the one
level-0 cap. -/
theorem verify_refines (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (extra : Session → List (Model.Claim 𝔽 S.m))
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
      OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
      OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) := by
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
  have hok₀ := opensOK_of msId sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) hcaps₀ hreqs₀ fun ℓ j => rfl
  have hok₁ := opensOK_of msId sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) hcaps₁ hreqs₁ fun ℓ j => rfl
  have hcapeq : p₁.cap = p₀.cap := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans hc₀)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  rw [hcapeq] at hok₁
  exact ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀,
    hok₁⟩

open Classical in
/-- **The verifier refines the model's table, as one run** (§5). When `Flock.verify` accepts a one-table session, with the extra claims the
model's `extraClaims` at link points `pts` read off the record, some message list makes `tableAfter` accept. The run is
played on the two streams' coins, rep 0's then rep 1's, and its openings are the proofs'. The messages are existential
and the schedule is rep 0 first, while the live server may interleave the streams: a bound on the live interaction
lifts from `verify_refines`, which names each rep's messages and coins. -/
theorem verify_tableAfter (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (ptLocal mPts : ℕ) (pts : Session → Fin 2 → Fin mPts → 𝔽)
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess →
      ExtraRealizes st S sess.points (Model.extraClaims S ptLocal mPts (pts sess)))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutC 𝔽 Dig),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (tableAfter execArith H512 enc512 S (schedOf sch) hm ptLocal mPts (capDig p₀.cap) (pts sess)) ms
        (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
      out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch := by
  obtain ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    verify_refines spec st S hm hmS hkS hfold msId hms (fun sess => Model.extraClaims S ptLocal mPts (pts sess)) record
      proofs hext h
  obtain ⟨ms, hint⟩ := Plays.interleave_first hpl₀ hpl₁
  refine ⟨sess, sch, p₀, p₁, s₀, s₁, ms ++ [⟨Opens 𝔽 Dig, opensOf p₀ p₁ sch⟩],
    ⟨pend₀.verdict (opensOf p₀ p₁ sch) && pend₁.verdict (opensOf p₀ p₁ sch) &&
      decide (OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
        OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch)),
      capDig p₀.cap, (pend₀, pend₁), opensOf p₀ p₁ sch⟩, hsess, hf, hp₀, hp₁, hs₀, hs₁, ?_, ?_, rfl⟩
  · unfold tableAfter
    exact Plays.bind' (f := fun pend : PendL 𝔽 Dig × PendL 𝔽 Dig => Game.send (Opens 𝔽 Dig) fun o =>
        Game.ret (⟨pend.1.verdict o && pend.2.verdict o &&
          decide (OpensOK H512 enc512 (capDig p₀.cap) 0 pend.1 o ∧ OpensOK H512 enc512 (capDig p₀.cap) 1 pend.2 o),
          capDig p₀.cap, pend, o⟩ : OutC 𝔽 Dig)) hint (Plays.send _ (Plays.ret _)) rfl (by simp)
  · simp only [hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
    exact ⟨hok₀, hok₁⟩

end FlockSoundness.Refine
