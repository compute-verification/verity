import FlockSoundness.Refine.Paths
import FlockSoundness.Refine.Seam
import FlockSoundness.Session

/-!
# The verifier refines the model's table (PROTOCOL.md §3, §5)

`verify_refines`: when the executable's `Flock.verify` accepts a session record and its two proof files, under the
unsalted SHA-512 scheme, the record and both proofs decode. Then at the proofs' openings (`opensOf`), each rep's
compiled game `repC`, played with its proof's messages and its stream's coins, accepts. Both reps' paths also verify
against the one level-0 cap. That is `tableAfter`'s verdict, and `verify_tableAfter` states it as one run of
`tableAfter`.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

theorem bytes_eq_of_beq {a b : ByteArray} (h : (a == b) = true) : a = b := ByteArray.ext (beq_iff_eq.1 h)

theorem flat_data (L : List ByteArray) : ∀ acc : ByteArray,
    (L.foldl (· ++ ·) acc).data.toList = acc.data.toList ++ (L.map fun b => b.data.toList).flatten := by
  induction L with
  | nil => intro acc; simp
  | cons b L ih =>
      intro acc
      rw [List.foldl_cons, ih, ByteArray.data_append, Array.toList_append]
      simp

/-- Equally many 64-byte entries that flatten to the same bytes are the same entries. -/
theorem entries_eq_of_flat : ∀ (a b : List ByteArray), (∀ x ∈ a, x.size = 64) → (∀ x ∈ b, x.size = 64) →
    a.length = b.length → (a.map fun b => b.data.toList).flatten = (b.map fun b => b.data.toList).flatten → a = b
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, hl, _ => by simp at hl
  | x :: a, y :: b, ha, hb, hl, h => by
      simp only [List.map_cons, List.flatten_cons] at h
      obtain ⟨h1, h2⟩ := List.append_inj h (by
        rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data,
          ha x (List.mem_cons_self), hb y (List.mem_cons_self)])
      have hxy : x = y := ByteArray.ext (Array.ext' h1)
      rw [hxy, entries_eq_of_flat a b (fun z hz => ha z (List.mem_cons_of_mem _ hz))
        (fun z hz => hb z (List.mem_cons_of_mem _ hz)) (by simpa using hl) h2]

/-- The proof files' openings (§14): rep `r`'s opened rows and siblings, level by level. -/
def opensOf (p₀ p₁ : Proof) (sch : Flock.Schedule) : Opens 𝔽 Dig :=
  fun r ℓ j => if r = 0 then openRow p₀.ligerito sch ℓ j else openRow p₁.ligerito sch ℓ j

/-- **The verifier refines the model's table** (§3, §5, §9–§14). When `Flock.verify` accepts a session record and its
two proof files under the unsalted SHA-512 scheme, on a statement whose `m`, `k_log`, circuit fold and extra claims are
the model's `S` and `extra`, the record and both proofs decode. Then at the proofs' openings, each rep's compiled game,
played with its proof's messages and every coin of its stream, accepts, and both reps' paths verify against the one
level-0 cap. -/
theorem verify_refines (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m) (h14 : 14 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (extra : Session → List (Model.Claim 𝔽 S.m))
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode st.spec record = .ok sess → ExtraRealizes st S sess.points (extra sess))
    (hwf : ∀ sess, Record.decode st.spec record = .ok sess → ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h : Flock.verify st record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.table ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.table ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
      OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) := by
  unfold Flock.verify at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  cases hf : Flock.fast100 st.m with
  | none => rw [hf] at h; cases h
  | some sch =>
      rw [hf] at h
      dsimp only at h
      by_cases c1 : (proofs.size != 2) = true
      · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c1] at h
      split at h
      · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      obtain ⟨_, _, h⟩ := ebind_ok h
      obtain ⟨c0, h0, h⟩ := ebind_ok h
      obtain ⟨c1', h1, h⟩ := ebind_ok h
      by_cases c3 : (c0 != c1') = true
      · rw [ite_eq_left c3] at h; cases h
      have c3' : c0 = c1' := by
        have hb : (c0 == c1') = true := by
          cases hb : (c0 == c1')
          · exact absurd (by simp [bne, hb]) c3
          · rfl
        exact bytes_eq_of_beq hb
      obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
      have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
        fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
      have hschm : sch.m = S.m := hsm.trans hmS
      obtain ⟨s₀, p₀, hs₀, hp₀, hc₀, -, hsz₀, hw₀, hrest₀⟩ := rep_refines st S hm h14 sch hmS hkS hfold (extra sess)
        sess (hext sess hsess) hdim hlv hschm (hwf sess hsess) 0 _ _ h0
      obtain ⟨s₁, p₁, hs₁, hp₁, hc₁, -, hsz₁, hw₁, hrest₁⟩ := rep_refines st S hm h14 sch hmS hkS hfold (extra sess)
        sess (hext sess hsess) hdim hlv hschm (hwf sess hsess) 1 _ _ h1
      obtain ⟨pend₀, hpl₀, hcaps₀, hreqs₀, hver₀⟩ := hrest₀ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
      obtain ⟨pend₁, hpl₁, hcaps₁, hreqs₁, hver₁⟩ := hrest₁ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
      rw [hms] at hreqs₀ hreqs₁ hw₀ hw₁
      have hok₀ := opensOK_of msId sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) hcaps₀ hreqs₀ fun ℓ j => rfl
      have hok₁ := opensOK_of msId sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) hcaps₁ hreqs₁ fun ℓ j => rfl
      have hcapeq : p₁.cap = p₀.cap := by
        apply Array.ext'
        refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
        have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans (c3'.symm.trans hc₀))
        simp only [← Array.foldl_toList, flat_data] at e
        simpa using e
      rw [hcapeq] at hok₁
      exact ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, rfl, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀,
        hok₁⟩

open Classical in
/-- **The verifier refines the model's table, as one run** (§5). When `Flock.verify` accepts, with the extra claims the
model's `extraClaims` at link points `pts` read off the record, `tableAfter` has an accepting run. It is played on the
two streams' coins, rep 0's then rep 1's; its messages interleave the proofs' messages, and end with the openings. -/
theorem verify_tableAfter (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m) (h14 : 14 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (ptLocal mPts : ℕ) (pts : Session → Fin 2 → Fin mPts → 𝔽)
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode st.spec record = .ok sess →
      ExtraRealizes st S sess.points (Model.extraClaims S ptLocal mPts (pts sess)))
    (hwf : ∀ sess, Record.decode st.spec record = .ok sess → ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h : Flock.verify st record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutC 𝔽 Dig),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.table ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.table ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (tableAfter execArith H512 enc512 S (schedOf sch) hm ptLocal mPts (capDig p₀.cap) (pts sess)) ms
        (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
      out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch := by
  obtain ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    verify_refines st S hm h14 hmS hkS hfold msId hms (fun sess => Model.extraClaims S ptLocal mPts (pts sess)) record
      proofs hext hwf h
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
