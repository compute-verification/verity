import Proofs.Flock.Claims.Recursion.RepFramesBind

/-!
# What a rep's frames fix of its proof, before Ligerito

Two decoded proofs of one shape whose rep frames (`repFrames`) are the same carry the same values in every field the
statement binding, the zerocheck, the lincheck and the ring switch absorb, and have the same Ligerito frames
(`repFrames_fields`). The zerocheck's `finalC` is in no frame. With `rep_frames_bind` (`RepFramesBind`), two proofs that
`verifyRep` accepts as one rep on one retained stream agree there (`rep_bind_fields`).

Ligerito's fields from equal `ligFrames` are `RepFramesLig`'s; the openings and `initialCap` are in no frame.
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine

/-- Blocks of one length, one per index, are recovered from their concatenation. -/
theorem flatMap_range'_inj {α : Type} {F G : ℕ → List α} :
    ∀ (n a : ℕ), (List.range' a n).flatMap F = (List.range' a n).flatMap G → (∀ i, (F i).length = (G i).length) →
      ∀ i, a ≤ i → i < a + n → F i = G i
  | 0, _, _, _, _, h1, h2 => by omega
  | n + 1, a, h, hl, i, h1, h2 => by
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨h0, hr⟩ := List.append_inj h (hl a)
    by_cases hi : i = a
    · exact hi ▸ h0
    · exact flatMap_range'_inj n (a + 1) hr hl i (by omega) (by omega)

theorem array_ext_getElem! {α : Type} [Inhabited α] {a b : Array α} (hs : a.size = b.size)
    (h : ∀ i < a.size, a[i]! = b[i]!) : a = b :=
  Array.ext hs fun i h1 h2 => by
    have := h i h1
    rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

theorem foldl_append_data (a : List ByteArray) (b : ByteArray) :
    (a.foldl (· ++ ·) b).data.toList = b.data.toList ++ (a.map fun e => e.data.toList).flatten := by
  induction a generalizing b with
  | nil => simp
  | cons e a ih => simp [List.foldl_cons, ih, ByteArray.data_append, List.append_assoc]

theorem flatten_inj_len {α : Type} {s : ℕ} : ∀ {L L' : List (List α)}, L.length = L'.length →
    (∀ x ∈ L, x.length = s) → (∀ x ∈ L', x.length = s) → L.flatten = L'.flatten → L = L'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, h, _, _, _ => by simp at h
  | _ :: _, [], h, _, _, _ => by simp at h
  | x :: L, x' :: L', hl, hs, hs', h => by
    rw [List.flatten_cons, List.flatten_cons] at h
    obtain ⟨h0, hr⟩ := List.append_inj h (by rw [hs x List.mem_cons_self, hs' x' List.mem_cons_self])
    rw [h0, flatten_inj_len (by simpa using hl) (fun y hy => hs y (List.mem_cons_of_mem _ hy))
      (fun y hy => hs' y (List.mem_cons_of_mem _ hy)) hr]

/-- **A cap of digests of one length is its concatenation.** -/
theorem cap_inj {a a' : Array ByteArray} {s : ℕ} (hl : a.size = a'.size) (hs : ∀ e ∈ a, e.size = s)
    (hs' : ∀ e ∈ a', e.size = s)
    (h : a.foldl (· ++ ·) ByteArray.empty = a'.foldl (· ++ ·) ByteArray.empty) : a = a' := by
  have hd := congrArg (fun b : ByteArray => b.data.toList) h
  simp only [← Array.foldl_toList, foldl_append_data] at hd
  have hsz : ∀ {c : Array ByteArray}, (∀ e ∈ c, e.size = s) →
      ∀ x ∈ c.toList.map (fun e => e.data.toList), x.length = s := by
    intro c hc x hx
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hx
    rw [Array.length_toList]
    exact hc e (Array.mem_toList_iff.1 he)
  have hL := flatten_inj_len (by simp [hl]) (hsz hs) (hsz hs') (List.append_cancel_left hd)
  have hinj : Function.Injective (fun e : ByteArray => e.data.toList) :=
    fun x y hxy => ByteArray.ext (Array.toList_inj.1 hxy)
  exact Array.toList_inj.1 (List.map_injective_iff.2 hinj hL)

theorem zcFrames_inj {m : ℕ} {d c c' : ByteArray} {z z' : ZerocheckProof} (hs : z.rounds.size = m - 6)
    (hs' : z'.rounds.size = m - 6) (h : zcFrames m d c z = zcFrames m d c' z') :
    c = c' ∧ z.round1ab = z'.round1ab ∧ z.round1c = z'.round1c ∧ z.rounds = z'.rounds ∧ z.finalA = z'.finalA ∧
      z.finalB = z'.finalB := by
  unfold zcFrames at h
  obtain ⟨h12, h3⟩ := List.append_inj h (by simp)
  obtain ⟨h1, h2⟩ := List.append_inj h12 (by simp)
  simp only [List.cons.injEq, Frame.bytes.injEq, Frame.vec.injEq, Frame.one.injEq, true_and, and_true] at h1 h3
  obtain ⟨hc, ha, hb⟩ := h1
  have hr := flatMap_range'_inj (m - 6) 0 h2 (by intro _; rfl)
  refine ⟨hc, by simpa using ha, by simpa using hb, array_ext_getElem! (hs.trans hs'.symm) fun i hi => ?_, h3.1, h3.2⟩
  have := hr i (Nat.zero_le _) (by omega)
  simp only [List.cons.injEq, Frame.one.injEq, and_true] at this
  exact Prod.ext this.1 this.2

theorem lcFrames_inj {k : ℕ} {l l' : LincheckProof} (hs : l.rounds.size = k - 6) (hs' : l'.rounds.size = k - 6)
    (h : lcFrames k l = lcFrames k l') : l = l' := by
  unfold lcFrames at h
  obtain ⟨h12, h3⟩ := List.append_inj h (by simp)
  obtain ⟨-, h2⟩ := List.append_inj h12 (by simp)
  simp only [List.cons.injEq, Frame.vec.injEq, and_true] at h3
  have hr := flatMap_range'_inj (k - 6) 0 h2 (by intro _; rfl)
  have e₁ : l.rounds = l'.rounds := array_ext_getElem! (hs.trans hs'.symm) fun i hi => by
    have := hr i (Nat.zero_le _) (by omega)
    simp only [List.cons.injEq, Frame.one.injEq, and_true] at this
    exact Prod.ext this.1 this.2
  have e₂ : l.zPartial = l'.zPartial := by simpa using h3
  cases l
  cases l'
  cases e₁
  cases e₂
  rfl

theorem rsFrames_inj {n : ℕ} {rs rs' : Array (Array F128)} (hs : rs.size = n) (hs' : rs'.size = n)
    (h : rsFrames n rs = rsFrames n rs') : rs = rs' := by
  unfold rsFrames at h
  obtain ⟨h1, -⟩ := List.append_inj (List.cons.inj h).2 (by simp)
  have hr := flatMap_range'_inj n 0 h1 (by intro _; rfl)
  refine array_ext_getElem! (hs.trans hs'.symm) fun k hk => ?_
  have := hr k (Nat.zero_le _) (by omega)
  simp only [List.cons.injEq, Frame.vec.injEq, true_and, and_true] at this
  simpa using this

/-- A circuit's setup folds at its own `kLog`. -/
theorem ofCircuit_fold_kLog (s : Stmt) : (Setup.ofCircuit s).fold.kLog = (Setup.ofCircuit s).kLog := rfl

/-- **Two decoded proofs of one shape with one rep's frames agree before Ligerito**: the same cap, zerocheck but for
`finalC`, lincheck and ring switches, and the same Ligerito frames. -/
theorem repFrames_fields {st : Setup} {sch : Flock.Schedule} {nExtra : ℕ} {p p' : Proof}
    (hp : DecodeZL.ProofSizes (shapeOf st sch nExtra) p) (hp' : DecodeZL.ProofSizes (shapeOf st sch nExtra) p')
    (hk : st.fold.kLog = st.kLog) {domain : ByteArray} {T T' : F128}
    (h : repFrames st sch domain (2 + nExtra) p T = repFrames st sch domain (2 + nExtra) p' T') :
    p.cap = p'.cap ∧ p.zerocheck.round1ab = p'.zerocheck.round1ab ∧ p.zerocheck.round1c = p'.zerocheck.round1c ∧
      p.zerocheck.rounds = p'.zerocheck.rounds ∧ p.zerocheck.finalA = p'.zerocheck.finalA ∧
      p.zerocheck.finalB = p'.zerocheck.finalB ∧ p.lincheck = p'.lincheck ∧ p.ringSwitches = p'.ringSwitches ∧
      ligFrames sch T p.cap p.ligerito = ligFrames sch T' p'.cap p'.ligerito := by
  unfold repFrames at h
  obtain ⟨h123, h4⟩ := List.append_inj (List.cons.inj h).2 (by simp [zcFrames, lcFrames, rsFrames])
  obtain ⟨h12, h3⟩ := List.append_inj h123 (by simp [zcFrames, lcFrames])
  obtain ⟨h1, h2⟩ := List.append_inj h12 (by simp [zcFrames])
  obtain ⟨hc, ha, hb, hr, hA, hB⟩ := zcFrames_inj hp.zcRounds hp'.zcRounds h1
  have hcap := cap_inj (hp.cap.trans hp'.cap.symm) hp.capEach hp'.capEach hc
  refine ⟨hcap, ha, hb, hr, hA, hB, lcFrames_inj (by rw [hk]; exact hp.lcRounds) (by rw [hk]; exact hp'.lcRounds) h2,
    rsFrames_inj hp.rs hp'.rs h3, h4⟩

/-- **Two reps accepted on one stream agree before Ligerito**: two proof files that `verifyRep` accepts as rep `r` of
table `i` of one session, where the rep's stream retained every round's bytes, decode at one shape to proofs with the
same cap, zerocheck but for `finalC`, lincheck and ring switches, and the same Ligerito frames. -/
theorem rep_bind_fields (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : RunRecord) (i r : ℕ)
    (bytes bytes' cap cap' : ByteArray) (hm : 14 ≤ st.m) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) (hk : st.fold.kLog = st.kLog)
    (h : verifyRep spec st sch sess i r bytes = .ok cap) (h' : verifyRep spec st sch sess i r bytes' = .ok cap')
    (hret : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream →
        AllRetained stream.rounds) :
    ∃ (extras : Array Flock.Claim) (p p' : Proof) (T T' : F128), st.extra sess.points = .ok extras ∧
      decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧ decodeProof (shapeOf st sch extras.size) bytes' = .ok p' ∧
      p.cap = p'.cap ∧ p.zerocheck.round1ab = p'.zerocheck.round1ab ∧ p.zerocheck.round1c = p'.zerocheck.round1c ∧
      p.zerocheck.rounds = p'.zerocheck.rounds ∧ p.zerocheck.finalA = p'.zerocheck.finalA ∧
      p.zerocheck.finalB = p'.zerocheck.finalB ∧ p.lincheck = p'.lincheck ∧ p.ringSwitches = p'.ringSwitches ∧
      ligFrames sch T p.cap p.ligerito = ligFrames sch T' p'.cap p'.ligerito := by
  obtain ⟨extras, p, p', T, T', hex, hd, hd', hF⟩ :=
    rep_frames_bind spec st sch sess i r bytes bytes' cap cap' hm hdim hlv h h' hret
  exact ⟨extras, p, p', T, T', hex, hd, hd',
    repFrames_fields (DecodeZL.decodeProof_sizes _ _ _ hd) (DecodeZL.decodeProof_sizes _ _ _ hd') hk hF⟩

end FlockSoundness.Discharge.Exec
