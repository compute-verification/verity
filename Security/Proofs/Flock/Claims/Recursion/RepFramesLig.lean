import Proofs.Flock.Claims.Recursion.RepFramesFields

/-!
# What a rep's Ligerito frames fix of its proof

Two Ligerito proofs with the same frames (`ligFrames`), at one schedule of at least two levels and with the lengths a
decoded proof has (`DecodeZL.ProofSizes`), carry the same recursive caps, sumcheck messages, out-of-domain values,
query, claim and consistency nonces, and clear residual `yr` (`ligFrames_fields`). The frames read each of those lists
at every index the sizes allow:

* the header reads message 0, the folding challenges' messages `1 … k0`, two intros (`k0 + 1`, `k0 + 2`) and the query
  message `k0 + 3`, out-of-domain values `0 … 2`, claim nonces `0 … 3`, level 0's query and consistency nonces, and
  the first recursive cap;
* level `i` of the loop (`levelFr`) reads messages `k0 + 4 + 7i …`, seven of them below the last level and four at
  it, its query and consistency nonces `1 + i`, and below the last level the cap `i + 1`, out-of-domain values
  `3 + 2i`, `4 + 2i` and claim nonces `4 + 3i … 6 + 3i`; at the last, `yr` and claim nonce `4 + 3i`.

With `rep_bind_fields`, two proofs `verifyRep` accepts as one rep on one retained stream agree in every field but
`params`, `zerocheck.finalC`, `ligerito.initialCap` and the openings (`rep_bind_lig`).
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine

theorem f256_ext {a b : F256} (h0 : a.c0 = b.c0) (h1 : a.c1 = b.c1) : a = b := by
  cases a
  cases b
  cases h0
  cases h1
  rfl

theorem kFr_inj {m m' : F256 × F256} (h : kFr m = kFr m') : m = m' := by
  simp only [kFr, List.cons.injEq, Frame.vec.injEq, and_true] at h
  exact Prod.ext (f256_ext h.1.1 h.1.2) (f256_ext h.2.1 h.2.2)

theorem chFr_length (p : LigeritoProof) : ∀ (msg k : ℕ), (chFr p msg k).length = 3 * k
  | _, 0 => rfl
  | msg, k + 1 => by
    simp only [chFr, kFr, List.length_cons, List.length_append, List.length_nil, chFr_length p (msg + 1) k]
    omega

theorem chFr_inj {p p' : LigeritoProof} : ∀ (msg k : ℕ), chFr p msg k = chFr p' msg k →
    ∀ j, msg ≤ j → j < msg + k → p.messages[j]! = p'.messages[j]!
  | _, 0, _, j, h1, h2 => by omega
  | msg, k + 1, h, j, h1, h2 => by
    simp only [chFr] at h
    obtain ⟨h0, hr⟩ := List.append_inj h (by simp [kFr])
    by_cases hj : j = msg
    · subst hj
      exact kFr_inj (List.cons.inj h0).2
    · exact chFr_inj (msg + 1) k hr j (by omega) (by omega)

theorem introFr_inj {p p' : LigeritoProof} {dim ood msg cbn : ℕ}
    (h : introFr p dim ood msg cbn = introFr p' dim ood msg cbn) :
    p.oodValues[ood]! = p'.oodValues[ood]! ∧ p.messages[msg]! = p'.messages[msg]! ∧
      p.claimNonces[cbn]! = p'.claimNonces[cbn]! := by
  unfold introFr at h
  obtain ⟨h0, h1⟩ := List.append_inj h (by simp [kFr])
  obtain ⟨-, h0⟩ := List.cons.inj h0
  obtain ⟨hy, hk⟩ := List.cons.inj h0
  simp only [List.cons.injEq, Frame.bytes.injEq, and_true] at h1
  exact ⟨Frame.one.inj hy, kFr_inj hk, le64_inj h1⟩

theorem queryFr_inj {p p' : LigeritoProof} {lv : Flock.Level} {qn cn : ℕ}
    (h : queryFr p lv qn cn = queryFr p' lv qn cn) :
    p.queryNonces[qn]! = p'.queryNonces[qn]! ∧ p.consistencyNonces[cn]! = p'.consistencyNonces[cn]! := by
  simp only [queryFr, List.cons.injEq, Frame.bytes.injEq, true_and, and_true] at h
  exact ⟨le64_inj h.1, le64_inj h.2⟩

/-- A recursive cap of a decoded proof is its frame. -/
theorem rcap_inj {sh : Shape} {p p' : Proof} (hp : DecodeZL.ProofSizes sh p) (hp' : DecodeZL.ProofSizes sh p')
    {j : ℕ} (hj : j < sh.sched.levels.size - 1)
    (h : capFr p.ligerito.recursiveCaps[j]! = capFr p'.ligerito.recursiveCaps[j]!) :
    p.ligerito.recursiveCaps[j]! = p'.ligerito.recursiveCaps[j]! := by
  have hj1 : j < p.ligerito.recursiveCaps.size := hp.rcaps ▸ hj
  have hj2 : j < p'.ligerito.recursiveCaps.size := hp'.rcaps ▸ hj
  obtain ⟨hs1, he1⟩ := hp.rcapsEach j hj1
  obtain ⟨hs2, he2⟩ := hp'.rcapsEach j hj2
  rw [getElem!_pos _ j hj1, getElem!_pos _ j hj2] at h ⊢
  exact cap_inj (hs1.trans hs2.symm) he1 he2 (Frame.bytes.inj h)

/-- **What level `i`'s frames fix**: its messages, query and consistency nonces and claim nonce `4 + 3i`; below the
last level the next cap's frame, two out-of-domain values and two more claim nonces; at the last, `yr`. -/
theorem levelFr_inj {sch : Flock.Schedule} {p p' : LigeritoProof} {i : ℕ} (h : levelFr sch p i = levelFr sch p' i)
    (hyr : p.finalProof.yr.size = p'.finalProof.yr.size) :
    (∀ j, sch.k0 + 4 + 7 * i ≤ j → j < sch.k0 + 8 + 7 * i → p.messages[j]! = p'.messages[j]!) ∧
      p.queryNonces[1 + i]! = p'.queryNonces[1 + i]! ∧
      p.consistencyNonces[1 + i]! = p'.consistencyNonces[1 + i]! ∧
      p.claimNonces[4 + 3 * i]! = p'.claimNonces[4 + 3 * i]! ∧
      (i + 1 < sch.levels.size - 1 →
        capFr p.recursiveCaps[i + 1]! = capFr p'.recursiveCaps[i + 1]! ∧
        p.oodValues[3 + 2 * i]! = p'.oodValues[3 + 2 * i]! ∧ p.oodValues[4 + 2 * i]! = p'.oodValues[4 + 2 * i]! ∧
        p.messages[sch.k0 + 8 + 7 * i]! = p'.messages[sch.k0 + 8 + 7 * i]! ∧
        p.messages[sch.k0 + 9 + 7 * i]! = p'.messages[sch.k0 + 9 + 7 * i]! ∧
        p.messages[sch.k0 + 10 + 7 * i]! = p'.messages[sch.k0 + 10 + 7 * i]! ∧
        p.claimNonces[5 + 3 * i]! = p'.claimNonces[5 + 3 * i]! ∧
        p.claimNonces[6 + 3 * i]! = p'.claimNonces[6 + 3 * i]!) ∧
      (¬ i + 1 < sch.levels.size - 1 → p.finalProof.yr = p'.finalProof.yr) := by
  unfold levelFr at h
  obtain ⟨hch, hrest⟩ := List.append_inj h (by rw [chFr_length, chFr_length])
  have hm : ∀ j, sch.k0 + 4 + 7 * i ≤ j → j < sch.k0 + 8 + 7 * i → p.messages[j]! = p'.messages[j]! :=
    fun j h1 h2 => chFr_inj _ 4 hch j h1 (by omega)
  by_cases hc : i + 1 < sch.levels.size - 1
  · simp only [hc, ↓reduceIte] at hrest
    obtain ⟨hcap, hY⟩ := List.cons.inj hrest
    obtain ⟨hY1, hb⟩ := List.append_inj hY (by simp [introFr, queryFr, kFr])
    obtain ⟨hY2, hk⟩ := List.append_inj hY1 (by simp [introFr, queryFr, kFr])
    obtain ⟨hY3, hq⟩ := List.append_inj hY2 (by simp [introFr, kFr])
    obtain ⟨hi1, hi2⟩ := List.append_inj hY3 (by simp [introFr, kFr])
    simp only [List.cons.injEq, Frame.bytes.injEq, and_true] at hb
    obtain ⟨o1, m1, c1⟩ := introFr_inj hi1
    obtain ⟨o2, m2, c2⟩ := introFr_inj hi2
    obtain ⟨q1, q2⟩ := queryFr_inj hq
    exact ⟨hm, q1, q2, c1, fun _ => ⟨hcap, o1, o2, m1, m2, kFr_inj hk, c2, le64_inj hb⟩, fun h => absurd hc h⟩
  · simp only [hc, ↓reduceIte] at hrest
    obtain ⟨hY, hb⟩ := List.append_inj hrest (by simp [queryFr, hyr])
    obtain ⟨hy, hq⟩ := List.append_inj hY (by simp [hyr])
    simp only [List.cons.injEq, Frame.bytes.injEq, and_true] at hb
    obtain ⟨q1, q2⟩ := queryFr_inj hq
    have hyl : p.finalProof.yr.toList = p'.finalProof.yr.toList :=
      List.map_injective_iff.2 (fun _ _ e => Frame.one.inj e) hy
    exact ⟨hm, q1, q2, le64_inj hb, fun h => absurd h hc, fun _ => Array.toList_inj.1 hyl⟩

/-- **Two decoded proofs of one shape with one rep's Ligerito frames have one Ligerito transcript**: the same
recursive caps, sumcheck messages, out-of-domain values, query, claim and consistency nonces, and `yr`, at a schedule
of at least two levels. -/
theorem ligFrames_fields {sh : Shape} {p p' : Proof} (hp : DecodeZL.ProofSizes sh p)
    (hp' : DecodeZL.ProofSizes sh p') (hL : 2 ≤ sh.sched.levels.size) {T T' : F128}
    (h : ligFrames sh.sched T p.cap p.ligerito = ligFrames sh.sched T' p'.cap p'.ligerito) :
    p.ligerito.recursiveCaps = p'.ligerito.recursiveCaps ∧ p.ligerito.messages = p'.ligerito.messages ∧
      p.ligerito.queryNonces = p'.ligerito.queryNonces ∧ p.ligerito.oodValues = p'.ligerito.oodValues ∧
      p.ligerito.claimNonces = p'.ligerito.claimNonces ∧
      p.ligerito.consistencyNonces = p'.ligerito.consistencyNonces ∧
      p.ligerito.finalProof.yr = p'.ligerito.finalProof.yr := by
  have hyr : p.ligerito.finalProof.yr.size = p'.ligerito.finalProof.yr.size := hp.yr.trans hp'.yr.symm
  unfold ligFrames at h
  obtain ⟨-, h⟩ := List.append_inj h rfl
  obtain ⟨hA, hB⟩ := List.append_inj h (by simp [kFr, chFr_length])
  obtain ⟨hA1, hch⟩ := List.append_inj hA (by simp [kFr])
  obtain ⟨h4, hk0⟩ := List.append_inj hA1 rfl
  simp only [List.cons.injEq, Frame.one.injEq, Frame.bytes.injEq, true_and, and_true] at h4
  obtain ⟨hrc0, hY⟩ := List.cons.inj hB
  obtain ⟨hY1, hFM⟩ := List.append_inj hY (by simp [introFr, queryFr, kFr])
  obtain ⟨hY2, hb3⟩ := List.append_inj hY1 (by simp [introFr, queryFr, kFr])
  obtain ⟨hY3, hk3⟩ := List.append_inj hY2 (by simp [introFr, queryFr, kFr])
  obtain ⟨hY4, hq0⟩ := List.append_inj hY3 (by simp [introFr, kFr])
  obtain ⟨hi1, hi2⟩ := List.append_inj hY4 (by simp [introFr, kFr])
  simp only [List.cons.injEq, Frame.bytes.injEq, and_true] at hb3
  obtain ⟨o1, m1, c1⟩ := introFr_inj hi1
  obtain ⟨o2, m2, c2⟩ := introFr_inj hi2
  obtain ⟨q0, n0⟩ := queryFr_inj hq0
  have hlv := flatMap_range'_inj (sh.sched.levels.size - 1) 0 hFM fun i => by
    simpa using congrArg List.length (levelFr_hds hp hp' sh.sched i)
  have L := fun i (hi : i < sh.sched.levels.size - 1) => levelFr_inj (hlv i (Nat.zero_le _) (by omega)) hyr
  have hch' := chFr_inj 1 sh.sched.k0 hch
  refine ⟨array_ext_getElem! (hp.rcaps.trans hp'.rcaps.symm) fun j hj => ?_,
    array_ext_getElem! (hp.messages.trans hp'.messages.symm) fun j hj => ?_,
    array_ext_getElem! (hp.queryNonces.trans hp'.queryNonces.symm) fun j hj => ?_,
    array_ext_getElem! (hp.oodValues.trans hp'.oodValues.symm) fun j hj => ?_,
    array_ext_getElem! (hp.claimNonces.trans hp'.claimNonces.symm) fun j hj => ?_,
    array_ext_getElem! (hp.consistencyNonces.trans hp'.consistencyNonces.symm) fun j hj => ?_,
    (L (sh.sched.levels.size - 2) (by omega)).2.2.2.2.2 (by omega)⟩
  · rw [hp.rcaps] at hj
    refine rcap_inj hp hp' hj ?_
    rcases j with _ | j
    · exact hrc0
    · exact ((L j (by omega)).2.2.2.2.1 (by omega)).1
  · rw [hp.messages] at hj
    by_cases h0 : j = 0
    · subst h0; exact kFr_inj hk0
    by_cases h1 : j ≤ sh.sched.k0
    · exact hch' j (by omega) (by omega)
    by_cases h2 : j = sh.sched.k0 + 1
    · subst h2; exact m1
    by_cases h3 : j = sh.sched.k0 + 2
    · subst h3; exact m2
    by_cases h4 : j = sh.sched.k0 + 3
    · subst h4; exact kFr_inj hk3
    obtain ⟨i, r, hr, rfl⟩ : ∃ i r, r < 7 ∧ j = sh.sched.k0 + 4 + 7 * i + r :=
      ⟨(j - (sh.sched.k0 + 4)) / 7, (j - (sh.sched.k0 + 4)) % 7, Nat.mod_lt _ (by omega), by omega⟩
    have hi : i < sh.sched.levels.size - 1 := by omega
    by_cases hr4 : r < 4
    · exact (L i hi).1 _ (by omega) (by omega)
    have hc : i + 1 < sh.sched.levels.size - 1 := by omega
    obtain ⟨-, -, -, e8, e9, e10, -, -⟩ := (L i hi).2.2.2.2.1 hc
    rcases (by omega : r = 4 ∨ r = 5 ∨ r = 6) with rfl | rfl | rfl
    · simpa [show sh.sched.k0 + 4 + 7 * i + 4 = sh.sched.k0 + 8 + 7 * i by omega] using e8
    · simpa [show sh.sched.k0 + 4 + 7 * i + 5 = sh.sched.k0 + 9 + 7 * i by omega] using e9
    · simpa [show sh.sched.k0 + 4 + 7 * i + 6 = sh.sched.k0 + 10 + 7 * i by omega] using e10
  · rw [hp.queryNonces] at hj
    rcases j with _ | j
    · exact q0
    · simpa [Nat.add_comm] using (L j (by omega)).2.1
  · rw [hp.oodValues] at hj
    by_cases h0 : j = 0
    · subst h0; exact h4.1
    by_cases h1 : j = 1
    · subst h1; exact o1
    by_cases h2 : j = 2
    · subst h2; exact o2
    obtain ⟨i, r, hr, rfl⟩ : ∃ i r, r < 2 ∧ j = 3 + 2 * i + r :=
      ⟨(j - 3) / 2, (j - 3) % 2, Nat.mod_lt _ (by omega), by omega⟩
    obtain ⟨-, e3, e4, -⟩ := (L i (by omega)).2.2.2.2.1 (by omega)
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · simpa using e3
    · simpa [show 3 + 2 * i + 1 = 4 + 2 * i by omega] using e4
  · rw [hp.claimNonces] at hj
    by_cases h0 : j = 0
    · subst h0; exact le64_inj h4.2
    by_cases h1 : j = 1
    · subst h1; exact c1
    by_cases h2 : j = 2
    · subst h2; exact c2
    by_cases h3 : j = 3
    · subst h3; exact le64_inj hb3
    obtain ⟨i, r, hr, rfl⟩ : ∃ i r, r < 3 ∧ j = 4 + 3 * i + r :=
      ⟨(j - 4) / 3, (j - 4) % 3, Nat.mod_lt _ (by omega), by omega⟩
    have hi : i < sh.sched.levels.size - 1 := by omega
    rcases (by omega : r = 0 ∨ r = 1 ∨ r = 2) with rfl | rfl | rfl
    · simpa using (L i hi).2.2.2.1
    · obtain ⟨-, -, -, -, -, -, e5, -⟩ := (L i hi).2.2.2.2.1 (by omega)
      simpa [show 4 + 3 * i + 1 = 5 + 3 * i by omega] using e5
    · obtain ⟨-, -, -, -, -, -, -, e6⟩ := (L i hi).2.2.2.2.1 (by omega)
      simpa [show 4 + 3 * i + 2 = 6 + 3 * i by omega] using e6
  · rw [hp.consistencyNonces] at hj
    rcases j with _ | j
    · exact n0
    · simpa [Nat.add_comm] using (L j (by omega)).2.2.1

/-- **Two reps accepted on one stream agree in every field but `params`, `finalC`, `initialCap` and the openings**:
`rep_bind_fields` with `ligFrames_fields`, at a schedule of at least two levels. -/
theorem rep_bind_lig (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : RunRecord) (i r : ℕ)
    (bytes bytes' cap cap' : ByteArray) (hm : 14 ≤ st.m) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) (hk : st.fold.kLog = st.kLog)
    (h : verifyRep spec st sch sess i r bytes = .ok cap) (h' : verifyRep spec st sch sess i r bytes' = .ok cap')
    (hret : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream →
        AllRetained stream.rounds) :
    ∃ (extras : Array Flock.Claim) (p p' : Proof), st.extra sess.points = .ok extras ∧
      decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧ decodeProof (shapeOf st sch extras.size) bytes' = .ok p' ∧
      p.cap = p'.cap ∧ p.zerocheck.round1ab = p'.zerocheck.round1ab ∧ p.zerocheck.round1c = p'.zerocheck.round1c ∧
      p.zerocheck.rounds = p'.zerocheck.rounds ∧ p.zerocheck.finalA = p'.zerocheck.finalA ∧
      p.zerocheck.finalB = p'.zerocheck.finalB ∧ p.lincheck = p'.lincheck ∧ p.ringSwitches = p'.ringSwitches ∧
      p.ligerito.recursiveCaps = p'.ligerito.recursiveCaps ∧ p.ligerito.messages = p'.ligerito.messages ∧
      p.ligerito.queryNonces = p'.ligerito.queryNonces ∧ p.ligerito.oodValues = p'.ligerito.oodValues ∧
      p.ligerito.claimNonces = p'.ligerito.claimNonces ∧
      p.ligerito.consistencyNonces = p'.ligerito.consistencyNonces ∧
      p.ligerito.finalProof.yr = p'.ligerito.finalProof.yr := by
  obtain ⟨extras, p, p', T, T', hex, hd, hd', hc, ha, hb, hr, hA, hB, hl, hs, hF⟩ :=
    rep_bind_fields spec st sch sess i r bytes bytes' cap cap' hm hdim hlv hk h h' hret
  have hsz := DecodeZL.decodeProof_sizes _ _ _ hd
  have hsz' := DecodeZL.decodeProof_sizes _ _ _ hd'
  exact ⟨extras, p, p', hex, hd, hd', hc, ha, hb, hr, hA, hB, hl, hs, ligFrames_fields hsz hsz' hdim.2.1 hF⟩

end FlockSoundness.Discharge.Exec
