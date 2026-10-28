import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# Two reps accepted on one stream have one rep's frames

The core of the deployed plain verifier's binding of its last message's calls: two proof files that `verifyRep`
accepts as one rep of one session, on a stream whose rounds all retained their bytes, decode to proofs whose rep frames
(`repFrames`, every value the rep's transcript absorbs, in order) are the same.

* `Refine.encs_inj_hd`: frame lists with the same headers are recoverable from their bytes. `encs_inj` asks every
  frame's length to fit its header's eight bytes, which nothing in the logic bounds; two decoded proofs of one shape
  carry vectors of the same lengths (`DecodeZL.decodeProof_sizes`), so their frames have the same headers
  (`repFrames_hds`).
* `chunks_flatten`, `chunks_snd_of_closes`, `repFrames_last`: a rep's frames are their rounds' chunks, with nothing
  left pending, since Ligerito ends on a squeeze.
* `rep_frames_bind`: the rounds retained each chunk's bytes (`rep_rounds`), so the two proofs' frames have the same
  bytes, and so are the same frames.

What the frames don't carry is open: the openings (rows, siblings and salts at every level), which the Merkle checks
tie to the caps the frames carry, and the fields the transcript never absorbs (`params`, which `verifyRep` pins,
`finalC` and `initialCap`).
-/

namespace FlockSoundness.Refine

open Flock

namespace Frame

/-- A frame's header fields: its op, its kind and its length. -/
def hd (f : Frame) : UInt8 × UInt8 × ℕ := (f.op, f.kind, f.len)

@[simp] theorem hd_dom (b : ByteArray) : (Frame.dom b).hd = (1, 0, b.size) := rfl
@[simp] theorem hd_label (b : ByteArray) : (Frame.label b).hd = (2, 0, b.size) := rfl
@[simp] theorem hd_one (v : F128) : (Frame.one v).hd = (3, 1, 1) := rfl
@[simp] theorem hd_vec (vs : List F128) : (Frame.vec vs).hd = (3, 2, vs.length) := rfl
@[simp] theorem hd_bytes (b : ByteArray) : (Frame.bytes b).hd = (5, 0, b.size) := rfl
@[simp] theorem hd_sq (k : UInt8) (n : ℕ) : (Frame.sq k n).hd = (4, k, n) := rfl

/-- **Two frames with one header are told apart by their bytes**, with whatever follows them. -/
theorem enc_append_inj_hd {f f' : Frame} {r r' : List UInt8} (hh : f.hd = f'.hd) (h : f.enc ++ r = f'.enc ++ r') :
    f = f' ∧ r = r' := by
  have hop : f.op = f'.op := congrArg (fun x => x.1) hh
  have hk : f.kind = f'.kind := congrArg (fun x => x.2.1) hh
  have hl : f.len = f'.len := congrArg (fun x => x.2.2) hh
  have hlen : f.enc.length = f'.enc.length := by rw [enc_length, enc_length, payLen_eq hop hk hl]
  obtain ⟨he, hr⟩ := List.append_inj h hlen
  refine ⟨?_, hr⟩
  unfold enc at he
  exact eq_of_parts hop hk hl (List.append_inj he (by rw [header_length, header_length])).2

end Frame

/-- **Frames with the same headers are recoverable from their bytes.** -/
theorem encs_inj_hd : ∀ {fs fs' : List Frame}, fs.map Frame.hd = fs'.map Frame.hd → encs fs = encs fs' → fs = fs'
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | f :: fs, f' :: fs', hh, h => by
      simp only [List.map_cons, List.cons.injEq] at hh
      simp only [encs, List.flatMap_cons] at h
      obtain ⟨rfl, hr⟩ := Frame.enc_append_inj_hd hh.1 h
      rw [encs_inj_hd hh.2 hr]

/-- A round retains one list of bytes. -/
theorem Retains.unique {r : Round} {L L' : List UInt8} (h : Retains r L) (h' : Retains r L') : L = L' :=
  Option.some.inj (h.symm.trans h')

end FlockSoundness.Refine

namespace FlockSoundness.Discharge.Exec

open Flock Refine Transcript

/-! ## A rep's frames are its rounds' chunks -/

theorem chunks_flatten : ∀ fs : List Frame, (chunks fs).1.flatten ++ (chunks fs).2 = fs
  | [] => rfl
  | f :: fs => by
    have ih := chunks_flatten fs
    by_cases hf : closes f
    · simp only [chunks, hf, ite_true, List.flatten_cons, List.cons_append, List.nil_append]
      rw [ih]
    · simp only [chunks, hf, Bool.false_eq_true, ite_false]
      rcases h : (chunks fs).1 with _ | ⟨c, cs⟩
      · rw [h] at ih
        simp only [List.flatten_nil, List.nil_append] at ih ⊢
        rw [ih]
      · rw [h] at ih
        simp only [List.flatten_cons, List.cons_append, List.append_assoc] at ih ⊢
        rw [ih]

/-- Frames that end on a squeeze that closes a round leave nothing pending. -/
theorem chunks_snd_of_closes (fs : List Frame) {f : Frame} (hf : closes f = true) : (chunks (fs ++ [f])).2 = [] := by
  rw [chunks_append]
  have hc : chunks [f] = ([[f]], []) := by simp [chunks, hf]
  rw [hc]
  rfl

theorem encs_flatten : ∀ L : List (List Frame), encs L.flatten = (L.map encs).flatten
  | [] => rfl
  | c :: L => by rw [List.flatten_cons, encs_append, encs_flatten L, List.map_cons, List.flatten_cons]

theorem getLast?_flatMap_range' {F : ℕ → List Frame} {x : Frame} (hF : ∀ i, (F i).getLast? = some x) :
    ∀ (n a : ℕ) {y : Frame}, ((List.range' a n).flatMap F).getLast? = some y → y = x
  | 0, a, y, h => by simp at h
  | n + 1, a, y, h => by
    rw [List.range'_succ, List.flatMap_cons, List.getLast?_append, hF a] at h
    rcases h' : ((List.range' (a + 1) n).flatMap F).getLast? with _ | z
    · rw [h'] at h
      simpa using h.symm
    · rw [h'] at h
      simp only [Option.some_or, Option.some.injEq] at h
      exact h ▸ getLast?_flatMap_range' hF n (a + 1) h'

theorem levelFr_last (sch : Flock.Schedule) (p : LigeritoProof) (i : ℕ) :
    (levelFr sch p i).getLast? = some (.sq 1 1) := by
  unfold levelFr
  split <;> simp [List.getLast?_cons]

/-- **A rep's frames end on Ligerito's last batching coin**, a squeeze that closes a round. -/
theorem repFrames_last (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (p : Proof) (T : F128) :
    ∃ pre, repFrames st sch domain n p T = pre ++ [.sq 1 1] := by
  apply List.getLast?_eq_some_iff.1
  rcases h : ((List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p.ligerito)).getLast? with _ | y
  · simp [repFrames, ligFrames, h, List.getLast?_cons]
  · obtain rfl := getLast?_flatMap_range' (levelFr_last sch p.ligerito) _ 0 h
    simp [repFrames, ligFrames, h, List.getLast?_cons]

theorem repFrames_chunks_snd (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (p : Proof) (T : F128) :
    (chunks (repFrames st sch domain n p T)).2 = [] := by
  obtain ⟨pre, h⟩ := repFrames_last st sch domain n p T
  rw [h]
  exact chunks_snd_of_closes pre rfl

/-! ## Two decoded proofs of one shape: one header list -/

theorem le64_size (x : UInt64) : (le64 x).size = 8 := by
  have h := congrArg List.length (le64_data x)
  simpa using h

theorem foldl_append_size (a : List ByteArray) (b : ByteArray) :
    (a.foldl (· ++ ·) b).size = b.size + (a.map ByteArray.size).sum := by
  induction a generalizing b with
  | nil => simp
  | cons e a ih => simp only [List.foldl_cons, ih, ByteArray.size_append, List.map_cons, List.sum_cons]; omega

theorem capFr_size {a : Array ByteArray} {s : ℕ} (h : ∀ e ∈ a, e.size = s) :
    (a.foldl (· ++ ·) ByteArray.empty).size = a.size * s := by
  rw [← Array.foldl_toList, foldl_append_size, ByteArray.size_empty, Nat.zero_add,
    List.map_congr_left fun e he => h e (Array.mem_toList_iff.1 he), List.map_const', List.sum_replicate,
    Array.length_toList, smul_eq_mul]

theorem chFr_hds (p p' : LigeritoProof) : ∀ (k msg msg' : ℕ),
    (chFr p msg k).map Frame.hd = (chFr p' msg' k).map Frame.hd
  | 0, _, _ => rfl
  | k + 1, msg, msg' => by
    simp only [chFr, kFr, List.map_cons, List.map_append, Frame.hd_sq, Frame.hd_vec, List.length_cons,
      List.length_nil, chFr_hds p p' k (msg + 1) (msg' + 1)]

theorem introFr_hds (p p' : LigeritoProof) (dim ood msg cbn : ℕ) :
    (introFr p dim ood msg cbn).map Frame.hd = (introFr p' dim ood msg cbn).map Frame.hd := by
  simp [introFr, kFr, le64_size]

theorem queryFr_hds (p p' : LigeritoProof) (lv : Flock.Level) (qn cn : ℕ) :
    (queryFr p lv qn cn).map Frame.hd = (queryFr p' lv qn cn).map Frame.hd := by
  simp [queryFr, le64_size]

/-- The lengths of a decoded proof's recursive caps, as one byte string each, are its shape's. -/
theorem rcap_size {sh : Shape} {p : Proof} (hp : DecodeZL.ProofSizes sh p) (i : ℕ) :
    (p.ligerito.recursiveCaps[i]!.foldl (· ++ ·) ByteArray.empty).size =
      if i < sh.sched.levels.size - 1 then 2 ^ (sh.sched.levels.getD (i + 1) default).cap * sh.hashLen else 0 := by
  by_cases hi : i < p.ligerito.recursiveCaps.size
  · obtain ⟨hsz, he⟩ := hp.rcapsEach i hi
    rw [getElem!_pos _ i hi, capFr_size he, hsz, ite_eq_left (hp.rcaps ▸ hi)]
  · rw [getElem!_neg _ i hi, ite_eq_right (by rw [← hp.rcaps]; exact hi)]
    rfl

theorem rs_size {sh : Shape} {p : Proof} (hp : DecodeZL.ProofSizes sh p) (k : ℕ) :
    (p.ringSwitches[k]!).size = if k < sh.claims then 128 else 0 := by
  by_cases hk : k < p.ringSwitches.size
  · rw [getElem!_pos _ k hk, hp.rsEach _ (Array.getElem_mem hk), ite_eq_left (hp.rs ▸ hk)]
  · rw [getElem!_neg _ k hk, ite_eq_right (by rw [← hp.rs]; exact hk)]
    rfl

theorem levelFr_hds {sh : Shape} {p p' : Proof} (hp : DecodeZL.ProofSizes sh p) (hp' : DecodeZL.ProofSizes sh p')
    (sch : Flock.Schedule) (i : ℕ) :
    (levelFr sch p.ligerito i).map Frame.hd = (levelFr sch p'.ligerito i).map Frame.hd := by
  unfold levelFr
  rw [List.map_append, List.map_append, chFr_hds p.ligerito p'.ligerito]
  congr 1
  split
  · simp only [List.map_cons, List.map_append, capFr, Frame.hd_bytes, rcap_size hp, rcap_size hp',
      introFr_hds p.ligerito p'.ligerito, queryFr_hds p.ligerito p'.ligerito]
    simp [kFr, le64_size]
  · simp only [List.map_append, List.map_map, queryFr_hds p.ligerito p'.ligerito]
    have hy : (p.ligerito.finalProof.yr.toList.map (Frame.hd ∘ Frame.one)) =
        (p'.ligerito.finalProof.yr.toList.map (Frame.hd ∘ Frame.one)) := by
      have e : (Frame.hd ∘ Frame.one) = fun _ => ((3 : UInt8), (1 : UInt8), 1) := rfl
      rw [e, List.map_const', List.map_const', Array.length_toList, Array.length_toList, hp.yr, hp'.yr]
    rw [hy]
    simp [le64_size]

/-- **Two decoded proofs of one shape have frames with the same headers**, whatever the batched values. -/
theorem repFrames_hds {st : Setup} {sch : Flock.Schedule} {nExtra : ℕ} {p p' : Proof}
    (hp : DecodeZL.ProofSizes (shapeOf st sch nExtra) p) (hp' : DecodeZL.ProofSizes (shapeOf st sch nExtra) p')
    (domain : ByteArray) (T T' : F128) :
    (repFrames st sch domain (2 + nExtra) p T).map Frame.hd =
      (repFrames st sch domain (2 + nExtra) p' T').map Frame.hd := by
  have hcap : (p.cap.foldl (· ++ ·) ByteArray.empty).size = (p'.cap.foldl (· ++ ·) ByteArray.empty).size := by
    rw [capFr_size hp.capEach, capFr_size hp'.capEach, hp.cap, hp'.cap]
  have hrs : ∀ k : ℕ, (p.ringSwitches[k]!).size = (p'.ringSwitches[k]!).size := fun k => by
    rw [rs_size hp, rs_size hp']
  have hlv : (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (levelFr sch p.ligerito i).map Frame.hd) =
      (List.range' 0 (sch.levels.size - 1)).flatMap (fun i => (levelFr sch p'.ligerito i).map Frame.hd) := by
    congr 1
    funext i
    exact levelFr_hds hp hp' sch i
  simp only [repFrames, zcFrames, lcFrames, rsFrames, ligFrames, capFr, List.map_cons, List.map_append,
    List.map_flatMap, Frame.hd_bytes, Frame.hd_vec, Frame.hd_one, Frame.hd_sq, Frame.hd_label, Frame.hd_dom,
    Array.length_toList, hp.r1ab, hp'.r1ab, hp.r1c, hp'.r1c, hp.zPartial, hp'.zPartial, hcap, hrs, hlv,
    introFr_hds p.ligerito p'.ligerito, queryFr_hds p.ligerito p'.ligerito, rcap_size hp, rcap_size hp']
  rw [chFr_hds p.ligerito p'.ligerito sch.k0 1 1]
  simp [kFr, le64_size]

/-! ## Two reps accepted on one stream -/

/-- **Two reps accepted on one stream have one rep's frames**: two proof files that `verifyRep` accepts as rep `r` of
table `i` of one session, where the rep's stream retained every round's bytes, decode at one shape to proofs whose
frames are the same. -/
theorem rep_frames_bind (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : RunRecord) (i r : ℕ)
    (bytes bytes' cap cap' : ByteArray) (hm : 14 ≤ st.m) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (h : verifyRep spec st sch sess i r bytes = .ok cap) (h' : verifyRep spec st sch sess i r bytes' = .ok cap')
    (hret : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream →
        AllRetained stream.rounds) :
    ∃ (extras : Array Flock.Claim) (p p' : Proof) (T T' : F128), st.extra sess.points = .ok extras ∧
      decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧ decodeProof (shapeOf st sch extras.size) bytes' = .ok p' ∧
      repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T =
        repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p' T' := by
  obtain ⟨stream, p, extras, T, hs, hex, hp, hR⟩ := rep_rounds spec st sch sess i r bytes cap hm hdim hlv h
  obtain ⟨stream', p', extras', T', hs', hex', hp', hR'⟩ := rep_rounds spec st sch sess i r bytes' cap' hm hdim hlv h'
  rw [hs] at hs'
  cases hs'
  rw [hex] at hex'
  cases hex'
  refine ⟨extras, p, p', T, T', hex, hp, hp', ?_⟩
  obtain ⟨hn, hc⟩ := hR (hret stream hs)
  obtain ⟨hn', hc'⟩ := hR' (hret stream hs)
  generalize hF : repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T = F at hn hc
  generalize hF' : repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p' T' = F' at hn' hc'
  have hhd : F.map Frame.hd = F'.map Frame.hd := by
    rw [← hF, ← hF']
    exact repFrames_hds (DecodeZL.decodeProof_sizes _ _ _ hp) (DecodeZL.decodeProof_sizes _ _ _ hp') _ T T'
  have h2 : (chunks F).2 = [] := hF ▸ repFrames_chunks_snd _ _ _ _ _ _
  have h2' : (chunks F').2 = [] := hF' ▸ repFrames_chunks_snd _ _ _ _ _ _
  have hmap : (chunks F).1.map encs = (chunks F').1.map encs := by
    refine List.ext_getElem (by rw [List.length_map, List.length_map, hn, hn']) fun j hj hj' => ?_
    rw [List.length_map, hn] at hj
    have e₁ := hc j hj
    have e₂ := hc' j hj
    rw [List.getElem_map, List.getElem_map, ← List.getD_eq_getElem _ [], ← List.getD_eq_getElem _ []]
    exact Retains.unique e₁ e₂
  refine encs_inj_hd hhd ?_
  rw [← chunks_flatten F, ← chunks_flatten F', h2, h2', List.append_nil, List.append_nil, encs_flatten,
    encs_flatten, hmap]

end FlockSoundness.Discharge.Exec
