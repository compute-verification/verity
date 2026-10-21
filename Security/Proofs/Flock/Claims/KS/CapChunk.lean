import Proofs.Flock.Claims.Recursion.RepFramesBind
import Proofs.Flock.Claims.KS.Rounds

/-!
# The round of a rep's frames that absorbs a recursive cap

A rep's frames (`repFrames`) absorb Ligerito's `ℓ`-th recursive cap as one frame (`capFr`), right before the squeeze
of the next level's first out-of-domain point. So the round that squeeze closes holds the cap, `preLig +
capRound k0 (ℓ + 1)` rounds into the rep's stream (`Rounds.lean`), and two reps whose frames have one header list and
whose rounds there retained the same bytes absorbed one cap.

* `chunks_length`: a frame list has one round per squeeze that closes one.
* `chunks_getD_split`: the round a squeeze closes right after a frame is the frames pending before that frame, the
  frame and the squeeze.
* `chunks_getD_hd`: frame lists with one header list have rounds with one header list.
* `repFrames_capSplit`: a rep's frames split at the `ℓ`-th recursive cap, for `ℓ + 1` below the schedule's levels,
  with `preLig + capRound k0 (ℓ + 1)` closing squeezes before it.
* `frame_eq_of_chunk`, `capFr_eq_of_chunk`: two such splits of frame lists with one header list, whose rounds at the
  split have the same bytes, split at one frame, so at one cap.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine FlockSoundness.Discharge.Exec

/-! ## Rounds of a frame list -/

theorem chunks_length : ∀ fs : List Frame, (chunks fs).1.length = fs.countP closes
  | [] => rfl
  | f :: fs => by
    have ih := chunks_length fs
    by_cases hf : closes f = true
    · simp only [chunks, hf, ↓reduceIte, List.length_cons, List.countP_cons, ih]
    · simp only [Bool.not_eq_true] at hf
      simp only [chunks, hf, Bool.false_eq_true, ite_false, List.countP_cons, ← ih]
      rcases h : (chunks fs).1 with _ | ⟨c, cs⟩ <;> simp

theorem chunks_getD_split (a b : List Frame) {c s : Frame} (hc : closes c = false) (hs : closes s = true) :
    (chunks (a ++ c :: s :: b)).1.getD (chunks a).1.length [] = (chunks a).2 ++ [c, s] := by
  rw [chunks_append]
  have hx : chunks (c :: s :: b) = ([c, s] :: (chunks b).1, (chunks b).2) := by
    simp only [chunks, hc, hs, Bool.false_eq_true, ite_false, ite_true]
  rw [hx]
  unfold glue
  simp only [List.getD_eq_getElem?_getD, List.getElem?_append_right (le_refl _), Nat.sub_self,
    List.getElem?_cons_zero, Option.getD_some]

theorem closes_of_hd {f f' : Frame} (h : f.hd = f'.hd) : closes f = closes f' := by
  cases f <;> cases f' <;>
    simp only [Frame.hd_dom, Frame.hd_label, Frame.hd_one, Frame.hd_vec, Frame.hd_bytes, Frame.hd_sq,
      Prod.mk.injEq] at h <;>
    simp_all [closes]

theorem chunks_hd : ∀ {fs fs' : List Frame}, fs.map Frame.hd = fs'.map Frame.hd →
    (chunks fs).1.map (List.map Frame.hd) = (chunks fs').1.map (List.map Frame.hd) ∧
      (chunks fs).2.map Frame.hd = (chunks fs').2.map Frame.hd
  | [], [], _ => ⟨rfl, rfl⟩
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | f :: fs, f' :: fs', h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨ih1, ih2⟩ := chunks_hd h.2
    have hc := closes_of_hd h.1
    by_cases hf : closes f = true
    · have hf' : closes f' = true := hc ▸ hf
      simp only [chunks, hf, hf', ↓reduceIte, List.map_cons, List.map_nil, ih1, ih2, h.1, and_self]
    · have hf0 : closes f = false := by simpa using hf
      have hf0' : closes f' = false := hc ▸ hf0
      simp only [chunks, hf0, hf0', Bool.false_eq_true, ite_false]
      rcases e : (chunks fs).1 with _ | ⟨c, cs⟩ <;> rcases e' : (chunks fs').1 with _ | ⟨c', cs'⟩ <;>
        simp only [e, e', List.map_nil, List.map_cons, reduceCtorEq, List.cons.injEq] at ih1 <;>
        simp_all

theorem chunks_getD_hd {fs fs' : List Frame} (h : fs.map Frame.hd = fs'.map Frame.hd) (i : ℕ) :
    ((chunks fs).1.getD i []).map Frame.hd = ((chunks fs').1.getD i []).map Frame.hd := by
  have e := congrArg (fun L => L[i]?) (chunks_hd h).1
  simp only [List.getElem?_map] at e
  simp only [List.getD_eq_getElem?_getD]
  rcases h1 : (chunks fs).1[i]? with _ | a <;> rcases h2 : (chunks fs').1[i]? with _ | b <;>
    simp_all

/-! ## Squeezes that close a round, stage by stage -/

theorem countP_flatMap_range' {F : ℕ → List Frame} {c : ℕ} :
    ∀ (n a : ℕ), (∀ i, a ≤ i → i < a + n → (F i).countP closes = c) →
      ((List.range' a n).flatMap F).countP closes = n * c
  | 0, a, _ => by simp
  | n + 1, a, hF => by
    rw [List.range'_succ, List.flatMap_cons, List.countP_append, hF a (le_refl a) (by omega),
      countP_flatMap_range' n (a + 1) fun i h1 h2 => hF i (by omega) (by omega)]
    ring

theorem countP_kFr (m : F256 × F256) : (kFr m).countP closes = 0 := rfl

theorem countP_chFr (p : LigeritoProof) : ∀ (k msg : ℕ), (chFr p msg k).countP closes = k
  | 0, _ => rfl
  | k + 1, msg => by
    simp only [chFr, List.cons_append, List.countP_cons, List.countP_append, countP_kFr, countP_chFr p k (msg + 1)]
    simp [closes]

theorem countP_introFr (p : LigeritoProof) {dim : ℕ} (hd : dim ≠ 0) (ood msg cbn : ℕ) :
    (introFr p dim ood msg cbn).countP closes = 2 := by
  simp [introFr, kFr, closes, hd, List.countP_cons]

theorem countP_queryFr (p : LigeritoProof) {lv : Flock.Level} (hq : lv.q ≠ 0) (qn cn : ℕ) :
    (queryFr p lv qn cn).countP closes = 2 := by
  simp [queryFr, closes, hq, List.countP_cons]

theorem countP_levelFr (sch : Flock.Schedule) (p : LigeritoProof) {i : ℕ} (hi : i + 1 < sch.levels.size - 1)
    (hq : (sch.levels.getD (i + 1) default).q ≠ 0) : (levelFr sch p i).countP closes = 11 := by
  unfold levelFr
  rw [ite_eq_left hi]
  simp only [List.countP_append, List.countP_cons, countP_chFr,
    countP_introFr p (show (sch.levels.getD (i + 1) default).cols + 1 ≠ 0 by omega), countP_queryFr p hq, countP_kFr]
  simp [closes, capFr]

theorem countP_zcFrames {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof) :
    (zcFrames m digest cap p).countP closes = 3 + (m - 6) := by
  have hb : ((List.range' 0 (m - 6)).flatMap
      (fun i => [Frame.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1])).countP closes = (m - 6) * 1 :=
    countP_flatMap_range' (m - 6) 0 fun i _ _ => by simp [closes, List.countP_cons]
  simp only [zcFrames, List.countP_append, hb]
  have h13 : m - 13 ≠ 0 := by omega
  simp [closes, h13, List.countP_cons]

theorem countP_lcFrames (kLog : ℕ) (p : LincheckProof) : (lcFrames kLog p).countP closes = 3 + (kLog - 6) := by
  have hb : ((List.range' 0 (kLog - 6)).flatMap
      (fun j => [Frame.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])).countP closes = (kLog - 6) * 1 :=
    countP_flatMap_range' (kLog - 6) 0 fun i _ _ => by simp [closes, List.countP_cons]
  simp only [lcFrames, List.countP_append, hb]
  simp [closes, List.countP_cons]
  omega

theorem countP_rsFrames {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) : (rsFrames n rs).countP closes = n + 1 := by
  have hb : ((List.range' 0 n).flatMap
      (fun k => [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7])).countP closes = n * 1 :=
    countP_flatMap_range' n 0 fun i _ _ => by simp [closes, List.countP_cons]
  simp only [rsFrames, List.countP_cons, List.countP_append, hb]
  simp [closes, hn]

/-! ## The round that absorbs a recursive cap -/

theorem range'_split {i n : ℕ} (h : i < n) :
    List.range' 0 n = List.range' 0 i ++ i :: List.range' (i + 1) (n - (i + 1)) := by
  obtain ⟨k, rfl⟩ : ∃ k, n = i + (k + 1) := ⟨n - (i + 1), by omega⟩
  rw [← List.range'_append_1, Nat.zero_add, List.range'_succ, show i + (k + 1) - (i + 1) = k by omega]

/-- **A rep's frames split at its `ℓ`-th recursive cap**, for `ℓ + 1` below the schedule's levels: the cap's frame is
followed by a squeeze that closes a round, and `preLig + capRound k0 (ℓ + 1)` squeezes that close one come before it
(zerocheck's `3 + (m − 6)`, lincheck's `3 + (kLog − 6)`, the ring switch's `n + 1`, then Ligerito's). -/
theorem repFrames_capSplit (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (p : Proof) (T : F128)
    (hm : 14 ≤ st.m) (hn : n ≠ 0) (hn0 : sch.n ≠ 0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0)
    {ℓ : ℕ} (hℓ : ℓ + 1 < sch.levels.size) :
    ∃ a b s, repFrames st sch domain n p T = a ++ capFr p.ligerito.recursiveCaps[ℓ]! :: s :: b ∧
      closes s = true ∧
      a.countP closes = (3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (n + 1) + capRound sch.k0 (ℓ + 1) := by
  rcases ℓ with _ | i
  · refine ⟨.dom domain :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
        lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++
        ([Frame.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap] ++
          ([.sq 2 sch.n, .one p.ligerito.oodValues[0]!, .bytes (le64 p.ligerito.claimNonces[0]!), .sq 1 1] ++
            kFr p.ligerito.messages[0]! ++ chFr p.ligerito 1 sch.k0))),
      (.one p.ligerito.oodValues[1]! :: kFr p.ligerito.messages[sch.k0 + 1]! ++
          [.bytes (le64 p.ligerito.claimNonces[1]!), .sq 1 1]) ++
        introFr p.ligerito (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
        queryFr p.ligerito (sch.levels.getD 0 default) 0 0 ++ kFr p.ligerito.messages[sch.k0 + 3]! ++
        [.bytes (le64 p.ligerito.claimNonces[3]!), .sq 1 1] ++
        (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p.ligerito),
      .sq 2 (sch.n - sch.k0 + 1), ?_, by simp [closes], ?_⟩
    · simp only [repFrames, ligFrames, introFr, List.append_assoc, List.cons_append]
    · simp only [List.countP_cons, List.countP_append, countP_zcFrames hm, countP_lcFrames, countP_rsFrames hn,
        countP_kFr, countP_chFr]
      simp [closes, capFr, hn0, capRound]
      omega
  · have hi : i + 1 < sch.levels.size - 1 := by omega
    have hL : levelFr sch p.ligerito i = chFr p.ligerito (sch.k0 + 4 + 7 * i) 4 ++
        capFr p.ligerito.recursiveCaps[i + 1]! ::
          (introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (3 + 2 * i) (sch.k0 + 8 + 7 * i)
              (4 + 3 * i) ++
            introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i)
              (5 + 3 * i) ++
            queryFr p.ligerito (sch.levels.getD (i + 1) default) (1 + i) (1 + i) ++
            kFr p.ligerito.messages[sch.k0 + 10 + 7 * i]! ++
            [.bytes (le64 p.ligerito.claimNonces[6 + 3 * i]!), .sq 1 1]) := by
      unfold levelFr
      rw [ite_eq_left hi]
    have hlev : ((List.range' 0 i).flatMap (levelFr sch p.ligerito)).countP closes = i * 11 :=
      countP_flatMap_range' i 0 fun j _ hj => countP_levelFr sch p.ligerito (by omega) (hq (j + 1) (by omega))
    refine ⟨.dom domain :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
        lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++
        ([Frame.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap] ++
          ([.sq 2 sch.n, .one p.ligerito.oodValues[0]!, .bytes (le64 p.ligerito.claimNonces[0]!), .sq 1 1] ++
            kFr p.ligerito.messages[0]! ++ chFr p.ligerito 1 sch.k0 ++ capFr p.ligerito.recursiveCaps[0]! ::
            (introFr p.ligerito (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++
              introFr p.ligerito (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
              queryFr p.ligerito (sch.levels.getD 0 default) 0 0 ++ kFr p.ligerito.messages[sch.k0 + 3]! ++
              [.bytes (le64 p.ligerito.claimNonces[3]!), .sq 1 1] ++
              (List.range' 0 i).flatMap (levelFr sch p.ligerito) ++ chFr p.ligerito (sch.k0 + 4 + 7 * i) 4)))),
      (.one p.ligerito.oodValues[3 + 2 * i]! :: kFr p.ligerito.messages[sch.k0 + 8 + 7 * i]! ++
          [.bytes (le64 p.ligerito.claimNonces[4 + 3 * i]!), .sq 1 1]) ++
        introFr p.ligerito ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i)
          (5 + 3 * i) ++
        queryFr p.ligerito (sch.levels.getD (i + 1) default) (1 + i) (1 + i) ++
        kFr p.ligerito.messages[sch.k0 + 10 + 7 * i]! ++
        [.bytes (le64 p.ligerito.claimNonces[6 + 3 * i]!), .sq 1 1] ++
        (List.range' (i + 1) (sch.levels.size - 1 - (i + 1))).flatMap (levelFr sch p.ligerito),
      .sq 2 ((sch.levels.getD (i + 1) default).cols + 1), ?_, by simp [closes], ?_⟩
    · rw [repFrames, ligFrames, range'_split (show i < sch.levels.size - 1 by omega), List.flatMap_append,
        List.flatMap_cons, hL]
      simp only [introFr, List.append_assoc, List.cons_append]
    · simp only [List.countP_cons, List.countP_append, countP_zcFrames hm, countP_lcFrames, countP_rsFrames hn,
        countP_kFr, countP_chFr, countP_introFr p.ligerito (show sch.n - sch.k0 + 1 ≠ 0 by omega),
        countP_queryFr p.ligerito (hq 0 (by omega)), hlev]
      simp [closes, capFr, hn0, capRound]
      omega

/-! ## Two reps' frames at the round that absorbs a cap -/

/-- **Two frame lists with one header list, split before a squeeze that closes a round after as many closing squeezes,
whose rounds at the split have the same bytes, split at one frame.** -/
theorem frame_eq_of_chunk {F F' : List Frame} (hhd : F.map Frame.hd = F'.map Frame.hd)
    {a b a' b' : List Frame} {c c' s s' : Frame} (hF : F = a ++ c :: s :: b) (hF' : F' = a' ++ c' :: s' :: b')
    (hc : closes c = false) (hc' : closes c' = false) (hs : closes s = true) (hs' : closes s' = true)
    (hT : a.countP closes = a'.countP closes)
    (he : encs ((chunks F).1.getD (a.countP closes) []) = encs ((chunks F').1.getD (a.countP closes) [])) :
    c = c' := by
  have hch := encs_inj_hd (chunks_getD_hd hhd _) he
  rw [hF, ← chunks_length, chunks_getD_split a b hc hs, hF', chunks_length, hT, ← chunks_length,
    chunks_getD_split a' b' hc' hs'] at hch
  have := (List.append_inj' hch rfl).2
  simp only [List.cons.injEq] at this
  exact this.1

/-- **Two frame lists with one header list, split at caps after as many closing squeezes, whose rounds at the split
have the same bytes, split at one cap.** -/
theorem capFr_eq_of_chunk {F F' : List Frame} (hhd : F.map Frame.hd = F'.map Frame.hd)
    {a b a' b' : List Frame} {c c' : Array ByteArray} {s s' : Frame} (hF : F = a ++ capFr c :: s :: b)
    (hF' : F' = a' ++ capFr c' :: s' :: b') (hs : closes s = true) (hs' : closes s' = true)
    (hT : a.countP closes = a'.countP closes)
    (he : encs ((chunks F).1.getD (a.countP closes) []) = encs ((chunks F').1.getD (a.countP closes) [])) :
    c.foldl (· ++ ·) ByteArray.empty = c'.foldl (· ++ ·) ByteArray.empty := by
  have := frame_eq_of_chunk hhd hF hF' rfl rfl hs hs' hT he
  simpa [capFr] using this

end FlockDeploy.KS
