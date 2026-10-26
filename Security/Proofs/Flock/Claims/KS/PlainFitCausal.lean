import Proofs.Flock.Claims.Recursion.NextRFits

/-!
# A fit's round `j` is fixed by the rows through round `j`

`readRep` reads a rep's proof from all its opened rounds, but a prover sends round `j` before it sees the later
coins. Here the frames of round `j` are fixed by the rows through round `j` alone (`fit_chunk`): two fits of rows that
agree through round `j` have one `j`-th chunk.

* **The caps a prefix reads** (`repFrames_split0`, `repFrames_splitS`): a rep's frames start with the header `hdFr`,
  which reads no recursive cap, and then, for each level `i` below the last, a prefix `preFr i` that reads the
  recursive caps `0..i`. The header closes `|schedHd|` chunks (`nCh_hdFr`), and `preFr i` closes `11` more per level
  and `11` more for its own (`nCh_preFr`).
* **The tops the rows fix** (`tops_hd`, `tops_pre`): the schedule's rounds through the one after the header commit one
  top, and those through the one after `preFr i` commit `i + 2`, so rows that agree through round `j` agree on every
  recursive cap a chunk before `j + 1` reads (`topGroups_take`, `topGroups_length`).
* **Skeleton and values** (`chunks_map_skel`, `repFrames_skel`, `RowsAt.vals_eq`): `p` with `p'`'s recursive caps has
  `p'`'s skeleton, and round `j`'s rows fix the values its chunk absorbs.
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Discharge.Exec (chunks closes glue chunks_append repFrames zcFrames lcFrames rsFrames ligFrames
  levelFr introFr queryFr chFr kFr capFr)

/-! ## Chunks of a prefix -/

theorem chunks_fst_append (F G : List Frame) : ∃ Xs, (chunks (F ++ G)).1 = (chunks F).1 ++ Xs := by
  rw [chunks_append]
  unfold glue
  split
  · exact ⟨[], by simp⟩
  · exact ⟨_, rfl⟩

theorem chunks_getD_append {F : List Frame} (G : List Frame) {j : ℕ} (hj : j < (chunks F).1.length) :
    (chunks (F ++ G)).1.getD j [] = (chunks F).1.getD j [] := by
  obtain ⟨Xs, hX⟩ := chunks_fst_append F G
  rw [hX, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_left hj]

theorem roundPat_length (F : List Frame) : (roundPat F).1.length = (chunks F).1.length := by
  simp [roundPat, patOf]

theorem glueP_cons' (a : List Bool × Bool) (c : Bool) (cs : List Bool) (y : Bool) :
    glueP a (c :: cs, y) = (a.1 ++ (a.2 || c) :: cs, y) := rfl

theorem chunks_map_skel : ∀ F : List Frame,
    chunks (F.map skel) = ((chunks F).1.map (List.map skel), (chunks F).2.map skel)
  | [] => rfl
  | f :: fs => by
    rw [List.map_cons]
    simp only [chunks, closes_skel, chunks_map_skel fs]
    cases closes f
    · rcases (chunks fs).1 with _ | ⟨c, cs⟩ <;> rfl
    · rfl

theorem chunks_map_skel_fst (F : List Frame) : (chunks (F.map skel)).1 = (chunks F).1.map (List.map skel) := by
  rw [chunks_map_skel]

theorem flatMap_congr_mem {α β : Type} {f g : α → List β} : ∀ {l : List α}, (∀ a ∈ l, f a = g a) →
    l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a (by simp), flatMap_congr_mem fun b hb => h b (by simp [hb])]

/-! ## A proof with other recursive caps -/

/-- `p` with the recursive caps `rc`. -/
def withRc (p : Flock.Proof) (rc : Array (Array ByteArray)) : Flock.Proof :=
  { p with ligerito := { p.ligerito with recursiveCaps := rc } }

/-- A rep's frames up to level 0's recursive cap: they read no recursive cap. -/
def hdFr (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128) :
    List Frame :=
  .dom d :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
    lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++
    ([.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap] ++
      ([.sq 2 sch.n, .one p.ligerito.oodValues[0]!, .bytes (le64 p.ligerito.claimNonces[0]!), .sq 1 1] ++
        kFr p.ligerito.messages[0]! ++ chFr p.ligerito 1 sch.k0)))

/-- Level 0's frames after its recursive cap. -/
def y0Fr (sch : Flock.Schedule) (p : Flock.LigeritoProof) : List Frame :=
  introFr p (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++ introFr p (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
    queryFr p (sch.levels.getD 0 default) 0 0 ++ kFr p.messages[sch.k0 + 3]! ++
    [.bytes (le64 p.claimNonces[3]!), .sq 1 1]

/-- A rep's frames up to level `i + 1`'s recursive cap: they read the recursive caps `0..i`. -/
def preFr (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128) (i : ℕ) :
    List Frame :=
  hdFr st sch d n p T ++ capFr p.ligerito.recursiveCaps[0]! :: (y0Fr sch p.ligerito ++
    (List.range' 0 i).flatMap (levelFr sch p.ligerito) ++ chFr p.ligerito (sch.k0 + 4 + 7 * i) 4)

theorem repFrames_split0 (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof)
    (T : F128) :
    repFrames st sch d n p T = hdFr st sch d n p T ++ capFr p.ligerito.recursiveCaps[0]! ::
      (y0Fr sch p.ligerito ++ (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p.ligerito)) := by
  simp only [repFrames, ligFrames, hdFr, y0Fr, List.append_assoc, List.cons_append, List.nil_append]

theorem repFrames_splitS (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof)
    (T : F128) {i : ℕ} (hi : i < sch.levels.size - 1) :
    ∃ R, repFrames st sch d n p T = preFr st sch d n p T i ++ R := by
  obtain ⟨k, hk⟩ : ∃ k, sch.levels.size - 1 = i + 1 + k := ⟨sch.levels.size - 1 - (i + 1), by omega⟩
  have hr : List.range' 0 (sch.levels.size - 1) = List.range' 0 i ++ i :: List.range' (i + 1) k := by
    rw [hk, show i + 1 + k = i + (k + 1) by omega, ← List.range'_append_1, List.range'_succ, Nat.zero_add]
  obtain ⟨Xs, hX⟩ : ∃ Xs, levelFr sch p.ligerito i = chFr p.ligerito (sch.k0 + 4 + 7 * i) 4 ++ Xs := ⟨_, rfl⟩
  refine ⟨Xs ++ (List.range' (i + 1) k).flatMap (levelFr sch p.ligerito), ?_⟩
  rw [repFrames_split0, hr, List.flatMap_append, List.flatMap_cons, hX]
  simp only [preFr, List.append_assoc, List.cons_append]

theorem hdFr_withRc (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128)
    (rc : Array (Array ByteArray)) : hdFr st sch d n (withRc p rc) T = hdFr st sch d n p T := by
  unfold hdFr
  rw [chFr_congr (p := (withRc p rc).ligerito) (p' := p.ligerito) rfl]
  all_goals rfl

theorem y0Fr_withRc (sch : Flock.Schedule) (p : Flock.Proof) (rc : Array (Array ByteArray)) :
    y0Fr sch (withRc p rc).ligerito = y0Fr sch p.ligerito := rfl

theorem levelFr_withRc (sch : Flock.Schedule) (p : Flock.Proof) (rc : Array (Array ByteArray)) (i : ℕ)
    (h : i + 1 < sch.levels.size - 1 → rc[i + 1]! = p.ligerito.recursiveCaps[i + 1]!) :
    levelFr sch (withRc p rc).ligerito i = levelFr sch p.ligerito i := by
  have hr : (withRc p rc).ligerito.recursiveCaps = rc := rfl
  unfold levelFr
  rw [chFr_congr (p := (withRc p rc).ligerito) (p' := p.ligerito) rfl]
  split_ifs with hi
  · rw [hr, h hi]
    all_goals rfl
  · rfl

theorem preFr_withRc (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128)
    (rc : Array (Array ByteArray)) (i : ℕ) (h : ∀ t ≤ i, rc[t]! = p.ligerito.recursiveCaps[t]!) :
    preFr st sch d n (withRc p rc) T i = preFr st sch d n p T i := by
  have hl : (List.range' 0 i).flatMap (levelFr sch (withRc p rc).ligerito) =
      (List.range' 0 i).flatMap (levelFr sch p.ligerito) :=
    flatMap_congr_mem fun i' hi' => levelFr_withRc sch p rc i' fun _ =>
      h (i' + 1) (by have := List.mem_range'_1.1 hi'; omega)
  have hr : (withRc p rc).ligerito.recursiveCaps = rc := rfl
  unfold preFr
  rw [hdFr_withRc, y0Fr_withRc, hl, chFr_congr (p := (withRc p rc).ligerito) (p' := p.ligerito) rfl, hr,
    h 0 (Nat.zero_le _)]

/-! ## How many rounds a prefix closes -/

theorem nCh_hdFr {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128)
    (hm : 14 ≤ st.m) (hk : 0 < sch.k0) (hn : sch.n ≠ 0) (hn2 : n ≠ 0) :
    (chunks (hdFr st sch d n p T)).1.length =
      st.m - 6 + 3 + (st.fold.kLog - 6 + 3) + (n + 1) + 2 + sch.k0 := by
  rw [← roundPat_length]
  obtain ⟨j, hj⟩ : ∃ j, sch.k0 = j + 1 := ⟨sch.k0 - 1, by omega⟩
  unfold hdFr
  rw [hj]
  simp only [roundPat_dom, roundPat_append, roundPat_zc hm, roundPat_lc, roundPat_rs hn2, roundPat_label,
    roundPat_one, roundPat_capFr, roundPat_sq _ hn, roundPat_bytes, roundPat_sq _ Nat.one_ne_zero, roundPat_kFr,
    roundPat_chFr, roundPat_nil, glueP_cons, glueP_closed, glueP_nil, Bool.true_or, Bool.false_or, Bool.or_false,
    List.nil_append, List.cons_append, List.append_assoc]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate]
  omega

/-- Pass `i` of the levels' loop, at the passes below the last: eleven rounds, one top. -/
theorem passT_lt (sch : Flock.Schedule) {r i : ℕ} (hc : i + 1 < r) :
    passT sch r i = [(false, []), (true, []), (true, []), (true, [])] ++ lvT sch (i + 1) := by
  simp only [passT, hc, ↓reduceIte]

/-- The tops a round list commits, as a count. -/
def topsN (l : List (Bool × List ℕ)) : ℕ := (l.map fun x => x.2.length).sum

theorem topsN_append (a b : List (Bool × List ℕ)) : topsN (a ++ b) = topsN a + topsN b := by
  simp [topsN]

theorem topsN_take_mono (l : List (Bool × List ℕ)) {a b : ℕ} (h : a ≤ b) : topsN (l.take a) ≤ topsN (l.take b) := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [List.take_add, topsN_append]
  omega

theorem flatG_len (sch : Flock.Schedule) (r : ℕ) : ∀ i, i < r →
    ((List.range' 0 i).flatMap fun i' => (passT sch r i').map Prod.fst).length = 11 * i
  | 0, _ => by simp
  | i + 1, hi => by
    have hc : i + 1 < r := hi
    have hl : ((passT sch r i).map Prod.fst).length = 11 := by simp [passT_lt sch hc, lvT]
    rw [← List.range'_append_1, List.flatMap_append, List.length_append, flatG_len sch r i (by omega), Nat.zero_add,
      List.range'_one, List.flatMap_cons, List.flatMap_nil, List.append_nil, hl] <;> omega

theorem passT_flat (sch : Flock.Schedule) (r : ℕ) : ∀ i, i < r →
    ((List.range' 0 i).flatMap (passT sch r)).length = 11 * i ∧
      topsN ((List.range' 0 i).flatMap (passT sch r)) = i
  | 0, _ => by simp [topsN]
  | i + 1, hi => by
    obtain ⟨h1, h2⟩ := passT_flat sch r i (by omega)
    have hc : i + 1 < r := hi
    have hl : (passT sch r i).length = 11 := by simp [passT_lt sch hc, lvT]
    have ht : topsN (passT sch r i) = 1 := by simp [passT_lt sch hc, lvT, topsN, topsAt]
    rw [← List.range'_append_1, List.flatMap_append, List.length_append, topsN_append, h1, h2, Nat.zero_add,
      List.range'_one, List.flatMap_cons, List.flatMap_nil, List.append_nil, hl, ht]
    exact ⟨by omega, rfl⟩

theorem nCh_preFr {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (n : ℕ) (p : Flock.Proof) (T : F128)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {i : ℕ} (hi : i < sch.levels.size - 1) :
    (chunks (preFr st sch d n p T i)).1.length = (chunks (hdFr st sch d n p T)).1.length + 7 + 11 * i + 4 := by
  have hloop := roundPat_flatMap (F := levelFr sch p.ligerito)
    (G := fun i' => (passT sch (sch.levels.size - 1) i').map Prod.fst) 0 i
    (fun i' _ h => roundPat_levelFr p.ligerito hq (by omega))
  have hfl := flatG_len sch (sch.levels.size - 1) i hi
  rw [← roundPat_length, ← roundPat_length]
  unfold preFr y0Fr
  simp only [roundPat_append, roundPat_capFr, hloop, roundPat_introFr p.ligerito (Nat.add_one_ne_zero _),
    roundPat_queryFr p.ligerito (hq 0 (by omega)), roundPat_kFr, roundPat_bytes, roundPat_sq _ Nat.one_ne_zero,
    roundPat_nil, roundPat_chFr4, glueP_cons, glueP_closed, Bool.true_or, Bool.or_false, Bool.false_or,
    List.nil_append, List.cons_append, List.append_assoc, glueP_cons']
  simp only [List.length_append, List.length_cons, List.length_nil, hfl]
  omega

/-! ## How many tops the schedule's first rounds commit -/

/-- `schedPat`'s rounds before level 0's seven. -/
def hdS (sch : Flock.Schedule) (m kLog claims : ℕ) : List (Bool × List ℕ) :=
  [(false, []), (false, []), (true, [])] ++ List.replicate (m - 5) (true, []) ++ [(false, [])] ++
    List.replicate (kLog - 6) (true, []) ++ [(true, [])] ++ List.replicate claims (true, []) ++ [(false, [])] ++
    [(true, []), (true, [])] ++ List.replicate sch.k0 (true, [])

theorem schedPat_hd (sch : Flock.Schedule) (m kLog claims : ℕ) :
    schedPat sch m kLog claims = hdS sch m kLog claims ++
      (lvT sch 0 ++ (List.range' 0 (sch.levels.size - 1)).flatMap (passT sch (sch.levels.size - 1))) := by
  rw [← List.append_assoc]
  all_goals rfl

theorem hdS_length (sch : Flock.Schedule) (m kLog claims : ℕ) :
    (hdS sch m kLog claims).length = 3 + (m - 5) + 1 + (kLog - 6) + 1 + claims + 1 + 2 + sch.k0 := by
  simp only [hdS, List.length_append, List.length_cons, List.length_nil, List.length_replicate]

theorem topsN_hdS (sch : Flock.Schedule) (m kLog claims : ℕ) : topsN (hdS sch m kLog claims) = 0 := by
  simp [topsN, hdS]

theorem tops_hd (sch : Flock.Schedule) (m kLog claims : ℕ) :
    topsN ((schedPat sch m kLog claims).take ((hdS sch m kLog claims).length + 1)) = 1 := by
  rw [schedPat_hd, List.take_length_add_append, topsN_append, topsN_hdS,
    List.take_append_of_le_length (show 1 ≤ (lvT sch 0).length by simp [lvT])]
  all_goals rfl

theorem tops_pre (sch : Flock.Schedule) (m kLog claims : ℕ) {i : ℕ} (hi : i + 1 < sch.levels.size - 1) :
    topsN ((schedPat sch m kLog claims).take ((hdS sch m kLog claims).length + 7 + 11 * i + 5)) = i + 2 := by
  obtain ⟨k, hk⟩ : ∃ k, sch.levels.size - 1 = i + 1 + k := ⟨sch.levels.size - 1 - (i + 1), by omega⟩
  have hr : List.range' 0 (sch.levels.size - 1) = List.range' 0 i ++ i :: List.range' (i + 1) k := by
    rw [hk, show i + 1 + k = i + (k + 1) by omega, ← List.range'_append_1, List.range'_succ, Nat.zero_add]
  obtain ⟨hl, ht⟩ := passT_flat sch (sch.levels.size - 1) i (by omega)
  have h5 : topsN ((passT sch (sch.levels.size - 1) i).take 5) = 1 := by rw [passT_lt sch hi]; rfl
  have hlen : 5 ≤ (passT sch (sch.levels.size - 1) i).length := by rw [passT_lt sch hi]; simp [lvT]
  have hlv : topsN (lvT sch 0) = 1 := rfl
  have e : (hdS sch m kLog claims).length + 7 + 11 * i + 5 = (hdS sch m kLog claims).length +
      ((lvT sch 0).length + (((List.range' 0 i).flatMap (passT sch (sch.levels.size - 1))).length + 5)) := by
    rw [hl, show (lvT sch 0).length = 7 from rfl] <;> omega
  rw [e, schedPat_hd, hr, List.flatMap_append, List.flatMap_cons, List.take_length_add_append, topsN_append,
    List.take_length_add_append, topsN_append, List.take_length_add_append, topsN_append,
    List.take_append_of_le_length hlen, topsN_hdS, hlv, ht, h5] <;> omega

/-! ## The tops rows fix -/

theorem cutLens_length {α : Type} : ∀ (ns : List ℕ) (l : List α), (cutLens ns l).length = ns.length
  | [], _ => rfl
  | n :: ns, l => by simp [cutLens, cutLens_length ns]

theorem topGroups_take : ∀ (s : List RecVStar.SchedRound) (r : List (List ByteArray)) (k : ℕ),
    ∃ R, topGroups s r = topGroups (s.take k) (r.take k) ++ R
  | s, r, 0 => ⟨topGroups s r, by simp [topGroups]⟩
  | [], r, k + 1 => ⟨[], by simp [topGroups]⟩
  | a :: s, [], k + 1 => ⟨[], by simp [topGroups]⟩
  | a :: s, b :: r, k + 1 => by
    obtain ⟨R, hR⟩ := topGroups_take s r k
    refine ⟨R, ?_⟩
    simp only [topGroups, List.take_succ_cons, List.zip_cons_cons, List.flatMap_cons, List.append_assoc] at hR ⊢
    rw [hR]

theorem topGroups_length : ∀ (s : List RecVStar.SchedRound) (r : List (List ByteArray)), s.length ≤ r.length →
    (topGroups s r).length = topsN (s.map flagTop)
  | [], _, _ => by simp [topGroups, topsN]
  | _ :: _, [], h => by simp at h
  | a :: s, b :: r, h => by
    have ih := topGroups_length s r (by simp at h; omega)
    simp only [topGroups, List.zip_cons_cons, List.flatMap_cons, List.length_append, List.length_map] at ih ⊢
    rw [ih, cutLens_length]
    simp [topsN, flagTop]

theorem rc_agree {a b : Array (Array ByteArray)} {G R R' : List (List ByteArray)}
    (ha : a.toList.map Array.toList = G ++ R) (hb : b.toList.map Array.toList = G ++ R') {t : ℕ}
    (ht : t < G.length) : a[t]! = b[t]! := by
  have h1 : (a.toList.map Array.toList)[t]? = (b.toList.map Array.toList)[t]? := by
    rw [ha, hb, List.getElem?_append_left ht, List.getElem?_append_left ht]
  rw [List.getElem?_map, List.getElem?_map, Array.getElem?_toList, Array.getElem?_toList] at h1
  have hta : t < a.size := by
    have := congrArg List.length ha
    simp only [List.length_map, Array.length_toList, List.length_append] at this
    omega
  have htb : t < b.size := by
    have := congrArg List.length hb
    simp only [List.length_map, Array.length_toList, List.length_append] at this
    omega
  rw [Array.getElem?_eq_getElem hta, Array.getElem?_eq_getElem htb] at h1
  rw [getElem!_pos a t hta, getElem!_pos b t htb]
  simpa [Array.toList_inj] using h1

/-! ## Round `j` -/

/-- **Round `j` of `p` with other recursive caps** is `p`'s where the caps agree on the tops the schedule's rounds
through `j` commit: the chunks before the header's end read no recursive cap, and those before `preFr i`'s end read the
caps `0..i` only. -/
theorem chunk_withRc {c : RepCtx} (hsc : SchedOf c) {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128}
    (h : FitsRep c rows p T) {rc : Array (Array ByteArray)} (hsz : rc.size = p.ligerito.recursiveCaps.size) {j : ℕ}
    (hrc : ∀ t < topsN ((c.sched.take (j + 1)).map flagTop), rc[t]! = p.ligerito.recursiveCaps[t]!) :
    (chunksOf c (withRc p rc) T).getD j [] = (chunksOf c p T).getD j [] := by
  obtain ⟨hs, -, rs, hrs, hsc'⟩ := hsc
  obtain ⟨e, hn, hp⟩ := h.sizes
  obtain ⟨h22, -, hn7, hk0, -⟩ := RecDeploy.fast100_sizes hs
  have hk : 0 < c.sch.k0 := by rw [hk0]; split_ifs <;> omega
  have hq := q_ne_zero hs
  have hrcsz : p.ligerito.recursiveCaps.size = c.sch.levels.size - 1 := hp.rcaps
  have hS : (c.sched.take (j + 1)).map flagTop = (schedPat c.sch c.st.m c.st.fold.kLog c.n).take (j + 1) := by
    rw [List.map_take, hsc', repRounds_flagTop hs hrs]
  rw [hS] at hrc
  obtain ⟨G, hG⟩ : ∃ G, topsN ((schedPat c.sch c.st.m c.st.fold.kLog c.n).take (j + 1)) = G := ⟨_, rfl⟩
  rw [hG] at hrc
  have hd0 := nCh_hdFr (st := c.st) (sch := c.sch) c.dom c.n p T (by omega) hk (by omega) (by omega)
  have hhd := hdS_length c.sch c.st.m c.st.fold.kLog c.n
  unfold chunksOf
  by_cases hGL : c.sch.levels.size - 1 ≤ G
  · have he : rc = p.ligerito.recursiveCaps := by
      refine Array.ext hsz fun t h1 h2 => ?_
      have := hrc t (by omega)
      rwa [getElem!_pos rc t h1, getElem!_pos p.ligerito.recursiveCaps t h2] at this
    subst he
    rfl
  · rcases Nat.eq_zero_or_pos G with hG0 | hGpos
    · have hj : j < (chunks (hdFr c.st c.sch c.dom c.n p T)).1.length := by
        by_contra hn'
        have h1 := tops_hd c.sch c.st.m c.st.fold.kLog c.n
        have h2 := topsN_take_mono (schedPat c.sch c.st.m c.st.fold.kLog c.n)
          (show (hdS c.sch c.st.m c.st.fold.kLog c.n).length + 1 ≤ j + 1 by omega)
        omega
      rw [repFrames_split0, repFrames_split0, hdFr_withRc, chunks_getD_append _ hj, chunks_getD_append _ hj]
    · obtain ⟨i, hi⟩ : ∃ i, G = i + 1 := ⟨G - 1, by omega⟩
      have hiL : i < c.sch.levels.size - 1 := by omega
      have hi1 : i + 1 < c.sch.levels.size - 1 := by omega
      have hdi := nCh_preFr (st := c.st) (sch := c.sch) c.dom c.n p T hq hiL
      have hj : j < (chunks (preFr c.st c.sch c.dom c.n p T i)).1.length := by
        by_contra hn'
        have h1 := tops_pre c.sch c.st.m c.st.fold.kLog c.n hi1
        have h2 := topsN_take_mono (schedPat c.sch c.st.m c.st.fold.kLog c.n)
          (show (hdS c.sch c.st.m c.st.fold.kLog c.n).length + 7 + 11 * i + 5 ≤ j + 1 by omega)
        omega
      obtain ⟨R, hR⟩ := repFrames_splitS c.st c.sch c.dom c.n p T hiL
      obtain ⟨R', hR'⟩ := repFrames_splitS c.st c.sch c.dom c.n (withRc p rc) T hiL
      rw [hR, hR', preFr_withRc c.st c.sch c.dom c.n p T rc i (fun t ht => hrc t (by omega)),
        chunks_getD_append _ hj, chunks_getD_append _ hj]

/-- **A fit's round `j` is fixed by the rows through round `j`**: two fits of one rep's rows that agree through round
`j` have one `j`-th chunk. `p` with `p'`'s recursive caps has `p`'s `j`-th chunk (`chunk_withRc`: the caps agree on
the tops the rounds through `j` commit) and `p'`'s skeleton (`repFrames_skel`), and round `j`'s rows fix its values
(`RowsAt.vals_eq`). -/
theorem fit_chunk {c : RepCtx} (hsc : SchedOf c) {rows rows' : List (List ByteArray)} {p p' : Flock.Proof}
    {T T' : F128} (h : FitsRep c rows p T) (h' : FitsRep c rows' p' T') {j : ℕ} (hj : j < rows.length)
    (hpre : rows.take (j + 1) = rows'.take (j + 1)) :
    (chunksOf c p T).getD j [] = (chunksOf c p' T').getD j [] := by
  obtain ⟨e, hn, hp⟩ := h.sizes
  obtain ⟨e', hn', hp'⟩ := h'.sizes
  have hee : e' = e := by omega
  rw [hee] at hp'
  have hj' : j < rows'.length := by
    have := congrArg List.length hpre
    simp only [List.length_take] at this
    omega
  have hrow : rows.getD j [] = rows'.getD j [] := by
    have := congrArg (fun l : List (List ByteArray) => l[j]?) hpre
    simp only [List.getElem?_take, Nat.lt_add_one, ↓reduceIte] at this
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, this]
  obtain ⟨R, hR⟩ := topGroups_take c.sched rows (j + 1)
  obtain ⟨R', hR'⟩ := topGroups_take c.sched rows' (j + 1)
  rw [← hpre] at hR'
  have hGl : (topGroups (c.sched.take (j + 1)) (rows.take (j + 1))).length =
      topsN ((c.sched.take (j + 1)).map flagTop) :=
    topGroups_length _ _ (by simp only [List.length_take]; have := h.sched; omega)
  have hrc : ∀ t < topsN ((c.sched.take (j + 1)).map flagTop),
      p'.ligerito.recursiveCaps[t]! = p.ligerito.recursiveCaps[t]! :=
    fun t ht => rc_agree (h'.tops.trans hR') (h.tops.trans hR) (by omega)
  have hsz : p'.ligerito.recursiveCaps.size = p.ligerito.recursiveCaps.size := hp'.rcaps.trans hp.rcaps.symm
  have hA := chunk_withRc hsc h hsz hrc
  have hpq : ProofSizes (FlockSoundness.Refine.shapeOf c.st c.sch e) (withRc p p'.ligerito.recursiveCaps) :=
    ⟨hp.r1ab, hp.r1c, hp.zcRounds, hp.lcRounds, hp.zPartial, hp.rs, hp.rsEach, hp.cap, hp.capEach, hp'.rcaps,
      hp'.rcapsEach, hp.yr, hp.messages, hp.queryNonces, hp.oodValues, hp.claimNonces, hp.consistencyNonces⟩
  have hcap : (withRc p p'.ligerito.recursiveCaps).cap = p'.cap := Array.toList_inj.1 (h.cap.trans h'.cap.symm)
  have hqn := zeros_eq (hp.queryNonces.trans hp'.queryNonces.symm) h.nonces.1 h'.nonces.1
  have hcn := zeros_eq (hp.claimNonces.trans hp'.claimNonces.symm) h.nonces.2.1 h'.nonces.2.1
  have hsn := zeros_eq (hp.consistencyNonces.trans hp'.consistencyNonces.symm) h.nonces.2.2 h'.nonces.2.2
  have hsk := repFrames_skel (st := c.st) (sch := c.sch) (d := c.dom) (n := c.n) (T := T) (T' := T') hpq hp' hcap
    rfl hcn hqn hsn
  have hsk' : ((chunksOf c (withRc p p'.ligerito.recursiveCaps) T).getD j []).map skel =
      ((chunksOf c p' T').getD j []).map skel := by
    unfold chunksOf
    rw [getD_map_skel, getD_map_skel, ← chunks_map_skel_fst, ← chunks_map_skel_fst, hsk]
  have hv : vals ((chunksOf c (withRc p p'.ligerito.recursiveCaps) T).getD j []) =
      vals ((chunksOf c p' T').getD j []) := by
    rw [hA]
    have h2 := h'.elems j hj'
    rw [← hrow] at h2
    exact (h.elems j hj).vals_eq h2 (by rw [← hA, ← vals_length_skel, hsk', vals_length_skel])
  rw [← hA]
  exact eq_of_skel_vals hsk' hv

end FlockDeploy
