import Proofs.Flock.Claims.KS.ConflictMerkle

/-!
# Two passing Merkle checks at one cap and positions open alike, or collide

`Flock.opens_binding` collides two openings of one position against one cap whose rows differ. A rep's openings also
carry salts and siblings, and two reps the verifier accepts on one stream agree in everything but their openings
(`RepFramesLig.rep_bind_lig`), so what the calls half needs is the whole opening: two accepted openings that differ in
their row, their salt or their siblings collide among their SHA-512 inputs (`opens_coll_of`), under either scheme the
verifier admits for its tables.

* Different leaves climb to one cap entry: the climb's extractor collides (`climb_coll`).
* One leaf, different rows or salts: under `hm96-sha512/v1` the row digests or `Hm96.leafPair`'s inputs collide
  (`leaf_coll`); a plain leaf ignores its salt, which `opens` forces empty.
* One leaf, one row and salt, different siblings: the first node where the paths part (`climb_sibs_coll`).

Two passing Merkle checks at one cap and positions then carry the same rows, siblings and salts, or some query's two
openings collide (`merkleCheck_coll`), and a collision among two lists of inputs that a decision's calls log is a
collision among those calls (`CollIn.tied`).
-/

namespace FlockDeploy.KS

open Flock

attribute [local irreducible] Sha512.hash

/-- Two lists of SHA-512 inputs collide across: an input of the first and one of the second differ and hash alike. -/
def CollIn (A B : List ByteArray) : Prop := ∃ u ∈ A, ∃ v ∈ B, u ≠ v ∧ Sha512.hash u = Sha512.hash v

theorem CollIn.mono {A B A' B' : List ByteArray} (h : CollIn A B) (hA : ∀ x ∈ A, x ∈ A')
    (hB : ∀ x ∈ B, x ∈ B') : CollIn A' B' := by
  obtain ⟨u, hu, v, hv, hne, heq⟩ := h
  exact ⟨u, hA u hu, v, hB v hv, hne, heq⟩

/-- **A collision across two logged lists is a collision among the calls**: each list's inputs are inputs of `l`. -/
theorem CollIn.tied {A B : List ByteArray} {l : List (Claims.Msg × Claims.Msg)} (h : CollIn A B)
    (hA : ∀ b ∈ A, Claims.toMsg b ∈ l.map Prod.fst) (hB : ∀ b ∈ B, Claims.toMsg b ∈ l.map Prod.fst) :
    Claims.TiedColl sha512 l := by
  obtain ⟨u, hu, v, hv, hne, heq⟩ := h
  refine ⟨_, _, ⟨fun e => hne (toMsg_injective e), ?_⟩, hA _ hu, hB _ hv⟩
  show Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _))) =
    Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _)))
  rw [Claims.ofMsg_toMsg, Claims.ofMsg_toMsg, heq]

/-! ## The climb -/

/-- One step of a climb, as `climbIns` logs it. -/
theorem climb_cons (acc s : ByteArray) (idx : ℕ) (rest : List ByteArray) :
    climb MerkleScheme.hm96Sha512 acc idx (s :: rest) =
      climb MerkleScheme.hm96Sha512 (Sha512.hash (if idx % 2 = 0 then acc ++ s else s ++ acc)) (idx / 2) rest := by
  by_cases hp : idx % 2 = 0
  · simp only [climb, hp, ↓reduceIte]
    rfl
  · simp only [climb, hp, ↓reduceIte]
    rfl

/-- **Different starts that climb to one digest collide on the way.** -/
theorem climb_coll (s1 s2 : List ByteArray) (a b : ByteArray) (idx : ℕ) (hl : s1.length = s2.length)
    (hw1 : ∀ x ∈ s1, x.size = 64) (hw2 : ∀ x ∈ s2, x.size = 64) (ha : a.size = 64) (hb : b.size = 64)
    (hab : a ≠ b) (h : climb MerkleScheme.hm96Sha512 a idx s1 = climb MerkleScheme.hm96Sha512 b idx s2) :
    CollIn (climbIns a idx s1) (climbIns b idx s2) := by
  have hc := climb_binding MerkleScheme.hm96Sha512 MerkleScheme.hm96Sha512_sized s1 s2 a b idx hl hw1 hw2 ha hb hab h
  obtain ⟨hne, heq⟩ := MerklePair.hm96_inputs _ hc
  obtain ⟨m1, m2⟩ := climbPair_mem s1 s2 a b idx hc
  exact ⟨_, m1, _, m2, hne, heq⟩

/-- The two children of a node, as `climbIns` concatenates them, are two different inputs when the siblings differ. -/
theorem step_ne {a s t : ByteArray} (idx : ℕ) (hs : s.size = 64) (ht : t.size = 64) (hst : s ≠ t) :
    (if idx % 2 = 0 then a ++ s else s ++ a) ≠ (if idx % 2 = 0 then a ++ t else t ++ a) := by
  by_cases hp : idx % 2 = 0
  · simp only [hp, ↓reduceIte]
    exact fun e => hst ((ByteArray.append_right_inj a).mp e)
  · simp only [hp, ↓reduceIte]
    exact fun e => hst (ByteArray.append_inj_left e (by rw [hs, ht]))

/-- **One start, different siblings, one digest: the paths collide where they part.** -/
theorem climb_sibs_coll : ∀ (s1 s2 : List ByteArray) (a : ByteArray) (idx : ℕ), s1.length = s2.length →
    (∀ x ∈ s1, x.size = 64) → (∀ x ∈ s2, x.size = 64) → s1 ≠ s2 →
    climb MerkleScheme.hm96Sha512 a idx s1 = climb MerkleScheme.hm96Sha512 a idx s2 →
    CollIn (climbIns a idx s1) (climbIns a idx s2)
  | [], [], _, _, _, _, _, hne, _ => (hne rfl).elim
  | [], _ :: _, _, _, hl, _, _, _, _ => by simp at hl
  | _ :: _, [], _, _, hl, _, _, _, _ => by simp at hl
  | s :: rest, t :: rest2, a, idx, hl, hw1, hw2, hne, h => by
    have hl' : rest.length = rest2.length := by simpa using hl
    have hr1 : ∀ x ∈ rest, x.size = 64 := fun x hx => hw1 x (List.mem_cons_of_mem _ hx)
    have hr2 : ∀ x ∈ rest2, x.size = 64 := fun x hx => hw2 x (List.mem_cons_of_mem _ hx)
    rw [climb_cons, climb_cons] at h
    show CollIn ((if idx % 2 = 0 then a ++ s else s ++ a) ::
        climbIns (Sha512.hash (if idx % 2 = 0 then a ++ s else s ++ a)) (idx / 2) rest)
      ((if idx % 2 = 0 then a ++ t else t ++ a) ::
        climbIns (Sha512.hash (if idx % 2 = 0 then a ++ t else t ++ a)) (idx / 2) rest2)
    by_cases hst : s = t
    · subst hst
      have hrest : rest ≠ rest2 := fun e => hne (by rw [e])
      exact (climb_sibs_coll rest rest2 _ (idx / 2) hl' hr1 hr2 hrest h).mono
        (fun x hx => List.mem_cons_of_mem _ hx) (fun x hx => List.mem_cons_of_mem _ hx)
    · have hu := step_ne (a := a) idx (hw1 s List.mem_cons_self) (hw2 t List.mem_cons_self) hst
      by_cases hn : Sha512.hash (if idx % 2 = 0 then a ++ s else s ++ a) =
          Sha512.hash (if idx % 2 = 0 then a ++ t else t ++ a)
      · exact ⟨_, List.mem_cons_self, _, List.mem_cons_self, hu, hn⟩
      · exact (climb_coll rest rest2 _ _ (idx / 2) hl' hr1 hr2 (Sha512.hash_size _) (Sha512.hash_size _) hn h).mono
          (fun x hx => List.mem_cons_of_mem _ hx) (fun x hx => List.mem_cons_of_mem _ hx)

/-! ## The leaf -/

/-- **An `hm96-sha512/v1` leaf binds its row and its salt** among its SHA-512 inputs. -/
theorem leaf_coll {r1 r2 : Array F128} {y1 y2 : ByteArray} (hne : (r1, y1) ≠ (r2, y2))
    (h : MerkleScheme.hm96Sha512.leaf r1 y1 = MerkleScheme.hm96Sha512.leaf r2 y2) :
    CollIn (leafIns r1 y1) (leafIns r2 y2) := by
  by_cases hxy : (Sha512.hash (MerkleScheme.rowBytes r1), y1) = (Sha512.hash (MerkleScheme.rowBytes r2), y2)
  · obtain ⟨hx, rfl⟩ := Prod.mk.inj hxy
    have hr : r1 ≠ r2 := fun e => hne (by rw [e])
    exact ⟨_, List.mem_cons_self, _, List.mem_cons_self, fun e => hr (rowBytes_inj e), hx⟩
  · have hc := Hm96.leafPair_collides Hm96.Default512.leafPrefix Hm96.Default512.saltPrefix Hm96.Default512.mask
      (Sha512.hash (MerkleScheme.rowBytes r1)) (Sha512.hash (MerkleScheme.rowBytes r2)) y1 y2 (Sha512.hash_size _)
      (Sha512.hash_size _) hxy h
    refine ⟨_, ?_, _, ?_, hc.1, hc.2⟩
    · unfold Hm96.leafPair
      dsimp only
      split
      · exact List.mem_cons_of_mem _ List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)
    · unfold Hm96.leafPair
      dsimp only
      split
      · exact List.mem_cons_of_mem _ List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)

/-! ## The opening -/

theorem eq_empty_of_size {b : ByteArray} (h : b.size = 0) : b = ByteArray.empty := by
  cases b with
  | mk data =>
    have : data = #[] := Array.eq_empty_of_size_eq_zero h
    subst this
    rfl

/-- **Two accepted `hm96-sha512/v1` openings of one position against one cap that differ in their row, salt or
siblings collide among their SHA-512 inputs.** -/
theorem opens_coll_hm96 {cap : Array ByteArray} {c d lanes pos : ℕ} {r1 r2 : Array F128} {y1 y2 : ByteArray}
    {s1 s2 : List ByteArray} (hw1 : ∀ x ∈ s1, x.size = 64) (hw2 : ∀ x ∈ s2, x.size = 64)
    (h1 : opens MerkleScheme.hm96Sha512 cap c d lanes pos r1 y1 s1 = true)
    (h2 : opens MerkleScheme.hm96Sha512 cap c d lanes pos r2 y2 s2 = true) (hne : (r1, y1, s1) ≠ (r2, y2, s2)) :
    CollIn (opensIns r1 y1 pos s1) (opensIns r2 y2 pos s2) := by
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq := h1.2.trans h2.2.symm
  have hl : s1.length = s2.length := by rw [h1.1.1.2, h2.1.1.2]
  have hsz := MerkleScheme.hm96Sha512_sized.2
  by_cases hlf : MerkleScheme.hm96Sha512.leaf r1 y1 = MerkleScheme.hm96Sha512.leaf r2 y2
  · by_cases hry : (r1, y1) = (r2, y2)
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hry
      have hs : s1 ≠ s2 := fun e => hne (by rw [e])
      exact (climb_sibs_coll s1 s2 _ pos hl hw1 hw2 hs heq).mono
        (fun x hx => List.mem_append_right _ hx) (fun x hx => List.mem_append_right _ hx)
    · exact (leaf_coll hry hlf).mono (fun x hx => List.mem_append_left _ hx) (fun x hx => List.mem_append_left _ hx)
  · exact (climb_coll s1 s2 _ _ pos hl hw1 hw2 (hsz _ _) (hsz _ _) hlf heq).mono
      (fun x hx => List.mem_append_right _ hx) (fun x hx => List.mem_append_right _ hx)

/-- **The same under plain SHA-512 leaves**, whose salts `opens` forces empty. -/
theorem opens_coll_plain {cap : Array ByteArray} {c d lanes pos : ℕ} {r1 r2 : Array F128} {y1 y2 : ByteArray}
    {s1 s2 : List ByteArray} (hw1 : ∀ x ∈ s1, x.size = 64) (hw2 : ∀ x ∈ s2, x.size = 64)
    (h1 : opens (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") cap c d lanes pos r1 y1 s1 = true)
    (h2 : opens (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") cap c d lanes pos r2 y2 s2 = true)
    (hne : (r1, y1, s1) ≠ (r2, y2, s2)) : CollIn (opensInsP r1 y1 pos s1) (opensInsP r2 y2 pos s2) := by
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  rw [climb_sha512] at h1 h2
  have heq := h1.2.trans h2.2.symm
  have hl : s1.length = s2.length := by rw [h1.1.1.2, h2.1.1.2]
  have hy : y1 = y2 := (eq_empty_of_size h1.1.2).trans (eq_empty_of_size h2.1.2).symm
  subst hy
  by_cases hlf : Sha512.hash (MerkleScheme.rowBytes r1) = Sha512.hash (MerkleScheme.rowBytes r2)
  · by_cases hr : r1 = r2
    · subst hr
      have hs : s1 ≠ s2 := fun e => hne (by rw [e])
      exact (climb_sibs_coll s1 s2 _ pos hl hw1 hw2 hs heq).mono
        (fun x hx => List.mem_append_right _ hx) (fun x hx => List.mem_append_right _ hx)
    · exact ⟨_, List.mem_append_left _ List.mem_cons_self, _, List.mem_append_left _ List.mem_cons_self,
        fun e => hr (rowBytes_inj e), hlf⟩
  · exact (climb_coll s1 s2 _ _ pos hl hw1 hw2 (Sha512.hash_size _) (Sha512.hash_size _) hlf heq).mono
      (fun x hx => List.mem_append_right _ hx) (fun x hx => List.mem_append_right _ hx)

/-- **Either admitted scheme.** -/
theorem opens_coll_of {ms : MerkleScheme}
    (hms : ms = MerkleScheme.hm96Sha512 ∨ ms = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    {cap : Array ByteArray} {c d lanes pos : ℕ} {r1 r2 : Array F128} {y1 y2 : ByteArray} {s1 s2 : List ByteArray}
    (hw1 : ∀ x ∈ s1, x.size = ms.digestLen) (hw2 : ∀ x ∈ s2, x.size = ms.digestLen)
    (h1 : opens ms cap c d lanes pos r1 y1 s1 = true) (h2 : opens ms cap c d lanes pos r2 y2 s2 = true)
    (hne : (r1, y1, s1) ≠ (r2, y2, s2)) : CollIn (opensInsOf ms r1 y1 pos s1) (opensInsOf ms r2 y2 pos s2) := by
  rcases hms with rfl | rfl
  · rw [opensInsOf_hm96]
    exact opens_coll_hm96 hw1 hw2 h1 h2 hne
  · rw [opensInsOf_plain]
    exact opens_coll_plain hw1 hw2 h1 h2 hne

/-! ## The Merkle check -/

/-- An accepted Merkle check has its shape: the cap's, rows', siblings' and salts' counts. -/
theorem merkleCheck_shape {ms : MerkleScheme} {cap : Array ByteArray} {c d : ℕ} {positions : Array ℕ}
    {rows : Array (Array F128)} {siblings salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheck ms cap c d positions rows siblings salts lanes = .ok ()) :
    cap.size = 2 ^ c ∧ rows.size = positions.size ∧ siblings.size = positions.size * (d - c) ∧
      salts.size = saltCount ms positions.size := by
  unfold merkleCheck at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
  · rename_i hs
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hs
    exact ⟨hs.1.1.1, hs.1.1.2, hs.1.2, hs.2⟩

/-- Siblings of `n` queries, `w` each, are determined by each query's `w`. -/
theorem sibs_ext {a b : Array ByteArray} {n w : ℕ} (ha : a.size = n * w) (hb : b.size = n * w)
    (h : ∀ j < n, sibsOf a w j = sibsOf b w j) : a = b := by
  apply Array.ext (ha.trans hb.symm)
  intro i hi hi'
  have hw : 0 < w := by
    rcases Nat.eq_zero_or_pos w with hw | hw
    · rw [hw, Nat.mul_zero] at ha
      omega
    · exact hw
  have hj : i / w < n := by
    rw [Nat.div_lt_iff_lt_mul hw]
    omega
  have hk : i % w < w := Nat.mod_lt _ hw
  have hi2 : i / w * w + i % w = i := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod i w
  have e := h _ hj
  unfold sibsOf at e
  have hlen : i % w < (a.extract (i / w * w) (i / w * w + w)).toList.length := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have := List.getElem_of_eq e hlen
  simp only [Array.getElem_toList, Array.getElem_extract] at this
  simpa only [hi2] using this

/-- **Two passing Merkle checks at one cap and positions open alike, or some query's openings collide**, under either
admitted scheme. -/
theorem merkleCheck_coll {ms : MerkleScheme}
    (hms : ms = MerkleScheme.hm96Sha512 ∨ ms = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    {cap : Array ByteArray} {c d : ℕ} {positions : Array ℕ} {rows1 rows2 : Array (Array F128)}
    {sib1 sib2 salt1 salt2 : Array ByteArray} {lanes : ℕ}
    (h1 : merkleCheck ms cap c d positions rows1 sib1 salt1 lanes = .ok ())
    (h2 : merkleCheck ms cap c d positions rows2 sib2 salt2 lanes = .ok ()) :
    (rows1 = rows2 ∧ sib1 = sib2 ∧ salt1 = salt2) ∨
      ∃ j < positions.size, CollIn
        (opensInsOf ms rows1[j]! (salt1.getD j ByteArray.empty) positions[j]! (sibsOf sib1 (d - c) j))
        (opensInsOf ms rows2[j]! (salt2.getD j ByteArray.empty) positions[j]! (sibsOf sib2 (d - c) j)) := by
  by_cases hall : ∀ j < positions.size, (rows1[j]!, salt1.getD j ByteArray.empty, sibsOf sib1 (d - c) j) =
      (rows2[j]!, salt2.getD j ByteArray.empty, sibsOf sib2 (d - c) j)
  · left
    obtain ⟨_, hr1, hs1, hy1⟩ := merkleCheck_shape h1
    obtain ⟨_, hr2, hs2, hy2⟩ := merkleCheck_shape h2
    refine ⟨?_, sibs_ext hs1 hs2 fun j hj => (Prod.mk.inj (Prod.mk.inj (hall j hj)).2).2, ?_⟩
    · apply Array.ext (hr1.trans hr2.symm)
      intro i hi hi'
      have e := (Prod.mk.inj (hall i (hr1 ▸ hi))).1
      rwa [getElem!_pos rows1 i hi, getElem!_pos rows2 i hi'] at e
    · apply Array.ext (hy1.trans hy2.symm)
      intro i hi hi'
      have hn : salt1.size ≤ positions.size := by
        rw [hy1]
        unfold saltCount
        split <;> omega
      have e := (Prod.mk.inj (Prod.mk.inj (hall i (by omega))).2).1
      simp only [Array.getD, hi, hi', ↓reduceDIte] at e
      exact e
  · right
    push Not at hall
    obtain ⟨j, hj, hne⟩ := hall
    exact ⟨j, hj, opens_coll_of hms (merkleCheck_sibs ms cap c d positions rows1 sib1 salt1 lanes h1 j)
      (merkleCheck_sibs ms cap c d positions rows2 sib2 salt2 lanes h2 j)
      (merkleCheck_opens ms cap c d positions rows1 sib1 salt1 lanes h1 j hj)
      (merkleCheck_opens ms cap c d positions rows2 sib2 salt2 lanes h2 j hj) hne⟩

end FlockDeploy.KS
