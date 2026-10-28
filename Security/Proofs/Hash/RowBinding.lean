import Proofs.Hash.TreeBinding

/-!
# `Hash.SecurityProofs.RowBinding : Hash.Guarantees.RowBinding`

Down from the root, each step gives the next equality or a collision among what the two checks hashed:

1. **The tree** (`Hash.SecurityProofs.TreeBinding`): the two frame leaves' digests are equal.
2. **The frame leaf**: equal digests come from one frame-leaf message, whose value, the hm96 leaf, is then equal
   (`leafMsg_inj`).
3. **The hm96 leaf** (`hm96_bind`): one leaf message gives one commit string `b ‖ c`; one `c` gives one salt; and with
   one salt, one `b = x ⊕ M·y` gives one inner digest `x`.
4. **The row digest** (`inner_bind`). Under `sha512/row/v2` (`rowV2_bind`), one message gives one bit length (`P(ℓ)`) and
   one row's bytes, so one row. Under `sha512/row-seg/v1` (`rowSeg_bind`), one outer message gives one bit length
   (`Q(ℓ)`) and one list of segment digests. Each segment digest comes from one segment message, so from one segment,
   and the segments cover the row.
-/

namespace Hash.BindingProofs

open RowLeaf

/-- Two rows of one length with the same segments are equal. -/
theorem eq_of_segments {r₁ r₂ : Row} (hl : r₁.length = r₂.length)
    (h : ∀ j < RowSeg.segCount r₁.length, RowSeg.segment r₁ j = RowSeg.segment r₂ j) : r₁ = r₂ :=
  eq_of_chunks RowSeg.SEG_BITS (by simp [RowSeg.SEG_BITS]) hl h

theorem lt_256_pow_8 {n : ℕ} (h : n < 2 ^ 64) : n < 256 ^ 8 := lt_of_lt_of_eq h (by norm_num)

/-- **`sha512/row/v2` binds the row.** -/
theorem rowV2_bind (G : Msg → Digest) {r₁ r₂ : Row} (h₁ : r₁.length < 2 ^ 64) (h₂ : r₂.length < 2 ^ 64)
    (h : RowV2.digest G r₁ = RowV2.digest G r₂) : r₁ = r₂ ∨ CollisionIn G (RowV2.hashed r₁ ++ RowV2.hashed r₂) := by
  rcases eq_or_collisionIn (xs := RowV2.hashed r₁ ++ RowV2.hashed r₂) (x := RowV2.msg r₁) (y := RowV2.msg r₂) (by simp [RowV2.hashed])
    (by simp [RowV2.hashed]) h with hm | hc
  · obtain ⟨hp, hb⟩ := List.append_inj hm (by simp [RowV2.rowPrefix, padTo_length])
    have hℓ : r₁.length = r₂.length :=
      be_inj_lt (lt_256_pow_8 h₁) (lt_256_pow_8 h₂) (List.append_cancel_left (padTo_inj (by simp) hp))
    exact Or.inl (packBits_inj hℓ hb)
  · exact Or.inr hc

/-- **`sha512/row-seg/v1` binds the row.** -/
theorem rowSeg_bind (G : Msg → Digest) {r₁ r₂ : Row} (h₁ : r₁.length < 2 ^ 64) (h₂ : r₂.length < 2 ^ 64)
    (h : RowSeg.digest G r₁ = RowSeg.digest G r₂) :
    r₁ = r₂ ∨ CollisionIn G (RowSeg.hashed G r₁ ++ RowSeg.hashed G r₂) := by
  rcases eq_or_collisionIn (xs := RowSeg.hashed G r₁ ++ RowSeg.hashed G r₂) (x := RowSeg.outerMsg G r₁) (y := RowSeg.outerMsg G r₂) (by simp [RowSeg.hashed])
    (by simp [RowSeg.hashed]) h with hm | hc
  swap
  · exact Or.inr hc
  obtain ⟨hp, hd⟩ := List.append_inj hm (by simp [RowSeg.outerPrefix, padTo_length])
  have hℓ : r₁.length = r₂.length := by
    have hq := padTo_inj (by simp) hp
    simp only [List.append_assoc, List.append_cancel_left_eq] at hq
    exact be_inj_lt (lt_256_pow_8 h₁) (lt_256_pow_8 h₂) (List.append_inj hq (by simp)).1
  have hmap : (RowSeg.segmentMsgs r₁).map G = (RowSeg.segmentMsgs r₂).map G :=
    flatMap_bytes_inj (by simpa [List.flatMap_map, Function.comp_def] using hd)
  have hsl : ∀ j, (RowSeg.segment r₁ j).length = (RowSeg.segment r₂ j).length := fun j => by
    simp [RowSeg.segment, hℓ]
  have mem : ∀ (r : Row), ∀ j < RowSeg.segCount r.length, RowSeg.segmentMsg r j ∈ RowSeg.segmentMsgs r :=
    fun r j hj => List.mem_map.mpr ⟨j, List.mem_range.mpr hj, rfl⟩
  by_cases hall : ∀ j < RowSeg.segCount r₁.length, RowSeg.segmentMsg r₁ j = RowSeg.segmentMsg r₂ j
  · refine Or.inl (eq_of_segments hℓ fun j hj => ?_)
    have hs := hall j hj
    simp only [RowSeg.segmentMsg] at hs
    rw [hsl, hℓ] at hs
    exact packBits_inj (hsl j) (List.append_cancel_left hs)
  · simp only [not_forall] at hall
    obtain ⟨j, hj, hne⟩ := hall
    have hj₂ : j < RowSeg.segCount r₂.length := hℓ ▸ hj
    have hG : G (RowSeg.segmentMsg r₁ j) = G (RowSeg.segmentMsg r₂ j) := by
      have := congrArg (fun l => l[j]?) hmap
      simpa [RowSeg.segmentMsgs, hj, hj₂] using this
    refine Or.inr ((eq_or_collisionIn ?_ ?_ hG).resolve_left hne)
    · exact List.mem_append_left _ (List.mem_append_left _ (mem r₁ j hj))
    · exact List.mem_append_right _ (List.mem_append_left _ (mem r₂ j hj₂))

/-- **A format's row digest binds the row.** -/
theorem inner_bind (G : Msg → Digest) (f : Format) {r₁ r₂ : Row} (h₁ : r₁.length < 2 ^ 64)
    (h₂ : r₂.length < 2 ^ 64) (h : f.digest G r₁ = f.digest G r₂) :
    r₁ = r₂ ∨ CollisionIn G (f.hashed G r₁ ++ f.hashed G r₂) := by
  cases f
  · exact rowSeg_bind G h₁ h₂ h
  · exact rowV2_bind G h₁ h₂ h

/-- **The hm96 leaf binds its inner digest.** -/
theorem hm96_bind (H : Msg → Digest) (key : Msg) {x₁ x₂ : Digest} {y₁ y₂ : Msg}
    (h : Hm96.leaf H key x₁ y₁ = Hm96.leaf H key x₂ y₂) :
    x₁ = x₂ ∨ CollisionIn H (Hm96.hashed H key x₁ y₁ ++ Hm96.hashed H key x₂ y₂) := by
  rcases eq_or_collisionIn (xs := Hm96.hashed H key x₁ y₁ ++ Hm96.hashed H key x₂ y₂)
    (x := Hm96.leafPrefix H key ++ Hm96.commitString H key x₁ y₁)
    (y := Hm96.leafPrefix H key ++ Hm96.commitString H key x₂ y₂) (by simp [Hm96.hashed])
    (by simp [Hm96.hashed]) h with hm | hc
  swap
  · exact Or.inr hc
  have hcs := List.append_cancel_left hm
  simp only [Hm96.commitString] at hcs
  obtain ⟨hb, hc⟩ := List.append_inj hcs (by simp)
  rcases eq_or_collisionIn (xs := Hm96.hashed H key x₁ y₁ ++ Hm96.hashed H key x₂ y₂)
    (x := Hm96.saltPrefix ++ y₁) (y := Hm96.saltPrefix ++ y₂) (by simp [Hm96.hashed])
    (by simp [Hm96.hashed]) (bytes_injective hc) with hy | hcoll
  swap
  · exact Or.inr hcoll
  obtain rfl := List.append_cancel_left hy
  refine Or.inl (funext fun i => ?_)
  exact (UInt8.xor_left_inj _).mp (congrFun (List.ofFn_injective hb) i)

end Hash.BindingProofs

open Hash Hash.BindingProofs in
/-- **Row binding** (`Hash.Guarantees.RowBinding`). -/
theorem Hash.SecurityProofs.RowBinding : Hash.Guarantees.RowBinding := by
  intro H G V root rank o₁ o₂ h₁ h₂
  have sub : ∀ (s : RowLeaf.Opening → List Msg), (∀ o, s o ⊆ RowLeaf.outerHashed H G V rank o) →
      s o₁ ++ s o₂ ⊆ RowLeaf.outerHashed H G V rank o₁ ++ RowLeaf.outerHashed H G V rank o₂ :=
    fun s hs z hz => by
      simp only [List.mem_append] at hz ⊢
      exact hz.imp (fun h => hs o₁ h) (fun h => hs o₂ h)
  rcases Hash.SecurityProofs.TreeBinding H V.domain root rank _ _ o₁.sibs o₂.sibs h₁.2.2 h₂.2.2 with hl | hc
  · rcases eq_or_collisionIn
      (xs := RowLeaf.outerHashed H G V rank o₁ ++ RowLeaf.outerHashed H G V rank o₂) (x := RowLeaf.leafMsg H G V rank o₁) (y := RowLeaf.leafMsg H G V rank o₂)
      (by simp [RowLeaf.outerHashed]) (by simp [RowLeaf.outerHashed]) hl with hL | hc
    · rcases hm96_bind H V.key (leafMsg_inj hL) with hx | hc
      · rcases inner_bind G V.format h₁.1 h₂.1 hx with hr | hc
        · exact Or.inl hr
        · exact Or.inr (Or.inr hc)
      · refine Or.inr (Or.inl (collisionIn_mono ?_ hc))
        exact sub (fun o => Hm96.hashed H V.key (V.format.digest G o.row) o.salt)
          fun o z hz => by simp only [RowLeaf.outerHashed, List.mem_append]; tauto
    · exact Or.inr (Or.inl hc)
  · refine Or.inr (Or.inl (collisionIn_mono ?_ hc))
    exact sub (fun o => Tree.hashed H V.domain rank (H (RowLeaf.leafMsg H G V rank o)) o.sibs)
      fun o z hz => by simp only [RowLeaf.outerHashed, List.mem_append]; tauto
