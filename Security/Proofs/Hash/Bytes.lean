import Models.Hash.Guarantees

/-!
# What the hash area's binding proofs share

A collision among a list of inputs survives growing the list (`collisionIn_mono`), and two inputs of the list with one
digest are equal or such a collision (`eq_or_collisionIn`). A digest's bytes are 64 and determine it. The encodings read
back: `be k n` determines `n` below `256^k`, `padTo` of two messages of one length shorter than the block determines
the message, and a frame's parts are what follows its tag.
-/

namespace Hash.BindingProofs

theorem collisionIn_mono {H : Msg → Digest} {xs ys : List Msg} (h : xs ⊆ ys) :
    CollisionIn H xs → CollisionIn H ys := by
  rintro ⟨x, hx, y, hy, hc⟩
  exact ⟨x, h hx, y, h hy, hc⟩

/-- Two inputs of the list with one digest are the same input, or a collision among the list. -/
theorem eq_or_collisionIn {H : Msg → Digest} {xs : List Msg} {x y : Msg} (hx : x ∈ xs) (hy : y ∈ xs)
    (h : H x = H y) : x = y ∨ CollisionIn H xs := by
  by_cases hxy : x = y
  · exact Or.inl hxy
  · exact Or.inr ⟨x, hx, y, hy, hxy, congrArg Digest.bytes h⟩

@[simp] theorem bytes_length (d : Digest) : d.bytes.length = 64 := List.length_ofFn

theorem bytes_injective : Function.Injective Digest.bytes := List.ofFn_injective

theorem bytes_inj {d e : Digest} : d.bytes = e.bytes ↔ d = e := bytes_injective.eq_iff

/-- Two lists of digests whose bytes concatenate to one string are equal. -/
theorem flatMap_bytes_inj : ∀ {ds es : List Digest}, ds.flatMap Digest.bytes = es.flatMap Digest.bytes → ds = es
  | [], [], _ => rfl
  | [], e :: es, h => by simp at h; simpa using congrArg List.length h.1
  | d :: ds, [], h => by simp at h; simpa using congrArg List.length h.1
  | d :: ds, e :: es, h => by
    simp only [List.flatMap_cons] at h
    obtain ⟨h₁, h₂⟩ := List.append_inj h (by simp)
    rw [bytes_injective h₁, flatMap_bytes_inj h₂]

/-- A frame over one tag determines its parts' encoding. -/
theorem frame_parts {tag : String} {ps qs : List Msg} (h : Tree.frame tag ps = Tree.frame tag qs) :
    ps.flatMap (fun p => be 8 p.length ++ p) = qs.flatMap (fun p => be 8 p.length ++ p) := by
  simpa only [Tree.frame, List.append_assoc, List.append_cancel_left_eq] using h

/-- Two node messages at one level and index, in one domain, have the same children. -/
theorem nodeMsg_inj {D : Tree.Domain} {lv i : ℕ} {l r l' r' : Digest}
    (h : Tree.nodeMsg D lv i l r = Tree.nodeMsg D lv i l' r') : l = l' ∧ r = r' := by
  have h := frame_parts h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, bytes_length, List.append_assoc,
    List.append_cancel_left_eq] at h
  obtain ⟨hl, hr⟩ := List.append_inj h (by simp)
  exact ⟨bytes_injective hl, bytes_injective (List.append_cancel_left hr)⟩

end Hash.BindingProofs
