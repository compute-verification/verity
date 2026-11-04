import Definitions.Hash.Guarantees

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
  · exact Or.inr ⟨x, hx, y, hy, hxy, congrArg Verity.Digest.bytes h⟩

@[simp] theorem bytes_length (d : Digest) : d.bytes.length = 64 := List.length_ofFn

theorem bytes_injective : Function.Injective Verity.Digest.bytes := List.ofFn_injective

theorem bytes_inj {d e : Digest} : d.bytes = e.bytes ↔ d = e := bytes_injective.eq_iff

/-- Two lists of digests whose bytes concatenate to one string are equal. -/
theorem flatMap_bytes_inj : ∀ {ds es : List Digest}, ds.flatMap Verity.Digest.bytes = es.flatMap Verity.Digest.bytes → ds = es
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

/-- Two frame leaves at one rank, in one domain and under one schema, over 64-byte values, have the same value. -/
theorem leafMsg_inj {D : Tree.Domain} {rank : ℕ} {schema : String} {v w : Digest}
    (h : Tree.leafMsg D rank schema v.bytes = Tree.leafMsg D rank schema w.bytes) : v = w := by
  have h := frame_parts h
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, bytes_length, List.append_assoc,
    List.append_cancel_left_eq] at h
  exact bytes_injective h

/-! ## The encodings read back -/

@[simp] theorem be_length (k n : ℕ) : (be k n).length = k := by simp [be]

theorem toUInt8_inj {a b : ℕ} (ha : a < 256) (hb : b < 256) (h : a.toUInt8 = b.toUInt8) : a = b := by
  have := congrArg UInt8.toNat h
  simpa only [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.reducePow, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] using this

theorem be_succ (k n : ℕ) : be (k + 1) n = be k (n / 256) ++ [(n % 256).toUInt8] := by
  simp only [be, List.range_succ, List.map_append, List.map_cons, List.map_nil, Nat.add_sub_cancel,
    Nat.sub_self, pow_zero, Nat.div_one]
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi : i < k := List.mem_range.mp hi
  rw [show k - i = (k - 1 - i) + 1 by omega, pow_succ', Nat.div_div_eq_div_mul]

/-- `be k` determines its number modulo `256^k`. -/
theorem be_inj : ∀ {k a b : ℕ}, be k a = be k b → a % 256 ^ k = b % 256 ^ k
  | 0, _, _, _ => by rw [pow_zero, Nat.mod_one, Nat.mod_one]
  | k + 1, a, b, h => by
    rw [be_succ, be_succ] at h
    obtain ⟨h₁, h₂⟩ := List.append_inj h (by simp)
    have hl := toUInt8_inj (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (List.cons.inj h₂).1
    have hk := be_inj h₁
    rw [pow_succ', Nat.mod_mul, Nat.mod_mul, hl, hk]

theorem be_inj_lt {k a b : ℕ} (ha : a < 256 ^ k) (hb : b < 256 ^ k) (h : be k a = be k b) : a = b := by
  simpa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] using be_inj h

theorem padTo_length (n : ℕ) (m : Msg) : (padTo n m).length = m.length + (n - m.length) := by simp [padTo]

theorem padTo_inj {n : ℕ} {m₁ m₂ : Msg} (hl : m₁.length = m₂.length) (h : padTo n m₁ = padTo n m₂) : m₁ = m₂ :=
  (List.append_inj h hl).1

/-- Two lists of one length that agree on every chunk of `n` are equal. -/
theorem eq_of_chunks {α : Type} (n : ℕ) (hn : 0 < n) {l₁ l₂ : List α} (hl : l₁.length = l₂.length)
    (h : ∀ j < (l₁.length + n - 1) / n, (l₁.drop (n * j)).take n = (l₂.drop (n * j)).take n) : l₁ = l₂ := by
  refine List.ext_getElem hl fun i h₁ h₂ => ?_
  have hj : i / n < (l₁.length + n - 1) / n := by
    have e : (l₁.length + n - 1) / n = (l₁.length - 1) / n + 1 := by
      rw [show l₁.length + n - 1 = (l₁.length - 1) + n by omega, Nat.add_div_right _ hn]
    have := Nat.div_le_div_right (c := n) (show i ≤ l₁.length - 1 by omega)
    omega
  have hc := congrArg (fun l => l[i % n]?) (h (i / n) hj)
  have hin : i % n < n := Nat.mod_lt _ hn
  simp only [List.getElem?_take, hin, ↓reduceIte, List.getElem?_drop, Nat.div_add_mod] at hc
  simpa [List.getElem?_eq_getElem h₁, List.getElem?_eq_getElem h₂] using hc

theorem bitsVal_lt : ∀ l : List Bool, RowSeg.bitsVal l < 2 ^ l.length
  | [] => by simp [RowSeg.bitsVal]
  | b :: bs => by
    have := bitsVal_lt bs
    simp only [RowSeg.bitsVal, List.length_cons, pow_succ]
    split <;> omega

theorem bitsVal_inj : ∀ {l₁ l₂ : List Bool}, l₁.length = l₂.length → RowSeg.bitsVal l₁ = RowSeg.bitsVal l₂ → l₁ = l₂
  | [], [], _, _ => rfl
  | b₁ :: l₁, b₂ :: l₂, hl, h => by
    simp only [RowSeg.bitsVal] at h
    have hb : b₁ = b₂ := by cases b₁ <;> cases b₂ <;> simp at h ⊢ <;> omega
    subst hb
    have : RowSeg.bitsVal l₁ = RowSeg.bitsVal l₂ := by split at h <;> omega
    rw [bitsVal_inj (by simpa using hl) this]

/-- Two rows of one length with the same bytes are the same bits. -/
theorem packBits_inj {l₁ l₂ : Row} (hl : l₁.length = l₂.length) (h : RowSeg.packBits l₁ = RowSeg.packBits l₂) :
    l₁ = l₂ := by
  simp only [RowSeg.packBits, hl] at h
  refine eq_of_chunks 8 (by norm_num) hl fun j hj => ?_
  have hj' : j ∈ List.range ((l₂.length + 7) / 8) := List.mem_range.mpr (by rw [← hl]; omega)
  have hv := (List.map_inj_left.mp h) j hj'
  have hlen : ∀ l : Row, ((l.drop (8 * j)).take 8).length ≤ 8 := fun l => by simp
  have b₁ := lt_of_lt_of_le (bitsVal_lt ((l₁.drop (8 * j)).take 8)) (Nat.pow_le_pow_right (by norm_num) (hlen l₁))
  have b₂ := lt_of_lt_of_le (bitsVal_lt ((l₂.drop (8 * j)).take 8)) (Nat.pow_le_pow_right (by norm_num) (hlen l₂))
  exact bitsVal_inj (by simp [hl]) (toUInt8_inj (by simpa using b₁) (by simpa using b₂) hv)

end Hash.BindingProofs
