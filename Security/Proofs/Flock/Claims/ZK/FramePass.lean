import Proofs.Flock.Claims.ZK.FrameShape

/-!
# A device round that passes the frame's check

The frame passes a device round `v` in a scheduled round `r`'s place when `v` is `r` byte for byte outside the words
`roundFree` frees, the values `r` absorbs in the algebraic phase (`Flock.Firewall.Frame.passes`). With `r` the bytes of
frames `fs` and its squeeze, this file shows `v` is the bytes of frames alike to `fs` (`Alike`) and the same squeeze
(`passes_frames`), so that the frame's round at `v` is its round at `r` but for the values (`roundFree_alike`):

* the free words are the frames' values (`roundFree_free`, over `stepAt_free` and `foldlM_absorb1_free`);
* any bytes agreeing with frames' outside their values are alike frames' (`encs_refill`, `frame_refill`), since any 16
  bytes are a value's as the frame reads them (`toBytes_readWord`, `u64OfLE_byte`).
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 u64OfLE)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree inFree firstDiff passes)
open FlockSoundness.Refine (Frame encs colBytes toBytes_length)
open FlockSoundness.Discharge.ZkView (get!_eq u64OfLE_eq)

/-! ## Any 16 bytes are a value's -/

-- new

theorem toUInt64_8 (k : ℕ) (hk : k < 8) : (8 * k).toUInt64.toNat = 8 * k := by
  rw [Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' (by simp [UInt64.size]; omega)]

theorem foldl_bytes_testBit (d : ByteArray) (o : ℕ) : ∀ (k : ℕ) (_ : k ≤ 8) (j : ℕ),
    ((List.range' 0 k).foldl (fun acc i => acc ||| (d.get! (o + i)).toUInt64 <<< (8 * i).toUInt64) 0).toNat.testBit j =
      (decide (j < 8 * k) && (d.get! (o + j / 8)).toNat.testBit (j % 8))
  | 0, _, j => by simp
  | k + 1, hk, j => by
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, UInt64.toNat_or, Nat.testBit_or,
      foldl_bytes_testBit d o k (by omega) j, show 0 + 1 * k = k by omega, UInt64.toNat_shiftLeft,
      toUInt64_8 k (by omega), UInt8.toNat_toUInt64, Nat.mod_eq_of_lt (show 8 * k < 64 by omega),
      Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
    rcases Nat.lt_or_ge j (8 * k) with h | h
    · simp [h, show j < 8 * (k + 1) by omega, show ¬ 8 * k ≤ j by omega]
    rcases Nat.lt_or_ge j (8 * (k + 1)) with h' | h'
    · simp [h', show ¬ j < 8 * k by omega, show j < 64 by omega, show j / 8 = k by omega,
        show j % 8 = j - 8 * k by omega, h]
    · have hb : (d.get! (o + k)).toNat < 2 ^ (j - 8 * k) :=
        Nat.lt_of_lt_of_le (d.get! (o + k)).toNat_lt (Nat.pow_le_pow_right (by omega) (by omega))
      simp [show ¬ j < 8 * k by omega, show ¬ j < 8 * (k + 1) by omega, Nat.testBit_lt_two_pow hb]

/-- **The frame's read of a word gives its bytes back**: byte `i` of `u64OfLE d o` is `d`'s byte `o + i`. -/
theorem u64OfLE_byte (d : ByteArray) (o i : ℕ) (hi : i < 8) :
    (u64OfLE d o >>> (8 * i).toUInt64).toUInt8 = d.get! (o + i) := by
  apply UInt8.toNat_inj.1
  rw [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, toUInt64_8 i hi, Nat.mod_eq_of_lt (show 8 * i < 64 by omega)]
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftRight, u64OfLE_eq, foldl_bytes_testBit d o 8 (Nat.le_refl _)]
  rcases Nat.lt_or_ge j 8 with h | h
  · simp [h, show 8 * i + j < 64 by omega, show (8 * i + j) / 8 = i by omega, show (8 * i + j) % 8 = j by omega]
  · have hb : (d.get! (o + i)).toNat < 2 ^ j :=
      Nat.lt_of_lt_of_le (d.get! (o + i)).toNat_lt (Nat.pow_le_pow_right (by omega) h)
    simp [show ¬ j < 8 by omega, Nat.testBit_lt_two_pow hb]

/-- The value whose bytes are `w`'s first sixteen, as the frame reads it. -/
def readWord (w : List UInt8) : F128 := ⟨u64OfLE (ByteArray.mk w.toArray) 0, u64OfLE (ByteArray.mk w.toArray) 8⟩

theorem list_ext_get! {a b : List UInt8} (hl : a.length = b.length) (h : ∀ i < a.length, a[i]! = b[i]!) : a = b := by
  apply List.ext_getElem hl
  intro i h1 h2
  have := h i h1
  rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

/-- **Any sixteen bytes are a value's.** -/
theorem toBytes_readWord (w : List UInt8) (hw : w.length = 16) : (readWord w).toBytes.data.toList = w := by
  apply list_ext_get! (by rw [toBytes_length, hw])
  intro i hi
  rw [toBytes_length] at hi
  rw [toBytes_list_get _ i hi]
  unfold readWord
  split
  · rw [u64OfLE_byte _ _ _ (by omega), get!_list, Nat.zero_add]
  · rw [u64OfLE_byte _ _ _ (by omega), get!_list, show 8 + (i - 8) = i by omega]


/-- **Any `16 m` bytes are a row's** of `m` values. -/
theorem colBytes_read : ∀ (m : ℕ) (w : List UInt8), w.length = 16 * m → ∃ vs : List F128, vs.length = m ∧ colBytes vs = w
  | 0, w, h => ⟨[], rfl, by rw [List.eq_nil_of_length_eq_zero (by omega : w.length = 0)]; rfl⟩
  | m + 1, w, h => by
    obtain ⟨vs, hl, hc⟩ := colBytes_read m (w.drop 16) (by rw [List.length_drop]; omega)
    refine ⟨readWord (w.take 16) :: vs, by rw [List.length_cons, hl], ?_⟩
    rw [show colBytes (readWord (w.take 16) :: vs) = (readWord (w.take 16)).toBytes.data.toList ++ colBytes vs from
      List.flatMap_cons, toBytes_readWord _ (by rw [List.length_take]; omega), hc, List.take_append_drop]

/-! ## A round that agrees with frames but for their values is alike frames' -/

/-- Whether byte `i` of frame `f`'s bytes is one of its values': an observe's body. -/
def valByte : Frame → ℕ → Prop
  | .one _, i => 16 ≤ i ∧ i < 32
  | .vec vs, i => 16 ≤ i ∧ i < 16 + 16 * vs.length
  | _, _ => False

/-- Whether byte `i` of the bytes of frames `fs` is one of their values'. -/
def valBytes : List Frame → ℕ → Prop
  | [], _ => False
  | f :: fs, i => (i < f.enc.length ∧ valByte f i) ∨ (f.enc.length ≤ i ∧ valBytes fs (i - f.enc.length))

theorem head_of_agree (w H P : List UInt8) (hH : H.length = 16)
    (h : ∀ i < 16, w[i]? = (H ++ P)[i]?) : w = H ++ w.drop 16 := by
  conv => lhs; rw [← List.take_append_drop 16 w]
  congr 1
  apply List.ext_getElem?
  intro i
  rw [List.getElem?_take]
  split
  · rw [h i (by omega), List.getElem?_append_left (by omega)]
  · rw [List.getElem?_eq_none (by omega)]

theorem enc_one (a : F128) : (Frame.one a).enc = (header 3 1 1).data.toList ++ a.toBytes.data.toList := rfl

theorem enc_vec (vs : List F128) : (Frame.vec vs).enc = (header 3 2 vs.length).data.toList ++ colBytes vs := rfl

/-- **A frame's bytes but for its values** are an alike frame's. -/
theorem frame_refill (f : Frame) (w : List UInt8) (hl : w.length = f.enc.length)
    (h : ∀ i, ¬ valByte f i → w[i]? = f.enc[i]?) : ∃ f', Alike f f' ∧ w = f'.enc := by
  cases f with
  | dom b => exact ⟨.dom b, rfl, List.ext_getElem? fun i => h i id⟩
  | label b => exact ⟨.label b, rfl, List.ext_getElem? fun i => h i id⟩
  | bytes b => exact ⟨.bytes b, rfl, List.ext_getElem? fun i => h i id⟩
  | sq k n => exact ⟨.sq k n, ⟨rfl, rfl⟩, List.ext_getElem? fun i => h i id⟩
  | one v =>
    rw [Frame.enc_length] at hl
    simp only [Frame.payLen] at hl
    rw [enc_one] at h
    have hw := head_of_agree w _ _ (Frame.header_length 3 1 1) fun i hi => h i fun hv => by
      simp only [valByte] at hv; omega
    refine ⟨.one (readWord (w.drop 16)), trivial, ?_⟩
    rw [enc_one, toBytes_readWord _ (by rw [List.length_drop]; omega)]
    exact hw
  | vec vs =>
    rw [Frame.enc_length] at hl
    simp only [Frame.payLen] at hl
    rw [enc_vec] at h
    have hw := head_of_agree w _ _ (Frame.header_length 3 2 vs.length) fun i hi => h i fun hv => by
      simp only [valByte] at hv; omega
    obtain ⟨vs', hl', hc⟩ := colBytes_read vs.length (w.drop 16) (by rw [List.length_drop]; omega)
    refine ⟨.vec vs', hl'.symm, ?_⟩
    rw [enc_vec, hl', hc]
    exact hw

/-- **A round's bytes but for its frames' values** are alike frames'. -/
theorem encs_refill : ∀ (fs : List Frame) (tail w : List UInt8), w.length = (encs fs ++ tail).length →
    (∀ i, ¬ valBytes fs i → w[i]? = (encs fs ++ tail)[i]?) → ∃ fs', List.Forall₂ Alike fs fs' ∧ w = encs fs' ++ tail
  | [], tail, w, _, h => ⟨[], .nil, List.ext_getElem? fun i => h i id⟩
  | f :: fs, tail, w, hl, h => by
    have hn : f.enc.length ≤ w.length := by
      rw [hl, encs_cons, List.length_append, List.length_append]; omega
    obtain ⟨f', hf, he⟩ := frame_refill f (w.take f.enc.length) (by rw [List.length_take]; omega) fun i hi => by
      rw [List.getElem?_take]
      split
      · rw [h i fun hv => hv.elim (fun hv => hi hv.2) (fun hv => by omega), encs_cons, List.append_assoc,
          List.getElem?_append_left (by omega)]
      · rw [List.getElem?_eq_none (by omega)]
    obtain ⟨fs', hfs, he'⟩ := encs_refill fs tail (w.drop f.enc.length)
      (by rw [List.length_drop, hl, encs_cons, List.length_append, List.length_append, List.length_append]; omega)
      fun i hi => by
        rw [List.getElem?_drop, h _ fun hv => hv.elim (fun hv => by omega)
          (fun hv => hi (by rw [show f.enc.length + i - f.enc.length = i by omega] at hv; exact hv.2)),
          encs_cons, List.append_assoc, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
    refine ⟨f' :: fs', .cons hf hfs, ?_⟩
    calc w = w.take f.enc.length ++ w.drop f.enc.length := (List.take_append_drop _ w).symm
      _ = encs (f' :: fs') ++ tail := by rw [he, he', encs_cons, List.append_assoc]

theorem midFrame_of_alike : ∀ {f f' : Frame}, Alike f f' → midFrame f → midFrame f'
  | .dom _, .dom _, h, hm => by simp only [Alike] at h; simp only [midFrame] at hm ⊢; omega
  | .label _, .label _, h, hm => by simp only [Alike] at h; rw [← h]; exact hm
  | .bytes _, .bytes _, h, hm => by simp only [Alike] at h; simp only [midFrame] at hm ⊢; omega
  | .one _, .one _, _, _ => trivial
  | .vec _, .vec _, h, hm => by simp only [Alike] at h; simp only [midFrame] at hm ⊢; omega
  | .sq _ _, .sq _ _, h, hm => by simp only [Alike] at h; obtain ⟨rfl, rfl⟩ := h; exact hm
  | .dom _, .label _, h, _ | .dom _, .one _, h, _ | .dom _, .vec _, h, _ | .dom _, .bytes _, h, _
  | .dom _, .sq _ _, h, _ | .label _, .dom _, h, _ | .label _, .one _, h, _ | .label _, .vec _, h, _
  | .label _, .bytes _, h, _ | .label _, .sq _ _, h, _ | .one _, .dom _, h, _ | .one _, .label _, h, _
  | .one _, .vec _, h, _ | .one _, .bytes _, h, _ | .one _, .sq _ _, h, _ | .vec _, .dom _, h, _
  | .vec _, .label _, h, _ | .vec _, .one _, h, _ | .vec _, .bytes _, h, _ | .vec _, .sq _ _, h, _
  | .bytes _, .dom _, h, _ | .bytes _, .label _, h, _ | .bytes _, .one _, h, _ | .bytes _, .vec _, h, _
  | .bytes _, .sq _ _, h, _ | .sq _ _, .dom _, h, _ | .sq _ _, .label _, h, _ | .sq _ _, .one _, h, _
  | .sq _ _, .vec _, h, _ | .sq _ _, .bytes _, h, _ => (h : False).elim

theorem midFrames_of_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') (hm : ∀ f ∈ fs, midFrame f) :
    ∀ f ∈ fs', midFrame f := by
  induction h with
  | nil => intro f hf; cases hf
  | cons ha _ ih =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact midFrame_of_alike ha (hm _ (List.mem_cons_self ..))
    · exact ih (fun g hg => hm g (List.mem_cons_of_mem _ hg)) f hf

/-! ## The free offsets are the frames' values -/

theorem foldlM_absorb1_free (c : Cfg) : ∀ (vs : List F128) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    vs.foldlM (absorb1 c) r = .ok r' → ∀ q ∈ r'.2.2.1, q ∈ r.2.2.1 ∨ ∃ j < vs.length, q = r.2.2.2 + 16 * j
  | [], r, r', h => by cases h; intro q hq; exact .inl hq
  | v :: vs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases ha : absorb1 c r v with
    | error e => rw [ha] at h; cases h
    | ok a =>
      rw [ha] at h
      have hpos : a.2.2.2 = r.2.2.2 + 16 ∧ ∀ q ∈ a.2.2.1, q ∈ r.2.2.1 ∨ q = r.2.2.2 := by
        unfold absorb1 at ha
        cases hb : r.1.absorb c v with
        | error e => rw [hb] at ha; cases ha
        | ok b =>
          rw [hb] at ha
          cases ha
          refine ⟨rfl, fun q hq => ?_⟩
          dsimp only at hq
          split at hq
          · exact Array.mem_push.1 hq
          · exact .inl hq
      intro q hq
      rcases foldlM_absorb1_free c vs a r' h q hq with h1 | ⟨j, hj, rfl⟩
      · rcases hpos.2 q h1 with h2 | rfl
        · exact .inl h2
        · exact .inr ⟨0, by simp, by omega⟩
      · exact .inr ⟨j + 1, by simp only [List.length_cons]; omega, by rw [hpos.1]; omega⟩

theorem stepAt_free (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (f : Frame) (h : stepAt c r f = .ok r') :
    ∀ q ∈ r'.2.2.1, q ∈ r.2.2.1 ∨
      ∀ i, q ≤ i → i < q + 16 → r.2.2.2 ≤ i ∧ i - r.2.2.2 < f.enc.length ∧ valByte f (i - r.2.2.2) := by
  intro q hq
  cases f with
  | dom b => cases h; exact .inl hq
  | label b =>
    simp only [stepAt] at h
    cases hl : r.1.label c b with
    | error e => rw [hl] at h; cases h
    | ok x => rw [hl] at h; cases h; exact .inl hq
  | bytes b => cases h; exact .inl hq
  | sq k n => cases h; exact .inl hq
  | one v =>
    rcases foldlM_absorb1_free c [v] _ r' h q hq with h1 | ⟨j, hj, rfl⟩
    · exact .inl h1
    · refine .inr fun i hi hi' => ?_
      rw [Frame.enc_length]
      simp only [List.length_singleton] at hj
      dsimp only at hi hi'
      simp only [Frame.payLen, valByte]
      omega
  | vec vs =>
    rcases foldlM_absorb1_free c vs _ r' h q hq with h1 | ⟨j, hj, rfl⟩
    · exact .inl h1
    · refine .inr fun i hi hi' => ?_
      rw [Frame.enc_length]
      dsimp only at hi hi'
      simp only [Frame.payLen, valByte]
      omega

theorem foldlM_stepAt_free (c : Cfg) : ∀ (fs : List Frame) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    fs.foldlM (stepAt c) r = .ok r' →
      ∀ q ∈ r'.2.2.1, q ∈ r.2.2.1 ∨ ∀ i, q ≤ i → i < q + 16 → r.2.2.2 ≤ i ∧ valBytes fs (i - r.2.2.2)
  | [], r, r', h => by cases h; intro q hq; exact .inl hq
  | f :: fs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases ha : stepAt c r f with
    | error e => rw [ha] at h; cases h
    | ok a =>
      rw [ha] at h
      have hp := stepAt_pos c r a f ha
      intro q hq
      rcases foldlM_stepAt_free c fs a r' h q hq with h1 | h1
      · rcases stepAt_free c r a f ha q h1 with h2 | h2
        · exact .inl h2
        · refine .inr fun i hi hi' => ?_
          obtain ⟨x, y, z⟩ := h2 i hi hi'
          exact ⟨x, Or.inl ⟨y, z⟩⟩
      · refine .inr fun i hi hi' => ?_
        obtain ⟨x, y⟩ := h1 i hi hi'
        refine ⟨by omega, Or.inr ⟨by omega, ?_⟩⟩
        rwa [show i - r.2.2.2 - f.enc.length = i - a.2.2.2 by omega]

/-- **The words a round frees are its frames' values.** -/
theorem roundFree_free (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (hq : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q) :
    ∀ p ∈ q.2.2, ∀ i, p ≤ i → i < p + 16 → valBytes fs i := by
  rw [roundFree_encs c s fs k n hs hm hk h0 hn] at hq
  cases ha : fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) with
  | error e => rw [ha] at hq; cases hq
  | ok a =>
    rw [ha] at hq
    have := exMap_ok_eq hq
    subst this
    intro p hp i hi hi'
    rcases foldlM_stepAt_free c fs _ a ha p hp with h | h
    · simp at h
    · simpa using (h i hi hi').2

/-! ## The pass decision -/

theorem passes_none {c : Cfg} {s : Masker} {v r : ByteArray} {n : ℕ} (h : passes c s v r n = .ok none) :
    ∃ q, roundFree c s r n = .ok q ∧ v.size = r.size ∧ firstDiff v r q.2.2 = none := by
  unfold passes at h
  cases hr : roundFree c s r n with
  | error e => rw [hr] at h; cases h
  | ok q =>
    rw [hr] at h
    refine ⟨q, rfl, ?_⟩
    by_cases hs : v.size = r.size
    · simp only [bind, Except.bind, hs, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, pure, Except.pure] at h
      refine ⟨hs, ?_⟩
      cases hd : firstDiff v r q.2.2 with
      | none => rfl
      | some i => rw [hd] at h; cases h
    · have hb : (v.size != r.size) = true := by simpa using hs
      simp only [bind, Except.bind, pure, Except.pure, hb, ↓reduceIte] at h
      cases h

theorem firstDiff_none {v r : ByteArray} {free : Array ℕ} (h : firstDiff v r free = none) :
    ∀ i < v.size, inFree free i = false → v.get! i = r.get! i := by
  intro i hi hf
  unfold firstDiff at h
  split at h
  · rename_i he
    have he' : (v.data == r.data) = true := he
    rw [ByteArray.ext (eq_of_beq he')]
  · have := List.find?_eq_none.1 h i (List.mem_range.2 hi)
    simpa [hf] using this

theorem getElem?_data (v : ByteArray) (i : ℕ) (hi : i < v.size) : v.data.toList[i]? = some (v.get! i) := by
  have hi' : i < v.data.size := hi
  rw [get!_eq, getElem!_pos v.data i hi', Array.getElem?_toList, Array.getElem?_eq_getElem hi']

/-- **A device round that passes the frame's check** at the scheduled round of frames `fs` and its squeeze is the bytes
of frames alike to `fs`, each before the squeeze, and the same squeeze. -/
theorem passes_frames (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (v : ByteArray)
    (h : passes c s v (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok none) :
    ∃ fs', List.Forall₂ Alike fs fs' ∧ (∀ f ∈ fs', midFrame f) ∧
      v = ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray := by
  obtain ⟨q, hq, hsz, hd⟩ := passes_none h
  have hfree := roundFree_free c s fs k n hs hm hk h0 hn hq
  have hag := firstDiff_none hd
  have hsz' : v.size = (encs fs ++ (Frame.sq k n).enc).length := by
    rw [hsz]; simp [ByteArray.size]
  obtain ⟨fs', hA, he⟩ := encs_refill fs (Frame.sq k n).enc v.data.toList (by rw [Array.length_toList]; exact hsz')
    fun i hi => by
      rcases Nat.lt_or_ge i v.size with hlt | hge
      · have hf : inFree q.2.2 i = false := by
          unfold inFree
          apply Bool.eq_false_iff.2
          intro ha
          obtain ⟨p, hp, hpi⟩ := Array.any_eq_true'.1 ha
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hpi
          exact hi (hfree p hp i hpi.1 hpi.2)
        rw [getElem?_data v i hlt, hag i hlt hf, get!_list,
          getElem!_pos _ i (by omega), List.getElem?_eq_getElem (by omega)]
      · rw [List.getElem?_eq_none (by rw [Array.length_toList]; exact hge), List.getElem?_eq_none (by omega)]
  refine ⟨fs', hA, midFrames_of_alike hA hm, ?_⟩
  apply ByteArray.ext
  apply Array.ext'
  rw [he]

end Claims.ZKServer
