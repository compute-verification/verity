import Properties.Identity

/-!
# Proofs that core's identity hash is C-Flock's SHA-512 (`Properties.Identity`)

A state `Rep`resents the message it absorbed: its length, its pending bytes (the message's tail after its whole blocks)
and its chaining value (`chain`, the compressions over those blocks). `init` represents the empty message and `update`
keeps the invariant (`rep_update`): the compression reads only the 128 bytes of its block (`compress_congr`), so the
blocks the state compresses out of its buffer and the new data are the message's own. Finalizing a state that represents
`m` compresses `m`'s padding (`L_padding`, `L_padData`) after the same blocks, which is `Flock.Sha512.hash m`
(`Rep.finalize`). The kernel never unfolds a compression: the proofs rewrite with `compress_congr` and `chain`'s
equations instead of comparing compressions by evaluation.
-/

namespace Properties.Identity

open Flock Identity.Sha512

/-- A byte string's bytes. -/
abbrev L (b : ByteArray) : List UInt8 := b.data.toList

theorem L_length (b : ByteArray) : (L b).length = b.size := by simp [L]

theorem L_append (a b : ByteArray) : L (a ++ b) = L a ++ L b := by simp [L, ByteArray.data_append]

theorem get!_eq (b : ByteArray) (k : Nat) : b.get! k = ((L b)[k]?).getD 0 := by
  obtain ⟨bs⟩ := b
  simp only [L, Array.getElem?_toList, ByteArray.get!, getElem!_def]
  cases bs[k]? <;> rfl

/-! ## The compression reads only its block -/

theorem word_congr {d d' : ByteArray} {o o' : Nat} (h : ∀ i < 8, (L d)[o + i]? = (L d')[o' + i]?) :
    Sha512.word d o = Sha512.word d' o' := by
  simp only [Sha512.word, get!_eq]
  rw [h 0 (by decide), h 1 (by decide), h 2 (by decide), h 3 (by decide), h 4 (by decide), h 5 (by decide),
    h 6 (by decide), h 7 (by decide)]

theorem compress_congr (h : Array UInt64) {d d' : ByteArray} {off off' : Nat}
    (hd : ∀ k < 128, (L d)[off + k]? = (L d')[off' + k]?) : Sha512.compress h d off = Sha512.compress h d' off' := by
  have hw : ∀ j < 16, Sha512.word d (off + 8 * j) = Sha512.word d' (off' + 8 * j) := fun j hj =>
    word_congr fun i hi => by rw [Nat.add_assoc, Nat.add_assoc]; exact hd _ (by omega)
  unfold Sha512.compress
  rw [hw 0 (by decide), hw 1 (by decide), hw 2 (by decide), hw 3 (by decide), hw 4 (by decide), hw 5 (by decide),
    hw 6 (by decide), hw 7 (by decide), hw 8 (by decide), hw 9 (by decide), hw 10 (by decide), hw 11 (by decide),
    hw 12 (by decide), hw 13 (by decide), hw 14 (by decide), hw 15 (by decide)]

/-! ## Chains of compressions -/

attribute [local irreducible] Sha512.compress

/-- The chaining value after `n` blocks of `d` from offset `off`, starting at `h`. -/
def chain (h : Array UInt64) (d : ByteArray) (off n : Nat) : Array UInt64 :=
  (List.range' 0 n).foldl (fun h b => Sha512.compress h d (off + 128 * b)) h

theorem foldl_eq_chain (h : Array UInt64) (d : ByteArray) (off n : Nat) :
    (List.range' 0 n).foldl (fun h b => Sha512.compress h d (off + 128 * b)) h = chain h d off n := rfl

theorem chain_succ (h : Array UInt64) (d : ByteArray) (off n : Nat) :
    chain h d off (n + 1) = Sha512.compress (chain h d off n) d (off + 128 * n) := by
  rw [chain, chain, List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]

theorem chain_congr (h : Array UInt64) {d d' : ByteArray} {off off' : Nat} (n : Nat)
    (hd : ∀ k < 128 * n, (L d)[off + k]? = (L d')[off' + k]?) : chain h d off n = chain h d' off' n := by
  induction n with
  | zero => simp only [chain, List.range'_zero, List.foldl_nil]
  | succ n ih =>
    rw [chain_succ, chain_succ, ih fun k hk => hd k (by omega)]
    exact compress_congr _ fun k hk => by rw [Nat.add_assoc, Nat.add_assoc]; exact hd _ (by omega)

theorem chain_add (h : Array UInt64) (d : ByteArray) (off n k : Nat) :
    chain h d off (n + k) = chain (chain h d off n) d (off + 128 * n) k := by
  induction k with
  | zero => simp only [chain, List.range'_zero, List.foldl_nil, Nat.add_zero]
  | succ k ih => rw [← Nat.add_assoc, chain_succ, chain_succ, ih, Nat.mul_add, Nat.add_assoc]

/-! ## A state as the message it absorbed -/

/-- `s` is the state after absorbing `m`: its length, its pending bytes the tail of `m` after its whole blocks, and its
chaining value the chain over those blocks. -/
structure Rep (s : State) (m : ByteArray) : Prop where
  len : s.len = m.size
  buf : L s.buf = (L m).drop (128 * (m.size / 128))
  h : s.h = chain Sha512.iv m 0 (m.size / 128)

theorem Rep.buf_size {s : State} {m : ByteArray} (r : Rep s m) : s.buf.size = m.size % 128 := by
  have := congrArg List.length r.buf
  simp only [L_length, List.length_drop] at this
  omega

theorem rep_init : Rep init .empty := ⟨rfl, rfl, rfl⟩

theorem getElem?_L_append_left (m d : ByteArray) {k : Nat} (hk : k < m.size) : (L (m ++ d))[k]? = (L m)[k]? := by
  rw [L_append, List.getElem?_append_left (by rw [L_length]; exact hk)]

theorem getElem?_L_append_right (m d : ByteArray) (k : Nat) : (L (m ++ d))[m.size + k]? = (L d)[k]? := by
  rw [L_append, List.getElem?_append_right (by rw [L_length]; omega), L_length, Nat.add_sub_cancel_left]

theorem L_extract (d : ByteArray) (a : Nat) : L (d.extract a d.size) = (L d).drop a := by
  simp only [L, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop]
  rw [List.take_of_length_le (by simp)]

theorem L_extract_zero (d : ByteArray) (t : Nat) : L (d.extract 0 t) = (L d).take t := by
  simp only [L, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero,
    Nat.sub_zero]

theorem rep_update {s : State} {m : ByteArray} (r : Rep s m) (d : ByteArray) : Rep (s.update d) (m ++ d) := by
  have hbs := r.buf_size
  have hsz : (m ++ d).size = m.size + d.size := ByteArray.size_append
  have hm : (L m).length = m.size := L_length m
  unfold State.update
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, bind_pure_comp,
    map_pure, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, foldl_eq_chain]
  by_cases hb : s.buf.size > 0
  · simp only [hb, ite_true]
    generalize ht : min (128 - s.buf.size) d.size = t
    have ht' : t ≤ d.size := by omega
    have hs1 : (s.buf ++ d.extract 0 t).size = s.buf.size + t := by
      rw [ByteArray.size_append, ByteArray.size_extract]; omega
    have hL1 : L (s.buf ++ d.extract 0 t) = (L m).drop (128 * (m.size / 128)) ++ (L d).take t := by
      rw [L_append, L_extract_zero, r.buf]
    have e1 : chain Sha512.iv (m ++ d) 0 (m.size / 128) = chain Sha512.iv m 0 (m.size / 128) :=
      chain_congr _ _ fun k hk => getElem?_L_append_left m d (by omega)
    by_cases hA : s.buf.size + t = 128
    · simp only [hs1, hA, beq_self_eq_true, ite_true, ByteArray.size_empty, Id.run_pure]
      have hq : (m.size + d.size) / 128 = m.size / 128 + 1 + (d.size - t) / 128 := by omega
      refine ⟨by simp [r.len, hsz], ?_, ?_⟩
      · rw [L_extract, hsz, hq, L_append, List.drop_append, List.drop_eq_nil_of_le (as := L m) (by rw [hm]; omega),
          List.nil_append, hm]
        congr 1
        omega
      · rw [hsz, hq, chain_add, chain_succ, e1, ← r.h]
        have hc : Sha512.compress s.h (m ++ d) (0 + 128 * (m.size / 128)) =
            Sha512.compress s.h (s.buf ++ d.extract 0 t) 0 := by
          refine compress_congr _ fun k hk => ?_
          rw [hL1, Nat.zero_add, Nat.zero_add, L_append]
          by_cases hk' : k < s.buf.size
          · rw [List.getElem?_append_left (by rw [hm]; omega), List.getElem?_append_left (by simp; omega),
              List.getElem?_drop]
          · rw [List.getElem?_append_right (by rw [hm]; omega), List.getElem?_append_right (by simp; omega),
              List.getElem?_take_of_lt (by simp; omega), hm, List.length_drop, hm]
            congr 1
            omega
        rw [hc]
        refine chain_congr (d := d) (d' := m ++ d) _ _ fun k _ => ?_
        rw [show 0 + 128 * (m.size / 128 + 1) + k = m.size + (t + k) by omega, getElem?_L_append_right]
    · have htd : t = d.size := by omega
      have h0 : ¬ s.buf.size + t = 0 := by omega
      simp only [hs1, hA, h0, beq_iff_eq, ite_false, Id.run_pure]
      have hq : (m.size + d.size) / 128 = m.size / 128 := by omega
      refine ⟨by simp [r.len, hsz], ?_, ?_⟩
      · rw [hL1, hsz, hq, L_append, List.drop_append_of_le_length (by rw [hm]; omega), htd,
          List.take_of_length_le (by rw [L_length]; exact Nat.le_refl _)]
      · rw [hsz, hq, e1, r.h]
  · simp only [hb, ite_false]
    have h0 : s.buf.size = 0 := by omega
    simp only [h0, beq_self_eq_true, ite_true, Id.run_pure]
    have hr : m.size = 128 * (m.size / 128) := by omega
    have hq : (m.size + d.size) / 128 = m.size / 128 + d.size / 128 := by omega
    refine ⟨by simp [r.len, hsz], ?_, ?_⟩
    · rw [Nat.zero_add, L_extract, hsz, hq, L_append, List.drop_append, List.drop_eq_nil_of_le (as := L m) (by rw [hm]; omega),
        List.nil_append, hm]
      congr 1
      omega
    · rw [hsz, hq, chain_add, r.h]
      have e1 : chain Sha512.iv (m ++ d) 0 (m.size / 128) = chain Sha512.iv m 0 (m.size / 128) :=
        chain_congr _ _ fun k hk => getElem?_L_append_left m d (by omega)
      rw [e1]
      refine chain_congr (d := d) (d' := m ++ d) _ _ fun k _ => ?_
      simp only [Nat.zero_add]
      rw [← hr, getElem?_L_append_right]


/-- The 16 bytes of a message length `len` (in bytes): SHA-512's 128-bit big-endian bit count. -/
def lenBytes (len : Nat) : List UInt8 := (List.range' 0 16).map fun a => ((len * 8) >>> (8 * (15 - a)) % 256).toUInt8

/-- The number of zero bytes SHA-512's padding puts after a `k`-byte message and `0x80`. -/
def zeroPad (k : Nat) : Nat := (112 + 128 - (k + 1) % 128) % 128

/-- `Flock.Sha512.hash`'s padded message: `m ‖ 0x80 ‖ 0…0 ‖ len`. -/
def padData (m : ByteArray) : ByteArray :=
  (List.range' 0 16).foldl (fun b a => b.push ((m.size * 8) >>> (8 * (15 - a)) % 256).toUInt8)
    ((List.range' 0 ((112 + 128 - (m.push 128).size % 128) % 128)).foldl (fun b _ => b.push 0) (m.push 128))

theorem chain_zero (h : Array UInt64) (d : ByteArray) (n : Nat) :
    chain h d 0 n = (List.range' 0 n).foldl (fun h b => Sha512.compress h d (128 * b)) h := by
  simp only [chain, Nat.zero_add]

theorem hash_eq (m : ByteArray) : Flock.Sha512.hash m = digest (chain Sha512.iv (padData m) 0 ((padData m).size / 128)) := by
  simp only [Flock.Sha512.hash, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero,
    Nat.add_sub_cancel, Nat.div_one]
  rw [chain_zero]
  rfl

theorem finalize_eq (s : State) : s.finalize = digest (chain s.h s.padding 0 (s.padding.size / 128)) := by
  simp only [State.finalize, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rw [chain_zero]
  rfl

theorem foldl_push (f : Nat → UInt8) : ∀ (l : List Nat) (b : ByteArray),
    L (l.foldl (fun b a => b.push (f a)) b) = L b ++ l.map f
  | [], b => by simp
  | a :: l, b => by
    rw [List.foldl_cons, foldl_push f l, List.map_cons]
    simp [L, ByteArray.data_push]

theorem L_padData (m : ByteArray) : L (padData m) = L m ++ 128 :: List.replicate (zeroPad m.size) 0 ++ lenBytes m.size := by
  rw [padData, foldl_push, foldl_push (fun _ => 0)]
  simp [L, ByteArray.data_push, zeroPad, lenBytes, List.map_const']

theorem L_padding (s : State) : L s.padding = L s.buf ++ 128 :: List.replicate (zeroPad s.buf.size) 0 ++ lenBytes s.len := by
  simp only [State.padding, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, pure_bind]
  show L (List.foldl _ _ _) = _
  rw [foldl_push]
  simp [L, ByteArray.data_push, ByteArray.data_append, zeroPad, lenBytes]

theorem Rep.finalize {s : State} {m : ByteArray} (r : Rep s m) : s.finalize = Flock.Sha512.hash m := by
  have hbs := r.buf_size
  have hm : (L m).length = m.size := L_length m
  have hz : zeroPad s.buf.size = zeroPad m.size := by unfold zeroPad; omega
  have hP := L_padData m
  have hQ : L s.padding = (L m).drop (128 * (m.size / 128)) ++ (128 :: List.replicate (zeroPad m.size) 0 ++ lenBytes m.size) := by
    rw [L_padding, hz, r.len, r.buf, List.append_assoc]
  rw [List.append_assoc] at hP
  generalize hT : 128 :: List.replicate (zeroPad m.size) 0 ++ lenBytes m.size = T at hP hQ
  have hPs : (padData m).size = m.size + T.length := by rw [← L_length, hP, List.length_append, hm]
  have hQs : s.padding.size = m.size % 128 + T.length := by
    rw [← L_length, hQ, List.length_append, List.length_drop, hm]; omega
  have hn : (padData m).size / 128 = m.size / 128 + s.padding.size / 128 := by omega
  rw [finalize_eq, hash_eq, hn, chain_add]
  have e1 : chain Sha512.iv (padData m) 0 (m.size / 128) = s.h := by
    rw [r.h]
    refine chain_congr _ _ fun k hk => ?_
    rw [hP, List.getElem?_append_left (by rw [hm]; omega)]
  rw [e1]
  refine congrArg digest (chain_congr _ _ fun k hk => ?_).symm
  rw [hP, hQ, Nat.zero_add, Nat.zero_add]
  by_cases hk : k < m.size % 128
  · rw [List.getElem?_append_left (by rw [hm]; omega), List.getElem?_append_left (by simp; omega),
      List.getElem?_drop]
  · rw [List.getElem?_append_right (by rw [hm]; omega), List.getElem?_append_right (by simp; omega), hm,
      List.length_drop, hm, show 128 * (m.size / 128) + k - m.size = k - (m.size - 128 * (m.size / 128)) by omega]

theorem update_update (hd p : ByteArray) : ((init.update hd).update p).finalize = Flock.Sha512.hash (hd ++ p) := by
  have := (rep_update (rep_update rep_init hd) p).finalize
  rwa [ByteArray.empty_append] at this

theorem taggedSha512_eq (tag : String) (payload : ByteArray) :
    Identity.taggedSha512 tag payload = Flock.Sha512.hash <$> taggedInput tag payload := by
  unfold Identity.taggedSha512 Identity.taggedState taggedInput
  cases h : Identity.taggedHeader tag payload.size with
  | error e => rfl
  | ok hd =>
    simp only [bind, Except.bind, pure, Except.pure, Functor.map, Except.map]
    rw [update_update]

theorem taggedHeader_ok {tag : String} {n : Nat} {hd : ByteArray} :
    Identity.taggedHeader tag n = .ok hd ↔ (0 < tag.toUTF8.size ∧ tag.toUTF8.size < 2 ^ 32 ∧ n < 2 ^ 64) ∧
      hd = Identity.TAGGED_SHA512 ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++ Flock.be64 n.toUInt64 := by
  unfold Identity.taggedHeader
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw, beq_iff_eq]
  by_cases h1 : tag.toUTF8.size = 0
  · simp only [h1, ite_true, reduceCtorEq, false_iff]
    omega
  by_cases h2 : tag.toUTF8.size ≥ 2 ^ 32
  · simp only [h1, h2, ite_true, ite_false, reduceCtorEq, false_iff]
    omega
  by_cases h3 : n ≥ 2 ^ 64
  · simp only [h1, h2, h3, ite_true, ite_false, reduceCtorEq, false_iff]
    omega
  simp only [h1, h2, h3, ite_false, Except.ok.injEq]
  constructor
  · intro h; exact ⟨by omega, h.symm⟩
  · intro h; exact h.2.symm

theorem taggedInput_ok {tag : String} {payload x : ByteArray} :
    taggedInput tag payload = .ok x ↔ (0 < tag.toUTF8.size ∧ tag.toUTF8.size < 2 ^ 32 ∧ payload.size < 2 ^ 64) ∧
      x = Identity.TAGGED_SHA512 ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++
        Flock.be64 payload.size.toUInt64 ++ payload := by
  unfold taggedInput
  cases h : Identity.taggedHeader tag payload.size with
  | error e =>
    simp only [bind, Except.bind, reduceCtorEq, false_iff]
    intro hx
    cases (taggedHeader_ok.mpr ⟨hx.1, rfl⟩).symm.trans h
  | ok hd =>
    obtain ⟨hc, rfl⟩ := taggedHeader_ok.mp h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq]
    exact ⟨fun e => ⟨hc, e.symm⟩, fun e => e.2.symm⟩

theorem rep_foldl : ∀ (ms : List ByteArray) {s : State} {m : ByteArray}, Rep s m →
    Rep (ms.foldl State.update s) (ms.foldl (· ++ ·) m)
  | [], _, _, r => r
  | d :: ms, _, _, r => rep_foldl ms (rep_update r d)

theorem streamIsFlock : StreamIsFlock := fun ms => (rep_foldl ms rep_init).finalize

theorem taggedIsFlock : TaggedIsFlock := taggedSha512_eq

/-- What a tagged digest is, when there is one. -/
theorem taggedSha512_ok {tag : String} {payload dg : ByteArray} :
    Identity.taggedSha512 tag payload = .ok dg ↔ ∃ x, taggedInput tag payload = .ok x ∧ dg = Flock.Sha512.hash x := by
  rw [taggedSha512_eq]
  cases taggedInput tag payload with
  | error e => simp only [Functor.map, Except.map, reduceCtorEq, false_and, exists_false]
  | ok x => simp only [Functor.map, Except.map, Except.ok.injEq]; exact ⟨fun h => ⟨x, rfl, h.symm⟩, fun ⟨_, h, e⟩ => by
      cases h; exact e.symm⟩

/-- A tag and payload that fit the header's fields have a tagged digest: SHA-512 of their tagged input. -/
theorem taggedSha512_of {tag : String} {payload : ByteArray} (h0 : 0 < tag.toUTF8.size) (h1 : tag.toUTF8.size < 2 ^ 32)
    (h2 : payload.size < 2 ^ 64) :
    ∃ x, taggedInput tag payload = .ok x ∧ Identity.taggedSha512 tag payload = .ok (Flock.Sha512.hash x) := by
  refine ⟨_, taggedInput_ok.mpr ⟨⟨h0, h1, h2⟩, rfl⟩, ?_⟩
  rw [taggedSha512_eq, taggedInput_ok.mpr ⟨⟨h0, h1, h2⟩, rfl⟩]
  rfl

end Properties.Identity
