import Proofs.Flock.Soundness.Discharge.ZkView.Decode
import Flock.Firewall.Frame

/-!
# The frame's round over a round's frames

The Lean frame parses each round the transport hands it (`Flock.Firewall.Frame.roundFree`): a loop over the round's
bytes that reads each frame's 16-byte header, takes its payload, and labels, copies or absorbs it, until the round's
squeeze. The analysis models a round's bytes as its frames (`FlockSoundness.Refine.Frame`, `encs`). This file runs the
loop on those bytes:

* **The frame's step** (`stepAt`): one frame before the round's squeeze, from the rep's state, the release so far, its
  free offsets and where the frame starts: a domain is skipped, a label goes through `Masker.label`, bytes are copied, and
  an observe absorbs its values one at a time (`absorb1`), each free while the algebraic phase masks.
* **The round** (`roundFree_encs`): at the bytes of frames `fs` followed by the squeeze of `n` coins of kind `k`,
  `roundFree` is the fold of `stepAt` over `fs`, then the inner proof's own round where one is open (`innerOwn`) and the
  squeeze's header (`lastAt`). The frames before the squeeze are `midFrame`s: lengths that fit their headers, and a
  squeeze among them is a zero-coin one.
* The loops as folds (`forIn_range'_foldlM`, `forIn_range'_foldlM_all`), and an observe's inner loop as the fold of
  `absorb1` over the values its body holds (`observe_loop`), read through the frame's own byte reads (`header_at`,
  `body_at`, `word_lo`, `word_hi`).
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 u64OfLE)
open _root_.Flock.Transcript (header padded)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS bytesFrame innerOwn roundFree)
open FlockSoundness.Refine (Frame encs colBytes toBytes_length header_data le64_data padded_data)
open FlockSoundness.Discharge.ZkView (get!_eq u64OfLE_eq foldl_le64)

/-! ## Loops as folds -/

/-- A `for` over `[a, a + |fs|)` whose body is `step` on each of `fs`, as long as it runs: the fold of `step`. -/
theorem forIn_range'_foldlM_all {τ ρ γ : Type} (g : ℕ → τ → V (ForInStep τ)) (emb : ρ → τ)
    (step : ρ → γ → V ρ) (Inv : ℕ → ρ → List γ → Prop)
    (hstep : ∀ a r f fs, Inv a r (f :: fs) → g a (emb r) = (fun r' => ForInStep.yield (emb r')) <$> step r f)
    (hinv : ∀ a r f fs r', Inv a r (f :: fs) → step r f = .ok r' → Inv (a + 1) r' fs) :
    ∀ (fs : List γ) (r : ρ) (a : ℕ) (x : τ), x = emb r → Inv a r fs →
      forIn (List.range' a fs.length) x g = emb <$> fs.foldlM step r := by
  intro fs
  induction fs with
  | nil => intro r a x hx _; subst hx; rfl
  | cons f fs ih =>
    intro r a x hx hI
    subst hx
    rw [List.length_cons, List.range'_succ, List.forIn_cons, hstep a r f fs hI, List.foldlM_cons]
    cases hs : step r f with
    | error e => rfl
    | ok r' =>
      have := ih r' (a + 1) _ rfl (hinv a r f fs r' hI hs)
      simp only [Functor.map, Except.map, bind, Except.bind]
      exact this

/-- A `for` of more than `|fs|` turns whose body is `step` on each of `fs`, then returns `last` of the state. -/
theorem forIn_range'_foldlM {τ ρ γ : Type} (g : ℕ → τ → V (ForInStep τ)) (emb : ρ → τ)
    (step : ρ → γ → V ρ) (last : ρ → V τ) (Inv : ρ → List γ → Prop)
    (hstep : ∀ r f fs i, Inv r (f :: fs) → g i (emb r) = (fun r' => ForInStep.yield (emb r')) <$> step r f)
    (hinv : ∀ r f fs r', Inv r (f :: fs) → step r f = .ok r' → Inv r' fs)
    (hlast : ∀ r i, Inv r [] → g i (emb r) = ForInStep.done <$> last r) :
    ∀ (fs : List γ) (r : ρ) (a m : ℕ) (x : τ), x = emb r → Inv r fs → fs.length < m →
      forIn (List.range' a m) x g = fs.foldlM step r >>= last := by
  intro fs
  induction fs with
  | nil =>
    intro r a m x hx hI hm
    subst hx
    obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by simp at hm; omega⟩
    rw [List.range'_succ, List.forIn_cons, hlast r a hI, List.foldlM_nil]
    show _ = last r
    cases last r <;> rfl
  | cons f fs ih =>
    intro r a m x hx hI hm
    subst hx
    obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by simp at hm; omega⟩
    rw [List.range'_succ, List.forIn_cons, hstep r f fs a hI, List.foldlM_cons]
    cases hs : step r f with
    | error e => rfl
    | ok r' =>
      have := ih r' (a + 1) m _ rfl (hinv r f fs r' hI hs) (by simp at hm; omega)
      simp only [Functor.map, Except.map, bind, Except.bind]
      exact this

/-! ## Reading a frame's bytes -/

theorem get!_list (L : List UInt8) (i : ℕ) : (ByteArray.mk L.toArray).get! i = L[i]! := by
  rw [get!_eq]
  simp

theorem get!_drop (L : List UInt8) (p i : ℕ) : (ByteArray.mk L.toArray).get! (p + i) = (L.drop p)[i]! := by
  rw [get!_list]
  simp [getElem!_def, List.getElem?_drop]

theorem extract_list (L : List UInt8) (a b : ℕ) :
    (ByteArray.mk L.toArray).extract a b = ByteArray.mk ((L.drop a).take (b - a)).toArray := by
  apply ByteArray.ext
  simp [ByteArray.data_extract]

theorem size_list (L : List UInt8) : (ByteArray.mk L.toArray).size = L.length := by simp [ByteArray.size]

theorem u64OfLE_of_get (d : ByteArray) (o : ℕ) (x : UInt64)
    (h : ∀ i < 8, d.get! (o + i) = (x >>> (8 * i).toUInt64).toUInt8) : u64OfLE d o = x := by
  rw [u64OfLE_eq, List.foldl_ext _
    (fun acc i => acc ||| (x >>> (8 * i).toUInt64).toUInt8.toUInt64 <<< (8 * i).toUInt64) _ fun acc i hi => by
      have hi8 := (List.mem_range'_1.1 hi).2
      rw [h i (by omega)]]
  apply UInt64.toNat_inj.1
  rw [foldl_le64 x 8 (Nat.le_refl _)]
  exact Nat.mod_eq_of_lt x.toNat_lt

/-- A frame's header at `p` reads back as its op, kind and length. -/
theorem header_at (L : List UInt8) (p : ℕ) (f : Frame) (rest : List UInt8) (hd : L.drop p = f.enc ++ rest)
    (hw : f.len < 2 ^ 64) :
    (ByteArray.mk L.toArray).get! p = f.op ∧ (ByteArray.mk L.toArray).get! (p + 1) = f.kind ∧
      (u64OfLE (ByteArray.mk L.toArray) (p + 8)).toNat = f.len := by
  have h0 := get!_drop L p 0
  have h1 := get!_drop L p 1
  rw [hd, Frame.enc, header_data] at h0 h1
  refine ⟨by simpa using h0, by simpa using h1, ?_⟩
  rw [u64OfLE_of_get _ _ f.len.toUInt64 fun i hi => ?_]
  · simpa [Nat.mod_eq_of_lt hw] using hw
  · rw [show p + 8 + i = p + (8 + i) by omega, get!_drop, hd, Frame.enc, header_data, le64_data]
    simp only [getElem!_def, List.append_assoc]
    rw [List.getElem?_append_right (by simp), List.getElem?_append_left (by simp; omega)]
    simp [hi]

theorem room (L rest : List UInt8) (p : ℕ) (f : Frame) (hd : L.drop p = f.enc ++ rest) :
    p + f.enc.length + rest.length = L.length := by
  have := congrArg List.length hd
  rw [List.length_drop, List.length_append, Frame.enc_length] at this
  rw [Frame.enc_length]
  omega

theorem drop_body (L rest : List UInt8) (p : ℕ) (f : Frame) (hd : L.drop p = f.enc ++ rest) :
    L.drop (p + 16) = f.pay ++ rest := by
  rw [← List.drop_drop, hd, Frame.enc, List.append_assoc, List.drop_left' (by rw [header_data, le64_data]; simp)]

theorem body_at (L rest : List UInt8) (p w : ℕ) (f : Frame) (hd : L.drop p = f.enc ++ rest) :
    (ByteArray.mk L.toArray).extract (p + 16) (p + 16 + w) = ByteArray.mk ((f.pay ++ rest).take w).toArray := by
  rw [extract_list, drop_body L rest p f hd, show p + 16 + w - (p + 16) = w by omega]

theorem take_padded (b : ByteArray) (rest : List UInt8) :
    ByteArray.mk (((padded b).data.toList ++ rest).take b.size).toArray = b := by
  rw [padded_data, List.append_assoc, List.take_left' (by rw [Array.length_toList]; rfl)]

theorem toBytes_list_get (a : F128) (i : ℕ) (hi : i < 16) :
    a.toBytes.data.toList[i]! =
      if i < 8 then (a.lo >>> (8 * i).toUInt64).toUInt8 else (a.hi >>> (8 * (i - 8)).toUInt64).toUInt8 := by
  have h : a.toBytes.data.toList[i]? = some
      (if i < 8 then (a.lo >>> (8 * i).toUInt64).toUInt8 else (a.hi >>> (8 * (i - 8)).toUInt64).toUInt8) := by
    rw [Array.getElem?_toList]
    unfold F128.toBytes
    rw [Array.getElem?_ofFn]
    simp [hi]
  rw [getElem!_def, h]

theorem word_lo (pre post : List UInt8) (a : F128) :
    u64OfLE (ByteArray.mk (pre ++ a.toBytes.data.toList ++ post).toArray) pre.length = a.lo := by
  apply u64OfLE_of_get
  intro i hi
  rw [get!_list, getElem!_def, List.append_assoc, List.getElem?_append_right (by omega),
    show pre.length + i - pre.length = i by omega, List.getElem?_append_left (by simp [F128.toBytes]; omega),
    ← getElem!_def, toBytes_list_get a i (by omega)]
  simp [hi]

theorem word_hi (pre post : List UInt8) (a : F128) :
    u64OfLE (ByteArray.mk (pre ++ a.toBytes.data.toList ++ post).toArray) (pre.length + 8) = a.hi := by
  apply u64OfLE_of_get
  intro i hi
  rw [get!_list, getElem!_def, List.append_assoc, List.getElem?_append_right (by omega),
    show pre.length + 8 + i - pre.length = 8 + i by omega, List.getElem?_append_left (by simp [F128.toBytes]; omega),
    ← getElem!_def, toBytes_list_get a (8 + i) (by omega)]
  simp [show ¬ 8 + i < 8 by omega]

theorem encs_cons (f : Frame) (fs : List Frame) : encs (f :: fs) = f.enc ++ encs fs := by simp [encs]

theorem encs_length (fs : List Frame) : 16 * fs.length ≤ (encs fs).length := by
  induction fs with
  | nil => simp
  | cons f fs ih => rw [encs_cons, List.length_append, Frame.enc_length, List.length_cons]; omega

/-! ## The frame's steps -/

/-- A frame a round holds before its squeeze: its length fits its header, and a squeeze there is a zero-coin one. -/
def midFrame : Frame → Prop
  | .dom b => b.size < 2 ^ 64
  | .label b => b.size < 2 ^ 64
  | .bytes b => b.size < 2 ^ 64
  | .one _ => True
  | .vec vs => vs.length < 2 ^ 64
  | .sq k n => k = 2 ∧ n = 0

theorem midFrame_len {f : Frame} (h : midFrame f) : f.len < 2 ^ 64 := by
  cases f <;> simp_all [midFrame, Frame.len]

/-- One value of an observe whose body is at `r.2.2.2`: free while the algebraic phase masks, then absorbed. -/
def absorb1 (c : Cfg) (r : Masker × ByteArray × Array ℕ × ℕ) (v : F128) : V (Masker × ByteArray × Array ℕ × ℕ) :=
  (fun q : Masker × F128 => (q.1, F128.pushBytes r.2.1 q.2,
      if (r.1.on && decide (r.1.pos < c.nPiop)) = true then r.2.2.1.push r.2.2.2 else r.2.2.1, r.2.2.2 + 16)) <$>
    r.1.absorb c v

/-- **The frame's step** at a frame before the round's squeeze whose header is at `r.2.2.2`: the state, the round's
release so far, its free offsets, and where the next frame starts. -/
def stepAt (c : Cfg) (r : Masker × ByteArray × Array ℕ × ℕ) : Frame → V (Masker × ByteArray × Array ℕ × ℕ)
  | .dom b => pure (r.1, r.2.1, r.2.2.1, r.2.2.2 + 16 + (b.size + 15) / 16 * 16)
  | .label b => (fun q => (q.1, r.2.1 ++ q.2, r.2.2.1, r.2.2.2 + 16 + (b.size + 15) / 16 * 16)) <$> r.1.label c b
  | .bytes b => pure (r.1, r.2.1 ++ bytesFrame b, r.2.2.1, r.2.2.2 + 16 + (b.size + 15) / 16 * 16)
  | .one v => [v].foldlM (absorb1 c) (r.1, r.2.1 ++ header 3 1 1, r.2.2.1, r.2.2.2 + 16)
  | .vec vs => vs.foldlM (absorb1 c) (r.1, r.2.1 ++ header 3 2 vs.length, r.2.2.1, r.2.2.2 + 16)
  | .sq _ _ => pure (r.1, r.2.1 ++ header 4 2 0, r.2.2.1, r.2.2.2 + 16)

theorem foldlM_absorb1_pos (c : Cfg) : ∀ (vs : List F128) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    vs.foldlM (absorb1 c) r = .ok r' → r'.2.2.2 = r.2.2.2 + 16 * vs.length
  | [], r, r', h => by cases h; simp
  | v :: vs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases ha : absorb1 c r v with
    | error e => rw [ha] at h; cases h
    | ok q =>
      rw [ha] at h
      have hq : q.2.2.2 = r.2.2.2 + 16 := by
        unfold absorb1 at ha
        cases hb : r.1.absorb c v with
        | error e => rw [hb] at ha; cases ha
        | ok b => rw [hb] at ha; cases ha; rfl
      rw [foldlM_absorb1_pos c vs q r' h, hq, List.length_cons]
      omega

theorem stepAt_pos (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (f : Frame) (h : stepAt c r f = .ok r') :
    r'.2.2.2 = r.2.2.2 + f.enc.length := by
  rw [Frame.enc_length]
  cases f with
  | dom b => cases h; simp [Frame.payLen]; omega
  | label b =>
    simp only [stepAt] at h
    cases hl : r.1.label c b with
    | error e => rw [hl] at h; cases h
    | ok q => rw [hl] at h; cases h; simp [Frame.payLen]; omega
  | bytes b => cases h; simp [Frame.payLen]; omega
  | one v => rw [foldlM_absorb1_pos c [v] _ r' h]; simp [Frame.payLen]
  | vec vs => rw [foldlM_absorb1_pos c vs _ r' h]; simp [Frame.payLen]; omega
  | sq k n => cases h; simp [Frame.payLen]

/-- **An observe's loop** over the values `vs` its body `B` holds, from offset `q`: the fold of `absorb1`. -/
theorem observe_loop (c : Cfg) (q N : ℕ) (vs : List F128) (hN : N = vs.length) (B : ByteArray)
    (hB : B.data.toList = colBytes vs) (m : Masker) (o : ByteArray) (fr : Array ℕ) :
    (forIn (List.range' 0 N) (m, o, fr) (fun i (__s : Masker × ByteArray × Array ℕ) =>
      if (__s.fst.on && decide (__s.fst.pos < c.nPiop)) = true then do
        let __x ← Masker.absorb c __s.fst { lo := u64OfLE B (16 * i), hi := u64OfLE B (16 * i + 8) }
        pure (ForInStep.yield (__x.fst, F128.pushBytes __s.snd.fst __x.snd, __s.snd.snd.push (q + 16 * i)))
      else do
        let __x ← Masker.absorb c __s.fst { lo := u64OfLE B (16 * i), hi := u64OfLE B (16 * i + 8) }
        pure (ForInStep.yield (__x.fst, F128.pushBytes __s.snd.fst __x.snd, __s.snd.snd))) :
        V (Masker × ByteArray × Array ℕ)) =
      (fun r => (r.1, r.2.1, r.2.2.1)) <$> vs.foldlM (absorb1 c) (m, o, fr, q) := by
  subst hN
  refine forIn_range'_foldlM_all _ (fun r => (r.1, r.2.1, r.2.2.1)) (absorb1 c)
    (fun a r ws => r.2.2.2 = q + 16 * a ∧ ∃ pre, colBytes vs = pre ++ colBytes ws ∧ pre.length = 16 * a)
    ?_ ?_ vs (m, o, fr, q) 0 _ rfl ⟨rfl, [], by simp, rfl⟩
  · intro a r w ws hI
    obtain ⟨hq, pre, hpre, hl⟩ := hI
    obtain ⟨m', o', fr', q'⟩ := r
    dsimp only at hq ⊢
    subst hq
    have hw : B = ByteArray.mk (pre ++ w.toBytes.data.toList ++ colBytes ws).toArray := by
      apply ByteArray.ext
      apply Array.ext'
      rw [hB, hpre]
      simp [colBytes]
    rw [hw, ← hl, word_lo, word_hi]
    unfold absorb1
    dsimp only
    cases hc : (m'.on && decide (m'.pos < c.nPiop)) <;> simp only [Bool.false_eq_true, ↓reduceIte] <;>
      cases m'.absorb c w <;> rfl
  · intro a r w ws r' hI hs
    obtain ⟨hq, pre, hpre, hl⟩ := hI
    unfold absorb1 at hs
    cases hab : r.1.absorb c w with
    | error e => rw [hab] at hs; cases hs
    | ok x =>
      rw [hab] at hs
      cases hs
      refine ⟨by dsimp only; rw [hq]; omega, pre ++ w.toBytes.data.toList, by rw [hpre]; simp [colBytes], ?_⟩
      rw [List.length_append, hl, toBytes_length]
      omega

theorem observe_tail (c : Cfg) (vs : List F128) (r0 : Masker × ByteArray × Array ℕ × ℕ) (q : ℕ)
    (hq : q = r0.2.2.2 + 16 * vs.length) :
    (do
      let __s ← (fun r => (r.1, r.2.1, r.2.2.1)) <$> vs.foldlM (absorb1 c) r0
      pure (ForInStep.yield ((none : Option (ByteArray × Masker × Array ℕ)), __s.1, __s.2.1, __s.2.2, q)) : V _) =
      (fun r' => ForInStep.yield (none, r'.1, r'.2.1, r'.2.2.1, r'.2.2.2)) <$> vs.foldlM (absorb1 c) r0 := by
  cases h : vs.foldlM (absorb1 c) r0 with
  | error e => rfl
  | ok r' =>
    have := foldlM_absorb1_pos c vs r0 r' h
    obtain ⟨a, b, d, e⟩ := r'
    dsimp only at this
    subst hq this
    rfl

/-- The round's final squeeze, of `n` coins of kind `k`: the inner proof's own round when one is open, then the
squeeze's header after the round's release, and the state that owes the round's coins. -/
def lastAt (c : Cfg) (C : ByteArray) (k : UInt8) (n : ℕ) (r : Masker × ByteArray × Array ℕ × ℕ) :
    V (Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ) :=
  (fun _ => (some (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1), r.1, r.2.1, r.2.2.1, r.2.2.2 + 16)) <$>
    (match r.1.inner with
     | some j => innerOwn c r.1 j C n
     | none => pure () : V Unit)

/-- **The frame's round over a round's frames**: `roundFree` at the bytes of frames `fs` and the round's squeeze of `n`
coins of kind `k` is the fold of the frame's step over `fs`, then the squeeze. -/
theorem roundFree_encs (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ)
    (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
      (do
        let r ← fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0)
        (fun (_ : Unit) => (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1)) <$>
          (match r.1.inner with
           | some j => innerOwn c r.1 j (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n
           | none => pure () : V Unit)) := by
  unfold roundFree
  simp only [show (s.inner == some INNER_ROUNDS) = false from by simpa using hs, Bool.false_eq_true, ↓reduceIte,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [forIn_range'_foldlM _ (fun r => (none, r.1, r.2.1, r.2.2.1, r.2.2.2)) (stepAt c)
    (lastAt c (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) k n)
    (fun r fs' => (encs fs ++ (Frame.sq k n).enc).drop r.2.2.2 = encs fs' ++ (Frame.sq k n).enc ∧
      ∀ f ∈ fs', midFrame f)
    _ _ _ fs (s, ByteArray.empty, #[], 0) 0 _ _ rfl ⟨by simp, hm⟩ ?bound]
  · cases List.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) fs with
    | error e => rfl
    | ok r =>
      obtain ⟨m, o, fr, p⟩ := r
      cases hi : m.inner with
      | none => simp only [bind, Except.bind, lastAt, hi]; rfl
      | some j =>
        simp only [bind, Except.bind, lastAt, hi]
        cases innerOwn c _ j _ n <;> rfl
  · intro r f fs' i ⟨hd, hmf⟩
    obtain ⟨m, o, fr, p⟩ := r
    dsimp only at hd ⊢
    have hfm : midFrame f := hmf f (by simp)
    rw [encs_cons, List.append_assoc] at hd
    obtain ⟨hop, hkd, hln⟩ := header_at _ p f _ hd (midFrame_len hfm)
    have hL := room _ _ p f hd
    have hb := fun w => body_at _ _ p w f hd
    simp only [size_list, hop, hkd, hln, hb]
    have hr16 : 16 ≤ (encs fs' ++ (Frame.sq k n).enc).length := by
      rw [List.length_append, Frame.enc_length]; simp [Frame.payLen]
    rw [Frame.enc_length] at hL
    rw [ite_cond_eq_false _ _ (eq_false (by omega))]
    cases f with
    | dom b =>
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 1 = 1 from rfl]
      simp only []
      have hc : ¬ (p + 16 + (b.size + 15) / 16 * 16 > (encs fs ++ (Frame.sq k n).enc).length) := by omega
      simp only [hc, ↓reduceIte]
      rfl
    | label b =>
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 2 = 2 from rfl]
      simp only []
      have hc : ¬ (p + 16 + b.size > (encs fs ++ (Frame.sq k n).enc).length) := by omega
      simp only [hc, ↓reduceIte]
      rw [take_padded, pure_bind]
      simp only [stepAt]
      cases m.label c b <;> rfl
    | bytes b =>
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 5 = 5 from rfl]
      simp only []
      have hc : ¬ (p + 16 + b.size > (encs fs ++ (Frame.sq k n).enc).length) := by omega
      simp only [hc, ↓reduceIte]
      rw [take_padded]
      rfl
    | one v =>
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 3 = 3 from rfl]
      simp only []
      have hc : ¬ (p + 16 + 16 * 1 > (encs fs ++ (Frame.sq k n).enc).length) := by omega
      simp only [hc, ↓reduceIte, pure_bind,
        show (!((1 : UInt8) == 2 || (1 : UInt8) == 1 && (1 == 1))) = false from rfl, Bool.false_eq_true]
      rw [observe_loop c (p + 16) _ [v] ?hN _ ?hB]
      · exact observe_tail c [v] _ _ rfl
      · rfl
      · show (List.take 16 (v.toBytes.data.toList ++ _)).toArray.toList = _
        rw [List.take_left' (toBytes_length v)]
        simp [colBytes]
    | vec vs =>
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 3 = 3 from rfl]
      simp only []
      have hc : ¬ (p + 16 + 16 * vs.length > (encs fs ++ (Frame.sq k n).enc).length) := by omega
      simp only [hc, ↓reduceIte, pure_bind,
        show (!((2 : UInt8) == 2 || (2 : UInt8) == 1 && (vs.length == 1))) = false from rfl, Bool.false_eq_true]
      rw [observe_loop c (p + 16) _ vs ?hN' _ ?hB']
      · exact observe_tail c vs _ _ rfl
      · omega
      · show (List.take (16 * vs.length) (colBytes vs ++ _)).toArray.toList = _
        rw [List.take_left' (Frame.colBytes_length vs)]
    | sq k' n' =>
      obtain ⟨rfl, rfl⟩ := hfm
      simp only [Frame.op, Frame.kind, Frame.len, Frame.payLen, Frame.pay] at hL ⊢
      rw [show UInt8.toNat 4 = 4 from rfl]
      rfl
  · intro r f fs' r' ⟨hd, hmf⟩ hst
    refine ⟨?_, fun g hg => hmf g (List.mem_cons_of_mem _ hg)⟩
    rw [stepAt_pos c r r' f hst, ← List.drop_drop, hd, encs_cons, List.append_assoc, List.drop_left]
  · intro r i hI
    obtain ⟨hd, -⟩ := hI
    obtain ⟨m, o, fr, p⟩ := r
    dsimp only at hd ⊢
    have hd' : (encs fs ++ (Frame.sq k n).enc).drop p = (Frame.sq k n).enc ++ [] := by
      rw [hd, List.append_nil]; rfl
    obtain ⟨hop, hkd, hln⟩ := header_at _ p _ [] hd' hn
    have hL := room _ _ p _ hd'
    simp only [size_list, hop, hkd, hln]
    simp only [Frame.op, Frame.kind, Frame.len, Frame.enc_length, Frame.payLen, List.length_nil] at hL ⊢
    have hc : ¬ (p + 16 > (encs fs ++ (Frame.sq k n).enc).length) := by omega
    have h1 : (n == 0 && k == 2) = false := by
      rcases hk with rfl | ⟨rfl, rfl⟩
      · simpa using fun h => h0 ⟨h, rfl⟩
      · rfl
    have h2 : (p + 16 != (encs fs ++ (Frame.sq k n).enc).length) = false := by
      rw [show p + 16 = (encs fs ++ (Frame.sq k n).enc).length by omega]
      exact bne_self_eq_false _
    have h4 : (!(k == 2 || k == 1 && n == 1)) = false := by rcases hk with rfl | ⟨rfl, rfl⟩ <;> rfl
    rw [show UInt8.toNat 4 = 4 from rfl]
    simp only [hc, h1, h2, h4, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
    cases hi : m.inner with
    | none => simp only [lastAt, hi]; rfl
    | some j =>
      simp only [lastAt, hi]
      cases innerOwn c _ j _ n <;> rfl
  · simp only [size_list, List.length_append, Frame.enc_length]
    have := encs_length fs
    omega

end Claims.ZKServer
