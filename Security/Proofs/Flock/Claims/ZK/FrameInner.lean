import Proofs.Flock.Claims.ZK.FrameOwn
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Soundness.Refine.Arith

/-!
# What the frame takes at the inner proof's rounds (Stage B, R3)

At the inner proof's rounds the frame fixes every frame and value it computes from its draws and the coins
(`innerOwn`), so a round it takes there is the firewall's own (`ownRound`) at its configuration and state, whatever
`final_c'` and `τ`'s commitment it carries: `innerOwn_inv`, the converse of `innerOwn_ownRound`.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame wordBytes innerOwn)

/-- A round fixed but for `f` bytes after `pre` is `pre`, those bytes, then `post`. -/
theorem fixedBut_inv {xb pre post : ByteArray} {f : ℕ}
    (h : (xb.size == pre.size + f + post.size && xb.extract 0 pre.size == pre &&
      xb.extract (pre.size + f) xb.size == post) = true) :
    xb = pre ++ xb.extract pre.size (pre.size + f) ++ post ∧ (xb.extract pre.size (pre.size + f)).size = f := by
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨h0, h1⟩, h2⟩ := h
  have hs : xb.size = pre.size + f + post.size := beq_iff_eq.1 h0
  have e1 : xb.extract 0 pre.size = pre :=
    ByteArray.ext (beq_iff_eq.1 (show ((xb.extract 0 pre.size).data == pre.data) = true from h1))
  have e2 : xb.extract (pre.size + f) xb.size = post :=
    ByteArray.ext (beq_iff_eq.1 (show ((xb.extract (pre.size + f) xb.size).data == post.data) = true from h2))
  refine ⟨?_, by rw [ByteArray.size_extract]; omega⟩
  conv_lhs => rw [← ByteArray.extract_zero_size (b := xb)]
  rw [ByteArray.extract_eq_extract_append_extract (a := xb) (i := 0) (k := xb.size) pre.size (by omega) (by omega),
    ByteArray.extract_eq_extract_append_extract (a := xb) (i := pre.size) (k := xb.size) (pre.size + f) (by omega)
      (by omega), e1, e2, ByteArray.append_assoc]

theorem eq_of_bne_false {xb yb : ByteArray} (h : (xb != yb) = false) : xb = yb := by
  have h' : (xb == yb) = true := by simpa only [bne, Bool.not_eq_false'] using h
  exact ByteArray.ext (beq_iff_eq.1 (show (xb.data == yb.data) = true from h'))

/-- **What the frame takes at the inner proof's round `k` is the firewall's own** (`ownRound`, at the frame's
configuration and state), at some 16-byte `final_c'` and 64-byte `τ` commitment: the converse of `innerOwn_ownRound`. -/
theorem innerOwn_inv {c : Cfg} {s : Masker} {xb : ByteArray} {n : ℕ} :
    ∀ {k : ℕ}, innerOwn c s k xb n = .ok () →
      k < 4 ∧ ∃ w t, w.size = 16 ∧ t.size = 64 ∧ xb = ownRound c s w t n k
  | 0, h => by
    refine ⟨by omega, ?_⟩
    cases hB : (xb.size == (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size + 16 +
        (header 4 1 1).size &&
        xb.extract 0 (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size ==
          labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 &&
        xb.extract ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size + 16) xb.size ==
          header 4 1 1) with
    | false =>
      unfold innerOwn at h
      simp only [hB, Bool.not_false, ↓reduceIte] at h
      cases h
    | true =>
      obtain ⟨hX, hw⟩ := fixedBut_inv hB
      exact ⟨_, ByteArray.mk (Array.replicate 64 0), hw, by simp [ByteArray.size], hX⟩
  | 1, h => by
    refine ⟨by omega, ByteArray.mk (Array.replicate 16 0), ByteArray.mk (Array.replicate 64 0),
      by simp [ByteArray.size], by simp [ByteArray.size], ?_⟩
    unfold innerOwn at h
    cases hB : (xb != header 3 2 2 ++ wordBytes #[s.eps * c.h[c.lay.fa]! + c.h[c.lay.tx]!,
        c.h[c.lay.fb]! + c.h[c.lay.ty]!] ++ header 4 2 n) with
    | true =>
      simp only [hB, ↓reduceIte] at h
      cases h
    | false => exact eq_of_bne_false hB
  | 2, h => by
    refine ⟨by omega, ?_⟩
    cases hB : (xb.size == (header 5 0 64).size + 64 + (header 4 1 1).size &&
        xb.extract 0 (header 5 0 64).size == header 5 0 64 &&
        xb.extract ((header 5 0 64).size + 64) xb.size == header 4 1 1) with
    | false =>
      unfold innerOwn at h
      simp only [hB, Bool.not_false, ↓reduceIte] at h
      cases h
    | true =>
      obtain ⟨hX, ht⟩ := fixedBut_inv hB
      exact ⟨ByteArray.mk (Array.replicate 16 0), _, by simp [ByteArray.size], ht, hX⟩
  | 3, h => by
    refine ⟨by omega, ByteArray.mk (Array.replicate 16 0), ByteArray.mk (Array.replicate 64 0),
      by simp [ByteArray.size], by simp [ByteArray.size], ?_⟩
    unfold innerOwn at h
    cases hB : (xb != header 3 2 ((c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b).size ++
        wordBytes ((c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b) ++ header 4 2 n) with
    | true =>
      simp only [hB, ↓reduceIte] at h
      cases h
    | false => exact eq_of_bne_false hB
  | k + 4, h => by
    unfold innerOwn at h
    cases h

/-! ## The firewall's own rounds as frames -/

open FlockSoundness.Refine (Frame encs colBytes padded_data)

/-- The frames the firewall's own round `k` absorbs (`ownRound`), before its squeeze. -/
def ownFrames (c : Cfg) (s : Masker) (w t : ByteArray) : ℕ → List Frame
  | 0 => [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one (readWord w.data.toList)]
  | 1 => [.vec (#[s.eps * c.h[c.lay.fa]! + c.h[c.lay.tx]!, c.h[c.lay.fb]! + c.h[c.lay.ty]!] : Array F128).toList]
  | 2 => [.bytes t]
  | _ => [.vec ((c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b).toList]

/-- The squeeze that ends the firewall's own round `k`. -/
def ownSq (n : ℕ) : ℕ → Frame
  | 0 => .sq 1 1
  | 1 => .sq 2 n
  | 2 => .sq 1 1
  | _ => .sq 2 n

theorem wordBytes_data (v : Array F128) : (wordBytes v).data.toList = colBytes v.toList :=
  FlockSoundness.Refine.rowBytes_data v

theorem enc_sq (k : UInt8) (n : ℕ) : (Frame.sq k n).enc = (header 4 k n).data.toList := List.append_nil _

/-- **The firewall's own round is frames:** `ownFrames`, then `ownSq`. -/
theorem ownRound_encs {c : Cfg} {s : Masker} {w t : ByteArray} {n : ℕ} (hw : w.size = 16) (ht : t.size = 64) :
    ∀ {k : ℕ}, k < 4 → (ownRound c s w t n k).data.toList = encs (ownFrames c s w t k) ++ (ownSq n k).enc
  | 0, _ => by
    have e1 : (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).data.toList =
        (Frame.label _root_.Flock.Zk.LABEL_INNER.toUTF8).enc := by
      unfold labelFrame; rw [ByteArray.toList_data_append]; rfl
    have e2 : (Frame.one (readWord w.data.toList)).enc = (header 3 1 1).data.toList ++ w.data.toList := by
      rw [enc_one, toBytes_readWord _ (by rw [Array.length_toList]; exact hw)]
    simp only [ownRound, ownFrames, ownSq, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      ByteArray.toList_data_append, e1, e2, enc_sq, List.append_assoc]
  | 1, _ => by
    simp only [ownRound, ownFrames, ownSq, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      ByteArray.toList_data_append, enc_vec, wordBytes_data, enc_sq, List.append_assoc]
    rfl
  | 2, _ => by
    have e : (Frame.bytes t).enc = (header 5 0 64).data.toList ++ t.data.toList := by
      simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay, padded_data, ht]
    simp only [ownRound, ownFrames, ownSq, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      ByteArray.toList_data_append, e, enc_sq, List.append_assoc]
  | 3, _ => by
    simp only [ownRound, ownFrames, ownSq, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      ByteArray.toList_data_append, enc_vec, wordBytes_data, enc_sq, List.append_assoc, Array.length_toList]

/-! ## At zero draws the firewall's own rounds read no coin -/

open FlockSoundness.Refine (fOf_injective fOf_mul fOf_add fOf_zero)
open _root_.Flock.Firewall.Frame (Draws words)

theorem f128_mul_zero (a : F128) : a * 0 = 0 := fOf_injective (by rw [fOf_mul, fOf_zero, mul_zero])

theorem f128_zero_add_zero : (0 : F128) + 0 = 0 := fOf_injective (by rw [fOf_add, fOf_zero, add_zero])

/-- Every word of `a` is zero. -/
def AllZero (a : Array F128) : Prop := ∀ x ∈ a, x = 0

theorem AllZero.get! {a : Array F128} (h : AllZero a) (i : ℕ) : a[i]! = 0 := by
  by_cases hi : i < a.size
  · rw [getElem!_pos a i hi]; exact h _ (Array.getElem_mem hi)
  · rw [getElem!_neg a i hi]; rfl

theorem AllZero.set! {a : Array F128} (h : AllZero a) (j : ℕ) {v : F128} (hv : v = 0) : AllZero (a.set! j v) :=
  fun x hx => by
    rw [Array.set!_eq_setIfInBounds] at hx
    rcases Array.mem_or_eq_of_mem_setIfInBounds hx with hx | rfl
    · exact h x hx
    · exact hv

theorem AllZero.append {a b : Array F128} (ha : AllZero a) (hb : AllZero b) : AllZero (a ++ b) := fun x hx => by
  rcases Array.mem_append.1 hx with hx | hx
  · exact ha x hx
  · exact hb x hx

theorem zeros_get! (N i : ℕ) : (Loop.zeros N).get! i = 0 := by
  rw [FlockSoundness.Discharge.ZkView.get!_eq]
  by_cases hi : i < N
  · rw [getElem!_pos _ i (by simpa [Loop.zeros] using hi)]
    simp [Loop.zeros]
  · rw [getElem!_neg _ i (by simpa [Loop.zeros] using hi)]
    rfl

theorem u64OfLE_zeros (N o : ℕ) : _root_.Flock.u64OfLE (Loop.zeros N) o = 0 := by
  rw [FlockSoundness.Discharge.ZkView.u64OfLE_eq]
  simp only [zeros_get!]
  decide

theorem words_zeros (N o n : ℕ) : AllZero (words (Loop.zeros N) o n) := fun x hx => by
  unfold words at hx
  obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hx
  rw [u64OfLE_zeros, u64OfLE_zeros]
  rfl

/-- A configuration whose pads and rows are zero: the placeholder's, at zero draws. -/
structure ZeroCfg (c : Cfg) : Prop where
  h : AllZero c.h
  rowH : AllZero c.rowH
  rowMu : AllZero c.rowMu

theorem zeroCfg_ofDraws (l : _root_.Flock.Zk.PadLayout) (N : ℕ) (m : Option _root_.Flock.Firewall.Frame.MaskSlot) :
    ZeroCfg (Cfg.ofDraws l (Draws.ofBytes l (Loop.zeros N)) m) := by
  have h0 := words_zeros N 0 l.len
  have h1 := h0.set! l.hab (v := (words (Loop.zeros N) 0 l.len)[l.fa]! * (words (Loop.zeros N) 0 l.len)[l.fb]!)
    (by rw [h0.get! l.fb, f128_mul_zero])
  have h2 := h1.set! l.tw
    (v := ((words (Loop.zeros N) 0 l.len).set! l.hab
      ((words (Loop.zeros N) 0 l.len)[l.fa]! * (words (Loop.zeros N) 0 l.len)[l.fb]!))[l.tx]! *
      ((words (Loop.zeros N) 0 l.len).set! l.hab
      ((words (Loop.zeros N) 0 l.len)[l.fa]! * (words (Loop.zeros N) 0 l.len)[l.fb]!))[l.ty]!)
    (by rw [h1.get! l.ty, f128_mul_zero])
  exact ⟨h2, h2.append (words_zeros _ _ _), (words_zeros _ _ _).append (words_zeros _ _ _)⟩

theorem zip_map_zero {a b : Array F128} (ha : AllZero a) (hb : AllZero b) (β : F128) :
    ((a.zip b).map fun (x, y) => x + β * y) = (a.zip b).map fun _ => (0 : F128) := by
  refine Array.ext (by simp only [Array.size_map]) fun i h1 _ => ?_
  have hi : i < (a.zip b).size := by simpa using h1
  have hia : i < a.size := by simp only [Array.size_zip] at hi; omega
  have hib : i < b.size := by simp only [Array.size_zip] at hi; omega
  rw [Array.getElem_map, Array.getElem_map, Array.getElem_zip]
  simp only [ha _ (Array.getElem_mem hia), hb _ (Array.getElem_mem hib), f128_mul_zero, f128_zero_add_zero]

theorem ownRound_zero0 {c : Cfg} (s s' : Masker) (w t : ByteArray) (n : ℕ) :
    ownRound c s w t n 0 = ownRound c s' w t n 0 := by
  simp only [ownRound]

theorem ownRound_zero1 {c : Cfg} (hz : ZeroCfg c) (s s' : Masker) (w t : ByteArray) (n : ℕ) :
    ownRound c s w t n 1 = ownRound c s' w t n 1 := by
  simp only [ownRound, hz.h.get!, f128_mul_zero]

theorem ownRound_zero2 {c : Cfg} (s s' : Masker) (w t : ByteArray) (n : ℕ) :
    ownRound c s w t n 2 = ownRound c s' w t n 2 := by
  simp only [ownRound]

theorem ownRound_zero3 {c : Cfg} (hz : ZeroCfg c) (s s' : Masker) (w t : ByteArray) (n : ℕ) :
    ownRound c s w t n 3 = ownRound c s' w t n 3 := by
  simp only [ownRound, zip_map_zero hz.rowH hz.rowMu]

/-- **At zero pads the firewall's own round reads no coin**: it is the same at any `ε` and `β`. -/
theorem ownRound_zero {c : Cfg} (hz : ZeroCfg c) (s s' : Masker) (w t : ByteArray) (n : ℕ) :
    ∀ {k : ℕ}, k < 4 → ownRound c s w t n k = ownRound c s' w t n k
  | 0, _ => ownRound_zero0 s s' w t n
  | 1, _ => ownRound_zero1 hz s s' w t n
  | 2, _ => ownRound_zero2 s s' w t n
  | 3, _ => ownRound_zero3 hz s s' w t n

/-- **At zero pads the frame takes the firewall's own round at any state's coins.** -/
theorem innerOwn_zero {c : Cfg} (hz : ZeroCfg c) (s s₀ : Masker) {w t : ByteArray} (hw : w.size = 16)
    (ht : t.size = 64) (n : ℕ) {k : ℕ} (hk : k < 4) : innerOwn c s k (ownRound c s₀ w t n k) n = .ok () := by
  rw [ownRound_zero hz s₀ s w t n hk]
  exact innerOwn_ownRound c s w t hw ht n k hk

/-! ## A round the frame takes ends in its squeeze -/

open _root_.Flock (u64OfLE)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS roundFree)

theorem forIn_ok_inv {τ : Type} (P : τ → Prop) (g : ℕ → τ → V (ForInStep τ))
    (hg : ∀ i x y, P x → g i x = .ok y → P y.value) :
    ∀ (l : List ℕ) (x : τ), P x → ∀ y, forIn l x g = .ok y → P y
  | [], x, hx, y, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact hx
  | a :: l, x, hx, y, h => by
    rw [List.forIn_cons] at h
    cases hga : g a x with
    | error e => simp only [hga, bind, Except.bind, reduceCtorEq] at h
    | ok z =>
      simp only [hga, bind, Except.bind] at h
      cases z with
      | done b =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact hg a x _ hx hga
      | yield b => exact forIn_ok_inv P g hg l b (hg a x _ hx hga) y h

/-- What the frame's round leaves where it took a round `xb` of `n` coins: its squeeze is the round's last 16 bytes, a
header of `n` coins of a kind the frame squeezes, and the inner proof's own round passed where one is open. -/
def SqOk (c : Cfg) (xb : ByteArray) (n : ℕ) (r : ByteArray × Masker × Array ℕ) : Prop :=
  16 ≤ xb.size ∧ (xb.get! (xb.size - 16)).toNat = 4 ∧ (u64OfLE xb (xb.size - 16 + 8)).toNat = n ∧
    ((u64OfLE xb (xb.size - 16 + 8)).toNat == 0 && xb.get! (xb.size - 16 + 1) == 2) = false ∧
    (xb.get! (xb.size - 16 + 1) == 2 || xb.get! (xb.size - 16 + 1) == 1 && n == 1) = true ∧
    ∀ k, r.2.1.inner = some k → innerOwn c r.2.1 k xb n = .ok ()

theorem bindV_ok {α β : Type} {m : V α} {f : α → V β} {b : β} (h : (m >>= f) = .ok b) : ∃ a, m = .ok a ∧ f a = .ok b := by
  cases hm : m with
  | error e => simp only [hm, bind, Except.bind, reduceCtorEq] at h
  | ok a => exact ⟨a, rfl, by simpa only [hm, bind, Except.bind] using h⟩

theorem innerOwn_owed (c : Cfg) (s : Masker) (o : Option ℕ) (k : ℕ) (xb : ByteArray) (n : ℕ) :
    innerOwn c { s with owed := o } k xb n = innerOwn c s k xb n := rfl

theorem sqOk_of {c : Cfg} {xb : ByteArray} {n : ℕ} (p : ℕ) {o : ByteArray} {s : Masker} {fr : Array ℕ}
    (hp : ¬(p + 16 != xb.size) = true) (h4 : (xb.get! p).toNat = 4) (hl : ¬((u64OfLE xb (p + 8)).toNat != n) = true)
    (hz : ¬((u64OfLE xb (p + 8)).toNat == 0 && xb.get! (p + 1) == 2) = true)
    (hk : ¬(!(xb.get! (p + 1) == 2 || xb.get! (p + 1) == 1 && n == 1)) = true)
    (hi : ∀ k, s.inner = some k → innerOwn c s k xb n = .ok ()) : SqOk c xb n (o, { s with owed := some n }, fr) := by
  have hp' : p + 16 = xb.size := by simpa using hp
  unfold SqOk
  rw [show xb.size - 16 = p by omega]
  have hk2 : (xb.get! (p + 1) == 2 || xb.get! (p + 1) == 1 && n == 1) = true := by
    generalize (xb.get! (p + 1) == 2 || xb.get! (p + 1) == 1 && n == 1) = b at hk ⊢
    cases b <;> simp_all
  refine ⟨by omega, h4, by simpa using hl, by simpa using hz, hk2, fun k hk' => ?_⟩
  rw [innerOwn_owed]
  exact hi k hk'

set_option maxHeartbeats 1000000 in
/-- **A round the frame takes ends in its squeeze** (`SqOk`): `roundFree`'s loop returns only there. -/
theorem roundFree_sq {c : Cfg} {s : Masker} {xb : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (hs : s.inner ≠ some INNER_ROUNDS) (h : roundFree c s xb n = .ok q) : SqOk c xb n q := by
  unfold roundFree at h
  simp only [show (s.inner == some INNER_ROUNDS) = false from by simpa using hs, Bool.false_eq_true, ↓reduceIte,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨y, hy, hk⟩ := bindV_ok h
  have hP := forIn_ok_inv (fun x => ∀ r, x.1 = some r → SqOk c xb n r) _ ?hg _ _ (fun r hr => by cases hr) y hy
  · clear h hy
    revert hk hP
    obtain ⟨o, rest⟩ := y
    cases o with
    | none => intro hk _; cases hk
    | some r => intro hk hP; cases hk; exact hP _ rfl
  · clear h hy hk
    intro i x z hx hz r hr
    beta_reduce at hz
    by_cases h1 : x.2.2.2.2 + 16 > xb.size
    · rw [ite_cond_eq_true _ _ (eq_true h1)] at hz
      rw [ite_eq_iff] at hz
      rcases hz with ⟨_, hz⟩ | ⟨_, hz⟩ <;> obtain ⟨_, h, _⟩ := bindV_ok hz <;> cases h
    · rw [ite_cond_eq_false _ _ (eq_false h1)] at hz
      split at hz
      all_goals
        repeat' (first
          | (obtain ⟨_, _, hz⟩ := bindV_ok hz; try beta_reduce at hz)
          | (rw [ite_eq_iff] at hz; rcases hz with ⟨_, hz⟩ | ⟨_, hz⟩)
          | (split at hz))
      all_goals first | (cases hz; cases hr; done) | (cases hz; done) | contradiction | skip
      all_goals first
        | (cases hz; cases hr
           exact sqOk_of x.2.2.2.2 ‹_› ‹_› ‹_› ‹_› ‹_› fun k hk => (‹∀ i, x.2.1.inner = some i → False› k hk).elim)
        | (rename_i j hj hz0 _ hio
           rw [hio] at hz0
           cases hz0; cases hr
           refine sqOk_of x.2.2.2.2 ‹_› ‹_› ‹_› ‹_› ‹_› fun k hk => ?_
           have e : x.2.1.inner = some k := hk
           rw [hj] at e
           obtain rfl := Option.some.inj e
           exact hio)

end Claims.ZKServer
