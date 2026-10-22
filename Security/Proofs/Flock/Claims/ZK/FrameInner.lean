import Proofs.Flock.Claims.ZK.FrameOwn
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Claims.ZK.FrameProg
import Proofs.Flock.Soundness.Refine.Arith

/-!
# What the frame takes at the inner proof's rounds (Stage B, R3)

At the inner proof's rounds the frame fixes every frame and value it computes from its draws and the coins
(`innerOwn`), so a round it takes there is the firewall's own (`ownRound`) at its configuration and state, whatever
`final_c'` and `τ`'s commitment it carries: `innerOwn_inv`, the converse of `innerOwn_ownRound`.

* `ownRound_encs`: the firewall's own round `k` is frames (`ownFrames`), then its squeeze (`ownSq`).
* `innerOwn_zero`: at zero pads (`ZeroCfg`, the placeholder's, `zeroCfg_ofDraws`) the own round reads no coin.
* `roundFree_sq`, `roundFree_inner_some`: a round the frame takes ends in its squeeze (`SqOk`), and inside the inner
  proof keeps its count, so there it is the firewall's own round's frames (`roundFree_own_frames`).
* `roundFree_own_ok`: the placeholder's own round at zero pads passes where the deployed frame took a round of its
  shape, from the deployed state but for `stopped`, and leaves the same state but for `stopped`; so where the deployed
  frame's check releases a round past the inner proof's label (`check_own_ok`), and at it where the scheduled round in
  its place is the firewall's own round 0 at some `final_c'` (`check_own0_ok`).
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

theorem loopZeros_get! (N i : ℕ) : (Loop.zeros N).get! i = 0 := by
  rw [FlockSoundness.Discharge.ZkView.get!_eq]
  by_cases hi : i < N
  · rw [getElem!_pos _ i (by simpa [Loop.zeros] using hi)]
    simp [Loop.zeros]
  · rw [getElem!_neg _ i (by simpa [Loop.zeros] using hi)]
    rfl

theorem u64OfLE_zeros (N o : ℕ) : _root_.Flock.u64OfLE (Loop.zeros N) o = 0 := by
  rw [FlockSoundness.Discharge.ZkView.u64OfLE_eq]
  simp only [loopZeros_get!]
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

/-! ## The placeholder takes its own round where the deployed frame took one of its shape -/

open _root_.Flock.Firewall.Frame (isLabel)

theorem relM_ok_left {α β : Type} {f : α → β} {y : V α} {a : α} (h : RelM f (.ok a) y) :
    ∃ b, y = .ok b ∧ f a = f b := by
  cases y with
  | ok b => exact ⟨b, rfl, h⟩
  | error e => exact (h : False).elim

theorem midFrame_forall₂ {fs fs' : List Frame} (hA : List.Forall₂ Alike fs fs') :
    (∀ f ∈ fs, midFrame f) → ∀ f ∈ fs', midFrame f := by
  induction hA with
  | nil => exact fun _ f hf => by cases hf
  | cons ha _ ih =>
    intro hm g hg
    rcases List.mem_cons.mp hg with rfl | hg
    · exact Sched.midFrame_alike ha (hm _ List.mem_cons_self)
    · exact ih (fun x hx => hm x (List.mem_cons_of_mem _ hx)) g hg

/-- The firewall's own rounds at two configurations of one row size are of one shape, whatever `final_c'`, `τ` and the
coins. -/
theorem ownFrames_alike {c c' : Cfg} (hH : c.rowH.size = c'.rowH.size) (hM : c.rowMu.size = c'.rowMu.size)
    (s s' : Masker) {w w' t t' : ByteArray} (ht : t.size = t'.size) :
    ∀ k, List.Forall₂ Alike (ownFrames c s w t k) (ownFrames c' s' w' t' k)
  | 0 => .cons (show Alike (.label _) (.label _) from rfl) (.cons True.intro .nil)
  | 1 => .cons (show List.length _ = List.length _ by simp) .nil
  | 2 => .cons ht .nil
  | k + 3 => .cons (show List.length _ = List.length _ by simp [hH, hM]) .nil

theorem ownFrames_notInner (c : Cfg) (s : Masker) (w t : ByteArray) :
    ∀ {k : ℕ}, 0 < k → ∀ f ∈ ownFrames c s w t k, f ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8
  | 0, h, _, _ => (Nat.lt_irrefl 0 h).elim
  | 1, _, f, hf => by rw [List.mem_singleton.mp hf]; nofun
  | 2, _, f, hf => by rw [List.mem_singleton.mp hf]; nofun
  | _ + 3, _, f, hf => by rw [List.mem_singleton.mp hf]; nofun

theorem isLabel_inner : isLabel _root_.Flock.Zk.LABEL_INNER.toUTF8 _root_.Flock.Zk.LABEL_INNER = true := by
  unfold isLabel
  exact beq_self_eq_true _

/-- The inner proof's label starts its count. -/
theorem label_inner_zero (c : Cfg) (s : Masker) {q : Masker × ByteArray}
    (h : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q) : q.1.inner = some 0 := by
  have key : (s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8).map (fun q => q.1.inner) =
      (s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8).map (fun _ => some 0) := by
    unfold Masker.label
    cases hm : c.mask <;>
    simp only [isLabel_inner, ↓reduceIte, apply_ite (Except.map _), exMap_bind, exMap_pure, Sched.exMap_throw,
      Sched.throw_bindV]
  rw [h] at key
  exact Except.ok.inj key

/-- The frame's steps over the firewall's own round 0 start the inner proof's count. -/
theorem foldlM_own0_inner (c : Cfg) (s₀ : Masker) (w t : ByteArray) (r q : Masker × ByteArray × Array ℕ × ℕ)
    (h : (ownFrames c s₀ w t 0).foldlM (stepAt c) r = .ok q) : q.1.inner = some 0 := by
  simp only [ownFrames, List.foldlM_cons, List.foldlM_nil] at h
  obtain ⟨a, ha, h⟩ := bindV_ok h
  obtain ⟨b, hb, h⟩ := bindV_ok h
  obtain rfl : b = q := Except.ok.inj h
  rw [Sched.stepAt_inner c a b _ (by nofun) hb]
  simp only [stepAt] at ha
  cases hl : r.1.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 with
  | error e => rw [hl] at ha; cases ha
  | ok l =>
    rw [hl] at ha
    cases ha
    exact label_inner_zero c r.1 hl

/-- The frame's steps over the firewall's own rounds past the first leave the inner proof's count as it was. -/
theorem foldlM_own_inner (c : Cfg) (s₀ : Masker) (w t : ByteArray) {i : ℕ} (hi : 0 < i)
    (r q : Masker × ByteArray × Array ℕ × ℕ) (h : (ownFrames c s₀ w t i).foldlM (stepAt c) r = .ok q) :
    q.1.inner = r.1.inner :=
  Sched.foldlM_stepAt_inner c _ r q (ownFrames_notInner c s₀ w t hi) h

/-- **The placeholder takes its own round where the deployed frame took a round of its shape**: at zero pads
(`ZeroCfg`), from the deployed state but for `stopped`, the placeholder's own round `i` (`ownRound` at zero `final_c'`
and `τ`) passes `roundFree` and leaves the deployed state but for `stopped`, where the placeholder's frames reach the
inner proof's round `i` (`hI`). -/
theorem roundFree_own_ok {c c' : Cfg} (hc : CfgAlike c c') (hz : ZeroCfg c') (s s₀ : Masker) (b : Bool) {i n : ℕ}
    {k : UInt8} (hi : i < 4) (hs : s.inner ≠ some INNER_ROUNDS) (hsq : ownSq n i = .sq k n)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {fs : List Frame}
    (hm : ∀ f ∈ fs, midFrame f) (hA : List.Forall₂ Alike fs (ownFrames c' s₀ (Loop.zeros 16) (Loop.zeros 64) i))
    (hI : ∀ q, (ownFrames c' s₀ (Loop.zeros 16) (Loop.zeros 64) i).foldlM (stepAt c')
      ({ s with stopped := b }, ByteArray.empty, #[], 0) = .ok q → q.1.inner = some i)
    {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q) :
    ∃ q', roundFree c' { s with stopped := b } (ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n i) n = .ok q' ∧
      unstop q.2.1 = unstop q'.2.1 := by
  have hz16 : (Loop.zeros 16).size = 16 := by simp [Loop.zeros, ByteArray.size]
  have hz64 : (Loop.zeros 64).size = 64 := by simp [Loop.zeros, ByteArray.size]
  have hX : ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n i = ByteArray.mk (encs (ownFrames c' s₀
      (Loop.zeros 16) (Loop.zeros 64) i) ++ (Frame.sq k n).enc).toArray := by
    rw [← hsq, ← ownRound_encs hz16 hz64 hi, Array.toArray_toList]
  have hm' := midFrame_forall₂ hA hm
  rw [roundFree_encs c s fs k n hs hm hk h0 hn] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hR := foldlM_stepAt_alike hc fs _ hA (s, ByteArray.empty, #[], 0)
    ({ s with stopped := b }, ByteArray.empty, #[], 0) rfl
  rw [hr] at hR
  obtain ⟨r', hr', he⟩ := relM_ok_left hR
  obtain rfl := exMap_ok_eq h
  have hin := hI r' hr'
  rw [hX, roundFree_encs c' { s with stopped := b } _ k n hs hm' hk h0 hn, hr']
  simp only [bind, Except.bind, hin]
  rw [← hX, innerOwn_zero hz r'.1 s₀ hz16 hz64 n hi]
  refine ⟨_, rfl, ?_⟩
  have hu : unstop r.1 = unstop r'.1 := congrArg Prod.fst he
  rw [← hin]
  exact congrArg (fun x : Masker => { x with owed := some n }) hu

/-! ## Inside the inner proof a round keeps its count -/

/-- The inner proof's label inside the inner proof is refused. -/
theorem label_inner_twice (c : Cfg) (s : Masker) {j : ℕ} (hs : s.inner = some j) {q : Masker × ByteArray}
    (h : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q) : False := by
  have key : (s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8).map (fun _ => true) =
      (s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8).map (fun _ => false) := by
    unfold Masker.label
    cases hm : c.mask <;>
    simp only [isLabel_inner, hs, Option.isSome_some, ↓reduceIte, apply_ite (Except.map _), exMap_bind,
      Sched.exMap_throw, Sched.throw_bindV]
  rw [h] at key
  exact Bool.noConfusion (Except.ok.inj key)

/-- Inside the inner proof a label keeps the inner proof's count. -/
theorem label_inner_some (c : Cfg) (s : Masker) {j : ℕ} (hs : s.inner = some j) (b : ByteArray)
    {q : Masker × ByteArray} (h : s.label c b = .ok q) : q.1.inner = some j := by
  by_cases hb : b = _root_.Flock.Zk.LABEL_INNER.toUTF8
  · subst hb
    exact (label_inner_twice c s hs h).elim
  · rw [Sched.label_inner c s b hb h, hs]

set_option maxHeartbeats 1000000 in
/-- **Inside the inner proof a round the frame takes keeps the inner proof's count.** -/
theorem roundFree_inner_some {c : Cfg} {s : Masker} {xb : ByteArray} {n j : ℕ} {q : ByteArray × Masker × Array ℕ}
    (hs : s.inner ≠ some INNER_ROUNDS) (hj : s.inner = some j) (h : roundFree c s xb n = .ok q) :
    q.2.1.inner = some j := by
  unfold roundFree at h
  simp only [show (s.inner == some INNER_ROUNDS) = false from by simpa using hs, Bool.false_eq_true, ↓reduceIte,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨y, hy, hk⟩ := bindV_ok h
  have hP := forIn_ok_inv (fun x => x.2.1.inner = some j ∧ ∀ r, x.1 = some r → r.2.1.inner = some j) _ ?hg _ _
    ⟨hj, fun r hr => by cases hr⟩ y hy
  · clear h hy
    revert hk hP
    obtain ⟨o, rest⟩ := y
    cases o with
    | none => intro hk _; cases hk
    | some r => intro hk hP; cases hk; exact hP.2 _ rfl
  · clear h hy hk
    intro i x z hx hz
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
      all_goals first
        | contradiction
        | (cases hz; done)
        | (cases hz; refine ⟨hx.1, fun r hr => ?_⟩; cases hr; done)
        | (cases hz; refine ⟨hx.1, fun r hr => ?_⟩; cases hr; exact hx.1)
        | (cases hz
           refine ⟨label_inner_some c x.2.1 hx.1 _ ‹Masker.label c x.2.1 _ = .ok _›, fun r hr => ?_⟩
           cases hr)
        | (cases hz
           refine ⟨forIn_ok_inv (fun y : Masker × ByteArray × Array ℕ => y.1.inner = some j) _
             (fun i y y' hy hy' => ?_) _ _ hx.1 _ ‹forIn _ _ _ = Except.ok _›, fun r hr => ?_⟩
           · beta_reduce at hy'
             split at hy' <;>
               (obtain ⟨_, ha, hy'⟩ := bindV_ok hy'
                cases hy'
                exact (Sched.absorb_inner c _ _ ha).trans hy)
           · cases hr)

/-- A squeeze the frame took ends the round's frames in the count it asked for, of a kind it squeezes. -/
theorem sqOk_frames {c : Cfg} {xb : ByteArray} {n : ℕ} {r : ByteArray × Masker × Array ℕ} {fs : List Frame}
    {k : UInt8} {m : ℕ} (hx : xb.data.toList = encs fs ++ (Frame.sq k m).enc) (hm : m < 2 ^ 64) (h : SqOk c xb n r) :
    m = n ∧ (k = 2 ∨ (k = 1 ∧ n = 1)) ∧ ¬ (n = 0 ∧ k = 2) ∧ n < 2 ^ 64 := by
  have hl : (Frame.sq k m).enc.length = 16 := by rw [Frame.enc_length]; rfl
  have hsz : xb.size = (encs fs).length + 16 := by
    rw [← hl, ← List.length_append, ← hx, Array.length_toList]
    rfl
  have e : xb = ByteArray.mk xb.data.toList.toArray := by rw [Array.toArray_toList]
  have hd : xb.data.toList.drop (encs fs).length = (Frame.sq k m).enc ++ [] := by
    rw [hx, List.drop_left, List.append_nil]
  obtain ⟨-, hkind, hlen⟩ := header_at xb.data.toList _ _ [] hd hm
  rw [← e] at hkind hlen
  simp only [Frame.kind, Frame.len] at hkind hlen
  obtain ⟨-, -, h3, h4, h5, -⟩ := h
  have hp : xb.size - 16 = (encs fs).length := by omega
  rw [hp] at h3 h4 h5
  rw [hlen] at h3 h4
  rw [hkind] at h4 h5
  have h3' : m = n := h3
  subst h3'
  refine ⟨rfl, by simpa using h5, fun ⟨h0, hk⟩ => ?_, hm⟩
  subst h0 hk
  simp at h4

theorem ownSq_sq {n : ℕ} (hn : n < 2 ^ 64) : ∀ i, ∃ k m, ownSq n i = .sq k m ∧ m < 2 ^ 64
  | 0 => ⟨1, 1, rfl, by norm_num⟩
  | 1 => ⟨2, n, rfl, hn⟩
  | 2 => ⟨1, 1, rfl, by norm_num⟩
  | _ + 3 => ⟨2, n, rfl, hn⟩

/-- **Inside the inner proof the frame takes only the firewall's own round**: at the inner proof's round `i`, a round
`roundFree` takes is `ownRound`'s frames at the state it leaves, at some `final_c'` and `τ`, and squeezes as the
firewall's own round does. -/
theorem roundFree_own_frames {c : Cfg} {s : Masker} {xb : ByteArray} {n i : ℕ} {q : ByteArray × Masker × Array ℕ}
    (hi : i < 4) (hj : s.inner = some i) (h : roundFree c s xb n = .ok q) :
    ∃ w t, w.size = 16 ∧ t.size = 64 ∧ xb.data.toList = encs (ownFrames c q.2.1 w t i) ++ (ownSq n i).enc ∧
      ∃ k, ownSq n i = .sq k n ∧ (k = 2 ∨ (k = 1 ∧ n = 1)) ∧ ¬ (n = 0 ∧ k = 2) ∧ n < 2 ^ 64 := by
  have hs : s.inner ≠ some INNER_ROUNDS := by
    rw [hj]
    intro e
    have : i = INNER_ROUNDS := Option.some.inj e
    rw [this] at hi
    exact absurd hi (by decide)
  have hq := roundFree_sq hs h
  have hin := roundFree_inner_some hs hj h
  obtain ⟨-, w, t, hw, ht, hx⟩ := innerOwn_inv (hq.2.2.2.2.2 i hin)
  have hX := ownRound_encs (c := c) (s := q.2.1) (n := n) hw ht hi
  rw [← hx] at hX
  have hn : n < 2 ^ 64 := by rw [← hq.2.2.1]; exact UInt64.toNat_lt _
  obtain ⟨k, m, hkm, hm⟩ := ownSq_sq hn i
  have hX' := hX
  rw [hkm] at hX'
  obtain ⟨rfl, hk, h0, hn'⟩ := sqOk_frames hX' hm hq
  exact ⟨w, t, hw, ht, hX, k, hkm, hk, h0, hn'⟩

theorem ownFrames_mid {c : Cfg} (hrow : c.rowH.size < 2 ^ 64) (s : Masker) (w t : ByteArray) (ht : t.size = 64) :
    ∀ {i : ℕ}, 0 < i → ∀ f ∈ ownFrames c s w t i, midFrame f
  | 0, h, _, _ => (Nat.lt_irrefl 0 h).elim
  | 1, _, f, hf => by
    rw [List.mem_singleton.mp hf]
    show List.length _ < _
    simp
  | 2, _, f, hf => by
    rw [List.mem_singleton.mp hf]
    show t.size < _
    rw [ht]
    norm_num
  | _ + 3, _, f, hf => by
    rw [List.mem_singleton.mp hf]
    show List.length _ < _
    simp only [Array.length_toList, Array.size_map, Array.size_zip]
    exact lt_of_le_of_lt (Nat.min_le_left _ _) hrow

open _root_.Flock.Firewall.Frame (check) in
/-- **Inside the inner proof the placeholder takes its own round where the deployed frame's check released one**: past
the inner proof's label, whether the device's round passed or the scheduled round stood in, the placeholder's own round
at zero pads, from the deployed state but for `stopped`, leaves the deployed state but for `stopped`. -/
theorem check_own_ok {c c' : Cfg} (hc : CfgAlike c c') (hz : ZeroCfg c') (hH : c.rowH.size = c'.rowH.size)
    (hM : c.rowMu.size = c'.rowMu.size) (hrow : c'.rowH.size < 2 ^ 64) (m s₀ : Masker) (st : Bool) {i n : ℕ}
    (hi : i < 4) (hi0 : 0 < i) (hj : m.inner = some i) {v r out : ByteArray} {s' : Masker} {why : Option String}
    (h : check c { m with stopped := st } v r n = .ok (out, s', why)) :
    ∃ q', roundFree c' m (ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n i) n = .ok q' ∧
      unstop s' = unstop q'.2.1 := by
  have hz64 : (Loop.zeros 64).size = 64 := by simp [Loop.zeros, ByteArray.size]
  have key : ∀ (b : Bool) (xb : ByteArray) (fr : Array ℕ),
      roundFree c { m with stopped := b } xb n = .ok (out, s', fr) →
      ∃ q', roundFree c' m (ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n i) n = .ok q' ∧
        unstop s' = unstop q'.2.1 := by
    intro b xb fr hf
    obtain ⟨w, t, -, ht, hX, k, hkm, hk, h0, hn⟩ := roundFree_own_frames (s := { m with stopped := b }) hi hj hf
    dsimp only at hX
    have e : xb = ByteArray.mk (encs (ownFrames c s' w t i) ++ (Frame.sq k n).enc).toArray := by
      rw [← hkm, ← hX, Array.toArray_toList]
    rw [e] at hf
    have hA := ownFrames_alike (w := w) (w' := Loop.zeros 16) hH hM s' s₀ (ht.trans hz64.symm) i
    have hm := midFrame_forall₂ (Sched.forall₂_alike_symm hA) (ownFrames_mid hrow s₀ _ _ hz64 hi0)
    have hs : ({ m with stopped := b } : Masker).inner ≠ some INNER_ROUNDS := by
      show m.inner ≠ _
      rw [hj]
      intro e'
      have : i = INNER_ROUNDS := Option.some.inj e'
      rw [this] at hi
      exact absurd hi (by decide)
    obtain ⟨q', hq', hu⟩ := roundFree_own_ok hc hz { m with stopped := b } s₀ m.stopped hi hs hkm hk h0 hn hm hA
      (fun q hq => by rw [foldlM_own_inner c' s₀ _ _ hi0 _ q hq]; exact hj) hf
    rw [Sched.with_stopped_self m rfl] at hq'
    exact ⟨q', hq', hu⟩
  cases why with
  | none =>
    obtain ⟨fr, hfr⟩ := Sched.roundFree_round (check_none h)
    exact key st v fr hfr
  | some w =>
    obtain ⟨fr, hfr⟩ := Sched.roundFree_round (check_some h)
    exact key true r fr hfr

theorem labelInner_size : _root_.Flock.Zk.LABEL_INNER.toUTF8.size < 2 ^ 64 := by decide

theorem ownFrames0_mid (c : Cfg) (s : Masker) (w t : ByteArray) : ∀ f ∈ ownFrames c s w t 0, midFrame f := by
  intro f hf
  simp only [ownFrames, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl
  · exact labelInner_size
  · trivial

open _root_.Flock.Firewall.Frame (check) in
/-- **At the inner proof's label the placeholder takes its own round where the deployed frame's check released one**,
where the scheduled round in its place is the firewall's own round 0 at some `final_c'` (`r`): whether the device's
round passed or `r` stood in, the placeholder's own round 0 at zero pads, from the deployed state but for `stopped`,
leaves the deployed state but for `stopped`. -/
theorem check_own0_ok {c c' : Cfg} (hc : CfgAlike c c') (hz : ZeroCfg c') (m s₀ : Masker) (st : Bool) {n : ℕ}
    (hj : m.inner = none) {w : ByteArray} (hw : w.size = 16) {v r out : ByteArray} {s' : Masker}
    {why : Option String}
    (hr : r.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList)
    (h : check c { m with stopped := st } v r n = .ok (out, s', why)) :
    ∃ q', roundFree c' m (ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n 0) n = .ok q' ∧
      unstop s' = unstop q'.2.1 := by
  have hz64 : (Loop.zeros 64).size = 64 := by simp [Loop.zeros, ByteArray.size]
  have hR : r = ByteArray.mk (encs (ownFrames c m w (Loop.zeros 64) 0) ++ (Frame.sq 1 1).enc).toArray := by
    have e := ownRound_encs (c := c) (s := m) (n := n) hw hz64 (k := 0) (by norm_num)
    simp only [ownRound, ownSq] at e
    rw [← e, ← hr, Array.toArray_toList]
  have hfs : List.Forall₂ Alike (ownFrames c m w (Loop.zeros 64) 0)
      (ownFrames c' s₀ (Loop.zeros 16) (Loop.zeros 64) 0) :=
    .cons (show Alike (.label _) (.label _) from rfl) (.cons True.intro .nil)
  have hs : ∀ b : Bool, ({ m with stopped := b } : Masker).inner ≠ some INNER_ROUNDS := fun b => by
    show m.inner ≠ _
    rw [hj]
    nofun
  have hn1 : ∀ (b : Bool) (q : ByteArray × Masker × Array ℕ), roundFree c { m with stopped := b } r n = .ok q →
      n = 1 := fun b q hq =>
    (sqOk_frames (xb := r) (by rw [hR]) (by norm_num) (roundFree_sq (hs b) hq)).1.symm
  have key : ∀ (b : Bool) (fs : List Frame) (fr : Array ℕ), (∀ f ∈ fs, midFrame f) →
      List.Forall₂ Alike fs (ownFrames c' s₀ (Loop.zeros 16) (Loop.zeros 64) 0) → n = 1 →
      roundFree c { m with stopped := b } (ByteArray.mk (encs fs ++ (Frame.sq 1 1).enc).toArray) n =
        .ok (out, s', fr) →
      ∃ q', roundFree c' m (ownRound c' s₀ (Loop.zeros 16) (Loop.zeros 64) n 0) n = .ok q' ∧
        unstop s' = unstop q'.2.1 := by
    intro b fs fr hm hA hn hf
    subst hn
    obtain ⟨q', hq', hu⟩ := roundFree_own_ok hc hz { m with stopped := b } s₀ m.stopped (i := 0) (by norm_num)
      (hs b) rfl (.inr ⟨rfl, rfl⟩) (by decide) (by norm_num) hm hA
      (fun q hq => foldlM_own0_inner c' s₀ _ _ _ q hq) hf
    rw [Sched.with_stopped_self m rfl] at hq'
    exact ⟨q', hq', hu⟩
  cases why with
  | none =>
    have hp := check_none_passes h
    obtain ⟨q0, hq0, -⟩ := passes_none hp
    have h1 := hn1 st q0 hq0
    subst h1
    rw [hR] at hp
    obtain ⟨fs', hA', hm', rfl⟩ := passes_frames c { m with stopped := st } _ 1 1 (hs st) (ownFrames0_mid c m w _)
      (.inr ⟨rfl, rfl⟩) (by decide) (by norm_num) v hp
    obtain ⟨fr, hfr⟩ := Sched.roundFree_round (check_none h)
    exact key st fs' fr hm' (Sched.forall₂_alike_trans (Sched.forall₂_alike_symm hA') hfs) rfl hfr
  | some _ =>
    obtain ⟨fr, hfr⟩ := Sched.roundFree_round (check_some h)
    have h1 := hn1 true _ hfr
    subst h1
    rw [hR] at hfr
    exact key true _ fr (ownFrames0_mid c m w _) hfs rfl hfr

end Claims.ZKServer
