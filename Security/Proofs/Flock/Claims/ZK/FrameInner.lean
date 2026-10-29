import Proofs.Flock.Claims.ZK.FrameOwn
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Claims.ZK.FrameProg
import Proofs.Flock.Soundness.Refine.Arith

/-!
# What the frame takes at the inner proof's rounds

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
* `repState_own_inner`, `check_inner_done`, `repState_past`: along the loop's requests the rep's state counts the
  inner proof's rounds (`run_ownReqs_inner`), the frame's check refuses its round 4, and past its last coins the state
  owes none. A round the frame refuses leaves the rep owing nothing, so it ends at the next coins.
* `repState_own_check`, `progOf_own_step`: the loop's frame state one round past the rep's schedule is the deployed
  check's but for `stopped`; so `ofLoop_progChecks`: `ProgChecks` at `ofLoop`, from `ofLoop_progChecks_of`'s premises
  but `hO`, the placeholder's pad row within a `u64`'s count (`hRow`) and `L₀`'s round at the inner proof's round 0
  the firewall's own at some `final_c'` (`hL0`).
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

/-! ## The drive's requests at the inner proof's rounds -/

theorem take_pre_rounds {α : Type} (A B C D : List α) {n : ℕ} (hn : n ≤ C.length) :
    (A ++ B ++ C ++ D).take (A.length + B.length + n) = A ++ B ++ C.take n := by
  rw [show A ++ B ++ C ++ D = (A ++ B) ++ (C ++ D) by simp only [List.append_assoc],
    show A.length + B.length + n = (A ++ B).length + n by rw [List.length_append], List.take_length_add_append,
    List.take_append_of_le_length hn]

/-- **The drive's requests up to the inner proof's round `i`**: the earlier rep's coins, every scheduled round with its
coins, the inner proof's rounds before `i` with theirs, then round `i` without its coins. -/
theorem reqs_take_own (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (dev : ℕ → ByteArray)
    (coins : Array (Array F128)) {i : ℕ} (hi : i < own.size) :
    (_root_.Flock.Firewall.Drive.reqs prior sched own dev coins).take
        ((if prior.isEmpty then 0 else 1) + 2 * (sched.size + i) + 1) =
      (if prior.isEmpty then [] else [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]) ++
      (List.range sched.size).flatMap (fun k =>
        [.round (sched.getD k (.empty, 0)).2 (dev k) (.sched (sched.getD k (.empty, 0)).1),
          .coins (.ok (coins.getD k #[]))]) ++
      (List.range i).flatMap (fun j =>
        [.round (own.getD j (.empty, 0)).2 (own.getD j (.empty, 0)).1 .own,
          .coins (.ok (coins.getD (sched.size + j) #[]))]) ++
      [.round (own.getD i (.empty, 0)).2 (own.getD i (.empty, 0)).1 .own] := by
  unfold _root_.Flock.Firewall.Drive.reqs
  have hl : ∀ k, ([.round (sched.getD k (.empty, 0)).2 (dev k) (.sched (sched.getD k (.empty, 0)).1),
      .coins (.ok (coins.getD k #[]))] : List _root_.Flock.Firewall.Frame.Req).length = 2 := fun _ => rfl
  have hl' : ∀ j, ([.round (own.getD j (.empty, 0)).2 (own.getD j (.empty, 0)).1 .own,
      .coins (.ok (coins.getD (sched.size + j) #[]))] : List _root_.Flock.Firewall.Frame.Req).length = 2 :=
    fun _ => rfl
  have hP : (if prior.isEmpty then ([] : List _root_.Flock.Firewall.Frame.Req) else
      [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]).length =
      if prior.isEmpty then 0 else 1 := by
    split <;> rfl
  rw [← hP, show ∀ p : ℕ, p + 2 * (sched.size + i) + 1 = p + ((List.range sched.size).flatMap fun k =>
      ([.round (sched.getD k (.empty, 0)).2 (dev k) (.sched (sched.getD k (.empty, 0)).1),
        .coins (.ok (coins.getD k #[]))] : List _root_.Flock.Firewall.Frame.Req)).length + (2 * i + 1) from
      fun p => by rw [Loop.flatMap_range_length _ hl]; omega,
    take_pre_rounds _ _ _ _ (by rw [Loop.flatMap_range_length _ hl']; omega), Loop.flatMap_range_take _ hl' hi]
  simp only [List.append_assoc]
  rfl

namespace Loop

/-- **A rep's requests up to the inner proof's round `i`**, the scheduled rounds themselves in the device's place
(`reqs_take_own`). -/
def ownReqs (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128)) (i : ℕ) :
    List _root_.Flock.Firewall.Frame.Req :=
  (if prior.isEmpty then [] else [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]) ++
    (List.range sched.size).flatMap (fun k =>
      [.round (sched.getD k (.empty, 0)).2 (sched.getD k (.empty, 0)).1 (.sched (sched.getD k (.empty, 0)).1),
        .coins (.ok (cr.getD k #[]))]) ++
    (List.range i).flatMap (fun j =>
      [.round (own.getD j (.empty, 0)).2 (own.getD j (.empty, 0)).1 .own,
        .coins (.ok (cr.getD (sched.size + j) #[]))]) ++
    [.round (own.getD i (.empty, 0)).2 (own.getD i (.empty, 0)).1 .own]

theorem ownReqs_succ (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128)) (i : ℕ) :
    ownReqs prior sched own cr (i + 1) = ownReqs prior sched own cr i ++
      [.coins (.ok (cr.getD (sched.size + i) #[])),
        .round (own.getD (i + 1) (.empty, 0)).2 (own.getD (i + 1) (.empty, 0)).1 .own] := by
  unfold ownReqs
  rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
  simp only [List.append_assoc, List.cons_append, List.nil_append]

theorem ownRounds_size (c : Cfg) (ci : Array (Array F128)) (ns : ℕ → ℕ) : (ownRounds c ci ns).size = INNER_ROUNDS := by
  unfold ownRounds
  rw [Array.size_map, Array.size_range]

/-- **A rep's frame state after the inner proof's round `i`**: `Frame.run` over its requests up to it (`ownReqs`), where
the schedule on the rep's coins before the last is `sched`. -/
theorem repState_own_at (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (i : ℕ)
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hi4 : i < INNER_ROUNDS) :
    repState c t rep dom xs hi cr0 cr ns (sched.size + i + 1) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        (ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
          (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr i)
      if r.2 then {} else r.1.m := by
  unfold repState
  rw [hal]
  simp only [bind, Except.bind, hs, pure, Except.pure]
  rw [show ∀ p : ℕ, p + 2 * (sched.size + i + 1) - 1 = p + 2 * (sched.size + i) + 1 from fun p => by omega,
    reqs_take_own _ _ _ _ _ (by rw [ownRounds_size]; exact hi4)]
  rfl

/-- At zero pads the inner proof's rounds read none of their coins. -/
theorem ownRounds_zero {c : Cfg} (hz : ZeroCfg c) (ci ci' : Array (Array F128)) (ns : ℕ → ℕ) :
    ownRounds c ci ns = ownRounds c ci' ns := by
  unfold ownRounds
  refine Array.ext (by simp) fun j h1 _ => ?_
  have hj : j < INNER_ROUNDS := by simpa using h1
  simp only [Array.getElem_map, Array.getElem_range]
  rw [ownRound_zero hz { eps := (ci.getD 0 #[]).getD 0 0, beta := (ci.getD 2 #[]).getD 0 0 }
    { eps := (ci'.getD 0 #[]).getD 0 0, beta := (ci'.getD 2 #[]).getD 0 0 } _ _ _ hj]

theorem ownRounds_getD (c : Cfg) (ci : Array (Array F128)) (ns : ℕ → ℕ) {j : ℕ} (hj : j < INNER_ROUNDS) :
    (ownRounds c ci ns).getD j (.empty, 0) =
      (ownRound c { eps := (ci.getD 0 #[]).getD 0 0, beta := (ci.getD 2 #[]).getD 0 0 } (zeros 16) (zeros 64) (ns j) j,
        ns j) := by
  unfold ownRounds
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, hj, ↓reduceIte, Option.map_some,
    Option.getD_some]

/-- **The frame's coins, then the inner proof's round**: where the frame takes the coins `cs` and the round `r` there,
its state after both is the round's. -/
theorem run_coins_own (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (cs : Array F128) (n : ℕ) (r : ByteArray)
    {s1 s' : Masker} {out : ByteArray} {fr : Array ℕ} (ha : st.m.answer c cs = .ok s1)
    (hr : roundFree c s1 r n = .ok (out, s', fr)) (hin : s'.inner.isSome) :
    _root_.Flock.Firewall.Frame.run c st [.coins (.ok cs), .round n r .own] = ({ m := s', begun := true }, false) := by
  have ho := answer_owed ha
  have hr' := round_of_roundFree hr
  have hn : s'.inner.isNone = false := by cases h : s'.inner <;> simp_all
  simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha, ho, hr', hn, bind, Except.bind,
    pure, Except.pure, Bool.false_eq_true, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends]

/-- Coins inside the inner proof count one more of its rounds. -/
theorem answer_inner_some {c : Cfg} {s s' : Masker} {cs : Array F128} {k : ℕ} (hs : s.inner = some k)
    (h : s.answer c cs = .ok s') : s'.inner = some (k + 1) := by
  unfold Masker.answer at h
  simp only [bind, Except.bind, pure, Except.pure, hs] at h
  repeat' split at h
  all_goals (cases h <;> rfl)

theorem ownReqs_push (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128))
    (a : Array F128) {i : ℕ} (hi : sched.size + i ≤ cr.size) :
    ownReqs prior sched own (cr.push a) i = ownReqs prior sched own cr i := by
  have hg : ∀ k, k < cr.size → (cr.push a).getD k #[] = cr.getD k #[] := fun k hk => by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, show k ≠ cr.size by omega, ↓reduceIte]
  unfold ownReqs
  rw [flatMap_range_congr (i := sched.size) fun k hk => by rw [hg k (by omega)],
    flatMap_range_congr (i := i) fun j hj => by rw [hg (sched.size + j) (by omega)]]

/-- **The loop's frame state one round on, inside the inner proof**: where the rep's schedule is complete on its coins
`cr` and on those before the last, `cr` covers the inner proof's round `i`, the frame takes that round's coins at the
rep's state after it, at the state `m`, and takes the inner proof's round `i + 1` there, the rep's state after round
`i + 1` on one more round's coins is that round's. -/
theorem repState_own_step {c : Cfg} (hz : ZeroCfg c) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ)
    (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ)
    (a : Array F128) (i : ℕ) {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched)
    (hcr : cr.size = sched.size + i + 1) (hi4 : i + 1 < INNER_ROUNDS) (s₀ : Masker) {m s' : Masker}
    {out : ByteArray} {fr : Array ℕ}
    (ha : (repState c t rep dom xs hi cr0 cr ns (sched.size + i + 1)).answer c (cr.getD (sched.size + i) #[]) = .ok m)
    (hr : roundFree c m (ownRound c s₀ (zeros 16) (zeros 64) (ns (sched.size + (i + 1))) (i + 1))
      (ns (sched.size + (i + 1))) = .ok (out, s', fr)) (hin : s'.inner.isSome) :
    repState c t rep dom xs hi cr0 (cr.push a) ns (sched.size + i + 2) = s' := by
  rw [show sched.size + i + 2 = sched.size + (i + 1) + 1 by omega,
    repState_own_at c t rep dom xs hi cr0 (cr.push a) ns (i + 1) hal (by rw [Array.pop_push]; exact hs') hi4,
    ownReqs_succ, ownReqs_push _ _ _ _ _ (by omega),
    ownRounds_zero hz ((cr.push a).extract sched.size (cr.push a).size) (cr.extract sched.size cr.size),
    ownRounds_getD _ _ _ hi4]
  rw [repState_own_at c t rep dom xs hi cr0 cr ns i hal hs (by omega)] at ha
  have hg : (cr.push a).getD (sched.size + i) #[] = cr.getD (sched.size + i) #[] := by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, show sched.size + i ≠ cr.size by omega, ↓reduceIte]
  rw [hg, ownRound_zero hz _ s₀ _ _ _ hi4]
  dsimp only at ha ⊢
  rw [run_append]
  generalize _root_.Flock.Firewall.Frame.run c {}
    (ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr i) = R at ha ⊢
  obtain ⟨S, b⟩ := R
  cases b with
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha ⊢
    rw [run_coins_own c S _ _ _ ha hr hin]
    rfl
  | true =>
    have e : (({} : Masker).answer c (cr.getD (sched.size + i) #[])) = .error "coins for no round" := rfl
    simp only [↓reduceIte] at ha
    rw [e] at ha
    cases ha

/-! ## The inner proof's count along the loop -/

/-- A round the frame takes comes before the inner proof's last. -/
theorem roundFree_ok_inner {c : Cfg} {s : Masker} {xb : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s xb n = .ok q) : s.inner ≠ some INNER_ROUNDS := by
  intro e
  have hb : (s.inner == some INNER_ROUNDS) = true := by rw [e]; exact beq_self_eq_true _
  unfold roundFree at h
  simp only [hb, ↓reduceIte] at h
  cases h

theorem roundFree_done {c : Cfg} {s : Masker} (hs : s.inner = some INNER_ROUNDS) (xb : ByteArray) (n : ℕ) :
    ∃ e, roundFree c s xb n = .error e := by
  have hb : (s.inner == some INNER_ROUNDS) = true := by rw [hs]; exact beq_self_eq_true _
  unfold roundFree
  simp only [hb, ↓reduceIte]
  exact ⟨_, rfl⟩

open _root_.Flock.Firewall.Frame (check standIn passes) in
/-- **The frame's check refuses every round past the inner proof's last**, whatever the device's and the scheduled
round. -/
theorem check_inner_done {c : Cfg} {s : Masker} (hs : s.inner = some INNER_ROUNDS) (v r : ByteArray) (n : ℕ)
    (q : ByteArray × Masker × Option String) : check c s v r n ≠ .ok q := by
  obtain ⟨e, he⟩ := roundFree_done (c := c) hs r n
  obtain ⟨e', he'⟩ := roundFree_done (c := c) (s := { s with stopped := true }) hs r n
  intro h
  unfold check passes standIn _root_.Flock.Firewall.Frame.round at h
  cases hst : s.stopped <;> simp only [hst, he, he', Bool.false_eq_true, ↓reduceIte, bind, Except.bind] at h <;>
    cases h

/-- The frame's steps over the firewall's own round 0, where it takes them, start the inner proof's count. -/
theorem roundFree_own0_inner {c : Cfg} {m s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64) {n : ℕ}
    {q : ByteArray × Masker × Array ℕ} (h : roundFree c m (ownRound c s w t n 0) n = .ok q) : q.2.1.inner = some 0 := by
  have hs := roundFree_ok_inner h
  have hR : ownRound c s w t n 0 = ByteArray.mk (encs (ownFrames c s w t 0) ++ (Frame.sq 1 1).enc).toArray := by
    have e := ownRound_encs (c := c) (s := s) (n := n) hw ht (k := 0) (by norm_num)
    simp only [ownSq] at e
    rw [← e, Array.toArray_toList]
  have hn : n = 1 := (sqOk_frames (xb := ownRound c s w t n 0) (by rw [hR]) (by norm_num) (roundFree_sq hs h)).1.symm
  subst hn
  rw [hR, roundFree_encs c m _ 1 1 hs (ownFrames0_mid c s w t) (.inr ⟨rfl, rfl⟩) (by decide) (by norm_num)] at h
  cases hf : (ownFrames c s w t 0).foldlM (stepAt c) (m, ByteArray.empty, #[], 0) with
  | error e => rw [hf] at h; cases h
  | ok r =>
    have hr := foldlM_own0_inner c s w t _ r hf
    rw [hf] at h
    simp only [bind, Except.bind, hr] at h
    cases hio : innerOwn c r.1 0 (ByteArray.mk (encs (ownFrames c s w t 0) ++ (Frame.sq 1 1).enc).toArray) 1 with
    | error e => rw [hio] at h; cases h
    | ok u =>
      rw [hio] at h
      cases h
      rfl

/-- **The frame's step at the inner proof's round**, where it neither ends the rep nor leaves it owing no coins: the
frame took the round, which kept the rep inside the inner proof, and the state is the round's. -/
theorem run_own1 (c : Cfg) (S : _root_.Flock.Firewall.Frame.State) (n : ℕ) (r : ByteArray)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.round n r .own]).2 = false)
    (ho : (_root_.Flock.Firewall.Frame.run c S [.round n r .own]).1.m.owed ≠ none) :
    ∃ out s' fr, roundFree c S.m r n = .ok (out, s', fr) ∧ s'.inner.isSome = true ∧
      (_root_.Flock.Firewall.Frame.run c S [.round n r .own]).1.m = s' := by
  cases hS : S.m.owed with
  | some k =>
    simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS,
      _root_.Flock.Firewall.Frame.Reply.ends] at h2
    cases h2
  | none =>
    cases hr : roundFree c S.m r n with
    | error e =>
      have hR : _root_.Flock.Firewall.Frame.round c S.m r n = .error e := by
        unfold _root_.Flock.Firewall.Frame.round; rw [hr]; rfl
      by_cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP = true
      · simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS, hR, bind, Except.bind,
          hst, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends] at h2
        cases h2
      · simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS, hR, bind, Except.bind,
          hst, Bool.false_eq_true, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends] at ho
        exact absurd rfl ho
    | ok p =>
      obtain ⟨out, s', fr⟩ := p
      have hR := round_of_roundFree hr
      cases hin : s'.inner.isNone with
      | true =>
        by_cases hst : ("a round with no scheduled round in its place, outside the inner proof").startsWith
            _root_.Flock.Firewall.Frame.STOP = true
        · simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS, hR, hin, bind,
            Except.bind, throw, throwThe, MonadExceptOf.throw, hst, ↓reduceIte,
            _root_.Flock.Firewall.Frame.Reply.ends] at h2
          cases h2
        · simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS, hR, hin, bind,
            Except.bind, throw, throwThe, MonadExceptOf.throw, hst, Bool.false_eq_true,
            ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends] at ho
          exact absurd rfl ho
      | false =>
        refine ⟨out, s', fr, rfl, by cases e : s'.inner <;> simp_all, ?_⟩
        simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hS, hR, hin, bind,
          Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends]

/-- Coins the frame takes without ending the rep are its answer to them. -/
theorem run_coins_cons (c : Cfg) (S : _root_.Flock.Firewall.Frame.State) (cs : Array F128)
    (L : List _root_.Flock.Firewall.Frame.Req) (h : (_root_.Flock.Firewall.Frame.run c S (.coins (.ok cs) :: L)).2 = false) :
    ∃ s1, S.m.answer c cs = .ok s1 ∧
      _root_.Flock.Firewall.Frame.run c S (.coins (.ok cs) :: L) = _root_.Flock.Firewall.Frame.run c { S with m := s1 } L := by
  cases ha : S.m.answer c cs with
  | error e =>
    exfalso
    have h' : (_root_.Flock.Firewall.Frame.run c S (.coins (.ok cs) :: L)).2 = true := by
      cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
        simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha, hst, Bool.false_eq_true,
          ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends]
    rw [h'] at h
    cases h
  | ok s1 =>
    exact ⟨s1, rfl, by
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha,
        _root_.Flock.Firewall.Frame.Reply.ends, Bool.false_eq_true, ↓reduceIte]⟩

/-- `run_own1` past any requests `L`. -/
theorem run_snoc_own (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (L : List _root_.Flock.Firewall.Frame.Req)
    (n : ℕ) (r : ByteArray) (h2 : (_root_.Flock.Firewall.Frame.run c st (L ++ [.round n r .own])).2 = false)
    (ho : (_root_.Flock.Firewall.Frame.run c st (L ++ [.round n r .own])).1.m.owed ≠ none) :
    ∃ out s' fr, roundFree c (_root_.Flock.Firewall.Frame.run c st L).1.m r n = .ok (out, s', fr) ∧
      s'.inner.isSome = true ∧ (_root_.Flock.Firewall.Frame.run c st (L ++ [.round n r .own])).1.m = s' := by
  rw [run_append] at h2 ho ⊢
  cases hb : (_root_.Flock.Firewall.Frame.run c st L).2
  · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ho ⊢
    exact run_own1 c _ n r h2 ho
  · simp only [hb, ↓reduceIte, Bool.true_eq_false] at h2

/-- The frame's coins, then the inner proof's round, past any requests `L`, where neither ends the rep nor leaves it
owing no coins: the frame took the coins at `L`'s state, then the round. -/
theorem run_snoc_coins_own (c : Cfg) (st : _root_.Flock.Firewall.Frame.State)
    (L : List _root_.Flock.Firewall.Frame.Req) (cs : Array F128) (n : ℕ) (r : ByteArray)
    (h2 : (_root_.Flock.Firewall.Frame.run c st (L ++ [.coins (.ok cs), .round n r .own])).2 = false)
    (ho : (_root_.Flock.Firewall.Frame.run c st (L ++ [.coins (.ok cs), .round n r .own])).1.m.owed ≠ none) :
    (_root_.Flock.Firewall.Frame.run c st L).2 = false ∧
      ∃ s1, (_root_.Flock.Firewall.Frame.run c st L).1.m.answer c cs = .ok s1 ∧
        ∃ out s' fr, roundFree c s1 r n = .ok (out, s', fr) ∧ s'.inner.isSome = true ∧
          (_root_.Flock.Firewall.Frame.run c st (L ++ [.coins (.ok cs), .round n r .own])).1.m = s' := by
  rw [run_append] at h2 ho ⊢
  cases hb : (_root_.Flock.Firewall.Frame.run c st L).2
  · refine ⟨rfl, ?_⟩
    simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ho ⊢
    obtain ⟨s1, ha, he⟩ := run_coins_cons c _ cs _ h2
    rw [he] at h2 ho ⊢
    exact ⟨s1, ha, run_own1 c _ n r h2 ho⟩
  · simp only [hb, ↓reduceIte, Bool.true_eq_false] at h2

/-- After coins, then `finish`, the frame's state owes no coins. -/
theorem run_coins_finish_owed (c : Cfg) (S : _root_.Flock.Firewall.Frame.State) (cs : Array F128) :
    (if (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .finish]).2 then ({} : Masker)
      else (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .finish]).1.m).owed = none := by
  cases ha : S.m.answer c cs with
  | error e =>
    have h : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .finish]).2 = true := by
      cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
        simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha, hst, Bool.false_eq_true,
          ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends]
    rw [h]
    rfl
  | ok s1 =>
    have ho := answer_owed ha
    cases hu : s1.unfinished c <;>
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha, hu, ho,
        _root_.Flock.Firewall.Frame.Reply.ends, Bool.false_eq_true, ↓reduceIte]

/-- **The inner proof's count along the loop's requests**: where the requests up to the inner proof's round `i` neither
end the rep nor leave it owing no coins, the rep is at the inner proof's round `i`. Round 0 is the firewall's own. -/
theorem run_ownReqs_inner (c : Cfg) (prior : Array F128) (sched own : Array (ByteArray × ℕ))
    (cr : Array (Array F128)) (h0 : ∃ s w t n, w.size = 16 ∧ t.size = 64 ∧ own.getD 0 (.empty, 0) = (ownRound c s w t n 0, n)) :
    ∀ i, (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).1.m.owed ≠ none →
      (_root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr i)).1.m.inner = some i
  | 0, h2, ho => by
    obtain ⟨s, w, t, n, hw, ht, he⟩ := h0
    have e : ownReqs prior sched own cr 0 =
        ((if prior.isEmpty then [] else [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]) ++
          (List.range sched.size).flatMap (fun k =>
            [.round (sched.getD k (.empty, 0)).2 (sched.getD k (.empty, 0)).1 (.sched (sched.getD k (.empty, 0)).1),
              .coins (.ok (cr.getD k #[]))])) ++ [.round n (ownRound c s w t n 0) .own] := by
      unfold ownReqs
      rw [he]
      simp only [List.range_zero, List.flatMap_nil, List.append_nil]
    rw [e] at h2 ho ⊢
    obtain ⟨out, s', fr, hr, -, hm⟩ := run_snoc_own c {} _ n _ h2 ho
    rw [hm]
    exact roundFree_own0_inner hw ht hr
  | i + 1, h2, ho => by
    rw [ownReqs_succ] at h2 ho ⊢
    obtain ⟨hb, s1, ha, out, s', fr, hr, -, hm⟩ := run_snoc_coins_own c {} _ _ _ _ h2 ho
    have hI := run_ownReqs_inner c prior sched own cr h0 i hb (fun e => answer_owed_none _ e ha)
    rw [hm]
    exact roundFree_inner_some (roundFree_ok_inner hr) (answer_inner_some hI ha) hr

theorem zeros_size (n : ℕ) : (zeros n).size = n := by simp [zeros, ByteArray.size]

/-- **The inner proof's count at the loop's state**: where the frame takes coins at the rep's state after the inner
proof's round `i`, it is at round `i + 1`. -/
theorem repState_own_inner (c c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) {i : ℕ}
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hi4 : i < INNER_ROUNDS) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (sched.size + i + 1)).answer c' cs = .ok m) : m.inner = some (i + 1) := by
  rw [repState_own_at c t rep dom xs hi cr0 cr ns i hal hs hi4] at ha
  dsimp only at ha
  generalize hR : _root_.Flock.Firewall.Frame.run c {}
    (ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr i) = R at ha
  obtain ⟨S, b⟩ := R
  cases b with
  | true =>
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    have hI := run_ownReqs_inner c _ sched _ cr ⟨_, zeros 16, zeros 64, _, zeros_size 16, zeros_size 64,
      ownRounds_getD c _ _ (show 0 < INNER_ROUNDS by decide)⟩ i (by rw [hR])
      (by rw [hR]; exact fun e => answer_owed_none cs e ha)
    rw [hR] at hI
    exact answer_inner_some hI ha

theorem ownReqs_zero (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {j : ℕ}
    (hS : sched.size = j + 1) :
    ownReqs prior sched own cr 0 = schedReqs prior sched cr j ++
      [.coins (.ok (cr.getD j #[])), .round (own.getD 0 (.empty, 0)).2 (own.getD 0 (.empty, 0)).1 .own] := by
  unfold ownReqs schedReqs
  rw [hS, List.range_succ, List.flatMap_append, List.flatMap_singleton]
  simp only [List.range_zero, List.flatMap_nil, List.append_nil, List.append_assoc, List.cons_append, List.nil_append]

/-- **The loop's frame state one round on, into the inner proof**: where the rep's schedule is complete on its coins
`cr` and on those before the last, its last round `j` is the last `cr` covers, the frame takes that round's coins at the
rep's state after it, at the state `m`, and takes the inner proof's round 0 there, the rep's state after it on one more
round's coins is that round's. -/
theorem repState_own0_step {c : Cfg} (hz : ZeroCfg c) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ)
    (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ)
    (a : Array F128) (j : ℕ) {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched)
    (hcr : cr.size = j + 1) (hS : sched.size = j + 1) (s₀ : Masker) {m s' : Masker} {out : ByteArray}
    {fr : Array ℕ} (ha : (repState c t rep dom xs hi cr0 cr ns (j + 1)).answer c (cr.getD j #[]) = .ok m)
    (hr : roundFree c m (ownRound c s₀ (zeros 16) (zeros 64) (ns (j + 1)) 0) (ns (j + 1)) = .ok (out, s', fr))
    (hin : s'.inner.isSome) :
    repState c t rep dom xs hi cr0 (cr.push a) ns (j + 2) = s' := by
  rw [show j + 2 = sched.size + 0 + 1 by omega,
    repState_own_at c t rep dom xs hi cr0 (cr.push a) ns 0 hal (by rw [Array.pop_push]; exact hs') (by decide),
    ownReqs_push _ _ _ _ _ (by omega), ownReqs_zero _ _ _ _ hS,
    ownRounds_zero hz ((cr.push a).extract sched.size (cr.push a).size) (cr.extract sched.size cr.size),
    ownRounds_getD _ _ _ (show 0 < INNER_ROUNDS by decide)]
  rw [repState_at c t rep dom xs hi cr0 cr ns j hal hs (by omega)] at ha
  rw [ownRound_zero hz _ s₀ _ _ _ (show 0 < 4 by decide)]
  dsimp only at ha ⊢
  rw [show sched.size + 0 = j + 1 by omega, run_append]
  generalize _root_.Flock.Firewall.Frame.run c {}
    (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j) = R at ha ⊢
  obtain ⟨S, b⟩ := R
  cases b with
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha ⊢
    rw [run_coins_own c S _ _ _ ha hr hin]
    rfl
  | true =>
    have e : (({} : Masker).answer c (cr.getD j #[])) = .error "coins for no round" := rfl
    simp only [↓reduceIte] at ha
    rw [e] at ha
    cases ha

theorem reqs_length (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (dev : ℕ → ByteArray)
    (coins : Array (Array F128)) :
    (_root_.Flock.Firewall.Drive.reqs prior sched own dev coins).length =
      (if prior.isEmpty then 0 else 1) + 2 * sched.size + 2 * own.size + 1 := by
  unfold _root_.Flock.Firewall.Drive.reqs
  simp only [List.length_append, List.length_singleton]
  rw [flatMap_range_length _ fun _ => rfl, flatMap_range_length _ fun _ => rfl]
  split <;> simp

/-- **Past the inner proof's last round's coins the loop's state owes no coins**: there the rep has taken every
round's coins, then `finish`. -/
theorem repState_past (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) {N : ℕ}
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hN : sched.size + INNER_ROUNDS + 1 ≤ N) :
    (repState c t rep dom xs hi cr0 cr ns N).owed = none := by
  unfold repState
  rw [hal]
  simp only [bind, Except.bind, hs, pure, Except.pure]
  generalize (if rep = 0 then (#[] : Array F128) else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) = prior
  generalize hown : ownRounds c (cr.extract sched.size cr.size) (fun j => ns (sched.size + j)) = own
  have ho4 : own.size = 4 := by rw [← hown, ownRounds_size]; rfl
  have hN5 : sched.size + 5 ≤ N := by
    have : INNER_ROUNDS = 4 := rfl
    omega
  rw [List.take_of_length_le (by rw [reqs_length, ho4]; split <;> omega)]
  have e : _root_.Flock.Firewall.Drive.reqs prior sched own (fun k => (sched.getD k (.empty, 0)).1) cr =
      ownReqs prior sched own cr 3 ++ [.coins (.ok (cr.getD (sched.size + 3) #[])), .finish] := by
    unfold _root_.Flock.Firewall.Drive.reqs ownReqs
    simp only [ho4, show List.range 4 = List.range 3 ++ [3] from rfl, List.flatMap_append, List.flatMap_singleton,
      List.append_assoc, List.cons_append, List.nil_append]
  rw [e, run_append]
  generalize _root_.Flock.Firewall.Frame.run c {} (ownReqs prior sched own cr 3) = R
  obtain ⟨S, b⟩ := R
  cases b with
  | true => rfl
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte]
    exact run_coins_finish_owed c S _

/-- Where the schedule on the coins `cr` is no longer than they are, it is the schedule on all of them but the last. -/
theorem schedSoFar_pop_eq {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {j : ℕ}
    {sched sched' : Array (ByteArray × ℕ)} (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched)
    (hs' : schedSoFar t rep dom xs hi cr = .ok sched') (hcr : cr.size = j + 1) (hle : sched'.size ≤ j + 1) :
    sched' = sched := by
  rw [Array.eq_push_pop_back!_of_size_ne_zero (show cr.size ≠ 0 by omega)] at hs'
  by_cases hd : sched.size ≤ cr.pop.size
  · exact schedSoFar_snoc_done t rep dom xs hi cr.pop cr.back! hs hd hs'
  · obtain ⟨u, rfl⟩ := schedSoFar_snoc t rep dom xs hi cr.pop cr.back! hs hs'
    rw [Array.size_pop, hcr] at hd
    rw [Array.size_append] at hle
    rw [Array.eq_empty_of_size_eq_zero (show u.size = 0 by omega), Array.append_empty]

theorem ofDraws_rowH_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) :
    (Cfg.ofDraws l (Draws.ofBytes l b) m).rowH.size = l.len + _root_.Flock.Zk.K_PAD := by
  show ((Draws.ofBytes l b).h ++ (Draws.ofBytes l b).padH).size = _
  rw [Array.size_append, ofBytes_h_size]
  show l.len + (words b (32 * l.len) _root_.Flock.Zk.K_PAD).size = _
  rw [words_size]

theorem ofDraws_rowMu_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) :
    (Cfg.ofDraws l (Draws.ofBytes l b) m).rowMu.size = l.len + _root_.Flock.Zk.K_PAD := by
  show (words b (16 * l.len) l.len ++ words b (32 * l.len + 16 * _root_.Flock.Zk.K_PAD) _root_.Flock.Zk.K_PAD).size = _
  rw [Array.size_append, words_size, words_size]

/-- Where the located rep's schedule so far ends before its next round, the loop has no round of its own in the
device's place. -/
theorem schedOf_none (real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (x : Msg)
    (cs : List FlockDeploy.Coin) {s0 : _} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (hreal : real (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (t, hi)) {sched : Array (ByteArray × ℕ)}
    (hs : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
      (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
        ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (wordsAt x cs 0))) hi
      ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
        fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched)
    (hle : sched.size ≤ (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2) :
    schedOf real x cs = none := by
  unfold schedOf
  rw [hx, hq]
  simp only [hreal, hs, Array.getElem?_eq_none hle, Option.map_none]

open _root_.Flock.Firewall.Frame (check) in
/-- **The loop's frame state one round on, past the rep's schedule, is the deployed check's** but for `stopped`: where
the rep's schedule is complete on its coins `cr` and on those before the last, no longer than they are, and its rounds
framed, the frame takes the last coins at the state `m`, and the deployed frame's check at an alike configuration of a
device round against `r` releases a round, the rep's state on one more round's coins is that check's, but for
`stopped`. At the inner proof's round 0 (the schedule as long as the coins), `r` is the firewall's own round 0 at some
`final_c'` (`hr0`); at its rounds 1 to 3 `r` is any; at its round 4 the check refuses (`check_inner_done`), and past it
the state owes no coins (`repState_past`). -/
theorem repState_own_check {c c' : Cfg} (hc : CfgAlike c c') (hz : ZeroCfg c') (hH : c.rowH.size = c'.rowH.size)
    (hM : c.rowMu.size = c'.rowMu.size) (hrow : c'.rowH.size < 2 ^ 64) (t : _root_.Flock.Firewall.Finish.Tab)
    (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128))
    (ns : ℕ → ℕ) (a : Array F128) (j : ℕ) {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched)
    (hcr : cr.size = j + 1) (hle : sched.size ≤ j + 1)
    (hF : ∀ k, k < sched.size → Sched.Framed (sched.getD k (.empty, 0))) {m : Masker}
    (ha : (repState c' t rep dom xs hi cr0 cr ns (j + 1)).answer c' (cr.getD j #[]) = .ok m)
    {st : Bool} {v r out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v r (ns (j + 1)) = .ok (out, s, why))
    (hr0 : sched.size = j + 1 → ∃ w : ByteArray, w.size = 16 ∧ r.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList) :
    repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2) =
      { s with stopped := (repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2)).stopped } := by
  by_cases hS : sched.size = j + 1
  · have hnone : m.inner = none := Sched.answer_inner (repState_inner c' t rep dom xs hi cr0 cr ns j hal hs
      (by omega) fun k hk => hF k (by omega)) ha
    obtain ⟨w, hw, hrw⟩ := hr0 hS
    obtain ⟨⟨out', s'', fr⟩, hq', hu⟩ := check_own0_ok hc hz m {} st hnone hw hrw h
    have hin : s''.inner.isSome := by rw [roundFree_own0_inner (zeros_size 16) (zeros_size 64) hq']; rfl
    rw [repState_own0_step hz t rep dom xs hi cr0 cr ns a j hal hs hs' hcr hS {} ha hq' hin]
    exact eq_unstop hu
  · obtain ⟨i, rfl⟩ : ∃ i, j = sched.size + i := ⟨j - sched.size, by omega⟩
    rcases Nat.lt_or_ge i 3 with h3 | h3
    · have hI := repState_own_inner c' c' t rep dom xs hi cr0 cr ns hal hs (i := i) (by show i < 4; omega) ha
      obtain ⟨⟨out', s'', fr⟩, hq', hu⟩ := check_own_ok hc hz hH hM hrow m {} st (i := i + 1) (by omega) (by omega)
        hI h
      have hin : s''.inner.isSome := by rw [roundFree_inner_some (roundFree_ok_inner hq') hI hq']; rfl
      have hq'' : roundFree c' m (ownRound c' {} (zeros 16) (zeros 64) (ns (sched.size + (i + 1))) (i + 1))
          (ns (sched.size + (i + 1))) = .ok (out', s'', fr) := by
        rw [← Nat.add_assoc]
        exact hq'
      rw [repState_own_step hz t rep dom xs hi cr0 cr ns a i hal hs hs' hcr (by show i + 1 < 4; omega) {} ha hq''
        hin]
      exact eq_unstop hu
    · rcases Nat.eq_or_lt_of_le h3 with h3 | h3
      · subst h3
        have hI : m.inner = some (3 + 1) :=
          repState_own_inner c' c' t rep dom xs hi cr0 cr ns hal hs (i := 3) (by decide) ha
        exact absurd h (check_inner_done (s := { m with stopped := st }) hI v r _ _)
      · exact absurd ha (answer_owed_none _ (repState_past c' t rep dom xs hi cr0 cr ns hal hs
          (by have : INNER_ROUNDS = 4 := rfl; omega)))

open _root_.Flock.Firewall.Frame (check) in
/-- **The loop's frame state one round on, past the rep's schedule, within a rep stream, is the deployed check's** but
for `stopped`: as `progOf_check_step`, where the placeholder's schedule so far is no longer than its coins
(`repState_own_check`). -/
theorem progOf_own_step (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') (hz : ZeroCfg c') (hH : c.rowH.size = c'.rowH.size) (hM : c.rowMu.size = c'.rowMu.size)
    (hrow : c'.rowH.size < 2 ^ 64) {j : ℕ} (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    {sched' : Array (ByteArray × ℕ)}
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched')
    (hle : sched'.size ≤ j + 1) (hF : ∀ k, k < sched'.size → Sched.Framed (sched'.getD k (.empty, 0)))
    {m : Masker} (ha : (progOf x cs).answer c (wordsAt x cs q) = .ok m)
    {st : Bool} {v r out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v r (loopNr x (q + 1)) = .ok (out, s, why))
    (hr0 : sched'.size = j + 1 → ∃ w : ByteArray, w.size = 16 ∧ r.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w ++ header 4 1 1).data.toList) :
    progOf x (cs ++ [d]) = { s with stopped := (progOf x (cs ++ [d])).stopped } := by
  have hle' := locate_snd_le (roundsOf spec sts) 0 q
  rw [progOf_at x cs hx hq hj hph, hj] at ha
  rw [progOf_snoc x cs d hx hq, hph]
  dsimp only
  rw [hj]
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hg : ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).getD j #[] = wordsAt x cs q := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simp), Option.getD_some, Array.getElem_map,
      Array.getElem_range]
    congr 1
    omega
  rw [answer_congr _ _ hc.lay hc.order, ← hg] at ha
  have hcr : ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).size = j + 1 := by simp
  obtain rfl := schedSoFar_pop_eq hs hs' hcr hle
  have hn : loopNr x (q + 1) = (fun i => loopNr x (q - (j + 1) + i + 1)) (j + 1) := by
    show _ = loopNr x (q - (j + 1) + (j + 1) + 1)
    congr 1
    omega
  rw [hn] at h
  exact repState_own_check hc hz hH hM hrow t _ dom xs hi _ _ _ _ j hal hs hs' hcr hle hF ha h hr0

end Loop

namespace LeanRelease

open _root_.Flock (Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (check)

open Loop in
/-- **The deployed configuration at the located rep and its placeholder's**: alike, of one row size each, the
placeholder's at zero pads. -/
theorem ofLoop_cfg_place (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h.length = q + 2)
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1)) {link : Array F128} {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts k link = .ok (c', t, dom, xs, hi)) :
    CfgAlike ((ofLoop T L₀ real).cfg x ρ h) c' ∧ ((ofLoop T L₀ real).cfg x ρ h).rowH.size = c'.rowH.size ∧
      ((ofLoop T L₀ real).cfg x ρ h).rowMu.size = c'.rowMu.size ∧ ZeroCfg c' := by
  obtain ⟨lay, mask, hst, hc', -, -⟩ := placeholder_ok hph
  have hrk : (ofLoop T L₀ real).rep x h.length = k := by
    show repOf x h.length = k
    unfold repOf
    rw [hx, hq]
    simp only [Nat.add_sub_cancel]
    rw [hki]
  have hso : startOf x k = .ok (lay, mask) := by
    unfold startOf
    rw [hx]
    exact hst
  have e1 : (ofLoop T L₀ real).lay x k = lay := by
    show ((startOf x k).toOption.map (·.1)).getD default = lay
    rw [hso]
    rfl
  have e2 : (ofLoop T L₀ real).slot x k = some mask := by
    show (startOf x k).toOption.map (·.2) = some mask
    rw [hso]
    rfl
  subst hc'
  refine ⟨?_, ?_, ?_, zeroCfg_ofDraws lay _ _⟩
  · unfold cfg drawsAt
    rw [hrk, e1, e2]
    exact cfgAlike_ofBytes lay _ _ (some mask)
  · unfold cfg drawsAt
    rw [hrk, e1, ofDraws_rowH_size, ofDraws_rowH_size]
  · unfold cfg drawsAt
    rw [hrk, e1, ofDraws_rowMu_size, ofDraws_rowMu_size]

open Loop in
/-- **`ProgChecks` past the rep's schedule, at the loop's own frames**: as `ofLoop_progChecks_step`, where the rep's
schedule so far at its own table and salts (`rsched`) ends before the next round, so the loop has no round of its own
in the device's place (`schedOf_none`) and `L₀`'s stands there. At the inner proof's round 0, the first past the
schedule, `L₀`'s round is the firewall's own round 0 at some `final_c'` (`hL0`). -/
theorem ofLoop_progChecks_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray}
    (hA : (ofLoop T L₀ real).answered x ρ h = .ok m)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ h m false v = .ok (out, s', why))
    {cs : List FlockDeploy.Coin} (hcs : h.tail.map (fun p => ZKMal.CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h.length = q + 2) {k j : ℕ}
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} (hph : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ h) c') (hz : ZeroCfg c')
    (hH : ((ofLoop T L₀ real).cfg x ρ h).rowH.size = c'.rowH.size)
    (hM : ((ofLoop T L₀ real).cfg x ρ h).rowMu.size = c'.rowMu.size) (hrow : c'.rowH.size < 2 ^ 64)
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    {rsched sched' : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x cs 0))) hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok rsched)
    (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t (k % 2) dom xs hi ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) =
      .ok sched')
    (hle : rsched.size ≤ j + 1)
    (hL0 : rsched.size = j + 1 → ∀ vals, ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList) :
    ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ h m st v = .ok (out, s', why) ∧
      (ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := ((ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)])).stopped } := by
  have hco : (ofLoop T L₀ real).coins x h.length = loopNr x (q + 1) := by
    show loopNr x (h.length - 1) = _
    rw [hq, show q + 2 - 1 = q + 1 by omega]
  obtain ⟨v, s', why, hv⟩ := hex
  refine ⟨v, false, s', why, hv, ?_⟩
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  cases h with
  | nil => simp only [List.length_nil] at hq; omega
  | cons e0 h1 =>
  obtain ⟨m0, r0⟩ := e0
  have hcs1 : h1.map (fun p => ZKMal.CoinLayer.rd p.2) = cs := hcs
  have hq1 : cs.length = q + 1 := by
    rw [← hcs1, List.length_map]
    simp only [List.length_cons] at hq
    omega
  have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x cs := by
    rw [← hcs1]
    exact loop_coinsRead loopNr T (withFrames T L₀ real) x m0 r0 h1
  have hF : (ofLoop T L₀ real).frameAnswer x ρ ((m0, r0) :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ (m0, r0) h1 hx (by rw [← hcs1, List.length_map] at hq1; exact hq1) hj,
    hcs1] at hF
  have hsch : schedOf (real x w ρ) x cs = none :=
    schedOf_none (real x w ρ) x cs hx hq1 (by rw [hk1]; exact hreal) (by rw [hk1, hj]; exact hrs0)
      (by rw [hj]; exact hle)
  have hv' : check ((ofLoop T L₀ real).cfg x ρ ((m0, r0) :: h1)) { m with stopped := false } ⟨v.toArray⟩
      ⟨(L₀.scheduled x w ρ ((m0, r0) :: h1)
        (computedOf ((ofLoop T L₀ real).computed x w ρ ((m0, r0) :: h1)))).toArray⟩ (loopNr x (q + 1)) =
      .ok (out, s', why) := by
    unfold checkAt at hv
    have hsc : ∀ vals, (ofLoop T L₀ real).scheduled x w ρ ((m0, r0) :: h1) vals =
        L₀.scheduled x w ρ ((m0, r0) :: h1) vals := fun vals => by
      show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
        | some r => r.data.toList
        | none => L₀.scheduled x w ρ ((m0, r0) :: h1) vals) = _
      rw [hcs1, hsch]
    rw [hsc] at hv
    rw [← hco]
    exact hv
  have hcs' : (((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map (fun p => ZKMal.CoinLayer.rd p.2) =
      cs ++ [ZKMal.CoinLayer.rd a] := by
    simp only [← hcs1, List.cons_append, List.tail_cons, List.map_append, List.map_cons, List.map_nil]
  show progOf x ((((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2) =
    { s' with stopped :=
      (progOf x ((((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2)).stopped }
  rw [hcs']
  rw [hcr, readCoins_eq x cs hq1] at hF
  have hsz := roundsAt_size hrs
  exact progOf_own_step x cs (ZKMal.CoinLayer.rd a) hx hq1 (by rw [hk1]; exact hph) hc hz hH hM hrow hj
    (by rw [hk1]; exact hs') (by rw [← hsz]; exact hle) (fun k' hk' => roundsAt_framed hrs hk') hF hv'
    (fun hS => by
      obtain ⟨w', hw', he⟩ := hL0 (by rw [hsz]; exact hS)
        (computedOf ((ofLoop T L₀ real).computed x w ρ ((m0, r0) :: h1)))
      exact ⟨w', hw', by rw [← he]⟩)

open Loop in
/-- **`ProgChecks` at the loop's own frames** (`ofLoop_progChecks_of`, its round past the rep's schedule proved,
`ofLoop_progChecks_own`): from the bounds the placeholder's schedule reads (`hS`), the rep's own tables where `real`
answers the loop's (`hR`), the tables' link points (`hP`), and two named premises:
* `hRow`: the placeholder's row of pads, which the inner proof's rounds 1 to 3 frame, within a `u64`'s count;
* `hL0`: at the inner proof's round 0, the first past the rep's schedule on its own table, `L₀`'s round in the
  device's place is the firewall's own round 0 at some `final_c'`. -/
theorem ofLoop_progChecks (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (hS : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) (link : Array F128) (c' : Cfg)
      (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → k < spec.repStreams.size →
      link.size ≤ _root_.Flock.Firewall.Frames.COIN_MAX → placeholder spec sts k link = .ok (c', t, dom, xs, hi) →
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64)
    (hR : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (hP : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → k < spec.repStreams.size →
      sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧ spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX)
    (hRow : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) (link : Array F128) (c' : Cfg)
      (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → k < spec.repStreams.size →
      placeholder spec sts k link = .ok (c', t, dom, xs, hi) → c'.rowH.size < 2 ^ 64)
    (hL0 : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (h : List (Msg × Msg)) (vals : _) (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (q k j : ℕ)
      (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray) (rsched : Array (ByteArray × ℕ)),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
      ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1) → real x w ρ k = .ok (tr, hr) →
      schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
          (wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hr
        ((Array.range (j + 1)).map
          fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
      rsched.size = j + 1 →
      ∃ w' : ByteArray, w'.size = 16 ∧ L₀.scheduled x w ρ h vals =
        (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList) :
    ProgChecks (ofLoop T L₀ real) x w := by
  refine ofLoop_progChecks_of T L₀ real x w hS hR hP ?_
  intro ρ h a m out s0 spec sts q k j c' t dom xs hi tr hr rsched hx hq hki hph hreal hrs hle hA hkd hex
  have hqs := loop_round_in_plan T (withFrames T L₀ real) x ρ h hA (kindAt_masked hkd).2.2 hx hq
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 q hqs
    rw [hki] at this
    simpa [roundsOf] using this
  have hls : (wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    match h, hq with
    | [], hq => simp only [List.length_nil] at hq; omega
    | [_], hq => simp only [List.length_cons, List.length_nil] at hq; omega
    | _ :: _ :: _, _ => simp [wordsAt, FlockDeploy.Coin.vals]
  obtain ⟨hdom, hxs⟩ := hS s0 spec sts k _ c' t dom xs hi hx hk (by rw [hls]; exact Nat.min_le_right _ _) hph
  have hT := Count.real_tabAlike (hR ρ s0 spec sts hx) hph hreal
  obtain ⟨hdg, hm, hn, hlq, hp, hd⟩ := Count.real_bounds hph hT
  obtain ⟨-, -, -, -, hdm, hxe⟩ := placeholder_ok hph
  have hR' := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi
    ((Array.range (j + 1)).map fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1))
    hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hm hxs hn hlq hp hd
  rw [hrs] at hR'
  cases hs' : schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1))
    with
  | error e => rw [hs'] at hR'; exact (hR' : False).elim
  | ok sched' =>
  rw [hs'] at hR'
  obtain ⟨hc, hH, hM, hz⟩ := ofLoop_cfg_place T L₀ real x ρ h hx hq hki hph
  have hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hr
      ((Array.range (j + 1)).map
        fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched := by
    rw [← hdm, ← hxe]
    exact hrs
  exact ofLoop_progChecks_own T L₀ real x w ρ h a hA hex rfl hx hq hki hph hc hz hH hM
    (hRow s0 spec sts k _ c' t dom xs hi hx hk hph) hreal hrs0 hR' hs' hle
    (fun hS vals => hL0 ρ h vals s0 spec sts q k j tr hr rsched hx hq hki hreal hrs0 hS)

end LeanRelease

end Claims.ZKServer
