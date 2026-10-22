import Proofs.Flock.Claims.ZK.FrameOwn
import Proofs.Flock.Claims.ZK.FramePass

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
theorem fixedBut_inv {X pre post : ByteArray} {f : ℕ}
    (h : (X.size == pre.size + f + post.size && X.extract 0 pre.size == pre &&
      X.extract (pre.size + f) X.size == post) = true) :
    X = pre ++ X.extract pre.size (pre.size + f) ++ post ∧ (X.extract pre.size (pre.size + f)).size = f := by
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨h0, h1⟩, h2⟩ := h
  have hs : X.size = pre.size + f + post.size := beq_iff_eq.1 h0
  have e1 : X.extract 0 pre.size = pre :=
    ByteArray.ext (beq_iff_eq.1 (show ((X.extract 0 pre.size).data == pre.data) = true from h1))
  have e2 : X.extract (pre.size + f) X.size = post :=
    ByteArray.ext (beq_iff_eq.1 (show ((X.extract (pre.size + f) X.size).data == post.data) = true from h2))
  refine ⟨?_, by rw [ByteArray.size_extract]; omega⟩
  conv_lhs => rw [← ByteArray.extract_zero_size (b := X)]
  rw [ByteArray.extract_eq_extract_append_extract (a := X) (i := 0) (k := X.size) pre.size (by omega) (by omega),
    ByteArray.extract_eq_extract_append_extract (a := X) (i := pre.size) (k := X.size) (pre.size + f) (by omega)
      (by omega), e1, e2, ByteArray.append_assoc]

theorem eq_of_bne_false {X Y : ByteArray} (h : (X != Y) = false) : X = Y := by
  have h' : (X == Y) = true := by simpa only [bne, Bool.not_eq_false'] using h
  exact ByteArray.ext (beq_iff_eq.1 (show (X.data == Y.data) = true from h'))

/-- **What the frame takes at the inner proof's round `k` is the firewall's own** (`ownRound`, at the frame's
configuration and state), at some 16-byte `final_c'` and 64-byte `τ` commitment: the converse of `innerOwn_ownRound`. -/
theorem innerOwn_inv {c : Cfg} {s : Masker} {X : ByteArray} {n : ℕ} :
    ∀ {k : ℕ}, innerOwn c s k X n = .ok () →
      k < 4 ∧ ∃ w t, w.size = 16 ∧ t.size = 64 ∧ X = ownRound c s w t n k
  | 0, h => by
    refine ⟨by omega, ?_⟩
    cases hB : (X.size == (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size + 16 +
        (header 4 1 1).size &&
        X.extract 0 (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size ==
          labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 &&
        X.extract ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1).size + 16) X.size ==
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
    cases hB : (X != header 3 2 2 ++ wordBytes #[s.eps * c.h[c.lay.fa]! + c.h[c.lay.tx]!,
        c.h[c.lay.fb]! + c.h[c.lay.ty]!] ++ header 4 2 n) with
    | true =>
      simp only [hB, ↓reduceIte] at h
      cases h
    | false => exact eq_of_bne_false hB
  | 2, h => by
    refine ⟨by omega, ?_⟩
    cases hB : (X.size == (header 5 0 64).size + 64 + (header 4 1 1).size &&
        X.extract 0 (header 5 0 64).size == header 5 0 64 &&
        X.extract ((header 5 0 64).size + 64) X.size == header 4 1 1) with
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
    cases hB : (X != header 3 2 ((c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b).size ++
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

end Claims.ZKServer
