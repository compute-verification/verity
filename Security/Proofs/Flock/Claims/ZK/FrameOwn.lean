import Proofs.Flock.Claims.ZK.FrameShape

/-!
# The frame takes the firewall's own rounds of the inner proof

At the inner proof's rounds the frame fixes every frame and
every value it can compute from its draws and the coins (`innerOwn`): only `final_c'` and `τ`'s commitment are free. Were
the frame to refuse one of the firewall's own rounds, the rep's next coins would be refused ("coins for no round"), a
stop that reads the draws. So:

* `ownRound`: the firewall's own round `k` of the inner proof at a configuration and state, as `innerOwn` fixes it.
* **`innerOwn_ownRound`**: at every configuration and state, so at the rep's draws and at any other draws alike, the
  frame's check of the inner proof takes it.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame wordBytes innerOwn)

/-- **The firewall's own round `k` of the inner proof** at the configuration `c` and the state `s` (its `ε` and `β`),
squeezing `n` coins, as the frame fixes it (`innerOwn`): round 0 the inner label, `final_c'`'s word `w` and `ε`'s
squeeze; round 1 `(ρ, σ) = (ε h_fa + h_tx, h_fb + h_ty)` and the batching coins; round 2 `τ`'s 64-byte commitment `t`
and `β`'s squeeze; round 3 `Y = (h ‖ pad_h) + β (μ ‖ pad_μ)` and the queries. -/
def ownRound (c : Cfg) (s : Masker) (w t : ByteArray) (n : ℕ) : ℕ → ByteArray
  | 0 => (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1) ++ w ++ header 4 1 1
  | 1 => header 3 2 2 ++ wordBytes #[s.eps * c.h[c.lay.fa]! + c.h[c.lay.tx]!, c.h[c.lay.fb]! + c.h[c.lay.ty]!] ++
      header 4 2 n
  | 2 => header 5 0 64 ++ t ++ header 4 1 1
  | _ =>
    let y := (c.rowH.zip c.rowMu).map fun (a, b) => a + s.beta * b
    header 3 2 y.size ++ wordBytes y ++ header 4 2 n

theorem byteArray_beq_self (a : ByteArray) : (a == a) = true := by
  show (a.data == a.data) = true
  simp

theorem byteArray_bne_self (a : ByteArray) : (a != a) = false := by
  simp only [bne, byteArray_beq_self, Bool.not_true]

/-- A round that is `pre`, then `free` bytes, then `post`, is fixed but for those bytes. -/
theorem fixedBut_of (pre w post : ByteArray) :
    ((pre ++ w ++ post).size == pre.size + w.size + post.size &&
      (pre ++ w ++ post).extract 0 pre.size == pre &&
      (pre ++ w ++ post).extract (pre.size + w.size) (pre ++ w ++ post).size == post) = true := by
  have e1 : (pre ++ w ++ post).extract 0 pre.size = pre := by
    rw [ByteArray.append_assoc]
    exact ByteArray.extract_append_eq_left rfl
  have e2 : (pre ++ w ++ post).extract (pre.size + w.size) (pre ++ w ++ post).size = post :=
    ByteArray.extract_append_eq_right ByteArray.size_append.symm ByteArray.size_append
  rw [e1, e2, byteArray_beq_self, byteArray_beq_self, ByteArray.size_append, ByteArray.size_append]
  simp

/-- **The frame takes the firewall's own rounds of the inner proof** (R3): at any configuration and state, round
`k < 4` as `ownRound` builds it, from a 16-byte `final_c'` and a 64-byte `τ` commitment, is the round `innerOwn` fixes. -/
theorem innerOwn_ownRound (c : Cfg) (s : Masker) (w t : ByteArray) (hw : w.size = 16) (ht : t.size = 64) (n : ℕ) :
    ∀ k, k < 4 → innerOwn c s k (ownRound c s w t n k) n = .ok ()
  | 0, _ => by
    unfold innerOwn ownRound
    have h := fixedBut_of (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1) w (header 4 1 1)
    rw [hw] at h
    simp only [h, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    rfl
  | 1, _ => by
    unfold innerOwn ownRound
    simp only [byteArray_bne_self, Bool.false_eq_true, ↓reduceIte]
    rfl
  | 2, _ => by
    unfold innerOwn ownRound
    have h := fixedBut_of (header 5 0 64) t (header 4 1 1)
    rw [ht] at h
    simp only [h, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    rfl
  | 3, _ => by
    unfold innerOwn ownRound
    simp only [byteArray_bne_self, Bool.false_eq_true, ↓reduceIte]
    rfl

end Claims.ZKServer
