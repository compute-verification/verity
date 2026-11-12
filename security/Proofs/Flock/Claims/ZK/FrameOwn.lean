module

public import Flock.Firewall.Frame
public import Mathlib.Data.Nat.Notation
public import Mathlib.Tactic.ToAdditive
public import Mathlib.Tactic.ToDual
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import VCVio.Prelude.Core

/-!
# The frame takes the ZK server's own rounds of the inner proof

At the inner proof's rounds the frame fixes every frame and
every value it can compute from its samples and the coins (`innerOwn`): only `final_c'` and `τ`'s commitment are free. Were
the frame to refuse one of the ZK server's own rounds, the rep's next coins would be refused ("coins for no round"), a
stop that reads the samples. So:

* `ownRound` (the frame's `Frame.ownRound`): the ZK server's own round `k` of the inner proof at a configuration and
  state, as `innerOwn` fixes it, which the session loop's frames schedule (`Frame.innerSched`).
* **`innerOwn_ownRound`**: at every configuration and state, so at the rep's samples and at any other samples alike, the
  frame's check of the inner proof takes it.
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker labelFrame wordBytes innerOwn)

export _root_.Flock.Firewall.Frame (ownRound)

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

/-- **The frame takes the ZK server's own rounds of the inner proof** (R3): at any configuration and state, round
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
