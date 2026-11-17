module

public import Flock.ZkServer.Frame
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

/-! The frame takes the ZK server's own rounds of the inner proof, since refusing one would refuse the rep's
next coins, a stop that reads the samples. -/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.ZkServer.Frame (Cfg Masker labelFrame wordBytes innerOwn)

export _root_.Flock.ZkServer.Frame (ownRound)

theorem byteArray_beq_self (a : ByteArray) : (a == a) = true := by
  show (a.data == a.data) = true
  simp

theorem byteArray_bne_self (a : ByteArray) : (a != a) = false := by
  simp only [bne, byteArray_beq_self, Bool.not_true]

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
