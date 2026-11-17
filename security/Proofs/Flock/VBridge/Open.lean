module

public import Proofs.Flock.VBridge.Climb
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
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import Proofs.Flock.Verifier.Hash

/-! V*'s opening climbs the row's leaf to the top (`sound_climbRow`); `acc_out`, the other half of `_steps`, is not stated here. -/

@[expose] public section

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

def sibForms (aux : List Lin) (H : ℕ) : List (List Lin) := (List.range H).map fun h => (aux.drop (512 * h)).take 512

def climbRow (H : ℕ) (row aux d : List Lin) : M (List Lin) := do
  let leaf ← shaFrom (preCv [] 0) row 0
  let top ← climbFrom d 0 leaf (sibForms aux H)
  pure (top ++ List.replicate 512 Lin.zero)

theorem carries_inj {z : Var → Bool} {w : List Lin} {b b' : ByteArray} (h : Carries z w b) (h' : Carries z w b') :
    b = b' := by
  have hs : b.size = b'.size := by have := h.1.symm.trans h'.1; omega
  have hl : b.data.toList.length = b'.data.toList.length := by
    rw [Array.length_toList, Array.length_toList]; exact hs
  have he := eq_of_bitL hl fun t ht => by
    rw [Array.length_toList] at ht
    rw [← h.2 t (by rw [h.1]; exact ht), h'.2 t (by rw [h.1]; exact ht)]
  cases b
  cases b'
  simp only [ByteArray.mk.injEq]
  exact Array.toList_inj.1 he

theorem carries_zeros (z : Var → Bool) (n : ℕ) :
    Carries z (List.replicate (8 * n) Lin.zero) ⟨Array.replicate n 0⟩ := by
  refine ⟨by simp [ByteArray.size], fun t ht => ?_⟩
  rw [List.length_replicate] at ht
  rw [bit, List.getElem?_replicate, bitL, Array.toList_replicate, List.getD_eq_getElem _ _ (by simp; omega),
    List.getElem_replicate]
  simp [ht]

theorem sound_climbRow (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (H : ℕ)
    (row aux d : List Lin) : Sound (climbRow H row aux d) fun z out => ∀ (rb : ByteArray) (idx : ℕ) (ss : List ByteArray),
      Carries z row rb → List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (sibForms aux H) ss →
      (∀ h < H, ev z (d.getD h Lin.zero) = idx.testBit h) →
      Carries z out (Flock.climb ms (Sha512.hash rb) idx ss ++ ⟨Array.replicate 64 0⟩) := by
  unfold climbRow
  refine Sound.bind' (sound_shaIv row) fun leaf => Sound.bind' (sound_climbFrom ms hms d 0 leaf _) fun top =>
    Sound.pure' fun z htop hleaf rb idx ss hrow hss hdir => ?_
  refine carries_append (htop _ idx ss (Sha512.hash_size _) (hleaf rb hrow) hss fun j hj => ?_) (carries_zeros z 64)
  rw [Nat.zero_add]
  exact hdir j (by simpa [sibForms] using hj)

end FlockVBridge
