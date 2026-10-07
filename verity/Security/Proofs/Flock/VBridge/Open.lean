import Proofs.Flock.VBridge.Climb
import Proofs.Flock.Soundness.Discharge.Hm.Commit

/-!
# V*'s opening: the row's leaf climbs to the top

`_climb` of `verity_flock.rec_open`'s `RecOpen_v3` (rec-step3) and `RecOpen_v4` (#1347), which share it, as the
verifier's builder (`Flock.HmNets`) states it: the leaf is plain SHA-512 of the row's bits from the IV (the inner
statement's `flock-leaf/sha512-unsalted`), it climbs `H` levels through the siblings in `aux` (512 bits each, bottom up)
at the directions `d` (`climbFrom`), and `out` is the node reached and 512 zero bits.

`sound_climbRow`: whenever the rows hold, if the row's forms carry its bytes, `aux`'s sibling forms carry 64-byte siblings
and the directions are the position's low bits, `out` carries the verifier of record's climb of the row's leaf,
`Flock.climb ms (SHA-512(row))`, then 64 zero bytes: `rec_open.expected` of that node.

`out` is half of `_steps`; the other half, `acc_out` (the opening's share of the consistency sums, `cons_structure`), is
`rec_residuals.residual_forms` at `cons_structure(LANES)` and waits on that gadget's statement.
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## The builder -/

/-- `aux`'s `H` siblings, 512 bits each from bit `512 h` (`aux[16 · NODE_WORDS · h : 16 · NODE_WORDS · (h + 1)]`). -/
def sibForms (aux : List Lin) (H : ℕ) : List (List Lin) := (List.range H).map fun h => (aux.drop (512 * h)).take 512

/-- `RecOpen_v3`'s and `RecOpen_v4`'s `_climb`: SHA-512 of `row` from the IV climbs `H` levels through `aux`'s
siblings at the directions `d`; the node reached, then 512 zero bits. -/
def climbRow (H : ℕ) (row aux d : List Lin) : M (List Lin) := do
  let leaf ← shaFrom (preCv [] 0) row 0
  let top ← climbFrom d 0 leaf (sibForms aux H)
  pure (top ++ List.replicate 512 Lin.zero)

/-! ## What the rows force -/

/-- Forms carry one byte string: two strings they carry are equal. -/
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

/-- **V*'s opening climbs the row's leaf to the node `out` names.** For a Merkle scheme `ms` whose node is `SHA-512(l ‖ r)`,
whenever the rows of `climbRow H row aux d` hold under `z`: if `row` carries the bytes `rb`, the `H` sibling forms of `aux`
carry 64-byte siblings `ss`, and the direction bit at level `h < H` is bit `h` of the position `idx`, `out` carries
`Flock.climb ms (SHA-512(rb)) idx ss` followed by 64 zero bytes. -/
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
