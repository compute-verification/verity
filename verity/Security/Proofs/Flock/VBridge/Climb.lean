import Proofs.Flock.VBridge.Sha
import Proofs.Flock.Verifier.Hash
import Flock.Merkle

/-!
# V*'s climb through the tree's nodes

The loop of `verity_flock.rec_open._climb`, as the verifier's builder (`Flock.HmNets`) states it: at level `h` the
direction bit `d[h]` swaps the current digest and the sibling (`_Forms.swap`: `t = d ∧ (cur ⊕ sib)` bit by bit, then
`(cur ⊕ t, sib ⊕ t)`), and the node is SHA-512 of the pair from the IV (`_sha(ops, IV, left + right, 0)`).

`sound_climbFrom`: whenever the rows hold, if the forms carry the bytes of a 64-byte digest and of 64-byte siblings
(`Carries`) and the direction bits are the position's low bits, the output carries the verifier of record's climb,
`Flock.climb ms` of the digest at the position through the siblings, for any scheme `ms` whose node is `SHA-512(l ‖ r)`
(`MerkleScheme.hm96Sha512`, and `MerkleScheme.sha512`'s unsalted leaves).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## The builder -/

/-- `_Forms.swap`: `t = d ∧ (a ⊕ s)` bit by bit, then `(a ⊕ t, s ⊕ t)`, which is `(a, s)` when `d` is 0 and `(s, a)`
when it is 1. -/
def swap (d : Lin) (a s : List Lin) : M (List Lin × List Lin) := do
  let t ← (a.zip s).mapM fun x => Flock.HmNets.and d (x.1 ^^^ x.2)
  pure (List.zipWith (· ^^^ ·) a t, List.zipWith (· ^^^ ·) s t)

/-- `_climb`'s loop from level `h`: the digest `cur` climbs through the siblings `sibs`, the direction at level `h + j`
being `d[h + j]`, each node SHA-512 of the pair from the IV. -/
def climbFrom (d : List Lin) : ℕ → List Lin → List (List Lin) → M (List Lin)
  | _, cur, [] => pure cur
  | h, cur, s :: rest => do
    let lr ← swap (d.getD h Lin.zero) cur s
    let cur ← shaFrom (preCv [] 0) (lr.1 ++ lr.2) 0
    climbFrom d (h + 1) cur rest

/-- The forms `w` carry the bytes `b`: one form a bit, bit `t % 8` of byte `t / 8`. -/
def Carries (z : Var → Bool) (w : List Lin) (b : ByteArray) : Prop :=
  w.length = 8 * b.size ∧ ∀ t < w.length, bit z w t = bitL b.data.toList t

/-! ## What the rows force -/

theorem carries_append {z : Var → Bool} {w w' : List Lin} {b b' : ByteArray} (h : Carries z w b)
    (h' : Carries z w' b') : Carries z (w ++ w') (b ++ b') := by
  obtain ⟨hl, hb⟩ := h
  obtain ⟨hl', hb'⟩ := h'
  have hbl : b.data.toList.length = b.size := Array.length_toList
  refine ⟨by rw [List.length_append, ByteArray.size_append, hl, hl', Nat.mul_add], fun t ht => ?_⟩
  rw [List.length_append] at ht
  rw [ByteArray.toList_data_append]
  by_cases htw : t < w.length
  · rw [bitL_append_left _ (by omega), ← hb t htw, bit, bit, List.getElem?_append_left htw]
  · rw [bitL_append_right _ (by omega), bit, List.getElem?_append_right (by omega),
      show t - w.length = t - 8 * b.data.toList.length by omega, ← bit, hb' _ (by omega)]

/-- SHA-512 from the IV of forms that carry `x` carries `SHA-512(x)`. -/
theorem sound_shaIv (bits : List Lin) : Sound (shaFrom (preCv [] 0) bits 0) fun z out =>
    ∀ x : ByteArray, Carries z bits x → Carries z out (Sha512.hash x) :=
  Sound.mono (sound_shaFrom (preCv [] 0) bits 0) fun z out h x hx => by
    obtain ⟨h1, h2⟩ := h [] x.data.toList rfl (by rw [Array.length_toList]; exact hx.1) hx.2 rfl
    have hx' : catBytes [] x.data.toList = x := by simp [catBytes]
    exact ⟨by rw [h1, Sha512.hash_size], fun t ht => by rw [h2 t (by omega), hx']⟩

theorem sound_swap (d : Lin) (a s : List Lin) : Sound (swap d a s) fun z lr => ∀ x y : ByteArray,
    Carries z a x → Carries z s y → x.size = y.size →
      Carries z lr.1 (if ev z d then y else x) ∧ Carries z lr.2 (if ev z d then x else y) := by
  unfold swap
  refine Sound.bind' (sound_mapM _ (fun z x => ev z d && (ev z x.1 ^^ ev z x.2))
    (fun x => Sound.mono (sound_and d _) fun z r hr => by rw [hr, ev_xor]) _)
    fun t => Sound.pure' fun z ht x y hx hy hxy => ?_
  obtain ⟨htl, htb⟩ := ht
  dsimp only
  have hl : a.length = s.length := by rw [hx.1, hy.1, hxy]
  have hz : (a.zip s).length = a.length := by simp [hl]
  have hlen1 : (List.zipWith (· ^^^ ·) a t).length = a.length := by simp [htl, hz]
  have hlen2 : (List.zipWith (· ^^^ ·) s t).length = a.length := by simp [htl, hz, hl]
  have hbit : ∀ i < a.length, bit z (List.zipWith (· ^^^ ·) a t) i = (if ev z d then bit z s i else bit z a i) ∧
      bit z (List.zipWith (· ^^^ ·) s t) i = (if ev z d then bit z a i else bit z s i) := fun i hi => by
    have hit : i < t.length := by omega
    have his : i < s.length := by omega
    have hi1 : i < (List.zipWith (· ^^^ ·) a t).length := by omega
    have hi2 : i < (List.zipWith (· ^^^ ·) s t).length := by omega
    have hti := htb i (by omega)
    rw [bit_of_lt hit] at hti
    simp only [List.getElem_zip] at hti
    rw [bit_of_lt hi1, bit_of_lt hi2, bit_of_lt hi, bit_of_lt his, List.getElem_zipWith, List.getElem_zipWith, ev_xor,
      ev_xor, hti]
    cases ev z d <;> cases ev z a[i] <;> cases ev z s[i] <;> decide
  refine ⟨⟨?_, fun i hi => ?_⟩, ⟨?_, fun i hi => ?_⟩⟩
  · rw [hlen1]; cases ev z d <;> simp [hx.1, hxy]
  · rw [(hbit i (by omega)).1]; cases ev z d <;> simp [hx.2 i (by omega), hy.2 i (by omega)]
  · rw [hlen2]; cases ev z d <;> simp [hx.1, hxy]
  · rw [(hbit i (by omega)).2]; cases ev z d <;> simp [hx.2 i (by omega), hy.2 i (by omega)]

/-- **V*'s climb is the verifier of record's.** For a Merkle scheme `ms` whose node is `SHA-512(l ‖ r)`, whenever the
rows of `climbFrom d h cur sibs` hold under `z`: if `cur` carries a 64-byte digest `acc`, each sibling's forms carry a
64-byte sibling of `ss`, and the direction bit at level `h + j` is bit `j` of the position `idx`, the output carries
`Flock.climb ms acc idx ss`. -/
theorem sound_climbFrom (ms : MerkleScheme) (hms : ∀ l r, ms.node l r = Sha512.hash (l ++ r)) (d : List Lin) :
    ∀ (h : ℕ) (cur : List Lin) (sibs : List (List Lin)),
    Sound (climbFrom d h cur sibs) fun z out => ∀ (acc : ByteArray) (idx : ℕ) (ss : List ByteArray),
      acc.size = 64 → Carries z cur acc → List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) sibs ss →
      (∀ j < sibs.length, ev z (d.getD (h + j) Lin.zero) = idx.testBit j) →
      Carries z out (Flock.climb ms acc idx ss)
  | _, cur, [] => Sound.pure' fun z acc idx ss _ hcur hss _ => by
      cases hss
      exact hcur
  | h, cur, s :: rest => by
      unfold climbFrom
      refine Sound.bind' (sound_swap _ cur s) fun lr => Sound.bind' (sound_shaIv _) fun cur' =>
        Sound.mono (sound_climbFrom ms hms d (h + 1) cur' rest) fun z out hih hsha hsw acc idx ss hacc hcur hss hdir => ?_
      obtain _ | ⟨⟨hsb, hs⟩, hss⟩ := hss
      rename_i sb ss
      obtain ⟨h1, h2⟩ := hsw acc sb hcur hs (by rw [hacc, hsb])
      have hd : ev z (d.getD h Lin.zero) = idx.testBit 0 := hdir 0 (by simp)
      have hnode : (if idx % 2 = 0 then ms.node acc sb else ms.node sb acc) =
          Sha512.hash ((if ev z (d.getD h Lin.zero) then sb else acc) ++
            (if ev z (d.getD h Lin.zero) then acc else sb)) := by
        rw [hd, Nat.testBit_zero, hms, hms]
        rcases Nat.mod_two_eq_zero_or_one idx with hp | hp <;> simp [hp]
      rw [Flock.climb, hnode]
      exact hih _ (idx / 2) ss (Sha512.hash_size _) (hsha _ (carries_append h1 h2)) hss fun j hj => by
        rw [Nat.testBit_div_two, ← hdir (j + 1) (by simp; omega), show h + 1 + j = h + (j + 1) by omega]

end FlockVBridge
