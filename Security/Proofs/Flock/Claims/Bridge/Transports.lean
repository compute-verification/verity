import Proofs.Flock.Claims.Bridge.BindsColl
import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH

/-!
# The deployed check opens shaped rows only

The game's binding (`HmRowsZX`, at the hidden-output layout) opens a registered row with any value `v : BitVal W`, of
any width, and hashes `encPortH p v`: at a `sha512/row/v2` port, `rowEncP (rowPre true) v`, a row's `P(ℓ) ‖ bytes` when
`v` is `RowShaped` and `0^{128+W} ‖ bitCode v` otherwise. The deployed sampling commitment (`SamplingDeploy.commit c`)
hashes only `RowV2.enc` of a row of its place's width. So the openings the deployed check passes are the shaped ones:

* `row_eq_rowEncP`: a deployed opening's row is `rowEncP (rowPre true)` of its value (`valOf`), which is `RowShaped`
  of its place's width (`valOf_shaped`), at a place of `16 w` bits.
* `row_ne_misshaped`: **a misshaped value's bytes are no deployed opening's row**: `rowEncP` is injective.
* `opens_sha512`: at SHA-512 the deployed check passes exactly when the verifier's own port at the root `at_` names
  opens the hm96 leaf of that row (`Flock.Registered.opensLeaf`), so `opens_shaped`: the leaf it opens is the game's
  leaf of a shaped value.
* `leaf_misshaped`: **a misshaped game opening's leaf is the leaf of a deployed opening only by a SHA-512 collision**
  (`Binding.hm96Pair_spec`).
-/

namespace SamplingDeploy

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open FlockSoundness.Discharge.Layout (BitVal RowShaped rowEncP rowBytes widthOf padBits)
open Properties.NetworkCertifier.Bytes (bs)

variable {W : ℕ} (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (o : Opening Bool)

/-- A deployed opening's row bits, by index: the opened value at its place, the data's bits elsewhere. -/
def rowFn : Fin (place o.pos).1 → Bool := fun i => if i = (place o.pos).2 then o.val else bits o.data i

theorem rowBits_eq : rowBits place bits o = List.ofFn (rowFn place bits o) := rfl

/-- **The game value a deployed opening's row is**: its place's width and its bits. -/
def valOf (W : ℕ) : BitVal W := (widthOf W (place o.pos).1, padBits (rowFn place bits o))

variable {place bits o}

theorem valOf_width (hW : (place o.pos).1 ≤ W) : (valOf place bits o W).1.val = (place o.pos).1 := by
  simp only [valOf, widthOf]
  omega

/-- **A deployed opening's value is shaped**, of its place's width, at a place of `16 w` bits. -/
theorem valOf_shaped (h16 : (place o.pos).1 % 16 = 0) (hW : (place o.pos).1 ≤ W) :
    RowShaped (valOf place bits o W) ∧ (valOf place bits o W).1.val = (place o.pos).1 := by
  have hw := valOf_width (bits := bits) hW
  refine ⟨⟨by rw [hw]; exact h16, fun i hi => ?_⟩, hw⟩
  rw [hw] at hi
  simp only [valOf, padBits, show ¬ i.val < (place o.pos).1 by omega, dite_false]

theorem bitsVal_eq_sum : ∀ m : List Bool,
    Flock.HmRow.bitsVal m = ∑ t : Fin m.length, if m[t.val] then 2 ^ t.val else 0
  | [] => by simp [Flock.HmRow.bitsVal]
  | b :: m => by
    show _ = ∑ t : Fin (m.length + 1), _
    rw [Fin.sum_univ_succ]
    simp only [Flock.HmRow.bitsVal, bitsVal_eq_sum m, List.length_cons, Fin.val_zero, List.getElem_cons_zero,
      pow_zero, Fin.val_succ, List.getElem_cons_succ, Finset.mul_sum]
    congr 1
    refine Finset.sum_congr rfl fun t _ => ?_
    split_ifs <;> ring

/-- `sha512/row/v2`'s bytes of a row of `16 k` bits are `rowBytes` of its value: bit `8 j + t` is bit `t` of byte `j`
in both. -/
theorem packBits_eq_rowBytes {ℓ : ℕ} (f : Fin ℓ → Bool) (h16 : ℓ % 16 = 0) (hW : ℓ ≤ W) :
    Flock.HmRow.packBits (List.ofFn f) = rowBytes ((widthOf W ℓ, padBits f) : BitVal W) := by
  have hw : (widthOf W ℓ).val = ℓ := by simp only [widthOf]; omega
  apply List.ext_getElem
  · simp only [Flock.HmRow.packBits, List.length_map, List.length_range, List.length_ofFn, rowBytes, hw]
    omega
  · intro j h1 h2
    have hj : 8 * j + 8 ≤ ℓ := by
      simp only [rowBytes, List.length_ofFn, hw] at h2
      omega
    simp only [Flock.HmRow.packBits, List.getElem_map, List.getElem_range, rowBytes, List.getElem_ofFn, byteOf]
    rw [bitsVal_eq_sum]
    have hl : ((List.drop (8 * j) (List.ofFn f)).take 8).length = 8 := by
      simp only [List.length_take, List.length_drop, List.length_ofFn]
      omega
    show UInt8.ofNat _ = UInt8.ofNat _
    congr 1
    rw [← Fin.sum_congr' _ hl]
    refine Finset.sum_congr rfl fun t _ => ?_
    have ht : 8 * j + t.val < ℓ := by have := t.isLt; omega
    simp only [Fin.val_cast, List.getElem_take, List.getElem_drop, List.getElem_ofFn,
      FlockSoundness.Discharge.Layout.BitVal.get, padBits]
    rw [dif_pos (show 8 * j + t.val < W by omega), dif_pos ht]

/-- **A deployed opening's row is `rowEncP` of its value**, at a place of `16 w` bits. -/
theorem row_eq_rowEncP (h16 : (place o.pos).1 % 16 = 0) (hW : (place o.pos).1 ≤ W) :
    row place bits o = rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) (valOf place bits o W) := by
  obtain ⟨hs, hw⟩ := valOf_shaped (bits := bits) h16 hW
  unfold rowEncP
  rw [ite_cond_eq_true _ _ (eq_true hs), hw]
  unfold row FlockLevel3.RowV2.enc
  rw [rowBits_eq, List.length_ofFn]
  congr 1
  · unfold FlockSoundness.Discharge.Hidden.rowPre
    rw [ite_cond_eq_true _ _ rfl, show 16 * ((place o.pos).1 / 16) = (place o.pos).1 by omega]
  · exact packBits_eq_rowBytes _ h16 hW

/-- **A misshaped value's bytes are no deployed opening's row**: a value that isn't `RowShaped` of the place's width. -/
theorem row_ne_misshaped (h16 : (place o.pos).1 % 16 = 0) (hW : (place o.pos).1 ≤ W) {v : BitVal W}
    (hv : ¬ (RowShaped v ∧ v.1.val = (place o.pos).1)) :
    row place bits o ≠ rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v := by
  rw [row_eq_rowEncP h16 hW]
  intro h
  have e := FlockSoundness.Discharge.Layout.rowEncP_inj (fun w => FlockSoundness.Discharge.Hidden.size_rowPre true w) h
  rw [← e] at hv
  exact hv (valOf_shaped h16 hW)

/-- **At SHA-512 the deployed check is the verifier's**: it passes exactly when the verifier's own port whose root
`at_` names opens the hm96 leaf of the opening's row at `at_`'s index through `π`. -/
theorem opens_sha512 (c : Code) (roots : List Msg) (o : Opening Bool) :
    (commit c).Opens sha512Msg roots o = true ↔
      ∃ ri reg, c.at_ roots o.pos = some ri ∧ c.own.find? (fun p => decide (bs p.root = ri.1)) = some reg ∧
        Flock.Registered.opensLeaf reg (bytes (digBytes (H512 (hm96Input (row c.place c.bits o) (c.salt o)))))
          ri.2 (bytes (c.π o)) = true := by
  unfold Commitment.Opens commit pathCommit
  dsimp only
  cases hat : c.at_ roots o.pos with
  | none => simp [OC.eval]
  | some ri =>
    simp only [ocBind_eval, hm96Leaf_sha512]
    rw [regPath_sha512]
    simp

/-- **The deployed check opens shaped values' leaves**: the leaf it opens is the game's hm96 leaf of the opening's value,
`RowShaped` of its place's width. -/
theorem opens_shaped (c : Code) (roots : List Msg) (o : Opening Bool) (h16 : (c.place o.pos).1 % 16 = 0)
    (hW : (c.place o.pos).1 ≤ W) :
    (commit c).Opens sha512Msg roots o = true ↔
      ∃ ri reg, c.at_ roots o.pos = some ri ∧ c.own.find? (fun p => decide (bs p.root = ri.1)) = some reg ∧
        Flock.Registered.opensLeaf reg (bytes (digBytes (H512 (hm96Input
          (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) (valOf c.place c.bits o W)) (c.salt o)))))
          ri.2 (bytes (c.π o)) = true := by
  rw [opens_sha512, row_eq_rowEncP h16 hW]

/-- **The deployed check refuses misshaped openings**: a game opening of a value that isn't `RowShaped` of the place's
width has the hm96 leaf of a deployed opening's row only by a SHA-512 collision. -/
theorem leaf_misshaped (c : Code) (o : Opening Bool) (h16 : (c.place o.pos).1 % 16 = 0) (hW : (c.place o.pos).1 ≤ W)
    {v : BitVal W} (hv : ¬ (RowShaped v ∧ v.1.val = (c.place o.pos).1)) (y : List UInt8)
    (h : H512 (hm96Input (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v) y) =
      H512 (hm96Input (row c.place c.bits o) (c.salt o))) :
    Collides (hm96Pair (rowEncP (FlockSoundness.Discharge.Hidden.rowPre true) v) y (row c.place c.bits o) (c.salt o)) :=
  hm96Pair_spec _ _ _ _ (Ne.symm (row_ne_misshaped h16 hW hv)) h

end SamplingDeploy
