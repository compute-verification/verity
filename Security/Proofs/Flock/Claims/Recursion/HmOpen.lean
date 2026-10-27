import Proofs.Flock.Claims.Recursion.SlotsCommit
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# An hm96 slot's two calls open its commit string (`IVStar`'s salt and row checks)

`hmCalls_true` gives a slot's two recorded calls from its commit string. This file goes the other way, to the form
`IVStar` checks a commit string `b ‖ c` against its opening `(row, salt)` at the deployed bound record
(`IVCheck.Holds`' `.salt` and `.row`, at `bdDeployed`'s `saltPrefix`, `rowInDeployed` and `mask`):

* the salt's call, `SHA-512(salt_prefix ‖ y) = c`, is the salt check at the salt read whole (`saltOf`): `saltCheck_of`;
* the row's call, `SHA-512(row_prefix ‖ row) = b ⊕ Φy`, is the row check `SHA-512(rowIn row) = b ⊕ M·y`, where the
  port's prefix is the one `rowInDeployed` reads off the row's length (`rowCheck_of`): `Φy` is `M·y` (`maskV_filter`,
  `bitL_mask`).
-/

namespace RecDeploy.HmOpen

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB (readForms formBit)
open FlockSoundness.Discharge.Hm (bitL)

/-! ## Bytes -/

theorem bitL_drop64 (l : List UInt8) (t : ℕ) : bitL (l.drop 64) t = bitL l (512 + t) := by
  unfold bitL
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_drop,
    show (512 + t) / 8 = 64 + t / 8 by omega, show (512 + t) % 8 = t % 8 by omega]

theorem bitL_take64 (l : List UInt8) {t : ℕ} (ht : t < 512) : bitL (l.take 64) t = bitL l t := by
  unfold bitL
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by omega)]

theorem bitL_xorMask {b : List UInt8} (hb : b.length = 64) (m : Fin 64 → UInt8) {t : ℕ} (ht : t < 512) :
    bitL (Claims.Recursion.xorMask b m) t = (bitL b t ^^ bitL (List.ofFn m) t) := by
  have hm : (List.ofFn m).length = 64 := List.length_ofFn
  have hb8 : t / 8 < b.length := by omega
  have hm8 : t / 8 < (List.ofFn m).length := by omega
  unfold bitL Claims.Recursion.xorMask
  simp only [List.getD_eq_getElem?_getD, List.getElem?_mapIdx, List.getElem?_eq_getElem hb8,
    List.getElem?_eq_getElem hm8, Option.map_some, Option.getD_some, UInt8.toNat_xor, Nat.testBit_xor]

theorem length_readForms (w : Claims.Flock.Wit) (fs : List HmNets.Lin) :
    (readForms w fs).length = (fs.length + 7) / 8 := by
  simp [readForms, Claims.Recursion.VStar.bytesOf]

theorem hmCommit_len (c : Circuit) (q n : ℕ) : (hmCommit c q n).length = 1024 := by
  rw [hmCommit, List.length_map, List.length_range]

theorem length_commit (w : Claims.Flock.Wit) (c : Circuit) (q n : ℕ) :
    (readForms w (hmCommit c q n)).length = 128 := by
  rw [length_readForms, hmCommit, List.length_map, List.length_range]

theorem length_salt (w : Claims.Flock.Wit) (c : Circuit) (q n : ℕ) :
    (readForms w (hmSalt c q n)).length = 192 := by
  rw [length_readForms, hmSalt, List.length_map, List.length_range]

/-- **A salt read whole is its bytes.** -/
theorem ofFn_saltOf {m : Msg} (h : m.length = 192) : List.ofFn (DeployedR.saltOf m) = m := by
  refine List.ext_getElem (by rw [List.length_ofFn, h]; rfl) fun i h1 h2 => ?_
  rw [List.getElem_ofFn]
  simp only [DeployedR.saltOf, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2, Option.getD_some]

theorem ofFn_saltOf_array {m : Msg} (h : m.length = 192) : Array.ofFn (DeployedR.saltOf m) = m.toArray := by
  apply Array.toList_inj.mp
  rw [Array.toList_ofFn, ofFn_saltOf h, List.toList_toArray]

/-- `bdDeployed`'s mask, at a salt read whole, as bytes: `M·y` at the salt's bytes. -/
theorem ofFn_mask {m : Msg} (h : m.length = 192) :
    List.ofFn (fun i : Fin 64 => (Hm96.Default512.mask ⟨Array.ofFn (DeployedR.saltOf m)⟩).get! i.val) =
      (Hm96.Default512.mask ⟨m.toArray⟩).data.toList := by
  rw [ofFn_saltOf_array h]
  have hs : (Hm96.Default512.mask ⟨m.toArray⟩).data.toList.length = 64 := by
    rw [FlockSoundness.Refine.length_data, FlockSoundness.Refine.mask_size]
  refine List.ext_getElem (by rw [List.length_ofFn, hs]) fun i h1 h2 => ?_
  rw [List.getElem_ofFn, FlockSoundness.Discharge.Hm.get!_eq_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem h2, Option.getD_some]

/-! ## The two checks -/

/-- **The salt's call is the salt check**: where `SHA-512(salt_prefix ‖ y)` is the commit string's second half (the
call's output form, `hmCommit`'s last 512 bits), the salt read whole hashes under `bdDeployed`'s prefix to the commit
string's last 64 bytes. -/
theorem saltCheck_of {w : Claims.Flock.Wit} {c : Circuit} {q n : ℕ}
    (hS : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512)) :
    FlockDeploy.sha512 (Claims.toMsg Hm96.Default512.saltPrefix ++
        List.ofFn (DeployedR.saltOf (readForms w (hmSalt c q n)))) =
      (readForms w (hmCommit c q n)).drop 64 := by
  rw [ofFn_saltOf (length_salt w c q n)]
  refine hS.trans (readForms_eq_of_bits ?_ fun t ht => ?_)
  · rw [List.length_drop, List.length_drop, length_commit, hmCommit_len]
  · have ht' : t < 512 := by simpa [hmCommit_len] using ht
    rw [bitL_drop64, List.getElem_drop, bitL_readForms (by rw [hmCommit_len]; omega)]

/-- **The row's call is the row check**: where `SHA-512(row_prefix ‖ row)` is the digest form `b ⊕ Φy` (`hmDigest`)
and the port's prefix is the one `rowInDeployed` reads off the row's length, the row's digest input hashes to the
commit string's first half under `bdDeployed`'s mask of the salt read whole, `b ⊕ M·y`. -/
theorem rowCheck_of {w : Claims.Flock.Wit} {c : Circuit} {nm : String} {q n : ℕ}
    (hpfx : rowPfx c q = (HmRow.rowPrefix
      ((readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).length / 2)).data.toList)
    (hR : FlockDeploy.sha512 (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)) =
      readForms w (hmDigest c q n)) :
    FlockDeploy.sha512 (rowInDeployed (readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))) =
      Claims.Recursion.xorMask ((readForms w (hmCommit c q n)).take 64)
        (fun i : Fin 64 =>
          (Hm96.Default512.mask ⟨Array.ofFn (DeployedR.saltOf (readForms w (hmSalt c q n)))⟩).get! i.val) := by
  rw [rowInDeployed, ← hpfx, hR]
  have hb : ((readForms w (hmCommit c q n)).take 64).length = 64 := by
    rw [List.length_take, length_commit]
    rfl
  refine readForms_eq_of_bits ?_ fun i hi => ?_
  · rw [Claims.Recursion.xorMask, List.length_mapIdx, hb]
    simp [hmDigest]
  · have hi' : i < 512 := by simpa [hmDigest] using hi
    have hy : ∀ j (hj : j < 1536), w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
        bitL (readForms w (hmSalt c q n)) j := fun j hj => by
      rw [bitL_readForms (fs := hmSalt c q n) (by simpa [hmSalt] using hj)]
      simp [hmSalt, formBit_var]
    have hmap : ((((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
        fun j => slotHid c n (hmBase c q n) (hmIn c 2 + j)).map fun v => w.hid.getD v false) =
        ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
          (bitL (readForms w (hmSalt c q n))) := by
      rw [List.map_map]
      exact List.map_congr_left fun j hj => hy j (List.mem_range.1 (List.mem_filter.1 hj).1)
    rw [bitL_xorMask hb _ hi', bitL_take64 _ hi', ofFn_mask (length_salt w c q n),
      FlockSoundness.Discharge.Hm.bitL_mask _ hi', maskV_filter, ← hmap,
      bitL_readForms (fs := hmCommit c q n) (by rw [hmCommit_len]; omega)]
    simp only [hmDigest, hmCommit, List.getElem_map, List.getElem_range, formBit, List.foldr_cons, List.foldr_nil,
      List.foldr_map, HmNets.Lin.var, Bool.xor_false]

/-- **A slot's two true calls open its commit string**: the salt check and the row check at the slot's opening, where
the port's prefix is the one `rowInDeployed` reads off the row's length. -/
theorem checks_of_calls {w : Claims.Flock.Wit} {c : Circuit} {nm : String} {q n : ℕ}
    (hT : ∀ p ∈ hmCalls c nm q n, FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2)
    (hpfx : rowPfx c q = (HmRow.rowPrefix
      ((readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).length / 2)).data.toList) :
    FlockDeploy.sha512 (Claims.toMsg Hm96.Default512.saltPrefix ++
        List.ofFn (DeployedR.saltOf (readForms w (hmSalt c q n)))) =
      (readForms w (hmCommit c q n)).drop 64 ∧
    FlockDeploy.sha512 (rowInDeployed (readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))) =
      Claims.Recursion.xorMask ((readForms w (hmCommit c q n)).take 64)
        (fun i : Fin 64 =>
          (Hm96.Default512.mask ⟨Array.ofFn (DeployedR.saltOf (readForms w (hmSalt c q n)))⟩).get! i.val) := by
  have hc : hmCalls c nm q n =
      [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
        (rowPfx c q, ((rowHidden c nm n).getD []).map HmNets.Lin.var, hmDigest c q n)] := rfl
  rw [hc] at hT
  simp only [List.forall_mem_cons] at hT
  obtain ⟨h1, h2, -⟩ := hT
  have h1' : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512) := h1
  have h2' : FlockDeploy.sha512 (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)) =
      readForms w (hmDigest c q n) := h2
  have e1 := saltCheck_of h1'
  have e2 := rowCheck_of hpfx h2'
  exact ⟨e1, e2⟩

end RecDeploy.HmOpen
