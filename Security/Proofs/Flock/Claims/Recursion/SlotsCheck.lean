import Proofs.Flock.Claims.Recursion.SlotsCommit

/-!
# The hm96 slots' commit strings at the outer relation

Toward `HmSlotsCommit` from `StagedPart` and `Flock.check`: at a staged part, the outer relation's model is satisfied at
the witness's table share (`layout_of_check`, `zOf`), and there `Hm.row_bc_flat` gives each held row's `b ‖ c` as the
commit string of its row and salt, at the model's columns (`bcCol`, `saltCol`, `Typed.msgCol`). This file maps those
columns onto the witness's hidden values that `SlotCommits` reads (`slotHid`, `rowHidden`).
-/

namespace RecDeploy

/-- **A block's bit at the table share is the witness's hidden value** at the share's interleaved address. -/
theorem blockBit_zOf (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (J t : ℕ) (o : Fin (2 ^ S.nbl))
    {y : ℕ} (hy : y < 2 ^ S.kLog) :
    FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf J w t S.m) o y =
      w.hid.getD ((o.val * 2 ^ S.kLog + y) * J + t) false := by
  rw [FlockSoundness.Discharge.Hm.blockBit_of_lt S _ o hy]
  exact Flock.Readable.bit_toZ _

/-- At one table, the share's address is the block's own: `slotHid`'s form. -/
theorem blockBit_zOf_one (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (o : Fin (2 ^ S.nbl))
    {y : ℕ} (hy : y < 2 ^ S.kLog) :
    FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) o y =
      w.hid.getD (o.val * 2 ^ S.kLog + y) false := by
  rw [blockBit_zOf S w 1 0 o hy, Nat.mul_one, Nat.add_zero]

/-- The hm96 net's output column, as `Hm.bcCol` reads it, is `hmOutCol`'s. -/
theorem hmOut_eq (c : Flock.Circuit) : FlockSoundness.Discharge.Hm.hmOut c = hmOutCol c := by
  unfold FlockSoundness.Discharge.Hm.hmOut hmOutCol
  rw [Array.getD_eq_getD_getElem?]
  cases (Flock.portCols (c.nets.getD c.hmNet default).2.outGroups)[0]? <;> rfl

/-- The hm96 net's `k`-th input column, as `Hm.saltCol` reads it, is `hmIn`'s. -/
theorem hmIn_eq (c : Flock.Circuit) (k : ℕ) : FlockSoundness.Discharge.Hm.hmIn c k = hmIn c k := by
  unfold FlockSoundness.Discharge.Hm.hmIn hmIn
  rw [Array.getD_eq_getD_getElem?]
  cases (Flock.portCols (c.nets.getD c.hmNet default).2.inGroups)[k]? <;> rfl

/-- **Port `q`'s `b ‖ c` at instance `n`, bit `t`**, as `Hm.row_bc_flat` reads it in block `n / g` (VU `n % g`), is
the hidden value at the slot's out column `t` (`hmCommit`'s). -/
theorem bc_bit (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (c : Flock.Circuit) {q n t : ℕ}
    (hk : 2 ^ S.kLog = 2 ^ c.kLog) (hb : n / c.g < 2 ^ S.nbl) (hy : hmBase c q n + (hmOutCol c + t) < 2 ^ c.kLog)
    (ht : t < 1024) :
    FlockSoundness.Discharge.Hm.bitL (FlockSoundness.Discharge.Hm.bytesOf
      (FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) ⟨n / c.g, hb⟩) 128
      (FlockSoundness.Discharge.Hm.bcCol c (n % c.g) q)) t =
      w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false := by
  have he : FlockSoundness.Discharge.Hm.bcCol c (n % c.g) q (t / 8) (t % 8) = hmBase c q n + (hmOutCol c + t) := by
    unfold FlockSoundness.Discharge.Hm.bcCol hmBase
    rw [hmOut_eq]
    omega
  rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by omega), he, blockBit_zOf_one S w _ (by rw [hk]; exact hy), hk]
  rfl

/-- **Port `q`'s salt at instance `n`, bit `j`**, as `Hm.row_bc_flat` reads it in block `n / g`, is the hidden value
at the slot's salt column `j` (`hmSalt`'s). -/
theorem salt_bit (S : FlockSoundness.Model.Statement) (w : Claims.Flock.Wit) (c : Flock.Circuit) {q n j : ℕ}
    (hk : 2 ^ S.kLog = 2 ^ c.kLog) (hb : n / c.g < 2 ^ S.nbl) (hy : hmBase c q n + (hmIn c 2 + j) < 2 ^ c.kLog)
    (hj : j < 1536) :
    FlockSoundness.Discharge.Hm.bitL (FlockSoundness.Discharge.Hm.bytesOf
      (FlockSoundness.Discharge.Hm.blockBit S (Flock.Readable.zOf 1 w 0 S.m) ⟨n / c.g, hb⟩) 192
      (FlockSoundness.Discharge.Hm.saltCol c (n % c.g) q)) j =
      w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false := by
  have he : FlockSoundness.Discharge.Hm.saltCol c (n % c.g) q (j / 8) (j % 8) = hmBase c q n + (hmIn c 2 + j) := by
    unfold FlockSoundness.Discharge.Hm.saltCol hmBase
    rw [hmIn_eq]
    omega
  rw [FlockSoundness.Discharge.Hm.bitL_bytesOf _ _ (by omega), he, blockBit_zOf_one S w _ (by rw [hk]; exact hy), hk]
  rfl

end RecDeploy
