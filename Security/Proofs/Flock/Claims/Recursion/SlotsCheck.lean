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

end RecDeploy
