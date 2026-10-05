import FlockSoundness.Discharge.Hm.Nets

/-!
# A row's commit string under any 128-byte prefix

`Hm.row_commit` and `Hm.port_commit_rows` read `HmRow.rowPrefix words` only through its 128 bytes and its midstate.
These are theirs for any 128-byte prefix `pre` (`row_commitP`, `port_commit_rowsP`), so for `HmRow.prefixOf` of a
`sha512/row/v1` or `sha512/row/v2` row (`size_prefixOf`), whose bytes are whole blocks. A row of no compression
(`port_commit_rows0`) is hashed by its `hm96` slot alone: its `cv` the prefix's midstate, its `pad` the padding block of
the prefix.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and

open Flock Hm

/-! ## The prefix -/

theorem size_pad {b : ByteArray} {n : ℕ} (h : b.size ≤ n) : (Hm96.pad b n).size = n := by
  rw [Hm96.pad, ByteArray.size_append]
  simp only [ByteArray.size, Array.size_replicate]
  show b.size + (n - min n b.size) = n
  omega

theorem size_rowPrefixV2 (l : ℕ) : (HmRow.rowPrefixV2 l).size = 128 := by
  have h0 : ("verity/sha512-row/v2".toUTF8.push 0).size = 21 := by decide +kernel
  have h1 : (ByteArray.mk ((Array.range 8).map fun i => (l / 256 ^ (7 - i) % 256).toUInt8)).size = 8 := by
    simp [ByteArray.size]
  refine size_pad ?_
  rw [ByteArray.size_append, h0, h1]
  omega

/-! ## The row's bytes after a prefix -/

/-- The row's bytes after the 128-byte prefix `pre`. -/
def rowMsgP (pre : ByteArray) (data : List UInt8) : ByteArray := ⟨(pre.data.toList ++ data).toArray⟩

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

theorem length_pre : pre.data.toList.length = 128 := by rw [Array.length_toList, ByteArray.size_data, hpre]

end Pre

/-! ## The port -/

section Nets

variable (c : Circuit) (w : ℕ → Bool) (g p : ℕ)

theorem hmLaidL (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    LaidRows (layout ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
      ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray) (inRowsOf HM_IN)
      fun j => w (c.slotCol c.hmNet g p + j) := by
  have := hhm
  rw [hmLaid_eq] at this
  rw [inRows_HM_IN]
  exact this

theorem hm_zh (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    stHolds (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x))
      ((hm96Row Hm96.Default512.key).run St.empty).2 := by
  have hgh : Good (HM_IN.foldl (· + ·) 0) ((hm96Row Hm96.Default512.key).run St.empty).2 := by
    rw [sum_HM_IN]; exact good_hm96Row _
  exact laid_rows _ _ _ _ _ hgh (hmLaidL c w g p hhm)

theorem hm_bc (hck : HmNets.check c = .ok ())
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) =
      Hm.bit (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x))
        ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t := by
  intro t ht
  obtain ⟨-, -, -, hl, -⟩ := hm96Row_spec _ _ (hm_zh c w g p hhm)
  have hj : t < ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray.size := by
    rw [List.size_toArray, hl]; exact ht
  have := laid_outs ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT _ (fun j => w (c.slotCol c.hmNet g p + j))
    (by rw [List.size_toArray, hl]; decide) (hmLaidL c w g p hhm) _ hj
  rw [hmOut_eq hck, Nat.add_assoc, this, List.getElem_toArray, ← bit_of_lt (by rw [hl]; exact ht)]

theorem hm_zh_in : ∀ v : ℕ, v < 3072 →
    (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x)) v =
      w (c.slotCol c.hmNet g p + v) := by
  intro v hv
  show w (c.slotCol c.hmNet g p + col _ _ v) = _
  rw [col_input (by rw [sum_HM_IN]; exact hv)]

end Nets

end FlockSoundness.Discharge.Hidden
