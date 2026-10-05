/-
A row's hashing across its slots: the row's bytes `rowPrefix words ‖ data` (`data` in 128-byte chunks), its in-circuit
compressions in `sha512x3` slots (the first from the prefix's midstate, each next one from the previous one's output, each
over one chunk of `data`), and its `hm96` slot (`cv` from the last compression, `pad` the padding block, the salt). When
every slot's rows hold and the slots' inputs carry what Δ copies into them (`HmRow.delta`: the midstate and the padding
block from the constants, the data bytes at `msgCol`, `HmRow.rowWires`' chain), the hm96 slot outputs the row's commit
string (`row_commit`).
-/
import Flock.HmRow
import FlockSoundness.Protocol.Binding.Leaf
import Mathlib.Data.List.GetD
import FlockSoundness.Protocol.Discharge.Hm.Hash

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## The sha512x3 slot on given inputs -/

/-! ## The row's bytes -/

/-! ## The chain over the row's blocks -/

theorem wordsBE_getElem! {b : ByteArray} {i : ℕ} (hi : i < 16) :
    (HmRow.wordsBE b 16)[i]! = wordL b.data.toList 0 i := by
  rw [HmRow.wordsBE, getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range, wordL]
  refine List.foldl_ext _ _ _ fun x j _ => ?_
  rw [get!_eq_getD, Nat.zero_add]

/-! ## Δ's bits -/

/-- Bit `q` of a word array as Δ reads a constant: bit `q % 64` of word `q / 64` (`HmRow.delta`). -/
def wordsBit (w : Array UInt64) (q : ℕ) : Bool := (w[q / 64]! >>> (q % 64).toUInt64) &&& 1 == 1

theorem wordsBit_eq (w : Array UInt64) {n q : ℕ} (hq : q < 64 * n) :
    wordsBit w q = wbit ((List.range n).map fun i => w[i]!.toBitVec) q := by
  rw [wordsBit, wbit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some]
  have h1 : (q % 64).toUInt64.toNat % 64 = q % 64 := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  have : ∀ x : UInt64, (x &&& 1 == 1) = x.toNat.testBit 0 := by
    intro x
    have hx : (x &&& 1).toNat = x.toNat % 2 := by rw [UInt64.toNat_and]; exact Nat.and_one_is_mod _
    rw [Nat.testBit_zero]
    by_cases h : x.toNat % 2 = 1
    · have : x &&& 1 = 1 := UInt64.toNat_inj.mp (by rw [hx, h]; rfl)
      simp [this, h]
    · have : x &&& 1 ≠ 1 := fun e => h (by rw [← hx, e]; rfl)
      simp [this, h]
  rw [this, UInt64.toNat_shiftRight, Nat.testBit_shiftRight, h1, Nat.add_zero]
  rfl

/-! ## The row -/

end FlockSoundness.Discharge.Hm
