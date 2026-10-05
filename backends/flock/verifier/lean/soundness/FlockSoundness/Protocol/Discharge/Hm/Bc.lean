/-
A row's `b ‖ c` in a satisfying witness of the untyped statement `Stmt.setupH` accepted (`row_bc_setupH`): the commit
string of the row and salt its slots read. Δ's one copy at each `cv` and `pad` input bit (`inBit_copyL`) gives
`port_commit_rows` its four facts (`row_copies_flat`); the constant bits read the pin (`pin_bit`) and the unit range's
forced-zero row (`zero_bit`).
-/
import FlockSoundness.Protocol.Discharge.Hm.Flat

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)

end Bc

/-- A unit that returns the outputs has a slot per VU at least: the check's power of two of unit slots. -/
theorem upv_pos {c : Circuit} (C : CheckFacts c) (ho : c.outNet = c.unitNet) : 0 < c.upv := by
  have h := C.unit_pow2 ho
  rw [C.unit_range.2] at h
  unfold isPow2 at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact Nat.pos_of_mul_pos_left h.1

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

end FlockSoundness.Discharge.Hm
