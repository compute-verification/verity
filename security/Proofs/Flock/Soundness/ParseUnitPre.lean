module

public import Aesop.BuiltinRules
public import Flock.HmRow
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.ContinuousFunctionalCalculus
import Mathlib.Tactic.NormNum.GCD
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.ExecCircuit

/-! A unit net given ahead of the file (`pre := #[("unit", n)]`, as `HmOut.parseTmpl` reads a templated file) is the
circuit's unit. -/

@[expose] public section

open Flock

namespace FlockSoundness

theorem unit_of_parsePre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {n : Net}
    {c : Circuit} (h : HmRow.parse tags bytes tables (pre := #[("unit", n)]) = .ok c) :
    c.unitNet = 0 ∧ c.nets.getD 0 default = ("unit", n) ∧ c.unit = n := by
  obtain ⟨-, ⟨more, hm, -⟩, hu, -, -⟩ := parse_factsPre_nets h
  have hm' : c.nets.toList = ("unit", n) :: more := hm
  have h0 : c.nets.findIdx? (·.1 == "unit") = some 0 := by
    rw [← Array.toArray_toList (xs := c.nets), List.findIdx?_toArray, hm']
    simp [List.findIdx?_cons]
  have hu0 : c.unitNet = 0 := Option.some.inj (hu.symm.trans h0)
  have hg : c.nets.getD 0 default = ("unit", n) := by
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hm']
    rfl
  refine ⟨hu0, hg, ?_⟩
  show (c.nets.getD c.unitNet default).2 = n
  rw [hu0, hg]

end FlockSoundness
