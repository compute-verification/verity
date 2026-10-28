import Proofs.Flock.Soundness.ExecCircuit

/-!
# A unit given ahead of the file

`HmRow.parse … (pre := #[("unit", n)])` is how a templated file's circuit is read (`HmOut.parseTmpl`): the unit net `n`
comes first, ahead of the file's own nets, so it is the circuit's unit (`unit_of_parsePre`). The flat-parse consumers'
`…Pre` forms (`Layout.outShape_of_parsePre`, `Hidden.unitFacts_of_parsePre`) read their facts about the unit off `n`.
-/

open Flock

namespace FlockSoundness

/-- **The unit given ahead of the file is the circuit's unit**: the first net, named `unit`. -/
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
