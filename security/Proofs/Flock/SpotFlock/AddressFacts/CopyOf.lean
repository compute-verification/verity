module

public import Proofs.Flock.SpotFlock.AddressFacts.Copy
public import Proofs.Flock.SpotFlock.AddressFacts.WidthOf

/-!
# The copy fact at the address map

`copyFact_of_wiring`: `CopyFact` at the address map (`addressOf`), from what the entry and one tree a gate ask of the
held code (`Wiring`), each position's bit inside its port's row (`BitsInPorts`), the row fact for the units it
counts (`RowFact`), and one condition on the members' circuits alone, `CopiesAt`.

At an accepted run, a gate a checked unit needs is a bit `rp` of the unit, and the gate the wiring map sends it to is
at `rp`'s own bit (`target_of_wiring`): the address's port is the unit's first port of `rp`'s row and its place is
`rp`'s bit. So the address's copy condition (`CopyRef`) is the member's: the column `pickAny` gives the bit in the
unit's stacked rows copies that port's bit at that place (`CopiesAt`), which reads only the registration and no
wiring. The copy rows then carry it into the session's first table (`copy_of_copies`, `copies_of_ref`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Each unit's bit is copied at the column the layout reads it at**: at each bit `i` of member `t`'s unit `k`, at
row position `rp`, the column `pickAny` gives `i` in the unit's stacked rows copies, in the member's circuit, bit
`rp.bit` of the member's first port of `rp`'s row (`CopiesRef`). -/
def CopiesAt : Prop :=
  ∀ t k i rp, BitOf h t k i rp → ∃ m, h.ms[t]? = some m ∧
    CopiesRef m.1 (pickAny h t k i) (portNamed h t rp.row) rp.bit

variable {h}

namespace CheckedRun

variable {r : CheckedRun h.stage}

/-- **The copy at every gate a counted checked unit needs**, at the address map. -/
theorem copyAt_of_wiring (W : Wiring h) (hB : BitsInPorts h) (hC : CopiesAt h) (hr : r.Ok h) (hn : r.Counted h)
    {i : ℕ} (hi : r.s.prog.Needs r.b i) : r.CopyAt h (addressOf h i) i := by
  obtain ⟨hω, rp, tw, rq, hb, hw, hrb, hbit, -⟩ := target_of_wiring W hB hr hi
  obtain ⟨m, hm, hx⟩ := hC _ _ _ _ hb
  have haddr := (addressOf_eq h hω i).trans (addressAt_of_bitOf hb hw)
  have ep : (addressOf h i).port r.u = portNamed h (templateOf h.ms r.u) rp.row := by rw [haddr]
  have e : ((addressOf h i).place.2 : ℕ) = rp.bit := by
    rw [haddr]
    show ((placeAt h tw rq).2 : ℕ) = rp.bit
    rw [placeAt, dite_eq_left hbit]
    exact hrb
  have hx' : CopiesRef m.1 (r.pick h i) ((addressOf h i).port r.u) (addressOf h i).place.2 := by
    rw [ep, e]
    exact hx
  exact copy_of_copies hr (inBlock_of_counted hr hn) hm (copies_of_ref hr hm hx')

end CheckedRun

/-- **The copy, at the address map**: at every accepted run's checked unit, the bit the row of each needed gate's
address holds at its place, at the unit's instance, is the one the layout reads for the gate. -/
theorem copyFact_of_wiring {ω : ℕ → ℕ} {keep : ℕ → Bool} (W : Wiring h) (hB : BitsInPorts h)
    (hrow : RowFact h ω keep (addressOf h)) (hC : CopiesAt h) : CopyFact h (addressOf h) :=
  fun _ hr _ hi => CheckedRun.copyAt_of_wiring W hB hC hr (hrow _ hr _ hi).1 hi

end SpotFlock
