module

public import Proofs.Flock.SpotFlock.AddressOf

/-!
# Every gate a checked unit needs is one of its own unit's bits

The address map (`addressOf`) gives an address to a gate that has a commitment position: a source bit or a written
bit of a member (`bitAtGate`). The five facts are asked at every gate a checked unit's proof block needs side by
side, so each rests on what such a gate is. This file states it once, for all five:

* `BitOf h t k i rp`: gate `i` is a bit of member `t`'s unit `k`, at commitment position `rp`, the unit's own leaf.
  It is a source bit, an input gate of the members' programs, or a written bit, a gate the members' programs return
  that is no input; and the first entry at its position, among the source bits and then the written bits, is that
  one, so `bitAtGate` finds it.
* `NeededBits h`: every gate an accepted run's checked unit needs side by side is a bit of that unit.

A unit's proof block holds its rows' gates and the input gates it reads; each unit reads its own source gates, the
constant and the zero, and no other unit's gates. So a gate the block needs is one of its own that its boundary
commits: an input gate it reads, which has a commitment position where every input a row reads is a row port's bit,
or a gate the member's program returns, a returned output leaf's bit.

`bitAtGate_of_bitOf`, `addressAt_of_bitOf`: at such a gate the address map's port is the first port of the bit's row
in its member's circuit.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs)

variable (h : Held)

/-- **Gate `i` is a bit of member `t`'s unit `k`**, at commitment position `rp`, the unit's own leaf: a source bit,
the first entry at `i` among the source bits and an input gate of the members' programs; or a written bit, where
no source bit is at `i`, the first entry at `i` among the written bits, a gate the members' programs return that is
no input. -/
def BitOf (t k i : ℕ) (rp : Spot.Bits.RowPos) : Prop :=
  rp.leaf = k ∧
    (((sourceBits h.ms).find? (·.1 == i) = some (i, t, rp) ∧ (bitsProgs h.ms).gates[i]? = some .input) ∨
      ((sourceBits h.ms).find? (·.1 == i) = none ∧ (writtenBits h.ms).find? (·.1 == i) = some (i, t, rp) ∧
        i ∈ (bitsProgs h.ms).outs ∧ (bitsProgs h.ms).gates[i]? ≠ some .input))

/-- **Every gate a checked unit needs is one of its own unit's bits**: at every accepted run's checked unit, each
gate its proof block needs side by side is a bit of the unit's member, at the unit's place among the member's units
(`BitOf`). -/
def NeededBits : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    ∃ rp, BitOf h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) i rp

variable {h}

/-- A unit's bit is the commitment position the address map finds at its gate. -/
theorem bitAtGate_of_bitOf {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp) :
    bitAtGate h i = some (t, rp) := by
  rcases hb.2 with ⟨hs, -⟩ | ⟨hs, hw, -⟩
  · simp only [bitAtGate, hs]
  · simp only [bitAtGate, hs, hw, Option.map_some]

/-- **The address at a unit's bit**: its port is the first port of the bit's row in its member's circuit, and its
place and slot are those of the gate the wiring map sends it to, where that gate has a commitment position. -/
theorem addressAt_of_bitOf {t k i tw : ℕ} {rp rq : Spot.Bits.RowPos} (hb : BitOf h t k i rp)
    (hw : bitAtGate h (wireMap h i) = some (tw, rq)) :
    addressAt h i = ⟨fun _ => portNamed h t rp.row, placeAt h tw rq, slotAt h tw rq⟩ := by
  simp only [addressAt, bitAtGate_of_bitOf hb, hw]

/-- **A unit's bit the wiring map leaves alone has its own place and slot.** -/
theorem addressAt_of_fixed {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp) (hi : wireMap h i = i) :
    addressAt h i = ⟨fun _ => portNamed h t rp.row, placeAt h t rp, slotAt h t rp⟩ :=
  addressAt_of_bitOf hb (by rw [hi]; exact bitAtGate_of_bitOf hb)

/-- A written bit is no source bit, so the wiring map leaves it alone. -/
theorem wireMap_of_written {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp)
    (hg : (bitsProgs h.ms).gates[i]? ≠ some .input) : wireMap h i = i := by
  rcases hb.2 with ⟨-, hin⟩ | ⟨hs, -⟩
  · exact absurd hin hg
  · exact wireMap_of_not_source h hs

end SpotFlock
