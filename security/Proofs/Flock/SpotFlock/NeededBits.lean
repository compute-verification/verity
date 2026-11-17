module

public import Proofs.Flock.SpotFlock.AddressOf

/-! Every gate a checked unit needs is one of its own unit's bits (`NeededBits`), stated once for the five address
facts. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs)

variable (h : Held)

def BitOf (t k i : ℕ) (rp : Spot.Bits.RowPos) : Prop :=
  rp.leaf = k ∧
    (((sourceBits h.ms).find? (·.1 == i) = some (i, t, rp) ∧ (bitsProgs h.ms).gates[i]? = some .input) ∨
      ((sourceBits h.ms).find? (·.1 == i) = none ∧ (writtenBits h.ms).find? (·.1 == i) = some (i, t, rp) ∧
        i ∈ (bitsProgs h.ms).outs ∧ (bitsProgs h.ms).gates[i]? ≠ some .input))

def NeededBits : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    ∃ rp, BitOf h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) i rp

variable {h}

theorem bitAtGate_of_bitOf {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp) :
    bitAtGate h i = some (t, rp) := by
  rcases hb.2 with ⟨hs, -⟩ | ⟨hs, hw, -⟩
  · simp only [bitAtGate, hs]
  · simp only [bitAtGate, hs, hw, Option.map_some]

theorem addressAt_of_bitOf {t k i tw : ℕ} {rp rq : Spot.Bits.RowPos} (hb : BitOf h t k i rp)
    (hw : bitAtGate h (wireMap h i) = some (tw, rq)) :
    addressAt h i = ⟨fun _ => portNamed h t rp.row, placeAt h tw rq, slotAt h tw rq⟩ := by
  simp only [addressAt, bitAtGate_of_bitOf hb, hw]

theorem addressAt_of_fixed {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp) (hi : wireMap h i = i) :
    addressAt h i = ⟨fun _ => portNamed h t rp.row, placeAt h t rp, slotAt h t rp⟩ :=
  addressAt_of_bitOf hb (by rw [hi]; exact bitAtGate_of_bitOf hb)

theorem wireMap_of_written {t k i : ℕ} {rp : Spot.Bits.RowPos} (hb : BitOf h t k i rp)
    (hg : (bitsProgs h.ms).gates[i]? ≠ some .input) : wireMap h i = i := by
  rcases hb.2 with ⟨-, hin⟩ | ⟨hs, -⟩
  · exact absurd hin hg
  · exact wireMap_of_not_source h hs

end SpotFlock
