module

public import Proofs.Flock.SpotFlock.Honest.Rows
public import Proofs.Flock.SpotFlock.AddressFacts.Copy

/-!
# `CopyFact` at the one-template instance

The address facts' copy fact from the registration's copy condition (`copyFact_of_ref`), over the units whose proof
blocks need each gate at an accepted run: the row fact gives `Counted` (`rowFact_held`, from `RowGoal`), and at the
instance the copy condition is one on `R`'s circuit (`CopiesHeld`): at each bit of the one member, the column `pickAny`
gives the gate copies the bit the address's port holds at its place (`CopiesRef`), which a check of the member decides.
The hypotheses are `RowGoal`'s, `RefPorts R`, `NeededBits` and `CopiesHeld`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **`R`'s circuit copies each bit of the one member where the layout reads it**: at every bit `i` of the member's
unit `k` at row position `rp`, the column `pickAny` gives `i` copies bit `rp`'s place of the member's first port of
`rp`'s row (`CopiesRef`). -/
def CopiesHeld : Prop :=
  ∀ k i rp, BitOf (held R B desc q c₀) 0 k i rp →
    CopiesRef (stmtOf R.x) (pickAny (held R B desc q c₀) 0 k i) (portNamed (held R B desc q c₀) 0 rp.row)
      (placeAt (held R B desc q c₀) 0 rp).2

/-- **`CopyFact` at the instance.** -/
theorem copyFact_held
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) (hC : CopiesHeld R B desc q c₀) :
    CopyFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) :=
  copyFact_of_ref _ (rowFact_held R B desc q c₀ hG hR hN)
    (U := fun i u => ∃ r : CheckedRun (held R B desc q c₀).stage,
      r.Ok (held R B desc q c₀) ∧ r.u = u ∧ r.s.prog.Needs r.b i)
    (fun r hr i hi => ⟨r, hr, rfl, hi⟩)
    (fun i => by
      intro u hu
      obtain ⟨r, hr, rfl, hi⟩ := hu
      obtain ⟨rp, hb⟩ := hN r hr i hi
      have ht0 := (templateOf_held R B desc q c₀ hr).1
      rw [address_held R B desc q c₀ (bitAtGate_of_bitOf hb), ht0]
      rw [ht0] at hb
      exact ⟨(stmtOf R.x, #[0, 1]), rfl, hC _ _ _ hb⟩)

end SpotFlock.Honest
