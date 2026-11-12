module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressOf
public import Proofs.Flock.SpotFlock.AddressFactsGive

/-!
# The address premise at the one-template registration, through the five facts

`SpotFlock.Honest.held R B desc q c₀` has no wires, so its wiring map is the identity (`wireMap_held`) and the address
map the five facts are proved at, `addressOf`, is each gate's own position's address (`addressOf_held`,
`addressAt_held`): its own port, place and slot. `Five R B desc q c₀` is the five facts at that map
(`AddressFacts`), and `addressesConnected_held` gives the address premise from them through
`addressesConnected_of_facts`; with an accepted run they give `HonestExists` (`honest_of_five`).

What the wiring makes trivial here:
* one tree a gate's first half, a gate and the gate it is sent to sharing place and slot, is `addressOf_shares`;
* no proof block is always checked (`always_held`), and there is no wire (`wires_held`), so the facts' conditions on a
  wire's reader and writer ask nothing.

## What this does not show

The five facts themselves at the instance. Each is about every accepted run's checked unit and the gates its proof
block needs, so each rests on what an accepted run gives: that a needed gate is one of the block's own input bits or
returned output bits (`NeededBits`), the session's ports being the member's (the width), the member's copy rows (the
copy), the row and entry of each port's read, and the roots of a linked port's read being the statement's where the
interior record registers no value of its name.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The address map is `addressAt`**: the wiring map is the identity, so it sends no gate it moves anywhere. -/
theorem addressOf_held : addressOf (held R B desc q c₀) = addressAt (held R B desc q c₀) :=
  funext (addressOf_eq _ fun j => by rw [wireMap_held]; rfl)

/-- **A gate's address is its own position's**: its port, place and slot at its own member and row position;
`noAddress` at a gate with none. -/
theorem addressAt_held (i : ℕ) :
    addressAt (held R B desc q c₀) i =
      match bitAtGate (held R B desc q c₀) i with
      | some (t, r) => ⟨fun _ => portNamed (held R B desc q c₀) t r.row, placeAt (held R B desc q c₀) t r,
          slotAt (held R B desc q c₀) t r⟩
      | none => noAddress := by
  unfold addressAt
  rw [wireMap_held]
  simp only [id]
  cases hb : bitAtGate (held R B desc q c₀) i with
  | none => rfl
  | some a =>
    obtain ⟨t, r⟩ := a
    rfl

/-- **No proof block is always checked.** -/
theorem always_held (v : ℕ) : (held R B desc q c₀).reg.always v = false := rfl

/-- **There is no wire.** -/
theorem wires_held : (held R B desc q c₀).wires.getD #[] = #[] := rfl

/-- **One tree a gate's first half**: a gate and the gate the wiring map sends it to share place and slot. -/
theorem shares_held (i : ℕ) :
    (addressOf (held R B desc q c₀) i).place =
        (addressOf (held R B desc q c₀) (wireMap (held R B desc q c₀) i)).place ∧
      (addressOf (held R B desc q c₀) i).slot =
        (addressOf (held R B desc q c₀) (wireMap (held R B desc q c₀) i)).slot :=
  addressOf_shares _ i

/-- **The five facts at the instance's address map**, at its wiring map and the gates the boundary record holds. -/
def Five : Prop :=
  WidthFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) ∧
    CopyFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) ∧
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) ∧
    EntryFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) ∧
    OneTreeFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))

/-- **The address premise at the instance, through the five facts.** -/
theorem addressesConnected_held (hf : Five R B desc q c₀) : AddressesConnected (held R B desc q c₀) :=
  addressesConnected_of_facts _ fun _ => ⟨_, hf⟩

/-- **The five facts and an accepted run at some instance give `HonestExists`.** -/
theorem honest_of_five
    (hO : ∃ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code),
      Five R B desc q c₀ ∧ Accepts (held R B desc q c₀)) : HonestExists := by
  obtain ⟨R, B, desc, q, c₀, hf, hR⟩ := hO
  exact honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_held R B desc q c₀ hf, hR⟩

end SpotFlock.Honest
