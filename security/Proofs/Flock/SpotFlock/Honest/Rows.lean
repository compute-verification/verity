module

public import Proofs.Flock.SpotFlock.Honest.Entry
public import Proofs.Flock.SpotFlock.AddressFacts.RowGive

/-!
# The row fact's premises at the one-template instance

What `AddressRow.RowGoal` asks besides the roots (`rootsHold_held`), at the instance. At every gate an accepted run's
checked unit needs, a link names the gate's row and a registered read is named like its port (`named_held`), so:

* the read is a boundary link's (`kind_held`): the interior registration registers no value;
* the address's slot is the linked shape, the boundary port's row `ρ·per + x` the unit reads (`shapeFact_held`): the
  gate's leaf is the unit's place among its member's units (`BitOf`), so `slotAt`'s row is `linkShape`'s;
* no needed gate goes unregistered (`outputsAt_held`), and no proof block is always checked (`always_held`).

So the row fact at the instance is `RowGoal`'s conclusion (`rowFact_held`). The hypotheses are `RefPorts R` and
`NeededBits`.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **A needed gate's registered read at the instance**: the gate is a bit of the unit's member at the unit's place,
a link names its row, the address's slot is the slot of that position, every registered read named like the address's
port is of the row, and one is. -/
theorem named_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀))
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) {i : ℕ}
    (hi : r.s.prog.Needs r.b i) :
    ∃ rp l, BitOf (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u)
        (r.u - Properties.Spot.unitBase (held R B desc q c₀).ms (templateOf (held R B desc q c₀).ms r.u)) i rp ∧
      heldLink (held R B desc q c₀)
        ((held R B desc q c₀).names.getD (templateOf (held R B desc q c₀).ms r.u) "") rp.row = some l ∧
      (addressOf (held R B desc q c₀) i).slot =
        slotAt (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) rp ∧
      (∀ reg, regNamed r.ins r.sts ((addressOf (held R B desc q c₀) i).port r.u) = some reg → reg.name = rp.row) ∧
      regNamed r.ins r.sts ((addressOf (held R B desc q c₀) i).port r.u) ≠ none := by
  obtain ⟨rp, hb⟩ := hN r hr i hi
  obtain ⟨ht0, hports⟩ := templateOf_held R B desc q c₀ hr
  obtain ⟨hrowP, p, hp, hpn, own, reg₀, hown, hregm, hrn⟩ := bit_held R B desc q c₀ hR hr hb
  have haddr := address_held R B desc q c₀ (bitAtGate_of_bitOf hb)
  have hport : (addressOf (held R B desc q c₀) i).port r.u = p := by rw [haddr, hpn]
  obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 (heldLink_held R B desc q c₀ rp.row hrowP)
  refine ⟨rp, l, hb, ?_, by rw [haddr], fun reg hreg => ?_, fun hnone => ?_⟩
  · rw [ht0]
    exact hl
  · rw [hport] at hreg
    exact regNamed_name R B desc q c₀ hr hp hreg
  · rw [hport] at hnone
    unfold regNamed at hnone
    rw [hown, Option.bind_some, hports] at hnone
    apply Array.find?_eq_none.1 hnone reg₀ hregm
    have hq := List.find?_some hp
    simp only [beq_iff_eq] at hq ⊢
    rw [hrn, hq]

/-- **A needed gate's registered read is a boundary link's**: a link names its row, and the interior registration
registers no value. -/
theorem kind_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀))
    {r : CheckedRun (held R B desc q c₀).stage} (hr : r.Ok (held R B desc q c₀)) {i : ℕ}
    (hi : r.s.prog.Needs r.b i) {reg : _root_.Flock.Registered.Port}
    (hreg : regNamed r.ins r.sts ((addressOf (held R B desc q c₀) i).port r.u) = some reg) :
    AddressRow.kindHeld (held R B desc q c₀) (templateOf (held R B desc q c₀).ms r.u) reg.name reg.value =
      .boundaryLink := by
  obtain ⟨rp, l, -, hl, -, hname, -⟩ := named_held R B desc q c₀ hR hN hr hi
  have hl' : Spot.SecondStage.linkOf (AddressRow.heldOwn (held R B desc q c₀))
      ((held R B desc q c₀).names.getD (templateOf (held R B desc q c₀).ms r.u) "") reg.name = some l := by
    rw [hname reg hreg]
    exact hl
  unfold AddressRow.kindHeld
  simp only [hl']
  rfl

/-- **`NeededKinds` at the instance**: every needed gate's registered read is a boundary link's. -/
theorem neededKinds_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    AddressRow.NeededKinds (held R B desc q c₀) (addressOf (held R B desc q c₀)) :=
  fun _ hr _ hi _ hreg => Or.inl (kind_held R B desc q c₀ hR hN hr hi hreg)

/-- **`ShapeFact` at the instance**: the address's slot at a needed gate is the boundary port's row the unit reads. -/
theorem shapeFact_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    AddressRow.ShapeFact (held R B desc q c₀) (addressOf (held R B desc q c₀)) := by
  intro r hr i hi
  obtain ⟨rp, l, hb, hl, hslot, hname, hne⟩ := named_held R B desc q c₀ hR hN hr hi
  refine ⟨fun reg hreg => ⟨fun _ => ?_, fun hk => ?_⟩, fun hnone => absurd hnone hne⟩
  · rw [hslot, hname reg hreg]
    unfold AddressRow.linkShape AddressRow.rowShape AddressRow.RowShape.Fits slotAt
    rw [hl, hb.1]
    exact ⟨rfl, rfl, rfl, rfl⟩
  · rw [kind_held R B desc q c₀ hR hN hr hi hreg] at hk
    cases hk

/-- **`OutputsAt` at the instance**: no needed gate goes without a registered read named like its port. -/
theorem outputsAt_held (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    AddressRow.OutputsAt (held R B desc q c₀) (addressOf (held R B desc q c₀)) :=
  fun _ hr _ hi hnone => by
    obtain ⟨-, -, -, -, -, -, hne⟩ := named_held R B desc q c₀ hR hN hr hi
    exact absurd hnone hne

/-- **`RowFact` at the instance, from what gives it** (`AddressRow.RowGoal`): its premises all hold here. -/
theorem rowFact_held
    (hG : AddressRow.RowGoal (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)))
    (hR : RefPorts R) (hN : NeededBits (held R B desc q c₀)) :
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  hG (always_held R B desc q c₀) (neededKinds_held R B desc q c₀ hR hN) (shapeFact_held R B desc q c₀ hR hN)
    (outputsAt_held R B desc q c₀ hR hN) (rootsHold_held R B desc q c₀ hR hN)

end SpotFlock.Honest
