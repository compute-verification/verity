module

public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordTakes

/-!
# The circuit's θ and blocks, asked

Statements only, each to be proved in `RecordCircuit.lean`. A statement here is the form the assembly of
`Properties.Record.LaidOut` reads; a proof may take fewer hypotheses than it lists, never more.

C1 gives the two things the registered circuit takes beside the wired circuit `D`: θ, the gates of the rows held,
each an input gate, and the division of `D`'s computed gates into blocks, each gate's block its unit's slot. The
assembly registers `Verity.RegisteredCircuit.ofPartition D θ _ n U` of them. C2 is C1's hypothesis on the slots, on a
record `read` accepts: every unit has a slot, below the record's number of slots.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row)
open Lean (Json)

/-- **C1. θ and the blocks of a wired circuit.** Where `D` is the wired circuit (`LaysOut`), every unit's rows are
its own input gates (`OwnRows`), every computed gate lies in a unit (`GatesOwn`) and every unit has a slot below `n`:
some θ of input gates is the gates of the rows `held` holds (`Names`), and some division of `D`'s computed gates into
`n` blocks gives each gate its unit's slot (`Divides`). -/
theorem names_divides {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ} (hL : LaysOut P rs W S D at_) (hO : OwnRows P rs W)
    (hG : GatesOwn P rs) (held : Row → Bool) (slot : String → ℕ → Option ℕ) (n : ℕ)
    (hslot : ∀ tid u sc, UnitAt P rs tid u sc → ∃ s, slot tid u = some s ∧ s < n) :
    ∃ θ : Finset (Fin D.N), (∀ g ∈ θ, D.isInput g = true) ∧ ∃ U : Verity.Partition D n,
      Names P rs W held at_ θ ∧ Divides P rs slot at_ U := by
  sorry

/-- **C2. Every unit has a slot**, below the record's number of slots, on a record `read` accepts. -/
theorem slot_of_unit {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
    {ts : Array String} {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt c.f.P rs tid u sc) : ∃ s, c.slot tid u = some s ∧ s < c.S := by
  sorry

end NciCheck.Properties.Record.SecurityProofs
