module

public import Proofs.Flock.SpotFlock.AddressFacts

/-!
# The address map of a held code

`addressOf h`: where each gate of the members' programs side by side has its value, read off what the verifier holds
(`Held`) and nothing of a run. It is the map the five facts (`AddressFacts`) are to be proved at, so that
`addressesConnected_of_facts` gives the address premise.

A gate with a commitment position (`bitAtGate`: a source bit's or a written bit's member and row position) has

* **the port** that holds it in its member's sessions: the first port of the member's circuit named like the
  position's row (`portNamed`), the same for every unit;
* **the place and the slot** of the gate the wiring map sends it to (`wireMap`), so a wired source bit and the
  written bit it reads are at one place and slot. The place is the position's bit in a row of the port's width
  (`placeAt`). The slot (`slotAt`) is, at a port a link names, the boundary port the link names, at the row the
  position's unit reads there (its replay block's rows from `ρ · per`, then its cell position's row among them); at
  any other port, the member's port of the interior record, at the unit's place in its cell on a stride of the
  member's units a replay block, sized.

Every other gate has a default address, which no fact is asked of: a proof block needs only gates with a
commitment position.

The map is the default at every gate unless the wiring map sends each gate it moves to a gate it leaves alone
(`wireMap_idem`): then a gate and the gate it is sent to share their place and slot whatever the held code is
(`addressOf_shares`), and where the design evaluates and the wires' writers come first the wiring map is such a map.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Member `t`'s circuit**: its reference statement's statement circuit (`Held.ms`). -/
def memberCircuit (t : ℕ) : _root_.Flock.Circuit := stmtOf (h.xS₀ t)

/-- **The first port of member `t`'s circuit named `name`**; port `0` where none is. -/
def portNamed (t : ℕ) (name : String) : ℕ :=
  ((List.range (memberCircuit h t).ports.size).find? fun q => (memberCircuit h t).ports[q]!.name == name).getD 0

/-- **A gate's commitment position**: its member and its row position at the member's own units, as a source bit
(`sourceBits`), else as a written bit (`writtenBits`); none for a gate that is neither. -/
def bitAtGate (i : ℕ) : Option (ℕ × Spot.Bits.RowPos) :=
  match (sourceBits h.ms).find? (·.1 == i) with
  | some s => some s.2
  | none => ((writtenBits h.ms).find? (·.1 == i)).map (·.2)

/-- **The place of member `t`'s row position `r`**: bit `r.bit` of a row of its port's `16 w` bits; a one-bit row
where the bit is past the row. -/
def placeAt (t : ℕ) (r : Spot.Bits.RowPos) : (w : ℕ) × Fin w :=
  if hb : r.bit < 16 * ((memberCircuit h t).ports[portNamed h t r.row]!).words then
    ⟨16 * ((memberCircuit h t).ports[portNamed h t r.row]!).words, ⟨r.bit, hb⟩⟩
  else ⟨1, 0⟩

/-- **The row schema template `t`'s reference registered reads give port `port`** (`Held.refReads`, the key
`row_schema`); `hm96-sha512/row/v1`, the default, where they name none. -/
def refSchema (t : ℕ) (port : String) : String :=
  match Spot.getD (Spot.getD (Spot.getD (h.refReads t) "ports") port) _root_.Flock.Registered.SCHEMA_KEY with
  | .str s => s
  | _ => _root_.Flock.HmRow.ROW_SCHEMA

/-- **The slot of member `t`'s row position `r`**, a gate the wiring map leaves alone. At a port a link names, an
input a replay block reads or an output it returns: the boundary port the link names, at the reference reads' schema,
at the row the position's unit reads there, its replay block's `per` rows from `ρ · per` and its cell position's
row among them (`heldRows`). At any other port, an output only the interior record holds: the member's port of the
interior record, at the port's published schema, at the unit's place in its cell on a stride of the member's units
a replay block, sized. -/
def slotAt (t : ℕ) (r : Spot.Bits.RowPos) : Slot :=
  match heldLink h (h.names.getD t "") r.row with
  | some l =>
    { key := .port l.value
      schema := refSchema h t r.row
      pos := r.leaf / h.reg.ns.getD t 0 * ((heldRows h (h.names.getD t "") r.row).map (·.per)).getD 0 +
        ((heldRows h (h.names.getD t "") r.row).bind fun rows =>
          Spot.SecondStage.rowAt rows.runs (r.leaf % h.reg.ns.getD t 0)).getD 0 }
  | none =>
    { key := .interior (h.names.getD t "") r.row
      schema := _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t r.row)
      pos := r.leaf % h.reg.ns.getD t 0
      stride := h.reg.ns.getD t 0
      sized := true }

/-- **The address no fact is asked of.** -/
def noAddress : Address := ⟨fun _ => 0, ⟨1, 0⟩, { key := .reg "", schema := "", pos := 0 }⟩

/-- **Gate `i`'s address, from the commitment positions**: its own position's port, and the place and slot of the
position of the gate the wiring map sends it to; `noAddress` where either gate has no position. -/
def addressAt (i : ℕ) : Address :=
  match bitAtGate h i, bitAtGate h (wireMap h i) with
  | some (t, r), some (tw, rw) => ⟨fun _ => portNamed h t r.row, placeAt h tw rw, slotAt h tw rw⟩
  | _, _ => noAddress

open Classical in
/-- **The address map** (the module's docstring): `addressAt` where the wiring map sends each gate it moves to a
gate it leaves alone, `noAddress` at every gate otherwise. -/
noncomputable def addressOf (i : ℕ) : Address :=
  if ∀ j, wireMap h (wireMap h j) = wireMap h j then addressAt h i else noAddress

/-- **Where the wiring map sends each gate it moves to a gate it leaves alone, the address map is `addressAt`.** -/
theorem addressOf_eq (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (i : ℕ) :
    addressOf h i = addressAt h i := by
  classical
  exact if_pos hω

/-- **A gate and the gate the wiring map sends it to share their place and slot**, at every held code. -/
theorem addressOf_shares (i : ℕ) :
    (addressOf h i).place = (addressOf h (wireMap h i)).place ∧
      (addressOf h i).slot = (addressOf h (wireMap h i)).slot := by
  classical
  by_cases hω : ∀ j, wireMap h (wireMap h j) = wireMap h j
  · rw [addressOf_eq h hω, addressOf_eq h hω]
    unfold addressAt
    rw [hω i]
    cases hi : bitAtGate h i with
    | none =>
      have hs : (sourceBits h.ms).find? (·.1 == i) = none := by
        unfold bitAtGate at hi
        cases hf : (sourceBits h.ms).find? (·.1 == i) with
        | none => rfl
        | some s => simp [hf] at hi
      rw [wireMap_of_not_source h hs, hi]
      exact ⟨rfl, rfl⟩
    | some a =>
      obtain ⟨t, r⟩ := a
      cases hg : bitAtGate h (wireMap h i) with
      | none => exact ⟨rfl, rfl⟩
      | some b =>
        obtain ⟨tw, rw⟩ := b
        exact ⟨rfl, rfl⟩
  · have e : ∀ j, addressOf h j = noAddress := fun j => if_neg hω
    rw [e i, e (wireMap h i)]
    exact ⟨rfl, rfl⟩

end SpotFlock
