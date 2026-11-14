module

public import Definitions.Core.Compiler
public import Properties.Flock.Circuit.CircuitOf

/-!
# The committed files as one circuit: one statement's, and a window's members side by side

How the committed files are read. One statement's files are read as v2 reads them (`FileRead`: its tags, circuit
file, lookup tables and public file). A window of several members is read a member at a time, and the window's
circuit is the members' circuits side by side, in the members' order, with no wire between two members: what a member
takes from another is an input of its own circuit, and what holds it to that member's gate is the same registered
value at the same root, never a wire.

* **One statement's read** (`FileRead`, `circuit`, `units`, `parts`, `consts`): the committed files as core's circuit,
  as the verifier reads them (`Flock.Bridge.circuitOfFile`), the public file's population, its units, and the key
  program's constants (`parsedConsts`).
* **A circuit as an execution block over another** (`Circuit.Extension.ofCircuit C D`): `D`'s gates after `C`'s, each
  with `D`'s width and input flag, reading `D`'s gates at their places after `C`'s (`Fin.natAddEmb`) and applying
  `D`'s operation to the values there.
* **Two circuits side by side** (`Circuit.beside C D`): `C` with `D` appended as an execution block
  (`Circuit.append`), so `C`'s gates come first (`Fin.castAdd`), then `D`'s (`Fin.natAdd`), each gate with its own
  circuit's width, input flag, wires and operation, and both circuits' outputs. `Partition.beside P Q` is `P`'s
  units, then `Q`'s, each gate in its own circuit's unit.
* **The window's read** (`WindowFiles`): its first member's files and the later members'. A window has a member, so a
  window of one member is that member's read and nothing else (`WindowFiles.one`).
* **Its circuit** (`WindowFiles.circuit`, `units`, `parts`, `consts`): the members' circuits side by side, their units
  summed, each member's units after the earlier members' (`Partition.beside`), and each member's constants at its own
  gates. At a window of one member each is that member's, by definition (`WindowFiles.one_circuit`, `one_units`;
  `FileRead.circuitWith`, `unitsWith`, `partsWith` and `constsWith` at no later member).
* **A member's place** (`WindowFiles.base k`): where the member at place `k` starts in the window's units, the earlier
  members' units summed.

`Vocab.read` (`Properties.Verity.Game`) reads `WindowFiles`, and `circuitOf`, `units`, `partsOf`, `Opens` and
`extract` are over `WindowFiles.circuit`, `units`, `parts` and `consts`: the claim at a window is the claim at one
statement, over the members' circuits side by side. The lemmas about `beside` are
`Proofs.Properties.Verity.WindowFiles`'s.
-/

@[expose] public section

namespace Verity

/-- **How v2 reads the committed files**: its statement's tags (derived from the circuit file), the circuit file, its
lookup tables and the public file. -/
structure FileRead where
  tags : Flock.Tags
  circuitFile : ByteArray
  tables : List (String × ByteArray)
  publicFile : ByteArray

namespace FileRead

/-- The committed files as core's circuit, as the verifier reads them (`Flock.Bridge.circuitOfFile`). -/
noncomputable def circuit (r : FileRead) : Verity.Circuit :=
  Flock.Bridge.circuitOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' number of units, the public file's population. -/
noncomputable def units (r : FileRead) : ℕ := Flock.Bridge.unitsOfFile r.tags r.circuitFile r.tables r.publicFile

/-- The committed files' units, as core's partition of `circuit`. -/
noncomputable def parts (r : FileRead) : Verity.Partition r.circuit r.units :=
  Flock.Bridge.partsOfFile r.tags r.circuitFile r.tables r.publicFile

end FileRead

/-- **A parse's constants**: the key program's constant input gate holds `1` and its zero gate `0`
(`FlockSoundness.Discharge.Placed.oneGate`, `zeroGate`), public circuit structure that no proof block opens. -/
noncomputable def parsedConsts : (o : Option Flock.Bridge.Parsed) → Fin (Flock.Bridge.Parsed.circuit o).N → Option Bool
  | some r, g =>
    if g.val = (FlockSoundness.Discharge.Placed.oneGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some true
    else if g.val =
        (FlockSoundness.Discharge.Placed.zeroGate r.n (FlockSoundness.Discharge.Layout.keyB r.D.unroll r.hU)).val
    then some false
    else none
  | none, _ => none

/-- The committed files' constants. -/
noncomputable def FileRead.consts (r : FileRead) : Fin r.circuit.N → Option Bool :=
  parsedConsts (Flock.Bridge.parsedOf r.tags r.circuitFile r.tables r.publicFile)

namespace Circuit

variable (C D : Circuit)

/-- **`D` as an execution block over `C`**: `D`'s gates after `C`'s, each with `D`'s width and input flag, reading
`D`'s gates at their places after `C`'s and applying `D`'s operation to the values there. It reads no gate of `C`. -/
@[reducible] def Extension.ofCircuit : C.Extension where
  n := D.N
  w := D.w
  isInput := D.isInput
  reads j := (D.reads j).map (Fin.natAddEmb C.N)
  reads_lt j a ha := by
    obtain ⟨b, hb, rfl⟩ := Finset.mem_map.1 ha
    exact Nat.add_lt_add_left (Fin.lt_def.1 (D.reads_lt j b hb)) C.N
  input_reads j h := by rw [D.input_reads j h, Finset.map_empty]
  op j v := D.op j fun a => (v (Fin.natAdd C.N a)).cast (Fin.append_right C.w D.w a)
  op_congr j v v' h := D.op_congr j _ _ fun a ha => by
    rw [h (Fin.natAdd C.N a) (Finset.mem_map_of_mem (Fin.natAddEmb C.N) ha)]
  outputs := D.outputs.map (Fin.natAddEmb C.N)

/-- **Two circuits side by side**: `C` with `D` appended as an execution block (`Circuit.append`,
`Extension.ofCircuit`): `C`'s gates, then `D`'s, each gate with its own circuit's width, input flag, wires and
operation, no wire between the two, and both circuits' outputs. -/
@[reducible] def beside : Circuit := C.append (.ofCircuit C D)

end Circuit

/-- **Two partitions side by side**: `P`'s units, then `Q`'s, over the two circuits side by side, each gate in its own
circuit's unit. -/
def Partition.beside {C D : Circuit} {n m : ℕ} (P : Partition C n) (Q : Partition D m) :
    Partition (C.beside D) (n + m) where
  unit := Fin.addCases (m := C.N) (n := D.N) (motive := fun _ => Option (Fin (n + m)))
    (fun g => (P.unit g).map (Fin.castAdd m)) fun g => (Q.unit g).map (Fin.natAdd n)
  input_iff := by
    show ∀ g : Fin (C.N + D.N), Fin.addCases (motive := fun _ => Option (Fin (n + m))) _ _ g = none ↔
      Fin.addCases (motive := fun _ => Bool) C.isInput D.isInput g = true
    intro g
    induction g using Fin.addCases with
    | left i =>
      rw [Fin.addCases_left, Fin.addCases_left, Option.map_eq_none_iff]
      exact P.input_iff i
    | right j =>
      rw [Fin.addCases_right, Fin.addCases_right, Option.map_eq_none_iff]
      exact Q.input_iff j

namespace FileRead

/-- **The circuits of `r`'s files and each of `rs`'s, side by side in order.** -/
noncomputable def circuitWith : FileRead → List FileRead → Verity.Circuit
  | r, [] => r.circuit
  | r, s :: rs => r.circuit.beside (circuitWith s rs)

/-- The units of `r`'s files and each of `rs`'s, summed. -/
noncomputable def unitsWith : FileRead → List FileRead → ℕ
  | r, [] => r.units
  | r, s :: rs => r.units + unitsWith s rs

/-- **The units of `r`'s files and each of `rs`'s, as core's partition of `circuitWith`**: `r`'s units, then each of
`rs`'s in order. -/
noncomputable def partsWith : (r : FileRead) → (rs : List FileRead) →
    Verity.Partition (circuitWith r rs) (unitsWith r rs)
  | r, [] => r.parts
  | r, s :: rs => r.parts.beside (partsWith s rs)

/-- The constants of `r`'s files and each of `rs`'s, each at its own circuit's gates. -/
noncomputable def constsWith : (r : FileRead) → (rs : List FileRead) → Fin (circuitWith r rs).N → Option Bool
  | r, [] => r.consts
  | r, s :: rs => Fin.addCases (m := r.circuit.N) (n := (circuitWith s rs).N) r.consts (constsWith s rs)

end FileRead

/-- **How v2 reads a window's committed files**: its first member's files and the later members', in the members'
order, each as v2 reads one statement's (`FileRead`). -/
structure WindowFiles where
  first : FileRead
  rest : List FileRead

namespace WindowFiles

variable (w : WindowFiles)

/-- The members' reads, in the members' order. -/
def members : List FileRead := w.first :: w.rest

/-- **The window's committed files as core's circuit**: the members' circuits side by side, in the members' order. -/
noncomputable def circuit : Verity.Circuit := w.first.circuitWith w.rest

/-- The window's number of units: the members' summed. -/
noncomputable def units : ℕ := w.first.unitsWith w.rest

/-- **The window's units, as core's partition of `circuit`**: each member's units, after the earlier members'. -/
noncomputable def parts : Verity.Partition w.circuit w.units := w.first.partsWith w.rest

/-- The window's constants: each member's, at its own gates. -/
noncomputable def consts : Fin w.circuit.N → Option Bool := w.first.constsWith w.rest

/-- **A window of one member**: that member's read. Its circuit, units, partition and constants are the member's, by
definition. -/
def one (r : FileRead) : WindowFiles := ⟨r, []⟩

/-- A window of one member has that member's circuit, by definition. -/
theorem one_circuit (r : FileRead) : (one r).circuit = r.circuit := rfl

/-- A window of one member has that member's units, by definition. -/
theorem one_units (r : FileRead) : (one r).units = r.units := rfl

/-- **Where the member at place `k` starts in the window's units**: the earlier members' units summed. -/
noncomputable def base (k : ℕ) : ℕ := ((w.members.take k).map FileRead.units).sum

end WindowFiles

end Verity
