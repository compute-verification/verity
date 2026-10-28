import Definitions.Core.Boolean
import Definitions.Core.Partition
import Flock.Desc
import Properties.Flock.Circuit.Outs

/-!
# The registered circuit file as core's circuit

The circuit a C-Flock session checks is defined from its circuit file: the descriptor `D` the verifier parses
(`Flock.Desc.parse`), the unit's rows built from it (`hU : Built D.unroll.unit`) and the population `n` (the public
file's `pub.n`) give the key program `keyProg D.unroll hU n`, whose circuit and partition (`keyCircuit`,
`keyPartition`) the session's game is stated over. Here that circuit is core's:

* `Verity.Circuit.ofAudit C`: a game circuit (`FlockSoundness.Audit.Circuit`) as core's (`Verity.Circuit`), the same
  gates, one bit each; `Verity.Partition.ofAudit P`, the same partition.
* `circuitOf D hU n := .ofAudit (keyCircuit D hU n)` and `partsOf D hU n := .ofAudit (keyPartition D hU n)`: the
  parsed file as core's circuit and its `n` units.
* `circuitOfFile tags cf tables pf`, `unitsOfFile` and `partsOfFile`: the same at the committed files themselves, the
  circuit file `cf` and the public file `pf` the prover sends before any coin, read as the verifier reads them
  (`parsedOf`), and the empty circuit (`Verity.Circuit.empty`) when they don't read.

The lemmas about them are `Proofs.Flock.Bridge.CircuitOf`'s.
-/

namespace Flock.Bridge

/-- A game gate as the Boolean profile's: the same gate. -/
def coreOp {N : ℕ} : FlockSoundness.Audit.Op N → FlockSoundness.Audit.Op N
  | .input => .input
  | .and a b => .and a b
  | .xor a b => .xor a b
  | .not a => .not a
  | .row a b => .row a b

theorem args_coreOp {N : ℕ} (o : FlockSoundness.Audit.Op N) : (coreOp o).args = o.args := by
  cases o <;> rfl

theorem isInput_coreOp {N : ℕ} (o : FlockSoundness.Audit.Op N) : (coreOp o).isInput = o.isInput := by
  cases o <;> rfl

end Flock.Bridge

namespace Verity

open Flock.Bridge (coreOp args_coreOp isInput_coreOp)

/-- **A game circuit as core's**: the Boolean profile's circuit of it, one bit a gate, written out so that its gates are
the game circuit's. -/
abbrev Circuit.ofAudit (C : FlockSoundness.Audit.Circuit) : Circuit where
  N := C.N
  w _ := 1
  isInput g := (coreOp (C.op g)).isInput
  reads g := (coreOp (C.op g)).args.toFinset
  reads_lt g a ha := C.topo g a (by rwa [List.mem_toFinset, args_coreOp] at ha)
  input_reads g h := by simp [FlockSoundness.Audit.Op.args_of_isInput h]
  op g v := BitVec.ofBool ((coreOp (C.op g)).apply fun a => (v a).getLsbD 0)
  op_congr g v v' h := by
    rw [FlockSoundness.Audit.Op.apply_congr (coreOp (C.op g)) fun x hx => by rw [h x (List.mem_toFinset.2 hx)]]
  outputs := C.outputs

theorem Circuit.ofAudit_isInput {C : FlockSoundness.Audit.Circuit} (g : Fin C.N) :
    (Circuit.ofAudit C).isInput g = (C.op g).isInput :=
  isInput_coreOp _

/-- A game partition as core's: the same units. -/
def Partition.ofAudit {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) :
    Partition (Circuit.ofAudit C) n where
  unit := P.unit
  input_iff g := (P.input_iff g).trans (by rw [Circuit.ofAudit_isInput])

/-- The circuit with no gates: what files that don't parse register. -/
def Circuit.empty : Circuit where
  N := 0
  w := Fin.elim0
  isInput := Fin.elim0
  reads := Fin.elim0
  reads_lt g := g.elim0
  input_reads g := g.elim0
  op g := g.elim0
  op_congr g := g.elim0
  outputs := ∅

/-- The empty circuit's partition into no units. -/
def Partition.empty : Partition Circuit.empty 0 where
  unit := Fin.elim0
  input_iff g := g.elim0

end Verity

namespace Flock.Bridge

/-! ## The parsed file's circuit -/

/-- The key program's circuit at the descriptor `D`, its unit's rows `hU` and the population `n`. -/
noncomputable abbrev keyCircuit (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    FlockSoundness.Audit.Circuit :=
  (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).circuit (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)

/-- The key program's partition into its `n` units. -/
noncomputable abbrev keyPartition (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    FlockSoundness.Audit.Partition (keyCircuit D hU n) n :=
  (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
    (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)

/-- **The parsed circuit file as core's circuit**: the key program of the descriptor `D`, its unit's rows `hU` and the
population `n`, through the Boolean profile. -/
noncomputable def circuitOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) : Verity.Circuit :=
  .ofAudit (keyCircuit D hU n)

/-- **The parsed circuit file's units**, as core's partition of `circuitOf`. -/
noncomputable def partsOf (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) :
    Verity.Partition (circuitOf D hU n) n :=
  .ofAudit (keyPartition D hU n)

/-! ## The committed files' circuit -/

/-- **A registered circuit file, parsed**: the descriptor the verifier reads from it, its unit's rows built, and the
population its public file states. -/
structure Parsed where
  D : Flock.Desc
  hU : FlockSoundness.Built D.unroll.unit
  n : ℕ

open Classical in
/-- **What the committed files parse to**: the circuit file `cf` (with the lookup tables `tables`) as the verifier reads
it (`Flock.Desc.parse tags`), and the public file `pf` against its circuit (`HmOut.loadPublic`), when both read and the
unit's rows are built; else `none`. -/
noncomputable def parsedOf (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (pf : ByteArray) :
    Option Parsed :=
  match Flock.Desc.parse tags cf tables with
  | .error _ => none
  | .ok D =>
    match HmOut.loadPublic tags D.unroll pf with
    | .error _ => none
    | .ok pub => if hU : FlockSoundness.Built D.unroll.unit then some ⟨D, hU, pub.n⟩ else none

namespace Parsed

/-- A parse's circuit: `circuitOf` at it, the empty circuit at none. -/
noncomputable def circuit : Option Parsed → Verity.Circuit
  | some r => circuitOf r.D r.hU r.n
  | none => .empty

/-- A parse's number of units. -/
def units : Option Parsed → ℕ
  | some r => r.n
  | none => 0

/-- A parse's units, as core's partition of its circuit. -/
noncomputable def parts : (o : Option Parsed) → Verity.Partition (circuit o) (units o)
  | some r => partsOf r.D r.hU r.n
  | none => .empty

end Parsed

variable (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (pf : ByteArray)

/-- **The committed circuit file as core's circuit**: `circuitOf` at what the circuit file `cf` and the public file
`pf` parse to, the empty circuit when they don't. -/
noncomputable def circuitOfFile : Verity.Circuit := Parsed.circuit (parsedOf tags cf tables pf)

/-- **The committed files' number of units**: the public file's population. -/
noncomputable def unitsOfFile : ℕ := Parsed.units (parsedOf tags cf tables pf)

/-- **The committed files' units**, as core's partition of `circuitOfFile`. -/
noncomputable def partsOfFile : Verity.Partition (circuitOfFile tags cf tables pf) (unitsOfFile tags cf tables pf) :=
  Parsed.parts (parsedOf tags cf tables pf)

end Flock.Bridge
