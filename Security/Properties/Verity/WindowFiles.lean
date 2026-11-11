module

public import Properties.Verity.Game

/-!
# A window's committed files as one circuit

How the committed files are read at a window of several members: each member's files as v2 reads one statement's
(`FileRead`: its tags, circuit file, lookup tables and public file), and the window's circuit the members' circuits
side by side, in the members' order, with no wire between two members: what a member takes from another is an input
of its own circuit.

* **Two circuits side by side** (`Circuit.beside C D`, over core's `Verity.Circuit`): `C`'s gates, then `D`'s
  (`Fin.castAdd`, `Fin.natAdd`), each gate with its own circuit's width, input flag, wires and operation, and both
  circuits' outputs. `Partition.beside P Q` is `P`'s units, then `Q`'s, each gate in its own circuit's unit, and
  `Circuit.traceLeft X`, `Circuit.traceRight X` are a trace of the two at `C`'s gates and at `D`'s.
* **The window's read** (`WindowFiles`): its first member's files and the later members'. A window has a member, so a
  window of one member is that member's read and nothing else (`WindowFiles.one`).
* **Its circuit** (`WindowFiles.circuit`, `units`, `parts`, `consts`): the members' circuits side by side, their units
  summed, each member's units after the earlier members' (`Partition.beside`), and each member's constants at its own
  gates. At a window of one member each is that member's, by definition (`FileRead.circuitWith`, `unitsWith`,
  `partsWith` and `constsWith` at no later member).
* **A member's place** (`WindowFiles.base k`): where the member at place `k` starts in the window's units, the earlier
  members' units summed.

`Vocab.read` reads one statement's files (`FileRead`). At a window it is to read `WindowFiles`, with `circuitOf`,
`units`, `partsOf`, `Opens` and `extract` over `WindowFiles.circuit`, `units`, `parts` and `consts`: the claim at a
window is then the claim at one statement, over the members' circuits side by side. The lemmas about `beside` are
`Proofs.Properties.Verity.WindowFiles`'s.
-/

@[expose] public section

namespace Verity

namespace Circuit

variable (C D : Circuit)

/-- The gates' widths of two circuits side by side: `C`'s, then `D`'s. -/
def besideW : Fin (C.N + D.N) → ℕ := Fin.addCases C.w D.w

theorem besideW_left (i : Fin C.N) : besideW C D (Fin.castAdd D.N i) = C.w i := Fin.addCases_left i

theorem besideW_right (j : Fin D.N) : besideW C D (Fin.natAdd C.N j) = D.w j := Fin.addCases_right j

/-- The gates each gate reads, of two circuits side by side: a gate reads only gates of its own circuit. -/
def besideReads : Fin (C.N + D.N) → Finset (Fin (C.N + D.N)) :=
  Fin.addCases (fun g => (C.reads g).map (Fin.castAddEmb D.N)) fun g => (D.reads g).map (Fin.natAddEmb C.N)

/-- The gates' operations of two circuits side by side: a gate of `C` applies `C`'s to the values of `C`'s gates, a
gate of `D` applies `D`'s to the values of `D`'s. -/
def besideOp : (g : Fin (C.N + D.N)) → ((a : Fin (C.N + D.N)) → BitVec (besideW C D a)) → BitVec (besideW C D g) :=
  Fin.addCases
    (fun i v => (C.op i fun a => (v (Fin.castAdd D.N a)).cast (besideW_left C D a)).cast (besideW_left C D i).symm)
    fun j v => (D.op j fun a => (v (Fin.natAdd C.N a)).cast (besideW_right C D a)).cast (besideW_right C D j).symm

theorem besideOp_left (i : Fin C.N) (v : (a : Fin (C.N + D.N)) → BitVec (besideW C D a)) :
    besideOp C D (Fin.castAdd D.N i) v =
      (C.op i fun a => (v (Fin.castAdd D.N a)).cast (besideW_left C D a)).cast (besideW_left C D i).symm := by
  rw [besideOp, Fin.addCases_left]

theorem besideOp_right (j : Fin D.N) (v : (a : Fin (C.N + D.N)) → BitVec (besideW C D a)) :
    besideOp C D (Fin.natAdd C.N j) v =
      (D.op j fun a => (v (Fin.natAdd C.N a)).cast (besideW_right C D a)).cast (besideW_right C D j).symm := by
  rw [besideOp, Fin.addCases_right]

/-- **Two circuits side by side**: `C`'s gates, then `D`'s, each gate with its own circuit's width, input flag, wires
and operation, no wire between the two, and both circuits' outputs. -/
def beside : Circuit where
  N := C.N + D.N
  w := besideW C D
  isInput := Fin.addCases C.isInput D.isInput
  reads := besideReads C D
  reads_lt g := by
    induction g using Fin.addCases with
    | left i =>
      intro a ha
      rw [besideReads, Fin.addCases_left, Finset.mem_map] at ha
      obtain ⟨b, hb, rfl⟩ := ha
      exact C.reads_lt i b hb
    | right j =>
      intro a ha
      rw [besideReads, Fin.addCases_right, Finset.mem_map] at ha
      obtain ⟨b, hb, rfl⟩ := ha
      exact Nat.add_lt_add_left (D.reads_lt j b hb) C.N
  input_reads g := by
    induction g using Fin.addCases with
    | left i =>
      intro h
      rw [Fin.addCases_left] at h
      rw [besideReads, Fin.addCases_left, C.input_reads i h, Finset.map_empty]
    | right j =>
      intro h
      rw [Fin.addCases_right] at h
      rw [besideReads, Fin.addCases_right, D.input_reads j h, Finset.map_empty]
  op := besideOp C D
  op_congr g := by
    induction g using Fin.addCases with
    | left i =>
      intro v v' h
      rw [besideOp_left, besideOp_left]
      congr 1
      exact C.op_congr i _ _ fun a ha => by
        rw [h (Fin.castAdd D.N a) (by rw [besideReads, Fin.addCases_left]; exact Finset.mem_map_of_mem _ ha)]
    | right j =>
      intro v v' h
      rw [besideOp_right, besideOp_right]
      congr 1
      exact D.op_congr j _ _ fun a ha => by
        rw [h (Fin.natAdd C.N a) (by rw [besideReads, Fin.addCases_right]; exact Finset.mem_map_of_mem _ ha)]
  outputs := C.outputs.map (Fin.castAddEmb D.N) ∪ D.outputs.map (Fin.natAddEmb C.N)

variable {C D}

/-- **A trace of two circuits side by side, at `C`'s gates.** -/
def traceLeft (X : (C.beside D).Trace) : C.Trace := fun i => (X (Fin.castAdd D.N i)).cast (besideW_left C D i)

/-- **A trace of two circuits side by side, at `D`'s gates.** -/
def traceRight (X : (C.beside D).Trace) : D.Trace := fun j => (X (Fin.natAdd C.N j)).cast (besideW_right C D j)

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

/-- **Where the member at place `k` starts in the window's units**: the earlier members' units summed. -/
noncomputable def base (k : ℕ) : ℕ := ((w.members.take k).map FileRead.units).sum

end WindowFiles

end Verity
