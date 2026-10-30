/-
`ProvedScope.wiresAcyclic` on small circuits shaped like the tails' (RMSNorm's: each unit returns its aggregate to a
stage, which returns the row scalar to every unit; GEMM's: the units chained through their accumulators): the wires pass
where some order fills every slot's input ports from the leaves, port by port, and are refused at a cycle through a
slot's rows, at an input port no wire or leaf cut fills, and at a row or output reading an input column no port carries.
With hidden outputs those circuits are in scope (`scopeOkI`), with public outputs not (`scopeOk`). `proved_scope.rs`'s
`wires_are_lean_acyclic` and `hidden_outputs_take_the_general_instance_form` run the same circuits.
-/
import Flock.ProvedScope

namespace Flock.Test.Wires

/-- A net of `inWords` input words with port groups `gs`, then `rows` (A's columns, B's columns), then the constant; its
output ports, of widths `outs`, start at the first computed row. -/
def net (gs : Array Group) (inWords : Nat) (outs : Array Nat) (rows : Array (Array Nat × Array Nat))
    (lookup : Bool := false) : Net :=
  let n := inWords * WORD
  let k := n + rows.size
  let self := (Array.range n).map fun i => #[i]
  let ra := (self ++ rows.map (·.1)).push #[k]
  let rb := (self ++ rows.map (·.2)).push #[k]
  match Sparse.ofRows ra (2 ^ 10), Sparse.ofRows rb (2 ^ 10) with
  | .ok a, .ok b =>
    { name := "net", useful := k + 1, constPos := k, inGroups := gs, outGroups := #[⟨inWords, 1, outs⟩], inWords,
      aT := .pure a, bT := .pure b,
      unitLog := 10, lookup := if lookup then some default else none }
  | _, _ => default

/-- A unit: leaf port `x` (column 0) and cut port `s` (column 128); it returns `x·x` (port 0) and `x·s` (port 1). -/
def unit : Net := net #[⟨0, 1, #[1]⟩, ⟨1, 1, #[1]⟩] 2 #[1, 1] #[(#[0], #[0]), (#[0], #[128])]

/-- A stage of `k` one-bit input ports (columns `0, …, k − 1`), returning their XOR (port 0) and the constant (port 1,
column 130). -/
def stage (k : Nat) : Net :=
  net #[⟨0, 1, Array.replicate k 1⟩] 1 #[1, 1] #[(Array.range k, #[130]), (#[130], #[130])]

/-- The circuit of `nets` (the unit first, `upv` of them) and the rows' two nets after them. -/
def circ (nets : Array Net) (perVu : Array Nat) (wires : Array (Nat × Nat × Nat × Nat × Nat × Nat))
    (leafCuts : Array (Nat × Nat × Nat) := #[]) : Circuit :=
  { name := "wires", sha := .empty, nets := (nets ++ #[default, default]).map ("net", ·), ranges := #[], ports := #[],
    outPorts := #[], g := 1, kLog := 0, upv := perVu.getD 0 0, unitNet := 0, leavesIn := #[], leavesOut := #[],
    key := .empty, dummyDigests := #[], dummyOutputs := #[], leafScheme := "", perVu := perVu ++ #[1, 1],
    wires := wires.map fun (a, b, c, d, e, f) => ⟨(a, b, c), (d, e, f)⟩, shaNet := nets.size, hmNet := nets.size + 1,
    leafCuts }

/-- RMSNorm's shape, two units: each returns `x·x` to the stage, whose AND is both units' `s`. -/
def rms (w0 : Nat := 0) : Circuit :=
  circ #[unit, stage 2] #[2, 1] #[(1, 0, 0, 0, 0, w0), (1, 0, 1, 0, 1, w0), (0, 0, 1, 1, 0, 0), (0, 1, 1, 1, 0, 0)]

#guard ProvedScope.outReads unit == (#[#[0], #[0, 1]], true)
#guard ProvedScope.outReads (stage 2) == (#[#[0, 1], #[]], true)
#guard ProvedScope.wiresAcyclic (rms 0)
-- the stage reads `x·s`, which reads the stage's own output: a cycle through the units' rows
#guard !ProvedScope.wiresAcyclic (rms 1)

/-- GEMM's shape: unit 0's cut port from the stage's constant, unit `i + 1`'s from unit `i`'s `x·s`, the stage from the
last unit's. -/
def gemm (n : Nat) : Circuit :=
  circ #[unit, stage 1] #[n, 1]
    (#[(0, 0, 1, 1, 0, 1)] ++ (Array.range (n - 1)).map (fun i => (0, i + 1, 1, 0, i, 1)) ++ #[(1, 0, 0, 0, n - 1, 1)])

#guard ProvedScope.wiresAcyclic (gemm 4)
-- the wires listed last to first: one pass fills one port, and the passes still reach every port
#guard ProvedScope.wiresAcyclic { gemm 4 with wires := (gemm 4).wires.reverse }
-- unit 0 from the stage's AND, which reads unit 3, which reads unit 0
#guard !ProvedScope.wiresAcyclic { gemm 4 with wires := (gemm 4).wires.set! 0 ⟨(0, 0, 1), (1, 0, 0)⟩ }

-- a cut port no wire fills is never filled; a leaf cut fills it
#guard !ProvedScope.wiresAcyclic (circ #[unit] #[1] #[])
#guard ProvedScope.wiresAcyclic (circ #[unit] #[1] #[] #[(0, 1, 7)])
-- a wire from a slot past its net's
#guard !ProvedScope.wiresAcyclic (circ #[unit, stage 1] #[1, 1] #[(0, 0, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)])

/-- A stage of one input port (column 0) whose row reads input column 5 as well, which no port carries; or (`out`)
whose second output port is input column 1. -/
def loose (out : Bool) : Net :=
  if out then { net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])] with outGroups := #[⟨0, 1, #[1, 1]⟩] }
  else net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[5])]

#guard (ProvedScope.outReads (loose false)).2 == false
#guard (ProvedScope.outReads (loose true)).2 == false
#guard !ProvedScope.wiresAcyclic (circ #[unit, loose false] #[1, 1] #[(0, 0, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0)])
#guard !ProvedScope.wiresAcyclic (circ #[unit, loose true] #[1, 1] #[(0, 0, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0)])
-- a net with no slot is not read
#guard ProvedScope.wiresAcyclic (circ #[unit, loose false] #[1, 0] #[] #[(0, 1, 0)])

/-- A stage of one input port and one computed row (column 128), its constant column 129, whose output ports are its
first `k` columns from 128. -/
def past (k : Nat) : Net := { net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])] with outGroups := #[⟨1, 1, Array.replicate k 1⟩] }

-- an output port at the constant reads nothing; one past it, which no row computes, is refused
#guard ProvedScope.outReads (past 2) == (#[#[0], #[]], true)
#guard (ProvedScope.outReads (past 3)).2 == false
#guard !ProvedScope.wiresAcyclic (circ #[unit, past 3] #[1, 1] #[(0, 0, 1, 1, 0, 2), (1, 0, 0, 0, 0, 0)])

/-- A lookup slot (one input port, one output port, taken whole) between two stages; `back` wires the lookup from the
stage it feeds. -/
def lookup (back : Bool) : Circuit :=
  circ #[unit, stage 1, net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])] (lookup := true), stage 1] #[1, 1, 1, 1]
    #[(1, 0, 0, 0, 0, 0), (2, 0, 0, if back then 3 else 1, 0, 0), (3, 0, 0, 2, 0, 0), (0, 0, 1, 3, 0, 0)]

#guard ProvedScope.wiresAcyclic (lookup false)
#guard !ProvedScope.wiresAcyclic (lookup true)

-- where the wires feed the unit (a cut group) the check refuses a cycle; a unit of one group does not
#guard ProvedScope.wiresFeed (rms 1)
#guard !ProvedScope.wiresFeed (circ #[net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])]] #[1] #[])

/-- A unit of one group and a stage it feeds, the stage returning the outputs. -/
def tail : Circuit :=
  { circ #[net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])], stage 1] #[1, 1] #[(1, 0, 0, 0, 0, 0)] with outNet := 1 }

-- with hidden outputs, the general instance form: a unit reading a cut group, or a stage returning the outputs, once the
-- wires are acyclic; with public outputs, the unit's form alone
#guard ProvedScope.wiresAcyclic tail && ProvedScope.wiresFeed tail
#guard ProvedScope.scopeOkI (rms 0) && ProvedScope.scopeOkI tail
#guard !ProvedScope.scopeOk (rms 0) && !ProvedScope.scopeOk tail
#guard ProvedScope.scopeWhy false (rms 0) == "a unit reading 2 input groups"
#guard ProvedScope.scopeWhy false tail == "a circuit whose outputs a tail stage computes (META out_net)"

/-- A repeated last stage: two units, each wiring its own slot of a stage of two 16-bit output ports, the output words'
`(slot, port)` (META `out_net_ports` of `[slot, port]` pairs) `slots` and `ports`. -/
def slotted (slots ports : Array Nat) : Circuit :=
  { circ #[net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])], net #[⟨0, 1, #[1]⟩] 1 #[16, 16] (Array.replicate 32 (#[0], #[0]))]
      #[2, 2] #[(1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 1, 0)] with
    outNet := 1, outPorts := #[("y", 2)], outNetPorts := ports, outNetSlots := some slots }

def slottedOk (slots ports : Array Nat) : Bool :=
  HmRow.slottedOk (slotted slots ports) slots (portCols ((slotted slots ports).nets.getD 1 default).2.outGroups)

-- one 16-bit port of a slot per output word, each (slot, port) once: any slot and port of the stage, in any order
#guard slottedOk #[0, 1] #[0, 0] && slottedOk #[1, 0] #[0, 0] && slottedOk #[0, 0] #[0, 1] && slottedOk #[1, 1] #[1, 0]
#guard !slottedOk #[0, 0] #[0, 0] && !slottedOk #[0, 2] #[0, 0] && !slottedOk #[0, 1] #[0, 2] && !slottedOk #[0] #[0]
#guard ProvedScope.wiresAcyclic (slotted #[0, 1] #[0, 0]) && ProvedScope.scopeOkI (slotted #[0, 1] #[0, 0])
-- output word `o` at its slot's port
#guard (List.range 2).all fun o =>
  HmOut.outCol (slotted #[0, 1] #[0, 0]) 1 o == (slotted #[0, 1] #[0, 0]).slotCol 1 1 o + WORD
#guard HmOut.outCol (slotted #[1, 1] #[1, 0]) 0 0 == (slotted #[1, 1] #[1, 0]).slotCol 1 0 1 + WORD + 16

-- META `out_net_ports`: ports alone (slot 0), `[slot, port]` pairs, or a mix
#guard (HmRow.outNetEntries #[Lean.toJson 0, Lean.toJson 1]).toOption == some (#[0, 1], none)
#guard (HmRow.outNetEntries #[.arr #[Lean.toJson 0, Lean.toJson 0], .arr #[Lean.toJson 1, Lean.toJson 0]]).toOption ==
  some (#[0, 0], some #[0, 1])
#guard (HmRow.outNetEntries #[Lean.toJson 3, .arr #[Lean.toJson 1, Lean.toJson 0]]).toOption == some (#[3, 0], some #[0, 1])
#guard (HmRow.outNetEntries #[.arr #[Lean.toJson 1]]).toOption.isNone

end Flock.Test.Wires
