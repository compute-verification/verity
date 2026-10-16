/-
`ProvedScope.wiresAcyclic` on small circuits shaped like the tails' (RMSNorm's: each unit returns its aggregate to a
stage, which returns the row scalar to every unit; GEMM's: the units chained through their accumulators): the wires pass
where some order fills every slot's input ports from the leaves, port by port, and are refused at a cycle through a
slot's rows, at an input port no wire or leaf cut fills, and at a row or output reading an input column no port carries.
`proved_scope.rs`'s `wires_are_lean_acyclic` runs the same circuits.
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
    { name := "net", useful := k + 1, constPos := k, inGroups := gs, outGroups := #[⟨inWords, 1, outs⟩], inWords, a, b,
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

end Flock.Test.Wires
