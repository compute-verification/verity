/-
The claims' partition of a Flock circuit file, and the check that the registration's population is the file's units in
order (`Flock.Bridge.partOf` at the file's key program, `Placed.keyProg`):

* **The file's units.** The key program is `n` units of the circuit's template instance rows (the unit net's rows stacked
  `units_per_vu` times, `Placed.circuitRows`): `B` input columns and `m` computed rows a unit, on `2 + n·B` input gates
  (the constant, the zero, then `B` sources a unit: unit `u`'s input column `x` on gate `2 + u·B + b`, `b` the first
  column with `x`'s source key, `Aliased.keyIdx`, or the zero for the zero's key), then each unit's rows, a row lowered to
  three gates (`Bridge.progOf`). No unit reads another's sources, so unit `u`'s proof block (`Bridge.blockOf`) is an
  affine shift of one summary (`ofCircuit`): its rows' `3m` lowered gates from `2 + n·B + 3m·u`, then the sources `S` it
  owns at `2 + u·B + b`, `b ∈ S`, the first column of each source key some row reads that is not the zero's (`srcKey`
  mirrors `Aliased.srcKey`). The constant and the zero are constants (`Bridge.constsOf`), and so is a public input's gate
  at the live statement (`Bridge.constsOfP`), whose key is the zero's here. `ofCircuit` reads the unit's rows once and
  each stacked input's key once, O(the unit), never a unit per slot; `block u` is unit `u`'s proof block.
* **The population check** (`why`): the registration's strata over its program and query (`Partition.strata`), read
  stratum by stratum, are its units `0, 1, …, N − 1` in order, and the public file holds exactly those units in that
  order (`units.indices` is `0, …, N − 1`, read off the header v2 reads the file's other fields from, `header`), so slot
  `u` is the registration's unit `u` and the file's unit `u`, in strata
  of the sizes `sizes` gives, the `ns` of `partOf`. O(the strata's ranges + the public file's units), which the setup
  reads already.
-/
module

public import Flock.Circuit
public import Flock.RecResiduals
public import Flock.KnownRead
public import Flock.JsonParse
public import Flock.Norm
public import Spot.Partition
meta import Flock.Circuit
meta import Flock.RecResiduals
meta import Flock.KnownRead
meta import Flock.JsonParse
meta import Flock.Norm
meta import Spot.Partition

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.FilePart

open Lean (Json)
open Flock (V)

/-- **The column Δ copies into a unit input, without the template instance** (`Aliased.Src`). -/
inductive Src
  | zero
  | msg (p byte bit : Nat)
  | wire (net slot col : Nat)
  | lit (col : Nat)
  | pad (k off : Nat)
  deriving BEq, Hashable, Inhabited, Repr

/-- The last index of the ascending `s` whose entry is at most `x`, or 0
(`((List.range s.size).reverse.find? (s[·]! ≤ x)).getD 0`). -/
def lastLe (s : Array Nat) (x : Nat) : Nat := Id.run do
  let mut lo := 0
  let mut hi := s.size
  while lo < hi do
    let mid := (lo + hi) / 2
    if s.getD mid 0 ≤ x then lo := mid + 1 else hi := mid
  return lo - 1

/-- What `srcAt` looks up once a circuit: the row ports' first words, each `(copy, port)`'s first leaf cut and its first
wire into the unit net. -/
structure Index where
  starts : Array Nat
  cuts : Std.HashMap (Nat × Nat) Nat
  wires : Std.HashMap (Nat × Nat) (Nat × Nat × Nat)

def Index.of (c : Flock.Circuit) : Index :=
  { starts := Flock.KnownRead.starts c
    cuts := c.leafCuts.foldl (fun m (k, p, w) => if m.contains (k, p) then m else m.insert (k, p) w) {}
    wires := c.wires.foldl (fun m w =>
      if w.dst.1 == c.unitNet && !m.contains (w.dst.2.1, w.dst.2.2) then m.insert (w.dst.2.1, w.dst.2.2) w.src else m) {} }

/-- Leaf word `x`'s bit `t` as row port `q`'s message bit, where `q` is the port holding word `x`: within its bits, or
the zero. -/
private def msgAt (c : Flock.Circuit) (ix : Index) (x t : Nat) (wide : Bool) : Src :=
  let q := lastLe ix.starts x
  let s := ix.starts.getD q 0
  if (wide || t < 16) && 16 * (x - s) + t < (c.ports.getD q default).bits then .msg q (2 * (x - s) + t / 8) (t % 8)
  else .zero

/-- Copy `k`'s input port `p`, bit `t`: `Aliased.srcAt`. -/
def srcAt (c : Flock.Circuit) (ix : Index) (k p t : Nat) : Src :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    let l := (c.leavesIn.getD k #[]).getD p 0
    if l < 0 then .zero else msgAt c ix l.toNat t true
  else match ix.cuts[(k, p)]? with
    | some w => msgAt c ix w t false
    | none => match ix.wires[(k, p)]? with
      | some (net, slot, j) =>
        .wire net slot (((Flock.portCols (c.nets.getD net default).2.outGroups).getD j default).1 + t)
      | none => .lit 0

/-- **Stacked input `x`'s key** (`Aliased.srcKey`): copy `x / W`'s input at slot offset `x % W` (`W = inWords · WORD`),
`srcAt` at the port bit it is, or the padding row it is. -/
def srcKey (c : Flock.Circuit) (ix : Index) (x : Nat) : Src :=
  let W := c.unit.inWords * Flock.WORD
  let pc := Flock.portCols c.unit.inGroups
  let off := x % W
  let col := fun p => pc.getD p (0, 0)
  match (List.range pc.size).find? fun p => decide ((col p).1 ≤ off ∧ off < (col p).1 + (col p).2) with
  | some p => srcAt c ix (x / W) p (off - (col p).1)
  | none => .pad (x / W) off

/-- **The circuit file's partition at `n` units**: `n`, each unit's `B` input columns and `m` computed rows
(`Placed.circuitRows`), and the sources `S` each unit owns, ascending. -/
structure FilePart where
  n : Nat
  B : Nat
  m : Nat
  srcs : Array Nat
  deriving BEq, Repr

/-- **The summary of the circuit `c`'s key program at `n` units** (module docstring): the unit net's `nIn` input
columns and `nComp` computed rows (`Rows.ofNet`) stacked `upv` times, and the first column of each source key a computed
row reads (`RecResiduals.readIns`), the zero's aside. -/
def ofCircuit (c : Flock.Circuit) (n : Nat) : FilePart :=
  let g := c.unit
  let nIn := min (g.inWords * Flock.WORD) g.constPos
  let B := c.upv * nIn
  let rd := Flock.RecResiduals.readIns g
  let ix := Index.of c
  let keys := (Array.range B).map (srcKey c ix)
  let first := (Array.range B).foldl (fun (m : Std.HashMap Src Nat) x =>
    if m.contains keys[x]! then m else m.insert keys[x]! x) {}
  let owned := (Array.range B).foldl (fun (o : Array Bool) x =>
    if rd.getD (x % nIn) false && keys[x]! != .zero then o.set! (first.getD keys[x]! x) true else o)
    (Array.replicate B false)
  { n, B, m := c.upv * (g.constPos - nIn + 1), srcs := (Array.range B).filter (owned.getD · false) }

/-- The key program's input gates: the constant, the zero, and `B` sources a unit. -/
def FilePart.inGates (f : FilePart) : Nat := 2 + f.n * f.B

/-- **Unit `u`'s proof block** (`Bridge.blockOf`): its rows' lowered gates, then the gates of the sources it owns. -/
def FilePart.block (f : FilePart) (u : Nat) : Array Nat :=
  (Array.range (3 * f.m)).map (f.inGates + 3 * f.m * u + ·) ++ f.srcs.map (2 + u * f.B + ·)

/-- The strata's sizes, when their units read stratum by stratum are `0, 1, …` in order: each range starts where the
last ended. None when they are not. -/
def sizes (strata : Array (String × Array (Nat × Nat))) : Option (Array Nat) := do
  let mut at_ := 0
  let mut out := #[]
  for (_, rs) in strata do
    let lo := at_
    for (a, b) in rs do
      if a != at_ || b < a then none
      at_ := b
    out := out.push (at_ - lo)
  return out

/-- **Why the registration's population is not the circuit file's units in order** (module docstring), none when it is:
`strata` the registration's (`Partition.strata`), `indices` the public file's `units.indices`. -/
def why (strata : Array (String × Array (Nat × Nat))) (indices : Array Nat) : Option String :=
  match sizes strata with
  | none => some "partition: the registration's strata, read stratum by stratum, are not its units in order"
  | some ns =>
    let n := ns.foldl (· + ·) 0
    if indices != Array.range n then
      some s!"partition: the public file's {indices.size} units are not the registration's {n}, in order"
    else none

/-- The public file's header line: its bytes before the first newline, all of them when it has none. -/
def headerLine (b : ByteArray) : ByteArray := Id.run do
  for i in [0:b.size] do
    if b[i]! == 10 then return b.extract 0 i
  return b

/-- **The public file's header**, as `HmRow.loadPublic` reads it: its header line parsed by the verifier's total parser
(`Flock.JsonParse.parse`), then `Flock.norm`. The population check (`unitIndices`) and v2's other reads of the file
(`OneStageZk.publicHeld`, `OneStageZk.provedUnits`) read this one value. -/
def header (pubB : ByteArray) : V Json := do
  let some line := String.fromUTF8? (headerLine pubB) | throw "the public file's header is not UTF-8"
  Flock.norm <$> (Flock.JsonParse.parse line).mapError (s!"the public file's header is not JSON: {·}")

/-- A header's `units.indices`, a list of natural numbers. -/
def indicesOf (h : Json) : V (Array Nat) := do
  let .arr is := getD (getD h "units") "indices" | throw "the public file's header names no units.indices"
  is.mapM fun i => match nat? i with
    | some u => pure u
    | none => throw "the public file's units.indices is not a list of natural numbers"

/-- The public file's `units.indices`, off its header (`header`). -/
def unitIndices (pubB : ByteArray) : V (Array Nat) := do
  indicesOf (← header pubB)

/-- **The population check on the files** (`why`): the registration's strata over the verifier's program (its
descriptor's bytes) and query, against the public file's units. -/
def check (program : ByteArray) (query : Json) (pubB : ByteArray) : V Unit := do
  let strata ← Partition.strata (← Program.parse program) query
  if let some w := why strata (← unitIndices pubB) then throw w

/-- `{"program", "query", "indices"}` or `{"program", "query", "public"}` to `{"why"}`: `why` on the strata of the query
over the program, against the given `indices` or the units the public file (`public`, its bytes in hex) names
(`unitIndices`), null when the population is the units in order; refused where the strata or the file's units are. -/
def piece (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let indices ← match getD j "public" with
    | .null => do
      let .arr is := getD j "indices" | throw "indices is not a list"
      is.mapM fun i => match nat? i with
        | some u => pure u
        | none => throw "indices is not a list of natural numbers"
    | .str s => unitIndices (← Flock.unhex s)
    | _ => throw "public is not a string"
  let strata ← Partition.strata (← Program.parse (← Flock.canon (getD j "program")).toUTF8) (getD j "query")
  return Json.mkObj [("why", match why strata indices with | some w => Json.str w | none => Json.null)]

#guard sizes #[("a", #[(0, 3)]), ("b", #[(3, 5)])] == some #[3, 2]
#guard sizes #[("a", #[(0, 1), (2, 3)]), ("b", #[(1, 2)])] == none
#guard sizes #[] == some #[]
#guard why #[("a", #[(0, 2)])] #[0, 1] == none
#guard (why #[("a", #[(0, 2)])] #[0]).isSome
#guard (why #[("a", #[(0, 2)])] #[1, 0]).isSome
#guard (why #[("a", #[(0, 2)])] #[0, 1, 2]).isSome
#guard (unitIndices ("{\"circuit_sha512\": \"ab\", \"frame_v3\": {\"roots\": {\"x\": \"cd\"}}, \"instances\": 2, " ++
  "\"units\": {\"blocks\": [[0, 1]], \"classes\": [\"c\", \"c\"], \"indices\": [0, 1], \"units_per_vu\": 1, " ++
  "\"vus_per_block\": 2}}\n\n{").toUTF8).toOption == some #[0, 1]
#guard (unitIndices "{\"units\":{\"indices\":[0,1]}}".toUTF8).toOption == some #[0, 1]
#guard (unitIndices "{\"units\":{\"indices\":[0,-1]}}".toUTF8).toOption.isNone
#guard (unitIndices "{\"units\":{}}\n".toUTF8).toOption.isNone
#guard (unitIndices "{\"units\":".toUTF8).toOption.isNone
#guard (unitIndices (ByteArray.mk #[255, 10])).toOption.isNone
#guard lastLe #[0, 2, 2, 5] 2 == 2
#guard lastLe #[0, 2, 5] 7 == 2
#guard lastLe #[0, 2, 5] 0 == 0
#guard (FilePart.block { n := 2, B := 3, m := 1, srcs := #[0, 2] } 1) == #[11, 12, 13, 5, 7]

end Spot.FilePart
