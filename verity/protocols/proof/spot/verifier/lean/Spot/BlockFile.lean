module

public import Spot.BlockFront
public import Flock.Test.Compose

/-!
# A block's circuit file on C: its rows from the front, its unit a stand-in

Block `units` of the registered circuit `P` as a `Flock.Compose.Block`. Its input rows are the front's in(S) and its
output ports the front's out(S) (`Spot.BlockFront.front`, C's word-level descriptor gates in increasing index), cut where
a gate's writer changes (`writer`), each row its gates' bits concatenated, low bit first. Its unit (`standIn`) is a flat
net of the rows' widths that stands in for the block's lowered Program. The `block_file` piece writes the file, reads it
back through the door's reader and holds it to the block (`Flock.Test.Compose.agrees`).
-/

@[expose] public section

namespace Spot.BlockFile

open Flock Flock.Program
open Lean (Json toJson)

/-- Gate `g`'s writer: its root node and, for a node of a composite, its member; a node of a primitive, an input among
them, is one writer. -/
def writer (P : Prog) (g : Nat) : V (Nat × Nat) := do
  let root := P.defs[P.root]!
  let offs := root.body.offsets
  let some k := (List.range root.body.nodes.size).find? fun k => offs.getD k 0 ≤ g && g < offs.getD (k + 1) 0
    | throw s!"gate {g} is past C's root"
  let fn := P.defs[(root.body.nodes[k]!).fn]!
  return (k, if fn.prim then 0 else (g - offs.getD k 0) / max fn.gates 1)

/-- A row: its name `g<first gate>`, its length in bits and its gates. -/
abbrev Row := String × Nat × Array Nat

/-- `ports`, in increasing index, cut into rows where the writer changes. -/
def rows (P : Prog) (ports : Array BlockFront.Port) : V (Array Row) := do
  let mut out : Array Row := #[]
  let mut last : Option (Nat × Nat) := none
  for p in ports do
    let g := BlockFront.Port.gate p
    let w ← writer P g
    if last == some w then
      out := out.modify (out.size - 1) fun (n, b, gs) => (n, b + p.width, gs.push g)
    else
      out := out.push (s!"g{g}", p.width, #[g])
    last := some w
  return out

/-- The stand-in unit: `nIn` 16-bit input leaves and `nOut` output words, output bit `i` input bit `i` AND input bit
`i + 1`, XOR input bit `i + 2`, the input's bits read cyclically. -/
def standIn (nIn nOut : Nat) : HmNets.Laid :=
  let p : HmNets.M (Array HmNets.Lin) := do
    let x := (← HmNets.input (16 * nIn)).toArray
    let at_ := fun i => x.getD (i % max x.size 1) default
    (Array.range (16 * nOut)).mapM fun i => do
      pure ((← HmNets.and (at_ i) (at_ (i + 1))) ^^^ at_ (i + 2))
  let (outs, st) := p.run HmNets.St.empty
  HmNets.layout st (Array.replicate nIn 16) (Array.replicate nOut 16) outs

/-- A block's front, its input rows, its output rows and the `Compose.Block` they make. -/
structure Cut where
  front : BlockFront.Front
  ins : Array Row
  outs : Array Row
  block : Compose.Block

/-- Block `units` of `P` cut into rows, with the stand-in unit. -/
def cut (P : Prog) (units : Array Nat) : V Cut := do
  let f ← BlockFront.front P units
  let ins ← rows P f.ins
  let outs ← rows P f.outs
  let nIn := ins.foldl (fun a (_, b, _) => a + (b + 15) / 16) 0
  let ow := outs.map fun (n, b, _) => (n, (b + 15) / 16)
  let nOut := ow.foldl (· + ·.2) 0
  return { front := f, ins, outs,
           block := { name := "block", body := .flat (standIn nIn nOut), ins := ins.map fun (n, b, _) => (n, b), outs := ow } }

/-- The `block_file` piece: `{descriptor, units, partition}` (`block_front`'s input and the partition's digest)
to `{ins, outs, rows, out_rows, meta}`: the front's ports `[gate, width]`, each row `[name, bits, gates]`, and META as
the door's reader read it from the file, which it refuses unless the file reads back as the block. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let some pd := (j.getObjValAs? String "partition").toOption | throw "partition is not a string"
  let P ← decode d.toUTF8
  let x ← cut P us
  let file ← Compose.compose x.block (hexOf (Sha512.hash d.toUTF8)) pd
  let c ← ProvedScope.circuitOf (Zk.tags Tags.circuit (Zk.maskLogOf file) 1) file []
  if !(← Test.Compose.agrees x.block c) then throw "the block's file read back is not the block"
  let ports := fun (ps : Array BlockFront.Port) => Json.arr (ps.map fun p => Json.arr #[toJson (BlockFront.Port.gate p), toJson p.width])
  let rowsJ := fun (rs : Array Row) => Json.arr (rs.map fun (n, b, gs) => Json.arr #[.str n, toJson b, toJson gs])
  return Json.mkObj [("ins", ports x.front.ins), ("outs", ports x.front.outs), ("rows", rowsJ x.ins),
    ("out_rows", rowsJ x.outs), ("meta", c.metaJ)]

end Spot.BlockFile
