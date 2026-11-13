module

public import CoreProtocol.Blobs
public import Flock.Program
public import Flock.ProvedScope
public import Flock.Zk
public import Spot.Bits
meta import CoreProtocol.Blobs
meta import Flock.Program
meta import Flock.ProvedScope
meta import Flock.Zk
meta import Spot.Bits

/-!
# The scan's circuit files held to its templates

The scan's sampled round proves a circuit file a template of the scan's root. The period's check reads the scan off its
description (`NetworkCheck.Period.scanOf`); this module holds each template's circuit file to that description, so what
the description says of a template's inputs and returns is said of the gates the round proves.

A file is named by the SHA-512 of its bytes and found among the caller's blobs by it (`Core.Blobs.get`), so the bytes
read are the named ones. It is read as the sampled verifier reads a session's statement circuit
(`Flock.ProvedScope.circuitOf` under `verity/flock-circuit`'s tags with `--zk`'s hooks, `tagsOf`), a flat file. A
templated file (`flock-circuit/tmpl`) is refused by name, on its head line, before anything parses it (`templated`):
its unit's rows exist only unrolled from its template, and a check that unrolls them works in proportion to the
unrolled circuit.

Then, a file (`fileWhy`):

* Bits lays it out as its template (`Spot.Bits.check`): its input rows are the template's parameters in order, its
  output rows the template's return, one template instance a unit, and every returned output bit sits on a computed
  row. This reads the unit net's header and never its rows;
* for a template with ingress rows, the unit's rows read every bit of them, and read no input column that no port
  holds (`Spot.Bits.readsWhy` on `Spot.Bits.unitReads`). In Bits' layout a source bit no row reads is the constant
  false, so an ingress bit no row read would be no input gate. `unitReads` is the one pass over a unit's rows, and it
  runs once a file with ingress rows: a template with none has no pass.

`why` is the scan's finding: the first of its templates, in the root's order, whose file is not held. Two templates
naming one file are refused, so the passes over rows are as many as the templates with ingress rows (`passes`),
whatever the link-window's buckets or the period's link-windows.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.ScanFiles

open Flock (V)

/-- **A template of the scan's root with its circuit file**: its descriptor id (the Definition's key in the Program),
its ingress rows' parameters (none for a template that reads no ingress row), and the SHA-512 of its circuit file,
lowercase hex. -/
structure Template where
  key : String
  rows : List String
  circuit : String

/-- **Whether the bytes begin a templated circuit file**: the head line's format is `flock-circuit/tmpl`
(`Flock.Tags.circuitTmpl`). It reads the format's bytes and one more, whatever the file's size. -/
def templated (b : ByteArray) : Bool :=
  let pre := (Flock.Tags.circuitTmpl.fileFormat ++ " ").toUTF8
  b.extract 0 pre.size == pre

/-- **The tags a circuit file is read under**: `verity/flock-circuit`'s with `--zk`'s hooks in a session of one table,
as `verify --zk --statement verity/flock-circuit` reads a session's statement circuit. -/
def tagsOf (b : ByteArray) : Flock.Tags := Flock.Zk.tags Flock.Tags.circuit (Flock.Zk.maskLogOf b) 1

/-- **The circuit file of template `key`**, read from its bytes: a templated file is refused by name first, then the
file is parsed as the sampled verifier parses a statement's circuit. -/
def circuitOf (key : String) (b : ByteArray) : V Flock.Circuit := do
  if templated b then
    throw s!"scan: the circuit file of template {key} is a templated file ({Flock.Tags.circuitTmpl.fileFormat}): the \
      period's check reads a flat file's unit and never unrolls a template's rows"
  (Flock.ProvedScope.circuitOf (tagsOf b) b []).mapError fun e => s!"scan: the circuit file of template {key}: {e}"

/-- **Why the circuit file `c` is not template `key` of the Program `P`**, or none: Bits doesn't lay it out as the
template (`Spot.Bits.check`), or, `rows` being the template's ingress rows' parameters, its unit's rows don't read
every bit of them or read an input column no port holds (`Spot.Bits.readsWhy`). The unit's rows are read only where
`rows` names a parameter, once (`Spot.Bits.unitReads`). -/
def fileWhy (P : Flock.Program.Prog) (key : String) (rows : List String) (c : Flock.Circuit) : Option String :=
  match Spot.Bits.check c P key with
  | some why => some why
  | none =>
    if rows.isEmpty then none
    else match P.defs.find? (·.key == key) with
      | some d => Spot.Bits.readsWhy c d (Spot.Bits.unitReads c) rows
      | none => some s!"the circuit file's template {key} is no Definition of the Program"

/-- **The passes over a unit's rows** `why` makes at templates `ts`: one a template with ingress rows. -/
def passes (ts : Array Template) : Nat := (ts.filter fun t => !t.rows.isEmpty).size

/-- **Why the scan's circuit files are not its templates'**, or none: two templates name one file, or, at the first
template in order whose file is not held, no blob is the named file, the file is templated or doesn't parse
(`circuitOf`), or it is not the template as Bits lays it out with its ingress rows read (`fileWhy`). The files are
read one after another, none held past its own finding. -/
def why (P : Flock.Program.Prog) (ts : Array Template) (bs : Core.Blobs) : Option String :=
  if !decide (ts.toList.map (·.circuit)).Nodup then
    some "scan: two templates name one circuit file: a template has a circuit file of its own"
  else ts.findSome? fun t =>
    match bs.get t.circuit s!"scan: the circuit file of template {t.key}" >>= circuitOf t.key with
    | .error e => some e
    | .ok c => (fileWhy P t.key t.rows c).map fun w => s!"scan: the circuit file of template {t.key}: {w}"

/-! ## Checks -/

private def refuses (why : String) : Option String → Bool := Option.any (·.startsWith why)

private def fails (why : String) : V Flock.Circuit → Bool
  | .error e => e.startsWith why
  | .ok _ => false

#guard templated "flock-circuit/tmpl scan\nMETA {}\n".toUTF8 && !templated "flock-circuit scan\nMETA {}\n".toUTF8 &&
  !templated "flock-circuit/tmplx scan\n".toUTF8 && !templated "flock-circuit/tmpl".toUTF8 && !templated .empty

#guard fails "scan: the circuit file of template T_v1 is a templated file (flock-circuit/tmpl)"
    (circuitOf "T_v1" "flock-circuit/tmpl scan\nMETA {}\n".toUTF8) &&
  fails "scan: the circuit file of template T_v1: circuit: not a flock-circuit file" (circuitOf "T_v1" "x\n".toUTF8) &&
  fails "scan: the circuit file of template T_v1: circuit: no META line"
    (circuitOf "T_v1" "flock-circuit scan\n".toUTF8)

/-- A primitive template of `Value` parameters `ps` (name, width) returning `Value ret`. -/
private def toyDef (key : String) (ps : List (String × Nat)) (ret : Nat) : Flock.Program.Defn :=
  { key, id := key, prim := true, params := (ps.map fun (p, w) => (p, Flock.Program.Ty.value w)).toArray,
    ret := .value ret, gates := 1, paramLeaves := ps.length, outLeaves := 1 }

/-- An ingress template on a 16-bit `row` and a 16-bit `sin`, and an egress template on a 16-bit `state` and a 16-bit
`pad`, each returning 16 bits. -/
private def toyP : Flock.Program.Prog :=
  { defs := #[toyDef "TIngress_v1" [("row", 16), ("sin", 16)] 16, toyDef "TEgress_v1" [("state", 16), ("pad", 16)] 16]
    index := {}, root := 0, programFn := "", sha512 := .empty }

/-- A unit net of one input word holding two 16-bit ports, and 16 computed rows, row `i` the XOR of the input columns
`r i` times the constant, returned from column 128 as one 16-bit leaf. -/
private def netOf (r : Nat → Array Nat) : Flock.Net :=
  let a := Array.replicate Flock.WORD #[] ++ (Array.range 16).map r
  let b := Array.replicate Flock.WORD #[] ++ Array.replicate 16 #[Flock.WORD + 16]
  match Flock.Sparse.ofRows a (Flock.WORD + 17), Flock.Sparse.ofRows b (Flock.WORD + 17) with
  | .ok a, .ok b =>
    { (default : Flock.Net) with
      constPos := Flock.WORD + 16, inWords := 1, aT := .pure a, bT := .pure b,
      inGroups := #[⟨0, 1, #[16, 16]⟩], outGroups := #[⟨1, 1, #[16]⟩] }
  | _, _ => default

private def toyPort (name : String) : Flock.Port := { name, words := 1, chunks := 1, nb := 1, v2 := true }

/-- A file whose unit reads rows `x` and `y`, a word each, through the rows `r`, and writes the output row `o`. -/
private def toyFile (x y : String) (r : Nat → Array Nat) : Flock.Circuit :=
  { (default : Flock.Circuit) with
    nets := #[("unit", netOf r)], upv := 1, ports := #[toyPort x, toyPort y, toyPort "o"], outRowPorts := #[1],
    outPorts := #[("o", 1)], leavesIn := #[#[0, 1]], leavesOut := #[#[0]] }

/-- An ingress file whose rows read both parameters, one whose rows read only `sin`, one whose rows read an input
column no port holds, and an egress file whose rows read only `state`. -/
private def toyIngress : Flock.Circuit := toyFile "row" "sin" fun i => #[i, 16 + i]
private def toyUnread : Flock.Circuit := toyFile "row" "sin" fun i => #[16 + i]
private def toyPad : Flock.Circuit := toyFile "row" "sin" fun i => #[i, 40]
private def toyEgress : Flock.Circuit := toyFile "state" "pad" fun i => #[i]

#guard fileWhy toyP "TIngress_v1" ["row"] toyIngress == none &&
  refuses "no row of the circuit file's unit reads bit 0 of its parameter row"
    (fileWhy toyP "TIngress_v1" ["row"] toyUnread) &&
  refuses "a row of the circuit file's unit reads its input column 40, which no port holds"
    (fileWhy toyP "TIngress_v1" ["row"] toyPad) &&
  refuses "the circuit file's input rows are not its template's parameters in order"
    (fileWhy toyP "TEgress_v1" ["row"] toyIngress) &&
  refuses "the circuit file's template TOther_v1 is no Definition of the Program"
    (fileWhy toyP "TOther_v1" ["row"] toyIngress)

-- a template with no ingress row has no pass over its unit's rows: a parameter bit no row reads is not refused
#guard fileWhy toyP "TEgress_v1" [] toyEgress == none && fileWhy toyP "TIngress_v1" [] toyUnread == none &&
  fileWhy toyP "TIngress_v1" [] toyPad == none

private def digestOf (b : ByteArray) : String := Flock.hexOf (Flock.Sha512.hash b)

private def tmplFile : ByteArray := "flock-circuit/tmpl scan\nMETA {}\n".toUTF8
private def noFile : ByteArray := "x\n".toUTF8

private def toyTs (a b : ByteArray) : Array Template :=
  #[⟨"TIngress_v1", ["row"], digestOf a⟩, ⟨"TEgress_v1", [], digestOf b⟩]

#guard passes (toyTs tmplFile noFile) == 1 && passes #[] == 0 &&
  passes #[⟨"A", ["row"], ""⟩, ⟨"B", ["row"], ""⟩, ⟨"C", [], ""⟩] == 2

#guard why toyP #[] (Core.Blobs.ofArray #[]) == none &&
  refuses "scan: the circuit file of template TIngress_v1: no blob has the SHA-512"
    (why toyP (toyTs tmplFile noFile) (Core.Blobs.ofArray #[noFile])) &&
  refuses "scan: the circuit file of template TIngress_v1 is a templated file (flock-circuit/tmpl)"
    (why toyP (toyTs tmplFile noFile) (Core.Blobs.ofArray #[noFile, tmplFile])) &&
  refuses "scan: the circuit file of template TIngress_v1: circuit: not a flock-circuit file"
    (why toyP (toyTs noFile tmplFile) (Core.Blobs.ofArray #[noFile, tmplFile])) &&
  refuses "scan: two templates name one circuit file" (why toyP (toyTs noFile noFile) (Core.Blobs.ofArray #[noFile]))

end NetworkCheck.ScanFiles
