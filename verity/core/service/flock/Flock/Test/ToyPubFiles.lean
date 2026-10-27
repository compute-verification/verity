/-
A toy `verify --zk` session's files and what they parse to: one unit of `toy-copy`, whose input port `x` and output port
`y` are rows of 64 words, and whose one computed output bit is `x`'s bit 0 (`y`'s bit 0; its other fifteen output bits
are zero). `circuitFile` is V*'s emission of that unit (`VStar.emitCircuit`, as `RecVStar.circuitFile` emits a unit),
`publicFile` one instance's public file (unit 0, zero rows, its outputs hidden), `draw` a `subset` draw of that one unit.

`circuit` and `public` state what the files parse to: every field the analysis reads is written out (the unit net's
rows, the ports, the copies, the leaves), and the rest (the circuit's digests, its META, its ranges and wires, the
`sha512x3` and `hm96` nets the verifier builds itself, the public file's header and digest) is read off the parse, so
that `Desc.parse … circuitFile [] = .ok (.single circuit)` says the written fields are the file's. `Flock.Test.ToyPub`
checks that, with the public file's load and the draw, by running the verifier's functions on the files.
-/
import Flock.RecVStar
import Flock.Witness
import Flock.Desc
import Flock.ProvedScope
import Flock.Registered

namespace Flock.Test.ToyPubFiles

open Lean Flock.HmNets

/-- The unit: 1024 input bits (`x`), one output word whose bit 0 is input bit 0 and the rest zero. -/
def unit : VStar.UnitC :=
  let r := (input 1024).run St.empty
  { name := "toy-copy", inPorts := [64], outPorts := [1], st := r.2,
    outs := #[(r.1.headD Lin.zero)] ++ Array.replicate 15 Lin.zero }

def hex128 : String := "".pushn 'a' 128

def bindings : VStar.Bindings := { partition := hex128, program := hex128, programDigests := Json.mkObj [] }

def sub : VStar.Sub := { id := "toy", template := "toy", inputs := ["x"], outputs := ["y"] }

/-- V*'s emission of the unit. -/
def circuitFile : ByteArray :=
  match (do
      let l ← unit.laid
      let e ← VStar.emitCircuit (m := StateM String) (fun x => modify (· ++ x)) unit l sub bindings
      pure (Id.run (e.run "")).2.toUTF8 : Except String ByteArray) with
  | .ok b => b
  | .error _ => ByteArray.empty

/-- The tags `verify --zk` sets the session up under, at public file `pf`. -/
def tagsAt (pf : ByteArray) : Tags := Witness.zkTags ⟨Tags.circuit, circuitFile, pf, 1, [], none, none⟩

/-- What the circuit file parses to, as the verifier parses it. -/
def parsed : Circuit := match Desc.parse (tagsAt ByteArray.empty) circuitFile [] with
  | .ok D => D.unroll
  | .error _ => default

/-- The unit net's A rows: the 1024 input rows, the copy (row 1024 reads input column 0), and the constant. -/
def rowsA : Array (Array Nat) :=
  (Array.range 1024).map (#[·]) ++ #[#[0]] ++ Array.replicate 127 #[] ++ #[#[1152]]

/-- The unit net's B rows: the 1024 input rows, the 16 output rows against the constant, and the constant. -/
def rowsB : Array (Array Nat) :=
  (Array.range 1024).map (#[·]) ++ Array.replicate 16 #[1152] ++ Array.replicate 112 #[] ++ #[#[1152]]

/-- A column's four little-endian bytes, as `pushLe32` writes them. -/
def le32 (c : Nat) : List UInt8 :=
  [(c % 256).toUInt8, (c / 256 % 256).toUInt8, (c / 65536 % 256).toUInt8, (c / 16777216 % 256).toUInt8]

/-- Each row's first byte, rows packed from `base`. -/
def starts (base : Nat) : List (Array Nat) → List Nat
  | [] => []
  | r :: rs => base :: starts (base + 4 * r.size) rs

/-- What `Sparse.ofRows rows 2048` builds, written out: every row's columns' bytes in order, and each row's start and
length. Stated without `ofRows`'s loop, so that the kernel reads a row of it by indexing. -/
def sparseOf (rows : Array (Array Nat)) : Sparse :=
  ⟨rows.size, 2048, ⟨(rows.toList.flatMap fun r => r.toList.flatMap le32).toArray⟩, (starts 0 rows.toList).toArray,
    rows.map Array.size⟩

/-- The unit net. -/
def unitNet : Net where
  name := "toy-copy"
  useful := 1153
  constPos := 1152
  inGroups := #[⟨0, 8, Array.replicate 64 16⟩]
  outGroups := #[⟨8, 1, #[16]⟩]
  inWords := 8
  aT := .pure (sparseOf rowsA)
  bT := .pure (sparseOf rowsB)
  unitLog := 11

def portX : Port := { name := "x", words := 64, chunks := 1, nb := 0 }
def portY : Port := { name := "y", words := 64, chunks := 1, nb := 0 }

/-- What the circuit file parses to: the written fields, and the rest off the parse. -/
def circuit : Circuit :=
  { parsed with
    name := "toy-copy"
    nets := #[("unit", unitNet), parsed.nets.getD 1 default, parsed.nets.getD 2 default]
    ports := #[portX, portY]
    outPorts := #[("y", 1)]
    outRowPorts := #[1]
    g := 4
    kLog := 22
    upv := 1
    unitNet := 0
    leavesIn := #[(Array.range 64).map Int.ofNat]
    leavesOut := #[#[0]]
    key := ByteArray.empty
    dummyOutputs := #[0]
    leafScheme := "hm96-sha512/v1"
    perVu := #[1, 1, 2]
    shaNet := 1
    hmNet := 2
    outNet := 0
    outNetPorts := #[]
    outNetSlots := none
    leafCuts := #[]
    typed := none
    cls := none
    tmpl := none }

/-- One instance's public file: unit 0, with zero `b ‖ c` for both ports and, its outputs hidden, no output word. -/
def publicFile : ByteArray :=
  let t := tagsAt ByteArray.empty
  let header := Json.mkObj [("format", Json.str t.publicFormat), ("statement", Json.str t.statement),
    (t.publicCircuitKey, Json.str (hexOf parsed.sha)), ("instances", Lean.toJson (1 : Nat)), ("rows", Json.bool false),
    ("units", HmRow.unitsEntry parsed #[0])]
  (header.compress ++ "\n").toUTF8 ++ ⟨Array.replicate (2 * HmRow.BC_BYTES) 0⟩

/-- The session's tags. -/
def tags : Tags := tagsAt publicFile

/-- What the public file loads to at `circuit`, as the verifier loads it. -/
def loaded : Public := match HmOut.loadPublic tags circuit publicFile with
  | .ok p => p
  | .error _ => default

/-- What the public file loads to: one instance, its two zero `b ‖ c`, no output word and no shared rows, and its
header and digest off the load. -/
def pub : Public :=
  { loaded with
    n := 1
    digests := #[⟨Array.replicate HmRow.BC_BYTES 0⟩, ⟨Array.replicate HmRow.BC_BYTES 0⟩]
    outputs := #[]
    population := none
    tables := none }

/-- Port `q`'s registered root over the public file's one row: its frame-v3 root, at its name as the domain and its leaf
schema, over the `hm96` tree leaf of its `b ‖ c`. -/
def root (q : Nat) : ByteArray :=
  let dom := ((circuit.ports[q]?.map (·.name)).getD "").toUTF8
  FV3.root dom ((Array.range 1).map fun u =>
    Registered.leafAs ((circuit.ports[q]?.map Registered.portSchema).getD HmRow.ROW_SCHEMA_V2) dom u
      (Hm96.Default512.treeLeaf (pub.digest u q circuit.ports.size))) Sha512.hash

/-- A `subset` draw of the one unit, as a record carries it (`norm`). -/
def draw : Json :=
  norm (Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson (1 : Nat)), ("k", Lean.toJson (1 : Nat)),
    ("units", Json.arr #[Lean.toJson (0 : Nat)])])

end Flock.Test.ToyPubFiles
