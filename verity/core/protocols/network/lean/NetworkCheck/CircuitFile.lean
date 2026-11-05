import Flock.Hash
import Flock.HmRow
import Flock.JsonParse
import Flock.ProvedScope
import Flock.Tags

/-!
# A circuit file's digest and header, read in chunks, or the file read whole

`network-period check P C E CIRCUIT…` reads each circuit file its parameters list in chunks of a fixed size: it hashes
the whole file (`Hasher`, SHA-512) and keeps only its head line, its META line and each block's keyword line and its
net's header line (`Lines`), never a net's rows, so a file of any size is read in memory of its header. `ofHeader` is
the `Flock.Circuit` those lines describe, each net its header with no rows: what `SampledProofs.Bits.check` and the
period's `circuitsWhy` read of an audit's file. The file is a flat `flock-circuit` file (`Flock.Tags.circuit`); a typed
or templated block's file is refused by name.

A scan's file is read whole instead (`whole`), by the very function a session's verifier of `Flock.Tags.circuit`'s
statement parses its circuit with (`Flock.ProvedScope.circuitOf`, `whole_circuit`), so the period reads the rows the
scan's sessions prove.
-/

namespace NetworkCheck.CircuitFile

open Lean (Json)

/-! ## SHA-512 over chunks -/

/-- SHA-512 between chunks: the chaining value, the bytes of a block not yet compressed, and the length so far. -/
structure Hasher where
  h : Array UInt64 := Flock.Sha512.iv
  buf : ByteArray := .empty
  len : Nat := 0

/-- The chunk `b` appended: every whole block compressed. -/
def Hasher.update (s : Hasher) (b : ByteArray) : Hasher := Id.run do
  let data := s.buf ++ b
  let n := data.size / 128
  let mut h := s.h
  for blk in [0:n] do h := Flock.Sha512.compress h data (128 * blk)
  return { h, buf := data.extract (128 * n) data.size, len := s.len + b.size }

/-- The digest: the last block padded as `Flock.Sha512.hash` pads a message. -/
def Hasher.finish (s : Hasher) : ByteArray := Id.run do
  let mut data := s.buf.push 0x80
  for _ in [0:(112 + 128 - data.size % 128) % 128] do data := data.push 0
  let bits := s.len * 8
  for i in [0:16] do
    data := data.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  let mut h := s.h
  for blk in [0:data.size / 128] do
    h := Flock.Sha512.compress h data (128 * blk)
  return ⟨Array.ofFn (n := 64) fun i => (h[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩

/-! ## The lines that are a file's header -/

/-- The first newline of `b` at or after `i`, or `b.size`. -/
def nextNewline (b : ByteArray) (i : Nat) : Nat := Id.run do
  let mut j := i
  while j < b.size && b[j]! != 10 do j := j + 1
  return j

/-- A file's lines as its chunks arrive: those kept (the head, META, and each block's keyword line and its net's header
line), the current line's bytes while it is kept, the lines still to skip (a net's rows after its header) and whether a
skipped line has begun, after a block's keyword line its count of lines, and the first refusal. -/
structure Lines where
  kept : Array ByteArray := #[]
  cur : ByteArray := .empty
  skip : Nat := 0
  begun : Bool := false
  block : Option Nat := none
  err : Option String := none

/-- The current line ends: it is kept, and the next line is kept, or a net's rows are skipped after its header. -/
def Lines.close (s : Lines) (kw : String) : Lines :=
  let s := { s with kept := s.kept.push s.cur, cur := .empty }
  if s.kept.size ≤ 2 then s
  else match s.block with
    | some n => { s with block := none, skip := n - 1 }
    | none =>
      let h := Flock.tokens ((String.fromUTF8? s.kept.back!).getD "")
      if h.size != 3 || h[0]! != kw then { s with err := some s!"expected a {kw} line" }
      else match h[2]!.toNat? with
        | some (n + 1) => { s with block := some (n + 1) }
        | _ => { s with err := some s!"{h[1]!}: not a {kw} block of at least one line" }

/-- The chunk `b` read: lines skipped by counting newlines, kept lines' bytes gathered. -/
def Lines.feed (s : Lines) (kw : String) (b : ByteArray) : Lines := Id.run do
  let mut s := s
  let mut i := 0
  while i < b.size && s.err.isNone do
    let j := nextNewline b i
    if j == b.size then
      if s.skip == 0 then s := { s with cur := s.cur ++ b.extract i j }
      else s := { s with begun := true }
      i := j
    else
      if 0 < s.skip then s := { s with skip := s.skip - 1, begun := false }
      else s := { s with cur := s.cur ++ b.extract i j }.close kw
      i := j + 1
  return s

/-- The kept lines at the end of the file, a last line without its newline among them, or why the file is not whole. -/
def Lines.finish (s : Lines) (kw : String) : Flock.V (Array String) := do
  let s := if 0 < s.skip then (if s.begun then { s with skip := s.skip - 1 } else s)
    else if s.cur.isEmpty then s else s.close kw
  if let some e := s.err then throw e
  if 0 < s.skip || s.block.isSome then throw "a block is truncated"
  s.kept.mapM fun l => match String.fromUTF8? l with
    | some t => pure t
    | none => throw "a header line is not UTF-8"

/-! ## The circuit its header describes -/

private def ctx : String := "circuit META"

private def num (h : Array String) (i : Nat) : Flock.V Nat :=
  match h[i]?.bind String.toNat? with
  | some n => pure n
  | none => throw s!"flock-ir-unit/v2: header field {i} is not a number"

private def groups (h : Array String) (at0 : Nat) : Flock.V (Array Flock.Group × Nat) := do
  let count ← num h at0
  let mut at_ := at0 + 1
  let mut out := #[]
  for _ in [0:count] do
    let col ← num h at_
    let words ← num h (at_ + 1)
    let np ← num h (at_ + 2)
    let mut bits := #[]
    for i in [0:np] do bits := bits.push (← num h (at_ + 3 + i))
    if !(Flock.isPow2 words && col % words == 0) then
      throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
    out := out.push ⟨col, words, bits⟩
    at_ := at_ + 3 + np
  return (out, at_)

/-- **A net from its header line**, as `Flock.Net.parse` reads and checks the header, with no rows. -/
def headerNet (line : String) : Flock.V Flock.Net := do
  let h := Flock.tokens line
  if h[0]? != some "flock-ir-unit/v2" then throw "not a flock-ir-unit/v2 circuit"
  let name := h.getD 1 ""
  let useful ← num h 2
  let constPos ← num h 3
  let (inGroups, at1) ← groups h 4
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw "flock-ir-unit/v2: trailing header fields"
  if outGroups.isEmpty then throw "flock-ir-unit/v2: no output group"
  if constPos + 1 != useful then throw "flock-ir-unit/v2: the constant is not the last useful column"
  let inWords := (inGroups.back?.map fun g => g.col + g.words).getD 0
  if inGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * Flock.WORD || g.col + g.words > inWords)
      || outGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * Flock.WORD) then
    throw "flock-ir-unit/v2: a port group's ports overrun its words, or an input group the input rows"
  if inWords * Flock.WORD > constPos then throw "flock-ir-unit/v2: the input rows reach the constant"
  return { (default : Flock.Net) with name, useful, constPos, inGroups, outGroups, inWords,
                                      unitLog := max 7 (Flock.log2ceil useful) }

private def intLists (j : Json) (k : String) : Flock.V (Array (Array Int)) := do
  (← Flock.J.arr j k ctx).mapM fun u => do (← Flock.J.asArr u s!"{ctx}.{k}").mapM fun x => Flock.J.asInt x s!"{ctx}.{k}"

/-- **META's row ports**, input rows then output rows: each one's name and words, and its bits, a bit row's length
(`row_schemas`) or 16 per word, as `Flock.HmRow.rowPorts` reads them; the rows' constants, which the C-Flock verifier
checks on the same bytes, are not read. -/
def rowPorts (mj : Json) : Flock.V (Array Flock.Port) := do
  let ports ← (← Flock.J.arr mj "ports" ctx).mapM fun p => do
    let a ← Flock.J.asArr p s!"{ctx}: a port"
    if a.size < 2 then throw s!"{ctx}: a port is not [name, words, …]"
    pure ({ name := ← Flock.J.asStr a[0]! "port name", words := ← Flock.J.asNat a[1]! "port words", chunks := 0,
            nb := 0 } : Flock.Port)
  match mj.getObjVal? "row_schemas" with
  | .error _ => pure ports
  | .ok s =>
    let s ← Flock.J.asArr s "row_schemas"
    if s.size != ports.size then throw "META: row_schemas is not one [schema, bits] per port"
    (ports.zip s).mapM fun (p, x) => do
      let a ← Flock.J.asArr x "row_schemas"
      if a.size != 2 then throw "META: row_schemas: a port's is not [schema, bits]"
      pure { p with bits := ← Flock.J.asNat a[1]! "a bit row's length", v2 := true }

/-- **META's output rows**: each output row's run of output ports (`out_row_ports`), else one row a port but the pinned
ones (`Flock.HmRow.pinnedCount`), as `Flock.HmRow` reads them. -/
def outRows (mj : Json) (outPorts : Nat) : Flock.V (Array Nat) := do
  match mj.getObjVal? "out_row_ports" with
  | .error _ => pure (Array.replicate (outPorts - Flock.HmRow.pinnedCount mj) 1)
  | .ok v => (← Flock.J.asArr v "out_row_ports").mapM (Flock.J.asNat · "out_row_ports")

/-- **The circuit file's header as a `Flock.Circuit`**: its name, SHA-512 `sha`, nets (each its header, no rows), units
per VU, unit net, the net returning the outputs (META `out_net`, the unit by default), row ports (`rowPorts`), output
ports and output rows (`outRows`), leaf maps, wires and leaf cuts (META `wires` and `leaf_cuts`, read as
`Flock.Circuit.parse` and `Flock.HmRow` read them) and META. A file of another format than `Flock.Tags.circuit`'s, a
typed template's (`flock-circuit/types`) among them, is refused. -/
def ofHeader (sha : ByteArray) (lines : Array String) : Flock.V Flock.Circuit := do
  let tags := Flock.Tags.circuit
  let head := (lines.getD 0 "").trimAsciiEnd.toString
  if !head.startsWith (tags.fileFormat ++ " ") then
    throw s!"not a {tags.fileFormat} file: the period reads flat circuit files, not a typed or templated block's"
  let name := (head.drop (tags.fileFormat.length + 1)).toString
  let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
  if !metaLine.startsWith "META " then throw "no META line"
  let mj ← match Flock.JsonParse.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  if (← Flock.J.str mj "format" ctx) != tags.fileFormat || (← Flock.J.str mj "statement" ctx) != tags.statement then
    throw s!"META: not {tags.statement}"
  let mut nets : Array (String × Flock.Net) := #[]
  for i in [0:(lines.size - 2) / 2] do
    let nm := (Flock.tokens (lines.getD (2 + 2 * i) "")).getD 1 ""
    match headerNet (lines.getD (3 + 2 * i) "") with
    | .ok n => nets := nets.push (nm, n)
    | .error e => throw s!"{nm}: {e}"
  let some unitNet := nets.findIdx? (·.1 == "unit") | throw "no unit circuit"
  let outName := (mj.getObjValAs? String "out_net").toOption.getD "unit"
  let some outNet := nets.findIdx? (·.1 == outName) | throw s!"META: unknown circuit {outName}"
  let leavesOut ← (← intLists mj "leaves_out").mapM fun u => u.mapM fun x =>
    if x < 0 then (throw "META: a negative output leaf" : Flock.V Nat) else pure x.toNat
  let netOf := fun (n : String) => match nets.findIdx? (·.1 == n) with
    | some i => (pure i : Flock.V Nat)
    | none => throw s!"META: unknown circuit {n}"
  let wires ← match mj.getObjVal? "wires" with
    | .ok w => (← Flock.J.asArr w "wires").mapM fun x => do
      let a ← Flock.J.asArr x "wire"
      if a.size != 6 then throw "META: a wire is not six fields"
      pure (⟨(← netOf (← Flock.J.asStr a[0]! "wire"), ← Flock.J.asNat a[1]! "wire", ← Flock.J.asNat a[2]! "wire"),
        (← netOf (← Flock.J.asStr a[3]! "wire"), ← Flock.J.asNat a[4]! "wire", ← Flock.J.asNat a[5]! "wire")⟩ :
        Flock.Wire)
    | .error _ => pure #[]
  let leafCuts ← match mj.getObjVal? "leaf_cuts" with
    | .ok (.arr a) => a.mapM fun w => do
      let x ← Flock.J.asArr w "a leaf cut"
      if x.size != 4 then throw "META: a leaf cut is not [circuit, unit, port, leaf]"
      if (x[0]!.getStr?).toOption != some "unit" then throw "META: a leaf cut into anything but a unit"
      pure (← Flock.J.asNat x[1]! "a leaf cut", ← Flock.J.asNat x[2]! "a leaf cut", ← Flock.J.asNat x[3]! "a leaf cut")
    | _ => pure #[]
  let outPorts ← (← Flock.J.arr mj "out_ports" ctx).mapM fun p => do
    let a ← Flock.J.asArr p "out port"
    if a.size != 2 then throw "META: an out port is not [name, words]"
    pure (← Flock.J.asStr a[0]! "out port name", ← Flock.J.asNat a[1]! "out port words")
  return { (default : Flock.Circuit) with name, sha, nets, upv := ← Flock.J.nat mj "units_per_vu" ctx, unitNet,
                                          outNet, ports := ← rowPorts mj, outPorts,
                                          outRowPorts := ← outRows mj outPorts.size, leavesIn := ← intLists mj "leaves_in",
                                          leavesOut, wires, leafCuts, metaJ := mj }

/-- A whole file's SHA-512 and its header's circuit, from its hash and lines. -/
def result (hs : Hasher) (ls : Lines) : Flock.V (ByteArray × Flock.Circuit) := do
  let sha := hs.finish
  pure (sha, ← ofHeader sha (← ls.finish Flock.Tags.circuit.blockKeyword))

/-- The circuit file's bytes `b`, read in chunks of `chunk ≥ 1` bytes: its SHA-512 and its header's circuit, or why not. -/
def ofChunks (b : ByteArray) (chunk : Nat) : Flock.V (ByteArray × Flock.Circuit) :=
  let (hs, ls) := (List.range ((b.size + chunk - 1) / chunk)).foldl (fun (hs, ls) k =>
    let c := b.extract (k * chunk) (k * chunk + chunk)
    (hs.update c, ls.feed Flock.Tags.circuit.blockKeyword c)) (({} : Hasher), ({} : Lines))
  result hs ls

/-- **Read the circuit file at `path`** in chunks of `chunk` bytes, as `ofChunks` does: its SHA-512 and its header's
circuit, or why not. -/
def read (path : System.FilePath) (chunk : Nat := 1 <<< 20) : IO (Flock.V (ByteArray × Flock.Circuit)) := do
  let h ← IO.FS.Handle.mk path .read
  let mut hs : Hasher := {}
  let mut ls : Lines := {}
  repeat
    let c ← h.read chunk.toUSize
    if c.isEmpty || ls.err.isSome then break
    hs := hs.update c
    ls := ls.feed Flock.Tags.circuit.blockKeyword c
  return result hs ls

/-! ## A file read whole -/

/-- **The circuit file's bytes read whole**, as a session's verifier of `Flock.Tags.circuit`'s statement reads its circuit
(`Flock.ProvedScope.circuitOf`, no tables: every net's rows parsed and the rows checked): its SHA-512 and the circuit, or
why not. -/
def whole (bytes : ByteArray) : Flock.V (ByteArray × Flock.Circuit) :=
  (Flock.ProvedScope.circuitOf Flock.Tags.circuit bytes []).map fun c => (c.sha, c)

/-- `whole`'s circuit is the session verifier's own parse of the same bytes. -/
theorem whole_circuit (bytes : ByteArray) :
    (whole bytes).map (·.2) = Flock.ProvedScope.circuitOf Flock.Tags.circuit bytes [] := by
  unfold whole
  cases Flock.ProvedScope.circuitOf Flock.Tags.circuit bytes [] <;> rfl

/-- **Read the circuit file at `path` whole** (`whole`). -/
def readWhole (path : System.FilePath) : IO (Flock.V (ByteArray × Flock.Circuit)) :=
  return whole (← IO.FS.readBinFile path)

/-! ## Checks -/

private def bytes (n : Nat) : ByteArray := ⟨(Array.range n).map (·.toUInt8)⟩

#guard [0, 1, 111, 112, 127, 128, 129, 300].all fun n => [0, 1, 64, 127, 128, 200].all fun k =>
  let m := bytes n
  ((({} : Hasher).update (m.extract 0 k)).update (m.extract k n)).finish.data == (Flock.Sha512.hash m).data

/-- A unit net of one input word holding two 16-bit ports, sixteen computed rows on them, one on the first computed
row, and the constant: 146 useful rows, its output group the computed rows' word. -/
private def toyNet : Array String :=
  #["flock-ir-unit/v2 toy 146 145 1 0 1 2 16 16 1 1 1 1 16"] ++ (Array.range 146).map fun i =>
    if i < 32 then s!"1 {i} 1 {i}" else if i < 128 then "0 0" else if i < 144 then s!"2 {i - 128} {i - 112} 1 145"
    else if i == 144 then "1 128 1 145" else "1 145 1 145"

#guard match Flock.Net.parse toyNet, headerNet toyNet[0]! with
  | .ok a, .ok b => a.name == b.name && a.useful == b.useful && a.constPos == b.constPos && a.inWords == b.inWords &&
      a.unitLog == b.unitLog && (a.inGroups ++ a.outGroups).map (fun g => (g.col, g.words, g.bits)) ==
        (b.inGroups ++ b.outGroups).map (fun g => (g.col, g.words, g.bits))
  | _, _ => false

/-- A file of that unit net at two units a VU, each instance returning its own output word, an input row `x` of four
words and an output row `o` of two, its META's other fields `extra`. -/
private def toyFileWith (extra : List (String × Json)) : ByteArray :=
  let ns (xs : List Nat) : Json := Json.arr (xs.toArray.map Lean.toJson)
  let mj := Json.mkObj ([("format", .str Flock.Tags.circuit.fileFormat),
    ("statement", .str Flock.Tags.circuit.statement), ("units_per_vu", Lean.toJson (2 : Nat)),
    ("ports", Json.arr #[Json.arr #["x", Lean.toJson (4 : Nat), Lean.toJson (0 : Nat)],
      Json.arr #["o", Lean.toJson (2 : Nat), Lean.toJson (0 : Nat)]]),
    ("out_ports", Json.arr #[Json.arr #["o", Lean.toJson (2 : Nat)]]),
    ("leaves_in", Json.arr #[ns [0, 1], ns [2, 3]]), ("leaves_out", Json.arr #[ns [0], ns [1]])] ++ extra)
  (String.intercalate "\n" ([s!"{Flock.Tags.circuit.fileFormat} toy", s!"META {mj.compress}",
    s!"{Flock.Tags.circuit.blockKeyword} unit {toyNet.size}"] ++ toyNet.toList) ++ "\n").toUTF8

private def toyFile : ByteArray := toyFileWith []

private def jn (k : Nat) : Json := Lean.toJson k

#guard match ofChunks (toyFileWith [("wires", Json.arr #[Json.arr #["unit", jn 1, jn 2, "unit", jn 0, jn 0]]),
    ("leaf_cuts", Json.arr #[Json.arr #["unit", jn 0, jn 2, jn 5]])]) 64 with
  | .ok (_, c) => c.wires.map (fun w => (w.dst, w.src)) == #[((0, 1, 2), (0, 0, 0))] && c.leafCuts == #[(0, 2, 5)]
  | .error _ => false
#guard match ofChunks (toyFileWith [("wires", Json.arr #[Json.arr #["stage", jn 1, jn 2, "unit", jn 0, jn 0]])]) 64 with
  | .error e => e == "META: unknown circuit stage"
  | .ok _ => false

#guard [1, 7, 64, 1000, toyFile.size].all fun k => match ofChunks toyFile k with
  | .ok (sha, c) => sha.data == (Flock.Sha512.hash toyFile).data && c.name == "toy" && c.upv == 2 && c.unitNet == 0 &&
      c.outNet == 0 && c.unit.constPos == 145 && c.leavesOut == #[#[0], #[1]] && c.leavesIn.size == 2 &&
      c.ports.map (fun p => (p.name, p.words, p.bits)) == #[("x", 4, 64), ("o", 2, 32)] && c.inWords == 6 &&
      c.outPorts == #[("o", 2)] && c.outRowPorts == #[1]
  | .error _ => false
#guard match ofChunks (toyFileWith [("row_schemas", Json.arr #[Json.arr #["hm96-sha512/row/v2", jn 60],
    Json.arr #["hm96-sha512/row/v2", jn 32]]), ("out_row_ports", Json.arr #[jn 1])]) 64 with
  | .ok (_, c) => c.ports.map (·.bits) == #[60, 32] && c.outRowPorts == #[1]
  | .error _ => false
#guard match ofChunks (toyFile.extract 0 (toyFile.size - 40)) 64 with
  | .error e => e == "a block is truncated"
  | .ok _ => false
#guard match ofChunks ("flock-circuit/types toy\n").toUTF8 8 with
  | .error e => e.startsWith "not a flock-circuit file"
  | .ok _ => false
#guard match whole toyFile with
  | .error _ => true
  | .ok _ => false

end NetworkCheck.CircuitFile
