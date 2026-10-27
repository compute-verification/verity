/-
PROTOCOL.md §16.10 (registered reads), a value the verifier registers at known content (`Registered.KNOWN_KEY`): the known
content binds a statement's row only where the circuit reads or returns it. `check` refuses a circuit one of whose ports
the `--registered` file names at known content is not a `sha512/row/v2` row of the known rows' words, or has a row bit
that no computed row of the unit reads (an input row) or no copy returns (an output row): the statement would then not
depend on that bit, and its known content would bind nothing there.

For each such bit it finds a copy and a column holding it (`cands`, `outCands`, searches the proof does not rely on) and
accepts the bit only where the parsed circuit itself says so. At an input row: the copy is one of the unit's (`upv`), a
computed row reads the column (`RecResiduals.readIns`), and Δ's source key at the column is that bit of the port
(`srcMsg`, which mirrors the proof's `srcKey` at a message bit from a leaf). At an output row: the copy is one of the
unit's, one of its leaf's outputs is the bit's output word (`outAt`), and the bit's column of the output group is a
computed one. Its cost is the known bits times the unit's ports, never the unit's gates.

A port the file marks `read_in_full` (`READ_KEY`) is checked the same way, as one row of its registered row's words
(`fullRows`): its every bit is read, at the registered width, so two statements that read one registered row in full
depend on the same bits of it.
-/
import Flock.RegKnown
import Flock.RecResiduals
import Flock.ProvedScope
import Flock.TmplReads

namespace Flock
namespace KnownRead

/-- Each row port's first word among the message's words (`HmRow.delta`'s `starts`). -/
def starts (c : Circuit) : Array Nat :=
  (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1

/-- The input port an input column `off` is a bit of: the first whose columns hold it. -/
def portOf (pc : Array (Nat × Nat)) (off : Nat) : Nat :=
  ((List.range pc.size).find? fun p => decide (pc[p]!.1 ≤ off ∧ off < pc[p]!.1 + pc[p]!.2)).getD 0

/-- **The message bit copy `k`'s input port `p`, bit `t` is copied from**, where Δ copies one into it from the leaf word
the copy's leaves name (`leavesIn`): bit `bit` of byte `byte` of row port `q`; none where Δ copies the zero, and at a port
past the leaf group (a cut's or a wire's, which a statement of one input group has none of: `scopeOkH`), where it counts
no bit as read. -/
def srcAtMsg (c : Circuit) (k p t : Nat) : Option (Nat × Nat × Nat) :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    if (c.leavesIn.getD k #[])[p]! < 0 then none else
      let x := ((c.leavesIn.getD k #[])[p]!).toNat
      let q := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ x)).getD 0
      if 16 * (x - (starts c)[q]!) + t < c.ports[q]!.bits then
        some (q, 2 * (x - (starts c)[q]!) + t / 8, t % 8)
      else none
  else none

/-- **The message bit copy `u`'s input column `y` is copied from**, at the port bit it is (`mask`, the unit's
`ProvedScope.portMask`; `pc`, its `portCols`); none at an input padding column. -/
def srcMsg (c : Circuit) (mask : Array Bool) (pc : Array (Nat × Nat)) (u y : Nat) : Option (Nat × Nat × Nat) :=
  if mask.getD y false then srcAtMsg c u (portOf pc y) (y - pc[portOf pc y]!.1) else none

/-- **Where each message word sits among the copies' leaf inputs**: word `x`'s `(copy, input port)` pairs, every copy
below `upv` and every port of the unit's leaf group whose leaf is `x`. -/
def cands (c : Circuit) (words : Nat) : Array (List (Nat × Nat)) :=
  let nb := (c.unit.inGroups.getD 0 default).bits.size
  (List.range c.upv).foldl (fun m u =>
    (List.range nb).foldl (fun m p =>
      let l := (c.leavesIn.getD u #[]).getD p (-1)
      if l < 0 then m else m.modify l.toNat ((u, p) :: ·)) m) (Array.replicate words [])

/-- **Bit `k` of row port `q` is read**: at one of the candidates `cs`, copy `u` is a copy of the unit, a computed row
reads its input column `y` (`rd`, `RecResiduals.readIns`), and Δ copies bit `k` of `q` into it (`srcMsg`). -/
def bitRead (c : Circuit) (mask : Array Bool) (pc : Array (Nat × Nat)) (rd : Array Bool) (cs : List (Nat × Nat))
    (q k : Nat) : Bool :=
  cs.any fun (u, p) =>
    let y := pc[p]!.1 + k % 16
    decide (u < c.upv) && rd.getD y false && decide (srcMsg c mask pc u y = some (q, k / 8, k % 8))

/-- Output port `j`'s first word among the circuit's output words. -/
def outAt (c : Circuit) (j : Nat) : Nat := ((c.outPorts.toList.take j).map (·.2)).sum

/-- **Where each output word sits among the copies' returned outputs**: word `w`'s `(copy, output)` pairs, every copy
below `upv` and every output of its leaf whose word is `w`. -/
def outCands (c : Circuit) : Array (List (Nat × Nat)) :=
  (List.range c.upv).foldl (fun m u =>
    let lo := c.leavesOut.getD u #[]
    (List.range lo.size).foldl (fun m j => m.modify lo[j]! ((u, j) :: ·)) m) (Array.replicate c.outWords [])

/-- **Bit `k` of output row port `q` is returned**: its word is word `k / 16` of output port `q - nIn`, and at one of the
candidates `cs` copy `u` of the unit returns that word as its output `j`, bit `k % 16` at a computed column of the
unit's output group. -/
def bitOut (c : Circuit) (cs : List (Nat × Nat)) (q k : Nat) : Bool :=
  let w := outAt c (q - HmOut.nIn c) + k / 16
  decide (w < outAt c (q - HmOut.nIn c + 1)) && cs.any fun (u, j) =>
    let col := (c.unit.outGroups.getD 0 default).col * WORD + (16 * j + k % 16)
    decide (u < c.upv) && decide (j < (c.leavesOut.getD u #[]).size) && decide ((c.leavesOut.getD u #[])[j]! = w) &&
      decide (c.unit.inWords * WORD ≤ col) && decide (col < c.unit.constPos)

/-- **A known port's row is a known value's**: a `sha512/row/v2` row without segments, of each known row's words. -/
def shapeOk (p : Port) (ks : Array (Array Nat × ByteArray)) : Bool :=
  p.v2 && p.segs.isNone && ks.all (·.1.size == p.words)

/-- **Each known value names one port of the circuit**: where the `--registered` file names known content, the circuit
names each of its ports once (`Registered.dupName`), and has a port of each name the known content names. -/
def namesOk (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) : V Unit :=
  if known.isEmpty then pure ()
  else match Registered.dupName c with
    | some q => throw s!"registered reads: the circuit names port {c.ports[q]!.name} twice, so its known content names no one row"
    | none => known.toList.forM fun (name, _) =>
        if c.ports.any (·.name == name) then pure ()
        else throw s!"registered reads: known port {name}: the circuit has no port {name}, so its known content binds nothing"

/-- **Every bit of every known port is read or returned**, the input columns read taken from `rd`: each port of the
circuit that the `--registered` file names at known content is a `sha512/row/v2` row without segments of each known row's
words, and has each bit of its `16 · words`-bit row read by a computed row of the unit at some copy (`bitRead`), at an
input row (`HmOut.nIn`), or returned by some copy (`bitOut`), at an output row. -/
def readsOkAt (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) (rd : Array Bool) : V Unit :=
  let mask := ProvedScope.portMask c.unit.inGroups (c.unit.inWords * WORD)
  let pc := portCols c.unit.inGroups
  let st := starts c
  let cs := cands c (c.ports.foldl (· + ·.words) 0)
  let co := outCands c
  known.toList.forM fun (name, ks) =>
    (List.range c.ports.size).forM fun q =>
      if c.ports[q]!.name = name then
        if !shapeOk c.ports[q]! ks then
          throw s!"registered reads: known port {name}: the circuit's row is not a {HmRow.ROW_SCHEMA_V2} row without segments of its known rows' {c.ports[q]!.words} words"
        else if q < HmOut.nIn c then
          match (List.range (16 * c.ports[q]!.words)).find?
              (fun k => !bitRead c mask pc rd (cs.getD (st[q]! + k / 16) []) q k) with
          | some k =>
            throw s!"registered reads: known port {name}: no computed row of the circuit reads bit {k} of its row, so its known content binds nothing there"
          | none => pure ()
        else
          match (List.range (16 * c.ports[q]!.words)).find?
              (fun k => !bitOut c (co.getD (outAt c (q - HmOut.nIn c) + k / 16) []) q k) with
          | some k =>
            throw s!"registered reads: known port {name}: no copy of the unit returns bit {k} of its row, so its known content binds nothing there"
          | none => pure ()
      else pure ()

/-- The input columns a templated unit's description says a computed row surely reads (`Tmpl.readsSure`), where the
circuit's unit is its template's net (by pointer, else compared) of at most `2^32` rows (`ProvedScope`'s bound); `none`
elsewhere. -/
def sureReads (c : Circuit) : Option (Array Bool) :=
  match c.tmpl with
  | some tu =>
    if decide (c.unit.unitLog ≤ 32) &&
        withPtrEq c.unit tu.net (fun _ => decide (c.unit = tu.net)) (fun h => by simp [h]) then
      some tu.t.readsSure
    else none
  | none => none

/-- **Every bit of every known port is read or returned** (`readsOkAt`): at the input columns a templated unit's
description surely reads (`sureReads`), so its rows are not built, else, and where those don't cover every known bit, at
the input columns its computed rows read (`RecResiduals.readIns`). -/
def readsOk (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) : V Unit :=
  match sureReads c with
  | some rd =>
    match readsOkAt c known rd with
    | .ok () => pure ()
    | .error _ => readsOkAt c known (RecResiduals.readIns c.unit)
  | none => readsOkAt c known (RecResiduals.readIns c.unit)

/-- **The known content binds the circuit's rows** (PROTOCOL.md §16.10): each known value names one port of the
circuit (`namesOk`), and every bit of every known port is read or returned (`readsOk`). -/
def check (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) : V Unit :=
  namesOk c known >>= fun _ => readsOk c known

open Lean (Json)

/-- The key of a port entry whose row a computed row of the unit reads in every bit: `true`, or absent (false). -/
def READ_KEY : String := "read_in_full"

/-- **The ports a `--registered` file names read in full**: each port entry whose `read_in_full` (`READ_KEY`) is true. -/
def readOf (j : Json) : V (Array String) := do
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw "registered reads: ports is not an object"
  kvs.toArray.filterMapM fun (name, e) =>
    match e.getObjVal? READ_KEY with
    | .error _ => pure none
    | .ok (.bool b) => pure (if b then some name else none)
    | .ok _ => throw s!"registered reads: port {name}: {READ_KEY} is true or false"

/-- A registered row's u16 words: `words` leaves of `wordBits` bits each. -/
def regWords (reg : Registered.Port) : Nat := reg.words.getD 0 * (reg.wordBits / 16)

/-- **Ports read in full, as `check` reads a known port**: each named port as one row of its registered row's u16
words (`regWords` of `own`'s first port of that name), zero, a row no known-content check reads. So `check` holds the
circuit's port to a `sha512/row/v2` row without segments of the registered row's words, read in every bit by a computed
row of the unit: two statements that read one registered value's rows in full read them at one width, each bit. -/
def fullRows (own : Array Registered.Port) (names : Array String) : Array (String × Array (Array Nat × ByteArray)) :=
  names.map fun nm => (nm, #[(Array.replicate (((own.toList.find? (·.name == nm)).map regWords).getD 0) 0, ByteArray.empty)])

end KnownRead
end Flock
