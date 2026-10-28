/-
`KnownRead.check` on a unit of one input word, eight 16-bit ports: circuit ports `k0` (2 words) and `k1` (1 word),
copy 0's leaf ports 0–2 their three words, the rest forced zeros, and an output row `o` (1 word). A known input port passes
where the unit's computed rows read every bit of its row, at some copy, and is refused at the first bit none reads; a
port the file does not name is not checked. A known name no port has, or a circuit naming a port twice, is refused where
the file names known content, and not checked where it names none. A bit read only at a copy past `upv`, or only
through another copy's leaf, is not read. A known output row passes where some copy returns each of its bits at a
computed column, and is refused at the first bit none does. A known port that is no `sha512/row/v2` row, or whose
known rows are not its words, is refused. A port marked `read_in_full` is checked as one row of its registered row's
words (`fullRows`): refused at another width, or at a bit no computed row reads.
-/
import Flock.KnownRead

namespace Flock.Test.KnownRead

open Flock

/-- A net of one input word whose computed rows read, in A, the columns of `reads` (one row each, B empty), its inputs
eight 16-bit ports, its outputs the group `og`. -/
def net (reads : List (List Nat)) (og : Array Group := #[]) : Net :=
  let nIn := WORD
  let rows := Array.replicate nIn #[] ++ (reads.map List.toArray).toArray
  let w := nIn + reads.length + 1
  match Sparse.ofRows rows w, Sparse.ofRows (Array.replicate rows.size #[]) w with
  | .ok a, .ok b =>
    let inGroups : Array Group := #[⟨0, 1, Array.replicate 8 16⟩]
    { (default : Net) with constPos := nIn + reads.length, inWords := 1, aT := .pure a, bT := .pure b, inGroups := inGroups,
                           outGroups := og }
  | _, _ => default

/-- Copy `u`'s leaves: `leaves u` for each copy, words 0–2 at its first three ports by default; with `outs`, an output
row `o` of one word, which copy `u` returns as its outputs `louts u`, from column 128 of the unit (word 1). -/
def circ (reads : List (List Nat)) (upv : Nat := 1)
    (leaves : List (Array Int) := [#[0, 1, 2, -1, -1, -1, -1, -1]]) (outs : Bool := false)
    (louts : List (Array Nat) := [#[0]]) (v2 : Bool := true) : Circuit :=
  { (default : Circuit) with
    nets := #[("unit", net reads (if outs then #[⟨1, 1, #[16]⟩] else #[]))]
    ports := #[{ name := "k0", words := 2, chunks := 1, nb := 1, v2 }, { name := "k1", words := 1, chunks := 1, nb := 1, v2 }]
      ++ (if outs then #[{ name := "o", words := 1, chunks := 1, nb := 1, v2 }] else #[])
    outRowPorts := if outs then #[1] else #[]
    outPorts := if outs then #[("o", 1)] else #[]
    upv, leavesIn := leaves.toArray, leavesOut := if outs then louts.toArray else #[] }

def known (names : List String) : Array (String × Array (Array Nat × ByteArray)) :=
  names.toArray.map fun n => (n, #[])

def why (c : Circuit) (names : List String) : String :=
  match KnownRead.check c (known names) with
  | .error e => e
  | .ok () => ""

def why' (c : Circuit) (known : Array (String × Array (Array Nat × ByteArray))) : String :=
  match KnownRead.check c known with
  | .error e => e
  | .ok () => ""

#guard KnownRead.srcMsg (circ []) (ProvedScope.portMask (net []).inGroups WORD) (portCols (net []).inGroups) 0 21 ==
  some (0, 2, 5)
#guard KnownRead.srcMsg (circ []) (ProvedScope.portMask (net []).inGroups WORD) (portCols (net []).inGroups) 0 40 ==
  some (1, 1, 0)
#guard KnownRead.srcMsg (circ []) (ProvedScope.portMask (net []).inGroups WORD) (portCols (net []).inGroups) 0 50 == none

-- every bit of both rows read
#guard (KnownRead.check (circ [List.range 48]) (known ["k0", "k1"])).toBool
#guard (KnownRead.check (circ [List.range 24, List.range' 24 24]) (known ["k0", "k1"])).toBool
-- k1's last bit unread: refused by name, and k0 alone still passes
#guard why (circ [List.range 47]) ["k0", "k1"] ==
  "registered reads: known port k1: no computed row of the circuit reads bit 15 of its row, so its known content binds nothing there"
#guard (KnownRead.check (circ [List.range 32]) (known ["k0"])).toBool
#guard why (circ [List.range' 1 47]) ["k0"] ==
  "registered reads: known port k0: no computed row of the circuit reads bit 0 of its row, so its known content binds nothing there"
-- no known port passes; a name no port has is refused
#guard (KnownRead.check (circ []) (known [])).toBool
#guard why (circ [List.range 48]) ["k0", "x"] ==
  "registered reads: known port x: the circuit has no port x, so its known content binds nothing"
-- a port named twice: refused where the file names known content, of whichever port
#guard why { circ [List.range 48] with ports := (circ []).ports.push { name := "k0", words := 1, chunks := 1, nb := 1 } }
    ["k1"] ==
  "registered reads: the circuit names port k0 twice, so its known content names no one row"
#guard (KnownRead.check { circ [] with ports := (circ []).ports.push { name := "k0", words := 1, chunks := 1, nb := 1 } }
  (known [])).toBool
-- k1's word held only at copy 1's leaves: read where copy 1 is a copy of the unit, not past `upv`
#guard (KnownRead.check (circ [List.range 48] 2 [#[0, 1, -1, -1, -1, -1, -1, -1], #[-1, -1, 2, -1, -1, -1, -1, -1]])
  (known ["k1"])).toBool
#guard !(KnownRead.check (circ [List.range 48] 1 [#[0, 1, -1, -1, -1, -1, -1, -1], #[-1, -1, 2, -1, -1, -1, -1, -1]])
  (known ["k1"])).toBool
-- a forced zero holds no bit
#guard !(KnownRead.check (circ [List.range 48] 1 [#[0, -1, 2, -1, -1, -1, -1, -1]]) (known ["k0"])).toBool

-- the output row's 16 bits returned by copy 0, at columns 128–143, computed where the unit has 16 computed rows
#guard (KnownRead.check (circ (List.replicate 16 []) (outs := true)) (known ["o"])).toBool
-- 15 computed rows: column 143 is the constant's, so bit 15 is not returned
#guard why (circ (List.replicate 15 []) (outs := true)) ["o"] ==
  "registered reads: known port o: no copy of the unit returns bit 15 of its row, so its known content binds nothing there"
-- no copy returns the word
#guard why (circ (List.replicate 16 []) (outs := true) (louts := [#[]])) ["o"] ==
  "registered reads: known port o: no copy of the unit returns bit 0 of its row, so its known content binds nothing there"
-- returned only by copy 1: returned where copy 1 is a copy of the unit, not past `upv`
#guard (KnownRead.check (circ (List.replicate 16 []) 2 [#[0, 1, 2, -1, -1, -1, -1, -1], #[]] (outs := true)
  (louts := [#[], #[0]])) (known ["o"])).toBool
#guard !(KnownRead.check (circ (List.replicate 16 []) 1 [#[0, 1, 2, -1, -1, -1, -1, -1], #[]] (outs := true)
  (louts := [#[], #[0]])) (known ["o"])).toBool
-- the inputs are still checked beside an output row
#guard (KnownRead.check (circ (List.range 48 :: List.replicate 15 []) (outs := true)) (known ["k0", "k1", "o"])).toBool

-- the row's shape: a `sha512/row/v1` row, or known rows of other than its words, is refused
#guard why (circ [List.range 48] (v2 := false)) ["k1"] ==
  "registered reads: known port k1: the circuit's row is not a hm96-sha512/row/v2 row without segments of its known rows' 1 words"
#guard (KnownRead.check (circ [List.range 48]) #[("k1", #[(#[7], ByteArray.empty)])]).toBool
#guard !(KnownRead.check (circ [List.range 48]) #[("k1", #[(#[7, 8], ByteArray.empty)])]).toBool
#guard !(KnownRead.check (circ [List.range 48]) #[("k0", #[(#[7, 8], ByteArray.empty), (#[7], ByteArray.empty)])]).toBool

/-- A registered port `name` of `words` leaves of `bits` bits. -/
def reg (name : String) (words : Option Nat) (bits : Nat := 16) : Registered.Port :=
  { name, value := "v", root := ByteArray.empty, leaves := 1, words, wordBits := bits, domain := ByteArray.empty,
    readsRel := #[some 0] }

/-- The ports a file's JSON marks read in full, or its refusal. -/
def marked (t : String) : String :=
  match Lean.Json.parse t >>= Flock.KnownRead.readOf with
  | .ok ns => toString ns
  | .error e => e

-- the ports marked read in full, by name; false or absent is not, and a value other than true or false is refused
#guard marked "{\"ports\": {\"k0\": {\"read_in_full\": true}, \"k1\": {}, \"x\": {\"read_in_full\": false}}}" ==
  "#[k0]"
#guard marked "{\"ports\": {\"k0\": {\"read_in_full\": 1}}}" ==
  "registered reads: port k0: read_in_full is true or false"
-- a port read in full is one row of its registered row's u16 words, 32-bit leaves two each
#guard KnownRead.fullRows #[reg "k0" (some 2)] #["k0"] == #[("k0", #[(#[0, 0], ByteArray.empty)])]
#guard KnownRead.fullRows #[reg "k0" (some 1) 32] #["k0"] == #[("k0", #[(#[0, 0], ByteArray.empty)])]
-- read in every bit at its registered width: passes; at another width, or with a bit unread, refused
#guard (KnownRead.check (circ [List.range 48]) (KnownRead.fullRows #[reg "k0" (some 2), reg "k1" (some 1)] #["k0", "k1"])).toBool
#guard why' (circ [List.range 48]) (KnownRead.fullRows #[reg "k1" (some 2)] #["k1"]) ==
  "registered reads: known port k1: the circuit's row is not a hm96-sha512/row/v2 row without segments of its known rows' 1 words"
#guard why' (circ [List.range 47]) (KnownRead.fullRows #[reg "k1" (some 1)] #["k1"]) ==
  "registered reads: known port k1: no computed row of the circuit reads bit 15 of its row, so its known content binds nothing there"

end Flock.Test.KnownRead
