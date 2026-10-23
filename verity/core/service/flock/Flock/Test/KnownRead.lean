/-
`KnownRead.check` on a unit of one input word, eight 16-bit ports: circuit ports `k0` (2 words) and `k1` (1 word),
copy 0's leaf ports 0–2 their three words, the rest forced zeros. A known port passes where the unit's computed rows read
every bit of its row, at some copy, and is refused at the first bit none reads; a port the file does not name, a name
no port has, and an output row are not checked. A bit read only at a copy past `upv`, or only through another copy's leaf,
is not read.
-/
import Flock.KnownRead

namespace Flock.Test.KnownRead

open Flock

/-- A net of one input word whose computed rows read, in A, the columns of `reads` (one row each, B empty), its inputs
eight 16-bit ports. -/
def net (reads : List (List Nat)) : Net :=
  let nIn := WORD
  let rows := Array.replicate nIn #[] ++ (reads.map List.toArray).toArray
  let w := nIn + reads.length + 1
  match Sparse.ofRows rows w, Sparse.ofRows (Array.replicate rows.size #[]) w with
  | .ok a, .ok b =>
    let inGroups : Array Group := #[⟨0, 1, Array.replicate 8 16⟩]
    { (default : Net) with constPos := nIn + reads.length, inWords := 1, a := a, b := b, inGroups := inGroups }
  | _, _ => default

/-- Copy `u`'s leaves: `leaves u` for each copy, words 0–2 at its first three ports by default. -/
def circ (reads : List (List Nat)) (upv : Nat := 1)
    (leaves : List (Array Int) := [#[0, 1, 2, -1, -1, -1, -1, -1]]) (outs : Nat := 0) : Circuit :=
  { (default : Circuit) with
    nets := #[("unit", net reads)]
    ports := #[{ name := "k0", words := 2, chunks := 1, nb := 1 }, { name := "k1", words := 1, chunks := 1, nb := 1 }]
    outRowPorts := Array.replicate outs default
    upv, leavesIn := leaves.toArray }

def known (names : List String) : Array (String × Array (Array Nat × ByteArray)) :=
  names.toArray.map fun n => (n, #[])

def why (c : Circuit) (names : List String) : String :=
  match KnownRead.check c (known names) with
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
-- no known port, or a name no port has
#guard (KnownRead.check (circ []) (known [])).toBool
#guard (KnownRead.check (circ []) (known ["x"])).toBool
-- an output row is not read
#guard (KnownRead.check (circ [] (outs := 1)) (known ["k1"])).toBool
#guard !(KnownRead.check (circ [] (outs := 1)) (known ["k0"])).toBool
-- k1's word held only at copy 1's leaves: read where copy 1 is a copy of the unit, not past `upv`
#guard (KnownRead.check (circ [List.range 48] 2 [#[0, 1, -1, -1, -1, -1, -1, -1], #[-1, -1, 2, -1, -1, -1, -1, -1]])
  (known ["k1"])).toBool
#guard !(KnownRead.check (circ [List.range 48] 1 [#[0, 1, -1, -1, -1, -1, -1, -1], #[-1, -1, 2, -1, -1, -1, -1, -1]])
  (known ["k1"])).toBool
-- a forced zero holds no bit
#guard !(KnownRead.check (circ [List.range 48] 1 [#[0, -1, 2, -1, -1, -1, -1, -1]]) (known ["k0"])).toBool

end Flock.Test.KnownRead
