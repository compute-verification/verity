import NetworkCheck.Circuit
import Std.Data.HashMap

/-!
# A stated record's members

`interface._members`' layout of a stated circuit record (`Stated`), the window's partition into statements: each
template's units, in the partition's order, cut where a wire's reader units begin or end and where a registered read's,
table read's or opening's units do (`interface.committed_reads`), then in chunks of at most the record's cap. Each chunk
is one member, the statement `<descriptor id>/c<j>`, `j` counting across its template; its first unit's VU is its
template's first, the units of the templates before it (`populationOk` holds the record's templates to the Program's
template nodes, the partition's order), plus its offset among its template's units.

The layout is a function of the record and the Program, both public, so the period derives it rather than reading it:
the registration's roots and leaves are held to it. The work is per wire, read and member, never per unit (AGENTS.md
"Design rules").
-/

namespace NetworkCheck.Circuit

open Lean (Json toJson)
open SampledProofs.Wires (field)

/-- One member (`rows.Member`): units `[start, start + n)` among template `template`'s, the statement
`<template>/c<chunk>`, its first unit VU `base` of the partition. -/
structure Member where
  template : String
  chunk : Nat
  start : Nat
  base : Nat
  n : Nat
  deriving Repr, BEq, Inhabited

def Member.name (m : Member) : String := s!"{m.template}/c{m.chunk}"

/-- By reader template, the unit ranges `(first, units)` of the wires it reads and of the registered reads, table reads
and openings of its units, in no order. -/
def readerRanges (r : Stated) : Std.HashMap String (List (Nat × Nat)) :=
  let add := fun (m : Std.HashMap String (List (Nat × Nat))) (tid : String) (lo n : Nat) =>
    m.insert tid ((lo, n) :: m.getD tid [])
  let m := r.wiring.wires.foldl (fun m w => add m w.reader w.start w.readerUnits) {}
  let m := r.registered.reads.foldl (fun m x => add m x.pat.reader x.pat.start x.pat.readerUnits) m
  let m := r.registered.roots.foldl (fun m x => add m x.reader x.start x.units) m
  r.registered.openings.foldl (fun m x => add m x.reader x.start x.units) m

/-- A sorted array without its repeats. -/
def dedupSorted (xs : Array Nat) : Array Nat :=
  xs.foldl (fun out x => if out.back? == some x then out else out.push x) #[]

/-- Template `tid`'s cut points: 0, its `n` units and each end of `ranges`, sorted and distinct. -/
def cutPoints (n : Nat) (ranges : List (Nat × Nat)) : Array Nat :=
  dedupSorted ((#[0, n] ++ (ranges.flatMap fun (a, k) => [a, a + k]).toArray).qsort (· < ·))

/-- Template `tid`'s members, appended to `out`: each segment `[a, b)` between consecutive cut points `es` in chunks of
`cap` units from `a`, the last the rest, chunk `j` counting from 0 across the template; `lo` its first VU. -/
def chunks (tid : String) (lo cap : Nat) (es : Array Nat) (out : Array Member) : Array Member := Id.run do
  let mut out := out
  let mut j := 0
  for i in [0:es.size - 1] do
    let a := es[i]!
    let b := es[i + 1]!
    for c in [0:(b - a + cap - 1) / cap] do
      let s := a + c * cap
      out := out.push { template := tid, chunk := j, start := s, base := lo + s, n := min cap (b - s) }
      j := j + 1
  return out

/-- **The members** of stated record `r` (`interface._members`), templates in the record's order: each template's units
cut at its `cutPoints`, then chunked (`chunks`). Refused, by name, at a cap of 0, or where a wire or registered read
cuts a template's units past their end (which the record's checks refuse first). -/
def members (r : Stated) : Except String (Array Member) := do
  if r.cap == 0 then throw "the circuit's cap is 0: a statement holds at least one unit"
  let ranges := readerRanges r
  let mut out := #[]
  let mut lo := 0
  for (tid, n) in r.templates do
    let es := cutPoints n (ranges.getD tid [])
    if let some e := es.find? (n < ·) then
      throw s!"template {tid}: a wire or registered read cuts its units at {e}, past its {n}"
    out := chunks tid lo r.cap es out
    lo := lo + n
  return out

/-- `network-records verdicts`' `members`: on `{"circuit", "order", "program"}` as `circuitFn` reads them, the record's
checks (`checked`), then its `members`, each `{"base", "n", "name", "start"}`. -/
def membersFn (j : Json) : Except String Json := do
  let P ← Flock.Program.decode (← Flock.canon (← field j "program")).toUTF8
  let (r, _) ← checked P (← field j "circuit") (← field j "order")
  let ms ← members r
  return .arr (ms.map fun m => Json.mkObj [("base", toJson m.base), ("n", toJson m.n), ("name", .str m.name),
    ("start", toJson m.start)])

/-! ## The layout on a toy record -/

private def toyWire (reader : String) (start k : Nat) (cuts : Nat := 1) : Wire :=
  { reader, port := "x", writer := "A", out := "o0", start, radix := if cuts == 1 then [k] else [k, cuts], base := 0,
    step := if cuts == 1 then [1] else [1, 0], cut0 := 0, cuts }

private def toy (cap : Nat) (ws : List Wire) (reg : Registered := {}) : Stated :=
  { templates := [("A", 10), ("B", 4)], cap, wiring := { templates := [], cuts := [], wires := ws }, registered := reg }

private def layout (r : Stated) : Option (List (String × Nat × Nat × Nat)) :=
  (members r).toOption.map fun ms => ms.toList.map fun m => (m.name, m.start, m.base, m.n)

#guard layout (toy 16 []) == some [("A/c0", 0, 0, 10), ("B/c0", 0, 10, 4)]
#guard layout (toy 4 []) == some [("A/c0", 0, 0, 4), ("A/c1", 4, 4, 4), ("A/c2", 8, 8, 2), ("B/c0", 0, 10, 4)]
#guard layout (toy 4 [toyWire "A" 3 4]) ==
  some [("A/c0", 0, 0, 3), ("A/c1", 3, 3, 4), ("A/c2", 7, 7, 3), ("B/c0", 0, 10, 4)]
#guard layout (toy 2 [toyWire "A" 3 4, toyWire "B" 1 2 2]) ==
  some [("A/c0", 0, 0, 2), ("A/c1", 2, 2, 1), ("A/c2", 3, 3, 2), ("A/c3", 5, 5, 2), ("A/c4", 7, 7, 2),
        ("A/c5", 9, 9, 1), ("B/c0", 0, 10, 1), ("B/c1", 1, 11, 2), ("B/c2", 3, 13, 1)]
#guard layout (toy 16 [toyWire "A" 0 10]) == some [("A/c0", 0, 0, 10), ("B/c0", 0, 10, 4)]
#guard layout (toy 16 [] { roots := [⟨"B", "r", "t", "root", 1, 2⟩], openings := [⟨"A", "t", "k", ["p"], "o1", 5, 5⟩] }) ==
  some [("A/c0", 0, 0, 5), ("A/c1", 5, 5, 5), ("B/c0", 0, 10, 1), ("B/c1", 1, 11, 2), ("B/c2", 3, 13, 1)]
#guard layout (toy 16 [] { reads := [⟨"w", toyWire "A" 2 3⟩] }) ==
  some [("A/c0", 0, 0, 2), ("A/c1", 2, 2, 3), ("A/c2", 5, 5, 5), ("B/c0", 0, 10, 4)]
#guard members (toy 0 []) matches .error "the circuit's cap is 0: a statement holds at least one unit"
#guard members (toy 4 [toyWire "B" 3 2]) matches .error "template B: a wire or registered read cuts its units at 5, past its 4"

end NetworkCheck.Circuit
