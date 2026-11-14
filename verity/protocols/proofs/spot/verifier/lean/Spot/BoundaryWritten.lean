module

public import Spot.Json

/-!
# An output field's leaves, each written once by dense reads (`Spot.BoundaryWritten`)

Round 1's `link-written`: a window circuit's boundary reads (`verity.commitments.windows.interface.Circuit.boundary_reads`)
that name an output field `o<j>` of the replay Definition write each of the field's leaves exactly once. A read covers
reader index `i < Π radix`, whose source is leaf `base + Σ d_k · step_k` (`d` the digits of `i` in mixed radix `radix`), and
the row there holds `row` leaves from its source on (16-bit leaves, the field's leaf order).

A read is **dense** when its digits of radix above 1 have steps that, sorted, are `row`, then `row · r₀`, then
`row · r₀ · r₁` and so on, each the one before times the radix of the digit before (`r₀` the radix of the smallest step).
Its rows then lay `[base, base + row · Π radix)` end to end, each leaf once (`extent?`). A digit of radix 1 reads no other
row, whatever its step. A digit of radix 0, a row of no leaves, a fan-out (a step of 0), a negative step and rows with a gap
between them are no dense read. A field's dense reads, sorted by base, must tile `[0, leaves)` with no gap and no overlap
(`written`).

The work is per read, never per reader index or leaf (AGENTS.md "Design rules"). `writtenRef` is the reference it is held
to, every leaf of every read listed; the difftest (the `boundary_written` and `boundary_written_unrolled` pieces) holds both
to one Python reference. No check calls this module yet: round 1 reads each read from the window's record and the field's
leaves from the replay Definition's signature.
-/

@[expose] public section

namespace Spot.BoundaryWritten

open Lean (Json toJson)

/-- A read: its first leaf `base`, its digits `(radix, step)` (the first the slowest) and the leaves a row holds. -/
abbrev Read := Int × List (Nat × Int) × Nat

/-- The digits of radix above 1, sorted by step. -/
def moving (digits : List (Nat × Int)) : List (Nat × Int) :=
  (digits.filter (·.1 > 1)).mergeSort fun a b => decide (a.2 ≤ b.2)

/-- The leaves sorted digits lay from a run of `run` leaves, each step the run so far, or none at the first that isn't. -/
def chain : Int → List (Nat × Int) → Option Int
  | run, [] => some run
  | run, (r, s) :: t => if s == run then chain (run * r) t else none

/-- The leaves `[lo, hi)` a dense read lays end to end, or none when it is no dense read. -/
def extent? (base : Int) (digits : List (Nat × Int)) (row : Nat) : Option (Int × Int) :=
  if row == 0 || digits.any (·.1 == 0) then none
  else (chain row (moving digits)).map fun n => (base, base + n)

/-- Whether extents sorted by their first leaf lay `[at, leaves)` end to end. -/
def tilesFrom : Int → List (Int × Int) → Nat → Bool
  | at, [], leaves => at == leaves
  | at, (lo, hi) :: t, leaves => lo == at && tilesFrom hi t leaves

/-- `link-written` for one output field of `leaves` leaves: every read is dense, and together they write each leaf of
`[0, leaves)` once. Every extent holds a leaf, so two that meet can't both start where the one before ends. -/
def written (leaves : Nat) (reads : List Read) : Bool :=
  match reads.mapM fun (b, ds, r) => extent? b ds r with
  | none => false
  | some es => tilesFrom 0 (es.mergeSort fun a b => decide (a.1 ≤ b.1)) leaves

/-- A read's leaves, every reader index's row listed. -/
def leavesRef (base : Int) (digits : List (Nat × Int)) (row : Nat) : List Int :=
  let srcs := digits.foldl (fun acc (r, s) => acc.flatMap fun x => (List.range r).map fun d => x + d * s) [base]
  srcs.flatMap fun x => (List.range row).map fun k => x + k

/-- Whether a read is dense, read off its leaves: a leaf, no digit of radix 0 or negative step but at radix 1, and its
leaves a run from the least, each once. -/
def denseRef (base : Int) (digits : List (Nat × Int)) (row : Nat) : Bool :=
  let ls := (leavesRef base digits row).mergeSort fun a b => decide (a ≤ b)
  !ls.isEmpty && digits.all (fun (r, s) => r ≥ 1 && (r == 1 || s ≥ 0)) &&
    ls == (List.range ls.length).map fun k => ls.headD 0 + k

/-- `extent?`'s reference: a dense read's least leaf and one past its last, read off its leaves. -/
def extentRef? (base : Int) (digits : List (Nat × Int)) (row : Nat) : Option (Int × Int) :=
  let ls := leavesRef base digits row
  if denseRef base digits row then some (ls.foldl min (ls.headD 0), ls.foldl min (ls.headD 0) + ls.length) else none

/-- `written`'s reference: every read dense by `denseRef`, and their leaves together `[0, leaves)`, each once. -/
def writtenRef (leaves : Nat) (reads : List Read) : Bool :=
  reads.all (fun (b, ds, r) => denseRef b ds r) &&
    (reads.flatMap fun (b, ds, r) => leavesRef b ds r).mergeSort (fun a b => decide (a ≤ b)) ==
      (List.range leaves).map Int.ofNat

/-- An example field: four served ids of two 16-bit leaves each, a unit writing one id. -/
def ids : List Read := [(0, [(4, 2)], 2)]

#guard written 8 ids
#guard extent? 0 [(4, 2)] 2 == some (0, 8)
-- A window built at the request's own three ids writes three of the class's four (table (iii), `link-written`).
#guard !written 8 [(0, [(3, 2)], 2)]
-- A fan-out, and a negative step even where its rows would lay the field end to end.
#guard !written 8 [(0, [(4, 0)], 2)]
#guard !written 8 [(6, [(4, -2)], 2)] && writtenRef 8 [(6, [(4, -2)], 2)] == false
-- Two reads, in any order, each half the field; and the same reads overlapping, or past the field.
#guard written 8 [(4, [(2, 2)], 2), (0, [(2, 2)], 2)]
#guard !written 8 [(0, [(4, 2)], 2), (6, [(1, 0)], 2)]
#guard !written 8 [(0, [(5, 2)], 2)]
-- Two interleaved reads whose leaves together are the field once, neither dense.
#guard !written 8 [(0, [(2, 4)], 2), (2, [(2, 4)], 2)]
-- A cut port's row digit, slow or fast; a digit of radix 1 at any step; a row of no leaves; a digit of radix 0.
#guard written 8 [(0, [(2, 4), (2, 2)], 2)] && written 8 [(0, [(2, 2), (2, 4)], 2)]
#guard written 8 [(0, [(1, -5), (4, 2)], 2)]
#guard !written 0 [(0, [(1, 0)], 0)] && !written 0 [(0, [(0, 2)], 2)] && written 0 []
#guard [ids, [(0, [(3, 2)], 2)], [(4, [(2, 2)], 2), (0, [(2, 2)], 2)], [(0, [(2, 4)], 2), (2, [(2, 4)], 2)],
    [(0, [(2, 4), (2, 2)], 2)], [(0, [(1, -5), (4, 2)], 2)], [(6, [(4, -2)], 2)]].all fun rs => written 8 rs == writtenRef 8 rs

/-- A read from `{base, radix, step, row}`, its radix and step lists of one length. -/
def readOf? (j : Json) : Option Read := do
  let .arr rs := getD j "radix" | none
  let .arr ss := getD j "step" | none
  guard (rs.size == ss.size)
  let ds ← (rs.toList.zip ss.toList).mapM fun (r, s) => do pure ((← nat? r), (← int? s))
  pure ((← int? (getD j "base")), ds, (← nat? (getD j "row")))

/-- A piece: `{leaves, reads}` to `{written}` by `check`, and each read's extent by `extent` (null where it is no dense
read). -/
def pieceBy (check : Nat → List Read → Bool) (extent : Int → List (Nat × Int) → Nat → Option (Int × Int)) (j : Json) :
    Except String Json := do
  let some leaves := nat? (getD j "leaves") | throw "leaves is not a natural"
  let .arr rj := getD j "reads" | throw "reads is not a list"
  let some reads := rj.toList.mapM readOf?
    | throw "reads is not a list of {base, radix, step, row}, radix and step lists of one length"
  return Json.mkObj [("written", .bool (check leaves reads)),
    ("extents", .arr (reads.map fun (b, ds, r) => match extent b ds r with
      | some (lo, hi) => Json.arr #[toJson lo, toJson hi]
      | none => Json.null).toArray)]

/-- The `boundary_written` piece: `written` and `extent?`. -/
def piece : Json → Except String Json := pieceBy written extent?

/-- The `boundary_written_unrolled` piece: `writtenRef` and `extentRef?`. -/
def pieceRef : Json → Except String Json := pieceBy writtenRef extentRef?

end Spot.BoundaryWritten
