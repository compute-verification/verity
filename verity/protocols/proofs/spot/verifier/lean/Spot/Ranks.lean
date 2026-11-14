/-
**A joined circuit's ranks** (`verity.commitments.windows.interface.Ranks`, the record's `ranks`): the record of a node's
circuit joined from its ranks states, in rank order, each rank's window by its circuit's digest `H_r` (the SHA-512 of the
canonical JSON of its record less its run's fields, `circuit_of`), its units of each template and the count of the joined
wires that are its own. Serving commits one window a rank, never the join; the verifier reads the joined record as its
own copy and each rank's registration, and runs no join.

* **The unit-to-member rule** (`memberOf`): rank `r`'s units of template `t` follow rank `r − 1`'s (`Ranks.segment`), and
  the member that holds a unit is the one of rank `r`'s registered members `t/c<j>` whose range holds it, the members
  read from rank `r`'s registration (their leaves by prefix sums), never from the joined record.
* **Its checks**, each a code:
  - `ranks-shape`: the field is not at least one rank, each a digest of 128 lowercase hex digits, a count of units by
    template and of wires;
  - `ranks-window`: a served rank record's `H` is not its rank's `window`, or the SHA-512 of its bytes is not its
    registration's `window.circuit`;
  - `ranks-run`: the ranks' registrations are not one serve's: a rank's run is not `<root>/rank<r>` with one root for
    all, or a rank has no registration in the log before the period's challenge. Round 2's reader over the joined
    record (`Spot.SecondWindow`) rests its dropping of the interior registration on this check;
  - `ranks-members`: rank `r`'s members of a template don't tile its units in key order, or one member's output ports
    disagree on their leaves;
  - `ranks-units`: a template's units over the ranks are not the joined record's count of it;
  - `ranks-wires`: one of the ranks' own wires is not its rank's wire shifted by its rank's offsets, or one of the join's
    has its reader and its writers in one rank;
  - `ranks-reads`: a registered read named `r<r>/x` is not rank `r`'s read of `x`, shifted, or a registered name is no
    rank's.
  `ranks-wires` and `ranks-reads` hold the joined record to the ranks' records, all fixed before the serve: they run
  each round until the verifier has a step where a class is registered, and then there, once.

Built: the field's reader (`Ranks.ofJson?`) and the segments (`Ranks.segment`). Every other check refuses with its code
(`NOT_BUILT`) until built, so nothing passes, and no guarantee reads this module.
-/
module

public import Spot.Json
meta import Spot.Json

@[expose] public section

namespace Spot.Ranks

open Lean (Json)

/-- How a rank's registered names are its own: `r<rank>/` before the name its window registers (`RANK_PREFIX`). -/
def PREFIX : String := "r<rank>/"

/-- The detail of a check whose body is not built: it refuses. -/
def NOT_BUILT : String := "not yet built: refused"

/-- The codes of this module's checks, in the order they run. -/
def CODES : List String :=
  ["ranks-shape", "ranks-window", "ranks-run", "ranks-members", "ranks-units", "ranks-wires", "ranks-reads"]

/-- One rank of a joined circuit (`Rank`): its window's circuit digest, its units of each template, and the count of
the joined wires that are its own. -/
structure Rank where
  window : String
  units : Array (String × Nat)
  wires : Nat
  deriving Repr, Inhabited

/-- A joined circuit's ranks, in rank order (`Ranks`). -/
structure Ranks where
  of : Array Rank
  deriving Repr, Inhabited

/-- One rank's registration as round 2 reads it: its rank, its record's `H`, the digest of its record's bytes it binds
(`window.circuit`), its run, and its members `(key, base, units)` in key order. -/
structure Registered where
  rank : Nat
  digest : String
  circuit : String
  run : String
  members : Array (String × Nat × Nat)
  deriving Repr, Inhabited

/-- A rank member: its rank, template, index `j` (its key `t/c<j>`), its first unit in the rank's window and its count. -/
structure Member where
  rank : Nat
  template : String
  index : Nat
  base : Nat
  units : Nat
  deriving Repr, Inhabited

/-- Rank `q`'s units of template `t`, none for a template it doesn't name. -/
def Rank.unitsOf (q : Rank) (t : String) : Nat := ((q.units.find? (·.1 == t)).map (·.2)).getD 0

/-- Rank `r`'s units of template `t` in the joined circuit, `[lo, hi)`: the earlier ranks' come first. -/
def Ranks.segment (rs : Ranks) (t : String) (r : Nat) : Nat × Nat :=
  let lo := ((rs.of.extract 0 r).map (·.unitsOf t)).foldl (· + ·) 0
  (lo, lo + ((rs.of[r]?).map (·.unitsOf t)).getD 0)

/-- One rank of the record's `ranks` field: exactly `units`, `window` and `wires`, its window 128 lowercase hex digits,
its units an object of naturals and its wires a natural; `none` otherwise. -/
def Rank.ofJson? (j : Json) : Option Rank := do
  if keys? j != some #["units", "window", "wires"] then none
  let .str w := getD j "window" | none
  if !(w.length == 128 && isHex w) then none
  let .obj us := getD j "units" | none
  let units ← us.toArray.mapM fun (t, n) => (nat? n).map fun k => (t, k)
  let wires ← nat? (getD j "wires")
  pure ⟨w, units, wires⟩

/-- The ranks of the record's `ranks` field (`Ranks.of_doc`): exactly `of` and `prefix`, the prefix `PREFIX`, and at
least one rank (`Rank.ofJson?`); `none` otherwise, which `ranks-shape` refuses. -/
def Ranks.ofJson? (j : Json) : Option Ranks := do
  if keys? j != some #["of", "prefix"] then none
  let .str p := getD j "prefix" | none
  if p != PREFIX then none
  let .arr rs := getD j "of" | none
  if rs.isEmpty then none
  let of ← rs.mapM Rank.ofJson?
  pure ⟨of⟩

#guard
  let rank (w : String) := Json.mkObj [("units", Json.mkObj [("T", 3)]), ("window", Json.str w), ("wires", 2)]
  let field (w p : String) := Json.mkObj [("of", Json.arr #[rank w]), ("prefix", Json.str p)]
  (Ranks.ofJson? (field ("".pushn 'a' 128) PREFIX)).map (fun rs => (rs.of.size, rs.segment "T" 0, rs.segment "T" 1))
      == some (1, (0, 3), (3, 3)) &&
    (Ranks.ofJson? (field ("".pushn 'A' 128) PREFIX)).isNone && (Ranks.ofJson? (field ("".pushn 'a' 128) "r/")).isNone

/-- The member of rank registrations `regs` that holds unit `u` of template `t` of the joined circuit (the module's
unit-to-member rule). Not built: none. -/
def memberOf (_ : Ranks) (_ : Array Registered) (_ : String) (_ : Nat) : Option Member := none

/-- `ranks-window` of served rank records `served` (each its `H` and the digest of its bytes) against registrations
`regs`. Not built: refused. -/
def windowFailures (_ : Ranks) (_ : Array Registered) (_ : Array (String × String)) : List (String × Json) :=
  [("ranks-window", Json.str NOT_BUILT)]

/-- `ranks-run` of registrations `regs` at the serve's root `root`, `logged` whether a registration's digest is in the
log before the period's challenge. Not built: refused. -/
def runFailures (_ : Ranks) (_ : Array Registered) (_ : String) (_ : String → Bool) : List (String × Json) :=
  [("ranks-run", Json.str NOT_BUILT)]

/-- `ranks-members` of registrations `regs`. Not built: refused. -/
def membersFailures (_ : Ranks) (_ : Array Registered) : List (String × Json) :=
  [("ranks-members", Json.str NOT_BUILT)]

/-- `ranks-units`, `ranks-wires` and `ranks-reads` of the joined record `rec` against its ranks' records `records`.
Not built: refused. -/
def recordFailures (_ : Ranks) (_ : Json) (_ : Array Json) : List (String × Json) :=
  [("ranks-units", Json.str NOT_BUILT), ("ranks-wires", Json.str NOT_BUILT), ("ranks-reads", Json.str NOT_BUILT)]

/-- Every failure of the joined record `rec`'s ranks, read against its ranks' records, served records and
registrations at the serve's root, `ranks-shape` first. -/
def failures (rec : Json) (records : Array Json) (served : Array (String × String)) (regs : Array Registered)
    (root : String) (logged : String → Bool) : List (String × Json) :=
  match Ranks.ofJson? (getD rec "ranks") with
  | none => [("ranks-shape", Json.str NOT_BUILT)]
  | some rs =>
    windowFailures rs regs served ++ runFailures rs regs root logged ++ membersFailures rs regs ++
      recordFailures rs rec records

/-- **The ranks' check** (module docstring): each code once, with the detail it first failed with. -/
def check (rec : Json) (records : Array Json) (served : Array (String × String)) (regs : Array Registered)
    (root : String) (logged : String → Bool) : Check :=
  (failures rec records served regs root logged).foldl (fun c x => c.fail x.1 x.2) {}

end Spot.Ranks
