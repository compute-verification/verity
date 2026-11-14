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

Built: the field's reader (`Ranks.ofJson?`), the segments (`Ranks.segment`), the unit-to-member rule (`memberOf`),
`ranks-window`, `ranks-run`, `ranks-members` and `ranks-units`. `ranks-wires` and `ranks-reads` refuse with their codes
(`NOT_BUILT`) until built, so nothing passes yet, and no guarantee reads this module.
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

/-- One member of a rank's registration: its key `t/c<j>`, its first unit in the rank's window, its count of units and
each output port's leaves. -/
structure Held where
  key : String
  base : Nat
  units : Nat
  outs : Array (String × Nat)
  deriving Repr, Inhabited

/-- One rank's registration as round 2 reads it: its rank, its record's `H`, the digest of its record's bytes it binds
(`window.circuit`), its run, and its members. -/
structure Registered where
  rank : Nat
  digest : String
  circuit : String
  run : String
  members : Array Held
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

def WINDOW_WHY : String :=
  "a served rank record's digest is not its rank's window, or its registration binds another digest or other bytes"

def RUN_WHY : String :=
  "the ranks' registrations are not one serve's, one a rank at <root>/rank<r>, each in the log before the challenge"

def MEMBERS_WHY : String :=
  "a rank's members of a template don't tile its units in key order, name no template of the rank's, or disagree " ++
  "on an output port's leaves a unit"

def UNITS_WHY : String := "a template's units over the ranks are not the joined record's count of it"

/-- `j` where `key` is `t/c<j>`, none for a key of another template's. -/
def indexOf (t key : String) : Option Nat :=
  let p := (t ++ "/c").toList
  let k := key.toList
  if k.take p.length == p then (String.ofList (k.drop p.length)).toNat? else none

/-- Registration `reg`'s members of template `t`, each with its index, in index order. -/
def membersOf (reg : Registered) (t : String) : Array (Nat × Held) :=
  (reg.members.filterMap fun h => (indexOf t h.key).map fun j => (j, h)).qsort fun a b => a.1 < b.1

/-- Members `ms` in index order are `0, 1, …`, each of at least one unit from where the one before it ends, together
`[0, n)`. -/
def tiles (ms : Array (Nat × Held)) (n : Nat) : Bool :=
  let ends := ms.foldl (init := some (0, 0)) fun acc x => acc.bind fun p =>
    if x.1 == p.1 && x.2.base == p.2 && x.2.units ≥ 1 then some (p.1 + 1, p.2 + x.2.units) else none
  ends == some (ms.size, n)

/-- Members `ms` name the same output ports, each with the same leaves a unit. -/
def leavesAgree (ms : Array (Nat × Held)) : Bool :=
  match ms[0]? with
  | none => true
  | some x0 => ms.all fun x => x.2.outs.map (·.1) == x0.2.outs.map (·.1) &&
      (x.2.outs.zip x0.2.outs).all fun ab => ab.1.2 * x0.2.units == ab.2.2 * x.2.units

/-- The member of rank registrations `regs` that holds unit `u` of template `t` of the joined circuit (the module's
unit-to-member rule): the rank whose segment holds it, and that rank's member whose range holds its place there. -/
def memberOf (rs : Ranks) (regs : Array Registered) (t : String) (u : Nat) : Option Member := do
  let r ← (List.range rs.of.size).find? fun r =>
    decide ((rs.segment t r).1 ≤ u) && decide (u < (rs.segment t r).2)
  let reg ← regs.find? (·.rank == r)
  let v := u - (rs.segment t r).1
  let x ← (membersOf reg t).find? fun x => decide (x.2.base ≤ v) && decide (v < x.2.base + x.2.units)
  pure ⟨r, t, x.1, x.2.base, x.2.units⟩

/-- `ranks-window` of served rank records `served` (each its `H` and the digest of its bytes, in rank order) against
registrations `regs`: each is its rank's window, and its rank's registration binds both. -/
def windowFailures (rs : Ranks) (regs : Array Registered) (served : Array (String × String)) : List (String × Json) :=
  let ok := served.size == rs.of.size && (List.range rs.of.size).all fun r =>
    match served[r]?, rs.of[r]?, regs.find? (·.rank == r) with
    | some hc, some q, some reg => hc.1 == q.window && reg.digest == hc.1 && reg.circuit == hc.2
    | _, _, _ => false
  if ok then [] else [("ranks-window", Json.str WINDOW_WHY)]

/-- `ranks-run` of registrations `regs` at the serve's root `root`: one a rank, its run `<root>/rank<r>`, and in the log
before the period's challenge (`logged`, of the record digest it binds). -/
def runFailures (rs : Ranks) (regs : Array Registered) (root : String) (logged : String → Bool) :
    List (String × Json) :=
  let ok := regs.size == rs.of.size && (List.range rs.of.size).all fun r =>
    match regs.filter (·.rank == r) with
    | #[reg] => reg.run == root ++ "/rank" ++ toString r && logged reg.circuit
    | _ => false
  if ok then [] else [("ranks-run", Json.str RUN_WHY)]

/-- `ranks-members` of registrations `regs`: each rank's members of each template it names tile its units in key order
(`tiles`) and agree on their output ports' leaves a unit (`leavesAgree`), and each member is of a template it names. -/
def membersFailures (rs : Ranks) (regs : Array Registered) : List (String × Json) :=
  let ok := (List.range rs.of.size).all fun r =>
    match regs.find? (·.rank == r), rs.of[r]? with
    | some reg, some q =>
      reg.members.all (fun h => q.units.any fun tn => (indexOf tn.1 h.key).isSome) &&
        q.units.all fun tn => tiles (membersOf reg tn.1) tn.2 && leavesAgree (membersOf reg tn.1)
    | _, _ => false
  if ok then [] else [("ranks-members", Json.str MEMBERS_WHY)]

/-- The joined record's units of each template (its `templates`), none for a malformed list. -/
def templatesOf (rec : Json) : Option (Array (String × Nat)) :=
  match getD rec "templates" with
  | .arr ts => ts.mapM fun x => match x with
    | .arr #[.str t, n] => (nat? n).map fun k => (t, k)
    | _ => none
  | _ => none

/-- `ranks-units`: the ranks name only the joined record's templates, and each template's units over them are its
count. -/
def unitsOk (rs : Ranks) (rec : Json) : Bool :=
  match templatesOf rec with
  | some ts => rs.of.all (fun q => q.units.all fun tn => ts.any (·.1 == tn.1)) &&
      ts.all fun tn => (rs.of.map (·.unitsOf tn.1)).foldl (· + ·) 0 == tn.2
  | none => false

/-- `ranks-units`, then `ranks-wires` and `ranks-reads` of the joined record `rec` against its ranks' records
`records`; the last two not built: refused. -/
def recordFailures (rs : Ranks) (rec : Json) (_ : Array Json) : List (String × Json) :=
  (if unitsOk rs rec then [] else [("ranks-units", Json.str UNITS_WHY)]) ++
    [("ranks-wires", Json.str NOT_BUILT), ("ranks-reads", Json.str NOT_BUILT)]

#guard
  let rs : Ranks := ⟨#[⟨"", #[("T", 3)], 0⟩, ⟨"", #[("T", 2)], 0⟩]⟩
  let reg (r : Nat) (ms : Array Held) : Registered := ⟨r, "", "", "", ms⟩
  let r1 (leaves : Nat) := reg 1 #[⟨"T/c0", 0, 2, #[("o0", leaves)]⟩]
  let regs := #[reg 0 #[⟨"T/c0", 0, 2, #[("o0", 32)]⟩, ⟨"T/c1", 2, 1, #[("o0", 16)]⟩], r1 32]
  let held (u : Nat) := (memberOf rs regs "T" u).map fun m => (m.rank, m.index, m.base)
  held 2 == some (0, 1, 2) && held 4 == some (1, 0, 0) && held 5 == none && (membersFailures rs regs).isEmpty &&
    !(membersFailures rs #[regs[0]!, r1 31]).isEmpty &&
    unitsOk rs (Json.mkObj [("templates", Json.arr #[Json.arr #[Json.str "T", 5]])]) &&
    !unitsOk rs (Json.mkObj [("templates", Json.arr #[Json.arr #[Json.str "T", 4]])])

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
