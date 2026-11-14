/-
**Round 2's reader over the joined record**: a two-stage run's second-stage sessions held to the joined circuit's record
(the verifier's own copy, its digest the Definition's `window`) and to its ranks' registrations, beside
`Spot.SecondStage`, which reads an interior registration and stays as it is until a later switch with its proof.

* **Links** are the record's `boundary_reads` (each a reader template, port, units and field, row `i` the field's leaves
  from `base + Σ d_k · step_k`), read at round 1's boundary port for that field (`linksOf?`).
* **Wires** are the record's `wires`, each a reader span and a writer in closed form (`Spot.Wires`): the first
  `Σ wires_r` are the ranks' own, the rest the join's (`wiresOf?`).
* **Rows** are each declaration's pattern at the units the second sample draws, once per declaration and sampled unit,
  never unrolled (`rowsOf?`).
* **Parts** are rank members (`Spot.Ranks.memberOf`): each sampled unit is proved in the session of the rank member that
  holds it, and that session's sample is the sampled block's units in that member.
* **No interior registration.** This reader drops the interior registration and its four checks (`second-interior-order`,
  `second-interior-names`, `second-return`, `second-inputs`) because every row round 2 opens is in a rank window whose
  registration is in the log before the period's challenge, which `ranks-run` checks (`Spot.Ranks`). Without that
  check the interior registration's checks would still be needed.
* **Its checks**: the ranks' (`Spot.Ranks.check`); per session, `Spot.SecondStage`'s `second-header`,
  `second-population`, `second-roots`, `second-sample`, `second-read`, `second-values`, `second-refs`, `second-tables`
  and `second-verified`, read against the rank member in place of an interior part, with the window-domain binding at
  the rank's program, partition digest and run (`portBinding`); and `ranks-wire-tree`: the writer rows a sampled reader
  member reads through one wire lie in one writer member.
* **The form**: it runs behind `Spot.SecondStage.FORM`, the one switch, and is refused last (`two-stage-form`) while
  that is off.

Built: the readers, the parts, `ranks-wire-tree` and `second-table-rows` (`Spot.SecondStage.tableRowsFailures` over the
joined record's wires, so a registered read that lists `table_rows` is refused here as round 2's reader refuses it).
The other per-session checks refuse with their codes (`Spot.Ranks.NOT_BUILT`), so nothing passes, and no guarantee
reads this module.
-/
module

public import Spot.Ranks
public import Spot.SecondStage
meta import Spot.Ranks
meta import Spot.SecondStage

@[expose] public section

namespace Spot.SecondWindow

open Lean (Json)

/-- The per-session codes of `Spot.SecondStage` this reader reuses, read against a rank member. -/
def SESSION_CODES : List String :=
  ["second-header", "second-population", "second-roots", "second-sample", "second-read", "second-values",
    "second-refs", "second-tables", "second-verified", "second-table-rows"]

/-- A read's record (`{reader, port, param, at, radix, base, step}`, `cut` optional) as a wire's, its `param` for the
writer and no output port, so one reader (`Spot.Wires.Wire.ofJson`) reads both. -/
def asWire (r : Json) : Json :=
  Json.mkObj ((entries r).toList.filter (·.1 != "param") ++ [("writer", getD r "param"), ("out", .str "")])

def absent (j : Json) : Bool := match j with | .null => true | _ => false

/-- A read from its record, as a wire whose `writer` is its parameter (`asWire`); none unless it is a read's record. -/
def readOf? (r : Json) : Option Spot.Wires.Wire :=
  if absent (getD r "writer") && absent (getD r "out") then (Spot.Wires.Wire.ofJson (asWire r)).toOption else none

def listOf? (rec : Json) (k : String) : Option (Array Json) :=
  match getD rec k with
  | .null => some #[]
  | .arr xs => some xs
  | _ => none

/-- The joined record `rec`'s boundary reads (`boundary_reads`, none stated: none). -/
def boundaryOf? (rec : Json) : Option (Array Spot.Wires.Wire) := (listOf? rec "boundary_reads").bind (·.mapM readOf?)

/-- The joined record `rec`'s wires (`wires`), the ranks' own first. -/
def patternsOf? (rec : Json) : Option (Array Spot.Wires.Wire) :=
  (listOf? rec "wires").bind (·.mapM fun w => (Spot.Wires.Wire.ofJson w).toOption)

def distinct {α : Type} [BEq α] (xs : Array α) : Array α :=
  xs.foldl (fun acc x => if acc.contains x then acc else acc.push x) #[]

/-- The links of the joined record `rec`: each boundary read's reader template and port, reading round 1's boundary
port of the field it names, once each; none where one port names two fields. -/
def linksOf? (rec : Json) : Option (Array Spot.SecondStage.Link) := do
  let ls := distinct ((← boundaryOf? rec).map fun r => (⟨r.reader, r.port, r.writer⟩ : Spot.SecondStage.Link))
  if ls.any fun l => ls.any fun m => l.template == m.template && l.port == m.port && l.value != m.value then none
  pure ls

/-- The wires of the joined record `rec`: each wire's reader template and port and its writer's output port, once
each. A port may read several writers, each over its own units (a select's first step and the rest). -/
def wiresOf? (rec : Json) : Option (Array Spot.SecondStage.Wire) := do
  pure (distinct ((← patternsOf? rec).map fun w => (⟨w.reader, w.port, w.writer, w.out⟩ : Spot.SecondStage.Wire)))

/-- A declaration's row at a sampled unit: template `template`'s unit `unit` (the joined circuit's), row `row` of its
port `port`, reads `source` of `of`: the leaf of a parameter or boundary field (`out` empty), or the unit of writer
template `of` at its output port `out`. -/
structure Row where
  template : String
  port : String
  unit : Nat
  row : Nat
  of : String
  out : String
  source : Int
  deriving Repr, BEq

/-- Pattern `p`'s rows at sampled units `sample` (`(template, unit)`): one a sampled unit it covers and row of its
cut. -/
def rowsAt (p : Spot.Wires.Wire) (sample : Array (String × Nat)) : Array Row :=
  sample.foldl (fun acc (t, u) =>
    if t == p.reader && decide (p.start ≤ u) && decide (u < p.start + p.readerUnits) then
      acc ++ ((List.range p.cuts).map fun c =>
        ({ template := t, port := p.port, unit := u, row := p.cut0 + c, of := p.writer, out := p.out,
           source := p.writerAt ((u - p.start) * p.cuts + c) } : Row)).toArray
    else acc) #[]

/-- The rows of the joined record `rec`'s declarations (its wires, boundary reads and registered reads) at the sampled
units `sample`, `(template, unit)` each: once per declaration and sampled unit, never unrolled. -/
def rowsOf? (rec : Json) (sample : Array (String × Nat)) : Option (Array Row) := do
  let reads ← (listOf? (getD rec "registered") "reads").bind (·.mapM readOf?)
  let ps := (← patternsOf? rec) ++ (← boundaryOf? rec) ++ reads
  pure (ps.foldl (fun acc p => acc ++ rowsAt p sample) #[])

def sameMember (a b : Spot.Ranks.Member) : Bool := a.rank == b.rank && a.template == b.template && a.index == b.index

/-- The rank members that hold the sampled units `sample` (`(template, unit)`, the joined circuit's units), each once,
in the order of its first sampled unit (`Spot.Ranks.memberOf`); none where a unit is in no member. -/
def partsOf? (rs : Spot.Ranks.Ranks) (regs : Array Spot.Ranks.Registered) (sample : Array (String × Nat)) :
    Option (Array Spot.Ranks.Member) := do
  let ms ← sample.mapM fun (t, u) => Spot.Ranks.memberOf rs regs t u
  pure (ms.foldl (fun acc m => if acc.any (sameMember m) then acc else acc.push m) #[])

def WIRE_TREE_WHY : String :=
  "a sampled reader member reads through one wire writer rows that lie in more than one writer member, or in none"

/-- Whether wire `w`'s writer rows that member `m`'s units read lie in one writer member: the closed-form box on the
writers of the reader indices in `m` (`Spot.Wires.Wire.box`) has both ends in one member, which holds every unit
between them. -/
def treeOk (rs : Spot.Ranks.Ranks) (regs : Array Spot.Ranks.Registered) (m : Spot.Ranks.Member)
    (w : Spot.Wires.Wire) : Bool :=
  let lo := (rs.segment m.template m.rank).1 + m.base
  if !w.readsIn (m.template, lo, m.units) then true else
  let (a, b) := w.indicesIn lo m.units
  let (x, y) := w.box a b
  let writerOf := Spot.Ranks.memberOf rs regs w.writer
  decide (0 ≤ x) && match writerOf x.toNat, writerOf y.toNat with
    | some p, some q => sameMember p q
    | _, _ => false

/-- `ranks-wire-tree` of the sampled members `ms` over the joined record `rec`: the writer rows a sampled reader member
reads through one wire lie in one writer member, so one writer session holds every row the reader's session reads
through it. Work per sampled member and wire, never per unit. -/
def wireTreeFailures (rs : Spot.Ranks.Ranks) (regs : Array Spot.Ranks.Registered) (rec : Json)
    (ms : Array Spot.Ranks.Member) : List (String × Json) :=
  match patternsOf? rec with
  | some ws => if ms.all fun m => ws.all (treeOk rs regs m) then [] else [("ranks-wire-tree", Json.str WIRE_TREE_WHY)]
  | none => [("ranks-wire-tree", Json.str WIRE_TREE_WHY)]

/-- What `Spot.SecondStage`'s statement checks read of the joined record `rec` that this reader builds: its wires. -/
def ownOf (rec : Json) : Spot.SecondStage.Own :=
  { sample := #[], parts := none, interior := .null, boundary := .null, drawn := #[], wires := wiresOf? rec }

/-- The per-session failures of statements `ss` against members `ms`: `second-table-rows` as `Spot.SecondStage` reads
it (`tableRowsFailures`), over the joined record's wires; the rest refused until built. -/
def sessionFailures (rec : Json) (_ : Array Spot.Ranks.Member) (ss : List Json) : List (String × Json) :=
  ((SESSION_CODES.filter (· != "second-table-rows")).map fun c => (c, Json.str Spot.Ranks.NOT_BUILT)) ++
    Spot.SecondStage.tableRowsFailures (ownOf rec) ss

section Example

def num (k : Nat) : Json := Lean.toJson k

def wireJ (a b : Nat) (port : String := "x") : Json :=
  Json.mkObj [("at", num a), ("base", num b), ("out", .str "o0"), ("port", .str port), ("radix", .arr #[num 2]),
    ("reader", .str "A"), ("step", .arr #[num 1]), ("writer", .str "B")]

def readJ (param : String) (step : Nat) : Json :=
  Json.mkObj [("at", num 0), ("base", num 0), ("param", .str param), ("port", .str "p"), ("radix", .arr #[num 2]),
    ("reader", .str "A"), ("step", .arr #[num step])]

def exRanks : Spot.Ranks.Ranks := ⟨#[⟨"", #[("A", 2), ("B", 2)], 1⟩, ⟨"", #[("A", 2), ("B", 2)], 1⟩]⟩

def exRegs : Array Spot.Ranks.Registered := #[0, 1].map fun r =>
  ⟨r, "", "", "", #[⟨"A/c0", 0, 2, #[]⟩, ⟨"B/c0", 0, 2, #[]⟩]⟩

end Example

#guard (linksOf? (Json.mkObj [("boundary_reads", .arr #[readJ "o0" 2, readJ "o0" 2])])).map (·.toList) ==
    some [({ template := "A", port := "p", value := "o0" } : Spot.SecondStage.Link)] &&
  (linksOf? (Json.mkObj [("boundary_reads", .arr #[readJ "o0" 2, readJ "o1" 2])])).isNone &&
  (linksOf? (Json.mkObj [("boundary_reads", .arr #[wireJ 0 0])])).isNone &&
  (linksOf? (Json.mkObj [])).map (·.size) == some 0
#guard (wiresOf? (Json.mkObj [("wires", .arr #[wireJ 0 0, wireJ 2 2])])).map (·.toList) ==
  some [({ template := "A", port := "x", writer := "B", wport := "o0" } : Spot.SecondStage.Wire)]
#guard (rowsOf? (Json.mkObj [("wires", .arr #[wireJ 0 0, wireJ 2 2]), ("boundary_reads", .arr #[readJ "o0" 2])])
    #[("A", 1), ("A", 2)]).map (·.toList.map fun r => (r.unit, r.of, r.source)) ==
  some [(1, "B", 1), (2, "B", 2), (1, "o0", 2)]
#guard (partsOf? exRanks exRegs #[("A", 3), ("A", 1), ("A", 2)]).map (·.toList.map fun m => (m.rank, m.index)) ==
    some [(1, 0), (0, 0)] && (partsOf? exRanks exRegs #[("A", 4)]).isNone
-- Rank 0's member `A/c0` reads writer units 1 and 2 through the third wire: rank 0's `B/c0` and rank 1's.
#guard
  let ms := (partsOf? exRanks exRegs #[("A", 1), ("A", 3)]).getD #[]
  let recOf (ws : Array Json) : Json := Json.mkObj [("wires", .arr ws)]
  (wireTreeFailures exRanks exRegs (recOf #[wireJ 0 0, wireJ 2 2]) ms).isEmpty &&
    (wireTreeFailures exRanks exRegs (recOf #[wireJ 0 0, wireJ 2 2, wireJ 0 1 "y"]) ms).map (·.1) ==
      ["ranks-wire-tree"]
-- A registered read listing `table_rows` at its key port: `KEY_ROWS_WHY` where the key is wired, else `TABLE_ROWS_WHY`.
#guard
  let stmt := Json.mkObj [("draw", Json.mkObj []), ("header", Json.mkObj []),
    ("registered", Json.mkObj [("ports", Json.mkObj [("tok", Json.mkObj [("table_rows", num 4)])])]),
    ("template", .str "A")]
  let why (j : Json) : List String := (sessionFailures j #[] [stmt]).filterMap fun (c, d) =>
    if c == "second-table-rows" then (match getD d "why" with | .str w => some w | _ => none) else none
  why (Json.mkObj [("wires", .arr #[wireJ 0 0 "tok"])]) == [Spot.SecondStage.KEY_ROWS_WHY] &&
    why (Json.mkObj [("wires", .arr #[wireJ 0 0])]) == [Spot.SecondStage.TABLE_ROWS_WHY]

def PARTS_WHY : String := "a sampled unit is in no rank member"

/-- Every failure of statements `stmts` over the joined record `rec`, its ranks' records `records`, served records
`served` and registrations `regs` at the serve's root `root`, `logged` whether a registration is in the log before the
period's challenge; when there is none and the form isn't allowed, its refusal. -/
def failures (allow : Bool) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array (String × Nat))
    (stmts : List Json) : List (String × Json) :=
  let ranks := Spot.Ranks.failures rec records served regs root logged
  let found := match Spot.Ranks.Ranks.ofJson? (getD rec "ranks") with
    | none => ranks
    | some rs => match partsOf? rs regs sample with
      | none => ranks ++ [("ranks-members", Json.str PARTS_WHY)]
      | some ms => ranks ++ wireTreeFailures rs regs rec ms ++ sessionFailures rec ms stmts
  if found.isEmpty && !allow then [("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)] else found

/-- **Round 2's check over the joined record** (module docstring), the form allowed when `allow` (`Spot.SecondStage.FORM`
in the run): each code once, with the detail it first failed with. -/
def check (allow : Bool) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array (String × Nat))
    (stmts : List Json) : Check :=
  (failures allow rec records served regs root logged sample stmts).foldl (fun c x => c.fail x.1 x.2) {}

end Spot.SecondWindow
