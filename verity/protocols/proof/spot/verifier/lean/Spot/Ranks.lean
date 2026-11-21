/-
**A joined circuit's ranks** (`verity.commitments.periods.interface.Ranks`, the record's `ranks`): the record of a node's
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
  - `ranks-wires`: one of the ranks' own wires is not its rank's wire moved to its rank's units, or one of the join's
    has its reader units in no one rank or all its writers in its reader's rank;
  - `ranks-reads`: a registered declaration named `r<r>/x` is not rank `r`'s of `x` moved to its units, or a registered
    name is no rank's.
  `ranks-wires` and `ranks-reads` hold the joined record to the ranks' records, all fixed before the serve: they run
  each round until the verifier has a step where a class is registered, and then there, once.

`ranks-wires` and `ranks-reads` hold the joined record to the form the join gives it: each rank's wires moved to its
units, in rank order, then the join's; and each rank's registered declarations named `r<r>/…` and moved to its units, in
rank order. No guarantee reads this module.
-/
module

public import Spot.Json
public import Spot.Wires
meta import Spot.Json
meta import Spot.Wires

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

/-- Template `t`'s first unit of each rank in the joined circuit, then its count: the earlier ranks' come first. -/
def Ranks.bounds (rs : Ranks) (t : String) : Array Nat :=
  rs.of.foldl (fun b q => b.push ((b.back?.getD 0) + q.unitsOf t)) #[0]

/-- Rank `r`'s units `[lo, hi)` by `Ranks.bounds`' `b`. -/
def segmentIn (b : Array Nat) (r : Nat) : Nat × Nat :=
  let lo := b.getD (min r (b.size - 1)) 0
  (lo, b.getD (r + 1) lo)

/-- Rank `r`'s units of template `t` in the joined circuit, `[lo, hi)`: the earlier ranks' come first. -/
def Ranks.segment (rs : Ranks) (t : String) (r : Nat) : Nat × Nat := segmentIn (rs.bounds t) r

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

def SHAPE_WHY : String :=
  "the ranks are not {of, prefix}, the prefix r<rank>/, at least one rank, each {window, units, wires}: 128 lowercase " ++
  "hex digits, naturals by template, a natural"

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
def executionPhaseFailures (rs : Ranks) (regs : Array Registered) (served : Array (String × String)) : List (String × Json) :=
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
(`tiles`), every rank's members of a template agree on their output ports' leaves a unit (`leavesAgree`), and each
member is of a template its rank names. -/
def membersFailures (rs : Ranks) (regs : Array Registered) : List (String × Json) :=
  let ok := (List.range rs.of.size).all fun r =>
    match regs.find? (·.rank == r), rs.of[r]? with
    | some reg, some q =>
      reg.members.all (fun h => q.units.any fun tn => (indexOf tn.1 h.key).isSome) &&
        q.units.all fun tn =>
          tiles (membersOf reg tn.1) tn.2 && leavesAgree (regs.foldl (fun acc g => acc ++ membersOf g tn.1) #[])
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

/-- `j` with key `k` set to `v`; `j` itself where it is no object. -/
def setKey (j : Json) (k : String) (v : Json) : Json :=
  match j with
  | .obj _ => j.setObjVal! k v
  | _ => j

/-- `j` with its natural at `k` raised by `d`; `j` itself where it has none. -/
def raiseNat (j : Json) (k : String) (d : Nat) : Json :=
  match nat? (getD j k) with
  | some n => setKey j k (Lean.toJson (n + d))
  | none => j

/-- `j` with its integer at `k` raised by `d`; `j` itself where it has none. -/
def raiseInt (j : Json) (k : String) (d : Nat) : Json :=
  match int? (getD j k) with
  | some n => setKey j k (Lean.toJson (n + (d : Int)))
  | none => j

/-- `j` with `p` before its string at `k`; `j` itself where it has none. -/
def prefixAt (j : Json) (k p : String) : Json :=
  match getD j k with
  | .str s => setKey j k (.str (p ++ s))
  | _ => j

def strAt (j : Json) (k : String) : String := match getD j k with | .str s => s | _ => ""

def arrAt (j : Json) (k : String) : Array Json := match getD j k with | .arr a => a | _ => #[]

/-- Rank `r`'s prefix of its registered names, `r<r>/` (`PREFIX`). -/
def rankPrefix (r : Nat) : String := "r" ++ toString r ++ "/"

/-- Rank `r`'s first unit of template `t` in the joined circuit. -/
def Ranks.first (rs : Ranks) (t : String) (r : Nat) : Nat := (rs.segment t r).1

/-- The rank whose units `[b[r], b[r + 1])` (`Ranks.bounds`) hold units `lo` to `hi`, both included. -/
def rankIn (b : Array Nat) (lo hi : Nat) : Option Nat :=
  (List.range (b.size - 1)).find? fun r => decide (b.getD r 0 ≤ lo) && decide (lo ≤ hi) && decide (hi < b.getD (r + 1) 0)

/-- The rank whose units of template `t` hold units `lo` to `hi`, both included. -/
def Ranks.rankOf (rs : Ranks) (t : String) (lo hi : Nat) : Option Nat := rankIn (rs.bounds t) lo hi

/-- Rank `r`'s wires of its record `doc` as the joined record holds them: each one's reader units and its writers
moved by rank `r`'s first unit of their template. -/
def shiftedWires (rs : Ranks) (r : Nat) (doc : Json) : Array Json :=
  (arrAt doc "wires").map fun w =>
    raiseInt (raiseNat w "at" (rs.first (strAt w "reader") r)) "base" (rs.first (strAt w "writer") r)

/-- A wire of the join's: it reads at least one reader unit, every one in some rank's units, and in each rank whose
units it reads, its writers of those units (`Spot.Wires.Wire.box` on their indices) are not all that rank's: a read
within one rank is that rank's own wire, which its window states. A wire may read several ranks' units, as `join`'s
coalesced wires do; the check is one box per wire and rank, over each template's bounds summed once a wire. -/
def joinWireOk (rs : Ranks) (j : Json) : Bool :=
  match Spot.Wires.Wire.ofJson j with
  | .ok w =>
    let br := rs.bounds w.reader
    let bw := rs.bounds w.writer
    decide (w.readerUnits ≥ 1) && decide (w.start + w.readerUnits ≤ (segmentIn br (rs.of.size - 1)).2) &&
      (List.range rs.of.size).all fun r =>
        let (lo, hi) := segmentIn br r
        if !w.readsIn (w.reader, lo, hi - lo) then true else
        let (a, b) := w.indicesIn lo (hi - lo)
        let (x, y) := w.box a b
        !(decide (0 ≤ x) && rankIn bw x.toNat y.toNat == some r)
  | .error _ => false

def WIRES_WHY : String :=
  "the joined record's first wires are not each rank's own wires moved to its units, a rank's count of them, in rank " ++
  "order, or a wire after them reads no reader unit, or one in no rank, or in some rank whose units it reads has " ++
  "all its writers of them in that rank"

/-- `ranks-wires`: the joined record `rec`'s wires are each rank's record's wires (`records`, in rank order) moved to
its units (`shiftedWires`), as many as the rank states, then the join's (`joinWireOk`). -/
def wiresOk (rs : Ranks) (rec : Json) (records : Array Json) : Bool :=
  let joined := arrAt rec "wires"
  let own := (List.range rs.of.size).foldl (fun acc r => acc ++ shiftedWires rs r (records.getD r .null)) #[]
  records.size == rs.of.size &&
    (List.range rs.of.size).all (fun r =>
      (arrAt (records.getD r .null) "wires").size == ((rs.of[r]?).map (·.wires)).getD 0) &&
    decide (own.size ≤ joined.size) && (own.zip (joined.extract 0 own.size)).all (fun ab => jeq ab.1 ab.2) &&
    (joined.extract own.size joined.size).all (joinWireOk rs)

/-- What a record's `registered` reads as where it states none. -/
def NO_REGISTERED : Json :=
  Json.mkObj [("openings", .arr #[]), ("params", Json.mkObj []), ("reads", .arr #[]), ("roots", .arr #[]),
    ("tables", .arr #[])]

/-- The keys of a rank record's `registered` the join names, each by its rule (`joinedRegistered`): a rank's record
with any other (values, histories, supplied parameters) is refused until the join names it. -/
def REGISTERED_KEYS : List String := ["formats", "openings", "params", "reads", "roots", "tables"]

def registeredOf (doc : Json) : Json := match getD doc "registered" with | .null => NO_REGISTERED | g => g

/-- The joined record's `registered` as the ranks' records (`records`, in rank order) make it: each rank's parameters,
tables and formats named `r<r>/…`, and its registered reads, table reads and openings at its units, naming them so,
in rank order; none where a rank's has a key the join names no rule for (`REGISTERED_KEYS`). -/
def joinedRegistered (rs : Ranks) (records : Array Json) : Option Json := do
  let regs := (List.range rs.of.size).map fun r => (r, registeredOf (records.getD r .null))
  if regs.any fun (_, g) => (entries g).any fun (k, _) => !REGISTERED_KEYS.contains k then none
  let each (f : Nat → Json → Array Json) : Array Json := regs.foldl (fun acc (r, g) => acc ++ f r g) #[]
  let named (k : String) : Array (String × Json) :=
    regs.foldl (fun acc (r, g) => acc ++ (entries (getD g k)).map fun (n, v) => (rankPrefix r ++ n, v)) #[]
  let moved (r : Nat) (j : Json) : Json := raiseNat j "at" (rs.first (strAt j "reader") r)
  let formats := named "formats"
  pure <| Json.mkObj <|
    [("openings", .arr (each fun r g => (arrAt g "openings").map fun o => moved r (prefixAt o "table" (rankPrefix r)))),
     ("params", Json.mkObj (named "params").toList),
     ("reads", .arr (each fun r g => (arrAt g "reads").map fun x => moved r (prefixAt x "param" (rankPrefix r)))),
     ("roots", .arr (each fun r g => (arrAt g "roots").map fun x => moved r (prefixAt x "table" (rankPrefix r)))),
     ("tables", .arr (each fun r g => (arrAt g "tables").map fun t =>
       prefixAt (prefixAt t "name" (rankPrefix r)) "param" (rankPrefix r)))] ++
    (if formats.isEmpty then [] else [("formats", Json.mkObj formats.toList)])

def READS_WHY : String :=
  "the joined record's registered declarations are not its ranks', each named r<r>/... and at its rank's units, in " ++
  "rank order, or a rank's record declares what the join names no rule for (values, histories, supplied parameters)"

/-- `ranks-reads`: the joined record `rec`'s `registered` is the one its ranks' records make (`joinedRegistered`). -/
def readsOk (rs : Ranks) (rec : Json) (records : Array Json) : Bool :=
  records.size == rs.of.size && match joinedRegistered rs records with
    | some g => jeq g (registeredOf rec)
    | none => false

/-- `ranks-units`, `ranks-wires` and `ranks-reads` of the joined record `rec` against its ranks' records `records`, in
rank order. -/
def recordFailures (rs : Ranks) (rec : Json) (records : Array Json) : List (String × Json) :=
  (if unitsOk rs rec then [] else [("ranks-units", Json.str UNITS_WHY)]) ++
    (if wiresOk rs rec records then [] else [("ranks-wires", Json.str WIRES_WHY)]) ++
    (if readsOk rs rec records then [] else [("ranks-reads", Json.str READS_WHY)])

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

#guard
  let rs : Ranks := ⟨#[⟨"", #[("A", 2), ("B", 2)], 1⟩, ⟨"", #[("A", 2), ("B", 2)], 1⟩]⟩
  let n (k : Nat) : Json := Lean.toJson k
  let wire (a b : Nat) : Json := Json.mkObj [("at", n a), ("base", n b), ("out", .str "o0"), ("port", .str "x"),
    ("radix", .arr #[n 2]), ("reader", .str "A"), ("step", .arr #[n 1]), ("writer", .str "B")]
  let read (p : String) (a : Nat) : Json := Json.mkObj [("at", n a), ("base", n 0), ("param", .str p),
    ("port", .str "w"), ("radix", .arr #[n 2]), ("reader", .str "B"), ("step", .arr #[n 0])]
  let reg (ps : List (String × Json)) (rd : Array Json) : Json := Json.mkObj [("openings", .arr #[]),
    ("params", Json.mkObj ps), ("reads", .arr rd), ("roots", .arr #[]), ("tables", .arr #[])]
  let rank := Json.mkObj [("registered", reg [("w", n 8)] #[read "w" 0]), ("wires", .arr #[wire 0 0])]
  let joined (ws rd : Array Json) : Json := Json.mkObj [("registered", reg [("r0/w", n 8), ("r1/w", n 8)] rd),
    ("templates", .arr #[.arr #[.str "A", n 4], .arr #[.str "B", n 4]]), ("wires", .arr ws)]
  let codes (j : Json) : List String := (recordFailures rs j #[rank, rank]).map (·.1)
  let reads := #[read "r0/w" 0, read "r1/w" 2]
  let wide (b : Nat) (s : Int) : Json := Json.mkObj [("at", n 0), ("base", n b), ("out", .str "o0"), ("port", .str "x"),
    ("radix", .arr #[n 4]), ("reader", .str "A"), ("step", .arr #[Lean.toJson s]), ("writer", .str "B")]
  -- A join wire over both ranks' readers, each rank's reading the other's writers: accepted; over both ranks' readers,
  -- rank 0's reading its own writers: refused.
  codes (joined #[wire 0 0, wire 2 2, wide 3 (-1)] reads) == [] &&
    codes (joined #[wire 0 0, wire 2 2, wide 0 1] reads) == ["ranks-wires"] &&
    codes (joined #[wire 0 0, wire 2 2, wire 2 0] reads) == [] &&
    codes (joined #[wire 0 0, wire 2 0] reads) == ["ranks-wires"] &&
    codes (joined #[wire 0 0, wire 2 2, wire 0 0] reads) == ["ranks-wires"] &&
    codes (joined #[wire 0 0, wire 2 2] #[read "r0/w" 0, read "r1/w" 0]) == ["ranks-reads"]

/-- Every failure of the joined record `rec`'s ranks, read against its ranks' records, served records and
registrations at the serve's root, `ranks-shape` first. -/
def failures (rec : Json) (records : Array Json) (served : Array (String × String)) (regs : Array Registered)
    (root : String) (logged : String → Bool) : List (String × Json) :=
  match Ranks.ofJson? (getD rec "ranks") with
  | none => [("ranks-shape", Json.str SHAPE_WHY)]
  | some rs =>
    executionPhaseFailures rs regs served ++ runFailures rs regs root logged ++ membersFailures rs regs ++
      recordFailures rs rec records

/-- **The ranks' check** (module docstring): each code once, with the detail it first failed with. -/
def check (rec : Json) (records : Array Json) (served : Array (String × String)) (regs : Array Registered)
    (root : String) (logged : String → Bool) : Check :=
  (failures rec records served regs root logged).foldl (fun c x => c.fail x.1 x.2) {}

end Spot.Ranks
