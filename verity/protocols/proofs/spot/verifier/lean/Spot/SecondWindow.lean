/-
**Round 2's reader over the joined record**: a two-stage run's second-stage sessions held to the joined circuit's record
(the verifier's own copy, its digest the Definition's `window`) and to its ranks' registrations, beside
`Spot.SecondStage`, which reads an interior registration and stays as it is until a later switch with its proof.

* **Links** are the record's `boundary_reads` (each a reader template, port, units and field, row `i` the field's leaves
  from `base + Σ d_k · step_k`), read at round 1's boundary port for that field (`linksOf?`).
* **Wires** are the record's `wires`, each a reader span and a writer in closed form (`Spot.Wires`): the first
  `Σ wires_r` are the ranks' own, the rest the join's (`wiresOf?`).
* **Rows** are each declaration's pattern at the units the second sample draws, once per declaration and sampled proof block,
  never unrolled (`rowsOf?`).
* **Parts** are rank members (`Spot.Ranks.memberOf`): each sampled proof block is proved in the session of the rank member that
  holds it, and that session's sample is the sampled block's units in that member.
* **No interior registration.** This reader drops the interior registration and its four checks (`second-interior-order`,
  `second-interior-names`, `second-return`, `second-inputs`) because every row round 2 opens is in a rank window whose
  registration is in the log before the period's challenge, which `ranks-run` checks (`Spot.Ranks`). Without that
  check the interior registration's checks would still be needed.
* **Its checks**: the ranks' (`Spot.Ranks.check`); per session, `Spot.SecondStage`'s `second-header`,
  `second-population`, `second-roots`, `second-sample`, `second-read`, `second-values`, `second-refs`, `second-tables`
  and `second-verified`, read against the rank member in place of an interior part, with the window-domain binding at
  the rank's program, partition digest and run (`portBinding`); and `ranks-wire-tree`: the writer rows a sampled reader
  member reads through a port lie in one rank's window, so its session reads the port from one committed tree.
* **The form**: it runs behind `Spot.SecondStage.FORM`, the one switch, and is refused last (`two-stage-form`) while
  that is off.

* **A member's session** (`ownFor`) is `Spot.SecondStage`'s part of one block: the member's units in its rank's
  window, the interior its rank's docket record (`Given.dockets`). Each port it reads through a wire reads through the
  one its units read it through (`wireFor`, the one rule every check reads a wire by), so `second-header`'s bindings,
  `second-values`' run fields and `second-read` agree on it. `Spot.SecondStage.stmtCodes` runs as it is but for
  `second-read`, which reads the window's rows (`windowReadOk`): a wired port the writer's output port in the docket
  of the writer unit's rank, at that unit's rank-local row. A port a boundary read links is refused until round 1's
  boundary rows are read here. Statements are the sampled members', by position (`second-statement`, `second-order`,
  `second-cover`), and `second-table-rows` is `Spot.SecondStage.tableRowsFailures` over the joined record's wires.

No guarantee reads this module.
-/
module

public import Spot.Ranks
public import Spot.SecondStage
public import Spot.SecondStagePass
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

/-- A read from its record, as a wire whose `writer` is its parameter (`asWire`); none unless it is a read's record. -/
def readOf? (r : Json) : Option Spot.Wires.Wire :=
  if (field? r "writer").isNone && (field? r "out").isNone then (Spot.Wires.Wire.ofJson (asWire r)).toOption else none

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

/-- A declaration's row at a sampled proof block: template `template`'s unit `unit` (the joined circuit's), row `row` of its
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

/-- Pattern `p`'s rows at sampled proof blocks `sample` (`(template, unit)`): one a sampled proof block it covers and row of its
cut. -/
def rowsAt (p : Spot.Wires.Wire) (sample : Array (String × Nat)) : Array Row :=
  sample.foldl (fun acc (t, u) =>
    if t == p.reader && decide (p.start ≤ u) && decide (u < p.start + p.readerUnits) then
      acc ++ ((List.range p.cuts).map fun c =>
        ({ template := t, port := p.port, unit := u, row := p.cut0 + c, of := p.writer, out := p.out,
           source := p.writerAt ((u - p.start) * p.cuts + c) } : Row)).toArray
    else acc) #[]

/-- The rows of the joined record `rec`'s declarations (its wires, boundary reads and registered reads) at the sampled
proof blocks `sample`, `(template, unit)` each: once per declaration and sampled proof block, never unrolled. -/
def rowsOf? (rec : Json) (sample : Array (String × Nat)) : Option (Array Row) := do
  let reads ← (listOf? (getD rec "registered") "reads").bind (·.mapM readOf?)
  let ps := (← patternsOf? rec) ++ (← boundaryOf? rec) ++ reads
  pure (ps.foldl (fun acc p => acc ++ rowsAt p sample) #[])

def sameMember (a b : Spot.Ranks.Member) : Bool := a.rank == b.rank && a.template == b.template && a.index == b.index

/-- The rank members that hold the sampled proof blocks `sample` (`(template, unit)`, the joined circuit's units), each once,
in the order of its first sampled proof block (`Spot.Ranks.memberOf`); none where a unit is in no member. -/
def partsOf? (rs : Spot.Ranks.Ranks) (regs : Array Spot.Ranks.Registered) (sample : Array (String × Nat)) :
    Option (Array Spot.Ranks.Member) := do
  let ms ← sample.mapM fun (t, u) => Spot.Ranks.memberOf rs regs t u
  pure (ms.foldl (fun acc m => if acc.any (sameMember m) then acc else acc.push m) #[])

/-- The least `i < n` where `p i` holds, `p` false and then true over `[0, n)`; `n` where it never holds. -/
def bisect (n : Nat) (p : Nat → Bool) : Nat :=
  go 0 n (n + 1)
where
  go (lo hi : Nat) : Nat → Nat
    | 0 => lo
    | fuel + 1 =>
      if lo < hi then
        let mid := (lo + hi) / 2
        if p mid then go lo mid fuel else go (mid + 1) hi fuel
      else lo

/-- **The joined record's wires as the per-session checks read them**: sorted by reader template, port and first reader
unit (ties in the record's order), and at each place the most end (`start + readerUnits`) of its port's wires up to
there. The wires of a port that meet some units are then one run of it, found by two bisections (`Index.meets`), so a
sampled member costs a logarithm of the wires and the wires it meets, never a pass over the wires. -/
structure Index where
  ws : Array Spot.Wires.Wire
  ends : Array Nat

def pairLt (a b : String × String) : Bool := decide (a.1 < b.1) || (a.1 == b.1 && decide (a.2 < b.2))

def Index.of (ws : Array Spot.Wires.Wire) : Index :=
  let s := (ws.zipIdx.qsort fun (a, i) (b, j) =>
    pairLt (a.reader, a.port) (b.reader, b.port) ||
      (a.reader == b.reader && a.port == b.port && (a.start < b.start || (a.start == b.start && i < j)))).map (·.1)
  let ends := (s.foldl (fun ((prev, acc) : Option Spot.Wires.Wire × Array Nat) w =>
    let e := w.start + w.readerUnits
    let m := match prev with
      | some v => if v.reader == w.reader && v.port == w.port then max (acc.back?.getD 0) e else e
      | none => e
    (some w, acc.push m)) (none, #[])).2
  ⟨s, ends⟩

/-- The run of the index holding template `t`'s port `port`'s wires. -/
def Index.block (ix : Index) (t port : String) : Nat × Nat :=
  let key (i : Nat) : String × String := ((ix.ws[i]?).map fun w => (w.reader, w.port)).getD ("", "")
  (bisect ix.ws.size fun i => !pairLt (key i) (t, port), bisect ix.ws.size fun i => pairLt (t, port) (key i))

/-- The wires reading template `t`'s port `port` that meet units `[lo, lo + n)`, by first unit. -/
def Index.meets (ix : Index) (t port : String) (lo n : Nat) : Array Spot.Wires.Wire :=
  let (a, b) := ix.block t port
  let i := a + bisect (b - a) fun k => decide (lo < ix.ends.getD (a + k) 0)
  let j := a + bisect (b - a) fun k => decide (lo + n ≤ ((ix.ws[a + k]?).map (·.start)).getD 0)
  (ix.ws.extract i j).filter (·.readsIn (t, lo, n))

/-- Template `t`'s wired ports, each once, by name. -/
def Index.ports (ix : Index) (t : String) : Array String :=
  let a := bisect ix.ws.size fun i => ((ix.ws[i]?).map fun w => !decide (w.reader < t)).getD true
  let b := bisect ix.ws.size fun i => ((ix.ws[i]?).map fun w => decide (t < w.reader)).getD true
  (ix.ws.extract a b).foldl (fun acc w => if acc.back? == some w.port then acc else acc.push w.port) #[]

-- The index's wires of a port meeting some units are the scan's, by first unit; its ports are the template's.
#guard
  let wire (t p : String) (a n : Nat) : Spot.Wires.Wire :=
    { reader := t, port := p, writer := "W", out := "o0", start := a, radix := [n], base := 0, step := [1], cut0 := 0,
      cuts := 1 }
  let ws := #[wire "B" "x" 4 2, wire "A" "y" 0 8, wire "A" "x" 6 1, wire "A" "x" 0 3, wire "A" "x" 3 2, wire "A" "y" 2 1]
  let ix := Index.of ws
  let scan (t p : String) (lo n : Nat) :=
    ((ws.filter fun w => w.reader == t && w.port == p && w.readsIn (t, lo, n)).qsort fun a b => a.start < b.start).map
      (·.start)
  ([("A", "x", 0, 1), ("A", "x", 2, 2), ("A", "x", 5, 1), ("A", "x", 7, 3), ("A", "y", 3, 1), ("A", "y", 9, 1),
    ("B", "x", 0, 9), ("C", "x", 0, 9)].all fun (t, p, lo, n) =>
      (ix.meets t p lo n).map (·.start) == scan t p lo n) &&
    ix.ports "A" == #["x", "y"] && ix.ports "B" == #["x"] && ix.ports "C" == #[]

def WIRE_TREE_WHY : String :=
  "a sampled reader member reads one port through writer rows in more than one rank's window, or in none: its " ++
  "session reads each wired port from one rank's committed tree"

/-- The ranks whose units hold the writer rows member `m`'s units read through wire `w`: none read, `some []`; all in
one rank, that rank, by the closed-form box (`Spot.Wires.Wire.box`) on the writers of the reader indices in `m`, whose
two ends in one rank put every writer between them there; else none. -/
def writerRanks (rs : Spot.Ranks.Ranks) (m : Spot.Ranks.Member) (w : Spot.Wires.Wire) : Option (List Nat) :=
  let lo := (rs.segment m.template m.rank).1 + m.base
  if !w.readsIn (m.template, lo, m.units) then some [] else
  let (a, b) := w.indicesIn lo m.units
  let (x, y) := w.box a b
  if x < 0 then none else
  match rs.rankOf w.writer x.toNat x.toNat, rs.rankOf w.writer y.toNat y.toNat with
  | some p, some q => if p == q then some [p] else none
  | _, _ => none

/-- Member `m` reads each wired port of its template from one rank's tree: the writer rows every wire of that port
reads in `m` lie in one rank (`writerRanks`). Writer members don't matter: a rank's docket commits one tree a port over
all its members, each member a range of its leaves. -/
def treeOk (rs : Spot.Ranks.Ranks) (ix : Index) (m : Spot.Ranks.Member) : Bool :=
  let lo := (rs.segment m.template m.rank).1 + m.base
  (ix.ports m.template).all fun port =>
    match (ix.meets m.template port lo m.units).toList.mapM (writerRanks rs m) with
    | some rss => (distinct rss.flatten.toArray).size ≤ 1
    | none => false

/-- `ranks-wire-tree` of the sampled members `ms` over the joined record `rec`: each reads every wired port of its
template from one rank's committed tree (`treeOk`), the one tree its session's registered read of the port names.
Work per sampled member and wire, never per unit. -/
def wireTreeFailures (rs : Spot.Ranks.Ranks) (rec : Json) (ms : Array Spot.Ranks.Member) : List (String × Json) :=
  match patternsOf? rec with
  | some ws => if ms.all (treeOk rs (Index.of ws)) then [] else [("ranks-wire-tree", Json.str WIRE_TREE_WHY)]
  | none => [("ranks-wire-tree", Json.str WIRE_TREE_WHY)]

/-- What `Spot.SecondStage`'s statement checks read of the joined record `rec` that this reader builds: its wires. -/
def ownOf (rec : Json) : Spot.SecondStage.Own :=
  { sample := #[], parts := none, interior := .null, boundary := .null, drawn := #[], wires := wiresOf? rec }

/-- What round 2 holds of the run beside the joined record and its ranks, as `Spot.SecondStage.Own` holds it of an
interior run: each rank's docket record, in rank order (its ports' roots and domains, its program, partition and
`window.run`); round 1's boundary record; what the verdicts of record name of each session they accepted
(`Spot.SecondStage.verifiedOf`); and the auditor's reference header and registered reads of each template, taken as
given. -/
structure Given where
  dockets : Array Json
  boundary : Json := .null
  verified : Array Spot.SecondStage.Checked := #[]
  refs : Array (String × Json) := #[]
  refReads : Array (String × Json) := #[]

/-- **The wire a member reads a port through**, the one rule every per-session check reads a wire by: of the joined
record's wires reading template `t`'s port `port`, those meeting the member's units `[lo, lo + n)` (the joined
circuit's; `Index.meets`). None meets them: `some none`, the port is no wired port of the member's. All of one writer's output port,
uncut, together covering every unit: `some (some w)`. Else none: the member reads the port through two writers, or
partly through none, and no one session's header binds the port. A port two writer templates write, each over its
own reader units (a select's first step and the rest), is then one writer a member wherever the member rule cuts at
their bounds. Work per wire, never per unit. -/
def wireFor (ix : Index) (t port : String) (lo n : Nat) : Option (Option Spot.SecondStage.Wire) :=
  let meet := ix.meets t port lo n
  match meet[0]? with
  | none => some none
  | some w0 =>
    let covered := meet.foldl (fun c w => c + (min (lo + n) (w.start + w.readerUnits) - max lo w.start)) 0
    if meet.all (fun w => w.writer == w0.writer && w.out == w0.out && w.cuts == 1) && covered == n then
      some (some ⟨t, port, w0.writer, w0.out⟩)
    else none

/-- Member `m`'s wires, its units from `lo` in the joined circuit: one a wired port of its template (`wireFor`); none
where a port has no one wire. -/
def memberWires (ix : Index) (m : Spot.Ranks.Member) (lo : Nat) : Option (Array Spot.SecondStage.Wire) :=
  (ix.ports m.template).foldlM (fun acc p => do
    match ← wireFor ix m.template p lo m.units with
    | some w => pure (acc.push w)
    | none => pure acc) #[]

/-- The writer template and unit joined unit `u` of template `t` reads at port `port`: the one of the joined record's
wires (`Index.meets`) whose reader units hold `u`, at its first row. -/
def writerUnit (ix : Index) (t port : String) (u : Nat) : Option (String × Int) := do
  let w ← (ix.meets t port u 1)[0]?
  pure (w.writer, w.writerAt ((u - w.start) * w.cuts))

/-- Member `m`'s session as `Spot.SecondStage` reads a part's: one block (`sample` `#[0]`), its part `m`'s units in
its rank's window, the interior its rank's docket record, its sample the sampled proof blocks `drawn` of its template in that
window, its links the joined record's `links` of its template, and its wires `wires` (`memberWires`). -/
def ownFor (g : Given) (links : Array Spot.SecondStage.Link) (m : Spot.Ranks.Member) (drawn : Array Nat)
    (wires : Array Spot.SecondStage.Wire) : Spot.SecondStage.Own :=
  { sample := #[0], parts := some #[⟨m.template, m.base, m.units⟩], interior := g.dockets.getD m.rank .null,
    boundary := g.boundary, drawn, links := some (links.filter (·.template == m.template)), verified := g.verified,
    refs := g.refs, refReads := g.refReads, wires := some wires }

/-- A wired port's registered read `e` in the window form, for part `p` of rank `r` reading through `w`: the writer's
output port of the docket of the rank that holds the writer unit the part's first unit reads, at its root, domain and
leaves, each slot only its writer unit's row there (`writerUnit`, rank-local), every slot's writer in that rank. -/
def wiredReadOk (rs : Spot.Ranks.Ranks) (ix : Index) (dockets : Array Json)
    (p : Spot.SecondStage.Part) (r : Nat) (w : Spot.SecondStage.Wire) (v : String) (e : Json) : Bool :=
  let first := (rs.segment p.template r).1
  (do
    let (_, x0) ← writerUnit ix p.template w.port (first + p.lo)
    let r' ← if x0 < 0 then none else rs.rankOf w.writer x0.toNat x0.toNat
    let d := dockets.getD r' .null
    let wfirst := (rs.segment w.writer r').1
    let (_, x) ← (Spot.Record.byPort d).find? (·.1 == v)
    let leaves ← nat? (getD x "leaves")
    pure (v == Spot.SecondStage.portOf (Spot.SecondStage.byTemplate d) w.writer w.wport &&
      Spot.SecondStage.sameAt e x && Spot.SecondStage.placedOk 1 p e leaves fun u => do
        let (_, y) ← writerUnit ix p.template w.port (first + u)
        if decide ((wfirst : Int) ≤ y) && rs.rankOf w.writer y.toNat y.toNat == some r' then some (y.toNat - wfirst)
        else none) : Option Bool).getD false

/-- **`second-read` in the window form**: member part `p` of rank `r`'s session port `port` may read registered-reads
entry `e`: a port a wire names (`wireFor`), only the writer's output port in its rank's docket (`wiredReadOk`); a port
no link or wire names, its own port of its rank's docket, each slot only its own first row in a tree of the rank's
units of its template, or a value either record registers, never a boundary port; a port a boundary read links,
refused until round 1's boundary rows are read here. -/
def windowReadOk (rs : Spot.Ranks.Ranks) (ix : Index) (dockets : Array Json)
    (own : Spot.SecondStage.Own) (named : Bool) (p : Spot.SecondStage.Part) (r : Nat) (port : String) (e : Json) :
    Bool :=
  let units := (rs.segment p.template r).2 - (rs.segment p.template r).1
  match getD e "value", Spot.SecondStage.linkOf own p.template port, Spot.SecondStage.wireOf own p.template port with
  | .str _, some _, _ => false
  | .str v, none, some w => wiredReadOk rs ix dockets p r w v e
  | .str v, none, none =>
    match (Spot.Record.byPort own.interior).find? (·.1 == v), (Spot.Record.byPort own.boundary).find? (·.1 == v) with
    | some (_, x), _ => v == Spot.SecondStage.portOf named p.template port && Spot.SecondStage.sameAt e x &&
        Spot.SecondStage.rowsOk 1 p e units fun u => some u
    | none, some _ => false
    | none, none =>
      match Spot.SecondStage.registeredEntry own.boundary v <|> Spot.SecondStage.registeredEntry own.interior v with
      | some x => Spot.SecondStage.sameAt e x
      | none => false
  | _, _, _ => false

/-- Member `m`'s position among its rank's docket members, the `k` of its session's input set `served/<run>/m<k>`. -/
def positionOf (regs : Array Spot.Ranks.Registered) (m : Spot.Ranks.Member) : Nat :=
  match regs.find? (·.rank == m.rank) with
  | some g => (g.members.findIdx? fun h => h.key == m.template ++ "/c" ++ toString m.index).getD g.members.size
  | none => 0

/-- Statement `s`'s failing checks against member `m`'s session `own`: `Spot.SecondStage.stmtCodes` of its part, but
`second-read` in the window form (`windowReadOk`, with `linkedRead` and `wiredRead`). -/
def memberCodes (rs : Spot.Ranks.Ranks) (ix : Index) (g : Given) (own : Spot.SecondStage.Own)
    (m : Spot.Ranks.Member) (k : Nat) (s : Spot.SecondStage.Stmt) : List String :=
  let named := Spot.SecondStage.byTemplate own.interior
  let p : Spot.SecondStage.Part := ⟨m.template, m.base, m.units⟩
  let reads := entries (getD s.registered "ports")
  let read := reads.all (fun (port, e) => windowReadOk rs ix g.dockets own named p m.rank port e) &&
    Spot.SecondStage.linkedRead own m.template reads && Spot.SecondStage.wiredRead own m.template reads
  ((Spot.SecondStage.stmtCodes own named k p s).filter (· != "second-read")) ++ (if read then [] else ["second-read"])

def TWO_WRITERS_WHY : String :=
  "the member reads a port through two writers, or partly through none: no one session's header binds it"

def DESIGN_WHY : String := "the joined record's wires or boundary reads are none: no session reads by them"

/-- The per-session failures of statements `ss` against the sampled members `ms`, the `i`-th statement the `i`-th
member's session (`partsOf?`'s order): `second-statement`, `second-order` (a statement for another template than its
member's, or for no member) and `second-cover` (a member no statement is for), else its member's codes
(`memberCodes`); then `second-table-rows` as `Spot.SecondStage` reads it over the joined record's wires. -/
def sessionFailures (g : Given) (rs : Spot.Ranks.Ranks) (regs : Array Spot.Ranks.Registered) (rec : Json)
    (ms : Array Spot.Ranks.Member) (sample : Array (String × Nat)) (ss : List Json) : List (String × Json) :=
  let detail := Spot.SecondStage.detailOf
  let each := match linksOf? rec, patternsOf? rec with
    | some links, some ws => let ix := Index.of ws; (List.range (max ms.size ss.length)).flatMap fun i =>
      match ms[i]?, ss[i]? with
      | some m, none => [("second-cover", detail i (.str m.template))]
      | none, some _ => [("second-order", detail i .null)]
      | none, none => []
      | some m, some j => match Spot.SecondStage.Stmt.ofJson? j with
        | none => [("second-statement", detail i .null)]
        | some s =>
          if s.template != m.template then [("second-order", detail i (.str s.template))] else
          let first := (rs.segment m.template m.rank).1
          let lo := first + m.base
          match memberWires ix m lo with
          | none => [("second-read", Json.mkObj [("statement", Lean.toJson i), ("template", .str s.template),
              ("why", .str TWO_WRITERS_WHY)])]
          | some wires =>
            let mine := sample.filter fun (t, u) => t == m.template && decide (lo ≤ u) && decide (u < lo + m.units)
            let drawn := mine.map fun (_, u) => u - first
            (memberCodes rs ix g (ownFor g links m drawn wires) m (positionOf regs m) s).map
              fun c => (c, detail i (.str s.template))
    | _, _ => [("second-read", Json.str DESIGN_WHY)]
  each ++ Spot.SecondStage.tableRowsFailures (ownOf rec) ss

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
  ⟨r, "", "", "", #[⟨"A/c0", 0, 2, #[]⟩, ⟨"B/c0", 0, 1, #[]⟩, ⟨"B/c1", 1, 1, #[]⟩]⟩

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
-- Rank 0's member `A/c0` reads writer units 0 and 1, two writer members of one rank (a step's two selects), through its
-- wire: accepted. Through the third wire it reads writer units 1 and 2, of two ranks' trees; through `z`, two wires
-- read one port from two ranks: both refused.
#guard
  let ms := (partsOf? exRanks exRegs #[("A", 1), ("A", 3)]).getD #[]
  let recOf (ws : Array Json) : Json := Json.mkObj [("wires", .arr ws)]
  let one (a b : Nat) : Json := Json.mkObj [("at", num a), ("base", num b), ("out", .str "o0"), ("port", .str "z"),
    ("radix", .arr #[num 1]), ("reader", .str "A"), ("step", .arr #[num 1]), ("writer", .str "B")]
  (wireTreeFailures exRanks (recOf #[wireJ 0 0, wireJ 2 2]) ms).isEmpty &&
    (wireTreeFailures exRanks (recOf #[wireJ 0 0, wireJ 2 2, wireJ 0 1 "y"]) ms).map (·.1) == ["ranks-wire-tree"] &&
    (wireTreeFailures exRanks (recOf #[one 0 0, one 1 1]) ms).isEmpty &&
    (wireTreeFailures exRanks (recOf #[one 0 0, one 1 2]) ms).map (·.1) == ["ranks-wire-tree"]
-- A registered read listing `table_rows` at its key port: `KEY_ROWS_WHY` where the key is wired, else `TABLE_ROWS_WHY`.
#guard
  let stmt := Json.mkObj [("draw", Json.mkObj []), ("header", Json.mkObj []),
    ("registered", Json.mkObj [("ports", Json.mkObj [("tok", Json.mkObj [("table_rows", num 4)])])]),
    ("template", .str "A")]
  let why (j : Json) : List String :=
    (sessionFailures { dockets := #[] } exRanks exRegs j #[] #[] [stmt]).filterMap fun (c, d) =>
      if c == "second-table-rows" then (match getD d "why" with | .str w => some w | _ => none) else none
  why (Json.mkObj [("wires", .arr #[wireJ 0 0 "tok"])]) == [Spot.SecondStage.KEY_ROWS_WHY] &&
    why (Json.mkObj [("wires", .arr #[wireJ 0 0])]) == [Spot.SecondStage.TABLE_ROWS_WHY]

-- A port two select templates write, `F` over step 1's reader unit and `L` over the rest: a member of step 1's unit
-- reads it through `F`, one of the rest through `L`, one spanning both through neither, nor one partly unwired.
#guard
  let sel (writer : String) (a : Nat) : Spot.Wires.Wire :=
    { reader := "A", port := "live", writer, out := "o0", start := a, radix := [1], base := 0, step := [1], cut0 := 0,
      cuts := 1 }
  let ws := #[sel "F" 0, sel "L" 1]
  let w (x : String) : Option (Option Spot.SecondStage.Wire) := some (some ⟨"A", "live", x, "o0"⟩)
  let ix := Index.of ws
  wireFor ix "A" "live" 0 1 == w "F" && wireFor ix "A" "live" 1 1 == w "L" && wireFor ix "A" "live" 0 2 == none &&
    wireFor ix "A" "x" 0 2 == some none && wireFor (Index.of #[sel "F" 0]) "A" "live" 0 2 == none
-- Rank 0's member `A/c0` reads port `x` through a join wire from rank 1's `B` units 2 and 3: rank 1's docket's `B/o0`
-- at rows 0 and 1; port `y`, unwired, its own rank's `A/y` at its own rows.
#guard
  let root (port r : String) : Json :=
    Json.mkObj [("domain", .str ("d" ++ r)), ("leaves", num 2), ("port", .str port), ("root", .str r)]
  let dockets := #[Json.mkObj [("roots", .arr #[root "A/y" "a0"])], Json.mkObj [("roots", .arr #[root "B/o0" "b1"])]]
  let ws : Array Spot.Wires.Wire :=
    #[{ reader := "A", port := "x", writer := "B", out := "o0", start := 0, radix := [2], base := 2, step := [1],
        cut0 := 0, cuts := 1 }]
  let own := ownFor { dockets } #[] ⟨0, "A", 0, 0, 2⟩ #[0] #[⟨"A", "x", "B", "o0"⟩]
  let e (v r : String) (reads : List Nat) : Json := Json.mkObj [("domain", .str ("d" ++ r)), ("leaves", num 2),
    ("reads", .arr (reads.map num).toArray), ("root", .str r), ("value", .str v)]
  let ok (port : String) (j : Json) : Bool :=
    windowReadOk exRanks (Index.of ws) dockets own true ⟨"A", 0, 2⟩ 0 port j
  ok "x" (e "B/o0" "b1" [0, 1]) && !ok "x" (e "B/o0" "b1" [1, 0]) && !ok "x" (e "A/y" "a0" [0, 1]) &&
    ok "y" (e "A/y" "a0" [0, 1]) && !ok "y" (e "A/y" "a0" [1, 0]) && !ok "y" (e "B/o0" "b1" [0, 1])
-- The statements by position: a member that reads `live` through both selects, a missing statement, one for another
-- template.
#guard
  let selJ (writer : String) (a : Nat) : Json := Json.mkObj [("at", num a), ("base", num 0), ("out", .str "o0"),
    ("port", .str "live"), ("radix", .arr #[num 1]), ("reader", .str "A"), ("step", .arr #[num 1]),
    ("writer", .str writer)]
  let rec2 := Json.mkObj [("wires", .arr #[selJ "F" 0, selJ "L" 1])]
  let stmt (t : String) : Json := Json.mkObj [("draw", Json.mkObj []), ("header", Json.mkObj []),
    ("registered", .null), ("template", .str t)]
  let codes (ss : List Json) : List String :=
    (sessionFailures { dockets := #[] } exRanks exRegs rec2 #[⟨0, "A", 0, 0, 2⟩] #[("A", 1)] ss).map (·.1)
  codes [stmt "A"] == ["second-read"] && codes [] == ["second-cover"] && codes [stmt "B"] == ["second-order"] &&
    codes [stmt "A", stmt "A"] == ["second-read", "second-order"]

def PARTS_WHY : String := "a sampled proof block is in no rank member"

/-- Every failure of statements `stmts` over the joined record `rec` and what round 2 holds beside it `g`, its ranks'
records `records`, served records `served` and registrations `regs` at the serve's root `root`, `logged` whether a
registration is in the log before the period's challenge, and the sampled proof blocks `sample`; when there is none and the
form isn't allowed, its refusal. -/
def failures (allow : Bool) (g : Given) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array (String × Nat))
    (stmts : List Json) : List (String × Json) :=
  let ranks := Spot.Ranks.failures rec records served regs root logged
  let found := match Spot.Ranks.Ranks.ofJson? (getD rec "ranks") with
    | none => ranks
    | some rs => match partsOf? rs regs sample with
      | none => ranks ++ [("ranks-members", Json.str PARTS_WHY)]
      | some ms => ranks ++ wireTreeFailures rs rec ms ++ sessionFailures g rs regs rec ms sample stmts
  if found.isEmpty && !allow then [("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)] else found

/-- **Round 2's check over the joined record** (module docstring), the form allowed when `allow` (`Spot.SecondStage.FORM`
in the run): each code once, with the detail it first failed with. -/
def check (allow : Bool) (g : Given) (rec : Json) (records : Array Json) (served : Array (String × String))
    (regs : Array Spot.Ranks.Registered) (root : String) (logged : String → Bool) (sample : Array (String × Nat))
    (stmts : List Json) : Check :=
  (failures allow g rec records served regs root logged sample stmts).foldl (fun c x => c.fail x.1 x.2) {}

/-- A registration's member from `{key, base, units, outs}`, `outs` a list of `[port, leaves]`. -/
def heldOf? (j : Json) : Option Spot.Ranks.Held := do
  let .str key := getD j "key" | none
  let .arr os := getD j "outs" | none
  let outs ← os.mapM fun (o : Json) => match o with
    | .arr #[.str p, n] => do pure (p, ← nat? n)
    | _ => none
  pure { key, base := ← nat? (getD j "base"), units := ← nat? (getD j "units"), outs }

/-- A rank's registration from `{rank, digest, circuit, run, members}`. -/
def registeredOf? (j : Json) : Option Spot.Ranks.Registered := do
  let .str digest := getD j "digest" | none
  let .str circuit := getD j "circuit" | none
  let .str run := getD j "run" | none
  let .arr ms := getD j "members" | none
  pure { rank := ← nat? (getD j "rank"), digest, circuit, run, members := ← ms.mapM heldOf? }

/-- A list of `[name, x]` pairs, each name a string, `x` read by `f`. -/
def pairsOf? {α : Type} (f : Json → Option α) (j : Json) : Option (Array (String × α)) := do
  let .arr xs := j | none
  xs.mapM fun (x : Json) => match x with
    | .arr #[.str k, v] => do pure (k, ← f v)
    | _ => none

/-- A verdict of record's `{header, registered, draw, tables}` (`Spot.SecondStage.Checked`). -/
def checkedOf? (j : Json) : Option Spot.SecondStage.Checked := do
  let .str header := getD j "header" | none
  pure { header, registered := getD j "registered", draw := getD j "draw", tables := getD j "tables" }

def isObj (j : Json) : Bool := match j with | .obj _ => true | _ => false

def str? (j : Json) : Option String := match j with | .str s => some s | _ => none

/-- The `second_window` piece: `{allow, given, record, records, served, registrations, root, logged, sample, statements}`
to `{failures}`, each `[code, detail]` in `failures`' order, `given` `{dockets, boundary, verified, refs, ref_reads}`. -/
def piece (j : Json) : Except String Json := do
  let .bool allow := getD j "allow" | throw "allow is not a boolean"
  let gj := getD j "given"
  let .arr dockets := getD gj "dockets" | throw "given.dockets is not a list"
  let some verified := (match getD gj "verified" with | .arr a => a.mapM checkedOf? | _ => none)
    | throw "given.verified is not a list of {header, registered, draw, tables}, header a string"
  let some refs := pairsOf? some (getD gj "refs") | throw "given.refs is not a list of [name, x]"
  let some refReads := pairsOf? some (getD gj "ref_reads") | throw "given.ref_reads is not a list of [name, x]"
  let rec_ := getD j "record"
  unless isObj rec_ do throw "record is not an object"
  let .arr records := getD j "records" | throw "records is not a list"
  unless records.all isObj do throw "records is not a list of objects"
  let some served := pairsOf? str? (getD j "served")
    | throw "served is not a list of [digest, circuit], strings"
  let some regs := (match getD j "registrations" with | .arr a => a.mapM registeredOf? | _ => none)
    | throw "registrations is not a list of {rank, digest, circuit, run, members}"
  let .str root := getD j "root" | throw "root is not a string"
  let some logged := (match getD j "logged" with | .arr a => a.mapM str? | _ => none)
    | throw "logged is not a list of strings"
  let some sample := pairsOf? nat? (getD j "sample") | throw "sample is not a list of [template, unit]"
  let .arr stmts := getD j "statements" | throw "statements is not a list"
  let g : Given := { dockets, boundary := getD gj "boundary", verified, refs, refReads }
  let fs := failures allow g rec_ records served regs root (logged.contains ·) sample stmts.toList
  return Json.mkObj [("failures", .arr (fs.map fun (c, d) => Json.arr #[.str c, d]).toArray)]

end Spot.SecondWindow
