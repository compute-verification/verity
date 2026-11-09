/-
**Round 2's statement check**: a two-stage run's second-stage sessions held, each by its statement, to the interior
registration and to the second draw's cells (`check`). `Run.run` adds it to the interior registration's audit record as
`checks.statements` (`attach`), and `service.Stream` runs its mirror (`second_stage`), held to it by difftest
(`second_statements`).

* **The parts and their cells** are the verifier's own: one part per stratum of the interior program under the check's
  query (`Partition.strata`, which `population-order` holds to the population's units, template by template), each one
  range of `Q·n_t` slots from `off_t`, `Q` the replay sample's size (`parts`). Template `t`'s cell in the sample's
  `i`-th replay block `S[i]` is its `n_t` slots from `off_t + i·n_t` (`Part.span`). One template's cells are
  `Select.secondLaw`'s strata.
* **A statement** (`Stmt`) is one second-stage session's, as its files state it: the template its circuit is an
  instance of (its stratum's name), its public file's header, its session record's sample (`unit_draw`), and the
  registered reads it was verified against (`--registered`), null for none.
* **Its checks**, each a code (`stmtCodes`, `stmtFailures`):
  - `second-statement`: it is not one;
  - `second-template`: its template is none of the parts', or an earlier statement's;
  - `second-header`: its header is not the reference header the auditor's params give its template (`Own.refs`, the
    registered reference statement's, fixed before the run) placed in the run (`atRun`), both with the run's fields
    blanked (`RUN_FIELDS`: `instances`, `units.indices`, `frame_v3.roots` and `frame_v3.domain_ids`, which the next two
    checks hold to the run; `units.blocks`, the indices cut into blocks, which the session's verifier holds to them;
    and `registered`, whose reads `second-read` holds), or the params give its template none.
    Placed in the run, the reference's `set` is the session's input set, `served/<run>/m<k>` (`setName`: `<run>` the
    interior record's `window.run`, `k` its template's position among the parts, `posOf`), and its
    `frame_v3.bindings` are each of its `frame_v3.schemas` ports' bindings in the run (`portBinding`: the window-domain
    rule over the interior record's program, partition and that port's leaves): the verifier computes both, so the
    prover chooses neither;
  - `second-population`: its header's `units.indices` are not its part's slots in order, or its `instances` not their
    count;
  - `second-roots`: its scheme is not the interior record's, or its `frame_v3` roots and `domain_ids`, its registered
    reads aside, are not the record's for its template's ports (port `p` of template `t` is the record's `t/p` when the
    record names its ports by template, else `p`), each of which it roots or reads;
  - `second-sample`: its sample is not the verifier's own, the second draw's units in its part as a subset of its slots
    (`ownDraw`);
  - `second-read`: a port's registered read (`readOk`) is not, for a port a link names, the boundary port it names,
    each slot reading only its cell's replay block's rows (`[S[i]·r, (S[i] + 1)·r)`, `r` the port's leaves over the
    boundary population); else its own port of the interior record, each slot `u` reading only its own rows
    (`[(u - off_t)·r, (u - off_t + 1)·r)`, `r` the port's leaves over the part's `Q·n_t` slots), or a value either
    registration registers; each at its root, domain and leaves. Or a port a link names is not read.
  - `second-refs`: its header shares rows (`shared_rows`: tables whose rows its instances read by the refs in its public
    file's body) at a port it doesn't read registered (`refsOk`). At a port it reads registered, the session's verifier
    holds each instance's ref to the row at the position the read gives its unit (`Flock.Registered.check`), and
    `second-read` holds the read to the slot's own rows or its cell's replay block's; at any other it only bounds a ref
    by its table, so a ref could point a drawn unit at another unit's rows.
* **The links** (`Link`) are the verifier's own, derived from the replay block's registered Definition and never
  supplied (`replayLinks`): a template its body calls once that reads one of its parameters alone reads that boundary
  port, and one whose output it returns alone writes its output's, or, where it returns several fields, each one
  template's whole output, its field's (`o<j>` for field `j`).
* **The interior record's order**: `second-interior-order` when the replay sample the interior record names
  (`window.replay_sample`, which the proof server writes) is not the verifier's own, `S`, in its order
  (`interiorOrderOk`). The record then states that its blocks are `S`'s, block `S[i]`'s cell `off_t + i·n_t`, rather
  than leaving it to convention. It is checked before any statement is read, and fails the check whatever else does.
* **Over the statements**: `second-cells` when the interior program's strata make no parts, `second-links` when the
  replay block's Definition makes no links (`linksOf`) or one names a template of no part, `second-statements` when no
  statements are given, `second-order` for each statement for a part other than the one at its position (round 2 reads
  the statements by position, the `i`-th the `i`-th part's, in the parts' order, the interior program's strata's),
  `second-cover` for each part no statement is for (every part has one, drawn by the second draw or not), and
  `second-reported` when the `{value: root}` the verifiers of record report the sessions read (`Record.sessionRegistered`)
  is not the statements' (`statementReads`). A passing check (`check_ok`) holds the interior record to the replay
  sample in its order (`interiorOrderOk_sample`), the statements to the parts, one each by position, every header to
  its template's reference placed in the run but for the run's fields (`check_ok_header`), every port a header shares
  rows at to a registered read (`check_ok_refs`), every session's sample to the second draw, cell by cell, and every
  read it reports to a statement's.
* **The form**: once every check passes, the run is refused last (`two-stage-form`) unless `FORM` allows it; while it
  doesn't, no run passes (`check_refused`).
-/
module

public import Flock.FV3
public import Flock.Program
public import Spot.Record
meta import Flock.FV3
meta import Flock.Program
meta import Spot.Record

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.SecondStage

open Lean (Json)

/-- **The switch**: whether the verifier accepts a two-stage run whose every check of round 2 passes. Off, it refuses
each one last (`two-stage-form`): the two-stage guarantee (`Flock.SecurityProofs.SampledKSTwo`) takes its sessions'
openings (`SessionOpensTwo`) as a fact of the code, and SPOT[Flock] derives that fact only from its layout's `Embeds` and
`FrameIs`, both open. -/
def FORM : Bool := false

def FORM_WHY : String :=
  "refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the soundness " ++
  "proof does not cover"

def CELLS_WHY : String :=
  "the interior program's strata are not one range each, of a positive multiple of the replay sample's size"

def STATEMENTS_WHY : String :=
  "a two-stage outcome names its second-stage sessions' statements: each one's template, header, draw and registered reads"

def REPORTED_WHY : String :=
  "the registered reads the verifiers of record report are not the statements' registered reads, value by value"

def LINKS_WHY : String :=
  "the replay block's Definition reads a parameter, or returns, other than through templates of the interior's " ++
  "parts, each called once and reading one parameter alone, returned alone, or returned whole as one field each"

def INTERIOR_ORDER_WHY : String :=
  "the interior record's replay sample (window.replay_sample) is not the verifier's own, in its order"

/-- A template's part of the interior population: `Q·n` slots from `lo`, `n` in each sampled replay block. -/
structure Part where
  template : String
  lo : Nat
  n : Nat
  deriving Repr, Inhabited, DecidableEq

/-- Slot `u` is one of part `p`'s, in a sample of `q` replay blocks. -/
def Part.holds (q : Nat) (p : Part) (u : Nat) : Bool := decide (p.lo ≤ u) && decide (u < p.lo + q * p.n)

/-- Slot `u`'s place in the sample: the index of its cell's replay block. -/
def Part.span (p : Part) (u : Nat) : Nat := (u - p.lo) / p.n

/-- A stratum's part in a sample of `q` replay blocks: its one range, of a positive multiple of `q` slots. -/
def partOf (q : Nat) (s : String × Array (Nat × Nat)) : Option Part :=
  match s.2.toList with
  | [(lo, hi)] => if 0 < q ∧ lo < hi ∧ (hi - lo) % q = 0 then some ⟨s.1, lo, (hi - lo) / q⟩ else none
  | _ => none

/-- **The parts**: one per stratum, none when a stratum is not such a range. -/
def parts (q : Nat) (strata : Array (String × Array (Nat × Nat))) : Option (Array Part) := strata.mapM (partOf q)

/-- One second-stage session's statement (module docstring). -/
structure Stmt where
  template : String
  header : Json
  draw : Json
  registered : Json

def STMT_KEYS : Array String := #["draw", "header", "registered", "template"]

/-- A statement from its JSON: exactly `STMT_KEYS`, a string, two objects and an object or null. -/
def Stmt.ofJson? (j : Json) : Option Stmt :=
  if keys? j != some STMT_KEYS then none else
  match getD j "template", getD j "header", getD j "draw", getD j "registered" with
  | .str t, .obj h, .obj d, .null => some ⟨t, .obj h, .obj d, .null⟩
  | .str t, .obj h, .obj d, .obj r => some ⟨t, .obj h, .obj d, .obj r⟩
  | _, _, _, _ => none

/-- A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
`value`. -/
structure Link where
  template : String
  port : String
  value : String
  deriving Repr, Inhabited, DecidableEq

def Link.toJson (l : Link) : Json :=
  Json.mkObj [("port", Json.str l.port), ("template", Json.str l.template), ("value", Json.str l.value)]

/-- Links from their JSON: a list of objects of exactly `template`, `port` and `value`, each a string. -/
def linksOf? : Json → Option (Array Link)
  | .arr ls => ls.mapM fun l =>
    if keys? l != some #["port", "template", "value"] then none else
    match getD l "template", getD l "port", getD l "value" with
    | .str t, .str p, .str v => some ⟨t, p, v⟩
    | _, _, _ => none
  | _ => none

/-- What the verifier holds of the run: the replay sample `S`, the parts (`parts`), the interior and boundary
registrations' records, the second draw's units, the design's links (`replayLinks`, none when the replay block's
Definition makes none), the `{value: root}` the verifiers of record report the sessions read
(`Record.sessionRegistered`), and the reference header of each template's statement, as the auditor's params give
it (`Run.templateCircuits`). -/
structure Own where
  sample : Array Nat
  parts : Option (Array Part)
  interior : Json
  boundary : Json
  drawn : Array Nat
  links : Option (Array Link) := some #[]
  reported : Option (Array (String × Json)) := none
  refs : Array (String × Json) := #[]

/-- The record names its ports by template, `<template>/<port>`. -/
def byTemplate (rec : Json) : Bool := (Record.byPort rec).any fun (q, _) => q.contains '/'

/-- The interior record's port for port `p` of template `t`'s circuit. -/
def portOf (named : Bool) (t p : String) : String := if named then t ++ "/" ++ p else p

/-- The interior record's ports of template `t`: those under `t/` when it names its ports by template, else all. -/
def templatePorts (named : Bool) (t : String) (rec : Json) : Array String :=
  ((Record.byPort rec).map (·.1)).filter fun q => !named || q.startsWith (t ++ "/")

/-- The output ports `o<j>` of a circuit returning `t`, one per field: a tuple's or a record's fields, else one. -/
def outputs : Flock.Program.Ty → Nat
  | .tuple fs => fs.size
  | .record fs => fs.size
  | _ => 1

/-- The fields of a circuit returning `t`, one per output port (`outputs`): a tuple's or a record's, else `t`. -/
def fields : Flock.Program.Ty → Array Flock.Program.Ty
  | .tuple fs => fs
  | .record fs => fs.map (·.2)
  | t => #[t]

/-- What part `q` of a body's reference reads: parameters (`true`) and nodes (`false`), by index. -/
def partSources : Flock.Program.Part → List (Bool × Nat)
  | .affine p i _ _ => [(p, i)]
  | .strided p i _ _ _ _ _ => [(p, i)]
  | .explicit items => items.toList.map fun r => (r.p, r.i)

/-- What reference `s` reads: its parts' (`partSources`). -/
def sources (s : Flock.Program.Seq) : List (Bool × Nat) := s.parts.toList.flatMap partSources

/-- The nodes of all of `defs`' bodies that call definition `c`. -/
def callers (defs : Array Flock.Program.Defn) (c : Nat) : Nat :=
  defs.foldl (fun n d => n + (d.body.nodes.filter (·.fn == c)).size) 0

/-- Node `x`'s template: the composite definition it calls or batches, when no other node of `defs` calls it. -/
def template? (defs : Array Flock.Program.Defn) (x : Flock.Program.Node) : Option Flock.Program.Defn := do
  let c ← defs[x.fn]?
  guard ((x.form == .call || x.form == .batch) && !c.prim && callers defs x.fn == 1)
  pure c

/-- The links node `x` of replay block `D`'s body makes: for each argument that reads one of `D`'s parameters, its
template's parameter there reads that one; none when the argument reads anything else too, or `x` is no template's
(`template?`). -/
def nodeLinks (defs : Array Flock.Program.Defn) (D : Flock.Program.Defn) (x : Flock.Program.Node) :
    Option (List Link) :=
  (List.range x.args.size).filterMapM fun a => do
    let arg ← x.args[a]?
    let src := sources arg.seq
    match src.find? (·.1) with
    | none => pure none
    | some (_, i) =>
      guard (src.all (· == (true, i)))
      let c ← template? defs x
      let (port, _) ← c.params[a]?
      let (name, _) ← D.params[i]?
      pure (some ⟨c.key, port, name⟩)

/-- The link field `j` of replay block `D`'s return, of type `f`, makes when it is part `r` of the return: when `r` is
node `k`'s whole output, `k` a template's (`template?`) of one output, that template's `o0` is `D`'s `o<j>`; with it,
`k`. None when it is anything else. -/
def fieldLink (defs : Array Flock.Program.Defn) (D : Flock.Program.Defn) (j : Nat) (f : Flock.Program.Ty) :
    Flock.Program.Part → Option (Nat × Link)
  | .affine false k 0 c => do
    let x ← D.body.nodes[k]?
    guard (c == f.leaves && c == x.out.leaves)
    let t ← template? defs x
    guard (outputs t.ret == 1)
    pure (k, ⟨t.key, "o0", "o" ++ toString j⟩)
  | _ => none

/-- The links replay block `D`'s return makes: when it returns node `x`'s output alone, `x` a template's (`template?`)
of one output and `D` of one, that template's `o0` is `D`'s; when `D` returns several fields, each part of its return
one field, each one node's whole output and no node's twice (`fieldLink`), each one's template's `o0` is its field's.
None when it returns anything else. -/
def returnLinks (defs : Array Flock.Program.Defn) (D : Flock.Program.Defn) : Option (List Link) :=
  match sources D.body.ret with
  | [] => some []
  | (p, k) :: rest =>
    if outputs D.ret == 1 then do
      guard (!p && rest.all (· == (false, k)))
      let x ← D.body.nodes[k]?
      let c ← template? defs x
      guard (outputs c.ret == 1)
      pure [⟨c.key, "o0", "o0"⟩]
    else do
      let fs := fields D.ret
      guard (D.body.ret.parts.size == fs.size)
      let ls ← (List.range fs.size).mapM fun j => do fieldLink defs D j (← fs[j]?) (← D.body.ret.parts[j]?)
      guard (ls.all fun (k, _) => (ls.map (·.1)).count k == 1)
      pure (ls.map (·.2))

/-- **The design's links**, from replay block `defs[d]`'s registered Definition, its parameters and its body: those its
nodes make (`nodeLinks`) and its return (`returnLinks`), each value its own port's name. None when its body reads a
parameter, or returns, any other way. -/
def linksOf (defs : Array Flock.Program.Defn) (d : Nat) : Option (Array Link) := do
  let D ← defs[d]?
  guard (!D.prim)
  let ins ← D.body.nodes.toList.mapM (nodeLinks defs D)
  let outs ← returnLinks defs D
  pure (ins.flatten ++ outs).toArray

/-- The links of the replay block program `P` defines under descriptor id `key` (`linksOf`), each value the boundary
registration's port for it, as record `rec` names its ports (`portOf` under `key`). -/
def replayLinks (P : Flock.Program.Prog) (key : String) (rec : Json) : Option (Array Link) := do
  let d ← P.index[key]?
  let ls ← linksOf P.defs d
  pure (ls.map fun l => { l with value := portOf (byTemplate rec) key l.value })

/-- Links `ls` are derived, each for a template one of parts `ps` is of. -/
def linksOk (ps : Array Part) : Option (Array Link) → Bool
  | some ls => ls.all fun l => ps.any (·.template == l.template)
  | none => false

/-- The link that names template `t`'s session port `port`. -/
def linkOf (own : Own) (t port : String) : Option Link :=
  (own.links.getD #[]).find? fun l => l.template == t && l.port == port

/-- **The run's fields** of a statement's header, each a path of keys: the only ones a header may differ from its
template's reference placed in the run in, which other checks hold instead: `instances` and `units.indices`
(`second-population`), `units.blocks`, the indices cut into blocks of `vus_per_block` (the session's verifier, which
refuses a header whose `units` aren't `Flock.HmRow.unitsEntry` of its indices), `frame_v3.roots` and
`frame_v3.domain_ids` (`second-roots`), and `registered`, the reads' values, positions and paths (`second-read`, which
holds each read to its root, domain and rows). A reference staged at other slots differs from the header only in the
indices and their blocks. -/
def RUN_FIELDS : List (List String) :=
  [["instances"], ["units", "indices"], ["units", "blocks"], ["frame_v3", "roots"], ["frame_v3", "domain_ids"],
    ["registered"]]

/-- Object `j` with key `k` set to `v`; anything else unchanged. -/
def setKey (j : Json) (k : String) (v : Json) : Json :=
  match j with
  | .obj _ => j.setObjVal! k v
  | _ => j

/-- `j` with the field at key path `path` set to `v` where its parent is an object, each key before the last followed
only to an object; anything else unchanged. -/
def putAt (v : Json) : Json → List String → Json
  | j, [] => j
  | j, [k] => setKey j k v
  | j, k :: ks => match field? j k with
    | some (.obj kvs) => setKey j k (putAt v (.obj kvs) ks)
    | _ => j

/-- `j` with the field at key path `path` null (`putAt`). -/
def blankAt (j : Json) (path : List String) : Json := putAt .null j path

/-- Header `h` with the run's fields (`RUN_FIELDS`) blanked, null. -/
def blankRun (h : Json) : Json := RUN_FIELDS.foldl blankAt h

/-- The reference header the auditor's params give template `t` (`Own.refs`). -/
def refOf (own : Own) (t : String) : Option Json := (own.refs.find? (·.1 == t)).map (·.2)

/-- The run the interior record names (`window.run`). -/
def runOf (rec : Json) : Option String :=
  match getD (getD rec "window") "run" with
  | .str r => some r
  | _ => none

/-- The input set of the `k`-th part's session in run `run`, as the served path names it: `served/<run>/m<k>`, the
interior's `k`-th member being the `k`-th part. -/
def setName (run : String) (k : Nat) : String := "served/" ++ run ++ "/m" ++ toString k

/-- The window-domain rule's tag (`verity.ml.commitments.window.WINDOW_DOMAIN`). -/
def WINDOW_DOMAIN : String := "verity/one-stage/served-domain/v1"

/-- The binding, in run `run`, of the interior record `rec`'s port `port` of row schema `schema`: the window-domain rule
(`verity.ml.commitments.window`), the identity digest (`tagged_sha512` of the canonical JSON) of the record's program
and partition, the port, its schema, the leaves the record gives it and the run. None when the record names no such
port, or no program or partition. -/
def portBinding (rec : Json) (port schema run : String) : Option String := do
  let .str program := getD rec "program" | none
  let .str partition := getD rec "partition" | none
  let (_, e) ← (Record.byPort rec).find? (·.1 == port)
  let leaves ← nat? (getD e "leaves")
  let m ← (Flock.canonUtf8 (Json.mkObj [("program", .str program), ("partition", .str partition), ("port", .str port),
    ("schema", .str schema), ("leaves", Lean.toJson leaves), ("run", .str run)])).toOption
  pure (Flock.hexOf (Flock.FV3.Sha512.taggedSha512 WINDOW_DOMAIN m.toUTF8))

/-- Template `t`'s reference header `r` placed in the run, for the session of the `k`-th part: its `set`
the run's input set (`setName`) and its `frame_v3.bindings` the binding of each port of its `frame_v3.schemas`
(`portBinding`, the interior record's port `portOf named t p`), which the verifier computes, so the prover chooses
neither. None when the interior record names no run, or a port the verifier can't bind. -/
def atRun (own : Own) (named : Bool) (k : Nat) (t : String) (r : Json) : Option Json := do
  let run ← runOf own.interior
  let bs ← (entries (getD (getD r "frame_v3") "schemas")).mapM fun (p, s) => do
    let .str s := s | none
    pure (p, Json.str (← portBinding own.interior (portOf named t p) s run))
  pure (putAt (Json.mkObj bs.toList) (putAt (.str (setName run k)) r ["set"]) ["frame_v3", "bindings"])

/-- Header `h` of a statement for the `k`-th part is template `t`'s reference header placed in the run (`atRun`) but for
the run's fields: the two the same with those blanked (`blankRun`). False for a template the params give no reference,
or one the verifier can't place in the run. -/
def headerOk (own : Own) (named : Bool) (k : Nat) (t : String) (h : Json) : Bool :=
  match refOf own t with
  | some r => match atRun own named k t r with
    | some r' => jeq (blankRun h) (blankRun r')
    | none => false
  | none => false

/-- The header's `units.indices` are part `p`'s slots in order, and its `instances` their count. -/
def populationOk (q : Nat) (p : Part) (h : Json) : Bool :=
  match getD (getD h "units") "indices", nat? (getD h "instances") with
  | .arr is, some m => m == q * p.n && is.toList.map nat? == (List.range (q * p.n)).map fun i => some (p.lo + i)
  | _, _ => false

/-- The header's scheme is the interior record's, its `frame_v3` roots outside the ports `reads` reads are the record's
for template `t`'s ports, each at the record's root, its `domain_ids` entry the record's domain, and each of those ports
is one it roots or reads. -/
def rootsOk (named : Bool) (rec : Json) (t : String) (h : Json) (reads : Array String) : Bool :=
  let fv := getD h "frame_v3"
  let roots := (entries (getD fv "roots")).filter fun (p, _) => !reads.contains p
  let reg := Record.byPort rec
  jeq (getD h "scheme") (getD rec "scheme") &&
    roots.all (fun (p, r) => match reg.find? (·.1 == portOf named t p) with
      | some (_, e) => jeq r (getD e "root") && jeq (getD (getD fv "domain_ids") p) (getD e "domain")
      | none => false) &&
    (templatePorts named t rec).all fun x =>
      roots.any (fun (p, _) => portOf named t p == x) || reads.any fun p => portOf named t p == x

/-- **The verifier's own sample** for part `p`'s session: the second draw's units among its slots, counted from its
first, as `unit_draw` writes a subset of its population. -/
def ownDraw (q : Nat) (p : Part) (drawn : Array Nat) : Json :=
  let units := (drawn.filter (p.holds q)).map (· - p.lo)
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson (q * p.n)), ("k", Lean.toJson units.size),
    ("units", natsJson units)]

/-- Value `v`'s entry among registration `rec`'s registered values. -/
def registeredEntry (rec : Json) (v : String) : Option Json :=
  (match getD rec "registered" with | .arr es => es | _ => #[]).find? fun e => jsonStrIs (getD e "value") v

/-- Entries `e` and `x` name one root, domain and leaf count. -/
def sameAt (e x : Json) : Bool :=
  jeq (getD e "root") (getD x "root") && jeq (getD e "domain") (getD x "domain") && jeq (getD e "leaves") (getD x "leaves")

/-- Every read of registered-reads entry `e` (slot `reads_from + k` reads row `reads[k]`, null for none) is by one of
part `p`'s slots `u`, of a row of the unit `unit u` names, in a tree of `units` units' rows, `leaves / units` each. -/
def rowsOk (q : Nat) (p : Part) (e : Json) (units : Nat) (unit : Nat → Option Nat) : Bool :=
  let first : Option Nat := match getD e "reads_from" with | .null => some 0 | x => nat? x
  match nat? (getD e "leaves"), getD e "reads", first with
  | some leaves, .arr rs, some f =>
    decide (0 < units) && leaves % units == 0 && (List.range rs.size).all fun k =>
      match rs.getD k .null with
      | .null => true
      | x => match nat? x, unit (f + k) with
        | some pos, some b =>
          p.holds q (f + k) && decide (b * (leaves / units) ≤ pos) && decide (pos < (b + 1) * (leaves / units))
        | _, _ => false
  | _, _, _ => false

/-- Template `p.template`'s session port `port` may read registered-reads entry `e`: a port a link names, only the
boundary port it names, each slot only its cell's replay block's rows; any other, its own port of the interior record,
each slot only its own rows, or a value either registration registers, never a boundary port. Each at its root, domain
and leaves. -/
def readOk (own : Own) (named : Bool) (p : Part) (port : String) (e : Json) : Bool :=
  let q := own.sample.size
  match getD e "value", linkOf own p.template port with
  | .str v, some l =>
    v == l.value && match (Record.byPort own.boundary).find? (·.1 == v), nat? (getD own.boundary "population") with
      | some (_, b), some m => sameAt e b && rowsOk q p e m fun u => own.sample[p.span u]?
      | _, _ => false
  | .str v, none =>
    match (Record.byPort own.interior).find? (·.1 == v), (Record.byPort own.boundary).find? (·.1 == v) with
    | some (_, x), _ => v == portOf named p.template port && sameAt e x && rowsOk q p e (q * p.n) fun u => some (u - p.lo)
    | none, some _ => false
    | none, none => match registeredEntry own.boundary v <|> registeredEntry own.interior v with
      | some x => sameAt e x
      | none => false
  | _, _ => false

/-- Every port a link names for template `t` is one of `reads`. -/
def linkedRead (own : Own) (t : String) (reads : Array (String × Json)) : Bool :=
  (own.links.getD #[]).all fun l => l.template != t || reads.any (·.1 == l.port)

/-- Statement `s` reads registered every port its header shares rows at (`shared_rows`: the tables whose rows its
instances read by the refs in its public file's body). At such a port the session's verifier holds each instance's ref
to the row at the position the read gives its unit (`Flock.Registered.check`); at any other it only bounds a ref by its
table. A header that shares none has no refs. -/
def refsOk (s : Stmt) : Bool :=
  let reads := entries (getD s.registered "ports")
  (entries (getD s.header "shared_rows")).all fun (port, _) => reads.any (·.1 == port)

/-- Statement `s`'s failing checks against its part `p` (module docstring), in order, `k` the part's position. -/
def stmtCodes (own : Own) (named : Bool) (k : Nat) (p : Part) (s : Stmt) : List String :=
  let q := own.sample.size
  let reads := entries (getD s.registered "ports")
  [("second-header", headerOk own named k s.template s.header),
   ("second-population", populationOk q p s.header),
   ("second-roots", rootsOk named own.interior s.template s.header (reads.map (·.1))),
   ("second-sample", jeq s.draw (ownDraw q p own.drawn)),
   ("second-read", reads.all (fun (port, e) => readOk own named p port e) && linkedRead own p.template reads),
   ("second-refs", refsOk s)].filterMap
    fun (code, ok) => if ok then none else some code

def detailOf (i : Nat) (t : Json) : Json := Json.mkObj [("statement", Lean.toJson i), ("template", t)]

/-- The templates statements `ss` are for, those that are statements. -/
def templates (ss : List Json) : List String := ss.filterMap fun j => (Stmt.ofJson? j).map (·.template)

/-- Template `t`'s position among the parts `ps`, its first part's: the index of its member in the interior window,
whose members are the parts in order (the interior program's strata's); `ps.size`, no member's, for none. -/
def posOf (ps : Array Part) (t : String) : Nat := (ps.findIdx? (·.template == t)).getD ps.size

/-- Statement `i` of `ss`'s failures, `(code, detail)` each: `second-statement` for one that is not a statement,
`second-template` for one for no part's template or an earlier statement's, else its failing checks, its header placed
at its template's position (`posOf`). -/
def stmtFailures (own : Own) (ps : Array Part) (ss : List Json) (i : Nat) : List (String × Json) :=
  match Stmt.ofJson? (ss.getD i .null) with
  | none => [("second-statement", detailOf i .null)]
  | some s =>
    match ps.find? (·.template == s.template) with
    | some p =>
      if (templates (ss.take i)).contains s.template then [("second-template", detailOf i (.str s.template))]
      else (stmtCodes own (byTemplate own.interior) (posOf ps s.template) p s).map
        fun c => (c, detailOf i (.str s.template))
    | none => [("second-template", detailOf i (.str s.template))]

/-- `second-order` for each statement of `ss` for a part, no earlier statement's, other than the part at its position:
round 2 reads the statements by position, the `i`-th the `i`-th part's (the parts' order, the interior program's
strata's). -/
def orderFailures (ps : Array Part) (ss : List Json) : List (String × Json) :=
  (List.range ss.length).filterMap fun i =>
    match Stmt.ofJson? (ss.getD i .null) with
    | some s =>
      if (ps.find? (·.template == s.template)).isSome && !(templates (ss.take i)).contains s.template &&
          ps[i]?.map (·.template) != some s.template then
        some ("second-order", detailOf i (.str s.template))
      else none
    | none => none

/-- `second-cover` for each part no statement is for: every part has one, drawn by the second draw or not. -/
def coverFailures (ps : Array Part) (ss : List Json) : List (String × Json) :=
  ps.toList.filterMap fun p =>
    if !(templates ss).contains p.template then some ("second-cover", Json.mkObj [("template", Json.str p.template)])
    else none

/-- The `{value: root}` statements `ss` read, merged as `Record.sessionRegistered` merges the verdicts': a value read
under two roots maps to null. -/
def statementReads (ss : List Json) : Array (String × Json) :=
  ss.foldl (fun acc j => match Stmt.ofJson? j with
    | some s => (entries (getD s.registered "ports")).foldl (fun acc (_, e) => match getD e "value" with
      | .str v =>
        let root := getD e "root"
        match acc.find? (·.1 == v) with
        | some (_, r) => acc.map fun (k, x) => if k == v then (k, if jeq r root then r else .null) else (k, x)
        | none => acc.push (v, root)
      | _ => acc) acc
    | none => acc) #[]

/-- `second-reported` when the reads the verifiers of record report are not those statements `ss` state. -/
def reportedFailures (own : Own) (ss : List Json) : List (String × Json) :=
  if Record.sameDict (own.reported.getD #[]) (statementReads ss) then [] else [("second-reported", Json.str REPORTED_WHY)]

/-- **The interior record's order**: the replay sample the interior record names (`window.replay_sample`, which the
proof server writes) is the verifier's own, `own.sample`, in its order, each entry a natural number (`nat?`, never a
boolean). -/
def interiorOrderOk (own : Own) : Bool :=
  match getD (getD own.interior "window") "replay_sample" with
  | .arr bs => bs.toList.map nat? == own.sample.toList.map some
  | _ => false

/-- `second-interior-order` unless the interior record names the replay sample in its order (`interiorOrderOk`). -/
def interiorFailures (own : Own) : List (String × Json) :=
  if interiorOrderOk own then [] else [("second-interior-order", Json.str INTERIOR_ORDER_WHY)]

/-- Every failure of statements `ss` over parts `ps`, the interior record's order's first, then, when there is none
and the form isn't allowed, its refusal. -/
def failures (allow : Bool) (own : Own) (ps : Array Part) (ss : List Json) : List (String × Json) :=
  let found := interiorFailures own ++ (List.range ss.length).flatMap (stmtFailures own ps ss) ++
    orderFailures ps ss ++ coverFailures ps ss ++ reportedFailures own ss
  if found.isEmpty && !allow then [("two-stage-form", Json.str FORM_WHY)] else found

/-- **Round 2's statement check** (module docstring) of statements `stmts`, the two-stage form allowed when `allow`
(`FORM` in the run): each code once, with the detail it first failed with. The interior record's order is checked
before any statement is read (`interiorFailures`). -/
def check (allow : Bool) (own : Own) (stmts : Option (Array Json)) : Check :=
  let found := match own.parts, stmts with
    | none, _ => interiorFailures own ++ [("second-cells", Json.str CELLS_WHY)]
    | some ps, none =>
      interiorFailures own ++ if linksOk ps own.links then [("second-statements", Json.str STATEMENTS_WHY)]
        else [("second-links", Json.str LINKS_WHY)]
    | some ps, some ss =>
      if linksOk ps own.links then failures allow own ps ss.toList
      else interiorFailures own ++ [("second-links", Json.str LINKS_WHY)]
  found.foldl (fun c x => c.fail x.1 x.2) {}

/-- The interior registration's audit record `rec` with round 2's check `c` as `checks.statements`, failing where `c`
does as a failed check of its own fails it (`Record.record`): not accepted, and no profile, integrity profile or wrong
units. -/
def attach (rec : Json) (c : Check) : Json :=
  let r := rec.setObjVal! "checks" ((getD rec "checks").setObjVal! "statements" c.toJson)
  if c.codes.isEmpty then r
  else (((r.setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal! "integrity_profile" .null).setObjVal!
    "wrong_units" .null

theorem mem_fail (c : Check) (code : String) (d : Json) (x : String) :
    x ∈ (c.fail code d).codes ↔ x ∈ c.codes ∨ x = code := by
  unfold Check.fail
  split
  · rename_i h
    constructor
    · exact Or.inl
    · rintro (hx | rfl)
      · exact hx
      · simpa using h
  · simp

theorem mem_fold (l : List (String × Json)) (c : Check) (x : String) :
    x ∈ (l.foldl (fun c x => c.fail x.1 x.2) c).codes ↔ x ∈ c.codes ∨ ∃ d, (x, d) ∈ l := by
  induction l generalizing c with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons, ih, mem_fail]
    constructor
    · rintro ((h | rfl) | ⟨d, hd⟩)
      · exact Or.inl h
      · exact Or.inr ⟨a.2, by simp⟩
      · exact Or.inr ⟨d, by simp [hd]⟩
    · rintro (h | ⟨d, hd⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hd with rfl | hd
        · exact Or.inl (Or.inr rfl)
        · exact Or.inr ⟨d, hd⟩

/-- A fold of `Check.fail` has no code only over no failure. -/
theorem fold_codes_empty (l : List (String × Json)) :
    (l.foldl (fun c x => c.fail x.1 x.2) ({} : Check)).codes = #[] ↔ l = [] := by
  constructor
  · intro h
    cases l with
    | nil => rfl
    | cons a l =>
      have hm := (mem_fold (a :: l) {} a.1).2 (Or.inr ⟨a.2, by simp⟩)
      rw [h] at hm
      simp at hm
  · rintro rfl
    rfl

theorem failures_ne_nil (own : Own) (ps : Array Part) (ss : List Json) : failures false own ps ss ≠ [] := by
  simp only [failures]
  split
  · simp
  · rename_i h
    intro he
    simp_all

/-- **Off, round 2 never passes**: with the two-stage form refused, every run's check fails. -/
theorem check_refused (own : Own) (stmts : Option (Array Json)) : (check false own stmts).codes ≠ #[] := by
  simp only [check, Ne, fold_codes_empty]
  split
  · simp
  · split <;> simp
  · split <;> simp [failures_ne_nil]

/-- **By position**: with no `second-order`, statement `i`, one for part `p` and no earlier statement's, is for the
part at its position. -/
theorem orderFailures_nil {ps : Array Part} {ss : List Json} (h : orderFailures ps ss = []) {i : Nat}
    (hi : i < ss.length) {s : Stmt} {p : Part} (hs : Stmt.ofJson? (ss.getD i .null) = some s)
    (hp : ps.find? (·.template == s.template) = some p) (hd : (templates (ss.take i)).contains s.template = false) :
    ps[i]?.map (·.template) = some s.template := by
  have h' := List.filterMap_eq_nil_iff.1 h i (List.mem_range.2 hi)
  simp only [hs] at h'
  by_cases he : ps[i]?.map (·.template) = some s.template
  · exact he
  · have hb : (ps[i]?.map (·.template) != some s.template) = true := bne_iff_ne.2 he
    simp only [hp, hd, hb, Option.isSome_some] at h'
    simp at h'

/-- With no `second-interior-order`, the interior record names the replay sample in its order. -/
theorem interiorFailures_nil {own : Own} (h : interiorFailures own = []) : interiorOrderOk own = true := by
  unfold interiorFailures at h
  split at h
  · assumption
  · simp at h

/-- **A passing `second-interior-order`**: the interior record's `window.replay_sample` is a list whose entries are the
verifier's own replay sample's blocks, in its order. -/
theorem interiorOrderOk_sample {own : Own} (h : interiorOrderOk own = true) :
    ∃ bs, getD (getD own.interior "window") "replay_sample" = .arr bs ∧
      bs.toList.map nat? = own.sample.toList.map some := by
  unfold interiorOrderOk at h
  split at h
  · rename_i bs hb
    exact ⟨bs, hb, by simpa using h⟩
  · simp at h

/-- **What a passing check says**: the form is allowed, the interior record names the replay sample in its order
(`interiorOrderOk_sample`), the interior program's strata make parts, the replay block's Definition makes links, each
for a part's template, each statement is one for a part's template, no earlier statement's, the part at its position
(`orderFailures_nil`), and passes every check at its position (`stmtCodes_nil`), every part has a statement, and the
reads the verifiers of record report are the statements'. -/
theorem check_ok {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    allow = true ∧ interiorOrderOk own = true ∧ ∃ ps, own.parts = some ps ∧ linksOk ps own.links = true ∧
      (∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧ (templates (ss.toList.take i)).contains s.template = false ∧
        ps[i]?.map (·.template) = some s.template ∧
        stmtCodes own (byTemplate own.interior) (posOf ps s.template) p s = []) ∧
      (∀ p ∈ ps.toList, (templates ss.toList).contains p.template = true) ∧
      Record.sameDict (own.reported.getD #[]) (statementReads ss.toList) = true := by
  simp only [check, fold_codes_empty] at h
  split at h
  · simp at h
  · split at h <;> simp at h
  · rename_i ps _ hp hs
    cases hs
    by_cases hl : linksOk ps own.links = true
    case neg => rw [if_neg hl] at h; simp at h
    rw [if_pos hl] at h
    simp only [failures] at h
    split at h
    · simp at h
    · rename_i hne
      simp only [h, List.isEmpty_nil, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hne
      simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
      obtain ⟨hint, hst, hord, hcov, hrep⟩ := h
      refine ⟨by simpa using hne, interiorFailures_nil hint, ps, hp, hl, ?_, ?_, ?_⟩
      · intro i hi
        have hi' : i < ss.toList.length := by simpa using hi
        have := hst i hi'
        simp only [stmtFailures] at this
        split at this
        · simp at this
        · rename_i s hs
          split at this
          · rename_i p hp'
            split at this
            · simp at this
            · rename_i hdup
              have hd : (templates (ss.toList.take i)).contains s.template = false := by simpa using hdup
              exact ⟨s, p, hs, hp', hd, orderFailures_nil hord hi' hs hp' hd, by simpa using this⟩
          · simp at this
      · intro p hp'
        have := List.filterMap_eq_nil_iff.1 hcov p hp'
        split at this
        · simp at this
        · rename_i hc
          simpa using hc
      · unfold reportedFailures at hrep
        split at hrep
        · assumption
        · simp at hrep

/-- A passing statement's checks: its header at its position, population, roots, sample and registered reads, every
port a link names among them, and every port its header shares rows at read registered. -/
theorem stmtCodes_nil {own : Own} {named : Bool} {k : Nat} {p : Part} {s : Stmt} (h : stmtCodes own named k p s = []) :
    headerOk own named k s.template s.header = true ∧ populationOk own.sample.size p s.header = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      jeq s.draw (ownDraw own.sample.size p own.drawn) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true ∧
      linkedRead own p.template (entries (getD s.registered "ports")) = true ∧ refsOk s = true := by
  simpa [stmtCodes, List.filterMap_eq_nil_iff, and_assoc] using h

/-- The reference header `refOf` gives template `t` is one the auditor's params give it. -/
theorem refOf_mem {own : Own} {t : String} {r : Json} (h : refOf own t = some r) : (t, r) ∈ own.refs := by
  unfold refOf at h
  cases hf : own.refs.find? (·.1 == t) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    obtain ⟨t', r'⟩ := e
    have ht : t' = t := by simpa using Array.find?_some hf
    have hr : r' = r := by simpa using h
    subst ht hr
    exact Array.mem_of_find?_eq_some hf

/-- **A passing `second-header`** names the reference header the auditor's params give template `t`, which header `h`
of a statement for the `k`-th part is, placed in the run (`atRun`), but for the run's fields. -/
theorem headerOk_ref {own : Own} {named : Bool} {k : Nat} {t : String} {h : Json}
    (hh : headerOk own named k t h = true) :
    ∃ r r', (t, r) ∈ own.refs ∧ atRun own named k t r = some r' ∧ jeq (blankRun h) (blankRun r') = true := by
  unfold headerOk at hh
  cases hf : refOf own t with
  | none => simp [hf] at hh
  | some r =>
    cases ha : atRun own named k t r with
    | none => simp [hf, ha] at hh
    | some r' =>
      simp only [hf, ha] at hh
      exact ⟨r, r', refOf_mem hf, ha, hh⟩

/-- **Every header is its template's reference placed in the run** but for the run's fields (`RUN_FIELDS`): in a
passing check, the `i`-th statement is for the template of the part at its position, and its header with those fields
blanked is the reference header the auditor's params give that template placed in the run at the template's position
(`atRun` at `posOf`: its `set` the run's `served/<run>/m<k>`, its `frame_v3.bindings` the verifier's), blanked the same
(`second-header`). -/
theorem check_ok_header {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s r r', Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps[i]?.map (·.template) = some s.template ∧ (s.template, r) ∈ own.refs ∧
      atRun own (byTemplate own.interior) (posOf ps s.template) s.template r = some r' ∧
      jeq (blankRun s.header) (blankRun r') = true := by
  obtain ⟨-, -, ps, hp, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, -, -, hi', hc⟩ := hst i hi
  obtain ⟨r, r', hr, ha, hj⟩ := headerOk_ref (stmtCodes_nil hc).1
  exact ⟨s, r, r', hs, hi', hr, ha, hj⟩

/-- **A passing `second-refs`**: every port statement `s`'s header shares rows at is one of its registered reads. -/
theorem refsOk_read {s : Stmt} (h : refsOk s = true) {port : String} {c : Json}
    (hp : (port, c) ∈ entries (getD s.header "shared_rows")) :
    ∃ e, (port, e) ∈ entries (getD s.registered "ports") := by
  dsimp only [refsOk] at h
  have h' : (entries (getD s.registered "ports")).any (·.1 == port) = true :=
    Array.all_eq_true_iff_forall_mem.mp h _ hp
  obtain ⟨⟨q, e⟩, hm, hq⟩ := Array.any_eq_true'.mp h'
  obtain rfl : q = port := by simpa using hq
  exact ⟨e, hm⟩

/-- **Every port a header shares rows at is a read round 2 holds**: in a passing check, every port the `i`-th
statement's header shares rows at (`shared_rows`) is one of its registered reads, each `readOk` at its part: for a port
a link names, the boundary port it names, each slot reading only its cell's replay block's rows; else its own port of
the interior record, each slot reading only its own rows, or a value either registration registers. The session's
verifier holds each instance's ref at such a port to the row at the position the read gives its unit
(`Flock.Registered.check`), so no ref there points a drawn unit at another unit's rows. -/
theorem check_ok_refs {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    ∃ ps, own.parts = some ps ∧ ∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
      ps.find? (·.template == s.template) = some p ∧ ∀ port c, (port, c) ∈ entries (getD s.header "shared_rows") →
        ∃ e, (port, e) ∈ entries (getD s.registered "ports") ∧ readOk own (byTemplate own.interior) p port e = true := by
  obtain ⟨-, -, ps, hp, -, hst, -, -⟩ := check_ok h
  refine ⟨ps, hp, fun i hi => ?_⟩
  obtain ⟨s, p, hs, hf, -, -, hc⟩ := hst i hi
  obtain ⟨-, -, -, -, hr, -, hrefs⟩ := stmtCodes_nil hc
  refine ⟨s, p, hs, hf, fun port c hm => ?_⟩
  obtain ⟨e, he⟩ := refsOk_read hrefs hm
  exact ⟨e, he, Array.all_eq_true_iff_forall_mem.mp hr _ he⟩

private def exRoot (i : Nat) : String := String.ofList (List.replicate 128 (Char.ofNat (48 + i)))

private def exRec (population : Nat) (roots : List (String × Nat × Nat)) (window : Json := Json.mkObj []) : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("population", Lean.toJson population),
    ("program", "program-0"), ("partition", "partition-0"), ("window", window),
    ("roots", Json.arr (roots.toArray.map fun (p, r, d) =>
      Json.mkObj [("port", p), ("root", exRoot r), ("domain", exRoot d), ("leaves", Lean.toJson population)]))]

private def exRun : String := "run-0"

/-- The interior registration's window: the run, and the replay sample `sample` it lays out. -/
private def exWindow (sample : Json) (run : Json := .str exRun) : Json :=
  Json.mkObj [("kind", "served"), ("run", run), ("interior", Json.bool true), ("replay_sample", sample)]

private def exInterior (window : Json := exWindow (natsJson #[1, 3])) : Json :=
  exRec 6 [("x", 1, 2), ("out", 3, 4)] window

private def exSchemas : Json := Json.mkObj [("x", "hm96-sha512/row/v2"), ("out", "hm96-sha512/row/v2")]

/-- Port `p`'s binding in run `exRun`, which the served header names. -/
private def exBinding (p : String) : Json := .str ((portBinding exInterior p "hm96-sha512/row/v2" exRun).getD "")

/-- `exStmt`'s reference header, fixed before the run: its scheme, the set it was staged under, its ports' schemas and
their bindings there, and none of the run's fields. -/
private def exRef : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", "unit-Pair_v0"), ("units", Json.mkObj []),
    ("frame_v3", Json.mkObj [("schemas", exSchemas), ("bindings", Json.mkObj [("x", "00"), ("out", "00")])])]

/-- A replay sample of blocks 1 and 3 of 4, three proof blocks of one template in each, units 2 and 4 drawn. -/
private def exOwn : Own :=
  { sample := #[1, 3], parts := parts 2 #[("Pair_v0", #[(0, 6)])], interior := exInterior,
    boundary := exRec 4 [("bx", 5, 6)], drawn := #[2, 4], refs := #[("Pair_v0", exRef)] }

private def exStmt (indices units : List Nat) : Json :=
  Json.mkObj [("template", "Pair_v0"), ("registered", .null),
    ("header", Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("set", .str (setName exRun 0)),
      ("instances", Lean.toJson indices.length), ("units", Json.mkObj [("indices", natsJson indices.toArray)]),
      ("frame_v3", Json.mkObj [("roots", Json.mkObj [("x", exRoot 1), ("out", exRoot 3)]),
        ("domain_ids", Json.mkObj [("x", exRoot 2), ("out", exRoot 4)]), ("schemas", exSchemas),
        ("bindings", Json.mkObj [("x", exBinding "x"), ("out", exBinding "out")])])]),
    ("draw", Json.mkObj [("law", "subset"), ("population", (6 : Nat)), ("k", Lean.toJson units.length),
      ("units", natsJson units.toArray)])]

-- Python's `window_binding("program-0", "partition-0", "x", "hm96-sha512/row/v2", 6, "run-0")`
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun == some
  "009d0ab56f7953138bd8853c525c88a1ede372e270f1243e8c7b1980de7390130205daa6b324fe330e87a215a663b19a011fd6dbe3391eb47c5e7116ca32192f"
#guard portBinding exInterior "x" "hm96-sha512/row/v2" exRun != portBinding exInterior "out" "hm96-sha512/row/v2" exRun
#guard portBinding exInterior "y" "hm96-sha512/row/v2" exRun == none
#guard setName exRun 2 == "served/run-0/m2"

#guard exOwn.parts == some #[{ template := "Pair_v0", lo := 0, n := 3 }]
#guard (check false exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes == #["two-stage-form"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes.isEmpty
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 6] [2, 4]])).codes == #["second-population"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]])).codes == #["second-sample"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4], exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes ==
  #["second-template"]
#guard (check true exOwn (some #[])).codes == #["second-cover"]
#guard (check true exOwn none).codes == #["second-statements"]
#guard (check true { exOwn with parts := parts 4 #[("Pair_v0", #[(0, 6)])] } none).codes == #["second-cells"]

/-- `exStmt`'s honest statement, its header `f` of the honest one. -/
private def exHeader (f : Json → Json) : Json :=
  let s := exStmt [0, 1, 2, 3, 4, 5] [2, 4]
  s.setObjVal! "header" (f (getD s "header"))

/-- Header `h` with its object `k`'s field `f` set to `v`. -/
private def exIn (k f : String) (v : Json) (h : Json) : Json := h.setObjVal! k ((getD h k).setObjVal! f v)

/-- Round 2's codes, the form allowed, for `exHeader f` alone. -/
private def exCodes (own : Own) (f : Json → Json) : Array String := (check true own (some #[exHeader f])).codes

#guard (exCodes exOwn id).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", getD (exHeader id) "header")] } id).isEmpty
#guard (exCodes exOwn (·.setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [])]))).isEmpty
#guard exCodes exOwn (·.setObjVal! "set" (.str "served/other/m0")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "set" (.str (setName exRun 1))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x"), ("out", exBinding "x")])) ==
  #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "bindings" (Json.mkObj [("x", exBinding "x")])) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "schemas" (Json.mkObj [("x", "hm96-sha512/row/v2")])) == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) (.str "run-1")) } id ==
  #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1, 3]) .null) } id == #["second-header"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (natsJson #[1])) } id == #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow (.arr #[.bool true, Lean.toJson (3 : Nat)])) } id ==
  #["second-interior-order"]
#guard exCodes { exOwn with interior := exInterior (exWindow .null) } id == #["second-interior-order"]
#guard (check false { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } (some #[exHeader id])).codes ==
  #["second-interior-order"]
#guard (check true { exOwn with interior := exInterior (exWindow (natsJson #[3, 1])) } none).codes ==
  #["second-interior-order", "second-statements"]
#guard (check true { exOwn with interior := exInterior (exWindow .null), links := none } none).codes ==
  #["second-interior-order", "second-links"]
#guard (exCodes exOwn (exIn "units" "blocks" (natsJson #[0]))).isEmpty
#guard (exCodes { exOwn with refs := #[("Pair_v0", exIn "units" "blocks" (natsJson #[9]) exRef)] } id).isEmpty
#guard exCodes exOwn (exIn "units" "units_per_vu" (Lean.toJson (2 : Nat))) == #["second-header"]
#guard exCodes exOwn (exIn "frame_v3" "owner" (.str "other")) == #["second-header"]
#guard exCodes exOwn (·.setObjVal! "scheme" (.str "frame-v3-sha512/other")) == #["second-header", "second-roots"]
#guard exCodes exOwn (·.setObjVal! "instances" (Lean.toJson (7 : Nat))) == #["second-population"]
#guard exCodes exOwn (exIn "units" "indices" (natsJson #[1, 2, 3, 4, 5, 6])) == #["second-population"]
#guard exCodes exOwn (exIn "frame_v3" "roots" (Json.mkObj [("x", exRoot 1), ("out", exRoot 5)])) == #["second-roots"]
#guard exCodes exOwn (exIn "frame_v3" "domain_ids" (Json.mkObj [("x", exRoot 2)])) == #["second-roots"]
#guard exCodes { exOwn with refs := #[] } id == #["second-header"]
#guard exCodes { exOwn with refs := #[("Other_v0", exRef)] } id == #["second-header"]
#guard (check false { exOwn with refs := #[] } (some #[exHeader id])).codes == #["second-header"]

/-- Port `x`'s statement reading `value`'s tree of `leaves` leaves at root `root`, domain `dom`, slot `k` its `reads[k]`. -/
private def exReading (value : String) (root dom leaves : Nat) (reads : List Nat) : Json :=
  (exStmt [0, 1, 2, 3, 4, 5] [2, 4]).setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [
    ("value", value), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson leaves),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])])])

private def exReported (value : String) (root : Nat) (own : Own := exOwn) : Own :=
  { own with reported := some #[(value, .str (exRoot root))] }

private def exLinked : Own := { exOwn with links := some #[⟨"Pair_v0", "x", "bx"⟩] }

#guard (check true (exReported "x" 1) (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes.isEmpty
#guard (check true exOwn (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-reported"]
#guard (check true (exReported "x" 1) (some #[exReading "x" 1 2 6 [1, 1, 2, 3, 4, 5]])).codes == #["second-read"]
#guard (check true (exReported "out" 3) (some #[exReading "out" 3 4 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-read"]
#guard (check true (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes.isEmpty
#guard (check true (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 2]])).codes == #["second-read"]
#guard (check true (exReported "bx" 5) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes == #["second-read"]
#guard (check false (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes ==
  #["two-stage-form"]
#guard (check true (exReported "x" 1 exLinked) (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes ==
  #["second-read"]
#guard (check true exLinked (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes == #["second-read"]
#guard (check true { exOwn with links := none } (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes == #["second-links"]
#guard (check true { exOwn with links := none } none).codes == #["second-links"]
#guard (check true { exOwn with links := some #[⟨"Other_v0", "x", "bx"⟩] } none).codes == #["second-links"]

/-- Port `p`'s read of the interior record's own port `p` at root `root`, domain `dom`, slot `k` reading row `reads[k]`. -/
private def exOwnRead (p : String) (root dom : Nat) (reads : List Nat := [0, 1, 2, 3, 4, 5]) : String × Json :=
  (p, Json.mkObj [("value", .str p), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson (6 : Nat)),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])

/-- Tables of six rows at `ports`, as a header shares them (`shared_rows`). -/
private def exTables (ports : List String) : Json := Json.mkObj (ports.map fun p => (p, Lean.toJson (6 : Nat)))

/-- The honest statement, its header sharing rows at `shared`, reading `reads` registered (`none`: no registered
reads). -/
private def exSharing (shared : List String) (reads : Option (List (String × Json))) : Json :=
  (exHeader (·.setObjVal! "shared_rows" (exTables shared))).setObjVal! "registered"
    ((reads.map fun rs => Json.mkObj [("ports", Json.mkObj rs)]).getD .null)

/-- `exOwn` with its template's reference sharing rows at `shared`, and the verifiers of record reporting reads of
`read`, each value at its root. -/
private def exShares (shared : List String) (read : List (String × Nat)) : Own :=
  { exOwn with
    refs := #[("Pair_v0", exRef.setObjVal! "shared_rows" (exTables shared))],
    reported := some (read.toArray.map fun (v, r) => (v, .str (exRoot r))) }

-- a header sharing rows at `out` and `x` reads both registered, each slot its own row
#guard (check true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  (some #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2])])).codes.isEmpty
#guard (check true (exShares ["x"] [("x", 1)]) (some #[exSharing ["x"] (some [exOwnRead "x" 1 2])])).codes.isEmpty
-- `out` shared and not read registered: its refs are bounded only, so one could name another unit's row
#guard (check true (exShares ["out", "x"] [("x", 1)]) (some #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])])).codes ==
  #["second-refs"]
#guard (check true (exShares ["out", "x"] []) (some #[exSharing ["out", "x"] none])).codes == #["second-refs"]
-- a shared port read at another slot's row, which the session's verifier would hold slot 0's ref to
#guard (check true (exShares ["out", "x"] [("out", 3), ("x", 1)])
  (some #[exSharing ["out", "x"] (some [exOwnRead "out" 3 4, exOwnRead "x" 1 2 [1, 0, 2, 3, 4, 5]])])).codes ==
  #["second-read"]
-- rows shared where the template's reference shares none
#guard (check true (exReported "x" 1) (some #[exSharing ["out", "x"] (some [exOwnRead "x" 1 2])])).codes ==
  #["second-header", "second-refs"]

/-- A statement for template `t`'s part. -/
private def exFor (t : String) : Json := (exStmt [] []).setObjVal! "template" (.str t)

private def exTwoParts : Array Part := #[⟨"Pair_v0", 0, 3⟩, ⟨"Triple_v0", 6, 1⟩]

#guard [posOf exTwoParts "Pair_v0", posOf exTwoParts "Triple_v0", posOf exTwoParts "Other_v0"] == [0, 1, 2]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Triple_v0"]).isEmpty
#guard (orderFailures exTwoParts [exFor "Triple_v0", exFor "Pair_v0"]).map (·.1) == ["second-order", "second-order"]
#guard (orderFailures exTwoParts [exFor "Triple_v0"]).map (·.1) == ["second-order"]
#guard (orderFailures exTwoParts [exFor "Pair_v0", exFor "Pair_v0", exFor "Other_v0", .null]).isEmpty
#guard (check true { exOwn with parts := some exTwoParts } (some #[exFor "Triple_v0", exFor "Pair_v0"])).codes.contains
  "second-order"
#guard (check true { exOwn with parts := some exTwoParts, drawn := #[2] } (some #[exStmt [0, 1, 2, 3, 4, 5] [2]])).codes ==
  #["second-cover"]

private def exArg (p : Bool) (i : Nat) : Flock.Program.Arg :=
  { ty := .value 8, seq := Flock.Program.Seq.ofParts #[.affine p i 0 1] }

private def exNode (form : Flock.Program.Form) (fn : Nat) (args : Array Flock.Program.Arg) : Flock.Program.Node :=
  { form, fn, args, out := .value 8 }

private def exDefn (key : String) (prim : Bool) (params : Array String) (nodes : Array Flock.Program.Node)
    (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty := .value 8) : Flock.Program.Defn :=
  { key, id := key, prim, params := params.map (·, .value 8), ret := rty, gates := 0, paramLeaves := 0, outLeaves := 0,
    body := { params := params.map (·, .value 8), retType := rty, nodes, ret := .ofParts ret, offsets := #[] } }

/-- A template of one parameter `x` (0), calling a primitive (2); a replay block of parameters `x` and `w` (1), of body
`block` returning `ret`. -/
private def exDefs (block : Array Flock.Program.Node) (ret : Array Flock.Program.Part)
    (rty : Flock.Program.Ty := .value 8) : Array Flock.Program.Defn :=
  #[exDefn "Pair_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1],
    exDefn "Block_v0" false #["x", "w"] block ret rty, exDefn "Xor_v1" true #["a"] #[] #[]]

#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
/-- An argument reading `parts`. -/
private def exParts (parts : Array Flock.Program.Part) : Flock.Program.Arg :=
  { ty := .value 8, seq := .ofParts parts }

#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.explicit #[⟨true, 1, 0⟩, ⟨true, 1, 3⟩]]]] #[]) 1 ==
  some #[⟨"Pair_v0", "x", "w"⟩]
#guard linksOf (exDefs #[exNode .call 0 #[exArg false 0]] #[.affine false 0 0 1]) 1 ==
  some #[⟨"Pair_v0", "o0", "o0"⟩]
#guard linksOf (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 0 #[exArg true 1]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .call 0 #[exParts #[.affine true 0 0 1, .affine false 1 0 1]]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .prim 2 #[exArg true 0]] #[]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine true 1 0 1]) 1 == none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  none
#guard linksOf (exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1]) 2 == none

/-- `exDefs` with a second template (3), the replay block calling `Pair_v0` on `x` and it on `w`, returning `ret` of
type `rty`. -/
private def exTwo (ret : Array Flock.Program.Part) (rty : Flock.Program.Ty) : Array Flock.Program.Defn :=
  (exDefs #[exNode .call 0 #[exArg true 0], exNode .call 3 #[exArg true 1]] ret rty).push
    (exDefn "Triple_v0" false #["x"] #[exNode .prim 2 #[exArg true 0]] #[.affine false 0 0 1])

#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .value 8])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Pair_v0", "o0", "o0"⟩, ⟨"Triple_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 1 0 1, .affine false 0 0 1] (.record #[("a", .value 8), ("b", .value 8)])) 1 ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Triple_v0", "x", "w"⟩, ⟨"Triple_v0", "o0", "o0"⟩, ⟨"Pair_v0", "o0", "o1"⟩]
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine true 1 0 1] (.tuple #[.value 8, .value 8])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1, .affine false 1 0 1] (.tuple #[.value 8, .array 2 (.value 8)])) 1 == none
#guard linksOf (exTwo #[.affine false 0 0 1] (.tuple #[.value 8, .value 8])) 1 == none

private def exProg : Flock.Program.Prog :=
  { defs := exDefs #[exNode .batch 0 #[exArg true 0]] #[.affine false 0 0 1], root := 1, programFn := "Block_v0",
    index := Std.HashMap.ofList [("Pair_v0", 0), ("Block_v0", 1), ("Xor_v1", 2)], sha512 := ByteArray.empty }

#guard replayLinks exProg "Block_v0" (exRec 4 [("Block_v0/x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "Block_v0/x"⟩, ⟨"Pair_v0", "o0", "Block_v0/o0"⟩]
#guard replayLinks exProg "Block_v0" (exRec 4 [("x", 5, 6)]) ==
  some #[⟨"Pair_v0", "x", "x"⟩, ⟨"Pair_v0", "o0", "o0"⟩]
#guard replayLinks exProg "Other_v0" (exRec 4 [("x", 5, 6)]) == none

end Spot.SecondStage
