/-
SPOT's one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, its `program` C, plus `proofs`, exactly `{"public", "registered"}`: the SHA-512 the auditor
  committed for the public file (lowercase hex of 64 bytes) and theta's registered file (`verity-verify-proof
  --registered`'s object, which states no position, or null for a program that reads no registered value). The partition must not be null: the
  sessions are verified against the verifier's own partition object; and v1's `challenger`, with its `window_root`, must
  be named (`NO_CHALLENGER`), its key at the deployed LMS parameters, `Lms.deployed` (`KEY_PARAMS`). The sampling
  design must be a `subset` one, or a rate stated as text that names one over the population (`rate:NUM/DEN[:T]`): a
  `stratified` or `work` one, written as an object or as a rate, is refused by its kind (`SUBSET_ONLY`,
  `paramsOfJson_subset`), since
  the one statement's soundness statement covers a uniform sample of a fixed count alone;
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, its `coins` within the challenger evidence's limits (`Challenger.withinAt`
  at `Challenger.LIMITS`), plus `proofs`, exactly `{"sessions"}`: for each of v1's `sessions` in order,
  `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and `{file: SHA-512}` naming at least each rep
  file the statement's session reads (`<stream with / as .>.bin`). The sampled proof blocks are proved in one session, whose
  statement is theirs: the check refuses evidence naming none or several (`oneSession`) before it reads a tree;
* **blobs**: the files those digests name (`Core.Blobs`). The registered circuit names the lookup tables its session
  reads (its META's `lookups`, `lookupsOf`), so blobs whose pinned MUFU tables (`tablesOf`, each under its SHA-512 as
  `verity-verify-proof --archive` finds one) are other than those refuse (`tablesHeld`, `NO_TABLES`) before the session is set
  up: the tables the session `run` sets up reads are the registered circuit's. The per-member form holds each member
  to its own circuit's tables instead (below, **per member**);
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.sampleCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a sample other than the one they give before v1's `check`;
* **trees** (`treesOf`, `orderHeld`, `treesHeld`): the coin trees that same session's signed release log shows the
  challenger sampled and opened (`Challenger.trustedTreesOf`). The log must release the sample, then the link, then the
  other rounds: no coin of the parameters' challenger family after the registration's digest released after a round is
  opened, and the first round opened a tree's `session` stream (`Challenger.sampleLinkFirst`, `SAMPLE_LINK_FIRST`). Each
  session's coins must be one tree's, a tree no other session's, its record's `coin_tree`, link points and every
  round's coins as the challenger logged them, its link's `root_b` naming exactly the session's tables
  (`sessionTables`), the roots the link points are opened after, each entry the length its table's statement fixes,
  `2^cap₀` digests (`rootBTables`, `Challenger.LINK_ROOT_LEN`), refused before the link's message is hashed, which is
  what makes that message parse one way, and every tree logged must be a session's (`Challenger.sessionsTreeReason`);
  else the check refuses, before any session is verified. A session whose statement sets up no table has no length to
  hold, and its own verdict rejects;
* **population** (`popHeld`): the registration's population held to the verifier's program: under its query, the
  program's proof blocks read stratum by stratum must be the record's `population` units in order (`Slots.popOk`), as
  the service's run refuses at registration (`Run.run`, `population-order`), else the check refuses, before any session
  is verified. v1's R2 holds only their count, and the session's knowledge soundness reads slot `u` as proof block `u`;
* **public** (`publicHeld`): the public file's header held to the registration (`Record.matchesPublicBy`): its scheme,
  roots and their domains and its instance count the record's, and its `units.indices` the population's units in
  order (`indicesHeld`, `indicesHeld_of_publicHeld`), so no header written after the sample relabels an instance as
  another unit; else the check refuses, before any session is verified. The public file names its ports bare; a window
  of one member registers them `<member>/<port>`, and the header is held to that member's roots with its name taken
  off (`publicPorts`); a registration of several members' roots refuses (`NOT_ONE_MEMBER`);
* **check**: `verity-verify-proof verify --zk --statement verity/flock-circuit` with `--circuit` C_S (`restrict`),
  `--public`, `--partition`, `--program` C and `--registered` theta's registration, the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the Flock library calls `verifyCmd` makes on that path, its last the files' ports, domains and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV3` covers, or at a shared-row public file
  `SampledKSLiveV5`'s, `Flock.KsScope.checkOf`), and a session's verdict holds only where the registration's population
  is the circuit file's units in order (`FilePart.check`: its strata over the verifier's program and query, read stratum
  by stratum, are its units in order, and the public file holds exactly those, so slot `u` is the file's unit `u`, as
  `Flock.Bridge.partOf` reads it); then v1's `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `verity-verify-proof` can't read is. So does a length a session's
  statement can't hold, since it writes each in an 8-byte word: the circuit or public file, the lookup tables, the
  program, the partition object, the registered reads or a session's sample at `2^64` bytes or more (`fitsWhy`,
  `sampleFitsWhy`, the JSON ones as `Flock.RecordText.printT` prints them);
* **per member** (`checkMembersAt`): the form for a population of several members, a statement a sampled member. It
  accepts where every statement's session accepted and v1's check at the coins accepts on their verdicts, as the one
  statement's check does (`checkMembersAt_accept`). v1's check reads the members there
  (`membersParamsOfJson_linked`), so its registration check refuses a registration under which a member's sampled
  proof blocks give two counts of link points (`link-points`); a member a sample misses has no statement. `protocol`
  reads either form (`paramsEitherOfJson`, `evidenceEitherOf`): `proofs` of the one statement's keys is read by
  `paramsOfJson` and `evidenceOf`, whose
  parameters name no members (`paramsOfJson_members`) and whose evidence names no statements
  (`evidenceOf_statements`), and `proofs` of the form's keys by the form's own parsers (`membersParamsOfJson`,
  `membersEvidenceOf`). The check at the coins (`checkBlobsAt`, which `live` runs) is the one statement's
  (`checkOneAt`) on parameters that name no members with evidence that names no statements, and `checkMembersAt` on
  parameters that name members with evidence that names statements (`checkBlobsAt_accept`); parameters of one form
  with evidence of the other reject. Its parameters' `proofs` is exactly
  `{"members"}`: `members` (`membersOf`) an object from each member's name to exactly
  `{"public", "registered", "block", "vus_per_block"}`: its public file and theta's registered file, as the one
  statement's, its block of the partition object's `blocks`, whose range is its units, no unit in two members', and
  its circuit's units a proof block (a member handed `units` is refused by name); its circuit file is C_S, derived from C and the partition (`restrict`). Its evidence's `proofs` is exactly `{"statements"}` (`statementsOf`), a list of
  exactly `{"member", "sessions"}`, the statements' sessions in order being v1's `sessions`. The statements are held as
  public structure before any session is read (`membersHeld`): each names a member of the parameters that no earlier
  statement named, in the members' order (by first unit), with exactly one session, on live coins and never retried;
  a member the sample missed has no statement: a statement of a member the sample holds no unit of refuses
  (`sampledHeld`, at the evidence's `proved_units`), so the statements are exactly the members the sample holds, one
  session each (`membersRun_sampled`). Each session's coins are one tree's, as the one statement's are, and its tree's
  last opening is its closing round (`Challenger.memberTreeReason`): one more `session` round, of no coins, opened after
  the SHA-512 of the session's evidence (`sessionEvidence`: its record's bytes, then each rep file's name and bytes,
  each under an 8-byte length, `evidenceBytes`, which `evidenceParts` decodes), so the files the verdict reads are
  fixed before the next member's coins; and the log holds the trees one after another in the sessions' order, no line
  of a tree, the one that samples it or one opening a round of it, before the closing round of the one before it or
  after its own (`Challenger.membersOrder`). Then every
  member's files are read, sampled or not (`membersFiles`): its statement must read as a sampled member's does
  (`Member.statement`), on its own lookup tables, the blobs' pinned tables its own circuit looks up (`ownTables`), each
  of which the blobs must hold (`tablesMatch`, `NO_TABLES`): the members of a window may look up different tables, and
  a member's inputs hold no table its circuit does not look up (`Properties.Spot.TablesRegistered`). Its files pass
  what a session's verdict checks of the files alone (`StepAt.filesHeld`: the
  scope, the files' ports and roots, the form the sampled audit's knowledge soundness covers without a session's
  sample, `Flock.KsScope.filesOf`, the public file's header and rows as a session's setup takes them,
  `Flock.PortRoots.headerOf`, and the population check), so a member the sample missed is refused where its
  session's verdict would have refused its files (`membersRun_files`). A member's circuit file is parsed once for
  them (`StepAt.ofOwn`), and the statement keeps each check's result, never the parse. The compiled pass
  (`membersFilesKept`, equal to it) reads the blobs' lookup tables once and a circuit file once for the members that
  name it, at the tables of them it looks up, and keeps of the file the circuit's own verdicts and the circuit without
  its rows (`Spot.CircuitKept`),
  so its work on the rows is by the file and not by the member; a member's public file it loads once for the four
  checks that read it, and a program it parses and hashes once for the members in a row that read it. Then each
  statement's session is verified as the one statement's is, on its member's own files as that pass read them
  (`runMembers`), after the statements before it, so no coin tree serves two statements: its circuit's units a proof
  block must be the ones the parameters name, and its verdict holds only where its strata are the registration's, each
  restricted to its units (`Member.sizes`: the parameters fix each stratum's size in a member, and nothing the prover
  sends moves a unit between strata), and its public file's units are exactly its member's, in order
  (`Member.population`), so every unit its session proves is its member's; the units the statements proved must be the
  evidence's `proved_units` (`membersRun_proved`). The registration is held to the members before any
  statement's session: the members' units are its population, each member one range of it (`laidHeld`), and each of its roots is
  exactly one member's, the member whose name, then `/`, the root's port begins with, or the window's tree of a
  member's template's field, `<template>/<field>` (`rootsOf`; any other root that is no member's, or two members',
  refuses, and so does a window's row tables' root, `ROW_TABLES`). A member's public file is held to its slice of the
  registration (`Member.slice`, `headerHeld`): the window's scheme, the member's roots and their domains under their
  bare ports, its template's fields' trees at the output ports it has no root of (`ownFieldsOf`), and the member's
  count of units as its instances. Where C has an edge, a wired port with no tree under the member is its writer's
  output field's tree, by C's wiring under the partition (`derivedWiring`: one tree for each writer template and output
  field, the record's root `<template>/<field>`, `fieldTreesOf`), which the slice holds under the reader's port
  (`membersRootsOf`, `fieldRoots`); a port of the public file with no root in the slice is refused by name
  (`UNWIRED_WHY`). Every committed tree a member's registered file opens, its own `committed/<port>`, is the record's
  root of it, its root, domain and leaf count (`treePortsHeld`): a genuine tree of another window or another member's
  refuses; a port whose value is a writer's field tree's name refuses (`WIRED_TREE_WHY`). A member may read registered values, with its own
  registered file as the one statement's, and a value is one tree for every member (`classHeld`): two members' ports of
  one value that differ in its root, leaves, row width, domain or leaf schema refuse, whether or not either is sampled.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.FilePart
public import Spot.CircuitKept
public import Spot.OneStageZkStatement
public import Spot.OneStageZkRun
public import Spot.OneTree
public import Spot.MembersWires
public import Flock
public import Flock.JsonParse
public import Std.Data.HashMap
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.Slots
meta import Spot.FilePart
meta import Spot.OneStageZkStatement
meta import Spot.OneStageZkRun
meta import Spot.OneTree
meta import Spot.MembersWires
meta import Flock
meta import Flock.JsonParse
meta import Std.Data.HashMap

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-! ## The checks and the commands

The parameters, one statement's check, a member's statement and the files' pass are in `Spot/OneStageZkStatement.lean`,
the pass with each circuit file read once in `Spot/OneStageZkKept.lean`, and the run over the members and the evidence's
parts in `Spot/OneStageZkRun.lean`, in this namespace. -/

/-- **A registered value is one tree for every member**: over the parameters' members, sampled or not, every port of
one value names one root, one count of leaves, one row width, one domain and one leaf schema (`classStep`), so the
values the members read are a function of the parameters, each with one root. It is one pass over the parameters'
members, each member's registered file read once, and it reads no blob. -/
def classHeld (ms : Array Member) : V Unit :=
  ms.foldlM classStep {} >>= fun _ => pure ()

/-- `treePortsHeld` at the reader rows `wired` gives each member (`wiredRows`) and the linked ports `linked`. -/
def treePortsHeldAt (wired : Member → Array (String × String)) (ms : Array Member) (rec : Json)
    (linked : Array Docket.Linked := #[]) : V Unit :=
  let byPort := rootsByPort rec
  ms.forM fun m =>
    registeredTreesHeld byPort s!"member {m.name}" m.own.registered ((templateOf m.name).map (·.1)) (wired m) linked

/-- **Every member's registered file opens only its window's committed trees** (`registeredTreesHeld` at the record
`rec`'s roots by port, `rootsByPort`, built once): a member's own committed port at another root, domain or leaf
count than the record's (another window's tree, another member's) refuses by name, and so does a port whose value is a
writer's field tree's name at a row no wire C states into the member covers (`WIRED_TREE_WHY`, at `wiredRowsOf`).
One pass over the record's roots and the members' registered files. -/
def treePortsHeld (p : Params) (ms : Array Member) (rec : Json) : V Unit :=
  treePortsHeldAt (wiredRowsOf p ms) ms rec p.base.verifier.linked

/-- **The per-member form's structure**, held before its run on the parameters `p`, its members `ms`, the
registration `c`, the evidence `e` and its statements `sts`: each member's name once (`namesOnce`), the statements'
structure (`membersHeld`), their sessions
v1's in order, the sessions' coins held to the trees the challenger logged in the order it must release them, each
tree closed after its session's evidence before the next is opened (`treesOf`, `orderHeld`, `membersTreesHeld`), the
registration's population held to the program (`popHeld`) and the members laid over it (`laidHeld`), each registered
value the members read one tree (`classHeld`), each committed tree a member's registered file opens its window's
(`treePortsHeld`), and, the program marking a registered input the verifier does not expect, a weights' registration
of record R8 accepts on these parameters (`Docket.readPins`, else `Docket.readPinsWhy`); then every port of a
registered input at its registration of record (`pinsHeld`, over the partition's ranges, `Partition.evaluate`). -/
def membersStructure (p : Params) (ms : Array Member) (c : OneStage.Commitment) (e : Evidence)
    (sts : Array StatementFiles) (bs : Core.Blobs) : V Unit := do
  namesOnce ms
  let held ← membersHeld ms sts
  if (sts.flatMap (·.sessions)).map (·.name) != e.base.sessions then
    throw "the statements' sessions are not the evidence's sessions, in order"
  let ts ← treesOf p.base e.base
  orderHeld p c e
  membersTreesHeld ts (← linesOf e) (sts.zip held) bs
  popHeld p.base.verifier c.record
  laidHeld ms c.record
  classHeld ms
  treePortsHeld p ms c.record
  let v := p.base.verifier
  match Docket.readPins v with
  | none => throw (Docket.readPinsWhy v)
  | some rp => pinsHeld v.program.prog (← Partition.evaluate v.program v.query) (Docket.markedGates v) rp ms

/-- The refusal of a member's root of its own at a port C wires into it (`fieldRoots`): a crossing wire is committed
once, in its writer's output field's tree, and the reader's row opens that tree by path. -/
def OWN_ROOT_WHY (member port : String) : String :=
  s!"member {member}: port {port} is a row of in(S) (C wires it from another block's outputs), and the registration " ++
    "holds a root of its own for it: a crossing wire is committed once, in its writer's output field's tree"

/-- The refusal of a member's row `row` C wires into it that is named as its writer's output field: the slice holds
a member's own output and a wired row each at a field tree, and `headerHeldAt` tells them apart by that name. -/
def WIRED_AS_FIELD_WHY (member row : String) : String :=
  s!"member {member}: its row {row} of in(S) is named as the writer's output field it reads, and a member's slice " ++
    "names a field tree's field only at the member's own output"

/-- **The members' roots with their wired ports'**: each member's roots (`roots`), then, for each wire of the wiring
`W` whose reader units meet the member's (a wire's are its template's own count, so a range's are those of its
template's earlier ranges on), its writer's output field's tree among the window's trees `ft` (`fieldTreesOf`,
`fieldRootIn`) under the reader row the wire covers (`wireRow`: the port, or its cut row `<port>#<j>`), the first
wire's of each row, so every cut row of a wire gets its writer's tree. A wired row named as its writer's field
(`WIRED_AS_FIELD_WHY`) and one the member holds a root of its own of (`OWN_ROOT_WHY`) are refused by name. A
reader's wired rows are references with paths into that tree; one whose writer's field the record names no tree of
gets none, so its member's public file naming it is refused by name (`UNWIRED_WHY`). Its cost is the
members' and the wires' numbers, each lookup among one template's fields. -/
def fieldRoots (W : Spot.Wires.Wiring) (rs : Array (String × Nat × Nat)) (ms : Array Member)
    (ft : Std.HashMap String (Array (String × Json))) (roots : Std.HashMap String (Array Json)) :
    V (Std.HashMap String (Array Json)) :=
  let lbase := localBases rs
  ms.foldlM (fun acc m => do
    let inRange := wireReaches rs lbase m
    let own := roots.getD m.name #[]
    if let some w := W.wires.find? fun w => inRange w && wireRow W w == w.out then
      throw (WIRED_AS_FIELD_WHY m.name w.out)
    if let some w := W.wires.find? fun w => inRange w && own.any (getD · "port" == Json.str (wireRow W w)) then
      throw (OWN_ROOT_WHY m.name (wireRow W w))
    let mine := W.wires.foldl (fun (got : Array Json) w =>
      if got.any (getD · "port" == Json.str (wireRow W w)) || !inRange w then got
      else match fieldRootIn ft w.writer w.out (wireRow W w) with
        | some r => got.push r
        | none => got) own
    pure (acc.insert m.name mine)) roots

/-- The refusal of a registration whose trees of its templates' output fields are not the ones C and the partition
give (`fieldTreesHeld`, `derivedFieldTrees`), `what` saying which. -/
def FIELD_TREES_WHY (what : String) : String :=
  s!"roots: the registration's trees of its templates' output fields are not the ones C and the partition give " ++
    s!"({what}): a window registers exactly one tree <template>/o<j> for each output field of each member's " ++
    "template, its leaves the template's units in C"

/-- The one Definition of the partition's ranges `rs` over C that the member `m`'s units meet; none where they meet
no range or ranges of two. -/
def Member.tidIn (m : Member) (rs : Array Flock.TemplateQuery.TRange) : Option String :=
  match m.units.foldl (fun acc (lo, hi) => rs.foldl (fun acc r =>
      if r.first < hi && lo < r.first + r.count && !acc.contains r.template then acc.push r.template else acc) acc)
      (#[] : Array String) with
  | #[t] => some t
  | _ => none

/-- **The window's trees of its templates' output fields, from C and the partition** (`rows.Binding.trees`' output
trees): for each member's template (`templateOf`), in the members' order, `<template>/o<j>` for each return field `j`
(`fieldsOf`) of the Definition its members' units are instances of in C (`Member.tidIn`), its leaves that
Definition's units under the partition. Refuses a member whose name names no template or whose units are not one
Definition's, a template whose members' units are two Definitions', and a Definition whose units are two templates'. -/
def derivedFieldTrees (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (ms : Array Member) :
    V (Array (String × Nat)) := do
  let mut tids : Array (String × String) := #[]
  for m in ms do
    let some (t, _) := templateOf m.name | throw (FIELD_TREES_WHY s!"the member {m.name} names no template")
    let some tid := m.tidIn rs | throw (FIELD_TREES_WHY s!"the member {m.name}'s units are not one Definition's in C")
    match tids.find? (·.1 == t) with
    | some (_, d) => if d != tid then throw (FIELD_TREES_WHY s!"the template {t}'s members' units are two Definitions'")
    | none =>
      if tids.any (·.2 == tid) then throw (FIELD_TREES_WHY s!"the Definition {tid}'s units are two templates'")
      tids := tids.push (t, tid)
  tids.foldlM (fun acc (t, tid) => do
    let some fn := P.index[tid]? >>= (P.defs[·]?)
      | throw (FIELD_TREES_WHY s!"the template {t}'s Definition {tid} is no Definition of C")
    let n := (Spot.RecordText.rangesAt rs tid).foldl (fun a r => a + r.count) 0
    pure (acc ++ ((Flock.Program.fieldsOf fn.ret).toArray.mapIdx fun j _ => (s!"{t}/o{j}", n)))) #[]

/-- **The registration's trees of its templates' output fields are exactly `want`** (`derivedFieldTrees`): each
derived tree is a root of the record `rec`, once, at its derived leaf count, and the record holds no other root
`<template>/<field>` of a template of `ts` (`fieldTreesOf`'s). Refuses a tree omitted, held twice, held at another
leaf count, and a root that is no derived tree, by name. -/
def fieldTreesHeld (want : Array (String × Nat)) (ts : Std.HashSet String) (rec : Json) : V Unit := do
  let rs := match getD rec "roots" with | .arr rs => rs | _ => #[]
  let got := rs.filterMap fun r => match getD r "port" with
    | .str port => if (templateOf port).any (ts.contains ·.1) then some (port, getD r "leaves") else none
    | _ => none
  for (tree, n) in want do
    match got.filter (·.1 == tree) with
    | #[] => throw (FIELD_TREES_WHY s!"it omits {tree}")
    | #[(_, l)] => if nat? l != some n then throw (FIELD_TREES_WHY s!"it holds {tree} at {l.compress} leaves, not {n}")
    | _ => throw (FIELD_TREES_WHY s!"it holds {tree} more than once")
  if let some (port, _) := got.find? fun (port, _) => !want.any (·.1 == port) then
    throw (FIELD_TREES_WHY s!"it holds {port}, no output field's tree")

/-- The refusal of a member's root of its own at a port whose argument C reads from a run input node `w`
(`runInputRootsAt`): the node's rows are committed once, in `input/n<w>`, which every unit reading them opens. -/
def RUN_OWN_ROOT_WHY (member port : String) (w : Nat) : String :=
  s!"member {member}: port {port} reads C's run input node {w}, and the registration holds a root of its own for it: " ++
    s!"a run input node's rows are committed once, in {Flock.Registered.freshValue w}"

/-- **A member's ports at C's run input nodes' trees**: for each parameter member `m`'s units read at a run input node
`w` of C (`memberInputs`: every proof block it holds; a form `regRows?` refuses, two nodes or a mixed read refused by
name), the record's root of `input/n<w>` (`Registered.freshValue`, at `byPort`, `rootsByPort`) under the port, with that
tree, which `headerHeldAt` holds as it holds a field tree every member of a template shares (`SHARED_UNREAD`), its
leaves and domain C's and the window's (`runTreeHeld`). A record holding no such root is refused by name
(`RUN_TREE_MISSING_WHY`), and so is a root of the member's own at the port (`RUN_OWN_ROOT_WHY`). A node inside a tied
parameter's gates (`tied`) is not read here: every reader opens it under one root (`tiedRootsAt`), and that the root is
the right one is the later period check's, as R8 publishes. -/
def runInputRootsAt (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (prog : String) (R : Option String)
    (tied : Array (Nat × Nat)) (m : Member) (own : Array Json) (byPort : Std.HashMap String Json) : V (Array Json) := do
  (← memberInputs P rs m).foldlM (init := own) fun got (p, w, len) => do
    if !Flock.Registered.runNode P w || tied.any fun (first, count) => Partition.within P first count w then
      return got
    let t := Flock.Registered.freshValue w
    let some r := byPort.get? t | throw (RUN_TREE_MISSING_WHY m.name p w)
    if own.any (getD · "port" == .str p) then throw (RUN_OWN_ROOT_WHY m.name p w)
    runTreeHeld P prog R w len r
    return got.push (Json.mkObj [("domain", getD r "domain"), ("leaves", getD r "leaves"), ("port", .str p),
      ("root", getD r "root"), ("tree", .str t)])

/-- The refusal of two roots for one tied run input node `w` (`tiedRootsAt`): every reader of a tied node opens one
tree. -/
def TIED_ROOTS_WHY (w : Nat) : String :=
  s!"members: C's tied run input node {w} is read under two roots: every reader of a tied node opens one tree"

/-- `seen`, the root each tied run input node is read under so far, with each node inside the tied gates `tied` that
member `m` reads (`memberInputs`) at the root `own` holds for its port; refused (`TIED_ROOTS_WHY`) where `seen` holds
another root for the node. That the root is the right one is the later period check's. -/
def tiedRootsAt (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange) (tied : Array (Nat × Nat))
    (m : Member) (own : Array Json) (seen : Array (Nat × Json)) : V (Array (Nat × Json)) := do
  (← memberInputs P rs m).foldlM (init := seen) fun seen (p, w, _) => do
    if !Flock.Registered.runNode P w || !tied.any fun (first, count) => Partition.within P first count w then
      return seen
    match (own.find? (getD · "port" == .str p)).map (getD · "root") with
    | none => return seen
    | some r => match seen.find? (·.1 == w) with
      | some (_, r') => if r' == r then return seen else throw (TIED_ROOTS_WHY w)
      | none => return seen.push (w, r)

/-- `runInputRootsAt` for each member of `ms`, over the partition's ranges `rs`, the roots `roots` and the record `rec`,
under the verifier `v`'s program digest, window open root and tied parameters' gates (`Docket.tiedGates`), each tied
node read under one root across the members (`tiedRootsAt`). Its cost is the members' ranges times their Definitions'
parameters. -/
def runInputRoots (v : Docket.Verifier) (P : Flock.Program.Prog) (rs : Array Flock.TemplateQuery.TRange)
    (ms : Array Member) (rec : Json) (roots : Std.HashMap String (Array Json)) :
    V (Std.HashMap String (Array Json)) := do
  let byPort := rootsByPort rec
  let prog := Flock.hexOf v.program.desc.sha512
  let tied := Docket.tiedGates v
  let mut acc := roots
  let mut seen : Array (Nat × Json) := #[]
  for m in ms do
    seen ← tiedRootsAt P rs tied m (acc.getD m.name #[]) seen
    acc := acc.insert m.name (← runInputRootsAt P rs prog v.windowRoot tied m (acc.getD m.name #[]) byPort)
  return acc

/-- **The registration's roots each member's statement reads**: `rootsOf`'s, the window's trees of its templates'
output fields held to the ones C and the partition give (`fieldTreesHeld`, `derivedFieldTrees`), less each output
port a link names, which round 1's record holds (`Docket.Verifier.linked`), each member's own
output ports at its template's fields' trees (`ownFieldsOf`, among the window's trees `fieldTreesOf` reads), each
member's ports at C's run input nodes' trees (`runInputRoots`), and where
C has an edge, each member's wired ports at their writers' output fields' trees (`fieldRoots`, at C's wiring under the
verifier's partition, `derivedWiring`). -/
def membersRootsWired (p : Params) (ms : Array Member) (rec : Json) : V (Std.HashMap String (Array Json)) := do
  let roots ← rootsOf ms rec
  let ts := templatesOf (ms.toList.map (·.name))
  let ft := fieldTreesOf ts rec
  let P ← Program.parse p.own.program
  let rs ← Partition.evaluate P (getD p.own.partition "query")
  fieldTreesHeld ((← derivedFieldTrees P.prog rs ms).filter fun (t, _) =>
    !p.base.verifier.linked.any fun l => s!"{l.template}/{l.port}" == t) ts rec
  let roots := ms.foldl (fun acc m => acc.insert m.name (ownFieldsOf ft m.name (acc.getD m.name #[]))) roots
  let roots ← runInputRoots p.base.verifier P.prog rs ms rec roots
  if !hasEdge P.prog then return roots
  let W ← derivedWiring P.prog rs
  fieldRoots W (rs.map fun r => (r.template, r.first, r.count)) ms ft roots

/-- The refusal of a member's slice holding a root that names no port. -/
def SLICE_NO_PORT_WHY (member : String) : String :=
  s!"member {member}: a root of its slice names no port"

/-- The refusal of a member's slice whose roots name its port `p` twice (`Member.portsOnce`). -/
def PORT_TWICE_WHY (member p : String) : String :=
  s!"member {member}: its slice names port {p} twice: a member's slice holds one root of each port, the one its " ++
    "public file's root there is held to"

/-- The ports of the slice `s`'s roots, each its string, in the slice's order; none where the roots are not a list or
one names no port as a string. -/
def slicePorts (s : Json) : Option (List String) :=
  match getD s "roots" with
  | .arr rs => rs.toList.mapM fun r => match getD r "port" with
    | .str p => some p
    | _ => none
  | _ => none

/-- **A member's slice `s` names each port once**, its ports compared as strings (`slicePorts`): a root that names no
port refuses (`SLICE_NO_PORT_WHY`), and so does a port named twice (`PORT_TWICE_WHY`), whatever the two roots are. -/
def Member.portsOnce (m : Member) (s : Json) : V Unit :=
  match slicePorts s with
  | none => throw (SLICE_NO_PORT_WHY m.name)
  | some ps =>
    match portTwice ps with
    | some p => throw (PORT_TWICE_WHY m.name p)
    | none => pure ()

/-- `Member.portsOnce` on each member of `ms` at its slice `slice m`, in order. -/
def portsOnceAll (slice : Member → Json) : List Member → V Unit
  | [] => pure ()
  | m :: ms => m.portsOnce (slice m) >>= fun _ => portsOnceAll slice ms

/-- **The registration's roots each member's statement reads, each port once**: `membersRootsWired`'s, then every
member's slice of the registration `rec` at them (`Member.slice`), sampled or not, naming each port once
(`portsOnceAll`). -/
def membersRootsOf (p : Params) (ms : Array Member) (rec : Json) : V (Std.HashMap String (Array Json)) :=
  membersRootsWired p ms rec >>= fun roots =>
  portsOnceAll (fun m => m.slice rec roots) ms.toList >>= fun _ =>
  pure roots

/-- **The per-member form's sessions' verdicts** on the parameters `p`, its members `ms`, the registration `c`, the
evidence `e` and its statements `sts`: the form's structure (`membersStructure`), the registration's roots each one
member's, with each member's wired ports at their writers' and its run inputs' at their one trees
(`membersRootsOf`), then every member's files and each statement's session on the blobs, at its member's slice of the registration (`Member.slice`, `runMembers`). -/
def membersRun (p : Params) (ms : Array Member) (c : OneStage.Commitment) (e : Evidence) (sts : Array StatementFiles)
    (bs : Core.Blobs) : V Run :=
  membersStructure p ms c e sts bs >>= fun _ =>
  membersRootsOf p ms c.record >>= fun roots =>
  runMembers (fun m => m.slice c.record roots) ms sts e.base.provedUnits bs

/-- **Where the per-member form's run returns, every member's files are held, sampled or not**: at the roots
`membersRootsOf` gives each member, each member of the parameters has its statement read and its files pass what a
session's verdict checks of them (`Member.FilesHeld`), at its slice of the registration, whatever the evidence's
statements name and whatever the sample holds. -/
theorem membersRun_files {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    ∃ roots, membersRootsOf p ms c.record = .ok roots ∧ ∀ m ∈ ms, m.FilesHeld (m.slice c.record roots) bs := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨roots, hr, h⟩ := Flock.HmNets.bind_ok h
  exact ⟨roots, hr, runMembers_files h⟩

/-- **Where the per-member form's run returns with every session accepted, the units the sessions proved are the
evidence's `proved_units`** (`runMembers_proved`), which the record's checks hold to the sample (`Record.checks`'
`proved`): a sampled proof block whose member has no accepted session of this run refuses it. -/
theorem membersRun_proved {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r)
    (hall : r.all = true) : r.proved.qsort (· < ·) = e.base.provedUnits := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  exact runMembers_proved h hall

/-- **Where the per-member form's run returns, every statement is of a member the sample holds**
(`runMembers_sampled`): the statements' members in order each have a unit among the evidence's `proved_units`, so no
session of the run is of a member the sample missed. -/
theorem membersRun_sampled {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    ∃ held, membersHeld ms sts = .ok held ∧ ∀ m ∈ held, m.sampled e.base.provedUnits = true := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  exact runMembers_sampled h

/-- **Where the per-member form's run returns, its order refusal and its population refusal passed** (`orderHeld`,
`popHeld`): the two the one statement's check makes too, with the same arguments. -/
theorem membersRun_held {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    orderHeld p c e = .ok () ∧ popHeld p.base.verifier c.record = .ok () := by
  unfold membersRun at h
  obtain ⟨⟨⟩, h, -⟩ := Flock.HmNets.bind_ok h
  unfold membersStructure at h
  obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨held, -, h⟩ := Flock.HmNets.bind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, ht, -⟩ := Flock.HmNets.bind_ok h
    cases ht
  obtain ⟨ts, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hord, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨lines, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hpop, -⟩ := Flock.HmNets.bind_ok h
  exact ⟨hord, hpop⟩

/-- **The per-member form's check at the coins `ξ`** (module docstring, **per member**): the statements' sessions'
verdicts (`membersRun`), then v1's `checkAt` at `ξ` on them, as the one statement's check: v1 accepts only verdicts
that all accepted, and the run's own `all` says it again, so an acceptance reads the run. -/
def checkMembersAt (p : Params) (ms : Array Member) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (sts : Array StatementFiles) (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  if e.base.sessionRoots.isNone then .reject NO_SESSION_ROOTS else
  if e.base.served.isNone then .reject NO_SERVED else
  match membersRun p ms c e sts bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
    | .reject why => .reject why
    | .undecided why => .undecided why

/-- **What the per-member form's check accepts**: its run returned (`membersRun`) with every session accepted, and
v1's check at the coins accepts on the run's verdicts, with the check's public output and profile. -/
theorem checkMembersAt_accept {p : Params} {ms : Array Member} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkMembersAt p ms c ξ e sts bs = .accept pub prof) :
    ∃ r, membersRun p ms c e sts bs = .ok r ∧ r.all = true ∧
      OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } = .accept pub prof := by
  unfold checkMembersAt at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  · rename_i r hr
    split at h
    · rename_i pub' prof' hc
      split at h
      · rename_i hall
        cases h
        exact ⟨r, hr, hall, hc⟩
      · cases h
    · cases h
    · cases h

/-- The per-member form's check refuses evidence naming no session roots object, whatever else it names. -/
theorem checkMembersAt_noRoots {p : Params} {ms : Array Member} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} (h : e.base.sessionRoots = none) :
    checkMembersAt p ms c ξ e sts bs = .reject NO_SESSION_ROOTS := by
  simp [checkMembersAt, h]

/-- The per-member form's check refuses evidence naming session roots and no served units, whatever else it names. -/
theorem checkMembersAt_noServed {p : Params} {ms : Array Member} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} {roots : Lean.Json}
    (hr : e.base.sessionRoots = some roots) (h : e.base.served = none) :
    checkMembersAt p ms c ξ e sts bs = .reject NO_SERVED := by
  simp [checkMembersAt, hr, h]

/-- The refusal of parameters and evidence of two forms (`checkBlobsAt`). -/
def FORMS : String :=
  "proofs: the parameters' proofs and the evidence's are not of one form: one statement's files and its sessions, " ++
    "or the members and their statements"

/-- **The check at the coins `ξ`**, by the form of the parameters' `proofs` and the evidence's: the one statement's
(`checkOneAt`), or the per-member form's (`checkMembersAt`); parameters of one form with evidence of the other
reject (`FORMS`). -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p.members, e.statements with
  | none, none => checkOneAt p c ξ e bs
  | some ms, some sts => checkMembersAt p ms c ξ e sts bs
  | _, _ => .reject FORMS

/-- **What `checkBlobsAt` accepts is one form's accept**: the one statement's check accepts, on parameters that name
no members and evidence that names no statements; or the per-member form's check accepts, on parameters that name
members and evidence that names statements. -/
theorem checkBlobsAt_accept {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    (p.members = none ∧ e.statements = none ∧ checkOneAt p c ξ e bs = .accept pub prof) ∨
      ∃ ms sts, p.members = some ms ∧ e.statements = some sts ∧
        checkMembersAt p ms c ξ e sts bs = .accept pub prof := by
  unfold checkBlobsAt at h
  split at h
  · rename_i hm hs
    exact .inl ⟨hm, hs, h⟩
  · rename_i ms sts hm hs
    exact .inr ⟨ms, sts, hm, hs, h⟩
  · cases h

/-- **On parameters that name no members, what `checkBlobsAt` accepts the one statement's check accepts**, and the
evidence names no statements (`checkBlobsAt_accept`). -/
theorem checkBlobsAt_accept_one {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity} (hm : p.members = none)
    (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    checkOneAt p c ξ e bs = .accept pub prof ∧ e.statements = none := by
  rcases checkBlobsAt_accept h with ⟨-, hs, h1⟩ | ⟨ms, -, hms, -, -⟩
  · exact ⟨h1, hs⟩
  · cases hm.symm.trans hms

/-- **On parameters that name the members `ms`, what `checkBlobsAt` accepts the per-member form's check accepts**, at
the statements the evidence names (`checkBlobsAt_accept`). -/
theorem checkBlobsAt_accept_members {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity} {ms : Array Member} (hm : p.members = some ms)
    (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ sts, e.statements = some sts ∧ checkMembersAt p ms c ξ e sts bs = .accept pub prof := by
  rcases checkBlobsAt_accept h with ⟨hn, -, -⟩ | ⟨ms', sts, hms, hs, h1⟩
  · cases hm.symm.trans hn
  · obtain rfl : ms = ms' := Option.some.inj (hm.symm.trans hms)
    exact ⟨sts, hs, h1⟩

/-- On parameters that name no members and evidence that names no statements, the check is the one statement's. -/
theorem checkBlobsAt_one {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    (hm : p.members = none) (hs : e.statements = none) : checkBlobsAt p c ξ e bs = checkOneAt p c ξ e bs := by
  unfold checkBlobsAt
  simp only [hm, hs]

/-- The audit's check on the blobs: at the coins `coinsOf` reads (`checkBlobsAt`), refused with its reason when none
read. -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  Core.atCoins (coinsOf p c e) (fun ξ => checkBlobsAt p c ξ e bs)

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsEitherOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceEitherOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

/-- The audit on the blobs `bs` reads its sample's coins (`coinsOf`) and decides at them (`checkBlobsAt`). -/
def live (bs : Array ByteArray) : Core.Live (protocol.on bs) where
  Coins := ByteArray
  coins := coinsOf
  checkAt p c ξ e := checkBlobsAt p c ξ e (Core.Blobs.ofArray bs)
  factors _ _ _ := rfl

/-- **What `paramsOfJson` returns, `paramsParsed` returned, under a `subset` sampling design**: its one refusal of its
own is `SUBSET_ONLY`'s. -/
theorem paramsOfJson_ok {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    paramsParsed j = .ok p ∧ getD p.base.verifier.samplingDesign "law" = .str "subset" := by
  unfold paramsOfJson at h
  cases hp : paramsParsed j with
  | error why => simp [hp, bind, Except.bind] at h
  | ok q =>
    simp only [hp, bind, Except.bind] at h
    cases hl : getD q.base.verifier.samplingDesign "law" with
    | str s =>
      simp only [hl] at h
      by_cases hs : s = "subset"
      · subst hs
        simp [pure, Except.pure] at h
        subst h
        exact ⟨rfl, hl⟩
      · simp [hs, throw, throwThe, MonadExceptOf.throw] at h
    | _ => simp [hl, throw, throwThe, MonadExceptOf.throw] at h

/-- **What `paramsOfJson` returns, `paramsParsed` returned** (`paramsOfJson_ok`). -/
theorem paramsOfJson_parsed {j : Json} {p : Params} (h : paramsOfJson j = .ok p) : paramsParsed j = .ok p :=
  (paramsOfJson_ok h).1

/-- **Parsed parameters' sampling design is a `subset` one**: whatever `paramsOfJson` returns names a sampling design of
kind `subset`, the kind the sample reads (`Draw.deriveSamplingDesign`, `Draw.flockSamplingDesign`). -/
theorem paramsOfJson_subset {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    getD p.base.verifier.samplingDesign "law" = .str "subset" :=
  (paramsOfJson_ok h).2

/-- **Parsed parameters name no members**: whatever `paramsOfJson` returns is of the one statement's form. -/
theorem paramsOfJson_members {j : Json} {p : Params} (h : paramsOfJson j = .ok p) : p.members = none := by
  have h := paramsOfJson_parsed h
  unfold paramsParsed at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (rename_i ht _ _ _ _; simp [throw, throwThe, MonadExceptOf.throw] at ht; done)
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **Parsed evidence names no statements**: whatever `evidenceOf` returns is of the one statement's form. -/
theorem evidenceOf_statements {b : ByteArray} {e : Evidence} (h : evidenceOf b = .ok e) : e.statements = none := by
  unfold evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **The per-member form's parameters name members and hand them to the registration's check**: whatever
`membersParamsOfJson` returns names some, and the members its v1 part's verifier holds
(`Docket.Verifier.members`, which `link-points` reads) are those, each its units and its circuit's units a proof
block. -/
theorem membersParamsOfJson_linked {j : Json} {p : Params} (h : membersParamsOfJson j = .ok p) :
    ∃ ms : Array Member, p.members = some ms ∧ p.base.verifier.members = ms.map fun m => (m.units, m.g) := by
  unfold membersParamsOfJson at h
  split at h
  · obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨p0, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨ms, -, h⟩ := Flock.HmNets.bind_ok h
    cases h
    exact ⟨_, rfl, rfl⟩
  · cases h

/-- **The per-member form's parameters name members**: whatever `membersParamsOfJson` returns names some. -/
theorem membersParamsOfJson_members {j : Json} {p : Params} (h : membersParamsOfJson j = .ok p) :
    ∃ ms, p.members = some ms :=
  let ⟨ms, hm, _⟩ := membersParamsOfJson_linked h
  ⟨ms, hm⟩

/-- **Parameters of either form are parsed parameters but for the files and the members**: their v1 part is that of
parameters `paramsParsed` returned, the one statement's own (`paramsOfJson_parsed`) or the per-member form's, read at
`proofs` of no file, but for the members its registration's check reads (`Docket.Verifier.members`), which the
per-member form's parse hands it (`membersParamsOfJson_linked`). -/
theorem paramsEitherOfJson_base {j : Json} {p : Params} (h : paramsEitherOfJson j = .ok p) :
    ∃ j' p' ms, paramsParsed j' = .ok p' ∧
      p.base = { p'.base with verifier := { p'.base.verifier with members := ms } } := by
  unfold paramsEitherOfJson at h
  split at h
  · unfold membersParamsOfJson at h
    split at h
    · obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      obtain ⟨p0, hp0, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      obtain ⟨ms, -, h⟩ := Flock.HmNets.bind_ok h
      cases h
      exact ⟨_, p0, _, hp0, rfl⟩
    · cases h
  · exact ⟨j, p, p.base.verifier.members, paramsOfJson_parsed h, rfl⟩

/-- **Parameters of either form that name no members are the one statement's**: `paramsOfJson` returned them. -/
theorem paramsEitherOfJson_one {j : Json} {p : Params} (h : paramsEitherOfJson j = .ok p) (hm : p.members = none) :
    paramsOfJson j = .ok p := by
  unfold paramsEitherOfJson at h
  split at h
  · obtain ⟨ms, hms⟩ := membersParamsOfJson_members h
    cases hm.symm.trans hms
  · exact h

/-- **The per-member form's evidence names its statements, and its sessions are theirs in order**: whatever
`membersEvidenceOf` returns names statements whose sessions, one statement's after another's, are its sessions. -/
theorem membersEvidenceOf_sessions {b : ByteArray} {e : Evidence} (h : membersEvidenceOf b = .ok e) :
    ∃ sts, e.statements = some sts ∧ e.sessions = sts.flatMap (·.sessions) := by
  unfold membersEvidenceOf at h
  obtain ⟨j, -, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  split at h
  · obtain ⟨sts, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨one, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨e0, -, h⟩ := Flock.HmNets.bind_ok h
    cases h
    exact ⟨sts, rfl, rfl⟩
  · cases h

/-- **Evidence of either form that names the statements `sts` has their sessions, in order**: the per-member form's
parser returned it (`membersEvidenceOf_sessions`), since the one statement's names none (`evidenceOf_statements`). -/
theorem evidenceEitherOf_sessions {b : ByteArray} {e : Evidence} {sts : Array StatementFiles}
    (h : evidenceEitherOf b = .ok e) (hs : e.statements = some sts) : e.sessions = sts.flatMap (·.sessions) := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · obtain ⟨sts', hs', hse⟩ := membersEvidenceOf_sessions h
      obtain rfl : sts = sts' := Option.some.inj (hs.symm.trans hs')
      exact hse
    · cases hs.symm.trans (evidenceOf_statements h)
  · cases h

/-- **Evidence of either form has one statement's evidence's v1 part**: the one statement's own, or the per-member
form's, which `evidenceOf` read at `proofs` of the statements' sessions. -/
theorem evidenceEitherOf_base {b : ByteArray} {e : Evidence} (h : evidenceEitherOf b = .ok e) :
    ∃ b' e', evidenceOf b' = .ok e' ∧ e.base = e'.base := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · unfold membersEvidenceOf at h
      obtain ⟨j, -, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      split at h
      · obtain ⟨sts, -, h⟩ := Flock.HmNets.bind_ok h
        try dsimp only at h
        obtain ⟨one, -, h⟩ := Flock.HmNets.bind_ok h
        try dsimp only at h
        obtain ⟨e0, he0, h⟩ := Flock.HmNets.bind_ok h
        cases h
        exact ⟨_, e0, he0, rfl⟩
      · cases h
    · exact ⟨b, e, h, rfl⟩
  · cases h

/-- **Evidence of either form that names no statements is the one statement's**: `evidenceOf` returned it. -/
theorem evidenceEitherOf_one {b : ByteArray} {e : Evidence} (h : evidenceEitherOf b = .ok e)
    (hs : e.statements = none) : evidenceOf b = .ok e := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · obtain ⟨sts, hs', -⟩ := membersEvidenceOf_sessions h
      cases hs.symm.trans hs'
    · exact h
  · cases h

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the sessions' verdicts alone (`verdicts`, with no
proved units to bind), from the verifier's own inputs (`ownOfJson`) and the evidence's `proofs` (`proofsOf`), each file
canonical JSON; one line, `{"verdicts": [verdict, ...]}` in session order or `{"refused": why}`, exit 0 or 1. The
difftest holds it to `verity-verify-proof verify --zk` (`one_stage.proofs.verify`). `restrict C PARTITION BLOCK OUT`:
the circuit file of the block whose first unit is `BLOCK` (`restrict`) written to `OUT`, `PARTITION` canonical JSON; exit
0, or 1 on its refusal. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | "verdicts" :: own :: proofs :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V (Array (String × Json)) := do
      let sessions ← proofsOf (← Core.canonicalJson proofsB)
      verdicts (← ownOfJson (← Core.canonicalJson ownB)) sessions none (Core.Blobs.ofArray bs)
    let line := match r with
      | .ok vs => Json.mkObj [("verdicts", Json.arr (vs.map (·.2)))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | ["restrict", c, partition, block, out] =>
    let cB ← IO.FS.readBinFile c
    match Core.canonicalJson (← IO.FS.readBinFile partition), block.toNat? with
    | .ok pj, some b => match restrict cB pj b with
      | .ok bs => IO.FS.writeBinFile out bs; IO.println s!"RESTRICTED bytes={bs.size}"; return 0
      | .error e => IO.eprintln s!"restrict: {e}"; return 1
    | .error e, _ => IO.eprintln s!"restrict: the partition object: {e}"; return 2
    | _, none => IO.eprintln "restrict: BLOCK is not a unit index"; return 2
  | _ => protocol.main args

#guard registeredInputName "RegisteredInput96_v1"
#guard registeredInputName "RegisteredInput1_v1"
#guard !registeredInputName "RegisteredInput0_v1"
#guard !registeredInputName "RegisteredInput01_v1"
#guard !registeredInputName "RegisteredInput_v1"
#guard !registeredInputName "RegisteredInput96_v2"
#guard !registeredInputName "XRegisteredInput96_v1"
#guard readsRegistered (Json.mkObj [])
#guard !readsRegistered (Json.mkObj [("definitions", Json.mkObj [("Rope_v1", Json.null)])])
#guard readsRegistered (Json.mkObj [("definitions", Json.mkObj [("RegisteredInput96_v1", Json.null)])])
#guard (provedUnits "{\"units\":{\"indices\":[7,3,9]}}\nrest".toUTF8 "{\"unit_draw\":{\"units\":[2,0]}}").toOption
  == some #[7, 9]
#guard (provedUnits "{\"units\":{\"indices\":[7]}}".toUTF8 "{\"unit_draw\":{\"units\":[1]}}").toOption.isNone
#guard (provedHeld (#[9, 3] ++ #[7]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3, 7] ++ #[7, 9]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3] ++ #[9]) #[3, 7, 9]).toBool
#guard !((provedUnits "{\"units\":{\"indices\":[7,3,9]}}".toUTF8 "{\"unit_draw\":{\"units\":[2,2]}}") >>=
  (provedHeld · #[9])).toBool
#guard (oneSession #[⟨"s", "", #[]⟩]).toBool
#guard !(oneSession #[]).toBool
#guard !(oneSession #[⟨"s", "", #[]⟩, ⟨"t", "", #[]⟩]).toBool

private def heldRecord : Json :=
  (Flock.JsonParse.parse
    "{\"population\":2,\"roots\":[{\"domain\":\"dd\",\"port\":\"x\",\"root\":\"aa\"}],\"scheme\":\"s\"}").toOption.getD
    .null

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

#guard refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":[0,1]}}\nrest").toUTF8) == ""
#guard [("[1,0]" : String), "[0,2]", "[0]", "[0,1,2]", "[\"0\",1]"].all fun is => refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":" ++ is ++ "}}").toUTF8) == INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"ab\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots"
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"de\"},\"roots\":{\"x\":\"aa\",\"y\":\"bb\"}},\"instances\":3,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots, domains, population"
#guard (refusal (publicHeld heldRecord "not a public file".toUTF8)).startsWith "the public file's header is not JSON: "

/-- A registration of population 2 whose roots are at `ports`, roots `aa`, `ab`, … and domains `da`, `db`, … in order. -/
private def portsRecord (ports : List String) : Json :=
  Json.mkObj [("population", Lean.toJson (2 : Nat)), ("scheme", .str "s"), ("roots", .arr (ports.toArray.mapIdx
    fun i p => Json.mkObj [("domain", .str s!"d{i}"), ("port", .str p), ("root", .str s!"r{i}")]))]

/-- A public file of population 2 naming its roots at `ports`, as `portsRecord`'s in order. -/
private def portsPublic (ports : List String) : ByteArray :=
  let named (pre : String) := Json.mkObj (ports.mapIdx fun i p => (p, .str s!"{pre}{i}"))
  let header := Json.mkObj [("frame_v3", Json.mkObj [("domain_ids", named "d"), ("roots", named "r")]),
    ("instances", Lean.toJson (2 : Nat)), ("scheme", .str "s"),
    ("units", Json.mkObj [("indices", Lean.toJson #[(0 : Nat), 1])])]
  (header.compress ++ "\nrest").toUTF8

#guard commonMember ["T{K=1}/c0/x", "T{K=1}/c0/y"] == some "T{K=1}/c0"
#guard commonMember ["T/c0/x", "T/c1/x"] == none
#guard commonMember ["A/c0/x", "B/c0/x"] == none
#guard commonMember ["x", "T/c0/y"] == none
#guard commonMember [] == none
#guard refusal (publicHeld (portsRecord ["x", "y"]) (portsPublic ["x", "y"])) == ""
#guard refusal (publicHeld (portsRecord ["T{K=1}/c0/x", "T{K=1}/c0/y"]) (portsPublic ["x", "y"])) == ""
#guard refusal (publicHeld (portsRecord ["T{K=1}/c0/x", "T{K=1}/c0/y"]) (portsPublic ["y", "x"])) ==
  "public file: not the registration's: roots, domains"
#guard refusal (publicHeld (portsRecord ["T{K=1}/c0/x", "T{K=1}/c0/y"])
  (portsPublic ["T{K=1}/c0/x", "T{K=1}/c0/y"])) == "public file: not the registration's: roots, domains"
#guard [["T/c0/x", "T/c1/x"], ["A/c0/x", "B/c0/x"], ["x", "T/c0/y"], ["T/c0/x", "row-table/s0/t"]].all fun ps =>
  refusal (publicHeld (portsRecord ps) (portsPublic ["x", "y"])) == NOT_ONE_MEMBER

/-- A registered file reading port `x` at `value`. -/
private def xReadT (value : String) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT), ("ports", Json.mkObj [("x", Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson (2 : Nat)), ("root", Json.str ("".pushn 'a' 128)),
      ("value", Json.str value)])])]

-- a header port the registration holds no root of is not compared only where the verifier's registered file reads it
-- at a run input node's tree
#guard refusal (publicHeld (portsRecord ["y", "input/n3"]) (portsPublic ["y", "x"])) ==
  "public file: not the registration's: roots"
#guard refusal (publicHeld (portsRecord ["y", "input/n3"]) (portsPublic ["y", "x"]) (some (xReadT "input/n3"))) == ""
#guard refusal (publicHeld (portsRecord ["y", "input/n3"]) (portsPublic ["y", "x"]) (some (xReadT "w"))) ==
  "public file: not the registration's: roots"
-- a port the registration holds a root of is compared, whatever the registered file reads it at
#guard refusal (publicHeld (portsRecord ["input/n3", "x"]) (portsPublic ["x"]) (some (xReadT "input/n3"))) ==
  "public file: not the registration's: roots, domains"

/-- A registration holding run input node 3's tree at the root of `c`s, `d`'s domain and 2 leaves. -/
private def freshTreeT (c : Char) : Json :=
  Json.mkObj [("roots", .arr #[Json.mkObj [("domain", .str ("".pushn 'd' 128)), ("leaves", Lean.toJson (2 : Nat)),
    ("port", .str "input/n3"), ("root", .str ("".pushn c 128))]])]

-- what holds the port `publicHeld` leaves out: the registered file opens it under the registration's tree, and a
-- header root over other rows than that tree's, its registered file's root another, refuses by name
#guard refusal (registeredTreesHeld (rootsByPort (freshTreeT 'a')) "the verifier" (some (xReadT "input/n3"))) == ""
#guard refusal (registeredTreesHeld (rootsByPort (freshTreeT 'c')) "the verifier" (some (xReadT "input/n3"))) ==
  TREE_OTHER "the verifier" "x" "input/n3" "root"

/-- A circuit file whose META looks up the MUFU tables `names`. -/
private def lookingUp (names : List String) : ByteArray :=
  s!"circuit\nMETA {(Json.mkObj [("lookups", Json.mkObj (names.map (·, Json.mkObj [])))]).compress}\nrest".toUTF8

/-- The verifier's own files naming the circuit file `c` and no public file. -/
private def ownOf (c : ByteArray) : Own :=
  { programJ := Json.mkObj [("definitions", Json.mkObj [])], program := c, partition := Json.null, pub := "",
    registered := none }

/-- Blobs holding the circuit file `c` and, under their pinned SHA-512s whatever their bytes, the MUFU tables `names`. -/
private def withTables (c : ByteArray) (names : List String) : Core.Blobs :=
  ⟨#[(Flock.Sha512.hash c, c)] ++ (Flock.MUFU_TABLES.filter fun t =>
    names.contains t.name).toArray.map fun t => ((Flock.unhex t.sha512).toOption.getD .empty, .empty)⟩

private def held (lookups tables : List String) : String :=
  refusal (tablesHeld (ownOf (lookingUp lookups)) (withTables (lookingUp lookups) tables))

/-- The refusal of a statement's circuit file (`restrict`) on a partition object with no blocks. -/
private def NO_BLOCKS_T : String := refusal ((fun _ => ()) <$> restrict .empty Json.null 0)

#guard NO_BLOCKS_T == "restrict: the partition object has no blocks"
#guard held [] [] == NO_BLOCKS_T
#guard held ["ex2", "rcp"] ["rcp", "ex2"] == NO_BLOCKS_T
#guard (lookupsOf "circuit\nMETA {\"lookups\":[]}\n".toUTF8).toOption == none
#guard (lookupsOf "circuit\nMETA {\"lookups\":null}\n".toUTF8).toOption == some []
#guard (lookupsOf "circuit\nMETA {\"lookups\":{\"foo\":{},\"ex2\":{}}}\n".toUTF8).toOption == some ["ex2"]
#guard refusal ((fun _ => ()) <$> run (ownOf (lookingUp [])) #[] none (withTables (lookingUp []) ["rcp"])) == NO_BLOCKS_T
#guard (tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str
  "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc"])])).toOption ==
  some (Json.mkObj [("why", Json.str NO_BLOCKS_T)])
#guard !(tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str "0"])])).toBool

#guard (rootBLen { (default : Flock.Setup) with m := 25, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 16384
#guard (rootBLen { (default : Flock.Setup) with m := 26, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 8192

/-- The program and the partition the members' statements share, naming no file. -/
private def sharedT : Own :=
  { programJ := Json.null, program := .empty, partition := Json.null, pub := "", registered := none }

private def memberJ (block : Nat) : Json :=
  Json.mkObj [("block", Lean.toJson block), ("public", Json.str ("".pushn 'd' 128)), ("registered", Json.null),
    ("vus_per_block", Lean.toJson (1 : Nat))]

private def membersWhy (j : Json) : String :=
  match membersOf sharedT j with
  | .error e => e
  | .ok _ => ""

-- `sharedT`'s partition names no blocks, so every member's block is refused by name; one handed units, before that
#guard membersWhy (Json.mkObj [("a", memberJ 0)]) == NO_BLOCK "a" none
#guard membersWhy (Json.mkObj [("a", (memberJ 0).setObjVal! "units" (Json.arr #[]))]) ==
  s!"proofs: member a: {UNITS_HANDED}"
#guard membersWhy (Json.mkObj [("a", (memberJ 0).setObjVal! "block" (Lean.toJson (-1 : Int)))]) ==
  "proofs: member a's block is not a natural number"
#guard membersWhy (Json.mkObj [("", memberJ 0)]) == "proofs: a member's name is empty"
#guard membersWhy (Json.mkObj []) == "proofs: members names no member"
#guard membersWhy (Json.mkObj [("a", (memberJ 0).setObjVal! "vus_per_block" (Lean.toJson (0 : Nat)))]) ==
  "proofs: member a's vus_per_block is not a positive natural number"
#guard membersWhy (Json.mkObj [("a", (memberJ 0).setObjVal! "interface" Json.null)]) ==
  "the keys of member a are not block, public, registered, vus_per_block"
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str "a"),
  ("sessions", Json.arr #[])]])])).toOption.map (·.map (·.member)) == some #[some "a"]
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str ""),
  ("sessions", Json.arr #[])]])])).toOption.isNone
#guard (statementsOf (Json.mkObj [("sessions", Json.arr #[])])).toOption.isNone

private def paramsWhy (r : V Params) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

private def evidenceWhy (r : V Evidence) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

/-- Parameters of nothing but `proofs`: each parser refuses them, the one statement's past its `proofs`. -/
private def proofsOnly (pr : Json) : Json := Json.mkObj [("proofs", pr)]

private def oneProofs : Json :=
  let zeros := "".pushn '0' 128
  Json.mkObj [("public", Json.str zeros), ("registered", Json.null)]

#guard paramsWhy (paramsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  "the keys of proofs are not public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly oneProofs)) ==
  "the keys of proofs are not members"
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != ""
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != "the keys of proofs are not public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly oneProofs)) == paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("statements", Json.arr #[])]))) ==
  "the keys of proofs are not public, registered"
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  "the keys of the evidence's proofs are not statements"
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) != ""
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) != ""
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "not JSON".toUTF8) == evidenceWhy (evidenceOf "not JSON".toUTF8)

private def memberT (name : String) (units : Array (Nat × Nat)) : Member := { name, units, g := 1, own := sharedT }

private def statementT (member : String) (sessions : Nat := 1) : StatementFiles :=
  { member := some member, sessions := (List.replicate sessions { name := "s", record := "r", reps := #[] }).toArray }

private def membersT : Array Member :=
  #[memberT "m2" #[(2, 3)], memberT "m10" #[(10, 11)], memberT "m0" #[(0, 1), (5, 6)]]

private def whyT {α : Type} : V α → String
  | .error e => e
  | .ok _ => ""

#guard (memberT "m0" #[(0, 1), (5, 6)]).first == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).indices == #[2, 3, 7]
#guard ((membersHeld membersT #[statementT "m0", statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m0", "m2", "m10"]
#guard ((membersHeld membersT #[statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m2", "m10"]
#guard ((membersHeld membersT #[]).toOption.map (·.size)) == some 0
#guard (whyT (membersHeld membersT #[statementT "m2" 0, statementT "m10"])).startsWith
  "member m2: its statement has 0 sessions"
#guard (whyT (membersHeld membersT #[statementT "m2", statementT "m0"])).startsWith
  "member m0: its statement follows a member's whose units start after its own"
#guard (whyT (membersHeld membersT #[statementT "m0", statementT "m10", statementT "m2"])).startsWith
  "member m2: its statement follows a member's"
#guard whyT (membersHeld membersT #[statementT "m0", statementT "m0"]) ==
  "member m0: a second statement: a member's units are one statement's"
#guard (whyT (membersHeld membersT #[statementT "m0" 2])).startsWith "member m0: its statement has 2 sessions"
#guard whyT (membersHeld membersT #[statementT "m3"]) == "member m3: the parameters name no such member"
#guard (whyT (membersHeld membersT #[{ member := none, sessions := #[] }])).startsWith "a statement of the proofs names"
#guard whyT (namesOnce (membersT.push (memberT "m10" #[(12, 13)]))) == NAME_TWICE_WHY "m10"
#guard (memberT "m0" #[(0, 1), (5, 6)]).sampled #[5]
#guard !(memberT "m0" #[(0, 1), (5, 6)]).sampled #[1, 2, 6]
#guard !(memberT "m0" #[(0, 1)]).sampled #[]
#guard (sampledHeld #[memberT "m0" #[(0, 1), (5, 6)], memberT "m2" #[(2, 3)]] #[2, 5]).toBool
#guard whyT (sampledHeld #[memberT "m0" #[(0, 1), (5, 6)], memberT "m2" #[(2, 3)]] #[0, 3]) ==
  "member m2: the sample holds none of its units, and a member the sample missed has no statement"
#guard (sampledHeld #[] #[]).toBool

private def unitsJ (us : Array Nat) : Json := Json.mkObj [("units", Json.mkObj [("indices", natsJson us)])]

#guard unitsHeld #[2, 3, 7] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3, 7] (unitsJ #[2, 3])
#guard !unitsHeld #[2, 3] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3] (unitsJ #[3, 2])
#guard !unitsHeld #[] Json.null
#guard unitsIn #[(2, 4), (7, 8)] 3 8 == 2
#guard unitsIn #[(2, 4), (7, 8)] 4 7 == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 8)])] == some #[1, 2]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2), (2, 5)]), ("b", #[(5, 9)])] == some #[2, 1]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2)]), ("b", #[(2, 8)])] == some #[0, 3]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 7)])] == none
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 1), (2, 8)]), ("b", #[(1, 2)])] == none
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[2, 3, 7])) == ""
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[0, 1, 2])) ==
  MEMBER_UNITS
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 7)])] (unitsJ #[2, 3, 7])) ==
  MEMBER_STRATA
-- the strata PAIR, TRIPLE, PAIR: a cover, not the units in order stratum by stratum
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt
  #[("PAIR", #[(0, 2), (5, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7])) == ""
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt
  #[("PAIR", #[(0, 2), (5, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7])) == MEMBER_STRATA
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt
  #[("PAIR", #[(0, 2), (5, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[0, 1, 2])) == MEMBER_UNITS
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt
  #[("PAIR", #[(0, 3), (5, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7])) == MEMBER_COVER
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt
  #[("PAIR", #[(0, 2), (6, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7])) == MEMBER_COVER
#guard refusal ({ memberT "m" #[(2, 4), (7, 9)] with subset := true }.heldAt
  #[("PAIR", #[(0, 2), (5, 8)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7, 8])) == MEMBER_COVER
-- strata in order: a subset design's member is on the walk, as `popHeld` takes it first
#guard refusal ({ memberT "m" #[(2, 4), (7, 9)] with subset := true }.heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])]
  (unitsJ #[2, 3, 7, 8])) == MEMBER_STRATA
-- a range that ends before it starts: refused under a subset design, as the walk refuses it
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt
  #[("PAIR", #[(0, 2), (5, 8), (8, 7)]), ("TRIPLE", #[(2, 5)])] (unitsJ #[2, 3, 7])) == MEMBER_COVER
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt
  #[("a", #[(0, 3)]), ("b", #[(3, 8), (8, 7)])] (unitsJ #[2, 3, 7])) == MEMBER_STRATA
#guard refusal ({ memberT "m" #[(2, 4), (7, 8)] with subset := true }.heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])]
  (unitsJ #[2, 3, 7])) == ""
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).population Json.null
  "{\"units\":{\"indices\":[2,3,7]}}\nrest".toUTF8) != ""
#guard (runMembers (fun _ => Json.null) #[] #[] #[] ⟨#[]⟩).toOption.map (·.all) == some true
#guard whyT (runMembers (fun _ => Json.null) #[] #[] #[1] ⟨#[]⟩) ==
  "the units the sessions proved are not the evidence's proved_units"
#guard whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2"] #[2] ⟨#[]⟩) == s!"member m2: {NO_BLOCKS_T}"
#guard whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2"] #[] ⟨#[]⟩) == unsampled "m2"
#guard whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2", statementT "m10"] #[10] ⟨#[]⟩) ==
  unsampled "m2"
#guard whyT (runMembers (fun _ => Json.null) membersT #[] #[] ⟨#[]⟩) == s!"member m2: {NO_BLOCKS_T}"
#guard whyT (runMembers (fun _ => Json.null) #[memberT "m0" #[(0, 1)], memberT "m2" #[(2, 3)]] #[statementT "m2"]
  #[2] ⟨#[]⟩) == s!"member m0: {NO_BLOCKS_T}"
#guard whyT (membersFiles (fun _ => Json.null) ⟨#[]⟩ membersT.toList) == s!"member m2: {NO_BLOCKS_T}"
#guard ((membersFiles (fun _ => Json.null) ⟨#[]⟩ []).toOption.map (·.length)) == some 0
#guard whyT (statementAt [] (memberT "m2" #[(2, 3)])) == "the parameters name no such member"

/-- Blobs of a window of two members, one whose circuit file looks up no table and one whose looks up `rcp`, holding
the MUFU tables `names`. -/
private def twoT (names : List String) : Core.Blobs :=
  ⟨#[(Flock.Sha512.hash (lookingUp []), lookingUp [])] ++ (withTables (lookingUp ["rcp"]) names).entries⟩

/-- Why the statement of a member whose circuit file is `c` does not read on the blobs `bs`: its public file's
refusal, which these members name none of, where its tables pass. -/
private def statedT (c : ByteArray) (bs : Core.Blobs) : String :=
  whyT (({ name := "m", units := #[], g := 1, own := ownOf c } : Member).statement Json.null bs)

#guard statedT (lookingUp []) (twoT ["rcp"]) == NO_BLOCKS_T
#guard statedT (lookingUp ["rcp"]) (twoT []) == NO_BLOCKS_T
#guard (ownTables (ownOf (lookingUp [])) (twoT ["rcp", "ex2"])).toOption.isNone
#guard refusal (tablesMatch [("rcp", .empty)] []) == s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp"
#guard whyT (statementOf (ownOf (lookingUp [])) (fun _ => pure ()) [("rcp", .empty)] (twoT [])) == NO_BLOCKS_T

/-- A statement of no file whose scope, ports, form, header and population checks are the ones given. -/
private def stepT (scope ksFiles ksForm header population : V Unit) : Option StepAt :=
  (Flock.Tags.byName? STATEMENT).map fun tags =>
    { I := inputsOf tags sharedT .empty .empty ⟨#[]⟩, rd := { registered := none, known := #[], params := #[] }, scope,
      ksFiles, ksForm, header, g := none, population, programSha := none, want := Json.null }

#guard ((stepT (.error "scope") (.error "ports") (.error "form") (.error "header") (.error "population")).map fun a =>
  refusal a.filesHeld) == some "scope"
#guard ((stepT (.ok ()) (.error "ports") (.error "form") (.error "header") (.error "population")).map fun a =>
  refusal a.filesHeld) == some "ports"
#guard ((stepT (.ok ()) (.ok ()) (.error "form") (.error "header") (.error "population")).map fun a =>
  refusal a.filesHeld) == some "form"
#guard ((stepT (.ok ()) (.ok ()) (.ok ()) (.error "header") (.error "population")).map fun a =>
  refusal a.filesHeld) == some "header"
#guard ((stepT (.ok ()) (.ok ()) (.ok ()) (.ok ()) (.error "population")).map fun a => refusal a.filesHeld) ==
  some "population"
#guard ((stepT (.ok ()) (.ok ()) (.ok ()) (.ok ()) (.ok ())).map fun a => a.filesHeld.toOption.isSome) == some true

/-- The statement `StepAt.ofOwn` builds of no file: its form check is `Flock.KsScope.filesOf`'s refusal of a circuit
that does not parse, and it names no units a proof block. -/
private def ownT : Option StepAt :=
  (Flock.Tags.byName? STATEMENT).map fun tags =>
    StepAt.ofOwn sharedT tags .empty .empty [] { registered := none, known := #[], params := #[] } (.ok ())

#guard (ownT.map fun a => (refusal a.ksForm).startsWith "refused: ") == some true
#guard (ownT.map fun a => (refusal a.header).startsWith "refused: a circuit that does not parse (") == some true
#guard (ownT.map fun a => a.g) == some none
#guard whyT ((memberT "m0" #[(0, 1)]).blockHeld (some 2)) ==
  "vus_per_block: the member's circuit has 2 units a proof block, not the 1 the parameters name"
#guard ((memberT "m0" #[(0, 1)]).blockHeld (some 1)).toOption.isSome && ((memberT "m0" #[(0, 1)]).blockHeld none).toOption.isSome

private def rootJ (port root : String) : Json :=
  Json.mkObj [("domain", Json.str ("d" ++ root)), ("leaves", Lean.toJson (2 : Nat)), ("port", Json.str port),
    ("root", Json.str root)]

/-- A registration of the population `n` and the roots given, each a port and a root. -/
private def recordT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("population", Lean.toJson n), ("roots", Json.arr (roots.toArray.map fun (p, r) => rootJ p r)),
    ("scheme", Json.str "s")]

/-- Two members whose names hold a `/`. -/
private def keyedT : Array Member := #[memberT "t/c0" #[(0, 2)], memberT "t/c1" #[(2, 5)]]

private def strT : Json → String
  | .str s => s
  | _ => ""

/-- The roots `rootsOf` gives the member `name` of `keyedT`, each its bare port and its root. -/
private def rootsT (roots : List (String × String)) (name : String) : Option (Array (String × String)) :=
  (rootsOf keyedT (recordT 5 roots)).toOption.map fun rs =>
    (rs.getD name #[]).map fun r => (strT (getD r "port"), strT (getD r "root"))

#guard splitsOf "t/c0/x" == [("t", "c0/x"), ("t/c0", "x")]
#guard splitsOf "x" == []
#guard splitsOf "/x/" == [("", "x/"), ("/x", "")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0" == some #[("x", "aa"), ("y", "cc")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1" == some #[("y", "bb")]
#guard rootsT [] "t/c0" == some #[]
#guard (whyT (rootsOf #[memberT "t" #[(0, 2)], memberT "t/c0/u" #[(2, 5)]]
  (recordT 5 [("t/c0/u/y", "bb")]))).startsWith "roots: the registration's root t/c0/u/y is two members'"
#guard whyT (rootsOf #[memberT "t/c0" #[(0, 2)], memberT "t/c0/u" #[(2, 5)]] (recordT 5 [("t/c0/u/y", "bb")])) ==
  NAME_TEMPLATE "t/c0"
#guard rootsT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0" == some #[("x", "aa")]
#guard (whyT (rootsOf keyedT (recordT 5 [("u/o0", "aa")]))).startsWith
  "roots: the registration's root u/o0 is no member's"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c2/x", "aa")]))).startsWith
  "roots: the registration's root t/c2/x is no member's"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c0", "aa")]))).startsWith
  "roots: the registration's root t/c0 is no member's"
#guard whyT (rootsOf keyedT (recordT 5 [("t/c0/x", "aa"), ("row-table", "bb")])) == ROW_TABLES
#guard whyT (rootsOf keyedT (Json.mkObj [("roots", Json.arr #[Json.mkObj [("root", Json.str "aa")]])])) ==
  "roots: a root of the registration names no port"
#guard whyT (rootsOf keyedT Json.null) == "roots: the registration's roots are not a list"
#guard whyT (rootsOf keyedT (recordT 5 [("t/c0/y", "aa"), ("t/y", "bb")])) == OWN_FIELD "t/c0" "y"
#guard whyT (rootsOf keyedT (recordT 5 [("t/y", "bb"), ("t/c1/x", "cc"), ("t/c0/y", "aa")])) == OWN_FIELD "t/c0" "y"
#guard rootsT [("t/c0/x", "aa"), ("t/y", "bb")] "t/c0" == some #[("x", "aa")]
#guard whyT (rootsOf #[memberT "W" #[(0, 1)], memberT "W/c0" #[(1, 2)]] (recordT 2 [("W/y", "aa")])) ==
  NAME_TEMPLATE "W"
#guard whyT (rootsOf #[memberT "W/c0" #[(0, 1)], memberT "W" #[(1, 2)]] (recordT 2 [])) == NAME_TEMPLATE "W"
#guard whyT (rootsOf #[memberT "W/c0" #[(0, 1)], memberT "V" #[(1, 2)]] (recordT 2 [("W/y", "aa"), ("V/y", "bb")])) == ""
#guard whyT (publicPorts (recordT 2 [("t/c0/y", "aa"), ("t/y", "bb")])) == OWN_FIELD "t/c0" "y"
#guard whyT (publicPorts (recordT 2 [("t/c0/x", "aa"), ("t/y", "bb")])) == ""
-- a run input node's tree `input/n<w>` is the window's, no member's and no public file's
#guard rootsT [("t/c0/x", "aa"), ("input/n3", "ee")] "t/c0" == some #[("x", "aa")]
#guard rootsT [("t/c0/x", "aa"), ("input/n3", "ee")] "t/c1" == some #[]
#guard (whyT (rootsOf keyedT (recordT 5 [("input/n03", "ee")]))).startsWith
  "roots: the registration's root input/n03 is no member's"
#guard whyT (rootsOf #[memberT "input" #[(0, 2)], memberT "t/c1" #[(2, 5)]] (recordT 5 [])) == INPUT_MEMBER
#guard (publicPorts (recordT 2 [("x", "aa"), ("input/n3", "ee")])).toOption.map (·.map (·.1)) == some #["x"]
#guard (publicPorts (recordT 2 [("t/c0/x", "aa"), ("input/n3", "ee")])).toOption.map (·.map (·.1)) == some #["x"]
#guard (memberT "m" #[(2, 4), (7, 8)]).count == 3
#guard refusal (laidHeld keyedT (recordT 5 [])) == ""
#guard refusal (laidHeld keyedT (recordT 6 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT (recordT 4 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2), (3, 5)], memberT "b" #[(2, 3)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2)], memberT "b" #[(3, 6)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT Json.null) == "the record names no population"

/-- A registered file of the ports given, each its name, its value, the hex digit its root repeats and its leaves. -/
private def regT (ports : List (String × String × Char × Nat)) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT),
    ("ports", Json.mkObj (ports.map fun (name, value, c, leaves) => (name, Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson leaves),
      ("root", Json.str ("".pushn c 128)), ("value", Json.str value)])))]

/-- The member `name` with the registered file `reg`. -/
private def readerT (name : String) (reg : Json) : Member :=
  { memberT name #[(0, 1)] with own := { sharedT with registered := some reg } }

#guard refusal (classHeld keyedT) == ""
#guard refusal (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'a', 2), ("r", "v", 'b', 2)])]) == ""
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'b', 2)])]) ==
  classTwo "a" "c" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), readerT "c" (regT [("q", "w", 'a', 3)])]) ==
  classTwo "a" "c" "w" "leaves 2, against 3"
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2), ("q", "w", 'b', 2)])]) ==
  classTwo "a" "a" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard (whyT (classHeld #[memberT "a" #[(0, 1)], readerT "c" Json.null])).startsWith "member c: "

/-- The slice `rootsOf` and `Member.slice` give the member `name` of `keyedT` on the roots given. -/
private def sliceT (roots : List (String × String)) (name : String) : Json :=
  match rootsOf keyedT (recordT 5 roots), keyedT.find? (·.name == name) with
  | .ok rs, some m => m.slice (recordT 5 roots) rs
  | _, _ => Json.null

/-- A public file's header of the scheme `s`, `n` instances and the roots given, each a bare port and a root. -/
private def headerT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("frame_v3", Json.mkObj [("domain_ids", Json.mkObj (roots.map fun (p, r) => (p, Json.str ("d" ++ r)))),
    ("roots", Json.mkObj (roots.map fun (p, r) => (p, Json.str r)))]), ("instances", Lean.toJson n),
    ("scheme", Json.str "s")]

#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "bb")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "cc")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == UNWIRED_WHY "y"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0") (headerT 5 [("x", "aa")])) ==
  "public file: not the registration's: population"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa")] "t/c1") (headerT 3 [])) == ""

/-- `Member.portsOnce` of the member `name` of `keyedT` at its slice on the roots given (`sliceT`). -/
private def portsOnceT (roots : List (String × String)) (name : String) : String :=
  match keyedT.find? (·.name == name) with
  | some m => refusal (m.portsOnce (sliceT roots name))
  | none => "no member"

-- a slice naming port x twice, at its true root aa and at another, bb: refused by name, where `headerHeld` alone holds
-- a header at bb
#guard portsOnceT [("t/c0/x", "aa"), ("t/c0/x", "bb")] "t/c0" == PORT_TWICE_WHY "t/c0" "x"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c0/x", "bb")] "t/c0") (headerT 2 [("x", "bb")])) == ""
-- the same roots with x named once in the slice: accepted
#guard portsOnceT [("t/c0/x", "aa"), ("t/c1/x", "bb")] "t/c0" == ""
#guard portsOnceT [("t/c0/x", "aa"), ("t/c1/x", "bb")] "t/c1" == ""
#guard PORT_TWICE_WHY "t/c0" "x" == "member t/c0: its slice names port x twice: a member's slice holds one root of " ++
  "each port, the one its public file's root there is held to"
#guard refusal ((memberT "t/c0" #[(0, 2)]).portsOnce
  (Json.mkObj [("roots", Json.arr #[Json.mkObj [("port", (1 : Nat)), ("root", "aa")]])])) == SLICE_NO_PORT_WHY "t/c0"

/-- The member `name` of `keyedT`'s slice with its template's fields' trees at the fields it has no root of
(`ownFieldsOf`, as `membersRootsOf` gives them). -/
private def ownSliceT (roots : List (String × String)) (name : String) : Json :=
  let reg := recordT 5 roots
  match rootsOf keyedT reg, keyedT.find? (·.name == name) with
  | .ok rs, some m =>
    m.slice reg (rs.insert name (ownFieldsOf (fieldTreesOf (templatesOf ["t/c0", "t/c1"]) reg) name (rs.getD name #[])))
  | _, _ => Json.null

#guard refusal (headerHeld (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0") (headerT 2 [("x", "aa"), ("o0", "ee")]))
  == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0") (headerT 2 [("x", "aa"), ("o0", "ee")])) ==
  UNWIRED_WHY "o0"
#guard refusal (headerHeld (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0") (headerT 2 [("x", "aa"), ("o0", "ff")]))
  == "public file: not the registration's: roots, domains"

-- a member of a template of several members: its header's `o0` is its own tree's (Flock's `checkPublic`), held by its
-- registered read at the template's tree, so its own root is not compared
#guard refusal (headerHeldAt #[("o0", "t/o0")] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 2 [("x", "aa"), ("o0", "ff")])) == ""
-- the old header, its own root and no registered read at the shared tree: refused by name
#guard refusal (headerHeldAt #[] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 2 [("x", "aa"), ("o0", "ff")])) == SHARED_UNREAD "o0" "t/o0"
-- a registered read of `o0` at another tree: not the shared tree's opening, refused by name
#guard refusal (headerHeldAt #[("o0", "t/o1")] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 2 [("x", "aa"), ("o0", "ff")])) == SHARED_UNREAD "o0" "t/o0"
-- a template's one member, its own tree the registration's: compared as before, no read needed
#guard refusal (headerHeldAt #[] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 2 [("x", "aa"), ("o0", "ee")])) == ""
-- a header whose `instances` is not the slice's population: refused before any port is dropped
#guard refusal (headerHeldAt #[("o0", "t/o0")] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 5 [("x", "aa"), ("o0", "ff")])) == POPULATION_WHY
-- the member's own input at another root: still compared, and refused as before
#guard refusal (headerHeldAt #[("o0", "t/o0")] #[] (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
  (headerT 2 [("x", "ab"), ("o0", "ff")])) == "public file: not the registration's: roots, domains"

/-- The ports of the registered file `regT ports`. -/
private def heldT (ports : List (String × String × Char × Nat)) : Array Flock.Registered.Port :=
  (Flock.Registered.ofJson (regT ports)).toOption.getD #[]

/-- `headerHeldAt` on `t/c0`'s slice and a header of its port `x`, its tree at `o0` and `w`, at `reads` and the
registered file of `ports`. -/
private def heldAtT (reads : Array (String × String)) (ports : List (String × String × Char × Nat))
    (n : Nat := 2) (x : String := "aa") : String :=
  refusal (headerHeldAt reads (heldT ports) (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
    (headerT n [("x", x), ("o0", "ee"), ("w", "77")]))

-- a port the slice holds no root of, read at a value the verifier's registered file holds there as a registered value:
-- held by the reads, not compared
#guard heldAtT #[("w", "h/w")] [("w", "h/w", 'a', 2)] == ""
-- read at a value the file does not hold at that port, or holds at another port, or holds as a tree's name (a field
-- tree, a run input's tree, a committed tree): refused by name
#guard heldAtT #[("w", "h/z")] [("w", "h/w", 'a', 2)] == UNREGISTERED_WHY "w" "h/z"
#guard heldAtT #[("w", "h/w")] [("v", "h/w", 'a', 2)] == UNREGISTERED_WHY "w" "h/w"
#guard heldAtT #[("w", "h/w")] [] == UNREGISTERED_WHY "w" "h/w"
#guard [("t/o0" : String), "u/o1", "input/n3", "committed/w"].all fun v =>
  heldAtT #[("w", v)] [("w", v, 'a', 2)] == UNREGISTERED_WHY "w" v
-- no read of it: refused as before
#guard heldAtT #[] [("w", "h/w", 'a', 2)] == UNWIRED_WHY "w"
#guard heldAtT #[("v", "h/w")] [("v", "h/w", 'a', 2)] == UNWIRED_WHY "w"
-- a port the slice holds, read at a registered value: still compared to its root
#guard heldAtT #[("w", "h/w"), ("x", "h/x")] [("w", "h/w", 'a', 2), ("x", "h/x", 'a', 2)] (x := "ab") ==
  "public file: not the registration's: roots, domains"
-- the population first
#guard heldAtT #[("w", "h/w")] [("w", "h/w", 'a', 2)] (n := 5) == POPULATION_WHY

/-- `Member.populationOn` of member `t/c0` with the registered file of `ports`, on its slice and a public file whose
header names its units 0 and 1, its port `x`, its tree at `o0` and `w`: the reads and the ports `headerHeldAt` is
handed are the file's, as `Member.population` reads them. -/
private def populationT (ports : List (String × String × Char × Nat)) : String :=
  let m : Member := { memberT "t/c0" #[(0, 2)] with own := { sharedT with registered := some (regT ports) } }
  let h := (headerT 2 [("x", "aa"), ("o0", "ee"), ("w", "77")]).setObjVal! "units"
    (Json.mkObj [("indices", Lean.toJson [0, 1])])
  refusal (m.populationOn (pure #[("t", #[(0, 5)])]) (ownSliceT [("t/c0/x", "aa"), ("t/o0", "ee")] "t/c0")
    (h.compress ++ "\n").toUTF8)

#guard populationT [("w", "h/w", 'a', 2)] == ""
#guard populationT [("w", "committed/w", 'a', 2)] == UNREGISTERED_WHY "w" "committed/w"
#guard populationT [("w", "input/n3", 'a', 2)] == UNREGISTERED_WHY "w" "input/n3"
#guard populationT [("v", "h/w", 'a', 2)] == UNWIRED_WHY "w"

/-- A root of the port given at the hex digit `c` repeated, its domain `d`'s, two leaves. -/
private def hexRootT (port : String) (c : Char) : Json :=
  Json.mkObj [("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson (2 : Nat)), ("port", Json.str port),
    ("root", Json.str ("".pushn c 128))]

/-- A registration of template `W`'s field `y`'s tree at `a`'s and member `R/c0`'s own `z` at `b`'s. -/
private def treeRecT : Json :=
  Json.mkObj [("population", Lean.toJson (4 : Nat)), ("roots", Json.arr #[hexRootT "W/y" 'a', hexRootT "R/c0/z" 'b']),
    ("scheme", Json.str "s")]

/-- A registered file of one port `x` read as `value`, its root the hex digit `c` repeated, its domain `d`'s, `leaves`
leaves. -/
private def treeRegT (c : Char) (leaves : Nat := 2) (value : String := "committed/W/y") (d : Char := 'd') : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT), ("ports", Json.mkObj [("x", Json.mkObj [
      ("domain", Json.str ("".pushn d 128)), ("leaves", Lean.toJson leaves), ("root", Json.str ("".pushn c 128)),
      ("value", Json.str value)])])]

#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a')] treeRecT) == ""
-- a genuine tree of `W`'s `y` at another root (the last window's): refused
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'c')] treeRecT) == TREE_OTHER "member R/c0" "x" "W/y" "root"
-- the window's root under another domain (the domain binds the window): refused
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (d := 'e'))] treeRecT) ==
  TREE_OTHER "member R/c0" "x" "W/y" "domain"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' 3)] treeRecT) == TREE_OTHER "member R/c0" "x" "W/y" "leaf count"
-- the member's own tree `R/c0/z` named as the tree of `W`'s `y`: refused
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'b')] treeRecT) == TREE_OTHER "member R/c0" "x" "W/y" "root"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "committed/V/y"))] treeRecT) ==
  TREE_UNHELD "member R/c0" "x" "V/y"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'b' (value := "committed/R/c0/z"))] treeRecT) == ""
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "committed/R/c0/z"))] treeRecT) ==
  TREE_OTHER "member R/c0" "x" "R/c0/z" "root"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'c' (value := "w"))] treeRecT) == ""

/-- `R`'s port `x` linked to round 1's `logits`, round 1's record holding it at `f`'s root, `d`'s domain, two leaves. -/
private def linkedT : Array Docket.Linked := #[{ template := "R", port := "x", value := "logits", entry := hexRootT "logits" 'f' }]

-- a linked port reads round 1's tree, held to round 1's record in the place of the window's, which holds no `logits`
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "logits"))] treeRecT linkedT) == ""
-- cheat: at a root that is not round 1's logged root
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "logits"))] treeRecT linkedT) ==
  TREE_OTHER "member R/c0" "x" "logits" "root"
-- cheat: at a tree of round 1 the link does not name
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "bx"))] treeRecT linkedT) ==
  LINKED_TREE_WHY "member R/c0" "x" "logits"
-- cheat: at a tree of the second stage's own, its own output's tree among them
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'b' (value := "committed/R/c0/z"))] treeRecT
  linkedT) == LINKED_TREE_WHY "member R/c0" "x" "logits"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "R/o0"))] treeRecT linkedT) ==
  LINKED_TREE_WHY "member R/c0" "x" "logits"
-- a port no link names is held to the window's record as before
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a')] treeRecT
  (linkedT.map fun l => { l with port := "z" })) == ""
-- a linked port's rows of round 1's value by unit (`Docket.Linked.reads`): parsed where given, none where not
private def linkedJ (reads : List (String × Json)) : Json :=
  Json.mkObj ([("entry", hexRootT "o0" 'f'), ("port", Json.str "o0"), ("template", Json.str "R"),
    ("value", Json.str "o0")] ++ reads)
#guard ((OneStage.linkedOf (linkedJ [("reads", Json.arr #[Lean.toJson (3 : Nat), Json.null])])).toOption.map (·.reads))
  == some #[some 3, none]
#guard ((OneStage.linkedOf (linkedJ [])).toOption.map (·.reads)) == some #[]
#guard (OneStage.linkedOf (linkedJ [("reads", Json.arr #[Json.str "3"])])).toOption.isNone
-- a wired port, its value `W`'s field tree's name: refused whatever its root
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "W/o0"))] treeRecT) == WIRED_TREE_WHY "member R/c0" "x"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'a' (value := "committed/W/o0"))] treeRecT) ==
  TREE_UNHELD "member R/c0" "x" "W/o0"
#guard !fieldTreeName "o0" && !fieldTreeName "W/o" && !fieldTreeName "W/ox" && fieldTreeName "a/b/o12"

/-- A registration of template `W`'s output field `o0`'s tree at `a`'s, shared by its members `W/c0` and `W/c1`. -/
private def treeOutRecT : Json :=
  Json.mkObj [("population", Lean.toJson (4 : Nat)), ("roots", Json.arr #[hexRootT "W/o0" 'a']),
    ("scheme", Json.str "s")]

/-- A registered file of one port `port` read as `value`, at the root of `c`s, `d`'s domain and 2 leaves. -/
private def outRegT (port value : String) (c : Char) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT), ("ports", Json.mkObj [(port, Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson (2 : Nat)), ("root", Json.str ("".pushn c 128)),
      ("value", Json.str value)])])]

-- a member's own output, opened at its template's tree: held to the registration's root, domain and leaves
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "W/c0" (outRegT "o0" "W/o0" 'a')] treeOutRecT) == ""
-- its own output opened at another root than the registration's tree: refused
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "W/c0" (outRegT "o0" "W/o0" 'c')] treeOutRecT) ==
  TREE_OTHER "member W/c0" "o0" "W/o0" "root"
-- another template's output tree named as its own output: refused as a wired read
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "V/c0" (outRegT "o0" "W/o0" 'a')] treeOutRecT) == WIRED_TREE_WHY "member V/c0" "o0"
-- a wired input (port `x`) reading `W`'s tree: still refused
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "W/c0" (outRegT "x" "W/o0" 'a')] treeOutRecT) == WIRED_TREE_WHY "member W/c0" "x"

/-- `R/c0` with C's wires into its rows given (`wiredRows`), every other member none. -/
private def wiredAtT (rows : Array (String × String)) : Member → Array (String × String) :=
  fun m => if m.name == "R/c0" then rows else #[]

-- a wired input C wires from `W`'s field `o0`, opened at `W`'s tree: held to the registration's root, domain and leaves
#guard refusal (treePortsHeldAt (wiredAtT #[("x", "W/o0")]) #[readerT "R/c0" (outRegT "x" "W/o0" 'a')] treeOutRecT) == ""
#guard refusal (treePortsHeldAt (wiredAtT #[("x", "W/o0")]) #[readerT "R/c0" (outRegT "x" "W/o0" 'c')] treeOutRecT) ==
  TREE_OTHER "member R/c0" "x" "W/o0" "root"
-- a field of `W` C does not wire the port from: refused
#guard refusal (treePortsHeldAt (wiredAtT #[("x", "W/o0")]) #[readerT "R/c0" (outRegT "x" "W/o1" 'a')] treeOutRecT) ==
  WIRED_TREE_WHY "member R/c0" "x"
-- a cut row: the wired row `x#1` is admitted, the row `x#0` C wires from no field of `W` is refused
#guard refusal (treePortsHeldAt (wiredAtT #[("x#1", "W/o0")]) #[readerT "R/c0" (outRegT "x#1" "W/o0" 'a')] treeOutRecT) ==
  ""
#guard refusal (treePortsHeldAt (wiredAtT #[("x#1", "W/o0")]) #[readerT "R/c0" (outRegT "x#0" "W/o0" 'a')] treeOutRecT) ==
  WIRED_TREE_WHY "member R/c0" "x#0"
-- another member's wired row is not this one's
#guard refusal (treePortsHeldAt (wiredAtT #[("x", "W/o0")]) #[readerT "R/c1" (outRegT "x" "W/o0" 'a')] treeOutRecT) ==
  WIRED_TREE_WHY "member R/c1" "x"

-- a reader row's name: the port uncut, `<port>#<j>` where the wiring cuts it
#guard wireRow { templates := [], cuts := [("R", [("x", 2)])], wires := [] }
    { reader := "R", port := "x", writer := "W", out := "o0", start := 0, radix := [2], base := 0, step := [1],
      cut0 := 1, cuts := 1 } == "x#1"
#guard wireRow { templates := [], cuts := [], wires := [] }
    { reader := "R", port := "x", writer := "W", out := "o0", start := 0, radix := [2], base := 0, step := [1],
      cut0 := 0, cuts := 1 } == "x"

/-- `treeRecT` with run input node 3's tree, `input/n3`, at the root of `f`s. -/
private def treeFreshRecT : Json :=
  Json.mkObj [("population", Lean.toJson (4 : Nat)),
    ("roots", Json.arr #[hexRootT "W/y" 'a', hexRootT "R/c0/z" 'b', hexRootT "input/n3" 'f']), ("scheme", Json.str "s")]

-- a fresh input, node 3's tree, opened at the record's one root of it
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "input/n3"))] treeFreshRecT) == ""
-- the fresh row changed after the coin: its tree's root is not the record's
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'c' (value := "input/n3"))] treeFreshRecT) ==
  TREE_OTHER "member R/c0" "x" "input/n3" "root"
-- a block reading a run node the record holds no entry of: refused, with no fall back to the file's root
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "input/n4"))] treeFreshRecT) ==
  TREE_UNHELD "member R/c0" "x" "input/n4"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "input/n3"))] treeRecT) ==
  TREE_UNHELD "member R/c0" "x" "input/n3"
-- two blocks of one window opening node 3 under two roots: each held to the record's, and to each other's
#guard refusal (treePortsHeldAt (fun _ => #[]) #[readerT "R/c0" (treeRegT 'f' (value := "input/n3")),
  readerT "R/c1" (treeRegT 'c' (value := "input/n3"))] treeFreshRecT) == TREE_OTHER "member R/c1" "x" "input/n3" "root"
#guard (whyT (classHeld #[readerT "R/c0" (treeRegT 'f' (value := "input/n3")),
  readerT "R/c1" (treeRegT 'c' (value := "input/n3"))])).startsWith
  "registered: members R/c0 and R/c1 read the registered value input/n3 as two"
#guard refusal (treePortsHeldAt (fun _ => #[]) #[memberT "R/c0" #[(0, 1)]] treeRecT) == ""

/-- A writer `W/c0`, template `W`'s two units, and a reader `R/c0`, template `R`'s two after them. -/
private def wiredMs : Array Member := #[memberT "W/c0" #[(0, 2)], memberT "R/c0" #[(2, 4)]]

/-- `R`'s port `x` at unit `u` wired from the output `y` of the template `writer` at unit `u`, or no wire. -/
private def wiringT (wired : Bool) (writer : String := "W") : Spot.Wires.Wiring :=
  { templates := [("R", { units := 2, rows := [("x", { words := 1, outer := 1 })], outs := [] }),
                  ("W", { units := 2, rows := [], outs := [("y", 1)] })],
    cuts := [],
    wires := if wired then [{ reader := "R", port := "x", writer, out := "y", start := 0, radix := [2], base := 0,
                              step := [1], cut0 := 0, cuts := 1 }] else [] }

/-- The reader `R/c0`'s slice at the wiring `W`, on the registration of template `W`'s output field `y`'s tree (its
root `W/y`) and `R/c0`'s own `z`, `R/c0` with no tree of `x`. -/
private def wiredSliceT (W : Spot.Wires.Wiring) : Json :=
  let reg := recordT 4 [("R/c0/z", "bb"), ("W/y", "aa")]
  match rootsOf wiredMs reg >>= fieldRoots W #[("W", 0, 2), ("R", 2, 2)] wiredMs
      (fieldTreesOf (templatesOf ["W/c0", "R/c0"]) reg) with
  | .ok rs => (memberT "R/c0" #[(2, 4)]).slice reg rs
  | .error _ => Json.null

#guard refusal (headerHeld (wiredSliceT (wiringT true)) (headerT 2 [("x", "aa"), ("z", "bb")])) == ""
#guard refusal (headerHeld (wiredSliceT (wiringT true)) (headerT 2 [("x", "cc"), ("z", "bb")])) ==
  "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (wiredSliceT (wiringT false)) (headerT 2 [("x", "aa"), ("z", "bb")])) == UNWIRED_WHY "x"
#guard refusal (headerHeld (wiredSliceT (wiringT true "V")) (headerT 2 [("x", "aa"), ("z", "bb")])) ==
  UNWIRED_WHY "x"
-- the slice's wired root names its writer's tree, `W/y`: a header root over the member's own rows there is held by a
-- registered read of that tree, and refused by name without one
#guard (match getD (wiredSliceT (wiringT true)) "roots" with
  | .arr rs => rs.any (getD · "tree" == Json.str "W/y")
  | _ => false)
#guard refusal (headerHeldAt #[("x", "W/y")] #[] (wiredSliceT (wiringT true)) (headerT 2 [("x", "cc"), ("z", "bb")])) ==
  ""
#guard refusal (headerHeldAt #[] #[] (wiredSliceT (wiringT true)) (headerT 2 [("x", "cc"), ("z", "bb")])) ==
  SHARED_UNREAD "x" "W/y"
#guard refusal (headerHeldAt #[("x", "W/z")] #[] (wiredSliceT (wiringT true)) (headerT 2 [("x", "cc"), ("z", "bb")])) ==
  SHARED_UNREAD "x" "W/y"
-- the writer's own root at the wired row, with no read: the session would hold the reader's rows to the tree's leaves
-- in order, not C's, so it is refused by name too
#guard refusal (headerHeldAt #[] #[] (wiredSliceT (wiringT true)) (headerT 2 [("x", "aa"), ("z", "bb")])) ==
  SHARED_UNREAD "x" "W/y"
-- a wired row named as the writer's output field it reads: refused, so a slice's field-named entry is an own output
#guard whyT (fieldRoots { wiringT true with wires := (wiringT true).wires.map ({ · with port := "y" }) }
    #[("W", 0, 2), ("R", 2, 2)] wiredMs (fieldTreesOf (templatesOf ["W/c0", "R/c0"]) (recordT 4 [("W/y", "aa")])) {}) ==
  WIRED_AS_FIELD_WHY "R/c0" "y"
#guard WIRED_AS_FIELD_WHY "R/c0" "y" == "member R/c0: its row y of in(S) is named as the writer's output field it " ++
  "reads, and a member's slice names a field tree's field only at the member's own output"

/-- A row of 16 `v16` leaves, a `Gemm{K=16, N=16}` unit's `x` and its output. -/
private def rowT : Flock.Program.Ty := .array 16 (.value 16)

/-- The periods' `chain()`: two requests' inputs, then four root nodes of one template `G`, two members each, each
reading the node before it whole (layer 0 at step 0 reads the input), so units `u ≥ 2` read unit `u - 2`'s output. -/
private def chainP : Flock.Program.Prog :=
  let input : Flock.Program.Defn :=
    { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := .value 16, gates := 1,
      paramLeaves := 0, outLeaves := 1 }
  let gemm : Flock.Program.Defn :=
    { key := "G", id := "G", prim := true, params := #[("x", rowT)], ret := rowT, gates := 1, paramLeaves := 16,
      outLeaves := 16 }
  let layer (src : Nat) : Flock.Program.Node :=
    { form := .batch, fn := 1, out := .array 2 rowT, n := 2,
      args := #[{ ty := .array 2 rowT, seq := Flock.Program.Seq.ofParts #[.affine false src 0 32], dims := some (2, 16) }] }
  let nodes := #[{ form := .batch, fn := 0, args := #[], out := .array 32 (.value 16), n := 32 }, layer 0, layer 1,
    layer 2, layer 3]
  let root : Flock.Program.Defn :=
    { key := "R", id := "R", prim := false, params := #[], ret := rowT, gates := 0, paramLeaves := 0, outLeaves := 16,
      body := { params := #[], retType := rowT, nodes := nodes, ret := default, offsets := #[] } }
  { defs := #[input, gemm, root], index := (({} : Std.HashMap String Nat).insert "Input16_v1" 0).insert "G" 1,
    root := 2, programFn := "", sha512 := .empty }

private def chainRs : Array Flock.TemplateQuery.TRange := #[⟨1, "G", 0, 2⟩, ⟨2, "G", 2, 2⟩, ⟨3, "G", 4, 2⟩, ⟨4, "G", 6, 2⟩]

/-- `chain()`'s four members, one a root node's two units. -/
private def chainMs : Array Member := #[memberT "G/c0" #[(0, 2)], memberT "G/c1" #[(2, 4)], memberT "G/c2" #[(4, 6)],
  memberT "G/c3" #[(6, 8)]]

/-- `chain()`'s registration: `G`'s output field `o0`'s one tree (its root `G/o0`), and the member roots given. -/
private def chainRecT (roots : List (String × String)) : Json :=
  recordT 8 (roots ++ [("G/o0", "aa")])

/-- `fieldRoots` on `chain()` at C's wiring under its partition (`derivedWiring`), each member's roots. -/
private def chainRootsT (roots : List (String × String)) : V (Std.HashMap String (Array Json)) := do
  let W ← derivedWiring chainP chainRs
  fieldRoots W (chainRs.map fun r => (r.template, r.first, r.count)) chainMs
    (fieldTreesOf (templatesOf (chainMs.toList.map (·.name))) (chainRecT roots)) (← rootsOf chainMs (chainRecT roots))

#guard (ioCuts chainP chainRs).toOption == some []
#guard ((derivedWiring chainP chainRs).toOption.map (·.wires.map fun w => (w.start, w.radix, w.base, w.out))) ==
  some [(2, [2], 0, "o0"), (4, [2], 2, "o0"), (6, [2], 4, "o0")]
#guard ((chainRootsT [("G/c0/x", "bb")]).toOption.map fun r =>
    ["G/c0", "G/c1", "G/c3"].map fun m => (r.getD m #[]).map (getD · "root")) ==
  some [#[Json.str "bb"], #[Json.str "aa"], #[Json.str "aa"]]
#guard whyT (chainRootsT [("G/c0/x", "bb"), ("G/c2/x", "cc")]) == OWN_ROOT_WHY "G/c2" "x"
-- each wired root names its writer's tree, and C's wires into each member are the rows `portTreeHeld` admits
#guard ((chainRootsT []).toOption.map fun r => (r.getD "G/c1" #[]).map (getD · "tree")) == some #[Json.str "G/o0"]
#guard ((derivedWiring chainP chainRs).toOption.map fun W =>
    chainMs.map (wiredRows W (chainRs.map fun r => (r.template, r.first, r.count)))) ==
  some #[#[], #[("x", "G/o0")], #[("x", "G/o0")], #[("x", "G/o0")]]

/-- `chain()` with its first two root nodes one member, `G/c01`: units 2 and 3 read units 0 and 1 inside it. -/
private def chainMs01 : Array Member := #[memberT "G/c01" #[(0, 4)], memberT "G/c2" #[(4, 6)], memberT "G/c3" #[(6, 8)]]

private def chainRoots01 (roots : List (String × String)) : V (Std.HashMap String (Array Json)) := do
  let W ← derivedWiring chainP chainRs
  fieldRoots W (chainRs.map fun r => (r.template, r.first, r.count)) chainMs01
    (fieldTreesOf (templatesOf (chainMs01.toList.map (·.name))) (chainRecT roots)) (← rootsOf chainMs01 (chainRecT roots))

-- C states the wire into units 2 and 3, but a member over two ranges meets none (`fieldRoots`' `inRange`): `G/c01`'s
-- `x` gets no writer's tree and a root of its own there stands; its block is refused at the door instead
-- (`admittedBlock`'s `operands-outside`, `AA16`'s guard)
#guard ((chainRoots01 []).toOption.map fun r => (r.getD "G/c01" #[]).size) == some 0
#guard whyT (chainRoots01 [("G/c01/x", "bb")]) == ""

/-- `fieldTreesHeld` on `chain()`'s derived trees (`derivedFieldTrees`) and a registration of the roots given, each
its port and leaf count. -/
private def chainTreesT (roots : List (String × Nat)) : String :=
  let rec_ := Json.mkObj [("roots", Json.arr (roots.toArray.map fun (p, n) =>
    Json.mkObj [("leaves", Lean.toJson n), ("port", Json.str p)]))]
  whyT (derivedFieldTrees chainP chainRs chainMs >>= fun want =>
    fieldTreesHeld want (templatesOf (chainMs.toList.map (·.name))) rec_)

#guard (derivedFieldTrees chainP chainRs chainMs).toOption == some #[("G/o0", 8)]
#guard chainTreesT [("G/c0/x", 2), ("G/o0", 8)] == ""
#guard chainTreesT [("G/c0/x", 2)] == FIELD_TREES_WHY "it omits G/o0"
#guard chainTreesT [("G/c0/o0", 2), ("G/c1/o0", 2), ("G/c2/o0", 2), ("G/c3/o0", 2)] ==
  FIELD_TREES_WHY "it omits G/o0"
#guard chainTreesT [("G/o0", 4), ("G/o0", 4)] == FIELD_TREES_WHY "it holds G/o0 more than once"
#guard chainTreesT [("G/o0", 2)] == FIELD_TREES_WHY "it holds G/o0 at 2 leaves, not 8"
#guard chainTreesT [("G/o0", 8), ("G/o1", 8)] == FIELD_TREES_WHY "it holds G/o1, no output field's tree"
#guard whyT (derivedFieldTrees chainP chainRs #[memberT "G" #[(0, 2)]]) ==
  FIELD_TREES_WHY "the member G names no template"
#guard whyT (derivedFieldTrees chainP chainRs #[memberT "G/c0" #[(8, 9)]]) ==
  FIELD_TREES_WHY "the member G/c0's units are not one Definition's in C"
#guard whyT (derivedFieldTrees chainP chainRs #[memberT "G/c0" #[(0, 4)], memberT "H/c0" #[(4, 8)]]) ==
  FIELD_TREES_WHY "the Definition G's units are two templates'"
#guard whyT (derivedFieldTrees chainP (chainRs.push ⟨5, "H", 8, 2⟩) #[memberT "G/c0" #[(0, 8)], memberT "G/c1" #[(8, 10)]]) ==
  FIELD_TREES_WHY "the template G's members' units are two Definitions'"

#guard Flock.hexOf (evidenceBytes #["ab".toUTF8]) == "0000000000000001" ++ "0000000000000002" ++ "6162"
#guard (evidenceParts (evidenceBytes #["ab".toUTF8, .empty, "c".toUTF8])).map (·.map Flock.hexOf) ==
  some #["6162", "", "63"]
#guard (evidenceParts (evidenceBytes #[])).map (·.size) == some 0
#guard Flock.hexOf (evidenceBytes #["a".toUTF8, "b".toUTF8]) != Flock.hexOf (evidenceBytes #["ab".toUTF8])
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).push 0)).isNone
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).extract 0 17)).isNone
#guard (evidenceParts .empty).isNone
#guard (whyT (sessionEvidence ⟨#[]⟩ { name := "s", record := "".pushn 'a' 128, reps := #[] })).startsWith
  "session s's record: "

end Spot.OneStageZk
