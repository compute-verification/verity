module

public import NetworkVerifier.Check
public import NetworkVerifier.Clocks
public import NetworkVerifier.ScanOk
public import CoreProtocol.Profile
public import Flock.Program
public import Flock.Public
public import Flock.WindowRecord
public import InferenceOnlyVerifier.Desc
public import InferenceOnlyVerifier.DefinitionDigest
public import NetworkVerifier.Period.ConnectionRows
public import NetworkVerifier.Period.Start
meta import NetworkVerifier.Check
meta import NetworkVerifier.Clocks
meta import NetworkVerifier.ScanOk
meta import CoreProtocol.Profile
meta import Flock.Program
meta import Flock.Public
meta import Flock.WindowRecord
meta import InferenceOnlyVerifier.Desc
meta import InferenceOnlyVerifier.DefinitionDigest
meta import NetworkVerifier.Period.ConnectionRows
meta import NetworkVerifier.Period.Start

/-! The period check's structures and their decoding, which `NetworkCheck.Period` reads: its parameters, commitment,
evidence, facts and public, and the decoding of each from its JSON. The check and its bound are in
`NetworkCheck.Period`, whose module docstring describes both. -/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkVerifier.Period

open Lean (Json toJson)
open Flock (V)
open NetworkVerifier.Check (Certificate)
open InferenceOnlyVerifier.DefinitionDigest (closureDigest)

def protocolId : String := "verity/certifier/v0"
def commitmentTag : String := "verity-network-certifier/period/v0"
def oneStage : String := "verity/sampled-proofs/one-stage/v1"
/-- A proof block's public outputs are at most 16 bits (the values it commits can be wider). -/
def outBits : Nat := 16
def scheduleTag : String := "verity-network-warden/parameters/v3"
def domainTag : String := "verity-network-warden/cells-domain/v2"
def cellsTag : String := "verity-network-warden/cells/v2"
/-- The tag of the digest a session's start's certificate signs (`certificate.START_TAG`): its base is in milliseconds
of the key's life on the host's raw clock. -/
def startTag : String := "verity-network-certifier/start/v2"
def rowsFormat : String := "verity-network-warden/registration/v2"
def rowSchema : String := "hm96-sha512/row/v2"
/-- The cell kinds (`PAD`, `DATA`, `END`, `FAIL`, `REFUSED`) and the fail reasons the schedule's digest binds. -/
def cellKinds : List Nat := [0, 1, 32, 33, 34]
def failReasons : List String := ["late", "limit", "upstream", "ingress", "malformed", "format"]
/-- The fail reason a schedule with a response cap adds after `failReasons` (`network_certifier.REASONS`' last): a request past
the cap, or whose `max_tokens` doesn't parse. -/
def capReason : String := "cap"
/-- The egress form (`NetworkAttesterParams.egress`) of every schedule the check reads: the egress record's id form. A schedule
of another form has a digest no schedule here has. -/
def egressForm : String := "id"
/-- A cell's header: its kind u8, its connection's ordinal u32 and its payload's length u32. -/
def cellHeader : Nat := 9
/-- The byte of an id request past its `max_tokens` field (`id_form.MAX_TOKENS_AT + ID_DIGITS`): under a response cap a
cell's payload holds at least this much, so the network attester reads the field off the request's first cell. -/
def maxTokensEnd : Nat := 130

/-- A compute audit's layout terms, its free inputs it doesn't register and its monitoring period, and what its Program
gives: its SHA-512, its units (its root's template instances) and its slot entries'
templates (`entries`: each root template node calling a `SlotEntry_v1` or `SelectEntry_v1` Definition, `entryIds`, by
its descriptor id, members and closure digest, `closureDigest`). -/
structure Audit where
  free : Nat
  steps : Nat
  program : String
  units : Nat
  entries : Array (String × Nat × String)

/-- The network attester's schedule when it functions as an active network attester (`network_certifier.NetworkAttesterParams`). -/
structure Schedule where
  bucketMs : Nat
  cellBytes : Nat
  delay : Nat
  egressCells : Nat
  every : Nat
  ingressBuckets : Nat
  ingressCells : Nat
  lanes : Nat
  queue : Nat
  slots : Nat
  window : Nat
  /-- The response cap (`NetworkAttesterParams.max_tokens_cap`), or none. -/
  maxTokensCap : Option Nat := none
  /-- The lengths rows' form (`NetworkAttesterParams.lengths`), 1 or 2, or none. -/
  lengths : Option Nat := none
  /-- The content rows' classes (`NetworkAttesterParams.content`), each (P, O, rows) under the lengths rows' form 1 and (P, O)
  under form 2, or none. -/
  content : Option (Array (Array Nat)) := none
  /-- Whether the schedule has request words (`NetworkAttesterParams.request_words`, true or unset): a words row a release,
  in its class. -/
  requestWords : Bool := false

/-- The `fails` record's reasons under the schedule: `failReasons`, and `capReason` after them under a response cap
(`NetworkAttesterParams.reasons`). -/
def Schedule.reasons (s : Schedule) : List String :=
  if s.maxTokensCap.isSome then failReasons ++ [capReason] else failReasons

/-- S = lanes · r, a row's slots. -/
def Schedule.allocation (s : Schedule) : Nat := s.lanes * s.slots

/-- A slot stride's Program slots, one a lane (`NetworkAttesterParams.program_slots`). -/
def Schedule.programSlots (s : Schedule) : Nat := s.lanes

/-- A window's Program slots: T / `every` slot strides of `programSlots` (`NetworkAttesterParams.window_programs`). -/
def Schedule.executionPhasePrograms (s : Schedule) : Nat := s.window / s.every * s.programSlots

/-- An execution phase in milliseconds on the network attester's clock: its T buckets of `bucket_ms` each. -/
def Schedule.windowMs (s : Schedule) : Nat := s.window * s.bucketMs

/-- The most buckets from a connection's admission to its last egress cell's: D + ⌊(`egress_cells` − 1) / r⌋
(`NetworkAttesterParams.span`). With lengths rows a connection ends by then, and it is at most T. -/
def Schedule.span (s : Schedule) : Nat := s.delay + (s.egressCells - 1) / s.slots

/-- The most requests the network attester releases in one slot stride with lengths rows, a lengths row's slots
(`NetworkAttesterParams.stride_releases`): a released connection holds its lane at least D + 2 − `ingress_buckets` buckets from
its release, so that many buckets hold at most `lanes` releases, and it holds it D + 1 buckets around its release, so
a stride's releases are live at one of 1 + ⌈(`every` − 1) / (D + 1)⌉ buckets. It is read where ingress ends by the
delay (`ingress_buckets` ≤ D), as a schedule with lengths rows has it. -/
def Schedule.strideReleases (s : Schedule) : Nat :=
  s.lanes * min (divCeil s.every (s.delay + 2 - s.ingressBuckets)) (1 + divCeil (s.every - 1) (s.delay + 1))

/-- A bound on the requests the network attester releases in one execution phase with lengths rows
(`NetworkAttesterParams.phase_releases`): `lanes` · ⌈(T + `ingress_buckets` − 1) / (D + 1)⌉, as a release is at most
`ingress_buckets` − 1 buckets after its connection's admission and the connections admitted in any D + 1 buckets in
a row are live together, at most `lanes` of them. -/
def Schedule.executionPhaseReleases (s : Schedule) : Nat :=
  s.lanes * divCeil (s.window + s.ingressBuckets - 1) (s.delay + 1)

/-- Whether the lengths rows end every execution phase of a run, its last included: their form 2
(`NetworkAttesterParams.closes`). -/
def Schedule.closes (s : Schedule) : Bool := s.lengths == some 2

/-- Whether the schedule registers connection rows (`connection_rows`): its lengths rows, which a schedule with
content rows has too, as one with request words has content rows. Each is asked, since a `Schedule` the parse did not
build may have one without the other. -/
def Schedule.connectionRows (s : Schedule) : Bool := s.lengths.isSome || s.content.isSome || s.requestWords

/-- The rows form's cells over `windows` of a link's execution phases (`NetworkAttesterParams.rows_cells`): `egressCells` for
each of the `lanes · windows · window` connections the network attester can carry, as at most `lanes` are live in a
bucket and each is live in its admission bucket. -/
def Schedule.rowsCells (s : Schedule) (windows : Nat) : Array Nat :=
  Array.replicate (s.lanes * (windows * s.window)) s.egressCells

/-- **The rows form's frames read back** (`NetTiming.RowsFormat.Wf` at the network attester's frames,
`link_frame.encode_frame`: kind u8, ordinal u32, payload length u32, payload, zero fill): a cell's `B` bytes are under
2^32, so its payload's length is a u32, and the `lanes · windows · window` connections of a period of `windows` of a
link's execution phases, the k-th at ordinal k + 1 (0 is a refusal's), have distinct u32 ordinals, so no two frames
encode alike. -/
def rowsWf (s : Schedule) (windows : Nat) : Bool :=
  s.cellBytes < 2 ^ 32 && s.lanes * (windows * s.window) < 2 ^ 32

def rowsWfRefusal (s : Schedule) (windows : Nat) : String :=
  s!"rows format: a frame's ordinal and payload length are u32s, so the lanes · windows · window = \
    {s.lanes * (windows * s.window)} connections and a cell's {s.cellBytes} bytes must each be under 2^32, or two \
    frames could encode alike (rowsWf)"

/-- The scan's terms, read off its Program: its monitoring period (`T`, the window's buckets), its own free inputs
(none: it registers the rows, and its `prog` is a compute audit's output, `progTied`, whose free inputs an execution
phase's term counts, `tiedAudits`), the bits one ingress and one egress instance commit (its return: the outputs other
units read or the client receives, and its verdict), its slot entries' bits (`E_w`: `lanes · T / every` entries of
its `prog` ports' type), all a tied audit's incorrect proof blocks can set, and its state's full width (`start`: the
start unit's `sin`, the bits a period's first start that is not the run's first is charged, `scanStartCheck`). -/
structure Scan where
  steps : Nat
  free : Nat
  ingress : Nat
  egress : Nat
  entries : Nat
  start : Nat

/-- Which run the execution phases are: the rows form's, under one schedule (and its digest), read by the scan whose terms
are the last. -/
inductive Form where
  | rows (schedule : Schedule) (digest : ByteArray) (scan : Scan)

/-- The response cap of the form's schedule, its `maxTokensCap`. -/
def Form.maxTokensCap : Form → Option Nat
  | .rows s .. => s.maxTokensCap

/-- One entry of the link's network attester's listed keys (`"keys"`), a network attester session's: its LMS public key, its
session, and the start of its stretch of execution phases, in the numbers the period's `first` is in. -/
structure Entry where
  key : ByteArray
  session : ByteArray
  start : Nat
  deriving BEq

structure Params where
  audits : Array Audit
  cells : Array Nat
  first : Nat
  jitter : Nat
  link : String
  lms : Lms.Params
  entries : Array Entry
  program : String
  form : Form
  windows : Nat
  /-- The rows form's class circuit, its window circuit's Program's SHA-512, which every tied compute audit runs
  (`held`). -/
  circuit : Option String := none

/-- The one-stage window records of the compute audits and of the execution phases, in order. -/
structure Commitment where
  audits : Array Json
  windows : Array Json
  bytes : ByteArray

structure BucketSchedule where
  buckets : Nat
  slots : Nat
  connectionSlots : Nat
  firstBucket : Nat
  deriving BEq

/-- An execution phase's rows in one direction, as `network_certifier_rows.registration` writes them. -/
structure Rows where
  format : String
  value : String
  link : String
  direction : String
  window : Nat
  parameters : String
  cellBytes : Nat
  rows : Nat
  words : Nat
  wordBits : Nat
  rowSchema : String
  domain : String
  wiring : ByteArray
  root : ByteArray
  layout : BucketSchedule
  key : ByteArray
  digest : ByteArray
  signature : ByteArray

structure ExecutionPhase where
  answer : Core.Answer
  certificates : Array Certificate
  registrations : Array Rows

/-- A network attester session's signed start as the evidence shows it (`certificate.CertifiedStart`, as the run's
`deployment.json` holds it): where the session starts on its key's life, in milliseconds on the network
attester's clock; its ordinal, the number of the key's sessions its device state started before it; the parameters
digest of the schedule it runs under; and the certificate, which signs their start digest (`startDigest`). -/
structure Start where
  baseMs : Nat
  ordinal : Nat
  schedule : ByteArray
  certificate : Certificate

structure Evidence where
  audits : Array Core.Answer
  windows : Array ExecutionPhase
  /-- The listed entries' sessions' signed starts, in the entries' order, one an entry. -/
  starts : Array Start

/-- What an accepted audit establishes: its proof blocks, its sample, its δ, its bound on incorrect proof blocks at
that δ, and the sample's design (`design`), which bounds them at any other δ (`Facts.readAt`). -/
structure Facts where
  units : Nat
  drawn : Nat
  delta : Core.Ratio
  wrong : Nat
  /-- The sample's design, the registered one (`factsOf`); `subset 0`, none sampled, where none is read, at which
  every unit may be incorrect at any δ. -/
  design : Core.SamplingDesign := .subset 0

/-! ## Decoding -/

def isDigest (s : String) : Bool := s.length = 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')


def rowsParamsKeys : List String :=
  ["audits", "cells", "circuit", "first", "jitter", "keys", "link", "lms", "program", "scan", "schedule", "windows"]

/-- The keys every schedule has (`network_certifier.INTEGER_FIELDS`). -/
def scheduleKeys : List String :=
  ["bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells", "lanes", "queue",
    "slots", "window"]

/-- The keys of a schedule that sets the optional keys `set` holds, in the order an object lists its keys. -/
def scheduleKeysWith (set : String → Bool) : List String :=
  ["bucket_ms", "cell_bytes", "content", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells", "lanes",
    "lengths", "max_tokens_cap", "queue", "request_words", "slots", "window"].filter fun k =>
      !optionalScheduleKeys.contains k || set k

/-- What `content_rows.classes` refuses of a schedule's content rows' classes, or none: a schedule with content rows
has lengths rows; it lists 1 to `mostClasses` classes, or to `mostClassesWords` with request words, each (P, O, rows)
under the lengths rows' form 1 and (P, O) under form 2; P and O are each a class's size, O at most `mostIds` and P
at most the bytes of a request, a prompt id at least one (`ingress_cells` · a cell's payload); and under form 1 a
class's tables have 1 to T / `every` · `strideReleases` rows, the lengths table's slots, since an execution phase
releases no more. -/
def classesRefusal (s : Schedule) (classes : Array (Array Nat)) : Option String :=
  match s.lengths with
  | none => some "content: the content rows' tables are listed in the connection rows' head, so a schedule with \
      content has lengths rows (lengths is 1 or 2)"
  | some form =>
    let slots := s.window / s.every * s.strideReleases
    let most := if s.requestWords then mostClassesWords else mostClasses
    if classes.isEmpty || most < classes.size then
      some s!"content: 1 to {most} classes, not {classes.size}"
    else (classes.zipIdx.findSome? fun (c, i) =>
      let p := c.getD 0 0
      let o := c.getD 1 0
      if c.size != (if form == 2 then 2 else 3) then
        some s!"class {i} is (P, O) under the lengths rows' form 2 and (P, O, rows) under form 1, not {c.size} numbers"
      else if !(classSize p && classSize o) then
        some s!"class {i}: its sizes of prompt ids and of ids released are each a power of two of at least 2"
      else if mostIds < o then some s!"class {i}: a response row is at most {mostIds} ids, not O = {o}"
      else if s.ingressCells * (s.cellBytes - cellHeader) < p then
        some s!"class {i} has P = {p} prompt ids, and a request of the schedule is at most \
          {s.ingressCells * (s.cellBytes - cellHeader)} bytes (ingress_cells · a cell's payload), each id at least one"
      else if form != 2 && !(1 ≤ c.getD 2 0 && c.getD 2 0 ≤ slots) then
        some s!"class {i} has 1 to {slots} rows, the lengths table's slots, since an execution phase releases no more"
      else none).map ("content " ++ ·)

def scheduleOf (j : Json) : V Schedule := do
  exactly j (scheduleKeysWith (Flock.J.has j)) "schedule"
  let pos : String → V Nat := fun k => do
    let x ← Flock.J.nat j k "schedule"
    unless 1 ≤ x do throw s!"schedule: {k} is a positive integer"
    pure x
  let opt : String → V (Option Nat) := fun k => if Flock.J.has j k then some <$> pos k else pure none
  let content ← if Flock.J.has j "content" then do
      let classes ← (← Flock.J.arr j "content" "schedule").mapM fun c => do
        (← Flock.J.asArr c "schedule.content").mapM (Flock.J.asNat · "schedule.content")
      pure (some classes)
    else pure none
  let requestWords ← if Flock.J.has j "request_words" then do
      unless (← Flock.J.bool j "request_words" "schedule") do
        throw "schedule: request_words is true, for a schedule with request words, or unset"
      pure true
    else pure false
  let s : Schedule :=
    { bucketMs := ← pos "bucket_ms", cellBytes := ← pos "cell_bytes", delay := ← pos "delay",
      egressCells := ← pos "egress_cells", every := ← pos "every", ingressBuckets := ← pos "ingress_buckets",
      ingressCells := ← pos "ingress_cells", lanes := ← pos "lanes", queue := ← pos "queue", slots := ← pos "slots",
      window := ← pos "window", maxTokensCap := ← opt "max_tokens_cap", lengths := ← opt "lengths", content,
      requestWords }
  unless cellHeader + 1 < s.cellBytes do throw s!"schedule: a cell holds its {cellHeader}-byte header and a byte"
  unless s.cellBytes < 2 ^ 32 ∧ s.allocation < 2 ^ 32 ∧ s.window < 2 ^ 32 do
    throw "schedule: B, S = lanes · slots and T = window are u32s"
  unless s.window % s.every == 0 do throw "schedule: every divides the window: a window is whole timesteps"
  if let some cap := s.maxTokensCap then
    unless cap < 2 ^ 32 ∧ cap < s.egressCells do
      throw "schedule: max_tokens_cap is a u32 below egress_cells: a capped response's END is its egress cell \
        max_tokens"
    unless maxTokensEnd + cellHeader ≤ s.cellBytes do
      throw s!"schedule: under a response cap a cell is at least {maxTokensEnd + cellHeader} bytes, so it holds an id \
        request's max_tokens"
  if let some form := s.lengths then
    unless form == 1 ∨ form == 2 do throw s!"schedule: lengths is 1 or 2, the lengths rows' forms, not {form}"
    unless s.ingressBuckets ≤ s.delay do
      throw s!"schedule: with lengths rows ingress_buckets ({s.ingressBuckets}) is at most the delay ({s.delay}): a \
        released connection holds its lane from its release"
    unless s.span ≤ s.window ∧ s.egressCells ≤ 2 ^ 16 do
      throw s!"schedule: with lengths rows a connection ends within a window of its admission and its egress cells \
        count in a u16: delay + (egress_cells − 1) / slots ({s.span}) is at most the window ({s.window}) and \
        egress_cells at most {2 ^ 16}"
    unless firstPromptEnd + cellHeader ≤ s.cellBytes do
      throw s!"schedule: with lengths rows a cell is at least {firstPromptEnd + cellHeader} bytes, so its first holds \
        an id request's head and first prompt id"
    unless s.strideReleases ≤ mostSlots do
      throw s!"schedule: a lengths row is at most {mostSlots} slots, and this schedule's slot stride releases up to \
        {s.strideReleases} requests, a slot each"
  if s.requestWords then
    unless s.content.isSome ∧ s.closes ∧ s.maxTokensCap.isSome do
      throw "schedule: a words row is a release's in its class, one for each the class holds, and its max_tokens the \
        one its response is held to, so a schedule with request words has content rows under the lengths rows' form \
        2 and max_tokens_cap"
  if let some classes := s.content then
    if let some why := classesRefusal s classes then throw s!"schedule: {why}"
  pure s

/-- `NetworkAttesterParams.digest`: the schedule's canonical JSON, with its egress form, the cell kinds and the fail reasons,
tagged. A schedule without a response cap has neither the cap nor `capReason` in it, so its digest is the one it had
before the cap was a parameter. So with the lengths rows' form, the content rows' classes and the request words: a
schedule without one has nothing of it in its digest. -/
def Schedule.digest (s : Schedule) : V ByteArray :=
  Identity.identityDigestSha512 scheduleTag (Json.mkObj ([
    ("bucket_ms", toJson s.bucketMs), ("cell_bytes", toJson s.cellBytes), ("delay", toJson s.delay),
    ("egress", toJson egressForm), ("egress_cells", toJson s.egressCells), ("every", toJson s.every),
    ("ingress_buckets", toJson s.ingressBuckets),
    ("ingress_cells", toJson s.ingressCells), ("lanes", toJson s.lanes), ("queue", toJson s.queue),
    ("slots", toJson s.slots), ("window", toJson s.window), ("kinds", toJson cellKinds),
    ("reasons", toJson s.reasons)] ++ (s.maxTokensCap.map fun cap => ("max_tokens_cap", toJson cap)).toList ++
    (s.lengths.map fun form => ("lengths", toJson form)).toList ++
    (s.content.map fun classes => ("content", toJson classes)).toList ++
    (if s.requestWords then [("request_words", toJson true)] else [])))

/-- The scan's templates' parameters in the generator form: the start unit's (`ingress_scan.IngressScanStart`, the state
the execution phase's scan starts from, which it copies), each ingress bucket's (`ingress_scan.IngressScanBucket`) and
the egress bucket's (`egress.EgressBucket`), which computes its row and returns it. An ingress bucket's `prog<j>` is
lane
`j`'s slot entry, one a lane. -/
def progParams (lanes : Nat) : Array String := (Array.range lanes).map (s!"prog{·}")
def startParams : Array String := #["sin"]
def ingressParams (lanes : Nat) : Array String := #["row", "sin"] ++ progParams lanes
def egressParams : Array String := #["state"]

/-- A type's bits: its leaves' widths summed. -/
def bitsOf (t : Flock.Program.Ty) : Nat := t.widths.foldl (· + ·) 0

/-- Parameters' bits: their types' summed. -/
def paramBits (ps : Array (String × Flock.Program.Ty)) : Nat := ps.foldl (fun n p => n + bitsOf p.2) 0

/-- The collections a part of references names: its space and index, or each item's of an explicit part. -/
def partNames : Flock.Program.Part → Array (Bool × Nat)
  | .affine p i _ _ => #[(p, i)]
  | .strided p i _ _ _ _ _ => #[(p, i)]
  | .explicit it => it.map fun r => (r.p, r.i)

/-- Per Definition, whether every leaf it returns is a gate it computes, never a parameter's leaf passed through: each
part of its return names nodes of its body, none a scan, each calling a primitive or a Definition decoded before it of
which the same holds. Once per Definition, at the cost of its return's parts. -/
def ownReturns (P : Flock.Program.Prog) : Array Bool :=
  (List.range P.defs.size).foldl (fun a d =>
    let defn := P.defs[d]!
    a.push (!defn.prim && defn.body.ret.parts.all fun q => (partNames q).all fun (p, k) =>
      !p && match defn.body.nodes[k]? with
        | some n => n.form != .scan && ((P.defs[n.fn]!).prim || (n.fn < d && a[n.fn]!))
        | none => false)) #[]

/-- The root's template nodes, in order: those calling a composite. -/
def templateNodes (P : Flock.Program.Prog) : List Nat :=
  let nodes := (P.defs[P.root]!).body.nodes
  (List.range nodes.size).filter fun k => match P.defs[nodes[k]!.fn]? with
    | some d => !d.prim
    | none => false

/-- The root node a template's argument reads when the argument is that input node's leaves in order from its first
(one affine run), as the generator form binds each parameter to an input array of its own. -/
def inputArg (P : Flock.Program.Prog) (a : Flock.Program.Arg) : Option Nat :=
  match a.seq.parts.toList with
  | [.affine false k 0 _] => match (P.defs[P.root]!).body.nodes[k]? with
    | some n => if InferenceOnlyVerifier.Desc.isInputNode P n then some k else none
    | none => none
  | _ => none

/-- Whether the arguments `own` (the templates' row, state and egress arguments) read input nodes of their own: their
input nodes (`inputArg`) are pairwise distinct, and none is one of `pks`, the slot entries' input nodes. At the cost of
the arguments. -/
def argsApart (P : Flock.Program.Prog) (own : List Flock.Program.Arg) (pks : List Nat) : Bool :=
  let ks := own.map (inputArg P)
  decide ks.Nodup && pks.all fun pk => !ks.contains (some pk)

/-- Whether the root's input nodes hold what the scan reads and no other leaf: each argument of `own` is one run over
all its node's leaves (as many as the node has members), each node of `pks` has `entries` leaves, and every input node
of the root is one of `pks` or an argument's of `own`. At the cost of the root's nodes, each against the arguments. -/
def inputsRead (P : Flock.Program.Prog) (own : List Flock.Program.Arg) (pks : List Nat) (entries : Nat) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ks := own.map (inputArg P)
  own.all (fun a => match a.seq.parts.toList with
    | [.affine false k 0 c] => nodes[k]?.map (·.members) == some c
    | _ => false) &&
  pks.all (fun pk => nodes[pk]?.map (·.members) == some entries) &&
  (List.range nodes.size).all fun k => !InferenceOnlyVerifier.Desc.isInputNode P nodes[k]! || pks.contains k || ks.contains (some k)

/-- Whether two types have as many leaves, each of one width, the same in both (`Flock.Program.Widths.homo`): what a
tie of two rows of gates, leaf by leaf, needs of their types. From the types' structure, no walk over their leaves. -/
def sameLeaves (a b : Flock.Program.Ty) : Bool :=
  a.leaves == b.leaves && (Flock.Program.Widths.homo a).isSome &&
    Flock.Program.Widths.homo a == Flock.Program.Widths.homo b

/-- Whether `ret` is the generator form's ingress return, over the leaves the scan's two ties equate: a tuple of
exactly a u16, the state the next bucket reads and the view the bucket's egress unit reads, in that order. The state
has the leaves of `sin`, the ingress template's own parameter of that name, and the view the leaves of `state`, the
egress template's parameter (`sameLeaves`: a view of bits is read as u16 words, so the types themselves differ). -/
def ingressReturn (ret sin state : Flock.Program.Ty) : Bool :=
  match ret with
  | .tuple fs => fs.size == 3 && fs[0]! == .array outBits (.value 1) && sameLeaves fs[1]! sin &&
      sameLeaves fs[2]! state
  | _ => false

/-- Whether `ret` and `own`, the start template's return and its own parameter, are a state's, over the leaves its
two ties equate: `ret` a tuple of exactly one row, and that row and `own` each of the leaves of `sin`, an ingress
bucket's parameter of that name. The row it returns is the one bucket 0's `sin` reads, and its parameter is read
where an ingress bucket's returned state is committed (the last bucket's of the execution phase before), so each tie
equates rows of one length. -/
def startReturn (ret own sin : Flock.Program.Ty) : Bool :=
  match ret with
  | .tuple fs => fs.size == 1 && sameLeaves fs[0]! sin && sameLeaves own sin
  | _ => false

/-- Refuses with `why` unless `b`: one step of a `do` block, so what its acceptance gives is read off it step by step
(`Properties.NetworkScan`). -/
def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- `o`'s value, refusing with `why` where there is none, as one step. -/
def needSome {α : Type} (o : Option α) (why : String) : V α := match o with
  | some a => pure a
  | none => throw why

/-- The generator form's templates before the egress template's: the start unit's, then the ingress buckets'. -/
def beforeEgress : List (Flock.Program.Node × Flock.Program.Defn) →
    V ((Flock.Program.Node × Flock.Program.Defn) × (Flock.Program.Node × Flock.Program.Defn))
  | [st, b] => pure (st, b)
  | _ => throw "scan: not the generator form's templates: the start unit's, the ingress buckets' and the egress \
    bucket's"

/-- The slot entry's Definition (`io_shell.SlotEntry_v1`): one unit a due, occupied slot's Program, committing the
entry the scan's ingress reads at that slot (its `o0`). -/
def slotEntryId : String := "SlotEntry_v1"

/-- The slot's last position (`SelectEntry_v1`, pinned by `selectEntryPins`): one unit a due, occupied slot's Program,
whose select and collector end in `SlotEntry_v1`, so its `o0` is the slot's entry. -/
def selectEntryId : String := "SelectEntry_v1"

/-- The Definitions whose `o0` is a slot's entry: `SlotEntry_v1` and `SelectEntry_v1`. -/
def entryIds : List String := [slotEntryId, selectEntryId]

/-- The slot entry's Definitions the check takes, by closure digest (`closureDigest`): `io_shell.SlotEntry_v1` at
its circuit-check rows, `SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}` and `SlotEntry_v1{KO=64,IDB=17,NP=1024,STOP=[128001]}`
(a Llama-3 tokenizer's 17-bit ids and end-of-text id), in that order. A template under `SlotEntry_v1`'s name with
another body has neither digest, so its entries are no slot entries (`wired`). These are the rows a window circuit's
slot entries take as a template of their own; its `SlotEntry_v1:4` row (KO 4, 32 prompt ids) is no pin, since at KO 4
a slot's entry is a `SelectEntry_v1` unit's (`selectEntryPins`). A new body of `SlotEntry_v1`, or a binding a window
circuit takes as its slot entries' template, is pinned here. -/
def slotEntryPins : List String :=
  ["8c8a033047ec49110114023204a9c265720ae9d1b63963c51a98e14f0314ebd3\
    7b741b90d6a374aeb681a1a5fe364580b68ff60cf5514fdf57fdaff88b36de30",
   "2559f1c8e260aac6e8755c317cde07f990fcd3619dd394010d367a3b78156c2f\
    24b44567891a9ea2efb2f8a6830c9f9c957c14dc65041f8dd77bfd8d50c5b7ff"]

#guard slotEntryPins.all (·.length == 128)

/-- The slot's last position's Definitions the check takes, by closure digest (`closureDigest`, which covers the
`SlotEntry_v1` it calls): `SelectEntry_v1` at `SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4,STOP=[]}`,
`SelectEntry_v1{V=128256,C=16,KO=4,FIRST=false,IDB=17,NP=32,STOP=[128001]}` and
`SelectEntry_v1{V=128256,C=16,KO=64,FIRST=false,IDB=17,NP=1024,STOP=[128001]}`, in that order, each calling the
`SlotEntry_v1` that writes the scan's read bit: `io_shell.SelectEntry_v1` at its circuit-check rows. A new binding or
body is pinned here. -/
def selectEntryPins : List String :=
  ["f817250e0ac1d5028ea992bbe0c1ab12925bc1b709f020fdc54a57a0ebc0c9dc\
    bd6513d2bf72aa243c499833d1173fb14236e85bb27b4d03de88df93344007d2",
   "236d569741e3fe2026aaa65be2077b4da82d495a5b35322428051e4e07ca556b\
    b44d1c5f59f2aef22d7c4c5b563c3819569add04ed3d2d4d57563ad471cf71c4",
   "5460098589051bb756c5d7781e1727495952c296826b29315ec269b492ca3cea\
    bc394ee273a2dae92ffb8eef31d8d22e83208a9e280dd80e39be9073dfc7a68a"]

#guard selectEntryPins.all (·.length == 128)

/-- The entries' Definitions the check takes, by closure digest: `slotEntryPins` and `selectEntryPins`. -/
def entryPins : List String := slotEntryPins ++ selectEntryPins

/-- A compute audit's layout terms and its Program, from the Program's descriptor (`Flock.Program.decode` of its
canonical bytes): its units are its root's template instances, each committing its template's return; a root with no
template is refused. Its slot entries' templates are its root's template nodes whose Definition is `SlotEntry_v1` or
`SelectEntry_v1` (`entryIds`), each with its closure digest (`closureDigest`). -/
def auditOf (a : Json) (ctx : String) : V Audit := do
  exactly a ["free", "program", "steps"] ctx
  let pj ← Flock.J.get a "program" ctx
  let prog ← match Flock.Program.decode (← Flock.canon pj).toUTF8 with
    | .ok prog => pure prog
    | .error e => throw s!"{ctx}: program: {e}"
  let some root := prog.defs[prog.root]? | throw s!"{ctx}: program: no root"
  let templates := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if d.prim then none else some x.members
    | none => none
  if templates.isEmpty then throw s!"{ctx}: program: no template"
  let slots := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if !d.prim && entryIds.contains d.id then some (d.key, x.members) else none
    | none => none
  let entries ← slots.mapM fun (k, n) => do
    pure (k, n, ← (closureDigest pj k).mapError (s!"{ctx}: " ++ ·))
  pure { free := ← Flock.J.nat a "free" ctx, steps := ← Flock.J.nat a "steps" ctx, program := Flock.hexOf prog.sha512,
         units := templates.foldl (· + ·) 0, entries }

/-- Whether a batch node of ingress buckets `t0, t0 + 1, …` reads the slot entries at their slots: for each lane `j`
and member `m` (bucket `t0 + m`), its argument `o + j` (`prog<j>`) is row `⌊(t0 + m) / every⌋` of input node `ks[j]`,
lane `j`'s slot entries, `E` leaves a row, leaf by leaf (`Flock.Program.Arg.memberRef`). That row is lane `j`'s slot of
the boundary whose Programs bucket `t0 + m` reads, leaf `⌊(t0 + m) / every⌋` of the chunk `wired` holds lane `j`'s
tree to. -/
def readsEntries (x : Flock.Program.Node) (o t0 every E : Nat) (ks : List Nat) : Bool :=
  ks.zipIdx.all fun (k, j) => match x.args[o + j]? with
    | some a => (List.range x.n).all fun m => (List.range E).all fun l =>
        a.memberRef m l == { p := false, i := k, j := (t0 + m) / every * E + l }
    | none => false

/-- A batch of 3 members whose argument `1 + j` reads leaf `rows[m]` of node `k` at member `m`, one leaf a member, for
lane `j`'s `(k, rows)`. -/
def entriesProbe (lanes : List (Nat × List Nat)) : Flock.Program.Node :=
  let arg := fun (k : Nat) (rs : List Nat) => Flock.Program.Arg.mk (.value 1)
    (.ofParts (rs.toArray.map fun r => Flock.Program.Part.affine false k r 1)) (some (rs.length, 1))
  Flock.Program.Node.mk .batch 0 (#[arg 0 [0, 0, 0]] ++ (lanes.map fun (k, rs) => arg k rs).toArray) (.value 1) #[] 3 0

#guard readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [7] &&
  readsEntries (entriesProbe [(7, [0, 1, 1]), (8, [0, 1, 1])]) 1 1 2 1 [7, 8] &&
  !readsEntries (entriesProbe [(7, [0, 1, 0])]) 1 1 2 1 [7] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [7, 8] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [6] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1]), (8, [0, 1, 1])]) 1 1 2 1 [8, 7] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1]), (7, [2, 3, 3])]) 1 1 2 1 [7, 7]

/-- The scan's terms from its Program's descriptor (`Flock.Program.decode` of its canonical bytes), which must be the
registered Program (`program`, its SHA-512) and the generator form of the schedule's execution phase: its root batches
one instance of the start template, whose one parameter is `sin` and which returns the state alone, `T` of the ingress
buckets' template, each one's `row` the schedule's row (`S · B` bytes), then `T` of the egress template, which returns
a u16 and the schedule's row, and every input gate is a template's parameter. The types are the Boolean profile's, as
the scan's Definitions have them: a byte 8 bits, a u16 16. An egress template with a row among its parameters, the
checker form, is refused first, by name: there the egress row is an input gate, which `NetTiming.BoundaryGates.Fits`
refuses, and `K` would count the free inputs of a Program that only compares its egress rows.

Then what `NetTiming.BoundaryGates.Fits` asks of the layout the templates give (`Properties.NetworkScan.scanLayout`), read off
the description, at the cost of its nodes and returns: the Program is laid out as Flock's decoder lays one out and
within what its resolution walks (`InferenceOnlyCheck.Desc`'s `rootLaidOut`, `inputsAtRoot` and `walkable`); the root returns
every egress leaf (one affine run from the egress node's first leaf over all `T` instances'); the egress template
returns only gates it computes (`ownReturns`: no parameter's leaf passed through, so no egress gate is an input); the
ingress template's row, and every template's row, state and egress argument, is one input node's leaves (`inputArg`),
so each ingress gate is an input and a template reads only input gates outside its own; no egress template's argument
is the ingress rows' input node, so no egress gate depends on an ingress gate; and the Program is topological
(`Prog.orderViolation`, as `InferenceOnlyCheck.Desc.evaluate` checks it), so a gate reads only gates before it.

The ingress template's entry ports `prog<j>` all have one type, a slot entry's, and lane `j`'s read one input node of
its own, distinct from every other lane's and from the rows, lane `j`'s slot entries (`T / every` rows each, `lanes ·
T / every` counted once among the input gates): bucket `t`'s `prog<j>` is its row `⌊t / every⌋` (`readsEntries`), the
leaf the period check's `wired` holds to chunk `j` of an audit's `SlotEntry_v1` or `SelectEntry_v1` entries port;
their bits are the scan's `entries`.

The scan's ties equate rows of one length: a bucket's `sin` with the state the bucket before returned, bucket 0's
with the state the start unit returned, an egress bucket's `state` with the view its ingress bucket returned, and the
start unit's `sin` with the state the execution phase before's last bucket returned. So the ingress template returns
exactly a u16, a state of `sin`'s leaves and a view of the egress template's `state`'s leaves (`ingressReturn`), and
the start template takes and returns exactly a state of those leaves (`startReturn`); a return of another shape is
refused. A start unit so commits no more bits than an ingress bucket does, so the wider of an ingress and an egress
instance's bits (`Scan.terms`) bounds what any incorrect proof block of the scan commits.

An input gate is read as one thing: the row, state and egress arguments read input nodes of their own, none a lane's
slot entries' (`argsApart`). The input gates are counted as a sum of bits against the parameters' and the slot
entries', so two arguments on one node would leave that sum room for an input node nothing reads, and a state argument
on a lane's slot entries' node would make a slot entry a state gate.

And every input gate is read: each of those arguments is all its node's leaves, each lane's slot entries' node holds
its `T / every` entries and no other leaf, and the root has no other input node (`inputsRead`). That sum counts bits
and this counts leaves, node by node, so the scan's input gates are read off the check itself, with no appeal to the
widths the decoder holds a reference to. A slot entry's leaves are at most its bits (a leaf is at least a bit wide,
which the decoder gives and this holds), so the entries' bits bound the entries' gates.

The scan's own free inputs are none: it registers the rows, its `prog` is its tied audits' output, each bucket's
`sin` is the state the unit before it returned, and the start unit's `sin` is the record's carried start, which the
check holds (`scanStartCheck`). -/
def scanOf (j : Json) (program : String) (s : Schedule) : V Scan := do
  let prog ← (Flock.Program.decode (← Flock.canon j).toUTF8).mapError fun e => s!"scan: {e}"
  need (Flock.hexOf prog.sha512 == program) "scan: its SHA-512 is not the registered Program's (program)"
  let root ← needSome prog.defs[prog.root]? "scan: no root"
  let templates := (templateNodes prog).toArray.map fun k => (root.body.nodes[k]!, prog.defs[root.body.nodes[k]!.fn]!)
  let T := s.window
  let row : Flock.Program.Ty := .array (s.allocation * s.cellBytes) (.array 8 (.value 1))
  let (xe, de) ← needSome templates.back? "scan: no template"
  need (!de.params.any (fun p => p.1 == "row" || p.2 == row)) "scan: the egress template takes its row as an input \
    (the checker form): no egress gate is an input (NetTiming.BoundaryGates.Fits)"
  let ((xs, ds), (xi, di)) ← beforeEgress templates.pop.toList
  need (ds.params.map (·.1) == startParams && di.params.map (·.1) == ingressParams s.lanes &&
      de.params.map (·.1) == egressParams)
    s!"scan: the templates' parameters are not {startParams}, {ingressParams s.lanes} and {egressParams}"
  need (xs.form == .batch && xs.n == 1 && xi.form == .batch && xi.n == T && xe.form == .batch && xe.n == T)
    s!"scan: not the start unit, {T} ingress buckets and {T} egress buckets, the execution phase's"
  need (di.params[0]!.2 == row) "scan: the ingress template's row is not the schedule's"
  let ety := di.params[2]!.2
  need ((di.params.toList.drop 2).all (·.2 == ety)) "scan: the ingress template's entry ports' types differ"
  need (de.ret == .tuple #[.array outBits (.value 1), row])
    "scan: the egress template doesn't return a u16 and the schedule's row"
  need (ingressReturn di.ret di.params[1]!.2 de.params[0]!.2 && startReturn ds.ret ds.params[0]!.2 di.params[1]!.2)
    "scan: the ingress template doesn't return a u16, a state of its sin's leaves and a view of an egress bucket's \
      state's leaves, or the start template doesn't take and return a state of those leaves, so a tie of the scan's \
      state or view would equate rows of two lengths (NetworkVerifier.Period.ingressReturn, startReturn)"
  let inputs := root.body.nodes.foldl (fun n x => n + match prog.defs[x.fn]? with
    | some d => ((Flock.Program.inputWidth? d.key).map (· * x.members)).getD 0
    | none => 0) 0
  let nb := T / s.every
  need (inputs == paramBits ds.params + T * paramBits (di.params.extract 0 2) + T * paramBits de.params +
      s.lanes * nb * bitsOf ety)
    "scan: an input gate that is no template's row, state or egress parameter, nor one of the slot entries"
  need (InferenceOnlyVerifier.Desc.rootLaidOut prog && InferenceOnlyVerifier.Desc.inputsAtRoot prog && InferenceOnlyVerifier.Desc.walkable prog)
    "scan: not laid out as Flock's decoder lays a Program out, or past what its resolution walks \
      (NciCheck.Desc.rootLaidOut, inputsAtRoot and walkable)"
  let ke := (templateNodes prog).getLast?.getD 0
  need (root.body.ret.parts.any fun q => match q with
      | .affine false k 0 c => k == ke && T * de.outLeaves ≤ c
      | _ => false)
    "scan: the root doesn't return every egress leaf, so an egress gate would be one the Program doesn't return \
      (NetTiming.BoundaryGates.Fits)"
  need ((ownReturns prog)[xe.fn]!) "scan: the egress template returns a leaf it doesn't compute (a parameter's, passed \
    through), so an egress gate would be an input (NetTiming.BoundaryGates.Fits)"
  let rows := xi.args[0]?.bind (inputArg prog)
  need rows.isSome "scan: the ingress template's row is not one input node's leaves, so an ingress gate would be no \
    input (NetTiming.BoundaryGates.Fits)"
  need (xs.args.size == 1 && xi.args.size == 2 + s.lanes && xs.args.all (fun a => (inputArg prog a).isSome) &&
      (xi.args.extract 0 2).all (fun a => (inputArg prog a).isSome) && xe.args.all fun a => (inputArg prog a).isSome)
    "scan: a template's row, state or egress argument is not one input node's leaves"
  need (xe.args.all fun a => inputArg prog a != rows) "scan: an egress template's argument is the ingress rows' \
    input node, so an egress gate would depend on the ingress (NetTiming.BoundaryGates.Fits)"
  let a0 ← needSome ((List.range s.lanes).mapM fun j => xi.args[2 + j]?) "scan: bucket 0 has no entry port a lane"
  let pks := a0.map fun a => (a.memberRef 0 0).i
  need (a0.all (fun a => !(a.memberRef 0 0).p) && decide pks.Nodup && pks.all fun pk =>
      rows != some pk && match root.body.nodes[pk]? with
        | some n => InferenceOnlyVerifier.Desc.isInputNode prog n
        | none => false)
    "scan: bucket 0's prog<j> reads no input node of lane j's own, its slot entries', distinct from every other lane's"
  need (readsEntries xi 2 0 s.every ety.leaves pks)
    s!"scan: bucket t's prog<j> is not row ⌊t / {s.every}⌋ of lane j's slot entries' input node, its slot of the \
      boundary bucket t reads (NetworkVerifier.Period.readsEntries)"
  let own := (xs.args.extract 0 1 ++ xi.args.extract 0 2 ++ xe.args).toList
  need (argsApart prog own pks) "scan: two of the templates' row, state and egress arguments read one input node, or \
    one reads a lane's slot entries': an input node would be left that nothing reads, or a slot entry would be a \
    state gate (NetworkVerifier.Period.argsApart)"
  need (inputsRead prog own pks (nb * ety.leaves) && ety.leaves ≤ bitsOf ety)
    s!"scan: an input node with a leaf nothing reads: a row, state or egress argument that is not all its input \
      node's leaves, a lane's slot entries' node of other leaves than its {nb} entries' (each of at least a bit), \
      or an input node that is neither's (NetworkVerifier.Period.inputsRead)"
  need prog.orderViolation.isNone "scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program \
    is not topological, so a gate's operands could be gates after it"
  pure { steps := T, free := 0, ingress := bitsOf di.ret, egress := bitsOf de.ret,
         entries := s.lanes * nb * bitsOf ety, start := bitsOf ds.params[0]!.2 }

/-- The link's network attester's keys, as the parameters list them (`"keys"`): an entry for each certifier session,
its LMS public key under `lms`, its session and the start of its stretch of execution phases. At least one entry, and
the starts strictly increasing, so the stretches are in order and no two hold one execution phase: entry `i`'s runs from
its start up to the next entry's start, and the last entry's from its start on (`entryAt`). No two entries with one key
identifier (`I`, a key's first 16 bytes) and one session, whose execution phase `n` would be two of the period's. A
key's restart: the period reads each session's signed start (`startChecks`), so a key may be listed in several
sessions, each held to the key's first listed one, and an identifier still names one key. -/
def certifierEntriesOf (j : Json) (lms : Lms.Params) : V (Array Entry) := do
  let entries ← (← Flock.J.arr j "keys" "the object").mapM fun e => do
    match e with
    | .obj _ => exactly e ["key", "session", "start"] "keys"
    | _ => throw "keys: an entry is not an object: write each as {\"key\": hex(I ‖ T[1]), \"session\": hex, \"start\": a}"
    let key ← hexAt e "key" "keys"
    unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
    let session ← hexAt e "session" "keys"
    unless session.size = attachmentBytes do throw s!"a certifier session id is {attachmentBytes} bytes, not {session.size}"
    pure ({ key, session, start := ← Flock.J.nat e "start" "keys" } : Entry)
  unless 1 ≤ entries.size do throw "keys: no key: list the certifier's entries, at least one"
  for (x, y) in entries.zip (entries.extract 1) do
    if y.start < x.start then
      throw s!"keys: out of order: an entry starts at {y.start}, before the entry listed before it, at {x.start}: list \
        the entries by their stretches' starts"
    if y.start = x.start then
      throw s!"keys: two entries start at {x.start}: their stretches overlap, both holding that execution phase"
  let sessions := entries.toList.map fun e => (e.key.extract 0 16, e.session)
  unless sessions.eraseDups.length = sessions.length do
    throw "keys: one session listed twice: two entries hold one key identifier (a key's first 16 bytes) and one \
      session, so that session's execution phase n would be two of the period's"
  let ids := entries.toList.map (·.key.extract 0 16)
  unless (entries.toList.map (·.key)).eraseDups.length = ids.eraseDups.length do
    throw "keys: two keys with one key identifier (a key's first 16 bytes)"
  pure entries

def paramsOf (j : Json) : V Params := do
  match j with
  | .obj _ =>
    if (keysOf j).contains "sync_set" then throw "sync_set: the clock form is not one this check reads"
    if (keysOf j).contains "key" then
      throw "key: write \"keys\", the list of the certifier's entries, [{\"key\": hex(I ‖ T[1]), \"session\": hex, \
        \"start\": a}, …]"
    if (keysOf j).contains "session" then
      throw "session: write each key's session in its entry of \"keys\", {\"key\": hex(I ‖ T[1]), \"session\": hex, \
        \"start\": a}"
    if (keysOf j).contains "sessions" then
      throw "sessions: write each key's session in its entry of \"keys\", {\"key\": hex(I ‖ T[1]), \"session\": hex, \
        \"start\": a}"
    unless keysOf j == rowsParamsKeys do throw s!"the object: the keys must be {rowsParamsKeys}, not {keysOf j}"
  | _ => throw "the object: not an object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (auditOf · "an audit")
  let cells ← (← Flock.J.arr j "cells" "the object").mapM (Flock.J.asNat · "cells")
  let first ← Flock.J.nat j "first" "the object"
  let jitter ← Flock.J.nat j "jitter" "the object"
  unless 1 ≤ jitter do throw "jitter: the observations' jitter bound is at least 1"
  let link ← Flock.J.str j "link" "the object"
  unless link.toUTF8.size < 2 ^ 16 do throw "link: a name is at most 65,535 UTF-8 bytes"
  let l ← Flock.J.get j "lms" "the object"
  exactly l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let entries ← certifierEntriesOf j lms
  if let some e := entries[0]? then
    unless e.start ≤ first do
      throw s!"first: execution phase {first} is before the first listed entry's stretch, which starts at {e.start}"
  let program ← Flock.J.str j "program" "the object"
  unless isDigest program do throw "program: not a SHA-512 digest (128 lowercase hex digits)"
  let s ← scheduleOf (← Flock.J.get j "schedule" "the object")
  if s.connectionRows then throw connectionRowsRefusal
  let form := Form.rows s (← s.digest) (← scanOf (← Flock.J.get j "scan" "the object") program s)
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one execution phase"
  unless rowsWf s windows do throw (rowsWfRefusal s windows)
  unless cells == s.rowsCells windows do
    throw s!"cells: the rows form's cells are egress_cells = {s.egressCells} for each of the lanes · windows · \
      window = {s.lanes * (windows * s.window)} connections the schedule can carry (Schedule.rowsCells), not these \
      {cells.size}"
  let circuit ← Flock.J.str j "circuit" "the object"
  unless isDigest circuit do throw "circuit: not a SHA-512 digest (128 lowercase hex digits)"
  let circuit := some circuit
  pure { audits, cells, first, jitter, link, lms, entries, program, form, windows, circuit }

/-- The network attester's entries the parameters list (`"keys"`), each a key, its session and the start of its stretch, read
through the period's own decoder (`paramsOf`): what a composition that pins its network attester's keys (an `AuditorPin`
each, the pinned list `AuditorPins`) reads off the period's parameters, as `Core.Onsite.store` reads PoUS's root off its
commitment through PoUS's decoder. -/
def attesterEntries (j : Json) : V (Array Entry) := (·.entries) <$> paramsOf j

-- the parameters list the network attester's entries: two decode, in order, each with its key, its session and its start, a
-- key listed in two sessions among them. Refused, each by name: no entry; an entry below the one before it; two entries
-- with one start; one key identifier and one session twice; another key under a listed key's identifier; an entry that
-- is the key alone, or lacks its start; a period that starts before the first entry; and the earlier forms `"key"`,
-- `"session"` and `"sessions"`
#guard
  let hex := fun (b : UInt8) (n : Nat) => String.join (List.replicate n s!"0{b}")
  let (k, ses) := (fun (b : UInt8) => hex b 80, fun (b : UInt8) => hex b attachmentBytes)
  let entry := fun (key session : String) (start : Nat) =>
    Json.mkObj [("key", .str key), ("session", .str session), ("start", toJson start)]
  let listed := fun (es : Array Json) => certifierEntriesOf (Json.mkObj [("keys", .arr es)]) Lms.deployed
  let reason := fun (es : Array Json) => match listed es with | .ok _ => "" | .error e => e
  let params := fun (first : Nat) (more : List (String × Json)) => Json.mkObj (more ++ [
    ("audits", .arr #[]), ("cells", .arr #[]), ("circuit", .null), ("first", toJson first),
    ("jitter", toJson (1 : Nat)), ("link", .str "a"), ("lms", Json.mkObj [("H", toJson (1 : Nat)),
      ("N", toJson (64 : Nat)), ("W", toJson (4 : Nat)), ("toy", .bool false)]),
    ("program", .null), ("scan", .null), ("schedule", .null), ("windows", toJson (1 : Nat))])
  let refusal := fun (j : Json) => match paramsOf j with | .ok _ => "" | .error e => e
  let write := "{\"key\": hex(I ‖ T[1]), \"session\": hex, \"start\": a}"
  let (key, session) := (fun (b : UInt8) => (⟨Array.replicate 80 b⟩ : ByteArray),
    fun (b : UInt8) => (⟨Array.replicate attachmentBytes b⟩ : ByteArray))
  (listed #[entry (k 1) (ses 7) 0, entry (k 2) (ses 8) 3]).toOption ==
      some #[⟨key 1, session 7, 0⟩, ⟨key 2, session 8, 3⟩] &&
    (listed #[entry (k 1) (ses 7) 0, entry (k 2) (ses 8) 2, entry (k 1) (ses 9) 3]).toOption.map
      (·.map (·.start)) == some #[0, 2, 3] &&
    reason #[] == "keys: no key: list the certifier's entries, at least one" &&
    reason #[entry (k 1) (ses 7) 3, entry (k 2) (ses 8) 2] ==
      "keys: out of order: an entry starts at 2, before the entry listed before it, at 3: list the entries by their \
        stretches' starts" &&
    reason #[entry (k 1) (ses 7) 3, entry (k 2) (ses 8) 3] ==
      "keys: two entries start at 3: their stretches overlap, both holding that execution phase" &&
    reason #[entry (k 1) (ses 7) 0, entry (k 1) (ses 7) 3] ==
      "keys: one session listed twice: two entries hold one key identifier (a key's first 16 bytes) and one \
        session, so that session's execution phase n would be two of the period's" &&
    reason #[entry (k 1) (ses 7) 0, entry (hex 1 16 ++ hex 2 64) (ses 8) 3] ==
      "keys: two keys with one key identifier (a key's first 16 bytes)" &&
    reason #[.str (k 1)] == s!"keys: an entry is not an object: write each as {write}" &&
    reason #[Json.mkObj [("key", .str (k 1)), ("session", .str (ses 7))]] ==
      "keys: the keys must be [key, session, start], not [key, session]" &&
    refusal (params 2 [("keys", .arr #[entry (k 1) (ses 7) 3])]) ==
      "first: execution phase 2 is before the first listed entry's stretch, which starts at 3" &&
    refusal (params 0 [("key", .str (k 1))]) ==
      s!"key: write \"keys\", the list of the certifier's entries, [{write}, …]" &&
    refusal (params 0 [("keys", .arr #[entry (k 1) (ses 7) 0]), ("session", .str (ses 7))]) ==
      s!"session: write each key's session in its entry of \"keys\", {write}" &&
    refusal (params 0 [("keys", .arr #[entry (k 1) (ses 7) 0]), ("sessions", .arr #[.str (ses 7)])]) ==
      s!"sessions: write each key's session in its entry of \"keys\", {write}"

/-- The link the parameters name, the one whose records the period checks, read through the period's own decoder
(`paramsOf`): with `attesterEntries`, what a composition reads off the period's parameters. -/
def attesterLink (j : Json) : V String := (·.link) <$> paramsOf j

-- parameters the period refuses give no link
#guard (attesterLink (Json.mkObj [("link", .str "eth0")])).toOption == none

/-- A window record of one of `Flock.WindowRecord.FORMATS`. -/
def registrationOf (j : Json) (ctx : String) : V Json := do
  let format ← Flock.J.str j "format" ctx
  unless Flock.WindowRecord.FORMATS.contains format do
    throw s!"{ctx}: format is not {" or ".intercalate Flock.WindowRecord.FORMATS}"
  pure j

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (registrationOf · "an audit's registration")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (registrationOf · "an execution phase's registration")
  pure { audits, windows, bytes := b }

def rowsOf (j : Json) (ctx : String) : V Rows := do
  exactly j ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
    "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"] ctx
  let l ← Flock.J.get j "layout" ctx
  exactly l ["buckets", "connection_slots", "first_bucket", "slots"] s!"{ctx}.layout"
  let c ← Flock.J.get j "certificate" ctx
  exactly c ["digest", "key", "signature"] s!"{ctx}.certificate"
  pure { format := ← Flock.J.str j "format" ctx, value := ← Flock.J.str j "value" ctx,
         link := ← Flock.J.str j "link" ctx, direction := ← Flock.J.str j "direction" ctx,
         window := ← Flock.J.nat j "window" ctx,
         parameters := ← Flock.J.str j "parameters" ctx, cellBytes := ← Flock.J.nat j "cell_bytes" ctx,
         rows := ← Flock.J.nat j "rows" ctx, words := ← Flock.J.nat j "words" ctx,
         wordBits := ← Flock.J.nat j "word_bits" ctx, rowSchema := ← Flock.J.str j "row_schema" ctx,
         domain := ← Flock.J.str j "domain" ctx, wiring := ← hexAt j "wiring" ctx, root := ← hexAt j "root" ctx,
         layout := { buckets := ← Flock.J.nat l "buckets" ctx, slots := ← Flock.J.nat l "slots" ctx,
                     connectionSlots := ← Flock.J.nat l "connection_slots" ctx,
                     firstBucket := ← Flock.J.nat l "first_bucket" ctx },
         key := ← hexAt c "key" ctx, digest := ← hexAt c "digest" ctx, signature := ← hexAt c "signature" ctx }

def executionPhaseOf (j : Json) (ctx : String) : V ExecutionPhase := do
  exactly j ["answer", "certificates", "registrations"] ctx
  let answer ← Core.Answer.ofJson (← Flock.J.get j "answer" ctx) s!"{ctx}.answer"
  let certificates ← (← Flock.J.arr j "certificates" ctx).mapM (certificateOf · s!"{ctx}.certificates")
  let registrations ← (← Flock.J.arr j "registrations" ctx).mapM (rowsOf · s!"{ctx}.registrations")
  pure { answer, certificates, registrations }

/-- A session's signed start, as the run's `deployment.json` holds it (`"start"`). -/
def startOf (j : Json) (ctx : String) : V Start := do
  exactly j ["base_ms", "certificate", "ordinal", "schedule"] ctx
  pure { baseMs := ← Flock.J.nat j "base_ms" ctx, ordinal := ← Flock.J.nat j "ordinal" ctx,
         schedule := ← hexAt j "schedule" ctx,
         certificate := ← certificateOf (← Flock.J.get j "certificate" ctx) s!"{ctx}.certificate" }

/-- The evidence: the audits' answers and the execution phases', and the listed entries' sessions' signed starts where
it shows any (`"starts"`). Evidence that shows none has no `"starts"`, so it has one encoding. -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  let shown := keysOf j == ["audits", "starts", "windows"]
  unless shown do exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (Core.Answer.ofJson · "an audit's answer")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (executionPhaseOf · "an execution phase")
  let starts ← if shown then (← Flock.J.arr j "starts" "the object").mapM (startOf · "a start") else pure #[]
  if shown && starts.isEmpty then throw "starts: none: evidence that shows no session's start has no \"starts\""
  pure { audits, windows, starts }

end NetworkVerifier.Period
