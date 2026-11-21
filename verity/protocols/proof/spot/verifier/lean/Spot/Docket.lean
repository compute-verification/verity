/-
The docket (`one_stage/docket.py`): what serving is committed to, fixed before the verifier samples, its
digest, and R1–R8 against the verifier's own copies (`check`).

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <units the query derives>, "law": <the sampling design, expanded>, "scheme": "<leaf and tree scheme>",
     "roots": [{"port", "root", "leaves", "domain"}, ...], "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
     "anchors": [], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}

A v2 record (`verity/registration/v2`) adds `"registered": [{"value", "root", "leaves", "words", "domain"}, ...]` (with
`"word_bits"` for a value whose leaves are not 16 bits), and R7 holds it to the verifier's own entries
(`Registered.entries`), a value the verifier pinned before the record to its root (`pinned`: the weights' root recorded
before the salt), and each value an earlier record of the log registered under the domain it takes now to that record's
root (`heldRoots`). The verifier expects a v2 record exactly when it expects values registered or holds certified
entries (`Verifier.v2`). Its entry for a value
counts the value's rows in the row format the verifier holds for it (`schemas`; a bit row's `words` hold as many of its
leaves as their bits do), and in `row/v1` rows when it names none.

A verifier may hold certified entries (`certified`): the entry and root of a value that is none of its program's
parameters, which it built from a registration it read under the deployment's key (the certified rows of a link). R7
then holds the record's registered values to be its own and the certified ones, each once (`R7-values`), a certified
value's entry to the verifier's leaves, words, word bits and domain as its own values', and its root to the
verifier's (`certifiedHeld`, `R7-certified`). A certified value is not an output value, and no pinned root or root of
the log is read for it: its entries join the verifier's own after the output values' steps (`checkOutputs`).

A verifier that names its window's open root R (`windowRoot`, which it reads from the challenger's signed open of the
window, never from the record) takes every registered value's domain under R (`Registered.domainAt`) but the ones it
holds across windows: `held` (the weights), and its pinned, supplied and output values (`fixedOf`). A window-independent
program then registers each window's values under the same names: a per-window value is held to no earlier window's
root, and a root registered in another window opens under no domain R7 takes. One that names no R is refused before the
record is read when a value it doesn't hold across windows was registered by an earlier record under the domain it takes
now (`unwindowed`, `OneStage.paramsOfJson`). A value it fills itself (`supplied`,
`Supplied.root`: rows of the record's roots, positions after the log's, its coin bound to its binding and the record's
window and its own bytes, under public salts) is held to the root it recomputes from this record and the log instead
(`R7-supplied`), never to the log's.

The binding, the seed salt, the registered compiler, the declarations and the pool below are what SPOT's caller hands
the verifier beside the record and SPOT's own log. `check` reads none of them: the caller's check
(`VStar.Recursive.Caller.check`, in `spot_flock`) refuses by the codes named here, after SPOT's check accepts.

A verifier bound to a stream and a prover (`binding`: its stream's digest and the identity it registered the prover under)
holds the record's window to name both (`R6-stream`, `R6-prover`), and a window salt is bound to them, so a tile seeded
for one prover's stream is refused in any other's.

A verifier holds the record's window to name the window's seed salt it read from the challenger's log (`seedSalt`,
`Challenger.seedSaltOf`: the one coin of the seed-salt family released after the window's open root), and to name none
when it read none (`R6-seed-salt`), so a seed salt the server chose is refused by name.

A verifier that reads the registration log's registered compiler (`compiler`: its digest and the public length ℓ of
the advice it reads, or none when the log registers none) holds the record's window to name that compiler by its digest
(`window.compiler.digest`; `R6-compiler`) and advice of at most ℓ bits (`window.advice.bits`; `R6-advice-length`), and,
where the log registers none, to name neither a compiler nor advice (`R6-compiler`), so advice no registered compiler
reads is refused by name.

A verifier that reads the registration log's declarations (`declared`: the window's day and the log's `(day, program)`
declarations) holds the record's program to the day's one registered circuit (`declaredCode`): `R2-undeclared` when the day has no
declaration, `R2-declared-twice` when it has more than one, and `R2-declared` when the record's program is not the
day's.

A value that is an output of an earlier run (`outputs`; never the inference-only policy's baseline θ, which holds registered public data only)
reuses that run's commitment: its entry has the domain and root of the port in the `roots` of the log's record with that
digest, whose leaf count must be the value's rows (`R7-output` otherwise). One may instead reuse the previous record's
registered value (`Earlier.value`): its entry has the domain and root of that value's entry in the `registered` of the
log's record with that digest, which must be the log's last (`lastDigest`, the record R6 holds `previous` to), and whose
leaves, words and word bits must be the value's own (`R7-output` otherwise), so nothing is registered afresh, and the
record whose own check held the value's domain is the one the caller ties this record to (Pearl+'s window record W,
`Core.Remote.V1.executionPhase`).

An output value the parameters give only a port (`{"port"}`, a window's history) is that port of the previous record, the
log's last, which R6 holds `previous` to (`lastDigest`): the verifier names the record itself, so the value is held to the
previous window's root and domain for that port, and with no previous record it is `R7-output`.

A verifier with a pool (`pool`: the registered values a window opens its pool from and leaves it in, `pin` and `pout`,
the empty pool's root `genesis`, the public lag `lag`, its verdict log `verdicts`, serving's `reset` and the window's
`week`) holds each window to open the pool the previous window of record left (`Pool.chains`), whatever it made of that
window: its parameters give `pin` as the previous record's `pout`, reused (`OneStage.paramsParsed`). The chain restarts
from the empty pool, `pin` filled from `genesis`, a fixed root, by a public rule that takes no advice, at a
registration's first window, at a registration boundary (a previous record of another program) and after the verifier
refused the window `lag` windows of record before (`Pool.refused`, read from its own verdict log, `lag` the fixed public
lag at which serving learns a verdict), and by serving's own reset (`reset`), which an engine crash needs, at most one a
week. The record's window names that reset (`pool_reset`; `R6-pool-reset` otherwise), its week (`pool_week`;
`R6-pool-week`) and whether the week's reset is spent at it (`pool_spent`; `R6-pool-spent`): that it resets, or that
the previous record, of its week, named it spent (`Pool.spentBefore`). The caller refuses a reset where the week's is spent
before it (`R6-pool-reset-week`). The resets are serving's choice, so they are advice: the records name, of a week's n
windows, the one serving reset at or none, log₂(n + 1) bits a week (log₂ 1009, 9.98 bits, for 600-second windows),
which leakage charges; the spent marks and weeks are a function of the resets and the verifier's clock. The caller
refuses a record whose entry for `pin` is not the pool its window opens (`poolOpen`) by name (`R7-pool-chain`).

A verifier may give its verdict log (`verdicts`, the pool's form: each record digest it ran, with whether that run
accepted). The caller then refuses a record with an output value that reads a port of a record the log names refused
(`portsAccepted`, `R7-output-refused`): no start is carried past a window the verifier refused, a window's history
included. A record it holds no verdict for is not refused, as for the pool. A registered value an output value reuses
(`Earlier.value`, the pool's `pin` among them) is not held to it: the pool restarts by its own lag (`Pool.refused`).

Under a template query that applies, the fan-out joins no proof blocks through a value another proof block writes that
the program takes as an Input (`Partition.fanout`, which the registration piece publishes). In the variant with no members, R8
refuses a run Input node whose leaves the proof blocks don't each read alone, unless it is a parameter's the verifier
expects registered (`heldGates`) or a tied value's (`tiedGates`) (`Partition.readAlone`, `R8-shared-input`); the
per-member variant holds each reader to the node's one tree instead. A tied value
is a run Input whose rows a later check ties to the registrations that computed them; R8 refuses a tied name that isn't
a parameter of the program, neither registered nor held, whose gates are one interval of run Input nodes (`R8-tied`),
and the registration piece publishes, beside the fan-out, the tied names it let through (`tiedThrough`), since
acceptance rests on that later check. R7 refuses a registered value, a supplied one included (a table opening's root
and domain), whose parameter's gates hold a run Input node (`runInputNamed`, `R7-run-input`): a run Input's rows are
committed once, in its one tree, and are never a registered value. R7 refuses a supplied anchor, a mark of
`registered.root/v1` or `registered.domain/v1`, whose value V neither the verifier's `pinned` nor the weights'
registration of record pins (`anchorUnpinned`, `R7-anchor-unpinned`); a value the registration of record pins whose
record entry is not that pin in root, leaf count, words and domain, or an anchor's value it pins that the record
registers no entry of (`offPin`, `R7-pin`); and a parameter bound as a supplied input that the parameters' `supplied`
gives no source (`unsourced`, `R7-supplied-source`). `R7-anchor-unpinned` reads C's marks (`Verifier.marks`, from the
program descriptor's `inputs`), which no descriptor carries until circuits writes the marks into C: until then it
refuses nothing.

R8 then holds the program's registered inputs (the parameters it binds at registration, `Partition.regInputs`) to the
weights' registration of record the auditor's side hands it (`weightsOfRecord`: its bytes and the SHA-512 it holds; a
document at another digest, `R8-weights-digest`, or of another form, `R8-weights-form`, is refused alone), each pin's
domain the one its own document's program makes (`R8-weights-domain`) and its shape the parameter's
(`R8-weights-shape`). A program with a registered input, or a verifier with a registration, under a query that is no
template query that applies is refused (`R8-weights-query`). An input bound at registration that is neither pinned nor
expected registered and that a template parameter reads is refused at one reader as at several (`R8-unpinned`), and a pin
of no registered input excuses nothing. The pins it accepts of inputs the verifier does not expect registered (`weightsOf`) are no entry
R7 reads: the members' check holds each member's registered read of such an input to its pin's root, leaf count and
domain (`OneStageZk.pinsHeld`).

A verifier of the per-member form (`OneStageZk`) names its members (`members`: each its units and its circuit's units a
proof block, META `vus_per_block`), and after R8 the check refuses a member whose fewest and most sampled proof blocks under the
sampling design the sample runs on, over the samples that end, give different link points, a sampling design it can't
sample on, or one no sample of which ends (`link-points`, `linkRefused`): some sample ends, a sample that ends gives
each member's sampled proof blocks one count, and one that runs out of coins fails the window (`OneStageZk.RAN_OUT`).
-/
module

public import Flock.WindowRecord
public import Spot.Draw
public import Spot.Registered
public import Spot.Supplied
public import Spot.LinkPoints
public import Std.Data.HashMap

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Docket

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/- The record's formats and its digest, `SHA-512(format || 0 || canon(record))` with the record's own format as the tag:
the window record's, which every reader of the record shares. -/
export Flock.WindowRecord (FORMAT FORMAT_V2 digest digestHex)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v1"
def KINDS : List String := ["served", "stand-in"]

def KEYS : Array String := sortStrings #["format", "protocol", "program", "query", "partition", "population", "law",
  "scheme", "roots", "leaf_layer", "anchors", "window", "previous"]
def KEYS_V2 : Array String := sortStrings (KEYS.push "registered")
def ROOT_KEYS : Array String := sortStrings #["port", "root", "leaves", "domain"]

/-- `registration.query_digest`: SHA-512 of the canonical query. -/
def queryDigest (q : Json) : V String := canonSha512 q

/-- What an output value reuses of the log's earlier record: one of its ports (`roots`), or one of its registered values
(`registered`). -/
inductive Earlier where
  | port (name : String)
  | value (name : String)
  deriving Inhabited

/-- The verifier's pool (Pool chain, `registration._pool`): the registered value a window opens its pool from (`pin`),
the one it leaves the pool in (`pout`), the empty pool's root (`genesis`, lowercase hex of 64 bytes), the public lag at
which serving learns a verdict (`lag`, positive), the verifier's verdict log (`verdicts`: each record digest it ran,
with whether that run accepted), the reset serving chose for this window, which its record names (`reset`,
`pool_reset`), and the window's week on the verifier's clock, which its record names too (`week`, `pool_week`). -/
structure Pool where
  pin : String
  pout : String
  genesis : String
  lag : Nat
  verdicts : Array (String × Bool)
  reset : Bool
  week : Nat
  deriving Inhabited

/-- **Whether the verdict log `verdicts` names the verifier's run on the record with digest `d` refused**
(`registration.refused`). A record it holds no verdict for is not refused. -/
def refusedIn (verdicts : Array (String × Bool)) (d : String) : Bool := verdicts.any fun (r, a) => r == d && !a

/-- **Whether the verifier refused the window `lag` windows of record before this one** (`registration.pool_refused`):
its verdict log names its run on that record of the log refused (`refusedIn`). A log shorter than the lag, or a record
it holds no verdict for, is not refused, so where every run accepted the chain never restarts. -/
def Pool.refused (p : Pool) (log : Array Json) : Bool :=
  if p.lag ≤ log.size then
    match log[log.size - p.lag]? with
    | some e => match digestHex e with
      | .ok d => refusedIn p.verdicts d
      | .error _ => false
    | none => false
  else false

/-- **Whether the week's one reset is spent before this window** (`registration.pool_spent_before`): the log's last
record, the previous window of record, names it spent (`pool_spent`) in the verifier's week (`pool_week`), so a week
starts unspent. -/
def Pool.spentBefore (p : Pool) (log : Array Json) : Bool :=
  match log.back? with
  | some e => jeq (getD (getD e "window") "pool_spent") (.bool true) &&
      jeq (getD (getD e "window") "pool_week") (Lean.toJson p.week)
  | none => false

/-- The verifier's own copies that R1–R8 hold the record to. -/
structure Verifier where
  program : Program
  query : Json
  /-- the agreed sampling design: a rule, which `Draw.derive` expands -/
  samplingDesign : Json
  workTable : Option Json
  scheme : String
  /-- each port the caller gives as the verifier's, with its `{"leaves", "domain"}` -/
  ports : Array (String × Json)
  leafLayer : Option String
  anchors : Array Json
  /-- this prover's earlier records, oldest first -/
  log : Array Json
  /-- its own partition object, when it isn't the canonical one (a record made before the canonical envelope) -/
  partition : Option Json
  /-- the parameters it expects registered, `(parameter, words)` each; none: a v1 record -/
  registered : Array (String × Nat) := #[]
  /-- its program function's parameters, which `registered` names -/
  params : Array Registered.Param := #[]
  /-- of `registered`, the values it fills itself, each one's rows of sources (`Supplied.sources`) -/
  supplied : Array (String × Supplied.Rows) := #[]
  /-- of `registered`, values it doesn't fill whose roots it held before the record, `(value, root)` each -/
  pinned : Array (String × String) := #[]
  /-- of `registered`, values it neither fills nor pinned that are outputs of an earlier run or reuse an earlier record's
  registered value, `(value, record digest, its port or value)` each, sorted by value -/
  outputs : Array (String × String × Earlier) := #[]
  /-- the stream and prover it registers the record for, `(stream, prover)`, each lowercase hex -/
  binding : Option (String × String) := none
  /-- of `registered`, values it doesn't fill whose rows are not `row/v1` rows, `(value, schema)` each
  (`Registered.SCHEMAS`): a bit row's leaves are counted in its format (`Registered.entries`) -/
  schemas : Array (String × String) := #[]
  /-- the window's open root, lowercase hex of 64 bytes, from the challenger's signed open of the window -/
  windowRoot : Option String := none
  /-- of `registered`, values it holds across windows, whose domain leaves `windowRoot` out -/
  held : Array String := #[]
  /-- the window's day and the registration log's declarations as `(day, program)` each, from the log it holds when it
  samples; none when it reads no declarations -/
  declared : Option (Nat × Array (Nat × String)) := none
  /-- the window's seed salt, lowercase hex of 64 bytes: the coin of the challenger's seed-salt family it read from the
  challenger's log (`Challenger.seedSaltOf`), which the window must name; none when it read none -/
  seedSalt : Option String := none
  /-- the pool its windows carry, whose `pin` its parameters add to `outputs` or `supplied` (`OneStage.paramsParsed`)
  and the caller's check holds to the chain (`VStar.Recursive.Caller.poolChained`) and its window to name its reset,
  once a week; none for a
  program that carries none -/
  pool : Option Pool := none
  /-- its verdict log, the pool's form: each record digest it ran, with whether that run accepted, which R7 holds an
  output value's earlier port to (`portsAccepted`); empty when it gives none -/
  verdicts : Array (String × Bool) := #[]
  /-- run Inputs, by parameter name, whose rows a later check ties to the registrations that computed them, which R8 lets
  through though several proof blocks read them, on that later check alone (`tiedThrough`) -/
  tied : Array String := #[]
  /-- the per-member form's members, each its units as `[lo, hi)` ranges and its circuit's units a proof block, which
  `link-points` holds to one count of link points at every sample that ends (`linkRefused`); none in the one-statement
  form -/
  members : Array (Array (Nat × Nat) × Nat) := #[]
  /-- the registration log's registered compiler, its digest and the public length ℓ in bits of the advice it reads,
  from the log it holds when it samples: `some none` when the log registers none; none when it reads no log's
  compiler -/
  compiler : Option (Option (String × Nat)) := none
  /-- the certified entries, `(entry, root)` each: values that are none of `registered`, whose entry and root the
  verifier built from a registration it read under the deployment's key, which R7 holds the record's entry for the
  value to (`checkOutputs`, `certifiedHeld`) -/
  certified : Array (Registered.Entry × String) := #[]
  /-- the weights' registration of record: its bytes and the SHA-512 (hex) the auditor's side holds for it, which R8
  holds the program's registered inputs to (`weightsCodes`); none where it is handed none -/
  weightsOfRecord : Option (ByteArray × String) := none
  /-- C's supplied marks (`Supplied.marks`), by input, which R7 holds each anchor of to a pin (`anchorUnpinned`) -/
  marks : Array (String × Supplied.Mark) := #[]

/-- Whether the verifier expects a v2 record: it expects values registered or holds certified entries. -/
def Verifier.v2 (v : Verifier) : Bool := !(v.registered.isEmpty && v.certified.isEmpty)

/-- R6's registered compiler (`registration._compiler`): the window names the log's compiler by its digest, and names
neither a compiler nor advice where the log registers none; anything when the verifier reads no log's compiler. -/
def compilerOk (w : Json) : Option (Option (String × Nat)) → Bool
  | none => true
  | some none => jeq (getD w "compiler") .null && jeq (getD w "advice") .null
  | some (some (d, _)) => jeq (getD (getD w "compiler") "digest") (.str d)

/-- R6's advice length (`registration._compiler`): the window's advice is a natural number of bits at most the log's
compiler's public length ℓ; anything when the log registers no compiler or the verifier reads none. -/
def adviceOk (w : Json) : Option (Option (String × Nat)) → Bool
  | some (some (_, ℓ)) => match nat? (getD (getD w "advice") "bits") with
    | some bits => decide (bits ≤ ℓ)
    | none => false
  | _ => true

/-- The programs the declarations declare for `day`, in the log's order. -/
def declaredOf (day : Nat) (ds : Array (Nat × String)) : List String :=
  (ds.filter (·.1 == day)).toList.map (·.2)

def optStr : Option String → Json
  | some s => Json.str s
  | none => Json.null

/-- The Input gates `(first, count)` of the parameters `v` expects registered, which R8 reads as registered values
(`registered.held_gates`). -/
def heldGates (v : Verifier) : Array (Nat × Nat) :=
  v.params.filterMap fun p => if v.registered.any (·.1 == p.name) then p.gates else none

/-- The registered names, supplied ones included, whose parameter's gates hold a run Input node
(`Partition.anyRunInput`; `registration._run_input_named`), which R7 refuses (`R7-run-input`). -/
def runInputNamed (v : Verifier) : Array String :=
  (v.registered.map (·.1)).filter fun t =>
    ((v.params.find? (·.name == t)).bind (·.gates)).any fun (first, count) =>
      Partition.anyRunInput v.program.prog first count

/-- The Input gates `(first, count)` of tied name `t` where R8 may let it through: a parameter of the program that `v`
neither expects registered nor holds, whose gates are one non-empty interval of run Input nodes
(`Partition.runInputs`; `registration._tied`). A tied value is never a registered value: R7 holds nothing of it. -/
def tiedGate (v : Verifier) (t : String) : Option (Nat × Nat) :=
  if v.registered.any (·.1 == t) || v.held.contains t then none
  else ((v.params.find? (·.name == t)).bind (·.gates)).filter fun (first, count) =>
    Partition.runInputs v.program.prog first count

/-- The gates of the tied names R8 may let through (`tiedGate`). -/
def tiedGates (v : Verifier) : Array (Nat × Nat) := v.tied.filterMap (tiedGate v)

/-- The tied names R8 can't let through (`R8-tied`). -/
def tiedBad (v : Verifier) : Array String := v.tied.filter fun t => (tiedGate v t).isNone

/-- The tied names R8 let through (`tiedGate`). The registration piece publishes them beside the fan-out: acceptance
rests on the later check that ties their rows. -/
def tiedThrough (v : Verifier) : Array String := v.tied.filter fun t => (tiedGate v t).isSome

/-- A pin of the weights' registration of record: its value, leaves, words, word bits, domain and root. -/
structure Pin where
  value : String
  leaves : Nat
  words : Nat
  wordBits : Nat
  domain : String
  root : String
  deriving Inhabited

/-- An entry's leaves, words and word bits (16 where it leaves `word_bits` out), each a natural, when its keys are the
entry's with or without `word_bits`; none otherwise (`R8-weights-form`). -/
def pinForm (e : Json) : Option (Nat × Nat × Nat) :=
  if !(keys? e == some Registered.ENTRY_KEYS || keys? e == some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY)))
  then none
  else match nat? (getD e "leaves"), nat? (getD e "words"),
      (match field? e Registered.WIDTH_KEY with | none => some Registered.WORD_BITS | some j => nat? j) with
    | some l, some w, some b => some (l, w, b)
    | _, _, _ => none

/-- The documents of the weights' registration of record, each `(program, values)`: one object, or a list of them,
each exactly a string `program` and an object `values` of well-formed entries (`pinForm`), no value in two; none
otherwise. -/
def pinDocs (doc : Json) : Option (Array (String × Array (String × Json))) := do
  let docs := match doc with | .arr ds => ds | d => #[d]
  let docs ← docs.mapM fun d => do
    if keys? d != some #["program", "values"] then none
    match getD d "program", getD d "values" with
    | .str p, .obj kvs => if kvs.toArray.all (fun (_, e) => (pinForm e).isSome) then some (p, kvs.toArray) else none
    | _, _ => none
  let names := docs.flatMap fun (_, vs) => vs.map (·.1)
  if (sortStrings names).toList.eraseDups.length != names.size then none
  return docs

/-- `docket._of_record`: the weights' registration of record's pins, after its digest (`R8-weights-digest`), its form
(`R8-weights-form`), each one refused by the first code; then each pin whose value, domain or root is not its own
(`R8-weights-domain`): its domain is the one its own document's program, value and shape make
(`Registered.domain`), its root lowercase hex of 64 bytes. The codes and the pins that pass. -/
def pinsOf (body : ByteArray) (digest : String) : Except String (Array String × Array Pin) := do
  if Flock.hexOf (Flock.Sha512.hash body) != digest then throw "R8-weights-digest"
  let some text := String.fromUTF8? body | throw "R8-weights-form"
  let .ok doc := Json.parse text | throw "R8-weights-form"
  let some docs := pinDocs doc | throw "R8-weights-form"
  let mut bad := false
  let mut pins := #[]
  for (program, vs) in docs do
    for (v, e) in vs do
      let some (leaves, words, wordBits) := pinForm e | throw "R8-weights-form"
      let pin := match Registered.domain program v words leaves wordBits, getD e "root" with
        | .ok d, .str r =>
          if jeq (getD e "value") (Json.str v) && jeq (getD e "domain") (Json.str (Flock.hexOf d)) &&
              r.length == 128 && isHex r
          then some { value := v, leaves, words, wordBits, domain := Flock.hexOf d, root := r : Pin } else none
        | _, _ => none
      match pin with
      | some x => pins := pins.push x
      | none => bad := true
  return (if bad then #["R8-weights-domain"] else #[], pins)

/-- Whether parameter `p` is a registered input of the program: its gates are one interval of root input nodes bound at
registration (`Partition.regInputs`). -/
def marked (v : Verifier) (p : Registered.Param) : Bool :=
  match p.gates with
  | some (first, count) => Partition.regInputs v.program.prog first count
  | none => false

/-- Whether pin `x` is parameter `p`'s shape: the leaves and word bits the verifier's own entry for `p` at `x`'s words
gives (`Registered.entries`, Python's `registered.shape` and `width`). -/
def pinShaped (v : Verifier) (p : Registered.Param) (x : Pin) : Bool :=
  match Registered.entries v.program.desc #[p] #[(p.name, x.words)] with
  | .ok #[e] => e.leaves == x.leaves && e.wordBits == x.wordBits
  | _ => false

/-- The values whose domain leaves the window's open root out: `held`, then the pinned, supplied and output values. -/
def fixedOf (v : Verifier) : Array String :=
  v.held ++ v.pinned.map (·.1) ++ v.supplied.map (·.1) ++ v.outputs.map (·.1)

/-- The verifier's own entries (`registration._check_registered`'s `registered_entries`): each value's domain under the
window's open root, but `fixedOf`'s. -/
def ownEntries (v : Verifier) : V (Array Registered.Entry) :=
  Registered.entries v.program.desc v.params v.registered v.schemas v.windowRoot (fixedOf v)

/-- `registration.held_roots`, read at the verifier's own entries (`mine`, none where they don't compute): `(value,
root)` for each of its values an earlier record of the log registered under the domain `mine` gives it, the root of
its first registration under that domain. -/
def heldRoots (log : Array Json) (mine : V (Array Registered.Entry)) : Array (String × String) :=
  let mine := match mine with | .ok m => m | .error _ => #[]
  log.foldl (fun held rec => match getD rec "registered" with
    | .arr es => es.foldl (fun held e => match getD e "value", getD e "root", getD e "domain" with
      | .str v, .str r, .str d =>
        if held.any (·.1 == v) || !mine.any (fun w => w.value == v && w.domain == d) then held else held.push (v, r)
      | _, _, _ => held) held
    | _ => held) #[]

/-- The roots the verifier itself holds of its values, as R7 holds them (`R7-held`): its `pinned`, then those an earlier
record of the log registered under the domain each takes now (`heldRoots`). -/
def ownHeld (v : Verifier) : Array (String × String) := v.pinned ++ heldRoots v.log (ownEntries v)

/-- `docket.held_entries`: each of the verifier's own entries whose root it holds itself (`ownHeld`), with that root;
none where its entries don't compute. -/
def heldEntries (v : Verifier) : Array (Registered.Entry × String) :=
  match ownEntries v with
  | .error _ => #[]
  | .ok mine => mine.filterMap fun e => ((ownHeld v).find? (·.1 == e.value)).map fun (_, r) => (e, r)

/-- **R8's hold on the program's registered inputs** (`docket.weights_held`): its codes in order, the weights'
registration of record's (`pinsOf`), `R8-weights-shape` for a pinned registered input whose shape isn't its pin's, then
`R8-weights-query` for a registered input or a registration of record under a query that is no template query that
applies; else `R8-unpinned` for a registered input neither pinned nor one whose root the verifier itself holds
(`heldEntries`) that a template parameter reads (`Partition.inputReads`), at one reader as at several: that the verifier
expects it registered by name excuses nothing. Then the accepted pins, each an excused input (marked, pinned and
shaped) as an entry and its root, which the members' check holds each member's registered read of it to
(`OneTree.pinsHeld`). None for a verifier with no registration of record and no registered input. -/
def weightsHeld (v : Verifier) (codes : Array String) (pins : Array Pin) :
    Array String × Array (Registered.Entry × String) :=
  let P := v.program.prog
  let marks := v.params.filter (marked v)
  let pinned := marks.filterMap fun p => (pins.find? (·.value == p.name)).map (p, ·)
  let codes := if pinned.all fun (p, x) => pinShaped v p x then codes else codes.push "R8-weights-shape"
  let held := (heldEntries v).map (·.1.value)
  match Partition.templateRanges? v.program v.query with
  | none => (if v.weightsOfRecord.isSome || !marks.isEmpty then codes.push "R8-weights-query" else codes, #[])
  | some rs =>
    let gated (p : Registered.Param) := (p.gates.map fun (first, count) => #[(p.name, first, count)]).getD #[]
    let loose := marks.filter fun p => !(pins.any (·.value == p.name)) && !held.contains p.name
    let codes := if (Partition.inputReads P rs (loose.flatMap gated)).isEmpty then codes else codes.push "R8-unpinned"
    let excused := pinned.filter fun (p, x) => pinShaped v p x
    (codes, excused.map fun (_, x) =>
      ({ value := x.value, leaves := x.leaves, words := x.words, wordBits := x.wordBits, domain := x.domain }, x.root))

/-- `weightsHeld` after the weights' registration of record's own refusals (`pinsOf`), a digest or form refusal alone
and no pin accepted. -/
def weightsOf (v : Verifier) : Array String × Array (Registered.Entry × String) :=
  match v.weightsOfRecord with
  | none => weightsHeld v #[] #[]
  | some (body, digest) => match pinsOf body digest with
    | .error code => (#[code], #[])
    | .ok (codes, pins) => weightsHeld v codes pins

/-- The supplied functions whose value is a registered value's root or domain (`anchorUnpinned`). -/
def ANCHOR_FUNCTIONS : Array String := #["registered.root/v1", "registered.domain/v1"]

/-- The weights' registration of record's pins (`pinsOf`), none where it is absent or refused. -/
def recordPins (v : Verifier) : Array Pin :=
  match v.weightsOfRecord with
  | some (body, digest) => match pinsOf body digest with
    | .ok (_, pins) => pins
    | .error _ => #[]
  | none => #[]

/-- The values V of C's supplied anchors (a mark of `ANCHOR_FUNCTIONS`), each once, in the marks' order. -/
def anchorValues (v : Verifier) : Array String :=
  v.marks.foldl (init := (#[] : Array String)) fun out (mk : String × Supplied.Mark) =>
    let value : Option Json := (mk.2.args.find? (·.1 == "value")).map Prod.snd
    match ANCHOR_FUNCTIONS.contains mk.2.fn, value with
    | true, some (Json.str x) => if out.contains x then out else out.push x
    | _, _ => out

/-- `docket.anchor_unpinned`: the values V of C's supplied anchors (a mark of `ANCHOR_FUNCTIONS`) that are neither in
the verifier's `pinned` nor pinned by the weights' registration of record (`pinsOf`'s pins), each once, in the marks'
order; R7 refuses them (`R7-anchor-unpinned`). It reads C's marks (`Verifier.marks`), a field no circuit description
carries today, so it refuses nothing yet. -/
def anchorUnpinned (v : Verifier) : Array String :=
  let ofRecord := (recordPins v).map (·.value)
  (anchorValues v).filter fun x => !(v.pinned.any (·.1 == x) || ofRecord.contains x)

/-- `docket.off_pin`: the values the weights' registration of record pins (`recordPins`, in its order) whose record entry
(`rec`'s last `registered` entry of the value) is not that pin in root, leaf count, words and domain, or that a supplied
anchor of C names (`anchorValues`) and the record registers no entry of; R7 refuses them (`R7-pin`). A value the
verifier's `pinned` names is held by `R7-held` and R7's own entry (`checkRegistered`). -/
def offPin (rec : Json) (v : Verifier) : Array String :=
  let got : Array Json := match getD rec "registered" with | .arr es => es | _ => #[]
  let anchors := anchorValues v
  (recordPins v).filterMap fun p =>
    let off := match got.reverse.find? (fun e => getD e "value" == Json.str p.value) with
      | none => anchors.contains p.value
      | some e => !(getD e "root" == Json.str p.root && nat? (getD e "leaves") == some p.leaves
          && nat? (getD e "words") == some p.words && getD e "domain" == Json.str p.domain)
    if off then some p.value else none

/-- Whether the root nodes inside gates `[first, first + count)` of `P` are all primitive supplied inputs
(`SuppliedInput<w>_v1`), as `Partition.regInputs` reads registered ones. -/
def suppliedInputs (P : Flock.Program.Prog) (first count : Nat) : Bool :=
  let nodes := Flock.Registered.rootNodes P
  let inside := (List.range nodes.size).filter (Partition.within P first count)
  !inside.isEmpty && inside.all fun i => match nodes[i]? >>= fun x => P.defs[x.fn]? with
    | some f => f.prim && f.binding == some Flock.Program.SUPPLIED
    | none => false

/-- `docket.unsourced_supplied`: the parameters bound as supplied inputs (`suppliedInputs`) that the parameters'
`supplied` (C's marked rows among them) gives no source and that is no output of an earlier record (`outputs`, a pool's
reused `pin` among them); R7 refuses them (`R7-supplied-source`). -/
def unsourced (v : Verifier) : Array String :=
  (v.params.filter fun p => match p.gates with
    | some (first, count) => suppliedInputs v.program.prog first count && !v.supplied.any (·.1 == p.name)
        && !v.outputs.any (·.1 == p.name)
    | none => false).map (·.name)

/-- The members' check's refusal of a program marking a registered input the verifier does not expect, under parameters
whose weights' registration of record is absent or one R8 refuses (`readPins`). -/
def WEIGHTS_UNREGISTERED_WHY : String :=
  "registered: the program marks registered inputs the verifier does not expect, and the parameters carry no weights' " ++
    "registration of record R8 accepts: a marked input is read only at a pin R8 accepted"

/-- The members' check's refusal of a program marking a registered input the verifier does not expect, under a
partition that is no template query applying to the program, whose ranges the members' reads are read off (`readPins`). -/
def WEIGHTS_QUERY_WHY : String :=
  "registered: the program marks registered inputs the verifier does not expect, and the partition is no template " ++
    "query that applies to the program, whose ranges its members' reads of them are read off: a marked input is read " ++
    "only at a pin R8 accepted"

/-- What the members' check holds each registered read to (`readPins`). -/
structure ReadPins where
  /-- the marked inputs the verifier does not expect -/
  loose : Array String
  /-- their gates, `(value, first, count)` each -/
  held : Array (String × Nat × Nat)
  /-- the template query's ranges whose instance node reads one (`Partition.memberReads`) -/
  ranges : Array Flock.TemplateQuery.TRange
  /-- R8's accepted pins -/
  pins : Array (Registered.Entry × String)
  /-- the roots the verifier itself holds (`heldEntries`) -/
  roots : Array (Registered.Entry × String)

/-- What the members' check holds each registered read to, on the docket's own derivation (`weightsOf v`, of the one
parameters' verifier), with the roots the verifier itself holds (`heldEntries`). None where the program marks an input
the verifier does not expect and the verifier holds no weights' registration of record, or R8 refuses it; the first
three empty where every marked input is one the verifier expects. -/
def readPins (v : Verifier) : Option ReadPins :=
  let loose := (v.params.filter (marked v)).filter fun p => !v.registered.any (·.1 == p.name)
  let roots := heldEntries v
  if loose.isEmpty then some ⟨#[], #[], #[], (weightsOf v).2, roots⟩ else
  let (codes, pins) := weightsOf v
  if v.weightsOfRecord.isNone || !codes.isEmpty then none else
  match Partition.templateRanges? v.program v.query with
  | none => none
  | some rs =>
    let held := loose.flatMap fun p => (p.gates.map fun (first, count) => #[(p.name, first, count)]).getD #[]
    some ⟨loose.map (·.name), held,
      rs.filter fun r => !(Partition.memberReads v.program.prog held r 0 r.count).isEmpty, pins, roots⟩

/-- Why `readPins v` is none: no template query applying (`WEIGHTS_QUERY_WHY`), else no registration R8 accepts
(`WEIGHTS_UNREGISTERED_WHY`). -/
def readPinsWhy (v : Verifier) : String :=
  if (Partition.templateRanges? v.program v.query).isNone then WEIGHTS_QUERY_WHY else WEIGHTS_UNREGISTERED_WHY

/-- Every marked parameter's gates, `(value, first, count)` each, the expected ones' with the loose ones'
(`docket.marked_gates`): the members' check holds each port reading one to its entry of record. -/
def markedGates (v : Verifier) : Array (String × Nat × Nat) :=
  (v.params.filter (marked v)).flatMap fun p => (p.gates.map fun (first, count) => #[(p.name, first, count)]).getD #[]

/-- R8's codes on the program's registered inputs (`weightsOf`). -/
def weightsCodes (v : Verifier) : Array String := (weightsOf v).1

/-- `registration.unwindowed`: for a verifier that names no window root, its values it doesn't hold across windows
(`fixedOf`) that an earlier record of the log registered under the domain it takes now, a window-independent program's
fixed-name per-window values from its second window on; none with a window root, or where its entries don't compute. -/
def unwindowed (v : Verifier) : Array String :=
  if v.windowRoot.isSome then #[] else
    let fixed := fixedOf v
    ((heldRoots v.log (ownEntries v)).map (·.1)).filter (!fixed.contains ·)

/-- **The previous record's digest**, the one R6 holds `previous` to: the log's last record's, and empty, which names no
record, when the log is empty or that record has no digest. An output value the parameters give only a port is that port
of this record (`OneStage.paramsParsed`), so a window's history is the previous window's output whatever else the log
holds. -/
def lastDigest (log : Array Json) : String :=
  match log.back? with
  | some e => (digestHex e).toOption.getD ""
  | none => ""

/-- `registration._output_port`: the `roots` entry for `port` of the log's first record with digest `record` (its last,
as a dict reads it). -/
def outputPort (log : Array Json) (record port : String) : Option Json := do
  let r ← log.find? fun r => match digestHex r with | .ok h => h == record | .error _ => false
  let .arr roots := getD r "roots" | none
  roots.reverse.find? fun e => (e matches .obj _) && jeq (getD e "port") (Json.str port)

/-- The domain and root of `port` in the log's record with digest `record` (`outputPort`), when its leaf count is
`leaves` and both are strings. -/
def outputRoot (log : Array Json) (leaves : Nat) (record port : String) : Option (String × String) :=
  (outputPort log record port).bind fun r =>
    match int? (getD r "leaves") == some (leaves : Int), getD r "domain", getD r "root" with
    | true, .str dom, .str root => some (dom, root)
    | _, _, _ => none

/-- `registration._output_port` on `registered`: the entry for `value` in the `registered` of the log's first record with
digest `record` (its last naming it, as a dict reads it). -/
def outputValue (log : Array Json) (record value : String) : Option Json := do
  let r ← log.find? fun r => match digestHex r with | .ok h => h == record | .error _ => false
  let .arr es := getD r "registered" | none
  es.reverse.find? fun e => (e matches .obj _) && jeq (getD e "value") (Json.str value)

/-- An entry's word bits: its `word_bits`, an integer, or 16 when it leaves it out. -/
def widthOf (e : Json) : Option Int :=
  match field? e Registered.WIDTH_KEY with
  | none => some (Registered.WORD_BITS : Int)
  | some j => int? j

/-- The domain and root of `value`'s entry in the log's record with digest `record` (`outputValue`), when its leaves,
words and word bits are `leaves`, `words` and `bits` and both are strings. -/
def valueRoot (log : Array Json) (leaves words bits : Nat) (record value : String) : Option (String × String) :=
  (outputValue log record value).bind fun e =>
    match int? (getD e "leaves") == some (leaves : Int) && int? (getD e "words") == some (words : Int)
        && widthOf e == some (bits : Int), getD e "domain", getD e "root" with
    | true, .str dom, .str root => some (dom, root)
    | _, _, _ => none

/-- The domain and root an output value with entry `want` takes from the log's record with digest `record`: its port's
(`outputRoot`, at `want`'s leaf count) or its registered value's (`valueRoot`, at `want`'s leaves, words and bits), the
latter of the log's last record alone (`lastDigest`). -/
def earlierRoot (log : Array Json) (want : Registered.Entry) (record : String) : Earlier → Option (String × String)
  | .port p => outputRoot log want.leaves record p
  | .value u => if record == lastDigest log then valueRoot log want.leaves want.words want.wordBits record u else none

/-- One output value `(v, record, earlier)` with an entry in `mine` takes its earlier record's port or registered value
(`earlierRoot`): its domain in `mine` and its root ahead of what `held` holds, or `R7-output` in `c`. -/
def applyOutput (log : Array Json) :
    Array String × Array Registered.Entry × Array (String × String) → String × String × Earlier →
      Array String × Array Registered.Entry × Array (String × String)
  | (c, mine, held), (v, d, p) =>
    match mine.find? (·.value == v) with
    | none => (c, mine, held)
    | some want => match earlierRoot log want d p with
      | some (dom, root) =>
        (c, mine.map fun e => if e.value == v then { e with domain := dom } else e, #[(v, root)] ++ held)
      | none => (addCode c "R7-output", mine, held)

/-- `registration._check_registered`'s output values, before the rest of R7: `applyOutput` for each, in order. -/
def applyOutputs (c : Array String) (mine : Array Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Earlier)) (log : Array Json) :
    Array String × Array Registered.Entry × Array (String × String) :=
  outputs.foldl (applyOutput log) (c, mine, held)

/-- `registration._check_registered`, R7: the record's `registered` against the verifier's own entries (`mine`, an error
where `registered_entries` raises), the roots the log holds, and each supplied value's root to the one `rootOf` recomputes
(an error where `supplied_root` raises). -/
def checkRegistered (c : Array String) (got : Json) (mine : V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → V String) : Array String := Id.run do
  let .ok mine := mine | return addCode c "R7-registered"
  let widthKeys := sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY)
  let .arr got := got | return addCode c "R7-registered"
  if !got.all (fun e => keys? e == some Registered.ENTRY_KEYS || keys? e == some widthKeys) then
    return addCode c "R7-registered"
  let mut c := c
  -- Python's `{e["value"]: e}`: a value that isn't a string is none of the verifier's, which are parameter names
  let values := got.map (getD · "value")
  let names := values.filterMap fun v => match v with | .str s => some s | _ => none
  let distinct := names.toList.eraseDups
  if names.size != values.size || distinct.length != values.size || distinct.length != mine.size
      || !distinct.all (fun v => mine.any (·.value == v)) then
    c := addCode c "R7-values"
  for want in mine do
    -- a value registered twice reads as its last entry, as Python's dict does
    let some e := got.reverse.find? (fun e => jeq (getD e "value") (Json.str want.value)) | continue
    if !(match getD e "root" with | .str s => s.length == 128 && isHex s | _ => false) then c := addCode c "R7-root"
    -- a count is an integer, never a boolean equal to it
    if int? (getD e "leaves") != some (want.leaves : Int) then c := addCode c "R7-leaves"
    if int? (getD e "words") != some (want.words : Int) then c := addCode c "R7-words"
    if !jeq (getD e "domain") (Json.str want.domain) then c := addCode c "R7-domain"
    -- 16 is named by leaving word_bits out: one record digest per registration
    let badWidth := match field? e Registered.WIDTH_KEY with
      | none => want.wordBits != Registered.WORD_BITS
      | some j => match int? j with
        | some w => w != want.wordBits || w == Registered.WORD_BITS
        | none => true
    if badWidth then c := addCode c "R7-word_bits"
    if !supplied.any (·.1 == want.value) then
      if let some (_, r) := held.find? (·.1 == want.value) then
        if !jeq (getD e "root") (Json.str r) then c := addCode c "R7-held"
  for (v, rs) in supplied do
    match mine.find? (·.value == v) with
    | none => c := addCode c "R7-supplied"
    | some want =>
      if let some e := got.reverse.find? (fun e => jeq (getD e "value") (Json.str v)) then
        match rootOf v rs want with
        | .error _ => c := addCode c "R7-supplied"
        | .ok r => if !jeq (getD e "root") (Json.str r) then c := addCode c "R7-supplied"
  return c

/-- R7 with output values: `checkRegistered` on the entries and held roots `applyOutputs` leaves, then the certified
entries (`certified`), which no output step reads, after its codes. -/
def checkOutputs (c : Array String) (got : Json) (mine : V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → V String)
    (outputs : Array (String × String × Earlier)) (log : Array Json)
    (certified : Array Registered.Entry := #[]) : Array String :=
  match mine with
  | .error _ => addCode c "R7-registered"
  | .ok mine =>
    let (c, mine, held) := applyOutputs c mine held outputs log
    checkRegistered c got (.ok (mine ++ certified)) held supplied rootOf

/-- **Whether a window carries the previous window's pool** (`registration.pool_chains`): neither the verifier refused
the window the public lag back (`Pool.refused`) nor serving reset it (`reset`), and the log's last record, the previous
window of record, registered its program (`program`, its digest as R2 reads it). Otherwise it opens the empty pool. -/
def Pool.chains (p : Pool) (log : Array Json) (program : String) : Bool :=
  match log.back? with
  | some e => !p.reset && !p.refused log && jeq (getD e "program") (Json.str program)
  | none => false

/-- The domain and root of the pool the record's window opens (`registration.pool_open`): the previous record's entry
for `pout` (its last naming it) when the window carries the previous window's pool (`Pool.chains`), else the empty
pool's: the domain the verifier gives `pin` (`ownEntries`) and the root R7 recomputes for it filled from `genesis`
(`Supplied.root`); none where there is none. -/
def poolOpen (rec : Json) (v : Verifier) (p : Pool) : Option (String × String) :=
  if p.chains v.log (Flock.hexOf v.program.sha512) then
    match v.log.back?.map (getD · "registered") with
    | some (.arr es) =>
      match es.reverse.find? fun e => (e matches .obj _) && jeq (getD e "value") (Json.str p.pout) with
      | some e => match getD e "domain", getD e "root" with
        | .str d, .str r => some (d, r)
        | _, _ => none
      | none => none
    | _ => none
  else match ownEntries v with
    | .ok mine => match mine.find? (·.value == p.pin) with
      | some want =>
        match Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1))
            #[] p.pin #[#[("bytes", p.genesis)]] want with
        | .ok r => some (want.domain, r)
        | .error _ => none
      | none => none
    | .error _ => none

/-- The record's `registered` entry for `value`, its last naming it, as R7 reads a value registered twice. -/
def entryOf (rec : Json) (value : String) : Option Json :=
  match getD rec "registered" with
  | .arr es => es.reverse.find? fun e => jeq (getD e "value") (Json.str value)
  | _ => none

/-- R7's certified roots (`registration._certified_held`): the record's entry for each certified value (`entryOf`) has
the root the verifier holds for it; `R7-certified` otherwise. -/
def certifiedHeld (rec : Json) (v : Verifier) : Bool :=
  v.certified.all fun x => match entryOf rec x.1.value with
    | some e => jeq (getD e "root") (Json.str x.2)
    | none => false

/-- The verdicts the caller's check holds (`VStar.Recursive.Caller.check`, `caller._ports_refused`), not `check`: no
output value reads a port of a record the verifier's verdict log names refused (`refusedIn`); `R7-output-refused`
otherwise. An output value that reuses a registered value is not held to it. -/
def portsAccepted (v : Verifier) : Bool :=
  v.outputs.all fun (_, d, e) => match e with
    | .port _ => !refusedIn v.verdicts d
    | .value _ => true

/-- R3's index of the record's roots, each root's port canonicalized once: the last root naming each canonical port, and
how many distinct ports the roots name, a port with no canonical form counting as one of its own (`jeq` equals it to
nothing). -/
structure PortIndex where
  last : Std.HashMap String Json := {}
  count : Nat := 0

/-- One root into the index. The index is taken apart first and the map read and written in one call, so a map held
once is updated in place: reading it after the insert would copy it, root by root. -/
def PortIndex.add : PortIndex → Json → PortIndex
  | ⟨last, count⟩, r =>
    match (Flock.canon (getD r "port")).toOption with
    | none => ⟨last, count + 1⟩
    | some k =>
      let p := last.containsThenInsert k r
      ⟨p.2, if p.1 then count else count + 1⟩

def portIndex (roots : Array Json) : PortIndex := roots.foldl PortIndex.add {}

/-- The last root naming the verifier's port `p`. -/
def PortIndex.find (ix : PortIndex) (p : String) : Option Json :=
  match Flock.canon (Json.str p) with
  | .ok k => ix.last[k]?
  | .error _ => none

def PortIndex.has (ix : PortIndex) (p : String) : Bool := (ix.find p).isSome

/-- `registration.link_refused`: the verifier names members, and the sampling design the sample runs on doesn't derive, no sample under
it ends, or some member's fewest and most sampled proof blocks under it, over the samples that end, give different link points
(`LinkPoints.refused`). -/
def linkRefused (v : Verifier) : Bool :=
  !v.members.isEmpty &&
    match LinkPoints.samplingDesignOf v.program v.query v.samplingDesign v.workTable >>= fun (l, n) => LinkPoints.refused l n v.members with
    | .ok rs => !rs.isEmpty
    | .error _ => true


/-- The codes `c`, with `code` added unless `ok`. It names `c` once, where `if ok then c else addCode c code` names it
twice, so `check_eq_checkRef`'s `simp`, which leaves it folded, doesn't copy the codes before it into both cases. -/
def kept (ok : Bool) (c : Array String) (code : String) : Array String := if ok then c else addCode c code

/-- R8's codes added to `c`: a tied name it can't let through (`R8-tied`, `tiedBad`), then, in the variant with no
members, a run Input node the proof blocks don't each read alone outside the registered and tied parameters' gates
(`R8-shared-input`, `Partition.readAlone`), then R8's hold on the program's registered inputs (`weightsCodes`). One
term rather than two lines, since `check_eq_checkRef`'s `simp` copies every line after a join point into each of its
branches. -/
def r8Codes (v : Verifier) (c : Array String) : Array String :=
  (weightsCodes v).foldl addCode
    (kept (!v.members.isEmpty || Partition.readAlone v.program v.query (heldGates v ++ tiedGates v))
      (kept (tiedBad v).isEmpty c "R8-tied") "R8-shared-input")

/-- `registration.check`'s failure codes, R1–R8 in order, then `link-points` (`linkRefused`): empty when the record
holds. -/
def check (rec : Json) (v : Verifier) : Array String := Id.run do
  let v2 := v.v2
  if keys? rec != some (if v2 then KEYS_V2 else KEYS) then return #["keys"]
  let get := getD rec
  let mut c : Array String := #[]
  -- R1
  if !(jeq (get "format") (if v2 then FORMAT_V2 else FORMAT) && jeq (get "protocol") PROTOCOL) then
    c := addCode c "R1-protocol"
  match Draw.derive v.program v.query v.samplingDesign v.workTable with
  | .error _ => c := addCode c "R1-law"
  | .ok mine => if !Draw.sameSamplingDesign (get "law") mine then c := addCode c "R1-law"
  let w := get "window"
  if !(w matches .obj _) || !KINDS.any (jeq (getD w "kind") ·) then c := addCode c "window"
  -- R2
  let program := Flock.hexOf v.program.sha512
  let part := v.partition.getD (Partition.build program v.query)
  if !jeq (getD part "program") program then c := addCode c "R2-program"
  let partDigest := Partition.digest part
  if partDigest.isNone then c := addCode c "R2-partition"
  let mine : List (String × Option Json) :=
    [("program", some (getD part "program")), ("query", (queryDigest (getD part "query")).toOption.map Json.str),
     ("partition", partDigest.map Json.str)]
  for (name, m) in mine do
    if let some m := m then
      if !jeq (get name) m then c := addCode c s!"R2-{name}"
  match Partition.units v.program v.query with
  | .error _ => c := addCode c "R2-query"
  | .ok population => if !jeq (get "population") (Lean.toJson population) then c := addCode c "R2-population"
  -- R3
  if !jeq (get "scheme") v.scheme then c := addCode c "R3-scheme"
  let roots := match get "roots" with | .arr rs => rs | _ => #[]
  let ix := portIndex roots
  if roots.isEmpty || !roots.all (keys? · == some ROOT_KEYS) then c := addCode c "R3-roots"
  else
    if ix.count != roots.size || ix.count != v.ports.size || !v.ports.all (fun (p, _) => ix.has p) then
      c := addCode c "R3-ports"
    for (p, want) in v.ports do
      -- a port registered twice reads as its last root, as Python's dict does
      let some r := ix.find p | continue
      if !(match getD r "root" with | .str s => isHex s | _ => false) then c := addCode c "R3-root"
      if !jeq (getD r "leaves") (getD want "leaves") then c := addCode c "R3-leaves"
      if !jeq (getD r "domain") (getD want "domain") then c := addCode c "R3-domain"
  -- R4: when the verifier holds the leaf layer, the record must name it (a null leaf layer is refused)
  if !jeq (get "leaf_layer") (optStr v.leafLayer) then c := addCode c "R4-leaf-layer"
  -- R5
  if !jeq (get "anchors") (Json.arr v.anchors) then c := addCode c "R5-anchors"
  -- R6
  let previous := match v.log.back? with
    | some e => (digestHex e).toOption.map Json.str |>.getD Json.null
    | none => Json.null
  if !jeq (get "previous") previous then c := addCode c "R6-previous"
  if v.log.any (fun e => jeq (getD e "window") (get "window")) then c := addCode c "R6-window"
  -- R7
  if v2 then
    c := checkOutputs c (get "registered") (ownEntries v) (v.pinned ++ heldRoots v.log (ownEntries v)) v.supplied
      (Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1))
        v.schemas)
      v.outputs v.log (v.certified.map (·.1))
    c := kept (certifiedHeld rec v) c "R7-certified"
    c := kept (runInputNamed v).isEmpty c "R7-run-input"
    c := kept (anchorUnpinned v).isEmpty c "R7-anchor-unpinned"
    c := kept (offPin rec v).isEmpty c "R7-pin"
  -- a supplied input's rows have a source whatever the record registers
  c := kept (unsourced v).isEmpty c "R7-supplied-source"
  -- R8
  c := r8Codes v c
  return if linkRefused v then addCode c "link-points" else c

/-- `check` as first written, R3 quadratic in the ports: each root's port compared, re-canonicalized, with every port
before it and every verifier port with every root. The reference `check` is proved equal to (`check_eq_checkRef`). -/
def checkRef (rec : Json) (v : Verifier) : Array String := Id.run do
  let v2 := v.v2
  if keys? rec != some (if v2 then KEYS_V2 else KEYS) then return #["keys"]
  let get := getD rec
  let mut c : Array String := #[]
  -- R1
  if !(jeq (get "format") (if v2 then FORMAT_V2 else FORMAT) && jeq (get "protocol") PROTOCOL) then
    c := addCode c "R1-protocol"
  match Draw.derive v.program v.query v.samplingDesign v.workTable with
  | .error _ => c := addCode c "R1-law"
  | .ok mine => if !Draw.sameSamplingDesign (get "law") mine then c := addCode c "R1-law"
  let w := get "window"
  if !(w matches .obj _) || !KINDS.any (jeq (getD w "kind") ·) then c := addCode c "window"
  -- R2
  let program := Flock.hexOf v.program.sha512
  let part := v.partition.getD (Partition.build program v.query)
  if !jeq (getD part "program") program then c := addCode c "R2-program"
  let partDigest := Partition.digest part
  if partDigest.isNone then c := addCode c "R2-partition"
  let mine : List (String × Option Json) :=
    [("program", some (getD part "program")), ("query", (queryDigest (getD part "query")).toOption.map Json.str),
     ("partition", partDigest.map Json.str)]
  for (name, m) in mine do
    if let some m := m then
      if !jeq (get name) m then c := addCode c s!"R2-{name}"
  match Partition.units v.program v.query with
  | .error _ => c := addCode c "R2-query"
  | .ok population => if !jeq (get "population") (Lean.toJson population) then c := addCode c "R2-population"
  -- R3
  if !jeq (get "scheme") v.scheme then c := addCode c "R3-scheme"
  let roots := match get "roots" with | .arr rs => rs | _ => #[]
  if roots.isEmpty || !roots.all (keys? · == some ROOT_KEYS) then c := addCode c "R3-roots"
  else
    let distinct := roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]
    if distinct.size != roots.size || distinct.size != v.ports.size
        || !v.ports.all (fun (p, _) => distinct.any (jeq · p)) then
      c := addCode c "R3-ports"
    for (p, want) in v.ports do
      -- a port registered twice reads as its last root, as Python's dict does
      let some r := roots.reverse.find? (fun r => jeq (getD r "port") p) | continue
      if !(match getD r "root" with | .str s => isHex s | _ => false) then c := addCode c "R3-root"
      if !jeq (getD r "leaves") (getD want "leaves") then c := addCode c "R3-leaves"
      if !jeq (getD r "domain") (getD want "domain") then c := addCode c "R3-domain"
  -- R4: when the verifier holds the leaf layer, the record must name it (a null leaf layer is refused)
  if !jeq (get "leaf_layer") (optStr v.leafLayer) then c := addCode c "R4-leaf-layer"
  -- R5
  if !jeq (get "anchors") (Json.arr v.anchors) then c := addCode c "R5-anchors"
  -- R6
  let previous := match v.log.back? with
    | some e => (digestHex e).toOption.map Json.str |>.getD Json.null
    | none => Json.null
  if !jeq (get "previous") previous then c := addCode c "R6-previous"
  if v.log.any (fun e => jeq (getD e "window") (get "window")) then c := addCode c "R6-window"
  -- R7
  if v2 then
    c := checkOutputs c (get "registered") (ownEntries v) (v.pinned ++ heldRoots v.log (ownEntries v)) v.supplied
      (Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1))
        v.schemas)
      v.outputs v.log (v.certified.map (·.1))
    c := kept (certifiedHeld rec v) c "R7-certified"
    c := kept (runInputNamed v).isEmpty c "R7-run-input"
    c := kept (anchorUnpinned v).isEmpty c "R7-anchor-unpinned"
    c := kept (offPin rec v).isEmpty c "R7-pin"
  -- a supplied input's rows have a source whatever the record registers
  c := kept (unsourced v).isEmpty c "R7-supplied-source"
  -- R8
  c := r8Codes v c
  return if linkRefused v then addCode c "link-points" else c

/-! ## `check` is `checkRef` -/

theorem jeq_of_canon {x y : Json} {k : String} (h : Flock.canon y = .ok k) :
    jeq x y = ((Flock.canon x).toOption == some k) := by
  unfold jeq
  rw [h]
  cases Flock.canon x <;> simp [Except.toOption]

theorem jeq_of_canon_error {x y : Json} {e : String} (h : Flock.canon y = .error e) : jeq x y = false := by
  unfold jeq
  rw [h]
  cases Flock.canon x <;> rfl

private def distinctStep (acc : Array Json) (r : Json) : Array Json :=
  if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")

private def keyIs (k : String) (x : Json) : Bool := (Flock.canon x).toOption == some k

private def Inv (acc : Array Json) (ix : PortIndex) (seen : List Json) : Prop :=
  ix.count = acc.size ∧ (∀ k, ix.last.contains k = acc.any (keyIs k)) ∧
    ∀ k, ix.last[k]? = seen.find? fun r => keyIs k (getD r "port")

private theorem inv_step {acc : Array Json} {ix : PortIndex} {seen : List Json} (h : Inv acc ix seen) (r : Json) :
    Inv (distinctStep acc r) (ix.add r) (r :: seen) := by
  obtain ⟨hc, hk, hl⟩ := h
  obtain ⟨last, count⟩ := ix
  unfold distinctStep PortIndex.add
  simp only [Std.HashMap.containsThenInsert_fst, Std.HashMap.containsThenInsert_snd] at *
  rcases hx : Flock.canon (getD r "port") with e | k₀
  · have hn : acc.any (jeq · (getD r "port")) = false := by simp [jeq_of_canon_error hx]
    have hkey : ∀ k, keyIs k (getD r "port") = false := fun k => by simp [keyIs, hx, Except.toOption]
    simp only [hn, Except.toOption, Bool.false_eq_true, ↓reduceIte]
    refine ⟨by simp [hc], fun k => by simp [hk, hkey], fun k => by simp [hkey, hl]⟩
  · have hany : acc.any (jeq · (getD r "port")) = last.contains k₀ := by
      rw [hk]; congr 1; funext y; rw [jeq_of_canon hx]; rfl
    have hkey : ∀ k, keyIs k (getD r "port") = (k₀ == k) := fun k => by
      simp [keyIs, hx, Except.toOption]
    simp only [Except.toOption, hany]
    refine ⟨?_, fun k => ?_, fun k => ?_⟩
    · cases h0 : last.contains k₀ <;> simp [hc]
    · rw [Std.HashMap.contains_insert]
      cases h0 : last.contains k₀
      · simp only [Bool.false_eq_true, ↓reduceIte, Array.any_push, ← hk, hkey, Bool.or_comm]
      · simp only [↓reduceIte, ← hk]
        by_cases he : k₀ = k
        · subst he; simp [h0]
        · simp [he]
    · rw [Std.HashMap.getElem?_insert, List.find?_cons, hkey, hl]
      by_cases he : k₀ = k
      · simp [he]
      · rw [show (k₀ == k) = false from by simp [he]]; rfl

private theorem inv_foldl (l : List Json) :
    ∀ acc ix seen, Inv acc ix seen → Inv (l.foldl distinctStep acc) (l.foldl PortIndex.add ix) (l.reverse ++ seen) := by
  induction l with
  | nil => intro acc ix seen h; simpa using h
  | cons r l ih =>
    intro acc ix seen h
    simpa [List.foldl_cons, List.reverse_cons, List.append_assoc] using ih _ _ _ (inv_step h r)

private theorem inv_portIndex (roots : Array Json) :
    Inv (roots.foldl distinctStep #[]) (portIndex roots) roots.toList.reverse := by
  unfold portIndex
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  simpa using inv_foldl roots.toList #[] {} [] ⟨rfl, fun k => by simp, fun k => by simp⟩

theorem portIndex_count (roots : Array Json) :
    (portIndex roots).count = (roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]).size :=
  (inv_portIndex roots).1

theorem portIndex_has (roots : Array Json) (p : String) :
    (portIndex roots).has p = (roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]).any (jeq · (Json.str p)) := by
  unfold PortIndex.has PortIndex.find
  rcases hp : Flock.canon (Json.str p) with e | k
  · simp [jeq_of_canon_error hp]
  · show (portIndex roots).last[k]?.isSome = _
    rw [Std.HashMap.isSome_getElem?_eq_contains, (inv_portIndex roots).2.1 k]
    congr 1; funext y; rw [jeq_of_canon hp]; rfl

theorem portIndex_find (roots : Array Json) (p : String) :
    (portIndex roots).find p = roots.reverse.find? (fun r => jeq (getD r "port") (Json.str p)) := by
  unfold PortIndex.find
  rcases hp : Flock.canon (Json.str p) with e | k
  · simp only [jeq_of_canon_error hp]
    exact (Array.find?_eq_none.2 (by simp)).symm
  · show (portIndex roots).last[k]? = _
    rw [(inv_portIndex roots).2.2 k, ← Array.find?_toList, Array.toList_reverse]
    congr 1; funext r; rw [jeq_of_canon hp]; rfl

-- `simp` unfolds both whole bodies, the link-points step included, and each join point of
-- both checks into both of its branches. The link-points step names `c` in both of its branches, so the steps before
-- it are rewritten twice, past `simp`'s default step limit
set_option maxHeartbeats 1600000 in
/-- **R3 at `O(n)` hash-map operations is R3 as first written**: everything a guarantee reads of `check` is `checkRef`'s. -/
theorem check_eq_checkRef (rec : Json) (v : Verifier) : check rec v = checkRef rec v := by
  simp (maxSteps := 800000) only [check, checkRef, portIndex_count, portIndex_has, portIndex_find]

end Spot.Docket
