module

public import NciCheck.Protocol
public import NciCheck.RecordRows
public import NciCheck.RecordChosen
public import CoreProtocol.Blobs
public import Spot.OneStageZk
public import Spot.FilePart
public import Std.Data.HashMap

/-!
# The inference-only policy's check in rows, as a core protocol: `verity/inference-only/rows/v1`

The inference-only policy asks that the registered circuit divide into inference blocks of bounded size, each
reading a bounded share of fresh input. v0 (`Protocol`) decides it on a Program that holds every weight it reads and
every wire between its units, counted in the Program's gates. An execution phase's Program holds neither, and its
units are proved from their templates' circuit files, a row of a file at a time. This protocol decides the rule on
an execution phase's circuit record (`Record`), with a block counted in the rows of its templates' circuit files
(`RecordRows`). It is a `Core.BlobProtocol`: the circuit files are its blobs. It has no rounds and reads no coins,
and it is never undecided.

## The parts

* **params**: the verifier's own copies, canonical JSON with exactly these keys:
  * `gamma`: `Γ`, the most rows a block holds;
  * `alpha`: `α` as `[num, den]`, `den` not 0, the most fresh bits a row;
  * `circuits`: `[[template id, SHA-512], ...]`, one pair for each template of the record, in the record's
    `templates` order, the digest that of the template's circuit file, in lower-case hex;
  * `log`: the deployment's registration records before the run's own, oldest first, as v0's;
  * `pins`: the closure digests of the Definitions the policy takes as readers of a table opening
    (`Record`'s `pins`);
  * `program`: the Program, a `verity-ir/descriptor/v1` text.
* **commitment**: the run's own registration record, read by v0's decoder (`Protocol.commitmentOf`): canonical
  JSON, an object, with its digest.
* **evidence**: canonical JSON, `{"record"}` or `{"order", "record"}`: the execution phase's circuit record, as the
  text the run's registration record commits, and the order of its unit ranges where the record states none
  (`Record`'s `order`).
* **blobs**: each template's circuit file, and the pinned lookup tables a file looks up. A blob is read only under
  a digest `circuits` names, or under a pinned table's own.

The rows table is no part: each template's rows of one unit are read from the file `circuits` names for it. `Γ`
and `α` are in rows, and the rule in the Program's gates (`Record.rate`) is not decided here.

## The check

1. The record is read once (`Record.read`), with the params' Program, pins and log and the run's own registration
   record: every stage of the record form, the wires' check among them.
2. `circuits` is held to the record's templates.
3. **A template's rows of one unit** are `Spot.FilePart.ofCircuit`'s `m` on the circuit its file holds, the file
   read as SPOT's one-stage check reads the file of a statement it verifies (`circuitOf`): the unit circuit's
   computed rows, stacked `units_per_vu` times. So a file is parsed whole, each of its rows once, and that is this
   check's cost beside the record's: the files' size, one parse a template.
4. **A block's rows** are the sum over the templates of a template's units in a block times its rows of one unit
   (`RecordRows.ofSummary` on `Record.summarize`), and a block's fresh bits are the summary's.
5. **The rule** is `RecordRows.rate` on that one pair: every block has the same rows, and the summary's fresh bits
   are the greatest over the blocks.

An accept publishes nothing (`{}`). Its profile is `{"blocks", "chosen", "fresh", "rows"}`: the blocks, the bits
the prover chose over the execution phase (`RecordChosen.chosen`), and a block's fresh bits and rows. The four are
counts of the record, which is private: where a system shows a profile, they are shown.

## What it does not hold

* That a file `circuits` names is the file SPOT's proofs of that template's units are verified against. Inside
  this protocol a file with another SHA-512 is never read; that `circuits` is the list SPOT's check takes is a tie
  between two protocols' params, stated where the two are composed.
* The rule in the Program's gates, and any value of `Γ` and `α`: both are params.
* What `Record` does not: its docstring says what an accepted record leaves to SPOT's proofs and to registration.

## What rejects, in the order checked

A reject's reason opens with the name of what refused, and a colon follows where it says more.

1. `params: `, `commitment: `, `evidence: `: a part does not decode as stated above (`Core.Protocol.answer`
   writes the prefix).
2. `Record.read`'s own, in its order: `instance`, `params`, `program`, `not-topological`, `record`, `trusted`,
   `run-program`, `run-circuit`, `templates`, `own-rows`, `wires`, `leaf-widths`, `inputs`, `blocks`,
   `reader-layout`, `reader-roles`. The first two cannot arise: the decoders hold `gamma`, `alpha` and `pins` to
   the form `Record.parse` asks, and the instance has every field.
3. `circuits`: `circuits` does not name the record's templates, each once, in the record's order.
4. `circuit-file`: for a template, in the record's order, no blob has the digest `circuits` names, or the file
   cannot be counted. A file that cannot be counted is never counted as no rows.
5. `size`, then `nci`: the rule in rows. A block has more than `Γ` rows; else its fresh bits exceed `α` times its
   rows.
-/

@[expose] public section

namespace InferenceOnly.RowsProtocol

open Lean (Json toJson)
open Flock (V)
open NciCheck

def protocolId : String := "verity/inference-only/rows/v1"

/-! ## The parts -/

structure Params where
  /-- `Γ`: the most rows a block holds. -/
  gamma : Nat
  /-- `α`'s numerator: `α = num/den` is the most fresh bits a row. -/
  num : Nat
  /-- `α`'s denominator, not 0. -/
  den : Nat
  /-- Per template of the record, in the record's order: its id and its circuit file's SHA-512 in lower-case hex. -/
  circuits : Array (String × String)
  log : Array Json
  /-- The closure digests of the pinned reader Definitions, each a SHA-512 in lower-case hex. -/
  pins : Array String
  program : String

structure Evidence where
  /-- The execution phase's circuit record, as text. -/
  record : String
  /-- The order of the record's unit ranges, read where the record states none. -/
  order : Option Json

def PARAMS_KEYS : Array String := #["alpha", "circuits", "gamma", "log", "pins", "program"]

def ALPHA_FORM : String := "the params' alpha is not [num, den], two natural numbers, den not 0"
def PINS_FORM : String := "the params' pins are not a list of SHA-512s in lower-case hex"
def CIRCUITS_FORM : String :=
  "the params' circuits are not a list of [template id, SHA-512 in lower-case hex] pairs"

/-- One pin: a SHA-512 in lower-case hex. -/
def pinOf? : Json → Option String
  | .str s => if Record.isDigest s then some s else none
  | _ => none

/-- One entry of `circuits`: a template's id and its circuit file's SHA-512. -/
def entryOf? : Json → Option (String × String)
  | .arr #[.str tid, .str digest] => if Record.isDigest digest then some (tid, digest) else none
  | _ => none

/-- The params. `gamma`, `alpha` and `pins` are held to the form `Record.parse` asks of them. -/
def paramsOf (j : Json) : V Params := do
  if Baseline.keys? j != some PARAMS_KEYS then
    throw s!"the keys of the params are not {", ".intercalate PARAMS_KEYS.toList}"
  let .str program := Baseline.getD j "program" | throw "the params' program is not a string (the descriptor's text)"
  let .arr log := Baseline.getD j "log" | throw "the params' log is not a list of registration records"
  let some gamma := natOf? (Baseline.getD j "gamma") | throw "the params' gamma is not a natural number"
  let some (a, b) := pairOf? (Baseline.getD j "alpha") | throw ALPHA_FORM
  let some num := natOf? a | throw ALPHA_FORM
  let some den := natOf? b | throw ALPHA_FORM
  if den = 0 then throw ALPHA_FORM
  let .arr pins := Baseline.getD j "pins" | throw PINS_FORM
  let some pins := pins.mapM pinOf? | throw PINS_FORM
  let .arr circuits := Baseline.getD j "circuits" | throw CIRCUITS_FORM
  let some circuits := circuits.mapM entryOf? | throw CIRCUITS_FORM
  pure { gamma, num, den, circuits, log, pins, program }

/-- The evidence: the circuit record's text, and the order where it is given. -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  let ks := Baseline.keys? j
  if ks != some #["record"] && ks != some #["order", "record"] then
    throw "the keys of the evidence are not record, or order and record"
  let .str record := Baseline.getD j "record"
    | throw "the evidence's record is not a string (the circuit record's text)"
  pure { record, order := (j.getObjVal? "order").toOption }

/-! ## The rejects of the check -/

/-- `circuits` does not name the record's templates. -/
def CIRCUITS : String :=
  "circuits: the params' circuits do not name the record's templates, each once, in the record's order"

/-- The template `tid`'s circuit file is not read: `why` says what refused it. -/
def circuitFile (tid why : String) : String := s!"circuit-file: template {tid}: {why}"

/-- No blob is the file `circuits` names. -/
def noBlob (digest : String) : String := s!"no blob has the SHA-512 {digest}"

/-- A file whose unit has no computed row. No file the reader takes is one; it is refused and not counted. -/
def NO_ROWS : String := "one unit of the file has no computed row"

/-- A failure of the rule in rows, by `RecordRows.rate`'s code. -/
def failureOf (code : String) : String :=
  if code == "size" then "size: a block has more than Γ rows"
  else "nci: a block reads more than α fresh bits a row"

/-! ## The record -/

/-- The instance `Record.read` reads. Its `gamma` and `alpha` are the params', in rows: `Record.read` holds them to
their form and applies neither. -/
def instanceOf (p : Params) (e : Evidence) : Json :=
  Json.mkObj ([("alpha", Json.arr #[toJson p.num, toJson p.den]), ("gamma", toJson p.gamma),
    ("pins", Json.arr (p.pins.map Json.str)), ("program", Json.str p.program), ("record", Json.str e.record)] ++
    match e.order with
    | some o => [("order", o)]
    | none => [])

/-- The trusted input `Record.read` reads: the log through the run's own record, which is the run. Its `params` is
not read. -/
def trustedOf (p : Params) (c : Protocol.Commitment) : Json :=
  Json.mkObj [("log", Json.arr (p.log.push c.record)), ("params", Json.arr #[]), ("run", Json.str c.hex)]

/-- A reject by `Record.read`'s refusal `code`, with what to fix where the refusal says it (`Record.messageOf`). -/
def reasonOf (j tr : Json) (code : String) : String :=
  match Record.messageOf j tr code with
  | some m => s!"{code}: {m}"
  | none => code

/-- Whether `circuits` names the templates `ts`, each once, in order. -/
def namesTemplates (circuits : Array (String × String)) (ts : List RecordText.Tmpl) : Bool :=
  circuits.toList.map (·.1) == ts.map (·.tid)

/-! ## The circuit files -/

/-- The circuit a file holds, read as SPOT's one-stage check reads the file of a statement it verifies
(`Spot.OneStageZk.run`): `Flock.ProvedScope.circuitOf` under that statement's tags (`Spot.OneStageZk.STATEMENT`)
with one session's hooks at the file's own mask log (`Flock.Witness.zkTags`), `tables` the pinned lookup tables the
file may look up. The file is parsed whole: every row of every circuit it holds, the row commitment's circuits
held to the verifier's own. A file in another statement's form is refused. -/
def circuitOf (tables : List (String × ByteArray)) (file : ByteArray) : V Flock.Circuit := do
  let some tags := Flock.Tags.find? Spot.OneStageZk.STATEMENT
    | throw s!"the statement {Spot.OneStageZk.STATEMENT} is not known"
  Flock.ProvedScope.circuitOf (Flock.Zk.tags tags (Flock.Zk.maskLogOf file) 1) file tables

/-- **The rows of one unit of a file's template**: `Spot.FilePart.ofCircuit`'s `m` on the circuit the file holds
(`circuitOf`), the unit circuit's computed rows, stacked `units_per_vu` times. Refused where the reader refuses the
file, and where the count would be 0. -/
def rowsOf (tables : List (String × ByteArray)) (file : ByteArray) : V Nat := do
  let m := (Spot.FilePart.ofCircuit (← circuitOf tables file) 1).m
  if m = 0 then throw NO_ROWS
  pure m

/-- The blobs by their SHA-512 in lower-case hex. -/
def filesOf (bs : Core.Blobs) : Std.HashMap String ByteArray :=
  bs.entries.foldl (fun files (d, b) => files.insert (Flock.hexOf d) b) {}

/-- **The rows table**: per template of `circuits`, by its id, the rows of one unit of the file its digest names
among the blobs (`rowsOf`); or the reject for the first template whose file is no blob or is not counted. One lookup
and one file a template, the blobs indexed once. The cost is one parse for each template that names a file, so a file
two templates name is parsed twice; a memo by digest is the mend if a record ever shares a file between templates. -/
def tableOf (circuits : Array (String × String)) (bs : Core.Blobs) : Except String (Std.HashMap String Nat) :=
  let files := filesOf bs
  let tables := Spot.OneStageZk.tablesOf bs
  circuits.foldlM (init := {}) fun table (tid, digest) =>
    match files[digest]? with
    | none => throw (circuitFile tid (noBlob digest))
    | some file => match rowsOf tables file with
      | .ok m => pure (table.insert tid m)
      | .error why => throw (circuitFile tid why)

/-! ## The check -/

/-- The accepted check's report: the blocks, a block's rows and its fresh bits, and the bits the prover chose over
the execution phase. -/
def reportOf (blocks rows fresh chosen : Nat) : Json :=
  Json.mkObj [("blocks", toJson blocks), ("chosen", toJson chosen), ("fresh", toJson fresh), ("rows", toJson rows)]

/-- The check, in the module docstring's order. The record is read once (`Record.read`), and the summary, the rule
and the count of what the prover chose are taken of that one reading. With `circuits` held to the record's
templates, the table has every template's rows, so a block's rows are counted (`RecordRows.ofSummary` gives none
only for a template the table lacks). -/
def checkBlobs (p : Params) (c : Protocol.Commitment) (e : Evidence) (bs : Core.Blobs) : Core.Outcome Unit Json :=
  let j := instanceOf p e
  let tr := trustedOf p c
  match Record.read j tr with
  | .error code => .reject (reasonOf j tr code)
  | .ok r =>
    if !namesTemplates p.circuits r.f.ts then .reject CIRCUITS else
    match tableOf p.circuits bs with
    | .error why => .reject why
    | .ok table =>
      let s := Record.summarize r
      match RecordRows.ofSummary r.f.P (table[·]?) s with
      | none => .reject CIRCUITS
      | some rows =>
        match RecordRows.rate p.gamma p.num p.den rows s.fresh with
        | some code => .reject (failureOf code)
        | none =>
          .accept () (reportOf s.blocks rows s.fresh (RecordChosen.chosen (RecordChosen.executionPhaseBits r)))

def protocol : Core.BlobProtocol where
  id := protocolId
  Params := Params
  Commitment := Protocol.Commitment
  Evidence := Evidence
  Public := Unit
  Profile := Json
  params := paramsOf
  commitment := Protocol.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson _ := Json.mkObj []
  profileJson := id
  checkBlobs := checkBlobs

end InferenceOnly.RowsProtocol
