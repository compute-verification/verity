module

public import NciCheck.Protocol
public import NciCheck.Record
public import CoreProtocol.Blobs

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

open Lean (Json)
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

end InferenceOnly.RowsProtocol
