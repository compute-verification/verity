module

public import InferenceOnlyVerifier.Protocol
public import InferenceOnlyVerifier.Record
public import InferenceOnlyVerifier.RecordChosen

/-!
# The inference-only policy's check of a circuit record, as a core protocol: `verity/inference-only/record/v1`

The inference-only policy asks that the registered circuit divide into inference blocks of bounded size, each
reading a bounded share of fresh input. v0 (`Protocol`) decides it on a Program that holds every weight it reads and
every wire between its units. An execution phase's Program holds neither: its circuit record declares which unit feeds
which and which rows hold registered data. This protocol decides the rule on that record with the record check
(`Record`), a block counted in the Program's gates that no wire covers. It reads no circuit file and no blob, has no
rounds, reads no coins, and is never undecided.

## The parts

* **params**: the verifier's own copies, canonical JSON with exactly these keys:
  * `gamma`: `Γ`, the most gates a block holds;
  * `alpha`: `α` as `[num, den]`, `den` not 0, the most fresh bits a gate;
  * `log`: the deployment's registration records before the run's own, oldest first, as v0's;
  * `pins`: the closure digests of the Definitions the policy takes as readers of a table opening
    (`Record`'s `pins`);
  * `program`: the Program, a `verity-ir/descriptor/v1` text.
* **commitment**: the run's own registration record, read by v0's decoder (`Protocol.commitmentOf`): canonical
  JSON, an object, with its digest.
* **evidence**: canonical JSON, `{"record"}` or `{"order", "record"}`: the execution phase's circuit record, as the
  text the run's registration record commits, and the order of its unit ranges where the record states none
  (`Record`'s `order`).

## The check

1. The record is read once (`Record.read`), with the params' Program, pins and log and the run's own registration
   record: every stage of the record form, the wires' check among them.
2. **The rule** is `Record.rate` on the record's summary (`Record.summarize`): a block's gates are at most `Γ`, and
   its fresh bits at most `α` times its gates.

An accept publishes nothing (`{}`). Its profile is `{"blocks", "chosen", "fresh", "gates"}`: the blocks, the bits
the prover chose over the execution phase (`RecordChosen.chosen`), and a block's fresh bits and gates. The four are
counts of the record, which is private: where a system shows a profile, they are shown.

## What it does not hold

* Any value of `Γ` and `α`: both are params.
* What `Record` does not: its docstring says what an accepted record leaves to SPOT's proofs and to registration.

## What rejects, in the order checked

A reject's reason opens with the name of what refused, and a colon follows where it says more.

1. `params: `, `commitment: `, `evidence: `: a part does not decode as stated above (`Core.Protocol.answer`
   writes the prefix).
2. `Record.read`'s own, in its order: `instance`, `params`, `program`, `not-topological`, `record`, `trusted`,
   `run-program`, `run-circuit`, `templates`, `own-rows`, `wires`, `leaf-widths`, `inputs`, `blocks`,
   `reader-layout`, `reader-roles`. The first two cannot arise: the decoders hold `gamma`, `alpha` and `pins` to
   the form `Record.parse` asks, and the instance has every field.
3. `size`, then `inference_only`: the rule. A block has more than `Γ` gates; else its fresh bits exceed `α` times its
   gates.
-/

@[expose] public section

namespace InferenceOnly.RecordProtocol

open Lean (Json toJson)
open Flock (V)
open InferenceOnlyVerifier

def protocolId : String := "verity/inference-only/record/v1"

/-! ## The parts -/

structure Params where
  /-- `Γ`: the most gates a block holds. -/
  gamma : Nat
  /-- `α`'s numerator: `α = num/den` is the most fresh bits a gate. -/
  num : Nat
  /-- `α`'s denominator, not 0. -/
  den : Nat
  log : Array Json
  /-- The closure digests of the pinned reader Definitions, each a SHA-512 in lower-case hex. -/
  pins : Array String
  program : String

structure Evidence where
  /-- The execution phase's circuit record, as text. -/
  record : String
  /-- The order of the record's unit ranges, read where the record states none. -/
  order : Option Json

def PARAMS_KEYS : Array String := #["alpha", "gamma", "log", "pins", "program"]

def ALPHA_FORM : String := "the params' alpha is not [num, den], two natural numbers, den not 0"
def PINS_FORM : String := "the params' pins are not a list of SHA-512s in lower-case hex"

/-- One pin: a SHA-512 in lower-case hex. -/
def pinOf? : Json → Option String
  | .str s => if Record.isDigest s then some s else none
  | _ => none

/-- The params. `gamma`, `alpha` and `pins` are held to the form `Record.parse` asks of them. -/
def paramsOf (j : Json) : V Params := do
  if Theta.keys? j != some PARAMS_KEYS then
    throw s!"the keys of the params are not {", ".intercalate PARAMS_KEYS.toList}"
  let .str program := Theta.getD j "program" | throw "the params' program is not a string (the descriptor's text)"
  let .arr log := Theta.getD j "log" | throw "the params' log is not a list of registration records"
  let some gamma := natOf? (Theta.getD j "gamma") | throw "the params' gamma is not a natural number"
  let some (a, b) := pairOf? (Theta.getD j "alpha") | throw ALPHA_FORM
  let some num := natOf? a | throw ALPHA_FORM
  let some den := natOf? b | throw ALPHA_FORM
  if den = 0 then throw ALPHA_FORM
  let .arr pins := Theta.getD j "pins" | throw PINS_FORM
  let some pins := pins.mapM pinOf? | throw PINS_FORM
  pure { gamma, num, den, log, pins, program }

/-- The evidence: the circuit record's text, and the order where it is given. -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  let ks := Theta.keys? j
  if ks != some #["record"] && ks != some #["order", "record"] then
    throw "the keys of the evidence are not record, or order and record"
  let .str record := Theta.getD j "record"
    | throw "the evidence's record is not a string (the circuit record's text)"
  pure { record, order := (j.getObjVal? "order").toOption }

/-! ## The check -/

/-- A failure of the rule, by `Record.rate`'s code. -/
def failureOf (code : String) : String :=
  if code == "size" then "size: a block has more than Γ gates"
  else "inference_only: a block reads more than α fresh bits a gate"

/-- The instance `Record.read` reads, with the params' `gamma` and `alpha`. -/
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

/-- The accepted check's report: the blocks, a block's gates and its fresh bits, and the bits the prover chose over
the execution phase. -/
def reportOf (blocks gates fresh chosen : Nat) : Json :=
  Json.mkObj [("blocks", toJson blocks), ("chosen", toJson chosen), ("fresh", toJson fresh), ("gates", toJson gates)]

/-- The check, in the module docstring's order. The record is read once (`Record.read`), and the summary, the rule
and the count of what the prover chose are taken of that one reading. -/
def check (p : Params) (c : Protocol.Commitment) (e : Evidence) : Core.Outcome Unit Json :=
  let j := instanceOf p e
  let tr := trustedOf p c
  match Record.read j tr with
  | .error code => .reject (reasonOf j tr code)
  | .ok r =>
    let s := Record.summarize r
    match Record.rate p.gamma p.num p.den s with
    | some code => .reject (failureOf code)
    | none => .accept () (reportOf s.blocks s.gates s.fresh (RecordChosen.chosen (RecordChosen.executionPhaseBits r)))

def protocol : Core.Protocol where
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
  check := check
  publicJson _ := Json.mkObj []
  profileJson := id

end InferenceOnly.RecordProtocol
