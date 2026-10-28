import NciCheck.Baseline
import CoreProtocol.Protocol

/-!
# The inference-only policy as a core protocol: `verity/inference-only/v0`

The inference-only policy (`Nci.Complies Γ α`, `Security/Definitions/Nci.lean`) asks that the declared circuit divide
into inference blocks of at most `Γ` gates, each reading at most `α` bits of fresh input per gate: input from outside
the block that is not registered information. `NciCheck.Baseline.evaluate` is the policy's check: the division's check
at the size of the description (`NciCheck.Desc.evaluate`), then the provenance of the registered information against
the deployment's registration log. This module makes that check a `Core.Protocol`, which onsite runs as its
registration check, not as a component. v0 publishes the division and isn't zero-knowledge, so no composite may run it;
only v1, the private form, may be composed. It has no rounds and reads no coins: it answers from the declared circuit,
the declared division and the registered information it is handed, and is never undecided.

Its case splits the check's instance and trusted input as sampled proofs' one-stage audit splits its own
(`SampledProofs.OneStage`):

* **params**: the verifier's own copies, canonical JSON with exactly the keys `alpha` (`α` as `[num, den]`), `gamma`
  (`Γ`), `log` (the deployment's registration records before the run's own, oldest first, each one's `previous` the
  digest of the one before; empty for the deployment's first run), `parameters` (the declared circuit's program
  function's `[[name, type], ...]`, which the descriptor does not carry) and `program` (the declared circuit, a
  `verity-ir/descriptor/v1` Program, as text);
* **commitment**: the run's own registration record, canonical JSON, an object, fixed before the run: the record whose
  digest sampled proofs' one-stage audit commits to. Its digest is `SHA-512(format || 0 || canon(record))`
  (`NciCheck.Baseline.digest`, as one-stage's). Its `program` must be the declared circuit's SHA-512 (`names`), and
  θ's provenance reads the roots its `registered` entries give the program function's parameters;
* **evidence**: canonical JSON with exactly the keys `owner` (the declared division: per root node of the declared
  circuit, its inference block or `"input"`), `provenance` (per `theta` entry, the registration record, value and root
  it was registered under) and `theta` (the root input nodes that hold registered information), as `NciCheck.Desc` and
  `NciCheck.Baseline` read them;
* **check**: first that the record names the declared circuit, its `program` the circuit's SHA-512 (`names`), so that
  the commitment binds the circuit whether or not the case has registered information; then
  `NciCheck.Baseline.evaluate` on the instance `{alpha, gamma, owner, program, provenance, theta}` (`instanceOf`) and
  the trusted input `{log: log ++ [record], params: parameters, run: the record's digest}` (`trustedOf`). It rejects
  when the record names another circuit, with the check's refusal code (and its fix, where the check has one) when the
  instance or the trusted input is refused, and naming the first inference block over its budget when the division
  doesn't comply, and accepts exactly when it complies;
* on acceptance, **public** is nothing beyond the verdict, `{}` (`Γ`, `α`, the declared circuit and its parameters are
  the params), and the **profile** is the check's report, `{"blocks", "gates"}`: per inference block
  `[gates, fresh-input bits]`, and the declared circuit's gates.
-/

namespace NciCheck.Protocol

open Lean (Json)
open Flock (V)

def protocolId : String := "verity/inference-only/v0"

structure Params where
  alpha : Json
  gamma : Json
  log : Array Json
  parameters : Json
  program : String

/-- The run's own registration record, with its digest as `NciCheck.Baseline.digest` writes it and as bytes. -/
structure Commitment where
  record : Json
  hex : String
  digest : ByteArray

structure Evidence where
  owner : Json
  provenance : Json
  theta : Json

private def exactly (j : Json) (ks : Array String) (what : String) : V Unit :=
  if Baseline.keys? j != some ks then throw s!"the keys of {what} are not {", ".intercalate ks.toList}"
  else pure ()

def PARAMS_KEYS : Array String := #["alpha", "gamma", "log", "parameters", "program"]
def EVIDENCE_KEYS : Array String := #["owner", "provenance", "theta"]

def paramsOf (j : Json) : V Params := do
  exactly j PARAMS_KEYS "the params"
  let .str program := Baseline.getD j "program" | throw "the params' program is not a string (the descriptor's text)"
  let .arr log := Baseline.getD j "log" | throw "the params' log is not a list of registration records"
  pure { alpha := Baseline.getD j "alpha", gamma := Baseline.getD j "gamma", log,
         parameters := Baseline.getD j "parameters", program }

def commitmentOf (b : ByteArray) : V Commitment := do
  let record ← Core.canonicalJson b
  unless record matches .obj _ do throw "the registration record is not an object"
  let hex ← Baseline.digest record
  pure { record, hex, digest := ← Flock.unhex hex }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  exactly j EVIDENCE_KEYS "the evidence"
  pure { owner := Baseline.getD j "owner", provenance := Baseline.getD j "provenance",
         theta := Baseline.getD j "theta" }

/-- The instance the policy's check reads. -/
def instanceOf (p : Params) (e : Evidence) : Json :=
  Json.mkObj [("alpha", p.alpha), ("gamma", p.gamma), ("owner", e.owner), ("program", .str p.program),
    ("provenance", e.provenance), ("theta", e.theta)]

/-- The trusted input the policy's check reads: the log through the run's own record, which is the run. -/
def trustedOf (p : Params) (c : Commitment) : Json :=
  Json.mkObj [("log", .arr (p.log.push c.record)), ("params", p.parameters), ("run", .str c.hex)]

/-- A refusal's code for the instance `j`, with the fix its `message` gives where there is one. -/
def reasonOf (j : Json) (code : String) : String :=
  let r := Baseline.refusal j code
  match r.getObjVal? "error", r.getObjVal? "message" with
  | .ok (.str e), .ok (.str m) => s!"{e}: {m}"
  | .ok (.str e), _ => e
  | _, _ => code

/-- Why a verdict that doesn't comply rejects: the first inference block over its budget, and which budget. -/
def failureOf (out : Json) : String :=
  match Baseline.getD out "failure" with
  | .arr #[u, .str "size"] => s!"the division does not comply: inference block {u.compress} has more than Γ gates"
  | .arr #[u, .str _] =>
    s!"the division does not comply: inference block {u.compress} reads more than α bits of fresh input per gate"
  | _ => "the division does not comply"

/-- The check's report on a complying division: per inference block `[gates, fresh-input bits]`, and the declared
circuit's gates. -/
def reportOf (out : Json) : Json :=
  Json.mkObj [("blocks", Baseline.getD out "ius"), ("gates", Baseline.getD out "gates")]

/-- The run's record names the declared circuit: its `program` is the circuit's SHA-512 (`Prog.sha512`), what sampled
proofs' one-stage audit asks of its own record and program (R2-program). θ's provenance asks it only of a case with
registered information; here every case is bound to its circuit. A program that doesn't decode is left to the check,
which refuses it (`program`). -/
def names (p : Params) (c : Commitment) : Bool :=
  match Flock.Program.decode p.program.toUTF8 with
  | .ok P => Baseline.strAt c.record "program" == some (Flock.hexOf P.sha512)
  | .error _ => true

def UNNAMED : String :=
  "the run's registration record is not the declared circuit's: its program is not the circuit's SHA-512"

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Unit Json :=
  if !names p c then .reject UNNAMED else
  match Baseline.evaluate (instanceOf p e) (trustedOf p c) with
  | .error code => .reject (reasonOf (instanceOf p e) code)
  | .ok out =>
    match out.getObjVal? "complies" with
    | .ok (.bool true) => .accept () (reportOf out)
    | _ => .reject (failureOf out)

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Unit
  Profile := Json
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check := check
  publicJson _ := Json.mkObj []
  profileJson := id

private def answer (params commitment evidence : String) : Core.Answer :=
  protocol.answer params.toUTF8 commitment.toUTF8 evidence.toUTF8

#guard (answer "{}" "{}" "{}").reason ==
  some "params: the keys of the params are not alpha, gamma, log, parameters, program"
#guard (answer "{\"alpha\":[1,1],\"gamma\":1,\"log\":[],\"parameters\":[],\"program\":\"\"}" "[]" "{}").reason ==
  some "commitment: the registration record is not an object"
#guard (answer "{\"alpha\":[1,1],\"gamma\":1,\"log\":[],\"parameters\":[],\"program\":\"\"}" "{\"previous\":null}"
  "{\"owner\":[],\"provenance\":[],\"theta\":[]}").reason == some "program"
#guard (answer "{\"alpha\":[1,1],\"gamma\":1,\"log\":[],\"parameters\":[],\"program\":\"\"}" "{\"previous\":null}"
  "{\"owner\":[],\"provenance\":[],\"theta\":[]}").digest.isSome
#guard failureOf (Json.mkObj [("failure", .arr #[Json.num 0, .str "size"])]) ==
  "the division does not comply: inference block 0 has more than Γ gates"
#guard failureOf (Json.mkObj [("failure", .arr #[Json.num 2, .str "nci"])]) ==
  "the division does not comply: inference block 2 reads more than α bits of fresh input per gate"

end NciCheck.Protocol
