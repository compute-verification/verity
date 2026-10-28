import NciCheck.Baseline
import CoreProtocol.Protocol

/-!
# The NCI check, a core protocol: `verity/nci/v1`

`NciCheck.Baseline.evaluate`, the NCI check of record (the policy check at the size of the description, then θ's
provenance), as a `Core.Protocol`, so that onsite checks its registration with it (`Core.Onsite.protocol`, a
`Core.Composition.registeredB`). Its case splits the check's
instance and trusted input as sampled proofs' one-stage audit splits its own (`SampledProofs.OneStage`):

* **params**: the verifier's own copies, canonical JSON with exactly the keys `alpha` (`α` as `[num, den]`), `gamma`
  (`Γ`), `log` (the deployment's registration records before the run's own, oldest first, as R6 keeps them; empty for
  the deployment's first run), `parameters` (its program function's `[[name, type], ...]`, which the descriptor does not
  carry) and `program` (the declaration's `verity-ir/descriptor/v1` Program, as text);
* **commitment**: the run's own registration record, canonical JSON, an object, fixed before the run; its digest is
  `SHA-512(format || 0 || canon(record))` (`NciCheck.Baseline.digest`), as one-stage's;
* **evidence**: canonical JSON with exactly the keys `owner` (the declared division into isolation units, one entry per
  root node), `provenance` (one entry per θ entry) and `theta` (θ's root nodes), as `NciCheck.Desc` and
  `NciCheck.Baseline` read them;
* **check**: `NciCheck.Baseline.evaluate` on the instance `{alpha, gamma, owner, program, provenance, theta}`
  (`instanceOf`) and the trusted input `{log: log ++ [record], params: parameters, run: the record's digest}`
  (`trustedOf`). It rejects with the refusal's code (and its fix, where the check has one) when the instance or the
  trusted input is refused, rejects naming the first failing isolation unit and its rule when the division doesn't
  comply, and accepts exactly when it complies. It is never undecided: the check has no rounds and reads no coins;
* on acceptance, **public** is nothing beyond the verdict, `{}` (`Γ`, `α`, the Program and its parameters are the
  params), and the **profile** is the check's report, `{"gates", "ius"}`: the root's gates, and per isolation unit
  `[size, novelReads]`.
-/

namespace NciCheck.Protocol

open Lean (Json)
open Flock (V)

def protocolId : String := "verity/nci/v1"

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

/-- The instance the check of record reads. -/
def instanceOf (p : Params) (e : Evidence) : Json :=
  Json.mkObj [("alpha", p.alpha), ("gamma", p.gamma), ("owner", e.owner), ("program", .str p.program),
    ("provenance", e.provenance), ("theta", e.theta)]

/-- The trusted input the check of record reads: the log through the run's own record, which is the run. -/
def trustedOf (p : Params) (c : Commitment) : Json :=
  Json.mkObj [("log", .arr (p.log.push c.record)), ("params", p.parameters), ("run", .str c.hex)]

/-- A refusal's code for the instance `j`, with the fix its `message` gives where there is one. -/
def reasonOf (j : Json) (code : String) : String :=
  let r := Baseline.refusal j code
  match r.getObjVal? "error", r.getObjVal? "message" with
  | .ok (.str e), .ok (.str m) => s!"{e}: {m}"
  | .ok (.str e), _ => e
  | _, _ => code

/-- Why a verdict that doesn't comply rejects: the first failing isolation unit and its rule. -/
def failureOf (out : Json) : String :=
  match Baseline.getD out "failure" with
  | .arr #[u, .str rule] => s!"the division does not comply: isolation unit {u.compress} fails its {rule} bound"
  | _ => "the division does not comply"

/-- The check's report on a complying division: the root's gates and per isolation unit `[size, novelReads]`. -/
def reportOf (out : Json) : Json :=
  Json.mkObj [("gates", Baseline.getD out "gates"), ("ius", Baseline.getD out "ius")]

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Unit Json :=
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

end NciCheck.Protocol
