/-
A composition of core protocols, the common part of onsite and remote Verity: one `Core.Protocol` whose check runs each
component's own `answer` on that component's case, and decides by reject dominance.

* It rejects when any component rejects, whatever the others answer, so an undecided component never hides a rejection.
  It also rejects when the cases don't fit the composition (a component unknown, missing, repeated or out of order, or a
  case run under parameters other than the composition's), and when the commitment doesn't bind the components' digests.
* It is undecided when none of that holds and some component is undecided.
* It accepts when every component accepts. It then makes public each component's public outputs and keeps each one's
  profile, both keyed by the component's id. Nothing is public otherwise.

Its parameters name each component with its parameters, exactly the composition's components in the order of their ids:
`{"components":[{"id":ID,"params":PARAMS},...]}`. Its commitment is the 64-byte `binding` of the components' digests,
which the prover sends before the challenge; it is its own digest. Its evidence is each component's case, the line that
component's own `serve` reads, under its id, in the same order:
`{"components":[{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS},...]}`. All three are canonical JSON or bytes,
and an object with a key missing or a key more is refused.
-/
import CoreProtocol.Protocol
import CoreProtocol.Protocol.Canonical
import Flock.Hash

namespace Core.Composition

open Lean (Json)
open Flock (V)

/-- A component as the parameters name it: its id and its parameters, as canonical JSON text. -/
structure Named where
  id : String
  params : String
  deriving BEq, Repr

/-- A component's case: its id and the three things its `answer` reads, the parameters as canonical JSON text. -/
structure Case where
  id : String
  params : String
  commitment : ByteArray
  evidence : ByteArray

/-- Each component's id with its answer, in the composition's order. -/
abbrev Answers := List (String × Answer)

/-- Why `got`, the ids the parameters or the evidence name, are not `want`, the composition's: the first unknown one,
the first missing one, or else their order (a repeat, or ids out of order). -/
def misfit (what : String) (want got : List String) : String :=
  match got.find? (fun i => !want.contains i) with
  | some i => s!"{what}: {i} is not a component of this composition"
  | none =>
    match want.find? (fun i => !got.contains i) with
    | some i => s!"{what}: {i} is missing"
    | none => s!"{what}: name each component once, in the order of their ids ({", ".intercalate want})"

/-- Each component's answer on its case, under the parameters the composition names for it. -/
def answersOf (cs : List Protocol) (ps : List Named) (ev : List Case) : Answers :=
  (cs.zip (ps.zip ev)).map fun (P, n, k) => (P.id, P.answer n.params.toUTF8 k.commitment k.evidence)

/-- Each component's answer on its case, or why the cases don't fit the composition: the parameters and the evidence must
name exactly the components `cs`, in their order, and each case must carry the parameters the composition names for it. -/
def run (cs : List Protocol) (ps : List Named) (ev : List Case) : V Answers :=
  let want := cs.map (·.id)
  if ps.map (·.id) ≠ want then .error (misfit "params" want (ps.map (·.id)))
  else if ev.map (·.id) ≠ want then .error (misfit "evidence" want (ev.map (·.id)))
  else
    match (cs.zip (ps.zip ev)).find? (fun (_, n, k) => k.params ≠ n.params) with
    | some (P, _, _) => .error s!"evidence: {P.id}'s case carries params other than the composition's"
    | none => .ok (answersOf cs ps ev)

/-- The commitment the composition's prover sends: SHA-512 over the composition's id, a zero byte, and the canonical JSON
`{"components":[{"commitment":HEX,"id":ID},...]}` of its components' digests in order. -/
def bind (id : String) (digests : List (String × ByteArray)) : ByteArray :=
  let body := "{\"components\":[" ++ ",".intercalate (digests.map fun (i, d) =>
    "{\"commitment\":" ++ Flock.canonStr (Flock.hexOf d) ++ ",\"id\":" ++ Flock.canonStr i ++ "}") ++ "]}"
  Flock.Sha512.hash (id.toUTF8 ++ ByteArray.mk #[0] ++ body.toUTF8)

/-- `bind` of the answers' digests; `none` when a component has no digest: its commitment didn't decode, and it
rejected. -/
def binding (id : String) (answers : Answers) : Option ByteArray :=
  (answers.mapM fun ((i, a) : String × Answer) => a.digest.map (i, ·)).map (bind id)

private def reasons (xs : Answers) : String :=
  "; ".intercalate (xs.map fun (i, a) => s!"{i} {a.verdict.name}: {a.reason.getD ""}")

/-- Public outputs and profiles, each keyed by the component's id. -/
abbrev Keyed := List (String × Json)

/-- The verdict over the components' answers, reject first: any rejection rejects, then a commitment that doesn't bind
their digests, then any undecided component leaves it undecided; otherwise every component accepted. -/
def judge (id : String) (c : ByteArray) (answers : Answers) : Outcome Keyed Keyed :=
  if answers.any (·.2.verdict = .reject) then .reject (reasons (answers.filter (·.2.verdict = .reject)))
  else if binding id answers ≠ some c then .reject "the commitment does not bind the components' digests"
  else if answers.any (·.2.verdict = .undecided) then .undecided (reasons (answers.filter (·.2.verdict = .undecided)))
  else .accept (answers.map fun (i, a) => (i, a.published.getD .null)) (answers.map fun (i, a) => (i, a.profile.getD .null))

def check (id : String) (cs : List Protocol) (ps : List Named) (c : ByteArray) (ev : List Case) : Outcome Keyed Keyed :=
  match run cs ps ev with
  | .error r => .reject r
  | .ok answers => judge id c answers

private def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

/-- `j` is an object with exactly the keys `want` (sorted, as canonical JSON writes them). -/
private def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

private def entries (j : Json) (ctx : String) : V (List Json) := do
  exactly j ["components"] ctx
  pure (← Flock.J.arr j "components" ctx).toList

def decodeParams (j : Json) : V (List Named) := do
  (← entries j "the composition's params").mapM fun x => do
    exactly x ["id", "params"] "a component's params"
    pure { id := ← Flock.J.str x "id" "a component", params := ← Flock.canon (← Flock.J.get x "params" "a component") }

def decodeCommitment (b : ByteArray) : V ByteArray :=
  if b.size = 64 then .ok b else .error s!"the binding of the components' digests is 64 bytes, not {b.size}"

/-- Read in place (`Core.Canonical`), as `canonicalJson` and `Flock.unhex` would read it: a component's evidence can be
gigabytes of hex (PoUS's `C*`). -/
def decodeEvidence (b : ByteArray) : V (List Case) := do
  (← entries (← Canonical.read b).1 "the composition's evidence").mapM fun x => do
    exactly x ["commitment", "evidence", "id", "params"] "a component's case"
    pure { id := ← Flock.J.str x "id" "a case", params := ← Flock.canon (← Flock.J.get x "params" "a case"),
           commitment := ← Canonical.unhex (← Flock.J.str x "commitment" "a case"),
           evidence := ← Canonical.unhex (← Flock.J.str x "evidence" "a case") }

/-- The composition `id` of the components `cs`, which must be listed in the order of their ids, each once. -/
def product (id : String) (cs : List Protocol) : Protocol where
  id := id
  Params := List Named
  Commitment := ByteArray
  Evidence := List Case
  Public := Keyed
  Profile := Keyed
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest c := c
  check := check id cs
  publicJson := Json.mkObj
  profileJson := Json.mkObj

/-- The components' ids are strictly increasing, which `product` requires of its list: each product checks it by
`#guard`. -/
def ordered (cs : List Protocol) : Bool :=
  let ids := cs.map (·.id)
  (ids.zip ids.tail).all fun (a, b) => a < b

end Core.Composition
