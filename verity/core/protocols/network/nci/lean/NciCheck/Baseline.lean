import NciCheck.Desc

/-!
# θ's provenance: the baseline is registered

`NciCheck.Desc.evaluate` takes θ as given: every entry is a registered input node (`theta-not-registered`), but nothing
says what value it holds, so a developer could put values computed on unmonitored hardware in θ and read them for free.
`Nci.Assumptions.BaselineRegistered` states what θ must hold: registered public data, registered at the start of the
deployment. An output of an earlier run, certified or not, is never θ (registration happens at the start
of a deployment and essentially never changes). One registration log is one deployment, and θ holds what its first
record registers: the first run's θ cites its own record, which opens the log, and every
later run's cites that same record, so new public data takes a new log, a new deployment the auditor sees as such.
`evaluate` checks it: the policy check (`NciCheck.Desc.evaluate`) first, unchanged, then each θ entry's provenance
against what the proof service holds.

## The instance and the trusted input

The instance is `NciCheck.Desc`'s with one more field:
* `provenance`: one object per θ entry, in θ's order, each `{"kind": "registered", "registration": D, "value": V,
  "root": R}`: registered public data, the value `V` that the registration record with digest `D`, the log's first,
  registers (its `registered` list, `verity/registration/v2`) under the root `R`. There is no other kind.

The trusted input, beside the instance, is what the service holds, never the prover's word: an object of exactly
* `log`: the deployment's registration records, oldest first, as the service's R6 keeps them (each record's `previous`
  the digest of the one before, `null` for the first, which opens the deployment; the digest
  `SHA-512(format || 0 || canon(record))`, `digest`);
* `run`: the digest of the run under check's own record, the log's first for the deployment's first run;
* `params`: the verifier's own copy of its program function's parameters, `[[name, type], ...]` as a body's params are
  written. The descriptor names no parameters (`verity/ml/circuits/PROTOCOL.md` §5.1: the root starts with each
  parameter's input nodes, in parameter order), so these say which parameter each θ node holds a part of: they must
  lower onto the root's input nodes exactly (`ownersOf`), as `SampledProofs.Registered.params` requires.

## The binding

Each θ entry lies in a root input node `k` whose gates are `w` bits each, part of the parameter `p` whose input nodes hold
`n_p` gates. The run's own record registers `p` (R7) under a hiding root, and the entry is bound when that root is the
cited `R`: the frame-v3-sha512 root over hm96-sha512 rows determines the rows, short of a SHA-512 collision among the
inputs hashed along the two trees (`Nci.Assumptions.RootsBind`, the binding premise the reduction names). That is
R7's `held` rule: a value keeps its root across records (`one_stage/registration.py`).

## Refusals

In the order checked, after every code of `NciCheck.Desc.evaluate` (whose θ codes come first: `theta-unknown`,
`theta-not-input`, `theta-not-registered`, `theta-duplicate`; then its owners' and `empty-iu`), so that the policy check's
statements (`NciCheck.Properties`) stand as they are:
* `theta-provenance`: `provenance` missing, not an array of one entry per θ entry, or an entry not of that form (an
  entry of any other kind, `output` among them);
* `trusted`: the trusted input is not as stated: its keys, a log record that is no object or breaks the chain, `run` the
  digest of no record of the log, or `params` not distinct names of types;
* `run-program`: the run's own record is not this Program's (its `program` is not `Prog.sha512`), whatever θ is. The
  run's record is the registration that sampled proofs commit to, so it names the Program checked even when θ is empty
  and no entry reads it;
* then θ entry by entry:
  - `theta-record-unknown`: `D` is the digest of no record of the log;
  - `theta-not-at-deployment-start`: `D` is a record of the log other than its first. It replaces the rule that `D`
    precede the run's own record: the first record is at or before every run's own, the first run's being that record
    itself, so this one code is the whole timing rule;
  - `theta-root`: `D` registers no value `V` whose root is `R` (its last entry for `V`, as Python's dict reads it);
  - `theta-unbound`: `params` don't lower onto the root's input nodes, or the run's record has no `registered` entry for
    `p` whose root is `R`;
  - `theta-width`: `D`'s rows and words don't give `p`'s `n_p` gates of `w` bits, rounded up to whole u16 words.

## Cost

O(log + params' input nodes + θ · (log + one record)): one digest per record of the log, one comparison of the run's
record's program, one lowering of the parameters onto the root body's input nodes, and per θ entry lookups in two
records. Nothing here walks a gate.
-/

namespace NciCheck.Baseline

open Lean (Json toJson)
open Flock.Program

/-! ## JSON -/

/-- Python's `d.get(k)`: a missing key, or a `j` that is no object, reads as `null`. -/
def getD (j : Json) (k : String) : Json := (j.getObjVal? k).toOption.getD Json.null

def keys? : Json → Option (Array String)
  | .obj kvs => some ((kvs.toArray.map (·.1)).qsort (· < ·))
  | _ => none

/-- `j` as a string, `none` where it is none. A field is compared with a string through it, never with `Json`'s `==`,
which is `partial`, so that what the check compares is what `Properties.Nci.Baseline` can state. -/
def strOf? : Json → Option String
  | .str s => some s
  | _ => none

/-- Field `k` of `j` as a string: `strOf?` of `getD j k`. -/
def strAt (j : Json) (k : String) : Option String := strOf? (getD j k)

/-- The last entry of `j`'s array `f` whose `k` is the string `v`: the entry for `v` of a list keyed by `k`, as Python's
`{e[k]: e for e in …}` reads it. -/
def lastWith (j : Json) (f k v : String) : Option Json :=
  match getD j f with
  | .arr es => es.reverse.find? fun e => strAt e k == some v
  | _ => none

/-! ## The log -/

def FORMAT : String := "verity/registration/v1"
def FORMAT_V2 : String := "verity/registration/v2"

/-- A registration record's digest, `SHA-512(format || 0 || canon(record))` with its own format (v1 or v2) as the tag
(`registration.digest`, `SampledProofs.Registration.digest`), as lowercase hex. -/
def digest (rec : Json) : Except String String := do
  let tag := if strAt rec "format" == some FORMAT_V2 then FORMAT_V2 else FORMAT
  return Flock.hexOf (Flock.Sha512.hash (tag.toUTF8.push 0 ++ (← Flock.canon rec).toUTF8))

/-! ## The parameters, lowered onto the input nodes (§5.1, as `SampledProofs.Registered.lower`) -/

private theorem lt_array (n : Nat) (e : Ty) : sizeOf e < sizeOf (Ty.array n e) := by simp; omega

private theorem lt_tuple {f : Ty} {fs : Array Ty} (h : f ∈ fs) : sizeOf f < sizeOf (Ty.tuple fs) := by
  have := Array.sizeOf_lt_of_mem h; simp; omega

private theorem lt_record {f : String × Ty} {fs : Array (String × Ty)} (h : f ∈ fs) :
    sizeOf f.2 < sizeOf (Ty.record fs) := by
  have := Array.sizeOf_lt_of_mem h; cases f; simp at *; omega

/-- `Type.width_set`: the leaf widths, computed structurally (an `Array` of no elements has none). -/
def widthSet : Ty → List Nat
  | .value w => [w]
  | .array n e => if n == 0 then [] else widthSet e
  | .tuple fs => (fs.toList.attach.flatMap fun ⟨f, _⟩ => widthSet f).eraseDups
  | .record fs => (fs.toList.attach.flatMap fun ⟨f, _⟩ => widthSet f.2).eraseDups
termination_by t => sizeOf t
decreasing_by
  · exact lt_array _ _
  · exact lt_tuple (Array.mem_def.mpr ‹_›)
  · exact lt_record (Array.mem_def.mpr ‹_›)

/-- The input nodes a parameter of type `t` lowers to, `(width, n)` each: one node when its leaves are one width, else
each field or element in order. -/
def lower (t : Ty) : Array (Nat × Nat) :=
  match widthSet t with
  | [w] => #[(w, t.leaves)]
  | _ => match t with
    | .array n e => let l := lower e; (Array.range n).foldl (fun acc _ => acc ++ l) #[]
    | .tuple fs => fs.attach.foldl (fun acc ⟨f, _⟩ => acc ++ lower f) #[]
    | .record fs => fs.attach.foldl (fun acc ⟨f, _⟩ => acc ++ lower f.2) #[]
    | .value _ => #[]
termination_by sizeOf t
decreasing_by
  · exact lt_array _ _
  · exact lt_tuple ‹_›
  · exact lt_record ‹_›

/-- Per root node, the parameter (its index in `ps`) whose input nodes it is one of, `none` for a Call: when the
parameters, lowered in order, are exactly the root body's input nodes in body order, each one's width and member count. -/
def ownersOf (P : Prog) (ps : Array (String × Ty)) : Option (Array (Option Nat)) := Id.run do
  let nodes := (P.defs[P.root]!).body.nodes
  let flat := (ps.mapIdx fun i p => (lower p.2).map (i, ·)).foldl (· ++ ·) #[]
  let mut own : Array (Option Nat) := Array.replicate nodes.size none
  let mut at_ := 0
  for k in [0:nodes.size] do
    let n := nodes[k]!
    if !Desc.isInputNode P n then continue
    let some w := inputWidth? (P.defs[n.fn]!).key | return none
    match flat[at_]? with
    | some (i, (w', m)) =>
      if w' != w || m != n.members then return none
      own := own.set! k (some i)
      at_ := at_ + 1
    | none => return none
  if at_ != flat.size then return none
  return some own

/-- Per parameter, the gates of its input nodes: root node `k` holds gates `[offs[k], offs[k + 1])`. -/
def countsOf (offs : Array Nat) (own : Array (Option Nat)) (n : Nat) : Array Nat :=
  (List.range own.size).foldl (fun c k => match own[k]! with
    | some i => c.modify i (· + (offs[k + 1]! - offs[k]!))
    | none => c) (Array.replicate n 0)

/-! ## The trusted input -/

def TRUSTED_KEYS : Array String := #["log", "params", "run"]

/-- The trusted input, read: the log and each record's digest, the run's position in the log, and the program function's
parameters. -/
structure Trusted where
  log : Array Json
  digests : Array String
  run : Nat
  params : Array (String × Ty)

/-- Whether a record's `previous` is `d`, the digest of the record before it, or `null` where there is none. -/
def prevIs (prev : Json) : Option String → Bool
  | some d => strOf? prev == some d
  | none => match prev with
    | .null => true
    | _ => false

/-- A parameter of `params`, `[name, type]`. -/
def paramOf (p : Json) : Except String (String × Ty) := do
  let .arr a := p | throw "trusted"
  if a.size != 2 then throw "trusted"
  let .str k := a[0]! | throw "trusted"
  match Ty.ofJson a[1]! with
  | .ok ty => pure (k, ty)
  | .error _ => throw "trusted"

/-- The trusted input, refused (`trusted`) where it is not as the docstring states. -/
def trustedOf (tr : Json) : Except String Trusted := do
  if keys? tr != some TRUSTED_KEYS then throw "trusted"
  let .arr log := getD tr "log" | throw "trusted"
  let mut digests : Array String := #[]
  for rec in log do
    if !(rec matches .obj _) then throw "trusted"
    if !prevIs (getD rec "previous") digests.back? then throw "trusted"
    let .ok d := digest rec | throw "trusted"
    digests := digests.push d
  let .str run := getD tr "run" | throw "trusted"
  let some r := digests.findIdx? (· == run) | throw "trusted"
  let .arr ps := getD tr "params" | throw "trusted"
  let params ← ps.mapM paramOf
  if (params.map (·.1)).toList.eraseDups.length != params.size then throw "trusted"
  return { log, digests, run := r, params }

/-! ## Provenance -/

/-- One θ entry's provenance: the registration record `D` it cites, the value `V` and the root `R`. -/
structure Prov where
  registration : String
  value : String
  root : String
  deriving Inhabited

def REGISTERED_KEYS : Array String := #["kind", "registration", "root", "value"]

/-- An entry of `provenance`: of the `registered` form, every field a string. An entry of any other kind is none. -/
def provOf (p : Json) : Option Prov := do
  match ← strAt p "kind" with
  | "registered" =>
    guard (keys? p == some REGISTERED_KEYS)
    pure { registration := ← strAt p "registration", value := ← strAt p "value", root := ← strAt p "root" }
  | _ => none

/-- `provenance`: one entry per θ entry (`theta-provenance` otherwise). -/
def provenanceOf (j : Json) (n : Nat) : Except String (Array Prov) := do
  let .arr ps := getD j "provenance" | throw "theta-provenance"
  if ps.size != n then throw "theta-provenance"
  ps.mapM fun p => match provOf p with
    | some q => pure q
    | none => throw "theta-provenance"

/-- A registered entry's rows (`leaves`), `words` and `word_bits` (16 where the key is absent), each a natural. -/
def dims (e : Json) : Option (Nat × Nat × Nat) := do
  let r ← natOf? (getD e "leaves")
  let w ← natOf? (getD e "words")
  let b ← match (e.getObjVal? "word_bits").toOption with
    | none => pure 16
    | some j => natOf? j
  return (r, w, b)

/-- A θ entry in root node `k`, whose gates are `width` bits each, with provenance `p`: checked against the trusted
input `T`, each root node's parameter `own` and each parameter's gate count `counts`. -/
def entryStep (T : Trusted) (own : Array (Option Nat)) (counts : Array Nat) (k width : Nat) (p : Prov) :
    Except String Unit := do
  let some d := T.digests.findIdx? (· == p.registration) | throw "theta-record-unknown"
  if d != 0 then throw "theta-not-at-deployment-start"
  let some e := lastWith T.log[0]! "registered" "value" p.value | throw "theta-root"
  if strAt e "root" != some p.root then throw "theta-root"
  let run := T.log[T.run]!
  let some (some i) := own[k]? | throw "theta-unbound"
  let some mine := lastWith run "registered" "value" T.params[i]!.1 | throw "theta-unbound"
  if strAt mine "root" != some p.root then throw "theta-unbound"
  let bits := 16 * ((width + 15) / 16)
  let fits := match dims e with
    | some (r, w, b) => r * w == counts[i]! && b == bits
    | none => false
  if !fits then throw "theta-width"

/-- θ's provenance on an instance the policy check gave a verdict, θ entry `t` lying in root node `node offs θ[t]`
(`offs` the root body's offsets). -/
def provenance (j tr : Json) (node : Array Nat → Nat → Nat) : Except String Unit := do
  let i ← parse j
  let P ← match decode i.program.toUTF8 with
    | .ok P => pure P
    | .error _ => throw "program"
  let ps ← provenanceOf j i.theta.size
  let T ← trustedOf tr
  if strAt T.log[T.run]! "program" != some (Flock.hexOf P.sha512) then throw "run-program"
  let body := (P.defs[P.root]!).body
  let own := (ownersOf P T.params).getD (Array.replicate body.nodes.size none)
  let counts := countsOf body.offsets own T.params.size
  for t in [0:i.theta.size] do
    let k := node body.offsets ((natOf? i.theta[t]!).getD 0)
    let width := (inputWidth? (P.defs[(body.nodes[k]!).fn]!).key).getD 0
    entryStep T own counts k width ps[t]!

/-- The check of record: `NciCheck.Desc.evaluate` (θ and the owners per root node), then θ's provenance against the
trusted input `tr`; the verdict is the policy check's. -/
def evaluate (j tr : Json) : Except String Json := do
  let out ← Desc.evaluate j
  provenance j tr fun _ k => k
  return out

/-- The reference: `NciCheck.evaluate` (θ and the owners per gate), then θ's provenance, gate `g` of θ lying in the root
node that holds it (`Desc.nodeOf`). -/
def evaluateUnrolled (j tr : Json) : Except String Json := do
  let out ← NciCheck.evaluate j
  provenance j tr Desc.nodeOf
  return out

/-- The fix a refusal's `message` gives, for the codes of θ's provenance. -/
def fixOf : String → Option String
  | "theta-provenance" => some ("give `provenance`, one entry per theta entry in theta's order: {kind: registered, " ++
      "registration, value, root}; theta holds registered public data only, never an earlier run's output")
  | "trusted" => some ("the trusted input is the service's: {log: the deployment's registration records in R6's " ++
      "chain, run: the run's own record's digest, params: [[name, type], ...]}")
  | "run-program" => some ("register the run under the Program checked: the run's own record's program is the " ++
      "Program's SHA-512, whatever theta is")
  | "theta-record-unknown" => some "cite a record of the prover's registration log by its digest"
  | "theta-not-at-deployment-start" => some "cite the registration log's first record"
  | "theta-root" => some "cite the root the record registers for that value"
  | "theta-unbound" => some ("register the theta node's parameter in the run's own record under the cited root (R7: " ++
      "a value keeps its root)")
  | "theta-width" => some "the cited value's rows, words and word bits must give the theta node's parameter"
  | _ => none

/-- A refusal of `j` as JSON: `{"error": code}`, with its `message` (`fixOf`, else as `NciCheck.Desc.refusal` writes it). -/
def refusal (j : Json) (code : String) : Json :=
  match fixOf code with
  | some m => Json.mkObj [("error", Json.str code), ("message", Json.str m)]
  | none => Desc.refusal j code

def check (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => r
  | .error code => refusal j code

def checkUnrolled (j tr : Json) : Json :=
  match evaluateUnrolled j tr with
  | .ok r => r
  | .error code => refusal j code

end NciCheck.Baseline
