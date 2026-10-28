import Std.Data.HashSet
import Flock.Program

/-!
# The NCI policy's check of a declared division

`Nci.Complies Γ α` (`Security/Definitions/Nci.lean`) holds of a declaration `(D, θ)` when `D`'s computed gates divide
into isolation units (IUs) of at most `Γ` gates, each reading at most `α` bits of novel information per gate:
`novelReads R = width(into(R) \ θ) ≤ α · size R`. The verifier's side is the check of one declared division: `evaluate`
decides that it is a partition of the computed gates into IUs (`Verity.Partition`) and that every IU meets both bounds,
and reports per IU its size and novel reads, and the first IU that fails, with the rule it fails.

## The input, and how descriptor v1 is the definition's circuit

An instance is one JSON object:
* `program`: the declaration's circuit `D`, a `verity-ir/descriptor/v1` Program as text (`verity/ml/circuits/PROTOCOL.md`),
  read with C-Flock's decoder (`Flock.Program.decode`). Its canonical layout (§6) is `D`: gate `g` of the layout is
  `Fin D.N` gate `g`, one primitive call returning one `Value<w>`, and `w g` is that `w`, in bits (`Circuit.w`, which
  the definition calls `verity.ir`'s `Value<w>`). The input gates are the input nodes, `Input<w>_v1` and
  `RegisteredInput<w>_v1` (§5). `reads g` is the set of gates `g`'s operands name (`Flock.Program.Prog.gate`: the
  layout's operands; a repeat names the same gate). The definition's gates read only earlier gates (`reads_lt`), so a
  Program with a gate reading itself or a later one is refused (`not-topological`), and its input gates read nothing
  (`input_reads`), so an input node with operands is refused (`program`; descriptor v1's input primitives take no
  parameters, so the decoder admits none, but the check does not rely on it). Nothing here reads a gate's operation or
  the outputs, and neither does `novelReads`.
* `theta`: θ, a list of gate ids (canonical root-scoped gate indices), each an input gate bound at registration
  (`RegisteredInput<w>_v1`, binding `registration` in `Flock.Program.inputId?`), none twice. The baseline is what the
  developer committed before the run, so an input bound per run (`Input<w>_v1`) is refused (`theta-not-registered`):
  in θ it would make any value the run brings, recommitted, free to read.
* `owner`: the division, one entry per gate as Flock's partition check takes it (`Flock.Partition.Owner`): `"input"` for
  an input gate, which is in no IU, and `u` for a computed gate of IU `u`. This is the definition's `Partition.unit`,
  with `none` written `"input"`, so a gate in two IUs has no encoding. The IUs are `0 … n - 1`, `n` one more than the
  largest `u`, and each holds a gate (`empty-iu`): an empty unit has size and reads 0 and meets both bounds, so a
  division with empty units complies exactly when the one with them dropped and the rest renumbered does.
* `gamma`: `Γ`, a natural number; `alpha`: `α` as `[num, den]`, naturals with `den > 0`. `novelReads R ≤ α · size R`
  is checked as `novelReads R · den ≤ num · size R` in ℕ.

`evaluate` refuses a malformed instance with the first failing code, in the order checked: `instance` (a field missing
or of the wrong JSON kind), `params` (`Γ` or `α`), `program` (the decoder refuses it, among them a reference that
reads a leaf at another width than it was committed with, `E_REINTERPRET`, with a `message`: `decodeProgram`), then gate
by gate `program` or `not-topological`, then θ entry by entry (`theta-unknown`: not a gate id, `theta-not-input`, `theta-not-registered`: an input bound per run,
`theta-duplicate`), then the owners (`owner-length`: not one per gate, so an id outside the circuit or a gate left out;
then gate by gate `owner-entry`: neither `"input"` nor a natural, `input-in-iu`, `gate-in-no-iu`), then `empty-iu`.
`check` writes a refusal as `{"error": code}`, with a `message` giving the fix where one is the declaration's to make
(`fixOf`), or for a `program` refusal of a reinterpretation, where (`refusalOf`).
-/

namespace NciCheck

open Lean (Json toJson)

/-- A natural number in JSON: an integer with no exponent, at least 0. -/
def natOf? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- Whether a primitive id is an input bound at registration, `RegisteredInput<w>_v1` (`Flock.Program.inputId?`). -/
def isRegisteredId (id : String) : Bool :=
  (Flock.Program.inputId? id).map (·.1) == some Flock.Program.REGISTRATION

/-- `D` as the definition reads it, gate by gate: the width of its value, whether it is an input gate, and the gates its
operands name (repeats kept: only the set is read); and, which the definition doesn't read but θ does, whether it is an
input bound at registration. -/
structure Circuit where
  w : Array Nat
  isInput : Array Bool
  registered : Array Bool
  reads : Array (Array Nat)

def Circuit.N (c : Circuit) : Nat := c.w.size

/-- Gate `g` of the Program's canonical circuit: its width, whether it is an input gate, whether it is one bound at
registration, and its operands. Refused (`program`) where its primitive can't be located, returns more than one value, or
is an input node with operands, and (`not-topological`) where it reads a gate not before it. -/
def gateOf (P : Flock.Program.Prog) (g : Nat) : Except String (Nat × Bool × Bool × Array Nat) := do
  let (p, ops, _) ← match P.gate g with
    | .ok x => pure x
    | .error _ => throw "program"
  let defn := P.defs[p]!
  let .value w := defn.ret | throw "program"
  let inp := Flock.Program.isInputId defn.key
  if inp && !ops.isEmpty then throw "program"
  if ops.any (· ≥ g) then throw "not-topological"
  return (w, inp, isRegisteredId defn.key, ops)

def Circuit.push (c : Circuit) : Nat × Bool × Bool × Array Nat → Circuit
  | (w, inp, reg, ops) =>
    { w := c.w.push w, isInput := c.isInput.push inp, registered := c.registered.push reg, reads := c.reads.push ops }

/-- The Program's canonical circuit, gate by gate, refused at the first gate `gateOf` refuses. -/
def circuitOf (P : Flock.Program.Prog) : Except String Circuit :=
  (List.range (P.defs[P.root]!).gates).foldlM (fun c g => c.push <$> gateOf P g) ⟨#[], #[], #[], #[]⟩

/-- θ and the division, checked: θ as a set of gates, each gate's IU (`none` for an input gate), and the number of IUs. -/
structure Division where
  theta : Std.HashSet Nat
  owner : Array (Option Nat)
  n : Nat

/-- One entry of θ, added to the entries before it. -/
def thetaStep (c : Circuit) (s : Std.HashSet Nat) (t : Json) : Except String (Std.HashSet Nat) := do
  let some g := natOf? t | throw "theta-unknown"
  if g ≥ c.N then throw "theta-unknown"
  if !c.isInput[g]! then throw "theta-not-input"
  if !c.registered[g]! then throw "theta-not-registered"
  if s.contains g then throw "theta-duplicate"
  return s.insert g

def isInputTag : Json → Bool
  | .str s => s == "input"
  | _ => false

/-- Gate `g`'s owner entry `j`: `none` for `"input"`, `some u` for IU `u`. -/
def ownerOf (c : Circuit) (g : Nat) (j : Json) : Except String (Option Nat) := do
  let o ← if isInputTag j then pure none else match natOf? j with
    | some u => pure (some u)
    | none => throw "owner-entry"
  if o.isSome && c.isInput[g]! then throw "input-in-iu"
  if o.isNone && !c.isInput[g]! then throw "gate-in-no-iu"
  return o

/-- The number of IUs: one more than the largest IU an owner names. -/
def iuCount (own : Array (Option Nat)) : Nat :=
  own.foldl (fun m o => match o with | some u => max m (u + 1) | none => m) 0

/-- The IUs an owner names. -/
def usedOf (own : Array (Option Nat)) : Std.HashSet Nat :=
  own.foldl (fun s o => match o with | some u => s.insert u | none => s) {}

def divisionOf (c : Circuit) (theta owner : Array Json) : Except String Division := do
  let mask ← (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {}
  if owner.size != c.N then throw "owner-length"
  let own ← (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[]
  let n := iuCount own
  -- more IUs than gates leaves one empty, and the check below stays as small as the circuit
  if n > c.N then throw "empty-iu"
  let used := usedOf own
  if !(List.range n).all used.contains then throw "empty-iu"
  return { theta := mask, owner := own, n }

/-- Per IU `u`, in one pass over the gates: its number of gates, and the gates outside it that a gate of it reads,
`into(R)`. -/
def tally (c : Circuit) (d : Division) : Array (Nat × Std.HashSet Nat) :=
  (List.range c.N).foldl (fun acc g => match d.owner[g]! with
    | some u => acc.modify u fun (s, into) =>
        (s + 1, c.reads[g]!.foldl (fun into a => if d.owner[a]! = some u then into else into.insert a) into)
    | none => acc) (Array.replicate d.n (0, {}))

/-- Per IU `u`, with gates `R`: `(size R, novelReads R)`, `novelReads R` the width of the gates of `into(R)` not in θ. -/
def ius (c : Circuit) (d : Division) : Array (Nat × Nat) :=
  (tally c d).map fun (s, into) => (s, into.fold (fun x a => if d.theta.contains a then x else x + c.w[a]!) 0)

/-- The first IU that breaks the policy, with the rule: `size` when it has more than `Γ` gates, else `nci` when its
novel reads exceed `num/den` times its size. -/
def failure (gamma num den : Nat) (r : Array (Nat × Nat)) : Option (Nat × String) :=
  (List.range r.size).findSome? fun u =>
    let (size, novel) := r[u]!
    if size > gamma then some (u, "size") else if novel * den > num * size then some (u, "nci") else none

/-- An instance's fields: the program's text, θ, the owners, `Γ`, and `α = num/den`. -/
structure Instance where
  program : String
  theta : Array Json
  owner : Array Json
  gamma : Nat
  num : Nat
  den : Nat

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error _ => throw "instance"

/-- A JSON array of two entries. -/
def pairOf? : Json → Option (Json × Json)
  | .arr a => if a.size = 2 then some (a[0]!, a[1]!) else none
  | _ => none

/-- The instance's fields, refused (`instance`) where one is missing or of the wrong JSON kind, and (`params`) where `Γ`
or `α` is not as stated. -/
def parse (j : Json) : Except String Instance := do
  let .str program ← field j "program" | throw "instance"
  let .arr theta ← field j "theta" | throw "instance"
  let .arr owner ← field j "owner" | throw "instance"
  let gammaJ ← field j "gamma"
  let alphaJ ← field j "alpha"
  let some gamma := natOf? gammaJ | throw "params"
  let some (a, b) := pairOf? alphaJ | throw "params"
  let some num := natOf? a | throw "params"
  let some den := natOf? b | throw "params"
  if den = 0 then throw "params"
  return { program, theta, owner, gamma, num, den }

/-- The verdict as JSON: `{gates, ius: [[size, novelReads], …], complies, failure: null | [iu, rule]}`. -/
def verdict (gates : Nat) (r : Array (Nat × Nat)) (f : Option (Nat × String)) : Json :=
  Json.mkObj [("gates", toJson gates),
    ("ius", Json.arr (r.map fun (s, x) => Json.arr #[toJson s, toJson x])),
    ("complies", Json.bool f.isNone),
    ("failure", match f with
      | none => Json.null
      | some (u, rule) => Json.arr #[toJson u, Json.str rule])]

/-- The instance's Program as C-Flock's decoder reads it, refused (`program`) where the decoder refuses it, among them a
reference that reads a leaf at another width than it was committed with (`Flock.Program.Widths`, `E_REINTERPRET`;
`refusalOf` gives its message). -/
def decodeProgram (text : String) : Except String Flock.Program.Prog :=
  match Flock.Program.decode text.toUTF8 with
  | .ok P => .ok P
  | .error _ => .error "program"

/-- The check on one instance: its verdict, or the first failing code of a malformed instance. -/
def evaluate (j : Json) : Except String Json := do
  let i ← parse j
  let P ← decodeProgram i.program
  let c ← circuitOf P
  let d ← divisionOf c i.theta i.owner
  let r := ius c d
  return verdict c.N r (failure i.gamma i.num i.den r)

/-- The fix a refusal's `message` gives, for a code whose fix is the declaration's to make. -/
def fixOf : String → Option String
  | "theta-not-registered" => some ("theta names an input bound per run (Input<w>_v1), and the baseline holds only " ++
      "what was committed at registration: declare that parameter registered (RegisteredInput<w>_v1, the Program's " ++
      "`registered`) or drop it from theta")
  | _ => none

/-- A refusal as JSON: `{"error": code}`, with `message` where there is one. -/
def refusalWith (code : String) (message : Option String) : Json :=
  Json.mkObj (("error", Json.str code) :: match message with
    | some m => [("message", Json.str m)]
    | none => [])

/-- A refusal as JSON, with its `message` where `fixOf` has one. -/
def refusal (code : String) : Json :=
  refusalWith code (fixOf code)

/-- Where the decoder refuses `j`'s Program for a reference that reads a leaf at another width (`E_REINTERPRET`), its
message. -/
def reinterpretationOf (j : Json) : Option String :=
  match parse j with
  | .ok i => match Flock.Program.decode i.program.toUTF8 with
    | .error e => if (e.splitOn "E_REINTERPRET").length > 1 then some e else none
    | .ok _ => none
  | .error _ => none

/-- `refusal`, a `program` refusal of `j` with the reinterpretation that `decodeProgram` refused as its `message`. -/
def refusalOf (j : Json) (code : String) : Json :=
  match (if code == "program" then reinterpretationOf j else none) with
  | some m => refusalWith code (some m)
  | none => refusal code

/-- `evaluate`, with a refusal as `refusalOf` writes it. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => refusalOf j code

end NciCheck
