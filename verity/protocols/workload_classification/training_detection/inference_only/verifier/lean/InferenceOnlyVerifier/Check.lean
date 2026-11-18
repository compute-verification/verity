module

public import Std.Data.HashSet
public import Circuits.Program

/-!
# The inference-only policy's check of a declared division

`InferenceOnly.CompliesRate Γ α` (`security/Definitions/InferenceOnly.lean`) holds of a declaration `(D, θ)` when `D`'s computed gates
divide into inference blocks of at most `Γ` gates, each reading at most `α` fresh input bits per gate:
`freshInput R = width(into(R) \ θ) ≤ α · size R`. The verifier's side is the check of one declared division: `evaluate`
decides that it is a partition of the computed gates into inference blocks (`Verity.Partition`) and that every inference block meets both bounds,
and reports per inference block its size and fresh input bits, and the first inference block that fails, with the rule it fails.

## The input, and how descriptor v1 is the definition's circuit

An instance is one JSON object:
* `program`: the declaration's circuit `D`, a `verity-ir/descriptor/v1` Program as text (`verity/circuits/PROTOCOL.md`),
  read with Flock's decoder (`Flock.Program.decode`). Its canonical layout (§6) is `D`: gate `g` of the layout is
  `Fin D.N` gate `g`, one primitive call returning one `Value<w>`, and `w g` is that `w`, in bits (`Circuit.w`, which
  the format calls `Value<w>`). The input gates are the input nodes, `Input<w>_v1` and
  `RegisteredInput<w>_v1` (§5). `reads g` is the set of gates `g`'s operands name (`Flock.Program.Prog.gate`: the
  layout's operands; a repeat names the same gate). The definition's gates read only earlier gates (`reads_lt`), so a
  Program with a gate reading itself or a later one is refused (`not-topological`), and its input gates read nothing
  (`input_reads`), so an input node with operands is refused (`program`; descriptor v1's input primitives take no
  parameters, so the decoder admits none, but the check does not rely on it). Nothing here reads the outputs, and
  `freshInput` reads no gate's operation. The allowance reads one thing of it: whether its primitive holds no gate
  (`weighsNothing`).
* `theta`: θ, a list of gate ids (canonical root-scoped gate indices), each an input gate bound at registration
  (`RegisteredInput<w>_v1`, binding `registration` in `Flock.Program.inputId?`), none twice. The baseline is what the
  developer committed before the run, so an input bound per run (`Input<w>_v1`) is refused (`theta-not-registered`):
  in θ it would make any value the run brings, recommitted, free to read.
* `owner`: the division, one entry per gate as Flock's partition check takes it (`Flock.Partition.Owner`): `"input"` for
  an input gate, which is in no inference block, and `u` for a computed gate of inference block `u`. This is the definition's `Partition.unit`,
  with `none` written `"input"`, so a gate in two inference blocks has no encoding. The inference blocks are `0 … n - 1`, `n` one more than the
  largest `u`, and each holds a gate (`empty-iu`): an empty unit has size and reads 0 and meets both bounds, so a
  division with empty units complies exactly when the one with them dropped and the rest renumbered does.
* `gamma`: `Γ`, a natural number; `alpha`: `α` as `[num, den]`, naturals with `den > 0`. The allowance is checked as
  `freshInput R · den ≤ num · weighed R` in ℕ, where `weighed R ≤ size R` counts `R`'s gates whose primitive holds a gate:
  a constant, a wiring between leaf widths or another primitive of `GATELESS` counts in `size R ≤ Γ` and buys no fresh
  bit. So every division the check accepts meets `freshInput R ≤ α · size R`.

`evaluate` refuses a malformed instance with the first failing code, in the order checked: `instance` (a field missing
or of the wrong JSON kind), `params` (`Γ` or `α`), `program` (the decoder refuses it, among them a reference that
reads a leaf at another width than it was committed with, `E_REINTERPRET`, with a `message`: `decodeProgram`), then gate
by gate `program` or `not-topological`, then θ entry by entry (`theta-unknown`: not a gate id, `theta-not-input`, `theta-not-registered`: an input bound per run,
`theta-duplicate`), then the owners (`owner-length`: not one per gate, so an id outside the circuit or a gate left out;
then gate by gate `owner-entry`: neither `"input"` nor a natural, `input-in-iu`, `gate-in-no-iu`), then `empty-iu`.
`check` writes a refusal as `{"error": code}`, with a `message` giving the fix where one is the declaration's to make
(`fixOf`), or for a `program` refusal of a reinterpretation, where (`refusalOf`).
-/

@[expose] public section

namespace InferenceOnlyVerifier

open Lean (Json toJson)
open Flock.Program (GateTable)

/-- A natural number in JSON: an integer with no exponent, at least 0. -/
def natOf? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- Whether a primitive id is an input bound at registration, `RegisteredInput<w>_v1` (`Flock.Program.inputId?`). -/
def isRegisteredId (id : String) : Bool :=
  (Flock.Program.inputId? id).map (·.1) == some Flock.Program.REGISTRATION

/-- A decimal as the wiring ids write one: digits, with no leading zero but `0` itself. -/
def decimal : List Char → Bool
  | [] => false
  | ['0'] => true
  | '0' :: _ => false
  | cs => cs.all Char.isDigit

/-- A decimal of at least 1. -/
def positive (cs : List Char) : Bool := decimal cs && cs != ['0']

/-- `cs` less its version, `_v1` or `_v2`. -/
def unversioned (cs : List Char) : Option (List Char) :=
  match cs.reverse with
  | '1' :: 'v' :: '_' :: r | '2' :: 'v' :: '_' :: r => some r.reverse
  | _ => none

/-- `cs` on either side of its one `x`. -/
def splitX (cs : List Char) : Option (List Char × List Char) :=
  match cs.span (· != 'x') with
  | (w, 'x' :: i) => if i.contains 'x' then none else some (w, i)
  | _ => none

/-- Whether a primitive id is the wiring between leaf widths (`verity.circuits.population`): `BitOf{W}x{I}`, bit `I` of a
`W`-bit word, or `WordOf{W}`, a word of `W` bits, at version 1 or its Boolean version 2. -/
def wiringId (id : String) : Bool :=
  match id.toList with
  | 'B' :: 'i' :: 't' :: 'O' :: 'f' :: r => match (unversioned r).bind splitX with
    | some (w, i) => positive w && decimal i
    | none => false
  | 'W' :: 'o' :: 'r' :: 'd' :: 'O' :: 'f' :: r => ((unversioned r).map positive).getD false
  | _ => false

/-- The primitives other than constants and `wiringId`'s that the gate table prices at no Boolean gate: each selects,
extends or shifts its operand's bits. -/
def GATELESS : List String :=
  ["Bf16ToF32_v1", "E8m0ShlBf16_v1", "F32BitsShl23_v1", "F32Fabs_v1", "I64ToU8_v1", "PouwDigitLo_v1", "PouwLow8_v1",
   "PouwSx8_v1", "S32LowByte_v1", "U32Hi16_v1", "U8ToI64_v1", "U8ToU16_v1", "U8x4ToU32_v1"]

/-- Whether a call of `defn` weighs nothing in the rate rule's allowance: a primitive that holds no gate, a constant (no
parameters), a wiring between leaf widths (`wiringId`) or one of `GATELESS`. Such a call still counts in a block's size. -/
def weighsNothing (defn : Flock.Program.Defn) : Bool :=
  defn.prim && (defn.params.isEmpty || wiringId defn.key || GATELESS.contains defn.key)

/-- A node's calls that weigh nothing, `f` giving each Definition's: its callee's, times its members. -/
def nodeUnweighed (f : Array Nat) (nd : Flock.Program.Node) : Nat :=
  if nd.form == .prim then f[nd.fn]! else nd.members * f[nd.fn]!

/-- Per Definition, the calls of one activation of it that weigh nothing (`weighsNothing`), callees first, its nodes'
(`nodeUnweighed`) as the decoder sums its gates. -/
def unweighed (P : Flock.Program.Prog) : Array Nat :=
  P.defs.foldl (fun f defn => f.push (if defn.prim then (if weighsNothing defn then 1 else 0)
    else defn.body.nodes.foldl (fun s nd => s + nodeUnweighed f nd) 0)) #[]

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

/-- `D` as the definition reads it, gate by gate (`Flock.Program.GateTable`): the width of its value, whether it is an input
gate, and the gates its operands name (repeats kept: only the set is read); and, which the definition doesn't read but θ
does, whether it is an input bound at registration. Refused at the first gate `gateOf` refuses. -/
def circuitOf (P : Flock.Program.Prog) : Except String GateTable :=
  (List.range (P.defs[P.root]!).gates).foldlM (fun c g => c.push <$> gateOf P g) ⟨#[], #[], #[], #[]⟩

/-- θ and the division, checked: θ as a set of gates, each gate's inference block (`none` for an input gate), and the number of inference blocks. -/
structure Division where
  theta : Std.HashSet Nat
  owner : Array (Option Nat)
  n : Nat

/-- One entry of θ, added to the entries before it. -/
def thetaStep (c : GateTable) (s : Std.HashSet Nat) (t : Json) : Except String (Std.HashSet Nat) := do
  let some g := natOf? t | throw "theta-unknown"
  if g ≥ c.N then throw "theta-unknown"
  if !c.isInput[g]! then throw "theta-not-input"
  if !c.registered[g]! then throw "theta-not-registered"
  if s.contains g then throw "theta-duplicate"
  return s.insert g

def isInputTag : Json → Bool
  | .str s => s == "input"
  | _ => false

/-- Gate `g`'s owner entry `j`: `none` for `"input"`, `some u` for inference block `u`. -/
def ownerOf (c : GateTable) (g : Nat) (j : Json) : Except String (Option Nat) := do
  let o ← if isInputTag j then pure none else match natOf? j with
    | some u => pure (some u)
    | none => throw "owner-entry"
  if o.isSome && c.isInput[g]! then throw "input-in-iu"
  if o.isNone && !c.isInput[g]! then throw "gate-in-no-iu"
  return o

/-- The number of inference blocks: one more than the largest inference block an owner names. -/
def iuCount (own : Array (Option Nat)) : Nat :=
  own.foldl (fun m o => match o with | some u => max m (u + 1) | none => m) 0

/-- The inference blocks an owner names. -/
def usedOf (own : Array (Option Nat)) : Std.HashSet Nat :=
  own.foldl (fun s o => match o with | some u => s.insert u | none => s) {}

def divisionOf (c : GateTable) (theta owner : Array Json) : Except String Division := do
  let mask ← (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {}
  if owner.size != c.N then throw "owner-length"
  let own ← (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[]
  let n := iuCount own
  -- more inference blocks than gates leaves one empty, and the check below stays as small as the circuit
  if n > c.N then throw "empty-iu"
  let used := usedOf own
  if !(List.range n).all used.contains then throw "empty-iu"
  return { theta := mask, owner := own, n }

/-- Per inference block `u`, in one pass over the gates: its number of gates, and the gates outside it that a gate of it reads,
`into(R)`. -/
def tally (c : GateTable) (d : Division) : Array (Nat × Std.HashSet Nat) :=
  (List.range c.N).foldl (fun acc g => match d.owner[g]! with
    | some u => acc.modify u fun (s, into) =>
        (s + 1, c.reads[g]!.foldl (fun into a => if d.owner[a]! = some u then into else into.insert a) into)
    | none => acc) (Array.replicate d.n (0, {}))

/-- Per inference block `u`, with gates `R`: `(size R, freshInput R)`, `freshInput R` the width of the gates of `into(R)` not in θ. -/
def ius (c : GateTable) (d : Division) : Array (Nat × Nat) :=
  (tally c d).map fun (s, into) => (s, into.fold (fun x a => if d.theta.contains a then x else x + c.w[a]!) 0)

/-- Whether `P`'s gate `g` is a call that weighs nothing (`weighsNothing`); a gate the walk can't locate weighs. -/
def unweighedAt (P : Flock.Program.Prog) (g : Nat) : Bool :=
  match P.gate g with
  | .ok (p, _, _) => weighsNothing P.defs[p]!
  | .error _ => false

/-- Per inference block, its weighed gates: those of its gates that are no call weighing nothing (`unweighedAt`). -/
def weighs (P : Flock.Program.Prog) (c : GateTable) (d : Division) : Array Nat :=
  (List.range c.N).foldl (fun acc g => match d.owner[g]! with
    | some u => if unweighedAt P g then acc else acc.modify u (· + 1)
    | none => acc) (Array.replicate d.n 0)

/-- The first inference block that breaks the policy, with the rule: `size` when it has more than `Γ` gates, else `inference_only` when its
fresh input bits exceed `num/den` times its weighed gates `w[u]`, or its size where that is less. -/
def failure (gamma num den : Nat) (r : Array (Nat × Nat)) (w : Array Nat) : Option (Nat × String) :=
  (List.range r.size).findSome? fun u =>
    let (size, novel) := r[u]!
    if size > gamma then some (u, "size") else if novel * den > num * min size w[u]! then some (u, "nci") else none

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

/-- The verdict as JSON: `{gates, ius: [[size, freshInput], …], weighed: [weighed, …], complies, failure: null | [iu, rule]}`. -/
def verdict (gates : Nat) (r : Array (Nat × Nat)) (w : Array Nat) (f : Option (Nat × String)) : Json :=
  Json.mkObj [("gates", toJson gates),
    ("ius", Json.arr (r.map fun (s, x) => Json.arr #[toJson s, toJson x])),
    ("weighed", Json.arr (w.map toJson)),
    ("complies", Json.bool f.isNone),
    ("failure", match f with
      | none => Json.null
      | some (u, rule) => Json.arr #[toJson u, Json.str rule])]

/-- The instance's Program as Flock's decoder reads it, refused (`program`) where the decoder refuses it, among them a
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
  let w := weighs P c d
  return verdict c.N r w (failure i.gamma i.num i.den r w)

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

end InferenceOnlyVerifier
