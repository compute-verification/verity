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
  layout's operands, repeats dropped). The definition's gates read only earlier gates (`reads_lt`), so a Program with a
  gate reading itself or a later one is refused (`not-topological`). Nothing here reads a gate's operation or the
  outputs, and neither does `novelReads`.
* `theta`: θ, a list of gate ids (canonical root-scoped gate indices), each an input gate, none twice.
* `owner`: the division, one entry per gate as Flock's partition check takes it (`Flock.Partition.Owner`): `"input"` for
  an input gate, which is in no IU, and `u` for a computed gate of IU `u`. This is the definition's `Partition.unit`,
  with `none` written `"input"`, so a gate in two IUs has no encoding. The IUs are `0 … n - 1`, `n` one more than the
  largest `u`, and each holds a gate (`empty-iu`): an empty unit has size and reads 0 and meets both bounds, so a
  division with empty units complies exactly when the one with them dropped and the rest renumbered does.
* `gamma`: `Γ`, a natural number; `alpha`: `α` as `[num, den]`, naturals with `den > 0`. `novelReads R ≤ α · size R`
  is checked as `novelReads R · den ≤ num · size R` in ℕ.

`evaluate` refuses a malformed instance with the first failing code, in the order checked: `instance` (a field missing
or of the wrong JSON kind), `params` (`Γ` or `α`), `program` (the decoder refuses it), `not-topological`, then θ entry by
entry (`theta-unknown`: not a gate id, `theta-not-input`, `theta-duplicate`), then the owners (`owner-length`: not one
per gate, so an id outside the circuit or a gate left out; then gate by gate `owner-entry`: neither `"input"` nor a
natural, `input-in-iu`, `gate-in-no-iu`), then `empty-iu`.
-/

namespace NciCheck

open Lean (Json toJson)

/-- A natural number in JSON: an integer with no exponent, at least 0. -/
def natOf? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- `D` as the definition reads it, gate by gate: the width of its value, whether it is an input gate, and the gates it
reads, ascending and distinct. -/
structure Circuit where
  w : Array Nat
  isInput : Array Bool
  reads : Array (Array Nat)
  deriving Inhabited

def Circuit.N (c : Circuit) : Nat := c.w.size

def distinct (xs : Array Nat) : Array Nat :=
  (xs.qsort (· < ·)).foldl (fun a x => if a.back? == some x then a else a.push x) #[]

/-- The Program's canonical circuit, refused (`program`) where a gate's primitive can't be located or returns more than
one value, and (`not-topological`) where a gate reads a gate not before it. -/
def circuitOf (P : Flock.Program.Prog) : Except String Circuit := do
  let n := (P.defs[P.root]!).gates
  let mut c : Circuit := { w := Array.mkEmpty n, isInput := Array.mkEmpty n, reads := Array.mkEmpty n }
  for g in [0:n] do
    let (p, ops, _) ← match P.gate g with
      | .ok x => pure x
      | .error _ => throw "program"
    let defn := P.defs[p]!
    let .value w := defn.ret | throw "program"
    if ops.any (· ≥ g) then throw "not-topological"
    c := { w := c.w.push w, isInput := c.isInput.push (Flock.Program.isInputId defn.key),
           reads := c.reads.push (distinct ops) }
  return c

/-- θ and the division, checked: θ as a set of gates, each gate's IU (`none` for an input gate), and the number of IUs. -/
structure Division where
  theta : Array Bool
  owner : Array (Option Nat)
  n : Nat
  deriving Inhabited

def divisionOf (c : Circuit) (theta owner : Array Json) : Except String Division := do
  let N := c.N
  let mut mask := Array.replicate N false
  for t in theta do
    let some g := natOf? t | throw "theta-unknown"
    if g ≥ N then throw "theta-unknown"
    if !c.isInput[g]! then throw "theta-not-input"
    if mask[g]! then throw "theta-duplicate"
    mask := mask.set! g true
  if owner.size != N then throw "owner-length"
  let mut own : Array (Option Nat) := Array.mkEmpty N
  for g in [0:N] do
    let o ← match owner[g]! with
      | .str "input" => pure none
      | j => match natOf? j with
        | some u => pure (some u)
        | none => throw "owner-entry"
    if o.isSome && c.isInput[g]! then throw "input-in-iu"
    if o.isNone && !c.isInput[g]! then throw "gate-in-no-iu"
    own := own.push o
  let n := own.foldl (fun m o => match o with | some u => max m (u + 1) | none => m) 0
  -- more IUs than gates leaves one empty, and `used` stays as small as the circuit
  if n > N then throw "empty-iu"
  let mut used := Array.replicate n false
  for o in own do
    if let some u := o then used := used.set! u true
  if used.any (! ·) then throw "empty-iu"
  return { theta := mask, owner := own, n }

/-- Per IU `u`, with gates `R`: `(size R, novelReads R)`, where `into(R)` is the gates outside `R` that a gate of `R`
reads and `novelReads R` the width of those not in θ. -/
def ius (c : Circuit) (d : Division) : Array (Nat × Nat) := Id.run do
  let mut members : Array (Array Nat) := Array.replicate d.n #[]
  for g in [0:c.N] do
    if let some u := d.owner[g]! then members := members.modify u (·.push g)
  let mut out := Array.mkEmpty d.n
  for u in [0:d.n] do
    let R := members[u]!
    let mut into : Std.HashSet Nat := {}
    for g in R do
      for a in c.reads[g]! do
        if d.owner[a]! != some u then into := into.insert a
    out := out.push (R.size, into.fold (fun s a => if d.theta[a]! then s else s + c.w[a]!) 0)
  return out

/-- The first IU that breaks the policy, with the rule: `size` when it has more than `Γ` gates, else `nci` when its
novel reads exceed `num/den` times its size. -/
def failure (gamma num den : Nat) (r : Array (Nat × Nat)) : Option (Nat × String) :=
  (List.range r.size).findSome? fun u =>
    let (size, novel) := r[u]!
    if size > gamma then some (u, "size") else if novel * den > num * size then some (u, "nci") else none

/-- The check on one instance: `{gates, ius: [[size, novelReads], …], complies, failure: null | [iu, rule]}`, or the
first failing code of a malformed instance. -/
def evaluate (j : Json) : Except String Json := do
  let field := fun (k : String) => match j.getObjVal? k with
    | .ok v => (pure v : Except String Json)
    | .error _ => throw "instance"
  let .str program ← field "program" | throw "instance"
  let .arr theta ← field "theta" | throw "instance"
  let .arr owner ← field "owner" | throw "instance"
  let gammaJ ← field "gamma"
  let alphaJ ← field "alpha"
  let some gamma := natOf? gammaJ | throw "params"
  let (num, den) ← match alphaJ with
    | .arr #[a, b] => match natOf? a, natOf? b with
      | some a, some b => if b > 0 then pure (a, b) else throw "params"
      | _, _ => throw "params"
    | _ => throw "params"
  let P ← match Flock.Program.decode program.toUTF8 with
    | .ok P => pure P
    | .error _ => throw "program"
  let c ← circuitOf P
  let d ← divisionOf c theta owner
  let r := ius c d
  let f := failure gamma num den r
  return Json.mkObj [("gates", toJson c.N),
    ("ius", Json.arr (r.map fun (s, x) => Json.arr #[toJson s, toJson x])),
    ("complies", Json.bool f.isNone),
    ("failure", match f with
      | none => Json.null
      | some (u, rule) => Json.arr #[toJson u, Json.str rule])]

/-- `evaluate`, with a refusal as `{"error": code}`. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => Json.mkObj [("error", Json.str code)]

end NciCheck
