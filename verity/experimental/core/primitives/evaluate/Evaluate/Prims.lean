import Flock.Program

/-!
# The primitives the evaluator implements

E0 is the Boolean profile (`verity/ml/circuits/PROTOCOL.md` §9, `verity.ml.circuits.boolean`): `And<n>_v1` and `Xor<n>_v1`
for every n ≥ 2, `Not_v1`, and the constant bits `Const1[0x0]_v1` and `Const1[0x1]_v1`; and the input primitives `Input<w>_v1`
(§5.2), whose gates take the values bound per run. Each gate's value is the Python evaluator's, transcribed on `Nat`:
`_and`, `_xor` and `NOT`'s lambda take their operands unmasked, as the Python does.

A primitive is recognized by its id and must declare the signature the registry holds for it, parameter names included
(`codec.decode_program`'s check). Every other primitive is refused by name, never guessed: `RegisteredInput<w>_v1`
(registered inputs and their tables are a later step) and every primitive outside the profile.
-/

namespace Evaluate

open Flock.Program

/-- A primitive the evaluator implements. -/
inductive Prim where
  | and (n : Nat)
  | xor (n : Nat)
  | not
  | const (b : Nat)
  | input (w : Nat)
  deriving Inhabited, BEq

/-- `boolean._and`: `int(all(b & 1 for b in x))`. -/
def andBits (xs : Array Nat) : Nat := if xs.all (fun b => b &&& 1 == 1) then 1 else 0

/-- `boolean._xor`: the XOR of the operands, masked to its low bit. -/
def xorBits (xs : Array Nat) : Nat := xs.foldl (· ^^^ ·) 0 &&& 1

/-- `boolean.NOT`: `(a ^ 1) & 1`. -/
def notBit (a : Nat) : Nat := (a ^^^ 1) &&& 1

/-- The gate's value on its operands, in operand order; an input gate has none (its value is bound, not computed). -/
def Prim.apply : Prim → Array Nat → Nat
  | .and _, xs => andBits xs
  | .xor _, xs => xorBits xs
  | .not, xs => notBit xs[0]!
  | .const b, _ => b
  | .input _, _ => 0

/-- `n` of `boolean._N_ARY_ID`, `^(And|Xor)([2-9]|[1-9][0-9]+)_v1$`: decimal without a leading zero, at least 2. -/
private def arity? (pre id : String) : Option Nat :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let n := ((id.drop pre.length).dropEnd 3).toString
  if n.length > 0 && n.all Char.isDigit && n.front != '0' then n.toNat?.filter (· ≥ 2) else none

/-- The primitive an id denotes, with the signature the registry holds for it, or none. -/
def byId? (id : String) : Option (Prim × Array (String × Ty) × Ty) :=
  let bit := Ty.value 1
  match arity? "And" id, arity? "Xor" id with
  | some n, _ => some (.and n, #[("x", .array n bit)], bit)
  | _, some n => some (.xor n, #[("x", .array n bit)], bit)
  | none, none =>
    if id == "Not_v1" then some (.not, #[("a", bit)], bit)
    else if id == "Const1[0x0]_v1" then some (.const 0, #[], bit)
    else if id == "Const1[0x1]_v1" then some (.const 1, #[], bit)
    else none

private def sameParams (a b : Array (String × Ty)) : Bool :=
  a.size == b.size && (a.zip b).all fun (x, y) => x.1 == y.1 && x.2 == y.2

/-- The implementation of a primitive Definition of the program, or its refusal by name. -/
def ofDefn (d : Defn) : Flock.V Prim := do
  match inputId? d.key with
  | some (b, w) =>
    if b == REGISTRATION then
      throw s!"primitive {d.key}: a registered input, which this evaluator refuses (E0 binds run inputs only)"
    return .input w
  | none =>
    let some (p, params, ret) := byId? d.key
      | throw s!"primitive {d.key}: not implemented by this evaluator (E0: the Boolean profile, PROTOCOL.md §9)"
    if !(sameParams d.params params && d.ret == ret) then throw s!"primitive {d.key} signature differs from the registry"
    return p

end Evaluate
