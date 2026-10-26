import Properties.NetworkScan

/-!
# A sampling statement's gates: its descriptor flattened

`Claims.Sampling.Stmt.prog` is "the descriptor flattened". This module defines that flattening for a Boolean Program
(`verity/ml/circuits/PROTOCOL.md` §9), gate for gate in its canonical layout, and the statement reader whose statements'
gates are it:

* `gateOf? key ops`: the Boolean-profile gate (`Claims.Sampling.BGate`) a primitive gate of id `key` on the operands
  `ops` is: `.input` for an input id (`Input1_v1`, `RegisteredInput1_v1`), `.const` for `Const1[0x0]_v1` and
  `Const1[0x1]_v1`, `.not` for `Not_v1`, and `.and` and `.xor` over the operands for `And<n>_v1` and `Xor<n>_v1` with
  `n` operands. None for any other primitive, or for operands its signature doesn't have.
* `flatOf P`: every gate of `P`'s root, located (`Flock.Program.Prog.gate`) and read by `gateOf?`, returning the gates
  the root's return resolves to (`Flock.Program.resolve`). None where a gate doesn't locate or isn't Boolean, or a
  returned reference doesn't resolve.
* `parseOf rd`: the statement whose descriptor, query, design and roots the framing `rd` reads from the message, with
  its descriptor decoded (`Flock.Program.decode`) and flattened as its gates. None where `rd` reads nothing, the
  descriptor doesn't decode or `flatOf` refuses it.

The statements, which `Proofs.Properties.Flock.Circuit.Flatten` proves:

* `FlatOfFlattens`: `flatOf P = some C` gives `Flattens P C`.
* `ParseOfFlattens`: a deployment whose parser is `parseOf rd` has `StmtFlattens`, for every framing `rd`.
-/

namespace Properties.Flock.Circuit

open Properties.NetworkScan

/-- `n` of a family id `<pre><n>_v1`, `n ≥ 2` in decimal without a leading zero (PROTOCOL.md §9: `And<n>_v1`,
`Xor<n>_v1`), as the Lean evaluator reads it (`Evaluate.byId?`). -/
def arity? (pre id : String) : Option ℕ :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let n := ((id.drop pre.length).dropEnd 3).toString
  if n.length > 0 && n.all Char.isDigit && n.front != '0' then n.toNat?.filter (· ≥ 2) else none

/-- **The Boolean-profile gate** a primitive gate of id `key` on the operands `ops` is, or none when `key` isn't a
Boolean primitive or `ops` doesn't fit its signature. -/
def gateOf? (key : String) (ops : Array ℕ) : Option Claims.Sampling.BGate :=
  if _root_.Flock.Program.isInputId key then (if ops = #[] then some .input else none)
  else if key = "Const1[0x0]_v1" then (if ops = #[] then some (.const false) else none)
  else if key = "Const1[0x1]_v1" then (if ops = #[] then some (.const true) else none)
  else if key = "Not_v1" then (match ops with | #[a] => some (.not a) | _ => none)
  else match arity? "And" key, arity? "Xor" key with
    | some n, _ => if ops.size = n then some (.and ops.toList) else none
    | none, some n => if ops.size = n then some (.xor ops.toList) else none
    | none, none => none

/-- Gate `g` of `P`'s canonical layout as a Boolean-profile gate (`gateOf?`), none where it doesn't locate. -/
def flatGate (P : _root_.Flock.Program.Prog) (g : ℕ) : Option Claims.Sampling.BGate :=
  match P.gate g with
  | .ok (p, ops, _) => gateOf? (P.defs[p]!).key ops
  | .error _ => none

/-- The gate the root's returned reference `j` resolves to, none where it doesn't. -/
def flatOut (P : _root_.Flock.Program.Prog) (j : ℕ) : Option ℕ :=
  (_root_.Flock.Program.resolve P P.rootScope ((P.defs[P.root]!).body.ret.get j)).toOption

/-- **The Program flattened**: its root's gates in canonical layout order, each the Boolean-profile gate of its
primitive on its operands, returning the gates its root's return resolves to. -/
def flatOf (P : _root_.Flock.Program.Prog) : Option Claims.Sampling.Prog :=
  match (List.range (P.defs[P.root]!).gates).mapM (flatGate P) with
  | none => none
  | some gates => match (List.range (P.defs[P.root]!).body.ret.count).mapM (flatOut P) with
    | none => none
    | some outs => some ⟨gates, outs⟩

/-- **The statement reader**: the descriptor, query, design and roots the framing `rd` reads from the message, with
the descriptor decoded and flattened (`flatOf`) as the statement's gates. -/
def parseOf (rd : Claims.Msg → Option (Claims.Msg × Claims.Sampling.Query × Claims.Sampling.Design × List Claims.Msg))
    (x : Claims.Msg) : Option Claims.Sampling.Stmt :=
  match rd x with
  | none => none
  | some (desc, q, d, roots) =>
    match _root_.Flock.Program.decode ⟨desc.toArray⟩ with
    | .ok P => (flatOf P).map fun C => (⟨desc, C, q, d, roots⟩ : Claims.Sampling.Stmt)
    | .error _ => none

/-- **`flatOf` flattens**: a Program's flattening, where `flatOf` gives one, is the Program flattened (`Flattens`). -/
def FlatOfFlattens : Prop :=
  ∀ (P : _root_.Flock.Program.Prog) (C : Claims.Sampling.Prog), flatOf P = some C → Flattens P C

/-- **A deployment reading its statements with `parseOf` has its statements' gates their descriptors flattened.** -/
def ParseOfFlattens : Prop :=
  ∀ {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) rd, D.parse = parseOf rd → StmtFlattens D

end Properties.Flock.Circuit
