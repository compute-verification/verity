module

public import Flock.Lower

/-!
# A Boolean Definition's value

`Prog.evalBool P d x` is Definition `d` of the decoded Program `P` evaluated on its parameter bits `x`: the one evaluator of
a Boolean Definition, whose every primitive is one of PROTOCOL.md §9's gates, and of a word primitive, which evaluates as
its Boolean Definition in `P`'s map (`P.bodyOf?`, PROTOCOL.md §4.9). It is the meaning `Flock.Lower.lower`'s template is
held to and the operation of the word-level circuit's gates (`boolSem`), defined once here. Its one step (`walkStep`) is
generic in what a wire is (`Wires`): a bit here (`Wires.bool`), an emitted gate in C's inlining (`Circuit.Inline`).

Its body's nodes run in order, each node's value the list of its output leaves, low leaf first, as PROTOCOL.md §6 resolves
a reference: parameter `i`'s leaf `j` is bit `start i + j` of `x`, node `i`'s leaf `j` that node's leaf `j`.
- **A primitive** is its gate on every operand of every argument: `And<n>` their conjunction, `Xor<n>` their sum, `Not`
  the negation of its one, a `Const1` its constant. A primitive that is no gate is its Boolean Definition on the same
  bits; a body is decoded after its word primitive, so this is the one call to a Definition not before its caller, and a
  Boolean Definition reaches no word primitive, so a chain of calls is at most twice as deep as `P` has Definitions.
- **A call** is the callee on its arguments' leaves, in order.
- **A batch** of `n` is the callee on each member's arguments, its members' values one after another. An argument mapped
  over an axis of `(d, L)` gives member `m` its leaf `t` at reference `(t / L · d) · L + t % L + m · L`; one not mapped
  is the same for every member.
- **A scan** of `n` iterations threads its carry, the callee's first parameter: iteration 0 reads `init`, iteration `m`
  the carry iteration `m − 1` returned, an `xs` argument its leaf `t` at reference `t + m · cnt`, a shared one the same
  every iteration. Its value is the last carry, then each iteration's `y`.

A reference reads its sequence's references in order (`Seq.toArray`). Where it meets a primitive that is neither a §9 gate
nor in `P`'s map (an input), or a node's callee not before its caller, its value is `[]`.
-/

@[expose] public section

namespace Flock.Program

/-- A §9 gate's value on its operands. -/
def gateVal : Flock.Lower.Gate → List Bool → Bool
  | .and_, ops => ops.all id
  | .xor_, ops => ops.foldl xor false
  | .not_, ops => !(ops.headD false)
  | .const_ b, _ => b

/-- Each parameter's first bit. -/
def paramStarts (params : Array (String × Ty)) : Array Nat :=
  (params.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.2.leaves)) (#[], 0)).1

/-- Reference `r`'s wire, given the parameter wires `x` (parameter `i` from wire `st[i]`) and the nodes' values so far;
`default` (`false`) where it reads past them. -/
def refVal {W : Type} [Inhabited W] (x : List W) (st : Array Nat) (outs : Array (List W)) (r : Ref) : W :=
  if r.p then x.getD (st.getD r.i 0 + r.j) default else (outs.getD r.i []).getD r.j default

/-- Wires `k 0, …, k (cnt − 1)` of argument `a`'s references, in order (`Seq.toArray`). -/
def argVals {W : Type} (v : Ref → W) (a : Arg) (cnt : Nat) (k : Nat → Nat) : List W :=
  let rs := a.seq.toArray
  (List.range cnt).map fun t => v (rs.getD (k t) default)

/-- **What a wire is**, for the walk: a primitive's §9 gate on its operands' wires (`gate`), in a monad `m` with
failure, which a lookup's `Option` enters by `lift`. -/
structure Wires (m : Type → Type) (W : Type) where
  gate : Flock.Lower.Gate → List W → m W
  lift : {α : Type} → Option α → m α

/-- Wires as bits: the gate's value. -/
def Wires.bool : Wires Option Bool := ⟨fun g x => some (gateVal g x), fun o => o⟩

/-- **One call of the walk**, its recursive call `ev`: Definition `d` on its parameter wires `x`, its return's wires. -/
def walkStep {m : Type → Type} [Monad m] [Alternative m] {W : Type} [Inhabited W] (I : Wires m W) (P : Prog)
    (ev : Nat → List W → m (List W)) (d : Nat) (x : List W) : m (List W) := do
  let defn ← I.lift P.defs[d]?
  if defn.prim then
    match Flock.Lower.gateOf? defn.id with
    | some g => return [← I.gate g x]
    | none => return (← ev (← I.lift (P.bodyOf? defn.key)) x)
  let callOf (e : Nat) (ins : List W) : m (List W) := ev e ins
  let st := paramStarts defn.params
  let outs ← defn.body.nodes.toList.foldlM (init := (#[] : Array (List W))) fun outs nd => do
    let v := refVal x st outs
    if d ≤ nd.fn then failure
    let callee ← I.lift P.defs[nd.fn]?
    let leaves (i : Nat) : Nat := ((callee.params.getD i default).2).leaves
    match nd.form with
    | .prim | .call => outs.push <$> callOf nd.fn (nd.args.toList.flatMap fun a => argVals v a a.seq.count id)
    | .batch =>
      let ms ← (List.range nd.n).mapM fun m => callOf nd.fn ((List.range nd.args.size).flatMap fun i =>
        let a := nd.args.getD i default
        match a.dims with
        | none => argVals v a a.seq.count id
        | some (dd, L) => argVals v a (leaves i) fun t => (t / L * dd) * L + t % L + m * L)
      return outs.push ms.flatten
    | .scan =>
      let c := leaves 0
      let init := argVals v (nd.args.getD 0 default) c id
      let (carry, ys) ← (List.range nd.n).foldlM (init := (init, ([] : List W))) fun (carry, ys) m => do
        let ins := carry ++ ((List.range nd.args.size).drop 1).flatMap fun i =>
          let a := nd.args.getD i default
          if i ≤ nd.nxs then argVals v a (leaves i) fun t => t + m * leaves i else argVals v a (leaves i) id
        let o ← callOf nd.fn ins
        return (o.take c, ys ++ o.drop c)
      return outs.push (carry ++ ys)
  let rs := defn.body.ret.toArray
  return (List.range defn.body.ret.count).map fun t => refVal x st outs (rs.getD t default)

/-- Definition `d`'s value on `x`, at most `fuel` calls deep: the walk on bits. -/
def evalDef? (P : Prog) : Nat → Nat → List Bool → Option (List Bool)
  | 0, _, _ => none
  | fuel + 1, d, x => walkStep Wires.bool P (evalDef? P fuel) d x

/-- **Boolean Definition `d`'s value on its parameter bits `x`**, low bit first in its parameters' order: its return's
bits, `[]` where it is not Boolean; a word primitive's is its Boolean Definition's. -/
def Prog.evalBool (P : Prog) (d : Nat) (x : List Bool) : List Bool :=
  (evalDef? P (2 * P.defs.size + 2) d x).getD []

end Flock.Program
