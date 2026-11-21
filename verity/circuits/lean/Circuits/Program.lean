/-
`Flock.Program`'s canonical layout (§6) and scope resolution (§7) over a Program `Circuits.Descriptor` decodes.
-/
module

public import Circuits.Descriptor

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock.Program

open Lean (Json JsonNumber)


/-! ### §6 The canonical layout and §7 scope resolution -/

/-- An activation: the root (no parameters), a Call cut as its Definition's activation in a one-Call Program (its parameter `i`'s
leaf `j` is Input gate `starts[i] + j`), or child `(node, member)` of an activation. -/
inductive Scope where
  | top (d : Nat) (off : Nat) (starts : Option (Array Nat))
  | child (parent : Scope) (d : Nat) (off : Nat) (node : Nat) (member : Nat)
  deriving Inhabited

/-- The Definition whose activation `sc`'s chain of parents starts from. -/
def Scope.base : Scope → Nat
  | .top d _ _ => d
  | .child parent _ _ _ _ => parent.base

def Scope.d : Scope → Nat
  | .top d _ _ => d
  | .child _ d _ _ _ => d

def Scope.off : Scope → Nat
  | .top _ o _ => o
  | .child _ _ o _ _ => o

/-- Reference `k` of member `m` of a node's argument (§3): the whole argument when shared or unmapped. -/
@[inline] def Arg.memberRef (a : Arg) (m k : Nat) : Ref :=
  match a.dims with
  | none => a.seq.get k
  | some (d, L) => a.seq.get ((k / L * d + m) * L + k % L)

/-- One step of `layout.resolve`: the gate a body-scoped reference denotes in an activation when it is a prescribed input's or
a primitive's, else the reference it denotes one activation up (through a parameter) or down (through a node's output). -/
def resolveStep (P : Prog) : Scope × Ref → V (Nat ⊕ (Scope × Ref))
  | (.top _ _ none, ⟨true, _, _⟩) => throw "the root has no parameters (prescribed inputs are Input gates)"
  | (.top _ _ (some starts), ⟨true, i, j⟩) => match starts[i]? with
    | some s => pure (.inl (s + j))
    | none => throw s!"parameter {i} of the Call does not exist"
  | (.child parent d off k m, ⟨true, i, j⟩) => do
    let node := (P.defs[parent.d]!).body.nodes[k]!
    let a := node.args[i]!
    match node.form with
    | .scan =>
      if i == 0 then
        if m == 0 then pure (.inr (parent, a.seq.get j))
        else pure (.inr (.child parent d (off - (P.defs[d]!).gates) k (m - 1), (P.defs[d]!).body.ret.get j))
      else pure (.inr (parent, if i ≤ node.nxs then a.seq.get (m * (a.dims.map (·.2)).getD 0 + j) else a.seq.get j))
    | .batch => pure (.inr (parent, a.memberRef m j))
    | _ => pure (.inr (parent, a.seq.get j))
  | (sc, ⟨false, k, j⟩) => do
    let body := (P.defs[sc.d]!).body
    let node := body.nodes[k]!
    let o := sc.off + body.offsets[k]!
    let callee := P.defs[node.fn]!
    match node.form with
    | .prim => pure (.inl o)
    | .call => pure (.inr (.child sc node.fn o k 0, callee.body.ret.get j))
    | .batch =>
      let L := callee.outLeaves
      if callee.prim then pure (.inl (o + j / L))
      else pure (.inr (.child sc node.fn (o + j / L * callee.gates) k (j / L), callee.body.ret.get (j % L)))
    | .scan =>
      if node.n == 0 then throw s!"a scan with no iterations has no output to read"
      let .tuple #[carry, y] := callee.ret | throw "a scan step returns (carry, y)"
      let c := carry.leaves
      let (m, jj) := if j < c then (node.n - 1, j) else ((j - c) / y.leaves, c + (j - c) % y.leaves)
      pure (.inr (.child sc node.fn (o + m * callee.gates) k m, callee.body.ret.get jj))

/-- The gate layout's refusal of a root that reads an input node recut (`Body.recut`). -/
def recutGates (P : Prog) : String :=
  s!"recut-gates: input nodes {((P.defs[P.root]!).body.recut.map (·.1)).toList} are read in leaves of part " ++
    "of their width, which the gate layout doesn't lay out"

/-- `layout.resolve`: the gate a body-scoped reference denotes in an activation, up through parameters and down through node
outputs until a primitive. Refused (`recut-gates`, `Program.circuit`'s refusal) in an activation under the root's while
the root reads an input node recut (`Body.recut`): a gate is one leaf of its width, and the layout has no operand of part
of a gate. An activation of a Definition taken on its own (`Scope.top d _ (some starts)`, `d` not the root) resolves to
its own parameters and gates, and never reads the root's. -/
def resolve (P : Prog) (sc : Scope) (r : Ref) : V Nat :=
  if sc.base != P.root || (P.defs[P.root]!).body.recut.isEmpty then Steps.run "scope resolution" (resolveStep P) Steps.FUEL (sc, r)
  else throw (recutGates P)

theorem resolve_run {P : Prog} {sc : Scope} {r : Ref} {a : Nat} (h : resolve P sc r = .ok a) :
    Steps.run "scope resolution" (resolveStep P) Steps.FUEL (sc, r) = .ok a := by
  unfold resolve at h
  split at h
  · exact h
  · cases h

theorem resolve_of {P : Prog} {sc : Scope} (h : sc.base ≠ P.root ∨ (P.defs[P.root]!).body.recut = #[]) (r : Ref) :
    resolve P sc r = Steps.run "scope resolution" (resolveStep P) Steps.FUEL (sc, r) := by
  unfold resolve
  rw [if_pos]
  rcases h with h | h <;> simp [h]

theorem resolve_eq {P : Prog} (h : (P.defs[P.root]!).body.recut = #[]) (sc : Scope) (r : Ref) :
    resolve P sc r = Steps.run "scope resolution" (resolveStep P) Steps.FUEL (sc, r) :=
  resolve_of (.inr h) r

theorem resolve_cond {P : Prog} {sc : Scope} {r : Ref} {a : Nat} (h : resolve P sc r = .ok a) :
    sc.base ≠ P.root ∨ (P.defs[P.root]!).body.recut = #[] := by
  unfold resolve at h
  split at h
  · next hc =>
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Array.isEmpty_iff] at hc
    exact hc
  · cases h

theorem resolve_recut {P : Prog} {sc : Scope} {r : Ref} {a : Nat} (hb : sc.base = P.root)
    (h : resolve P sc r = .ok a) : (P.defs[P.root]!).body.recut = #[] :=
  (resolve_cond h).resolve_left (· hb)

/-- `partition_object.order_violation`: the first reference naming neither a parameter of its body nor an earlier node of it
(a body's `ret` may name any node), over every body the root reaches, root first and callees in node order, each
Definition once; its detail `{definition, node, arg, reference, why}`, or none when the program is topological. -/
def Prog.orderViolation (P : Prog) : Option Json := Id.run do
  let mut seen : Std.HashSet Nat := {}
  let mut todo : Array Nat := #[P.root]
  let mut at_ := 0
  for _ in [0:Steps.FUEL] do
    if at_ ≥ todo.size then break
    let d := todo[at_]!
    at_ := at_ + 1
    if seen.contains d then continue
    seen := seen.insert d
    let defn := P.defs[d]!
    if defn.prim then continue
    let body := defn.body
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      for i in [0:node.args.size] do
        for part in node.args[i]!.seq.parts do
          let names : Array (Bool × Nat) := match part with
            | .affine p idx _ c => if c > 0 then #[(p, idx)] else #[]
            | .strided p idx _ _ _ _ c => if c > 0 then #[(p, idx)] else #[]
            | .explicit it => it.map fun r => (r.p, r.i)
            | .grid _ lanes => lanes.map fun l => (l.p, l.i)
          for (p, idx) in names do
            if !(if p then idx < body.params.size else idx < k) then
              return some (Json.mkObj [("definition", Json.str defn.key), ("node", Lean.toJson k), ("arg", Lean.toJson i),
                ("reference", Json.arr #[Json.str (if p then "p" else "n"), Lean.toJson idx]), ("why", Json.str "not topological")])
      todo := todo.push node.fn
  if at_ < todo.size then return some (Json.mkObj [("why", Json.str "a walk of more steps than this verifier follows")])
  return none

/-- The root body's Calls (`partition_object.calls`): every node that is not an input node, with its root-body index.
A node whose callee is no Definition is a Call, which its reader refuses at its callee; a root that is no Definition has
none. -/
def Prog.calls (P : Prog) : Array Nat :=
  match P.defs[P.root]? with
  | none => #[]
  | some r => r.body.nodes.zipIdx.filterMap fun (x, k) => match P.defs[x.fn]? with
    | some f => if f.prim && isInputId f.key then none else some k
    | none => some k

def Prog.rootScope (P : Prog) : Scope := .top P.root 0 none

/-- The activation of root node `k`'s member `m`. -/
def Prog.child (P : Prog) (sc : Scope) (k m : Nat) : Scope :=
  let body := (P.defs[sc.d]!).body
  let node := body.nodes[k]!
  .child sc node.fn (sc.off + body.offsets[k]! + m * (P.defs[node.fn]!).gates) k m

/-- Member `m` of a primitive node, or the node itself: its operands, per argument in order, repeats kept (§6). Refused
where `resolve` refuses, an argument or none. -/
def operands (P : Prog) (sc : Scope) (node : Node) (m : Nat) : V (Array Nat) := do
  if sc.base == P.root && !(P.defs[P.root]!).body.recut.isEmpty then throw (recutGates P)
  let callee := P.defs[node.fn]!
  let mut ops := #[]
  for i in [0:node.args.size] do
    let a := node.args[i]!
    let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
    for t in [0:cnt] do ops := ops.push (← resolve P sc (if node.form == .batch then a.memberRef m t else a.seq.get t))
  return ops

theorem operands_of {P : Prog} {sc : Scope} (h : sc.base ≠ P.root ∨ (P.defs[P.root]!).body.recut = #[]) (node : Node)
    (m : Nat) : operands P sc node m = (do
      let callee := P.defs[node.fn]!
      let mut ops := #[]
      for i in [0:node.args.size] do
        let a := node.args[i]!
        let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
        for t in [0:cnt] do ops := ops.push (← resolve P sc (if node.form == .batch then a.memberRef m t else a.seq.get t))
      return ops : V (Array Nat)) := by
  have hc : (sc.base == P.root && !(P.defs[P.root]!).body.recut.isEmpty) = false := by
    rcases h with h | h <;> simp [h]
  unfold operands
  rw [hc]
  rfl

theorem operands_cond {P : Prog} {sc : Scope} {node : Node} {m : Nat} {ops : Array Nat}
    (h : operands P sc node m = .ok ops) : sc.base ≠ P.root ∨ (P.defs[P.root]!).body.recut = #[] := by
  cases hc : (sc.base == P.root && !(P.defs[P.root]!).body.recut.isEmpty)
  · simp only [Bool.and_eq_false_iff, beq_eq_false_iff_ne, Bool.not_eq_false', Array.isEmpty_iff] at hc
    exact hc
  · unfold operands at h
    rw [hc] at h
    cases h

/-- One step of locating gate `g` from an activation: its primitive, operands and activation when the node holding it is a
primitive's, else the activation of the node's member that holds it. -/
def gateStep (P : Prog) (g : Nat) (sc : Scope) : V ((Nat × Array Nat × Scope) ⊕ Scope) := do
  let body := (P.defs[sc.d]!).body
  let rel := g - sc.off
  let some k := (Array.range body.nodes.size).find? fun k => body.offsets[k]! ≤ rel && rel < body.offsets[k + 1]!
    | throw s!"gate {g} outside the circuit"
  let node := body.nodes[k]!
  let callee := P.defs[node.fn]!
  let m := if node.form == .prim || node.form == .call then 0 else (rel - body.offsets[k]!) / callee.gates
  if callee.prim then return .inl (node.fn, ← operands P sc node m, sc)
  return .inr (P.child sc k m)

/-- A gate's primitive and operands (`layout.Circuit.gate`): located through the offsets along the call hierarchy. -/
def Prog.gate (P : Prog) (g : Nat) : V (Nat × Array Nat × Scope) := do
  if !(P.defs[P.root]!).body.recut.isEmpty then throw (recutGates P)
  if g ≥ (P.defs[P.root]!).gates then throw s!"gate {g} outside the circuit"
  Steps.run "gate location" (gateStep P g) Steps.FUEL P.rootScope

theorem Prog.gate_of {P : Prog} (h : (P.defs[P.root]!).body.recut = #[]) (g : Nat) : P.gate g = (do
    if g ≥ (P.defs[P.root]!).gates then throw s!"gate {g} outside the circuit"
    Steps.run "gate location" (gateStep P g) Steps.FUEL P.rootScope : V (Nat × Array Nat × Scope)) := by
  unfold Prog.gate
  rw [h]
  rfl

theorem Prog.gate_recut {P : Prog} {g : Nat} {x : Nat × Array Nat × Scope} (h : P.gate g = .ok x) :
    (P.defs[P.root]!).body.recut = #[] := by
  cases hc : (P.defs[P.root]!).body.recut.isEmpty
  · unfold Prog.gate at h
    rw [hc] at h
    cases h
  · exact Array.isEmpty_iff.mp hc

/-- A Program's canonical layout (§6) read gate by gate, as `Prog.gate` locates each gate: the width of its value in bits,
whether it is an input gate, whether it is an input bound at registration (`RegisteredInput<w>_v1`, §5.2), and the gates
its operands name (repeats kept). It is `Verity.Circuit`'s wiring and widths, with no operation. -/
structure GateTable where
  w : Array Nat
  isInput : Array Bool
  registered : Array Bool
  reads : Array (Array Nat)

def GateTable.N (c : GateTable) : Nat := c.w.size

/-- `c` with one more gate. -/
def GateTable.push (c : GateTable) : Nat × Bool × Bool × Array Nat → GateTable
  | (w, inp, reg, ops) =>
    { w := c.w.push w, isInput := c.isInput.push inp, registered := c.registered.push reg, reads := c.reads.push ops }

end Flock.Program
