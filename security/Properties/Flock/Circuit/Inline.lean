module

public import Definitions.Core.Boolean
public import Properties.Flock.Circuit.OfProgram
public import Properties.Flock.DecodeBoolean

/-!
# C's Boolean inlining (`inline`)

`inline P` is C (`ofProgram P`) with each gate replaced by its primitive's Boolean body, a function of C read only inside
a block's statement and for counts, never to name a block, a boundary or a leaf. Walking C's gates in increasing index:

* an input gate of width `w` becomes `w` one-bit input gates, consecutive, bit `i` at the gate's start plus `i`;
* a computed gate becomes its primitive's Boolean body expanded in place, its gates in the evaluator's order
  (`Prog.evalBool`: a word primitive as its Boolean Definition in `P`'s map, a call, a batch's members and a scan's
  iterations one after another), each parameter leaf wired to the inlined gate of its operand's bit. `bitGate g i` is the
  inlined gate of the body's return leaf `i`, which may be an earlier gate where the body returns a parameter's bit.

Core's Boolean profile has two-input gates and no constant gate, so a §9 `And<n>` or `Xor<n>` is `n - 1` two-input gates
in a chain, `Not` one gate, a `0` bit one constant gate (a row of two empty XORs) and a `1` bit two (its NOT). Where the
evaluator's value is `[]` or shorter than the gate's width, the missing bits are `0` gates, and the circuit ends in one
`0` gate. So the inlining mirrors the evaluator on every description (`InlineEval`).

`bodyAnds` is a Definition's AND count by circuits' rule (`verity/circuits/count.py`): `And<n>` `n - 1`, every other §9
gate, constant bit and input none, a word primitive its body's, a composite the sum over its nodes of members times its
callee's. A figure that credits reads `andSize (inline P)` itself; a report or a limit may read the sum (`InlineAnds`).
-/

@[expose] public section

namespace Flock.OfProgram

open Flock.Program

/-- A gate of the inlining as it is built, its operands by index. -/
inductive RawOp where
  | input
  | and (a b : ℕ)
  | xor (a b : ℕ)
  | not (a : ℕ)
  | zero
  deriving Inhabited

/-- The inlining as it is built: its gates so far. -/
abbrev Build := StateM (Array RawOp)

def emit (o : RawOp) : Build ℕ := modifyGet fun s => (s.size, s.push o)

def zeroW : Build ℕ := emit .zero

def oneW : Build ℕ := do emit (.not (← zeroW))

/-- A §9 gate on the wires `xs`, as `gateVal` reads its operands. -/
def gateW : Flock.Lower.Gate → List ℕ → Build ℕ
  | .and_, [] => oneW
  | .and_, x :: xs => xs.foldlM (fun acc y => emit (.and acc y)) x
  | .xor_, [] => zeroW
  | .xor_, x :: xs => xs.foldlM (fun acc y => emit (.xor acc y)) x
  | .not_, [] => oneW
  | .not_, x :: _ => emit (.not x)
  | .const_ false, _ => zeroW
  | .const_ true, _ => oneW

/-- Reference `r`'s wire, as `refVal` reads its bit; a `0` gate where `refVal` reads `false` for want of a bit. -/
def refW (x : List ℕ) (st : Array ℕ) (outs : Array (List ℕ)) (r : Ref) : Build ℕ :=
  match (if r.p then x[st.getD r.i 0 + r.j]? else (outs.getD r.i [])[r.j]?) with
  | some w => pure w
  | none => zeroW

/-- The wires of `argVals`'s bits. -/
def argW (v : Ref → Build ℕ) (a : Arg) (cnt : ℕ) (k : ℕ → ℕ) : Build (List ℕ) :=
  let rs := a.seq.toArray
  (List.range cnt).mapM fun t => v (rs.getD (k t) default)

/-- `evalDef?` on wires: Definition `d` expanded on its parameter wires `x`, at most `fuel` calls deep, its return's
wires; none where `evalDef?` is none. -/
def inlDef? (P : Prog) : ℕ → ℕ → List ℕ → OptionT Build (List ℕ)
  | 0, _, _ => failure
  | fuel + 1, d, x => do
    let defn ← OptionT.mk (pure P.defs[d]?)
    if defn.prim then
      match Flock.Lower.gateOf? defn.id with
      | some g => return [← (gateW g x : Build ℕ)]
      | none => return (← inlDef? P fuel (← OptionT.mk (pure (P.bodyOf? defn.key))) x)
    let callOf (e : ℕ) (ins : List ℕ) : OptionT Build (List ℕ) := inlDef? P fuel e ins
    let st := paramStarts defn.params
    let outs ← defn.body.nodes.toList.foldlM (init := (#[] : Array (List ℕ))) fun outs nd => do
      let v : Ref → Build ℕ := refW x st outs
      if d ≤ nd.fn then failure
      let callee ← OptionT.mk (pure P.defs[nd.fn]?)
      let leaves (i : ℕ) : ℕ := ((callee.params.getD i default).2).leaves
      match nd.form with
      | .prim | .call =>
        let ins ← (nd.args.toList.mapM fun a => argW v a a.seq.count id : Build (List (List ℕ)))
        outs.push <$> callOf nd.fn ins.flatten
      | .batch =>
        let ms ← (List.range nd.n).mapM fun m => do
          let ins ← ((List.range nd.args.size).mapM fun i =>
            let a := nd.args.getD i default
            match a.dims with
            | none => argW v a a.seq.count id
            | some (dd, L) => argW v a (leaves i) fun t => (t / L * dd) * L + t % L + m * L : Build (List (List ℕ)))
          callOf nd.fn ins.flatten
        return outs.push ms.flatten
      | .scan =>
        let c := leaves 0
        let init ← (argW v (nd.args.getD 0 default) c id : Build (List ℕ))
        let (carry, ys) ← (List.range nd.n).foldlM (init := (init, ([] : List ℕ))) fun (carry, ys) m => do
          let rest ← (((List.range nd.args.size).drop 1).mapM fun i =>
            let a := nd.args.getD i default
            if i ≤ nd.nxs then argW v a (leaves i) fun t => t + m * leaves i else argW v a (leaves i) id
            : Build (List (List ℕ)))
          let o ← callOf nd.fn (carry ++ rest.flatten)
          return (o.take c, ys ++ o.drop c)
        return outs.push (carry ++ ys)
    let rs := defn.body.ret.toArray
    let ws ← ((List.range defn.body.ret.count).mapM fun t => refW x st outs (rs.getD t default) : Build (List ℕ))
    return ws

/-- Gate `g`'s operands below it, in order, as indices (`opsBelow`). -/
def opsBelowN (P : Prog) (g : ℕ) : List ℕ :=
  if inputAt P g then [] else
  match gateAt P g with
  | some (_, ops) => ops.toList.filter (· < g)
  | none => []

/-- What the walk records of one gate of C: its bits' wires, and the inlined gates `[lo, hi)` it emitted. -/
structure GateW where
  bits : Array ℕ
  lo : ℕ
  hi : ℕ
  deriving Inhabited

/-- Gate `g` of C inlined, given what the walk recorded of the gates below it. A body that doesn't evaluate emits
nothing of its own, and its gate's bits are `0` gates. -/
def gateStep (P : Prog) (done : Array GateW) (g : ℕ) : Build GateW := do
  let lo := (← get).size
  let w := widthAt P g
  if inputAt P g then
    let ws ← (List.range w).mapM fun _ => emit .input
    return { bits := ws.toArray, lo, hi := (← get).size }
  match gateAt P g with
  | none => return { bits := #[], lo, hi := lo }
  | some (p, _) =>
    let xs := (opsBelowN P g).flatMap fun a => ((done.getD a default).bits).toList
    let s ← get
    let r ← (inlDef? P (2 * P.defs.size + 2) p xs).run
    let ys ← match r with
      | some ys => pure ys
      | none => do set s; pure []
    let ws ← (List.range w).mapM fun i => match ys[i]? with
      | some y => pure y
      | none => zeroW
    return { bits := ws.toArray, lo, hi := (← get).size }

/-- The walk over C's gates in increasing index, then the closing `0` gate. -/
def walk (P : Prog) : Array GateW × Array RawOp :=
  let r := ((List.range (size P)).foldlM (init := (#[] : Array GateW))
    (fun done g => done.push <$> gateStep P done g) : Build (Array GateW)).run #[]
  (r.1, r.2.push .zero)

/-- A raw gate as core's, at index `g`; an operand not below `g` makes it a `0` gate, which the walk never emits. -/
def RawOp.toOp {N : ℕ} (g : Fin N) : RawOp → Verity.Boolean.Op N
  | .input => .input
  | .and a b => if h : a < g.val ∧ b < g.val then .and ⟨a, lt_trans h.1 g.isLt⟩ ⟨b, lt_trans h.2 g.isLt⟩ else .row [] []
  | .xor a b => if h : a < g.val ∧ b < g.val then .xor ⟨a, lt_trans h.1 g.isLt⟩ ⟨b, lt_trans h.2 g.isLt⟩ else .row [] []
  | .not a => if h : a < g.val then .not ⟨a, lt_trans h g.isLt⟩ else .row [] []
  | .zero => .row [] []

theorem RawOp.toOp_topo {N : ℕ} (g : Fin N) (o : RawOp) : ∀ a ∈ (o.toOp g).args, a < g := by
  intro a ha
  cases o with
  | input => simp [RawOp.toOp, Verity.Boolean.Op.args] at ha
  | zero => simp [RawOp.toOp, Verity.Boolean.Op.args] at ha
  | and x y =>
    simp only [RawOp.toOp] at ha
    split_ifs at ha with h
    · simp only [Verity.Boolean.Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
      rcases ha with rfl | rfl
      · exact h.1
      · exact h.2
    · simp [Verity.Boolean.Op.args] at ha
  | xor x y =>
    simp only [RawOp.toOp] at ha
    split_ifs at ha with h
    · simp only [Verity.Boolean.Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
      rcases ha with rfl | rfl
      · exact h.1
      · exact h.2
    · simp [Verity.Boolean.Op.args] at ha
  | not x =>
    simp only [RawOp.toOp] at ha
    split_ifs at ha with h
    · simp only [Verity.Boolean.Op.args, List.mem_cons, List.not_mem_nil, or_false] at ha
      rcases ha with rfl
      exact h
    · simp [Verity.Boolean.Op.args] at ha

/-- The inlining's gate count: the walk's gates and the closing `0` gate. -/
def inlineN (P : Prog) : ℕ := (walk P).2.size

theorem inlineN_pos (P : Prog) : 0 < inlineN P := by
  unfold inlineN walk
  simp

/-- A wire as a gate of the inlining; past its end, the closing `0` gate, which the walk never records. -/
def wireAt (P : Prog) (x : ℕ) : Fin (inlineN P) :=
  if h : x < inlineN P then ⟨x, h⟩ else ⟨inlineN P - 1, by have := inlineN_pos P; omega⟩

/-- Bit `i` of gate `g` of C, as a wire of the walk; the closing `0` gate where the walk recorded none. -/
def bitW (P : Prog) (g i : ℕ) : ℕ := ((walk P).1.getD g default).bits.getD i (inlineN P - 1)

/-- **C's Boolean inlining.** Its outputs are the inlined gates of C's outputs' bits. -/
def inline (P : Prog) : Verity.Boolean.Circuit where
  N := inlineN P
  op g := ((walk P).2.getD g.val default).toOp g
  topo g := RawOp.toOp_topo g _
  outputs := Finset.univ.filter fun x =>
    ∃ g ∈ (ofProgram P).outputs, ∃ i < (ofProgram P).w g, wireAt P (bitW P g.val i) = x

/-- Where bit `i` of C's gate `g` is in the inlining: an input gate's bit `i`, else its body's return leaf `i`. -/
def bitGate (P : Prog) (g : Fin (ofProgram P).N) (i : Fin ((ofProgram P).w g)) : Fin (inline P).N :=
  wireAt P (bitW P g.val i)

/-- The inlining's inputs read as C's input words: bit `i` of gate `g` is the inlining's gate `bitGate g i`. -/
def inputWords (P : Prog) (a : Fin (inline P).N → Bool) : (ofProgram P).Trace :=
  fun g => Verity.ofBits fun i => a (bitGate P g i)

/-- The inlined gates of C's gates `S`: the gates each one's inlining emitted. -/
def blockGates (P : Prog) (S : Finset (Fin (ofProgram P).N)) : Finset (Fin (inline P).N) :=
  Finset.univ.filter fun x => ∃ g ∈ S, ((walk P).1.getD g.val default).lo ≤ x.val ∧
    x.val < ((walk P).1.getD g.val default).hi

/-- Definition `d`'s ANDs by circuits' rule, at most `fuel` calls deep. -/
def andsDef (P : Prog) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | fuel + 1, d =>
    let defn := P.defs[d]!
    if defn.prim then
      match gateKind? defn.id with
      | some (.and n) => n - 1
      | some _ => 0
      | none => match P.bodyOf? defn.key with
        | some b => andsDef P fuel b
        | none => 0
    else (defn.body.nodes.toList.map fun nd => nd.members * andsDef P fuel nd.fn).sum

/-- **A Definition's ANDs** (`count.py`'s `ands`), at the evaluator's depth. -/
def bodyAnds (P : Prog) (p : ℕ) : ℕ := andsDef P (2 * P.defs.size + 2) p

/-- Gate `g`'s ANDs: its primitive's, none for an input. -/
def gateAnds (P : Prog) (g : Fin (ofProgram P).N) : ℕ :=
  if inputAt P g then 0 else
  match gateAt P g with
  | some (p, _) => bodyAnds P p
  | none => 0

/-- **`InlineEval`, (I): the inlining computes C's bits.** On every description, the inlining's gate `bitGate g i` on
inputs `a` is bit `i` of C's gate `g` on the input words `a` reads. -/
def InlineEval : Prop :=
  ∀ (P : Prog) (a : Fin (inline P).N → Bool) (g : Fin (ofProgram P).N) (i : Fin ((ofProgram P).w g)),
    (inline P).eval a (bitGate P g i) = ((ofProgram P).eval (inputWords P a) g).getLsbD i

/-- **`InlineAnds`, (A): the inlining's ANDs are at most the bodies' ANDs**, over its computed gates and over the
inlined gates of each block of `partOf P blocks`, on a description `decode` accepts. -/
def InlineAnds : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P → ∀ blocks : Array (Array (ℕ × ℕ)),
    (inline P).toCircuit.andSize (inline P).toCircuit.computed ≤ (∑ g : Fin (ofProgram P).N, gateAnds P g) ∧
      ∀ u, (inline P).toCircuit.andSize (blockGates P ((partOf P blocks).gates u)) ≤
        ∑ g ∈ (partOf P blocks).gates u, gateAnds P g

end Flock.OfProgram
