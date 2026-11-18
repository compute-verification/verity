module

public import Flock.TmplUnroll
public import Circuits.Program

/-!
# A Boolean Definition lowered to its template

`lower P d inPorts outPorts` is the `flock-ir-tmpl/v2` template (`Flock.Tmpl`) of Definition `d` of the decoded Program
`P`: every leaf `Value<1>` and every primitive it reaches one of PROTOCOL.md §9's gates. It is built from the
Definition's own structure, in the description's size, and never from unrolled rows.

- **One Definition a template Definition.** Each composite Definition `d` reaches is lowered once (`lowerDef`), its
  callees first. A primitive batched by a node is a template Definition of one gate (`gateDef`), lowered once per
  primitive.
- **A body, node by node.** A wire is a GF(2) sum of the body's references (`Form`). A primitive node's gate is computed in
  place: XOR and NOT add forms and write no row, a constant is a form, and an AND of two forms neither constant writes one
  row `(c, A, B)`, `z_c = (A·z)(B·z)`; an AND of more inputs is that, left to right. A `call` node is a `CALL` of one copy,
  and a `batch` node one `CALL` of `n` copies whose inputs step per copy, the steps read off the argument's references
  (`stepping`), never member by member. A `scan` node is two `CALL`s: its first iteration on `init`, then the rest, each
  reading the previous one's carry (`Work.scan`).
- **Bindings and returns are columns.** A call's input that is not one reference, and a Definition's returned leaf that is
  not one of its own columns, is committed first: one row `(c, form, 1)`.
- **The block.** The root holds the input rows (`z_c = z_c·z_c`, bit `i` at column `i`), places `d` once at the end of
  the input rows' words, copies `d`'s returns to the output rows, aligned to the output group's words, and ends with the
  constant, as `Flock.ReadNet.readTmpl` lays a read out. Input bit `i` is port bit `i`, the ports `inPorts` concatenated.

`lower` ends by running the reader's checks on what it wrote (`selfCheck`), so its output passes them
(`lower_checked`) and unrolls (`FlockSoundness.Lower.lower_unrolls`). A refusal there is `SELF_CHECK`'s, a completeness
failure of the lowering, never a refusal of its input; the tests hold it to never happening on an honest Definition.

`render` writes the template as `Tmpl.parse` reads it, and `inBit`/`outBit` name port bit `k`'s column. That the
unrolled template evaluates as the Definition is a named premise of the guarantees over the door, not stated here.
-/

@[expose] public section

namespace Flock.Lower

open Flock.Program

/-- A GF(2) sum of a Definition's references before its columns are counted: own column `c` is atom `2c`, input `i` atom
`2i + 1`, each named once, ascending; and whether the constant 1 is in it. -/
structure Form where
  atoms : Array Nat := #[]
  one : Bool := false
  deriving Inhabited, BEq

def Form.add (f g : Form) : Form := ⟨Tmpl.canon (f.atoms ++ g.atoms), f.one != g.one⟩
def Form.own (c : Nat) : Form := ⟨#[2 * c], false⟩
def Form.input (i : Nat) : Form := ⟨#[2 * i + 1], false⟩
def Form.const (b : Bool) : Form := ⟨#[], b⟩

/-- The one atom a form is, with no constant. -/
def Form.single? (f : Form) : Option Nat := if f.atoms.size == 1 && !f.one then f.atoms[0]? else none

/-- A §9 gate, by its primitive's id. -/
inductive Gate where
  | and_ | xor_ | not_ | const_ (b : Bool)

def family? (pre id : String) : Bool :=
  id.startsWith pre && id.endsWith "_v1" &&
    let n := ((id.drop pre.length).dropEnd 3).toString
    n.length > 0 && n.all Char.isDigit && n.front != '0'

def gateOf? (id : String) : Option Gate :=
  if id == "Not_v1" then some .not_
  else if id == "Const1[0x0]_v1" then some (.const_ false)
  else if id == "Const1[0x1]_v1" then some (.const_ true)
  else if family? "And" id then some .and_
  else if family? "Xor" id then some .xor_
  else none

/-- Where a node's output leaves are: a primitive node's one gate, or `count` copies at own columns `base + m · stride` of
a Definition whose leaf `l` is its column `rets[l]`. -/
inductive Out where
  | gate (f : Form)
  | copies (base stride : Nat) (rets : Array Nat)
  /-- a scan's `n` iterations at `base + m · stride`, each returning its `c` carry leaves then its `y`: the node's leaf
  `j < c` is the last iteration's carry leaf `j`, leaf `c + m · |y| + t` iteration `m`'s `y` leaf `t` -/
  | scan (base stride : Nat) (rets : Array Nat) (c n : Nat)
  deriving Inhabited

/-- A call's binding of its callee's input before the caller's columns are counted: copy `m` reads atom `a + m · step`. -/
structure Bind where
  a : Nat
  step : Nat
  deriving Inhabited

structure Pending where
  def_ : Nat
  count : Nat
  base : Nat
  stride : Nat
  binds : Array Bind

/-- A Definition being lowered: its rows and calls so far, its next own column, and its nodes' outputs. -/
structure Work where
  rows : Array (Nat × Form × Form) := #[]
  calls : Array Pending := #[]
  next : Nat := 0
  outs : Array Out := #[]

def Work.row (w : Work) (a b : Form) : Work × Form :=
  ({ w with rows := w.rows.push (w.next, a, b), next := w.next + 1 }, Form.own w.next)

/-- `f` as one atom: itself if it is one, else a row `(c, f, 1)` committing it. -/
def Work.commit (w : Work) (f : Form) : Work × Nat :=
  match f.single? with
  | some x => (w, x)
  | none => let (w', _) := w.row f (Form.const true); (w', 2 * w.next)

/-- `f` as one of the Definition's own columns. -/
def Work.commitOwn (w : Work) (f : Form) : Work × Nat :=
  match f.single? with
  | some x => if x % 2 == 0 then (w, x / 2) else let (w', _) := w.row f (Form.const true); (w', w.next)
  | none => let (w', _) := w.row f (Form.const true); (w', w.next)

def Work.and (w : Work) (a b : Form) : Work × Form :=
  if a.atoms.isEmpty then (w, if a.one then b else {})
  else if b.atoms.isEmpty then (w, if b.one then a else {})
  else w.row a b

def Work.op (w : Work) (g : Gate) (ops : Array Form) : V (Work × Form) :=
  match g with
  | .const_ b => pure (w, Form.const b)
  | .not_ => match ops[0]? with
    | some o => if ops.size == 1 then pure (w, o.add (Form.const true)) else throw "Not_v1: not one operand"
    | none => throw "Not_v1: not one operand"
  | .xor_ => if ops.size < 2 then throw "Xor: fewer than two operands" else pure (w, ops.foldl Form.add {})
  | .and_ => match ops[0]? with
    | some o0 => if ops.size < 2 then throw "And: fewer than two operands" else
        pure ((ops.extract 1 ops.size).foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) (w, o0))
    | none => throw "And: fewer than two operands"

/-- Each parameter's first input. -/
def starts (params : Array (String × Ty)) : Array Nat :=
  (params.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.2.leaves)) (#[], 0)).1

/-- Reference `r`'s form, the same for every copy. -/
def Work.formOf (w : Work) (st : Array Nat) (r : Ref) : V Form :=
  if r.p then match st[r.i]? with
    | some s => pure (Form.input (s + r.j))
    | none => throw s!"parameter {r.i}: none"
  else match w.outs[r.i]? with
    | some (.gate f) => if r.j == 0 then pure f else throw s!"node {r.i}: a gate has one leaf, not leaf {r.j}"
    | some (.copies base stride rets) =>
      if rets.isEmpty then throw s!"node {r.i}: returns nothing"
      else pure (Form.own (base + r.j / rets.size * stride + rets[r.j % rets.size]!))
    | some (.scan base stride rets c n) =>
      let y := rets.size - c
      if r.j < c then pure (Form.own (base + (n - 1) * stride + rets[r.j]!))
      else if y == 0 then throw s!"node {r.i}: a scan's leaf {r.j} past its carry, of no y"
      else pure (Form.own (base + (r.j - c) / y * stride + rets[c + (r.j - c) % y]!))
    | none => throw s!"node {r.i}: not an earlier node"

/-- Reference `r` stepping by `s` leaves a copy, as one atom and its step: a parameter's leaves, or a node's copies whose
members the step crosses whole. -/
def Work.stepOf (w : Work) (st : Array Nat) (r : Ref) (s : Nat) : V Bind :=
  if r.p then match st[r.i]? with
    | some s0 => pure ⟨2 * (s0 + r.j) + 1, 2 * s⟩
    | none => throw s!"parameter {r.i}: none"
  else match w.outs[r.i]? with
    | some (.copies base stride rets) =>
      let L := rets.size
      if L == 0 || s % L != 0 then throw s!"node {r.i}: a batch argument stepping {s} leaves across its copies of {L}"
      else pure ⟨2 * (base + r.j / L * stride + rets[r.j % L]!), 2 * (s / L * stride)⟩
    | some (.scan base stride rets c _) =>
      let y := rets.size - c
      if r.j < c || y == 0 || s % y != 0 then
        throw s!"node {r.i}: an argument stepping {s} leaves through a scan's carry, or across its iterations' y of {y}"
      else pure ⟨2 * (base + (r.j - c) / y * stride + rets[c + (r.j - c) % y]!), 2 * (s / y * stride)⟩
    | some (.gate _) => throw s!"node {r.i}: a batch argument stepping across one gate"
    | none => throw s!"node {r.i}: not an earlier node"

/-- The part of `q` holding reference `k`. -/
def partIx (q : Seq) (k : Nat) : Nat :=
  (List.range q.parts.size).foldl (fun i x => if q.starts[x]! ≤ k then x else i) 0

/-- References `k0 + m · L` of `q` for `m < n`, as reference `k0` stepping `L` leaves a copy: within one affine part. -/
def stepping (q : Seq) (k0 n L : Nat) (what : String) : V (Ref × Nat) :=
  if n ≤ 1 then pure (q.get k0, 0) else
  let i := partIx q k0
  if partIx q (k0 + (n - 1) * L) != i then throw s!"{what} across two parts of its references"
  else match q.parts[i]! with
    | .affine p ix b _ => pure (⟨p, ix, b + (k0 - q.starts[i]!)⟩, L)
    | _ => throw s!"{what} stepping through references that are not affine"

/-- Copy `m`'s reference `t` of argument `a` of a batch of `n`, for every `m` at once: where copy 0's is, and the leaves
it steps a copy. An argument mapped over an axis steps by its element's leaves within one affine part; one not mapped is
the same for every copy. -/
def bindRef (a : Arg) (n t : Nat) : V (Ref × Nat) :=
  match a.dims with
  | none => pure (a.seq.get t, 0)
  | some (d, L) =>
    stepping a.seq ((t / L * d) * L + t % L) n L "a batch argument"

/-- A Definition's resolved rows and calls, its columns counted. -/
def Work.finish (w : Work) (ins : Nat) : Tmpl.Def :=
  let cols := w.next
  let ref := fun (x : Nat) => if x % 2 == 0 then x / 2 else cols + x / 2
  let refs := fun (f : Form) => Tmpl.canon (f.atoms.map ref ++ (if f.one then #[cols + ins] else #[]))
  let seg := fun (b : Bind) => ({ len := 1, at_ := ref b.a, step := b.step / 2, reps := 1, rstride := 0 } : Tmpl.Seg)
  let merge := fun (ss : Array Tmpl.Seg) (s : Tmpl.Seg) => match ss.back? with
    | some l =>
      if l.reps == 1 && l.step == s.step && l.at_ + l.len == s.at_ && (decide (l.at_ < cols) == decide (s.at_ < cols)) then
        ss.pop.push { l with len := l.len + 1 }
      else ss.push s
    | none => ss.push s
  { cols, ins, rows := w.rows.map fun (c, a, b) => (c, refs a, refs b),
    calls := w.calls.map fun c =>
      { def_ := c.def_, count := c.count, base := c.base, stride := c.stride, segs := (c.binds.map seg).foldl merge #[] } }

/-- The template so far: its Definitions, and each lowered Definition of `P` (or primitive batched) by index: its template
Definition and its returns' columns. -/
structure St where
  defs : Array Tmpl.Def := #[]
  done : Std.HashMap Nat (Nat × Array Nat) := {}
  gates : Std.HashMap Nat (Nat × Array Nat) := {}

abbrev M := StateT St V

def emit (d : Tmpl.Def) (rets : Array Nat) : M (Nat × Array Nat) := do
  let s ← get
  set { s with defs := s.defs.push d }
  return (s.defs.size, rets)

/-- A primitive batched by a node, as a template Definition of its one gate on its operands. -/
def gateDef (P : Prog) (f : Nat) : M (Nat × Array Nat) := do
  if let some r := (← get).gates[f]? then return r
  let callee := P.defs[f]!
  let some g := gateOf? callee.id | throw s!"{callee.id}: not a Boolean gate (PROTOCOL.md §9)"
  let ins := callee.paramLeaves
  let (w, o) ← ({} : Work).op g ((Array.range ins).map Form.input)
  let (w, c) := w.commitOwn o
  let r ← emit (w.finish ins) #[c]
  modify fun s => { s with gates := s.gates.insert f r }
  return r

/-- A call of `count` copies of template Definition `e` of `cols` columns, binding its inputs `bs`: placed after every
column the bindings name. -/
def Work.place (w : Work) (e cols count : Nat) (bs : Array Bind) (rets : Array Nat) : Work :=
  { w with calls := w.calls.push { def_ := e, count, base := w.next, stride := cols, binds := bs },
           next := w.next + count * cols, outs := w.outs.push (.copies w.next cols rets) }

/-- Node `x`'s bindings of its callee's inputs, copy 0's and their steps: each argument's references in order (a mapped
argument's as many as its parameter's leaves), committing a form a copy reads that is not one reference. -/
def Work.binds (w : Work) (st : Array Nat) (callee : Defn) (x : Node) : V (Work × Array Bind) := do
  let mut w := w
  let mut out := #[]
  let n := if x.form == .batch then x.n else 1
  for i in [0:x.args.size] do
    let a := x.args[i]!
    let cnt := if x.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
    for t in [0:cnt] do
      let (r, s) ← if x.form == .batch then bindRef a n t else pure (a.seq.get t, 0)
      if s == 0 then
        let (w', at_) := w.commit (← w.formOf st r)
        w := w'
        out := out.push ⟨at_, 0⟩
      else out := out.push (← w.stepOf st r s)
  return (w, out)

/-- Reference `r` bound for every copy, stepping `s` leaves a copy: committed first when it is a form the same for every
copy (`s = 0`). -/
def Work.bindOne (w : Work) (st : Array Nat) (r : Ref) (s : Nat) : V (Work × Bind) := do
  if s == 0 then
    let (w', at_) := w.commit (← w.formOf st r)
    return (w', ⟨at_, 0⟩)
  else return (w, ← w.stepOf st r s)

/-- A scan node's `n` iterations of template Definition `e` (`cols` columns, its returns `rets`, the first `c` its carry):
copy 0 bound to `init`, copies `1 … n − 1` one `CALL` whose carry inputs are the previous copy's carry returns, stepping
one copy; an `xs` argument steps its element's leaves an iteration, a shared one binds every copy alike. -/
def Work.scan (w : Work) (st : Array Nat) (callee : Defn) (x : Node) (e cols : Nat) (rets : Array Nat) : V Work := do
  let n := x.n
  if n == 0 then throw s!"{callee.id}: a scan of no iterations"
  let c := (callee.params[0]!.2).leaves
  let mut w := w
  let mut first := #[]
  let mut rest := #[]
  let mut carries := #[]
  for i in [0:x.args.size] do
    let a := x.args[i]!
    let cnt := (callee.params[i]!.2).leaves
    for t in [0:cnt] do
      if i == 0 then
        let (w', b) ← w.bindOne st (a.seq.get t) 0
        w := w'
        first := first.push b
        carries := carries.push t
        rest := rest.push b
      else if i ≤ x.nxs then
        let (r, s) ← stepping a.seq t n cnt "a scan's xs"
        let (w', b) ← w.bindOne st r s
        w := w'
        first := first.push b
        rest := rest.push ⟨b.a + b.step, b.step⟩
      else
        let (w', b) ← w.bindOne st (a.seq.get t) 0
        w := w'
        first := first.push b
        rest := rest.push b
  let base := w.next
  rest := (Array.range rest.size).map fun k =>
    if k < c then ⟨2 * (base + rets[carries[k]!]!), 2 * cols⟩ else rest[k]!
  let calls := w.calls.push { def_ := e, count := 1, base, stride := cols, binds := first }
  let calls := if n > 1 then calls.push { def_ := e, count := n - 1, base := base + cols, stride := cols, binds := rest }
    else calls
  return { w with calls, next := base + n * cols, outs := w.outs.push (.scan base cols rets c n) }

/-- Composite Definition `d` lowered, its callees first: its template Definition and its returns' columns. `fuel` bounds the
call depth; a callee has a smaller index than its caller. -/
def lowerDef (P : Prog) : Nat → Nat → M (Nat × Array Nat)
  | 0, _ => throw "the call hierarchy is deeper than the program's Definitions"
  | fuel + 1, d => do
    if let some r := (← get).done[d]? then return r
    let defn := P.defs[d]!
    if defn.prim then throw s!"{defn.id}: a primitive is not lowered as a Definition"
    let body := defn.body
    let st := starts defn.params
    let mut w : Work := {}
    for x in body.nodes do
      let callee := P.defs[x.fn]!
      if x.form == .scan then
        if callee.prim then throw s!"{defn.id}: a scan of a primitive"
        let (e, rets) ← lowerDef P fuel x.fn
        let cols := ((← get).defs[e]!).cols
        w ← liftM (w.scan st callee x e cols rets)
        continue
      if callee.prim then
        let some g := gateOf? callee.id | throw s!"{callee.id}: not a Boolean gate (PROTOCOL.md §9)"
        if x.form == .batch then
          let (e, rets) ← gateDef P x.fn
          let cols := ((← get).defs[e]!).cols
          let (w', bs) ← liftM (w.binds st callee x)
          w := w'.place e cols x.n bs rets
        else
          let mut ops := #[]
          for a in x.args do
            for t in [0:a.seq.count] do ops := ops.push (← liftM (w.formOf st (a.seq.get t)))
          let (w', f) ← liftM (w.op g ops)
          w := { w' with outs := w'.outs.push (.gate f) }
      else
        let (e, rets) ← lowerDef P fuel x.fn
        let cols := ((← get).defs[e]!).cols
        let (w', bs) ← liftM (w.binds st callee x)
        w := w'.place e cols (if x.form == .batch then x.n else 1) bs rets
    let mut rets := #[]
    for t in [0:body.ret.count] do
      let (w', c) := w.commitOwn (← liftM (w.formOf st (body.ret.get t)))
      w := w'
      rets := rets.push c
    let r ← emit (w.finish defn.paramLeaves) rets
    modify fun s => { s with done := s.done.insert d r }
    return r
where
  liftM {α : Type} (x : V α) : M α := StateT.lift x

/-- The output rows' first column: past `n` columns, aligned to `span`. -/
def alignUp (n span : Nat) : Nat := (n + span - 1) / span * span

/-- **The block's template**: Definition `d` of `P` and every Definition it reaches, one template Definition each, callees
first, then the root: the input rows (port bit `i` at column `i`, every port bit a row), `d` placed once on them, its returns copied to the
output rows, and the constant. `inPorts`/`outPorts` are the port widths in bits, ports concatenated, low bit first. -/
def build (P : Prog) (d : Nat) (inPorts outPorts : Array Nat) : V Tmpl.Tmpl := do
  let some defn := P.defs[d]? | throw s!"no Definition {d}"
  let N := defn.paramLeaves
  let Mo := defn.outLeaves
  let inBits := inPorts.foldl (· + ·) 0
  let outBits := outPorts.foldl (· + ·) 0
  if inBits < N then throw s!"{defn.id}: {N} input bits, ports of {inBits}"
  if outBits < Mo then throw s!"{defn.id}: {Mo} output bits, ports of {outBits}"
  let ((e, rets), s) ← (lowerDef P (P.defs.size + 1) d).run {}
  let cols := (s.defs[e]!).cols
  let inW := Nat.nextPowerOfTwo ((inBits + WORD - 1) / WORD)
  let base := inW * WORD
  let outW := Nat.nextPowerOfTwo ((outBits + WORD - 1) / WORD)
  let span := outW * WORD
  let oStart := alignUp (base + cols) span
  let const := oStart + span
  let one := const + 1
  let root : Tmpl.Def :=
    { cols := const + 1, ins := 0,
      rows := ((List.range inBits).map (fun c => (c, #[c], #[c]))).toArray ++
        ((List.range Mo).map fun k => (oStart + k, #[base + rets[k]!], #[one])).toArray,
      calls := #[{ def_ := e, count := 1, base, stride := cols,
                   segs := if N == 0 then #[] else #[{ len := N, at_ := 0, step := 0, reps := 1, rstride := 0 }] }] }
  return { name := defn.key, root := s.defs.size, inGroups := #[⟨0, inW, inPorts⟩],
           outGroups := #[⟨oStart / WORD, outW, outPorts⟩], defs := s.defs.push root }

/-- The prefix of a refusal at `selfCheck`: the lowering wrote a template the reader refuses, a completeness failure of
`lower`, never a refusal of its input. -/
def SELF_CHECK : String := "lower: completeness failure"

/-- The first of the reader's five checks `t` fails. -/
def selfWhy (t : Tmpl.Tmpl) : String :=
  if !t.wellFormed then "not well-formed"
  else if !t.apart then "a Definition's rows or copies overlap or leave its columns"
  else if !t.ordered then "a row reads a column out of order, or an input column that is not a port bit"
  else if !t.inputsT then "an input row is not a self row or empty, or an input port bit has no self row"
  else "a port group is not an aligned power of two of words holding its ports"

/-- `t` if it passes the checks the reader runs on every template (`Tmpl.unrollOk`'s, the layouts' aside), else refused
under `SELF_CHECK`. -/
def selfCheck (t : Tmpl.Tmpl) : V Tmpl.Tmpl :=
  if t.wellFormed && t.apart && t.ordered && t.inputsT && t.groupsOk then pure t
  else throw s!"{SELF_CHECK}: the template of {t.name} written by `lower` fails the reader's check: {selfWhy t}"

/-- **The block's template** (`build`), refused under `SELF_CHECK` unless it passes the reader's checks. -/
def lower (P : Prog) (d : Nat) (inPorts outPorts : Array Nat) : V Tmpl.Tmpl :=
  build P d inPorts outPorts >>= selfCheck

/-- **What `lower` writes is a template the verifier's checks pass**: well formed, its copies apart, its rows and
bindings in order, its input rows self rows over its port bits, its port groups aligned. -/
theorem lower_checked {P : Prog} {d : Nat} {inP outP : Array Nat} {t : Tmpl.Tmpl}
    (hl : lower P d inP outP = .ok t) :
    t.wellFormed = true ∧ t.apart = true ∧ t.ordered = true ∧ t.inputsT = true ∧ t.groupsOk = true := by
  unfold lower at hl
  cases hb : build P d inP outP with
  | error e => simp [hb, bind, Except.bind] at hl
  | ok t' =>
    simp only [hb, bind, Except.bind, selfCheck] at hl
    split at hl
    · rename_i h
      cases hl
      simpa [Bool.and_eq_true, and_assoc] using h
    · cases hl

/-! ## The text -/

/-- A port group list in a header: the count, then each group's column, words, port count and widths. -/
def groupsText (gs : Array Group) : List String :=
  toString gs.size :: gs.toList.flatMap fun g =>
    [toString g.col, toString g.words, toString g.bits.size] ++ g.bits.toList.map toString

/-- **The template as `flock-ir-tmpl/v2` text**, a line an item, the carried `LEAVES` line last (`Tmpl.parse` reads it). -/
def render (t : Tmpl.Tmpl) : String := Id.run do
  let nums := fun (l : List Nat) => " ".intercalate (l.map toString)
  let mut out := " ".intercalate ([Tmpl.FORMAT, t.name, toString t.root] ++ groupsText t.inGroups ++
    groupsText t.outGroups) ++ "\n"
  for df in t.defs do
    out := out ++ s!"DEF {df.cols} {df.ins} {df.rows.size} {df.calls.size}\n"
    for r in df.rows do
      out := out ++ nums ([r.1, r.2.1.size] ++ r.2.1.toList ++ [r.2.2.size] ++ r.2.2.toList) ++ "\n"
    for c in df.calls do
      out := out ++ "CALL " ++ nums ([c.def_, c.count, c.base, c.stride, c.segs.size] ++
        c.segs.toList.flatMap fun s => [s.len, s.at_, s.step, s.reps, s.rstride]) ++ "\n"
  if let some l := t.leaves then out := out ++ l ++ "\n"
  return out

/-- The lines `render` writes, as `Tmpl.parse` takes them. -/
def renderLines (t : Tmpl.Tmpl) : Array String := ((render t).splitOn "\n").toArray.pop

/-! ## The ports -/

/-- Input port bit `i`'s column: the input rows hold bit `i` at column `i`. -/
def inBit (_t : Tmpl.Tmpl) (i : Nat) : Nat := i

/-- Output port bit `k`'s column: the output group's first column, plus `k`. -/
def outBit (t : Tmpl.Tmpl) (k : Nat) : Nat := ((t.outGroups[0]?.map (·.col)).getD 0) * WORD + k

end Flock.Lower
