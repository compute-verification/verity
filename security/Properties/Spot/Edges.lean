module

public import Mathlib.Data.Finset.Card
public import Spot.Edges

/-!
# The wires `Spot.Edges.edgesOf` returns, as the Program's references between root nodes

* `RootRef P k i x w l`: root node `k`'s argument `i` reads, at its index `x`, leaf `l` of root node `w`, neither an
  input node.
* `memberIndex a m t`: argument `a`'s index of member `m`'s leaf `t` (the circuit format's members of a batch argument).
* `Edges.fieldOffset t f`: the first leaf of field `f` of the return `t`, its first word in the unit's output row `out`
  (the code's, which `edgesOf` writes a wire's `offset` by).
* `WireReads P rs W e k i x w l`: wire `e` covers that read: a reader unit of `e` is member `mk` of node `k`, its row
  `cut0` of port `i` holds index `x` at leaf `t`, and the writer unit is member `mw` of node `w`, whose output `out`,
  from the first leaf of field `f`, word `e.offset`, holds leaf `l` at `t`, the row being field `f` whole.
* `EdgesExact P rs W ws`: the wires `ws` are exactly the references between root nodes: each is covered by a wire of
  `ws`, each wire covers only references, and no two wires cover one.
* `EdgesOfExact`: the wires `edgesOf` accepts are exact, an open statement.
-/

@[expose] public section

namespace Properties.Spot.Edges

open _root_.Spot
open Flock.Program (Prog Arg Ty fieldsOf)
open Flock.TemplateQuery (TRange)

/-- **A reference between root nodes**: root node `k`'s argument `i` reads, at index `x`, leaf `l` of root node `w`,
neither an input node. -/
def RootRef (P : Prog) (k i x w l : ℕ) : Prop :=
  let b := (P.defs[P.root]!).body
  k < b.nodes.size ∧ Edges.isInputNode P k = false ∧ i < (b.nodes[k]!).args.size ∧
    x < ((b.nodes[k]!).args[i]!).seq.count ∧ ((b.nodes[k]!).args[i]!).seq.get x = ⟨false, w, l⟩ ∧
    w < b.nodes.size ∧ Edges.isInputNode P w = false

/-- Argument `a`'s index of member `m`'s leaf `t`: `((t / L)·d + m)·L + t % L` for an argument mapped over an axis of
dimensions `(d, L)`, else `t`. -/
def memberIndex (a : Arg) (m t : ℕ) : ℕ :=
  match a.dims with
  | some (d, L) => ((t / L) * d + m) * L + t % L
  | none => t

/-- **Wire `e` covers a read**: reader unit `e.start + m` is member `mk` of root node `k`, whose row `e.cut0` of port
`i` (of `len` leaves) holds argument index `x` at its leaf `t`; writer unit `e.base + m·step` is member `mw` of root
node `w`, whose output `out`, from the first leaf of field `f`, word `e.offset`, holds leaf `l` at `t`, the row of `len`
leaves being field `f` whole. -/
def WireReads (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (k i x w l : ℕ) : Prop :=
  let b := (P.defs[P.root]!).body
  let node := b.nodes[k]!
  let ty := ((P.defs[node.fn]!).params[i]!).2
  let len := ty.leaves / W.cutsOf e.reader e.port
  let wd := P.defs[(b.nodes[w]!).fn]!
  e.radix.length = 1 ∧ e.step.length = 1 ∧ e.cuts = 1 ∧ ((P.defs[node.fn]!).params[i]!).1 = e.port ∧ e.out = "out" ∧
    ∃ rb wb m mk mw t f, (Edges.firstUnits rs)[k]? = some (e.reader, rb) ∧
      (Edges.firstUnits rs)[w]? = some (e.writer, wb) ∧ m < e.radix.headD 0 ∧ e.start + m = rb + mk ∧
      mk < node.members ∧ t < len ∧ x = memberIndex (node.args[i]!) mk (e.cut0 * len + t) ∧
      f < (fieldsOf wd.ret).length ∧ e.offset = Edges.fieldOffset wd.ret f ∧ len = ((fieldsOf wd.ret)[f]!).leaves ∧
      ((wb + mw : ℕ) : ℤ) = e.base + (m : ℤ) * e.step.headD 0 ∧ mw < (b.nodes[w]!).members ∧
      l = mw * wd.outLeaves + Edges.fieldOffset wd.ret f + t

/-- **The wires are exactly the references between root nodes**: each reference is covered by a wire of `ws`, each
wire covers only references, and no two wires of `ws` cover one. -/
def EdgesExact (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (ws : List Wires.Wire) : Prop :=
  (∀ k i x w l, RootRef P k i x w l → ∃ e ∈ ws, WireReads P rs W e k i x w l) ∧
    (∀ e ∈ ws, ∀ k i x w l, WireReads P rs W e k i x w l → RootRef P k i x w l) ∧
    ws.Pairwise fun e e' => ∀ k i x w l, ¬ (WireReads P rs W e k i x w l ∧ WireReads P rs W e' k i x w l)

/-- **The wires `edgesOf` accepts are exact**: for every Program, ranges and wiring, wires `edgesOf` returns are
exactly the references between root nodes. -/
def EdgesOfExact : Prop :=
  ∀ (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (ws : List Wires.Wire),
    Edges.edgesOf P rs W = .ok ws → EdgesExact P rs W ws

end Properties.Spot.Edges
