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
* `WireReadsAt P rs W e u row t k i x w l`, `WireReads`: wire `e` covers that read: reader unit `u` is member `mk` of
  node `k`, its row `row` of port `i` holds index `x` at leaf `t`, the wire's own lookup (`Circuits.Wire.leafAt`) gives
  the row the writer unit of member `mw` of node `w`, and its output `out` holds leaf `l` at `t` from `e.offset`, the
  first leaf of field `f`, the row being field `f` whole.
* `EdgesExact P rs W ws`: the wires `ws` are exactly the references between root nodes: each is covered by a wire of
  `ws`, every row a wire looks up is a port's row read whole, each read is a reference, and no two wires cover one.
* `EdgesOfExact`: the wires `edgesOf` accepts of a decoded Program, under its templates' ranges, are exact, an open
  statement.
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

/-- The leaves of one row of root node `k`'s port `i`, its template `tid` cutting the port into rows by the wiring. -/
def rowLen (P : Prog) (W : Wires.Wiring) (tid : String) (k i : ℕ) : ℕ :=
  let p := (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).params[i]!
  p.2.leaves / W.cutsOf tid p.1

/-- **Wire `e` reads a reference at reader unit `u`'s row `row`, leaf `t`**: `u` is member `mk` of root node `k`, whose
row `row` of port `i` holds argument index `x` at leaf `t`; the wire's own lookup (`Circuits.Wire.leafAt`, by its
index, reader steps and cuts) gives that row writer unit `wb + mw`, member `mw` of root node `w`, whose output `out`
holds leaf `l` at `t` from the wire's word `e.offset`, the first leaf of field `f`, the row being field `f` whole. -/
def WireReadsAt (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (u row t k i x w l : ℕ) : Prop :=
  let b := (P.defs[P.root]!).body
  let node := b.nodes[k]!
  let len := rowLen P W e.reader k i
  let wd := P.defs[(b.nodes[w]!).fn]!
  ((P.defs[node.fn]!).params[i]!).1 = e.port ∧ e.out = "out" ∧
    ∃ rb wb mk mw f, (Edges.firstUnits rs)[k]? = some (e.reader, rb) ∧
      (Edges.firstUnits rs)[w]? = some (e.writer, wb) ∧ u = rb + mk ∧ mk < node.members ∧
      row < W.cutsOf e.reader e.port ∧ t < len ∧ e.leafAt u row = some (wb + mw) ∧ mw < (b.nodes[w]!).members ∧
      x = memberIndex (node.args[i]!) mk (row * len + t) ∧ f < (fieldsOf wd.ret).length ∧
      e.offset = Edges.fieldOffset wd.ret f ∧ len = ((fieldsOf wd.ret)[f]!).leaves ∧ l = mw * wd.outLeaves + e.offset + t

/-- **Wire `e` covers a read**: at some reader unit, row and leaf (`WireReadsAt`). -/
def WireReads (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (k i x w l : ℕ) : Prop :=
  ∃ u row t, WireReadsAt P rs W e u row t k i x w l

/-- **The wires are exactly the references between root nodes**: each reference is covered by a wire of `ws`; every
row a wire's lookup gives a writer is a row of a root node's port, read whole; each read is a reference; and no two
wires of `ws` cover one. -/
def EdgesExact (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (ws : List Wires.Wire) : Prop :=
  (∀ k i x w l, RootRef P k i x w l → ∃ e ∈ ws, WireReads P rs W e k i x w l) ∧
    (∀ e ∈ ws, ∀ u row v, e.leafAt u row = some v → ∃ k i, 0 < rowLen P W e.reader k i ∧
      ∀ t < rowLen P W e.reader k i, ∃ x w l, WireReadsAt P rs W e u row t k i x w l) ∧
    (∀ e ∈ ws, ∀ u row t k i x w l, WireReadsAt P rs W e u row t k i x w l → RootRef P k i x w l) ∧
    ws.Pairwise fun e e' => ∀ k i x w l, ¬ (WireReads P rs W e k i x w l ∧ WireReads P rs W e' k i x w l)

/-- **The wires `edgesOf` accepts are exact**: for every Program the circuit format decodes, the ranges of its
templates and any wiring, the wires `edgesOf` returns are exactly the references between root nodes. -/
def EdgesOfExact : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (ts : Array String) (rs : Array TRange) (W : Wires.Wiring) (ws : List Wires.Wire),
    Flock.Program.decode bytes = .ok P → Flock.TemplateQuery.evaluate (.ofProg P) ts = .ok rs →
      Edges.edgesOf P rs W = .ok ws → EdgesExact P rs W ws

end Properties.Spot.Edges
