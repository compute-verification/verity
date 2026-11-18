module

public import Mathlib.Data.Finset.Card
public import Spot.Edges

/-!
# The wires `Spot.Edges.edgesOf` returns, as the Program's references between root nodes

* `RootRef P k i x w l`: root node `k`'s argument `i` reads, at its index `x`, leaf `l` of root node `w`, neither an
  input node.
* `memberIndex a m t`: argument `a`'s index of member `m`'s leaf `t` (the circuit format's members of a batch argument).
* `fieldOffset t f`: the first leaf of field `f` of the return `t`.
* `WireReads P rs W e k i x w l`: wire `e` covers that read: a reader unit of `e` is member `mk` of node `k`, its row
  `cut0` of port `i` holds index `x` at leaf `t`, and the writer unit is member `mw` of node `w`, whose field `out`
  holds leaf `l` at `t`.
* `EdgesExact P rs W ws`: the wires `ws` are exactly the references between root nodes: each is covered by a wire of
  `ws`, each wire covers only references, and no two wires cover one.
* `PartsWithin P`: each argument of each root node has at most one part more than the spaces its parts read.

`Proofs.Properties.Spot.Edges` proves that a root `edgesOf` accepts is within `PartsWithin`. That `edgesOf P rs W =
.ok ws` gives `EdgesExact P rs W ws` is the open statement.
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

/-- The first leaf of field `f` of the return `t`. -/
def fieldOffset (t : Ty) (f : ℕ) : ℕ := (((fieldsOf t).take f).map Ty.leaves).sum

/-- **Wire `e` covers a read**: reader unit `e.start + m` is member `mk` of root node `k`, whose row `e.cut0` of port
`i` (of `len` leaves) holds argument index `x` at its leaf `t`; writer unit `e.base + m·step` is member `mw` of root
node `w`, whose field `f` (`e.out`) holds leaf `l` at `t`. -/
def WireReads (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (k i x w l : ℕ) : Prop :=
  let b := (P.defs[P.root]!).body
  let node := b.nodes[k]!
  let ty := ((P.defs[node.fn]!).params[i]!).2
  let len := ty.leaves / W.cutsOf e.reader e.port
  let wd := P.defs[(b.nodes[w]!).fn]!
  e.radix.length = 1 ∧ e.step.length = 1 ∧ e.cuts = 1 ∧ ((P.defs[node.fn]!).params[i]!).1 = e.port ∧
    ∃ rb wb m mk mw t f, (Edges.firstUnits rs)[k]? = some (e.reader, rb) ∧
      (Edges.firstUnits rs)[w]? = some (e.writer, wb) ∧ m < e.radix.headD 0 ∧ e.start + m = rb + mk ∧
      mk < node.members ∧ t < len ∧ x = memberIndex (node.args[i]!) mk (e.cut0 * len + t) ∧
      e.out = "o" ++ toString f ∧ f < (fieldsOf wd.ret).length ∧
      ((wb + mw : ℕ) : ℤ) = e.base + (m : ℤ) * e.step.headD 0 ∧ mw < (b.nodes[w]!).members ∧
      l = mw * wd.outLeaves + fieldOffset wd.ret f + t

/-- **The wires are exactly the references between root nodes**: each reference is covered by a wire of `ws`, each
wire covers only references, and no two wires of `ws` cover one. -/
def EdgesExact (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (ws : List Wires.Wire) : Prop :=
  (∀ k i x w l, RootRef P k i x w l → ∃ e ∈ ws, WireReads P rs W e k i x w l) ∧
    (∀ e ∈ ws, ∀ k i x w l, WireReads P rs W e k i x w l → RootRef P k i x w l) ∧
    ws.Pairwise fun e e' => ∀ k i x w l, ¬ (WireReads P rs W e k i x w l ∧ WireReads P rs W e' k i x w l)

/-- **Each root argument within the parts bound**: at most one part more than the spaces (`Edges.partKey`) its parts
read. -/
def PartsWithin (P : Prog) : Prop :=
  let b := (P.defs[P.root]!).body
  ∀ k < b.nodes.size, ∀ (j : ℕ) (hj : j < (b.nodes[k]!).args.size),
    ((b.nodes[k]!).args[j]).seq.parts.size ≤
      (((b.nodes[k]!).args[j]).seq.parts.toList.map Edges.partKey).toFinset.card + 1

end Properties.Spot.Edges
