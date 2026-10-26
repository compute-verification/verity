import Properties.NetworkScan
import NciCheck.Desc

/-!
# The walk lemma, and the scan's causality: no egress gate depends on a later bucket's ingress

Part (3) of `NetTiming.Layout.Fits` at `scanLayout`: no egress gate of step `t` depends on an ingress gate of a later
step `t'`. It follows from where C-Flock's walk (`Flock.Program.Prog.gate`, `Flock.Program.resolve`) takes a gate's
operands.

**The walk lemma** (`MemberReads`): in a Program C-Flock decodes, topological (`Flock.Program.Prog.orderViolation`) and
laid out (`NciCheck.Desc.bodiesLaidOut`), a gate in the activation of member `m` of a root node `k` calling a composite
(by a batch or a call: a scan's iteration reads the one before) reads only gates of that activation, or a gate one of
node `k`'s argument references resolves to from the root, what its arguments bind.

**The wiring** (`scanWired`, a check for `scanOf` to make): every reference of a template node's argument names a root
input node, and no input node is named by two template arguments, as the generator form wires them (each template
parameter its own batch of `Input1`). An egress member then reads only its activation's gates and its arguments' input
gates (its `state`), and a later bucket's ingress gates are its `row` argument's input gates, which no egress argument
names. `ScanCausal` states it at an accepted scan whose Program `Prog.orderViolation` passes and `NciCheck.Desc.walkable`
takes, so that the egress gates `scanLayout` reads resolve.
-/

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- What a part of a reference sequence names, as `Flock.Program.Prog.orderViolation` reads it: each reference's space
(`true`, a parameter of the body; `false`, a node) and index. -/
def partNames : Flock.Program.Part → Array (Bool × ℕ)
  | .affine p i _ c => if c > 0 then #[(p, i)] else #[]
  | .strided p i _ _ _ _ c => if c > 0 then #[(p, i)] else #[]
  | .explicit it => it.map fun r => (r.p, r.i)

/-- The names the template nodes' arguments give, each with its template node `k` and argument `i`: `((k, i), (p, x))`. -/
def templateArgNames (P : Flock.Program.Prog) : List ((ℕ × ℕ) × (Bool × ℕ)) :=
  let nodes := (P.defs[P.root]!).body.nodes
  (templateNodes P).flatMap fun k => (List.range nodes[k]!.args.size).flatMap fun i =>
    (nodes[k]!.args[i]!.seq.parts.toList.flatMap fun q => (partNames q).toList).map fun r => ((k, i), r)

/-- **Each template argument reads root inputs of its own**, the check `scanOf` is to make: every reference of a
template node's argument names a root node that is an input node, and no input node is named by two template
arguments. -/
def scanWired (P : Flock.Program.Prog) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ns := templateArgNames P
  (ns.all fun x => !x.2.1 && decide (x.2.2 < nodes.size) &&
      ((P.defs[nodes[x.2.2]!.fn]!).prim && Flock.Program.isInputId (P.defs[nodes[x.2.2]!.fn]!).key)) &&
    ns.all fun x => ns.all fun y => x.2.2 != y.2.2 || x.1 == y.1

/-- **The walk lemma.** In a Program C-Flock decodes, topological and laid out, a gate `g` in the activation of member
`m` of a root node `k` that calls a composite by a batch or a call reads (its operands, `Flock.Program.Prog.gate`) only
gates of that activation, or a gate a reference of one of node `k`'s arguments resolves to from the root. -/
def MemberReads : Prop :=
  ∀ (bytes : ByteArray) (P : Flock.Program.Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    NciCheck.Desc.bodiesLaidOut P = true →
    ∀ k < (P.defs[P.root]!).body.nodes.size, ∀ m < ((P.defs[P.root]!).body.nodes[k]!).members,
      (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false →
      ((P.defs[P.root]!).body.nodes[k]!).form ≠ .scan →
      ∀ g p ops sc, P.gate g = .ok (p, ops, sc) → (P.child P.rootScope k m).off ≤ g →
        g < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates →
        ∀ a ∈ ops,
          ((P.child P.rootScope k m).off ≤ a ∧
            a < (P.child P.rootScope k m).off + (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).gates) ∨
          ∃ i t, t < ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.count) ∧
            Flock.Program.resolve P P.rootScope ((((P.defs[P.root]!).body.nodes[k]!).args[i]!).seq.get t) = .ok a

/-- **The scan is causal** (`NetTiming.Layout.Fits`'s part (3)). If `scanOf` accepts the scan `j`, its canonical text
`c` decodes to `P`, `P` is topological, `NciCheck.Desc.walkable` takes it and its templates read root inputs of their
own (`scanWired`), then in every flattening `C` of `P` no egress gate of a step depends on an ingress gate of a later
step of the window. -/
def ScanCausal : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P →
    P.orderViolation = none → NciCheck.Desc.walkable P = true → scanWired P = true → Flattens P C →
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s).ingress t', ¬ NetTiming.Depends C i g

end Properties.NetworkScan
