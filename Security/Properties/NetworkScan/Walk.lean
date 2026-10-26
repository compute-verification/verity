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

**The scan** (`ScanCausal`): `scanOf` takes each template's argument to be one input node's leaves
(`NetworkCheck.Period.inputArg`) and no egress argument to be an ingress row's input node, so an egress member reads
only its activation's gates and its arguments' input gates, and a later bucket's ingress gates are its row's input
node's, which no egress argument names. The walk lemma reads the Program topological, which `scanOf` doesn't check:
`ScanCausal` takes `P.orderViolation = none`, the check `NciCheck.Desc.evaluate` makes (`prog.orderViolation.isNone`).
-/

namespace Properties.NetworkScan

open NetworkCheck.Period

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
`c` decodes to `P` and `P` is topological, then in every flattening `C` of `P` no egress gate of a step depends on an
ingress gate of a later step of the window. -/
def ScanCausal : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P →
    P.orderViolation = none → Flattens P C →
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s).ingress t', ¬ NetTiming.Depends C i g

end Properties.NetworkScan
