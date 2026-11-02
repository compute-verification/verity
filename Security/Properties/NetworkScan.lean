import NetworkCheck.Period
import Definitions.NetworkCertifier.SampledAudit

/-!
# The rows form's scan fits its layout: what `NetworkCheck.Period.scanOf`'s acceptance gives the sampled audit's scope

The sampled audit's scope at a link-window's statement (`NetTiming.ScopeAcc`) has two halves: the sampling scope
(`Claims.Sampling.Scope`, the sampled verifier's own, which `Claims.Sampling.ScopeMatches` equates with its scope
check), and the statement's layout fitting its program over its horizon (`NetTiming.BoundaryGates.Fits`). The network
check reads the second off the scan's Program (`scanOf`, the generator form); the first it doesn't read.

**The layout** (`scanLayout`) is the generator form's, at the schedule's `S = lanes · slots` slots of `B` bytes and
`T` buckets, over one egress link: at step `t`, slot `σ`'s presence gate is bit 0 of egress member `t`'s verdict (its
return's leaf 0), and its frame's bits are the return's leaves `16 + 8σB … 16 + 8(σ + 1)B − 1` (the row after the u16,
row bit `8k + i` bit `i` of byte `k`); step `t`'s ingress is bucket `t`'s `row` parameter, in bit order: bucket 0's
first-template member, bucket `t ≥ 1`'s later-template member `t − 1`, the templates being the root's nodes that call a
composite (`NetworkCheck.Period.templateNodes`, as `scanOf` reads them). Gates are `Flock.Program.resolve`'s (§6), the
canonical layout's. A statement's layout is its descriptor's (`layOf`), at horizon `T`.

**The flattening** (`Flattens`): the sampling statement's gates (`Claims.Sampling.AuditStatement.prog`, "the descriptor
flattened") are its descriptor's canonical layout, gate for gate: as many, an input gate exactly where its primitive is
an input node, reading exactly the gates its operands name, returning the gates its root's return resolves to.
`StmtFlattens` says it of every statement a deployment parses; the deployed parser is the sampled verifier's
(`SamplingDeploy.Code.parse`), so it is that code's premise.

* `ScanFits`: an accepted scan's layout fits every flattening of its Program over `T` steps.
* `ScanScopeAcc`: at a deployment whose statements are their descriptors flattened and whose scope check is its scope,
  a statement the scope check takes, whose descriptor is the accepted scan's canonical bytes, is in the sampled audit's
  scope at `layOf` and `T` (`ScopeAcc`). `Proofs.Properties.NetworkScan` derives it from `ScanFits`.
-/

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- The gate the reference `r` resolves to from the activation `sc` (`Flock.Program.resolve`), `0` where it doesn't
resolve. -/
def gateAt (P : Flock.Program.Prog) (sc : Flock.Program.Scope) (r : Flock.Program.Ref) : ℕ :=
  (Flock.Program.resolve P sc r).toOption.getD 0

/-- **The scan's layout** in its Program `P` at the schedule `s`, over one egress link: step `t`'s slot `σ` is bit 0 of
egress member `t`'s verdict and its frame's `8B` bits, and step `t`'s ingress is bucket `t`'s `row` parameter. -/
def scanLayout (P : Flock.Program.Prog) (s : Schedule) : NetTiming.BoundaryGates 1 s.allocation :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ts := NetworkCheck.Period.templateNodes P
  let ke := ts.getLast?.getD 0
  let le := (P.defs[nodes[ke]!.fn]!).outLeaves
  let bits := 8 * s.cellBytes
  { egress := fun t _ σ =>
      let leaf := fun j => gateAt P P.rootScope ⟨false, ke, t * le + j⟩
      (leaf 0, (List.range bits).map fun b => leaf (outBits + σ.val * bits + b))
    ingress := fun t =>
      let km := if t = 0 then (ts.headD 0, 0) else (ts.getD 1 0, t - 1)
      (List.range (s.allocation * bits)).map fun j => gateAt P (P.child P.rootScope km.1 km.2) ⟨true, 0, j⟩ }

/-- A statement's layout at the schedule `s`: its descriptor's scan layout, none where the descriptor doesn't decode. -/
def layOf (s : Schedule) (x : Claims.Sampling.AuditStatement) : NetTiming.BoundaryGates 1 s.allocation :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => scanLayout P s
  | .error _ => { egress := fun _ _ _ => (0, []), ingress := fun _ => [] }

/-- **`C` is the Program `P` flattened**, gate for gate in its canonical layout (`Flock.Program.Prog.gate`): `P`'s
gates, gate `g` an input gate exactly where its primitive is an input node and reading exactly the gates its operands
name, and `C` returning the gates the root's return resolves to. -/
def Flattens (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) : Prop :=
  C.gates.length = (P.defs[P.root]!).gates ∧
    (∀ g < C.gates.length, ∃ (p : ℕ) (ops : Array ℕ) (sc : Flock.Program.Scope), P.gate g = .ok (p, ops, sc) ∧
      (C.gates[g]? = some .input ↔ Flock.Program.isInputId (P.defs[p]!).key = true) ∧
      ∀ a, a ∈ (C.gates[g]?.map Claims.Sampling.BGate.reads).getD [] ↔ a ∈ ops) ∧
    ∀ g, g ∈ C.outs ↔ ∃ j < (P.defs[P.root]!).body.ret.count,
      Flock.Program.resolve P P.rootScope ((P.defs[P.root]!).body.ret.get j) = .ok g

/-- **The deployment's statements are their descriptors flattened**: every statement `D` parses has a descriptor
C-Flock decodes (`Flock.Program.decode`, as the sampled verifier's `SampledProofs.Program.parse` does), and its gates
are that Program's (`Flattens`). -/
def StmtFlattens {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ Flattens P s.prog

/-- **An accepted scan's layout fits its Program flattened.** If `scanOf` accepts the scan `j` as the registered
Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, then `P`'s scan layout fits every
flattening of `P` over the window's `T` steps: every egress gate is returned and not an input, every ingress gate is an
input, and no egress gate of a step depends on an ingress gate of a later step. -/
def ScanFits : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      (scanLayout P s).Fits C s.window

/-- **An accepted scan's statements are in the sampled audit's scope.** At a deployment `D` whose statements are their
descriptors flattened (`StmtFlattens`) and whose scope check is its sampling scope (`ScopeMatches`), a statement `x`
the scope check takes, every parse of which has the accepted scan's canonical bytes as its descriptor, is in
`ScopeAcc` at the layout `layOf s` and the horizon `T`. -/
def ScanScopeAcc : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (D : Claims.Sampling.Deploy Coin Col T) (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String)
    (x : Claims.Msg),
    StmtFlattens D → Claims.Sampling.ScopeMatches D → D.scopeCheck x = true →
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8) →
      NetTiming.ScopeAcc D (layOf s) (fun _ => s.window) x

end Properties.NetworkScan
