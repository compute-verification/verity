import NetworkCheck.Period
import Definitions.NetworkCertifier.SampledAudit

/-!
# A sampling statement's gates are its descriptor's Program flattened

`Flattens P C`: the sampling statement's gates (`Claims.Sampling.AuditStatement.prog`, "the descriptor flattened") are
the Program `P`'s canonical layout, gate for gate: as many, an input gate exactly where its primitive is an input node,
reading exactly the gates its operands name, returning the gates its root's return resolves to. `Properties.NetworkScan`
states the scan's fit over every flattening, and `Properties.NetworkScan.Assumptions` the deployment's premise that each
statement it parses is its descriptor flattened.
-/

namespace Properties.NetworkScan

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

end Properties.NetworkScan
