module

public import Proofs.Properties.Nci.Scratch.RecordAttack.Bridge
public import Proofs.Properties.Remote

/-!
# The bridge, as far as the remote theorem's policy

A scratch file, never merged. `policyMet_of_declaresBy`: where a check that `Complies` accepts a record that lays
out a declaration's circuit under the declaration's own blocks, with every row the check holds among the
declaration's registered inputs (`DeclaresBy`), the declaration meets the inference-only policy as the remote
theorem states it (`Properties.Remote.PolicyMet`), at the instance's `Γ` and `α`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)

/-- **The bridge.** -/
theorem policyMet_of_declaresBy {F : Form} (hc : Complies F) {j tr : Json} {c : F.Case} (hacc : Accepts F j tr c)
    {d₀ : Verity.RegisteredCircuit} (hd : DeclaresBy F c d₀ d₀.𝓘) :
    _root_.Properties.Remote.PolicyMet d₀ d₀.𝓘 (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) :=
  _root_.Properties.Remote.SecurityProofs.policyMet_of_meets fun u => meets_of_declaresBy hc hacc hd u

end NciCheck.Properties.Record.Attack
