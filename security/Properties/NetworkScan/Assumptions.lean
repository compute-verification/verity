module

public import Properties.NetworkScan.Flattens

/-!
# The compiled sampled verifier's premise under an accepted scan's scope

`Properties.NetworkScan.ScanScopeAcc` carries the scan's fit into the sampled audit's scope at a compiled sampled
verifier meeting one premise, a named `Prop` here: its statements are their descriptors flattened (`StmtFlattens`).
It is a premise on the deployed sampled verifier's parse (`SamplingDeploy.Code.parse`), which the network check
doesn't read, so the scan's guarantee takes it as it stands; at the deployed parsers it fails wherever the
descriptor's Program has an input gate at gate 0 (`StmtFlattens`).
-/

@[expose] public section

namespace Properties.NetworkScan

/-- **The compiled verifier's statements are their descriptors flattened**: every statement `D` parses has a
descriptor Flock decodes (`Flock.Program.decode`, as the sampled verifier's `Spot.Program.parse` does), and its
gates are that Program's (`Flattens`). The deployed parser is the sampled verifier's (`SamplingDeploy.Code.parse`), so
it is that code's premise. It is false at the deployed sampled verifiers' parsers, SPOT[Flock]'s and one-stage v2's, at
every statement whose descriptor's Program has an input gate at gate 0, since their statements (SPOT[Flock]'s over at
least one member) have the constant there (`Properties.NetworkScan.not_flattens_bitsProgs`,
`Properties.NetworkScan.not_flattens_stmtA`), so `ScanScopeAcc` says nothing of a deployed scan whose Program has one
until it is restated. -/
def StmtFlattens {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (D : Claims.SamplingGeneric.Compiled Coin A) :
    Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ Flattens P s.prog

end Properties.NetworkScan
