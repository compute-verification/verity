import Properties.NetworkScan.Flattens

/-!
# The deployment's premises under an accepted scan's scope

`Properties.NetworkScan.ScanScopeAcc` carries the scan's fit into the sampled audit's scope at a deployment meeting two
premises, each a named `Prop` here: its statements are their descriptors flattened (`StmtFlattens`), and its scope
check is its sampling scope (`ScopeMatches`). Both are the deployed sampled verifier's facts, of its parse
(`SamplingDeploy.Code.parse`) and its scope check, which the network check doesn't read, so the scan's guarantee takes
them as they stand.
-/

namespace Properties.NetworkScan

/-- **The deployment's statements are their descriptors flattened**: every statement `D` parses has a descriptor
C-Flock decodes (`Flock.Program.decode`, as the sampled verifier's `SampledProofs.Program.parse` does), and its gates
are that Program's (`Flattens`). The deployed parser is the sampled verifier's (`SamplingDeploy.Code.parse`), so it is
that code's premise. -/
def StmtFlattens {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ Flattens P s.prog

/-- **The deployment's scope check is its sampling scope** (`Claims.Sampling.ScopeMatches`): a statement is in the
sampled verifier's scope (`Claims.Sampling.Scope`) exactly when the scope check takes it, the premise the sampled
verifier's own guarantees take. -/
def ScopeMatches {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (D : Claims.Sampling.Deploy Coin Col T) : Prop :=
  Claims.Sampling.ScopeMatches D

end Properties.NetworkScan
