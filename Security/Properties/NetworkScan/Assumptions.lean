import Properties.NetworkScan.Flattens

/-!
# The deployment's premise under an accepted scan's scope

`Properties.NetworkScan.ScanScopeAcc` carries the scan's fit into the sampled audit's scope at a deployment whose scope
check is its sampling scope (`ScopeMatches`), a named `Prop` here: the deployed sampled verifier's fact of its scope
check, which the network check doesn't read, so the scan's guarantee takes it as it stands. That a statement's circuit
is its audit's members' joint program (`Flattens`) is no premise: the scan's statements take it of the statement.
-/

namespace Properties.NetworkScan

/-- **The deployment's scope check is its sampling scope** (`Claims.Sampling.ScopeMatches`): a statement is in the
sampled verifier's scope (`Claims.Sampling.Scope`) exactly when the scope check takes it, the premise the sampled
verifier's own guarantees take. -/
def ScopeMatches {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (D : Claims.Sampling.Deploy Coin Col T) : Prop :=
  Claims.Sampling.ScopeMatches D

end Properties.NetworkScan
