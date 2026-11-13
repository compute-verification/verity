module

public import Properties.NetworkScan.Flattens

/-!
# The compiled sampled verifier's premises under an accepted scan's scope

`Properties.NetworkScan.ScanScopeAcc` carries the scan's fit into the sampled audit's scope at a compiled sampled
verifier meeting two premises, each a named `Prop` here: its statements are their descriptors flattened
(`StmtFlattens`), and its scope check is its sampling scope (`ScopeMatches`). Both are the deployed sampled verifier's
facts, of its parse (`SamplingDeploy.Code.parse`) and its scope check, which the network check doesn't read, so the
scan's guarantee takes them as they stand.
-/

@[expose] public section

namespace Properties.NetworkScan

/-- **The compiled verifier's statements are their descriptors flattened**: every statement `D` parses has a
descriptor Flock decodes (`Flock.Program.decode`, as the sampled verifier's `Spot.Program.parse` does), and its
gates are that Program's (`Flattens`). The deployed parser is the sampled verifier's (`SamplingDeploy.Code.parse`), so
it is that code's premise. -/
def StmtFlattens {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (D : Claims.SamplingGeneric.Compiled Coin A) :
    Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ Flattens P s.prog

/-- **The compiled verifier's scope check is its sampling scope** (`Claims.ScopeMatches`): a statement is in the
compiled sampled verifier's scope (`Claims.SamplingGeneric.Compiled.Scope`) exactly when the scope check takes it, the
premise the sampled verifier's own guarantees take. -/
def ScopeMatches {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
    (D : Claims.SamplingGeneric.Compiled Coin A) : Prop :=
  Claims.ScopeMatches D.Scope D.scopeCheck

end Properties.NetworkScan
