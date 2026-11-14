module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile

/-!
# The circuit is in the proved scope and its ports' published domains are distinct

`Layout.scopeOkH` holds of `cH` (one 64-word `sha512/row/v2` output row, no input row), `ProvedScope.circuitOf` reads the
circuit file to `cH`, and `PortRoots.checkOf` passes at the public file: the one port's domain is 64 bytes and has no
twin. The walk of `scopeOkH` is the one the local probe of `filesWhy` evaluates; here it reads `CircuitFile`'s fields.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Scope

open Flock

theorem scopeOkH : FlockSoundness.Discharge.Layout.scopeOkH cH = true := by
  sorry -- piece P5 (scope and ports)

theorem circuitOf_eq : ProvedScope.circuitOf tagsZ circuitFile [] = .ok cH := by
  sorry -- piece P5 (scope and ports)

theorem ports_ok : PortRoots.checkOf (.ok cH) tagsZ publicFile = .ok () := by
  sorry -- piece P5 (scope and ports)

theorem ready : PortRoots.checkOf (ProvedScope.circuitOf tagsZ circuitFile []) tagsZ publicFile = .ok () := by
  rw [circuitOf_eq]
  exact ports_ok

end Verity.Sampled.Instance.HonestSession.Scope
