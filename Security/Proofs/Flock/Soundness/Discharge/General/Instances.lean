import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Flock.Test.Wires

/-!
# The general form at the tails' shapes

`Layout.scopeGenH` on `Flock.Test.Wires`'s circuits, by evaluation (`#guard`), against `Layout.scopeOkH`:

* RMSNorm (`rms 0`): per-warp units, each returning its square to a stage whose result (the row's rsq) is wired back to
  every unit's cut input. The units read two input groups, so `scopeOkH` refuses it and `scopeGenH` covers it, with the
  outputs returned by the units or (`outNet := 1`) by the stage.
* The GEMM accumulator chain (`gemm 4`): unit `i + 1`'s cut input is unit `i`'s accumulator, a stage reads the last.
* A lookup slot between two stages (`lookup false`).
* At a cycle through the wires (`rms 1`, `lookup true`) both refuse: `wiresAcyclic` is the general form's one refusal
  on the wires.
-/

namespace FlockSoundness.Discharge.General

open Flock Flock.Test.Wires

#guard Layout.scopeGenH (rms 0)
#guard !Layout.scopeOkH (rms 0)
#guard Layout.scopeGenH { rms 0 with outNet := 1 }
#guard !Layout.scopeOkH { rms 0 with outNet := 1 }
#guard !Layout.scopeGenH (rms 1)

#guard Layout.scopeGenH (gemm 4)
#guard !Layout.scopeOkH (gemm 4)
#guard Layout.scopeGenH { gemm 4 with outNet := 1 }
#guard !Layout.scopeGenH { gemm 4 with wires := (gemm 4).wires.set! 0 ⟨(0, 0, 1), (1, 0, 0)⟩ }

#guard Layout.scopeGenH (lookup false)
#guard !Layout.scopeGenH (lookup true)

end FlockSoundness.Discharge.General
