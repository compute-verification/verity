module

public import Flock.Test.Wires
public import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Positivity.Finset
import VCVio.Prelude.Core
meta import Flock.Test.Wires
meta import Proofs.Flock.Soundness.Discharge.Layout.Scope

/-! `Layout.scopeGenH` against `Layout.scopeOkH` on `Flock.Test.Wires`'s circuits, by evaluation; a cycle through the wires is the general form's one refusal on the wires. -/

@[expose] public section

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
