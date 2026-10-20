import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.SimPiopOf
import Proofs.Flock.Claims.ZK.PiopSrcSv

/-!
# `SimPiop`

`SimPiop` (`HybridStep 0`): the session's PIOP and the cut's send the same, from `simPiop_of` at `piopSrcE`. At every
live statement, witness in the deployed relation, tape, opened coins and stop with no refusal seen, table `j` has at
least `14` variables, and its rep `r` source from the cut is at the opened coins: its masked values at the zerocheck's
and the lincheck's pad indices (`src_r1ab` … `src_fc`, `src_lc0`, `src_lc1`, `src_zp`) and its switched vectors
wherever the rep sends them before the stop (`src_sv`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The deployed source is at the opened coins** (`PiopSrcE`). -/
theorem piopSrcE : PiopSrcE := by
  intro x hx w hw τ c n hn s spec hsx j r extras he
  rw [tabsOf_live x j] at he ⊢
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  refine ⟨le_trans (by norm_num) (hall j.val (lt_of_jOf j)).1, ?_⟩
  exact
    { r1ab := fun i hi => src_r1ab c hw τ hsx j r hi
      r1c := fun i hi => src_r1c c hw τ hsx j r hi
      ml0 := fun i hi => src_ml0 c hw τ hsx j r hi
      ml1 := fun i hi => src_ml1 c hw τ hsx j r hi
      fa := src_fa c hw τ hsx j r
      fb := src_fb c hw τ hsx j r
      fc := src_fc c hw τ hsx j r
      lc := fun _ hfold _ hi => ⟨src_lc0 c hw τ j r hfold hi, src_lc1 c hw τ j r hfold hi⟩
      zp := fun _ hi => src_zp c hw τ j r hi
      sv := src_sv c hw τ j r hx hn hsx he }

/-- **`SimPiop`**: `repMix 0` and `repMix 1` send the same at every live statement, witness, tape, opened coins and stop
with no refusal seen. -/
theorem simPiop : SimPiop := simPiop_of piopSrcE

end Masked

end Claims.ZKMal
