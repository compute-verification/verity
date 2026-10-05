import FlockSoundness.Protocol.Discharge.Exec.Merkle

/-!
# The link points' count of a `setupH` statement

`setupH` sets `m_pts = PT_LOCAL + nbl`, and `nbl` is read off the number of units the statement proves and the
circuit's `g` (`setupH_facts`, `mPtsOf`). For a drawn session that number is the draw's proved units, its drawn units
and their closure (`drawn_n`, `setupH_mPts_drawn`). The circuit is the circuit file's parse whatever the draw, so two
draws that prove as many units set up at the same `m_pts` (`setupH_mPts_eq`), namely `mPtsOf c.g N₀` for `N₀` proved
units (`setupH_mPts_of_proved`). A draw's proved units include its closure, so under a law with a closure `m_pts` can
differ between draws of the same size.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

/-- `m_pts` for `n` proved units of a circuit with `g` units per block: `PT_LOCAL + nbl`, `nbl` as `setupH` computes it. -/
def mPtsOf (g n : ℕ) : ℕ := PT_LOCAL + max 3 (log2ceil (max ((n + g - 1) / g) 1))

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

end FlockSoundness.Discharge.Exec
