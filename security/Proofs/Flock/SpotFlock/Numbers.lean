module

public import Proofs.Flock.SpotFlock.KS

/-! SPOT[Flock]'s inner numbers are Flock's at every unit's session statement at its template's `N₀`
(`innerNumbersAt_code`). -/

@[expose] public section

namespace SpotFlock

variable (h : Held)

/-- One `N₀` per statement (`InnerNumbers`) would hold only where every template's is the same. -/
theorem innerNumbersAt_code (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Sampling.InnerNumbersAt (SamplingDeploy.deploy rel (code h)) (code h).unitN₀ :=
  SamplingDeploy.innerNumbersAt_deploy rel (code h) (sessionsIn_code h)

end SpotFlock
