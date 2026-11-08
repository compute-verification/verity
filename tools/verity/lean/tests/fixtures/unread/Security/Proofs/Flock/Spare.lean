import Proofs.Flock.Main

namespace Spare

theorem quadruple (n : Nat) : Core.double (Core.double n) = 4 * n := by
  rw [Core.helper, Core.helper]
  omega

end Spare
