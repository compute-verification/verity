import Pouw.Guarantees.PearlC
import Pouw.SecurityProofs.PearlC.DeviceCapRev1Gamma

/-!
# Proofs of Pearl-C's guarantees

One theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G` per guarantee, which is the theorem in
`Pouw.PearlC` (`DeviceCapRev1Gamma`) that proves the same statement under its old name.
-/

namespace Pouw.SecurityProofs.PearlC

theorem GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun CM sem hTT => Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192 CM sem hTT

end Pouw.SecurityProofs.PearlC
