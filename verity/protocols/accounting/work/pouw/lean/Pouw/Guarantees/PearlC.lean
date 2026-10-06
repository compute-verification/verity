import Pouw.Protocol.PearlC.DeviceKernelWref
import Pouw.Protocol.PearlC.DevicePricesLoop
import Pouw.Assumptions.PearlC.TTOutRev1

/-!
# Pearl-C's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Pouw.Protocol`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`.
γ at the RTX PRO 6000's v1 record under rev1 at the cap 1/1,000 (`Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192`
in `Pouw.SecurityProofs.PearlC.DeviceCapRev1Gamma`), its binders written as a `∀`. `Pouw.CostModel` and `Pouw.Gγ` are
the game's, so Pearl-C's are named in full.
-/

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.PearlC.CostModel Q R S)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC

end Pouw.Guarantees.PearlC
