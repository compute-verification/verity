import Proofs.Accounting.CoinFlow.Flow

/-!
# The coin rules' guarantees

Each statement of `Definitions/Accounting/Guarantees/CoinFlow.lean`, proved in `CoinFlow/Flow.lean`: a noise injection's
fresh coins reach only its forward cone (rule (b)), a non-sink value reads the coins only through noise injections
(rule (a)), the forward cone is the one reader's, and so `CoinPointMass` holds of every view built from the rest of an
admitted Program.
-/

namespace Accounting.SecurityProofs.CoinFlow

theorem ViewIndependent : Accounting.Guarantees.CoinFlow.ViewIndependent := Accounting.CoinFlowProofs.viewIndependent
theorem CoinsThroughNoise : Accounting.Guarantees.CoinFlow.CoinsThroughNoise :=
  Accounting.CoinFlowProofs.coinsThroughNoise
theorem OneReader : Accounting.Guarantees.CoinFlow.OneReader := Accounting.CoinFlowProofs.oneReader
theorem PointMass : Accounting.Guarantees.CoinFlow.PointMass := Accounting.CoinFlowProofs.pointMass

end Accounting.SecurityProofs.CoinFlow
