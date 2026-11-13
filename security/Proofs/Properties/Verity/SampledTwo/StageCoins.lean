module

public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.StageCut

/-!
# The stages' coins agree with the device's releases

The two pieces of the projection that read the stages' coins (`ProjectionOf`'s `CoinsAgree` at `.first` and
`.second`), at the form's reading (`formReading`) and the cut's `LogIsDevice`: `coinsAgree_first_log` and
`coinsAgree_second_log`. Each rests on the coin map's reading of its rounds (`DevFirst`, `DevSecond`) and on
`StageCut`'s premises (`CaseIsSent`, `CutReads`, and `FirstDrawn` or `SecondAlone`).
-/

@[expose] public section

namespace Verity.SampledTwo

/-- **The first stage's coins agree with the device's releases**, where the play's log is the device's as the cut
reads it. -/
theorem coinsAgree_first_log (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
    (hC : CaseIsSent C) (hR : CutReads C) (hD : FirstDrawn C) (hDev : DevFirst C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) LogIsDevice .first :=
  coinsAgree_first_of A C S dev LogIsDevice (firstShown_of C hC hR hD) hDev

/-- **The interior's coins agree with the device's releases**, where the play's log is the device's as the cut reads
it. -/
theorem coinsAgree_second_log (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
    (hC : CaseIsSent C) (hR : CutReads C) (hA : SecondAlone C) (hDev : DevSecond A C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) LogIsDevice .second :=
  coinsAgree_second_of A C S dev LogIsDevice (secondShown_of C hC hR hA) hDev

end Verity.SampledTwo
