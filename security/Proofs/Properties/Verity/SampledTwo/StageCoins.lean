module

public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.StageAlone

/-! The stages' coins agree with the device's releases, at the form's reading and the cut's `LogIsDevice`. -/

@[expose] public section

namespace Verity.SampledTwo

theorem coinsAgree_first_log (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
    (hC : CaseIsSent C) (hR : CutReads C) (hD : FirstSampled C) (hDev : DevFirst C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) LogIsDevice .first :=
  coinsAgree_first_of A C S dev LogIsDevice (firstShown_of C hC hR hD) hDev

theorem coinsAgree_second_log (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
    (hC : CaseIsSent C) (hR : CutReads C) (hA : SecondAlone C) (hDev : DevSecond A C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) LogIsDevice .second :=
  coinsAgree_second_of A C S dev LogIsDevice (secondShown_of C hC hR hA) hDev

theorem coinsAgree_second_matches (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
    (hC : CaseIsSent C) (hR : CutReads C) (hM : SessionMatches C) (hDev : DevSecond A C dev) :
    CoinsAgree A (formReading A C S dev) (FormAccepts C) LogIsDevice .second :=
  coinsAgree_second_log A C S dev hC hR (secondAlone_of_matches C hC hM) hDev

end Verity.SampledTwo
