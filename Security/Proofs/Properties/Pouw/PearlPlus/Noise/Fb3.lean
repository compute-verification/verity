import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey

/-! Noise lines 48 to 63 of the test noise's F_B, one kernel computation (`Witness.test_fb3`). -/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_fb3 : ((((List.range 128).zip testFb.toList).drop (16 * 3)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 1 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
