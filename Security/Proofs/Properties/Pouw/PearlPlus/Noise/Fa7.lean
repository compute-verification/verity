import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey

/-! Noise lines 112 to 127 of the test noise's F_A, one kernel computation (`Witness.test_fa7`). -/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_fa7 : ((((List.range 128).zip testFa.toList).drop (16 * 7)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
