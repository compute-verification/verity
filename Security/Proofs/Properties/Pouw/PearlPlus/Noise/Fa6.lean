import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey

/-! Noise lines 96 to 111 of the test noise's F_A, one kernel computation (`Witness.test_fa6`). -/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_fa6 : ((((List.range 128).zip testFa.toList).drop (16 * 6)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
