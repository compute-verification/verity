import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey

/-! Noise lines 0 to 31 of the test noise's F_A and F_B and rows 0 to 7 of its G, one kernel computation
each (`Witness.test_fa0` to `Witness.test_gram1`), checked one after another, so that a build holds one
chunk's kernel state per module at a time. -/

set_option Elab.async false

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_fa0 : ((((List.range 128).zip testFa.toList).drop (16 * 0)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa1 : ((((List.range 128).zip testFa.toList).drop (16 * 1)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fb0 : ((((List.range 128).zip testFb.toList).drop (16 * 0)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 1 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fb1 : ((((List.range 128).zip testFb.toList).drop (16 * 1)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 1 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram0 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 0)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram1 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 1)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
