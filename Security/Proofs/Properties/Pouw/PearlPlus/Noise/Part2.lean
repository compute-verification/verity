import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey

/-! Noise lines 64 to 95 of the test noise's F_A and F_B and rows 16 to 23 of its G, one kernel computation
each (`Witness.test_fa4` to `Witness.test_gram5`), checked one after another, so that a build holds one
chunk's kernel state per module at a time. -/

set_option Elab.async false

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_fa4 : ((((List.range 128).zip testFa.toList).drop (16 * 4)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa5 : ((((List.range 128).zip testFa.toList).drop (16 * 5)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fb4 : ((((List.range 128).zip testFb.toList).drop (16 * 4)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 1 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fb5 : ((((List.range 128).zip testFb.toList).drop (16 * 5)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 1 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram4 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 4)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram5 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 5)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
