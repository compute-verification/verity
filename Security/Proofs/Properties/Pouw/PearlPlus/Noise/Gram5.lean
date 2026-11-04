import Properties.Pouw.PearlPlus.Witnesses

/-! Rows 20 to 23 of the test noise's G, one kernel computation (`Witness.test_gram5`). -/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_gram5 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 5)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
