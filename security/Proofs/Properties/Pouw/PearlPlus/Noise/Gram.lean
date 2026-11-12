module

public import Properties.Pouw.PearlPlus.Witnesses
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Range.Basic

/-! Rows 0 to 31 of the test noise's G, four to a kernel computation (`Witness.test_gram0` to `Witness.test_gram7`),
checked one after another, so that a build holds one chunk's kernel state for the module at a time. -/

@[expose] public section

set_option Elab.async false

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

theorem test_gram0 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 0)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram1 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 1)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram2 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 2)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram3 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 3)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram4 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 4)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram5 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 5)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram6 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 6)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

theorem test_gram7 : (((((Array.range R).map (Words.column testFb)).toList.zip testGram.toList).drop
    (4 * 7)).take 4).all (fun p => decide ((((Array.range R).map (Words.column testFa)).mapM fun faq =>
      Words.chain p.1 faq).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
