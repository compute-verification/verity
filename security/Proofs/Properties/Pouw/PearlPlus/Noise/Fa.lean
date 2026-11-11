module

public import Proofs.Properties.Pouw.PearlPlus.Noise.LineKey
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Range.Basic
import all Init.Data.String.Defs

/-! Noise lines 0 to 127 of the test noise's F_A, sixteen to a kernel computation (`Witness.test_fa0` to
`Witness.test_fa7`), checked one after another, so that a build holds one chunk's kernel state for the module at a
time. -/

@[expose] public section

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

theorem test_fa2 : ((((List.range 128).zip testFa.toList).drop (16 * 2)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa3 : ((((List.range 128).zip testFa.toList).drop (16 * 3)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa4 : ((((List.range 128).zip testFa.toList).drop (16 * 4)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa5 : ((((List.range 128).zip testFa.toList).drop (16 * 5)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa6 : ((((List.range 128).zip testFa.toList).drop (16 * 6)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

theorem test_fa7 : ((((List.range 128).zip testFa.toList).drop (16 * 7)).take 16).all
    (fun p => decide ((Words.sampleLine lineKeyB 0 1 p.1 Words.NORM_F).toOption = some p.2)) = true := by
  decide +kernel

end Witness

end Properties.Pouw.PearlPlus
