import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-mxfp4`: selected rows of `A′·F_B` and `B̃·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16m_af_0 : (List.ofFn fun q => ch4 sm120Mxf4 (rowA sm120Mxf4 t16m 0).1 (colF (basis t16m.k t16m.fb) q)
    (rowA sm120Mxf4 t16m 0).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.af.getD 0 []) := by decide +kernel

theorem t16m_af_1 : (List.ofFn fun q => ch4 sm120Mxf4 (rowA sm120Mxf4 t16m 1).1 (colF (basis t16m.k t16m.fb) q)
    (rowA sm120Mxf4 t16m 1).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.af.getD 1 []) := by decide +kernel

theorem t16m_af_2 : (List.ofFn fun q => ch4 sm120Mxf4 (rowA sm120Mxf4 t16m 2).1 (colF (basis t16m.k t16m.fb) q)
    (rowA sm120Mxf4 t16m 2).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.af.getD 2 []) := by decide +kernel

theorem t16m_af_3 : (List.ofFn fun q => ch4 sm120Mxf4 (rowA sm120Mxf4 t16m 3).1 (colF (basis t16m.k t16m.fb) q)
    (rowA sm120Mxf4 t16m 3).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.af.getD 3 []) := by decide +kernel

theorem t16m_bf_0 : (List.ofFn fun q => ch4 sm120Mxf4 (rowB sm120Mxf4 t16m 0).1 (colF (basis t16m.k t16m.fa) q)
    (rowB sm120Mxf4 t16m 0).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.bf.getD 0 []) := by decide +kernel

theorem t16m_bf_1 : (List.ofFn fun q => ch4 sm120Mxf4 (rowB sm120Mxf4 t16m 1).1 (colF (basis t16m.k t16m.fa) q)
    (rowB sm120Mxf4 t16m 1).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.bf.getD 1 []) := by decide +kernel

theorem t16m_bf_2 : (List.ofFn fun q => ch4 sm120Mxf4 (rowB sm120Mxf4 t16m 2).1 (colF (basis t16m.k t16m.fa) q)
    (rowB sm120Mxf4 t16m 2).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.bf.getD 2 []) := by decide +kernel

theorem t16m_bf_3 : (List.ofFn fun q => ch4 sm120Mxf4 (rowB sm120Mxf4 t16m 3).1 (colF (basis t16m.k t16m.fa) q)
    (rowB sm120Mxf4 t16m 3).2 (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.bf.getD 3 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
