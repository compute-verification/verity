import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-mxfp4`: rows 0 to 7 of `G = F_Bᵀ·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16m_g_0 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 0) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 0 []) := by decide +kernel

theorem t16m_g_1 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 1) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 1 []) := by decide +kernel

theorem t16m_g_2 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 2) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 2 []) := by decide +kernel

theorem t16m_g_3 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 3) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 3 []) := by decide +kernel

theorem t16m_g_4 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 4) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 4 []) := by decide +kernel

theorem t16m_g_5 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 5) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 5 []) := by decide +kernel

theorem t16m_g_6 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 6) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 6 []) := by decide +kernel

theorem t16m_g_7 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 7) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 7 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
