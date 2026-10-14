import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-mxfp4`: rows 16 to 23 of `G = F_Bᵀ·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16m_g_16 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 16) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 16 []) := by decide +kernel

theorem t16m_g_17 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 17) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 17 []) := by decide +kernel

theorem t16m_g_18 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 18) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 18 []) := by decide +kernel

theorem t16m_g_19 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 19) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 19 []) := by decide +kernel

theorem t16m_g_20 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 20) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 20 []) := by decide +kernel

theorem t16m_g_21 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 21) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 21 []) := by decide +kernel

theorem t16m_g_22 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 22) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 22 []) := by decide +kernel

theorem t16m_g_23 : (List.ofFn fun q' => ch4 sm120Mxf4 (colF (basis t16m.k t16m.fb) 23) (colF (basis t16m.k t16m.fa) q')
    (onesScales sm120Mxf4 t16m.k) (onesScales sm120Mxf4 t16m.k)).mapM id = some (t16m.g.getD 23 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
