import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-nvfp4`: rows 16 to 23 of `G = F_Bᵀ·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16n_g_16 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 16) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 16 []) := by decide +kernel

theorem t16n_g_17 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 17) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 17 []) := by decide +kernel

theorem t16n_g_18 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 18) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 18 []) := by decide +kernel

theorem t16n_g_19 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 19) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 19 []) := by decide +kernel

theorem t16n_g_20 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 20) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 20 []) := by decide +kernel

theorem t16n_g_21 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 21) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 21 []) := by decide +kernel

theorem t16n_g_22 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 22) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 22 []) := by decide +kernel

theorem t16n_g_23 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 23) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 23 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
