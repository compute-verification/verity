import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-nvfp4`: rows 24 to 31 of `G = F_Bᵀ·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16n_g_24 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 24) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 24 []) := by decide +kernel

theorem t16n_g_25 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 25) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 25 []) := by decide +kernel

theorem t16n_g_26 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 26) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 26 []) := by decide +kernel

theorem t16n_g_27 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 27) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 27 []) := by decide +kernel

theorem t16n_g_28 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 28) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 28 []) := by decide +kernel

theorem t16n_g_29 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 29) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 29 []) := by decide +kernel

theorem t16n_g_30 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 30) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 30 []) := by decide +kernel

theorem t16n_g_31 : (List.ofFn fun q' => ch4 sm120Nvf4 (colF (basis t16n.k t16n.fb) 31) (colF (basis t16n.k t16n.fa) q')
    (onesScales sm120Nvf4 t16n.k) (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.g.getD 31 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
