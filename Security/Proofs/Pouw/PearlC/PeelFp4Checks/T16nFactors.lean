import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-nvfp4`: selected rows of `A′·F_B` and `B̃·F_A` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16n_af_0 : (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 0).1 (colF (basis t16n.k t16n.fb) q)
    (rowA sm120Nvf4 t16n 0).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.af.getD 0 []) := by decide +kernel

theorem t16n_af_1 : (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 1).1 (colF (basis t16n.k t16n.fb) q)
    (rowA sm120Nvf4 t16n 1).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.af.getD 1 []) := by decide +kernel

theorem t16n_af_2 : (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 2).1 (colF (basis t16n.k t16n.fb) q)
    (rowA sm120Nvf4 t16n 2).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.af.getD 2 []) := by decide +kernel

theorem t16n_af_3 : (List.ofFn fun q => ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 3).1 (colF (basis t16n.k t16n.fb) q)
    (rowA sm120Nvf4 t16n 3).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.af.getD 3 []) := by decide +kernel

theorem t16n_bf_0 : (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 t16n 0).1 (colF (basis t16n.k t16n.fa) q)
    (rowB sm120Nvf4 t16n 0).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.bf.getD 0 []) := by decide +kernel

theorem t16n_bf_1 : (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 t16n 1).1 (colF (basis t16n.k t16n.fa) q)
    (rowB sm120Nvf4 t16n 1).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.bf.getD 1 []) := by decide +kernel

theorem t16n_bf_2 : (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 t16n 2).1 (colF (basis t16n.k t16n.fa) q)
    (rowB sm120Nvf4 t16n 2).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.bf.getD 2 []) := by decide +kernel

theorem t16n_bf_3 : (List.ofFn fun q => ch4 sm120Nvf4 (rowB sm120Nvf4 t16n 3).1 (colF (basis t16n.k t16n.fa) q)
    (rowB sm120Nvf4 t16n 3).2 (onesScales sm120Nvf4 t16n.k)).mapM id = some (t16n.bf.getD 3 []) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
