import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `prod-64-nvfp4`: selected cells' `C̃` chains (`ch4`), rows 0 and 1 (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem p64n_c_0_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 0).1 (rowB sm120Nvf4 p64n 0).1 (rowA sm120Nvf4 p64n 0).2
    (rowB sm120Nvf4 p64n 0).2 = some (cell p64n.c 0 0) := by decide +kernel

theorem p64n_c_0_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 0).1 (rowB sm120Nvf4 p64n 1).1 (rowA sm120Nvf4 p64n 0).2
    (rowB sm120Nvf4 p64n 1).2 = some (cell p64n.c 0 1) := by decide +kernel

theorem p64n_c_0_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 0).1 (rowB sm120Nvf4 p64n 2).1 (rowA sm120Nvf4 p64n 0).2
    (rowB sm120Nvf4 p64n 2).2 = some (cell p64n.c 0 2) := by decide +kernel

theorem p64n_c_0_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 0).1 (rowB sm120Nvf4 p64n 3).1 (rowA sm120Nvf4 p64n 0).2
    (rowB sm120Nvf4 p64n 3).2 = some (cell p64n.c 0 3) := by decide +kernel

theorem p64n_c_1_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 1).1 (rowB sm120Nvf4 p64n 0).1 (rowA sm120Nvf4 p64n 1).2
    (rowB sm120Nvf4 p64n 0).2 = some (cell p64n.c 1 0) := by decide +kernel

theorem p64n_c_1_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 1).1 (rowB sm120Nvf4 p64n 1).1 (rowA sm120Nvf4 p64n 1).2
    (rowB sm120Nvf4 p64n 1).2 = some (cell p64n.c 1 1) := by decide +kernel

theorem p64n_c_1_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 1).1 (rowB sm120Nvf4 p64n 2).1 (rowA sm120Nvf4 p64n 1).2
    (rowB sm120Nvf4 p64n 2).2 = some (cell p64n.c 1 2) := by decide +kernel

theorem p64n_c_1_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 1).1 (rowB sm120Nvf4 p64n 3).1 (rowA sm120Nvf4 p64n 1).2
    (rowB sm120Nvf4 p64n 3).2 = some (cell p64n.c 1 3) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
