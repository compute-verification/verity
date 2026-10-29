import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `prod-64-nvfp4`: selected cells' `C̃` chains (`ch4`), rows 2 and 3 (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem p64n_c_2_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 2).1 (rowB sm120Nvf4 p64n 0).1 (rowA sm120Nvf4 p64n 2).2
    (rowB sm120Nvf4 p64n 0).2 = some (cell p64n.c 2 0) := by decide +kernel

theorem p64n_c_2_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 2).1 (rowB sm120Nvf4 p64n 1).1 (rowA sm120Nvf4 p64n 2).2
    (rowB sm120Nvf4 p64n 1).2 = some (cell p64n.c 2 1) := by decide +kernel

theorem p64n_c_2_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 2).1 (rowB sm120Nvf4 p64n 2).1 (rowA sm120Nvf4 p64n 2).2
    (rowB sm120Nvf4 p64n 2).2 = some (cell p64n.c 2 2) := by decide +kernel

theorem p64n_c_2_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 2).1 (rowB sm120Nvf4 p64n 3).1 (rowA sm120Nvf4 p64n 2).2
    (rowB sm120Nvf4 p64n 3).2 = some (cell p64n.c 2 3) := by decide +kernel

theorem p64n_c_3_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 3).1 (rowB sm120Nvf4 p64n 0).1 (rowA sm120Nvf4 p64n 3).2
    (rowB sm120Nvf4 p64n 0).2 = some (cell p64n.c 3 0) := by decide +kernel

theorem p64n_c_3_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 3).1 (rowB sm120Nvf4 p64n 1).1 (rowA sm120Nvf4 p64n 3).2
    (rowB sm120Nvf4 p64n 1).2 = some (cell p64n.c 3 1) := by decide +kernel

theorem p64n_c_3_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 3).1 (rowB sm120Nvf4 p64n 2).1 (rowA sm120Nvf4 p64n 3).2
    (rowB sm120Nvf4 p64n 2).2 = some (cell p64n.c 3 2) := by decide +kernel

theorem p64n_c_3_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 p64n 3).1 (rowB sm120Nvf4 p64n 3).1 (rowA sm120Nvf4 p64n 3).2
    (rowB sm120Nvf4 p64n 3).2 = some (cell p64n.c 3 3) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
