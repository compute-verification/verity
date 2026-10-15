import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-nvfp4`: selected cells' `C̃` chains (`ch4`) (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16n_c_0_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 0).1 (rowB sm120Nvf4 t16n 0).1 (rowA sm120Nvf4 t16n 0).2
    (rowB sm120Nvf4 t16n 0).2 = some (cell t16n.c 0 0) := by decide +kernel

theorem t16n_c_0_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 0).1 (rowB sm120Nvf4 t16n 1).1 (rowA sm120Nvf4 t16n 0).2
    (rowB sm120Nvf4 t16n 1).2 = some (cell t16n.c 0 1) := by decide +kernel

theorem t16n_c_0_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 0).1 (rowB sm120Nvf4 t16n 2).1 (rowA sm120Nvf4 t16n 0).2
    (rowB sm120Nvf4 t16n 2).2 = some (cell t16n.c 0 2) := by decide +kernel

theorem t16n_c_0_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 0).1 (rowB sm120Nvf4 t16n 3).1 (rowA sm120Nvf4 t16n 0).2
    (rowB sm120Nvf4 t16n 3).2 = some (cell t16n.c 0 3) := by decide +kernel

theorem t16n_c_1_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 1).1 (rowB sm120Nvf4 t16n 0).1 (rowA sm120Nvf4 t16n 1).2
    (rowB sm120Nvf4 t16n 0).2 = some (cell t16n.c 1 0) := by decide +kernel

theorem t16n_c_1_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 1).1 (rowB sm120Nvf4 t16n 1).1 (rowA sm120Nvf4 t16n 1).2
    (rowB sm120Nvf4 t16n 1).2 = some (cell t16n.c 1 1) := by decide +kernel

theorem t16n_c_1_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 1).1 (rowB sm120Nvf4 t16n 2).1 (rowA sm120Nvf4 t16n 1).2
    (rowB sm120Nvf4 t16n 2).2 = some (cell t16n.c 1 2) := by decide +kernel

theorem t16n_c_1_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 1).1 (rowB sm120Nvf4 t16n 3).1 (rowA sm120Nvf4 t16n 1).2
    (rowB sm120Nvf4 t16n 3).2 = some (cell t16n.c 1 3) := by decide +kernel

theorem t16n_c_2_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 2).1 (rowB sm120Nvf4 t16n 0).1 (rowA sm120Nvf4 t16n 2).2
    (rowB sm120Nvf4 t16n 0).2 = some (cell t16n.c 2 0) := by decide +kernel

theorem t16n_c_2_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 2).1 (rowB sm120Nvf4 t16n 1).1 (rowA sm120Nvf4 t16n 2).2
    (rowB sm120Nvf4 t16n 1).2 = some (cell t16n.c 2 1) := by decide +kernel

theorem t16n_c_2_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 2).1 (rowB sm120Nvf4 t16n 2).1 (rowA sm120Nvf4 t16n 2).2
    (rowB sm120Nvf4 t16n 2).2 = some (cell t16n.c 2 2) := by decide +kernel

theorem t16n_c_2_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 2).1 (rowB sm120Nvf4 t16n 3).1 (rowA sm120Nvf4 t16n 2).2
    (rowB sm120Nvf4 t16n 3).2 = some (cell t16n.c 2 3) := by decide +kernel

theorem t16n_c_3_0 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 3).1 (rowB sm120Nvf4 t16n 0).1 (rowA sm120Nvf4 t16n 3).2
    (rowB sm120Nvf4 t16n 0).2 = some (cell t16n.c 3 0) := by decide +kernel

theorem t16n_c_3_1 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 3).1 (rowB sm120Nvf4 t16n 1).1 (rowA sm120Nvf4 t16n 3).2
    (rowB sm120Nvf4 t16n 1).2 = some (cell t16n.c 3 1) := by decide +kernel

theorem t16n_c_3_2 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 3).1 (rowB sm120Nvf4 t16n 2).1 (rowA sm120Nvf4 t16n 3).2
    (rowB sm120Nvf4 t16n 2).2 = some (cell t16n.c 3 2) := by decide +kernel

theorem t16n_c_3_3 : ch4 sm120Nvf4 (rowA sm120Nvf4 t16n 3).1 (rowB sm120Nvf4 t16n 3).1 (rowA sm120Nvf4 t16n 3).2
    (rowB sm120Nvf4 t16n 3).2 = some (cell t16n.c 3 3) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
