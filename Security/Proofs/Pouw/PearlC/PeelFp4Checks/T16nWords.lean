import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-nvfp4`: `P_A,i`, `P_B,j` and each cell's `uWord?` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16n_pa_0 : paRow? (List.ofFn (beA t16n 0)) (t16n.af.getD 0 []) =
    some (t16n.pa.getD 0 []) := by decide +kernel

theorem t16n_pa_1 : paRow? (List.ofFn (beA t16n 1)) (t16n.af.getD 1 []) =
    some (t16n.pa.getD 1 []) := by decide +kernel

theorem t16n_pa_2 : paRow? (List.ofFn (beA t16n 2)) (t16n.af.getD 2 []) =
    some (t16n.pa.getD 2 []) := by decide +kernel

theorem t16n_pa_3 : paRow? (List.ofFn (beA t16n 3)) (t16n.af.getD 3 []) =
    some (t16n.pa.getD 3 []) := by decide +kernel

theorem t16n_pb_0 : pbRow? (List.ofFn (beB t16n 0)) (t16n.bf.getD 0 []) t16n.g =
    some (t16n.pb.getD 0 []) := by decide +kernel

theorem t16n_pb_1 : pbRow? (List.ofFn (beB t16n 1)) (t16n.bf.getD 1 []) t16n.g =
    some (t16n.pb.getD 1 []) := by decide +kernel

theorem t16n_pb_2 : pbRow? (List.ofFn (beB t16n 2)) (t16n.bf.getD 2 []) t16n.g =
    some (t16n.pb.getD 2 []) := by decide +kernel

theorem t16n_pb_3 : pbRow? (List.ofFn (beB t16n 3)) (t16n.bf.getD 3 []) t16n.g =
    some (t16n.pb.getD 3 []) := by decide +kernel

theorem t16n_u_0_0 : uWord? (cell t16n.c 0 0) (t16n.pa.getD 0 []) (t16n.pb.getD 0 []) =
    some (cell t16n.u 0 0) := by decide +kernel

theorem t16n_u_0_1 : uWord? (cell t16n.c 0 1) (t16n.pa.getD 0 []) (t16n.pb.getD 1 []) =
    some (cell t16n.u 0 1) := by decide +kernel

theorem t16n_u_0_2 : uWord? (cell t16n.c 0 2) (t16n.pa.getD 0 []) (t16n.pb.getD 2 []) =
    some (cell t16n.u 0 2) := by decide +kernel

theorem t16n_u_0_3 : uWord? (cell t16n.c 0 3) (t16n.pa.getD 0 []) (t16n.pb.getD 3 []) =
    some (cell t16n.u 0 3) := by decide +kernel

theorem t16n_u_1_0 : uWord? (cell t16n.c 1 0) (t16n.pa.getD 1 []) (t16n.pb.getD 0 []) =
    some (cell t16n.u 1 0) := by decide +kernel

theorem t16n_u_1_1 : uWord? (cell t16n.c 1 1) (t16n.pa.getD 1 []) (t16n.pb.getD 1 []) =
    some (cell t16n.u 1 1) := by decide +kernel

theorem t16n_u_1_2 : uWord? (cell t16n.c 1 2) (t16n.pa.getD 1 []) (t16n.pb.getD 2 []) =
    some (cell t16n.u 1 2) := by decide +kernel

theorem t16n_u_1_3 : uWord? (cell t16n.c 1 3) (t16n.pa.getD 1 []) (t16n.pb.getD 3 []) =
    some (cell t16n.u 1 3) := by decide +kernel

theorem t16n_u_2_0 : uWord? (cell t16n.c 2 0) (t16n.pa.getD 2 []) (t16n.pb.getD 0 []) =
    some (cell t16n.u 2 0) := by decide +kernel

theorem t16n_u_2_1 : uWord? (cell t16n.c 2 1) (t16n.pa.getD 2 []) (t16n.pb.getD 1 []) =
    some (cell t16n.u 2 1) := by decide +kernel

theorem t16n_u_2_2 : uWord? (cell t16n.c 2 2) (t16n.pa.getD 2 []) (t16n.pb.getD 2 []) =
    some (cell t16n.u 2 2) := by decide +kernel

theorem t16n_u_2_3 : uWord? (cell t16n.c 2 3) (t16n.pa.getD 2 []) (t16n.pb.getD 3 []) =
    some (cell t16n.u 2 3) := by decide +kernel

theorem t16n_u_3_0 : uWord? (cell t16n.c 3 0) (t16n.pa.getD 3 []) (t16n.pb.getD 0 []) =
    some (cell t16n.u 3 0) := by decide +kernel

theorem t16n_u_3_1 : uWord? (cell t16n.c 3 1) (t16n.pa.getD 3 []) (t16n.pb.getD 1 []) =
    some (cell t16n.u 3 1) := by decide +kernel

theorem t16n_u_3_2 : uWord? (cell t16n.c 3 2) (t16n.pa.getD 3 []) (t16n.pb.getD 2 []) =
    some (cell t16n.u 3 2) := by decide +kernel

theorem t16n_u_3_3 : uWord? (cell t16n.c 3 3) (t16n.pa.getD 3 []) (t16n.pb.getD 3 []) =
    some (cell t16n.u 3 3) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
