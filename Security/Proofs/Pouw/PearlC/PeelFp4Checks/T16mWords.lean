import Proofs.Pouw.PearlC.PeelFp4Checks.Defs

/-! # `test-16-mxfp4`: `P_A,i`, `P_B,j` and each cell's `uWord?` (`PeelFp4Checks`, which joins them) -/

namespace Pouw.PearlC.PeelFp4Checks

open Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelFp4Proofs
  Pouw.PearlC.PeelFp4Vectors

theorem t16m_pa_0 : paRow? (List.ofFn (beA t16m 0)) (t16m.af.getD 0 []) =
    some (t16m.pa.getD 0 []) := by decide +kernel

theorem t16m_pa_1 : paRow? (List.ofFn (beA t16m 1)) (t16m.af.getD 1 []) =
    some (t16m.pa.getD 1 []) := by decide +kernel

theorem t16m_pa_2 : paRow? (List.ofFn (beA t16m 2)) (t16m.af.getD 2 []) =
    some (t16m.pa.getD 2 []) := by decide +kernel

theorem t16m_pa_3 : paRow? (List.ofFn (beA t16m 3)) (t16m.af.getD 3 []) =
    some (t16m.pa.getD 3 []) := by decide +kernel

theorem t16m_pb_0 : pbRow? (List.ofFn (beB t16m 0)) (t16m.bf.getD 0 []) t16m.g =
    some (t16m.pb.getD 0 []) := by decide +kernel

theorem t16m_pb_1 : pbRow? (List.ofFn (beB t16m 1)) (t16m.bf.getD 1 []) t16m.g =
    some (t16m.pb.getD 1 []) := by decide +kernel

theorem t16m_pb_2 : pbRow? (List.ofFn (beB t16m 2)) (t16m.bf.getD 2 []) t16m.g =
    some (t16m.pb.getD 2 []) := by decide +kernel

theorem t16m_pb_3 : pbRow? (List.ofFn (beB t16m 3)) (t16m.bf.getD 3 []) t16m.g =
    some (t16m.pb.getD 3 []) := by decide +kernel

theorem t16m_u_0_0 : uWord? (cell t16m.c 0 0) (t16m.pa.getD 0 []) (t16m.pb.getD 0 []) =
    some (cell t16m.u 0 0) := by decide +kernel

theorem t16m_u_0_1 : uWord? (cell t16m.c 0 1) (t16m.pa.getD 0 []) (t16m.pb.getD 1 []) =
    some (cell t16m.u 0 1) := by decide +kernel

theorem t16m_u_0_2 : uWord? (cell t16m.c 0 2) (t16m.pa.getD 0 []) (t16m.pb.getD 2 []) =
    some (cell t16m.u 0 2) := by decide +kernel

theorem t16m_u_0_3 : uWord? (cell t16m.c 0 3) (t16m.pa.getD 0 []) (t16m.pb.getD 3 []) =
    some (cell t16m.u 0 3) := by decide +kernel

theorem t16m_u_1_0 : uWord? (cell t16m.c 1 0) (t16m.pa.getD 1 []) (t16m.pb.getD 0 []) =
    some (cell t16m.u 1 0) := by decide +kernel

theorem t16m_u_1_1 : uWord? (cell t16m.c 1 1) (t16m.pa.getD 1 []) (t16m.pb.getD 1 []) =
    some (cell t16m.u 1 1) := by decide +kernel

theorem t16m_u_1_2 : uWord? (cell t16m.c 1 2) (t16m.pa.getD 1 []) (t16m.pb.getD 2 []) =
    some (cell t16m.u 1 2) := by decide +kernel

theorem t16m_u_1_3 : uWord? (cell t16m.c 1 3) (t16m.pa.getD 1 []) (t16m.pb.getD 3 []) =
    some (cell t16m.u 1 3) := by decide +kernel

theorem t16m_u_2_0 : uWord? (cell t16m.c 2 0) (t16m.pa.getD 2 []) (t16m.pb.getD 0 []) =
    some (cell t16m.u 2 0) := by decide +kernel

theorem t16m_u_2_1 : uWord? (cell t16m.c 2 1) (t16m.pa.getD 2 []) (t16m.pb.getD 1 []) =
    some (cell t16m.u 2 1) := by decide +kernel

theorem t16m_u_2_2 : uWord? (cell t16m.c 2 2) (t16m.pa.getD 2 []) (t16m.pb.getD 2 []) =
    some (cell t16m.u 2 2) := by decide +kernel

theorem t16m_u_2_3 : uWord? (cell t16m.c 2 3) (t16m.pa.getD 2 []) (t16m.pb.getD 3 []) =
    some (cell t16m.u 2 3) := by decide +kernel

theorem t16m_u_3_0 : uWord? (cell t16m.c 3 0) (t16m.pa.getD 3 []) (t16m.pb.getD 0 []) =
    some (cell t16m.u 3 0) := by decide +kernel

theorem t16m_u_3_1 : uWord? (cell t16m.c 3 1) (t16m.pa.getD 3 []) (t16m.pb.getD 1 []) =
    some (cell t16m.u 3 1) := by decide +kernel

theorem t16m_u_3_2 : uWord? (cell t16m.c 3 2) (t16m.pa.getD 3 []) (t16m.pb.getD 2 []) =
    some (cell t16m.u 3 2) := by decide +kernel

theorem t16m_u_3_3 : uWord? (cell t16m.c 3 3) (t16m.pa.getD 3 []) (t16m.pb.getD 3 []) =
    some (cell t16m.u 3 3) := by decide +kernel

end Pouw.PearlC.PeelFp4Checks
