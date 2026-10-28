import Definitions.Accounting.TS.J10R

/-!
# Checks of J10-R's noise against the Python

- `unit_row_is_call_row`, `unit_col_is_call_col`, `j10r_rows`, `j10r_cols`: at `unitBlocks`' bases a unit's row `i` and
  column `j` are the call-scale layout's row `R I + i` and column `C J + j`, for every seed, call, tile and `K`.
- `callSeed_vec` … `keys_vec`: `J10R` against `verity/ml/protocols/pouw/circuit/seeds.py` and `h1tile.py` (`call_seed`,
  `receipt_seed`, `seed`, `noise_block`, `j10r_blocks`, `j10r_noise`, `fold_keys`) on unit (1, 2) of an 8 × 32 call at
  `K = 64`, the call at position 40 and the unit at 46, salt `0, 1, …, 63` and roots `(7i + 3) mod 256` (A) and
  `(5i + 1) mod 256` (B), evaluated by the kernel. `test_j10r_lean_vectors.py` recomputes each.
- `layout_4096` … `prefill_k14336`: RB and CB, today's 612 compressions a unit at a warp and `K = 4096`, and J10-R's over a
  call at decode (32 × 4096: 41,506 over 128 units, 324.27 a unit) and prefill.
-/

namespace Accounting.TS.J10RChecks

open Accounting.TS.J10R

theorem unit_row_is_call_row (s : List ℕ) (M N k R C I J i : ℕ) :
    rowAt s k ((unitBlocks M k R C I J).1 + i * rb k) = rowAt s k ((Layout.mk M N k).row (R * I + i)) := by
  simp only [unitBlocks, Layout.row, Nat.add_mul]

theorem unit_col_is_call_col (s : List ℕ) (M N k R C I J j : ℕ) :
    colAt s k ((unitBlocks M k R C I J).2 + j * cb k) = colAt s k ((Layout.mk M N k).col (C * J + j)) := by
  simp only [unitBlocks, Layout.col, Nat.add_mul, Nat.add_assoc]

/-- A unit's rows under J10-R are the call-scale rows `R I .. R I + R - 1`. -/
theorem j10r_rows (s : List ℕ) (M N k R C I J : ℕ) :
    (j10rNoise s ⟨R, C, k⟩ (unitBlocks M k R C I J)).1 =
      (List.range R).map fun i => rowAt s k ((Layout.mk M N k).row (R * I + i)) := by
  unfold j10rNoise
  exact List.map_congr_left fun i _ => unit_row_is_call_row s M N k R C I J i

/-- A unit's columns under J10-R are the call-scale columns `C J .. C J + C - 1`. -/
theorem j10r_cols (s : List ℕ) (M N k R C I J : ℕ) :
    (j10rNoise s ⟨R, C, k⟩ (unitBlocks M k R C I J)).2 =
      (List.range C).map fun j => colAt s k ((Layout.mk M N k).col (C * J + j)) := by
  unfold j10rNoise
  exact List.map_congr_left fun j _ => unit_col_is_call_col s M N k R C I J j

/-! ## Against `seeds.py` and `h1tile.py` -/

/-- The vectors' salt. -/
def vSalt : List ℕ := List.range 64

/-- The vectors' A root. -/
def vRootA : List ℕ := (List.range 64).map fun i => (7 * i + 3) % 256

/-- The vectors' B root. -/
def vRootB : List ℕ := (List.range 64).map fun i => (5 * i + 1) % 256

theorem callSeed_vec : (callSeed vSalt 40 vRootA vRootB).take 8 = [180, 42, 172, 231, 58, 40, 21, 223] := by decide +kernel

theorem receiptSeed_vec : (receiptSeed vSalt 46 vRootA vRootB).take 8 = [67, 165, 128, 61, 208, 199, 201, 16] := by decide +kernel

theorem seed_vec : (seed vSalt 46 vRootA vRootB).take 8 = [153, 133, 229, 243, 77, 209, 250, 249] := by decide +kernel

theorem noiseBlock_vec : (noiseBlock (callSeed vSalt 40 vRootA vRootB) 24).take 8 = [230, 11, 151, 120, 76, 147, 166, 149] := by decide +kernel

theorem unitBlocks_vec : unitBlocks 8 64 4 8 1 2 = (4, 24) := by decide +kernel

theorem row_vec : ((unitNoise vSalt vRootA vRootB 40 ⟨4, 8, 64⟩ (4, 24)).1.take 1).map (fun r => (value (r.1.take 32), r.2.map value)) = [(3463785772, [985759600])] := by decide +kernel

theorem col_vec : (((unitNoise vSalt vRootA vRootB 40 ⟨4, 8, 64⟩ (4, 24)).2.drop 3).take 1).map (fun c => (value (c.1.take 32), c.2.take 8)) = [(3693213991, [0, 34, 239, 245, 217, 245, 227, 42])] := by decide +kernel

theorem keys_vec : (unitKeys vSalt vRootA vRootB 46 ⟨4, 8, 64⟩).map (·.take 4) = [[806148450, 2504443776, 561003295, 1859415271]] := by decide +kernel

/-! ## Compressions -/

theorem layout_4096 : rb 4096 = 9 ∧ cb 4096 = 9 ∧ (Layout.mk 32 32 4096).blocks = 576 := by decide +kernel

theorem today_4096 : unitCompressions ⟨32, 32, 4096⟩ = 612 := by decide +kernel

theorem decode_4096 : callCompressionsNum 32 4096 4096 32 32 = 41506 ∧ units 32 4096 32 32 = 128 := by decide +kernel

theorem prefill_4096 : callCompressionsNum 256 4096 4096 32 32 = 73986 ∧ units 256 4096 32 32 = 1024 := by decide +kernel

theorem prefill_k14336 : rb 14336 = 29 ∧ cb 14336 = 29 ∧ unitCompressions ⟨32, 32, 14336⟩ = 1892 ∧
    callCompressionsNum 256 4096 14336 32 32 = 161026 := by decide +kernel

end Accounting.TS.J10RChecks
