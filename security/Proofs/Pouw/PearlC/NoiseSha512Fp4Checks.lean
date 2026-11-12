module

public import Definitions.Pouw.PearlC.NoiseSha512Fp4

/-!
# Checks of Pearl-C4's SHA-512 noise

`seedB_vec`, `seedA_vec`, `lineEA_vec`, `lineFA_vec`, `lineEB_vec`, `lineFB_vec`: `NoiseSha512Fp4` against
`verity/protocols/pouw/prover/schemes/pearl_c4.py` (`seed_b`, `unit_seed_a`, `line`) with salt `0, 1, …, 31`, weight root
`(7i + 3) mod 256`, activation root `(5i + 1) mod 256` and unit 5, evaluated by the kernel. The lines are one of each
role, at line indices whose block (`⌊i/4⌋`) and quarter (`i mod 4`) differ: `E_A` line 1, `F_A` line 4, `E_B` line 0 and
`F_B` line 6. `F_A`'s and `F_B`'s are `pearl_c4.basis`'s rows 4 and 6 too.
-/

@[expose] public section

namespace Pouw.PearlC.NoiseSha512Fp4Checks

open Pouw.PearlC Pouw.PearlC.NoiseSha512Fp4

/-- The salt `0, 1, …, 31`. -/
def vSalt : List ℕ := List.range 32

/-- A 64-byte weight root. -/
def vRootB : List ℕ := (List.range 64).map fun i => (7 * i + 3) % 256

/-- A 64-byte activation root. -/
def vRootA : List ℕ := (List.range 64).map fun i => (5 * i + 1) % 256

theorem seedB_vec : (seedB vSalt vRootB).take 8 = [177, 14, 205, 181, 8, 238, 89, 194] := by decide +kernel

theorem seedA_vec : (seedA vSalt vRootA vRootB 5).take 8 = [135, 26, 134, 61, 41, 42, 67, 87] := by decide +kernel

theorem lineEA_vec : lineList (seedA vSalt vRootA vRootB 5) 0 1 =
    [11, 3, 6, 7, 0, 13, 1, 7, 12, 13, 1, 13, 13, 8, 6, 15, 0, 3, 10, 10, 8, 6, 1, 12, 1, 5, 3, 11, 12, 5, 3, 4] := by
  decide +kernel

theorem lineFA_vec : lineList (seedB vSalt vRootB) 1 4 =
    [3, 13, 14, 13, 2, 12, 9, 9, 6, 2, 8, 9, 5, 1, 8, 6, 1, 9, 1, 9, 15, 11, 13, 7, 5, 0, 6, 3, 5, 15, 13, 10] := by
  decide +kernel

theorem lineEB_vec : lineList (seedB vSalt vRootB) 2 0 =
    [0, 4, 9, 12, 0, 8, 12, 4, 15, 7, 1, 5, 13, 3, 2, 11, 11, 1, 9, 3, 1, 15, 10, 14, 1, 4, 10, 13, 1, 2, 14, 4] := by
  decide +kernel

theorem lineFB_vec : lineList (seedB vSalt vRootB) 3 6 =
    [1, 10, 2, 2, 7, 15, 12, 6, 15, 7, 9, 1, 1, 6, 4, 13, 2, 4, 2, 5, 8, 12, 15, 13, 4, 9, 9, 8, 7, 2, 7, 5] := by
  decide +kernel

end Pouw.PearlC.NoiseSha512Fp4Checks
