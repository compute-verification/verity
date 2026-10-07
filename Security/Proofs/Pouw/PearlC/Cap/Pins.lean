import Proofs.Pouw.PearlC.Cap.IntForm

/-!
# `_cap_constants` at pinned shapes

`capConstants` evaluated by the kernel at the circuit-check binding's shapes and at deployed 64 × 64 tiles, each on the
device's own `g`, prices and cap; one shape `_cap_constants` refuses; and `creditInt` where `credit_of` is called.
`verity/protocols/accounting/work/pouw/tests/test_cap_lean.py` reads these statements and checks each against `cap._cap_constants` and
`PearlC(forming="v1").credit_of`.
-/

namespace Pouw.PearlC.Cap

/-- `_cap_constants(k, tm, tn, n, rho, dev)` as its tuple: `(atom, zero, n/g, elem/g, l_max, table)`. -/
def capPin (p : IntPrices) (g k tm tn n rho : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × List ℕ :=
  let c := capConstants p g k tm tn n rho
  (c.atom, c.zero, c.nG, c.elemG, c.lMax, (List.range (tm + 1)).map c.table)

/-- The same with the table's last entry `table[tm]` alone. -/
def capPinTop (p : IntPrices) (g k tm tn n rho : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
  let c := capConstants p g k tm tn n rho
  (c.atom, c.zero, c.nG, c.elemG, c.lMax, c.table tm)

theorem pin_sm120_k256 : CapDomain 4 256 1 1 1 1000 ∧ CapFits IntPrices.sm120 4 256 1 1 1 1000 ∧
    capPin IntPrices.sm120 4 256 1 1 1 1000 = (136, 32, 1, 160, 75, [0, 75]) := by decide

theorem pin_sm120_k512 : CapDomain 4 512 2 2 4 1000 ∧ CapFits IntPrices.sm120 4 512 2 2 4 1000 ∧
    capPin IntPrices.sm120 4 512 2 2 4 1000 = (136, 32, 1, 80, 158, [0, 79, 158]) := by decide

theorem pin_h100_k384 : CapDomain 4 384 1 1 1 400 ∧ CapFits IntPrices.h100 4 384 1 1 1 400 ∧
    capPin IntPrices.h100 4 384 1 1 1 400 = (160, 128, 1, 256, 374, [0, 374]) := by decide

theorem pin_sm120_k4096 : CapDomain 4 4096 64 64 4096 1000 ∧ CapFits IntPrices.sm120 4 4096 64 64 4096 1000 ∧
    capPin IntPrices.sm120 4 4096 64 64 4096 1000 = (136, 32, 2, 5, 74448,
      [0, 2326, 4653, 6979, 9306, 11632, 13959, 16285, 18612, 20938, 23265, 25591, 27918, 30244, 32571, 34897, 37224,
       39550, 41877, 44204, 46530, 48857, 51183, 53510, 55836, 58163, 60489, 62816, 65142, 67469, 69795, 72122, 74448,
       76775, 79101, 81428, 83755, 86081, 88408, 90734, 93061, 95387, 97714, 100040, 102367, 104693, 107020, 109346,
       111673, 113999, 116326, 118652, 120979, 123305, 125632, 127959, 130285, 132612, 134938, 137265, 139591, 141918,
       144244, 146571, 148897]) := by decide

theorem pin_sm120_k14336 : CapDomain 4 14336 64 64 4096 1000 ∧ CapFits IntPrices.sm120 4 14336 64 64 4096 1000 ∧
    capPinTop IntPrices.sm120 4 14336 64 64 4096 1000 = (136, 32, 2, 5, 255655, 511311) := by decide

theorem pin_sm120_k8192_n16384 : CapDomain 4 8192 64 64 16384 1000 ∧ CapFits IntPrices.sm120 4 8192 64 64 16384 1000 ∧
    capPinTop IntPrices.sm120 4 8192 64 64 16384 1000 = (136, 32, 8, 5, 145162, 1161297) := by decide

theorem pin_h100_k4096 : CapDomain 4 4096 64 64 6144 400 ∧ CapFits IntPrices.h100 4 4096 64 64 6144 400 ∧
    capPinTop IntPrices.h100 4 4096 64 64 6144 400 = (160, 128, 3, 8, 216268, 648806) := by decide

theorem refuse_sm120_k65536 : CapDomain 4 65536 128 128 4096 1000 ∧
    ¬ CapFits IntPrices.sm120 4 65536 128 128 4096 1000 := by decide

theorem credit_sm120_k256_n1 : creditInt IntPrices.sm120 4 1 256 1 = 18824 := by decide
theorem credit_sm120_k512_n4 : creditInt IntPrices.sm120 4 1 512 4 = 39520 := by decide
theorem credit_h100_k384_n1 : creditInt IntPrices.h100 4 1 384 1 = 37440 := by decide
theorem credit_sm120_k4096_n4096 : creditInt IntPrices.sm120 4 1 4096 4096 = 18612224 := by decide
theorem credit_sm120_k14336_n4096 : creditInt IntPrices.sm120 4 1 14336 4096 = 63913984 := by decide
theorem credit_sm120_k8192_n16384 : creditInt IntPrices.sm120 4 1 8192 16384 = 145162240 := by decide
theorem credit_h100_k4096_n6144 : creditInt IntPrices.h100 4 1 4096 6144 = 32440320 := by decide

end Pouw.PearlC.Cap
