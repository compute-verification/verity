import Definitions.Pouw.Lead.Model
import Definitions.Pouw.Assumptions.Dimension.H100W
import Proofs.Pouw.Dimension.H100WProofs

/-!
# The register floor of W1's priced work

Under `H32`, a well-formed `MH100w` trace with no free operand costs at least 32 a register it writes: the registers it
writes are those the distinct-writes proof splits into `hiP` and `loP` (`regCount_hi_lo`), and `cost_hi_lo` prices
both at 32 under `H32`, which is `FloorTc 32 32`.
-/

namespace Pouw.SecurityProofs.Lead.MH100W

open Pouw.Lead Pouw.Dimension.H100

/-- A trace's registers are those it writes at `a` or more a register and those it writes below. -/
theorem regCount_hi_lo {G : Type} (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) :
    ∀ (P : List (Op100 G)) (Rs : List (G → ℚ)),
      regCount P = (hiP a fpSem P Rs).length + (loP a fpSem P Rs).length
  | [], _ => by simp [regCount, hiP, loP]
  | o :: P, Rs => by
    have ih := regCount_hi_lo a fpSem P (Rs ++ o.outs fpSem Rs)
    have ho : Pouw.Lead.Op100.regs o = (o.hi a fpSem Rs).length + (o.lo a).length := by
      cases o with
      | mma outs p => by_cases hp : a * outs.length ≤ p <;> simp [Pouw.Lead.Op100.regs, Op100.hi, Op100.lo, hp]
      | _ => simp [Pouw.Lead.Op100.regs, Op100.hi, Op100.lo]
    simp only [regCount, List.map_cons, List.sum_cons, hiP, loP, List.length_append] at ih ⊢
    omega

/-- **The register floor**: under `H32`, every instruction of a trace well formed with no free operand
costs at least 32 a register it writes (`Op100.regs`), so the trace's priced W1 work is at least `32 · regCount`. Reads
`Op100.price`, `Op100.WFw` and `Floor`. -/
theorem cost_ge_regs {G : Type} (pr : Prices100) (hpr : H32 pr) (P : List (Op100 G))
    (hwf : ∀ o ∈ P, o.WFw pr ∅) : 32 * regCount P ≤ cost100 pr P := by
  have h := cost_hi_lo 32 32 pr hpr ∅ (fun _ _ _ _ => 0) P [] hwf
  rw [regCount_hi_lo 32 (fun _ _ _ _ => 0) P []]
  omega

end Pouw.SecurityProofs.Lead.MH100W
