import Definitions.Accounting.JointBlocks
import Proofs.Pous.Obligation

/-!
# The demo's memory share is PoUS's

The joint memory term (`Accounting.BlocksRun.MemoryTerm`, `Accounting.BlocksLive.MemoryTerm`) is stated at the share
PoUS certifies for a block's store. The demo's share, `Accounting.demoMemoryShare`, is the one
`Pous.Guarantees.ObligationDemo` proves for the demo's store (`demoMemoryShare_obligationDemo`).
-/

namespace Accounting.SecurityProofs

/-- **The demo's memory share is PoUS's.** At the demo's store, `N = 1038·2^16` labels on a device of
`U = 139.64·2^33` bits, the bits of `U` above the state bound `S₀ = ⌊ρ·|C|⌋` are at most `demoMemoryShare · U`, the
bound `Pous.Guarantees.ObligationDemo` proves. So the joint memory term at `Accounting.demoMemoryShare` is
`ObligationDemo`'s, `undeclared ≤ 0.1163·U + S_chip + Σ cap·round`. -/
theorem demoMemoryShare_obligationDemo (N : ℕ) [NeZero N] (hN : N = 1038 * 2 ^ 16) :
    (13964 / 100 * 2 ^ 33 - (Pous.p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ : ℝ) ≤
      demoMemoryShare * (13964 / 100 * 2 ^ 33) := by
  obtain ⟨hS, hU, -⟩ := Pous.SecurityProofs.ObligationDemo N hN
  rw [hS]
  exact_mod_cast hU

end Accounting.SecurityProofs
