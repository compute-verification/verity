import Definitions.Accounting.JointBlocks
import Proofs.Pous.Obligation

/-!
# The demo's memory share is PoUS's

The joint memory term (`Accounting.BlocksRun.MemoryTerm`, `Accounting.BlocksLive.MemoryTerm`) is stated at the share
PoUS certifies for a block's store, times the device's memory. The demo's share, `Accounting.demoMemoryShare`, is the
one `Pous.Guarantees.ObligationDemo` proves for the demo's store (`demoMemoryShare_obligationDemo`).
-/

namespace Accounting.SecurityProofs

/-- **The demo's memory share is PoUS's.** At the demo's store, `N = 1038·2^16` labels, on a device whose memory `M` is
at most `139.64·2^33` bits, the bits of `M` above the state bound `S₀ = ⌊ρ·|C|⌋` are at most `demoMemoryShare · M`:
`Pous.Guarantees.ObligationDemo` proves it at `M = 139.64·2^33`, and it holds at every smaller `M` once it holds there.
So the joint memory term at `Accounting.demoMemoryShare` is `ObligationDemo`'s,
`undeclared ≤ 0.1163·M + S_chip + Σ cap·round`. -/
theorem demoMemoryShare_obligationDemo {ι : Type} (d : Device ι) (hd : (d.mem : ℝ) ≤ 13964 / 100 * 2 ^ 33)
    (N : ℕ) [NeZero N] (hN : N = 1038 * 2 ^ 16) :
    (d.mem - (Pous.p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Pous.Params.ρ : ℝ) ≤
      demoMemoryShare * d.mem := by
  obtain ⟨hS, hU, -⟩ := Pous.SecurityProofs.ObligationDemo N hN
  rw [hS]
  unfold demoMemoryShare
  push_cast
  linarith

end Accounting.SecurityProofs
