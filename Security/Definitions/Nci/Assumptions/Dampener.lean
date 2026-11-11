module

public import Definitions.Nci.Dampener
public import Definitions.Nci.Threads

/-!
# The compute dampener: the model's accounting, named

The dampener's statements (`Nci.Guarantees.DampenerThreaded`, `DampenerRemote`, `DampenerRemotePeriod`) take the
model's accounting of where counted multiplications run, and the shape of the servers' windows, as premises over
numbers. These name them. None is about the world: each is how the model counts.
-/

@[expose] public section

namespace Nci.Assumptions

open Verity

/-- **A window is `L` ticks at the rate**: at `α F` bits a tick, a window of `L` ticks takes in `R` bits, as much as the
state a wipe leaves. -/
def WindowLength (α F L R : ℝ) : Prop :=
  α * F * L = R

/-- **The servers' windows hold the residual and the declared computation**: summed over `n` servers' `⌈H / L⌉` windows,
each reading at most `K = R + α F L` bits and holding at most `K Λ(K) / b_a` counted multiplications of `T`'s bound,
`b_a (M_r + |M|) ≤ n ⌈H / L⌉ K Λ(K)`. -/
def WindowsHold {r : RegisteredCircuit} (T : Threading r) (Mr n : ℕ) (α F H L R b_w b_a : ℝ) (dd e : ℕ) : Prop :=
  b_a * ((Mr : ℝ) + T.M.card) ≤
    n * ⌈H / L⌉₊ * ((R + α * F * L) * T.bound (R + α * F * L) b_w dd e)

/-- **The parts fit in the monitored compute**: `∑ x_i ≤ C`, for PoUW's windows' capacities. -/
def WithinCompute {ι : Type} [Fintype ι] (x : ι → ℝ) (C : ℝ) : Prop :=
  ∑ i, x i ≤ C

/-- **Every counted multiplication has a source**: `M ≤ M_U + M_r + M_d + M_χ`, unmonitored compute, the monitored
residual, the declared computation and incorrect proof blocks. -/
def SourcesCover (M Mu Mr Md Mx : ℕ) : Prop :=
  M ≤ Mu + Mr + Md + Mx

/-- **Unmonitored compute trains like the unconstrained developer**, at model-FLOP utilization `MFU₀`:
`M_U ≤ MFU₀ κ U / 2`. -/
def UnmonitoredAtMost (Mu : ℕ) (mfu κ U : ℝ) : Prop :=
  (Mu : ℝ) ≤ mfu * κ * U / 2

/-- **The residual trains at full efficiency outside the work certified correctly and run**: at least `(1 − ε_W) W_cert`
of the monitored compute `C` is certified correctly, less the wrong windows' capacity `wc` and the skipped work, so
`M_r ≤ κ (C − (1 − ε_W) W_cert + wc + skipped) / 2`. -/
def ResidualAtMost (Mr : ℕ) (κ C εW W wc skipped : ℝ) : Prop :=
  (Mr : ℝ) ≤ κ * (C - (1 - εW) * W + wc + skipped) / 2

/-- **Incorrect proof blocks inject at most the cheating share** `χ`: `M_χ ≤ χ κ (C + U) / 2`. -/
def CheatingAtMost (Mx : ℕ) (κ χ C U : ℝ) : Prop :=
  (Mx : ℝ) ≤ κ * χ * (C + U) / 2

end Nci.Assumptions
