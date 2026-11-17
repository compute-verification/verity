module

public import Definitions.Poub.Assumptions.RemoteHeld

/-! The memory an accepted PoUB session shows held (`Held`, `HeldOverTime`, `HeldOverTimeLinks`), for a node that moves
whole blocks. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions

/-- **The snapshot held, under Identity.** On any trace whose node re-encodes at most `s/E` blocks a tick and one batch
of `Br` (`SquaringCeiling`), whose disks bring back at most `β` a tick and one read of `Bd` (`Disks`), whose chips hold
at most `S` registered blocks (`ChipCeiling`), into whose memory and chips blocks come only by those routes and its
links (`Routed`), none over the links (`Identity`), whose credited passes run in order inside `[s₀, e]` (`Causal`),
whose correct receipts were computed from blocks present during their passes except with `δs` (`SketchBound`), and
whose wrong receipts are at most `W` except with `δc` (`LiveCoins`), the probability that the session is accepted and
the registered blocks in node memory at the credited passes' starts, summed, are fewer than
`Σ_p |plan_p| − W − ((s/E + β)·(e − s₀) + (Br + Bd)) − P·S` is at most `δs + δc`. -/
def Held : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (S s₀ e : ℕ) (s E β Br Bd W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Causal T s₀ e →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      (T.planned ω : ℝ) - W - ((s / E + β) * ((e : ℝ) - s₀) + (Br + Bd)) - T.passes * S ≤ T.held ω

/-- **Held over time, under Identity.** On any paced trace (`Paced`, `Covers`: every credited pass reads every
registered block) meeting `Held`'s other premises, the probability that the session is accepted and the registered
blocks in node memory at each tick of the credited span `[e_0, s_C)` of length `Σ`, summed, are fewer than
`Σ·(N − S) − 2·L·((s/E + β)·(Σ + 2·L) + (Br + Bd)) − fill(W)` is at most `δs + δc`. -/
def HeldOverTime : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (K : Pace Ω) (S : ℕ) (s E β Br Bd W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ W →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

/-- **Held over time with the links charged** (no Identity). `HeldOverTime` with the node's boundary links bringing in
at most `βl` blocks a tick and one window of `Bl` (`Links`) in place of none. -/
def HeldOverTimeLinks : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (K : Pace Ω) (S : ℕ) (s E β βl Br Bd Bl W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ βl → 0 ≤ W →
    Links T βl Bl → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

end ResourceAccounting.Guarantees.Remote
