import Models.Accounting.Assumptions.RemoteHeld

/-!
# The memory a remote audit shows held (trusted)

What an accepted remote audit session certifies about the node's memory, over its registered store of `N` blocks, at
each outcome of the audit's coins (a per-run reading): the probability that the session is accepted and the bound
fails is at most the premises' errors summed. These are the protocol's declared memory line (`PROTOCOL.md` §7,
"Declared outputs"; `remote_bandwidth/ledger.py`), at `X = s/E + β` blocks a tick brought back by rebuilds and the
node's disks:

* **Held**, the snapshot: the registered blocks in node memory at the start of each of the `P` credited passes,
  summed over them, are at least `Σ_p |plan_p| − W − X·(e − s₀) − P·S`. Every registered block a pass reads correctly
  was in node memory or on a chip at its start (at most `S` on chips), or came back during the pass (at most
  `X·(e − s₀)` over all of them, the passes running in order inside `[s₀, e]`); the rest are among the `W` wrong.
* **HeldOverTime**, for a paced session: the registered blocks in node memory at each tick of the credited span
  `[e_0, s_C)` of length `Σ`, summed over it, are at least `Σ·(N − S) − 2·L·X·(Σ + 2·L) − fill(W)`. A block absent
  between two of its correct reads is either brought back, ending an absence of at most `2·L` beyond the groups whose
  passes read it wrong, or read wrong in every pass of each group it is absent across, each wrong receipt of group `g`
  accounting for at most `v_g` ticks (`Pace.rate`, `Pace.fill`).
* **HeldOverTimeLinks**: the same without Identity, the node's boundary links charged at their ceiling `β_l`, at
  `X = s/E + β + β_l`.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions

/-- **The snapshot held, under Identity.** On any trace whose node re-encodes at most `s/E` blocks a tick
(`SquaringCeiling`), whose disks bring back at most `β` (`Disks`), whose chips hold at most `S` registered blocks
(`ChipCeiling`), into whose memory and chips blocks come only by those routes and its links (`Routed`), none over the
links (`Identity`), whose credited passes run in order inside `[s₀, e]` (`Causal`), whose correct receipts were computed
from blocks present during their passes except with `δs` (`SketchBound`), and whose wrong receipts are at most `W` except
with `δc` (`LiveCoins`), the probability that the session is accepted and the registered blocks in node memory at the
credited passes' starts, summed, are fewer than `Σ_p |plan_p| − W − (s/E + β)·(e − s₀) − P·S` is at most `δs + δc`. -/
def Held : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (S s₀ e : ℕ) (s E β W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β →
    Identity T → SquaringCeiling T s E → Disks T β → ChipCeiling T S → Routed T → Causal T s₀ e →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      (T.planned ω : ℝ) - W - (s / E + β) * ((e : ℝ) - s₀) - T.passes * S ≤ T.held ω

/-- **Held over time, under Identity.** On any paced trace (`Paced`, `Covers`: every credited pass reads every
registered block) meeting `Held`'s other premises, the probability that the session is accepted and the registered
blocks in node memory at each tick of the credited span `[e_0, s_C)` of length `Σ`, summed, are fewer than
`Σ·(N − S) − 2·L·(s/E + β)·(Σ + 2·L) − fill(W)` is at most `δs + δc`. -/
def HeldOverTime : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (K : Pace Ω) (S : ℕ) (s E β W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ W →
    Identity T → SquaringCeiling T s E → Disks T β → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

/-- **Held over time with the links charged** (no Identity). `HeldOverTime` with the node's boundary links bringing in
at most `βl` blocks a tick (`Links`) in place of none. -/
def HeldOverTimeLinks : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (T : Trace Ω) (K : Pace Ω) (S : ℕ) (s E β βl W δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ βl → 0 ≤ W →
    Links T βl → SquaringCeiling T s E → Disks T β → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → LiveCoins T W δc →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω))
          - K.fill ω T.blocks W
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

end Accounting.Guarantees.Remote
