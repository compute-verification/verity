import Definitions.NetworkCertifier

/-!
# The trusted layer: the certifier assumptions and the Program's dataflow conditions

Each assumption is a named `Prop`, taken as a hypothesis by the theorems that use it, never an `axiom`. The names follow
`docs/network-transparency/timing-channel.md` §1.2 and §2.4. `NetTiming.SecurityProofs.Sanity` gives a satisfiability
witness for each.

**The same `Prop`s serve ingress,** about the certifier read inward (`NetTiming.Protocol.Ingress`): there `Regenerates` is
W2 for what enters X, `Packed` and `Allocated` are W4 for the rows the ingress links release into X, and
`JitterBounded` is J for their release times.

**W1 (fixed parameters) has no `Prop`,** because it is structural here. The grid shape (`L`, `K`, `S`), the window
structure (`winEnd`), the frame length `B`, the jitter bound `D` and the emission map `view` are arguments fixed before
any run. The `Certifier` type has no input for the secret, and `view` reads only the step and the observation. So nothing
in the model can make them depend on the secret or on traffic. A reviewer checks this by reading `NetTiming.Protocol`.
-/

namespace NetTiming

variable {L K S : ℕ} {In Act Wire : Type}

/-- **W2, regeneration and re-timing** (with W1's fixed emission times). Everything the certifiers put on the wire at step
`k` is a fixed function `view k` of what they record: the egress rows, the jitter class and the statuses. No physical
feature, ack, backpressure or other traffic passes except through the observation. This holds for every action,
faults included (part of W6). -/
def Regenerates (w : Certifier L K S In Act Wire) (view : ℕ → Obs L K S → Wire) : Prop :=
  ∀ k xs as, w.wire k xs as = view k (w.obs k xs as)

/-- **W3, padding.** Every frame the certifiers release is exactly `B` bits. (The accepted-run theorems do not use it;
it bounds the unbound capacity, which is not formalized here.) -/
def Padded (w : Certifier L K S In Act Wire) (B : ℕ) : Prop :=
  ∀ k xs as ℓ s f, (w.obs k xs as).rows ℓ s = some f → f.length = B

/-- **W4, packing (left-packed).** In every recorded row the occupied slots form a prefix. -/
def Packed (w : Certifier L K S In Act Wire) : Prop :=
  ∀ k xs as ℓ, RowLeftPacked fun s => ((w.obs k xs as).rows ℓ s).isSome

/-- **W4, allocation.** Link `ℓ` releases at most `r ℓ` real frames per bucket: under constant-rate release a bucket
carries `min(eligible queue, r ℓ)` real frames, and padding in the rest of its `r ℓ` slots. `Obs` gives every link the
same `S` slots, so a link whose allocation is below `S` always leaves its last `S − r ℓ` slots empty; `r ℓ ≥ S` says
nothing. -/
def Allocated (w : Certifier L K S In Act Wire) (r : Fin L → ℕ) : Prop :=
  ∀ k xs as ℓ, (rowCount ((w.obs k xs as).rows ℓ) : ℕ) ≤ r ℓ

/-- **W5, one untimed status per sampling block.** Statuses are published only at the last step of each window
(`winEnd k = k`). At every other step they are the constant `complete`, so a status never shows when within the
window something happened. -/
def UntimedStatus (w : Certifier L K S In Act Wire) (winEnd : ℕ → ℕ) : Prop :=
  ∀ k xs as, winEnd k ≠ k → ∀ κ, (w.obs k xs as).status κ = Status.complete

/-- **W6, fail-closed.** Suppose a fault happens at step `i`, whose window ends at `winEnd i`. Then:
1. every egress row the certifiers release after it, within the window, is empty (all padding);
2. at the window's end some boundary link's status is not complete.

That emissions stay a function of the observation after a fault is `Regenerates`, which quantifies over every action.
The histories `xs` and `as` are those of step `k`, with `as[i]` the fault. -/
def FailClosed (w : Certifier L K S In Act Wire) (winEnd : ℕ → ℕ) : Prop :=
  ∀ i k xs as (hi : i < as.length), as.length = k + 1 → i ≤ k → k ≤ winEnd i → w.fault as[i] →
    (i < k → ∀ ℓ s, (w.obs k xs as).rows ℓ s = none) ∧
    (k = winEnd i → ∃ κ, (w.obs k xs as).status κ ≠ Status.complete)

/-- **J, bounded jitter.** The adversary-influenced timing offset of a step takes one of `D` classes that any receiver
can tell apart. (`D = 1` is ideal bucketing; per step, `D` stands for the doc's `D^R` over its `R` emissions.) -/
def JitterBounded (w : Certifier L K S In Act Wire) (D : ℕ) : Prop :=
  ∀ k xs as, (w.obs k xs as).jitter < D

section Dataflow

variable {Ac B C : Type} {At : B → Type} {U : C → Type}

/-- **Theorem 3, condition 1: frames read no timing advice.** Take the advice as content advice `a.1` and timing
advice `a.2 β` per block `β`, and cells `c` (a link and a step) with the occupancy `occ c` that the Program predicts
there. The Program's rows at step `t` then read timing advice only through the occupancy it predicts at `t`: two
advice strings with the same content advice and the same occupancy at every cell of step `t` give the same rows.
Timing-driven batching that changes content is content advice. -/
def FramesReadNoTimingAdvice (P : Program L S In (Ac × ((β : B) → At β))) (step : C → ℕ)
    (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c) : Prop :=
  ∀ t ins (a a' : Ac × ((β : B) → At β)), a.1 = a'.1 →
    (∀ c, step c = t → occ c ins a = occ c ins a') → P.rows t ins a = P.rows t ins a'

/-- **Theorem 3, condition 2: occupancy reads only its own block.** The occupancy predicted at a cell reads, of the
timing advice, only its block's `a.2 (blk c)` (and the content advice and the ingress). -/
def OccupancyReadsOwnBlock (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c) : Prop :=
  ∀ c ins (a a' : Ac × ((β : B) → At β)), a.1 = a'.1 → a.2 (blk c) = a'.2 (blk c) → occ c ins a = occ c ins a'

end Dataflow

end NetTiming
