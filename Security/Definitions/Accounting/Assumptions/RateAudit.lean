import Definitions.Accounting.RateAuditPartial

/-!
# The rate audit's named assumptions (trusted)

What the rate audit with partial completion (`Definitions/Accounting/RateAuditPartial.lean`) assumes, each a named
`Prop`, taken as a hypothesis by the guarantees that use it, never an `axiom`:

* `WorkHardness b rounds c_root cost`: completing `x` of a word's `b` bits costs at least `x/b` of a word's price
  `rounds · c_root` (claim `rate-codec-work`, `WorkHardness(p, k)` in partial-completion form, at `k = rounds`). The
  price is a parameter. The remote audit's route `Accounting.Remote.Run.Rebuild` (`rebuilt · c_word ≤ b · rebuildWork`)
  is what it gives a run (`Accounting.Guarantees.RatePartialRebuild`).
* `RateComputeAdditive`: a joint run's compute account and its bandwidth game share no work (the actual-circuit
  model's compute additivity, W), the remote audit's `Accounting.Remote.Assumptions.ComputeAdditive` for the device
  model's joint run.
-/

namespace Accounting.Assumptions

open Accounting Accounting.Rate

/-- **`rate-codec-work`: `WorkHardness(p, k)` in partial-completion form, at `k = rounds` rounds of a root priced
`c_root`.** For every `x ≤ b`, completing `x` of a word's `b` bits costs at least `x/b` of a word's price
`c_word = rounds · c_root`:

  `x · (rounds · c_root) ≤ b · cost x`,

where `cost x` is the least compute, in the game's unit, in which the stated device obtains `x` bits of a word of the
store (its bottom `x` bits, or any other `x`-bit function of it) that it neither holds nor reads, from setup, the
payload the word encodes and anything else it holds, the word's other `b − x` bits among them, one word at a time (no
amortization across words). It is what `RatePartial`'s games (`Definitions/Accounting/RateAuditPartial.lean`) charge for
a completion of `x` bits. At `x = b` it says a whole rebuild costs at least `c_word`, `RateAudit`'s `c_enc`.

**The price is a parameter.** For Sloth over `p` at `k` rounds, decoding is `k` squarings a word and encoding `k`
modular square roots, about `(b − 2)·k` squarings, so a word's price is `k·c_root`, and the price of `x` bits is the red
team's `(k·x/b)·c_root`: at `p = 2^512 − 569` and `k = 2`, `WorkHardness 512 2 c_root cost` says completing `x` of 512
bits costs at least `(2x/b)·c_root`. A completion measured below that, by a factor `θ < 1`, is this `Prop` at
`θ·c_root` (in a unit fine enough to make it a whole number), and every guarantee taking it holds there, its rebuild
term divided by `θ`.

**The stated instance is the deployed codec's**, Sloth over `p = 2^1024 − 105`, masked, at `k = 3` (`codec.DEFAULT`
of `verity/core/protocols/remote_bandwidth/codec.py`, as ruled; `k = 2` at the same width the
fallback): `WorkHardness 1024 3 c_root cost`, `x` of a word's 1,024 bits at `(3x/b)·c_root`. The `Prop` reads the codec
only through `b`, `rounds` and `c_root`, so the mask changes no statement, only what the price rests on. The
guarantees hold at every `rounds`; `k = 3` is the one they are read at.

**Where it stands.** At `k = 2` the adversary keeps a word's top `b − x` bits and completes the rest from the known
payload by a 4-dimensional lattice, up to `x ≈ b/10` at every width, so the claim at `x = b/10 − 8` is 0.168 of a root
at 512 bits. fplll, the best of five reducers, costs 2.07 roots there, 12.3× the claim; Howgrave-Graham's larger
lattices are 2–5× worse per bit. At the reduction's operation-count floor (estimated from operation counts, not
measured) the margin is about 1.2× on the GPU (range 0.75–2.7×) and 1.6× on the host (0.99–3.6×): on that estimate 512
bits is the widest word unmasked `k = 2` protects, 640 bits is at parity and 768 is broken. For the upgrade path,
`p = 2^512 − 569` unmasked at `k = 2`, the residual conjecture is `WorkHardness 512 2 c_root cost`: no 4D reduction at
512 bits completes a dropped tail below `2x/b` of a root.

**The masked codec**, the deployed one. A public XOR `σ_K` between rounds defeats the 4D completion (none of 200 masked
words completed, against all 200 plain ones, at every width), and the price's justification becomes that no completion
skips the inner root. Its residual conjecture is the stated instance, `WorkHardness 1024 3 c_root cost`.

**At `k = 1` it is false at every width**. A word's dropped bottom third completes by a 2-dimensional lattice
whose floor is half a GMP extended GCD: 0.025 of a root at 512 bits, 0.019 at 640, 0.015 at 768, 0.009 at 1,024,
0.005 at 1,536 and 0.003 at 2,048, against the per-bit 0.32–0.33, a break of 13–110×. `WorkHardness b 1 c_root cost` is
outside the claim: a guarantee taking it says nothing about a deployed one-round codec, and at one round the claim
holds only for words whose payload is unknown, under `Incompressible(W)`, which no guarantee here takes. -/
def WorkHardness (b rounds cRoot : ℕ) (cost : ℕ → ℕ) : Prop :=
  ∀ x, x ≤ b → x * (rounds * cRoot) ≤ b * cost x

/-- **W, compute additivity, for a joint run's bandwidth game** (the actual-circuit model's obligation W:
compute adds by a direct-sum conjecture). At every
value of the audits' coins, the compute account's work over the compute window and the compute of the run's bandwidth
game (its tags at `c_tag` and its completions at `cost x` for `x` bits), which falls in the compute window, together
fit in the device's priced work over that window:

  `work + game compute ≤ F·T`.

So what the compute audit certifies is not available to complete words, and the adversary's idle compute, `F·T − W`
when the compute account did `W`, is all a window's completions get. In the device model it holds whenever the game's
ops are charged to the accounts other than the compute account (`Step.Fits` charges each op once), which is
`RealizesRate`'s compute field; on the actual circuit, whose gates carry no account, it is the conjecture that PoUW's
work and the completions share no gate. -/
def RateComputeAdditive (p : Price) (cTag : ℕ) (cost : ℕ → ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS} (real : RealizesRatePartial r) : Prop :=
  ∀ x y s', r.work p (x, y, s') + (real.dev (x, s')).compute cTag cost y.1 y.2.1.1 y.2.1.2 ≤
    d.priced p * (r.win .compute).len

end Accounting.Assumptions
