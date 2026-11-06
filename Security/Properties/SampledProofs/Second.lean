module

public import Definitions.Claims.SamplingTwo
public import SampledProofs.Select
public import Properties.SampledProofs.Draw

/-!
# Sampled proofs' second stage: it misses proof blocks no more often than its strata's `escCell` (`Select.selectSecond`)

The property of the challenger's second-stage draw as the verifier reads it (`Draw.proved` of `Select.selectSecond`'s
draw object, the interior audit's proved units): over the replay sample `sample`, `k` of the `n` proof blocks of each
sampled replay block, the draw `selectSecond` makes from `L` uniform bytes is accepted and proves no proof block of a
set `X` with probability at most the product, over the sampled replay blocks, of the claims' per-cell escape
`Claims.Sampling.escCell n k |X ∩ ρ|`. It takes no premise: the draw is a function of its inputs and its bytes, the
`L` bytes included, however few.

* **The units** are `secondLaw`'s: the `i`-th sampled replay block (`sample[i]`) holds `[i·n, (i+1)·n)`, its proof block
  `j` the unit `i·n + j`, one stratum each.
* **The bound** is the two-stage claims' (`CompiledTwo.Sessions`' second draw) on one stratum per replay block: the
  probability that `k` units drawn uniformly from a replay block's `n` miss the `|X ∩ ρ|` of `X` in it, multiplied over
  the sampled replay blocks. A refused draw proves nothing and is not counted, as in `DrawMisses`.
-/

@[expose] public section

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

/-- The proof blocks a second-stage draw proves (`Draw.proved`), none where it is refused. -/
def secondOf : Flock.V (Json × ℕ) → Array Int
  | .ok (draw, _) => Draw.proved draw
  | .error _ => #[]

/-- **The second stage misses `X` with probability at most its strata's escape**: for every replay sample `sample`,
number `n` of proof blocks per replay block, number `k` drawn in each, number `L` of uniform bytes and set `X` of units,
the draw `Select.selectSecond` makes from the bytes is accepted and proves no unit of `X` with probability at most
`∏_i escCell n k |X ∩ [i·n, (i+1)·n)|` over the sampled replay blocks `i`. -/
def SecondMisses : Prop :=
  ∀ (sample : Array ℕ) (n k L : ℕ) (X : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Select.selectSecond sample n k (bytesOf src)).isOk = true ∧
        ∀ u ∈ X, (u : Int) ∉ secondOf (Select.selectSecond sample n k (bytesOf src))) ≤
      ∏ i ∈ Finset.range sample.size,
        Claims.Sampling.escCell n k (X ∩ Finset.Ico (i * n) ((i + 1) * n)).card

/-- A second stage that accepts everything: its draw is `null`, which proves no unit. -/
def acceptAllSecond (_ : Array ℕ) (_ _ : ℕ) (_ : ByteArray) : Flock.V (Json × ℕ) := .ok (Json.null, 0)

end Properties.SampledProofs
