module

public import Definitions.Claims.SamplingTwo
public import Properties.SampledProofs.Draw

/-!
# Sampled proofs' second stage: it misses proof blocks no more often than the replay blocks' escapes (`Select.selectSecond`)

The property of a two-stage rule's second stage as the claims read it (`CompiledTwo.Sessions`' `secondDraws`): the draw
`Select.selectSecond` makes from `L` uniform bytes, given the first stage's replay sample, the `n` proof blocks of each
replay block and the `k` it draws from each, proves no proof block of a set `X` with probability at most the product,
over the replay sample's blocks, of a uniform `k`-subset's escape from the `n` proof blocks `[i·n, (i+1)·n)` of the
`i`-th (the claims' `escCell`). It takes no premise: the draw is a function of its inputs and its bytes, and a draw that
would read past the `L` bytes is refused, so it misses nothing. The first stage is `DrawMisses` at the two-stage rule.
-/

@[expose] public section

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

/-- The proof blocks the second stage's draw proves (`Draw.proved`), none where it is refused. -/
def secondOf : Flock.V (Json × ℕ) → Array Int
  | .ok (draw, _) => Draw.proved draw
  | .error _ => #[]

/-- **The second stage misses `X` with probability at most the replay blocks' escapes**: for every replay sample,
replay block size `n`, draw `k`, number `L` of uniform bytes and set `X` of proof blocks, the draw `Select.selectSecond`
makes from the bytes is accepted and proves no proof block of `X` with probability at most
`∏_i C(n − |X ∩ [i·n, (i+1)·n)|, k)/C(n, k)` (`escCell`), the `i`-th replay block of the sample holding
`[i·n, (i+1)·n)`. -/
def SecondMisses : Prop :=
  ∀ (sample : Array ℕ) (n k L : ℕ) (X : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Select.selectSecond sample n k (bytesOf src)).isOk = true ∧
        ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample n k (bytesOf src))) ≤
      ∏ i ∈ Finset.range sample.size,
        Claims.Sampling.escCell n k (X.filter fun j => i * n ≤ j ∧ j < (i + 1) * n).card

end Properties.SampledProofs
