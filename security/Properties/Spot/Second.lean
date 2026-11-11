module

public import Definitions.Claims.SamplingTwo
public import Properties.Spot.Draw

/-!
# The second stage misses proof blocks no more often than the cells' escapes (`Select.selectSecond`)

The property of a two-stage rule's second stage as the claims read it (`CompiledTwo.Sessions`' `secondDraws`): the sample
`Select.selectSecond` makes from `L` uniform bytes, given the first stage's replay sample, the `n_t` proof blocks of each
template `t` in a replay block and the `k_t` it samples from each such cell, proves no proof block of a set `X` with
probability at most the product, over the templates and the replay sample's blocks, of a uniform `k_t`-subset's escape
from the cell of the `i`-th block's instances of template `t`, `[off_t + i·n_t, off_t + (i+1)·n_t)` of the template-major
interior (`cellOff`; the claims' `escCell`). It takes no premise: the sample is a function of its inputs and its bytes,
and a sample that would read past the `L` bytes is refused, so it misses nothing. The first stage is `DrawMisses` at the
two-stage rule.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

/-- The proof blocks the second stage's sample proves (`Draw.proved`), none where it is refused. -/
def secondOf : Flock.V (Json × ℕ) → Array Int
  | .ok (draw, _) => Draw.proved draw
  | .error _ => #[]

/-- The first proof block of template `t`'s cells in the template-major interior over a replay sample of `q`, each
template `t'` before it holding `q · ns[t']`: `q · Σ_{t'<t} ns[t']`. -/
def cellOff (q : ℕ) (ns : Array ℕ) (t : ℕ) : ℕ := q * (ns.toList.take t).sum

/-- **The second stage misses `X` with probability at most the cells' escapes**: for every replay sample, cell sizes
`ns` and counts `ks` by template, number `L` of uniform bytes and set `X` of proof blocks, the sample
`Select.selectSecond` makes from the bytes is accepted and proves no proof block of `X` with probability at most
`∏_t ∏_i C(n_t − |X ∩ cell(i, t)|, k_t)/C(n_t, k_t)` (`escCell`), the cell of the replay sample's `i`-th block and
template `t` holding `[off_t + i·n_t, off_t + (i+1)·n_t)`, `off_t = cellOff |sample| ns t`. -/
def SecondMisses : Prop :=
  ∀ (sample ns ks : Array ℕ) (L : ℕ) (X : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Select.selectSecond sample ns ks (bytesOf src)).isOk = true ∧
        ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample ns ks (bytesOf src))) ≤
      ∏ t ∈ Finset.range ns.size, ∏ i ∈ Finset.range sample.size,
        Claims.Sampling.escCell (ns.getD t 0) (ks.getD t 0)
          (X.filter fun j => cellOff sample.size ns t + i * ns.getD t 0 ≤ j ∧
            j < cellOff sample.size ns t + (i + 1) * ns.getD t 0).card

/-- A second stage that accepts everything: its sample is `null`, which proves no proof block. -/
def acceptAllSecond (_ _ _ : Array ℕ) (_ : ByteArray) : Flock.V (Json × ℕ) := .ok (Json.null, 0)

end Properties.Spot
