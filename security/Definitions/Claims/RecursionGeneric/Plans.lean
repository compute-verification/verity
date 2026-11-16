module

public import Definitions.Claims.RecursionGeneric.Plan

/-!
# Run finders every recursion reads

Plans that need no inner protocol:

* `forkPlan k`: `B`'s plan, guess a round below `k(x)` and search a run and its fork there. At the composite's reruns it is
  `B_I` (`Claims.Recursion.BinnerV`'s finder).
* `linkPlan k M`: the deviation finder's plan, gated: guess a round `j`, ask whether a rerun forked at `j` from a shared
  prefix gives; only if it does, ask up to `M` more forks at `j` and search the first and the first of them that gives.
  At the composite's reruns it is `Claims.Recursion.FdevV`'s finder, which is gated by the first rerun in the same way
  (`linkOutV`, `linkCost`).

Flock's terms are plans too, which only its instance reads (`Proofs.Flock.RecGeneric.Plans`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims

variable {Coin : Type}

/-- **`B`'s plan** at `k` rounds: guess a round `j < k(x)`, and search the run `u₀` and its fork at `j` with `u₁`. -/
def forkPlan (k : Msg → ℕ) : RunFinder Coin where
  sel := k
  slots _ := 1
  plan _ j := .done [.fresh 0, .fork j.val (.fresh 0) (.fresh 1)]

/-- The link's loop: up to `n` forks at `j` of the shared prefix `u₀`, from the fresh run `i` on; the first that is
accepted is searched with `first`, and `first` alone where none is. -/
def linkLoop (j : ℕ) (first : RunExp) : ℕ → ℕ → Plan Coin
  | 0, _ => .done [first]
  | n + 1, i => .ask (.fork j (.fresh 0) (.fresh i)) fun b =>
    if b then .done [first, .fork j (.fresh 0) (.fresh i)] else linkLoop j first n (i + 1)

/-- **The link's plan** at `k` rounds and `M` tries: guess `j < k(x)`; ask whether `fork_j(u₀, u₁)` is accepted; if
it is, the loop over `fork_j(u₀, u₂), …, fork_j(u₀, u_{M+1})`; if it isn't, search nothing. -/
def linkPlan (k : Msg → ℕ) (M : ℕ) : RunFinder Coin where
  sel := k
  slots _ := M + 1
  plan _ j := .ask (.fork j.val (.fresh 0) (.fresh 1)) fun b =>
    if b then linkLoop j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2 else .done []

end Claims.RecGeneric
