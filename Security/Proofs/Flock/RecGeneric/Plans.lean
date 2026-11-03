module

public import Proofs.Flock.RecGeneric.Plan

/-!
# Run finders every recursion reads, and Flock's

Plans that need no inner protocol:

* `forkPlan k`: `B`'s plan, guess a round below `k(x)` and search a run and its fork there. At the composite's reruns it is
  `B_I` (`Claims.Recursion.BinnerV`'s finder).
* `linkPlan k M`: the deviation finder's plan, gated: guess a round `j`, ask whether a rerun forked at `j` from a shared
  prefix gives; only if it does, ask up to `M` more forks at `j` and search the first and the first of them that gives.
  At the composite's reruns it is `Claims.Recursion.FdevV`'s finder with its search gated by the first rerun (`FdevV`
  searches the loop even where the first rerun gives nothing, which its declared cost doesn't charge).

Flock's terms are plans too, at any caps (`Claims.Recursion.Caps`): `capPlan Cp M` (`FC`) and `conflictPlan k R` (`F0`).

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

/-- The caps' loop at cap position `q` of round `j`: up to `n` forks at `j` of `u₀`, from the fresh run `i` on; the
first that is accepted and whose coins query `q` (`opens`) is searched with `u₀`, and `u₀` twice where none is. -/
def capLoop (j : ℕ) (opens : (ℕ → Coin) → Bool) : ℕ → ℕ → Plan Coin
  | 0, _ => .done [.fresh 0, .fresh 0]
  | n + 1, i => .ask (.fork j (.fresh 0) (.fresh i)) fun b =>
    if b then .look (.fork j (.fresh 0) (.fresh i)) fun cs =>
      if opens cs then .done [.fresh 0, .fork j (.fresh 0) (.fresh i)] else capLoop j opens n (i + 1)
    else capLoop j opens n (i + 1)

/-- **`FC`'s plan** at the caps `Cp` and `M` tries: a cap position `q`; if `u₀` is accepted and its coins query `q`,
the caps' loop over `M` forks at `q`'s round; otherwise search nothing. -/
def capPlan (Cp : Claims.Recursion.Caps Coin) (M : ℕ) : RunFinder Coin where
  sel := Cp.width
  slots _ := M
  plan x q := .ask (.fresh 0) fun b =>
    if b then .look (.fresh 0) fun cs =>
      if Cp.opens x q cs then capLoop (Cp.round x q) (Cp.opens x q) M 1 else .done []
    else .done []

/-- One wait of at most `n` fresh runs from the run `i` on, stopping at its first accepted one, then `cont` with it. -/
def waitLoop (cont : Option RunExp → Plan Coin) : ℕ → ℕ → Plan Coin
  | 0, _ => cont none
  | n + 1, i => .ask (.fresh i) fun b => if b then cont (some (.fresh i)) else waitLoop cont n (i + 1)

/-- `w` waits of at most `R` fresh runs each, from the run `base` on, collecting each wait's accepted run after
`found`, then search them all. -/
def waitsPlan (R : ℕ) : ℕ → ℕ → List RunExp → Plan Coin
  | 0, _, found => .done found
  | w + 1, base, found => waitLoop (fun o => waitsPlan R w (base + R) (found ++ o.toList)) R base

/-- **`F0`'s plan** at `k(x)` waits of `R` runs: if `u₀` is accepted, `k(x)` waits of at most `R` fresh runs each,
and search `u₀` and the waits' accepted runs; otherwise search nothing. -/
def conflictPlan (k : Msg → ℕ) (R : ℕ) : RunFinder Coin where
  sel _ := 1
  slots x := k x * R
  plan x _ := .ask (.fresh 0) fun b => if b then waitsPlan R (k x) 1 [.fresh 0] else .done []

end Claims.RecGeneric
