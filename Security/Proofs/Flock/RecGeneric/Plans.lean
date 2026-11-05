module

public import Definitions.Claims.RecursionGeneric.Plans

/-!
# Flock's run finders, and a finder's expected cost

The generic statements read run finders only through `Proto`'s terms and the composite's `forkPlan` and `linkPlan`
(`Definitions.Claims.RecursionGeneric.Plans`). Flock's own terms are plans too, at any caps (`Claims.Recursion.Caps`):
`capPlan Cp M` (`FC`) and `conflictPlan k R` (`F0`). A finder's expected cost (`RunFinder.ExCost`) and its bound in
runs named (`RunFinder.MadeLe`) are what its bits read (`Cost`, `LinkBits`, `AtFlockBits`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims

section Finder

variable {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C]
  {n : ℕ}

/-- `E[F's cost]` in the model `m`: the expected SHA-512 calls of the runs it names, `0` where its selector is
empty. -/
noncomputable def RunFinder.ExCost (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) : ℝ :=
  if h : 0 < F.sel x then Ex fun s => ((F.finder m x h).cost s : ℝ) else 0

end Finder

/-- **`F` names at most `e` runs in expectation, each of cost at most `c`**: in every model whose runs cost at most
`c` calls each on average over every fork the plan can make, `E[F's cost] ≤ e(x) · c`. What a run finder's bits read
(`ExpectedCR` turns its expected cost into its collision bound). -/
def RunFinder.MadeLe {Coin : Type} [Fintype Coin] [Nonempty Coin] (F : RunFinder Coin) (e : Msg → ℕ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ) (C : Type) [Fintype C] [Nonempty C] (m : RunModel Coin Ω n C)
    (x : Msg) (c : ℝ), (∀ (ω : Ω) (j : ℕ) (u₀ : (Fin n → Coin) × C),
      Ex (fun v : (Fin n → Coin) × C =>
        (m.cost ω (Claims.Recursion.fork (m.cut j) u₀.1 v.1, v.2) : ℝ)) ≤ c) →
    (∀ ω, Ex (fun u => (m.cost ω u : ℝ)) ≤ c) →
    F.ExCost m x ≤ e x * c

variable {Coin : Type}

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
