module

public import Definitions.Claims.RecursionLin

/-!
# Run finders: an inner protocol's error terms, in a form the recursion carries

A recursion generic in its inner protocol needs the inner protocol's knowledge error stated over collision finders the
compiler's proof can rebuild at the composite without knowing the inner protocol. A finder that reads a run only
through whether it accepts and its coins, forks runs at rounds, and searches the calls of the runs it names for a
collision is such a finder: it is a plan (`Plan`) over runs named by how they were made (`RunExp`), and it runs in any
model of runs (`RunModel`). Its model at the inner protocol's own runs (`runModel`) is where the inner protocol's KS is
stated; its models at the composite's reruns (`Claims.RecGeneric.Comp.innerModel`, `Comp.verModel`) are where the
recursion's bound reads it.

`ReadsLe F d`: in every model and for every predicate `bad` of runs, a run of `F`'s plan asks about a run where `bad`
holds and the run accepts with probability at most `d · Pr[bad]`. It is a property of the plan alone, Wald's bound for
gated loops, and it is what bounds the cost of the runs where the two models disagree.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims

/-- **A run, named by how it was made**: the `i`-th fresh run, or `fork j u v`, the run whose coins before round `j`'s
cut are `u`'s and whose coins after it, and extra coins, are `v`'s. -/
inductive RunExp where
  | fresh (i : ℕ)
  | fork (j : ℕ) (u v : RunExp)
  deriving DecidableEq

/-- **A run finder's plan**: ask whether a run accepts and go on by the answer, look at a run's coins and go on by them,
or stop and name the runs whose calls are searched for a collision. -/
inductive Plan (Coin : Type) where
  | done (search : List RunExp)
  | ask (u : RunExp) (k : Bool → Plan Coin)
  | look (u : RunExp) (k : (ℕ → Coin) → Plan Coin)

/-- **A model of runs** at a tape space `Ω`: a run is its `n` coins and its extra coins `C`; `fork j u v` keeps
`u`'s coins before `cut j` and takes `v`'s after, with `v`'s extra coins. A run accepts or not (`acc`), shows its coins
(`coins`), the SHA-512 evaluations a collision is searched in (`log`) and costs `cost` calls. -/
structure RunModel (Coin Ω : Type) (n : ℕ) (C : Type) where
  cut : ℕ → ℕ
  acc : Ω → (Fin n → Coin) × C → Prop
  coins : (Fin n → Coin) × C → ℕ → Coin
  log : Ω → (Fin n → Coin) × C → List (Msg × Msg)
  cost : Ω → (Fin n → Coin) × C → ℕ

section Model

variable {Coin Ω C : Type} {n : ℕ}

/-- The run a run expression names, at the fresh runs `us`. -/
def RunModel.named (m : RunModel Coin Ω n C) (us : ℕ → (Fin n → Coin) × C) : RunExp → (Fin n → Coin) × C
  | .fresh i => us i
  | .fork j u v => (Claims.Recursion.fork (m.cut j) (m.named us u).1 (m.named us v).1, (m.named us v).2)

open Classical in
/-- **A plan's run** at the tape `ω` and the fresh runs `us`: the runs it asked about, the runs it looked at and the
runs it searched, in order. -/
noncomputable def Plan.exec (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C) :
    Plan Coin → List RunExp × List RunExp × List RunExp
  | .done l => ([], [], l)
  | .ask u k =>
    let r := (k (decide (m.acc ω (m.named us u)))).exec m ω us
    (u :: r.1, r.2.1, r.2.2)
  | .look u k =>
    let r := (k (m.coins (m.named us u))).exec m ω us
    (r.1, u :: r.2.1, r.2.2)

end Model

/-- **A run finder**: a uniform selector in `[0, sel x)` (a round guess, a cap position), `slots x + 1` fresh runs, and
the plan at each selector value. -/
structure RunFinder (Coin : Type) where
  sel : Msg → ℕ
  slots : Msg → ℕ
  plan : (x : Msg) → Fin (sel x) → Plan Coin

section Finder

variable {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C]
  {n : ℕ}

/-- The fresh runs a finder's coins give, the last repeated past them. -/
def slotRuns {s : ℕ} (us : Fin (s + 1) → (Fin n → Coin) × C) : ℕ → (Fin n → Coin) × C :=
  fun i => us ⟨min i s, Nat.lt_succ_of_le (Nat.min_le_right i s)⟩

/-- **The run finder in a model**, at the plan's statement `x`: its coins are the selector, a tape and the fresh runs;
it outputs the first collision among the searched runs' evaluations (`log`), and its cost is the calls of every run
it named, each once. -/
noncomputable def RunFinder.finder (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < F.sel x) :
    Finder where
  S := Fin (F.sel x) × Ω × (Fin (F.slots x + 1) → (Fin n → Coin) × C)
  ne := ⟨(⟨0, h⟩, Classical.arbitrary _, fun _ => Classical.arbitrary _)⟩
  out s :=
    findColl (((F.plan x s.1).exec m s.2.1 (slotRuns s.2.2)).2.2.flatMap fun e =>
      m.log s.2.1 (m.named (slotRuns s.2.2) e))
  cost s :=
    (((((F.plan x s.1).exec m s.2.1 (slotRuns s.2.2)).1 ++ ((F.plan x s.1).exec m s.2.1 (slotRuns s.2.2)).2.1 ++
      ((F.plan x s.1).exec m s.2.1 (slotRuns s.2.2)).2.2).dedup).map fun e =>
        m.cost s.2.1 (m.named (slotRuns s.2.2) e)).sum

/-- `Pr[F finds a collision]` in the model `m` at the plan's statement `x`, `0` where its selector is empty. -/
noncomputable def RunFinder.PrAt (H : Msg → Msg) (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) : ℝ :=
  if h : 0 < F.sel x then (F.finder m x h).Pr H else 0

/-- **`Pr[F's plan asks about a run where `q` holds]`**, over its coins, `0` where its selector is empty. -/
noncomputable def RunFinder.PrAsk (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg)
    (q : Ω → (Fin n → Coin) × C → Prop) : ℝ :=
  if 0 < F.sel x then
    Pr fun s : Fin (F.sel x) × Ω × (Fin (F.slots x + 1) → (Fin n → Coin) × C) =>
      ∃ e ∈ ((F.plan x s.1).exec m s.2.1 (slotRuns s.2.2)).1, q s.2.1 (m.named (slotRuns s.2.2) e)
  else 0

end Finder

/-- **`F`'s plan meets a bad run at most `d` times as often as one run is bad**: in every model, for every predicate
`bad`, the plan asks about a run where `bad` holds and the run accepts with probability at most
`d(x) · Pr_{ω, u}[bad ω u]`. A plan that asks nothing meets it at `0`; a gated loop of forks, at one per loop
(`Pr_gate_loop_fork_le`). -/
def RunFinder.ReadsLe {Coin : Type} [Fintype Coin] [Nonempty Coin] (F : RunFinder Coin) (d : Msg → ℕ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ) (C : Type) [Fintype C] [Nonempty C] (m : RunModel Coin Ω n C)
    (x : Msg) (bad : Ω → (Fin n → Coin) × C → Prop),
    F.PrAsk m x (fun ω u => bad ω u ∧ m.acc ω u) ≤ d x * Pr (fun t : Ω × ((Fin n → Coin) × C) => bad t.1 t.2)

section Runs

variable {Coin : Type} [Inhabited Coin]

/-- **The model of an inner protocol's runs** at the prover `P` that may abort by `keep`: a run is the protocol's coins
and its abort coins, a fork keeps the round prefix (`cut = id`), a run accepts where the protocol run as it aborts
accepts, shows the protocol's coins, and its evaluations are the verifier's calls in that run; its cost is both
parties' calls (`Claims.Recursion.abortCost`). -/
def runModel (H : Msg → Msg) (V : Verifier Coin) {Ω A : Type} (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : RunModel Coin Ω (V.k x) A where
  cut := id
  acc ω u := (Claims.Recursion.abortRun H P V x keep ω u).out = .acc
  coins u := coinSeq u.1
  log ω u := (Claims.Recursion.abortRun H P V x keep ω u).vLog
  cost ω u := Claims.Recursion.abortCost H V P x keep ω u

end Runs

end Claims.RecGeneric
