import Definitions.Pouw.Game.Defs

/-!
# TT per coin (trusted)

`TT` (`Pouw.Assumptions.TT`) bounds adversaries within a budget `(T, q)`, with `q ≤ 2^64` and a free error `ε(q, N)`
that holds grinding and collisions. `TTPerCoin` states it over every adversary, with no budget: on each coin
`(H, s)` it compares the adversary's own online cost with the work of its correct-checked units, and it charges
collisions of `H` to an explicit finder built from the adversary (`finderQueries`), which makes exactly the
adversary's queries. The error reads the adversary's own query count (`Adversary.queries`), a bound and not a cap.
-/

namespace Pouw

open Finset

variable {Q R S : Type}

namespace QTree

/-- The points the program queries against `H`, in order. -/
def queries {α : Type} (H : Q → R) : QTree Q R α → List Q
  | done _ => []
  | query q k => q :: queries H (k (H q))

end QTree

/-- `H` collides on `qs`: two distinct points of `qs` with the same answer. -/
def Collides (H : Q → R) (qs : List Q) : Prop :=
  ∃ x ∈ qs, ∃ y ∈ qs, x ≠ y ∧ H x = H y

/-- The online program's query tree on fixed inputs, as `CM.Queries` provides it. -/
noncomputable def CostModel.tree {CM : CostModel Q R S} (hQ : CM.Queries) (prog : CM.Prog) (U : Workload)
    (act : ℕ → Mat) (σ : State) (s : S) : QTree Q R Transcript :=
  Classical.choose (hQ prog U act σ s)

/-- **The adversary's collision finder `B^A`**: it replays `A`'s preprocessing and then its online query tree on
oracle `H` and salt `s`, and logs every point queried. It makes exactly `A`'s queries (`finderQueries_length`, in
Proofs). `H` collides on its log when two of them answer alike. -/
noncomputable def finderQueries {CM : CostModel Q R S} (hQ : CM.Queries) (A : Adversary CM) (U : Workload)
    (H : Q → R) (s : S) : List Q :=
  A.pre.queries H ++ (CM.tree hQ A.online U A.act (A.pre.eval H) s).queries H

/-- `A`'s query count: the most queries its preprocessing and online phase make together, over every oracle and
salt. -/
noncomputable def Adversary.queries [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    (A : Adversary CM) (U : Workload) : ℕ :=
  univ.sup fun x : (Q → R) × S => A.pre.calls x.1 + CM.calls A.online x.1 U A.act (A.pre.eval x.1) x.2

namespace Assumptions

/-- **Assumption TT per coin**, for a cost model `CM`, a protocol `P`, a domain `D`, a target `γ₀` and an error
`ε(q, N)`. The model's programs reach the oracle only through counted queries (`CostModel.Queries`). For every
in-domain workload and every adversary, with no budget and no cap on its queries, the probability over `(H, s)` that
the adversary's online cost on that coin is below `(1 − γ₀)` times the `m·k·n` work of its units with correct checked
values (`checkedSet`), while `H` has no collision on its finder's log, is at most `ε(q_A, N)`, at `q_A` its own query
count. -/
def TTPerCoin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  ∃ hQ : CM.Queries, ∀ U : Workload, U.InDomain D → ∀ A : Adversary CM,
    pr (fun x : (Q → R) × S =>
      CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 <
          (1 - γ₀) * ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ) ∧
        ¬ Collides x.1 (finderQueries hQ A U x.1 x.2))
      ≤ ε (A.queries U) U.layout.N

end Assumptions

end Pouw
