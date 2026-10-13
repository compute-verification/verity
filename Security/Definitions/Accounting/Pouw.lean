import Definitions.Accounting.Joint
import Definitions.Pouw.Game.Defs

/-!
# PoUW's compute audit as the compute audit of a joint run (trusted)

A run realizes PoUW's game at cost model `CM`, protocol `P` and workload `U` when its compute coins are the game's (the
oracle, the salt and the `t` tickets), and at each value of the other audits' coins `o` the device's PoUW interaction is
an adversary `adv o` of the game: the compute audit's verdict is PoUW's audit of that adversary's transcript, the
adversary's cost is covered by the compute account's priced work, and its oracle calls by the compute account's oracle
queries.

The cost field is where double counting is ruled out. The adversary's cost counts every op its transcript depends on,
so every op the PoUW transcript needs is charged to the compute account. An op charged to another account (a PoUS
decode's, say) is not credited to PoUW's work, and the device's limits (`Step.Fits`) bound the two accounts' ops
together.
-/

namespace Accounting

/-- The compute audit's coins in PoUW's game at protocol `P`, workload `U` and `t` tickets: the oracle, the salt and the
tickets. -/
abbrev PouwCoins {Q R S : Type} (P : Pouw.Protocol Q R S) (U : Pouw.Workload) (t : ℕ) : Type :=
  (Q → R) × S × (Fin t → Pouw.Ticket (P.Wref U.layout) U.layout.N)

/-- Run `r`'s compute audit is PoUW's audit of an adversary of the game at `CM`, `P` and `U`, one for each value of the
other audits' coins, whose cost the compute account's work at price `p` covers and whose oracle calls the compute
account's oracle queries cover. -/
structure RealizesPouw {Q R S : Type} (CM : Pouw.CostModel Q R S) (P : Pouw.Protocol Q R S) (U : Pouw.Workload)
    {t : ℕ} {ι : Type} {d : Device ι} {α ΩQ ΩS : Type} [Fintype α] (r : Run d α (PouwCoins P U t) ΩQ ΩS)
    (p : Price) where
  /-- the adversary the device is in PoUW's game at the other audits' coins -/
  adv : ΩQ × ΩS → Pouw.Adversary CM
  /-- the compute audit is PoUW's audit of the adversary's transcript -/
  accepts_iff : ∀ x o, r.accepts .compute (x, o) ↔
    Pouw.AuditAccepts (Pouw.correctSet P U x.1 x.2.1 ((adv o).transcript U x.1 x.2.1)) x.2.2
  /-- the compute account's priced work covers the adversary's online cost -/
  cost_le : ∀ x o, CM.cost (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤ (r.work p (x, o) : ℝ)
  /-- the compute account's oracle queries cover the adversary's oracle calls -/
  calls_le : ∀ x o, (adv o).pre.calls x.1 + CM.calls (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤
    (r.schedule (x, o)).ops (r.win .compute) (r.acct .compute) .oracle

end Accounting
