module

public import Definitions.Compose.Joint
public import Definitions.Pouw.Game.Defs

/-! PoUW's compute audit as the compute audit of a joint run, whose cost field rules out double counting. -/

@[expose] public section

namespace ResourceAccounting

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
  adv : ΩQ × ΩS → Pouw.Adversary CM
  accepts_iff : ∀ x o, r.accepts .compute (x, o) ↔
    Pouw.AuditAccepts (Pouw.correctSet P U x.1 x.2.1 ((adv o).trace U x.1 x.2.1)) x.2.2
  cost_le : ∀ x o, CM.cost (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤ (r.work p (x, o) : ℝ)
  calls_le : ∀ x o, (adv o).pre.calls x.1 + CM.calls (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤
    (r.schedule (x, o)).ops (r.win .compute) (r.acct .compute) .oracle

end ResourceAccounting
