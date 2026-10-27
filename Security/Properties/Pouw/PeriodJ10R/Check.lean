import Properties.Pouw.PeriodJ10R
import Properties.Pouw.PeriodCheck

/-!
# The period check's accept at `Credit.window`'s inputs (`Properties.Pouw.PeriodJ10R`'s `WindowAccepts`)

`H1J10RCheckedCreditPeriod` reads served H1's verdict through `WindowAccepts`, at a window's `Registered` and
`Answered` inputs. These read both off a window of the period check's evidence (`registeredOf`, `answeredOf`), and
state what an accept of `PouwCheck.Period.protocol` gives of them.

* `AcceptedWindows`: an accepted period's `n` windows, `n` the profile's count, are each credited by `Credit.window`
  at their inputs, with the credited calls the check reports for them. A `CreditWindows` whose `reg` and `ev` are
  `registeredOf` and `answeredOf` of the evidence's windows then has every window in `AcceptedUpTo`, and its
  `prefixLen` is `n` (`Proofs.Pouw.PeriodJ10R.Honest`'s `prefixLen_all`).
-/

namespace Properties.Pouw.PeriodJ10R.Check

open Lean (Json toJson)
open PouwCheck PouwCheck.Credit Properties.Pouw.PeriodCheck

/-- **What window `w` of the period's evidence registers**, read as `Credit.window` reads it, at the draw `K` and
table rows `T` its period's open committed. -/
def registeredOf (K T : ℕ) (w : Json) : Registered where
  window := get w "window"
  descriptor := get w "descriptor"
  entries := get w "entries"
  commit := get w "commit"
  calls := get w "calls"
  served := get w "served"
  salt := get w "salt"
  k := toJson K
  rows := toJson T

/-- **The rest of window `w`'s inputs**: the verifier's challenger key and main, the auditor's ledger and the host's
pins, the period's, and the window's session, round log, draw, proofs and live evidence. -/
def answeredOf (key ledger pins main w : Json) : Answered where
  key := key
  ledger := ledger
  pins := pins
  main := main
  session := get w "session"
  rounds := get w "rounds"
  draw := get w "draw"
  proofs := get w "proofs"
  live := get w "live"

/-- **An accepted period's windows are each credited by `Credit.window`**: on an accept, the evidence's windows are a
list `ws` of the profile's `n` windows, the open committed `K = pc.k` and some table rows `T`, and `WindowAccepts`
holds at every window `i`, at `registeredOf K T ws[i]` and `answeredOf … ws[i]`. -/
def AcceptedWindows : Prop :=
  ∀ (p : Period.Params) (c : Period.Commitment) (e : Period.Evidence) (pc : PeriodCredit) (prof : Period.Profile),
    Period.protocol.check p c e = .accept pc prof →
      ∃ ws : List Json, windowsOf e.windows = some ws ∧ prof.windows = ws.length ∧
        committedK c.ledger p.name = .ok pc.k ∧ ∃ T : ℕ, committedRows c.ledger p.name = .ok T ∧
          ∀ i : Fin ws.length,
            WindowAccepts (registeredOf pc.k T ws[i]) (answeredOf p.key c.ledger e.pins p.main ws[i])

end Properties.Pouw.PeriodJ10R.Check
