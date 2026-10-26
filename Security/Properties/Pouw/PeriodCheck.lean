import PouwCheck.Period

/-!
# PoUW's period check: what its Lean code (`PouwCheck.Period.protocol`) does

The properties of onsite's work component, `PouwCheck.Period.protocol` (`verity/pouw/period/v1`), stated against its
`check` as it runs, for every input of the types its decoders produce. The check is `Credit.periodCredit`, so these are
`periodCredit`'s meaning. Each is a `Prop`, and `Proofs.Properties.Pouw.PeriodCheck` proves every one.

* `PeriodCreditMeans`: `periodCredit` credits a period only when its evidence is a list of windows (`windowsOf`), those
  windows' roots are exactly the ones the auditor's ledger lists in the period's close, in its order (`Credit.period`),
  its open committed the draw `K` it reports and the table rows `T`, and every window is credited by
  `Credit.windowLayout` at that `K` and `T` (`windowAt`), its credited calls the ones it reports for the window, in
  the close's order. A refused window refuses the period: there's no partial credit.
* `AcceptMeans`: the same of the check's accept, with its profile the period's name and its number of windows.
* `NeverUndecided`: the check accepts or rejects.

These give the period guarantees' premise that every window of the period is accepted, at `Credit.window`'s verdict
(`Credit.window` is `windowLayout`'s first component): `Properties.Pouw.PeriodJ10R`'s `WindowAccepts` at each window,
so its accepted prefix is the whole period. What each window's acceptance means of its draw and its proofs is
`Credit.windowLayout`'s, stated beside its verdict.
-/

namespace Properties.Pouw.PeriodCheck

open Lean (Json toJson)
open PouwCheck PouwCheck.Credit

/-- **The period's windows as `periodCredit` reads them**: the evidence's `windows`, a list of objects each of exactly
`Credit.PERIOD_KEYS`; `none` for any other JSON. -/
def windowsOf (windows : Json) : Option (List Json) :=
  match windows with
  | .arr a => if a.all (fun w => keys w == some PERIOD_KEYS) then some a.toList else none
  | _ => none

/-- **The windows' roots**: each window's `window.root`, which `Credit.period` holds to the period's close. -/
def rootsOf (ws : List Json) : Json :=
  Json.arr (ws.toArray.map fun w => match get w "window" with
    | j@(.obj _) => get j "root"
    | _ => .null)

/-- **Window `w`'s verdict in its period**: `Credit.windowLayout` on the window's fields, at the challenger's key, the
auditor's ledger, the pins and main the period's, and the draw `K` and table rows `T` its period's open committed. -/
def windowAt (key ledger pins main : Json) (K T : Nat) (w : Json) : R (List Nat × Json × List Nat) :=
  windowLayout key ledger pins main (get w "window") (get w "session") (get w "salt") (get w "calls") (get w "served")
    (get w "commit") (get w "rounds") (get w "draw") (get w "descriptor") (get w "entries") (get w "proofs")
    (get w "live") (toJson K) (toJson T)

/-- **What `periodCredit` credits**: a period whose evidence is a list of windows, exactly its close's windows in its
order, under an open that committed the reported `K` and some table rows `T`, every window credited at them, its
credited calls the reported ones, one list a window. -/
def PeriodCreditMeans : Prop :=
  ∀ (key ledger pins main : Json) (name : String) (windows : Json) (pc : PeriodCredit),
    periodCredit key ledger pins main (.str name) windows = .ok pc →
      ∃ ws : List Json, windowsOf windows = some ws ∧ period key ledger (.str name) (rootsOf ws) = .ok () ∧
        committedK ledger name = .ok pc.k ∧ ∃ T : Nat, committedRows ledger name = .ok T ∧
          ws.length = pc.windows.length ∧
          ∀ q ∈ ws.zip pc.windows, ∃ layout rows, windowAt key ledger pins main pc.k T q.1 = .ok (q.2, layout, rows)

/-- **What the period check accepts**: `PeriodCreditMeans` at the verifier's key, main and period name, the auditor's
ledger and the evidence's pins and windows, with the profile the period's name and its number of windows. -/
def AcceptMeans : Prop :=
  ∀ (p : Period.Params) (c : Period.Commitment) (e : Period.Evidence) (pc : PeriodCredit) (prof : Period.Profile),
    Period.protocol.check p c e = .accept pc prof →
      prof.period = p.name ∧ prof.windows = pc.windows.length ∧
      ∃ ws : List Json, windowsOf e.windows = some ws ∧ period p.key c.ledger (.str p.name) (rootsOf ws) = .ok () ∧
        committedK c.ledger p.name = .ok pc.k ∧ ∃ T : Nat, committedRows c.ledger p.name = .ok T ∧
          ws.length = pc.windows.length ∧
          ∀ q ∈ ws.zip pc.windows,
            ∃ layout rows, windowAt p.key c.ledger e.pins p.main pc.k T q.1 = .ok (q.2, layout, rows)

/-- **The period check never leaves a period undecided.** -/
def NeverUndecided : Prop :=
  ∀ (p : Period.Params) (c : Period.Commitment) (e : Period.Evidence) (r : String),
    Period.protocol.check p c e ≠ .undecided r

end Properties.Pouw.PeriodCheck
