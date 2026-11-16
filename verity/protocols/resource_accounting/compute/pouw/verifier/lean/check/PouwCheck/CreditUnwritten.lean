module

public import PouwCheck.Credit

/-!
# The credit checks not yet written in Lean (`credit.period`, `credit.pearl_c_period_credit`)

A credit period's presented roots and a Pearl-C period's credit, each a function of `PouwCheck.Credit` that refuses every
input with why (`NOT_WRITTEN`) until its check is written (`compute-accounting/pouw-verdict-from-main`); the Python's
verdict of record on each (`verifier.not_written`) names it. Beside `PouwCheck.Credit`, which is past the lines a module
may run to (`tests/test_lean_packages.py`'s `LONG`).
-/

@[expose] public section

namespace PouwCheck.Credit

open Lean (Json)

/-- Why a check below refuses: its Lean is not yet written (`compute-accounting/pouw-verdict-from-main`). -/
def NOT_WRITTEN : String :=
  "this check of record is not yet written in Lean, so it refuses everything (compute-accounting/pouw-verdict-from-main)"

/-- A credit period's presented roots, exactly the sessions its close lists (`credit.period_twin`'s rule): refuses every
period until its check is written. -/
def periodRoots (_key _ledger _name _roots : Json) : R Unit := throw ("period", NOT_WRITTEN)

/-- A Pearl-C credit period's credit over its passes (`credit.pearl_c_period_credit_twin`'s rule): refuses every period
until its check is written. -/
def pearlCMonitoringPeriodCredit (_key _ledger _pins _main _name _passes _earlier : Json) : R Json := throw ("period", NOT_WRITTEN)

end PouwCheck.Credit
