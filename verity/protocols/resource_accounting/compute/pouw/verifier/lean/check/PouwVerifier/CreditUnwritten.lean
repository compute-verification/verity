module

public import PouwVerifier.Credit

/-!
# The credit check not yet written in Lean (`credit.pearl_c_period_credit`)

A Pearl-C period's credit, a function of `PouwCheck.Credit` that refuses every input with why (`NOT_WRITTEN`) until its
check is written (`compute-accounting/pouw-verdict-from-main`); the Python's verdict of record on it
(`verifier.not_written`) names it. Beside `PouwCheck.Credit`, which is past the lines a module may run to
(`tests/test_lean_packages.py`'s `LONG`).
-/

@[expose] public section

namespace PouwVerifier.Credit

open Lean (Json)

/-- Why the check below refuses: its Lean is not yet written (`compute-accounting/pouw-verdict-from-main`). -/
def NOT_WRITTEN : String :=
  "this check of record is not yet written in Lean, so it refuses everything (compute-accounting/pouw-verdict-from-main)"

/-- A Pearl-C credit period's credit over its passes (`credit.pearl_c_period_credit_twin`'s rule): refuses every period
until its check is written. -/
def pearlCMonitoringPeriodCredit (_key _ledger _pins _main _name _passes _earlier : Json) : R Json := throw ("period", NOT_WRITTEN)

end PouwVerifier.Credit
