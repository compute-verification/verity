/-
Sampled proofs' period, the audits' half: one-stage v2's parameters at each window of a monitoring period.

The period's check audits every window the ledger's half lists (`Spot.Period.windowsOf`) with the one-stage audit on
the proofs themselves (`OneStageZk`). The verifier holds one set of parameters for the whole monitoring period
(`AuditParams`): the ledger's half's, its copy of main's first-parent history (`main`), and `audit`, one-stage v2's
parameters less what a window sets. A case supplies nothing of the parameters that the check does not hold to the
ledger or to the registration.

**The parameters at a window** (`paramsAt`), key by key of one-stage v2's:

* `window_root`: the window's root, which the auditor's ledger opens the window's session on. Set by the check;
  `audit` naming one is refused.
* `challenger`: the period's own key and family (`AuditParams.ledger`'s `key` and `label`). Set by the check; `audit`
  naming one is refused.
* `log`: `audit`'s own log, then the records of the audits before this one in the ledger's order, each a registration
  the check audits. Set by the check from `audit`'s and its own earlier records, never the case's.
* `proofs.public`: the case's, the digest of the registration's public file. One-stage v2 holds the file to the
  registration's record (`OneStageZk.publicHeld`) and its rows to the header's roots (the session's verifier); `audit`
  naming one is refused.
* `seed_salt`, `day`, `declarations`: one window's own, read from its session or from the ledger, and not derived
  here: `audit` naming one is refused.
* every other key, `proofs.circuit` and `proofs.registered` among them: `audit`'s, the same at every window. A
  monitoring period whose windows differ in one of them is not covered: its windows are refused by one-stage v2 under
  the parameters they do not fit.
-/
module

public import Spot.Period
public import Spot.OneStageZk

@[expose] public section

namespace Spot.Period

open Lean (Json)
open Flock (V)

/-- The verifier's own parameters of a period's check (module docstring): the ledger's half's (`ledger`), the
challenger key's JSON as the verifier gave it (`keyJ`), the verifier's copy of main's first-parent history (`main`)
and one-stage v2's parameters less what a window sets (`audit`). -/
structure AuditParams where
  ledger : Params
  keyJ : Json
  main : Json
  audit : Json

/-- The keys of one-stage v2's parameters that the check sets at a window. -/
def WINDOW_KEYS : Array String := #["challenger", "window_root"]

/-- The keys of one-stage v2's parameters that are one window's own and that the check does not derive. -/
def UNDERIVED_KEYS : Array String := #["day", "declarations", "seed_salt"]

def AUDIT_FORM : String :=
  "the audit's parameters are one-stage v2's less what a window sets: an object with a log, proofs of exactly " ++
  "circuit and registered, and no challenger, window_root, day, declarations or seed_salt"

/-- One-stage v2's parameters at the window whose root is `root`, for the audit of the registration whose public
file's digest is `pub`, after the audits whose records are `before` (module docstring). -/
def paramsAt (p : AuditParams) (root pub : String) (before : Array Json) : Json :=
  let pr := getD p.audit "proofs"
  let log := match getD p.audit "log" with
    | .arr l => l
    | _ => #[]
  Json.mkObj (((entries p.audit).filter fun kv => kv.1 != "log" && kv.1 != "proofs").toList ++
    [("challenger", Json.mkObj [("key", p.keyJ), ("label", Json.str p.ledger.label)]),
     ("log", Json.arr (log ++ before)),
     ("proofs", Json.mkObj [("circuit", getD pr "circuit"), ("public", Json.str pub),
                            ("registered", getD pr "registered")]),
     ("window_root", Json.str root)])

/-- A root and a digest of the right form, at which `AuditParams.of` reads `audit` once. -/
def ZEROS : String := "".pushn '0' 128

/-- The parameters from their JSON, exactly `{"audit", "key", "label", "lane", "main", "start", "stop"}`: the ledger's
half's (`Params.of`), and `audit` of `AUDIT_FORM` that one-stage v2 reads at a window (`paramsAt` at `ZEROS`, read by
`OneStageZk.paramsOfJson`), so parameters one-stage v2 would refuse at every window are refused here. -/
def AuditParams.of (j : Json) : V AuditParams := do
  unless keys? j == some #["audit", "key", "label", "lane", "main", "start", "stop"] do
    throw "a period's parameters' keys are audit, key, label, lane, main, start, stop"
  let ledger ← Params.of (getD j "key") (getD j "label") (getD j "lane") (getD j "start") (getD j "stop")
  let audit := getD j "audit"
  let formed := (audit matches .obj _) && (getD audit "log" matches .arr _) &&
    keys? (getD audit "proofs") == some #["circuit", "registered"] &&
    (WINDOW_KEYS ++ UNDERIVED_KEYS).all fun k => (field? audit k).isNone
  unless formed do throw AUDIT_FORM
  let p : AuditParams := { ledger, keyJ := getD j "key", main := getD j "main", audit }
  let _ ← (OneStageZk.paramsOfJson (paramsAt p ZEROS ZEROS #[])).mapError (s!"audit: {·}")
  return p

end Spot.Period
