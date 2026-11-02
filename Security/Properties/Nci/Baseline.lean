import NciCheck.Baseline

/-!
# What θ's provenance check's acceptance means

`NciCheck.Baseline.evaluate` (`verity/core/protocols/network/nci/lean/NciCheck/Baseline.lean`) is the NCI check of
record: the policy check, then each θ entry's provenance against the trusted input, the service's registration log, the
run's own record and the program function's parameters. `Nci.Assumptions.ThetaRegistered`
(`Definitions/Nci/Assumptions/Baseline.lean`, [THETA-REGISTERED]) states what θ must hold: registered information,
public data registered at the start of the deployment, which the log's first record registers.

* `BaselineProvenance`: on every instance and trusted input it accepts, the trusted input's log is in R6's chain from a
  record that opens it (`Chained`), the run's own record is one of its records, the parameters are the trusted input's
  (`ParamsOf`), and every θ entry `t` is `HeldAtStart`: its provenance cites, by digest, the log's first record, which
  registers the cited value under the cited root, and the run's own record, of this program, registers the parameter
  whose input node θ names under that same root.

What this does not give, and `ThetaRegistered` still assumes, is what no check sees, since commitments are hiding:
that equal roots hold equal values (the binding premise the reduction would name, `Nci.Assumptions.RootsBind`, not
stated in Lean yet), that the data the log's first record registers was public and not computed off the books, that
the auditor knows which log is the deployment's, and how the reference circuit `c` takes its own `c.θ`.
`theta-width` (the cited value's dimensions give the parameter's gates) is checked but not stated here: under the
binding premise equal roots already give equal rows.

Proved as `NciCheck.Properties.SecurityProofs.BaselineProvenance : NciCheck.Properties.BaselineProvenance` in
`Proofs.Properties.Nci.Baseline`.
-/

namespace NciCheck.Properties

open Lean (Json)
open Flock.Program (Prog Ty)
open NciCheck.Baseline (getD strAt digest lastWith ownersOf)

/-- `log` is a registration log in R6's chain, oldest first: every record has a digest, the first record's `previous` is
`null` (it opens the deployment), and each later record's `previous` is the digest of the record before it. -/
def Chained (log : Array Json) : Prop :=
  ∀ n < log.size, (∃ d, digest log[n]! = .ok d) ∧ (n = 0 → getD log[n]! "previous" = .null) ∧
    ∀ m d, n = m + 1 → digest log[m]! = .ok d → strAt log[n]! "previous" = some d

/-- `ps` are the parameters the trusted input `tr` gives, `[[name, type], ...]` in order. -/
def ParamsOf (tr : Json) (ps : Array (String × Ty)) : Prop :=
  ∃ a, getD tr "params" = .arr a ∧ a.size = ps.size ∧
    ∀ n < a.size, ∃ b, a[n]! = .arr b ∧ b.size = 2 ∧ b[0]! = .str ps[n]!.1 ∧ Ty.ofJson b[1]! = .ok ps[n]!.2

/-- The trusted input `tr` holds the registration log `log`, in R6's chain, the run's own record `run`, one of the log's
records named by its digest, and the parameters `ps`. -/
def TrustedHolds (tr : Json) (log : Array Json) (run : Json) (ps : Array (String × Ty)) : Prop :=
  getD tr "log" = .arr log ∧ Chained log ∧
    (∃ n < log.size, log[n]! = run ∧ ∃ r, digest run = .ok r ∧ strAt tr "run" = some r) ∧ ParamsOf tr ps

/-- θ entry `t` of `j`, whose fields are `i`'s and whose Program is `P`, holds data registered at the deployment's start,
as the log `log`, the run's record `run` and the parameters `ps` show it. Its provenance, entry `t` of `j`'s
`provenance`, is of the `registered` kind and cites by digest `D` the log's first record, which registers the value `V`
under the root `R` (its last entry for `V`). θ names the root input node `k`, one of the parameter `ps[p]`'s input nodes
(`ownersOf`: the parameters lowered onto the root's input nodes). And the run's own record is this Program's and
registers that parameter under the same root `R`. -/
def HeldAtStart (j : Json) (i : Instance) (P : Prog) (log : Array Json) (run : Json) (ps : Array (String × Ty))
    (t : Nat) : Prop :=
  ∃ (prov : Array Json) (D V R : String),
    getD j "provenance" = .arr prov ∧ prov.size = i.theta.size ∧
    strAt prov[t]! "kind" = some "registered" ∧ strAt prov[t]! "registration" = some D ∧
    strAt prov[t]! "value" = some V ∧ strAt prov[t]! "root" = some R ∧
    0 < log.size ∧ digest log[0]! = .ok D ∧
    (∃ e, lastWith log[0]! "registered" "value" V = some e ∧ strAt e "root" = some R) ∧
    ∃ (k p : Nat) (own : Array (Option Nat)), natOf? i.theta[t]! = some k ∧ ownersOf P ps = some own ∧
      own[k]? = some (some p) ∧ strAt run "program" = some (Flock.hexOf P.sha512) ∧
      ∃ e', lastWith run "registered" "value" ps[p]!.1 = some e' ∧ strAt e' "root" = some R

/-- What acceptance on `j` with the trusted input `tr` means: `j`'s fields (`parse`, which reads them as
`Properties.Nci.Parses` states, `parse_ok`) and Program, and a trusted log, run and parameters, against which every θ
entry is `HeldAtStart`. -/
def ProvenanceHeld (j tr : Json) : Prop :=
  ∃ (i : Instance) (P : Prog) (log : Array Json) (run : Json) (ps : Array (String × Ty)),
    parse j = .ok i ∧ Flock.Program.decode i.program.toUTF8 = .ok P ∧ TrustedHolds tr log run ps ∧
      ∀ t < i.theta.size, HeldAtStart j i P log run ps t

/-- **θ holds what the deployment's first record registers.** On every instance `j` and trusted input `tr` that
`NciCheck.Baseline.evaluate` accepts, `ProvenanceHeld j tr`. -/
def BaselineProvenance : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out → ProvenanceHeld j tr

end NciCheck.Properties
