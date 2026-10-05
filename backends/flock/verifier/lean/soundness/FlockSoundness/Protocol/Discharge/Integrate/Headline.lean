import FlockSoundness.Discharge.Integrate.Accepted
import FlockSoundness.Discharge.Hm.Site
import FlockSoundness.Discharge.Layout.Scope

/-!
# The headline at the executable, with every open guarantee named

`Prog.flock_headline` is about the unsalted batched session. At the production statements (`hm96-sha512/v1` leaves) an
honest transcript fails its openings check, so the headline the executable instantiates is e2e-exec's salted, one-table
form at flock-verify's verdict, `Prog.flock_headline_live_on`, whose hypotheses are `Prog.flock_headline`'s (`pp`,
`hConst`, `hZero`, `lay`, `rs`, `hHm`, A3, `hKS`, `hCR`) with R11 discharged up to R11b (`hdec`, the decoder right at
the law's draws, `DecodesOn`).

`flock_headline_exec` instantiates it at the production program `Placed.keyProg c hU n` (the circuit's program on Δ's
source keys), its returned outputs' bits for the program's outputs (`Layout.outsOf`), and the one-table plan of the
session's inputs (`planAt`, `Sites.lean`), with:
* `pp := Placed.Sites.progPlaces_keyProg` at `sitesAt` (W6 and `Layout.Aliased` at keyProg's places);
* `hConst := Const.constCols_keyProg`, `hZero := Zero.zeroCols_keyProg`, at `ones := {oneG}`, `zeros := {zeroG}`;
* `lay := Layout.keyedLayout` (`layAt`) and `rs := Layout.keyedRowsOuts` (`rsAt`) at row widths `16 · words` and the
  rows' salt columns `Discharge.Hm.saltCol`, with `hsite` from `sitesAt_bcSite`;
* `hHm := Discharge.Hm.keyedRowsOuts_computes` (`rsAt_computes`).

Its hypotheses on the statement's form (`ht`, `hc`, `hU`, `hpub`, `hpt`, `hscope`) are ones flock-verify refuses a statement
failing (PROTOCOL.md §16.15), and `FailClosed.flock_headline_exec` takes none of them.
-/

namespace FlockSoundness.Discharge.Integrate

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {c : Flock.Circuit} (hU : Built c.unit) {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
  (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {pub : Public}
  (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)
  (hscope : Layout.scopeOk c = true) {Reg : Type}

/-- **`lay` at the session's tables**: e2e-layout's `keyedLayout` at `sitesAt`, the program's outputs the returned
outputs' bits, the rows' salts at the `hm96` slots' salt columns, the registration's bits the population's returned
outputs. -/
noncomputable abbrev layAt :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAt (Reg := Reg) ht dj x₀)
      (tabAt ht dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAt ht dj x₀) (tabAt ht dj x₀)
        (Layout.keyPlace (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))) :=
  Layout.keyedLayout (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n) Hm.saltCol fun _ => pub

/-- **`rs` at the session's tables**: e2e-layout's `keyedRowsOuts` at `sitesAt`, its circuit conditions from
`scopeOk` (`Layout.scopeOk_spec`), with `hsite` from `sitesAt_bcSite` at a public file without shared-row tables
(`hpt`). -/
noncomputable abbrev rsAt : Binding.HmRows (layAt hU ht hc dj x₀ (pub := pub) (Reg := Reg)) :=
  Layout.keyedRowsOuts (sitesAt_parse ht dj x₀ hc) Hm.saltCol hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1
    (sitesAt_bcSite ht dj x₀ hc (Layout.scopeOk_spec hU hscope).2.1 hpub hpt)

/-- **`hHm` at the session's rows**: e2e-hm's `keyedRowsOuts_computes`. In a satisfying witness of a drawn unit's
table, each row port's `b ‖ c` at the unit's site is the commit string of the row's bytes under the salt at its
`hm96` slot. -/
theorem rsAt_computes : (rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).Computes :=
  Hm.keyedRowsOuts_computes (sitesAt_parse ht dj x₀ hc) hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
    (sitesAt_bcSite ht dj x₀ hc (Layout.scopeOk_spec hU hscope).2.1 hpub hpt)

/-- The one-table plan's tables, as `Prog.flock_headline_live_on` takes them. -/
noncomputable abbrev tablesAt : Finset (Fin n) → Reg → TabSpec mPts := fun S _ => (tableAt ht dj x₀ S).T

end FlockSoundness.Discharge.Integrate
