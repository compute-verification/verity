import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.Layout.Scope

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

/-- **The headline at the executable.** Let `I` be a verify command's inputs (one table) whose one circuit file parses
to an untyped circuit `c` (`hc`, `ht`) and whose public file loads to `pub` with no shared-row tables (`hpt`, as
the circuit prover's per-instance public files load, under `@967b8d06` too, whose tag allows them), and let the
challenger write draw `S` as the JSON `dj S` (`x₀` one draw it sets up). Let the verifier draw the units with
`Flock.Draw.drawOS`'s stratified law on a byte source A3 makes uniform, and let each draw's session be the challenger's
one table (`liveTable`, A6), written out by `wr`. Against any live prover `τ`, flock-verify accepts with at least `K`
wrong units of the production program `Placed.keyProg c hU n`, at the simulated prover's committed values (1 on the
constant's gate, 0 on the zero's, the plurality of its openings elsewhere), with probability at most
`miss(K) + ε_ks + δ_link`.

Its hypotheses, past the session's inputs:
* the circuit is in `rs`'s scope, `hscope : Layout.scopeOk c = true`: a check that runs on the parsed circuit, that the
  unit has only its leaf group, returns the outputs, has `unitLog ≤ 32`, reads only port bits among its inputs, and
  has only `sha512/row/v1` rows (`v1Ports`: each port `v2 = false`, unsegmented, `bits = 16 · words`, `0 < chunks`).
  `Layout.scopeOk_spec` gives `rs`'s `h1`, `ho` and `PortReads`, `Layout.scopeOk_ports` gives `hHm`'s v1 rows;
  e2e-layout's `#eval` on the circuit prover's composed RoPE circuit returns `true`;
* `hdec` (`DecodesOn`, R11b, at the law's draws only), open: owner e2e-exec (the framing decoder, `wr` as the server's
  record, `opn`, `fin`, `capOf`). At a draw flock-verify accepts whose record carries the challenger's draw `dj S`,
  proving `S`, with link points `mPts`, the table is `stmtOf` of the statement it accepted (`tableAt_of_accepts`, by
  `model_eq_stmtOf`), the statement e2e-exec's acceptance plays. At an accepted draw whose link points are not `mPts`
  the table is unsatisfiable and `hdec` fails there, so `hdec` also needs every accepted draw of the law to set up at
  `mPts`; setupH's link points are `mPtsOf c.g |D.proved|` (e2e-exec's `setupH_mPts_drawn`, `setupH_mPts_eq`);
* the assumptions of record: A3 (`hA3`, `uniform/io-getrandombytes`), `cr/sha-512` for the knowledge finders (`hKS`),
  `ecr/sha-512` for the link finders (`hCR`); A6 (`uniform/flock-coin-server`) is the live session `liveTable` itself;
* numeric conditions and the strata's match (`hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht'`).

Scope. W6 is definitional on the untyped path: the unit's rows are the parsed net's (`Rows.ofNet`), and nothing verifies
that the unit net lowers its template's Definition. The tail stages of `tc_units` templates are outside every unit's
rows. The typed path is not production. `keyProg` has no edges between units: the headline bounds template instances
that fail at their own committed values, and a template instance reading its producers' outputs is carried by the copy
constraints and the region pins, not by an edge of the program. The proof coins are the challenger's per-round OS coins
(A6): seed-derived coins are outside the headline, so a circuit-prover session counts only under `FC_COINS=os`. The tag
is whatever the verifier passes as `--statement`: a session counts when
`flock-verify verify --statement verity/flock-circuit@967b8d06 --coins os` accepts it with one table, so the headline
covers only public-output `@967b8d06` sessions, from builds before #757. Since #757, main's prover always writes the
hidden-output statement (`verity/flock-circuit`, the default `--statement`, `hiddenOutputs = true`), under which
`setupTables` calls `setupHidden`, which this headline does not cover: `HiddenExec.flock_headline_exec_hidden`
does, and `FailClosed.flock_verify_sound` states both. Sessions of more than one table are outside it (the circuit
prover's default is one). -/
theorem flock_headline_exec
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAt (Reg := Reg) ht dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (planAt ht dj x₀) (tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAt ht dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAt ht dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planAt ht dj x₀)
              (tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAt ht dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAt ht dj x₀) (tabAt ht dj x₀)
          ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope)) k Rw M ρ t' :=
  (Placed.keyProg c hU n).flock_headline_live_on (Layout.outsOf c hU n) σs ks hks St hS Kd pop Lb src hA3
    (tablesAt ht dj x₀) (tabAt ht dj x₀)
    (Placed.Sites.progPlaces_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.oneG} (Const.constCols_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.zeroG} (Zero.zeroCols_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    (layAt hU ht hc dj x₀ (pub := pub)) (rsAt hU ht hc dj x₀ hpub hpt hscope)
    (rsAt_computes hU ht hc dj x₀ hpub hpt hscope) hk hRw hM hρ hr ht' K I wr
    opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.Integrate
