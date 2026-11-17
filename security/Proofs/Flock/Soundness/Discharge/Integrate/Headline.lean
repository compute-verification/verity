module

public import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
public import Proofs.Flock.Soundness.Discharge.Hm.Site
public import Proofs.Flock.Soundness.Discharge.Layout.Scope

/-! The headline at the executable (`flock_headline_exec`), in its salted one-table form because an honest transcript
at `hm96-sha512/v1` leaves fails the unsalted headline's openings check. -/

@[expose] public section

namespace FlockSoundness.Discharge.Integrate

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {c : Flock.Circuit} (hU : Built c.unit) {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
  (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (x₀ : SampleSetup I mPts S₀ (dj S₀)) {pub : Public}
  (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)
  (hscope : Layout.scopeOk c = true) {Reg : Type}

/-- **`lay` at the session's tables**: `Layout.keyedLayout` at `sitesAt`, the program's outputs the returned
outputs' bits, the rows' salts at the `hm96` slots' salt columns, the registration's bits the population's returned
outputs. -/
noncomputable abbrev layAt :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAt (Reg := Reg) ht dj x₀)
      (tabAt ht dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAt ht dj x₀) (tabAt ht dj x₀)
        (Layout.keyPlace (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))) :=
  Layout.keyedLayout (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n) Hm.saltCol fun _ => pub

/-- **`rs` at the session's tables**: `Layout.keyedRowsOuts` at `sitesAt`, its circuit conditions from
`scopeOk` (`Layout.scopeOk_spec`), with `hsite` from `sitesAt_bcSite` at a public file without shared-row tables
(`hpt`). -/
noncomputable abbrev rsAt : Binding.HmRows (layAt hU ht hc dj x₀ (pub := pub) (Reg := Reg)) :=
  Layout.keyedRowsOuts (sitesAt_parse ht dj x₀ hc) Hm.saltCol hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1
    (sitesAt_bcSite ht dj x₀ hc (Layout.scopeOk_spec hU hscope).2.1 hpub hpt)

theorem rsAt_computes : (rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).Computes :=
  Hm.keyedRowsOuts_computes (sitesAt_parse ht dj x₀ hc) hc (Layout.scopeOk_spec hU hscope).1
    (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
    (sitesAt_bcSite ht dj x₀ hc (Layout.scopeOk_spec hU hscope).2.1 hpub hpt)

/-- The one-table plan's tables, as `Prog.flock_headline_live_on` takes them. -/
noncomputable abbrev tablesAt : Finset (Fin n) → Reg → TabSpec mPts := fun S _ => (tableAt ht dj x₀ S).T

/-- Covers only one-table public-output `@967b8d06` sessions under `--coins tree`, and `hdec` is open. Nothing
verifies that the unit net lowers its template's Definition, and `keyProg` has no edges between units, so it bounds
only template instances that fail at their own committed values. -/
theorem flock_headline_exec
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
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
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (planAt ht dj x₀) (tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAt ht dj x₀ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAt ht dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planAt ht dj x₀)
              (tabAt ht dj x₀) ((rsAt hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAt_computes hU ht hc dj x₀ hpub hpt hscope))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAt ht dj x₀) (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAt ht dj x₀) (tabAt ht dj x₀)
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
