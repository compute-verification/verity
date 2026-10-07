import Proofs.Flock.Soundness.Discharge.Integrate.Headline
import Proofs.Flock.Soundness.Discharge.ScopeW.Keyed

set_option Elab.async false

/-!
# The headline at the executable, under the widened scope

`Integrate.flock_headline_exec` with `hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`, and `rs` the
public sites' rows each under its own schema (`rsAtW`, `keyedRowsOutsW`). The statement is the same expression at
`rsAtW`; `layAt`, `planAt`, `sitesAt` and every other hypothesis are `flock_headline_exec`'s, which read no row's shape.
Its proof is `Prog.flock_headline_live_on` at `HmRowsP.toHmRows`, which binds as `rsAtW` does
(`HmRowsP.toHmRows_binding`), from `keyedRowsOutsW_computes` and the encodings' lemmas (`RowEnc.lean`).
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Integrate
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {c : Flock.Circuit} (hU : Built c.unit) {n : ℕ} {I : Inputs} (ht : I.tags.typed = false)
  (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {pub : Public}
  (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)
  (hscope : scopeOkW c = true) {Reg : Type}

/-- **`rs` at the session's tables, at the widened scope**: `keyedRowsOutsW` at `sitesAt`, its circuit conditions from
`scopeOkW` (`scopeOkW_spec`), with `hsite` from `sitesAt_bcSite`. -/
noncomputable abbrev rsAtW : HmRowsP (layAt hU ht hc dj x₀ (pub := pub) (Reg := Reg)) :=
  keyedRowsOutsW (sitesAt_parse ht dj x₀ hc) Hm.saltCol hc (scopeOkW_spec hU hscope).1
    (scopeOkW_spec hU hscope).2.2 (scopeOkW_spec hU hscope).2.1
    (sitesAt_bcSite ht dj x₀ hc (scopeOkW_spec hU hscope).2.1 hpub hpt)

/-- **`hHm` at the session's rows, at the widened scope** (`keyedRowsOutsW_computes`). -/
theorem rsAtW_computes : (rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).Computes :=
  keyedRowsOutsW_computes (sitesAt_parse ht dj x₀ hc) hc (scopeOkW_spec hU hscope).1
    (scopeOkW_spec hU hscope).2.2 (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope)
    (sitesAt_bcSite ht dj x₀ hc (scopeOkW_spec hU hscope).2.1 hpub hpt)

/-- **The headline at the executable, under the widened scope.** `Integrate.flock_headline_exec`'s statement with
`hscope : scopeOkW c = true` (every row a `sha512/row/v1` row, a `sha512/row/v2` row of whole blocks or of one short
block, or a `sha512/row-seg/v1` row of whole 64-word blocks) and the binding `rsAtW`'s, each row under its own schema.
Its other hypotheses and its bound are `flock_headline_exec`'s; `hCR`'s link finders now also return a segment's two
messages (`segPair`). -/
theorem flock_headline_execW
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
      (planAt ht dj x₀) (Integrate.tabAt ht dj x₀) ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope))
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
              (Integrate.tabAt ht dj x₀) ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAt ht dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAt ht dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAt ht dj x₀)
          (Integrate.tabAt ht dj x₀)
          ((rsAtW hU ht hc dj x₀ hpub hpt hscope (Reg := Reg)).binding
        (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope)) k Rw M ρ t' :=
  (Placed.keyProg c hU n).flock_headline_live_on (Layout.outsOf c hU n) σs ks hks St hS Kd pop Lb src hA3
    (tablesAt ht dj x₀) (Integrate.tabAt ht dj x₀)
    (Placed.Sites.progPlaces_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.oneG} (Const.constCols_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.zeroG} (Zero.zeroCols_keyProg hU (sitesAt_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    (layAt hU ht hc dj x₀ (pub := pub))
    ((rsAtW hU ht hc dj x₀ hpub hpt hscope).toHmRows (rsAtW_computes hU ht hc dj x₀ hpub hpt hscope))
    (HmRowsP.toHmRows_computes _ _) hk hRw hM hρ hr ht' K I wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.ScopeW
