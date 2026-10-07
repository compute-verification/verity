import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.Discharge.ScopeW.KeyH
import Proofs.Flock.Soundness.Discharge.ScopeW.Sites

set_option Elab.async false

/-!
# The headline at the executable over hidden-output sessions, under the widened scope

`HiddenExec.flock_headline_exec_hidden` with `hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`, so its
circuits include every PoUW circuit whose rows have the four shapes `rowShapeW` admits (`Scope.lean`). The statement is
the same expression at:
* `sitesAtHW` (`Sites.lean`): the drawn tables' acceptances with their unit slots' rows from `unitRowsFlatW`;
* `layAtHW`, `rsAtHW` (`KeyH.lean`): the readers at output rows of runs of output ports (`outBitRowW`), each row hashed
  under its own schema (`HmRowsP`, `rowEncOf`), so the binding `(rsAtHW …).binding (rsAtHW_computes …)` binds a
  segmented row through its segments' digests, a SHA-512 collision when two of its values share an outer message
  (`segPair_spec`).

Its proof is `flock_headline_hidden` at `HmRowsP.toHmRows`, which binds as `rsAtHW` does (`HmRowsP.toHmRows_binding`),
from what `KeyH.lean`, `Sites.lean` and `RowEnc.lean` prove.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : scopeOkW c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

include hU hpub hpt in
/-- **The site fact at a drawn table under hidden outputs, at the widened scope**: `HiddenExec.tableH_bcSite` at
`accW`. -/
theorem tableH_bcSiteW {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (accW ht hc (scopeOkW_ports hscope) x.table) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, hpt, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (parse_untyped ht hc).1 (scopeOkW_spec hU hscope).2.1 (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable site** of the one-table hidden plan, at the widened scope
(`HiddenExec.sitesAtH_bcSite`). -/
theorem sitesAtHW_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtHW ht hc (scopeOkW_ports hscope) dj x₀).acc S R u h) pub
          ((sitesAtHW ht hc (scopeOkW_ports hscope) dj x₀).o S R u h)
          ((sitesAtHW ht hc (scopeOkW_ports hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH (accW ht hc (scopeOkW_ports hscope) (tableAtH dj x₀ S)) pub ((tableAtH dj x₀ S).o u h)
    ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteW ht hc hscope dj hU hpub hpt _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

/-- **`lay` at the hidden tables, at the widened scope**: `keyLayoutHW` at `sitesAtHW`. -/
noncomputable abbrev layAtHW :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtH (Reg := Reg) dj x₀)
      (tabAtH dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀)
        (keyPlaceH (sitesAtHW_parse ht hc (scopeOkW_ports hscope) dj x₀) (Layout.outsOf c hU n))) :=
  keyLayoutHW (sitesAtHW_parse ht hc (scopeOkW_ports hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the hidden tables, at the widened scope**: `keyRowsHW` at `sitesAtHW`, its circuit conditions from
`scopeOkW` (`scopeOkW_spec`, `scopeOkW_ports`, `Layout.keysRow_of_parse`), `hsite` from `sitesAtHW_bcSite`. -/
noncomputable abbrev rsAtHW : HmRowsP (layAtHW ht hc hscope dj x₀ hU (pub := pub) (Reg := Reg)) :=
  keyRowsHW (sitesAtHW_parse ht hc (scopeOkW_ports hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtHW_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (scopeOkW_spec hU hscope).1 (scopeOkW_spec hU hscope).2.2)

/-- **`hHm` at the hidden tables, at the widened scope** (`keyRowsHW_computes`). -/
theorem rsAtHW_computes : (rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsHW_computes (sitesAtHW_parse ht hc (scopeOkW_ports hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtHW_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (scopeOkW_spec hU hscope).1 (scopeOkW_spec hU hscope).2.2)

/-- **The headline at the executable over hidden-output sessions, under the widened scope.**
`HiddenExec.flock_headline_exec_hidden`'s statement with `hscope : scopeOkW c = true` (every row a `sha512/row/v1` row,
a `sha512/row/v2` row of whole blocks or of one short block, or a `sha512/row-seg/v1` row of whole 64-word blocks;
output rows one port or a run of output ports) and the binding `rsAtHW`'s, each row under its own schema. Its other
hypotheses and its bound are `flock_headline_exec_hidden`'s: `hdec`, A3 (`hA3`), `cr/sha-512` (`hKS`), `ecr/sha-512`
(`hCR`, whose link finders now also return a segment's two messages, `segPair`) and the numeric conditions. -/
theorem flock_headline_exec_hiddenW
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
        (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
                (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtHW ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
            (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt)) k Rw M ρ t' :=
  flock_headline_hidden dj x₀ (Placed.keyProg c hU n) (Layout.outsOf c hU n) σs ks hks St hS Kd pop Lb src hA3
    (ppHW ht hc (scopeOkW_ports hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.oneG} (hConstHW ht hc (scopeOkW_ports hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.zeroG} (hZeroHW ht hc (scopeOkW_ports hscope) dj x₀ hU (Layout.outsOf c hU n))
    (layAtHW ht hc hscope dj x₀ hU (pub := pub))
    ((rsAtHW ht hc hscope dj x₀ hU hpub hpt).toHmRows (rsAtHW_computes ht hc hscope dj x₀ hU hpub hpt))
    (HmRowsP.toHmRows_computes _ _) hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.ScopeW
