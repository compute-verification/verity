module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
public import Proofs.Flock.Soundness.Discharge.Layout.Scope

@[expose] public section

set_option Elab.async false

/-! The headline at the executable over hidden-output sessions of one table (`flock_headline_exec_hidden`), with the decoder premise `hdec` open. -/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

theorem rowsFit_of_scopeH {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hc : HmOut.parse tags cf tables = .ok c) (hscope : Layout.scopeOkH c = true) :
    Hidden.Any.RowsFitS c :=
  rowsFit_of_scope (parse_untyped ht hc).1 (parse_untyped ht hc).2 hscope

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

include hU hpub in
theorem tableH_bcSiteS {S : Finset (Fin n)} (x : SampleSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).2.1 (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

set_option linter.unusedSectionVars false in
include hU hpub hpt in
theorem tableH_bcSite {S : Finset (Fin n)} (x : SampleSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub (x.table.o u h) (x.table.vu u h) u.val :=
  tableH_bcSiteS ht hc hscope dj hU hpub x h

include hU hpub hpt in
theorem sitesAtH_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).acc S R u h) pub
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).o S R u h)
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableAtH dj x₀ S).acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub ((tableAtH dj x₀ S).o u h)
    ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (SampleSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSite ht hc hscope dj hU hpub hpt _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

/-- **`lay` at the hidden tables**: `keyLayoutH` at `sitesAtH`, the program's outputs the returned outputs' bits, the
registration the population's public file. -/
noncomputable abbrev layAtH :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtH (Reg := Reg) dj x₀)
      (tabAtH dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀)
        (keyPlaceH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the hidden tables**: `keyRowsH` at `sitesAtH`, its circuit conditions from `scopeOkH`
(`Layout.scopeOkH_spec`, `rowsFit_of_scopeH`, `Layout.keysRow_of_parse`), with `hsite` from `sitesAtH_bcSite` at a
public file without shared-row tables (`hpt`). -/
noncomputable abbrev rsAtH : Binding.HmRowsX (layAtH ht hc hscope dj x₀ hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsAtH_computes : (rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem flock_headline_exec_hidden {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dec one fin capOf
      (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 Lp
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
        (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
      (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      (planAtH dj x₀ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 Lp (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 Lp (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
                (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
              (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
            (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt)) k Rw M ρ t' :=
  flock_headline_hidden dj x₀ (Placed.keyProg c hU n) (Layout.outsOf c hU n) σs ks hks St hS Kd pop Lb src hA3
    (ppH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.oneG} (hConstH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.zeroG} (hZeroH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    (layAtH ht hc hscope dj x₀ hU (pub := pub)) (rsAtH ht hc hscope dj x₀ hU hpub hpt)
    (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt) hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.HiddenExec
