import FlockSoundness.Discharge.HiddenExec.HeadlineExec
import FlockSoundness.Discharge.ScopeG.Scope
import FlockSoundness.Discharge.ScopeG.Bind

set_option Elab.async false

/-!
# The headline at the executable over hidden-output sessions, at the VU's program

`HiddenExec.flock_headline_exec_hidden` and `ScopeW.flock_headline_exec_hiddenW` bound the wrong units of
`Placed.keyProg`, the unit net alone. Here the program is the VU's (`vuProg`, `Scope.lean`): its units, its tail's stages
and its lookup slots in one order, so a unit's cut input a wire copies is a computed column, and the returned outputs may
come from the tail's last stage.

* `VuBindH`: what `HiddenExec.flock_headline_hidden` takes beside the program and its outputs, at the VU's program and
  the hidden tables: the places (`pp`), the constant and zero columns (on `vuOneG`, `vuZeroG`), and the binding of each
  drawn unit's committed wires to registered rows (`lay`, `rs`, `hHm`).
* `flock_headline_exec_hiddenG_of`: the headline at any such binding (`flock_headline_hidden` at `vuProg`).
* `vuBindH_nonempty` (a unit that returns the outputs) and `vuBindHT_nonempty` (a tail's last stage that returns them),
  that the circuit has such a binding: from `Bind.lean`'s obligations, a placement of the VU's rows with its inputs on
  their unit slots and a binding of registered rows there (`vuBindH_of`: the places, constant and zero columns are
  `Places.lean`'s).
* `flock_headline_exec_hiddenG`, `flock_headline_exec_hiddenGT`: the headline at `scopeOkG` and at `scopeOkGT`, at the
  binding the obligation gives (`vuBindG`, `vuBindGT`).

A binding's fields make it the registered rows' (`Binding.Layout.decode_row`, `Binding.HmRows.bc_registered`,
`ExplicitLeaf.pair_spec`), so the headline at the chosen one says what it says at any. What the obligations need is
`ScopeW.KeyH`'s list at the VU's program: its places in the block (`Compose.placement_of_realizes` at the VU's block
positions, as `ExecTemplatePlace` places a typed unit's parts), its aliased inputs, Δ's copies at input rows and at
output rows (from the tail's last stage at `HmOut.outCol`), the rows' `b ‖ c` and single gates, and `HmRowComputes`.
-/

namespace FlockSoundness.Discharge.ScopeG

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- **A binding of the VU's program at the hidden tables**: the program's places in the drawn units' tables, its
constant and zero columns, and the binding of each drawn unit's committed wires to registered rows, as
`flock_headline_hidden` takes them. -/
structure VuBindH {n : ℕ} {I : Inputs} {mPts : ℕ} {c : Flock.Circuit} (hs : scopeOkGT c = true)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (Reg : Type) where
  pp : ProgPlaces (vuProg c hs n) (vuOutsOf c hs n) (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀)
  hConst : ((vuProg c hs n).partition (vuOutsOf c hs n)).ConstCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place
    {vuOneG c hs n}
  hZero : ((vuProg c hs n).partition (vuOutsOf c hs n)).ZeroCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place
    {vuZeroG c hs n}
  lay : Binding.Layout ((vuProg c hs n).partition (vuOutsOf c hs n)) (planAtH dj x₀) (tabAtH dj x₀)
    (((vuProg c hs n).partition (vuOutsOf c hs n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place)
  rs : Binding.HmRows lay
  hHm : rs.Computes

variable {n : ℕ} {I : Inputs} {mPts : ℕ} {c : Flock.Circuit}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **A binding of the VU's program from a placement and a binding of rows there**: the places, constant and zero
columns of `VuPlace.pp` (`Places.lean`), the rows' `lay`, `rs` and `hHm`. -/
noncomputable def vuBindH_of (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hs : scopeOkGT c = true) (P : VuPlace c hs (sitesG (Reg := Reg) dj x₀ ht hc hs))
    (R : VuRowsH hs (ppG dj x₀ ht hc hs P)) : VuBindH hs dj x₀ Reg where
  pp := ppG dj x₀ ht hc hs P
  hConst := P.hConst (sitesG_parse dj x₀ ht hc hs) (vuOutsOf c hs n)
  hZero := P.hZero (sitesG_parse dj x₀ ht hc hs) (vuOutsOf c hs n)
  lay := R.lay
  rs := R.rs
  hHm := R.hHm

/-! ## The obligations -/

/-- **A binding of the VU's program exists, when the unit returns the outputs**: at a hidden-output session's tables
of an untyped circuit at `scopeOkG` whose public file has no public tables, the VU's program has places, constant and
zero columns and a binding of registered rows (`VuBindH`). -/
theorem vuBindH_nonempty (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hscope : scopeOkG c = true) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hpt : pub.tables = none) : Nonempty (VuBindH (scopeOkG_GT hscope) dj x₀ Reg) := by
  obtain ⟨⟨P, R⟩⟩ := vuPlaceRowsH_nonempty dj x₀ (Reg := Reg) ht hc hscope hpub hpt
  exact ⟨vuBindH_of dj x₀ ht hc (scopeOkG_GT hscope) P R⟩

/-- **A binding of the VU's program exists, a tail's last stage returning the outputs included**: `vuBindH_nonempty`
at `scopeOkGT`, the output rows' copies from `HmOut.outCol`'s stage columns. -/
theorem vuBindHT_nonempty (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hs : scopeOkGT c = true) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hpt : pub.tables = none) : Nonempty (VuBindH hs dj x₀ Reg) := by
  obtain ⟨⟨P, R⟩⟩ := vuPlaceRowsHT_nonempty dj x₀ (Reg := Reg) ht hc hs hpub hpt
  exact ⟨vuBindH_of dj x₀ ht hc hs P R⟩

/-- The binding at `scopeOkG` (`vuBindH_nonempty`). -/
noncomputable def vuBindG (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hscope : scopeOkG c = true) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hpt : pub.tables = none) : VuBindH (scopeOkG_GT hscope) dj x₀ Reg :=
  Classical.choice (vuBindH_nonempty dj x₀ ht hc hscope hpub hpt)

/-- The binding at `scopeOkGT` (`vuBindHT_nonempty`). -/
noncomputable def vuBindGT (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hs : scopeOkGT c = true) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hpt : pub.tables = none) : VuBindH hs dj x₀ Reg :=
  Classical.choice (vuBindHT_nonempty dj x₀ ht hc hs hpub hpt)

/-! ## The headlines -/

/-- **The headline over hidden-output sessions at the VU's program, at any binding of it.**
`flock_headline_hidden` at `vuProg`, `vuOutsOf` and the binding `b`. -/
theorem flock_headline_exec_hiddenG_of (hs : scopeOkGT c = true) (b : VuBindH hs dj x₀ Reg)
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
    (hCR : LinkCRL ((vuProg c hs n).partition (vuOutsOf c hs n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) (b.rs.binding b.hHm)
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((vuProg c hs n).partition (vuOutsOf c hs n)).wrong
          (Xpub {vuOneG c hs n} {vuZeroG c hs n}
            (XplurL ((vuProg c hs n).partition (vuOutsOf c hs n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) (b.rs.binding b.hHm)
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((vuProg c hs n).partition (vuOutsOf c hs n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          (b.rs.binding b.hHm) k Rw M ρ t' :=
  flock_headline_hidden dj x₀ (vuProg c hs n) (vuOutsOf c hs n) σs ks hks St hS Kd pop Lb src hA3 b.pp
    {vuOneG c hs n} b.hConst {vuZeroG c hs n} b.hZero b.lay b.rs b.hHm hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf
    N hdec τ hCR hKS

variable (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {pub : Public}
  (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)

/-- **The headline at the executable over hidden-output sessions, at the VU's program, the unit returning the
outputs.** `HiddenExec.flock_headline_exec_hidden`'s statement for an untyped circuit at `scopeOkG` (cut inputs a wire
copies from the VU's own stages and lookup slots, or a leaf cut copies from a row word; every row of the four shapes
`rowShapeW` admits), with `vuProg`'s wrong units (a VU whose returned outputs are not its rows' evaluation on its row
words) and the binding `vuBindG`. Its other hypotheses and its bound are `flock_headline_exec_hidden`'s: `hdec`, A3
(`hA3`), `cr/sha-512` (`hKS`), `ecr/sha-512` (`hCR`) and the numeric conditions. -/
theorem flock_headline_exec_hiddenG (hscope : scopeOkG c = true)
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
    (hCR : LinkCRL ((vuProg c (scopeOkG_GT hscope) n).partition (vuOutsOf c (scopeOkG_GT hscope) n)) H512 enc512
      leaf512 (planAtH dj x₀) (tabAtH dj x₀)
      ((vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).rs.binding (vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).hHm)
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((vuProg c (scopeOkG_GT hscope) n).partition (vuOutsOf c (scopeOkG_GT hscope) n)).wrong
          (Xpub {vuOneG c (scopeOkG_GT hscope) n} {vuZeroG c (scopeOkG_GT hscope) n}
            (XplurL ((vuProg c (scopeOkG_GT hscope) n).partition (vuOutsOf c (scopeOkG_GT hscope) n)) H512 enc512
              leaf512 (planAtH dj x₀) (tabAtH dj x₀)
              ((vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).rs.binding (vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).hHm)
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((vuProg c (scopeOkG_GT hscope) n).partition (vuOutsOf c (scopeOkG_GT hscope) n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).rs.binding (vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)).hHm) k Rw M ρ t' :=
  flock_headline_exec_hiddenG_of dj x₀ (scopeOkG_GT hscope) (vuBindG dj x₀ ht hc hscope hpub hpt (Reg := Reg)) σs ks hks St hS Kd pop
    Lb src hA3 hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

/-- **The headline at the executable over hidden-output sessions, at the VU's program, a tail's last stage returning
the outputs included.** `flock_headline_exec_hiddenG` at `scopeOkGT` and the binding `vuBindGT`. -/
theorem flock_headline_exec_hiddenGT (hs : scopeOkGT c = true)
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
    (hCR : LinkCRL ((vuProg c hs n).partition (vuOutsOf c hs n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀)
      ((vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).rs.binding (vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).hHm)
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((vuProg c hs n).partition (vuOutsOf c hs n)).wrong
          (Xpub {vuOneG c hs n} {vuZeroG c hs n}
            (XplurL ((vuProg c hs n).partition (vuOutsOf c hs n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).rs.binding (vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).hHm)
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((vuProg c hs n).partition (vuOutsOf c hs n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).rs.binding (vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)).hHm) k Rw M ρ t' :=
  flock_headline_exec_hiddenG_of dj x₀ hs (vuBindGT dj x₀ ht hc hs hpub hpt (Reg := Reg)) σs ks hks St hS Kd pop Lb src hA3 hk hRw
    hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.ScopeG
