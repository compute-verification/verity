import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.Composed.Sound
import FlockSoundness.Discharge.Composed.Headline

/-!
# The instance's protocol-1 hypotheses from objective 1's session theorems

The private-circuit instance takes protocol 1's soundness and zero knowledge at the universal unit as Props
(`ZkSessionSound`, `ZkSessionView`). Here they are discharged from `zk_session_sound` and from the zero-knowledge half of
`zk_session_composed` (`zk_session_view`), at the sessions those theorems cover:

* `realSession`: the `UnitSession` of a `--zk` session as `zk_session_sound` reads it: the live game `auditLive`,
  flock-verify `--zk`'s acceptance, the stratified law, and the extracted values, `ε_ks`, `δ_link` and `cr/sha-512` of the
  simulated prover `simZ … τ`. Those read A6's three facts (`TagsOk`, `RecordsLiveZK`, `DrawOkZK` at the law's draws)
  through `runsZC_live`, so the session reads them by cases on `A6`.
* `zkSessionSound_real`: `ZkSessionSound realSession` from `zk_session_sound`, under the count curve's side conditions.
* `zkSessionView_of_zkViewAt`, `zkSessionView_real`: `ZkSessionView` at one table, from `ZkViewAt` at a covered coin
  vector and a valid witness whose registered values are the public ones; at a draw's `coinedOf` tables
  (`zk_session_view`), under `hash-derived-key`, at `δ₁ = 2^-193`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

section Sound

variable (I : Inputs) (ht : I.tags.typed = false)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (dg : ByteArray → ByteArray) (N : ℕ)
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
  {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
  (hscope : Layout.scopeOk c = true)
  {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
  (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
  (k Rw M : ℕ) (ρ t' qF qS : ℝ)

/-- A6 at a `--zk` session: the verify command's tags, the coin server's record of the live session, and its draw at
each of the law's draws. -/
def A6Z : Prop :=
  TagsOk I ∧ RecordsLiveZK I dj wr ∧
    ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)

/-- The simulated prover of `τ` in the compiled batched session, under A6. -/
noncomputable def simOf (h : A6Z I dj wr St Kd pop Lb src)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))) :=
  simZ execArith hs512 _ I wr _ dg _ _ _ _ N
    (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live h.1 h.2.1 h.2.2)) τ

open Classical in
/-- **The universal unit's `--zk` session, as `zk_session_sound` reads it.** -/
noncomputable def realSession :
    UnitSession ((keyProg c hU n).partition (Layout.outsOf c hU n)) Reg where
  Out := Reg × Finset (Fin n) × Exec.Transcript
  G := auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)
  Accepts := LiveAcceptsZK I
  regOf o := o.1
  Lm := Law.stratified σs ks hks
  vals τ _ := if h : A6Z I dj wr St Kd pop Lb src then
      XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
        (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) k Rw
    else fun _ => false
  εks τ := if h : A6Z I dj wr St Kd pop Lb src then
      ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
        (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))
    else ⊤
  δlink τ := if h : A6Z I dj wr St Kd pop Lb src then
      linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
        k Rw M ρ t' (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))
    else ⊤
  CR τ := if h : A6Z I dj wr St Kd pop Lb src then
      ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))) j
        (zcStrat execArith hs512 (planZAt dj y₀) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ) ω)) qF qS
    else False
  m := m
  Ω := Ω
  Lb := Lb
  src := src
  A6 := A6Z I dj wr St Kd pop Lb src

/-- **`ZkSessionSound` at the universal unit's `--zk` session, from `zk_session_sound`**, under the count curve's side
conditions. -/
theorem zkSessionSound_real (hS : Law.ExecStrata σs ks St) (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) (ht' : 0 ≤ t') :
    ZkSessionSound (realSession I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) := by
  intro hA3 hA6 τ hCR K₀
  have hA6' : A6Z I dj wr St Kd pop Lb src := hA6
  simp only [realSession, hA6', ↓reduceDIte] at hCR ⊢
  exact zk_session_sound I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St hS Kd pop Lb src hA3 hA6'.2.2 hA6'.1
    hA6'.2.1 hk hRw hM hρ hr ht' K₀ τ hCR

end Sound

/-- **`ZkSessionView` at one table from `ZkViewAt`**, at a covered coin vector `c` and a valid witness `w` whose
registered values are the public ones. -/
theorem zkSessionView_of_zkViewAt (C : CoinedSetup) {δ₁ : ℝ≥0∞} (h : ZkViewAt C δ₁) (c : CoinsZK C.Rest)
    (hc : C.Covered c) (w : Witness C.S.m) (hw : C.Valid w) (hreg : (C.T c).tab.sReg w = (C.T c).tab.pubReg) :
    ZkSessionView (fun _ : Fin 1 => C.T c) (fun _ => w) δ₁ :=
  fun w₀ _ enc E => h c hc enc (fun _ => w) w₀ (fun _ => hw) (fun _ => hreg) E

/-- **`ZkSessionView` at a draw's `--zk` table**, from `zk_session_view` (the zero-knowledge half of
`zk_session_composed`), for an untyped statement whose Δ and regions avoid the mask slot, under `hash-derived-key`, at
`δ₁ = 2^-193`. -/
theorem zkSessionView_real (P : LigProver) (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZK I mPts S (dj S)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S))
    (c : CoinsZK (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Rest)
    (hc : (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Covered c)
    (w : Witness (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).S.m)
    (hw : (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Valid w)
    (hreg : ((coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).tab.sReg w =
      ((coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).tab.pubReg) :
    ZkSessionView (fun _ : Fin 1 => (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c) (fun _ => w)
      ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  zkSessionView_of_zkViewAt _ (zk_session_view P I dj y₀ ht hdm hKey S R y) c hc w hw hreg

end FlockSoundness.Discharge.PrivateCircuit
