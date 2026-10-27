import Proofs.Flock.Claims.KS.PlainLeafClash
import Proofs.Flock.Claims.KS.PlainLeafClashRunK
import Proofs.Flock.Claims.KS.PlainLiveK

/-!
# The concrete simulator's fork bound at plain SHA-512 leaves

E14's fork chain (`PlainFork`, `PlainAlign`, `PlainSched`, `PlainLead`, `PlainForkClash`) bounds the strategies' fork
advantage at the hm96 leaf `leaf512`. Where hidden outputs sit under `flock-leaf/sha512-unsalted` leaves the deployed
decoders are right at `leafPlain` (`plain_hdecHUAt`), so the same chain is stated there: the definitions and the
leaf-generic steps are copies at `leafPlain` (suffix `U`), and the schedule (`plainLive_schedL`, `slotRSL_lead`) and
the live opening (`plain_liveOpening`) read no leaf.

* `PlainHDecU`, `plainSimAtU`, `plainAdvU`, `plainAdvAtU`, `PlainAlignQU`, `PlainLiveForkU`, `PlainForkQU`,
  `plainAdv_le_ofU`, `pairEv_of_forkQU`: the fork advantage at `leafPlain` from the alignment and the live fork's pair.
* `leadsAfter_plainSimAtU`, `PlainSlotQU`, `plainAlignQ_of_slotU`: the alignment from the live schedule.
* `plainForkQ_ofU`: the live fork's collision at the deployed later entries.
* `plainAdv_leU`: so the fork bound holds at every plain-leaf statement.
* `PlainHDecUK` … `plainAdv_leUK`: each of these at `VK keep` (`plainSimK keep`, `plainVerdictAtK keep`).
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.Live
open Flock FlockSoundness.Discharge
open Lean (Json)

/-- The decoders `plainAdvU` reads: `PlainHDec` at `leafPlain`. -/
abbrev PlainHDecU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
    DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j

/-- **The simulated strategy after `Commit` at tape `u`, at `leafPlain`**: `plainSimAt`'s. -/
noncomputable def plainSimAtU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) :=
  simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
    (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)

open Classical in
/-- **The strategies' fork advantage at the concrete simulator, at `leafPlain`**: `plainAdv`'s. -/
noncomputable def plainAdvU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) : ℝ :=
  (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m
      k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
      (simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u))) /
    Fintype.card (plainTape x)

/-- **`plainAdvU`'s term at tape `u`**: `plainAdvAt` at `leafPlain`. -/
noncomputable def plainAdvAtU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (u : plainTape x) : ℝ :=
  advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
    (plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The simulated fork's alignment at tape `u`, at `leafPlain`**: `PlainAlignQ`'s. -/
def PlainAlignQU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  plainAdvAtU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) ×
        List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val p.1 p.2) (a, a')) j
      _ (plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- **The live fork event at the concrete simulator, at `leafPlain`**: `PlainLiveFork`'s. -/
def PlainLiveForkU {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEv (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (plainVerdictAt I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)))
    (ForkEvL Refine.H512 Refine.enc512 Refine.leafPlain sch r.val ℓ)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
      (ω, u) r₁)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
      (ω, u) r₂)

/-- **The live fork event gives `P`'s two accepting runs a collision, at `leafPlain`**: `PlainForkQ`'s. -/
def PlainForkQU {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2)
    (ℓ : ℕ), PlainLiveForkU P x ω st T hw I d opn sch C u r₁ r₂ r ℓ →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₁))).out = .acc →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₂))).out = .acc →
    Claims.TiedColl sha512 ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₁))).vLog ++
      (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r₂))).vLog)

open Classical in
/-- **The fork advantage from the alignment and the live fork's pair, at `leafPlain`**: `plainAdv_le_of`'s proof, the simulation's steps being the leaf's own. -/
theorem plainAdv_le_ofU {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) (hL : plainN x + 2 = (V false).k x) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length)
    (halign : ∀ u, PlainAlignQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ
      (plainJ h0 hsch r ℓ).val u)
    (hpair : ∀ u r₁ r₂, PlainLiveForkU P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) hw I d opn
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (V false) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂))) :
    plainAdvU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ ≤
      Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ h0 hsch r ℓ).val := by
  have hj : (plainJ h0 hsch r ℓ).val < plainN x + 2 := by have := (plainJ h0 hsch r ℓ).isLt; omega
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAt_le execArith Refine.H512 Refine.enc512 Refine.leafPlain
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
    (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
    (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSim P x spec st) x (ω, u)) (plainSim_root P x spec st ω hw u)
    (forkEvL_opens Refine.H512 Refine.enc512 Refine.leafPlain _ r.val ℓ.val)
    (not_opensAtL_nul Refine.H512 Refine.enc512 Refine.leafPlain _ r.val ℓ.val) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStrat plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
    (ω, u) _ ⟨_, hj⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_ind fun h => hpair u _ _ h

/-- **A live fork event's pair, at `leafPlain`**: `pairEv_of_forkQ`'s. -/
theorem pairEv_of_forkQU {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)} {sch : Accounting.Schedule}
    {C : Fin (wordLen false x) → List 𝔽}
    (habort : PlainSimAbort P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSim P x spec st)
      (plainΦ x spec st) (plainVerdictAt I d C (plainRoot P x ω)))
    (hq : PlainForkQU P x ω st T hw I d opn sch) {u : plainTape x} {r₁ r₂ : Fin (plainN x + 2) → LCoin 65536}
    {r : Fin 2} {ℓ : ℕ} (h : PlainLiveForkU P x ω st T hw I d opn sch C u r₁ r₂ r ℓ) :
    Claims.Flock.pairEv sha512 (V false) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂)) := by
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have h₁ := plain_acc_of_abort habort ha.1
  have h₂ := plain_acc_of_abort habort ha'.1
  exact ⟨h₁, h₂, hq C u r₁ r₂ r ℓ h h₁ h₂⟩

end Fork

section Lead

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- **`plainSimAtU` leads with rep 0 for `n` coins** where the plain simulator's live strategy submits its next `n` rounds on rep 0 and the rest on rep 1: `leadsAfter_simAt` at `leafPlain`. -/
theorem leadsAfter_plainSimAtU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) {n : ℕ}
    (hS : ∀ pts, SchedL (plainDg spec) (plainLive x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) n
      ((plainLive x spec st hw P ω u).2 pts)) :
    LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) n
      (plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u) :=
  leadsAfter_simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl) (plainRoot P x ω)
    (plainLive x spec st hw P ω u) (plainSim_root P x spec st ω hw u) hS

end Lead

section Align

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- **The simulated strategy reaches the plain fork slot after `j` coins, at `leafPlain`**: `PlainSlotQ`'s. -/
def PlainSlotQU {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  SlotRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
    (plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- `plainAlignQ_of_slot` at `leafPlain`. -/
theorem plainAlignQ_of_slotU {spec : SessionSpec} {st : Stmt} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {I : Exec.Inputs} {d : Json}
    {x : Claims.Msg} {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : PlainHDecU L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn} {hw : 2 * spec.mPts ≤ 65536} {Ω : Type}
    {P : Claims.Prover Ω} {ω : Ω} {C : Fin (wordLen false x) → List 𝔽} {r : Fin 2}
    {ℓ : Fin (l₀ :: l₁ :: rest).length} {j : ℕ} {u : plainTape x}
    (h : PlainSlotQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u) :
    PlainAlignQU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u :=
  advRSL_le_forkC _ _ _ _ _ m k0 l₀ l₁ rest ext hk0 hm _ _ _ r ℓ.val j _ h

/-- **The live fork event's collision, at `leafPlain` decoders whose outputs record the deployed reps' caps and later openings**: `plainForkQ_of`'s proof, which reads no leaf (`plain_liveOpening`, `forkEvL_entries`). -/
theorem plainForkQ_ofU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAt sch ins.read.I d (wrPD ins.read.I d o)
          (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainForkQU P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn sch := by
  intro C u r₁ r₂ r ℓ hq hacc₁ hacc₂
  obtain ⟨a₁, a₂, ⟨hV₁, pend₁, hpl₁, rfl⟩, ⟨hV₂, pend₂, hpl₂, rfl⟩, hF⟩ := hq
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvL_entries hF
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hc₁ hc₂ hx₁ hx₂
  obtain ⟨st₁, sch₁, p₁, hsu₁, hf₁, hcap₁, hop₁⟩ := plain_liveOpening hs hin hset h hd P ω hacc₁ hV₁.1 r.isLt
    (hE _ hV₁.1 (pend₁ 0) (hpl₁ 0) r.val r.isLt)
  obtain ⟨st₂, sch₂, p₂, hsu₂, hf₂, hcap₂, hop₂⟩ := plain_liveOpening hs hin hset h hd P ω hacc₂ hV₂.1 r.isLt
    (hE _ hV₂.1 (pend₂ 0) (hpl₂ 0) r.val r.isLt)
  obtain rfl : st₁ = st₂ := by simpa using hsu₁.symm.trans hsu₂
  obtain rfl := Option.some.inj (hf₁.symm.trans hf₂)
  obtain ⟨-, hd₁⟩ := hcap₁ ℓ dc hc₁
  obtain ⟨-, hd₂⟩ := hcap₂ ℓ dc hc₂
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : p₁.ligerito.recursiveCaps[ℓ]! = p₂.ligerito.recursiveCaps[ℓ]! :=
    capDig_inj (hd₁.trans hd₂.symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

/-- **The fork bound at plain SHA-512 leaves**: at a plain statement in scope with hidden outputs under
`flock-leaf/sha512-unsalted` leaves, at the plain decoders' hidden openings (`plainOpnH`, `plain_hdecHUAt`), the
concrete simulator's fork advantage at each plain fork round is at most its simulated fork there. The alignment is the
live schedule's (`plainLive_schedL`, `leadsAfter_plainSimAtU`, `slotRSL_lead`), the pair the live fork's collision at
the deployed later entries (`plainForkQ_ofU`, `plain_laterEntriesHUAt`). -/
theorem plainAdv_leU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) :
    plainAdvU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r
        ℓ ≤
      Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ (tables_head_one h) hsch r ℓ).val := by
  have hS := fun u => plainLive_schedL hset h hsch (by simp) (plain_mPts hs hin hset).1 P ω u
  refine plainAdv_le_ofU (tables_head_one h) (plainN_add hset) L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _ _ _ P ω
    C r ℓ (fun u => plainAlignQ_of_slotU (slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch
      hk0 hm _ _ _ (leadsAfter_plainSimAtU L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x _
        (plain_hdecHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u
        (hS u)) r ℓ.val)) fun _ _ _ hq => pairEv_of_forkQU
      (plainSimAbortAt x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω))
      (plainForkQ_ofU hs hin hset h hd _ _ _
        (plain_laterEntriesHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω) hq

end Align

/-! ## At `VK keep` -/

section ForkK

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.Live
open Flock FlockSoundness.Discharge
open Lean (Json)

/-- `PlainHDecU` at `VK keep`. -/
abbrev PlainHDecUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
    DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAtK keep I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j

/-- `plainSimAtU` at `VK keep`. -/
noncomputable def plainSimAtUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) :=
  simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _
    (plainVerdictAtK keep I d C (plainRoot P x ω)) opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)

open Classical in
/-- `plainAdvU` at `VK keep`. -/
noncomputable def plainAdvUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hw : 2 * spec.mPts ≤ 65536) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) : ℝ :=
  (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m
      k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
      (simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
        (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
          (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u))) /
    Fintype.card (plainTape x)

/-- `plainAdvAtU` at `VK keep`. -/
noncomputable def plainAdvAtUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (u : plainTape x) : ℝ :=
  advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
    (plainSimAtUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- `PlainAlignQU` at `VK keep`. -/
def PlainAlignQUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  plainAdvAtUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ u ≤
    forkC (fun a a' => ind (fun p : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) ×
        List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
      ForkEvL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ.val p.1 p.2) (a, a')) j
      _ (plainSimAtUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- `PlainLiveForkU` at `VK keep`. -/
def PlainLiveForkUK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x)
    (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2) (ℓ : ℕ) : Prop :=
  LiveEv (Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (plainVerdictAtK keep I d C (plainRoot P x ω)) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)))
    (ForkEvL Refine.H512 Refine.enc512 Refine.leafPlain sch r.val ℓ)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
      (ω, u) r₁)
    (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
      (ω, u) r₂)

/-- `PlainForkQU` at `VK keep`. -/
def PlainForkQUK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (sch : Accounting.Schedule) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r₁ r₂ : Fin (plainN x + 2) → LCoin 65536) (r : Fin 2)
    (ℓ : ℕ), PlainLiveForkUK keep P x ω st T hw I d opn sch C u r₁ r₂ r ℓ →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₁))).out = .acc →
    (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₂))).out = .acc →
    Claims.TiedColl sha512 ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₁))).vLog ++
      (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r₂))).vLog)

open Classical in
/-- `plainAdv_le_ofU` at `VK keep`. -/
theorem plainAdv_le_ofUK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) (hL : plainN x + 2 = (V false).k x) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length)
    (halign : ∀ u, PlainAlignQUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ
      (plainJ (x := x) h0 hsch r ℓ).val u)
    (hpair : ∀ u r₁ r₂, PlainLiveForkUK keep P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) hw I d opn
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C u r₁ r₂ r ℓ.val →
      Claims.Flock.pairEv sha512 (VK keep) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂))) :
    plainAdvUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ ≤
      Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ (x := x) h0 hsch r ℓ).val := by
  have hj : (plainJ (x := x) h0 hsch r ℓ).val < plainN x + 2 := by have := (plainJ (x := x) h0 hsch r ℓ).isLt; omega
  unfold Claims.Flock.simForkPr
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ => ?_) (Nat.cast_nonneg _)
  refine ((halign u).trans (forkC_simAt_le execArith Refine.H512 Refine.enc512 Refine.leafPlain
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep I d C (plainRoot P x ω)) opn
    (plainDg spec) (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
    (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512
      (plainSimK keep P x spec st) x (ω, u)) (plainSim_rootK keep P x spec st ω hw u)
    (forkEvL_opens Refine.H512 Refine.enc512 Refine.leafPlain _ r.val ℓ.val)
    (not_opensAtL_nul Refine.H512 Refine.enc512 Refine.leafPlain _ r.val ℓ.val) _)).trans ?_
  refine le_trans (le_of_eq (forkC_liveStrat plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
    (ω, u) _ ⟨_, hj⟩)) ?_
  exact Claims.Ex_mono fun rr => ind_le_ind fun h => hpair u _ _ h

/-- `pairEv_of_forkQU` at `VK keep`. -/
theorem pairEv_of_forkQUK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)} {sch : Accounting.Schedule}
    {C : Fin (wordLen false x) → List 𝔽}
    (habort : PlainSimAbortK keep P x ω spec.mPts plainN (plainDg spec) (plainW spec hw) (plainSimK keep P x spec st)
      (plainΦ x spec st) (plainVerdictAtK keep I d C (plainRoot P x ω)))
    (hq : PlainForkQUK keep P x ω st T hw I d opn sch) {u : plainTape x} {r₁ r₂ : Fin (plainN x + 2) → LCoin 65536}
    {r : Fin 2} {ℓ : ℕ} (h : PlainLiveForkUK keep P x ω st T hw I d opn sch C u r₁ r₂ r ℓ) :
    Claims.Flock.pairEv sha512 (VK keep) P x ω (plainΦ x spec st (u, r₁)) (plainΦ x spec st (u, r₂)) := by
  obtain ⟨a, a', ha, ha', -⟩ := id h
  have h₁ := plain_acc_of_abortK keep habort ha.1
  have h₂ := plain_acc_of_abortK keep habort ha'.1
  exact ⟨h₁, h₂, hq C u r₁ r₂ r ℓ h h₁ h₂⟩

end ForkK

section LeadK

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- `leadsAfter_plainSimAtU` at `VK keep`. -/
theorem leadsAfter_plainSimAtUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) {n : ℕ}
    (hS : ∀ pts, SchedL (plainDg spec) (plainLiveK keep x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) n
      ((plainLiveK keep x spec st hw P ω u).2 pts)) :
    LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) n
      (plainSimAtUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u) :=
  leadsAfter_simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl) (plainRoot P x ω)
    (plainLiveK keep x spec st hw P ω u) (plainSim_rootK keep P x spec st ω hw u) hS

end LeadK

section AlignK

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- `PlainSlotQU` at `VK keep`. -/
def PlainSlotQUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2)
    (ℓ : Fin (l₀ :: l₁ :: rest).length) (j : ℕ) (u : plainTape x) : Prop :=
  SlotRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0
    hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) r.val ℓ.val j
    (plainSimAtUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u)

/-- `plainAlignQ_of_slotU` at `VK keep`. -/
theorem plainAlignQ_of_slotUK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {I : Exec.Inputs} {d : Json}
    {x : Claims.Msg} {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : PlainHDecUK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn} {hw : 2 * spec.mPts ≤ 65536} {Ω : Type}
    {P : Claims.Prover Ω} {ω : Ω} {C : Fin (wordLen false x) → List 𝔽} {r : Fin 2}
    {ℓ : Fin (l₀ :: l₁ :: rest).length} {j : ℕ} {u : plainTape x}
    (h : PlainSlotQUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u) :
    PlainAlignQUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay I d x opn hdec hw P ω C r ℓ j u :=
  advRSL_le_forkC _ _ _ _ _ m k0 l₀ l₁ rest ext hk0 hm _ _ _ r ℓ.val j _ h

/-- `plainForkQ_ofU` at `VK keep`. -/
theorem plainForkQ_ofUK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        ∀ r < 2, LaterEntriesAtK keep sch ins.read.I d (wrPDK keep ins.read.I d o)
          (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) r)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainForkQUK keep P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn sch := by
  intro C u r₁ r₂ r ℓ hq hacc₁ hacc₂
  obtain ⟨a₁, a₂, ⟨hV₁, pend₁, hpl₁, rfl⟩, ⟨hV₂, pend₂, hpl₂, rfl⟩, hF⟩ := hq
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvL_entries hF
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hc₁ hc₂ hx₁ hx₂
  obtain ⟨st₁, sch₁, p₁, hsu₁, hf₁, hcap₁, hop₁⟩ := plain_liveOpeningK keep hs hin hset h hd P ω hacc₁ hV₁.1 r.isLt
    (hE _ hV₁.1 (pend₁ 0) (hpl₁ 0) r.val r.isLt)
  obtain ⟨st₂, sch₂, p₂, hsu₂, hf₂, hcap₂, hop₂⟩ := plain_liveOpeningK keep hs hin hset h hd P ω hacc₂ hV₂.1 r.isLt
    (hE _ hV₂.1 (pend₂ 0) (hpl₂ 0) r.val r.isLt)
  obtain rfl : st₁ = st₂ := by simpa using hsu₁.symm.trans hsu₂
  obtain rfl := Option.some.inj (hf₁.symm.trans hf₂)
  obtain ⟨-, hd₁⟩ := hcap₁ ℓ dc hc₁
  obtain ⟨-, hd₂⟩ := hcap₂ ℓ dc hc₂
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : p₁.ligerito.recursiveCaps[ℓ]! = p₂.ligerito.recursiveCaps[ℓ]! :=
    capDig_inj (hd₁.trans hd₂.symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

/-- `plainAdv_leU` at `VK keep`. -/
theorem plainAdv_leUK (keep : Bool → Bool) {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽) (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) :
    plainAdvUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C r
        ℓ ≤
      Claims.Flock.simForkPr sha512 (VK keep) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
        (plainJ (x := x) (tables_head_one h) hsch r ℓ).val := by
  have hS := fun u => plainLive_schedLK keep hset h hsch (by simp) (plain_mPts hs hin hset).1 P ω u
  refine plainAdv_le_ofUK keep (tables_head_one h) (plainN_add hset) L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _ _ _ _ P ω
    C r ℓ (fun u => plainAlignQ_of_slotUK keep (slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch
      hk0 hm _ _ _ (leadsAfter_plainSimAtUK keep L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x _
        (plain_hdecHUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u
        (hS u)) r ℓ.val)) fun _ _ _ hq => pairEv_of_forkQUK keep
      (plainSimAbortAtK keep x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω))
      (plainForkQ_ofUK keep hs hin hset h hd _ _ _
        (plain_laterEntriesHUAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω) hq

end AlignK

end FlockDeploy.KS
