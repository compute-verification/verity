import Proofs.Flock.Soundness.Discharge.Composed.Headline
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundJ

/-!
# Zero knowledge of production's `--zk` session: hidden outputs, `J` tables

`zk_session_view` at the session production runs. Its tables are `setupHidden`'s (hidden outputs: the output rows, their
Δ pairs `HmOut.delta`, `HmIn`'s public-input filter, and every row's `b ‖ c` region) at `--session-tables J`, each table
with its own coin vector.

* `zk_session_viewH`: at one table, the view of `tablesAtZKH`'s table at every draw `verify --zk` sets up.
* `ZkViewAtJ`, `zk_session_viewHJ`: at `J` tables, the joint view of `planZHJ`'s tables (`zk_session_soundHJ`'s plan),
  at any coin vector per table, within `Σ_j 2·|Hid_j|·δ₁` of the per-table simulators' joint view, both ways, at any
  bound `δ₁` at which hm96 hides: the pinned key's hiding gap with no premise (`CROnly.hm96Hiding_gap`), `2^-193` under
  `hash-derived-key`.
* `zk_session_composedHJ`: `zk_session_soundHJ` and `zk_session_viewHJ` on the same plan.

`avoidsMask` is a check on the statement (`st.da`, `st.db`, `st.regions`), so at a `setupHidden` statement it covers the
output rows' Δ pairs and regions as at a `setupH` one. Registered reads (`checkRegistered`) add no prover message: they
hold the statement to the verifier's own files and only shrink the accepting event.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- An untyped `setupHidden` statement is a `MaskSetup`: `Layout` at its flat twin's Δ (`HiddenExec.flat_layout`),
`HmRow.check` on the circuit the session commits (`parseCommit_check`), and `m`. -/
theorem maskSetup_of_setupHidden {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : MaskSetup st :=
  ⟨⟨_, _, HiddenExec.flat_layout ht h⟩,
    let ⟨ls, hc⟩ := Hidden.parseCommit_check ht (Hidden.setupHidden_parseCommit h); ⟨_, ls, hc⟩,
    (Hidden.setupHidden_spec' h).m⟩

theorem tableZKH_eq {I : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y₁ y₂ : DrawSetupZKH I mPts S d) :
    y₁.table = y₂.table := by
  obtain ⟨⟨st₁, h₁, D₁, hD₁, hS₁, hg₁, hb₁, m₁, k₁, a₁, b₁, r₁, e₁, hs₁, hm₁, hp₁, w₁⟩, sch₁, t₁, hz₁, _, _, _, _, _, _⟩ :=
    y₁
  obtain ⟨⟨st₂, h₂, D₂, hD₂, hS₂, hg₂, hb₂, m₂, k₂, a₂, b₂, r₂, e₂, hs₂, hm₂, hp₂, w₂⟩, sch₂, t₂, hz₂, _, _, _, _, _, _⟩ :=
    y₂
  obtain rfl : st₁ = st₂ := Except.ok.inj (h₁.symm.trans h₂)
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz₁.symm.trans hz₂))
  rfl

/-- The hidden-output `--zk` tables at a draw `verify --zk` sets up are its table. -/
theorem tablesAtZKH_eq {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZKH I mPts S (dj S)) :
    tablesAtZKH (Reg := Reg) dj y₀ S R = y.table := by
  show tableAtZKH dj y₀ S = _
  rw [tableAtZKH, dif_pos ⟨y⟩]
  exact tableZKH_eq _ _

/-- Table `j` of the hidden-output plan, at a draw at which `verify --zk` sets table `j` up, is its table. -/
theorem planZHJ_get_eq {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (S : Finset (Fin n)) (R : Reg) (j : Fin (nTab I))
    (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) :
    (planZHJ (Reg := Reg) dj y₀ S R).get (Fin.cast (planZHJ_length dj y₀ S R).symm j) = y.table := by
  simp only [planZHJ, List.get_ofFn, Fin.cast_cast, Fin.cast_eq_self]
  rw [tableZHJ, dif_pos ⟨y⟩]
  exact DrawSetupZHJ.table_eq _ _

/-- **Zero knowledge at `J` `CoinedSetup`s, at `δ₁` per hidden leaf** (`ZkViewAt` at `J` tables): at any coin vector of
each table that the honest prover doesn't refuse (`CoinedSetup.Admits`), for any encoding `enc` of the joint view and any
event `E`, the real joint view at valid witnesses `w j` and the simulators' at any `w0 j`, each table's prover
randomness independent, are within `Σ_j 2·|Hid_j|·δ₁`, both ways. -/
def ZkViewAtJ {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest), (∀ j, (C j).Admits (c j)) → ∀ {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w w0 : ∀ j, Witness (C j).S.m), (∀ j, (C j).Valid (w j)) →
    ∀ (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁

section Nonempty

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [AddCommGroup Salt] [AddCommGroup Hi]

set_option synthInstance.maxHeartbeats 1000000 in
theorem rand_nonempty (T : FlockSoundness.ZK.Table F K W Salt (B × Salt) D Hi VL) : Nonempty T.Rand := by
  unfold FlockSoundness.ZK.Table.Rand; infer_instance

set_option synthInstance.maxHeartbeats 1000000 in
theorem simRand_nonempty (T : FlockSoundness.ZK.Table F K W Salt (B × Salt) D Hi VL) : Nonempty T.SimRand := by
  unfold FlockSoundness.ZK.Table.SimRand; infer_instance

end Nonempty

set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
/-- **Zero knowledge at `J` `CoinedSetup`s from their facts**: Lemma B (`table_shvzk_hm96`) at each table, their leaves'
hiding at any hm96 hiding bound `δ₁` (`hT1`), row (ii) `padOnto_M1`'s from `PadNonvanishing` (`ZK.padNonvanishing40`), composed over the tables' independent
randomness (`prCoin_pi_close`), where every valid witness's registered values are the public ones (`hreg`,
`coinedOf_reg`). -/
theorem zkViewAtJ_of_ok {J : ℕ} (C : Fin J → CoinedSetup) (hC : ∀ j, CoinedOK execArith (fOf Zk.PAD_C) (C j))
    (hreg : ∀ j c w, (C j).Valid w → ((C j).T c).tab.sReg w = ((C j).T c).tab.pubReg)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) :
    ZkViewAtJ C δ₁ := by
  intro c hc Out enc w w0 hv E
  have hreg := fun j => hreg j (c j) (w j) (hv j)
  have one : ∀ j (E' : ((C j).T (c j)).tab.View → Prop),
      Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E' (((C j).T (c j)).viewExe (w j) p.1 p.2))
        ≤ Game.prCoin (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E' (((C j).T (c j)).simExe (w0 j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E' (((C j).T (c j)).simExe (w0 j) p.1 p.2))
        ≤ Game.prCoin (fun p : ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E' (((C j).T (c j)).viewExe (w j) p.1 p.2)) + 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁ := by
    intro j E'
    obtain ⟨hL, ht, hS, hT⟩ := hC j
    obtain ⟨⟨hpos, hlt⟩, -, hrank, hGp, hW, hpads, hβ, hinner⟩ := hT (c j) (hc j)
    have hpad : ((C j).T (c j)).tab.PadOnto :=
      FlockSoundness.ZK.padOnto_M1 ((C j).T (c j)).tab execArith execArith_correct (C j).cols hL ht hS (fOf Zk.PAD_C)
        ((C j).pos (c j)) hpos hlt hGp (FlockSoundness.ZK.padNonvanishing40 _ hL)
    simp only [ZkView.Coined.viewExe_eq, ZkView.Coined.simExe_eq]
    exact ((C j).T (c j)).tab.table_shvzk_hm96 ((C j).T (c j)).hm96 hT1 hrank hpad hW hpads hβ (w j) (hreg j)
      (hinner (w j) (hv j) (hreg j)) (w0 j) E'
  have : Nonempty ZkView.Salt := ⟨0⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((rand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  have : ∀ j, Nonempty (((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) :=
    fun j => ⟨((simRand_nonempty ((C j).T (c j)).tab).some, fun _ => 0)⟩
  exact ⟨FlockSoundness.ZK.prCoin_pi_close (Ω := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) (Ω' := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).viewExe (w j) p.1 p.2) (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').1) fun v => E (enc v),
    FlockSoundness.ZK.prCoin_pi_close (Ω := fun j => ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) (Ω' := fun j => ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt)) (V := fun j => ((C j).T (c j)).tab.View)
      (fun j p => ((C j).T (c j)).simExe (w0 j) p.1 p.2) (fun j p => ((C j).T (c j)).viewExe (w j) p.1 p.2)
      (fun j => 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁) (fun j E' => (one j E').2) fun v => E (enc v)⟩

/-- **One hidden-output `--zk` sampled session is zero knowledge, at the coins it doesn't refuse** (`zk_session_view` under
hidden outputs).

Which verifiers: this is honest-verifier zero knowledge at each fixed coin vector, a zerocheck outer coin `1` among the
refused (`CoinedSetup.Admits`). It covers a verifier that chooses its coins because the coin server commits every coin at
`Hello`, before the prover's first message (`coin_tree.rs`), under `keyed-cr`, which no statement here models yet
(`ASSUMPTIONS.md`, zero knowledge's coverage). `avoidsMask` (`hdm`) holds at every setup `verify --zk` makes: its shape check
refuses a statement that meets the mask slot (`Zk.maskFree`). -/
theorem zk_session_viewH (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZKH I mPts S (dj S)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZKH I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZKH (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  fun S R y => zkViewAt_of_ok _ (coinedOf_setup_of P _ y.x.st y.x.wf.1 (by rw [tablesAtZKH_eq dj y₀ S R y]; rfl)
    (maskSetup_of_setupHidden ((ZkExec.zkTags_typed I).trans ht) y.x.h) (hdm S y)).2.2.2.2.2
    (coinedOf_reg P _ y.x.st y.x.wf.1 y.x.wf.2 (by rw [tablesAtZKH_eq dj y₀ S R y]; rfl) (setupHidden_whole y.x.h))
    (δ₁ := (2 : ℝ≥0∞) ^ (-193 : ℤ)) hKey

/-- **One hidden-output `--zk` sampled session of `J` tables is zero knowledge, at the coins it doesn't refuse**: at every
draw at which `verify --zk` sets every table up (`ys`), the joint view of `zk_session_soundHJ`'s plan `planZHJ`, table
`j` at its executable statement, at any coin vector per table that the honest prover doesn't refuse, within
`Σ_j 2·|Hid_j|·δ₁` at any bound `δ₁` at which hm96 hides (`hT1`).

Which verifiers: this is honest-verifier zero knowledge at each fixed coin vector, a zerocheck outer coin `1` among the
refused (`CoinedSetup.Admits`). It covers a verifier that chooses its coins because the coin server commits every coin at
`Hello`, before the prover's first message (`coin_tree.rs`), under `keyed-cr`, which no statement here models yet
(`ASSUMPTIONS.md`, zero knowledge's coverage). `avoidsMask` (`hdm`) holds at every setup `verify --zk` makes: its shape check
refuses a statement that meets the mask slot (`Zk.maskFree`, `FailClosed.DrawSetupZHJ.avoidsMask`). -/
theorem zk_session_viewHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) (δ₁) :=
  fun S R ys => zkViewAtJ_of_ok _ (fun j => (coinedOf_setup_of P _ (ys j).x.st (ys j).x.wf.1
    (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
    (maskSetup_of_setupHidden ((tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) (ys j).x.h)
    (hdm S j (ys j))).2.2.2.2.2)
    (fun j => coinedOf_reg P _ (ys j).x.st (ys j).x.wf.1 (ys j).x.wf.2 (by rw [planZHJ_get_eq dj y₀ S R j (ys j)]; rfl)
      (setupHidden_whole (ys j).x.h)) hT1

/-- **One hidden-output `--zk` sampled session of `J` tables, sound and zero knowledge**: `zk_session_soundHJ` and
`zk_session_viewHJ` on the same plan `planZHJ dj y₀`.

Which verifiers: this is honest-verifier zero knowledge at each fixed coin vector, a zerocheck outer coin `1` among the
refused (`CoinedSetup.Admits`). It covers a verifier that chooses its coins because the coin server commits every coin at
`Hello`, before the prover's first message (`coin_tree.rs`), under `keyed-cr`, which no statement here models yet
(`ASSUMPTIONS.md`, zero knowledge's coverage). `avoidsMask` (`hdm`) holds at every setup `verify --zk` makes: its shape check
refuses a statement that meets the mask slot (`Zk.maskFree`, `FailClosed.DrawSetupZHJ.avoidsMask`). -/
theorem zk_session_composedHJ
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOkH c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    -- every table's statement avoids the mask slot, and hash-derived-key
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  ⟨zk_session_soundHJ I ht dj y₀ wr dg N hc hU hpub hpt hscope kd hkd Lb src hA3 hDraw hTags hRec hk hRw hM hρ hr ht'
      K₀ τ, zk_session_viewHJ P I dj y₀ ht hdm (δ₁ := (2 : ℝ≥0∞) ^ (-193 : ℤ)) hKey⟩

end FlockSoundness.Discharge.ZkHidden
