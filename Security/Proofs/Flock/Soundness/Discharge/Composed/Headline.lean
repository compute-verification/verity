import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.Composed.RegWords
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.ZK.PadNonvanishing40

/-!
# The composed headline of one `--zk` sampled session

`zk_session_composed`: for any live prover `τ` of a public-output one-table `--zk` session,

* **soundness**: `Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`, the units those of
  the key program's partition (`keyProg c hU n`), each read at its `--zk` site and bound to the rows it commits
  (lp-zkbind's `sitesZAt`, `rsZAt`): `ZkBind.zk_flock_countC_execOS_at` at the simulated prover `simZ … τ`, with
  `δ_tree = 0` (`execAccept` is acceptance, so the tree's event is empty: `TreeRewind.ofNever`), carried to
  flock-verify's acceptance by `zk_headline_exec_on`;
* **zero knowledge**: at every draw `verify --zk` sets up, the verifier's view of the session's table at the executable's
  leaves (`coinedOf P (tablesAtZK dj y₀ S R)`) is within `2·|Hid|·2^-193` of `S_shvzk`'s, which reads only the public
  values, both ways (Lemma B, `session_shvzk_hm96`, with `PadOnto` from `PadNonvanishing` by `padOnto_M1`, which
`ZK.padNonvanishing40` proves at C-Flock ZK's padding constant).

What it rests on: `cr/sha-512` (`hCR`), `ecr/sha-512` (A2, inside `δ_link`: `⊤` at a prover whose link finders break
it), A3 (`hA3`), A6 and the challenger (`RecordsLiveZK`, and `DrawOkZK` at each of the law's draws),
and `hash-derived-key` (`Assumptions.HashDerivedKey` at hm96-sha512's key). `CoinedOfSetup` is proved
(`coinedOf_setup`), and `RunsZC` at the law's draws is `ZkExec.runsZC_live`.

The zero knowledge holds at every coin the prover doesn't refuse (`CoinedSetup.Admits`, a zerocheck outer coin `1` among
the refused), with the coins committed at `Hello` under `keyed-cr` (not modelled; `ASSUMPTIONS.md`, zero knowledge's
coverage).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

set_option maxRecDepth 8000 in
/-- **Zero knowledge at a `CoinedSetup` from its facts**: Lemma B (`session_shvzk_hm96`) at one table, its leaves'
hiding at any hm96 hiding bound `δ₁` (`hT1`), its row (ii) `padOnto_M1`'s from `PadNonvanishing`
(`ZK.padNonvanishing40`). -/
theorem zkViewAtReg_of_ok (C : CoinedSetup) (hC : CoinedOK execArith (fOf Zk.PAD_C) C) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ₁) :
    ZkViewAtReg C δ₁ := by
  obtain ⟨hL, ht, hS, hT⟩ := hC
  intro c hc Out enc w w0 hv hreg E
  obtain ⟨⟨hpos, hlt⟩, -, hrank, hGp, hW, hpads, hβ, hinner⟩ := hT c hc
  have hpad : (C.T c).tab.PadOnto :=
    FlockSoundness.ZK.padOnto_M1 (C.T c).tab execArith execArith_correct C.cols hL ht hS (fOf Zk.PAD_C) (C.pos c) hpos
      hlt hGp (ZK.padNonvanishing40 _ hL)
  simp only [ZkView.Coined.viewExe_eq, ZkView.Coined.simExe_eq]
  exact FlockSoundness.ZK.Session.session_shvzk_hm96 (fun _ => (C.T c).tab) (fun _ => (C.T c).hm96)
    (fun _ => hT1) (fun _ => hrank) (fun _ => hpad) (fun _ => hW) (fun _ => hpads) (fun _ => hβ) w hreg
    (fun j => hinner (w j) (hv j) (hreg j)) w0 fun v => E (enc v)

/-- `ZkViewAt` where every valid witness's registered values are the public ones (`hreg`, `coinedOf_reg`). -/
theorem zkViewAt_of_ok (C : CoinedSetup) (hC : CoinedOK execArith (fOf Zk.PAD_C) C)
    (hreg : ∀ c w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ₁) : ZkViewAt C δ₁ :=
  fun c hc _ enc w w0 hv E => zkViewAtReg_of_ok C hC hT1 c hc enc w w0 hv (fun j => hreg c (w j) (hv j)) E

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- **The registered values of a `setupH` draw's table are public at every valid witness** once the circuit's output is
the unit's (`ho`): its regions are whole-word (`setupH_whole`, untyped by `parse_facts`). -/
theorem coinedOf_reg_setupH (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {S : Finset (Fin n)}
    {d : Lean.Json} (Tz : TabZK 𝔽 mPts) (y : DrawSetupZK I mPts S d)
    (hS : Tz.S = stmtOf y.x.st y.x.wf.1 (regionsOf y.x.st)) :
    ∀ c' w, (coinedOf P Tz y.x.st).Valid w →
      ((coinedOf P Tz y.x.st).T c').tab.sReg w = ((coinedOf P Tz y.x.st).T c').tab.pubReg := by
  obtain rfl : c = y.x.st.c :=
    Except.ok.inj (hc.symm.trans (setupH_spec ((ZkExec.zkTags_typed I).trans ht) y.x.h).1)
  exact coinedOf_reg P Tz y.x.st y.x.wf.1 y.x.wf.2 hS (setupH_whole y.x.h ho (parse_facts hc).typed)

/-- **One `--zk` sampled session is zero knowledge.** At every draw `verify --zk` sets up, at every coin vector the honest
prover doesn't refuse (`refused_coins`; `CoinedSetup.Admits`: `β ≠ 0`, the mask slot's rank, `ρ₀, ρ₁` independent, no
zerocheck outer coin `1`), the verifier's view of the draw's table at the executable's leaves, at any valid witness, is
within `2·|Hid|·δ₁` of `S_shvzk`'s at any `w0`, both ways, where `δ₁` is any bound at which hm96 hides (`hT1`):
`hidingGap My saltHash` with no premise (`CROnly.hm96Hiding_gap`), `2^-193` under `hash-derived-key`.

For an untyped statement (`ht`, as every soundness theorem) whose output is the unit's (`ho`, which `Layout.scopeOk`
implies) and whose Δ and regions avoid the mask slot (`avoidsMask`, which `verify --zk`'s shape check implies at every
setup, `FailClosed.DrawSetupZK.avoidsMask`), and for any prover of Ligerito's levels ≥ 1 (`LigProver`).

At a refused coin the honest prover stops before it sends anything that depends on the coin (`flock-circuit.rs`'s
`refused_coins`; an outer coin `1` as it opens, `zk_veil::Masking`), and this says nothing about the view there; the
statement at every coin vector, refused ones included, is `ZkEveryCoin.zk_session_view_all`.

Which verifiers: this is honest-verifier zero knowledge at each fixed coin vector. It covers a verifier that chooses its
coins because the challenger commits every coin at `Hello`, before the prover's first message (`coin_tree.rs`; the
prover checks each opening against the commitment before it uses the coin), so the coins are fixed before anything the
prover sends; the binding is collision resistance of `x ↦ SHA-512(ν ‖ x)` (`keyed-cr`). No statement here models that
commitment yet. -/
theorem zk_session_view (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZK I mPts S (dj S)), avoidsMask y.x.st = true)
    {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ₁) :
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) δ₁ :=
  fun S R y => zkViewAt_of_ok _ (coinedOf_setup (Reg := Reg) P I dj y₀ ht hdm S R y).2.2.2.2.2
    (coinedOf_reg_setupH P I ht hc ho _ y (by rw [tablesAtZK_eq dj y₀ S R y]; rfl)) hT1

/-- **One `--zk` sampled session, sound and zero knowledge.** -/
theorem zk_session_composed
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the challenger's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the challenger's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS)
    -- the statement's Δ and regions avoid the mask slot, and hash-derived-key
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZK I mPts S (dj S)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) ∧
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  refine ⟨?_, zk_session_view P I dj y₀ ht hc (Layout.scopeOk_spec hU hscope).2.1 hdm
    (δ₁ := (2 : ℝ≥0∞) ^ (-193 : ℤ)) hKey⟩
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_at ht dj y₀ hc hU hpub hpt hscope hs512 σs ks hks St hS Kd pop Lb src hA3 hk hRw hM
    hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

end FlockSoundness.Discharge.Composed
