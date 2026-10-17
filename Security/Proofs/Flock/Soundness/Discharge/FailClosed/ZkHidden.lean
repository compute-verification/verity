import Proofs.Flock.Soundness.Discharge.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask

/-!
# The hidden-output `--zk` headlines, over what flock-verify accepts

Each theorem here is the `ZkHidden` headline of the same name over `VerifiesZK` (`LiveVerifiesZK`), what
`flock-verify verify --zk` accepts, with no hypothesis on the statement's form, as `FailClosed.Composed` is for public
outputs: the circuit (`circZ I`), the public file and the facts the `ZkHidden` headline took about them come from the
statement check at each statement it passes (`hs : ScopeZ I`), and at a statement the check refuses the event is empty.

What stays a hypothesis on the statement is which headline applies: hidden outputs (`hh`) and, for the stratified law,
one table (`hn`); `zk_session_soundHJ` covers every number of tables. Zero knowledge's `avoidsMask` comes from each
table's setup (`DrawSetupZHJ.avoidsMask`): `verify --zk` refuses a statement whose Δ or regions meet the mask slot
(`Zk.maskFree`, D12). The `ZkHidden` headlines remain, as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

theorem ScopeZ.tagsOkH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1) :
    TagsOkH I :=
  ⟨hh, h.retainRounds, h.merkleLeaf, hn⟩

theorem ScopeZ.tagsOkZHJ {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) : TagsOkZHJ I :=
  ⟨hh, h.retainRounds, h.merkleLeaf⟩

variable {mPts n : ℕ} {Reg : Type}

section

variable (P : LigProver)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts, sound and zero knowledge.** `ZkHidden.zk_session_composedHJ` over `VerifiesZK`; zero knowledge, a property of the honest prover, at every statement the check passes. -/
theorem zk_session_composedHJ
    (I : Inputs) (hh : I.tags.hiddenOutputs = true)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    -- hash-derived-key
    (hKey : Assumptions.Zk.HashDerivedKeyHm96)
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    (prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ))) ∧
    (ScopeZ I → ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ))) :=
  ⟨prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      ((FlockSoundness.Discharge.ZkHidden.zk_session_composedHJ P I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec hk hRw hM hρ hr ht' K₀ τ (fun _ _ y => DrawSetupZHJ.avoidsMask y) hKey).1 (hCR hs)),
    fun hs => ZkHidden.zk_session_viewHJ P I dj y₀ hs.typed (fun _ _ y => DrawSetupZHJ.avoidsMask y) hKey⟩

end

/-- **One hidden-output `--zk` sampled session that flock-verify accepts is sound.** `ZkHidden.zk_session_soundH` over `VerifiesZK`. -/
theorem zk_session_soundH
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I, ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hRec hDraw)) τ))) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundH I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH σs ks hks St hS Kd pop Lb src hA3 hDraw (hs.tagsOkH hh hn) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session that flock-verify accepts is sound, under `live-verifier`.** `ZkHidden.zk_session_soundH_custody` over `VerifiesZK`. -/
theorem zk_session_soundH_custody
    (I : Inputs) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I (mPtsOf (circZ I).g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf (circZ I).g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- live-verifier: the record and its draw file are the live session's
    (hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf (circZ I).g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I, ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf (circZ I).g (∑ s, ks s)) N wr)) τ ≤
      ⨆ hs : ScopeZ I, ((Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZH dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZH_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ tablesAtZKH_live (hs.tagsOkH hh hn) hCust.1
          (fun ω => drawOkZKH_of_exec I (hs.parseH hh) σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundH_custody I hs.typed (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH σs ks hks St hS Kd pop Lb src hA3 dj y₀ wr dg N (hs.tagsOkH hh hn) hCust hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound.** `ZkHidden.zk_session_soundHJ` over `VerifiesZK`, at every number of tables. -/
theorem zk_session_soundHJ
    (I : Inputs) (hh : I.tags.hiddenOutputs = true)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundHJ I hs.typed dj y₀ wr dg N (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ hh) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound, at the coin server's custody.** `ZkHidden.zk_session_soundHJ_custody` over `VerifiesZK`. -/
theorem zk_session_soundHJ_custody
    (I : Inputs) (hh : I.tags.hiddenOutputs = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf (circZ I).g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf (circZ I).g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's custody of the record and its draw file
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hCust.1
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj hCust.2 _)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hCust.1
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj hCust.2 _)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hCust.1
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj hCust.2 _)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundHJ_custody I hs.typed (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH kd hkd hdiv dj y₀ wr dg N Lb src hA3 (hs.tagsOkZHJ hh) hCust hk hRw hM hρ hr ht' K₀ τ (hCR hs)

/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound, at the record's custody and the printed draw file.** `ZkHidden.zk_session_soundHJ_custody_json` over `VerifiesZK`. -/
theorem zk_session_soundHJ_custody_json
    (I : Inputs) (hh : I.tags.hiddenOutputs = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf (circZ I).g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf (circZ I).g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, the coin server's custody of the record, and its draw file as `flock-verify draw`
    -- printed the verifier's draw
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (hs.parseH hh) (Layout.scopeOkH_spec hs.built hs.scopeH).2.1 dj y₀ wr _ (hs.tagsOkZHJ hh) hRec
          fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH hh) kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables).bindingZ (rsZHJ_computes dj y₀ hs.typed (hs.parseH hh) hs.scopeH hs.built (hs.loadH hh) hs.tables))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact FlockSoundness.Discharge.ZkHidden.zk_session_soundHJ_custody_json I hs.typed (hs.parseH hh) hs.built (hs.loadH hh) hs.tables hs.scopeH kd hkd hdiv dj y₀ wr dg N Lb src hA3 (hs.tagsOkZHJ hh) hRec hdj hk hRw hM hρ hr ht' K₀ τ (hCR hs)

end FlockSoundness.Discharge.FailClosed
