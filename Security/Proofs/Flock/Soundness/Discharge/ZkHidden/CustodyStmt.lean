import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundStmt
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundCustody
import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJJson

set_option Elab.async false

/-!
# The custody forms at the session's statement circuit

`zk_session_soundH_custody`, `zk_session_soundHJ_custody` and `zk_session_soundHJ_custody_json` at `HmOut.parseStmt`
(`hcS`, a templated file's committed circuit unrolled), with the rows' fit `hv1` given: `zk_session_soundHSt` and
`zk_session_soundHJSt` with the record's custody in place of `hRec` and `hDraw`, the draws from `drawOkZKH_of_execSt`
and `drawOkZKHJ_of_fileSt`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session is sound, under `live-verifier`**, at the session's statement circuit
(`zk_session_soundH_custody` at `HmOut.parseStmt`). -/
theorem zk_session_soundH_custodySt
    (I : Inputs) (ht : I.tags.typed = false)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (hv1 : Hidden.Any.RowsFitS c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOkH I) (hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveHSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_liveSt hTags hCust.1
          (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) := by
  exact zk_session_soundHSt I ht dj y₀ wr dg N hcS hv1 hU hpub hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKH_of_execSt I ht hcS σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr ht' K₀
    τ hCR

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at the coin server's custody**, at the session's
statement circuit (`zk_session_soundHJ_custody` at `HmOut.parseStmt`). -/
theorem zk_session_soundHJ_custodySt
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (hv1 : Hidden.Any.RowsFitS c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and the coin server's custody of the record and its draw file
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_fileSt I ht hcS kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJSt I ht dj y₀ wr dg N hcS hv1 hU hpub hscope kd hkd Lb src hA3
    (fun _ => drawOkZKHJ_of_fileSt I ht hcS kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at the record's custody and the printed draw
file**, at the session's statement circuit (`zk_session_soundHJ_custody_json` at `HmOut.parseStmt`). -/
theorem zk_session_soundHJ_custody_jsonSt
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (hv1 : Hidden.Any.RowsFitS c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, the coin server's custody of the record, and its draw file as `flock-verify draw`
    -- printed the verifier's draw
    (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hRec
          fun _ => drawOkZKHJ_of_fileSt I ht hcS kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJ_custodySt I ht hcS hv1 hU hpub hscope kd hkd hdiv dj y₀ wr dg N Lb src hA3 hTags
    ⟨hRec, drawFileZK_of_toJson_subset dj kd n hdj⟩ hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.ZkHidden
