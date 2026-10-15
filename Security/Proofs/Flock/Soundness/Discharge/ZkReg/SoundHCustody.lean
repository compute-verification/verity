import Proofs.Flock.Soundness.Discharge.ZkReg.SoundH
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundCustody

/-!
# One hidden-output `--zk --registered` sampled session, under the record's custody

`zk_session_soundHR_custody`: `zk_session_soundHR` with `hRec` and `hDraw` replaced by `RecordCustodyZK`
(`live-verifier`), exactly as `ZkHidden.zk_session_soundH_custody` replaces them in `zk_session_soundH`: the custody
assumption reads only the record and its draws, which neither the shared rows nor the registered check change.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Composed Placed StrictCR DecodeZL
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk --registered` sampled session is sound under `live-verifier`, and its registered reads
are checked.** `zk_session_soundHR` at the link points of the law's own draws, `mPtsOf c.g (∑ s, ks s)`, with the
record's custody (`RecordCustodyZK`) in place of `hRec` and `hDraw`, as `zk_session_soundH_custody` is
`zk_session_soundH`'s. -/
theorem zk_session_soundHR_custody
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
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
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hCust.1
          (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
          (rsShHS dj y₀ ht hc hscope hU hpub),
        OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) :=
  zk_session_soundHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKH_of_exec I hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr ht' K₀
    τ hCR

end FlockSoundness.Discharge.ZkReg
