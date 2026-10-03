import FlockSoundness.Discharge.ZkPubIn.Sound
import FlockSoundness.Assumptions.ZkPubIn

/-!
# One `--zk` session with the verifier's public inputs is sound, under the record's custody

`zk_session_soundHP_custody`, `zk_session_soundHRP_custody`: `zk_session_soundHP` and `zk_session_soundHRP` with their
two premises about the record, `hRec` (`RecordsLiveZKP`) and `hDraw` (`DrawOkZKHP` at each of `Law.execOS`'s draws),
replaced by one named assumption, `RecordCustodyZKP` (`live-verifier`), at `mPts := mPtsOf c.g (∑ s, ks s)`, the link
points flock-verify sets up at each of the law's draws (`drawOkZKHP_of_exec`), as `zk_session_soundH_custody` is
`zk_session_soundH`'s. They rest on five: `cr/sha-512` (`hCR`), `ecr/sha-512` (A2, inside `δ_link`), A3 (`hA3`), A6
and `live-verifier` (`hCust`).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session with the verifier's public inputs is sound, under `live-verifier`.** -/
theorem zk_session_soundHP_custody
    (I : Inputs) (ht : I.tags.typed = false) (own : ByteArray)
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOkH I)
    (hCust : Assumptions.ZkPubIn.RecordCustodyZKP I own dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKP I own o ∧
        K₀ ≤ (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
              (tabAtZKHP dj y₀)
              ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
              (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
              (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) :=
  zk_session_soundHP I ht own dj y₀ wr dg N hc hU hpi hpub hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr
    ht' K₀ τ hCR

/-- **One hidden-output `--zk --registered` sampled session with the verifier's public inputs is sound, and its
registered reads are checked, under `live-verifier`.** -/
theorem zk_session_soundHRP_custody
    (I : Inputs) (ht : I.tags.typed = false) (rports : Array Flock.Registered.Port) (own : ByteArray)
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOkH I)
    (hCust : Assumptions.ZkPubIn.RecordCustodyZKP I own dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKRP I rports own o ∧
        K₀ ≤ (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
              (tabAtZKHP dj y₀)
              ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
              (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
              (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hCust.1
          (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKRP I rports own o →
      ∀ rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU (pi := pi) (pub := pub))
          (rsShHP dj y₀ ht hc hscope hU hpi hpub),
        OwnReadsHP rports c pi pub o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsHP c pi pub o.2.2 rd.port) :=
  zk_session_soundHRP I ht rports own dj y₀ wr dg N hc hU hpi hpub hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr
    ht' K₀ τ hCR

end FlockSoundness.Discharge.ZkPubIn
