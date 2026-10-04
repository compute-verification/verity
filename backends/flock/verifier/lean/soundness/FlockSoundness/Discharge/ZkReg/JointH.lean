import FlockSoundness.Discharge.ZkReg.ValsH
import FlockSoundness.Discharge.ZkReg.Joint
import FlockSoundness.Discharge.ZkReg.Off

/-!
# The drawn units and the registered values of a hidden-output `--zk --registered` session

`zk_session_drawnRegValsR` and `zk_session_offR` at `zk_session_soundHR`'s session (hidden outputs, the rows `rsZHS`,
the shadow layout `layShH`), as `ValsH` is `Vals` there:

* `zk_session_drawnRegValsHR`: `zk_session_drawnHR`'s and `zk_session_regValsHR`'s events together, at one
  `ε_ks + δ_link`, at the committed values `zk_session_soundHR` reads;
* `zk_session_offHR`: no drawn unit reads a registered read whose leaf is not the registrant's, except with probability
  `qR²/2^513`, its check from `checked_of_acceptsH`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Composed Placed StrictCR DecodeZL
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **No drawn unit of a hidden-output `--zk --registered` session is wrong or off the registered values where it reads
the registrant's leaf, except with probability `ε_ks + δ_link + δ_reg`**: `zk_session_drawnRegValsR` at
`zk_session_soundHR`'s session, on `zk_session_regValsHR`'s assumptions. -/
theorem zk_session_drawnRegValsHR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHS dj y₀ ht hc hscope hU hpub))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
      ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
      (regOpenZH dj y₀ ht hc hscope hU hpub rd)
      (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
      (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        (¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
              ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
              (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
              (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw))
            o.2.1 ∨
          ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
            rd.isRead ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
            (layShH dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) =
              rd.leaf ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
            XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
              ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
              (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
              (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw g ≠
              (layShH dj y₀ ht hc hscope hU (pub := pub)).bit (rd.wv ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g)) g))
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH (Reg := Reg) dj y₀)
          (tabAtZKH dj y₀) ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW := by
  sorry

/-- **No drawn unit of a hidden-output `--zk --registered` session reads a registered read whose leaf is not the
registrant's**, except with probability `qR²/2^513`: `zk_session_offR` at `zk_session_soundHR`'s session. -/
theorem zk_session_offHR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the verify command's tags
    (hTags : TagsOkH I)
    -- the registered reads, the verifier's own at each accepted outcome's drawn units
    (rd : FlockSoundness.Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHS dj y₀ ht hc hscope hU hpub))
    (hown : ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      OwnReadsH own c pub o.2.2
        (rd.restrict (FlockSoundness.Registered.drawnReads (layShH dj y₀ ht hc hscope hU (pub := pub)) o.2.1)).isRead
        rd.port rd.readAt)
    -- the game, and cr/sha-512 for the registered-path finder
    (G : Game (Reg × Finset (Fin n) × Exec.Transcript)) (σ : Strategy G) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o =>
      (rd.restrict (FlockSoundness.Registered.drawnReads (layShH dj y₀ ht hc hscope hU (pub := pub)) o.2.1)).pick o.1
        (regPathsH c pub o.2.2 rd.port)).CR H512 qR) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          (layShH dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ≠
            rd.leaf ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g)) G σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.ZkReg
