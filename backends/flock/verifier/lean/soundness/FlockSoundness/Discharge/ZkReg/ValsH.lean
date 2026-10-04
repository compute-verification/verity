import FlockSoundness.Discharge.ZkReg.Vals
import FlockSoundness.Discharge.ZkReg.SoundH

/-!
# The drawn units and the registrant's values of a hidden-output `--zk --registered` session

`zk_session_drawnR` and `zk_session_regValsR` at `zk_session_soundHR`'s session: `flock-verify verify --zk --registered F`
sessions over hidden outputs (`Stmt.setupHidden`), at the committed values `zk_session_soundHR` reads (`XplurZC` at
`simZ … τ`, the rows `rsZHS`), with the registered reads of the shadow layout `layShH`. The generic layers
(`zk_headline_exec_drawn`, `zk_flock_drawnC_cr`, `zk_flock_regValsC_cr`) are the same; only the session's plan, tables,
places and rows change.

* `zk_session_drawnHR`: no drawn unit is wrong, except with probability `ε_ks + δ_link`; the `hwrong` of the bridge's
  `FlockBridge.Registered.registered_core` (`bridge/FlockBridge/Registered.lean`), with its `Off` false.
* `zk_session_regValsHR`: wherever a drawn unit reads the registrant's leaf at a registered read, the committed value is
  the registrant's, except with probability `ε_ks + δ_link + δ_reg`: the bridge's
  `FlockBridge.RegisteredReads.RegValsSound` (`bridge/FlockBridge/RegisteredReads.lean`) at `layShH`'s reads `rd`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Composed Placed StrictCR DecodeZL
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The registrant's opening** of each commit string of `rsZHS`'s value binding: the row and the salt `rd` reads
there. -/
def regOpenZH {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOk c = true)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (rd : FlockSoundness.Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHS dj y₀ ht hc hscope hU hpub)) :
    ((rsZHS (Reg := Reg) dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub)).Pos →
      ((rsZHS (Reg := Reg) dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub)).Opening :=
  fun p => (p, rd.wv p, rd.ws p)

/-- **No drawn unit of a hidden-output `--zk --registered` session is wrong, except with probability
`ε_ks + δ_link`**: `zk_session_drawnR` at `zk_session_soundHR`'s session, on `zk_session_soundHR`'s assumptions without
A3 (the law's count curve is not read). -/
theorem zk_session_drawnHR
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
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw))
          o.2.1)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)
  have hc' := zk_flock_drawnC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
    (tabAtZKH dj y₀) execArith_correct (placeZH dj y₀ ht hc hscope hU) ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub)) hk hRw hM hρ hr ht'
    (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawn execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun _ S => ¬ Disjoint
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀) ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) k Rw)) S) _ hc')

/-- **The committed values of a hidden-output `--zk --registered` session are the registered values wherever a drawn
unit reads the registrant's leaf**, except with probability `ε_ks + δ_link + δ_reg`: `zk_session_regValsR` at
`zk_session_soundHR`'s session, for every registered-reads record `rd` of `layShH`, on `zk_session_soundHR`'s
assumptions without A3 and with `cr/sha-512` at budget `qW` for the registered-value finders of `τ`'s simulation
(`hW`). -/
theorem zk_session_regValsHR
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
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          (layShH dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) =
            rd.leaf ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw g ≠
            (layShH dj y₀ ht hc hscope hU (pub := pub)).bit (rd.wv ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g)) g)
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
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)
  have hc' := zk_flock_regValsC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
    (tabAtZKH dj y₀) execArith_correct (placeZH dj y₀ ht hc hscope hU) ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
    (regOpenZH dj y₀ ht hc hscope hU hpub rd) hk hRw hM hρ hr ht' (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawn execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun R S => ∃ u ∈ S,
      ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        (layShH dj y₀ ht hc hscope hU (pub := pub)).regLeaf R ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) = rd.leaf ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀) ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) k Rw g ≠
          (layShH dj y₀ ht hc hscope hU (pub := pub)).bit (rd.wv ((layShH dj y₀ ht hc hscope hU (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, u, hu, g, hg, hread, hleaf, hne⟩
  refine ⟨hacc, u, hu, g, hg, ?_, hne⟩
  refine Prod.ext rfl ?_
  show ((rsZHS dj y₀ ht hc hscope hU hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZ.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

end FlockSoundness.Discharge.ZkReg
