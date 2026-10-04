import FlockSoundness.Discharge.ZkReg.ValsHP
import FlockSoundness.Discharge.ZkReg.Joint
import FlockSoundness.Discharge.ZkReg.Off

/-!
# The drawn units and the registered values of a `--zk --registered --public-inputs` session

`zk_session_drawnRegValsR` and `zk_session_offR` at `ZkPubIn.zk_session_soundHRP`'s session, at the public transcript
`Xpub (pubOnes …) (pubZeros …) XplurZC` at `simZA … τ` that `zk_session_soundHRP` reads, as `ValsHP` is `Vals` there:

* `zk_flock_drawnRegValsC_cr_pub`: `zk_flock_drawnRegValsC_cr` at the public transcript, under `ZkPubIn.PubDecodesZ`;
* `zk_session_drawnRegValsHRP`: `zk_session_drawnHRP`'s and `zk_session_regValsHRP`'s events together, at one
  `ε_ks + δ_link`; the bridge's `FlockBridge.RegisteredReads.RegJointSound` (`bridge/FlockBridge/RegisteredReads.lean`)
  at `layShHP`'s reads `rd`;
* `zk_session_offHRP`: no drawn unit reads a registered read whose leaf is not the registrant's, except with
  probability `qR²/2^513`, its check from `checked_of_acceptsHP`.
-/

namespace FlockSoundness.Discharge.ZkReg

open scoped ENNReal

section Generic

open Game Model Accounting Extract Audit Audit.Partition ZkSession ZkLink StrictCR ZkPubIn

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Audit.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The compiled `--zk` audit's drawn units are right and agree with the registered values at the public transcript,
at one `ε_ks + δ_link`** (`zk_flock_drawnRegValsC_cr` at `Xpub ones zeros XplurZC`, under `PubDecodesZ` at the places'
decoder). -/
theorem zk_flock_drawnRegValsC_cr_pub (hA : A.Correct) {L : Law n}
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (wo : vb.Pos → vb.Opening)
    (ones zeros : Finset (Fin C.N))
    (hpub : PubDecodesZ A P planZ tab (decodeOfPlace A P planZ tab place) ones zeros)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {qW : ℝ} (hW : RegCRZC A Hs P planZ tab vb wo (reg σ) (cont σ) k Rw qW) :
    prob (fun o => o.1 = true ∧
        (¬ Disjoint (P.wrong (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))) o.2 ∨
          ∃ u ∈ o.2, ∃ g ∈ P.io u, RegBadZ A P planZ tab vb wo (reg σ)
            (Xpub ones zeros (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) g))
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regTermZC A P planZ tab vb L k Rw ρ qW := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisRegZC A Hs P planZ tab vb wo (fun R τ => Xpub ones zeros (XplurZC A Hs P planZ tab vb R τ k Rw)) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (reg_knowledgeSoundZC A Hs P planZ tab vb wo _ _ hlow hj)
    (zk_linkSoundRegC_pub A Hs P planZ tab vb wo ones zeros hpub hlow hRw hM hρ hr hj ht qW) σ
  have hb : regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ) =
      regTermZC A P planZ tab vb L k Rw ρ qW := ite_eq_left hW
  refine le_trans (prob_mono ?_ _ _) (h.trans ?_)
  · rintro o ⟨h1, hw | ⟨u, hu, g, hg, hbad⟩⟩
    · obtain ⟨u, hw, hu⟩ := Finset.not_disjoint_iff.1 hw
      exact ⟨h1, Finset.not_disjoint_iff.2 ⟨u, Finset.mem_union_left _ hw, hu⟩⟩
    · exact ⟨h1, Finset.not_disjoint_iff.2 ⟨u, Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, hbad⟩), hu⟩⟩
  · show ksAvgZC A Hs planZ k _ (reg σ) (cont σ) +
        (linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
          regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ)) ≤ _
    rw [hb, ← add_assoc]
    exact add_le_add (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl) le_rfl

end Generic

section Session

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden ZkPubIn

variable {mPts n : ℕ} {Reg : Type}

/-- **No drawn unit of a `--zk --registered --public-inputs` session is wrong or off the registered values where it
reads the registrant's leaf, at its public transcript, except with probability `ε_ks + δ_link + δ_reg`**:
`zk_session_drawnRegValsR` at `zk_session_soundHRP`'s session, on `zk_session_regValsHRP`'s assumptions. -/
theorem zk_session_drawnRegValsHRP
    (I : Inputs) (ht : I.tags.typed = false) (rports : Array Flock.Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHP I own dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZKP I own dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
      ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
      (regOpenZHP dj y₀ ht hc hscope hU hpi hpub rd)
      (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
      (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKRP I rports own o ∧
        (¬ Disjoint (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
            (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
              (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
                ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
                (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
                (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)))
            o.2.1 ∨
          ∃ u ∈ o.2.1, ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
            rd.isRead ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
            (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf o.1 ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) =
              rd.leaf ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
            (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
              (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
                ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
                (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
                (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)) g ≠
              (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).bit (rd.wv ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g)) g))
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planAtZKHP dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        regTermZC execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP (Reg := Reg) dj y₀)
          (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW := by
  have hdec := decodesOnZKA_of_runsOnZCA execArith hs512 _ (AcceptsZKP I own) wr _ _ _ _
    (runsZC_liveHP I own ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKHP_live hTags hRec hDraw)
  have hc' := zk_flock_drawnRegValsC_cr_pub execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n))
    (planAtZKHP dj y₀) (tabAtZKHP dj y₀) execArith_correct (placeZHP dj y₀ ht hc hscope hU hpi)
    ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
    (regOpenZHP dj y₀ ht hc hscope hU hpi hpub rd)
    (pubOnes c hU n pi own) (pubZeros c hU n pi own) (pubDecodesZ_placeZHP dj y₀ ht hc hscope hU hpi hpub)
    hk hRw hM hρ hr ht' (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZKP_of_RP h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ (fun R S =>
      ¬ Disjoint (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀)
              (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
              (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ))
              (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) k Rw))) S ∨
      ∃ u ∈ S, ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
        (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf R ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) = rd.leaf ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
        (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
          (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planAtZKHP dj y₀) (tabAtZKHP dj y₀) ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N hdec τ)) k Rw)) g ≠
          (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).bit (rd.wv ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, hw | ⟨u, hu, g, hg, hread, hleaf, hne⟩⟩
  · exact ⟨hacc, Or.inl hw⟩
  refine ⟨hacc, Or.inr ⟨u, hu, g, hg, ?_, hne⟩⟩
  refine Prod.ext rfl ?_
  show ((rsZHP dj y₀ ht hc hscope hU hpi hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZ.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

/-- **No drawn unit of a `--zk --registered --public-inputs` session reads a registered read whose leaf is not the
registrant's**, except with probability `qR²/2^513`: `zk_session_offR` at `zk_session_soundHRP`'s session, in the live
audit under any law and live session, with `hown` at its outcomes (the shape of `ZkPubIn.ownAtDrawsHP_of_check`). -/
theorem zk_session_offHRP
    (I : Inputs) (ht : I.tags.typed = false) (rports : Array Flock.Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    -- the session's circuit, its public input ports and the public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the verify command's tags
    (hTags : TagsOkH I)
    -- the live audit: any law and live session
    (L : Audit.Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    -- the registered reads, the verifier's own at the draw of each accepted live outcome
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : ∀ (R : Reg) (ω : L.Ω) (lo : LiveOut mPts),
      LiveAcceptsZKRP I rports own (R, L.draw ω, wr (L.draw ω) R lo) →
      OwnReadsHP rports c pi pub (wr (L.draw ω) R lo)
        (rd.restrict
          (FlockSoundness.Registered.drawnReads (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)) (L.draw ω))).isRead
        rd.port rd.readAt)
    -- the live prover, and cr/sha-512 for the registered-path finder
    (σ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) {qR : ℝ}
    (hR : (StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) σ fun o =>
      (rd.restrict (FlockSoundness.Registered.drawnReads (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).pick
        o.1 (regPathsHP c pi pub o.2.2 rd.port)).CR H512 qR) :
    prob (fun o => LiveAcceptsZKRP I rports own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ∧
          (layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf o.1
              ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g) ≠
            rd.leaf ((layShHP dj y₀ ht hc hscope hU hpi (pub := pub)).pos g))
        (auditLive L Reg (liveOf dg mPts N wr)) σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end Session

end FlockSoundness.Discharge.ZkReg
