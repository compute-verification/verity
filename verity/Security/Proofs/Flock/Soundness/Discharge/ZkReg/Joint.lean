import Proofs.Flock.Soundness.Discharge.ZkReg.Vals

/-!
# The drawn units and the registered values of a `--zk --registered` session, in one event

`zk_session_drawnR` and `zk_session_regValsR` bound their events apart, each at `ε_ks + δ_link` (and the second at
`δ_reg` more). Both come from one extraction (`extraction_audit_drawn` at `analysisRegZC`), whose wrong units are the
drawn wrong units together with the units reading a registered-value disagreement, so their union costs
`ε_ks + δ_link + δ_reg` once:

* `zk_flock_drawnRegValsC_cr`: `zk_flock_regValsC_cr` with the drawn wrong units as a second disjunct;
* `zk_session_drawnRegValsR`: in `flock-verify verify --zk --registered F` sessions, at the committed values
  `zk_session_soundR` reads, an accepted session draws a wrong unit or a unit reading a registered read whose session
  leaf is the registrant's and whose committed value is not the registrant's, with probability at most
  `ε_ks + δ_link + δ_reg`. Its event is the bridge's `FlockBridge.RegisteredReads.RegJointSound`
  (`bridge/FlockBridge/RegisteredReads.lean`) at `laySh`'s registered reads `rd`, with `Xc` the committed values.
-/

namespace FlockSoundness.Discharge.ZkReg

section Generic

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
variable {Dh : Type} {Hc : List UInt8 → Dh} {L : Law n}

/-- **The compiled `--zk` audit's drawn units are right and agree with the registered values, at one
`ε_ks + δ_link`.** On `zk_flock_regValsC_cr`'s assumptions,
`Pr[accept ∧ (a drawn unit is wrong ∨ a drawn unit reads a wire disagreeing with its registered value)]
≤ ε_ks + δ_link + δ_reg`. -/
theorem zk_flock_drawnRegValsC_cr (hA : A.Correct)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (wo : vb.Pos → vb.Opening)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {qW : ℝ} (hW : RegCRZC A Hs P planZ tab vb wo (reg σ) (cont σ) k Rw qW) :
    prob (fun o => o.1 = true ∧
        (¬ Disjoint (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) o.2 ∨
          ∃ u ∈ o.2, ∃ g ∈ P.io u,
            RegBadZ A P planZ tab vb wo (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) g))
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regTermZC A P planZ tab vb L k Rw ρ qW := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisRegZC A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (reg_knowledgeSoundZC A Hs P planZ tab vb wo _ _ hlow hj)
    (zk_linkSoundRegC A Hs P planZ tab vb wo hlow hRw hM hρ hr hj ht qW) σ
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

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **No drawn unit of a `--zk --registered` session is wrong or off the registered values where it reads the
registrant's leaf, except with probability `ε_ks + δ_link + δ_reg`**: `zk_session_drawnR`'s and
`zk_session_regValsR`'s events together, on `zk_session_regValsR`'s assumptions, at one `ε_ks + δ_link`. -/
theorem zk_session_drawnRegValsR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
      ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
      (regOpenZ ht dj y₀ hc hU hpub hscope rd)
      (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
      (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        (¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
              ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
              (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
              (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw))
            o.2.1 ∨
          ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
            rd.isRead ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ∧
            (laySh ht dj y₀ hc hU (pub := pub)).regLeaf o.1 ((laySh ht dj y₀ hc hU (pub := pub)).pos g) =
              rd.leaf ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ∧
            XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
                (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
                (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw g ≠
              (laySh ht dj y₀ hc hU (pub := pub)).bit (rd.wv ((laySh ht dj y₀ hc hU (pub := pub)).pos g)) g))
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt (Reg := Reg) dj y₀)
          (tabZAt dj y₀) ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)
  have hc' := zk_flock_drawnRegValsC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n))
    (planZAt dj y₀) (tabZAt dj y₀) execArith_correct
    ((sitesZAt ht dj y₀).placeKey hU (sitesZAt_parse ht dj y₀ hc) (Layout.outsOf c hU n))
    ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
    (regOpenZ ht dj y₀ hc hU hpub hscope rd) hk hRw hM hρ hr ht'
    (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawn execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun R S =>
      ¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) k Rw)) S ∨
      ∃ u ∈ S, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ∧
        (laySh ht dj y₀ hc hU (pub := pub)).regLeaf R ((laySh ht dj y₀ hc hU (pub := pub)).pos g) =
          rd.leaf ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ∧
        XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) k Rw g ≠
          (laySh ht dj y₀ hc hU (pub := pub)).bit (rd.wv ((laySh ht dj y₀ hc hU (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, hw | ⟨u, hu, g, hg, hread, hleaf, hne⟩⟩
  · exact ⟨hacc, Or.inl hw⟩
  refine ⟨hacc, Or.inr ⟨u, hu, g, hg, ?_, hne⟩⟩
  refine Prod.ext rfl ?_
  show ((rsZAtS ht dj y₀ hc hU hpub hscope).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZ.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

end FlockSoundness.Discharge.ZkReg
