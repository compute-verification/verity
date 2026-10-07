import Proofs.Flock.Soundness.Discharge.Composed.PublicJ

/-!
# The drawn units of one `--zk` session of `J` tables, at flock-verify's acceptance

`zk_session_soundJ_custody_reg` is the count curve: it bounds flock-verify's acceptance with "at least `K₀` units wrong",
an event the prover's strategy fixes before the draw, because `zk_headline_exec_tables` carries only such events. This
file carries an event of the draw too, and bounds the drawn units' correctness:

* `execRefinesZK_tables_drawn`: `execRefinesZK_tables` with an event `E` of the draw on both sides. Both audits'
  outcomes carry the draw, and the refinement is one inequality at each of the law's draws, where `E` is fixed;
* `ZkLink.zk_flock_drawnC_reg`: `Audit.extraction_audit_drawn` at `analysisZCReg`, so in the compiled batched audit
  `Pr[accepts ∧ some drawn unit is in wrongRegZ] ≤ ε_ks + δ_link`;
* `zk_session_soundJ_custody_drawn_reg`: the same at flock-verify's acceptance, at the coin server's custody, with
  `ksAvgStrictZ` for `ε_ks`. Its bound has no miss term, so it needs no A3.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) {L : Law n}

/-- **R11 at `--zk` for sessions of `J` tables, with an event of the draw**: flock-verify `--zk` accepts the live audit
at a draw in `E` no likelier than the compiled batched audit, played by `simZJ … τ`, accepts at a draw in `E`. -/
theorem execRefinesZK_tables_drawn (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (E : Finset (Fin n) → Prop) :
    prob (fun o => LiveAcceptsZK I o ∧ E o.2.1) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤
      prob (fun o => o.1 = true ∧ E o.2) (audit L Reg (batchedSessionZC A Hs planZ))
        (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) := by
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => LiveAcceptsZK I o ∧ E o.2.1)
        ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true ∧ E o.2) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simZJ batchedSessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOfZJ
  rw [prob_map]
  classical
  dsimp only
  simp only [prob_and_const]
  split_ifs
  · refine (live_le_sessionZJ A Hs (planZ (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
      (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
    congr 1
    funext outs
    exact propext decide_eq_true_iff.symm
  · exact le_rfl

end Session

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) {L : Law n}

/-- **The drawn units of the compiled `--zk` audit, against the registration**: it accepts with a drawn unit in
`wrongRegZ` with probability at most `ε_ks + δ_link`, `zk_flock_countC_reg`'s terms without the miss term. -/
theorem zk_flock_drawnC_reg (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (wrongRegZ A P planZ tab vb (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgZC A Hs planZ k a (reg σ) (cont σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_drawn (analysisZCReg A Hs P planZ tab vb (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj)
    (zc_knowledgeSound_reg A Hs P planZ tab vb _ _ hlow hown hj)
    (zk_linkSoundC_reg A Hs P planZ tab vb hlow hown hRw hM hρ hr hj ht) σ

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The drawn units of one `--zk` sampled session of `J` tables, against the public file, at the coin server's
custody.** For any live prover `τ`, flock-verify `--zk` accepts with some drawn unit in `wrongRegZ` with probability at
most `ε_ks + δ_link`: `zk_session_soundJ_custody_reg`'s terms without the miss term, so without A3. -/
theorem zk_session_soundJ_custody_drawn_reg
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) (reg σ)
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
            (reg σ) (cont σ) k Rw))
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  let W := wrongRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
    ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) (reg σ)
    (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
      ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
      (reg σ) (cont σ) k Rw)
  refine (execRefinesZK_tables_drawn execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg
    (decAtZJ (planZJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (fun S => ∃ u ∈ S, u ∈ W)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin n) => o.1 = true ∧ ¬ Disjoint W o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_drawnC_reg execArith hs512 _ (planZJ dj y₀) (tabZJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZJ dj y₀) (tabZJ dj y₀)
      ((sitesZJ dj y₀ ht).placeKey hU (sitesZJ_parse dj y₀ ht hc) (Layout.outsOf c hU n)))
    (Layout.keyProg_readsOwn c hU n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZJ dj y₀) hk) ht' σ).trans ?_
  exact add_le_add (ksAvgZC_le_strict execArith hs512 (planZJ dj y₀) k (reg σ) (cont σ) hCR) le_rfl

end FlockSoundness.Discharge.Composed
