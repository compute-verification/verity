module

public import Proofs.Flock.EndToEnd.Joint.V5.Statement
public import Proofs.Flock.Bridge.V5.EndToEndHidden
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ

/-!
# A member's session alone: its sampled proof blocks at flock-verify `--zk --public-inputs`, at any sampling design

`SampledKSLiveV5`'s session with an event of the sample in place of the count curve, as
`zk_session_soundJ_custody_drawn_reg` is `zk_session_soundJ_custody_reg`'s:

* `execRefinesZK_tables_drawnA`: `execRefinesZK_tables_drawn` at any acceptance `acc` (`simZJA`);
* `ZkLink.zk_flock_drawnC_regC`: `Audit.extraction_audit_drawn` at `analysisZCRegC`, so in the compiled batched audit
  `Pr[accepts ∧ some drawn proof block is in wrongRegZc] ≤ ε_ks + δ_link`;
* `MemberV5.accepted_drawnWrong_le`: a member's session accepts at its `own` with a sampled proof block incorrect, at V5's
  canonical openings, with probability at most `ε_ks + ε_bind` at its sampling design. Custody at `own` and one table
  (`decodesZCJH_custodyAt`) holds at any sampling design whose samples have the session's link points, so the bound does
  too, and it needs no A3.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts))
  (acc : Exec.Transcript → Prop)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) {L : Law n}

/-- **`execRefinesZK_tables_drawn` at `acc`**: `acc` accepts the live audit at a sample in `E` no likelier than the
compiled batched audit, played by `simZJA … τ`, accepts at a sample in `E`. -/
theorem execRefinesZK_tables_drawnA (hdec : DecodesOnZKJA A Hs planZ acc wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (E : Finset (Fin n) → Prop) :
    prob (fun o => acc o.2.2 ∧ E o.2.1) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤
      prob (fun o => o.1 = true ∧ E o.2) (audit L Reg (batchedSessionZC A Hs planZ))
        (simZJA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ) := by
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => acc o.2.2 ∧ E o.2.1)
        ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true ∧ E o.2) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simZJA A Hs planZ acc wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simZJA batchedSessionZC
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

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (canon : Fin C.N → vb.Opening → Prop) {L : Law n}

/-- **The sampled proof blocks of the compiled `--zk` audit, against the registration with canonical openings**: it
accepts with a sampled proof block in `wrongRegZc` with probability at most `ε_ks + δ_link`, `zk_flock_countC_regC`'s
terms without the miss term. -/
theorem zk_flock_drawnC_regC (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (wrongRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))
          o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgZC A Hs planZ k a (reg σ) (cont σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_drawn (analysisZCRegC A Hs P planZ tab vb canon (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj)
    (zc_knowledgeSound_regC A Hs P planZ tab vb canon _ _ hlow hown hcan hj)
    (zk_linkSoundC_regC A Hs P planZ tab vb canon hlow hown hcan hRw hM hρ hr hj ht) σ

end FlockSoundness.Discharge.ZkLink

namespace Flock.Guarantees.MemberV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkSession
open scoped ENNReal

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

/-- **A member's session accepted with a sampled proof block incorrect**, played alone at the sampling design `L` by `τ`:
at most `ε_ks + ε_bind` at `L`. -/
theorem accepted_drawnWrong_le (L : Law m.n) (hL : m.OneClass L) (τ : Strategy (m.game L))
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZHJP m.dj m.y₀ (L.draw ω) (reg (m.sim L hL τ))) j
      (zcStrat execArith hs512 (planZHJP m.dj m.y₀) (cont (m.sim L hL τ) ω)) m.qF m.qS)
    (hA2 : LinkCRZC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb (reg (m.sim L hL τ))
      (cont (m.sim L hL τ)) m.k m.Rw m.M m.t') :
    prob (fun o => LiveAcceptsZKP m.I m.own o ∧ ∃ u ∈ o.2.1, u ∈ m.wrongSet L hL τ) (m.game L) τ ≤
      m.epsKs L + m.link L := by
  refine (execRefinesZK_tables_drawnA execArith hs512 (planZHJP m.dj m.y₀) (AcceptsZKP m.I m.own) m.wr
    (opnAtZHJP (planZHJP m.dj m.y₀) m.I m.own m.dj) m.dg (decAtZJ (planZHJP m.dj m.y₀)) (fun _ _ _ _ => true)
    (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) m.N
    (decodesZCJH_custodyAt m.I m.own m.ht m.hcS m.hcount _ m.dj m.y₀ m.wr L m.hTags m.hCust hL)
    τ (fun S => ∃ u ∈ S, u ∈ m.wrongSet L hL τ)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin m.n) => o.1 = true ∧ ¬ Disjoint (m.wrongSet L hL τ) o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_drawnC_regC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon
    (loweringSoundZK_of_place execArith _ (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀)
      (placeZHJP m.dj m.y₀ m.ht m.hcS m.hscope m.hU m.hpi))
    (Discharge.Layout.keyProgP_readsOwn _ m.hU m.n m.pi _)
    (Flock.Bridge.openingsCanonV5 m.n Reg m.I m.own m.ht m.D m.hDS m.hU m.pi m.hpi m.pub m.hpub m.hscope m.kd m.dj
      m.S₀ m.d₀ m.y₀)
    m.hRw m.hM m.hρ m.hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJP m.dj m.y₀) m.hk) m.ht'
    (m.sim L hL τ)).trans ?_
  refine add_le_add ((ksAvgZC_le_strict execArith hs512 (planZHJP m.dj m.y₀) m.k (reg (m.sim L hL τ))
    (cont (m.sim L hL τ)) hCR).trans (le_iSup (fun R => ksAvgStrictZ (planZHJP m.dj m.y₀) L m.k m.qF m.qS R) _)) ?_
  exact le_of_eq (ite_eq_left hA2)

end Flock.Guarantees.MemberV5
