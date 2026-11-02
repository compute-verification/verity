import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
import Proofs.Flock.Soundness.Discharge.Composed.Sound
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# The drawn units of one `--zk --registered` session

`exec_le_of_refines` and `zk_headline_exec_on` carry a bound on the compiled audit to the live audit only for a fact
`Q : Prop` the strategies fix. An event about the draw (`o.2.1`) and the registration (`o.1`) carries over too, since in
both audits the registration is the prover's first message and the draw the verifier's coin after it, and the
simulation keeps both (`simAuditZKOn`):

* `exec_le_of_refines_drawn`: draw by draw, if the live session's acceptance is at most the compiled session's, then
  any bound on `Pr[accept ∧ Q R S]` in the compiled audit, `R` its registration and `S` its draw, bounds
  `Pr[acc ∧ Q o.1 o.2.1]` in the live audit;
* `zk_headline_exec_drawn`: that at `--zk`, from the decoder at the law's draws; `zk_headline_exec_drawnJ` at sessions
  of `J` tables, and `simZHJ` the simulation it reads at hidden-output tables;
* `zk_flock_drawnC_cr`: the compiled audit's drawn bound, `Pr[accept ∧ a drawn unit is wrong] ≤ ε_ks + δ_link`
  (`extraction_audit_drawn` at `analysisZC`), every term on the accepted assumptions;
* `zk_session_drawnR`: `hwrong` for `flock-verify verify --zk --registered F` sessions: no drawn unit is wrong in the
  committed values `zk_session_soundR` reads, except with probability `ε_ks + δ_link`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open scoped ENNReal

section Transfer

variable {n : ℕ} {Reg : Type}

/-- **A bound on the audit carries to the live audit at an event of the registration and the draw.** If after each
draw the live session's event `acc` is at most as likely as the compiled session's acceptance (`hat`), the registrations
agreeing (`hreg`), then a bound on the compiled audit's `Pr[accept ∧ Q R S]` (`R` the registration, `S` the draw)
bounds the live audit's `Pr[acc ∧ Q o.1 o.2.1]`. -/
theorem exec_le_of_refines_drawn {L : Law n} {live : Finset (Fin n) → Reg → Game Exec.Transcript}
    {session : Finset (Fin n) → Reg → Game Bool} (τ : Strategy (auditLive L Reg live))
    (σ : Strategy (audit L Reg session)) (hreg : reg σ = τ.1) (acc : Exec.Transcript → Prop)
    (hat : ∀ ω, prob acc (live (L.draw ω) τ.1) (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)) ≤
      prob (fun ok => ok = true) (session (L.draw ω) (reg σ)) (cont σ ω))
    (Q : Reg → Finset (Fin n) → Prop) {B : ℝ≥0∞}
    (hB : prob (fun o => o.1 = true ∧ Q (reg σ) o.2) (audit L Reg session) σ ≤ B) :
    prob (fun o => acc o.2.2 ∧ Q o.1 o.2.1) (auditLive L Reg live) τ ≤ B := by
  classical
  obtain ⟨R, τ₂⟩ := τ
  obtain ⟨R', σ₂⟩ := σ
  obtain rfl : R' = R := hreg
  refine le_trans ?_ hB
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => acc o.2.2 ∧ Q o.1 o.2.1) ((live (L.draw ω) R').map fun t => (R', L.draw ω, t)) (τ₂ ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true ∧ Q R' o.2) ((session (L.draw ω) R').map fun ok => (ok, L.draw ω)) (σ₂ ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  rw [prob_map, prob_map]
  show prob (fun t => acc t ∧ Q R' (L.draw ω)) _ _ ≤ prob (fun ok => ok = true ∧ Q R' (L.draw ω)) _ _
  rw [prob_and_const, prob_and_const]
  split_ifs
  · exact hat ω
  · exact le_rfl

end Transfer

section On

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}

/-- **The `--zk` headline at an event of the registration and the draw**: a bound on the compiled batched audit's
`Pr[accept ∧ Q R S]` at `simZ … τ` bounds flock-verify `--zk`'s `Pr[accept ∧ Q o.1 o.2.1]`
(`zk_headline_exec_on` for a `Q` that reads the outcome). -/
theorem zk_headline_exec_drawn (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Reg → Finset (Fin n) → Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q (reg (simZ A Hs T I wr opn dg dec one fin capOf N hdec τ)) o.2)
      (audit L Reg (batchedSessionZC A Hs (planOf T))) (simZ A Hs T I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q o.1 o.2.1) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ Q (reg (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
          (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ))) o.2) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ Q (reg (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)) o.2)
          (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  have hB' := (hc _ (sessionZC_planOf A Hs T)).symm.trans_le hB
  refine exec_le_of_refines_drawn τ (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) rfl (AcceptsZK I)
    (fun ω => ?_) Q hB'
  have h : prob (LiveAcceptsZK I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
      prob (fun o => o.1 = true)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ).2 ω) := by
    unfold simAuditZKOn ZkExec.sessionZC
    rw [prob_strategy_map, prob_strategy_map, prob_map]
    unfold liveOf
    rw [prob_map]
    exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N
      (Gz A Hs T (L.draw ω) τ.1) (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _
  rw [prob_map] at h
  exact h.trans_eq (prob_map _ _ _ _)

end On

section OnJ

open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) {L : Law n}

/-- **The `--zk` headline at an event of the registration and the draw, for sessions of `J` tables**
(`zk_headline_exec_drawn` at `simZJ`). -/
theorem zk_headline_exec_drawnJ (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (Q : Reg → Finset (Fin n) → Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q (reg (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ)) o.2)
      (audit L Reg (batchedSessionZC A Hs planZ)) (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q o.1 o.2.1) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤ B := by
  refine exec_le_of_refines_drawn τ (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) rfl (AcceptsZK I)
    (fun ω => ?_) Q hB
  have h : prob (LiveAcceptsZK I) ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
      prob (fun o => o.1 = true) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ).2 ω) := by
    unfold simZJ batchedSessionZC
    rw [prob_strategy_map, prob_strategy_map, prob_map]
    unfold liveOfZJ
    rw [prob_map]
    refine (live_le_sessionZJ A Hs (planZ (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
      (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
    congr 1
    funext outs
    classical
    exact propext decide_eq_true_iff.symm
  rw [prob_map] at h
  exact h.trans_eq (prob_map _ _ _ _)

end OnJ

section Hidden

open ZkHidden (planZHJ DrawSetupZHJ TagsOkZHJ DrawOkZKHJ opnAtZHJ runsZCJH_live)
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {mPts n : ℕ} {Reg : Type}

/-- **The simulation of a hidden-output `--zk` session of `J` tables**: `simZJ` at the tables `planZHJ`, with the decoder
`runsZCJH_live` gives. -/
noncomputable abbrev simZHJ {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) {L : Law n} (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω))
    (τ : Strategy (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr))) :
    Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) :=
  simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
    (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr L hTags hRec hDraw)) τ

end Hidden

section Drawn

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Audit.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The compiled `--zk` audit's drawn units are right, every term on the accepted assumptions**
(`extraction_audit_drawn` at `analysisZC`, `zk_flock_countC_execOS_cr`'s terms): with each drawn unit read from its
table at its place (`place`) and the committed values the plurality value layer of `vb`,
`Pr[accept ∧ some drawn unit is wrong] ≤ ε_ks + δ_link`, `ε_ks = ksAvgStrictZ` under A1 for every table's finders after
each draw (`hCR`) and `δ_link = linkBoundZC` under A2 for the link finders. -/
theorem zk_flock_drawnC_cr (hA : A.Correct) {L : Law n}
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  exact (extraction_audit_drawn
      (analysisZC A Hs P planZ tab (decodeOfPlace A P planZ tab place)
        (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
      (zc_knowledgeSound A Hs P planZ tab _ _ _ hlow hj)
      (zk_linkSoundC A Hs P planZ tab vb hlow hRw hM hρ hr hj ht) σ).trans
    (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl)

end Drawn

variable {mPts n : ℕ} {Reg : Type}

/-- **No drawn unit of a `--zk --registered` session is wrong, except with probability `ε_ks + δ_link`**: in
`flock-verify verify --zk --registered F` sessions, `own` F's ports, at the committed values `zk_session_soundR` reads
(`XplurZC` at `simZ … τ`), `Pr[accept ∧ some drawn unit is wrong] ≤ ε_ks + δ_link`, on `zk_session_soundR`'s
assumptions without A3 (the law's count curve is not read). This is the `hwrong` a registered-reads core theorem takes. -/
theorem zk_session_drawnR
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
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw))
          o.2.1)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)
  have hc' := zk_flock_drawnC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀)
    (tabZAt dj y₀) execArith_correct ((sitesZAt ht dj y₀).placeKey hU (sitesZAt_parse ht dj y₀ hc) (Layout.outsOf c hU n))
    ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope)) hk hRw hM hρ hr ht'
    (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawn execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun _ S => ¬ Disjoint
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ)) (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ))
          k Rw)) S) _ hc')

end FlockSoundness.Discharge.ZkReg
