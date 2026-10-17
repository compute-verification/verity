import Proofs.Flock.Soundness.Discharge.ZkHidden.BindInst
import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ

set_option Elab.async false

/-!
# One hidden-output `--zk` sampled session of `J` tables, at the template instance's program

`zk_session_soundHJ`, `zk_session_soundHJ_custody`, `zk_session_soundHJ_custody_reg` and
`zk_session_soundHJ_custody_drawn_reg` at `instProg c n` in place of `Placed.keyProg c hU n`: the units are those of
the whole template instance's program, partitioned at the output words' bits (`outsOfI`), each read at its site in the
hidden-output table it is drawn into (`placeZHJI`) and bound to the rows it commits, input or output (`rsZHJI`).

In place of `scopeOkH` and `pub.tables = none`, the circuit conditions are the rows' fit (`hrf`), the unit's reads
(`hpr`) and the wires' order (`hw`): none on the unit's input groups, on where the outputs are returned (the live
tables' runs, `runsZCJH_liveI`, read none) or on shared-row tables.

* `instProg_readsOwn`: `instProg`'s units read only their own gates and input gates (`instProg_reads`).
* `rsZHJI_wrongRegZ`: `wrongRegZ` at `rsZHJI` is one set at every registration.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockSoundness.Discharge.Inst

/-- **`instProg`'s units read only their own gates and input gates.** -/
theorem instProg_readsOwn (c : Flock.Circuit) (n : ℕ) (outs : Finset (Fin (instN c n))) (u : Fin n) :
    ((instProg c n).partition outs).ReadsOwn u :=
  Layout.readsOwn_of_rows (fun u => Const.Prog.instC (instProg c n) outs u)
    (fun u y hy => instProg_reads outs u y hy) u

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at `instProg`.** `zk_session_soundHJ` with the
template instance's program, its circuit conditions `hrf`, `hpr` and `hw`. -/
theorem zk_session_soundHJI
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hRec hDraw)) τ _ _ ?_
  refine (zk_flock_countC execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

/-- **One hidden-output `--zk` sampled session of `J` tables is sound at the coin server's custody, at `instProg`.**
`zk_session_soundHJ_custody` with the template instance's program. -/
theorem zk_session_soundHJ_custodyI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
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
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJI I ht dj y₀ wr dg N hc hU hpub hrf hpr hw kd hkd Lb src hA3
    (fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

/-- **The registered commit strings are the public file's**, at `instProg`: the rows' value binding (`rsZHJI`) reads
the registration only through the public file `pub`, so its `wrongRegZ` is one set at every registration. -/
theorem rsZHJI_wrongRegZ (dj : Finset (Fin n) → Lean.Json) {I : Inputs} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hU : Built c.unit) (hpr : Layout.PortReads c hU)
    (hw : ProvedScope.wiresAcyclic c = true) (R R' : Reg) (Xv : Fin (instN c n) → Bool) :
    wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) R Xv =
      wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) R' Xv := by
  apply wrongRegZ_congr
  intro p
  rfl

/-- **One hidden-output `--zk` sampled session of `J` tables is sound against the registration, at the coin server's
custody, at `instProg`.** `zk_session_soundHJ_custody_reg` with the template instance's program: "wrong" is
`wrongRegZ` at the prover's registration `reg σ`, read through the rows' value binding (`rsZHJI`). -/
theorem zk_session_soundHJ_custody_regI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
        fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ _ _ ?_
  refine (zk_flock_countC_reg execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    (instProg_readsOwn c n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

/-- **One hidden-output `--zk` sampled session of `J` tables is sound at the units it draws, at `instProg`.**
`zk_session_soundHJ_custody_drawn_reg` with the template instance's program: the executable verifier accepts and some
unit the session drew is in `wrongRegZ`. The draw is the session's, so no law over it is assumed. -/
theorem zk_session_soundHJ_custody_drawn_regI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw))
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  let W := wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
    ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) (reg σ)
    (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
      (reg σ) (cont σ) k Rw)
  refine (execRefinesZK_tables_drawn execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
        fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (fun S => ∃ u ∈ S, u ∈ W)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin n) => o.1 = true ∧ ¬ Disjoint W o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_drawnC_reg execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    (instProg_readsOwn c n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' σ).trans ?_
  exact add_le_add (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR) le_rfl

end FlockSoundness.Discharge.ZkHidden
