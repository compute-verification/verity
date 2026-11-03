import Proofs.Flock.Soundness.Discharge.PlainHJ.Decode
import Proofs.Flock.Soundness.Discharge.PlainHJ.Bind
import Proofs.Flock.Soundness.Discharge.Tables.Headline

set_option Elab.async false

/-!
# The soundness half of one plain sampled session of `J` hidden-output tables

For any live prover `τ` of a hidden-output plain session of `J ≥ 1` tables
(`flock-verify verify --plain-tables --session-tables J`, no `--zk`, `J = nTab I`),
`Pr[flock-verify accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`, each unit read at its site in the table
it is drawn into (`sitesPHJI`, `tabPHJ`) and bound to the rows it commits, outputs included, at the simulated prover
`simAuditJ … τ` in the batched session of the `J` tables `planPHJ`:

* `plain_session_soundHJ`, where the wires feed neither the unit nor the outputs: the units the unit's key program's
  (`Placed.keyProg c hU n`, `rsPHJ`), the circuit condition `scopeOkH`, as `HiddenExec.flock_headline_exec_hidden`;
* `plain_session_soundHJI`, at the template instance's program (`instProg c n`, `rsPHJI`), its circuit conditions those
  of `HiddenExec.flock_headline_exec_hidden_inst`: the rows fit (`hrf`), every input a unit row reads is a port bit
  (`hpr`), and `wiresAcyclic` passed (`hw`).

Each is `PlainJ.plain_session_soundJ` with the hidden-output tables and rows in place of the public ones
(`Prog.flock_headline_live_tables_subset_missX`). The tags, the record and the draws are `TagsOkZHJ` (hidden outputs,
retained rounds, `hm96-sha512/v1` leaves), `DecodeZLJ.RecordsLiveJ` and `DrawOkPHJ`, the coin server's draw setting each
table up with setupHidden at its part.

The assumptions are `flock_headline_exec`'s: A3 (`hA3`), `cr/sha-512` for the knowledge finders (`hKS`) and
`ecr/sha-512` for the link finders (`hCR`), with A6 the live session itself.
-/

namespace FlockSoundness.Discharge.PlainHJ

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.Tables
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ)
open FlockSoundness.Discharge.DecodeZLJ (decJ RecordsLiveJ)
open FlockSoundness.Discharge.PlainJ (inputsPJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **One plain sampled session of `J` hidden-output tables is sound**, where the wires feed neither the unit nor the
outputs: at the unit's key program. -/
theorem plain_session_soundHJ
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planPHJ (Reg := Reg) dj x₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file, and the circuit condition
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkPHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveJ I dj (fun S R => (planPHJ (Reg := Reg) dj x₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planPHJ (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPHJ dj x₀) N wr)))
    {qF qS : ℝ} :
    let σ := simAuditJ H512 enc512 leaf512 (planPHJ dj x₀) I wr (opnPHJ I (planPHJ dj x₀) dj) dg
      (decJ (planPHJ dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnPHJ_exec dj x₀ ht hc _ wr hTags hRec hDraw) τ
    -- ecr/sha-512 for the link finders of its simulation
    LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planPHJ dj x₀)
      (tabPHJ dj x₀) ((rsPHJ dj x₀ ht hc hpub hU hscope).binding (rsPHJ_computes dj x₀ ht hc hpub hU hscope))
      (reg σ) (cont σ) k Rw M t' →
    -- cr/sha-512 for its knowledge finders
    (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planPHJ dj x₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 (planPHJ dj x₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planPHJ dj x₀)
              (tabPHJ dj x₀)
              ((rsPHJ dj x₀ ht hc hpub hU hscope).binding (rsPHJ_computes dj x₀ ht hc hpub hU hscope))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPHJ dj x₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planPHJ dj x₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.subset kd) n) (planPHJ (Reg := Reg) dj x₀) (tabPHJ (Reg := Reg) dj x₀)
          ((rsPHJ dj x₀ ht hc hpub hU hscope (Reg := Reg)).binding (rsPHJ_computes dj x₀ ht hc hpub hU hscope)) k Rw M ρ t' := by
  intro σ hCR hKS
  exact (Placed.keyProg c hU n).flock_headline_live_tables_subset_missX (Layout.outsOf c hU n) kd hkd Lb src hA3
    (planPHJ dj x₀) (tabPHJ dj x₀) (ppPHJ dj x₀ ht hc hU hscope)
    {Layout.oneG} (constCols_keyProgH hU (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope))
      (Layout.outsOf c hU n))
    {Layout.zeroG} (zeroCols_keyProgH hU (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope))
      (Layout.outsOf c hU n))
    (layPHJ dj x₀ ht hc hU hscope (pub := pub)) (rsPHJ dj x₀ ht hc hpub hU hscope)
    (rsPHJ_computes dj x₀ ht hc hpub hU hscope) hk hRw hM hρ hr ht' K I wr
    (opnPHJ I (planPHJ dj x₀) dj) dg (decJ (planPHJ dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N (decodesOnPHJ_exec dj x₀ ht hc _ wr hTags hRec hDraw) τ hCR hKS

/-- **One plain sampled session of `J` hidden-output tables is sound, at the template instance's program.** -/
theorem plain_session_soundHJI
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planPHJ (Reg := Reg) dj x₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file, and the circuit conditions
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c)
    {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkPHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveJ I dj (fun S R => (planPHJ (Reg := Reg) dj x₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planPHJ (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPHJ dj x₀) N wr)))
    {qF qS : ℝ} :
    let σ := simAuditJ H512 enc512 leaf512 (planPHJ dj x₀) I wr (opnPHJ I (planPHJ dj x₀) dj) dg
      (decJ (planPHJ dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnPHJ_exec dj x₀ ht hc _ wr hTags hRec hDraw) τ
    -- ecr/sha-512 for the link finders of its simulation
    LinkCRL ((instProg c n).partition (outsOfI c n)) H512 enc512 leaf512 (planPHJ dj x₀) (tabPHJ dj x₀)
      ((rsPHJI dj x₀ ht hc hrf hpub hU hpr hw).binding (rsPHJI_computes dj x₀ ht hc hrf hpub hU hpr hw))
      (reg σ) (cont σ) k Rw M t' →
    -- cr/sha-512 for its knowledge finders
    (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planPHJ dj x₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 (planPHJ dj x₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((instProg c n).partition (outsOfI c n)).wrong
          (Xpub {instOneG (c := c) (n := n)} {instZeroG (c := c) (n := n)}
            (XplurL ((instProg c n).partition (outsOfI c n)) H512 enc512 leaf512 (planPHJ dj x₀) (tabPHJ dj x₀)
              ((rsPHJI dj x₀ ht hc hrf hpub hU hpr hw).binding (rsPHJI_computes dj x₀ ht hc hrf hpub hU hpr hw))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPHJ dj x₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planPHJ dj x₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        ((instProg c n).partition (outsOfI c n)).linkBoundE
          (L := Law.execOS (n := n) src (.subset kd) n) (planPHJ (Reg := Reg) dj x₀) (tabPHJ (Reg := Reg) dj x₀)
          ((rsPHJI dj x₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).binding (rsPHJI_computes dj x₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' := by
  intro σ hCR hKS
  exact (instProg c n).flock_headline_live_tables_subset_missX (outsOfI c n) kd hkd Lb src hA3
    (planPHJ dj x₀) (tabPHJ dj x₀) (ppPHJI dj x₀ ht hc hrf)
    {instOneG (c := c) (n := n)} (constCols_instProgH (sitesPHJI_c dj x₀ ht hc hrf) hrf.1 (outsOfI c n))
    {instZeroG (c := c) (n := n)} (zeroCols_instProgH (sitesPHJI_c dj x₀ ht hc hrf) hrf.1 (outsOfI c n))
    (layPHJI dj x₀ ht hc hrf (pub := pub)) (rsPHJI dj x₀ ht hc hrf hpub hU hpr hw)
    (rsPHJI_computes dj x₀ ht hc hrf hpub hU hpr hw) hk hRw hM hρ hr ht' K I wr
    (opnPHJ I (planPHJ dj x₀) dj) dg (decJ (planPHJ dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N (decodesOnPHJ_exec dj x₀ ht hc _ wr hTags hRec hDraw) τ hCR hKS

end FlockSoundness.Discharge.PlainHJ
