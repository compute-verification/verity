import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
import Proofs.Flock.Soundness.Discharge.ZkRegV2.SoundHJ

/-!
# One hidden-output `--zk --registered` sampled session of `J` tables over `sha512/row/v2` rows, under the record's custody

`zk_session_soundHJR_custodyV2`: `zk_session_soundHJRV2` with `hRec` and `hDraw` replaced by `RecordCustodyZKJ`
(`live-verifier`), exactly as `zk_session_soundHJR_custody` is `zk_session_soundHJR`'s: the draws are `DrawOkZKHJ` by
`drawOkZKHJ_of_file`, which reads the record and its draws and no row.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  Composed Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk --registered` sampled session of `J` tables over `sha512/row/v2` rows is sound under
`live-verifier`, and its registered reads are checked.** `zk_session_soundHJR_custody` at a `scopeOkHV2` circuit, with
the rows `rsZHJSV2`. -/
theorem zk_session_soundHJR_custodyV2
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : ZkRegV2.scopeOkHV2 c = true)
    -- the law (the live subset draw), the tables dividing it
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- A3
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJSV2_computes dj y₀ ht hc hscope hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJSV2_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.Reads (layShHJV2 (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
          (rsShHJSV2 dj y₀ ht hc hscope hU hpub),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) :=
  zk_session_soundHJRV2 I ht own dj y₀ wr dg N hc hU hpub hscope kd hkd Lb src hA3
    (fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.ZkReg
