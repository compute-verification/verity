import FlockSoundness.Discharge.Composed.SoundJ
import FlockSoundness.Assumptions.Zk

/-!
# `zk_session_soundJ` at the coin server's custody

`zk_session_soundJ` takes `RecordsLiveZKJ` (`hRec`) and `DrawOkZKJ` at every draw of the live subset law (`hDraw`). As
at one table (`ZkLink/DrawOk.lean`), the record and its draw file are custody, and the rest of `DrawOkZKJ` is
flock-verify's own arithmetic. This file splits them:

* `drawOkZKJ_of_file`: from `DrawFileZK dj (subset kd) pop`, with `nTab I` dividing `kd`, `DrawOkZKJ` holds at
  `mPtsOf c.g (kd / nTab I)` link points at every draw `S`. `UnitDraw.ofJson` decodes a subset draw only at its `k`
  units, so a draw file it decodes is a draw of `kd` units, and part `j` (`partDraw`) of it decodes to its `kd / nTab I`
  units, which are the units `partOf` sends to table `j`. No fact about the law's runs is needed: the run that reads
  past the source's bytes draws every unit, whose draw file decodes only at `kd = n`.
* `zk_session_soundJ_custody`: `zk_session_soundJ` with `RecordCustodyZKJ` (`live-verifier`) for `hRec` and `hDraw`,
  at `mPtsOf c.g (kd / nTab I)` link points, and `nTab I ∣ kd` (flock-verify accepts a session of `count > 1` tables
  only at a draw that splits into `count` equal parts, `accepts_tables`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The coin server's draws are `DrawOkZKJ`.** On untyped `--statement` tags whose circuit file parses under the
`--zk` tags to `c`: when the record's draw file is the verifier's draw under the subset law `subset kd` over `[0, pop)`
(`DrawFileZK`) and the session's tables divide `kd`, every draw `S` is `DrawOkZKJ` at `mPtsOf c.g (kd / nTab I)` link
points. -/
theorem drawOkZKJ_of_file (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S := by
  sorry

/-- **One `--zk` sampled session of `J` tables is sound, at the coin server's custody.** `zk_session_soundJ` with
`RecordCustodyZKJ` (`live-verifier`: the record is the live session's, and its draw file the verifier's draw under the
subset law) in place of `hRec` and `hDraw`, at `mPtsOf c.g (kd / nTab I)` link points, the session's tables dividing
`kd`. -/
theorem zk_session_soundJ_custody
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and the coin server's custody of the record and its draw file
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) := by
  sorry

end FlockSoundness.Discharge.Composed
