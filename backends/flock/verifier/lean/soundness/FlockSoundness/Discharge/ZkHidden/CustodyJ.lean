import FlockSoundness.Discharge.ZkHidden.SoundJ
import FlockSoundness.Discharge.ZkHidden.SoundDrawOk
import FlockSoundness.Discharge.Composed.CustodyJ

/-!
# `zk_session_soundHJ` at the coin server's custody

`Composed/CustodyJ.lean` under hidden outputs. `zk_session_soundHJ` takes `RecordsLiveZKJ` (`hRec`) and `DrawOkZKHJ` at
every draw of the live subset law (`hDraw`); the record and its draw file are custody, and the rest of `DrawOkZKHJ` is
flock-verify's own arithmetic:

* `drawOkZKHJ_of_file`: `drawOkZKJ_of_file` at setupHidden. From `DrawFileZK dj (subset kd) pop`, with `nTab I`
  dividing `kd`, `DrawOkZKHJ` holds at `mPtsOf c.g (kd / nTab I)` link points at every draw `S`, `c` the circuit file's
  `HmOut.parse` (setupHidden's `m_pts`, `setupHidden_mPts_drawn` and `setupHidden_mPts_of_proved`).
* `zk_session_soundHJ_custody`: `zk_session_soundHJ` with `RecordCustodyZKJ` (`live-verifier`) for `hRec` and `hDraw`,
  at `mPtsOf c.g (kd / nTab I)` link points and `nTab I ∣ kd`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The coin server's draws are `DrawOkZKHJ`.** On untyped `--statement` tags whose circuit file `HmOut.parse`
accepts under the `--zk` tags as `c`: when the record's draw file is the verifier's draw under the subset law
`subset kd` over `[0, pop)` (`DrawFileZK`) and the session's tables divide `kd`, every draw `S` is `DrawOkZKHJ` at
`mPtsOf c.g (kd / nTab I)` link points. -/
theorem drawOkZKHJ_of_file (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S := by
  sorry

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at the coin server's custody.**
`zk_session_soundHJ` with `RecordCustodyZKJ` (`live-verifier`: the record is the live session's, and its draw file the
verifier's draw under the subset law) in place of `hRec` and `hDraw`, at `mPtsOf c.g (kd / nTab I)` link points, the
session's tables dividing `kd`. -/
theorem zk_session_soundHJ_custody
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
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
        (runsZCJH_live I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) := by
  sorry

end FlockSoundness.Discharge.ZkHidden
