import FlockSoundness.Discharge.Canonical.Session
import FlockSoundness.Discharge.Canonical.Verifier

set_option Elab.async false

/-!
# The session headline with no scope hypothesis

`zk_session_sound_canonical`: `zk_session_soundHJ_custody_json` with neither `Layout.scopeOk c = true` nor
`pub.tables = none`. The circuit's only conditions are the verifier's own: `HmOut.parse` accepted its file
(`hc`, which after V1 implies the canonical format, `parse_formatOk`) and the public file loaded (`hpub`). `ht` (an
untyped circuit) and `hU` (`Built c.unit`, which `ExecSetup.unit_built` gives from any accepted parse) are not scope
conditions; typed circuits are out of this headline (`DESIGN.md`).

**Proof work.** None in this file: `zk_session_soundC_custody_json` at `parse_formatOk`. `Session.lean` is proved, so
the headline rests on `parse_formatOk` alone, which is false until V1 lands; until then the headline is a frozen target,
not a result. What it does not yet say (a tail's outputs, public inputs, registered reads) is in `Gaps.lean`.
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, for every circuit the verifier accepts.** For
any live prover `τ` of a `--zk` session of `J ≥ 1` tables, at the record's custody and the printed draw file,
`Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`, with no condition on the circuit
but `HmOut.parse`'s acceptance (`hc`). -/
theorem zk_session_sound_canonical
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, the coin server's custody of the record, and its draw file as `flock-verify draw`
    -- printed the verifier's draw
    (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveC I ht hc (parse_formatOk (zkTyped ht) hc) dj y₀ wr _ hTags hRec
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZC dj y₀ ht hc (parse_formatOk (zkTyped ht) hc) hU hpub).bindingZ
              (rsZC_computes dj y₀ ht hc (parse_formatOk (zkTyped ht) hc) hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZC dj y₀ ht hc (parse_formatOk (zkTyped ht) hc) hU hpub).bindingZ
            (rsZC_computes dj y₀ ht hc (parse_formatOk (zkTyped ht) hc) hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundC_custody_json I ht hc hU hpub (parse_formatOk (zkTyped ht) hc) kd hkd hdiv dj y₀ wr dg N Lb src
    hA3 hTags hRec hdj hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.Canonical
