import FlockSoundness.Discharge.Canonical.Session
import FlockSoundness.Discharge.Canonical.Verifier
import FlockSoundness.Discharge.FailClosed.ZkHidden

set_option Elab.async false

/-!
# The canonical headline, over what flock-verify accepts

`zk_session_sound_canonical` is `zk_session_soundC_custody_json` over `VerifiesZK` (`LiveVerifiesZK`), what
`flock-verify verify --zk` accepts, as `FailClosed.zk_session_soundHJ_custody_json` is at `Layout.scopeOk`: the circuit
(`circZ I`), its public file and every fact `zk_session_soundC_custody_json` took about them come from the statement check
at each statement it passes (`hs : ScopeZ I`), and at a statement the check refuses the event is empty (`prob_scope`).
`zk_session_soundC_verifies` is `zk_session_soundC` the same way, at the draw and record facts (`hDraw`, `hRec`).

The one condition on the statement is which binding the claim is about: a circuit outside `Layout.scopeOk` (`hv`).
There the check accepts only the canonical format at a `--zk` statement with hidden outputs (`parse_formatOk`), so `hv`
gives hidden outputs (`ScopeZ.hidden`) and `formatOk` (`ScopeZ.formatOk`); a circuit in `Layout.scopeOk` is
`FailClosed.ZkHidden`'s. `FailClosed.flock_verify_sound` takes both. What the headline does not yet say (a tail's
outputs, public inputs, registered reads, all of which the check refuses) is in `Gaps.lean`.

**Proof work.** None: every declaration here is proved, and so is every one it uses.
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.FailClosed (ScopeZ LiveVerifiesZK circZ prob_scope)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **One canonical hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound.**
`zk_session_soundC` over `VerifiesZK`, at a circuit outside `Layout.scopeOk`. -/
theorem zk_session_soundC_verifies
    (I : Inputs) (hv : Layout.scopeOk (circZ I) = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the coin server's record of the live session
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    -- cr/sha-512 for the knowledge finders of its simulation
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n))
            (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ
              (rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built
                (hs.loadH (hs.hidden hv))))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec hDraw)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n))
          (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ
            (rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built
              (hs.loadH (hs.hidden hv))))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact zk_session_soundC I hs.typed dj y₀ wr dg N (hs.parseH (hs.hidden hv)) hs.built (hs.loadH (hs.hidden hv))
    (hs.formatOk hv) kd hkd Lb src hA3 hDraw (hs.tagsOkZHJ (hs.hidden hv)) hRec hk hRw hM hρ hr ht' K₀ τ (hCR hs)

omit mPts in
/-- **One hidden-output `--zk` sampled session of `J` tables that flock-verify accepts is sound, at every circuit outside
`Layout.scopeOk`.** For any live prover `τ` of a `--zk` session of `J ≥ 1` tables, at the record's custody and the printed
draw file, `Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`
(`zk_session_soundC_custody_json` over `VerifiesZK`). The check accepts such a circuit only in the canonical format, at
hidden outputs (`parse_formatOk`); no other hypothesis on the circuit or the statement's form. -/
theorem zk_session_sound_canonical
    (I : Inputs) (hv : Layout.scopeOk (circZ I) = false)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I (mPtsOf (circZ I).g (kd / nTab I)) S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf (circZ I).g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's custody of the record, and its draw file as `flock-verify draw` printed the verifier's draw
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec
            fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH (hs.hidden hv)) kd n hdiv dj
              (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        ∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) :
    prob (fun o => LiveVerifiesZK I o ∧ ∀ hs : ScopeZ I,
        let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec
            fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH (hs.hidden hv)) kd n hdiv dj
              (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        K₀ ≤ (((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n)).wrong
          (XplurZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n))
            (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ
              (rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built
                (hs.loadH (hs.hidden hv))))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ⨆ hs : ScopeZ I,
        (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
          (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
          (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_liveC I hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) dj y₀ wr _
            (hs.tagsOkZHJ (hs.hidden hv)) hRec
            fun _ => drawOkZKHJ_of_file I hs.typed (hs.parseH (hs.hidden hv)) kd n hdiv dj
              (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
        (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg (circZ I) hs.built n).partition (Layout.outsOf (circZ I) hs.built n))
          (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZC dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built (hs.loadH (hs.hidden hv))).bindingZ
            (rsZC_computes dj y₀ hs.typed (hs.parseH (hs.hidden hv)) (hs.formatOk hv) hs.built
              (hs.loadH (hs.hidden hv))))
          k Rw M ρ t' (reg σ) (cont σ)) := by
  refine prob_scope (A := LiveAcceptsZK I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact zk_session_soundC_custody_json I hs.typed (hs.parseH (hs.hidden hv)) hs.built (hs.loadH (hs.hidden hv))
    (hs.formatOk hv) kd hkd hdiv dj y₀ wr dg N Lb src hA3 (hs.tagsOkZHJ (hs.hidden hv)) hRec hdj hk hRw hM hρ hr ht' K₀ τ
    (hCR hs)

end FlockSoundness.Discharge.Canonical
