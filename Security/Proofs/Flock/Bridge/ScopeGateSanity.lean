import Proofs.Flock.Bridge.ScopeGate

/-!
# `SampledKSLiveV4` is not true of an accept-everything verifier (`SampledKSLiveV4_nontrivial`)

`SampledKSLiveV4`'s event is `verify --zk`'s acceptance (`Bridge.LiveAcceptsScoped`), the registration's population
(`Bridge.RunPopOk`), and the claims relation failing at the fixed extractor's values (`XplurZC`). The relation reads no
outcome of the game, so for a verifier that accepts everything the event is every outcome once the population passes and
the relation fails. `SampledKSLiveV4_nontrivial` states that V4's conclusion, with the acceptance replaced by `True`,
then fails in both branches:

* at files and a draw in `Bridge.KsScoped`, whenever the bound is below 1 (the event has probability 1);
* outside them, where V4 says acceptance has probability 0 (the event has probability 1, not 0).

So the bound holds because of what `verify --zk` checks, not for free.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV4`'s conclusion fails for an accept-everything verifier**: at `SampledKSLiveV4`'s binders, with
the registration's population passing, its conclusion with `Bridge.LiveAcceptsScoped` replaced by `True` is false. At
files and a draw in `Bridge.KsScoped` this needs the claims relation to fail at the extractor's values and the bound to
be below 1, with the cr/ecr antecedents holding. Outside them it needs nothing more. -/
theorem SampledKSLiveV4_nontrivial
    (n : ℕ) (Reg : Type) (I : Inputs)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (desc : Claims.Msg) (qr : Lean.Json) (hpop : Bridge.RunPopOk desc qr n)
    (kd : ℕ)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (ρ : ℝ) (t' : ℝ) (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ) (q : Claims.Sampling.Query) :
    if hs : Bridge.KsScoped I D pub kd then
      let L := Law.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hs.built n).partition (Discharge.Layout.outsOf D.unroll hs.built n)
      let vb := (rsZHJ dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables).bindingZ
        (rsZHJ_computes dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (Flock.Desc.parse_flat hD)
            (Discharge.Layout.scopeOkH_spec hs.built hs.scope).2.1 dj y₀ wr _ hs.tags hCust.1
            fun _ => drawOkZKHJ_of_file I hs.typed (Flock.Desc.parse_flat hD) kd n hs.div dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      let Cc := (keyProg D.unroll hs.built n).circuit (Discharge.Layout.outsOf D.unroll hs.built n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hs.built n
          (Discharge.Layout.outsOf D.unroll hs.built n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll n roots
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hs.built n g
      let bound : ℝ≥0∞ := ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
          ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
          ENNReal.ofReal ((∑ p : vb.Pos,
              (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
            (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
      (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
      LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
      ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) →
      bound < 1 →
      ¬ ((∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
            (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
          LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
          prob (fun _ => (True ∧ Bridge.RunPopOk desc qr n) ∧
              ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots)
                (Bridge.partOf P) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
            (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤ bound)
    else
      prob (fun _ => True ∧ Bridge.RunPopOk desc qr n)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≠ 0 := by
  split
  · intro L P vb σ Xr δ Cc rows roots Cd κ bound hT hL hrel hb h
    exact not_le.2 hb (le_trans ((prob_true _ τ).symm.le.trans
      (prob_mono (fun _ _ => ⟨⟨trivial, hpop⟩, hrel⟩) _ τ)) (h hT hL))
  · exact (zero_lt_one.trans_le ((prob_true _ τ).symm.le.trans
      (prob_mono (fun _ _ => ⟨trivial, hpop⟩) _ τ))).ne'

end Flock.Guarantees
