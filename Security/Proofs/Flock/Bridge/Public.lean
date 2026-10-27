import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# `SampledKSLiveV2` with a public error

`SampledKSLiveV2` bounds `Pr[flock-verify accepts ∧ ¬ RelS]` by three terms, two of which read the prover's run:
`ksAvgStrictZ (planZHJ dj y₀) L k q_F q_S (reg σ)`, at the prover's registration, and
`Q_s = Σ_c |Ω_c|/|Ω|` (`ΩcZ`), at its binding. Neither depends on the prover in value, and each is at most a number read
off the public parameters:

* `ε_ks ≤ ksClosedM1 k qF qS / (1 − ρ)`: every table of `planZHJ` is at C-Flock ZK's parameters (`planZHJ_M1W`, from
  what `verify --zk` checks at setup, at every registration: `planZHJ` reads none), so `ksAvgStrictZ_planZHJ_le_closed`
  applies, and its rate `2^23/(ek)` is at most `ρ`;
* `Q_s ≤ W`, `W = Σ_u |io u|` (`Bridge.ioSum`) the committed wires each unit of the circuit's partition computes or
  reads: a draw reads a commit string only through such a wire of a drawn unit (`qs_le_io`). It is `Σ_u`, not
  `k_d · max_u`, because the live subset draw on finite bytes draws every unit when it runs out (`Law.execOS`).

So `SampledKSLivePub` is `SampledKSLiveV2` with its error `Bridge.deltaPub`:
$$\binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \frac{\mathrm{ksClosedM1}(k,q_F,q_S)}{1-\rho}
  + \frac{W\,\big(2t'(1+k)/2^{256} + 1/(eM) + k/(eR_w)\big)}{1-\rho},$$
at the knowledge finders' budgets `q_F = 2(t' + v)` and `q_S = t' + v` (`SampledKSLiveV2`'s: the prover's per-run time `t'`
and one opening's verification `v`), a function of `n`, `k_d`, `K₀`, the analysis' `k`, `R_w`, `M`, `t'`, `v`, `ρ` and the
circuit's partition. Its rate hypothesis reads only `k` (`rateM1 k ≤ ρ`) where `SampledKSLiveV2`'s reads the plan, and every other
hypothesis, named assumption included, is `SampledKSLiveV2`'s.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The committed wires the units read or write, summed over the units**: `Σ_u |io u|`, a bound on the commit strings
any draw reads. -/
noncomputable def _root_.Flock.Bridge.ioSum {C : FlockSoundness.Audit.Circuit} {n : ℕ}
    (P : FlockSoundness.Audit.Partition C n) : ℕ :=
  ∑ u, (P.io u).card

/-- **The sampled audit's public error**: the draw's miss, the knowledge term at C-Flock ZK's parameters
(`ksClosedM1`), and the link term at `W` commit strings, each over `1 − ρ` where `SampledKSLiveV2` has it. -/
noncomputable def _root_.Flock.Bridge.deltaPub (n kd K₀ k : ℕ) (qF qS t' : ℝ) (M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + ksClosedM1 k qF qS / (1 - ρ) +
    (W : ℝ) * (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)

open Classical in
/-- **The sampled audit's knowledge soundness, live, with a public error.** `SampledKSLiveV2`'s setting and named
assumptions, at a rate cap `ρ < 1` with `2^23/(ek) ≤ ρ` (`rateM1`): for every live prover,
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witDep}(X))] \le
  \mathrm{deltaPub}(n, k_d, K_0, k, 2(t' + v), t' + v, t', M, R_w, W, \rho),$$
`W = Σ_u |io u|` over the circuit's partition (`Bridge.ioSum`), `t'` the prover's per-run time and `v` one opening's
verification. -/
def SampledKSLivePub : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    -- the statement's descriptor, and the registration's query, whose population is the game's n units in order
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one opening's verification, in SHA-512 evaluations
    (v : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- the circuit's constants, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost: two runs of the
    -- prover and two verifications for a rewinding finder, one of each for the self-clash finder
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    -- ecr/sha-512 for the prover's link finders
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      k Rw M t' →
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k (2 * (t' + v)) (t' + v) t' M Rw (Bridge.ioSum P) ρ)

/-- **`SampledKSLiveV2`'s error is at most `deltaPub`**: its draw term is `deltaPub`'s first, its knowledge term at most
the second (`ksAvgStrictZ_planZHJ_le_closed`), and its link term at most the third (`qs_le_io`). -/
theorem sampledKSLivePub_of (h : SampledKSLiveV2) : SampledKSLivePub := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' ht' K₀ hK₀ τ v q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  refine (h n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb
    src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v q hCR hLink).trans ?_
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hB : 0 ≤ 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) := by
    have : 0 ≤ t' * (1 + k) := mul_nonneg ht' (by positivity)
    positivity
  have hmiss : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) =
      ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) := by
    have hc : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkd
    rw [ENNReal.ofReal_div_of_pos hc, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have hks : ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) ≤ ENNReal.ofReal (ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ)) :=
    (ksAvgStrictZ_planZHJ_le_closed dj y₀ L k (2 * (t' + v)) (t' + v) (reg σ) (hρk.trans_lt hρ)).trans
      (ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_left (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ (by linarith)))
  have hQs : ∑ p : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) /
      Fintype.card L.Ω ≤ (Bridge.ioSum P : ℝ) :=
    qs_le_io execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L (Bridge.ioSum P) fun ω =>
      Finset.sum_le_sum_of_subset (Finset.subset_univ _)
  have hlink : ENNReal.ofReal ((∑ p : vb.Pos,
        (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) ≤
      ENNReal.ofReal ((Bridge.ioSum P : ℝ) *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) :=
    ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hQs hB) h1ρ.le)
  have hx : (0 : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by positivity
  have hy : 0 ≤ ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ) := div_nonneg (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ.le
  have hz : 0 ≤ (Bridge.ioSum P : ℝ) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg _) hB) h1ρ.le
  rw [hmiss]
  refine (add_le_add (add_le_add le_rfl hks) hlink).trans (le_of_eq ?_)
  unfold Bridge.deltaPub
  rw [ENNReal.ofReal_add (add_nonneg hx hy) hz, ENNReal.ofReal_add hx hy]

end Flock.Guarantees

/-- **The sampled audit's knowledge soundness, live, with a public error** (`Flock.Guarantees.SampledKSLivePub`). -/
theorem Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub :=
  Flock.Guarantees.sampledKSLivePub_of Flock.SecurityProofs.SampledKSLiveV2
