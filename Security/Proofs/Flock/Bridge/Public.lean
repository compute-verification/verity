import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.PubAvg
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# `Bridge.SampledKSLiveXr` with a public error

`Bridge.SampledKSLiveXr` bounds `Pr[flock-verify accepts ∧ ¬ RelS]` by three terms, two of which read the prover's run: its
knowledge finders' collision probabilities (`ksAvgPrZ`) and its link finders' (`linkPrZC`). `SampledKSLivePub` states
it in the additive form `δ + f(Pr[B^τ])`: a public error `δ` (`Bridge.deltaPub`) plus a loss `f` (`Bridge.lossPub`) in
the probability that one finder `B^τ` built from the prover (`Bridge.pubFinder`) outputs a SHA-512 collision, with no
hash premise. `B^τ` is the uniform mixture of C-Flock's finders at the prover's run (`Bridge.PubFinder`), `K` of them
(`K = card PubIx`), so each succeeds with probability at most `p = K·Pr[B^τ]` (`Claims.Finder.le_mix`). At that `p`:

* each table slot's knowledge finder succeeds with probability at most `p` averaged over the verifier's draw, since
  `B^τ`'s member for that slot draws the draw itself (`tabAvg_of_mix`), and every table of `planZHJ` is at C-Flock ZK's
  parameters (`planZHJ_M1W`, from what `verify --zk` checks at setup, at every registration: `planZHJ` reads none),
  so `ε_ks ≤ (2^-203 + J·((2·25638 + 2^14 + 2)·√p + p + √(kp)))/(1 − ρ)` (`ksAvgPrZ_le_avg`: the largest of the
  `J = nTab I` tables at most their sum, and Jensen under each square root), its rate `2^23/(ek)` at most `ρ`; with
  `J√p = √(J²p)` and `Jp ≤ J²p` that is at most the knowledge term at `J²p`;
* each link finder succeeds with probability at most `p ≤ J²p` (`linkFinderPrZC_le_mix`), and the draws read at most
  `W = Σ_u |io u|` commit strings (`Bridge.ioSum`, the committed wires each unit of the circuit's partition computes or
  reads: a draw reads a commit string only through such a wire of a drawn unit, `qs_le_io`). It is `Σ_u`, not
  `k_d · max_u`, because the live subset draw on finite bytes draws every unit when it runs out (`Law.execOS`).

So
$$\delta = \binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \frac{2^{-203}}{1-\rho}
  + \frac{W\,\big(1/(eM) + k/(eR_w)\big)}{1-\rho},\qquad
  f(p) = \frac{(2\cdot 25638 + 2^{14} + 2)\sqrt p + p + \sqrt{kp}}{1-\rho} + \frac{W p}{1-\rho},$$
`δ` a function of `n`, `k_d`, `K₀`, the analysis' `k`, `R_w`, `M`, `ρ` and the circuit's partition. `B^τ` runs one of
C-Flock's finders at its own budget: a rewinding knowledge finder at `2(t' + v)`, the self-clash finder at `t' + v`, a
link finder at `t'` per run of the prover (`Bridge.SampledKSLiveXr`'s: the prover's per-run time `t'` and one opening's
verification `v`). `f` is read at `J²·K·Pr[B^τ]`: `K` counts each table's slots, every committed position and one, and
`J²` is the price of the averaged premise, one factor `J` from the sum over the tables, which `√(J²p) = J√p` folds
under the square roots. The rate hypothesis reads only `k` (`rateM1 k ≤ ρ`) where `Bridge.SampledKSLiveXr`'s reads the plan,
and every other hypothesis, named assumption included, is `Bridge.SampledKSLiveXr`'s.
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

/-- **The sampled audit's public error `δ`**: the draw's miss, the knowledge term's constant at C-Flock ZK's parameters
(`ksClosedM1`'s `2^-203`), and the link term's waits at `W` commit strings, each over `1 − ρ` where `Bridge.SampledKSLiveXr`
has it. It reads no prover. -/
noncomputable def _root_.Flock.Bridge.deltaPub (n kd K₀ k M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(203 : ℝ)) / (1 - ρ) +
    (W : ℝ) * (1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)

/-- **The sampled audit's loss `f(p)`** at a bound `p` on each of C-Flock's finders' collision probabilities: the
knowledge term's (`ksClosedM1` at `q²/2^513 = p`, less its constant) and the link term's (`p` at each of `W` commit
strings), each over `1 − ρ`. -/
noncomputable def _root_.Flock.Bridge.lossPub (k W : ℕ) (ρ p : ℝ) : ℝ :=
  ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p)) / (1 - ρ) + (W : ℝ) * p / (1 - ρ)

open Classical in
/-- **The sampled audit's knowledge soundness, live, in the additive form.** `Bridge.SampledKSLiveXr`'s setting and named
assumptions, at a rate cap `ρ < 1` with `2^23/(ek) ≤ ρ` (`rateM1`): for every live prover,
$$\Pr[\text{flock-verify accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witDep}(X))] \le
  \mathrm{deltaPub}(n, k_d, K_0, k, M, R_w, W, \rho) + \mathrm{lossPub}(k, W, \rho, J^2 K\cdot\Pr[B^\tau]),$$
`J = nTab I` the plan's tables, `W = Σ_u |io u|` over the circuit's partition (`Bridge.ioSum`), `B^τ` the uniform
mixture of C-Flock's `K` finders at the prover's run (`Bridge.pubFinder`, at budgets `2(t' + v)`, `t' + v` and `t'`:
`t'` the prover's per-run time and `v` one opening's verification) and `Pr[B^τ]` its probability of outputting a
SHA-512 collision. -/
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
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
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
    -- `B^τ`: one of C-Flock's finders at the prover's run, picked uniformly with its own coins
    let B := Bridge.pubFinder dj y₀ P vb k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k M Rw (Bridge.ioSum P) ρ +
        Bridge.lossPub k (Bridge.ioSum P) ρ
          ((nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIx dj y₀ P vb L) * B.Pr Claims.Sampling.sha512Msg))

/-- **The additive bound is at most `δ + f(J²·K·Pr[B^τ])`**, with no hash premise: its draw term is `δ`'s first; at
`p = K·Pr[B^τ]` each of C-Flock's finders succeeds with probability at most `p` (`pubFam_le`), so each table slot's
finder does on average over the draw (`tabAvg_of_mix`), its knowledge term is at most
`(2^-203 + J·((2·25638 + 2^14 + 2)·√p + p + √(kp)))/(1 − ρ)` (`ksAvgPrZ_le_avg`), and its link term at most
`W·(p + 1/(eM) + k/(eR_w))/(1 − ρ)` (`linkFinderPrZC_le_mix`, `qs_le_io`); both at most their terms at `J²p`. -/
theorem _root_.Flock.Bridge.boundPr_le_pub {n mPts : ℕ} {Reg : Type} {I : Inputs}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {C : FlockSoundness.Audit.Circuit}
    (P : FlockSoundness.Audit.Partition C n)
    {decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀)}
    (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode H512)
    {L : Law n} {kd K₀ : ℕ} (hkd : kd ≤ n) {k Rw M : ℕ} {ρ : ℝ} (hρ : ρ < 1) (hρk : rateM1 k ≤ ρ)
    (qF qS t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) :
    ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) + ksAvgPrZ execArith hs512 (planZHJ dj y₀) k R τ +
        linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ R τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k M Rw (Bridge.ioSum P) ρ +
        Bridge.lossPub k (Bridge.ioSum P) ρ ((nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIx dj y₀ P vb L) *
          (Bridge.pubFinder dj y₀ P vb k Rw M qF qS t' R τ).Pr Claims.Sampling.sha512Msg)) := by
  have hfam := Bridge.pubFam_le dj y₀ P vb k Rw M qF qS t' R τ
  have hp0 : (0 : ℝ) ≤ Fintype.card (Bridge.PubIx dj y₀ P vb L) *
      (Bridge.pubFinder dj y₀ P vb k Rw M qF qS t' R τ).Pr Claims.Sampling.sha512Msg :=
    mul_nonneg (Nat.cast_nonneg _) (Claims.Pr_nonneg _)
  rw [mul_assoc ((nTab I : ℝ) ^ 2)]
  generalize (Fintype.card (Bridge.PubIx dj y₀ P vb L) : ℝ) *
      (Bridge.pubFinder dj y₀ P vb k Rw M qF qS t' R τ).Pr Claims.Sampling.sha512Msg = p at hfam hp0 ⊢
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hJ1 : (1 : ℝ) ≤ nTab I := by exact_mod_cast nTab_pos I
  have hJ0 : (0 : ℝ) ≤ nTab I := by linarith
  have hP2 : 0 ≤ (nTab I : ℝ) ^ 2 * p := mul_nonneg (sq_nonneg _) hp0
  have hpJ : p ≤ (nTab I : ℝ) ^ 2 * p := by
    calc p = 1 * (1 * p) := by ring
      _ ≤ (nTab I : ℝ) * ((nTab I : ℝ) * p) :=
          mul_le_mul hJ1 (mul_le_mul_of_nonneg_right hJ1 hp0) (by linarith) hJ0
      _ = (nTab I : ℝ) ^ 2 * p := by ring
  have hpJ' : (nTab I : ℝ) * p ≤ (nTab I : ℝ) ^ 2 * p := by
    calc (nTab I : ℝ) * p = 1 * ((nTab I : ℝ) * p) := by ring
      _ ≤ (nTab I : ℝ) * ((nTab I : ℝ) * p) := mul_le_mul_of_nonneg_right hJ1 (mul_nonneg hJ0 hp0)
      _ = (nTab I : ℝ) ^ 2 * p := by ring
  have hsq : Real.sqrt ((nTab I : ℝ) ^ 2 * p) = nTab I * Real.sqrt p := by
    rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hJ0]
  have hsqk : Real.sqrt (k * ((nTab I : ℝ) ^ 2 * p)) = nTab I * Real.sqrt (k * p) := by
    rw [show (k : ℝ) * ((nTab I : ℝ) ^ 2 * p) = (nTab I : ℝ) ^ 2 * (k * p) by ring, Real.sqrt_mul (sq_nonneg _),
      Real.sqrt_sq hJ0]
  have hks : ksAvgPrZ execArith hs512 (planZHJ dj y₀) k R τ ≤
      ENNReal.ofReal ((2 ^ (-(203 : ℝ)) + ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt ((nTab I : ℝ) ^ 2 * p) +
        (nTab I : ℝ) ^ 2 * p + Real.sqrt (k * ((nTab I : ℝ) ^ 2 * p)))) / (1 - ρ)) := by
    refine (Bridge.ksAvgPrZ_le_avg dj y₀ qF qS R τ hρ hρk hp0
        fun j s => Bridge.tabAvg_of_mix dj y₀ P vb k Rw M qF qS t' R τ hfam j s).trans
      (ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ h1ρ.le))
    rw [hsq, hsqk]
    linarith
  have hQs : ∑ c : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L c) : ℝ) /
      Fintype.card L.Ω ≤ (Bridge.ioSum P : ℝ) :=
    qs_le_io execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L (Bridge.ioSum P) fun ω =>
      Finset.sum_le_sum_of_subset (Finset.subset_univ _)
  have hB : 0 ≤ (nTab I : ℝ) ^ 2 * p + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) :=
    add_nonneg (add_nonneg hP2 (by positivity)) (by positivity)
  have hlink : linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ R τ ≤
      ENNReal.ofReal ((Bridge.ioSum P : ℝ) * ((nTab I : ℝ) ^ 2 * p + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) /
        (1 - ρ)) := by
    unfold linkPrZC
    refine ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ h1ρ.le)
    refine le_trans ?_ (mul_le_mul_of_nonneg_right hQs hB)
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left
      (by linarith [Bridge.linkFinderPrZC_le_mix dj y₀ P vb k Rw M qF qS t' R τ hp0 hfam c]) (by positivity)
  have hmiss : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) =
      ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) := by
    have hc : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkd
    rw [ENNReal.ofReal_div_of_pos hc, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have hx : (0 : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by positivity
  have hy : 0 ≤ (2 ^ (-(203 : ℝ)) + ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt ((nTab I : ℝ) ^ 2 * p) +
      (nTab I : ℝ) ^ 2 * p + Real.sqrt (k * ((nTab I : ℝ) ^ 2 * p)))) / (1 - ρ) :=
    div_nonneg (add_nonneg (by positivity) (add_nonneg (add_nonneg (by positivity) hP2) (Real.sqrt_nonneg _)))
      h1ρ.le
  have hz : 0 ≤ (Bridge.ioSum P : ℝ) * ((nTab I : ℝ) ^ 2 * p + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) /
      (1 - ρ) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg _) hB) h1ρ.le
  rw [hmiss]
  refine (add_le_add (add_le_add le_rfl hks) hlink).trans (le_of_eq ?_)
  rw [← ENNReal.ofReal_add hx hy, ← ENNReal.ofReal_add (add_nonneg hx hy) hz]
  congr 1
  unfold Bridge.deltaPub Bridge.lossPub
  ring

/-- **`Bridge.SampledKSLiveXr`'s error is at most `δ + f(J²·K·Pr[B^τ])`** (`Bridge.boundPr_le_pub`). -/
theorem sampledKSLivePub_of (h : Bridge.SampledKSLiveXr) : SampledKSLivePub := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' K₀ hK₀ τ v q L P vb σ Xr δ Cc rows roots Cd κ B
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  exact (h n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb
    src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q).trans
    (Bridge.boundPr_le_pub dj y₀ P vb hkd hρ hρk (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ))

end Flock.Guarantees

/-- **The sampled audit's knowledge soundness, live, in the additive form** (`Flock.Guarantees.SampledKSLivePub`). -/
theorem Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub :=
  Flock.Guarantees.sampledKSLivePub_of Flock.Bridge.sampledKSLiveXr
