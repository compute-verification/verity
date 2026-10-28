import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.PubAvg
import Proofs.Flock.Bridge.PubFinderRows
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ

/-!
# The sampled audit's knowledge soundness at the registrant's rows, with a public error

`SampledKSLiveV2Z` bounds `Pr[flock-verify accepts ∧ ¬ RelS]`, the claims read at the rows the registrant opens, by four
terms, three of which read the prover's run: its knowledge finders' collision probabilities (`ksAvgPrZ`), its link
finders' (`linkPrZC`) and the rows term (`rowsPrZC`). `SampledKSLivePub` states it in the additive form
`δ + f(Pr[B^τ])`: a public error `δ` (`Bridge.deltaPubR`) plus a loss `f` (`Bridge.lossPubR`) in the probability that
one finder `B^τ` built from the prover (`Bridge.pubFinderR`) outputs a SHA-512 collision, with no hash premise. `B^τ` is
the uniform mixture of C-Flock's finders at the prover's run and the registrant's openings (`Bridge.PubFinderRows`), `K`
of them (`K = card PubIxR`), so each succeeds with probability at most `p = K·Pr[B^τ]` (`Claims.Finder.le_mix`). At
that `p`:

* each table slot's knowledge finder succeeds with probability at most `p` averaged over the verifier's draw, since
  `B^τ`'s member for that slot draws the draw itself (`tabAvg_of_mix`), and every table of `planZHJ` is at C-Flock ZK's
  parameters (`planZHJ_M1W`, from what `verify --zk` checks at setup, at every registration: `planZHJ` reads none),
  so `ε_ks ≤ (2^-203 + J·((2·25638 + 2^14 + 2)·√p + p + √(kp)))/(1 − ρ)` (`ksAvgPrZ_le_avg`: the largest of the
  `J = nTab I` tables at most their sum, and Jensen under each square root), its rate `2^23/(ek)` at most `ρ`; with
  `J√p = √(J²p)` and `Jp ≤ J²p` that is at most the knowledge term at `J²p`;
* each link finder and each rows finder succeeds with probability at most `p ≤ J²p` (`linkFinderPrZC_le_mix`,
  `rowsFinderPrZC_le_mix`), and the draws read at most `W = Σ_u |io u|` commit strings (`Bridge.ioSum`, the committed
  wires each unit of the circuit's partition computes or reads: a draw reads a commit string only through such a wire
  of a drawn unit, `qs_le_io`). It is `Σ_u`, not `k_d · max_u`, because the live subset draw on finite bytes draws
  every unit when it runs out (`Law.execOS`).

The rows term is at the rows finders only where the registrant's openings open the registered commitments
(`rowsPrZC_le_pr`). The registered reads' leaves are the public file's at every registration, so whether some read is
off the registrant's leaf (`Registered.ReadsX.Off`) is one proposition. Where it is, the claims read the registrant's
rows only at an outcome whose registered reads the verifier owns (`OwnReadsHJ`): accepted there, the registered check
(`checked_of_acceptsHJ`) gives a SHA-512 collision at an off read, `B^τ`'s registered reads' finder's
(`Registered.ReadsX.pick`, at `p`); elsewhere the claims are `Bridge.SampledKSLiveXr`'s. So
$$\delta = \binom{n-K_0}{k_d}\Big/\binom{n}{k_d} + \frac{2^{-203}}{1-\rho}
  + \frac{W\,\big(1/(eM) + 2k/(eR_w)\big)}{1-\rho},\qquad
  f(p) = \frac{(2\cdot 25638 + 2^{14} + 2)\sqrt p + p + \sqrt{kp}}{1-\rho} + \frac{2 W p}{1-\rho} + p,$$
`δ` a function of `n`, `k_d`, `K₀`, the analysis' `k`, `R_w`, `M`, `ρ` and the circuit's partition. `B^τ` runs one of
C-Flock's finders at its own budget: a rewinding knowledge finder at `2(t' + v)`, the self-clash finder at `t' + v`, a
link or rows finder at `t'` per run of the prover, the registered reads' finder at `q_R`, one play of the audit
(`SampledKSLiveV2`'s setting: the prover's per-run time `t'`, one opening's verification `v`). `f` is read at
`J²·K·Pr[B^τ]`: `K` counts each table's slots, two per committed position and two, and `J²` is the price of the
averaged premise, one factor `J` from the sum over the tables, which `√(J²p) = J√p` folds under the square roots. The
rate hypothesis reads only `k` (`rateM1 k ≤ ρ`) where `SampledKSLiveV2`'s reads the plan, and every other hypothesis,
named assumption included, is `SampledKSLiveV2`'s.
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

/-- **`Bridge.SampledKSLiveXr`'s public error `δ`**: the draw's miss, the knowledge term's constant at C-Flock ZK's
parameters (`ksClosedM1`'s `2^-203`), and the link term's waits at `W` commit strings, each over `1 − ρ` where
`Bridge.SampledKSLiveXr` has it. It reads no prover. -/
noncomputable def _root_.Flock.Bridge.deltaPub (n kd K₀ k M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(203 : ℝ)) / (1 - ρ) +
    (W : ℝ) * (1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)

/-- **`Bridge.SampledKSLiveXr`'s loss `f(p)`** at a bound `p` on each of C-Flock's finders' collision probabilities: the
knowledge term's (`ksClosedM1` at `q²/2^513 = p`, less its constant) and the link term's (`p` at each of `W` commit
strings), each over `1 − ρ`. -/
noncomputable def _root_.Flock.Bridge.lossPub (k W : ℕ) (ρ p : ℝ) : ℝ :=
  ((2 * 25638 + 2 ^ 14 + 2) * Real.sqrt p + p + Real.sqrt (k * p)) / (1 - ρ) + (W : ℝ) * p / (1 - ρ)

/-- **The sampled audit's public error `δ` at the registrant's rows**: `Bridge.deltaPub` and the rows term's waits at
`W` commit strings, over `1 − ρ`. It reads no prover. -/
noncomputable def _root_.Flock.Bridge.deltaPubR (n kd K₀ k M Rw W : ℕ) (ρ : ℝ) : ℝ :=
  Bridge.deltaPub n kd K₀ k M Rw W ρ + (W : ℝ) * (k / (Real.exp 1 * Rw)) / (1 - ρ)

/-- **The sampled audit's loss `f(p)` at the registrant's rows**: `Bridge.lossPub`, the rows term's (`p` at each of `W`
commit strings, over `1 − ρ`) and the registered reads' (`p`). -/
noncomputable def _root_.Flock.Bridge.lossPubR (k W : ℕ) (ρ p : ℝ) : ℝ :=
  Bridge.lossPub k W ρ p + (W : ℝ) * p / (1 - ρ) + p

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, in the additive form.**
`SampledKSLiveV2`'s setting and named assumptions, with F's registered roots `own` (`verify --zk --registered F`), at a
rate cap `ρ < 1` with `2^23/(ek) ≤ ρ` (`rateM1`). The claims' values at an outcome (`vreg`) are, at each wire whose
commit string is a registered read, the bit of the registrant's row there (`rd.wv`) where the registrant's leaves are
the registered ones (`¬ rd.Off`) or the verifier owns the registered reads at the outcome's draw (`OwnReadsHJ`), and
at every other wire its value in the circuit's evaluation on the extracted plurality's inputs (`XplurZC`). For every
live prover,
$$\Pr[\text{flock-verify --registered accepts} \wedge \neg\,\mathrm{RelS}_\delta(\mathrm{witRows}(v_{reg}))] \le
  \mathrm{deltaPubR}(n, k_d, K_0, k, M, R_w, W, \rho) + \mathrm{lossPubR}(k, W, \rho, J^2 K\cdot\Pr[B^\tau]),$$
`J = nTab I` the plan's tables, `W = Σ_u |io u|` over the circuit's partition (`Bridge.ioSum`), `B^τ` the uniform
mixture of C-Flock's `K` finders at the prover's run and the registrant's openings (`Bridge.pubFinderR`, at budgets
`2(t' + v)`, `t' + v`, `t'` and `q_R`: `t'` the prover's per-run time, `v` one opening's verification and `q_R` one play
of the audit) and `Pr[B^τ]` its probability of outputting a SHA-512 collision. -/
def SampledKSLivePub : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
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
    -- the registrant's opening of the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
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
    -- one opening's verification, and one play of the audit, in SHA-512 evaluations
    (v qR : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
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
    -- the claims' values at an outcome: the registrant's row's bit at a registered read, where the registrant's leaves
    -- are the registered ones or the verifier owns the reads at the outcome's draw, and the evaluation on the extracted
    -- plurality's inputs at every other wire
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- `B^τ`: one of C-Flock's finders at the prover's run and the registrant's openings, picked uniformly with its own
    -- coins; its registered reads' finder plays the audit once and outputs the registered check's pair at an off read
    let B := Bridge.pubFinderR dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
      (Bridge.strictComp (StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR)
    prob (fun o => Discharge.ZkReg.LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ENNReal.ofReal (Bridge.deltaPubR n kd K₀ k M Rw (Bridge.ioSum P) ρ +
        Bridge.lossPubR k (Bridge.ioSum P) ρ
          ((nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIxR dj y₀ P vb L) * B.Pr Claims.Sampling.sha512Msg))

/-- **The additive bound is at most `δ + f(J²p)`** wherever each of `pubFinder`'s members succeeds with probability at
most `p ≥ 0`, with no hash premise: its draw term is `δ`'s first; each table slot's finder succeeds with probability at
most `p` on average over the draw (`tabAvg_of_mix`), so its knowledge term is at most
`(2^-203 + J·((2·25638 + 2^14 + 2)·√p + p + √(kp)))/(1 − ρ)` (`ksAvgPrZ_le_avg`), and its link term at most
`W·(p + 1/(eM) + k/(eR_w))/(1 − ρ)` (`linkFinderPrZC_le_mix`, `qs_le_io`); both at most their terms at `J²p`. -/
theorem _root_.Flock.Bridge.boundPr_le_pub_of {n mPts : ℕ} {Reg : Type} {I : Inputs}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {C : FlockSoundness.Audit.Circuit}
    (P : FlockSoundness.Audit.Partition C n)
    {decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀)}
    (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode H512)
    {L : Law n} {kd K₀ : ℕ} (hkd : kd ≤ n) {k Rw M : ℕ} {ρ : ℝ} (hρ : ρ < 1) (hρk : rateM1 k ≤ ρ)
    (qF qS t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {p : ℝ}
    (hp0 : 0 ≤ p)
    (hfam : ∀ i, (Bridge.pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤ p) :
    ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) + ksAvgPrZ execArith hs512 (planZHJ dj y₀) k R τ +
        linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ R τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k M Rw (Bridge.ioSum P) ρ +
        Bridge.lossPub k (Bridge.ioSum P) ρ ((nTab I : ℝ) ^ 2 * p)) := by
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

/-- **The additive bound is at most `δ + f(J²·K·Pr[B^τ])`**, with no hash premise: at `p = K·Pr[B^τ]` each of
C-Flock's finders succeeds with probability at most `p` (`pubFam_le`), and `Bridge.boundPr_le_pub_of` there. -/
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
  rw [mul_assoc ((nTab I : ℝ) ^ 2)]
  exact Bridge.boundPr_le_pub_of dj y₀ P vb hkd hρ hρk qF qS t' R τ
    (mul_nonneg (Nat.cast_nonneg _) (Claims.Pr_nonneg _)) (Bridge.pubFam_le dj y₀ P vb k Rw M qF qS t' R τ)

theorem _root_.Flock.Bridge.deltaPub_nonneg (n kd K₀ k M Rw W : ℕ) {ρ : ℝ} (hρ : ρ < 1) :
    0 ≤ Bridge.deltaPub n kd K₀ k M Rw W ρ := by
  have h := (sub_pos.2 hρ).le
  have hB : (0 : ℝ) ≤ 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) := by positivity
  unfold Bridge.deltaPub
  exact add_nonneg (add_nonneg (by positivity) (div_nonneg (by positivity) h))
    (div_nonneg (mul_nonneg (Nat.cast_nonneg _) hB) h)

theorem _root_.Flock.Bridge.lossPub_nonneg (k W : ℕ) {ρ p : ℝ} (hρ : ρ < 1) (hp : 0 ≤ p) :
    0 ≤ Bridge.lossPub k W ρ p := by
  have h := (sub_pos.2 hρ).le
  unfold Bridge.lossPub
  exact add_nonneg (div_nonneg (add_nonneg (add_nonneg (mul_nonneg (by norm_num) (Real.sqrt_nonneg _)) hp)
    (Real.sqrt_nonneg _)) h) (div_nonneg (mul_nonneg (Nat.cast_nonneg _) hp) h)

open Classical FlockSoundness.Discharge.ZkReg in
/-- **`SampledKSLivePub` from `Bridge.SampledKSLiveXr` and `SampledKSLiveV2Z`.** At `p = K·Pr[B^τ]` each member of
`B^τ` succeeds with probability at most `p` (`pubFamR_le`), so `pubFinder`'s do (`pubFam_le_R`) and the bound's first
three terms are at most `δ + f(J²p)` (`Bridge.boundPr_le_pub_of`). The registered reads' leaves are the public file's
at every registration, so `rd.Off` is one proposition.

* **Off**: an accepted outcome whose reads the verifier owns has a SHA-512 collision at an off read
  (`checked_of_acceptsHJ`, `Registered.ReadsX.offAt_conflict`), the registered reads' finder's output on that play, so
  such outcomes are at most `p`; at every other outcome the claims are `Bridge.SampledKSLiveXr`'s.
* **Not off**: the registrant's openings open the registered commitments at every read, the claims are
  `SampledKSLiveV2Z`'s at every outcome, and its rows term is at most `W·(p + k/(eR_w))/(1 − ρ)` (`rowsPrZC_le_pr`,
  `rowsFinderPrZC_le_mix`, `qs_le_io`). -/
theorem sampledKSLivePub_of (hX : Bridge.SampledKSLiveXr) (hV : SampledKSLiveV2Z) : SampledKSLivePub := by
  intro n Reg own I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω
    _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' K₀ hK₀ τ v qR q L G P vb wo σ Xr δ Cc rows roots Cd κ vreg B
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  set pk := fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
    rd.pick o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port)
  rw [mul_assoc ((nTab I : ℝ) ^ 2)]
  have hfamR : ∀ i, (Bridge.pubFamR dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
      (Bridge.strictComp (StrictCR.oneRun G τ pk) qR) i).Pr Claims.Sampling.sha512Msg ≤
      (Fintype.card (Bridge.PubIxR dj y₀ P vb L) : ℝ) * B.Pr Claims.Sampling.sha512Msg :=
    Bridge.pubFamR_le dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ) _
  have hp0 : (0 : ℝ) ≤ Fintype.card (Bridge.PubIxR dj y₀ P vb L) * B.Pr Claims.Sampling.sha512Msg :=
    mul_nonneg (Nat.cast_nonneg _) (Claims.Pr_nonneg _)
  generalize (Fintype.card (Bridge.PubIxR dj y₀ P vb L) : ℝ) * B.Pr Claims.Sampling.sha512Msg = p at hfamR hp0 ⊢
  have hbase := Bridge.boundPr_le_pub_of (K₀ := K₀) dj y₀ P vb hkd hρ hρk (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
    hp0 (Bridge.pubFam_le_R dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ) _ hfamR)
  have hJ1 : (1 : ℝ) ≤ nTab I := by exact_mod_cast nTab_pos I
  have hJ0 : (0 : ℝ) ≤ nTab I := by linarith
  have hpJ : p ≤ (nTab I : ℝ) ^ 2 * p := by
    calc p = 1 * (1 * p) := by ring
      _ ≤ (nTab I : ℝ) * ((nTab I : ℝ) * p) :=
          mul_le_mul hJ1 (mul_le_mul_of_nonneg_right hJ1 hp0) (by linarith) hJ0
      _ = (nTab I : ℝ) ^ 2 * p := by ring
  set x := (nTab I : ℝ) ^ 2 * p with hx
  have hx0 : 0 ≤ x := hp0.trans hpJ
  have hA0 : 0 ≤ Bridge.deltaPub n kd K₀ k M Rw (Bridge.ioSum P) ρ + Bridge.lossPub k (Bridge.ioSum P) ρ x :=
    add_nonneg (Bridge.deltaPub_nonneg _ _ _ _ _ _ _ hρ) (Bridge.lossPub_nonneg _ _ hρ hx0)
  have hWk : 0 ≤ (Bridge.ioSum P : ℝ) * (k / (Real.exp 1 * Rw)) / (1 - ρ) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg _) (by positivity)) h1ρ.le
  have hWx : 0 ≤ (Bridge.ioSum P : ℝ) * x / (1 - ρ) := div_nonneg (mul_nonneg (Nat.cast_nonneg _) hx0) h1ρ.le
  have hfin : ∀ y : ℝ, 0 ≤ y →
      y ≤ (Bridge.ioSum P : ℝ) * (k / (Real.exp 1 * Rw)) / (1 - ρ) + (Bridge.ioSum P : ℝ) * x / (1 - ρ) + x →
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k M Rw (Bridge.ioSum P) ρ + Bridge.lossPub k (Bridge.ioSum P) ρ x) +
          ENNReal.ofReal y ≤
        ENNReal.ofReal (Bridge.deltaPubR n kd K₀ k M Rw (Bridge.ioSum P) ρ +
          Bridge.lossPubR k (Bridge.ioSum P) ρ x) := by
    intro y hy hle
    rw [← ENNReal.ofReal_add hA0 hy]
    refine ENNReal.ofReal_le_ofReal ?_
    unfold Bridge.deltaPubR Bridge.lossPubR
    linarith
  have hcs : ∀ (R R' : Reg) c,
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R c =
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf R' c :=
    fun _ _ _ => rfl
  have hoffc : ∀ R R' : Reg, rd.Off R → rd.Off R' := by
    intro R R' h
    obtain ⟨c, hc, hne⟩ := h
    exact ⟨c, hc, fun h' => hne ((hcs R R' c).trans h')⟩
  by_cases hoff : rd.Off (reg σ)
  · -- some registered read is off the registrant's leaf
    have hXr := hX n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N
      Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q
    have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
        (LiveAcceptsZKR I own o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
            (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
        (LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
            (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)) ∨
        ((LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.Off o.1) := by
      rintro o ⟨hA, hrel⟩
      by_cases hown : OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt
      · exact Or.inr ⟨⟨hA, hown⟩, hoffc _ _ hoff⟩
      · refine Or.inl ⟨acceptsZK_of_R hA, ?_⟩
        have hv : vreg o = Cc.eval Xr := funext fun g =>
          if_neg fun h => h.1.elim (fun hn => hn (hoffc _ _ hoff)) hown
        rwa [Bridge.witDep_eq_witRows, ← hv]
    have hreg : prob (fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
        (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.Off o.1) G τ ≤ ENNReal.ofReal x := by
      have hcoll : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
          (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
            rd.Off o.1 → StrictCR.IsColl H512 (pk o) := by
        rintro o ⟨⟨hA, hO⟩, hoff'⟩
        have hc : rd.Checked o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port) :=
          checked_of_acceptsHJ I ht own dj y₀ (Flock.Desc.parse_flat hD) hU hpub hscope hTags o hA rd hO
        exact (rd.openings o.1 (regPathsHJ I D.unroll pub o.2.2 rd.port) (rd.offAt o.1)).isColl
          (rd.offAt_conflict hc hoff')
      have hle : prob (fun o : Reg × Finset (Fin n) × Discharge.Exec.Transcript =>
          (LiveAcceptsZKR I own o ∧ OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
            rd.Off o.1) G τ ≤ prob (fun o => StrictCR.IsColl H512 (pk o)) G τ :=
        prob_mono hcoll G τ
      have heq : (prob (fun o => StrictCR.IsColl H512 (pk o)) G τ).toReal =
          (Bridge.strictComp (StrictCR.oneRun G τ pk) qR).Pr Claims.Sampling.sha512Msg := by
        rw [Bridge.strictComp_Pr, StrictCR.oneRun_ev, prob_toReal]
        all_goals rfl
      have hB : (Bridge.strictComp (StrictCR.oneRun G τ pk) qR).Pr Claims.Sampling.sha512Msg ≤ p :=
        hfamR (some (.inr (.inr (.inr ()))))
      have hfin' : prob (fun o => StrictCR.IsColl H512 (pk o)) G τ ≠ ⊤ :=
        ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
      refine hle.trans ?_
      rw [← ENNReal.ofReal_toReal hfin', heq]
      exact ENNReal.ofReal_le_ofReal (hB.trans hpJ)
    refine (prob_mono hsub G τ).trans ((prob_or_le _ _ G τ).trans ((add_le_add (hXr.trans hbase) hreg).trans ?_))
    exact hfin x hx0 (by linarith)
  -- every registered read is on the registrant's leaf
  have hno : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript, ¬ rd.Off o.1 := fun o h => hoff (hoffc _ _ h)
  have hon : ∀ c, rd.isRead c → vb.com (wo c) = vb.cs (reg σ) c := by
    intro c hc
    have hl : (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub)).regLeaf (reg σ) c =
        rd.leaf c := by
      by_contra h
      exact hoff ⟨c, hc, h⟩
    refine Prod.ext rfl ?_
    show FlockSoundness.Binding.ExplicitLeaf.leaf (FlockSoundness.Discharge.ZkBind.HmRowsZX.scheme
      (rsZHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub) c) (rd.wv c) (rd.ws c) = _
    unfold FlockSoundness.Discharge.ZkBind.HmRowsZX.scheme
    split_ifs with hrow
    · exact hl.symm
    · exact absurd (rd.read_row c hc) hrow
  have hVi := hV n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg
    N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ q
  have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (LiveAcceptsZKR I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o))) →
      LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots fun g =>
            if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g) := by
    rintro o ⟨hA, hrel⟩
    refine ⟨acceptsZK_of_R hA, ?_⟩
    have hv : vreg o = fun g => if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g :=
      funext fun g => by
        show ite _ _ _ = ite _ _ _
        by_cases h : rd.isRead (vb.pos g)
        · exact (if_pos (And.intro (Or.inl (hno o)) h)).trans (if_pos h).symm
        · exact (if_neg (not_and_of_not_right _ h)).trans (if_neg h).symm
    rwa [hv] at hrel
  have hQs : ∑ c : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L c) : ℝ) /
      Fintype.card L.Ω ≤ (Bridge.ioSum P : ℝ) :=
    qs_le_io execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L (Bridge.ioSum P) fun ω =>
      Finset.sum_le_sum_of_subset (Finset.subset_univ _)
  have hkR : (0 : ℝ) ≤ k / (Real.exp 1 * Rw) := by positivity
  have hrowsB : rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo k ρ (reg σ) (cont σ) ≤
      ENNReal.ofReal ((Bridge.ioSum P : ℝ) * (x + k / (Real.exp 1 * Rw)) / (1 - ρ)) := by
    refine (rowsPrZC_le_pr execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo (reg σ) (cont σ) hRw hρ
      hon).trans ?_
    refine ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ h1ρ.le)
    refine le_trans ?_ (mul_le_mul_of_nonneg_right hQs (add_nonneg hx0 hkR))
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left
      (by linarith [Bridge.rowsFinderPrZC_le_mix dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ) _ hp0
            hfamR c]) (by positivity)
  refine (prob_mono hsub G τ).trans (hVi.trans ((add_le_add hbase hrowsB).trans ?_))
  refine hfin _ (div_nonneg (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hx0 hkR)) h1ρ.le) ?_
  have e : (Bridge.ioSum P : ℝ) * (x + k / (Real.exp 1 * Rw)) / (1 - ρ) =
      (Bridge.ioSum P : ℝ) * (k / (Real.exp 1 * Rw)) / (1 - ρ) + (Bridge.ioSum P : ℝ) * x / (1 - ρ) := by ring
  rw [e]
  linarith

end Flock.Guarantees

/-- **The sampled audit's knowledge soundness, live, at the registrant's rows, in the additive form**
(`Flock.Guarantees.SampledKSLivePub`). -/
theorem Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub :=
  Flock.Guarantees.sampledKSLivePub_of Flock.Bridge.sampledKSLiveXr Flock.Bridge.sampledKSLiveV2Z
