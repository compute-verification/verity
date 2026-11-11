module

public import Proofs.Flock.Bridge.KSLive
public import Proofs.Flock.Bridge.PublicDeployed
public import Proofs.Flock.Bridge.QsRows
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-!
# `SampledKSLivePub`'s bound at the rows a sample reads

`SampledKSLivePub`'s binding error counts `W = Σ_u |io u|` commit strings, every unit's, because the live subset sample
on finite bytes samples every unit when it runs out. On a byte source that holds the sample's first read, it runs out
with probability at most `k_d · 2⁻⁶⁴`, and otherwise reads the rows of `k_d` proof blocks and the zero's and the
constant's gates (`Bridge.qs_vbOf_execOS_le`). So under two more hypotheses the count is `k_d · n_p + 3`, `n_p` the
ports of the unrolled unit, whatever the population:

* `Bridge.sampledKSLivePubRows`: `SampledKSLivePub`'s conclusion at `W = k_d · n_p + 3`, for a byte source of at least
  `2 · k_d · width n + 64` bytes and `k_d · n · n_p ≤ 2^64`. Every other hypothesis is `SampledKSLivePub`'s.
* `Bridge.sampledKSLivePubRows_deployed`: at the deployment's parameters (`k = kDep`, `ρ = 1/2`, `M = R_w = 2^192`),
  a prover's per-run time `t' ≤ 2^80`, one opening's verification `v ≤ 2^80` and `k_d · n_p + 3 ≤ 2^40`, the bound is
  the sample's miss plus `2^-100` (`Bridge.deltaPub_le_deployed`).

Neither is a guarantee: each is a theorem over `SampledKSLiveV2`'s proof, and no lock lists it.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLivePub`'s bound at `W = k_d · n_p + 3` commit strings**, `n_p` the ports of the unrolled unit:
`SampledKSLivePub`'s hypotheses and conclusion, on a byte source that holds the live sample's first read
(`2 · k_d · width n + 64 ≤ Lb 0`) and at `k_d · n · n_p ≤ 2^64`, where the sample's running out adds at most one commit
string to the `k_d · n_p + 2` a sample of `k_d` proof blocks reads (`Bridge.qs_vbOf_execOS_le`). -/
theorem _root_.Flock.Bridge.sampledKSLivePubRows :
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
    -- the byte source holds the live sample's first read
    (_hLb : 2 * kd * Flock.Draw.width n + 64 ≤ Lb 0)
    -- the sample's running out adds at most one commit string
    (_hsz : kd * (n * D.unroll.ports.size) ≤ 2 ^ 64)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one opening's verification, in SHA-512 evaluations
    (v : ℝ) (_hv : 0 ≤ v)
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
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    -- ecr/sha-512 for the prover's link finders, at a run of the prover and a verification per run
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      k Rw M (t' + v) →
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
          (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k (2 * (t' + v)) (t' + v) (t' + v) M Rw
        (kd * D.unroll.ports.size + 3) ρ) := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hLb hsz hTags hCust k Rw M hk hRw hM ρ hρ hρk t' ht' K₀ hK₀ τ v hv q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  refine (Flock.SecurityProofs.SampledKSLiveV2 n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd
    hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v hv q hCR
    hLink).trans ?_
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hB : 0 ≤ 2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) := by
    have : 0 ≤ (t' + v) * (1 + k) := mul_nonneg (add_nonneg ht' hv) (by positivity)
    positivity
  have hmiss : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) =
      ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) := by
    have hc : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkd
    rw [ENNReal.ofReal_div_of_pos hc, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have hks : ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) ≤
      ENNReal.ofReal (ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ)) :=
    (ksAvgStrictZ_planZHJ_le_closed dj y₀ L k (2 * (t' + v)) (t' + v) (reg σ) (hρk.trans_lt hρ)).trans
      (ENNReal.ofReal_le_ofReal
        (div_le_div_of_nonneg_left (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ (by linarith)))
  -- the sample's running out adds at most one commit string
  have hout : (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * D.unroll.ports.size) ≤ 1 := by
    have h2 : ((kd * (n * D.unroll.ports.size) : ℕ) : ℝ) ≤ ((2 ^ 64 : ℕ) : ℝ) := Nat.cast_le.2 hsz
    rw [Nat.cast_mul, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat] at h2
    have h3 : (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * D.unroll.ports.size) =
        (kd : ℝ) * ((n : ℝ) * D.unroll.ports.size) / 2 ^ 64 := by ring
    rw [h3]
    exact div_le_one_of_le₀ h2 (by positivity)
  have hcast : ((kd * D.unroll.ports.size + 3 : ℕ) : ℝ) = (kd : ℝ) * D.unroll.ports.size + 2 + 1 := by
    rw [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
    ring
  have hQs : ∑ p : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) /
      Fintype.card L.Ω ≤ ((kd * D.unroll.ports.size + 3 : ℕ) : ℝ) := by
    rw [hcast]
    exact (Bridge.qs_vbOf_execOS_le (Reg := Reg) (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parse_flat hD)
      (hscope := hscope) (hU := hU) (hpub := hpub) (hpt := hpt) Lb src hA3 kd hkd hLb).trans
      (add_le_add le_rfl hout)
  have hlink : ENNReal.ofReal ((∑ p : vb.Pos,
        (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
      (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) ≤
      ENNReal.ofReal (((kd * D.unroll.ports.size + 3 : ℕ) : ℝ) *
        (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) :=
    ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hQs hB) h1ρ.le)
  have hx : (0 : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by positivity
  have hy : 0 ≤ ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ) :=
    div_nonneg (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ.le
  have hz : 0 ≤ ((kd * D.unroll.ports.size + 3 : ℕ) : ℝ) *
      (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg _) hB) h1ρ.le
  rw [hmiss]
  refine (add_le_add (add_le_add le_rfl hks) hlink).trans (le_of_eq ?_)
  unfold Bridge.deltaPub
  rw [ENNReal.ofReal_add (add_nonneg hx hy) hz, ENNReal.ofReal_add hx hy]

open Classical in
/-- **At the deployment's parameters the bound is the sample's miss plus `2^-100`, whatever the population.**
`Bridge.sampledKSLivePubRows` at `k = kDep`, `ρ = 1/2` and `M = R_w = 2^192`, for a prover's per-run time `t' ≤ 2^80`,
one opening's verification `v ≤ 2^80` and a sample that reads at most `2^40` commit strings
(`k_d · n_p + 3 ≤ 2^40`): `Bridge.deltaPub_le_deployed`. -/
theorem _root_.Flock.Bridge.sampledKSLivePubRows_deployed :
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
    -- the byte source holds the live sample's first read
    (_hLb : 2 * kd * Flock.Draw.width n + 64 ≤ Lb 0)
    -- the sample's running out adds at most one commit string
    (_hsz : kd * (n * D.unroll.ports.size) ≤ 2 ^ 64)
    -- a sample reads at most 2^40 commit strings
    (_hW : kd * D.unroll.ports.size + 3 ≤ 2 ^ 40)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    -- the prover's per-run time
    (t' : ℝ) (_ht' : 0 ≤ t') (_ht80 : t' ≤ 2 ^ 80) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one opening's verification, in SHA-512 evaluations
    (v : ℝ) (_hv : 0 ≤ v) (_hv80 : v ≤ 2 ^ 80)
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
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) Bridge.kDep (2 ^ 192)
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
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs, at their cost
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    -- ecr/sha-512 for the prover's link finders, at a run of the prover and a verification per run
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      Bridge.kDep (2 ^ 192) (2 ^ 192) (t' + v) →
    prob (fun o => LiveAcceptsZK I o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
          (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ))) := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hLb hsz hW hTags hCust t' ht' ht80 K₀ hK₀ τ v hv hv80 q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink
  refine (Bridge.sampledKSLivePubRows n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv
    dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hLb hsz hTags hCust Bridge.kDep (2 ^ 192) (2 ^ 192) Bridge.one_le_kDep
    Nat.one_le_two_pow Nat.one_le_two_pow (1 / 2) (by norm_num) Bridge.rateM1_kDep t' ht' K₀ hK₀ τ v hv q hCR
    hLink).trans ?_
  exact ENNReal.ofReal_le_ofReal (Bridge.deltaPub_le_deployed n kd K₀ (kd * D.unroll.ports.size + 3) hW
    (by linarith) (by linarith) (by linarith) (by linarith) (by linarith) (by linarith))

end Flock.Guarantees
