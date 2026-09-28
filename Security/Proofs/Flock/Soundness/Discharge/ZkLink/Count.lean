import Proofs.Flock.Soundness.Discharge.ZkLink.CompiledLink
import Proofs.Flock.Soundness.Binding.E2E

/-!
# The compiled `--zk` count curve

`extraction_audit_count` at the compiled analysis `analysisZC`, in the shape of `Binding.flock_e2e_count_hm96` and of
`zk_flock_count_execOS_m1` at the interactive-oracle layer:
$$\Pr[\text{accept}\wedge\text{at least }K\text{ wrong units}]\le\mathrm{miss}(K)+\varepsilon_{\rm ks}+\delta_{\rm link}
+\delta_{\rm tree},$$
with
* `miss(K)`: the law's count curve (`zk_flock_countC`), or `Law.stratified`'s at the live draw under A3
  (`zk_flock_countC_execOS`);
* `ε_ks = ksAvgZC` at a knowledge bound `a` with its joint inequality `hj : JointZC … a`, the named hypothesis this
  layer leaves open (the compiled `ε_ks`, the `--zk` counterpart of `joint_le_accBL` and `ksBoundAccB_le_strict`);
* `δ_link = linkBoundZC`, under A2 for the prover's link finders (`zk_linkSoundC`);
* `δ_tree = TreeRewind.bound`, under A1 (`cr/sha-512`) for the serving roots' rewinding finder (`hT`), for an
  acceptance event `execAccept` the compiled session's acceptance does not imply.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **The compiled `--zk` count curve**: at the plurality value layer, `δ(K) = miss K + ε_ks + δ_link`, with `ε_ks` the
compiled analysis's knowledge term at the bound `a` (`hj`) and `δ_link` under A2 (`linkBoundZC`). -/
theorem zk_flock_countC {L : Law n} (hlow : LoweringSoundZK A P planZ tab decode) {k Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_count (analysisZC A Hs P planZ tab decode (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow a hj)
    (zc_knowledgeSound A Hs P planZ tab decode _ _ hlow hj)
    (zk_linkSoundC A Hs P planZ tab vb hlow hRw hM hρ hr hj ht) K σ

/-- **The compiled `--zk` count curve at the live draw, with `δ_tree`.** With the units drawn by `Flock.Draw.drawOS`'s
stratified law on a byte source that A3 makes uniform (`hA3`), and an acceptance event `execAccept` whose excess over
the compiled session's acceptance a rewinding at the serving roots covers (`tr`, under A1 through `hT`):
`Pr[execAccept ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link + δ_tree`, `miss(K)` `Law.stratified`'s count
curve. -/
theorem zk_flock_countC_execOS (hlow : LoweringSoundZK A P planZ tab decode)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) + ENNReal.ofReal (tr.bound qT) := by
  refine (Binding.prob_split_accept _ σ execAccept _).trans (add_le_add ?_ (tr.le hT))
  refine (prob_mono (q := fun o => o.1 = true ∧
    K ≤ (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card) (fun o h => ⟨h.1.1, h.2⟩) _ _).trans ?_
  exact (zk_flock_countC A Hs P planZ tab vb hlow hRw hM hρ hr hj ht K σ).trans
    (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.ZkLink
