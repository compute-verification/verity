module

public import Proofs.Flock.Soundness.Discharge.ZkBind.Place
public import Proofs.Flock.Soundness.Discharge.ZkBind.Bind
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
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

/-! The compiled `--zk` count curve on the circuit's program on Δ's source keys (`zk_flock_countC_execOS_keyProg`). -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Audit ZkSession ZkLink StrictCR Placed Refine Binding
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith F K) (Hs : Hashes F D Sl) {n : ℕ} {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

theorem zk_flock_countC_execOS_keyProg (hA : A.Correct) {c : Flock.Circuit} (hU : FlockSoundness.Built c.unit)
    (si : SitesZ planZ tab)
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      Flock.HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))
    (lay : LayoutZ A ((keyProg c hU n).partition outs) planZ tab
      (decodeOfPlace A ((keyProg c hU n).partition outs) planZ tab (si.placeKey hU hparse outs)))
    (rs : HmRowsZ lay) (hHm : rs.Computes)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition outs).incorrect
        (XplurZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) (reg σ) (cont σ) k Rw)).card)
        (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr A Hs ((keyProg c hU n).partition outs) planZ tab hA σs ks hks St hS Kd pop Lb src hA3
    (si.placeKey hU hparse outs) (rs.bindingZ hHm) hk hRw hM hρ hr ht K σ hCR tr hT

end FlockSoundness.Discharge.ZkBind
