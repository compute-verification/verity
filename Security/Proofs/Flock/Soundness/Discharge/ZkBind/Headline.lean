import Proofs.Flock.Soundness.Discharge.ZkBind.Place
import Proofs.Flock.Soundness.Discharge.ZkBind.Bind
import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ

/-!
# The compiled `--zk` count curve on the circuit's program

`ZkLink.zk_flock_countC_execOS_cr` at the circuit's program on Δ's source keys (`Placed.keyProg c hU n`), with each
drawn unit placed at its site in its `--zk` table's accepted statement (`place := si.placeKey hU hparse outs`), and the
committed transcript the plurality value layer of the rows' binding (`vb := rs.bindingZ hHm`). Its `place` and `vb`
hypotheses become the sites (`si`), the parse (`hparse`), the layout (`lay`, `rs`) and `HmRowComputes` (`hHm`).
-/

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Audit ZkSession ZkLink StrictCR Placed Refine Binding
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith F K) (Hs : Hashes F D Sl) {n : ℕ} {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The compiled `--zk` count curve on the circuit's units, every term on the accepted assumptions.** If
`execAccept` holds, the probability that at least `K` units of the circuit's program (`keyProg c hU n`) are wrong at
the committed values (the plurality value layer of the rows' binding) is at most
`miss(K) + ε_ks + δ_link + δ_tree`, as in `zk_flock_countC_execOS_cr`. Each drawn unit is read at its template instance and block in its
`--zk` table's accepted statement (`si`, every such table's circuit file parsing to `c`), and the committed values are
the rows the drawn tables' witnesses hold under the registered leaves (`lay`, `rs`), bound by `HmRowComputes`
(`hHm`). -/
theorem zk_flock_countC_execOS_keyProg (hA : A.Correct) {c : Flock.Circuit} (hU : FlockSoundness.Built c.unit)
    (si : SitesZ planZ tab)
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      Flock.HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))
    (lay : LayoutZ A ((keyProg c hU n).partition outs) planZ tab
      (decodeOfPlace A ((keyProg c hU n).partition outs) planZ tab (si.placeKey hU hparse outs)))
    (rs : HmRowsZ lay) (hHm : rs.Computes)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs planZ)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition outs).wrong
        (XplurZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZC A Hs planZ)) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ planZ (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC A Hs ((keyProg c hU n).partition outs) planZ tab (rs.bindingZ hHm) k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr A Hs ((keyProg c hU n).partition outs) planZ tab hA σs ks hks St hS Kd pop Lb src hA3
    (si.placeKey hU hparse outs) (rs.bindingZ hHm) hk hRw hM hρ hr ht K σ hCR tr hT

end FlockSoundness.Discharge.ZkBind
