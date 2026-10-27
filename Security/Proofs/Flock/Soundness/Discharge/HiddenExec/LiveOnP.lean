import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
import Proofs.Flock.Soundness.Discharge.Exec.LiveOn

/-!
# The execution refinement for one-table sessions at the verifier's public inputs

`LiveOn.lean` at flock-verify's acceptance at the verifier's public inputs `own` (`AcceptsP I own`) in place of
`Accepts I`: `LiveAcceptsP`, `DecodesOnP`, `simAuditOnP`, `execRefines_onP` and `Prog.flock_headline_live_onXP`, with
the same bound. At `own = none` each is the deployed one (`acceptsP_none`). The headline reads the event only through
the refinement (`exec_le_of_refines`), so `flock_headline_live_onXP` is `Prog.flock_headline_LX` at the simulated
prover, carried to the live audit by `execRefines_onP`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

/-- flock-verify accepts the live audit's transcript at the verifier's public inputs `own`. -/
def LiveAcceptsP (I : Inputs) (own : Option ByteArray) (o : Reg × Finset (Fin n) × Transcript) : Prop :=
  AcceptsP I own o.2.2

theorem liveAcceptsP_none (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) :
    LiveAcceptsP I none o ↔ LiveAccepts I o := Iff.rfl

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (own : Option ByteArray)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts at `own`, at the law's draws** (`DecodesOn` at
`AcceptsP I own`). -/
def DecodesOnP (L : Law n) : Prop :=
  ∀ ω R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => AcceptsP I own (wr (L.draw ω) R o)) (·.accepted = true) (opn (L.draw ω) R) (dec (L.draw ω) R)
    (one (L.draw ω) R) (fin (L.draw ω) R) (capOf (L.draw ω) R) (repsOf (D := D) execArith (T (L.draw ω) R))

variable {L : Law n}

/-- **The simulated audit strategy at `own`** (`simAuditOn`). -/
noncomputable def simAuditOnP (hdec : DecodesOnP H E Lf T I own wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessL execArith H E Lf (T (L.draw ω) τ.1) (fun o => AcceptsP I own (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1)
      dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **The execution refinement for one-table sessions at `own`** (`execRefines_on`): flock-verify's acceptance at `own`
is no likelier than the salted audit's acceptance against `simAuditOnP … τ`, with no live event added. -/
theorem execRefines_onP (hdec : DecodesOnP H E Lf T I own wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAcceptsP I own) (fun _ => False)
      (audit L Reg (batchedSessionL H E Lf fun S R => [T S R]))
      (simAuditOnP H E Lf T I own wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsP I own) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionL H E Lf (fun S R => [T S R]) (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simAuditOnP H E Lf T I own wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditOnP batchedSessionL
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  refine (live_le_sessionBL execArith H E Lf (T (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

variable {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))

/-- **The headline at the executable at the verifier's public inputs, for one-table sessions**
(`Prog.flock_headline_live_onX` at `AcceptsP I own`): against any live prover `τ`, flock-verify accepts at `own` with
at least `K` wrong units of the simulated prover's committed values with probability at most
`miss(K) + ε_ks + δ_link`, under `Prog.flock_headline_live_onX`'s hypotheses with `hdec` at `own` (`DecodesOnP`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_onXP
    {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (T : Finset (Fin n) → Reg → TabSpec mPts)
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin [T S R].length)
    (pp : ProgPlaces p outs (fun S R => [T S R]) tab)
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols (fun S R => [T S R]) tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols (fun S R => [T S R]) tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (fun S R => [T S R]) tab
      ((p.partition outs).placeDecoder (fun S R => [T S R]) tab pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB [T S R] j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (own : Option ByteArray) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOnP H512 enc512 Lp T I own wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 Lp (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      [T ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ))] j
      (batchStratL H512 enc512 Lp (fun S R => [T S R])
        (cont (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsP I own o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 Lp (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := p.flock_headline_LX outs (fun S R => [T S R]) tab (execAccept := fun o => o.1 = true) σs ks hks St hS Kd
    pop Lb src hA3 pp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditOnP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := exec_le_of_refines (execRefines_onP H512 enc512 Lp T I own wr opn dg dec one fin capOf N hdec τ) _ hH
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

end FlockSoundness.Discharge.Exec
