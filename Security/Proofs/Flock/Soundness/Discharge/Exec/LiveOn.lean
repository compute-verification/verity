import Proofs.Flock.Soundness.Discharge.Exec.Live

/-!
# R11 for one-table sessions, with the decoder right at the law's draws only

`DecodesAll` asks `Decodes` at every draw `S` and registration. The audit plays only the law's draws `L.draw ω`, so
`DecodesOn L` asks it there and nowhere else, and `execRefines_on` and `Prog.flock_headline_live_on` take it in place
of `DecodesAll`, with the same bound. `DecodesAll` implies it (`decodesOn_of_all`), and at it the simulated prover is
the same (`simAudit_eq_on`). At a draw the law never makes, the table `T S R` and the decoder are unconstrained; at a
draw it makes, `T S R` must be the table flock-verify sets up there, `m_pts` included (`Pts.lean`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts, at the law's draws** (R11b): `DecodesAll` at the
draws `L.draw ω` only. -/
def DecodesOn (L : Law n) : Prop :=
  ∀ ω R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => Accepts I (wr (L.draw ω) R o)) (·.accepted = true) (opn (L.draw ω) R) (dec (L.draw ω) R)
    (one (L.draw ω) R) (fin (L.draw ω) R) (capOf (L.draw ω) R) (repsOf (D := D) execArith (T (L.draw ω) R))

omit [Nonempty D] [Inhabited D] [Inhabited Sl] in
theorem decodesOn_of_all (L : Law n) (h : DecodesAll H E Lf T I wr opn dec one fin capOf) :
    DecodesOn H E Lf T I wr opn dec one fin capOf L :=
  fun ω R => h (L.draw ω) R

variable {L : Law n}

/-- **The simulated audit strategy**, from the decoder at the law's draws: the live prover's registration, and at each
draw the simulated session strategy against its live session. -/
noncomputable def simAuditOn (hdec : DecodesOn H E Lf T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessL execArith H E Lf (T (L.draw ω) τ.1) (fun o => Accepts I (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1) dg
      (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **R11 for one-table sessions.** Against any live prover `τ` of the live audit whose session is `liveTable` written
out by `wr`, flock-verify's acceptance is no likelier than the salted audit's acceptance against the simulated prover
`simAuditOn … τ`, with no live event added (`bad = ⊥`), whenever the decoder is right on accepted sessions
at the law's draws (`DecodesOn`). -/
theorem execRefines_on (hdec : DecodesOn H E Lf T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAccepts I) (fun _ => False)
      (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) (simAuditOn H E Lf T I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAccepts I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionL H E Lf (fun S R => [T S R]) (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simAuditOn H E Lf T I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditOn batchedSessionL
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  refine (live_le_sessionBL execArith H E Lf (T (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

/-! ## The headline at the executable, for one-table sessions -/

variable {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))

/-- **The headline at the executable, for one-table sessions of the challenger.** Let the verifier draw the units
as in `Prog.flock_headline_L` (the live stratified draw under A3), let each draw's session be one table `T S R`, and let
the live session be `liveTable` (the challenger, its coins uniform per round and drawn after the round's retained
bytes), written out as a session directory by `wr`. Against any live prover `τ`, flock-verify accepts with at least `K`
wrong units of the committed values of the simulated prover (`simAuditOn … τ`) with probability at most
`miss(K) + ε_ks + δ_link`: `Prog.flock_headline_execL` with `hRef` discharged by `execRefines_on`, so `Pr[bad] = 0`.
Beyond `Prog.flock_headline_L`'s (`pp`, `hConst`, `hZero`, `lay`, `rs`, `hHm`, and the assumptions through `hA3`, `hKS`,
`hCR`), it takes `hdec`: the decoder is right on the sessions flock-verify accepts, at the draws the law makes (R11b). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_on
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
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB [T S R] j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 T I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      [T ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))] j
      (batchStratL H512 enc512 leaf512 (fun S R => [T S R])
        (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' := by
  have h := p.flock_headline_execL outs (fun S R => [T S R]) tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOf dg mPts N wr) τ
    (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_on H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) hCR hKS
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

/-- `Prog.flock_headline_live_on` at rows each under its own encoding (`Binding.HmRowsX`). -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_onX
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
    (K : ℕ) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 T I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      [T ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))] j
      (batchStratL H512 enc512 leaf512 (fun S R => [T S R])
        (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' := by
  have h := p.flock_headline_execLX outs (fun S R => [T S R]) tab σs ks hks St hS Kd pop Lb src hA3 pp ones hConst
    zeros hZero lay rs hHm hk hRw hM hρ hr ht K I (liveOf dg mPts N wr) τ
    (simAuditOn H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_on H512 enc512 leaf512 T I wr opn dg dec one fin capOf N hdec τ) hCR hKS
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

/-- At `DecodesAll`, the simulated prover is `simAudit`'s. -/
theorem simAudit_eq_on (hdec : DecodesAll H E Lf T I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    simAudit H E Lf T I wr opn dg dec one fin capOf N hdec τ =
      simAuditOn H E Lf T I wr opn dg dec one fin capOf N (decodesOn_of_all H E Lf T I wr opn dec one fin capOf L hdec) τ :=
  rfl

end FlockSoundness.Discharge.Exec
