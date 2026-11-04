import Proofs.Flock.Soundness.Discharge.ZkExec.Refines
import Proofs.Flock.Soundness.Discharge.ZkLink

/-!
# flock-verify `--zk`'s headline at the compiled `--zk` table

`Refines.lean`'s one-table refinement and headline at `Discharge/ZkLink`'s compiled `--zk` table. At each draw `S` and
registration `R` the table is `T S R : TabZK 𝔽 mPts`, its reps `ZkLink.repZCAt A Hs (T S R)` (`repZ`), the level-0 cap
first (`Fin 1 → Cap D`), the openings `ZkLink.OpensZ 𝔽 D Sl` last, and the verdict `ZkLink.acceptedZC` (`outZ`, `accZ`).
That table's `sessionZC` is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_repZ`, by
`ZkLink.batchedSessionZC_one`), every rep's game has strategies (`ZkLink.allInh_repZCAt`), and the simulated prover
`simAuditZK … τ` is a strategy of that audit (`simAuditZC`), where the compiled count curve `ZkLink.zk_flock_countC`
applies:

* `zk_headline_exec_zc`: against any live prover, at any law,
  `Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`;
* `zk_headline_exec_zc_execOS`: the same at the live draw (`Law.execOS`, `drawOS`'s stratified law) under A3
  (`UniformRandomBytes`), with `Law.stratified`'s count curve for `miss(K)`.

The wrong units are those of the simulated prover's committed values (`ZkLink.XplurZC` at `simAuditZC … τ`). Open,
named in the hypotheses: `hdec : DecodesZK …` (the per-transcript refinement, R11b at `--zk`), `hj : JointZC …`
(`ε_ks`), and `hlow : LoweringSoundZK …` with the value binding `vb`, as at `Discharge/ZkLink`; `δ_link` is `⊤` at a
prover outside A2 (`linkBoundZC`). Open and not a hypothesis: the coin tree (the live audit draws each round's coins
fresh; `Discharge/ZkCoins`), and `J > 1` (the plan has one table).
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Model Accounting ZkSession
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
  (A : Arith 𝔽 K) (Hs : ZkLink.Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)

/-- The one-table plan: table `T S R` alone at each draw and registration. -/
abbrev planOne : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S R => [T S R]

/-- The compiled `--zk` table's reps at each draw and registration (`Refine.modelTable`'s `G`). -/
noncomputable def repZ (S : Finset (Fin n)) (R : Reg) : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (ZkLink.PendZ 𝔽 D Sl) :=
  ZkLink.repZCAt A Hs (T S R)

/-- The compiled `--zk` table's output: the one table's `outZC` at its level-0 cap. -/
noncomputable def outZ (S : Finset (Fin n)) (R : Reg) (caps : Fin 1 → Cap D)
    (pend : ZkLink.PendZ 𝔽 D Sl × ZkLink.PendZ 𝔽 D Sl) (os : ZkLink.OpensZ 𝔽 D Sl) : List (ZkLink.OutZC 𝔽 D Sl) :=
  [ZkLink.outZC Hs (T S R) (caps 0) pend os]

open Classical in
/-- The compiled `--zk` table's verdict. -/
noncomputable def accZ (S : Finset (Fin n)) (R : Reg) (outs : List (ZkLink.OutZC 𝔽 D Sl)) : Bool :=
  decide (ZkLink.acceptedZC [T S R] outs)

omit [Fintype K] in
/-- **The compiled `--zk` table's game is the compiled audit's session at the one-table plan.** -/
theorem sessionZC_repZ :
    sessionZC (repZ A Hs T) (outZ Hs T) (accZ T) = ZkLink.batchedSessionZC A Hs (planOne T) := by
  funext S R
  exact ZkLink.batchedSessionZC_one A Hs T S R

theorem prob_cast {β : Type} {G₁ G₂ : Game β} (h : G₁ = G₂) (p : β → Prop) (σ : Strategy G₁) :
    prob p G₂ (cast (congrArg Strategy h) σ) = prob p G₁ σ := by
  subst h
  rfl

variable (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → ZkLink.OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}

/-- **The simulated prover as a strategy of the compiled `--zk` audit**: `simAuditZK … τ` at the compiled table, read
as a strategy of `audit L Reg (batchedSessionZC A Hs (planOne T))` through `sessionZC_repZ`. -/
noncomputable def simAuditZC (hdec : DecodesZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (ZkLink.batchedSessionZC A Hs (planOne T))) :=
  cast (congrArg Strategy (congrArg (audit L Reg) (sessionZC_repZ A Hs T)))
    (simAuditZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dg dec one fin capOf N
      (fun S R pts r => ZkLink.allInh_repZCAt A Hs (T S R) pts r) hdec τ)

variable {C : Circuit} (P : Partition C n) (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planOne T S R).length)
  {decode : DecoderZK (C := C) (planOne T) tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ZkLink.ValueBindingZ A P (planOne T) tab decode Hc)

/-- **The `--zk` headline at flock-verify's acceptance, at the compiled `--zk` table**: against any live prover `τ`,
flock-verify `--zk` accepts the live audit's transcript while the simulated prover's committed values have at least `K`
wrong units with probability at most `miss(K) + ε_ks + δ_link` of the compiled count curve at that prover. -/
theorem zk_headline_exec_zc (hlow : LoweringSoundZK A P (planOne T) tab decode) {k Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, ZkLink.rateZ (planOne T S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (ZkLink.sessionZC A Hs (planOne T S R)) →
      Fin (planOne T S R).length → ℝ}
    (hj : ZkLink.JointZC A Hs (planOne T) k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (hdec : DecodesZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    let σ := simAuditZC A Hs T I wr opn dg dec one fin capOf N hdec τ
    prob (fun o => LiveAcceptsZK I o ∧
        K ≤ (P.wrong (ZkLink.XplurZC A Hs P (planOne T) tab vb (reg σ) (cont σ) k Rw)).card)
        (auditLive L Reg (liveOf dg mPts N wr)) τ ≤
      L.miss K + ZkLink.ksAvgZC A Hs (planOne T) k a (reg σ) (cont σ) +
        ZkLink.linkBoundZC A Hs P (planOne T) tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ
  have hB := ZkLink.zk_flock_countC A Hs P (planOne T) tab vb hlow hRw hM hρ hr hj ht K σ
  have hc := prob_cast (congrArg (audit L Reg) (sessionZC_repZ A Hs T))
    (fun o => o.1 = true ∧ K ≤ (P.wrong (ZkLink.XplurZC A Hs P (planOne T) tab vb (reg σ) (cont σ) k Rw)).card)
    (simAuditZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dg dec one fin capOf N
      (fun S R pts r => ZkLink.allInh_repZCAt A Hs (T S R) pts r) hdec τ)
  exact zk_headline_exec_one (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dg dec one fin capOf N L _ hdec τ _ _
    (hc.symm.trans_le hB)

/-- **The `--zk` headline at flock-verify's acceptance, at the live draw under A3**: `zk_headline_exec_zc` with the
units drawn by `Flock.Draw.drawOS`'s stratified law on a byte source A3 makes uniform, `miss(K)` `Law.stratified`'s
count curve. -/
theorem zk_headline_exec_zc_execOS (hlow : LoweringSoundZK A P (planOne T) tab decode)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, ZkLink.rateZ (planOne T S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (ZkLink.sessionZC A Hs (planOne T S R)) →
      Fin (planOne T S R).length → ℝ}
    (hj : ZkLink.JointZC A Hs (planOne T) k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (hdec : DecodesZK (repZ A Hs T) (outZ Hs T) (accZ T) I wr opn dec one fin capOf)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr))) :
    let σ := simAuditZC A Hs T I wr opn dg dec one fin capOf N hdec τ
    prob (fun o => LiveAcceptsZK I o ∧
        K ≤ (P.wrong (ZkLink.XplurZC A Hs P (planOne T) tab vb (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K + ZkLink.ksAvgZC A Hs (planOne T) k a (reg σ) (cont σ) +
        ZkLink.linkBoundZC A Hs P (planOne T) tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  (zk_headline_exec_zc A Hs T I wr opn dg dec one fin capOf N P tab vb hlow hRw hM hρ hr hj ht K hdec τ).trans
    (add_le_add (add_le_add (Law.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.ZkExec
