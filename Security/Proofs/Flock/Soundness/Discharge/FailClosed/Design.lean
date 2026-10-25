import Proofs.Flock.Soundness.Discharge.FailClosed.General
import Proofs.Flock.Soundness.Discharge.Tables.Headline

set_option Elab.async false

/-!
# The hidden-output headlines without `--zk` at a draw design

flock-verify draws a one-table session without `--zk` by one of two laws (`ProvedScope.lawWhy`): the live stratified
draw, and at hidden outputs also the live subset draw, the draw every one_stage driver takes. Neither headline reads
the law but through its miss term, so one statement covers both: a **draw design** `DrawDesign n` names the live draw's
law and population and the ideal miss that bounds its count curve under A3 (`DrawDesign.miss_le`, which is
`Law.execOS_miss_le` at the stratified draw, at every assignment of units to strata that the strata describe, and
`Tables.execOS_subset_miss_le` at the subset draw).

* `Prog.flock_headline_live_onLX`: `Prog.flock_headline_live_onX` at any law `L`, its miss term `L.miss K`
  (`flock_e2e_count_hm96LX`, carried to the live audit by `execRefines_on`, as `Prog.flock_headline_live_tables_subset`
  does at `J` tables);
* `HiddenExec.flock_headline_hidden_design`: `flock_headline_hidden` at a design's live draw under A3, its miss term the
  design's ideal miss;
* `FailClosed.flock_headline_exec_hidden_design_dec` (the wires feed neither the unit nor the outputs, at
  `Placed.keyProg`) and `FailClosed.flock_headline_exec_hiddenG_design` (they feed either, at `instProg`): the two over
  `Verifies`, with the decoder discharged (`decodesOn_execH`, `decodesOn_execHI`, each at any law).

At `DrawDesign.stratified` each bounds by the infimum of its stratified form's bound
(`FailClosed.flock_headline_exec_hidden_dec`, `FailClosed.flock_headline_exec_hiddenG`) over the assignments the strata
describe, and at `DrawDesign.subset` by the live subset draw's. Each takes the same named
assumptions at either law: A3 as `UniformRandomBytes` on the design's byte streams, `RecordsLive`, `DrawOkH` at the
law's draws, `TableCRL`, `LinkCRL`.
-/

namespace FlockSoundness.Discharge

open Game Model Accounting Refine Audit

/-- **A draw design**: the law flock-verify's live draw of a one-table session without `--zk` is by. The stratified draw
at executable strata `St`, stratum `s` drawing `(St s).k` of its units, over `pop` units; or the subset draw of `kd` of
the `n` units. -/
inductive DrawDesign (n : ℕ) : Type
  | stratified {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  | subset (kd : ℕ) (hkd : kd ≤ n)

namespace DrawDesign

variable {n : ℕ}

/-- The number of byte streams the live draw reads: one per stratum, one for the subset draw. -/
def streams : DrawDesign n → ℕ
  | @stratified _ m _ _ _ => m
  | subset _ _ => 1

/-- The executable law the live draw runs (`Flock.Draw.drawOS`'s). -/
def law : DrawDesign n → Flock.Draw.Law
  | stratified St Kd _ => .stratified Kd (Array.ofFn St)
  | subset kd _ => .subset kd

/-- The population the live draw is over. -/
def pop : DrawDesign n → ℕ
  | stratified _ _ pop => pop
  | subset _ _ => n

/-- The ideal miss that bounds the live draw's count curve under A3 (`miss_le`). At the stratified draw it is the
stratified law's at every assignment `σs` of the units to the strata that the strata describe (`Law.ExecStrata`), the
infimum over them; at the subset draw it is the subset law's. -/
noncomputable def idealMiss : DrawDesign n → ℕ → ENNReal
  | @stratified _ m St _ _, K =>
    ⨅ σs : Fin n → Fin m, ⨅ _ : Law.ExecStrata σs (fun s => (St s).k) St,
    ⨅ hks : ∀ s, (St s).k ≤ (Law.stratum σs s).card, (Law.stratified σs (fun s => (St s).k) hks).miss K
  | subset kd hkd, K => (Law.subset n kd hkd).miss K

/-- **Under A3 the live draw misses no more than the ideal miss**: `Law.execOS_miss_le` at every assignment the strata
describe, at the stratified draw, and `Tables.execOS_subset_miss_le` at the subset draw. -/
theorem miss_le (D : DrawDesign n) : ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ)
    (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256), Assumptions.UniformRandomBytes Lb src → ∀ K : ℕ,
    (Law.execOS (n := n) src D.law D.pop).miss K ≤ D.idealMiss K := by
  cases D with
  | stratified St Kd pop =>
    intro Ω _ _ Lb src hA3 K
    exact le_iInf fun σs => le_iInf fun hS => le_iInf fun hks =>
      Law.execOS_miss_le σs (fun s => (St s).k) hks St hS Kd pop Lb src hA3 K
  | subset kd hkd =>
    intro Ω _ _ Lb src hA3 K
    exact Tables.execOS_subset_miss_le kd hkd Lb src hA3 K

end DrawDesign

end FlockSoundness.Discharge

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Refine Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}

/-- **The headline at the executable, for one-table sessions of the challenger, at any law.**
`Prog.flock_headline_live_onX` with the verifier's draw any law `L` and the miss term `L.miss K`. -/
theorem _root_.FlockSoundness.Prog.flock_headline_live_onLX {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {L : Law n}
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
    (hdec : DecodesOn H512 enc512 Lp T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 Lp (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      [T (L.draw ω)
        (reg (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ))] j
      (batchStratL H512 enc512 Lp (fun S R => [T S R])
        (cont (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 Lp (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ)) k Rw))).card)
        (auditLive L Reg (liveOf dg mPts N wr)) τ ≤
      L.miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) L k qF qS
          (reg (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := L)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' := by
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hH := flock_e2e_count_hm96LX (p.partition outs) H512 enc512 Lp (fun S R => [T S R]) tab
    (execAccept := fun o => o.1 = true) pp.derived ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K
    (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ) hCR hKS
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR H512 hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hH
  have h := flock_headline_exec_L (fun S R => [T S R]) I (liveOf dg mPts N wr) τ
    (simAuditOn H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ)
    (execRefines_on H512 enc512 Lp T I wr opn dg dec one fin capOf N hdec τ) _ hH
  rwa [prob_eq_zero_of_out (p := fun _ => False) (fun _ _ h => h) τ, add_zero] at h

end FlockSoundness.Discharge.Exec

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The headline over hidden-output sessions, at a draw design.** `flock_headline_hidden` with the verifier's draw the
design's live draw on byte streams A3 makes uniform, whose count curve is at most the design's ideal miss
(`DrawDesign.miss_le`). -/
theorem flock_headline_hidden_design {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))
    (D : DrawDesign n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ) (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (pp : ProgPlaces p outs (planAtH (Reg := Reg) dj x₀) (tabAtH dj x₀))
    (ones : Finset (Fin N')) (hConst : (p.partition outs).ConstCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place ones)
    (zeros : Finset (Fin N')) (hZero : (p.partition outs).ZeroCols (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (planAtH dj x₀) (tabAtH dj x₀)
      ((p.partition outs).placeDecoder (planAtH dj x₀) (tabAtH dj x₀) pp.derived.place))
    (rs : Binding.HmRowsX lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dec one fin capOf
      (Law.execOS (n := n) src D.law D.pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 Lp (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm)
      (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      (planAtH dj x₀ ((Law.execOS (n := n) src D.law D.pop).draw ω)
        (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 Lp (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 Lp (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm)
              (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)) τ ≤
      D.idealMiss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src D.law D.pop) k qF qS
          (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src D.law D.pop)
          (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm) k Rw M ρ t' :=
  (p.flock_headline_live_onLX outs (tablesAtH dj x₀) (tabAtH dj x₀) pp ones hConst zeros hZero lay rs hHm hk hRw hM hρ
    hr ht' K I wr opn dg dec one fin capOf N hdec τ hCR hKS).trans
    (add_le_add (add_le_add (D.miss_le Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.General (rsAtG rsAtG_computes)
open scoped ENNReal

/-- **One hidden-output non-ZK sampled session that flock-verify accepts at a draw design is sound, with the decoder
discharged.** `flock_headline_exec_hidden_dec` at any design's live draw. -/
theorem flock_headline_exec_hidden_design_dec
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = false) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
    (D : DrawDesign n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ) (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src D.law D.pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : Scope I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
        (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables))
      (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src D.law D.pop).draw ω)
        (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifies I o ∧ ∀ hs : Scope I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
                (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables))
              (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, (D.idealMiss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src D.law D.pop) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src D.law D.pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
            (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact flock_headline_hidden_design dj x₀ (Placed.keyProg (circS I) hs.built n) (Layout.outsOf (circS I) hs.built n)
    D Lb src hA3
    (ppH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    {Layout.oneG} (hConstH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    {Layout.zeroG} (hZeroH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    (layAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (pub := pubS I))
    (rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)
    (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)
    hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
    (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ
    (hCR hs) (hKS hs)

/-- **One hidden-output non-ZK sampled session that flock-verify accepts at a draw design, where the wires feed, is
sound.** `flock_headline_exec_hiddenG` at any design's live draw. -/
theorem flock_headline_exec_hiddenG_design
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = true) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
    (D : DrawDesign n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ) (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src D.law D.pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : Scope I, LinkCRL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀) (tabAtH dj x₀)
      ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
        (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built))
      (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : Scope I, ∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src D.law D.pop).draw ω)
        (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifies I o ∧ ∀ hs : Scope I,
        K ≤ (((instProg (circS I) n).partition (outsOfI (circS I) n)).wrong
          (Xpub {instOneG (c := circS I) (n := n)} {instZeroG (c := circS I) (n := n)}
            (XplurL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
              (tabAtH dj x₀)
              ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
                (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built))
              (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, (D.idealMiss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src D.law D.pop) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((instProg (circS I) n).partition (outsOfI (circS I) n)).linkBoundE
          (L := Law.execOS (n := n) src D.law D.pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
            (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact flock_headline_hidden_design dj x₀ (instProg (circS I) n) (outsOfI (circS I) n) D Lb src hA3
    (ppHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀)
    {instOneG (c := circS I) (n := n)} (constCols_instProgH (sitesAtH_parse hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
    {instZeroG (c := circS I) (n := n)} (zeroCols_instProgH (sitesAtH_parse hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
    (layAtHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀ (pub := pubS I))
    (rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
    (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
    hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
    (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ (hCR hs) (hKS hs)

end FlockSoundness.Discharge.FailClosed
