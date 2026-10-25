import Proofs.Flock.Soundness.Discharge.FailClosed.General
import Proofs.Flock.Soundness.Discharge.Tables.Headline

set_option Elab.async false

/-!
# The hidden-output headlines without `--zk` at the live subset draw

The one-table hidden-output headlines without `--zk` (`FailClosed.flock_headline_exec_hidden_dec`,
`FailClosed.flock_headline_exec_hiddenG`) are at the live stratified draw. flock-verify also accepts such a session at
the live subset draw (`ProvedScope.lawWhy`: a subset draw is covered under `--zk` and at hidden outputs), the draw every
one_stage driver takes, and here are the same headlines there:

* `Prog.flock_headline_live_onLX`: `Prog.flock_headline_live_onX` at any law `L`, its miss term `L.miss K`
  (`flock_e2e_count_hm96LX`, carried to the live audit by `execRefines_on`, as `Prog.flock_headline_live_tables_subset`
  does at `J` tables);
* `HiddenExec.flock_headline_hidden_subset`: `flock_headline_hidden` at the live `kd`-subset draw of the `n` units under
  A3, its count curve the uniform `kd`-subset's (`Tables.execOS_subset_miss_le`);
* `FailClosed.flock_headline_exec_hidden_subset_dec` (the wires feed neither the unit nor the outputs, at
  `Placed.keyProg`) and `FailClosed.flock_headline_exec_hiddenG_subset` (they feed either, at `instProg`): the two over
  `Verifies`, with the decoder discharged (`decodesOn_execH`, `decodesOn_execHI`, each at any law).

Each takes what its stratified form takes, with the subset law's parameters (`kd`, `hkd`, `Lb`, `src` over one byte
stream) in place of the strata's: the same named assumptions (A3 as `UniformRandomBytes`, `RecordsLive`, `DrawOkH` at the
law's draws, `TableCRL`, `LinkCRL`).
-/

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

/-- **The headline over hidden-output sessions, at the live subset draw.** `flock_headline_hidden` with the verifier's
draw the live `kd`-subset draw of the `n` units on a byte source A3 makes uniform, whose count curve is at most the
uniform `kd`-subset's (`Law.subset`). -/
theorem flock_headline_hidden_subset {Lp : Merkle.Leaf (List FlockLevel3.GF128) (List UInt8) Dig H512}
    {N' : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N'))
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
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
      (Law.execOS (n := n) src (.subset kd) n))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL (p.partition outs) H512 enc512 Lp (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm)
      (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 Lp
      (planAtH dj x₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω)
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
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditOn H512 enc512 Lp (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.subset kd) n)
          (planAtH dj x₀) (tabAtH dj x₀) (rs.binding hHm) k Rw M ρ t' :=
  (p.flock_headline_live_onLX outs (tablesAtH dj x₀) (tabAtH dj x₀) pp ones hConst zeros hZero lay rs hHm hk hRw hM hρ
    hr ht' K I wr opn dg dec one fin capOf N hdec τ hCR hKS).trans
    (add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K) le_rfl) le_rfl)

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.General (rsAtG rsAtG_computes)
open scoped ENNReal

/-- **One hidden-output non-ZK sampled session that flock-verify accepts at the live subset draw is sound, with the
decoder discharged.** `flock_headline_exec_hidden_dec` at the live `kd`-subset draw. -/
theorem flock_headline_exec_hidden_subset_dec
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = false) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)))
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
      (planAtH dj x₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω)
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
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src (.subset kd) n) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables (Reg := Reg)).binding
            (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact flock_headline_hidden_subset dj x₀ (Placed.keyProg (circS I) hs.built n) (Layout.outsOf (circS I) hs.built n)
    kd hkd Lb src hA3
    (ppH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    {Layout.oneG} (hConstH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    {Layout.zeroG} (hZeroH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built (Layout.outsOf (circS I) hs.built n))
    (layAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (pub := pubS I))
    (rsAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)
    (rsAtH_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) hs.tables)
    hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
    (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ
    (hCR hs) (hKS hs)

/-- **One hidden-output non-ZK sampled session that flock-verify accepts at the live subset draw, where the wires feed,
is sound.** `flock_headline_exec_hiddenG` at the live `kd`-subset draw. -/
theorem flock_headline_exec_hiddenG_subset
    {n : ℕ} {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = true) {mPts : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {Reg : Type}
    (kd : ℕ) (hkd : kd ≤ n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)))
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
      (planAtH dj x₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω)
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
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : Scope I, ((Law.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((instProg (circS I) n).partition (outsOfI (circS I) n)).linkBoundE
          (L := Law.execOS (n := n) src (.subset kd) n) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
            (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)) k Rw M ρ t') := by
  refine prob_scope (A := LiveAccepts I) (fun _ h => h.1) (fun _ h => h.2) _ _ _ _ fun hs => ?_
  exact flock_headline_hidden_subset dj x₀ (instProg (circS I) n) (outsOfI (circS I) n) kd hkd Lb src hA3
    (ppHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀)
    {instOneG (c := circS I) (n := n)} (constCols_instProgH (sitesAtH_parse hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
    {instZeroG (c := circS I) (n := n)} (zeroCols_instProgH (sitesAtH_parse hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
    (layAtHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀ (pub := pubS I))
    (rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
    (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
    hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
    (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ (hCR hs) (hKS hs)

end FlockSoundness.Discharge.FailClosed
