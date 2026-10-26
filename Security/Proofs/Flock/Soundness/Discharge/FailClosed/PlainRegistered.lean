import Proofs.Flock.Soundness.Discharge.PlainReg
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec
import Proofs.Flock.Soundness.Discharge.FailClosed.General
import Proofs.Flock.Soundness.Discharge.FailClosed.Design

/-!
# The hidden-output headlines without `--zk`, over what `verify --registered F` accepts

`flock-verify verify --registered F` without `--zk` (`VerifiesR`: the verdict `PlainReg.AcceptsR` models and the statement
check with the verifier's registered reads, `ScopeR`) accepts, at hidden outputs and `hm96-sha512/v1` leaves, a statement
whose population's public file carries shared rows, which `verify` without `--registered` refuses. Each headline here is
`FailClosed.Design`'s over `VerifiesR`, at any draw design (`DrawDesign`: the stratified draw or the live subset draw),
with the registered reads checked:

* `flock_headline_exec_hiddenR`: where the wires feed neither the unit nor the outputs, `HiddenExec.flock_headline_hidden_design`
  at the unit's layout and `PlainReg.rsAtHS` (the rows with or without shared rows), its decoder discharged as in
  `FailClosed.flock_headline_exec_hidden_design_dec`, and `PlainReg.checked_of_acceptsRH` at every accepted outcome;
* `flock_headline_exec_hiddenRG`: where they feed, `HiddenExec.flock_headline_hidden_design` at the template
  instance's program, as `FailClosed.flock_headline_exec_hiddenG_design` (whose rows never read `pub.tables`), and `PlainReg.checked_of_acceptsRHI` at every accepted outcome.

At public outputs the check with `--registered F` is the check without it (`ScopeR.scope`), so what `verify --registered
F` accepts there `verify` accepts (`verifies_of_R`), and `FailClosed.Exec`'s public headlines cover it.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Inst
open scoped ENNReal

theorem ScopeR.tagsOkH {I : Inputs} (h : ScopeR I) (hh : I.tags.hiddenOutputs = true) : HiddenDec.TagsOkHL I :=
  ⟨hh, h.retainRounds, h.leaf, h.count⟩

/-- **What `flock-verify verify --registered F` accepts** (no `--zk`), `own` F's ports: the setup, registered check and
verdict `PlainReg.AcceptsR` models, and the statement check with the verifier's registered reads. -/
def VerifiesR (I : Inputs) (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  PlainReg.AcceptsR I own t ∧ ScopeR I

variable {mPts n : ℕ} {Reg : Type}

/-- `VerifiesR` at the live session's transcript. -/
def LiveVerifiesR (I : Inputs) (own : Array Flock.Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) :
    Prop :=
  VerifiesR I own o.2.2

/-- **At public outputs, `--registered F` accepts no more than `verify`**: the registered check only refuses more, and the
statement check is the same (`ScopeR.scope`). -/
theorem verifies_of_R {I : Inputs} {own : Array Flock.Registered.Port} {t : Exec.Transcript} (h : VerifiesR I own t)
    (hh : I.tags.hiddenOutputs = false) : Verifies I t :=
  ⟨PlainReg.accepts_of_R h.1, h.2.scope hh⟩

section Hidden

open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.PlainReg (rsAtHS rsAtHS_computes checked_of_acceptsRH checked_of_acceptsRHI)
open FlockSoundness.Discharge.ZkReg (OwnReadsH regPathsH)
open FlockSoundness.Discharge.General (rsAtG rsAtG_computes)

/-- **One hidden-output non-ZK sampled session that `flock-verify verify --registered F` accepts is sound, and its
registered reads are checked.** The first conjunct is `FailClosed.flock_headline_exec_hidden_design_dec`'s bound
over `VerifiesR` at a draw design, at a public file with or without shared rows (`rsAtHS`); the second, the registered check's consequence at
every accepted outcome (`PlainReg.checked_of_acceptsRH`). -/
theorem flock_headline_exec_hiddenR
    {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = false)
    (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    (D : DrawDesign n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ) (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src D.law D.pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : ScopeR I, LinkCRL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtHS hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) (Reg := Reg)).binding
        (rsAtHS_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh)))
      (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : ScopeR I, ∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src D.law D.pop).draw ω)
        (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifiesR I own o ∧ ∀ hs : ScopeR I,
        K ≤ (((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)) H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtHS hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) (Reg := Reg)).binding
                (rsAtHS_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh)))
              (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
                (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => [])
                (fun _ _ b _ => capOfRoot b) N (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)) τ ≤
      ⨆ hs : ScopeR I, (D.idealMiss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src D.law D.pop) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((Placed.keyProg (circS I) hs.built n).partition (Layout.outsOf (circS I) hs.built n)).linkBoundE
          (L := Law.execOS (n := n) src D.law D.pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtHS hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh) (Reg := Reg)).binding
            (rsAtHS_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh))) k Rw M ρ t') ∧
    ∀ hs : ScopeR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveVerifiesR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layAtH (Reg := Reg) hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (pub := pubS I))
          (rsAtHS hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh)),
        OwnReadsH own (circS I) (pubS I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsH (circS I) (pubS I) o.2.2 rd.port) :=
  ⟨prob_scope (A := LiveAccepts I) (fun _ h => PlainReg.accepts_of_R h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      flock_headline_hidden_design dj x₀ (Placed.keyProg (circS I) hs.built n) (Layout.outsOf (circS I) hs.built n)
        D Lb src hA3
        (ppH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀ hs.built
          (Layout.outsOf (circS I) hs.built n))
        {Layout.oneG} (hConstH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀
          hs.built (Layout.outsOf (circS I) hs.built n))
        {Layout.zeroG} (hZeroH hs.typed (hs.parseH hh) (rowsFit_of_scopeH hs.typed (hs.parseH hh) (hs.scopeH hf)) dj x₀
          hs.built (Layout.outsOf (circS I) hs.built n))
        (layAtH hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (pub := pubS I))
        (rsAtHS hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh))
        (rsAtHS_computes hs.typed (hs.parseH hh) (hs.scopeH hf) dj x₀ hs.built (hs.loadH hh))
        hk hRw hM hρ hr ht' K wr (opnAtH (tablesAtH dj x₀) I dj) dg (decAt (tablesAtH dj x₀) wOP wLIG)
        (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execH hs.typed (hs.parseH hh) (Layout.outNet_of_wiresFeed hf) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw)
        τ (hCR hs) (hKS hs),
    fun hs o hv rd hown =>
      checked_of_acceptsRH I hs.typed own dj x₀ (hs.parseH hh) (hs.scopeH hf) hs.built (hs.loadH hh) hh hs.count o
        hv.1 rd hown⟩

/-- **One hidden-output non-ZK sampled session that `flock-verify verify --registered F` accepts, where the wires feed,
is sound, and its registered reads are checked.** The first conjunct is `FailClosed.flock_headline_exec_hiddenG_design`'s
bound over `VerifiesR` at a draw design (its rows `rsAtG` read no `pub.tables`); the second, the registered check's consequence at every
accepted outcome (`PlainReg.checked_of_acceptsRHI`). -/
theorem flock_headline_exec_hiddenRG
    {I : Inputs} (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = true)
    (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    (D : DrawDesign n)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin D.streams → ℕ) (src : Ω → (i : Fin D.streams) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    (hwr : RecordsLive I dj wr)
    (hdraw : ∀ ω, DrawOkH I dj mPts ((Law.execOS (n := n) src D.law D.pop).draw ω))
    (τ : Strategy (auditLive (Law.execOS (n := n) src D.law D.pop) Reg (liveOf dg mPts N wr)))
    (hCR : ∀ hs : ScopeR I, LinkCRL ((instProg (circS I) n).partition (outsOfI (circS I) n)) H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀) (tabAtH dj x₀)
      ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
        (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built))
      (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))
      (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
        (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ hs : ScopeR I, ∀ ω j, TableCRL execArith H512 enc512 (leafOfId I.tags.merkleLeaf)
      (planAtH dj x₀ ((Law.execOS (n := n) src D.law D.pop).draw ω)
        (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ))) j
      (batchStratL H512 enc512 (leafOfId I.tags.merkleLeaf) (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
          (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
          (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ) ω)) qF qS) :
    prob (fun o => LiveVerifiesR I own o ∧ ∀ hs : ScopeR I,
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
      ⨆ hs : ScopeR I, (D.idealMiss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src D.law D.pop) k qF qS
          (reg (simAuditOn H512 enc512 (leafOfId I.tags.merkleLeaf) (tablesAtH dj x₀) I wr (opnAtH (tablesAtH dj x₀) I dj) dg
            (decAt (tablesAtH dj x₀) wOP wLIG) (fun _ _ _ => true) (fun _ _ _ _ => []) (fun _ _ b _ => capOfRoot b) N
            (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ)) +
        ((instProg (circS I) n).partition (outsOfI (circS I) n)).linkBoundE
          (L := Law.execOS (n := n) src D.law D.pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built (Reg := Reg)).binding
            (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)) k Rw M ρ t') ∧
    ∀ hs : ScopeR I, ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveVerifiesR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX
          (layAtHI (Reg := Reg) hs.typed (hs.parseH hh)
            (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀ (pub := pubS I))
          (rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built),
        OwnReadsH own (circS I) (pubS I) o.2.2 rd.isRead rd.port rd.readAt →
          rd.Checked o.1 (regPathsH (circS I) (pubS I) o.2.2 rd.port) :=
  ⟨prob_scope (A := LiveAccepts I) (fun _ h => PlainReg.accepts_of_R h.1) (fun _ h => h.2) _ _ _ _ fun hs =>
      flock_headline_hidden_design dj x₀ (instProg (circS I) n) (outsOfI (circS I) n) D Lb src hA3
        (ppHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀)
        {instOneG (c := circS I) (n := n)} (constCols_instProgH (sitesAtH_c hs.typed (hs.parseH hh)
          (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
        {instZeroG (c := circS I) (n := n)} (zeroCols_instProgH (sitesAtH_c hs.typed (hs.parseH hh)
          (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)).1 (outsOfI (circS I) n))
        (layAtHI hs.typed (hs.parseH hh) (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) dj x₀ (pub := pubS I))
        (rsAtG hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
        (rsAtG_computes hs.typed (hs.parseH hh) (hs.scopeGen hh hf) dj x₀ (hs.loadH hh) hs.built)
        hk hRw hM hρ hr ht' K wr _ dg _ _ _ _ N
        (decodesOn_execHI hs.typed (hs.parseH hh) dj x₀ _ wr (hs.tagsOkH hh) hwr hdraw) τ (hCR hs) (hKS hs),
    fun hs o hv rd hown =>
      checked_of_acceptsRHI I hs.typed own dj x₀ (hs.parseH hh)
        (General.rowsFit_of_scopeGenH hs.typed (hs.parseH hh) (hs.scopeGen hh hf)) hs.built (hs.loadH hh)
        (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).1 (Layout.scopeGenH_spec hs.built (hs.scopeGen hh hf)).2
        hh hs.count o hv.1 rd hown⟩

end Hidden

end FlockSoundness.Discharge.FailClosed
