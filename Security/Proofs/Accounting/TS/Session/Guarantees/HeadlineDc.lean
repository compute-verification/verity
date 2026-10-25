import Proofs.Accounting.TS.Session.Assumptions
import Models.Accounting.Guarantees.HeadlineDc

/-!
# The H200's and B200's headlines at a window of C-Flock sessions (the statements)

`Sm90SheetHeadline`, `Sm90MeasuredHeadline`, `Sm90MeasuredI3Headline`, `Sm100SheetHeadline` and `Sm100MeasuredHeadline`
take proof soundness at `δ_ZK` (`ProofSoundAt pf δzk`) as a named premise. Here each is restated at a window whose
proofs are C-Flock sessions, with that premise replaced by `Accounting.Guarantees.TS.SessionSoundAt`'s hypotheses at
`δzk = δs + δ`, as `Session/Guarantees/Headline` does for sm_120's. Every other hypothesis, the bound and the event are
the headline's own.

They sit in the proofs package because the sessions do (`reads_exempt`), and bind no `X`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
open scoped ENNReal

/-- **The H200's headline at the data sheet, at a window of C-Flock sessions**: `Sm90SheetHeadline` for a proof layer
`pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm90SheetHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 34) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (sess : Paths Ω Sm90.checks → Fin Sm90.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm90.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm90.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm90.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm90.sheet (Pl.run x).run) → (∀ x, SheetWords Sm90.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm90.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw) Sm90.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm90.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm90.model.credit Sm90.flat < 1 - 944 / 100000) ≤
      (1 / 2) ^ 40

/-- **The H200's headline on its measured vector, at a window of C-Flock sessions**: `Sm90MeasuredHeadline` for a proof
layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm90MeasuredHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (sess : Paths Ω Sm90.checks → Fin Sm90.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm90Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw) Sm90Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf →
    RightFixed Pl pf Sm90Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90Measured.model.credit Sm90Measured.flat < 1 - 1412 / 100000) ≤ (1 / 2) ^ 40

/-- **The H200's headline on its measured vector under (I3) at 3/5, at a window of C-Flock sessions**:
`Sm90MeasuredI3Headline` for a proof layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its
`ProofSoundAt pf δzk` replaced by `SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm90MeasuredI3HeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (sess : Paths Ω Sm90.checks → Fin Sm90.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm90MeasuredI3.model (Pl.run x)) → (∀ x, ScalarRoute Sm90MeasuredI3.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90MeasuredI3.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90MeasuredI3.model.Nw) Sm90MeasuredI3.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf →
    RightFixed Pl pf Sm90MeasuredI3.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90MeasuredI3.model.credit Sm90Measured.flat < 1 - 1086 / 100000) ≤ (1 / 2) ^ 40

/-- **The B200's headline at the data sheet, at a window of C-Flock sessions**: `Sm100SheetHeadline` for a proof layer
`pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm100SheetHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 27) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V)
    (sess : Paths Ω Sm100.checks → Fin Sm100.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm100.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm100.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm100.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm100.sheet (Pl.run x).run) → (∀ x, SheetWords Sm100.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm100.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw)
      Sm100.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm100.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm100.model.credit Sm100.flat < 1 - 783 / 100000) ≤
      (1 / 2) ^ 40

/-- **The B200's headline on its measured vector, at a window of C-Flock sessions**: `Sm100MeasuredHeadline` for a proof
layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm100MeasuredHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V)
    (sess : Paths Ω Sm100.checks → Fin Sm100.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm100Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm100Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw) Sm100Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf →
    RightFixed Pl pf Sm100Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm100Measured.model.credit Sm100Measured.flat < 1 - 900 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
