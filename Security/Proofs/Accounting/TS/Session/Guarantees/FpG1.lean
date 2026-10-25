import Proofs.Accounting.TS.Session.Assumptions
import Models.Accounting.Guarantees.FpG1

/-!
# FP8-g1's and W4A8-g1's headlines on sm_120 at a window of C-Flock sessions (the statements)

`Accounting.Guarantees.FpG1`'s four headlines take proof soundness at `δ_ZK` (`ProofSoundAt pf δzk`) as a named premise.
Here each is restated at a window whose proofs are C-Flock sessions, with that premise replaced by
`Accounting.Guarantees.TS.SessionSoundAt`'s hypotheses at `δzk = δs + δ`, as `Session/Guarantees/Headline` does for
sm_120's. Every other hypothesis, the bound and the event are the headline's own.

They sit in the proofs package because the sessions do (`reads_exempt`), and bind no `X`.
-/

namespace Accounting.Guarantees.FpG1

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Pouw.FpG1
open scoped ENNReal

/-- **FP8-g1's headline on sm_120 at the data sheet, at a window of C-Flock sessions**: `Sm120FpG1SheetHeadline` for a
proof layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm120FpG1SheetHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin (nπ + 1) → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → (Leaves × W))
    (δbind δs δ : ℝ),
    (∀ x w i, (pf.committed x w i).1.Fp8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **W4A8-g1's headline on sm_120 at the data sheet, at a window of C-Flock sessions**: `Sm120W4a8G1SheetHeadline` for
a proof layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced
by `SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm120W4a8G1SheetHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin (nπ + 1) → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → (Leaves × W))
    (δbind δs δ : ℝ),
    (∀ x w i, (pf.committed x w i).1.W4a8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **FP8-g1's headline on sm_120's measured vector, at a window of C-Flock sessions**: `Sm120FpG1MeasuredHeadline` for
a proof layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced
by `SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm120FpG1MeasuredHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin (nπ + 1) → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → (Leaves × W))
    (δbind δs δ : ℝ),
    (∀ x w i, (pf.committed x w i).1.Fp8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

/-- **W4A8-g1's headline on sm_120's measured vector, at a window of C-Flock sessions**: `Sm120W4a8G1MeasuredHeadline`
for a proof layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk`
replaced by `SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm120W4a8G1MeasuredHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin (nπ + 1) → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → (Leaves × W))
    (δbind δs δ : ℝ),
    (∀ x w i, (pf.committed x w i).1.W4a8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.FpG1
