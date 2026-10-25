import Proofs.Accounting.TS.Session.Assumptions
import Models.Accounting.Guarantees.Headline

/-!
# sm_120's headlines at a window of C-Flock sessions (the statements)

`Accounting.Guarantees.TS.Sm120SheetHeadline` and `Sm120MeasuredHeadline` take proof soundness at `δ_ZK`
(`ProofSoundAt pf δzk`) as a named premise. Here each is restated at a window whose proofs are C-Flock sessions, with
that premise replaced by `Accounting.Guarantees.TS.SessionSoundAt`'s hypotheses at `δzk = δs + δ`:
`Flock.SecurityProofs.EndToEndRegisteredRows`' collision bounds at every session (`RowsSession.CR`) and the session
premises `AcceptedLive`, `WindowLaw`, `ServedChecksHold` and `ServedRootsLinked`. Every other hypothesis, the bound and
the event are the headline's own. No served path meets `AcceptedLive` yet (`SessionSoundAt`'s docstring).

They sit in the proofs package because the sessions do (`reads_exempt`), and bind no `X`: `Proofs.Flock.SoundAtBridge`'s
imports make it a token.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
open scoped ENNReal

/-- **sm_120's headline at the data sheet, at a window of C-Flock sessions**: `Sm120SheetHeadline` for a proof layer
`pf` whose proof `i` of round `w` on the path `x` is the C-Flock session `sess x w i` with outcome `out x w i`, its
`ProofSoundAt pf δzk` replaced by `SessionSoundAt`'s hypotheses at `δzk = δs + δ`, so its budget reads
`S·(√δ_fork + nπ·(δs + δ)) ≤ 2^-41`. -/
def Sm120SheetHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V)
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **sm_120's headline on its measured vector, at a window of C-Flock sessions**: `Sm120MeasuredHeadline` for a proof
layer `pf` whose proofs are the C-Flock sessions `sess` with outcomes `out`, its `ProofSoundAt pf δzk` replaced by
`SessionSoundAt`'s hypotheses at `δzk = δs + δ`. -/
def Sm120MeasuredHeadlineSessions : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V)
    (sess : Paths Ω Sm120.checks → Fin Sm120.checks → Fin nπ → Flock.SoundAtBridge.RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : Flock.SoundAtBridge.RowsSession, (Fin s.Circ.N → Bool) → V)
    (δfork δs δ : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * (δs + δ)) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork →
    (∀ x w i, (sess x w i).CR) → Accounting.Assumptions.TS.AcceptedLive pf sess out →
    Accounting.Assumptions.TS.WindowLaw sess out → Accounting.Assumptions.TS.ServedChecksHold pf sess out rows →
    Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    ProofsPin pf → RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
