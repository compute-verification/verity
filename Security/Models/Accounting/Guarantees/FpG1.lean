import Definitions.Accounting.Headline
import Definitions.Pouw.FpG1
import Models.Accounting.Assumptions.Sheet

/-!
# sm_120's headlines at FP8-g1 and W4A8-g1's windows (trusted)

FPX-DESIGN's g1 designs (note:20261019T0817Z-report-fpx-design, PR #2710) serve an FP8 model, or an FP4-weight model
beside Dither's int8 activations, through the record's own int8 transcript: each operand is an integer level on its
format's grid plus a live fair coin (`Definitions.Pouw.FpG1`), the int8 MMA multiplies those integers, and the epilogue
takes the coin's two rank-one terms off. On sm_120 that transcript is H1's: activations on E4M3's integers to 120,
weights on E4M3's integers to 60 or doubled E2M1 levels under integer block scales to 5 (`Grids.sm120`).

Each statement here is `Models.Accounting.Guarantees.Headline`'s, word for word, at a window whose proofs' relations
check g1's grids: every value a proof's relation accepts carries its GEMM operand entries (`ops`, any reading of a
value's entries, each with its side) on sm_120's g1 grids (`Grids.On`), the public property of the proof unit's
Program that the work was an FP8 model's (or an FP4-weight model's). That hypothesis is the only change. It adds no
premise: the record's statement quantifies over every value type `V` and every proof layer `pf`, relations and
derivations included, and its conclusion reads neither, so g1's windows are among the windows it covers, and the
statement here follows from it by instantiation (`Accounting.SecurityProofs.TS.G1Sm120SheetHeadline`). The hypothesis
is also vacuous at an empty `ι`: it restricts which windows the statement is about, never what it concludes.

What g1's windows need from the record's premises is no new Lean premise but a check of scope, outside these
statements: the luck premises (`OperandHMu`, `ReceiptLuckAt`, `WholeLuckAt`) are named Props over the receipt game,
and their cards have to hold at g1's committed values (E4M3 levels plus a coin, every level of magnitude at least 16
even: `Proofs.Pouw.FpG1`'s `e4m3Nat_even`); A7 (`SheetSemantics`) holds of every run here, and its evidence
(`benchmarks/pouw/theory/isa_escape.py`) checks its verdicts over every s8 activation and every H1 weight in
`[−63, 63]`, which contain g1's sm_120 entries (`Proofs.Pouw.FpG1`'s `sm120_act_int`, `sm120_wt_int`). B200 and
H200's g1 (`Grids.dc`, weights to 121) is Dither-2S's, whose headlines (#2699's `HeadlineDc`) are not on this branch.

The statements:

* `G1Sm120SheetHeadline`: `Sm120SheetHeadline` (0.656%) at every window whose relations check g1's sm_120 grids.
* `G1Sm120MeasuredHeadline`: `Sm120MeasuredHeadline` (0.591%) at every such window.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet, at FP8-g1 and W4A8-g1's windows**: `Sm120SheetHeadline`'s statement, word
for word, at a window of proofs `pf` whose relations check g1's sm_120 grids: every value `v` that a proof's relation
accepts carries its operand entries `ops v j` (each with its side) on `Grids.sm120`, E4M3's integers to 120 for
activations (or Dither's int8) and to 60 for weights (or doubled E2M1 levels under block scales to 5), each plus its
coin. Outside an event of probability at most `2^-40` every accepted window's time is at least `1 − 0.656%` of its
credit. The record's premises are its premises, unchanged; the grids' hypothesis restricts the windows, and the
statement is the record's at them. -/
def G1Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V ι : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V)
    (ops : V → ι → Pouw.FpG1.Side × Pouw.FpG1.Entry) (δbind δzk : ℝ),
    (∀ w i v, pf.rel w i v → ∀ j, Pouw.FpG1.Grids.sm120.On (ops v j).1 (ops v j).2) →
    (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFolded Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 656 / 100000) ≤
      (1 / 2) ^ 40

/-- **sm_120's headline on its measured vector, at FP8-g1 and W4A8-g1's windows**: `Sm120MeasuredHeadline`'s
statement, word for word, at a window of proofs `pf` whose relations check g1's sm_120 grids (as
`G1Sm120SheetHeadline`'s). Outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 0.591%` of its credit. -/
def G1Sm120MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V ι : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V)
    (ops : V → ι → Pouw.FpG1.Side × Pouw.FpG1.Entry) (δbind δzk : ℝ),
    (∀ w i v, pf.rel w i v → ∀ j, Pouw.FpG1.Grids.sm120.On (ops v j).1 (ops v j).2) →
    (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFolded Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
