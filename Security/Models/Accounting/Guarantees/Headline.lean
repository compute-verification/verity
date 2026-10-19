import Definitions.Accounting.Headline
import Models.Accounting.Assumptions.Sheet

/-!
# The families' headline guarantees (trusted)

One statement per family and reading: the family's credited work is at least `1 − ε` of its credit, `ε` the reading's
headline, on one device's machine, one design's TS model on it and its floor program ♭, all exact data
(`Definitions.Accounting.Headline`). Each is `Models.Accounting.Guarantees.Sheet`'s `SheetDIdealFloor` at that data
with fold skippers charged (below), its parameters fixed (`π*`, `p`, `p₀`, `S`, `δ`), with every hypothesis that the
data discharge gone and the rest left as they are: the named assumptions of `Models.Accounting.Assumptions` and the
proof layer's pieces. A measured reading is `Models.Accounting.Guarantees.TS`'s `DIdealFloor` at its data the same
way, its machine the measured vector's rows. Each is proved as `Accounting.SecurityProofs.TS.G` in
`Proofs.Accounting.Headline`.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after
the commitments. The sampled path's `ε_sample` (#2466's `DrawnLayer`) and the Fiat–Shamir premium (decision 122) are
separate lines, in none of these statements.

The luck (I7) is split at the receipt (LEDGER §1.3) into two named luck premises, and `LuckOfReceipt` gives (I7) at
`¾ + ¼·½ = 7/8` from them, the receipt game's openings pinned:

* `OperandHMu` at `¾`, the first: (H-μ) at the deficient words (decision 110), which Lemma B at fibre 1 turns into the
  operand luck at `¾`;
* `ReceiptLuckAt` at `½`, the second: HMU's B′ case, LEDGER's case (b), named and conjectured (ASK-9 = A).

Case (a), which would make (I7) `¾`, restricts the cheater and is no premise. Neither is `OperandLuckCube`'s
hypothesis (degree-2 operand errors on cube bits), which only affine-operand runs meet (RT-HEADLINE): it is a proved
route to the operand luck, not a premise here.

Fold skippers are charged (RT-COMPOSE's B1, RT-HEADLINE §3). FOLD-OPT is per word folded (`FoldOptFolded`, ASK-6 = A):
each run's fold is priced at the words it folds, `(1 − Σ_w e_w/S)·N_w`, where `Uf : Unfolded` gives each round's
share `e_w` of words whose fold the run skips. A round with a positive unfolded share passes only by keeping the honest
committed value without the fold, which the whole-unit luck bounds: B′ at a whole unit (`WholeLuckAt` at `p₀ = ½`,
ASK-9 = A, live coins only). The whole-unit guess bound `2^-64` isn't granted, so `p₀` is B′'s `½`. The luck term is
the larger of two branches':

* the folded branch's `ε_jl` at `p = 7/8`;
* the unfolded branch's `(N_w w* + N_w φ*)/B_♭ · ln(1/δ′)/(ln 2 · S)`.

The unfolded branch binds on sm_120's sheet (`0.656%`); on sm_120's measured vector the folded branch binds
(`0.591%`). `ProofLayer.Pins` is met by main's circuit Definitions since PIN (#2417).

The statements:

* `Sm120SheetHeadline`: sm_120 (RTX PRO 6000) at its data sheet, H1 + fx1 at SCALAR-ROUTE's ½ floors: outside an
  event of probability at most `2^-40`, every accepted window of 48,000 equal live rounds runs at least `1 − 0.656%`
  of its credit, through A2u (below).
* `Sm120MeasuredHeadline`: sm_120 on its measured vector (VF3r-16), H1 + fx1 at SCALAR-ROUTE's `9/8` floors: outside
  an event of probability at most `2^-40`, every accepted window of 48,000 equal live rounds runs at least
  `1 − 0.591%` of its credit.

The window is in the statements' type: `Sm120.checks = 48,000` equal live rounds, one check a round, so neither says
anything about a window of another shape. The ε needs that many rounds. RT-INT-1's floor (Compose's live line,
note:20261019T0528Z-draft-rt-int): ε < 1% needs at least 10,552 equal live rounds a window, whatever the draw. The same
two branches with `S` rounds in place of 48,000 need 9,496 (sheet) and 9,652 (measured), and exceed 100% below 41
rounds. Today's served windows have 18 rounds.

The sheet reading holds only through A2u (`SheetDispatch`: each uniform-datapath instruction takes an issue slot),
which no vendor document states. DS-120 has no uniform-pipe row, so without A2u the machine model's LP gives 2.79% with
a measured uniform-pipe row (0.766 a SMSP-cycle) and 100% with none, against its 0.585% with A2u (RT-MACHINE-4,
note:20261019T0456Z-draft-rt-machine); A2u was measured to hold on sm_120 beside an issue-bound stream
(r20261019-054518-8cd2).

sm_80 has no headline: FOLD-OPT is false at its ALU-pipe certificate, which doesn't price duo's `IMAD.HI.U32` rotate
route on the FMA pipe (RT-LOGIC-10).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet**, under full checking with live coins. On DS-120's machine (`Sm120.sheet`),
H1 + fx1's TS model (`Sm120.model`, `q = q_T = ½`, `N_w = 128`) and its floor program (`Sm120.flat`), at the issue
facet's certificate (`Sm120.piStar`), over a window of `Sm120.checks = 48,000` equal live rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, A1–A7 hold on every run and FOLD-OPT per word folded prices every run's fold at the words it
folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose openings are pinned
(`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`, the first luck
premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and whose whole-unit luck is at
most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), binding fails with
probability at most `δ_bind` a round, every proof is sound up to `δ_ZK` (`SoundAt`), the relations pin every committed
value (`Pins`) and E1(c) holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 656/100000` (`1 − 0.656%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a
hypothesis: H1 + fx1 has no filler, so it holds at any run. `ε_sample` and the Fiat–Shamir premium are separate lines.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. RT-INT-1's floor: ε < 1% needs at
least 10,552 live rounds a window (9,496 for its two branches at `S` rounds); today's served windows have 18.

A2u holds this ε: A2 (`SheetDispatch`) counts each uniform-datapath instruction in an issue slot, which no vendor
document states, and DS-120 has no uniform-pipe row. Without A2u the machine model's LP gives 2.79% with a measured
uniform-pipe row and 100% with none (RT-MACHINE-4). -/
def Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ),
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

/-- **sm_120's headline on its measured vector**, under full checking with live coins. On VF3r-16's machine
(`Sm120Measured.machine`), H1 + fx1's TS model on it (`Sm120Measured.model`: the MMA forms at their committed words an
instruction, `q = q_T = 9/8`, `b_T = 0`, `N_w = 128`) and its floor program (`Sm120Measured.flat`), at JI3s_0's
certificate (`Sm120Measured.piStar`), over a window of `Sm120.checks = 48,000` equal live rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, (I1) `FacetCeilings` holds on VF3r-16's rows, (I2)/(I4) `WordsWritten` at the model's `ω`,
(I3) `ScalarRoute` at its floors and (I5) `NoSum` hold on every run and (I6) FOLD-OPT per word folded prices every
run's fold at the words it folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose
openings are pinned
(`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`, the first luck
premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and whose whole-unit luck is at
most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), binding fails with
probability at most `δ_bind` a round, every proof is sound up to `δ_ZK` (`SoundAt`), the relations pin every committed
value (`Pins`) and E1(c) holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 591/100000` (`1 − 0.591%`) of the credit `T_cred = B_♭/f_max`. (I5) `SideUnits` and
FILL-OPT are not hypotheses: the model has no side unit and no filler, so they hold at any run. On the lane route (I3)
at `9/8` is what `ClassPSlots` gives (`1 + (32 − w)/130` slots a word, FLOORS, note:20261018T0230Z-draft-floors) once
writers create at most `w = 15` incidences, a cap nothing proves yet; on the table route it is SCALAR-ROUTE's, outside
`ClassPSlots`. `ε_sample` and the Fiat–Shamir premium are separate lines.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. RT-INT-1's floor: ε < 1% needs at
least 10,552 live rounds a window (9,652 for its two branches at `S` rounds); today's served windows have 18. -/
def Sm120MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ),
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
