import Definitions.Accounting.Headline
import Models.Accounting.Assumptions.Sheet

/-!
# The families' headline guarantees (trusted)

One statement per family and reading: the family's credited work is at least `1 − ε` of its credit, `ε` the reading's
headline, on one device's machine, one design's TS model on it and its floor program ♭, all exact data
(`Definitions.Accounting.Headline`). Each is `Models.Accounting.Guarantees.Sheet`'s `SheetDIdealFloor` at that data,
its parameters fixed (`π*`, `p`, `S`, `δ`), with every hypothesis that the data discharge gone and the rest left as
they are: the named assumptions of `Models.Accounting.Assumptions` and the proof layer's pieces. A measured reading is
`Models.Accounting.Guarantees.TS`'s `DIdealFloor` at its data the same way, its machine the measured vector's rows.
Each is proved as `Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.Headline`.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after
the commitments. The sampled path's `ε_sample` (#2466's `DrawnLayer`) and the Fiat–Shamir premium (decision 122) are
separate lines, in none of these statements.

The luck (I7) is split at the receipt (LEDGER §1.3) into two named luck premises, and `LuckOfReceipt` gives (I7) at
`¾ + ¼·½ = 7/8` from them, the receipt game's openings pinned:

* `OperandHMu` at `¾`, the first: (H-μ) at the deficient words (decision 110), which Lemma B at fibre 1 turns into the
  operand luck at `¾`;
* `ReceiptLuckAt` at `½`, the second: HMU's B′ case, LEDGER's case (b).

Case (a), which would make (I7) `¾`, restricts the cheater and is no premise. Neither is `OperandLuckCube`'s
hypothesis (degree-2 operand errors on cube bits), which only affine-operand runs meet (RT-HEADLINE): it is a proved
route to the operand luck, not a premise here.

Two caveats bound what the statements say (RT-HEADLINE, note:20261018T1057Z-draft-rt-headline):

* `∀ x, FoldOpt` at `μ_f = 0` excludes every run that skips a fold (RT-COMPOSE's B1). Charging fold skippers instead,
  FOLD-OPT per fold done with the luck split into folded rounds at `7/8` and unfolded ones at `½`, moves the luck term
  to `0.994·41/48,000 = 0.0849%`: `ε ≈ 0.656%` instead of `0.599%` on sm_120's sheet, a similar shift on its measured
  vector, and `ε ≈ 0.232%` instead of `0.147%` on sm_80 (hand arithmetic). That fix changes `DIdealFloor` and is a
  follow-up.
* `ProofLayer.Pins` is false for today's Definitions (`PouwSegCheck_v2`) until PIN (#2417) lands: until then no proof
  layer of today's Definitions meets that hypothesis, and each statement is conditional on PIN.

The statements:

* `Sm120SheetHeadline`: sm_120 (RTX PRO 6000) at its data sheet, H1 + fx1 at SCALAR-ROUTE's ½ floors: outside an
  event of probability at most `2^-40`, every accepted window runs at least `1 − 0.599%` of its credit.
* `Sm120MeasuredHeadline`: sm_120 on its measured vector (VF3r-16), H1 + fx1 at SCALAR-ROUTE's `9/8` floors: outside
  an event of probability at most `2^-40`, every accepted window runs at least `1 − 0.591%` of its credit.
* `Sm80SheetHeadline`, FOLD-OPT only: sm_80 (A100) at its data sheet, Dither-2S at L = 16 at SCALAR-ROUTE's ½ floors:
  outside an event of probability at most `2^-40`, every accepted window runs at least `1 − 0.147%` of its credit. Words
  cost nothing at its certificate (`κ = 0`), so skipping a fold is the only saving, and `FoldOpt` assumes no run does:
  its luck and proof-layer premises do nothing.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet**, under full checking with live coins. On DS-120's machine (`Sm120.sheet`),
H1 + fx1's TS model (`Sm120.model`, `q = q_T = ½`, `N_w = 128`) and its floor program (`Sm120.flat`), at the issue
facet's certificate (`Sm120.piStar`), over `Sm120.checks = 48,000` rounds on any finite coins `Ω`: for any window's
play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling
splits, A1–A7 and FOLD-OPT hold on every run, the play's ideal game is the receipt game `G`'s, whose openings are
pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`, the
first luck premise) and whose receipt luck is at most `½` (`ReceiptLuckAt`, the second), the play is the proofs'
(`Of`), binding fails with probability at most `δ_bind` a round, every proof is sound up to `δ_ZK` (`SoundAt`), the
relations pin every committed value (`Pins`, false for today's Definitions until #2417) and E1(c) holds
(`RightFixed`), then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 599/100000` (`1 − 0.599%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: H1 + fx1 has no
filler, so it holds at any run. `ε_sample` and the Fiat–Shamir premium are separate lines. FOLD-OPT's `∀`-run form
excludes fold skippers, and charging them gives about `0.656%` (the module doc). -/
def Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ),
    (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) → (∀ x, FoldOpt Sm120.model (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 599 / 100000) ≤
      (1 / 2) ^ 40

/-- **sm_120's headline on its measured vector**, under full checking with live coins. On VF3r-16's machine
(`Sm120Measured.machine`), H1 + fx1's TS model on it (`Sm120Measured.model`: the MMA forms at their committed words an
instruction, `q = q_T = 9/8`, `b_T = 0`, `N_w = 128`) and its floor program (`Sm120Measured.flat`), at JI3s_0's
certificate (`Sm120Measured.piStar`), over `Sm120.checks = 48,000` rounds on any finite coins `Ω`: for any window's
play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling
splits, (I1) `FacetCeilings` holds on VF3r-16's rows, (I2)/(I4) `WordsWritten` at the model's `ω`, (I3) `ScalarRoute`
at its floors, (I5) `NoSum` and (I6) FOLD-OPT hold on every run, the play's ideal game is the receipt game `G`'s,
whose openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`, the first luck premise) and whose receipt luck is at most `½` (`ReceiptLuckAt`, the second), the play
is the proofs' (`Of`), binding fails with probability at most `δ_bind` a round, every proof is sound up to `δ_ZK`
(`SoundAt`), the relations pin every committed value (`Pins`, false for today's Definitions until #2417) and E1(c)
holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 591/100000` (`1 − 0.591%`) of the credit `T_cred = B_♭/f_max`. (I5) `SideUnits` and FILL-OPT are not hypotheses:
the model has no side unit and no filler, so they hold at any run. On the lane route (I3) at `9/8` is what
`ClassPSlots` gives (`1 + (32 − w)/130` slots a word, FLOORS, note:20261018T0230Z-draft-floors) once writers create at
most `w = 15` incidences, a cap nothing proves yet; on the table route it is SCALAR-ROUTE's, outside `ClassPSlots`.
`ε_sample` and the Fiat–Shamir premium are separate lines. FOLD-OPT's `∀`-run form excludes fold skippers, and
charging them shifts `ε` as on the sheet (the module doc). -/
def Sm120MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ),
    (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) → (∀ x, FoldOpt Sm120Measured.model (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

/-- **sm_80's headline at the data sheet: FOLD-OPT only.** Under full checking with live coins, on DS-80's machine
(`Sm80.sheet`), Dither-2S at L = 16's TS model (`Sm80.model`, `q = q_T = ½`, `N_w = 128`) and its floor program
(`Sm80.flat`), at the ALU pipe's certificate (`Sm80.piStar`), over `Sm80.checks = 48,000` rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, A1–A7 and FOLD-OPT hold on every run, the play's ideal game is the receipt game `G`'s, whose
openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`, the first luck premise) and whose receipt luck is at most `½` (`ReceiptLuckAt`, the second), the play
is the proofs' (`Of`), binding fails with probability at most `δ_bind` a round, every proof is sound up to `δ_ZK`
(`SoundAt`), the relations pin every committed value (`Pins`, false for today's Definitions until #2417) and E1(c)
holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 147/100000` (`1 − 0.147%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: Dither-2S has no
filler, so it holds at any run. `ε_sample` and the Fiat–Shamir premium are separate lines.

It is FOLD-OPT restated. `w* = 0` at the ALU facet, so `κ = 0` and `ε_jl = 0`: words cost nothing at the certificate,
skipping a fold is the only saving, and `FoldOpt` assumes no run does. Its luck and proof-layer premises do nothing,
and it bounds no cheater that saves anything. Charging fold skippers gives about `0.232%` (the module doc). -/
def Sm80SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm80.checks (Fin 35) (Fin 23) Unit Empty) (G : ReceiptGame Ω Sm80.checks)
    {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm80.checks nπ V) (δbind δzk : ℝ),
    (Sm80.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm80.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm80.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm80.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm80.sheet (Pl.run x).run) → (∀ x, SheetWords Sm80.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm80.model (Pl.run x)) → (∀ x, FoldOpt Sm80.model (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm80.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm80.model.credit Sm80.flat < 1 - 147 / 100000) ≤
      (1 / 2) ^ 40

end Accounting.Guarantees.TS
