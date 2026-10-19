import Definitions.Accounting.HeadlineDc
import Models.Accounting.Assumptions.Sheet

/-!
# The data-centre families' headline guarantees (trusted)

The H200's and B200's headlines in `Models.Accounting.Guarantees.Headline`'s form: one statement per device and
reading, the family's credited work at least `1 − ε` of its credit, on one device's machine, one design's TS model on it
and its floor program ♭, all exact data (`Definitions.Accounting.HeadlineDc`, written by
`benchmarks/pouw/lean_headline_dc.py`). Each `ε` is the exporter's exact bound rounded up at `10^-5`. The hypotheses are
`Sm120SheetHeadline`'s and `Sm120MeasuredHeadline`'s, with two differences, each stated where it enters:

* **The measured readings' word premises.** Both measured models are at A6's ½ floors, so A6 (`SheetWords`) stands in
  for (I2)/(I4) `WordsWritten` and (I3) `ScalarRoute`: it gives the first directly and the second through
  `SheetScalarRouteHalf` (proved), on any machine. (I1) is `FacetCeilings` on the measured rows (DC-90-ri, DC-100-ri),
  whose card (`accounting-ts-facet-ceilings`) records sm_120's rows only; (I5) is `NoSum`.
* **The certificate.** `lamStar` in each data file is the certificate an LP over the Lean menu picks to minimise the
  bound, made exact; on DC-90-ri and DS-100 it is mixed (`IsCert` takes any). The statements don't name it: the proof
  does.

**What ♭ credits: only work a check reads** (RT-LOGIC-12, decision 99). That is the fold the served unit's Definition
recomputes, and a filler only at the theorem its check meets. `PouwWarpCheck`'s LI absorbs H1's call words with plain
LI-CP-W redux, and no check reads the H200's ffill chains. So the H200's design credits ffill at p = 1 (out of ♭;
honest still runs it), and the B200's credits plain LI-CP-W redux with no extra keyed round. No model has a filler, so
`χ* = 0` in `Y = (N_w w* + N_w φ* + χ*)/B_♭` everywhere, every statement is in the form `headline_of_branches` proves
(its unfolded branch has no `χ*`), and FILL-OPT is a hypothesis of none. A uniform writer heads 2 words
(`UIMAD.WIDE.U32` writes a UR pair, RT-HOPPER-7), so `UNI`'s ω is 2 on both devices' menus; it moves no bound.

Per device:

* The H200 (sm_90) runs `dither-2s-licpw-redux-p1-l32` (Dither-2S, LI-CP-W redux, ffill at p = 1). On DC-90-ri the
  joint rows price every tensor form but honest's IGMMA s8 m64n256k32 at the scalar coefficient (IGMMA_S8_N256 at 62.7
  on JI90_0, the bf16, binary and fp8 forms at about 1), a lower bound on their issue cost that no certificate prices
  above it, so the measured menu is honest's form and the uniform `UNI`, and `NoSum` carries every other form. Over
  the census's full menu the same certificate search gives `31.8%`. Without ffill in ♭, `setup`'s floor program on
  DC-90-ri heads 0.3% of its words on honest's lane baseline beside the IGMMA, so `Sm90Measured.flat` is honest's
  program, the IGMMA heading every word: its `B_♭` is larger (199.31 a unit against the floor's 199.11), so the bound
  is at least the machine model's (2.260% at `q = ½`). It is over 1%; crediting ffill would need a check that reads
  its chains (RT-HB-HEAD-1). ε_skip prices the P·V peel at sm_90a's own instruction counts (`eps_budget`'s `SASS`:
  RT-ISA-7's static nvdisasm counts with ncu's rule applied, not ncu): 0.846% on DS-90, 0.581% on DC-90-ri.
* The B200 (sm_100) runs `dither-2s-licpw-redux-ldtm0-l32` (Dither-2S, LI-CP-W redux, no filler): its fold is the
  one `PouwWarpCheck` recomputes (`PouwLiAbsorb_v2`, no extra keyed round), and its ♭ credits none of honest's 8
  `tcgen05.ld` a MMA: the operands-free cheater never pays them, so crediting them was all of DC-100-ri's
  `ε_machine`. Honest still runs them; a lower credit is sound for any program. DS-100's `f_max` (1965 MHz) is the
  driver's reported maximum: the data sheet gives no clock.

Outside every statement: `S8MmaExact`, that IGMMA s8 (sm_90a) and tcgen05 `kind::i8` (sm_100a) write `C + Σ a_k b_k`
exactly in s32 on the design's operands. The bounds read classes and words (A6), not values, so they don't read it;
completeness (honest passes) and A7's reconciliation do. H1-HB's capture of the s8 semantics decides it.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after
the commitments. The sampled path's `ε_sample` and the Fiat–Shamir premium are separate lines.

**The window these numbers are for.** Each statement is typed at `SmX.checks = 48,000` rounds, every round live and
fully checked (no service draw, so no `K_d`), and `RightFixed` charges each round an equal `1/S` share of the window's
words (`r = 1`). It says nothing about a window of any other shape, and no served window has this one. The same bound
at `S` rounds in place of 48,000 needs at least 26,367 (H200 sheet), 13,801 (B200 sheet) and 20,594 (B200 measured)
equal live rounds for `ε < 1%`; the H200's measured reading is over 1% at any `S` (`1 − Y` alone is 2.27%). Each is
above 100% at 18 rounds. Unequal rounds pay their max-over-mean share (RT-INT-2's `r`, an estimate, not a statement):
at `r = 2.87` and 48,000 rounds the B200's measured reading would be 1.05%.

**The premises the sheet numbers lean on hardest.** A2 (`SheetDispatch`) includes A2u, that a uniform-datapath
instruction takes an issue slot, which no vendor document states. It is measured on both devices: the H200 beside a
saturated wgmma, the B200 beside four `tcgen05.mma` in flight. Without it, the uniform route takes the sheet readings
to about 94% (H200, RT-HOPPER-3) and 100% (B200). FOLD-OPT (`FoldOptFolded`, conjectured) has 5.9% of slack on DS-90:
the H200's sheet reading holds while LI-CP-W redux's fold costs at least 0.992 lane instructions a word against its
1.055 (with ffill out of ♭, ♭ on DS_issue is the MMA and the fold alone). On DS-100 it has 53% on count: the cliff is
0.496 a word (RT-HB-HEAD-2). This statement's π* on DS-100 also prices the pipes apart, a LOP3 at 1.969 and an IMAD or
FFMA at 0.031, so there FOLD-OPT also says li's XOR of two words has no IMAD- or FP-pipe program of fewer than 63
instructions (the bit-serial one has 193; at the machine model's π*, a LOP3 at 1.979 and an IMAD at 0.021, it is 93,
as `headline.toml` and the design file state it). It is false for a keyed round's XOR with a constant, which takes
1 + 2|B| IMAD-pipe instructions (FOLD-MIN, #2751), but plain li has no such round: every half step absorbs a word. With
the fold priced at its instruction count alone, which needs no pipe premise, the B200's sheet reading is 1.408% here
(1.328% in the machine model). With only the proved fan-in floor in its place each reading is tens of percent
(decision 115).

The statements:

* `Sm90SheetHeadline`: the H200 at its data sheet (DS-90), at π* = DS_issue: outside an event of probability at most
  `2^-40`, every accepted window of 48,000 equal live rounds runs at least `1 − 0.931%` of its credit (the unfolded
  branch binds).
* `Sm90MeasuredHeadline`: the H200 on DC-90-ri, at a mixed π* (issue, JI90_0, JI90_1): at least `1 − 2.419%` (the
  folded branch binds), over 1%, and an upper reading: ♭ is honest's program, not the floor.
* `Sm100SheetHeadline`: the B200 at its data sheet (DS-100), at a mixed π* (DS_issue, DS_alu): at least `1 − 0.790%`
  (unfolded).
* `Sm100MeasuredHeadline`: the B200 on DC-100-ri, at π* = issue: at least `1 − 0.888%` (unfolded).

Each is proved as `Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.HeadlineDc`: with `χ* = 0` it is an instance
of `headline_of_branches` at the data's π*, each branch's floor bounded from below by lower bounds on `w*` and `φ*`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The H200's headline at the data sheet**, under full checking with live coins. On DS-90's machine
(`Sm90.sheet`), `dither-2s-licpw-redux-p1-l32`'s TS model (`Sm90.model`: every census MMA class a writer kind at its
committed words an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 4096`) and its floor program (`Sm90.flat`, ffill
out of it at p = 1), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof
layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 hold on
every run, FOLD-OPT per word folded prices every run's fold at the words it folds (its unfolded share `Uf`, the words
whose fold it skips), the play's ideal game is the receipt game `G`'s, whose openings are pinned (`Pinned`), whose
deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`), whose receipt luck is at most `½`
(`ReceiptLuckAt`, HMU's B′) and whose whole-unit luck is at most `½` (`WholeLuckAt`, B′ at a unit whose fold is
skipped), the play is the proofs' (`Of`), binding fails with probability at most `δ_bind` a round, every proof is sound
up to `δ_ZK` (`SoundAt`), the relations pin every committed value (`Pins`) and E1(c) holds (`RightFixed`), then outside
an event of probability at most `2^-40` every accepted window's time is at least `1 − 931/100000` (`1 − 0.931%`) of the
credit `T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: ♭ credits no filler. A3 (`SheetTensor`) is per SM clock at
the clock each tensor figure is stated at, the 1830 MHz tensor boost clock, not f_max's 1980 MHz: read per second at
f_max, the machine model puts this reading at 1.942%, since its ε first moves at tensor lines 1.063× the sheet's. -/
def Sm90SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 28) (Fin 24) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δbind δzk : ℝ),
    (Sm90.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm90.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm90.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm90.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm90.sheet (Pl.run x).run) → (∀ x, SheetWords Sm90.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm90.model (Pl.run x)) →
    (∀ x, FoldOptFolded Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm90.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm90.model.credit Sm90.flat < 1 - 931 / 100000) ≤
      (1 / 2) ^ 40

/-- **The H200's headline on its measured vector**, under full checking with live coins. On DC-90-ri's machine
(`Sm90Measured.machine`), `dither-2s-licpw-redux-p1-l32`'s TS model on it (`Sm90Measured.model`: honest's IGMMA s8
m64n256k32 and `UNI` at 2 the writer kinds, `q = q_T = ½`, `b_T = 0`, `N_w = 4096`) and honest's program as its ♭
(`Sm90Measured.flat`: `setup`'s floor heads 0.3% of its words on honest's lane baseline, so this ♭ is larger and the
bound at least the machine model's), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play
`Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits,
(I1) `FacetCeilings` holds on DC-90-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run, FOLD-OPT per word
folded prices every run's fold at the words it folds (its unfolded share `Uf`), and the luck and proof premises hold
as in `Sm90SheetHeadline`, then outside an event of probability at most `2^-40` every accepted window's time is at
least `1 − 2419/100000` (`1 − 2.419%`, over 1%: no check reads ffill's chains) of the credit `T_cred = B_♭/f_max`.
`NoSum` carries the tensor forms off the menu: DC-90-ri prices them at the scalar coefficient on its joint rows. Two of
them would move the reading if they were on it: the register-sourced IGMMA form, whose coefficient fitted from the
H200's probes gives about 1.64% in the machine model at #2699's earlier design, and DMMA, which at the scalar
coefficient heads a word for about 0.0077 against honest's 0.0153 (at its tensor-pipe coefficient, about 0.031, it
moves nothing). -/
def Sm90MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δbind δzk : ℝ),
    (Sm90.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm90Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm90Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90Measured.model.credit Sm90Measured.flat < 1 - 2419 / 100000) ≤ (1 / 2) ^ 40

/-- **The B200's headline at the data sheet**, under full checking with live coins. On DS-100's machine
(`Sm100.sheet`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model (`Sm100.model`: every census MMA class a writer
kind at its committed words an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 32768`) and its floor program
(`Sm100.flat`, crediting no `tcgen05.ld`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any
window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the
labelling splits, A1–A7 hold on every run and FOLD-OPT per word folded prices every run's fold at the words it folds
(its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline`, then outside an event of
probability at most `2^-40` every accepted window's time is at least `1 − 790/100000` (`1 − 0.790%`) of the credit
`T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: the design has no filler, so it holds at any run. -/
def Sm100SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 26) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δbind δzk : ℝ),
    (Sm100.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm100.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm100.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm100.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm100.sheet (Pl.run x).run) → (∀ x, SheetWords Sm100.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm100.model (Pl.run x)) →
    (∀ x, FoldOptFolded Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm100.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm100.model.credit Sm100.flat < 1 - 790 / 100000) ≤
      (1 / 2) ^ 40

/-- **The B200's headline on its measured vector**, under full checking with live coins. On DC-100-ri's machine
(`Sm100Measured.machine`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model on it (`Sm100Measured.model`: every
census MMA class and `UNI` at 2 the writer kinds, `q = q_T = ½`, `b_T = 0`, `N_w = 32768`) and its floor program
(`Sm100Measured.flat`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and
proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, (I1)
`FacetCeilings` holds on DC-100-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run and FOLD-OPT per word
folded prices every run's fold at the words it folds (its unfolded share `Uf`), and the luck and proof premises hold as
in `Sm90SheetHeadline`, then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 888/100000` (`1 − 0.888%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: the design has no
filler. -/
def Sm100MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 26) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δbind δzk : ℝ),
    (Sm100.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm100Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm100Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFolded Sm100Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm100Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm100Measured.model.credit Sm100Measured.flat < 1 - 888 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
