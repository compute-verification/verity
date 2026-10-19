import Definitions.Accounting.HeadlineDc
import Models.Accounting.Assumptions.Sheet

/-!
# The data-centre families' headline statements (trusted; statements, unproved)

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
  bound, made exact; on DS-100 and DC-90-ri it is mixed (`IsCert` takes any), so FOLD-OPT is asked at the data file's
  `lamStar` itself, where sm_120's statements ask it at a facet (`onFacet`). On DS-90 it is DS_issue alone and on
  DC-100-ri issue alone.

FOLD-OPT is asked at each statement's own `π*` (`FoldOptFoldedAt`), the one certificate the proof prices at, as on
sm_120 (RT2-LOGIC-1). At every certificate (`FoldOptFolded`) it is false on all four (RT3-LOGIC-6): ♭'s fold with each
IMAD (its one IMAD-pipe class) replaced by 32 ALU shift-adds prices below `N_w φ*` at 1/20 issue + 19/20 IMAD on the
sheets (105.9 against 128.3 on DS-90, 846.9 against 1026.0 on DS-100) and at 1/20 I + 19/20 UP on the measured rows,
where the ALU class is free (0.68 against 6.28 on DC-90-ri; on DC-100-ri the fold prices at 0 against 50.2, its
REDG32 having no line on either row). At `π*` the same fold costs far more: 2117 against 133.0 (DS-90), 22571 against
1066.7 (DS-100), 2072 against 130.3 (DC-90-ri) and 16936 against 1064.0 (DC-100-ri); on DS-90's `π*` an IMAD costs 1
and its 32 ALU 32, on DS-100's ⅔ and 42⅔. At `π*` it is a conjecture (`shape.conjectures`), which
`benchmarks/pouw/tests/test_lean_headline_dc.py` tests on ♭'s fold and that one, and in which every fold route prices
each class it issues above 0.

Binding is taken in its fork form at `δ_fork` (`BindingForkAt`, the 8 Oct ruling's, no advantage bound) and read at
`√δ_fork` (`BindingOfFork`), so the budget asks `δ_fork ≤ (2^-41/48,000)²` a round; no ε moves (RT2-LOGIC-2). The proof
service's other pieces are named premises too: proof soundness (`ProofSoundAt`, which C-Flock's bound doesn't discharge
until it states the rows link) and the Definitions' pinning (`ProofsPin`, which no Lean checks of the served relations:
`PouwSegCheck_v2` lacks it, PIN's checks of #2417 recompute every word they absorb), beside `RightFixed` (RT3-LOGIC-3).
`ProofLayer.Of` and `ReceiptGame.Pinned` stay hypotheses over `Definitions/`. Both measured readings are at A6's proved
½ floors, so RT3-LOGIC-2's caveat on sm_120's measured `9/8` floors has no counterpart here.

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
  the census's full menu the same certificate search gives `31.3%`. ♭ and the cheater fold by the cheapest mix of
  LI-CP-W redux and every bit-exact checkpoint loop the tree carries (the design's `[fold_routes]`): CP_SEG4 (four
  segments' checkpoint words in one STG.128) and CP_SEG4U (one XOR a checkpoint on the uniform datapath; RT2-HOPPER-9,
  -11), and rt3_dc's CP_ATS, CP_ATS2 and CP_ATS8 (the checkpoint's XOR into shared memory by 1, 2 or 8 ATOMS), CP_RED,
  CP_RED2 and CP_RED8 (into L2 by 1, 2 or 8 REDG) and CP_XT (a transpose through shared memory), each bit-exact on
  the H200 and the B200 (r20261019-183610-89bd, r20261019-192247-8771; RT3-DC-1, -4). On DS-90 ♭ folds by CP_RED2
  alone; on DC-90-ri by CP_SEG4U, CP_ATS8 and CP_RED2 (0.363, 0.302 and 0.335 of the words). Honest's kernel keeps
  REDUX, so γ is unchanged. Each memory, control and warp-reduce instruction those loops issue (ATOMS32, LDS128,
  STS128, STS32, STG128, STG32, REDUX, REDG32, ISETP) is its own census class at one issue slot on the issue rows
  and nothing on a pipe row, the sound price: each issues once, and no pipe row is asserted for a class nobody timed
  on it (`FacetCeilings` reads every line), so a class's price only falls short of its cost. The H200's REDG32 also
  costs 290 SMSP-cycles on `AT90`, the L2's atomic row (RT2-HOPPER's G5). Without ffill in ♭, `setup`'s floor program
  on DC-90-ri writes every word on the IGMMA, and `Sm90Measured.flat` is that floor, its shares exact (`B_♭` 195.51 a
  unit; the machine model's reading is 1.408% at `q = ½`). It is over 1%; crediting ffill would need a check that
  reads its chains (RT-HB-HEAD-1). ε_skip prices the P·V peel at sm_90a's own instruction counts (`eps_budget`'s
  `SASS`: RT-ISA-7's static nvdisasm counts with ncu's rule applied, not ncu): 0.858% on DS-90, 0.592% on DC-90-ri.
* The B200 (sm_100) runs `dither-2s-licpw-redux-ldtm0-l32` (Dither-2S, LI-CP-W redux, no filler): its fold is the
  one `PouwWarpCheck` recomputes (`PouwLiAbsorb_v2`, no extra keyed round), and its ♭ credits none of honest's 8
  `tcgen05.ld` a MMA: the operands-free cheater never pays them, so crediting them was all of DC-100-ri's
  `ε_machine`. Honest still runs them; a lower credit is sound for any program. Its fold menu is the H200's, at the
  same sound price (the same classes but ISETP, which its loops don't issue, with VIADD and NOROW, which MOV is, each
  at one issue slot; the B200 has no atomic row, so REDG32 is an issue slot alone there): ♭ folds by CP_RED2 and
  CP_RED8 (65/72 and 7/72 of the words) on DS-100 and by CP_RED2 alone on DC-100-ri. DS-100's `f_max` (1965 MHz) is
  the driver's reported maximum: the data sheet gives no clock.

Outside every statement: `S8MmaExact`, that IGMMA s8 (sm_90a) and tcgen05 `kind::i8` (sm_100a) write `C + Σ a_k b_k`
exactly in s32 on the design's operands. The bounds read classes and words (A6), not values, so they don't read it;
completeness (honest passes) and A7's reconciliation do. H1-HB's capture of the s8 semantics decides it.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after
the commitments. The sampled path's `ε_sample` and the Fiat–Shamir premium are separate lines.

**The window these numbers are for.** Each statement is typed at `SmX.checks = 48,000` rounds, every round live and
fully checked (no service draw, so no `K_d`), and `RightFixed` charges each round an equal `1/S` share of the window's
words (`r = 1`). It says nothing about a window of any other shape, and no served window has this one. The same bound
at `S` rounds in place of 48,000 needs at least 28,693 (H200 sheet), 13,485 (B200 sheet) and 21,918 (B200 measured)
equal live rounds for `ε < 1%`; the H200's measured reading is over 1% at any `S` (`1 − Y` alone is 1.33%). Each is
above 100% at 18 rounds. Unequal rounds pay their max-over-mean share (RT-INT-2's `r`, an estimate, not a statement):
at `r = 2.87` and 48,000 rounds the B200's measured reading would be 1.06%, and the H200's sheet reading 1.10%.

**The premises the sheet numbers lean on hardest.** A2 (`SheetDispatch`) includes A2u, that a uniform-datapath
instruction takes an issue slot, which no vendor document states. It is measured on both devices: the H200 beside a
saturated wgmma, the B200 beside four `tcgen05.mma` in flight. Without it, the uniform route takes the sheet readings
to about 94% (H200, RT-HOPPER-3) and 100% (B200). FOLD-OPT at `π*` (`FoldOptFoldedAt`) has no slack as a premise: it
says no run's fold costs less than `N_w φ*` at `π*`, and `φ*` is the cheapest menu route's price there, a fold the
tree runs bit-exact, so a fold cheaper than it by any amount falsifies it and the statement says nothing. The
readings' slack against 1% is small (RT3-DC-7): with ♭ held at its fold, a cheater whose every fold route is cheaper
by 0.058% takes the H200's sheet reading to 1% (CP_RED2 at 1.0385 instructions a word against its 1.0391), 0.219%
does the B200's and 0.102% the B200's measured reading; the H200's measured reading, over 1% already, reaches its 3%
target at 2.28% (RT2-HOPPER-10: the 5.9% and 53% once quoted here scaled ♭'s fold with the cheater's). DS-90's
`π*`, DS_issue alone, prices every instruction at one issue slot, so there FOLD-OPT is an instruction count with no
pipe premise. DS-100's, DS_issue ⅔ + DS_alu ⅓, prices the pipes apart, a LOP3 at 4/3 and an IMAD or FFMA at ⅔, so
there FOLD-OPT also says li's XOR of two words has no IMAD- or FP-pipe program of 1 instruction. That is checked within a modelled menu, not proved over the B200: no program of 1
or 2 instructions on DS-100's IMAD/FP menu computes li's `x ⊕ y ⊕ K` bit-exactly for any `K` (FOLD-XOR, k = 1 in
#2778, r20261019-125446-db4e; k = 2 in #2850, all 1,456 two-instruction sequences unsat, r20261019-183441-c6b3), so
such a program costs at least 2, above the LOP3's 4/3. The menu is IMAD, IMAD.HI, IDP.4A, IDP.2A and FFMA2;
IMAD.WIDE, HFMA2 and the whole ALU class are outside it. The cheapest program known is 78 instructions, all on the
IMAD pipe (33 IMAD, 45 IMAD.HI; #2769). FOLD-OPT is false for a keyed round's XOR with a constant, which takes 1 + 2|B|
IMAD-pipe instructions (FOLD-MIN, #2751), but plain li has no such round: every half step absorbs a word. With the
whole certificate at DS_issue alone, which needs no pipe premise, the sheet readings are 0.943% (H200, its `π*`) and
0.9998% (B200). With only the proved fan-in floor in its place each reading is tens of percent (decision 115).

The statements:

* `Sm90SheetHeadline`: the H200 at its data sheet (DS-90), at π* = DS_issue: outside an event of probability at most
  `2^-40`, every accepted window of 48,000 equal live rounds runs at least `1 − 0.944%` of its credit (the unfolded
  branch binds, at 0.9430159%; the folded gives 0.8618368%).
* `Sm90MeasuredHeadline`: the H200 on DC-90-ri, at a mixed π* (issue 0.003603 + JI90_0 0.965711 + JI90_1 0.029738 +
  AT90 0.000948): at least `1 − 1.479%` (the folded branch binds, at 1.4787388%; the unfolded gives 1.4114042%), over
  1%.
* `Sm100SheetHeadline`: the B200 at its data sheet (DS-100), at a mixed π* (DS_issue ⅔ + DS_alu ⅓): at least
  `1 − 0.783%` (the unfolded branch binds, at 0.7828807%; the folded gives 0.6983541%).
* `Sm100MeasuredHeadline`: the B200 on DC-100-ri, at π* = issue: at least `1 − 0.900%` (the unfolded branch binds, at
  0.8991826%; the folded gives 0.8149027%), under 1%. RT3-DC read 1.2401% here with each new class priced as SHFL on
  every row and NOP as ALU; at the sound price above (one issue slot, no pipe row) ♭ folds by CP_RED2, which costs
  the cheater as much as ♭, and the reading is `1 − Y` at ε_skip plus the unfolded branch's `41/S` share.

Each statement's docstring gives its binding branch exactly, the exporter's `bound`.

None is proved yet: `lean-audit.json` lists them under `shape.frozen` as `unstated` until their proofs land. With
`χ* = 0` each is an instance of `headline_of_branches` as #2482 proves it.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The H200's headline at the data sheet**, under full checking with live coins. On DS-90's machine
(`Sm90.sheet`), `dither-2s-licpw-redux-p1-l32`'s TS model (`Sm90.model`: every census MMA class a writer kind at its
committed words an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 4096`) and its floor program (`Sm90.flat`, ffill
out of it at p = 1), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof
layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 hold on
every run, FOLD-OPT per word folded at `π*` (`Sm90.lamStar`, DS_issue alone: at one issue slot an instruction it
is an instruction count, no pipe premise) prices every run's fold at the
words it folds (its unfolded share `Uf`, the words whose fold it skips), the play's ideal game is the receipt game
`G`'s, whose openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`), whose receipt luck is at most `½` (`ReceiptLuckAt`, HMU's B′) and whose whole-unit luck is at most `½`
(`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), two runs off one commitment open
it differently with probability at most `δ_fork` a round (`BindingForkAt`, binding at `√δ_fork`), every proof is sound
up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value (`ProofsPin`) and E1(c) holds (`RightFixed`),
then outside an event of probability at most `2^-40` every accepted window's time is at least `1 − 944/100000`
(`1 − 0.944%`) of the credit `T_cred = B_♭/f_max`. The unfolded branch binds, at exactly
`150882547268931/16000000000000000` (0.9430159%); the folded gives 0.8618368%. FOLD-OPT's premise has no slack
(`φ*` is CP_RED2's price, a fold the tree runs bit-exact); the reading's against 1% is 0.058% (every fold route that
much cheaper). FILL-OPT is not a hypothesis: ♭ credits no filler. A3 (`SheetTensor`) is per SM clock at the clock
each tensor figure is stated at, the 1830 MHz tensor boost clock, not f_max's 1980 MHz: read per second at f_max, the
machine model put this reading at 2.253% at #2695's menu (CP_SEG4 and CP_SEG4U its only routes), since its ε first
moves at tensor lines 1.060× the sheet's. -/
def Sm90SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 37) (Fin 24) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm90.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm90.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm90.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm90.sheet (Pl.run x).run) → (∀ x, SheetWords Sm90.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm90.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw) Sm90.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm90.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm90.model.credit Sm90.flat < 1 - 944 / 100000) ≤
      (1 / 2) ^ 40

/-- **The H200's headline on its measured vector**, under full checking with live coins. On DC-90-ri's machine
(`Sm90Measured.machine`), `dither-2s-licpw-redux-p1-l32`'s TS model on it (`Sm90Measured.model`: honest's IGMMA s8
m64n256k32 and `UNI` at 2 the writer kinds, `q = q_T = ½`, `b_T = 0`, `N_w = 4096`) and its floor program
(`Sm90Measured.flat`: `setup`'s floor, every word on the IGMMA and its fold by CP_SEG4U, CP_ATS8 and CP_RED2), over
`Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play
`Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits,
(I1) `FacetCeilings` holds on DC-90-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run, FOLD-OPT per word
folded at `π*` (`Sm90Measured.lamStar`: issue 0.003603 + JI90_0 0.965711 + JI90_1 0.029738 + AT90 0.000948, rounded)
prices every run's fold at the words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in
`Sm90SheetHeadline` (`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at
most `2^-40` every accepted window's time is at least `1 − 1479/100000` (`1 − 1.479%`, over 1%: no check reads ffill's
chains) of the credit `T_cred = B_♭/f_max`. The folded branch binds, at exactly
`17398361851154717150461150531652876297946986740216852498531/1176567614710668458506374737430060647266064453125000000000000`
(1.4787388%); the unfolded gives 1.4114042%. FOLD-OPT's premise has no slack; the reading reaches its 3% target with
every fold route 2.28% cheaper.
`NoSum` carries the tensor forms off the menu: DC-90-ri prices them at the scalar coefficient on its joint rows. Two of
them would move the reading if they were on it: the register-sourced IGMMA form, whose coefficient fitted from the
H200's probes gives about 1.64% in the machine model at #2699's earlier design, and DMMA, which at the scalar
coefficient heads a word for about 0.0077 against honest's 0.0153 (at its tensor-pipe coefficient, about 0.031, it
moves nothing). -/
def Sm90MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 37) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm90Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw) Sm90Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm90Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90Measured.model.credit Sm90Measured.flat < 1 - 1479 / 100000) ≤ (1 / 2) ^ 40

/-- **The B200's headline at the data sheet**, under full checking with live coins. On DS-100's machine
(`Sm100.sheet`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model (`Sm100.model`: every census MMA class a writer
kind at its committed words an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 32768`) and its floor program
(`Sm100.flat`, crediting no `tcgen05.ld`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any
window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the
labelling splits, A1–A7 hold on every run and FOLD-OPT per word folded at `π*` (`Sm100.lamStar`, DS_issue ⅔ + DS_alu
⅓) prices every run's fold at the words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in
`Sm90SheetHeadline` (`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at
most `2^-40` every accepted window's time is at least `1 − 783/100000` (`1 − 0.783%`) of the credit
`T_cred = B_♭/f_max`. The unfolded branch binds, at exactly
`40108546277242306023361573/5123200000000000000000000000` (0.7828807%); the folded gives 0.6983541%. At this `π*` a
LOP3 costs 4/3 and an IMAD or FFMA ⅔, so FOLD-OPT says, among other things, that li's XOR of two words has no IMAD-
or FP-pipe program of 1 instruction. Within a modelled menu (IMAD, IMAD.HI, IDP.4A, IDP.2A and FFMA2; IMAD.WIDE,
HFMA2 and the ALU class outside it) that is a checked fact, and so is the next count: no program of 1 or 2
instructions computes li's `x ⊕ y ⊕ K` for any `K` (FOLD-XOR k = 1, #2778, r20261019-125446-db4e; k = 2, #2850, all
1,456 two-instruction sequences unsat, r20261019-183441-c6b3), so such a program costs at least 2 here. It is not
proved over the B200's whole instruction set. FOLD-OPT's premise has no slack (`φ*` is CP_RED2's and CP_RED8's
price); the reading's against 1% is 0.219%. FILL-OPT is not a hypothesis: the design has no filler, so it holds at
any run. -/
def Sm100SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 36) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δfork δzk : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm100.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm100.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm100.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm100.sheet (Pl.run x).run) → (∀ x, SheetWords Sm100.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm100.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw)
      Sm100.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm100.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm100.model.credit Sm100.flat < 1 - 783 / 100000) ≤
      (1 / 2) ^ 40

/-- **The B200's headline on its measured vector**, under full checking with live coins. On DC-100-ri's machine
(`Sm100Measured.machine`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model on it (`Sm100Measured.model`: every
census MMA class and `UNI` at 2 the writer kinds, `q = q_T = ½`, `b_T = 0`, `N_w = 32768`) and its floor program
(`Sm100Measured.flat`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and
proof layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, (I1)
`FacetCeilings` holds on DC-100-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run and FOLD-OPT per word
folded at `π*` (`Sm100Measured.lamStar`, issue alone) prices every run's fold at the words it folds (its unfolded share
`Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline` (`BindingForkAt` at `δ_fork`, `ProofSoundAt`,
`ProofsPin`), then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 900/100000` (`1 − 0.900%`) of the credit `T_cred = B_♭/f_max`, under 1%. The unfolded branch binds, at exactly
`71934604043397/8000000000000000` (0.8991826%); the folded gives 0.8149027%. With each new class at one issue slot
and no pipe row (the sound price: no row is asserted that nobody timed), ♭ and the cheater both fold by CP_RED2, so
`1 − Y` is ε_skip alone (0.8145%); RT3-DC's pricing, each new class as SHFL on every row, read 1.2401% here, over 1%.
FOLD-OPT's premise has no slack; the reading's against 1% is 0.102%. FILL-OPT is not a hypothesis: the design has
no filler. -/
def Sm100MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 36) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δfork δzk : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm100Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm100Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw) Sm100Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm100Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm100Measured.model.credit Sm100Measured.flat < 1 - 900 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
