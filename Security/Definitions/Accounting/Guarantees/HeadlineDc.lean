import Definitions.Accounting.HeadlineDc
import Definitions.Accounting.Assumptions.Sheet

/-!
# The data-centre families' headline guarantees (trusted)

The H200's and B200's headlines in `Definitions.Accounting.Guarantees.Headline`'s form: one statement per device and
reading, and a twin of the H200's measured one at (I3)'s `q = q_T = 3/5`, the family's credited work at least `1 − ε` of
its credit, on one device's machine, one design's TS model on it
and its floor program ♭, all exact data (`Definitions.Accounting.HeadlineDc`, written by
`benchmarks/pouw/lean_headline_dc.py`). Each `ε` is the exporter's exact bound rounded up at `10^-5`. The hypotheses are
`Sm120SheetHeadline`'s and `Sm120MeasuredHeadline`'s, with two differences, each stated where it enters:

* **The measured readings' word premises.** Both measured models are at A6's ½ floors, so A6 (`SheetWords`) stands in
  for (I2)/(I4) `WordsWritten` and (I3) `ScalarRoute`: it gives the first directly and the second through
  `SheetScalarRouteHalf` (proved), on any machine. (I1) is `FacetCeilings` on the measured rows (DC-90-ri, DC-100-ri),
  whose card (`accounting-ts-facet-ceilings`) records sm_120's rows only; (I5) is `NoSum`. The H200's twin
  (`Sm90MeasuredI3Headline`) is DC-90-ri at `q = q_T = 3/5`, where A6 gives (I2)/(I4) but not (I3), so it takes
  `WordsWritten` and (I3) `ScalarRoute` (`accounting-ts-scalar-route`, assumed) as `Sm120MeasuredHeadline` does at
  `9/8`. It stands beside the proved ½ floor, not in its place.
* **The certificate.** `lamStar` in each data file is the certificate an LP over the Lean menu picks to minimise the
  bound, made exact; on DS-100 and DC-90-ri it is mixed (`IsCert` takes any), so FOLD-OPT is asked at the data file's
  `lamStar` itself, where sm_120's statements ask it at a facet (`onFacet`). On DS-90 it is DS_issue alone and on
  DC-100-ri issue alone.

FOLD-OPT is asked at each statement's own `π*` (`FoldOptFoldedAt`), the one certificate the proof prices at, as on
sm_120. At every certificate (`FoldOptFolded`) it is false on the B200's two: ♭'s fold with each IMAD (its one IMAD-pipe
class) replaced by 32 ALU shift-adds prices below `N_w φ*` at 1/20 issue + 19/20 IMAD on DS-100 (846.9 against 1026.0)
and at 1/20 I + 19/20 UP on DC-100-ri, where the ALU class is free (0 against 50.2, its memory classes having no line on
either row). On the H200's three that fold breaks it at no certificate: the H200's menu holds each route with LI's
multiplies as IMAD.WIDE, and `N_w` times one of those prices at most that fold on every row. At `π*` the same fold costs
far more: 2117 against 133.0 (DS-90), 22571 against 1066.7 (DS-100), 2073 against 130.2 (DC-90-ri), 2384 against 130.6
(its twin) and 16936 against 1064.0 (DC-100-ri); on DS-90's `π*` an IMAD costs 1 and its 32 ALU 32, on DS-100's ⅔ and
42⅔. At `π*` it is a conjecture (`shape.conjectures`), which `benchmarks/pouw/tests/test_lean_headline_dc.py` tests on
♭'s fold and that one, and in which every fold route prices each class it issues above 0. It is open at all five `π*`.
On the H200 the fold menu is split in two. ♭ folds by the built routes alone (`Design.built_routes`): the design's fold,
the 12 bit-exact loops and li-bulk at its measured cost. The cheater's menu holds those and every counted fold known to
come near them, none built (the design's `counted`): the moves of LI on li and each loop that take its XORs through
memory or its multiplies to IMAD.WIDE, and li-bulk's counted BT1024 and trimmed W = ∞ members. So there FOLD-OPT is a
premise only about folds off the cheater's menu, and a counted fold priced below what it would cost built only lowers
the proved credit `(1 − ε)·B_♭`: it can lower `φ*` at `π*`, which raises ε, and it never enters ♭, so `B_♭` stays. At
both DC-90-ri statements' `π*` the counted folds move the bound by under `10⁻⁹`: the cheapest built route prices at `φ*`
(at ½) or within `4.6·10⁻¹⁰` of it (at 3/5). FOLD-OPT at the twin's `π*` doesn't follow from FOLD-OPT at DC-90-ri's: the
two price the classes apart, and a class free at one `π*` would make FOLD-OPT there harder to meet, not easier. The
two-slot round trip (one ATOM.E.XOR and one LD.E back, at `φ*·(1 − 3.32·10⁻¹⁰)` at the twin's earlier `π*`) is left off
the menu: li's XOR has three inputs a half step, `y ⊕ w ⊕ k`, and those two instructions carry two, so it is a count,
not a fold; with the third input it is the three-slot round trip, which is on the menu.

Binding is taken in its fork form at `δ_fork` (`BindingForkAt`, with no advantage bound) and read at `√δ_fork`
(`BindingOfFork`), so the budget asks `δ_fork ≤ (2^-41/48,000)²` a round; no ε moves. The proof system's other pieces
are named premises too: proof soundness (`ProofSoundAt`, which C-Flock's bound doesn't discharge until it states the
rows link) and the Definitions' pinning (`ProofsPin`, which no Lean checks of the served relations: `PouwSegCheck_v2`
lacks it, and the pinning checks recompute every word they absorb), beside `RightFixed`. `ProofLayer.Of` and
`ReceiptGame.Pinned` stay hypotheses over `Definitions/`. Both measured readings are at A6's proved ½ floors, so the
caveat on sm_120's measured reading, whose `9/8` floors are (I3)'s, assumed, not A6's, has no counterpart in them; the
H200's twin carries it at 3/5.

**What ♭ credits: only work a check reads.** That is the fold the served unit's Definition recomputes, and a filler only
at the theorem its check meets. `PouwWarpCheck`'s LI absorbs H1's call words with plain LI-CP-W redux, and no check
reads the H200's ffill chains. So the H200's design credits ffill at p = 1 (out of ♭; honest still runs it), and the
B200's credits plain LI-CP-W redux with no extra keyed round. No model has a filler, so `χ* = 0` in
`Y = (N_w w* + N_w φ* + χ*)/B_♭` everywhere, every statement is in the form `headline_of_branches` proves (its unfolded
branch has no `χ*`), and FILL-OPT is a hypothesis of none. A uniform writer heads 2 words (`UIMAD.WIDE.U32` writes a UR
pair), so `UNI`'s ω is 2 on both devices' menus; it moves no bound.

The sheets are `sheet.toml` as `Definitions.Accounting.Sheets.H200` and `B200` state it, over the census's classes:
every MMA form it rates (the DMMA, 2:4 sparse and fp4 forms among them) is a writer kind at its sheet price, the uniform
pipe is at the census's measured UP line (`DS_uniform`, 128 over 1.96110716 SMSP-cycles an instruction), and the
memory, atomic and control classes the fold loops issue are their issue slot alone. None of those forms moves a
bound: at each `π*` the sparse and m128n256 forms head a word at honest's price and DMMA above it, and the four
certificates and ♭'s `B` are the same with or without them. On a measured vector an MMA form the reading doesn't time
is at the measured issue slot alone and off the menu, and `NoSum` carries its words.

Per device:

* The H200 (sm_90) runs `dither-2s-licpw-redux-p1-l32` (Dither-2S, LI-CP-W redux, ffill at p = 1). On DC-90-ri the joint
  rows price every tensor form but honest's IGMMA s8 m64n256k32 at the scalar coefficient (IGMMA_S8_N256 at 62.7 on
  JI90_0, the bf16, binary and fp8 forms at about 1), a lower bound on their issue cost that no certificate prices above
  it, so the measured menu is honest's form and the uniform `UNI`, and `NoSum` carries every other form. Over the
  census's full menu the same certificate search gives `31.4%`. ♭ folds by the cheapest mix of the built routes
  (`Design.built_routes`): LI-CP-W redux, li-bulk at its measured cost, and every bit-exact checkpoint loop the tree
  carries (the design's `[fold_routes]` but its `counted`): CP_SEG4 (four segments' checkpoint words in one STG.128) and
  CP_SEG4U (one XOR a checkpoint on the uniform datapath), and CP_ATS, CP_ATS2 and CP_ATS8 (the checkpoint's XOR into
  shared memory by 1, 2 or 8 ATOMS), CP_RED, CP_RED2 and CP_RED8 (into L2 by 1, 2 or 8 REDG) and CP_XT (a transpose
  through shared memory), each bit-exact on the H200 and the B200, and CP_RED, CP_RED2 and CP_RED8 through a generic
  address (`red.relaxed.gpu.xor.b32`, which ptxas emits as ATOM.E on sm_90a and sm_100a), bit-exact on the H200 and the
  B200 too (every global and generic red, red2 and red8 loop at 0 mismatches); li-bulk is li's XOR through the stores, a
  bulk XOR with the key slab and a load back, at its measured 3.27 SMSP-cycles a word (bit-exact on the H200 and
  L2-bound), LDG.128 and the bulk XOR in `UNLISTED`. The cheater folds by those and every counted fold known to come
  near them, none built, so the bound prices them and ♭ never folds by one: on li and each of those loops the moves of
  LI (its XORs as a memory round trip of three issue slots, an ATOM.E.XOR and two LD.E back; its multiplies as
  IMAD.WIDE; or both: a longer round trip and ptxas's IMAD.WIDE with its extra ALU price at least these class by class),
  and li-bulk's BT1024 and trimmed W = ∞ members. On DS-90 ♭ folds by CP_RED2 alone; on DC-90-ri by the generic CP_RED2
  and CP_RED8 (0.6632 and 0.3368 of the words). Honest's kernel keeps REDUX, so γ is unchanged. Each memory, control and
  warp-reduce instruction those loops issue (ATOMS32, LDS128, STS128, STS32, STG128, STG32, REDUX, REDG32, ATOME32,
  ISETP) is its own census class at one issue slot and nothing on a pipe row. On DC-90-ri each also carries the JI90
  lines every timed scalar class does, since a run timed it beside the IGMMA: ATOMS32, LDS128, REDG32, REDUX and
  ATOME32, the same reduction as REDG32 through a generic address (ATOM.E), and the stores and ISETP. That is the sound
  price: each issues once, and no row is asserted for a class nobody timed on it (`FacetCeilings` reads every line), so
  a class's price only falls short of its cost. The H200's REDG32 also costs 290 SMSP-cycles on `AT90`, the L2's atomic
  row; ATOME32 has no `AT90` line, since `AT90` was timed on REDG alone. Each sheet declares A2's catch-all, `NOROW` at
  its issue slot alone: every warp instruction no other class names counts there. Each measured reading declares
  `UNLISTED`, at one slot on the measured issue row, since NOROW carries SHFL's joint issue lines there; the
  instructions no class lists count in it. On DC-90-ri it carries the JI90 lines too: IMAD.WIDE and LD.E, which count in
  it, were timed beside the IGMMA. Without ffill in ♭, `setup`'s floor program on DC-90-ri writes every word on the
  IGMMA, and `Sm90Measured.flat` is that floor, its fold's shares exact (`B_♭` 195.30 a unit, binding on `JPA90_0`; the
  machine model's reading is 1.349% at `q = ½`). It is over 1%, and so is its twin; crediting ffill would need a check
  that reads its chains. Folding by built routes alone costs both readings 0.28 points: li-bulk's trimmed W = ∞ member,
  a counted fold, had carried 0.0195 of ♭'s words and held `B_♭` at 194.75 on JI90_0 (1.133% and 0.807%). The rest of
  the gap is A6's scalar floor: at `q = ½` the lane route writes words at the IGMMA's rate on JI90_0 (65.28 against
  65.27 a cycle), so `π*` stays on JI90_0, with a little on JI90_1 and issue, off ♭'s facet: ♭ has 0.47% slack there and
  its fold is 1.0076 `φ*`. At 3/5 `π*` puts 0.15 on JPA90_0, ♭'s slack is 0.33% and its fold is 1.0042 `φ*`: the twin
  reads 1.086%, over 1% (the machine model reads 1.022% at its `q* = 1.22`, over 1% too). ε_skip prices the P·V peel at
  sm_90a's own instruction counts (`eps_budget`'s `SASS`: static nvdisasm counts with ncu's rule applied, not ncu):
  0.858% on DS-90, 0.593% on DC-90-ri. DS-90's `f_max` (1980 MHz) is EPS1 DC's measured f_max at default clocks; the
  sheet quotes its TOPS at 1830 MHz.
* The B200 (sm_100) runs `dither-2s-licpw-redux-ldtm0-l32` (Dither-2S, LI-CP-W redux, no filler): its fold is the one
  `PouwWarpCheck` recomputes (`PouwLiAbsorb_v2`, no extra keyed round), and its ♭ credits none of honest's 8
  `tcgen05.ld` a MMA: the operands-free cheater never pays them, so crediting them was all of DC-100-ri's `ε_machine`.
  Honest still runs them; a lower credit is sound for any program. Its fold menu is the H200's bit-exact loops, without
  the H200's counted folds, at the same sound price (the same classes but ISETP, which its loops don't issue, with
  VIADD, at one issue slot, and MOV in NOROW, the sheet's catch-all; the B200 has no atomic row, so REDG32 is its issue
  lines there and ATOME32 the issue row alone): ♭ folds by CP_RED2 and CP_RED8 (65/72 and 7/72 of the words) on DS-100
  and by the global CP_RED2 alone on DC-100-ri. DS-100's `f_max` (1965 MHz) is the driver's reported maximum: the data
  sheet gives no clock.

Outside every statement: `S8MmaExact`, that IGMMA s8 (sm_90a) and tcgen05 `kind::i8` (sm_100a) write `C + Σ a_k b_k`
exactly in s32 on the design's operands. The bounds read classes and words (A6), not values, so they don't read it;
completeness (honest passes) and A7's reconciliation do. A capture of each device's s8 semantics decides it.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after
the commitments. The sampled path's `ε_sample` and the Fiat–Shamir premium are separate lines.

**The window these numbers are for.** Each statement is typed at `SmX.checks = 48,000` rounds, every round live and
fully checked (no service draw, so no `K_d`), and `RightFixed` charges each round an equal `1/S` share of the window's
words (`r = 1`). It says nothing about a window of any other shape, and no served window has this one. The same bound at
`S` rounds in place of 48,000 needs at least 28,693 (H200 sheet), 13,485 (B200 sheet), 21,918 (B200 measured) and
120,064 (the H200's twin, whose `1 − Y` alone is 0.943%) equal live rounds for `ε < 1%`; no number of rounds takes the
H200's measured reading under 1%, since its `1 − Y` alone is 1.269%. Each is above 100% at 18 rounds. Unequal rounds pay
their max-over-mean share (`r`, an estimate, not a statement): at `r = 2.87` and 48,000 rounds the B200's measured
reading would be 1.06%, and the H200's sheet reading 1.10%.

**The premises the sheet numbers lean on hardest.** A2 (`SheetDispatch`) includes A2u, that a uniform-datapath
instruction takes an issue slot, which no vendor document states. It is measured on both devices: the H200 beside a
saturated wgmma, the B200 beside four `tcgen05.mma` in flight. Without it, the uniform route takes the sheet readings to
about 94% (H200) and 100% (B200). FOLD-OPT at `π*` (`FoldOptFoldedAt`) has no slack as a premise: it says no run's fold
costs less than `N_w φ*` at `π*`, and `φ*` is the price there of the cheapest route on the cheater's menu, a fold the
tree runs bit-exact (on the H200's twin a counted one, li-bulk's trimmed W = ∞ member, which no run has built), so a
fold cheaper than it by any amount falsifies it and the statement says nothing. The readings' slack against 1% is small:
with ♭ held at its fold, a cheater whose every fold route is cheaper by 0.058% takes the H200's sheet reading to 1%
(CP_RED2 at 1.0385 instructions a word against its 1.0391), 0.219% does the B200's and 0.102% the B200's measured
reading; the H200's measured reading and its twin, over 1% already, reach their 3% target at 2.38% and 2.86%. DS-90's
`π*`, DS_issue alone, prices every instruction at one issue slot, so there FOLD-OPT is an instruction count with no pipe
premise. DS-100's, DS_issue ⅔ + DS_alu ⅓, prices the pipes apart, a LOP3 at 4/3 and an IMAD or FFMA at ⅔, so there
FOLD-OPT also says li's XOR of two words has no IMAD- or FP-pipe program of 1 instruction. That is checked within a
modelled menu, not proved over the B200: no program of 1 or 2 instructions on DS-100's IMAD/FP menu computes li's
`x ⊕ y ⊕ K` bit-exactly for any `K` (an exhaustive search: k = 1, and k = 2, all 1,456 two-instruction sequences unsat),
so such a program costs at least 2, above the LOP3's 4/3. The menu is IMAD, IMAD.HI, IDP.4A, IDP.2A and FFMA2;
IMAD.WIDE, HFMA2 and the whole ALU class are outside it. The cheapest program known is 78 instructions, all on the IMAD
pipe (33 IMAD, 45 IMAD.HI). FOLD-OPT is false for a keyed round's XOR with a constant, which takes 1 + 2|B| IMAD-pipe
instructions (`benchmarks/pouw/theory/fold_min.py`), but plain li has no such round: every half step absorbs a word.
With the whole certificate at DS_issue alone, which needs no pipe premise, the sheet readings are 0.943% (H200, its
`π*`) and 0.9998% (B200). With only the proved fan-in floor in its place each reading is tens of percent.

The statements:

* `Sm90SheetHeadline`: the H200 at its data sheet (DS-90), at π* = DS_issue: outside an event of probability at most
  `2^-40`, every accepted window of 48,000 equal live rounds runs at least `1 − 0.944%` of its credit (the unfolded
  branch binds, at 0.9430159%; the folded gives 0.8618368%).
* `Sm90MeasuredHeadline`: the H200 on DC-90-ri, at a mixed π* (issue 0.003598 + JI90_0 0.966699 + JI90_1 0.029703):
  at least `1 − 1.412%` (the folded branch binds, at 1.4110908% with `ln(8/7) ≥ 2/15`; the unfolded gives 1.3529949%),
  over 1%.
* `Sm90MeasuredI3Headline`: its twin under (I3) `ScalarRoute` at `q = q_T = 3/5` (assumed), at π* = JI90_0 0.819832 +
  JI90_1 0.033778 + JPA90_0 0.146390: at least `1 − 1.086%` (the folded branch binds, at 1.0858206%; the unfolded gives
  1.0274482%), over 1% too. Cite it with its premise: DC-90-ri 1.086% under (I3) at 3/5; 1.412% at the proved ½ floor;
  credit against the built ♭, the cheater priced at every known fold, counted ones included.
* `Sm100SheetHeadline`: the B200 at its data sheet (DS-100), at a mixed π* (DS_issue ⅔ + DS_alu ⅓): at least
  `1 − 0.783%` (the unfolded branch binds, at 0.7828807%; the folded gives 0.6983541%).
* `Sm100MeasuredHeadline`: the B200 on DC-100-ri, at π* = issue: at least `1 − 0.900%` (the unfolded branch binds, at
  0.8991826%; the folded gives 0.8149027%), under 1%. With each new class priced as SHFL on every row and NOP as ALU
  it reads 1.2401%; at the sound price above (one issue slot, no pipe row) ♭ folds by the global CP_RED2,
  which costs
  the cheater as much as ♭, and the reading is `1 − Y` at ε_skip plus the unfolded branch's `41/S` share.

Each statement's docstring gives its binding branch exactly, the exporter's `bound`.

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
machine model put this reading at 2.253% on a menu whose only routes were CP_SEG4 and CP_SEG4U, since its ε first
moves at tensor lines 1.060× the sheet's. -/
def Sm90SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 34) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
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
(`Sm90Measured.flat`: `setup`'s floor, every word on the IGMMA, and its fold by the generic CP_RED2 and CP_RED8, both
built), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer `pf`
whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, (I1) `FacetCeilings` holds on
DC-90-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run, FOLD-OPT per word folded at `π*`
(`Sm90Measured.lamStar`: issue 0.003598 + JI90_0 0.966699 + JI90_1 0.029703, rounded) prices every run's fold at the
words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline`
(`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every
accepted window's time is at least `1 − 1412/100000` (`1 − 1.412%`, over 1%: no check reads ffill's chains, and ♭ folds
by built routes alone) of the credit `T_cred = B_♭/f_max`. The folded branch binds, at exactly
`2495272303972424451357093036809987368800019/176832869229494800775000000000000000000000000` (1.4110908%), reading
`ln(8/7) ≥ 2/15` as `Sm120MeasuredHeadline` does; the unfolded gives 1.3529949%. The cheater's menu holds every counted
fold known to come near the bit-exact loops, none built, and ♭'s holds none of them: the moves of LI on li and each loop
(its XORs as an ATOM.E.XOR and two LD.E back, its multiplies as IMAD.WIDE, whose low word does li's multiply, or both),
whose classes carry the JI90 lines the H200 measured beside the IGMMA (ATOME32, and `UNLISTED`, where IMAD.WIDE and LD.E
count, each 0.9788 at `π*`, as an IMAD and a LOP3 are), and li-bulk at its measured cost, built, with its counted BT1024
and trimmed W = ∞ members. So FOLD-OPT at `π*` is a premise only about folds off the cheater's menu; the two-slot round
trip is off it, since li's XOR has three inputs a half step and one ATOM.E.XOR and one LD.E carry two (at this `π*` it
prices at 1.50 `φ*`). FOLD-OPT at `π*` is open: `φ*` is CP_RED's and CP_RED2's price (their generic and IMAD.WIDE
members tie), and every class a fold route issues prices above 0 there, the stores and ISETP at their measured JI90
lines. FOLD-OPT's premise has no slack; the reading reaches its 3% target with every fold route 2.38% cheaper. Its twin
at (I3)'s `q = q_T = 3/5` is `Sm90MeasuredI3Headline`. `NoSum` carries the tensor forms off the menu: DC-90-ri prices
them at the scalar coefficient on its joint rows. Three kinds would move the reading if they were on it: the
register-sourced IGMMA form, whose coefficient fitted from the H200's probes gives about 1.64% in the machine model at
an earlier design; DMMA, which at the scalar coefficient heads a word for about 0.0077 against honest's 0.0153 (at its
tensor-pipe coefficient, about 0.031, it moves nothing); and the 2:4-sparse forms, whose classes carry the issue row
alone (no run timed them beside the IGMMA), so that chained into a dense product they head a word for under `2·10⁻⁶` at
this `π*` and for 0 at the twin's. -/
def Sm90MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
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
      (Pl.run x).run.time / Sm90Measured.model.credit Sm90Measured.flat < 1 - 1412 / 100000) ≤ (1 / 2) ^ 40

/-- **The H200's headline on its measured vector under (I3) at 3/5**, the twin of `Sm90MeasuredHeadline`, under full
checking with live coins. On DC-90-ri's machine (`Sm90Measured.machine`), the same TS model at (I3)'s floors
(`Sm90MeasuredI3.model`: `Sm90Measured.model`'s writer kinds, vector classes and fold routes, `q = q_T = 3/5`,
`b_T = 0`, `N_w = 4096`) and the same floor program (`Sm90Measured.flat`), over `Sm90.checks = 48,000` rounds on any
finite coins `Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over
the window, if the labelling splits, (I1) `FacetCeilings` holds on DC-90-ri's rows, (I2)/(I4) `WordsWritten`, (I3)
`ScalarRoute` and (I5) `NoSum` hold on every run, FOLD-OPT per word folded at `π*` (`Sm90MeasuredI3.lamStar`: JI90_0
0.819832 + JI90_1 0.033778 + JPA90_0 0.146390, rounded) prices every run's fold at the words it folds (its unfolded
share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline` (`BindingForkAt` at `δ_fork`,
`ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every accepted window's time is at
least `1 − 1086/100000` (`1 − 1.086%`, over 1%) of the credit `T_cred = B_♭/f_max`. The folded branch binds, at exactly
`5625257050663081207487178157051972415029440436967/518065046570785549145507812500000000000000000000000`
(1.0858206%), reading `ln(8/7) ≥ 2/15`; the unfolded gives 1.0274482%.

So the 1.086% rests on (I3) at 3/5 (`accounting-ts-scalar-route`), an assumed premise, where `Sm90MeasuredHeadline`'s
1.412% rests on A6 alone, and both are over 1%. Nothing proved gives 3/5 here: A6 gives ½, since it lets a 64-bit head
(IMAD.WIDE, DFMA) write two words a lane, and `ClassPSlots`, which counts at least one slot a word, takes each output
writer's slot to write nothing else, the case those heads break. 3/5 is the least floor of denominator at most 5 at
which the bound is at its least: the same 1.086% at every `q` from about 0.5884 up, and under 1% at no `q` (sm_120's
measured headline reads `9/8`, the machine model `1.22`). At 3/5 the lane route no longer writes words at the IGMMA's
rate on JI90_0, so `π*` puts 0.15 on JPA90_0; `φ*` there is the price of li-bulk's trimmed W = ∞ member, a counted fold,
not built, with the built CP_RED2 `4.6·10⁻¹⁰` of it above, and ♭'s fold, by built routes alone, is 1.0042 `φ*`; `w*` is
`UNI`'s word, under honest's IGMMA's by `6.6·10⁻⁹` of it. `B_♭`, ♭ and the cheater's menu are `Sm90MeasuredHeadline`'s.
FOLD-OPT at this `π*` is open and doesn't follow from FOLD-OPT at DC-90-ri's: its JPA90_0 weight prices ALU and the
IGMMA alone, so it prices the classes apart from DC-90-ri's `π*`. Every class a fold route issues prices above 0 here (a
free class would make FOLD-OPT harder to meet, not easier). The two-slot round trip on CP_RED8, at `φ*·(1 − 3.32·10⁻¹⁰)`
at this statement's earlier `π*`, is off the menu, since li's XOR has three inputs a half step and one ATOM.E.XOR and
one LD.E carry two; at this `π*` it prices at 1.28 `φ*`, and the three-slot one, on the menu, at 1.69 `φ*`. FOLD-OPT's
premise has no slack; the reading, over 1% already, reaches its 3% target with every fold route 2.86% cheaper. Cite it
with its premise: DC-90-ri 1.086% under (I3) at 3/5; 1.412% at the proved ½ floor; credit against the built ♭, the
cheater priced at every known fold, counted ones included. -/
def Sm90MeasuredI3Headline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm90MeasuredI3.model (Pl.run x)) → (∀ x, ScalarRoute Sm90MeasuredI3.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90MeasuredI3.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90MeasuredI3.model.Nw) Sm90MeasuredI3.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm90MeasuredI3.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90MeasuredI3.model.credit Sm90Measured.flat < 1 - 1086 / 100000) ≤ (1 / 2) ^ 40

/-- **The B200's headline at the data sheet**, under full checking with live coins. On DS-100's machine (`Sm100.sheet`),
`dither-2s-licpw-redux-ldtm0-l32`'s TS model (`Sm100.model`: every census MMA class a writer kind at its committed words
an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 32768`) and its floor program (`Sm100.flat`, crediting no
`tcgen05.ld`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer
`pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 hold on every run
and FOLD-OPT per word folded at `π*` (`Sm100.lamStar`, DS_issue ⅔ + DS_alu ⅓) prices every run's fold at the words it
folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline` (`BindingForkAt` at
`δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every accepted window's
time is at least `1 − 783/100000` (`1 − 0.783%`) of the credit `T_cred = B_♭/f_max`. The unfolded branch binds, at
exactly `40108546277242306023361573/5123200000000000000000000000` (0.7828807%); the folded gives 0.6983541%. At this
`π*` a LOP3 costs 4/3 and an IMAD or FFMA ⅔, so FOLD-OPT says, among other things, that li's XOR of two words has no
IMAD- or FP-pipe program of 1 instruction. Within a modelled menu (IMAD, IMAD.HI, IDP.4A, IDP.2A and FFMA2; IMAD.WIDE,
HFMA2 and the ALU class outside it) that is a checked fact, and so is the next count: no program of 1 or 2 instructions
computes li's `x ⊕ y ⊕ K` for any `K` (an exhaustive search: k = 1, and k = 2, all 1,456 two-instruction sequences
unsat), so such a program costs at least 2 here. It is not proved over the B200's whole instruction set. FOLD-OPT's
premise has no slack (`φ*` is CP_RED2's and CP_RED8's price); the reading's against 1% is 0.219%. FILL-OPT is not a
hypothesis: the design has no filler, so it holds at any run. -/
def Sm100SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 27) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
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
(`Sm100Measured.machine`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model on it (`Sm100Measured.model`: every MMA class
DC-100-ri times and `UNI` at 2 the writer kinds, the forms it doesn't time off the menu, `q = q_T = ½`, `b_T = 0`,
`N_w = 32768`) and its floor program (`Sm100Measured.flat`), over `Sm100.checks = 48,000` rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, (I1) `FacetCeilings` holds on DC-100-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on
every run and FOLD-OPT per word folded at `π*` (`Sm100Measured.lamStar`, issue alone) prices every run's fold at the
words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline`
(`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every
accepted window's time is at least `1 − 900/100000` (`1 − 0.900%`) of the credit `T_cred = B_♭/f_max`, under 1%. The
unfolded branch binds, at exactly `71934604043397/8000000000000000` (0.8991826%); the folded gives 0.8149027%. With each
new class at one issue slot and no pipe row (the sound price: no row is asserted that nobody timed), ♭ folds by the
global CP_RED2, at `π*` the price of CP_RED2, the cheater's cheapest, so `1 − Y` is ε_skip alone (0.8145%); priced with
each new class as SHFL on every row, it reads 1.2401% here, over 1%. FOLD-OPT's premise has no slack; the reading's
against 1% is 0.102%. FILL-OPT is not a hypothesis: the design has no filler. -/
def Sm100MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
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
