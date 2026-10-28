import Definitions.Accounting.Headline
import Definitions.Accounting.Assumptions.Sheet

/-!
# The headline guarantees (trusted)

One statement per family and reading: the family's credited work is at least `1 − ε` of its credit, `ε` the reading's
headline, on one device's machine, one design's TS model on it and its floor program ♭, all exact data
(`Definitions.Accounting.Headline`). Each is `Definitions.Accounting.Guarantees.Sheet`'s `SheetDIdealFloor` at that data
with fold skippers charged (below), its parameters fixed (`π*`, `p`, `p₀`, `S`, `δ`), with every hypothesis that the
data discharge gone and the rest left as they are: the named assumptions of `Definitions.Accounting.Assumptions` and the
proof layer's pieces. A measured reading is `Definitions.Accounting.Guarantees.TS`'s `DIdealFloor` at its data the same
way, its machine the measured vector's rows. Each is proved as `Accounting.SecurityProofs.TS.G` in
`Proofs.Accounting.Headline`.

Each bound is per node: an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at least `1 − ε`
of the credited work, with the device that did it left to the attribution layer. The statements bound the run's time
against `T_cred`, the credited work priced on one device's machine, and nothing ties a committed tile to the GPU that
computed it: the transcript commits tile digests, so a helper can supply whole tiles. A per-GPU reading needs LOC's host
deduction (what the host CPU can compute of the credited work is deducted) and the peer-GPU premise, every billable GPU
on the node credited each window or the node as the unit (`benchmarks/pouw/machine_model/README.md`, "The headline is
per node"). Counted per GPU without them, the host's AMX and AVX-512 (and NVLink peers) take every device over 1% on
some real host.

Each covers full checking with live coins: `𝒫_full`, every round's units proved, against uniform coins drawn after the
commitments. The sampled path's `ε_sample` (`DrawnLayer`) and the Fiat–Shamir premium are separate lines, in none of
these statements.

The luck (I7) is split at the receipt (LEDGER §1.3) into two named luck premises, and `LuckOfReceipt` gives (I7) at
`¾ + ¼·½ = 7/8` from them, the receipt game's openings pinned:

* `OperandHMu` at `¾`, the first: (H-μ) at the deficient words, which Lemma B at fibre 1 turns into the operand luck at
  `¾`;
* `ReceiptLuckAt` at `½`, the second: HMU's B′ case, LEDGER's case (b), named and conjectured.

Case (a), which would make (I7) `¾`, restricts the cheater and is no premise. Neither is `OperandLuckCube`'s hypothesis
(degree-2 operand errors on cube bits), which only affine-operand runs meet: it is a proved route to the operand luck,
not a premise here.

FOLD-OPT is asked at the statement's own `π*` (`FoldOptFoldedAt`), the one certificate the proof prices at.
At every certificate (`FoldOptFolded`) it is false for a
bit-exact fold: the IMAD rotate costs 20 issue slots a unit against the menu's 16, and 14 against 16 halfway between
issue and the LOP pipe. At `π*` it is a conjecture (`shape.conjectures`), checked within a stated menu by
`benchmarks/pouw/tests/test_lean_headline.py`, at both headlines' `π*`:

* fx1's lane fold and its uniform route, the Lean menu itself (`Sm120.h1fx1.folds`);
* the bit-exact lane folds `FOLDS`: the IMAD, funnel and shift rotates and the IMAD.WIDE forms;
* the bulk-reduction floors, the exporter's `BULK_FLOORS` (sheet only): fx1's step `u = y ⊕ d′ ⊕ k` as two bulk
  reductions a word, `d′` moved by STS.128 or STG.256 and the reductions into global or shared memory;
* the twelve whole-block folds on the bulk engine (`fold_bulk.cu`, `fixtures/fold_bulk_loops.json`, sheet
  only), fbw4096_K4_BT256 among them.

A fold outside that menu isn't checked: FOLD-OPT says nothing is cheaper, and the menu is what has been tried.

Fold skippers are charged. FOLD-OPT is per word folded:
each run's fold is priced at the words it folds, `(1 − Σ_w e_w/S)·N_w`, where `Uf : Unfolded` gives each round's
share `e_w` of words whose fold the run skips. A round with a positive unfolded share passes only by keeping the honest
committed value without the fold, which the whole-unit luck bounds: B′ at a whole unit (`WholeLuckAt` at `p₀ = ½`,
live coins only). The whole-unit guess bound `2^-64` isn't granted, so `p₀` is B′'s `½`. The luck term is
the larger of two branches':

* the folded branch's `ε_jl` at `p = 7/8`;
* the unfolded branch's `(N_w w* + N_w φ*)/B_♭ · ln(1/δ′)/(ln 2 · S)`.

The `2^-40` event is split in two: binding and soundness cost at most `2^-41` over the window, and the luck's tail is
`δ′ = 2^-41`, so both branches' logarithm is `ln(1/δ′) = 41 ln 2`. Compose puts its logarithm at its
`δ_s = 2^-40` instead; at this convention its 0.744% is 0.748%.
Binding is taken in its fork form at `δ_fork` (`BindingForkAt`, with no advantage bound) and read at `√δ_fork`
(`BindingOfFork`), so the budget asks `δ_fork ≤ (2^-41/48,000)²` a round; no ε moves.

The unfolded branch binds on sm_120's sheet (`5.860%`); on sm_120's measured vector the folded branch binds (`0.591%`).
The proof service's pieces are named premises: binding's fork form (`BindingForkAt`), proof soundness (`ProofSoundAt`,
which C-Flock's bound doesn't discharge until it states the rows link) and the Definitions' pinning (`ProofsPin`, which
no Lean checks of the served relations: `PouwSegCheck_v2` lacks it, PIN's checks recompute every word they absorb),
beside `RightFixed`. `ProofLayer.Of` and `ReceiptGame.Pinned` stay hypotheses over `Definitions/`.

The statements:

* `Sm120SheetHeadline`: sm_120 (RTX PRO 6000) at its data sheet, H1 + fx1 at SCALAR-ROUTE's ½ floors: under its
  premises, A2u and A5b among them (below), outside an event of probability at most `2^-40` every accepted window of
  48,000 equal live rounds runs at least `1 − 5.860%` of its credit.
* `Sm120MeasuredHeadline`: sm_120 on its measured vector (VF3s-16), H1 + fx1 at SCALAR-ROUTE's `9/8` floors: under its
  premises, outside an event of probability at most `2^-40` every accepted window of 48,000 equal live rounds runs at
  least `1 − 0.591%` of its credit. The `9/8` floors are assumed, not proved: at A6's proved `½` floor the same bound
  is 7.791% (below).

The window is in the statements' type: `Sm120.checks = 48,000` equal live rounds, one check a round, so neither says
anything about a window of another shape. The ε needs that many rounds. The same two branches with `S` rounds in place
of 48,000 give ε < 1% only from 8,320 rounds on the measured vector, and on the sheet never: there ε falls to 5.779%,
`1 − N_w(w* + φ*)/B_♭`, and is under 6% only from 17,460 rounds. Both exceed 100% up to 40 rounds (sheet) and 41
(measured); these counts are `benchmarks/pouw/lean_headline.py`'s `branches` at `S`, not Lean statements.
Compose's live line has the same kind of floor: ε < 1% there needs
at least 10,552 equal live rounds a window, whatever the draw. The windows run so far have at most 18 rounds.

The window's shape asks four things of the protocol, which the statements carry in their types and the verifier must
enforce on every window:

* **An all-or-nothing verdict.** A window is accepted only if every proof of every round is (`ProofLayer.Of`), so one
  caught round voids the whole window's credit; the luck term `ln(1/δ′)/S` prices exactly that. Credited round by
  round, the same per-round luck lets a prover who skips every round keep about 7/8 of the credit. With no retries, an
  honest per-round failure rate `q` keeps `(1 − q)^48,000` of honest windows (62% at `q = 10⁻⁵`).
* **Equal rounds.** `RightFixed` charges each round `1/S` of the window's words: the equal shares are the protocol's.
* **Fresh coins.** Each round's coin is uniform and independent of the earlier ones (the product measure on `Ω^S`
  that the luck premises' "whatever the past" is over); one salt an epoch expanded into the rounds' coins isn't.
* **A cap on the cheater's hash calls over the window.** The budget asks `δ_ZK ≤ 2^-41/(48,000·n_π)` a proof and
  `δ_fork ≤ (2^-41/48,000)²` a round, and C-Flock's soundness and `cr/sha-512` give such bounds only for a cheater
  whose hash calls are bounded: against C-Flock's `t·2^-λ`, `t ≤ 2^(λ − 56.6)/n_π` over the window.

The sheet reading's ε rests on A2u (`SheetDispatch`: each uniform-datapath instruction takes an issue slot), which no
vendor document states. DS-120 has no uniform-pipe row, so without A2u the machine model's LP gives 2.79% with a
measured uniform-pipe row (0.766 a SMSP-cycle) and 100% with none, against its 0.585% with A2u.
A probe on sm_120 reads A2u as holding: past
an issue-bound stream's idle slots, at up to 0.954 of issue, each uniform instruction costs about one issue slot.
With uniform chains beside a saturating vector stream, at 8, 2 and 1 warps a SMSP, no kernel's total rate
exceeds 1 over 279 measurements (max 0.9999), so A2u holds at saturation, and nothing dual issues.

It rests on A5b too (`SheetSmem`): a bulk or TMA instruction holds its size `S` of shared-memory bank bytes at A5's
rate, counted `S/99 KB` in UTMABIG (`loads.bulk_counts`), which no vendor document states either. Without it DS-120
charges a bulk reduction one issue slot and no other row, and there is a
bit-exact fold below the menu: fold_bulk.cu's fbw4096_K4_BT256, fx1's fold with its XOR on
the bulk engine, costs 15.766 issue slots a unit against 16 at the issue facet, and its full-work run loads DS-120's
most-loaded row at 0.982 of ♭. The floor that moves `d′` by STG.256 loads no DS-120 row above the menu's fold and is
below it on issue (13.5) and the LOP pipe (8 to 16), so no certificate with weight on issue prices it at `φ*` without
the bytes. With A5b, `π*` takes 197/3761 on DS_smem, the least weight at which every bulk-reduction floor costs
`N_w φ*` (the STS.128 one exactly), and the ε is 5.860% where the issue facet alone gave 0.656%. No census program
exceeds 1 on DS_smem with A5b, and R6r's measured row already charges TMAred512 its 512 B. Even so the sheet admits a
run at 0.989 of ♭ (fbw on part of the words, the lane fold on the rest, on any bulk-byte reading), so no sheet ε
below 1.06% can hold; VF3s-16's `ROW_L2RD` loads fbw at 5.78 of ♭, so its real time doesn't come near.

sm_80 has no headline: FOLD-OPT is false at its ALU-pipe certificate, which doesn't price duo's `IMAD.HI.U32` rotate
route on the FMA pipe.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet**, under full checking with live coins. On DS-120's machine (`Sm120.sheet`),
H1 + fx1's TS model (`Sm120.model`, `q = q_T = ½`, `N_w = 128`) and its floor program (`Sm120.flat`), at its certificate
(`Sm120.piStar`: 3564/3761 on issue, 197/3761 on shared memory), over a window of `Sm120.checks = 48,000` equal live
rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most
`2^-41` over the window, if the labelling splits, A1–A7 hold on every run and FOLD-OPT per word folded at `π*` prices
every run's fold at the words it folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose
openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`, the first luck premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and
whose whole-unit luck is at most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs'
(`Of`), two runs off one commitment open it differently with probability at most `δ_fork` a round (`BindingForkAt`,
binding at `√δ_fork`), every proof is sound up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value
(`ProofsPin`) and E1(c) holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 5860/100000` (`1 − 5.860%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a
hypothesis: H1 + fx1 has no filler, so it holds at any run. `ε_sample` and the Fiat–Shamir premium are separate lines.
The bound is per node: an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at
least `1 − 5.860%` of the credited work. Which device did it is the attribution layer's: read per GPU, it needs LOC's
host deduction and every billable GPU on the node credited each window.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. With `S` rounds in place of
48,000 its two branches never give ε < 5.779%, and under 6% only from 17,460 rounds (Compose's live line is under 1%
only from 10,552); the windows run so far have at most 18.

The bound is tight to its own floor: `N_w(w* + φ*)` is `17·3564/3761 = 16.1095` a unit at `π*` against
`B_♭ = 17.0976` (`B_♭` and the credit don't move with `π*`), so the ε holds only while a cheater's accepted program
costs at least `(1 − 5.860%)·B_♭/(1 − 41/S) = 16.1094` a unit at `π*`, and its margin is FOLD-OPT's.
`π*`'s DS_smem weight is A5b's (above): fbw4096_K4_BT256 costs 16.79 a unit at
`π*` against `N_w φ* = 15.162`. A6's `SheetWords` sums over the design's `vec`, CUDA's lane classes and the 64-bit
ones (ALU64, CVT64, DMUL, IMADW, MOV64), so a run that heads vector words with IMAD.WIDE meets it.

This ε rests on A2u and A5b. A2 (`SheetDispatch`) counts each uniform-datapath instruction in an issue slot, which no
vendor document states, and DS-120 has no uniform-pipe row. Without A2u the machine model's LP gives 2.79% with a
measured uniform-pipe row and 100% with none. Measured, it holds at saturation: beside a saturating vector stream no
kernel's total rate exceeds 1 over 279 measurements. A2's catch-all counts an instruction the 122 classes don't name in
`NOROW`, at its issue slot alone, so no instruction is free at `π*`: a count of any class costs at least 3564/3761
there, and a bulk instruction is `S/99 KB` of a UTMABIG count. A5b (`SheetSmem`) holds a bulk or TMA instruction's size
`S` in the banks at A5's 32 B an SMSP-cycle (above). -/
def Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δfork δzk : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **sm_120's headline on its measured vector**, under full checking with live coins. On VF3s-16's machine
(`Sm120Measured.machine`), H1 + fx1's TS model on it (`Sm120Measured.model`: the MMA forms at their committed words an
instruction, `q = q_T = 9/8`, `b_T = 0`, `N_w = 128`) and its floor program (`Sm120Measured.flat`), at JI3t_0's
certificate (`Sm120Measured.piStar`), over a window of `Sm120.checks = 48,000` equal live rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, (I1) `FacetCeilings` holds on VF3s-16's rows, (I2)/(I4) `WordsWritten` at the model's `ω`,
(I3) `ScalarRoute` at its floors and (I5) `NoSum` hold on every run and (I6) FOLD-OPT per word folded at `π*` prices
every run's fold at the words it folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose
openings are pinned
(`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`, the first luck
premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and whose whole-unit luck is at
most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), two runs off one
commitment open it differently with probability at most `δ_fork` a round (`BindingForkAt`, binding at `√δ_fork`), every
proof is sound up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value (`ProofsPin`) and E1(c) holds
(`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 591/100000` (`1 − 0.591%`) of the credit `T_cred = B_♭/f_max`. (I5) `SideUnits` and
FILL-OPT are not hypotheses: the model has no side unit and no filler, so they hold at any run. On the lane route (I3)
at `9/8` is what `ClassPSlots` gives (`1 + (32 − w)/130` slots a word) once
writers create at most `w = 15` incidences, a cap nothing proves yet; on the table route it is SCALAR-ROUTE's, outside
`ClassPSlots`. `ε_sample` and the Fiat–Shamir premium are separate lines.
The bound is per node: an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at
least `1 − 0.591%` of the credited work. Which device did it is the attribution layer's: read per GPU, it needs LOC's
host deduction and every billable GPU on the node credited each window.

So the 0.591% rests on (I3) at `9/8`, an assumed premise, not on proved ones. At A6's `½` floor, which Lean proves from
A6 (`SheetScalarRouteHalf`, the floor the sheet headline reads), the same bound on the same vector is 7.791%: π* moves
to issue 0.645 + JI3t_0 0.355 and the vector route binds (10.24% on VF3r-16; the exporter's instance at `q = ½`). The
0.591% reads `9/8` until `ClassPSlots`' cap is proved.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. With `S` rounds in place of
48,000 its two branches give ε < 1% only from 8,320 rounds (Compose's live line only from 10,552), and the
windows run so far have at most 18.

The vector is VF3s-16, a refit of VF3r-16's joint issue rows, not model.toml's `current` (VF3r-16):
bit-exact census programs load VF3r-16's `π*`, JI3s_0, at up to 1.0383, and honest's IMMA16832 form at 1.0107, so
(I1) on its rows is false of runs that ran. No census program
loads a VF3s-16 row above 1 (`benchmarks/pouw/tests/test_lean_headline.py`). The credit is 2.9% lower: ♭'s bound
`B_♭` is 19.259 a unit at JI3t_0 against VF3r-16's 19.829 at JI3s_0. The ε reads `ln(8/7) ≥ 2/15`, its series'
first term, where the sheet's reads `1/8`: 0.588% at the data, under 0.591%.

(I1) has no catch-all: VF3s-16's rows price the 122 census classes it measured, and an instruction none of them names
is on no row. A2's catch-all is the sheet's: VF3s-16 itself prices some listed classes (UNI64, the TMA forms) at no
issue slot, so "at its issue slot alone" is no measured fact here. -/
def Sm120MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δfork δzk : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.TS
