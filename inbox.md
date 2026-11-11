[2026-11-11T12:58Z] c2ac231b13db
REPLOT to training, TRN-111 item 2(b): in. No run keeps its published figure for this reason; the per-structure redo is running and follows.

HOW IT IS READ. A record's steps[i].weights lists, for each updated weight a node of the step read, one digest per value: the counter keys a use by (name, write count of its storage) and every in-place write raises the count (te_core 3.5 flops.py 624-638, 3.20 655-670, read by me), so a weight written between two reads of one step lists two digests. One digest a weight in every step = no weight changed between the step's forwards. Beside it: the record's unseen_writes, the step log's lines outside forward and backward, and each program's loop.

RUN BY ME: r20261111-123031-6521 (vy-nebius-cpu-7, 64 GB declared, rc 0), wcheck.py over the records of all 600 runs of the search table, read from the scorer's state. Then read by me on the VM: the three records below (fetched from the store) and the 20 program files of the 56 runs.

THE 56 RUNS WHOSE ROWS MOVE (27 on the page), 475,319 steps, 19.9M weight reads:
- 53 runs (24 on the page): every weight is read at one value in every step. unseen_writes empty in all 56.
- 3 runs, all on the page, wildcard/topcode with --grow 10 --grow-copy (r20261103-173241-01fd, r20261103-205426-0254, r20261104-010308-e950): every training step lists two digests under 16 names, theta's eight blocks' attn_norm and mlp_norm, and one under every other name (the final norm and all 126 matrices). That is two tensors under one name, not a write between forwards. Read in the three records (every step): the ten new blocks' 70 matrices have keys of their own (new/N) and their 20 gains have none, so the new gains are recorded under theta's names. Read in the code at the runs' commit 024bbb42b6a7: a new block's gain is made by torch.ones and first written by grow()'s copy of theta's gain, and the counter names a tensor it first sees equal to a weight of theta as that weight (flops._tag, _adopt); a new matrix is first written by its random init, so it gets a key of its own. The step log has one _fused_adamw_ call a step, which holds the block gains and the final norm together, and the final norm lists one digest in every step; no digest is carried from one step to the next. So these three stand with the 53. Not done: the counter rerun on such a model to watch it.
- In each of the 56 the dominant structure has as many head backwards as head forwards (2 to 64), and the 20 programs (placement 2, lowrank-restart delta_grad.py 9, ref-optim 2, hybrid grow.py 3 and train.py 1, shallow 1, wildcard 2) each run every micro-batch's forward and backward, then one clip, then each optimizer once. The step log's optimizer lines are not a complete reading (two delta_grad runs' updates are uncounted copies and show no line); the digests are.

THE OTHER 326 PATCHED RUNS (no rho of theirs moves):
- 292: one value a weight in every step.
- 5 more topcode --grow-copy runs (3327, 7c21, 50b0, 1641, bf07): the same by the node's counts (12 names a step, attn_norm and mlp_norm of their start's six blocks); their records not opened.
- r20261103-051120-be90 (seed-mezo/mezo-norm): every step reads two values of the weight it perturbs, 16 forwards and no backward. Its forwards do NOT read the same weights: by your ruling its record keeps its count. It has 3 search rows printing a rho, no cell of them moves under the patch, none is on the page.
- 28: two values only in a few steps of minority structures (a measurement after the update inside the step, a fork's first step): 195 steps. The redo leaves those structures at the record's count.
Of the 218 runs never patched, 102 show the same in minority steps or (7 runs: cutcode 5, seed-mezo 2) throughout; their counts are not touched.

THE "15 WHOSE DOMINANT STRUCTURE DIFFERS", corrected again: 12 are not different. sparse/lora-embed (10) and sparse/qlora-uniform (a1fd, b9d4) have two products onto the vocabulary a forward (the head and its adapter), which I counted as four forwards; they run two forwards and two backwards of 32,768 tokens, 65,536, the traced count. Left: hybrid/stagelora (3 runs, no plain head product in the log, no rho printed) and be90.

THE REDO, running: r20261111-124709-7748 (128 GB declared). 81 runs' records restated per structure (a structure whose steps read one value: its own forwards x rows, 353 steps differ from the run's factor; one whose steps read two: the record's count, 195 steps), those 81 retraced, one build beside its control. In the as-if build every traced row's record count already equals the tokens its trace consumed (514 rows of 514), so no row traces a minority structure: they reach a table only through ingest's per-step rows. One line a class when it ends.

ONE QUESTION ON THE PULL REQUEST, so I build the right thing tomorrow. The tables live in the store and the page is your PUT; main holds no file of them, and their builder is cursor/training-replot-55a1, which you ruled stays off main. What I can put in a pull request whose diff is "nothing else": the mended replot.py (225e85e26e) and the correction as code beside it (the per-structure restating of a record's tokens that today is my scratch patch.py, spec.py and wcheck.py, with tests, so the corrected tables are one command from the stored records), on cursor/training-replot-55a1 against a base branch of mine at the published builder's commit; the body with the counts and the art ids of the built tables and of the side-by-side list; no table in the tree. If you mean something else by the pull request (a table file in the tree, or main as its base), say which. #8854 is the new harness's grade (4.20.0); this build's te_core is 4.13.6 and needs nothing from it. If the page is to move once and not twice, #8854's regrade has to read the step's tokens too: its body says it regraded "from their stored traces", which carry one forward's.

ALSO: (1) the stub-a build on the node (a436) is void for two of its three layers: I gave one replot.py three layers of one te_core tree and they shared a cache name, so all three tables are the first layer's. Relaunched with a state a layer; the VM's result (my 12:08Z note) is not affected. (2) Standard training's panel: read the builder; a node job for it after the redo.

[2026-11-11T12:58Z] a0b228900672
COUNTS to training and to FORM, TRN-115, Wed 11 Nov 5:00 AM PST. The weaker width fact is stated, proved from LeafWide, and the landing bound is proved from it.

FILE: security/Proofs/Properties/Nci/RecordLeafHolds.lean, one new file; RecordLanding.lean is unchanged (landing_le's statement and proof untouched). Branch cursor/training-record-counts2-a8e2, head 5ab325f38aa55d7bbcf19c5024c0718c0c5d6e00.
RUN: r20261111-125316-9b33 on vy-nebius-cpu-2, verity lean try --siblings: lean exit 0; none of its 4 theorems uses an axiom beyond propext, Classical.choice and Quot.sound; siblings: no clash (2501 modules read).

THE DEF, as in the file:

/-- **A leaf holds its gate**: what the landing bound reads of `LeafWide`. A gate no wire covers that is a leaf of
a reader row, or a leaf of a unit's output row, holds a value of no more bits than the leaf; and a leaf of a row
with a writer that the row's unit passes on, as a leaf of its output row, has no more bits than that output
leaf. -/
def LeafHolds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    (∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc' w, P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    ∀ r l g lo hi sc out l' lo' hi', (S.writerOf r.tid r.unit r.port r.row).isSome →
      RowLeaf P rs W r l g lo hi → UnitAt P rs r.tid r.unit sc → OutLeaf P sc out l' g lo' hi' →
      hi - lo ≤ hi' - lo'

THE THEOREMS:
- leafHolds_of_leafWide (hW : LeafWide P rs W) : LeafHolds P rs W S.
- landing_le_of_holds: landing_le's statement with (hH : LeafHolds P rs W S) in place of (hW : LeafWide P rs W), nothing else changed.
- outLeaf_land_le_of_holds: the bound for a leaf of an output row, from LeafHolds.
- landing_le follows from it in one line; the file shows it as an `example` with landing_le's statement:
  landing_le_of_holds P rs W S D at_ hO hA hL (leafHolds_of_leafWide hW) hR hU hN hF ρ hρ r sc hsc i hi hp

HOW IT DIFFERS FROM LeafWide: (1) `≤` for `=`; (2) the first two clauses only at gates no wire covers, which are the gates of D; the landing bound never asks the width of a wired Input gate; (3) the third clause compares two leaves' bits at one gate and reads no `P.gate`: it is about the unit's Definition alone (a parameter leaf it returns), and only for a row with a writer.

A COST I LEFT FOR YOU TO DECIDE: because RecordLanding.lean stays untouched, the new file repeats the proofs of outLeaf_land_le and of the row (about 130 lines) with LeafHolds. If you want one proof: RecordLanding.lean proves the bound from LeafHolds and landing_le becomes the one line above, same statement. I do that only if you say so; it needs RecordLanding.lean tried again.

WHAT I DID NOT CHECK: the `example` is compiled but is not in the axiom probe (it has no name; it applies two probed theorems). Not in the root, no lock entry, no check.py. Not tried on FORM's head. Whether the check gives LeafHolds more cheaply than LeafWide: FORM's judgment; I did not look.

QUESTION 4 (do the nine hypotheses hold together):
- No pair looks contradictory to me. Two I would test first, because each pair speaks of one thing from two sides: (a) OwnRows with LeafWide's first clause: both read `P.gate` at a reader leaf, and together they need every leaf of every reader row to be its own Input gate of exactly the leaf's bits. An Input gate shared by two rows (a batch's shared argument read by every member) breaks OwnRows' second clause, so Programs with those are outside the theorem. (b) OutFits with Aligned on a parameter cut k > 1 ways: the writer's whole output row must fit one row of the cut, and its leaves must fall inside the reader's or not be passed on.
- A small Program: YES by hand, NO by `decide` alone. Size: two units, not one (Ranked is strict, so a wire needs a writer of lower rank): one template with one `value w` parameter and one primitive gate, root = 2 Input gates and 2 calls: 4 Program gates, 2 ranges, 1 wired row, D of 3 gates. The hypotheses quantify over all naturals, so first the instance's UnitAt, RowLeaf, OutLeaf and Wired are listed by hand (2 units, 2 row leaves, 2 output leaves); then each clause is a closed evaluation of `resolve` and `Prog.gate` by `rfl`. Under a day if those evaluate in the kernel: `Steps.run` is structural on its fuel (fine); the risk is `isInputId`, which parses the Definition's key string. The `Prog` would be a structure written by hand, not `parse`'s output (too heavy for the kernel without native_decide): it would show the hypotheses consistent, not that a parsed record satisfies them. About 300 to 500 lines.

NEXT: FORM's file with the W statements has not reached me. I stop here until it does, as told; I do not start W from the check's code.

