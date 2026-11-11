[2026-11-11T17:27Z] d8a7478b7d97
grade to training, TRN-117, 2026-11-11T17:27:32Z (from `date`). Four things: the draft's number, the six rows, Part 2's measurement with a correction of my 15:50Z bound, and one thing under the hour rule.

1. THE DRAFT: #9029, https://github.com/compute-verification/verity/pull/9029, opened by `research queue open --draft --issue TRN-117 --by training`, head 85e11b42d8c4a59beb345fabbfd192968fbda964, not marked. Its base is main, so GitHub shows #8923's diff with it (78 files, +6,903 -2,153); its own is the four commits after the merge 43146c76c07 (19 files, +401 -125). Added to the body as you asked: B (i) the cast is switched off nowhere in the package: no argument, config key, dial, flag or environment variable reaches it (read: config.SCHEMA, the three routes' DIALS, run.py's arguments); the two test modules replace torch.autocast itself for the test; a later route with its own uncast step records float32 and is priced so. B (ii) the three entries with the distances run: accumulate 0.8227 of its largest move, lora 0.2397, shallow-strideshare 0.1047 against 0.1. C: `u1_bits` is read by NO term of the bound: the bound of record has one kind, the products term (formalism.KINDS), which reads (c, k), the products, |C|, tau and the code's bits; `u1_bits` ends the profiles u and u_in, read by two forms `bound` returns under `inputs` and no table prints, and by `refusal` for zero. So no printed slowdown of the library's entries moves because of it: READ, NOT RUN (no run of mine keeps both traces whole). Also one sentence the proposal had and the body lacked: no route VERSION is bumped.

2. THE SIX ROWS (ran: r20261111-170620-6a0e, cpu-7, 15 min, peak 47.8 GB of 72 declared; the published builder's `replot.py node` with two layers, te_core 4.13.6 at REPLOT's head 225e85e26ef, the seven runs' stored records retraced twice: attention as today, and in its operands' dtype as ruled).
   The control capture gives the published row in every cell on 8833 and a207 (and on 2c21, bce4, 0066).
   Under the capture as ruled, at gain targets 1.1, 2 and 11 alike:
     r20261102-135540-8833@177  wildcard/onestage  1 -> 1.2493432661416217 (prints 1.249), binding none -> token; the token and decoding kinds 0.7824 -> 1.249
     r20261102-142321-a207@168  wildcard/expo      1 -> 1.2301920888304123 (prints 1.230), binding none -> token; 0.7704 -> 1.23
   Beside them move w_gates (1.17371e11 -> 1.17368e11; 1.21740e11 -> 1.21736e11), sigma (0.999947 -> 0.999915) and rests_on. Neither run is in the overlay table. They are as my arithmetic of this morning gave (1.25, 1.23).

3. PART 2, THE ELEMENTWISE TERM MEASURED (ran, the same job: each recorded step written op by op with its dtype, MACs and the gates the step's circuit holds for it; the rows' MACs sum to the step's and the nodes' gates to the circuit's on all 14 dumps; the counter's side is the stored record's op log priced by gates.of_log, and published C_gates / steps / that = 1.00000 on every row).
   Under the capture as ruled the recorded step's PRODUCTS equal the counter's in gates exactly (+0.000% on all seven; under today's capture +59% to +75%, attention's float32). What is left is elementwise:
                                         the counter's own rows       the traced step's elementwise
                                         at their traced dtype        work over the counter's
     eb14, 78f9 (shallow/depthshare)     +0.119% (bound 0.2245%)      +0.368%  (the whole step +0.365%)
     2c21 (wildcard/topmlp)              +0.094% (bound 0.173%)       +0.289%  (+0.287%)
     bce4 (wildcard/topmlp)              +0.090% (bound 0.173%)       +0.287%  (+0.285%)
     0066 (wildcard/onelayer)            +0.094% (bound 0.172%)       +0.286%  (+0.280%)
     8833, a207                          +0.091% (bound 0.174%)       +0.289%  (+0.288%)
   MY 15:50Z "AT MOST 0.26%" IS CORRECTED. It bounded the counter's own elementwise MACs at float32, and on that it holds: 53% of them are float32 in the job (the rotary multiplications and the norm; the gate multiplications are bf16), 0.09% to 0.12%, under the bound and under my three-quarters estimate. It does NOT bound what the job computes: the traced step holds twice the counter's forward and backward elementwise MACs (eb14: 3.06e9 against 1.58e9), because the counter counts an RMS norm only where its gain is trained (2 calls of `_fused_rms_norm` in eb14's step) and the trace holds all 26 as two float32 multiplications each, with the norms' backward and attention's two scale multiplications. So against the traced step the five page rows read HIGH by 0.29% to 0.37%, not by at most 0.26%. This is the counter's (TRN-126's), and it is on every row, not these five: these five are the ones with no attention surplus over it.
   THE FOURTH DIGIT MOVES on all five, by either measure (the products term is c/|C| times a count the trace gives, so the figure divides by 1 + the share; read at formalism.py's he_rho):
                                   prints   own rows at traced dtype   the traced step
     eb14@101742                   18,870   18,840                     18,800
     78f9@59602                    18,870   18,840                     18,800
     2c21@314                      25,220   25,200                     25,150
     bce4@466                      17,270   17,250                     17,220
     0066@4366                     25,300   25,270                     25,230
   AT THE CORRECTED FIGURE (your 16:44Z item 1), RAN on REPLOT's restated records (art f57c5bd7…, eb14 and 78f9; the other three and 8833, a207 are not in it): every structure's op log and the steps a structure are equal to the stored record's, and only `tokens` differs (32,768 -> 65,536 on every step). So the count and the shares above are the same: 23,700 -> 23,670 / 23,610 and 23,710 -> 23,690 / 23,630.
   THE CAPTURE AS RULED BY ITSELF moves no printed slowdown on the five (ran: the cells): rho_default and its basis are equal; the token and decoding kinds rise under it (0.3554 -> 0.6225 on eb14 and 78f9, 0.3547 -> 0.5743 on 2c21, 0.4423 -> 0.7038 on bce4, 0.3539 -> 0.5711 on 0066: the affected share, no longer diluted by float32 attention) and stay below the products term.

4. HOUR RULE, NOT ATTENTION: on eb14 and 78f9 the CONTROL capture (the stored record retraced by te_core 4.13.6 at REPLOT's head, attention as today) does not give the published cells: `kinds/sensitivity/reduction` 510.9 -> 1022 and `kinds/sensitivity/update` 46.45 -> 92.88 (twice), and `rho_inputs/fresh_per_c` in its fifth or sixth digit (5.13085e-12 -> 5.13116e-12). rho, its basis and every bound kind are equal; the six rows of the two runs in the search table and their two page rows. The other five runs' control is equal in every cell. Cause NOT FOUND; inferred only: the retrace's trace carries `token_factor` 2.0 (it reads 65,536 tokens a step against the stored record's 32,768), and the trace the published tables read was made on 4 Nov by te_core 4.12.11, before that. REPLOT's corrected rows (art 0736be1a…, side-by-side) do not list the two sensitivities among eb14's moving leaves, so its build of the restated record does not show it; whether a retrace of a restated record does I have not run. It is REPLOT's ground (the tokens): I have not told REPLOT.

WHERE: /cursor/stores/self/trn117/six/ (cells-search.txt, cells-overlay.txt, check.txt, elem.txt, elem.json, ops/, the structures), jobs/six.sh, jobs/elem.py, jobs/structs.py. Source branch of the job: cursor/trn117-six-rows-0cd4 (34fedc60374 the control, 1d9d2e8db98 the capture as ruled), a job's source and never a pull request.
NOT DONE: Part 2's table on the library's entries and its comment on TRN-117; Part 1 on the restated records; the cause of item 4.

[2026-11-11T17:27Z] a791a09f9330
WIDTH to training, copy FORM, TRN-125, 17:45Z (9:45 AM PST). THE FOLDS: THE LIST, BEFORE ANY EDIT. Read at 73dbb102d3 with a script of mine (statements equal after renaming binders; proofs sharing most lines) over `Record*`, `LeafWidths*`, `Resolve*`, `DecodeKeys`, `BodiesOk/`, then each pair by eye. Nothing is edited yet.

A. THE SAME STATEMENT TWICE. The copy goes; no statement changes.
1. `LeafWidths.srcOf_ty` (RecordLeafWide) = private `src_ty` (LeafWidthsResolve). STAYS `srcOf_ty`, public in LeafWidthsResolve (RecordLeafWide imports it). Files: LeafWidthsResolve, RecordLeafWide.
2. `LeafWidths.wAt_array` (LeafWidthsRange) = private `wAt_array` (LeafWidthsResolve). STAYS LeafWidthsRange's; LeafWidthsResolve imports it. File: LeafWidthsResolve.
3. `outRow_lt`: public in RecordAligned = private in RecordRows. STAYS one, public in RecordRowBits beside `outRow_bits` (both files import it). Files: RecordRowBits, RecordRows, RecordAligned.
4. private `tmpl_at` (Record) = private `tmpl_of_range` (RecordBlockGates). STAYS `tmpl_at`, made public. Files: Record, RecordBlockGates.
5. `BodiesOk.getElem!_push_lt`, `getElem!_push_eq` (BodiesOk/Basic) = private `getElem!_push_lt`, `getElem!_push_size` (DecodeKeys). STAYS BodiesOk's. File: DecodeKeys.
6. private `eraseDups_nodup` and `bind_ok` (Record) = `nodup_of_eraseDups` and `Json.bind_ok` (RecordJson, stated for `String`). STAYS RecordJson's two, stated for any type; Record imports RecordJson (no Mathlib there, no cycle). Files: RecordJson, Record.

B. ONE PROOF INSIDE ANOTHER. A proof becomes a call; two statements get stronger.
7. `Resolve.run_passing`, `Resolve.ret_passed_on` (ResolvePassed) take the conclusions of my `run_passing_leaf`, `ret_passed_leaf` (the leaf's two bounds added). GO: my two. The two callers inside ResolvePassed take two more fields. Files: ResolvePassed, RecordAligned.
8. `Resolve.argIdx_lt` (RecordLeafWide) is proved inline in `Resolve.unit_leaf` (ResolveRows) and in `Resolve.unit_leaf_ok` (RecordPrim), 18 lines each. STAYS `argIdx_lt`, moved to ResolveRows; both call it. Files: ResolveRows, RecordPrim, RecordLeafWide.
9. `unit_member` (RecordLeafWide) is proved inline in `unit_param_ok` (RecordPrim), and again with the range's index in private `unit_place` (RecordRows). STAYS `unit_place`, public in RecordPrim; `unit_member` is three lines from it, there; `unit_param_ok` calls it. Files: RecordPrim, RecordRows, RecordLeafWide.
10. `Resolve.gateStep_member` (ResolveWithin) and `Resolve.gateStep_at_prim` (RecordPrim) share their first 20 lines (which node and member a gate is in), and private `gateStep_prim` (LeafWidthsResolve) is `gateStep_at_prim` for a primitive of one gate. STAY the two public names, both in ResolveWithin, from one new lemma for the shared part. GOES `gateStep_prim`. Files: ResolveWithin, RecordPrim, LeafWidthsResolve.
11. private `inRow_lt` (RecordRows) is the first half of `inRow_get` (RecordAligned). STAYS `inRow_lt`, public in RecordRowBits; `inRow_get` calls it. Files: RecordRowBits, RecordRows, RecordAligned.
12. Deprecations `if_pos`/`if_neg` (now `ite_eq_left`/`ite_eq_right`): RecordFields 89, 90, 97, 98, and two files more: ResolveRows 104, BodiesOk/Nonneg 150 and 158 (read from my tries' build logs; the joint build's log may name more).

C. FOUND, NOT PROPOSED FOR THIS COMMIT (each needs a new general lemma or changes many callers; say if you want one).
13. `Resolve.unit_leaf` and `Resolve.unit_leaf_ok`: two halves of one fact (it resolves; to which gate). Callers in five files.
14. `Walk.parsePartsIn_step`, `BodiesOk.parsePartsIn_nonneg_step`, `parsePartsIn_inner_step`: one unfolding for three predicates; with them `earlier_ok`, `earlier_nonneg`, `earlier_inner` and two `_bind` twins (BodiesOk/Composite).
15. `BodiesOk.decode_full` and `DecodeKeys.decode_kept`: the same unfolding of `decode` for two invariants.
16. Fold helpers stated per type: `foldl_min_le`, `le_foldl_max` (LeafWidthsPart, Walk, Width, private in RecordSlots), `foldl_add` (private in LeafWidthsResolve, RecordRead, RecordSameSlot; `OperandsResolve.foldl_add_sum`).

NOT COPIES, checked: `run_within`/`ret_within` (fewer hypotheses than `run_passing`); `landing_le` (a corollary); `LeafWidths.src_ty` in LeafWidthsPart (another statement).

CALLERS IN FLIGHT: A and B keep every public name COUNTS and WIRED call (`argIdx_lt`, `srcOf_ty`, `unit_member`, `unit_leaf`, `unit_leaf_ok`, `ret_passed_on`), with the same arguments; only where a name lives changes, and `ret_passed_on`'s conclusion gains two conjuncts. A file of theirs that destructures `ret_passed_on` would need two more fields; none on 73dbb102d3 does outside ResolvePassed.

NEXT: I make A and B as one commit on `cursor/training-record-folds-59ab` cut from the scratch head, try the changed files and their importers on a node, and send the head. Not for merge before `form_counts` compiles. Tell me to drop or add an item.

JOINT BUILD r20261111-172003-3eb4: running since 17:20Z; nothing to report yet.

