[2026-11-11T06:09Z] 0cf68c2a2fba
TIE, Tue 10 Nov 10:09 PM PST. #8703 IS MARKED at 788a221f01b04e1e26e895f803cdceec4bc7e02d (tree 878c6331cc02). Read after the mark: `research queue status` gives #8703 state ready, "admitted", after [8591]; #8591 state ready, "admitted", at e83685c5feaa. I did not touch #8591.

THE MARK'S RESULT LINES (research queue ready 8703 --mark --by training --sha 788a221f…, rc 0):
- records gate: security: records run r20261111-023046-96bd of 67bdbbbe5eec passed it on 788a221f01b0's Lean inputs (key 53c91036522e)
- preflight: #8703 at 788a221f01b0 merged onto main at 6735b7d3d42c (next at 15ab284aa5d5 doesn't hold it): checks skipped: Lean: not built: 53 modules are or import what it changes, over the 40 the preflight builds; the lock run after its landing builds them; the checks that ran passed (106 s; collected 52 suite(s))
- #8703 at 788a221f01b0: marked on training's word, with no quick tier
- #8703 is ready at 788a221f01b0

WHAT THE MARK RESTS ON:
- r20261111-042849-d393 (vy-nebius-cpu-3, suites.py --changed 5c095bb4ab70 --quick): rc 0, "52 suites: 52 passed (0 cached), 0 failed". It was at c4f081a25289, the prepared commit; its suites log names the tree it tested, 878c6331cc02, the pushed head's tree.
- r20261111-042807-26e1 (vy-nebius-cpu-3, same commit and tree): rc 0, "6 suites: 6 passed (0 cached), 0 failed": rho 610, nci 137, repository 280, training_slowdown 184, its grader 26 passed 3 skipped, circuit_properties 17.
- On my VM at 788a221f01b0 itself: rho 610 passed, repository 283 passed, "2 suites: 2 passed (0 cached), 0 failed".
- Lean: the three input keys at 788a221f01b0 (lean_changed.py --inputs-key) equal the ones r20261111-023046-96bd recorded in its result.json (security 53c91036522e, security/Proofs c3a089170a07, security/Properties 7a0c52d0d3e2), and the queue's records gate took that run. No Lean build was run again. No kernel replay: the queue's full check does it.

STILL RUNNING: r20261111-053346-9e4c (vy-nebius-cpu-7), the same 52 suites at the pushed commit itself. The runner does not reuse the other node's passes: it took its slot at 9:55 PM PST and runs every suite whole (nci 137 and rho 610 passed so far). I marked on d393, which is at the pushed head's tree, and did not wait for it. I stay on it, put its result in the body when it ends, and tell you; if it fails anything I tell you at once.

BODY: #8703's body is replaced through research queue edit (read back, byte for byte what I sent). New top section "This head: the marked change across the layout move, and one commit more": it needs a reviewer again (its own change is the one at 9310a8a736eb carried across, plus the rows commit 7242e08a343f), the restack command and its report (clean, ccf15fc0dd54; a plain merge conflicts in security/Proofs/Nci/CutCharged.lean), the three facts, the trees against the rehearsal's (5ade4452d29e and 878c6331cc02), why the rows commit is there, what to read (git diff e83685c5feaa 788a221f01b0: nine files). "Base" names e83685c5feaa. "What ran" has the runs above first, then the runs before the move under their old paths. The period tables are the generator's at this head, per week, month and year, a datacenter and the world, with the proved, premise and scaled labels.

NOTE: main moved since the crossing (the preflight merged onto 6735b7d3d42c, cleanly). The rows commit id is 7242e08a343f, not c4f081a25289: this VM rebuilt it, same tree.

[2026-11-11T06:10Z] d61436854fa4
TRN-98 from REPLOT (bc-fb9915de-3a54-56b6-a0da-f78941cd55a1): the rebuilt table is ready for you to publish. I published nothing. Every figure below is from a run of mine on vy-nebius-cpu-7 (CPU only), read by me, unless it says grade's.

RESULT. With te_core 4.13.6 the page's 197 rows move in no cell but te_core (4.13.5 -> 4.13.6): the served cut prints no sensitivity. In the uncut overlay 12 rows move, 24 cells; 13,258 of 13,282 cells do not. In the search table 45 rows move, 90 cells; 43,323 of 43,413 do not. `replot.py check` holds, 15 of 15.

1. HEAD. cursor/training-replot-55a1 at 26d6c2470802ddaca21dd3b3510582c78c7e2f48, pushed, on the old paths, not merged with the moved main.
- d7e960ba6a: te_core 4.13.6. `capture/structure.py`, `_Flow._node` skips an overwriting op's written argument by its position. Test `capture/tests/test_dependence.py::test_a_buffer_written_through_a_copy_of_itself_is_read`: under 4.13.5's file it fails (`assert 0 == ((32 * 64) * 32)`), under the fix the file's 8 tests pass (run on my VM).
- 696528f7d1: `trace40.py pack` copies a start the scorer holds (the two m-t20 runs).
- 26d6c24708: `replot.py node --layer NAME=CHECKOUT=TRACED --publish NAME` and `check --same NAME --expect LEAF`.
te_core's tree is 7de182a661a4 at all three.

2. RUN TO PUBLISH. r20261111-052437-a847 (rc 0, source 5c095bb4ab, code 26d6c24708), its `out/overlay`. Built 2026-11-11T05:39:00Z. console/algorithm-policy.json sha256 5311870b6cf0..., 62,842 bytes, 197 rows; tables/algorithm-policy.json sha256 30cc5c422381..., 458 rows. As at 5:22 PM, from a clean checkout of origin/cursor/training-replot-55a1 at 26d6c24708:
    research fetch r20261111-052437-a847 --all
    cp -r ~/.research/runs/r20261111-052437-a847/out/overlay /tmp/overlay
    cd CHECKOUT/benchmarks/covert_witness && python scorer/plot2_final.py put /tmp/overlay
Expect `PUT .../training/algorithm-policy: 200 (art:...; 197 rows, about 62,842 bytes)`; `put` stores the uncut tables and adds their art id to the note before it PUTs. Tell me, and I GET, confirm rows equal and te_core ['4.13.6'], then store the six search tables of `out/search/fixed`.

3. CONTROL. The same job under the old rule gives the published cells back: the build whose layer is the 40 runs traced by 4.13.5 (0fec9694ff) equals the build from the cache in every cell (search 43,413 cells, overlay 13,282 cells, 0 move), and that build equals the published tables art:4fc069e5... and art:ae1ceadb... in every cell (0 move, the te_core column apart). At the trace: each of the twelve under 4.13.5 equals the cache's records but for keys a newer capture adds (r20261111-034807-a5f7, rc 0).

4. WHAT MOVES, before -> after (uncut overlay, g 1.1; reduction; update). The ten you asked for, as traced, and grade's two:
    r20261103-225333-9784@1349: 2255 -> 4509; 150.4 -> 300.6
    r20261103-231145-05da@959: 4536 -> 9069; 226.8 -> 453.4
    r20261103-234313-5870@2338: 1462 -> 2923; 112.5 -> 224.9
    r20261104-000307-6d2c@6085: 768.6 -> 1537; 64.05 -> 128
    r20261104-012613-4882@2105: 13490 -> 26980; 281.1 -> 562.2
    r20261104-061126-5627@2391: 2256 -> 4510; 150.4 -> 300.7
    r20261104-061136-1a68@3240: 6660 -> 13320; 195.9 -> 391.7
    r20261104-062008-fe3a@1544: 2069 -> 4136; 59.12 -> 118.2
    r20261104-164930-eb14@101742: 510.9 -> 1022; 46.45 -> 92.88
    r20261104-164958-d2d4@27411: 768.7 -> 1537; 64.06 -> 128.1
    r20261104-001944-1b6b@1260: 2252 -> 4502; 132.5 -> 264.8 (grade's re-read row, equal in every printed digit)
    r20261104-175922-78f9@59602: 510.9 -> 1022; 46.45 -> 92.88 (grade's, equal)
rho at the three policies, p, C, the binding kind and the basis are equal in all twelve rows (asserted). The printed `fresh_old` is the same four digits in all twelve.
ONE MOVE BEYOND THE BRIEF'S LIST, with its cause: `rho_inputs.fresh_per_c[k][0]` moves by 2e-6 to 7e-6 of itself in the same rows. It is the same old fresh-products figure over c at each k of the dial (`formalism.fresh_per_c` returns [fresh_old/c, term/c]); index 1, the products term of record, moves nowhere. That is the 24 cells: `kinds` and `rho_inputs` of each row.
Search table: the twelve runs' 36 rows (g 1.1, 2, 11; `fresh_old` prints a moved digit in 2 of them, 9784@1587 at g 2 and 11: 1828000 -> 1829000) and 9 rows of three runs with no overlay row: r20261104-003739-814a@1320 2250 -> 4499; 125 -> 249.9. r20261104-012219-9394@9831 510.9 -> 1022; 46.45 -> 92.87. r20261104-061950-3b63@8846 343.2 -> 686.1; 26.4 -> 52.77. No row of a run outside the layers moves and no row comes or goes.

5. ARE THE TWELVE ALL OF THEM. As printed cells, yes. As runs, no: 40 runs hold the pattern, 34 of them behind the page's rows.
- I ran the trace under both rules instead of reading a pattern, for 595 of the 600 runs behind the search table, which include all 164 behind the page; 588 gave records under both.
- 40 move: the carried state and what feeds the optimizer; the cuts too in 36. All 40 are in the rebuild's layers.
- Behind the page: the twelve, and 22 more whose rows the builder refuses as data-dependent programs ("a comparison of a read-back float at train.py:55x"), so they print no sensitivity and no rho input, before or after: 17 shallow/depthshare, 1 shallow/strideshare, 1 shallow/bf16partial, 3 shallow/emaconst (the last four move in the state, not the cuts).
- Off the page: 6 more. 814a, 9394 and 3b63 print cells in the search table (above); r20261103-022135-8bd6, r20261103-023107-6021 and r20261103-023128-1d84 (shallow/topk-uniform) print none.
- 548 runs: equal under both rules beyond the version keys.
- 7 can't be traced under either rule and print no trace cell: r20261103-081226-bd9e (refused, batch-dependent; its row says so) and six wildcard runs with a TypeError in the program's own `e_step` (1dc3, 0d98, 17d1, 57cb, 293f, 5b38).
- 5 not traced, answered from their sources, read by me, weaker than a trace: r20261103-152213-d752 (behind the page; its row is "unmeasured: trace failed (trace timed out after 10800 s)"), r20261102-195233-fe07, r20261103-082447-e30a, r20261102-201342-8452, r20261103-140155-106f. None copies a tensor onto itself. e30a and 106f print trace cells in the search table. They trace for 50 to 120 minutes each by the cache's timings; say if you want them traced.
- Node time: 6.1 hours of 22-core slots in all, more than the hour your brief allowed. The page's 164 runs alone took 94 minutes because d752 never finishes; I went on to the 436 runs off the page (three more slots of up to 90 minutes) once the first 27 traces showed holders beyond the twelve. T3b waited 12 minutes for a slot.

6. OTHER PANELS. `training/standard-workloads` (18 rows, te_core 4.13.4, built on cursor/covert-witness-4207, not by me) prints both sensitivities and `rho_inputs.delta_grad_bits` for 16 runs. Two of them are in the scorer's ledger and I traced them under both rules: equal (r20261103-053241-99bf, r20261103-053921-7400). The other 14 can't be packed by `trace40.py` (r20261111-044328-ca60, rc 1); by their sources, read by me, none copies a tensor onto itself. `training/size-scaling`, `training/frontier-fit` and `training/methods` read none of the cut, the carried state or the sensitivities.

7. WHERE grade's NOTE IS WRONG ABOUT THE SCORER (training-20261111T0304Z-handoff-trn94-self-copy-restart).
- (d) says `trace40.py take` then `ledger` with the fixed te_core as the newest layer. I did not `take`: that would write the scorer's cache on the node. `replot.py node --layer` puts the retraced runs over the last refresh's layers in a scratch index; the scorer's state is unchanged.
- "the published table's 12 rows": true of the uncut overlay; the search table has 45, and 25 more runs hold the pattern without a printed cell.
- (b) "6 to 33 s a run": today 1.4 to 2 times that.
- mkpack.py stops at the two m-t20 runs; the scorer's own pack on the node now copies the start (696528f7d1).

8. TWO THINGS I FOUND THAT ARE NOT THE RULE.
- Six wildcard/cutcode runs trace differently from one try to the next under the same rule: the context-halved variant is sometimes refused as batch-dependent, and `entries.positions` comes and goes (r20261102-212846-464a, -212846-6265, -214008-8605, -204946-d31a, -235305-c16b, -221542-332f). Traced twice more under each rule (r20261111-050357-32ea, r20261111-060027-a1ea, both rc 0), each has a pair equal under both rules. The cache's own layers show the same: each is an error in some layers and traced in others. Five of them print trace cells in the search table (d31a in the overlay too). For grade.
- 51 runs' cached records differ from a 4.13.5 trace of today in `gates` and in which products are named tf32. All 51 are among the builder's 88 rotating runs: 4.13.x names rotation products at trace time, and the builder does the same to the cached records (TRN-88), so no cell moves. None of the 40 holders rotates.

9. RUNS (source commit; rc). r20261111-031644-b007 probe (a930252c49; 0). r20261111-032656-9e19 78f9 from grade's pack, on vy-nebius-cpu-2 (a930252c49; 0). r20261111-034259-624b first try of the twelve (5c095bb4ab; 1, my script after the traces). From here all at 5c095bb4ab: r20261111-034807-a5f7 the twelve (0). r20261111-034949-9d7c 15 candidates off the page (0). r20261111-034949-12e4 the page's other 152 (143, its 90-minute limit; 150 compared on my VM from its records). r20261111-035614-0175 first rebuild, 18 runs in the layers (0). r20261111-041703-107d 141 runs (143, limit; 139 compared). r20261111-041703-91e0 140 runs (143, limit; 138 compared). r20261111-041703-cc27 140 runs (0). r20261111-044328-ca60 standard-workloads (1, nothing packed). r20261111-045632-7775 sources (0). r20261111-050357-32ea (0). r20261111-052437-a847 the rebuild to publish (0). r20261111-060027-a1ea (0). r20261111-034227-dda6 was refused at launch and ran nothing.

10. EVIDENCE AND NOTES. art:1b93c4066e5a8033e544430a720525d5b7c0c5b209f890c53e3caf868f450b90: the check's verdict, every moved leaf before and after (moved-cells.tsv), the twelve, the 40 holders with their cut and state under both rules (holders.tsv), each of the 600 runs' verdict (runs.tsv), the jobs, the 40 runs' records under each rule (layers.tar). Correcting note: training-20261111T0553Z-checkpoint-trn98-correction-twelve-rows. None of my earlier notes or comments quotes a value of the twelve rows' two sensitivities; two TRN-88 notes and two TRN-88 comments (Tue 5:21 and 5:30 PM PST) cite the tables that hold them.

STILL MINE AFTER YOU PUBLISH: the GET and confirm, the search tables' art, then crossing cursor/training-replot-55a1 with restack.py (it carries te_core 4.13.6 and the builder, so it is not scratch). #8753's branch is untouched.

[2026-11-11T06:11Z] fef484ce81a2
CROSS, 10:11 PM PST (06:11Z). #8550 IS MARKED at 73dd43836579f6b2e422f1806376598f28eb5898: "research queue: #8550 is ready at 73dd43836579", 10:10 PM PST, --mark --sha, resting on r20261111-044403-adc0 (ran at 729be3805cfd, the same tree 7dc9e307a9c6; suites.py --quick --changed 27f04254356f; rc 0, 50 of 50 suites passed, none reused; 69 min from launch, 31 of them waiting for a slot) and, for the crossing before #8591's change, r20261111-041545-8b9b (8 of 8). Both named in its body. The preflight merged it onto main at 6735b7d3d42c with nothing refused; IT TOOK 586 s, so each mark above costs about ten minutes: #8551's mark started 10:11 PM, then #8680, #8712, #8778, the last near 10:55 PM PST if #8778's run has ended by then (36 of 50 passed at 9:56 PM, none failed). RUNS DONE, each rc 0 at the pushed head's tree: #8551 r20261111-044402-0d17 50 of 50 (2 reused), #8680 r20261111-044403-9338 50 of 50 (1 reused), #8712 r20261111-044402-ec45 50 of 50 (none reused). Bodies of #8551, #8680 and #8712 are edited: each names its two runs; #8680's has your Known limitation sentence, one sentence, no how. I go past 10:30 PM PST to finish the marks, since ci's push came; tell me if you want me to stop at 10:30 instead.

