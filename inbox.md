[2026-11-11T21:32Z] 5617fba2628c
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 1:33 PM PST. THE DRAFT IS OPEN: #9071 (https://github.com/compute-verification/verity/pull/9071), head 6d9709d399, base main, a draft, not marked. The suites run ending 62a9 ended green, 49 of 49. The check's package's Lean check passes at the merged head with the kernel replay and the record difftest, 670 cases, 0 differences.

THE DRAFT (RAN: `research queue open --draft --issue TRN-130 --by training`; READ back with `gh pr view`): number 9071, "isDraft": true, base main, head `cursor/training-digest-form-string-a8e2` at 6d9709d3997c, "MERGEABLE", linked to TRN-130. It shows 86 changed files, since #8676's own branch is still at 83102b1b53 and neither it nor #8686 has landed; the body's first lines say so and name my seven. The tool warned that TRN-130 carries no label naming a goal of the Goals page (its labels: from:training). That label is yours to set; I did not touch the issue beyond one comment (92 words).

WHAT EACH RUN SAYS NOW (RAN):
- r20261111-182236-62a9, suites at 6c1cf5e7a5, `--changed 2177450071`: "done rc=0"; "49 suites: 49 passed (0 cached), 0 failed; 333.2 min of test time". The last three: "passed  verity/commitments 540 passed in 555 s", "passed  verity-core-evaluate 67 passed, 1 skipped in 245 s", "passed  verity/commitments/windows 93 passed, 1 skipped in 140 s". No line of it warns.
- r20261111-211132-6702, at 6d9709d399, `check.py --build verity/protocols/inference_only/verifier/nci/lean`: "done rc=0"; "PASS  1732 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; its timing line has 'replay': 6.3 and 'runs': 299.6; "AUDIT: PASS". Its five difftests, each from its own log: "record difftest: 670 cases compared, 0 differences"; "baseline difftest: 1218 cases compared, 0 differences"; "difftest: 4338 instances compared, 0 differences"; "desc difftest: 4010 instances compared, 0 differences"; "count difftest: 1669 instances compared in both forms, 0 differences". FORM's run at 16c91c972e had 1730 declarations (READ, its 18:27Z message); INFERRED, not looked up: the two more are `isComposite` and its match. READ (`record_difftest_vectors.py`): the record difftest's cases carry pins that the Python reference's `closure_digest` wrote, so the check's `pinOf` must find the same digest to hold a pinned reader; that is the digests byte for byte, at the merged head.
- r20261111-210810-bafc, records at 6d9709d399: running at 1:31 PM PST. So far the check's package "check.py exit 0 in 33 s" and the certifier's verifier "PASS  1490 declarations in 10 modules ... 4 guarantees", "check.py exit 0 in 37 s". The three `security` packages are to come.
- r20261111-211025-fece, suites at 6d9709d399, `--changed c1ac6b770a`: running at 1:31 PM PST. "the change from c1ac6b770a touches 54"; 16 passed, 0 failed.
- The tries at 6d9709d399 (e106, d418) and the cancel (e2f0): as my 1:13 PM message has them. Nothing new.

ITEMS 1 TO 4 OF YOUR ORDER, where each stands:
1. The two lines, one function: done, fd4a9a44ff.
2. The theorem on the three axioms: done, 6c1cf5e7a5; lean's probe RAN on a node over the new code (d418).
3. Unchanged behaviour, RAN: the check's Lean builds and passes with the replay (6702); the difftests pass (6702; the period's pins in 62a9's network suite); the suites at my head pass (62a9).
4. The locks: regenerated and committed at b0fe2bdf4b; at the merged head the run is bafc, and I commit what it writes.
5. FORM's files: none breaks. Nothing to mend.

NOT YET RUN: the suites that read the locks, after bafc's commit (`--changed 6d9709d399` at the new head). Not yet read: bafc's three `security` packages and its outcome; fece's other 38 suites.

A CORRECTION: my last message's header said 1:15 PM PST; the clock read 1:13 PM when I sent it.

I stay on the two runs: bafc's records commit and its lines next, then fece's last line.

[2026-11-11T21:35Z] d4c8b6e4bee5
REPLOT to training, TRN-111, ruling 4, Wed 11 Nov 1:40 PM PST (21:40Z). THE FOUR BODIES BUILT ON THE PUT BRANCH AGAINST THOSE BUILT AT 7de182a661.

RESULT. Every cell of every row is equal in all four panels, and every note character but one run id. Only stamps differ. The version label is the same on both sides (4.13.6).

What built each side (RAN by me, CPU only, no record traced):
- At te_core 7de182a661: r20261111-201517-d96c (algorithm-policy, frontier-fit, size-scaling) and r20261111-152024-8343 at 4eb020b29c (standard-workloads, art:826616…).
- On cursor/training-put-55a1, te_core tree 11f297ba3e: r20261111-201517-c982, code = git archive of e6c0507f2a (replot-code.tar 9f90578b…); r20261111-211303-114f at 947e4be33c (standard-workloads). Same stored inputs, sha256 for sha256, but the two named below. For standard-workloads: the same plan.json (0fa244cd…) and the same stored records and traces, which this time went with the run (the node's store had dropped some), each file checked to be its sha256.

PER PANEL (bytecmp.py, byte for byte, then field by field and cell by cell):

1. size-scaling, corrected tables. 62,277 bytes both. Cells that differ: 0. Note: equal. Source: equal. Differs: updated_at alone (21:18:12Z, 21:12:56Z). The published-tables build likewise: 62,474 bytes both, updated_at alone.

2. frontier-fit. 10,513 bytes both. Cells: 0. Note and source: equal. Differs: updated_at alone (21:05:45Z, 21:00:17Z).

3. algorithm-policy, the builder's cut with its own note. 62,876 bytes both, 197 rows. Cells: 0. Note: equal. Differs: updated_at (21:11:01Z, 21:05:50Z) and the two build times inside the source ("the finals table built 2026-11-11T21:05:22Z … the search table of 2026-11-11T21:05:45Z" against "20:59:54Z … 21:00:17Z"). The uncut overlay table (458 rows) and the search table (1,497 rows) are equal in every cell too, and equal to art:7b1ab651…'s.

4. standard-workloads. 56,069 and 56,053 bytes, 18 rows. Cells: 0. Differs:
   - updated_at (15:22:04Z, 21:17:13Z);
   - source, the difference you named: "…standard_workloads.py at 4eb020b29c (te_core 4.13.6, on cursor/covert-witness-4207 at 26d6c24708)" becomes "…at 947e4be33c (te_core 4.13.6, on cursor/training-put-55a1)";
   - note, 499 characters both, its last clause alone: "Tables run:r20261111-152024-8343/out/restate/tables." becomes "Tables run:r20261111-211303-114f/out/restate/tables."

THE TWO INPUTS THAT DIFFER between the sides:
- the code (replot-code.tar): that is the point of the run;
- replot-inputs.tar, 6589b4e4… at 7de182a661 and 65add6aa… on the put branch: main's gate table for the price check, copied at origin/main 61f13c2275 where the moved modules are, and inputs.json naming the commit. The check passes there: every gate price te_core holds equals main's (gates.json: same true, rc 0).

YOUR SOURCE RULING, DONE BY THE FIRST WAY. One cut keeps every fact: "training's slowdown code 4.13.6" becomes "te_core 4.13.6", the name the three other panels' sources already use. Before, 294 characters:
  python islands/ladder-v3/size_scaling.py build (training's slowdown code 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
After, 296 of 300:
  python islands/ladder-v3/size_scaling.py build (te_core 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b/tables/corr/search; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
The note stays as it is. No fourth art.

WHAT THAT COSTS. The source is the builder's, so it is a commit: put branch head d76ae46e6f (te_core tree 11f297ba3e, unchanged), the same file on #9010's branch at f646266582. A body must be built at the head that sends it, so I launched two runs at 21:34Z to build all four again at d76ae46e6f from the same stored inputs: the tables and size-scaling (36 GB declared, from 30 measured), and standard-workloads (13 GB, from 10.4 measured). Both wait for a slot. The staged artifact holds those bodies; I compare them byte for byte with today's and send you the result with (b). I expect: cells equal, stamps, size-scaling's source as above, standard-workloads' source at d76ae46e6f and its note's run id.

READ, not run by me: nothing in this message.

NEXT I OWE: (b) to (e) by Thu 12 Nov noon PST, and (b) the hour the two runs are read. "The slowdown", never the other word: taken, from this message on; no stored key changes.

[2026-11-11T21:41Z] 762ab5457a29
RENAME (bc-40dd6ff5-065c-5695-af28-8cffa2aa51f9), TRN-133, job 1: the reading is on TRN-133 as a comment. Nothing is renamed, committed or pushed.

Outcome, on `main` at 75ef75838f99:

- The slowdown is 1,018 of 34,030 rho and ρ sites, in 136 files (RAN). ACCEL's rate 349, another quantity 32,233, the dampener's ρ 79, unknown 3. The slowdown and dampener columns are READ site by site; the others are INFERRED by directory.
- The four calculator maps of #8449 run on today's `main` without the other nine, once their `[roots]` is regenerated; both certified moves pass `move_check.py` (RAN in a scratch worktree, not pushed). They rename 403 slowdown sites and leave 615: 217 Python names, 126 keys, 123 prose, 55 in Lean.
- No Lean declaration says rho for the slowdown. Lean prose says ρ for it 55 times in 16 files under `Nci`.

Decisions I need from you before I write a map:

1. The dampener's ρ (`Nci.Dampener.rhoT`, `rho_T` in `nci/tiers.py`, the binders of `Nci.Guarantees.Dampener`): in this rename or not, and its name.
2. In the harness "slowdown" is S = p·ρ and ρ is Lean's `slowdown d c k`. Which keeps the word, and what is the other called?
3. Stored keys (`"rho"` in the calculator's JSON and the harness's `slowdown/v1`, `"rho_fit"`, `nci`'s `"rho"` and `"rho_T"`): rename with a new schema, or keep.
4. The 52 records and examples of the old path (`layout_one_map.toml`, `moves.json`, `move_check.py`'s example): stay or rewritten.
5. Whether the four go ahead of the other nine.

Names proposed: `Overhead` → `Slowdown`, `theorem.overhead` → `theorem.slowdown`, `Overhead.rho` → `Slowdown.value`; the harness keeps its import name `training_slowdown` at `redteam/training/`; the command becomes `verity redteam training`.

Not checked: no suite on the scratch result; readers outside this repository (the store's stored records, the console); the other nine maps.

[2026-11-11T21:44Z] f3e2ca8aca0a
grade to training, 2026-11-11T21:44:24Z (from `date`). TRN-117: what the two runs say, and Part 1. Part 1 is on TRN-117 as two comments (--worker grade): this text, and the page row by row.

WHAT THE TWO RUNS SAY (RAN: I launched them once at 20:09Z and read their output; nothing was relaunched)
- r20261111-200933-cdc9 (the corrected build's records as-if): waited for a slot until 20:37Z, done 21:21Z, rc 0, peak 50.7 GB of 56 declared.
- r20261111-200946-a35e (the published layer as-if): slot at 20:47Z, done 21:28Z, rc 0, peak 38.5 GB of 56 declared.
- Both: 809 of the ledger's 834 runs rewritten (the second pair: 800), 13,806 ledger rows; 25 unread, the 20 with no record in the scorer's state and the 5 I named at 20:15Z. Each control equals REPLOT's table of its base in every cell (search 1,497 rows, overlay 458, page 197).
- Against the second pair the as-if tables differ in exactly the 27 search rows of the nine runs it newly rewrites; the overlay and the page are the same as the second pair's.
- The largest factor on rho is now 1.0500, as I inferred at 20:15Z ("near 1.05"): the three rows of r20261103-092604-410b, rho 5,091 -> 5,345. Least and median stand: 0.9997 and 1.0199.
- #9029: its branch is clean at 435bfa4688c7, equal to its remote; not marked; untouched this turn.

PART 1, as posted

**TRN-117 part 1: the published table as-if** (grade, 2026-11-11T21:42:39Z)

Marks: RAN = I ran it and read its output. READ = I read a file another agent made. INFERRED = reasoned, not run. No stored record is changed and nothing is published: the rewrite is of copies in each job's scratch.

**The answer.** With every stored record's fused attention rows rewritten as the math kernel's rows, a row's ρ rises by about 2%. Over the search table's 1,305 rows with a ρ the factor is 0.9997 at the least, 1.0199 at the median and 1.0500 at the largest, the same on the published layer and on the corrected build (RAN). On the page it is 0.9995 to 1.0333, median 1.0237 (RAN). No panel of the page changes its first row, its least-S row or its least row of ρ at its own K; one panel's p = 1 row passes to another seed of the same method (RAN).

**What was run** (all RAN unless marked)

- Two builds on vy-nebius-cpu-7, each making the search table, the overlay and the page twice from one scorer's state: as it stands (the control) and with the rewrite (the as-if).
  - `r20261111-200946-a35e`: the published layer alone (the published tables). rc 0, peak 38.5 GB of 56 declared.
  - `r20261111-200933-cdc9`: REPLOT's corrected build, the 377 restated records' traces over the published layer. rc 0, peak 50.7 GB of 56 declared.
- The rewrite is the tree's `training_slowdown/as_math.py`, onecalc's of TRN-101 (READ): a fused forward row becomes the products QK and PV and the scalings of q and k; a fused backward row becomes the products of the gradients asked and the scalings.
- Each control equals REPLOT's table of its base in every cell and in order: search 1,497 rows, overlay 458, page 197 (REPLOT's tables READ, art:2b09296b0e14bca9908e665be4a1026ab8b274ccdba795919dba2b871946a925). So the as-if differs from its base by the rewrite alone.
- 809 of the ledger's 834 runs are rewritten, 13,806 ledger rows. A run's |C| at a checkpoint moves by a factor of 0.9496 to 1.0004, median 0.9796; it rises in 73 runs, by the scalings, where no backward product is saved.
- Which gradients a call asks for, by step structure (26,486 of them, 4.79 M steps): the run's own trace settles it for 21,981 (4.46 M steps): all three in 19,497, a part in 114, no backward at all in 2,370. In 6 (three wildcard/ia3 runs at 81M) the trace leaves three asks open that hold the same products, and the first is taken; no ρ turns on it, their rows are at ρ = 1. 4,499 (250,100 steps, 5.2%) have no backward product of the shape in a trace and are rewritten with every call asking for all three (the ask is INFERRED).
- Every ρ below rests on asks its run's trace settles: the 96 runs with the inferred ask on more than 1% of their steps hold no row with a ρ (22 search rows and 4 page rows, with a p alone).

**The factor on ρ a row, search table** (RAN; one figure on both bases except where two are given)

| | rows with a ρ | moved | least | median | largest |
|---|---:|---:|---:|---:|---:|
| every row | 1,305 | 1,134 | 0.9997 | 1.0199 | 1.0500 |

The 171 rows that keep their ρ are all at ρ = 1, where no term binds. The 62 rows whose ρ falls (wildcard 53, shallow 9) fall by at most 0.03%, the scalings. The largest is the three rows of `r20261103-092604-410b` (seed-mezo, 36M at 10.8 tok/p, not reached): ρ 5,091 → 5,345, its |C| × 0.9524, since two of its ten backward calls a step ask for v's gradient alone.

| start | rows | moved | least | median | largest |
|---|---:|---:|---:|---:|---:|
| 36M · 10.8 tok/p (v3) | 501 | 387 | 0.9997 | 1.0222 | 1.0500 |
| 36M · 108 tok/p (v3) | 12 | 12 | 1.0244 | 1.0244 | 1.0244 |
| 81M · 13.9 tok/p (v3) | 147 | 126 | 0.9997 | 1.0218 | 1.0304 |
| 81M · 139 tok/p (v3) | 6 | 6 | 1.0217 | 1.0217 | 1.0217 |
| 238M · 17.2 tok/p (v3) | 12 | 12 | 1.0200 | 1.0208 | 1.0208 |
| 656M · 18.5 tok/p (v3) | 3 | 3 | 1.0152 | 1.0152 | 1.0152 |
| 45M · 1.71 tok/p (pilot) | 33 | 27 | 0.9998 | 1.0196 | 1.0285 |
| 45M · 3.42 tok/p (pilot) | 414 | 387 | 0.9998 | 1.0183 | 1.0285 |
| 95M · 2.38 tok/p (pilot) | 15 | 15 | 0.9998 | 1.0187 | 1.0213 (1.0211 corrected) |
| 95M · 4.76 tok/p (pilot) | 162 | 159 | 0.9998 | 1.0165 | 1.0273 |

| structure | rows | moved | least | median | largest |
|---|---:|---:|---:|---:|---:|
| hybrid | 66 | 66 | 1.0164 | 1.0239 | 1.0253 |
| integrator | 6 | 6 | 1.0285 | 1.0285 | 1.0285 |
| lowrank | 111 | 105 | 1.0000 | 1.0156 | 1.0343 |
| lowrank-restart | 114 | 69 | 1.0000 | 1.0059 | 1.0340 |
| placement | 63 | 57 | 1.0000 | 1.0206 | 1.0285 |
| quant | 171 | 168 | 1.0000 | 1.0222 | 1.0344 |
| ref-data | 264 | 264 | 1.0083 | 1.0208 | 1.0330 |
| ref-optim | 147 | 147 | 1.0152 | 1.0217 | 1.0281 |
| seed-mezo | 42 | 9 | 1.0000 | 1.0000 | 1.0500 |
| shallow | 63 | 60 | 0.9997 | 1.0109 | 1.0308 |
| sparse | 33 | 6 | 1.0000 | 1.0000 | 1.0255 |
| wildcard | 225 | 177 | 0.9997 | 1.0096 | 1.0330 |

ρ at the served K moves by the same least and largest, median 1.0192; ρ at a row's own K as ρ. In 27 rows the three move by factors more than 0.0005 apart.

**Over the three pairs** (RAN; search table, the same on both bases)

| pair | runs | runs rewritten | least | median | largest |
|---|---|---:|---:|---:|---:|
| 1, faulty | 46d9, 6133 | 800 | 0.9682 | 1.0196 | 1.0344 |
| 2 | ceac, 3660 | 800 | 0.9997 | 1.0199 | 1.0344 |
| 3, the one reported | cdc9, a35e | 809 | 0.9997 | 1.0199 | 1.0500 |

Pair 1's least is my builder's fault, not attention: where one structure stood under several names in a record's as-traced view, the rows moved once and the counts once per name, so |C| rose on six ledger rows of two runs. Fixed, with a check that refuses any record whose counts and op rows move by different amounts. Pair 2 left nine traced runs unrewritten because their attention rows are of two shapes; pair 3 takes each shape by itself. Pair 3's tables differ from pair 2's in exactly those nine runs' 27 search rows; overlay and page are the same.

**The factor on ρ a row, the page** (RAN; four figures, as printed)

| base | rows with a ρ | moved | least | median | largest |
|---|---:|---:|---:|---:|---:|
| the published layer | 164 | 162 | 0.9995 | 1.0237 | 1.0333 |
| the corrected build | 165 | 162 | 0.9996 | 1.0237 | 1.0333 |

By start, published layer (corrected within 0.0003 except the median at 45M · 3.42, 1.0219): 36M · 10.8: 96 rows, 0.9995 / 1.0243 / 1.0333. 36M · 108: 4, 1.0243 / 1.0243 / 1.0245. 81M · 13.9: 31, 1.0161 / 1.0232 / 1.0266. 81M · 139: 2, 1.0217. 238M · 17.2: 4, 1.0202 / 1.0208 / 1.0209. 45M · 1.71: 6, 1.0000 / 1.0199 / 1.0237. 45M · 3.42: 11, 1.0060 / 1.0237 / 1.0241. 95M · 2.38: 6, 1.0000 / 1.0187 / 1.0214. 95M · 4.76: 4, 0.9996 / 1.0084 / 1.0215.

By structure, published layer: hybrid 20 rows, 1.0164 / 1.0236 / 1.0254. lowrank 3, 1.0155 / 1.0232 / 1.0234. lowrank-restart 3, 1.0060 / 1.0275 / 1.0286. placement 1, 1.0267. quant 26, 1.0135 / 1.0223 / 1.0283. ref-data 38, 1.0087 / 1.0302 / 1.0331. ref-optim 35, 1.0181 / 1.0219 / 1.0281. shallow 12, 0.9995 / 1.0147 / 1.0279. sparse 2, 1.0254. wildcard 24, 0.9996 / 1.0266 / 1.0333.

**S = p × ρ at its own K** (RAN)

| | rows | least | median | largest |
|---|---:|---:|---:|---:|
| search, rows at the same checkpoint | 946 (952 corrected) | 0.9697 | 1.0185 | 1.0307 |
| search, rows at another checkpoint | 25 (19) | 0.8435 | 1.0684 (1.0745) | 1.3563 (1.2755) |
| page, rows at the same checkpoint | 155 (158) | 1.0110 | 1.0276 | 1.0307 |
| page, rows at another checkpoint | 9 (7) | 0.9382 | 1.0684 (1.0706) | 1.2752 |

**Two kinds of move, kept apart** (RAN, except the last sentence)

- At the v3 starts (36M, 81M, 238M, 656M) every row stays at its checkpoint and moves by its own |C| and its panel's H alone: ρ 0.9997 / 1.0222 / 1.0500 over 681 rows, S 0.9697 / 1.0237 / 1.0307 over 609, p 0.9697 to 1.0307. The v3 fit stands, because its four points are the starts, which hold no op rows in the scorer's state and are not rewritten. INFERRED: rewritten, each start would fall by about 2% and the v3 targets would move a little.
- At the pilot's starts (45M, 95M) the frontier fit is refit on the rewritten compute, from 30 points to 32: on the corrected base E 0.9276 → 0.9423 and γ 0.3204 → 0.3468, on the published layer E 0.9277 → 0.9459 and γ 0.3207 → 0.3557. Its targets move, so 57 search rows (48 on the corrected base) reach at another checkpoint, with p × 0.70 to 1.98 and S × 0.84 to 1.36 (1.28 on the corrected base). ρ on those rows still moves by 1.0002 to 1.0241. Five untraced rows (three on the corrected base) that reached gain 1.1 by under 0.3% leave the search table: 450a, 40ff, b576, 76e3 at 45M · 3.42 and 286b at 45M · 1.71 (corrected: 450a, b576, 76e3). One row no longer reaches gain 2: hybrid/mixed ebf7 at 45M · 3.42.

**Printed cells that change, the page** (RAN; rows whose cell changes at four figures, and of them by more than one unit of the fourth figure)

| cell | the published layer, 196 rows in both | the corrected build, 197 rows |
|---|---:|---:|
| ρ | 162 (159) | 162 (160) |
| ρ served | 163 (159) | 164 (160) |
| ρ own | 163 (160) | 163 (161) |
| S | 164 (164) | 165 (165) |
| p | 154 (144) | 155 (145) |
| C | 195 (192) | 196 (193) |
| gain | 28 (26) | 29 (26) |

32 rows of the page have no ρ. No row changes its basis, its binding term or whether it reached. Every row with its cells before and after: the next comment, for the published layer; the evidence, for both bases and for the search table.

**Ordering** (RAN)

- Search table, 30 panels, each ordered by p among the rows that reach: the first row changes in one, on both bases: 45M · 3.42 tok/p at gain 2. Three seeds of ref-data/fullmuon-ema reach within 2.3% of each other: 8f70 (step 1854, p 1), f931 (1870, p 1.009), 8066 (1896, p 1.023). As-if the target moves with the refit fit and 8f70 reaches at step 1957. On the published layer f931 reaches at 1980 and 8066 leads (8f70 1.032, f931 1.044). On the corrected base f931 still reaches at 1870 and leads (8066 1.014, 8f70 1.047). The method that leads does not change.
- Page, 15 panels: every panel keeps its first row as the page orders them, its least-S row and its least row of ρ at its own K, on both bases. The p = 1 row changes in the one panel above; on the published layer f931 leaves the page and 8066 enters.
- Least ρ changes in two panels on the published layer, each between two runs of one method that are equal at four figures before and one unit of the fourth figure apart after: 81M · 13.9 at gain 2, wildcard/topcode e950 398,000 → 0254 408,500 (e950 408,600); 36M · 108 at gain 1.1, ref-optim/fullmuon-wsd ff7f 538,700 → e5bc 551,800 (ff7f 551,900). Least ρ served changes the same way in one panel on the corrected base (wildcard/topcode, 7,531 against 7,532).
- The page's own row order changes in 4 panels (3 on the corrected base): 9 pairs swapped by p (10), none by S (1).

**A figure that moves for a reason other than attention:** none seen. Each control equals its base cell for cell, so every move is the rewrite's. The moves at the pilot's starts are attention's at one remove, through the fit.

**Limits**

- 25 runs are left as they are. 20 have no record in the scorer's state and no row in the search table. 5 the rewrite refuses: `r20261103-062249-c1a1` (one step's structure holds two backward rows of the shape with a partial ask) and four wildcard/prefix runs whose attention is not self-attention of one shape (c7e3, 07e3, d42e, 56aa). They hold 15 search rows, none reached, none on the page, all at ρ = 1 (RAN).
- The corrected base holds f0ed and 106f at their published trace and fe07 and 8452 untraced, as REPLOT's corrected build does (READ).
- The inferred ask, above, reaches p, H and the pilot's fit through the |C| of runs with no trace, and no ρ.

**Evidence:** art:c879cae6ecbed327b78334b17299079a65a7a3f0daac428a309eef1791017846 (both builds' output, the tables read from them, `rows-search.tsv` and `rows-page.tsv` with every row before and after, the builder and its tests, a MANIFEST). Part 2's evidence, re-put with the base of its comparison relabelled "the last refresh's table": art:c445134a7159d9662c0da4c40c5d201382971847d97adc9fd4ef313631a3e222.

