[2026-11-11T21:24Z] 92a1a72e73e0
REPLOT to training, TRN-111 (a), Wed 11 Nov 1:27 PM PST (21:27Z). THE THREE COUNTS, size-scaling, from r20261111-201517-d96c (done rc 0 at 21:19:18Z, peak 29 GB of 64 declared).

RESULT. No slowdown falls in any of the three. The put moves 8 rows of 90: 4 rise (2 records), 4 change text only.

Built by the published builder (benchmarks/covert_witness/islands/ladder-v3/size_scaling.py build, sha256 ae655723…, te_core tree 7de182a661 at TE commit 696528f7d1, VERSION 4.13.6), twice in one run, same stored steps (steps.tar 79d3b0c3…; no new trace): once on the published search tables, once on the corrected. Each: 90 rows, note 486 characters, check 226 rows at console and grown, 0 mismatched, 0 reached not yet traced, 0 unmeasured. Bytes 62,474 (published tables, sha256 123c4c75…) and 62,277 (corrected, fb57849e…). RAN by me.

Served: GET by me 21:10:36Z, updated_at 2026-11-04T18:06:50Z, sha256 641fd4cb… of console.body, 63,465 bytes, 90 rows. The column te_core is left out of the counts (4.13.3 served, 4.13.6 built: it differs on every row that has one); numbers are compared as numbers.

1. SERVED AGAINST THE PUBLISHED-TABLES BUILD (already behind). Falls 0. Rises 0. 4 rows differ, in rests_on text alone, all four the serving policy's attack row: 36M gain 1.1, 36M gain 2, 81M gain 1.1, 238M gain 1.1. The counts in the text change and none says "not yet traced" any more: 36M gain 1.1 "60 of the 89 reached (40 partial, 37 unmeasured, 12 unmeasured: not yet traced)" becomes "73 of the 93 reached (51 unmeasured, 42 partial)"; 36M gain 2 "10 of the 15" becomes "13 of the 15"; 81M gain 1.1 "16 of the 25" becomes "17 of the 25"; 238M gain 1.1 "1 of the 2" becomes "2 of the 2". 86 more rows differ in te_core alone. Source differs (4.13.3 and art:e1c50d24… become 4.13.6 and art:4ba75358…).

2. PUBLISHED-TABLES BUILD AGAINST THE CORRECTED (the restating). Falls 0. Rises 4, two records, each at console and grown:
   - 81M gain 2, hybrid/grow-depth, r20261104-060904-d64c@7326: console rho = S 440,500 to 653,600 (+48.377%, the largest); grown 825,900 to 1,225,000 (+48.323%).
   - 36M gain 11, hybrid/grow-wide, r20261103-121552-4551@49926: console 465,800 to 595,500 (+27.845%); grown 404,200 to 516,700 (+27.833%).
   Each of the 4 also changes rho_products, products_inputs (subset "the weight gradients alone" becomes "M", F 1.0 becomes 1.155) and rests_on. Their serving rows stay "none with a full rho". No other row differs; source differs in the art alone.

3. SERVED AGAINST THE CORRECTED BUILD (what the put moves). Falls 0. Rises 4 (those of 2). 4 more rows differ in rests_on alone (those of 1). 8 rows of 90; 82 more differ in te_core alone.

238M GAIN 1.1, CONFIRMED from the build's output. Its search table has 4 reached rows there. Least p*rho at console in both builds: r20261103-132106-82a6@1804, method ref-optim/fullmuon-fe-wsd (rho_console 392,202 published, 659,625 corrected, p 1). r20261103-095000-6998@2652 is ref-optim/fullmuon-wsd too. The builder drops both (STANDARD = "ref-optim/"), and serves the least of the rest, r20261104-021052-b5a7@2017, ref-data/fullmuon-fe-wsd: p 1.125, console rho 665,100 S 748,400, grown rho 3,530,000 S 3,972,000, the same cells in served, published-tables and corrected builds. Were ref-optim not dropped, 82a6 would be the row in both. READ by me from out/build-{fixed,corr}/search and size-scaling.

THE SOURCE'S ART, your ruling's check: PASSES. The build's own search tables equal art:7b1ab651… tables/corr/search in every cell (algorithm-policy 1,497 rows, frontier-fit 46 rows; coverage, frontier-points and refused byte-equal), and the published-tables build's equal tables/fixed/search likewise. So the corrected body's source names art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b. One thing for you: that art holds three sets (published, fixed, corr) and the source is 294 of 300 characters, so it cannot say tables/corr/search.

COMMANDS (RAN by me, VM, scripts in the staged artifact): samecells.py on each table pair; bodycmp.py BASE VARIANT --skip te_core, three times. Logs: a-d96c-search-tables.log, a-d96c-three-counts.log.

RULING 4, early: the put branch's builds are both done (c982 at 21:14Z, standard-workloads r20261111-211303-114f at 21:17Z, rc 0, peak 10.4 GB of 13). Against the bodies built at 7de182a661, every cell of all four panels is equal. What differs, character by character, in my next message within the hour: updated_at on all four; algorithm-policy's source (two build times); standard-workloads' source (947e4be33c, on cursor/training-put-55a1) and its note's run id.

NEXT I OWE: ruling 4's message by 2:30 PM PST today; (b) to (e) by Thu 12 Nov noon PST. No run of mine is running.

