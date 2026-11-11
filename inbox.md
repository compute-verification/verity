[2026-11-11T22:24Z] ef784618b201
REPLOT to training, TRN-111 (b) to (e), Wed 11 Nov 2:23 PM PST (22:23Z). THE PUT IS STAGED. NOTHING WAS PUT.

RESULT. One artifact holds the four bodies, the four served bodies and the comparisons:
art:55b35cfbffb2831c03393c7dc250d030dc019350ec081692dad80ada171ac53f (116 objects, 3.5 MB, preserved). From a clean checkout of cursor/training-put-55a1 at d76ae46e6f005334c2e4d8c952a85282fdf48d4d (te_core tree 11f297ba3e, the publishing branch's) each of the four dry puts prints "console.put would refuse: nothing". No change under te_core was needed.

(b) THE FOUR BODIES, as the put would send them (RAN by me: built in r20261111-213443-e454, rc 0, peak 30 GB of 36 declared, and r20261111-213443-46e2, rc 0, peak 10.1 GB of 13; both preserved; no trace made, the stored steps as they are):
- algorithm-policy: sha256 864688df47ed85f42f0a573c22ce7d72d2ee799e982ec583c327784bb0d564e5, 62,867 bytes, 197 rows, note 482 characters (your file's text; the sha256 is with a stand-in id of 64 zeros, the put's own differs by the id).
- frontier-fit: a1c878ff75814936a1371f9cb63b0e7ceab6506201ee3a23ef3c3b0f89fab5a4, 10,513 bytes, 46 rows, note 477.
- standard-workloads: 78a409b4509f6cfa0013e0df4afc5109329b6c9e85f4f8b134f8aba5b90e2d82, 56,053 bytes, 18 rows, note 499.
- size-scaling: d59b340eedd613cee29f6d0bc028ef8e1df57c446c6f388238502f7427d7c781, 62,279 bytes, 90 rows, note 486.

SERVED, one GET by me at 22:14:25Z (sha256 of console.body, updated_at):
- algorithm-policy 5cfa4f00f315e7d1…, 2026-11-11T05:39:00Z. frontier-fit 92cd7967a1cbe65a…, 2026-11-05T04:47:24Z. standard-workloads 0eba91c5d94f4bce…, 2026-11-05T07:28:18Z. size-scaling 641fd4cb6430bbfc…, 2026-11-04T18:06:50Z. Full digests in served/meta.json.

SERVED AGAINST STAGED, cell by cell, numbers as numbers (RAN by me, bodycmp.py; compare/served-against-staged/NAME.{txt,json,md}). In every panel the rows and their order are the served ones.
- algorithm-policy: 81 rows of 197 differ. A slowdown falls on 24 rows of 19 runs (0.02% to 0.23%), rises on 37 of 27 runs (0.03% to 72.1%), 7 rows have both; 27 rows differ in other cells alone (the start's compute 29 rows in all, gain 6). Note and source differ.
- frontier-fit: prints no slowdown. 35 rows of 46 differ, every one the pilot's: compute falls on 34 rows by 0.04% to 0.13%; the pilot fit's E, A and gamma fall by 0.016%, 1.098% and 0.078%; gain moves on 23 rows (-0.15% to +0.06%). No v3 row differs. Note equal; source differs in the label (4.13.3 to 4.13.6).
- standard-workloads: 2 rows of 18. Falls: standard training 20M, each of its three slowdowns by 0.11% (539,339 to 538,753 default and console; 467,967 to 467,458 own; 10,282.2 to 10,271 served price). LoRA 20M differs in two stored inputs alone. The label 4.13.4 to 4.13.6 on all 18. Note differs in the label and its tables' run; source names d76ae46e6f on cursor/training-put-55a1.
- size-scaling: 8 rows of 90 (4 rise in two records, 4 text alone), no fall; the label 4.13.3 to 4.13.6 on all 90; the 45 standard-training rows equal the served ones in every cell but the label. Note equal; source differs.
- Served against the build without the restating (same artifact): algorithm-policy 0 rows, frontier-fit 0, standard-workloads 0 but the label, size-scaling the 4 text rows. So what moves is the restating.

RULING 4, FINAL (the source's cut made a new head, so all four were rebuilt at it). The four bodies built at d76ae46e6f against those built at 7de182a661 (r20261111-201517-d96c, r20261111-152024-8343), and against the put branch's earlier builds (r20261111-201517-c982, r20261111-211303-114f): EVERY CELL EQUAL in all four, columns equal, rows equal. RAN by me, bytecmp.py; compare/byte-for-byte/. What differs, in full:
- all four: updated_at.
- algorithm-policy: the two build times in its source; and the note, which is your file's where the builder's cut has its own.
- standard-workloads: source "at 4eb020b29c (te_core 4.13.6, on cursor/covert-witness-4207 at 26d6c24708)" becomes "at d76ae46e6f (te_core 4.13.6, on cursor/training-put-55a1)"; the note's run r20261111-152024-8343 becomes r20261111-213443-46e2.
- size-scaling: the source, as below.
The three counts from the put branch's build are the ones you accepted (4 / 4 / 8 rows; no fall; 4 rises). Its search tables equal art:7b1ab651…'s tables/corr/search and tables/fixed/search in every cell.

THE SOURCE, your ruling: done the first way. Words cut, no fact lost, note unchanged, no fourth art. Both strings are in the artifact's size-scaling.source.txt:
- before (294): python islands/ladder-v3/size_scaling.py build (training's slowdown code 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
- after (296 of 300): python islands/ladder-v3/size_scaling.py build (te_core 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b/tables/corr/search; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
"training's slowdown code" became "te_core", the name the other three sources use.

(c) THE FOUR LINES. In benchmarks/covert_witness of a clean checkout of cursor/training-put-55a1, S the artifact fetched (research data fetch art:55b35cfb… --to S), NOTE your file with {art} where the stored tables' id goes (your draft is S/algorithm-policy.note.txt):
  python scorer/plot2_final.py put S/algorithm-policy --note NOTE
  python scorer/plot2_final.py put-fit S/bodies/frontier-fit.json
  python islands/standard-workloads/standard_workloads.py --put S/bodies/standard-workloads.json --served S/served/panels.json
  python islands/ladder-v3/size_scaling.py put S/bodies/size-scaling.json
Each takes --dry: stores nothing, PUTs nothing, prints "dry: PANEL: sha256 …, N bytes, N rows, note N characters; console.put would refuse: …; nothing stored, nothing PUT". RAN by me, all four with --dry, from the clean checkout on the artifact fetched back from the store: the four sha256 above, refuse nothing, the checkout clean after. From #9010's own branch the dry form prints console.put's refusal (9 files of te_core differ from 225e85e26e). console.put's refusals are untouched. Not run by anyone: a put without --dry (its store and PUT steps are covered by tests with a stand-in route; 61 pass at both heads, RAN by me).

(d) NOTES (RAN by me against the staged body; logs/d-note-checks.log).
- algorithm-policy, your draft: 482 characters with a 68-character id, of 500. Its two counts and two percentages match the staged body: falls 24 rows, largest 0.227%; rises 37, largest 72.133%; both 7. "9 of 15 cells" is the staged build's figure: tables/final.json has 15 cells, 9 on the search split, 6 final (the served note says the same).
- frontier-fit: the builder's note as it is, 477, equal to the served one. Draft clause: " Pilot rows corrected." makes 499 (" Corrected: 35 rows." makes 497).
- size-scaling: the builder's note as it is, 486, equal to the served one; it does not say when the steps were traced. Draft clause: " 4 corrected." makes 499 (" Corrected." makes 497).
- standard-workloads: 499 as built. No clause fits.
- FOR YOU, if you take a draft: the staged bodies carry the builders' notes as they are. frontier-fit's note is written under te_core (frontier_fit.py), so a clause there is your decision, not a mend; the other way is a --note on put-fit and on size_scaling.py put, which is a commit on the put branch, a new head, and the four bodies built again there (standard-workloads' source names the commit that built it). Default: no clause, nothing rebuilt.

(e) #9010's body has the section "What is put" (52,700 bytes in all): what built the bodies, the line on the stored steps, the one input that differs with both sha256 (replot-inputs.tar c1596d40… on the put branch, 6589b4e4… at 7de182a661; the gate check passes on both), each panel with its rows that move, falls first, the three counts with your paragraph for size-scaling, which way the source was done, and the four lines. "What is in the diff" and "What was run" name the put's forms and these runs. The slowdown is "the slowdown" throughout; stored keys are in backticks and unchanged.

NOT DONE, AND WHAT TO KEEP IN MIND FOR THURSDAY:
- Nothing is PUT, nothing goes to main, #9010 stays a draft at f646266582.
- cursor/training-put-55a1 must stay at d76ae46e6f until the put: its bodies' sources name that commit. I commit nothing more to it.
- The served bodies were read at 22:14:25Z; the comparisons are against that GET. standard-workloads' put refuses, by its own rule, if the route's panel is no longer the one in S/served/panels.json (someone PUT since).
- fe07 and 8452 print no slowdown and are left; te_core's files are as they were; the 414 cells are TRN-128's.
- The section's generator in the artifact (scripts/whatisput.py) is one wording pass older than the one that wrote the body; the figures are the same.

NEXT I OWE: nothing with a date before your read Thursday afternoon. No run of mine is running.

