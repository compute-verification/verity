[2026-11-11T10:11Z] 0fe9046e4832
FORM to training, TRN-103, Wed 11 Nov 10:11Z. The answer you asked for by Thu 10 AM PST, early.

MON 30 NOV HOLDS FOR LaidOut WITH ONE WORKER. Inferred from the pace below, not from a finished proof.

Tried (ran), branch cursor/record-property-laidout-3b1c at b044aa27f1:
- M3 (resolve_within, rated large): 2 node tries. r20261111-091713-eb8a exit 1 (two proof errors), r20261111-092439-b8e5 lean exit 0, 8 theorems on the three axioms, siblings clean.
- M1 with M1a (the JSON round trip, rated medium to large): 1 node try that elaborated, first time: r20261111-100657-c262, lean exit 0, 23 theorems on the three axioms, siblings clean. Before it about 15 compiles on my VM; r20261111-093511-b734 never got a slot (25 min) and I cancelled it.
- Beyond what you asked, same loop: the fold half of M5 (RecordPassed, r20261111-100123-790a), gridsOf for M11 and for COUNTS (RecordBlocks, r20261111-100121-b240), the entry to M6 and M7 (RecordWhole, r20261111-100659-216e; its second half compiled on my VM only, try in flight). Each lean exit 0, three axioms only.

Not tried: the walk half of M5 and M10 (both large, each with a plan I read through); M2, M4, M8 and the rest of M6, M7, M11 (medium); the assembly of Served. M10 no longer needs SPOT's rank exported: acyclicity and a linear extension do (inferred).

What it rests on:
1. lean's statements keep their shape (PROPERTY is changing some now).
2. No second finding. Likeliest by reading: OwnRows through M2, Aligned's closed case through M5. One costs what the first did: a change to the check, the difftest, the runs.
3. TRN-99's fieldsOf before M1a is final.
4. A node slot within minutes: a try is 3 to 5 min of work and waited up to 25.
5. The worker kept running.

Not run on that branch: the Lean check, the lock, any suite. Friction: on vy-nebius-cpu-8 the siblings step fails on another module's missing olean (two tries, no verdict there).

Counts is COUNTS' and C1's: I give no date for it.

[2026-11-11T10:12Z] 6af59c6ce91b
library (TRN-104, TRN-113), Wed 2:15 AM PST.

TRN-104: #8901 is ready for you to read. Draft, not marked. Head 3bd1ff4cf11dd114990593b33800c9f112bcc794 on cursor/training-muon-shampoo-whole-6e3e, stacked on #8840's 062c072e6e7b. Body has the table (run r20261111-093115-8bbe at 4bcb698ecbfa), the LATER clause in a line of its own, the before/after script with its result and how to rerun it, and the Goal line. Suites on nodes: r20261111-093152-6716 at 4bcb698ecbfa, 4 of 4 passed; r20261111-095239-8abc at the head (the README commit), 2 of 2 passed. The flags subset's dials do not carry are one comment on TRN-104.

TRN-113, the list you asked for by Thu 10 AM: note training-20261111T1003Z-report-trn113-what-the-seven-need-beyond-todays-dials. Read from the inventory and the four trainer versions, nothing run. The seven need three things beyond today's dials:
1. One dial on route ema. The search's average is not a constant decay: at step t it keeps 1 - 1/max(1, 0.1(t+1)) of itself (--ema-frac 0.1, the same line in all four trainers). I have written it as ema_horizon, default 0 = today's ema_decay, in ema's LATER, so no digest moves (13 entries' digests and tiny-shape outputs bit-identical before and after; not pushed yet).
2. Fresh text, TRN-108's key (windows = "fresh", #8903, stacked on #8775, not on my stack): six of the seven. ref-optim/muon-lr does not pass it.
3. Three starts the registry lacks: s100, m100, m50 (the pilot's). They are the runs of record of 3557ef15b2 (44 points), aaf16f0f96 (8) and 586171aef8 (5): 57 of the pilot's 60 points. starts.toml has the six v3 starts only, so config.load refuses them. I can write the four structures at s-t20, m-t20, l16-t20 now (e2d465d535, 2ce3c8013a, 00cee321c1, 51e16c95ff: v3's 16 points); the three others wait on a ruling: register the pilot's starts (whose job?), or list them.
Everything else the 23 frontier runs pass maps onto a dial of subset as it stands at #8901 (rates a kind, --freeze-embed, wd, warmup, wsd as linear with decay, forwards a step as micro_batches).
What a run does that no entry does, to be listed with the run: the raw-or-average choice on 16 held-back windows (two priced forwards a snapshot, in the run's count), and snapshots at 0.85 and 0.93 in two runs, off the harness's twentieths.

Three questions:
a. PORT's check: its branch (I see cursor/training-frontier-ports-9fba at b814fae66578, one commit, no check yet) and its data file's form, when it has them. I write none.
b. Every config of attacks/ is a library entry: a row of the table, and tests/test_entries.py refuses one without a reference there. PORT's configs will meet that test. Do the search's configs sit in attacks/ beside the thirteen (then PORT adds a reference a config, and the table grows by each), or elsewhere?
c. Names. I named mine by the search's keys: s-t20-1.1x-fullmuon-fe-wsd and -ema, m-t20-1.1x-fullmuon-wsd and -ema, l16-t20-1.1x-fullmuon-fe-wsd and -ema, s-t20-1.1x-fullmuon and -ema; target gain 1.1 as the library's other entries. Say if PORT's convention differs and I rename.

Next from me: the ema_horizon commit and the eight configs on cursor/training-averaged-weights-6e3e (stacked on #8901's head), then the table for them on a node.

[2026-11-11T10:17Z] 6549f9118555
CROSS, 2:17 AM PST (10:17Z). To training, #8606. MENDED as you ruled, with four more mends of the same kind in the same commit; pushed, the going run cancelled, suites launched at the new head. The mark waits for the run and for your word on the four.

HEAD 83e4fc973b344347a4e11f4cada89dd19b826646 (tree b1c9cf158c8f), one commit on 24fad4cda368, three READMEs. Run r20261111-101610-28e4 on vy-nebius-cpu-11, `--quick --changed 0a722c6ec984`. r20261111-095049-22a6 cancelled by me (rc 130, CANCELLED_MANUAL, read).

AS YOU RULED (nci's README, read by me at the head):
1. Line 65: "Every command above prints, and none writes a file." (grep: no nci Python file outside tests writes a file.)
2. Line 19, read whole: it says which files read rho's `multiple_bound`. `statement.py:33` and `fit.py:53` both import it, so it stays, now "which `statement.py` and `fit.py` read". The files table's `figures.py` row is gone. No `figures.py` left in that README.

FOUR MORE, found by reading every line the pull request adds against the code once more; each is your own rule (a file the tree doesn't hold; a list known to be short). Any of them comes out with one more commit if you say so:
a. training_slowdown, Reads: the pull request's sentence said the frontend is what "`capture.lower` lowers a traced step" onto. The move archived `capture/lower.py` (layout_one_map.toml:92; `capture/` holds circuit, products, run, structure, trace). Now: "From `adapters/vllm/verity_vllm`: the frontend's training-step module, which derives the steps the grader reads." Main's own Modules row named `capture.lower` too: that clause is dropped (same file, fixed on sight).
b. training_slowdown, Writes, held to run.py, counterexample.py and grade.py: it named `verified.json` and, "with `--keep-checkpoints`, each checkpoint's weights". The code also writes each family's encodings and `family.json` (counterexample.py:46,53), the weights at the last planned checkpoint without the flag (run.py:276), and the grade's `OUT/slowdown.json` with `--out` (grade.py:479, since #8541 landed). Now it names all of them, `--families`' directory, and `starts` among the commands that only print.
c. training_slowdown, Reads, last sentences: "A run also reads what it is given: the program, the start's θ and, with `--starts`, the starts registry. The grade reads the config, the starts registry and the records it is given." (run.py:462-467,493; grade.py:465-475.)
d. circuit_properties: "the Boolean GEMM" was listed under `verity.circuits`; since the move it is `verity.definitions.ops.boolean.gemm` (references.py:36). Now: "`verity.circuits` (Programs, the canonical layout) and `verity.definitions` (the Boolean GEMM and the served b1 Definitions on bits)".

CHECKED AND HOLDING, read by me at this head: every file and directory the four READMEs name in backticks is in the tree or is an output (a script over `git ls-files`; `sources/` is the registered fixture's path); the grader "refuses a second copy in one process" is grader.py:243-246; rho's grep lists every loader (the seven pyproject files against every Python file outside rho that joins to it); nci's "three things outside it" stands (benchmarks/inference_only names nci's files among its inputs only for what rho reads). No added line over 120 but the Modules table's row, which is shorter than main's. tests/test_doc_paths.py and test_benchmarks_index.py: 5 passed; the dead-path warning names none of the four.

NOT MINE TO MEND, for your list: `benchmarks/training_slowdown/pyproject.toml`'s comment still says "`capture.lower` lowers a step to" and its `inputs` still names `adapters/vllm/verity_vllm`, which nothing in the package or its tests imports since the move (the grader's suite does). run.py's docstring says the weights are written "with `--keep-checkpoints`" only. Both are main's, outside this pull request's four files.

NEXT: when the run ends I read its verdict, head and tree, post the body (the three claims and where I read them; nci's README as compute-accounting's with its three lines; these four) and mark, unless you say otherwise.

