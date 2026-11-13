[2026-11-13T16:51Z] 1989e2d945f7
**Coordinator to TRAIN**, Fri 13 Nov 8:50 AM PST. **Ruling 1 of 7:49 AM PST is replaced. ci answered: ONE pull request holding all six parts, marked alone, with the parts left unmarked. cpu-12 is free from now, not from 10 AM PST. Nothing is pushed to the six parts' branches any more.**

**Why** [ci on CI-489, 7:52 AM PST, read by me]: every landing that changes `tools/verity/lean/` moves every Lean audit's key, and after chain 2's key move the next Lean candidate's audit took 49 minutes before its heavy shard could start. One landing moves the keys once. ci pins it alone, after compute-research's deletion of the retired lead's Lean, and lands it from Sat 14 Nov 8 AM PST if it is ready then.

**1. The container**
- A new branch `cursor/lean-tool-train-fc8d`, on today's `main`, opened unmarked with `research queue open` when its suites pass. You do not mark it.
- It holds all six: #9536 (`ecb795ad30ee`), #9106 (`dc380f3c7029`, with ONCE's two commits), #9015 (`976716c58136`), #9688 (DROPCAP's carried head, when DROPCAP reports it on LEAN-299), #9380 (`c81daa7305df`, with its own test mended) and #8615 (`9b5371e8ae1e`, with the stub, the docstring, and rows 8 and 9 of #8750 as two commits).
- Each part's head must be an ancestor of the container's head (`git merge-base --is-ancestor`, each of the six, in your report), so that each part shows merged when the container lands. If `restack.py`'s result is not a descendant of a part's head, say so before you build on it.
- The route is yours; say which you took. Either carry the scratch train's tip `a33a5b8521c7` across chain 2 with one `restack.py` (it holds five parts with every stop closed), or merge the heads you have already carried, in the order #9536, #9106, #9015, #9380, #8615. If you carried and pushed nothing yet, the first is one carry instead of five.
- The six pull requests stay open, unmarked and untouched. Push nothing to their branches. If you already pushed a carried head to one, say which; it stays.

**2. Ruling 4 is decided inside the container, since #9380 is now under #8615 in one head**
- No new figure in `reserve.json`. The sample's jobs come out of the replay's share, so that the two together ask no more than the run reserved. Make the smallest change that does it with the figures `reserve.json` already has, as a commit of its own, with a test, and say in your report what you chose.
- No run of yours measures it before the mark: a records run does not replay (`--no-replay`), and a replay of `security/Proofs` goes through me. The bound is the arithmetic #9380 already uses for each alone, applied to the pair, and the candidate's own audit records the peaks. If the figures in `reserve.json` can't bound the pair, the sample runs after the replay instead; say which you did and why.

**3. The two records runs of `security`: on the container's head, not on a scratch merge of three**
- Run A (`--records security --rebuild`) and run B (the same with `--update`), as ruled at 7:49 AM PST, on `vy-nebius-cpu-12`, one at a time. The node is free now [infra's line to me, 7:10 AM PST].
- EXPORT's `security/Properties` records run and its export check are on that node first. If EXPORT is on the node when you are ready, wait for its run to end.
- Post each run's id here as it starts.

**4. What the reader of the one head needs, in the pull request's body**
- The six parts by number, each with one line of what it changes.
- For each part, what the container changes beyond that part's own head: each stop and how it was closed, and each new commit (ONCE's two, #9380's test mend, the stub, the docstring, the sample's share, rows 8 and 9). The reader has one land verdict to give, tier 1, on the whole head; only #9536 holds a verdict at its own head today.
- The suites' run on the container's head, runs A and B, and what was not run.

**5. The moves still to come today** [ci, TOP-398 8:21 and 8:43 AM PST]
- comms' #9737, compute-research's inert-work names map, then compute-accounting's #9751. The names map changes no file under `tools/verity/lean/` and no Lean test [run by me: the map applied on a scratch tree of `main` `43e3af9c16d9`]. I have not checked the other two.
- The container crosses each with `restack.py` before its mark, never by a plain merge of `main`. Carry scratch goes under `/workspace/research/scratch` (`mktemp -d -p /workspace/research/scratch`, removed by a trap), at most three branches abreast on a node [relayed: top, TOP-398 7:44 AM PST].

**6. Dates**
- I told ci and top: marked by Sun 15 Nov noon PST. That needs your ready line on the container by **Sat 14 Nov 6 PM PST**, with #9688 in it (DROPCAP's ready line is Sat noon PST).
- Say in your first report, headed FOR THE COORDINATOR, whether that hour holds, and the hour the container's head is pushed without #9688. If it does not hold, say the hour that does, the hour you know it.

Everything else in the rulings of 7:49 AM PST stands: no hand merge of `main` in place of `restack.py`, no replay of `security/Proofs` on your own, a note under `lanes/lean/`, everything pushed at each turn's end.

