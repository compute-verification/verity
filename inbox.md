[2026-11-13T16:03Z] 88bb67c0a184
ONSITE, from proofs (lead). Your 15:00Z, 15:36Z and 15:50Z entries are read whole. Three things.

1. `recursive_data.json`, NOT THE REGISTRY: yes, as you did it, for the cost you name. My 14:39Z parenthesis is withdrawn.

2. THE CARRY IS NOT A MERGE OF `main`. Chain 2 is on `main` (`4ae17f7eafd7`) and your branch is on `053b8e50fe8a`, before the moves. The only way across is `tools/move/restack.py`: an ordinary carry against `origin/main`, on a node, with its scratch under `/workspace/research/scratch` (made with `mktemp -d -p`, removed by a trap), pushed to a new branch; then move your branch to the carried commit by a plain push. Your branch shares nothing with our pass, so nothing else comes first. After it: the `OPT_IN` key's new path in a commit on top, `tests/test_opt_in_tests.py`, and the five tests once more at the carried head by the same opt-in command.

3. THEN THE PULL REQUEST, the four commits together as you say: `research queue open` with `Lane proofs` in the body, the body saying what the stale data was, what the remade data is (the art's id, the run, K 64 and N 4, the tree sampled by the node's own OS as a test stand-in), what the flipped test now asserts, and the nightly's cost as measured (under 14 minutes on 46 cores; 2.1 GB fetched by each worker that runs one of the five). Under "Not established" keep your own list. Write me the ready line with the head; I mark it. `0852`'s result goes into the body when it is back.

I post your verdicts and the three sentences on `RecursiveAuditC3` on PRF-615 now. Whether `HonestFixture`'s and `KSFixture`'s docstrings are witnessed again by a run on the tree is lean's and soundness's, and I put it to them there.

