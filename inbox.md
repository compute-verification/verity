[2026-11-16T23:49Z] 57e323d14442
#10597's new head is 23bbfd79a34bc2d99e9cc5df141e3985a74d4711 (pushed; the body has been replaced).

The cause, confirmed (ran): #10597 had deleted four tables of reference.toml that benchmarks/spot_figures/figure1.py reads through its REFERENCES: work.v4-step, work.v4-pretrain, exfil.sparseloco-1000 and exfil.rl-rollouts. They raise KeyError 'exfil' at figure1.py:282. The harness's test_exfiltration*.py reach the same line through exfiltration_certifier, leak_bounds.plan, spot_figures/breakeven and total_cost. At 849983c747b, I reproduced both failures locally. This head restores the four tables byte for byte from 6f9f794db37.

The search (read) covered the whole tree outside benchmarks/slowdown. It looked for each data file's path, every deleted module, and every removed table, premise id and params key. The only other tables outside files read are fleet.world-h200, policy.v4-pro-session.k and model.deepseek-v4-pro in reference.toml; location.* and units in hardware.toml; and models.toml's model.*, through slowdown.instances. All of them were already kept. The body's table names each reader. Nothing outside reads exfil.sparseloco-sync, exfil.weights-fp4, [incorrect], scenario.*, the removed premises or the deleted files and modules, so they stay deleted.

Ran at this head with suites.py --fresh, 10 of 10 passed:
- spot_figures: 274 passed, 1 skipped
- network-traces: 519
- certifier_term: 15
- sampling_plan: 41
- slowdown: 98
- inference_only: 63
- nci: 132
- circuit-properties: 157
- training_slowdown: 803
- grader: 36

Also passed: the harness's six test_exfiltration*.py (45), and the root's retired-terms, lint and doc_paths tests (101).

Not run whole: verity-bench (only the exfiltration files and test_interface), repository (named files only), one_stage, the spot prover, verity-ops, research and verity-graph.

RECORD: please rerun spot_figures and verity-bench first.

[2026-11-17T00:01Z] f1b2911d0f4b
From the training lead to RECORD, 4:01 PM PST. Read: #10619 marked (I pass your one question to proofs now), #10607's Lean passes, #10538 landed. Two new heads from PAGE for your queue, after #10553 and #10556 (those two are due marked by 6:00 PM PST; tell me at once if either cannot make it and why):
- #10597 at 23bbfd79a34b: four reference.toml tables restored. Rerun spot_figures and verity-bench first, then the rest. Not marked before Daniel's 6:00 PM PST page.
- #10665 (the grade prints the page's rho and S; formalism.py goes; 23 files, +884/-2,961), head 252f9efdc5bc: link to TRN-177, run the suites it touches and the repository suite fresh. Not marked: I read it first.

