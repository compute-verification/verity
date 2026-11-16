[2026-11-16T23:28Z] cec9ebad0973
model: claude-opus-5-5-high; subscriptions removed: none held

STEP to the training lead, Mon 16 Nov 3:28 PM PST. Item 4 now: the 8 older uses of T1/T2 in `benchmarks/training_slowdown` on `main` `220767e4865` [ran: `git grep -w T1 T2`; sentences read]. Older T1 is update hardness, whose β is the narrowest cut from the step's gradient side to its new persistent values; older T2 is update dependence, the deep use of activations that depend on the run's own persistent values. Each line gives the file and line, the sentence, and the mend I propose [inferred] for the tiny PR after the map is on `main`:

1. `training_slowdown/capture/__init__.py:4`, the docstring's last clause: "…and the narrowest cut between the step's gradient side and its new persistent values, T1's β (`structure`)."
   - Mend: drop ", T1's β". The clause already says what the cut is.
2. `training_slowdown/capture/structure.py:15`: "Terms are the training ontology's, with T2 read as deep use of activations that depend on the run's own persistent values, however the circuit is written."
   - Mend: delete the sentence. It states the older T2.
3. `training_slowdown/capture/structure.py:43`: "- **Cut** (`cut_bits`, T1's witness): the narrowest vertex cut, in bits, between the step's gradient-side computation and the new values of the dependent slots its forward reads…"
   - Mend: drop ", T1's witness".
4. `training_slowdown/counterexample.py:1`: "Counterexamples to update hardness (T1): a family shows that some steps' updated weights rebuild from θ in few bits."
   - Mend: drop " (T1)". The premise is named `update-hardness` in `premises.toml`.
5. `tests/benchmarks/training_slowdown/test_dependence.py:1`: "training_slowdown 4.0's definitions, on toy steps: T2 reads update dependence, …"
6. `tests/benchmarks/training_slowdown/test_dependence.py:2`: "…however the circuit is written, and T1's β is the narrowest cut the step's own circuit has (structure.py's docstring)."
   - Mend for 5 and 6, which are one docstring: "Update dependence and the step's narrowest cut on toy steps: dependence however the circuit is written, and the narrowest cut the step's own circuit has (structure.py's docstring)."
7. `tests/benchmarks/training_slowdown/test_dependence.py:32`: "# --- T2: dependence ---"
   - Mend: "# --- dependence ---".
8. `tests/benchmarks/training_slowdown/test_dependence.py:135`: "# --- T1: the cut ---"
   - Mend: "# --- the cut ---".

None of these is the page's κ(A→B) between a step's inference block and the rest, so none takes T1/T2 or κ in the new sense. Each drops the older label, and 2 is deleted. I start items 1 to 3 now.

[2026-11-16T23:38Z] b076f6731e67
RECORD to the training lead, 23:38Z.

- Compose step #10619: marked at e9fbc9833ba0446f1081d88db87ac379dfa52857 [ran].
  - r20261116-222008-48f0 done: 12 suites passed fresh at that head, among them benchmarks/one_stage (1697) and benchmarks/inference_only (89) [ran].
  - The mark's preflight erred collecting tests/benchmarks/one_stage/interfaces/*, files this PR doesn't touch; that nested suite passed on the node in #10556's run [inferred: the preflight's own tree].
  - The queue says admitted, waiting on a land verdict; no reader yet. GitHub still shows it as a draft (opened with --draft); the queue admits it all the same.
  - Proofs: it is theirs to read (served_verdicts.sh). The body's "For proofs" asks one thing: the step leaves the compose's exit as it is when assemble.py refuses, and says so in its ASSEMBLE line; say if a refusal should fail the compose. Please pass that to proofs; I post nowhere.
- #10538 landed on main at 22:17Z as b75bb1ed81b. Its reader was review-82, a land verdict at 20:49Z [ran: the PR head's store labels]. Its two notes, malformed policy or circuit JSON giving a traceback and the count failing after the writes, are #10619's two follow-ups.
- #10607: `verity lean try` passed on both files [ran]:
  - Proofs/Slowdown/Cut.lean: r20261116-232921-99fb done, lean exit 0, 73 theorems, no axiom beyond the three;
  - Proofs/InferenceOnly/PartitionCovers.lean: r20261116-232921-5e99 done, exit 0, 1 theorem, same.
  - The lock still listed the two deleted guarantees (Slowdown.SecurityProofs.CoveringBound and CoveringBoundTwoCosts), which the update refuses. So I hand-removed them from guarantees, unwitnessed and shape (frozen, inline); their three open premises, which no other guarantee takes (Slowdown.Covering.Covers, Slowdown.Assumptions.ParameterHard and ParameterHardAt); and the four deleted modules' reads records. That is commit 6b73e913a4a on cursor/training-delete-covering-rest-4207, pushed. PartitionCovers' entries are untouched.
  - Records update r20261116-233713-7261 running; I'll check PartitionCovers' record is unchanged. No mark.
- #10553 (7ef39626a6c5, r20261116-224141-40b7) and #10556 (r20261116-215836-155e) are still running. #10553's research suite passed whole this time. #10556's research suite has the two main-shared treelock reds.

[2026-11-16T23:38Z] 3efb0d704459
RECORD to the training lead, 23:38Z.

- Compose step #10619: marked at e9fbc9833ba0446f1081d88db87ac379dfa52857 [ran].
  - r20261116-222008-48f0 done: 12 suites passed fresh at that head, among them benchmarks/one_stage (1697) and benchmarks/inference_only (89) [ran].
  - The mark's preflight erred collecting tests/benchmarks/one_stage/interfaces/*, files this PR doesn't touch; that nested suite passed on the node in #10556's run [inferred: the preflight's own tree].
  - The queue says admitted, waiting on a land verdict; no reader yet. GitHub still shows it as a draft (opened with --draft); the queue admits it all the same.
  - Proofs: it is theirs to read (served_verdicts.sh). The body's "For proofs" asks one thing: the step leaves the compose's exit as it is when assemble.py refuses, and says so in its ASSEMBLE line; say if a refusal should fail the compose. Please pass that to proofs; I post nowhere.
- #10538 landed on main at 22:17Z as b75bb1ed81b. Its reader was review-82, a land verdict at 20:49Z [ran: the PR head's store labels]. Its two notes, malformed policy or circuit JSON giving a traceback and the count failing after the writes, are #10619's two follow-ups.
- #10607: `verity lean try` passed on both files [ran]:
  - Proofs/Slowdown/Cut.lean: r20261116-232921-99fb done, lean exit 0, 73 theorems, no axiom beyond the three;
  - Proofs/InferenceOnly/PartitionCovers.lean: r20261116-232921-5e99 done, exit 0, 1 theorem, same.
  - The lock still listed the two deleted guarantees (Slowdown.SecurityProofs.CoveringBound and CoveringBoundTwoCosts), which the update refuses. So I hand-removed them from guarantees, unwitnessed and shape (frozen, inline); their three open premises, which no other guarantee takes (Slowdown.Covering.Covers, Slowdown.Assumptions.ParameterHard and ParameterHardAt); and the four deleted modules' reads records. That is commit 6b73e913a4a on cursor/training-delete-covering-rest-4207, pushed. PartitionCovers' entries are untouched.
  - Records update r20261116-233713-7261 running; I'll check PartitionCovers' record is unchanged. No mark.
- #10553 (7ef39626a6c5, r20261116-224141-40b7) and #10556 (r20261116-215836-155e) are still running. #10553's research suite passed whole this time. #10556's research suite has the two main-shared treelock reds.

[2026-11-16T23:50Z] 40edf3ea8f11
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

