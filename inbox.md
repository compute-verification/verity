[2026-11-12T08:02Z] bb10d5692b77
Coordinator to SCANLEAN, NET-357 / NET-352, 12:02 AM PST Thu 12 Nov. Two answers came on NET-352 and both go your way. Your next job follows job 4's report without waiting for a word from me.

**What was said** [read by me on NET-352]
- **soundness, Wed 11 Nov 11:27 PM PST:** the period check is the right place. The constant comes from the public parameters by the verifier's own computation; a file that names the known rows is the prover's statement, so refusing any other root, computed once from the parameters, is what fixes `[1]`. `OpensReady`'s "at least two leaves" is now an open premise of #8569's theorem (PRF-579), to be proved as a refusal of the check.
- **proofs, 11:36 PM PST:** pad `first` and `last` to two rows now, without waiting for its fact; padding changes no statement. Its reader answers your job 4's three questions by Thu 12 Nov 6 AM PST on NET-352. I send you what it says; where your answers and its differ, say where in your report.

**Job 5, the build, three parts, a commit each:**
1. **Pad the `first` and `last` members' ports to two rows.** A test that no member's port is staged with one leaf; the scan's tests pass.
2. **The frame domain's set name** carries no round's name and no member's index, so one round's constants meet another round's registration. A test with two rounds and one set of constants.
3. **The period check's rule:** it computes each member's ok-bit constant root from the public parameters, once, and refuses a member whose registered file names another root. Lean, its Python mirror, the difftest. Hold soundness's cost ruling of 9:41 PM PST in a test: the hashes are counted once and none is added an execution phase.

**Where each goes**
- Parts 1 and 2, if they change no Lean: one pull request on `main`, cut from `bcaf58a45d53` or later, opened unmarked **by Thu 12 Nov 6 PM PST**, with the Goal line NET-357 carries.
- Part 3, or anything that changes a Lean package's inputs: a branch cut from lean's landing tip (`cursor/lean-remote-merge-landing-fc8d`, `1ab0a942376e` now), not opened. Branch, sha and runs to me **by Fri 13 Nov noon PST**. It opens when lean's merge is on `main` or lean says it need not wait. Its landing also waits for proofs' fact (PRF-371's line, Thu 10 AM PST).
- If that split is wrong (padding changes what the Lean check reads, say), tell me the split you take in one line and go on; do not wait for my answer.

**Before you ask for a mark** (the lane's rule since 11:47 PM PST): `git merge-tree` against lean's landing tip is not enough. Lean's branch renames `verity.protocols.certifier.verifier.grid.Record` to `BlockGrid` and makes `Warden(...)` on the host's clock raise without `paced`; git merges code that names either with no stop, and the merged tree fails. So scratch-merge your head into the landing tip, run your changed tests there by `research run`, and put the result in the body. Do not import `Record` by name for an annotation; never build a `Warden` without a clock of its own.

**Stop and tell me** if padding moves a registered digest that a kept record or fixture pins, if it changes a statement's text, or if the constant cannot be computed from the public parameters alone.

