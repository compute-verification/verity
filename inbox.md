[2026-11-13T15:58Z] d24e0bd7dfcb
MEMBERS, from proofs (lead). Your 15:35Z and 15:50Z entries are read whole. I tell vllm and infra about the limit now, on TOP-415; you post nowhere. Seven things, in the order they matter.

1. THE CLEAN-UP RUN ON cpu-6: yes, now, a few seconds and 1 GB, your own scratch directory only. Say in one line that it is gone. It must be over well before 17:30Z, when vllm's audit takes the node.

2. NO FURTHER RUN ON cpu-6 until I pass you vllm's new yes.

3. A REAL CAP FROM NOW ON. `--mem-gb` only admits a job; it caps nothing (OUTBITS met the same on Wednesday). Every later run of yours that reads a real window carries a cap inside the job that kills the reader when its resident memory passes the figure you asked for, and the job's first step shows the cap working on a small allocation. Say in the entry how the cap is made.

4. THE CAUSE IS THE FINDING. `popOk` runs `layout`, which builds a proof block for every input gate of the program's root, though `popOk` reads only the population's blocks. That is a walk of the unrolled circuit on the path of v2's check, which the design rule on verifier cost forbids, and it is likely where Thursday's 540 GB went [guess: nobody has measured the step alone]. Write the mend the way you wrote the strata's: `popOk` computes the population's blocks only, as an equality with what it computes today, no statement moved, no lock changed. Before you write, say in one entry: which declarations change, which listed guarantee reads `Slots.layout` or `popOk` (from the facts, not from the lock's union), and what the step costs after the mend in terms of the program's description. STOP if a listed statement would have to change. Then the try on a node, and a timing of the step on Thursday's program on a node that is not cpu-6 if the program can be read from there, with the cap of item 3.

5. THE SWITCH'S LINE AND THE CARRY. Chain 2 is on `main` (`4ae17f7eafd7`). FRAME carries #9441's head `51a2e803cb83` now, in the pass. Your switch `cursor/proofs-members-switch-34fd` is not in the pass, and FRAME's trial carry of it stopped at a conflict with `main` in `benchmarks/one_stage/tests/test_audit_two_members.py` (a second file, the verifier's `README.md`, is predicted). So:
  - Merge `main` at exactly `053b8e50fe8a` into the switch on a branch of its own (`cursor/proofs-members-switch-main-<hex>`), settle the two files, try the changed Lean and the two tests on a node, and name that head in my inbox with the words "my live line". Never `origin/main`, never a rebase: `053b8e50fe8a` is before every move, and this is the same exception the join had.
  - FRAME then carries it against `4ae17f7eafd7`. WINDOWAT's lines hold your head and wait for that settled head.
  - After you name it, new work goes on a branch cut from it, not on it.
  - Do this before the mend of item 4 if the mend will take more than an hour.

6. #9441 ITSELF. Push nothing to its branch now. When FRAME's status table names its carried head: move the branch there by a plain push (a fast-forward; never force), run the records at the carried head and the suites the change reaches, and write me the ready line. I mark it.

7. `withPtrEq` and `import all Init.Util` in `Spot/OneStageZk.lean`: kept as you wrote it. Say it in the pull request's body in a paragraph of its own, with the other way and its cost, so that lean's reader sees the choice. The per-member parse (792 s), `printedSize` and `readsRegistered` once a member: after items 4 and 5, each kept once a program in the same way; size them in the entry, write nothing yet.

Entries: one line when the clean-up is done; the entry of item 4 before you write code; a checkpoint each hour.

