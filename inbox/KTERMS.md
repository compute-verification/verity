# KTERMS's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 18:40Z: your brief: #9299 takes `main` (proofs' #9234 has landed), its four counts recounted, its two suites run. One short job

**Who you are.** KTERMS, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You finish a small pull request
of the worker KSTEADY (bc-f944c53d-53aa-581d-ab39-a46f095b1cde), who wrote it and left the steps.

**The pull request.** #9299, branch `cursor/kernels-wide-leaf-terms-7c1e`, head `e1847fc0db7f`, unmarked,
`Refs HASH-119`. #9077 (the wide-leaf option of `benchmarks/one_stage/server_rate.py`, on `main`) added
eleven uses of words GLOSSARY.md retires (`flock-verify`, `draw`) to four files, and the retired-terms
ratchet says "fix it forward". #9299 brings the four files' counts back. proofs' #9234 rewrote some of
the same lines and landed today at 18:10Z, so #9299 now conflicts with `main` in
`benchmarks/one_stage/README.md` and `benchmarks/one_stage/server_rate.py`.

**The task, exactly as KSTEADY's handoff has it:** `~/.research/notes/lanes/kernels/ksteady-jobs/RESUME.md`
(after `research notes sync --path lanes/kernels`), the section "Where it stands (Thu 12 Nov 18:35Z)",
"Not done" item 1, with its "Restore" lines for the scripts (`h119/count.py`, `h119/launch.sh`,
`h119/pr-body.md`). In short:
1. Merge `origin/main` into the branch in a worktree (a merge; no rebase, no force-push).
2. In the two conflicted files keep proofs' text, and take the retired name out of whatever is left of
   #9077's own lines (say "the Lean verifier"). The recorded reason a row returns keeps the binary's
   name. No behaviour moves: the wide-leaf option, its refusals and every test's meaning stay.
3. Recount the four files with the lint's own `sites` (`h119/count.py afa939410f6f HEAD`). `main` now
   reads 20, 15, 7 and 24; **the targets are at most 19, 11, 2 and 23** (`main`'s count less what #9077
   added). Say each count, base against head.
4. Run the two suites that hold the files (`benchmarks/one_stage`, `verity-flock`) at the merged head in
   one named run on a CPU node (`--on NAME`; not `auto` blind, not `vy-nebius-cpu-7` blind).
5. Push the branch in a command of its own (`git -C <worktree> push -u origin
   cursor/kernels-wide-leaf-terms-7c1e`), then `research queue edit 9299 --body-file …` with the new
   counts and the run. **Mark nothing: the lead reads it and marks.**

**Report** on HASH-119 with `research msg linear comment HASH-119 'TEXT' --worker KTERMS` (at most 100
words), never through the Linear connector's `save_` tools. Your note is
`~/.research/notes/lanes/kernels/<stamp>-report-kterms.md`, a CHECKPOINT line at its top every 30
minutes, synced.

**Return by 20:30Z** with the head, the four counts, the run's id and verdict, and anything that was
not as the handoff says. If the merge needs a choice the handoff does not make, return at once with it.
