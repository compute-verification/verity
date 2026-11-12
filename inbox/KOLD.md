# KOLD's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 18:46Z: your brief: HASH-126, kernels' twelve pull requests that lean's merge brings from `remote`: keep each, or name what takes it out. A reading job, with a table tonight

**Who you are.** KOLD, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work).

**The task: HASH-126, whole; its "Done when" is the contract.** lean's merge (#9375,
`cursor/lean-remote-merge-onmain-fc8d`) brings to `main` the lines of 157 pull requests that landed on
the `remote` branch and were closed without reaching `main`. Twelve are kernels'; one of them adds
lines to a Definition or a lock (#5967): read it first. top asks each lane (LEAN-249, Thu 12 Nov
18:38Z): for each of yours that you have since withdrawn, replaced or rewritten on `main`, the number
of the pull request that takes it out; no line means keep.

**Where the rows are.** On LEAN-249 (read through the Linear connector; page back to them): top's
comment of 18:38Z, and lean's worker OLDNEXT's two tables of 17:33Z (the 52 that touch a Definition
or a lock) and 17:35Z (all 187, largest first, each with its files). Search them for "kernels".

**For each of the twelve, one row of a table:**
- the pull request, its title, what it changed (`gh pr view N`, its diff);
- what #9375's head has of it that `main` does not (`git diff origin/main origin/cursor/lean-remote-merge-onmain-fc8d -- <its files>`), in lines and in one sentence;
- whether kernels withdrew, replaced or rewrote that on `main` since. Look at least at: #9260 (35
  unused files of `benchmarks/hash` and `benchmarks/torch_gpu` deleted today, HASH-104), the layout
  move, #9225 (the wide row digests), #8662 (the store's decode moved into its package), #8669 (the
  READMEs' interface sections), and `git log origin/main -- <file>` for each file;
- **the verdict with its evidence:** keep; taken out already (by which pull request on `main`, and
  that the merged tree agrees); or it needs a pull request after #9375 lands (exactly what it
  deletes or restores). Say for each whether a file the merge brings back is loaded by anything
  (`git grep` on the merged tree).
**And the other direction:** for each file of kernels' (`benchmarks/hash`, `benchmarks/torch_gpu`,
`benchmarks/moe_align`, `benchmarks/pouw`'s pearl_c directories, `benchmarks/remote_session`,
`benchmarks/one_stage`'s `server_rate.py`) that `main` changed since the `remote` branch forked: does
the merged tree still hold `main`'s version? (OLDNEXT: the merge carries 4,222 deleted lines of 113
of these pull requests: lines `main` has and the merged tree lacks.) List every such file where it
does not, with the lines.

**You change nothing and open no pull request in this turn.** A follow-up that a verdict needs is
described in the table; I file it.

**Report:** the table in your note (`~/.research/notes/lanes/kernels/<stamp>-report-kold.md`, synced),
and one worker comment on HASH-126 (`research msg linear comment HASH-126 'TEXT' --worker KOLD`, at
most 100 words: the count by verdict and the table's place). Not on LEAN-249: I post kernels' line
there after reading yours.

**Return by Fri 13 Nov 02:00Z** with the table's place, the count by verdict, #5967's verdict in two
sentences, and every file where the merge lacks `main`'s lines. Every claim says how it is known:
read (file and line), run (the command), derived, or guess.
