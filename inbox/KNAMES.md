# KNAMES's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 18:26Z: your brief: HASH-125, the names lean's stage 4 left on a retired word: the list, a map, one move, then the lint

**Who you are.** KNAMES, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). The piece is proofs': it listed
ten pieces another lane can take (TOP-402, Thu 12 Nov 17:42Z) and top gave this one, piece 7, to kernels.
It serves proofs' goal `proofs/terminology-clean` (Fri 20 Nov).

**The task: HASH-125, whole; its steps and "Done when" are the contract.** After lean's stage 4 of the
terms move (LEAN-80), about 30 names in proofs' Lean still carry a retired word (`window`, `session` and
their siblings) because a sibling of theirs was renamed and they were not. Rename them in one move, built
now on lean's merge branch so that it is ready the hour that merge is on `main`.
1. **The list** (your first return, posted on HASH-125): from lean's stage 4 map (LEAN-80; PRF-486's row
   list) and a search of `security/` and the verifiers' Lean. proofs' count of about 30 is not recounted:
   yours is the count. For each name: the file and line, the retired word, the sibling that was renamed
   and the row lean used for it, the new name that row gives, and whether a listed guarantee's statement
   reads it (`lean-audit.json`).
2. **A map file** in the form `tools/move` takes (`tools/move/README.md`), one row a name.
3. **A dry run** on the merge branch's tip: the move applies, the Lean of every touched package builds,
   the records run says which lock records move; a guarantee's statement that only moved is declared with
   `--moved` (the lean-proofs skill).
4. **One pull request, the move alone,** opened unmarked against `main` with `Lane proofs` in its body.
   proofs marks it; you never mark.
5. **The lint** that refuses new uses of the retired words in those areas, as its own pull request after
   the move.
**Not yours to decide:** a name that a listed guarantee's statement reads. Say so on HASH-125 before
renaming it and wait for proofs' word on that name; go on with the others.

**One correction to the issue's text: the base is `cursor/lean-remote-merge-onmain-fc8d` (lean's #9375),
not `cursor/lean-remote-merge-landing-fc8d`.** lean, TOP-402, 17:34Z: the queue refuses any head that
reaches the landing branch or `remote`; #9375 is the same tree as one commit on `main`. Search and build
on #9375's head; cut `cursor/kernels-names-move-` and four hex from it
(`git merge-base --is-ancestor origin/cursor/lean-remote-merge-landing-fc8d HEAD` must fail). Its head
moves a few times until lean marks it (about 20:00Z today): merge it again before the dry run of record.
Push only to your own branch, each push in a command of its own (`git -C <checkout> push -u origin HEAD`).

**Pointers.** HASH-125; PRF-361 (the terms work and its tools: `tools/move/rename.py`,
`tools/move/terms_map.toml`, the ratchet lint of #7153); LEAN-80 (lean's map, its families and stages);
PRF-486 (soundness's yes to its rows, today); proofs' worker's reading of stage 4: the entry "PROGHELD,
Thu 12 Nov 15:25Z" in `inbox/proofs-coordinator.md` on `origin/cursor/verity-proofs-84b5`; the skills
`team` (how a move's pull request lands), `lean-proofs` (the edit loop, the lock, `--moved`, why two
agents never share a writable `.lake`), `testing`, `friction`.

**How.** Lean builds are heavy: run them with `research run` or `verity lean try` on a CPU node you name
(`--on NAME`; not `auto` blind, not `vy-nebius-cpu-7` blind), never on your VM. The research pool is
full today (39 runs in line at 17:56Z): build only the packages the move touches, and if a run has no
slot after 20 minutes say so on HASH-125 and end your turn naming its id. Every claim says how it is
known: read (file and line), run (the run's id), derived, or guess. Report on HASH-125 with
`research msg linear comment HASH-125 'TEXT' --worker KNAMES`, never through the Linear connector's
`save_` tools. Your note is `~/.research/notes/lanes/kernels/<stamp>-report-knames.md`, a CHECKPOINT line
at its top every 30 minutes, synced.

**Dates.** The list on HASH-125: proofs asks for Fri 13 Nov 20:00Z; **yours is tonight, by Fri 02:00Z**,
and you end your turn with it (proofs waits for it). The move's pull request open: proofs asks for Mon
16 Nov 20:00Z; say with the list what date holds once you have seen the names.
