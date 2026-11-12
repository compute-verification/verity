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

## Thu 12 Nov 18:43Z: from lean (HASH-125, 18:40Z): its own rename is rehearsed on the same head; the two maps must not hold one name twice

Read lean's comment on HASH-125 whole. Two things beyond your brief:
- **#9375's head moves today:** `main`'s Lean batch (`30d6fcc92ff7`) is being merged into it now, with
  one more merge at about 19:15Z; the ready line on LEAN-249 (due 19:25Z) names the head. Cut from
  that head, or merge it again before the list's search of record; say the head your list was made
  on.
- **lean's worker RESTAGE is rehearsing the period rows' rename on the same head** (LEAN-262, part of
  LEAN-80); its list is due on LEAN-262 today by 21:00Z. **In your list, mark every name RESTAGE's
  list also holds** (read LEAN-262 before you post), and put no such name into your map until lean
  says on HASH-125 whose move takes it. lean answers the day your list is posted.

## Thu 12 Nov 20:08Z: lean's #9375 is MARKED at `bafae2eaec2e5d6e04569b75c31d422b8c015d8c` (20:05Z) and does not move again until it is on `main`

lean (LEAN-249, 20:06Z): "From now until it is on `main` nothing is pushed to it; a fix is a new
pull request." So the head of `cursor/lean-remote-merge-onmain-fc8d` to build on, search and test
at is `bafae2eaec2e`: merge it once now and say that sha in what you report. Its candidate's check
is the first Lean build and the first suites' verdict on that tree, so it may still come back red;
if lean says so on LEAN-249 I tell you here. When it is on `main`, your base is `main`.

## Thu 12 Nov 20:53Z: your list is read and rung; lean answered (HASH-125, 20:52Z). Go on NOW with the 39 names that are settled; proofs' six stay out until proofs says

Read whole: your return and your comment on HASH-125; five rows checked by me at `bafae2eaec2e`
(the three period-game lines, `windowK` and `audit_window_split_of_record` with their lock entries,
`MAX_SESSION_ID` in both locks). Rung: proofs and lean, with your three questions. **A list of 45
with each name's sibling, row and lock reads, the maps run in both orders, and the one condition
that would have doubled three names found before anything was built: that is the reading the
move needed.**

**lean's answers (read its comment whole):**
1. **Its 8 go in your move**, each with the name you gave. The two short names that then stand
   in two namespaces (`challengePhaseNamed_ok`, `trustedChallengePhaseOf_ok`): the build is the
   judge; if a file that opens both namespaces stops resolving, those two keep their old names
   and you say so on HASH-125.
2. **The three modules are not in this move.**
3. **Your condition holds:** Sunday's stage-4 map is variant B, with `monitoringPeriodGood` in
   its keep strings.
4. **The hour:** lean's renames are cut on `main` on Sun 15 Nov at 16:00Z and land that evening.
   Your move is on `main` before Sun 16:00Z, or it lands after lean's. "If yours opens Sat noon
   PST and is not on `main` by Sat 10 PM PST, say so here and we set the order then."

**proofs has not answered on the six** (`windowK`, `audit_window_split_of_record`,
`MAX_SESSION_ID`, `SESSION_FORM`, `WINDOW_FORM`, `Supplied.WINDOW_SALT`). Do not wait for it:

- **Now: the map with the 39 settled names** (proofs' 31 that no listed guarantee's statement
  reads, and lean's 8) in the tree on your branch, in `tools/move`'s form; `resolve_check`; the
  dry run on `bafae2eaec2e` on a CPU node you name: the move applies, the Lean of every touched
  package builds, the records run says which lock records move (with these 39, none should:
  say what it finds).
- **Keep the six as rows of their own** (a second block or file, not applied), so that proofs'
  yes adds them with one more records run and `--moved`, and its no costs nothing. Note that a
  name of the 31 may be used inside one of the six's statements or proofs: the build says.
- **The pull request:** open it, unmarked, with `Lane proofs`, when the dry run holds: the base
  is `main` if #9375 has landed by then, else say that it waits for it. Aim for Sat 14 Nov
  20:00Z or earlier; say on HASH-125 if a slot or a build makes that slip.
- The lint is its own pull request after the move, as the issue says.

If a run has no slot after 20 minutes, say so on HASH-125 and try another named node once
(`--pool research`; the pool is full today); do not wait on a run for more than 40 minutes
inside a turn: end your turn naming it and I resume you.

Return when the dry run has its verdict, or by Fri 13 Nov 04:00Z with where it stands.

## Thu 12 Nov 21:16Z: PROOFS ANSWERED (HASH-125, 21:15Z): rename ALL SIX, in your move. So the move is all 45, with three conditions

Read proofs' comment whole. "Rename all six, in your move. The three module names stay for a
layout move of their own." Each to the name you gave: `windowK` → `executionPhaseK`;
`audit_window_split_of_record` → `audit_executionPhase_split_of_record`;
`Spot.Challenger.MAX_SESSION_ID` → `MAX_CHALLENGE_PHASE_ID`; `SESSION_FORM` →
`CHALLENGE_PHASE_FORM`; `WINDOW_FORM` → `RUN_FORM`; `Spot.Supplied.WINDOW_SALT` → `RUN_SALT`.

**So: one map with all 45, one dry run, not two.** If your dry run of the 39 is already out, let
it finish (it says whether the 39 build) and then run the 45; if it is not out yet, send the 45.

**proofs' three conditions:**
1. **Names only.** No statement's text changes apart from the names, and no constant's string
   value changes. The locks are regenerated with `--update --moved` and the moves file is in the
   pull request, so the listed guarantee's record moves and nothing is sent. **If the records
   run shows any record changing beyond its own name and the names it reads: stop, say so on
   HASH-125, and end your turn. That would be a change to what a guarantee reads, and it goes
   to Daniel through proofs; it is not yours or mine to wave through.**
2. **The Python twins keep their names**; SPOT's difftest reads Python's.
3. **Say on HASH-125 when the pull request opens** (a worker comment; tell me and I ring):
   proofs then names its open Lean pull requests that edit `Audit/Window.lean`,
   `Spot/ChallengerSession.lean` or `Spot/Supplied.lean`. proofs' own move of the configurations
   is due on `main` by Sun 06:00Z; it says by Fri 20:00Z whether its maps share a path with
   yours and sets the order.

In the pull request's body, for the reader: the table of the 45 (old, new, file, the sibling's
row), which records moved and that each moved by its name and the names it reads only (how you
checked: the diff of each lock, record by record), the moves file, the builds' and the records
run's ids.
