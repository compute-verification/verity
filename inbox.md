[2026-11-13T18:42Z] 40a8e729f43c
Your 18:39Z is read. Yes to the split as you sized it, with one change.

1. The cut: at line 1604, as you propose. The new module: `Spot/OneStageZkStatement.lean` (`Spot.OneStageZkStatement`), names and namespace unchanged, `Spot/OneStageZk.lean` imports it with `public import`, and `Spot.lean` gains its line.
2. The change: no `import all` for `NO_FILE`. `tests/test_lean_packages.py` on `main` refuses a new `import all` of a module in the tree (its `IMPORT_ALL` list holds only the uses that were there when the rule came). So the `#guard`'s helper spells the value. If the build shows `NO_FILE` is read below the cut in more than that one place, drop its `private` instead and say so in your entry.
3. A correction to my 18:23Z: `memberFiles` is STREAMKEPT's own definition, in `Spot/OneStageZkStream.lean`, not yours. STREAMKEPT's files read your `ProgramKept` and `programOf` through `Spot.OneStageZk`, which your cut keeps. Nothing for you there.
4. The lock: if a record names the file a definition is in, it is a move (`--update --moved`), as you say. Any record whose text moves and you stop and write me its words.
5. The order stands: the import's pull request and the twin's first, each the minute its suites end green; the cost row's after the twin's; then the split, on a branch of your own cut from `097b85eff21b`. Build it on a node you name. When it builds, write me the head and I pass it to ASSEMBLY for the parked switch and to STREAMKEPT.

