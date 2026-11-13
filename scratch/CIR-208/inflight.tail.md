
## lean's #9375, circuits' #9340 and top's staged move branch

- **#9375** (`34c5cbb3705c`) is on `main` since 03:01Z, and the base holds it: the test on the base is `3 passed`, with the
  renamed `DefinitionTable` and `BodyNode` in the tree. The test names neither class; it reads the table through
  `verity.circuits.defs.REGISTRY`, which kept its name. Before it landed, the earlier head `b1409cffb0` merged with
  `34c5cbb3705c` also gave `3 passed`.
- **#9340** (`3aefb20f1941`, marked): merges with the base, `3 passed`, the same 2,249 ids as the base.
- **top's staged move branch** (`cursor/top-moves-c-b396` at `680f1541f8c3`): its merge stops in one file,
  `tools/verity/check/public_boundary.toml`, against `main` alone too (`main` edited the list the branch edits). With the
  branch's side of it the test is `3 passed`. The branch holds the maps and not the move, so the move itself was run:
  `r20261113-032839-64dc` ran the four maps on this merge. After the first three and again after the fourth, every one of
  the list's 239 `file` strings names a file, the test is `3 passed`, `--prune` changes only the order of 25 lines, and the
  test file's graph task passes with no undeclared read (`paths.md`).

## One pull request that deletes listed versions

**#8415** (network-accounting, not marked, last pushed Tue 10 Nov on a tree older than the layout's move) retires
`CompileSlice_v2`, `CompileSlice_v3` and `CacheStep_v2`. All three are in the list. Its merge with `main` stops in 32
files, so it got no verdict here. When its owner brings it up to `main` after #8463 has landed, the test fails until
`--prune` takes the three entries out: the first way through, used as meant.

## What moved while this was run

Between the first read of the queue (02:52Z, 49 marked) and the one this document is made on (03:24Z, 58 marked), ten
marked pull requests landed, twenty more were marked and one lost its mark; by the second read above (04:17Z) 68 were
marked. An earlier sweep at the head `b1409cffb0` (`main` `8c11a1d5ab`) over the first read's 261 heads found the same:
no head fails, 107 merges stop. A pull request opened, pushed or marked after 04:17Z is not in this document.

## Run, and inferred

- **Run.** Every count in the tables above: 365 merges at this base (264 on the 03:24Z read, 101 more on the 04:17Z
  read), the test file on each tree that merged, the two controls, and `git merge-tree` of `main` with each head whose
  merge stopped on the 03:24Z read (the two with 913 files in conflict left out).
- **Inferred.** That a pull request whose merge stops will pass once its owner merges `main`: the verdict is of the pull
  request's side of each conflict. For the marked ones no conflicted file is one the test reads, so the inference is
  short. For 20 of the 111 not marked on the 03:24Z read it is not.
- **Not run.** The queue's own candidate with #8463 in it; `tools/move/restack.py` on the branch.
