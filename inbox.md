[2026-11-11T03:41Z] 33f1e7efb9d3
From training's coordinator, Tue 10 Nov 7:41 PM PST. THE MOVE IS ON main. For you: nothing changes for TRN-98 tonight. Go on on cursor/training-replot-55a1 on the old paths (its date is Wed noon PST; the table is published from the store, not from main); do not merge main into it. #8753 is marked and forecast clean: ci crosses it tonight, so push nothing to its branch. Once the comparison is with me and the table is published, cross cursor/training-replot-55a1 by hand with the command below (or say that the branch is scratch and can be dropped), and tell me. If your job needs a tool from main, run it from a checkout of the moved main.

## The move is on `main` (top, TOP-398, Tue 10 Nov 7:34 PM PST; read it there whole)

`main` is `5c095bb4ab70` since 7:26 PM PST: every file is where the approved architecture puts it (the calculator is `verity/protocols/inference_only/rho/`; the last `main` before the move is `a930252c4993`). Every open branch crosses the move; **a plain `git merge origin/main` is not a crossing** (where clean it is still wrong: a file you added under an old path stays there, and your own lines naming old paths are not rewritten).

- **ci crosses every MARKED head that crosses clean** and pushes the crossed head to its branch (about 9 PM PST; counts on CI-524). A mark and a land verdict carry when the head's own change is the old head's carried across; a head that was marked but not reviewed is marked again by its owner. **Push nothing to a marked head ci has not crossed yet.**
- **By hand: unmarked branches, and heads ci stops on.** From a fresh worktree of the moved `main`, for a branch `B`:

      git fetch origin main B
      python3 tools/move/restack.py origin/B --move-commit 6581bcf197 --move-commit c1ca25a247 --onto origin/main --kept --no-pins

  - It prints `clean` and a commit id, a descendant of your head: `git push origin ID:refs/heads/B`, never forced.
  - Name the two commits (without them the branch ends `based-after`); `--onto origin/main` is needed (without it the result lacks what is above the second move).
  - `conflicted` at the first step is an ordinary conflict with `main` before the move: merge `a930252c4993` into the branch on the old paths, then cross. If it still stops there it meets one of the move's hand-written lines: tell me the file.
  - `script-failed`, "tracked files under a retired path with no row in the map": the branch adds a file under a path the move retires. One commit on the branch first: `git mv` the file to its moved place, then cross.
  - **A stack crosses whole, before its lowest member lands**; once the lowest is on `main`, the ones above it are refused.
  - `--no-pins` skips the pins step, whose stale-pin command builds Lean. If the change touches a pinned input, drop the flag and run the crossing on a node, never on your VM.
  - About two minutes a branch on one core; the first on a machine a few minutes more.
- Before any merge on the moved tree, once: `git config merge.vllm-allowlist.driver 'python3 adapters/vllm/tests/lint/merge_allowlist.py %O %A %B %P'`.
- A branch that edits `tools/verity/check/public_boundary.toml` runs `public_equal.py` before its mark.
- Read `tools/move/README.md` at `origin/main` before the first crossing. `research` and the tools run from a checkout of the moved `main`.

**ci's forecast for training's heads** (made Tue 6:14 PM PST on the staged tree; read by the coordinator from `art:4c9ca5974557`): cross clean by the command: #8449, #8490, #8502, #8504, #8541, #8550, #8577, #8591, #8593, #8606, #8676 (at `34490b47`), #8703, #8753. **Stop at the layout** (the base of the second move conflicts: `benchmarks/README.md`'s rows, `benchmarks/training_slowdown/pyproject.toml`): #8551, #8589, #8680, #8712, and so what stands on them (#8778, #8775). #8595 was forecast at an older head.

