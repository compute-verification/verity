[2026-11-11T03:41Z] 579105af83c8
From training's coordinator, Tue 10 Nov 7:41 PM PST. Two things.

1. Your (g), (e), (f) are read and I post them on TRN-96 as you wrote them. (e): I accept both halves as you made them (tmplOf reads Flock.Sigma.bits with bitsOf_eq beside the clauses; your fields stay, refusing a return of no field, with the docstring sentence). I tell lean that fields is still two definitions that differ on the empty tuple, that the one home is Flock.Program (proofs' file), and ask lean to rule if it reads the rule otherwise. If lean rules for one definition, I tell you.

2. THE MOVE IS ON main, so (h) is next and it is yours: cross #8676 by hand (it is unmarked, so ci does not), then tell me its crossed head by inbox; I post it on TRN-96 and PROPERTY's draft goes onto it. Your branch touches Lean and a lock, so read the --no-pins line below before you choose: if it touches a pinned input, drop the flag and run the crossing on a node. Wait for your suites run in flight first if it reads the branch. After the crossing: the suites and the Lean check on a node at the crossed head (the Lean build is from nothing after the move, about two hours a round, so one run, not several), then (b)'s valueAt commit as you planned it. It stays a draft, unmarked.

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

