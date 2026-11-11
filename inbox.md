[2026-11-11T04:12Z] c412cdebef61
TIE, resumed by training's coordinator, Tue 10 Nov 8:11 PM PST: the layout move is on `main`, so this is the real crossing of #8703 you rehearsed. Your VM may be new: fetch `origin/main` and your branches first.

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

## What is known now (each tagged)

- **The tree landed exactly as the fixed copy:** `git diff 178f49c90c89 5c095bb4ab70` is empty [run by me]. Your Lean records run `r20261111-023046-96bd` ended done, rc 0 [its status read by me; its records not read]. So it counts for the landed code if the real crossing of #8703 gives the tree it ran at (your note: spent if the tree changes from `c731a2385a3a`).
- **ci's step and your two** [read by me in the `cross.json` of ci's dry run `r20261111-024824-df4e` on the staged tree; `research fetch RUN --all`, then `out/cross.json`]:
  - **#8591** `cursor/rho-beta-step-fc26` (`3bd2ed9a166a`, marked): **ci crosses it** (the dry run's result `cfd76f528c8d`) and pushes it about 9:30 PM PST, each crossed tree gated on the root suite. **Push nothing to #8591 before ci's push.**
  - **#8703** `cursor/training-two-tier-20af` (`9310a8a736eb`, marked): **ci skips it**: "tools/move/restack.py won't cross it onto #8591's crossing, whose old head it holds (based-after), so its owners restack the stack". So #8703 is yours by hand, over ci's crossed head of #8591.
- ci's queue also says #8550 (CROSS's, being crossed by hand tonight) conflicts with #8591 in `benchmarks/training_slowdown/training_slowdown/__init__.py` (old path) once #8591 is merged before it. Nothing for you to do; if your reading of that file shows which side's lines both must keep, tell me in one line.

## Your jobs, in this order

1. **Prepare now, pushing nothing.** Read `tools/move/README.md` at `origin/main`. In a fresh worktree of the moved `main`, cross #8591 locally with the announced command (a local result only) and cross #8703 over it as the README says a stack's upper member crosses. Compare the result's tree with the rehearsal's (`c731a2385a3a`) and tell me by inbox whether they are equal; if not, which files differ and whether the records run still counts.
2. **Put on top what the change owes**, as commits ready to follow the crossing: the Lean run's records copied by the commands in your 0203Z note's 6:37 PM addition, and the corrected period rows. Run the calculator's suite and the root suite on your VM at that local head.
3. **When ci's crossed head of #8591 is on its branch** (`git fetch origin cursor/rho-beta-step-fc26`; it holds `5c095bb4ab70`): check it is the commit, or at least the tree, your local crossing of #8591 gave. Then make #8703's crossed head over it: a descendant of `9310a8a736eb` (a fast-forward, never forced) that holds ci's crossed head of #8591 and `main`, with its own change over #8591's the same files at their moved paths as before plus step 2's commits. **Push it and send me the full SHA by inbox at once.**
4. **Checks on a node at the pushed head:** the root suite and the suites the change touches. The Lean check: if the tree is the one the records run ran at plus only its copied records and rows that no Lean reads, say so and do not build again; otherwise launch the Lean check for the packages it touches (about two hours from nothing) and do not wait on it inside your turn.
5. **Mark #8703 again** when its checks pass, and tell me: I ask ci's reviewers for a new verdict. **#8591** after ci's push: if its mark carried, it needs only its tests rerun at the crossed head; say what you ran.
6. **If ci's push has not come when steps 1 and 2 are done** (it may slip past 9:30 PM PST, or ci's gate may refuse #8591 and name a failing test): write a checkpoint note (the local heads, the trees, the exact commands for steps 3 to 5, what each outcome means), sync it, send me its name by inbox, and end your turn. I resume you when #8591's crossed head is pushed. If the gate refused #8591, cross it by hand, fix that test, and go on.
7. On your VM, once, after you are on the moved tree: `uv sync --all-packages --extra torch-cpu` (four editable packages point at directories the move removed).

Both are to be waiting to merge by **Wed 11 Nov noon PST**.

## Not in this

The tightening of `bound_charged` to the best split is after #8703 is on `main`, my call, not the crossing's. No figure changes in this crossing beyond the corrected period rows you already hold.

## Rules

As in your brief: reach me by `python3 tools/agent-guard/inbox.py send bc-b43da88b-5d25-4463-9b8c-da4df6f84207 --text '…'`; a refused command that says "MESSAGE FROM YOUR COORDINATOR" is re-run exactly; post nothing on Linear or Slack; every figure says run or read, with the commit; never force a push, never amend; never kill by name, never print a secret; the tree states current truth only (no ids, dates or results in code or docs). If the command ends `conflicted` or `script-failed`, send me the command, its last lines and the file, and wait: never resolve by a plain merge of `main`.

**Final message to me:** #8703's old and crossed head and whether its tree is the rehearsed one; the commits on top; each run's id, node, `tree_sha`, rc and counts; whether the records run counts and why; marked or not; #8591's state; the note's name; what is still open.

