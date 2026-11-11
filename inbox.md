[2026-11-11T04:12Z] 39226c86ff7c
bounds, resumed by training's coordinator, Tue 10 Nov 8:11 PM PST: the layout move is on `main`, and four of your five deletions are yours to carry across it by hand, starting now. Your VM may be new: fetch `origin/main` and your branches first.

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

## What ci's step does with your five (read by me in the `cross.json` of ci's two dry runs on the staged tree, `r20261111-024824-df4e` and its rerun `r20261111-032059-0023`; `research fetch RUN --all`, then `out/cross.json`)

| PR | branch | head | ci's step |
|---|---|---|---|
| #8490 transfers | `cursor/nci-transfers-delete-6fd9` | `ca2b9ade86fc` | **crosses it** (the rerun's result `248854aef3a9`; the first run stopped at its base merge in `tests/lint/allowlists/repo_retired_terms.json`). ci pushes about 9:30 PM PST, each crossed tree gated on the root suite. |
| #8502 activations | `cursor/nci-activations-delete-6fd9` | `be48898185c0` | **refuses it**: "restack.py crossed it clean as 1ab528ad8ca2, but its own change there (68cb6de2de70) isn't its old one mapped across the moves (its merge with b5354b9c2344, the base of the move 6581bcf197d3, changes its own change), so no mark or verdict would carry". |
| #8504 packed | `cursor/nci-packed-delete-6fd9` | `3f4c4916dc12` | skips it: "#8502 below it in its stack doesn't cross" |
| #8593 the chain | `cursor/nci-chain-delete-8ca5` | `a724af609a98` | skips it, the same |
| #8595 info and Fmt2 | `cursor/nci-info-fmt2-delete-8ca5` | `0607e1f58beb` | skips it, the same |

So #8502 is a head ci stops on, and it and the three on it are the owner's by hand (top's rule; circuits and vllm are doing the same class tonight on CI-524). **#8490 is ci's: push nothing to it before ci's push.**

## Your jobs, in this order

1. **Read `tools/move/README.md` at `origin/main`**, then cross **#8502** by hand and **#8504, #8593, #8595** over it, in that order, as the README says a stack crosses (each upper member over the lower one's script commits, so each holds the crossed head of the one below). They touch Lean, so decide the pins step by the rule above: if a pinned input is touched, the crossing runs on a node without `--no-pins`, never on your VM; if you run a pins step on your VM, say which command it ran and that it built no Lean.
2. **Before each push, check and write down:** the result is a descendant of the old head (a fast-forward, never forced); it holds `main` `5c095bb4ab70`; and its own change against the one below is the same files at their moved paths with the same insertions and deletions as before the crossing. For #8502, say in two lines what the carry check saw: which files of its own change differ from the old change mapped across, and what the merge with `b5354b9c2344` did to them. That goes in the PR's body too, since the review verdict does not carry.
3. **Push each** (`git push origin ID:refs/heads/BRANCH`) and **send me each full SHA by inbox the minute it is pushed.**
4. **Checks at the crossed heads, on nodes:** the root suite (`tests/`, as the quick tier runs it) and the suites each change touches; and the Lean check from nothing for the packages each change touches. Every machine builds Lean from nothing after the move (top puts a round at about two hours), so launch #8502's first, and the others' beside it only on slots that are free; two builds never share a writable `.lake`. If one check at the top of the stack would stand for the members below it, say why before relying on it; otherwise each head that is marked has its own.
5. **Mark each again** when its checks pass (`research queue ready N --on auto` or as the team skill says), and tell me: I ask ci's reviewers for new verdicts.
6. **#8490, after ci's push** (about 9:30 PM PST): read its branch. If it holds `5c095bb4ab70` and its mark carried, it needs nothing from you; say so. If ci's gate refused it, ci names the first failing test: cross it by hand and fix that, then steps 2 to 5.
7. On your VM, once, after you are on the moved tree: `uv sync --all-packages --extra torch-cpu` (four editable packages point at directories the move removed).

## Do not wait inside a turn on a two-hour run

Launch the runs, write a checkpoint note (each run's id, node, the head it is at, and what each outcome means for the mark), sync it, send me its name and the run ids by inbox, and end your turn. I resume you when the runs are done. All five are to be waiting to merge by **Thu 12 Nov noon PST**; tonight's part is the four crossed heads pushed and their checks launched.

## Rules

As in your brief: reach me by `python3 tools/agent-guard/inbox.py send bc-b43da88b-5d25-4463-9b8c-da4df6f84207 --text '…'`; a refused command that says "MESSAGE FROM YOUR COORDINATOR" is re-run exactly; post nothing on Linear or Slack; every figure says run or read, with the commit; never force a push, never amend; never kill by name, never print a secret; the tree states current truth only (no ids, dates or results in code or docs). If the command ends `conflicted` or `script-failed`, or the stack's instructions do not fit, send me the command, its last lines and the file, and wait: never resolve by a plain merge of `main`.

**Final message to me:** a table of the five (old head, crossed head, hand fixes if any, the carry check's difference for #8502, each run's id and state, marked or not), the note's name, and what is still open.

