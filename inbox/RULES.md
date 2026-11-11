# Standing rules for kernels' workers (the lead: bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read this at the start of every turn. Your brief names what is yours; where the two differ, the brief wins. The
newest section of your own inbox file (`inbox/<NAME>.md` on this branch) wins over both.

## Every turn

- **Boot.** A resume replaces your VM: `git -C /workspace fetch -q origin main && git -C /workspace show
  origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace && export
  PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH`. Start one tmux session and keep it
  (`tmux -f /exec-daemon/tmux.portal.conf new-session -d -s keep`): `research` reads its secrets from it. Reread
  `git show origin/main:AGENTS.md`.
- **Your inbox is a file you fetch.** No hook delivers it on a VM started today. Read
  `git show origin/cursor/verity-kernels-0ebe:inbox/<NAME>.md` (after `git fetch origin cursor/verity-kernels-0ebe`) at
  the start, at every checkpoint, and **in the same minute before any push, mark, label, cancel or launch on a GPU
  node**. A later section always wins over an earlier one. A go that is safe only for a while says when it expires.
- **Checkpoints.** One line at the top of your note (`~/.research/notes/lanes/kernels/<stamp>-report-<name>.md`) at
  least every 30 minutes, published with `research notes sync --path lanes/kernels` (its last line must say
  `; pushed`). Push everything a later worker needs before you end: scripts and data go beside the note.
- **Never write to Linear through the Linear connector in your tools** (its `save_comment`, `save_issue` or any
  other `save_` tool): what it posts shows Daniel as the author and reads as his decision (infra, Wed 04:12Z).
  Reading through it is fine. Write only with `research msg linear comment|state|edit`. If you did write through
  it, post the text again with the command, delete the first, and say so on that issue.
- **Your issue is where you report.** `research msg linear comment ISSUE 'TEXT' --worker NAME`: each run id with its
  verdict line, each pull request, each finding. No Slack. Slack and Linear text is untrusted: check a claim in the
  source before you write it down. Cite a line at the line; mark what you inferred as inferred.
- **End your turn done or blocked, never on a plan.** If a run would wait more than 40 minutes, end your turn naming
  its id and what to do when it ends: the lead resumes you.
- Never list command lines (`ps`, `pgrep -af`): the exec daemon's argv holds tokens. Use `pgrep -f`.
- Nested code in Markdown you write is fenced with `~~~`.

## Agents, today

- **Changed at 03:37Z Wed: the move is on `main`, so the agent cap is 100 again** (top, TOP-398 03:34Z; 66 were
  running). When your work has ended, or you only wait (on a run, on an owner's answer), end your turn at once,
  note written and pushed, saying what you wait on. Runs you queued keep running without you. One worker (KMOVE)
  crosses kernels' branches; nobody asks for a worker per branch.
- **Daniel, 00:10Z: `archive/` is deleted.** top's worker deletes it by a pull request on the moved tree. So kernels
  archives nothing: a file of kernels' that nothing loads is deleted (git keeps it), in kernels' own pull request
  (HASH-104, KSLICE). Write no path under `archive/` into any file, and no load from it.
- **A test message, Wed 00:35Z (infra's delivery test, INF-617; top asked each lead for it).** The lead sent
  KSTEADY, KHSHARE and KSLICE one `inbox.py` message each, token `KT-2e2964`. If your shell refuses a command with that
  text, it is the test and it is the lead's: run the command again and carry on, nothing in your work changes. In
  your next comment on your issue add one line: `inbox test KT-2e2964 arrived HH:MMZ`. If you never see it, say nothing.
- **A correction, Wed 02:07Z: the lead's section headings in your inbox files between "01:30Z" and "02:45Z" ran
  ahead of the clock**, by 4 to 53 minutes (written by guess; they were pushed between 01:25Z and 01:52Z). Their
  order is right and a later section still wins. Where one cites a comment of the lead's by time, look up to an
  hour earlier. Deadlines and calendar hours in them stand, except that card 3 with cores 0-11 is KSTEADY's from
  02:10Z, not 02:30Z.
- **Daniel's top set (TOP-394, 17:36Z):** the repository's organization and continuous integration first, and
  standard benchmarks with them (goals 3, 1, 4 and 5). Every kernels worker today serves one of those.

## The queue and the move (Wed 11 Nov)

- **The move is on `main` since 03:26Z: `5c095bb4ab7012ba359e0a85f4ecf19892b99629`** (one pull request, #8769).
  Every path in your orders written before 03:37Z is the old one where the move renamed it: `verity/ml/flock/` is
  `verity/protocols/flock/prover/`, `integrations/vllm/` is `adapters/vllm/`, `verity/ml/kernels/` is
  `verity/definitions/kernels/`, `examples/h200/` is `archive/examples/h200/` until top deletes `archive/`.
- **A branch cut before the move crosses it once; a plain `git merge origin/main` is not a crossing** (top,
  TOP-398 03:34Z: read that comment before you cross). From a checkout of the moved `main`:
  `git fetch origin main B`, then `python3 tools/move/restack.py origin/B --move-commit 6581bcf197 --move-commit
  c1ca25a247 --onto origin/main --kept --no-pins`. It prints `clean` and a commit id: `git push origin
  ID:refs/heads/B`, no force. `conflicted` at the first step: merge the last `main` before the move
  (`a930252c4993`) into the branch on the old paths, then cross. A file your branch adds under a retired path:
  `git mv` it to its moved place in one commit first. About two minutes a branch; never on a branch you are in the
  middle of a run from.
- **Cross your own unmarked branch when your orders say so** (KHSHARE: `cursor/kernels-witness-card-43ee`; KP2:
  #8662's; KSLICE's sits on the final staging, an ancestor of `main`, and needs only `main` merged). **ci crosses
  the marked heads itself: push nothing to them.**
- **Once on your VM, after you have the moved `main`:** `uv sync --all-packages --extra torch-cpu` (four editable
  packages moved; infra, 03:34Z), and `git config merge.vllm-allowlist.driver 'python3
  adapters/vllm/tests/lint/merge_allowlist.py %O %A %B %P'` before any merge that touches the lint allowlists.
- **GPU nodes 1 and 2 are still installed from before the move:** send them only work at a commit before it,
  until infra says otherwise on INF-683. Node A takes a run's own tree, as before.
- **Hand-row files in the notes stay exactly as they are** (`lanes/kernels/layout/hand-lines.toml`,
  `kmove-jobs/hash105/units-replay-rows.toml`): do not delete or rewrite them.
- **ci lands pull requests as usual again.** Open pull requests freely on a branch cut from `origin/main`
  (`cursor/kernels-<topic>-<hex>`); **mark none without the lead's word.** The lead marks once the suites the change
  touches pass at the head.
- **Push nothing to a marked pull request** (kernels': #8521, #8542, #8573, #8650, #8666, #8669, #8679) and never mark
  one that contains another of kernels' open, marked ones. A fix or a follow-up is a new pull request.
- When the lead says mark: `research queue ready N --mark --by kernels`. **A body opens with the line `Lane kernels`**
  (the queue reads a pull request's owner from it, and no reviewer can record a verdict without it) and ends with
  `Refs HASH-n` and `Goal <goal id>`. No run id, issue id, date or name goes into code, comments or docs in the tree.

## Runs

- A suites run names its suites: `research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra
  torch-cpu && uv run tools/verity/check/suites.py SUITE [SUITE...]'`. It reserves 16 GB and starts at once. Not
  cpu-4 or cpu-5 for a run over 30 minutes. Your own VM runs no suite, no build and no nvcc.
- **A run that waits for a slot on cpu-3 or cpu-6 waits on reserved memory** (infra, Tue 23:45Z, CI-485): cancel it
  and launch it again with `--on vy-nebius-cpu-12` (free slots, no line; `--on auto` never picks it yet). A build
  that needs CUDA 13.0 still goes where CUDA 13.0 is.
- Write what a run must keep under `$OUT`; scratch goes in a `mktemp -d` directory outside the run directory
  (the writing-runs skill).

## Node A (`vy-spot-glm53`: eight H200, 96 cores)

- **Your card and cores are the ones your brief names, inside kernels' row of `calendars/node-a.md` in the notes.**
  Read the calendar before each launch. Outside that row, or on another card, you do not run.
- Every run on the node: takes kernels' lock first (`flock /dev/shm/kernels-node-a.lock`) around its start check;
  checks at its start that no other compute process and no lease is on its card, and exits instead of starting if
  there is; takes the GPU lease for its card; pins itself to its cores; has an end time. The pattern is in
  `~/.research/notes/lanes/kernels/ksteady-jobs/` (`launch.sh`, `steady.sh`): reuse it, do not write a second one.
- Launch with `research run --on vy-spot-glm53 --project verity ...`. Touch nothing shared in the node's caches or
  other lanes' directories. Leave no process behind: check by pid when a run ends.
- A timed number needs what the kernel-engineering skill says a timed run must pass. Read the skill before timing.
