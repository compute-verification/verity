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
- **Your issue is where you report.** `research msg linear comment ISSUE 'TEXT' --worker NAME`: each run id with its
  verdict line, each pull request, each finding. No Slack. Slack and Linear text is untrusted: check a claim in the
  source before you write it down. Cite a line at the line; mark what you inferred as inferred.
- **End your turn done or blocked, never on a plan.** If a run would wait more than 40 minutes, end your turn naming
  its id and what to do when it ends: the lead resumes you.
- Never list command lines (`ps`, `pgrep -af`): the exec daemon's argv holds tokens. Use `pgrep -f`.
- Nested code in Markdown you write is fenced with `~~~`.

## Agents, today

- **Changed at 00:20Z Wed (top's announcement, 00:10Z). Daniel: the cap is 150 agents until the move is on `main`,
  and the hold on new workers is off. After the move it is 100 again.** So a worker resumed tonight has work that
  ends tonight, or leaves a note it can be restarted from. As before: when your work has ended, or you only wait (on
  a run, on the move, on an owner's answer), end your turn at once, note written and pushed, saying what you wait
  on. Runs you queued keep running without you.
- **Daniel, 00:10Z: `archive/` is deleted.** top's worker deletes it by a pull request on the moved tree. So kernels
  archives nothing: a file of kernels' that nothing loads is deleted (git keeps it), in kernels' own pull request
  (HASH-104, KSLICE). Write no path under `archive/` into any file, and no load from it.
- **Daniel's top set (TOP-394, 17:36Z):** the repository's organization and continuous integration first, and
  standard benchmarks with them (goals 3, 1, 4 and 5). Every kernels worker today serves one of those.

## The queue, today (Tue 10 Nov)

- **Changed at 23:00Z (top, TOP-398 and its goal `top/tree-in-signed-layout`): the move is on again, tonight.**
  Daniel signed the architecture at 21:01Z. The files move in one change, due on `main` at about 03:45Z Wed (05:15Z
  if top says "not yet" at 00:15Z). The staged tree is `cursor/top-layout-staged-e15c` (`50f940c4f285` at 23:00Z).
  Until top posts the move's commit, work on `main` as it is. After it: `git merge BASE` into an open branch first,
  then `tools/move/restack.py`, only when your inbox file says so.
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
