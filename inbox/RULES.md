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

- **top's hold is lifted (TOP-394, 17:36Z): 51 agents run, under Daniel's target of 60.** A worker that only waits
  still ends its turn, note written and pushed, saying what it waits on (a run's id and end time, an owner's
  answer): the lead resumes you. Runs you queued keep running without you.
- **Daniel's top set (TOP-394, 17:36Z):** the repository's organization and continuous integration first, and
  standard benchmarks with them (goals 3, 1, 4 and 5). Every kernels worker today serves one of those.

## The queue, today (Tue 10 Nov)

- **Changed at 17:39Z (top, TOP-394): the layout does not land today.** Daniel read the target tree and wants it
  changed first; the map will change once, and every open branch then crosses one time. **Nobody waits for the
  layout.** A step of yours that says "when the layout is on `main`", "restack" or "the moved tree" is void: work
  on `main` as it is, in today's paths. Do not run `tools/move/restack.py`. The stager's run ends as a rehearsal.
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
