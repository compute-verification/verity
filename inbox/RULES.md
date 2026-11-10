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

## The queue, today (Tue 10 Nov)

- The layout lands as one commit at about 21:00Z (1 PM PST). The loop stops building candidates at 19:00Z: what is in
  no candidate by then lands after the layout.
- **Open pull requests freely; mark none without the lead's word.** A result that is one more pull request waits
  behind about 140. Write on a branch cut from `origin/main` (`cursor/kernels-<topic>-<hex>`); when the layout is on
  `main`, `tools/move/restack.py` carries the branch across. `cursor/top-layout-staged-af72` shows the new paths.
- **Push nothing to a marked pull request** (kernels': #8521, #8542, #8619, #8650) and never mark one that contains
  another of kernels' open, marked ones. A fix or a follow-up is a new pull request.
- When the lead says mark: the suites the change touches pass first; `research queue ready N --mark --by kernels`,
  or, if that refuses a head for holding today's `main`, `research data label pr:N@FULLSHA ready true --by kernels`,
  said in the body. A body ends with `Refs HASH-n` and `Goal <goal id>`. No run id, issue id, date or name goes into
  code, comments or docs in the tree.

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
