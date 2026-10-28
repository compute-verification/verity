---
name: team
description: >
  How Verity's own agent team works, for its coordinators and workers (an outside contributor needs only AGENTS.md):
  the team's machines and where heavy work runs, how a PR lands through the queue, where work, notes, evidence and
  decisions live outside the tree, how coordinators and lanes are organized, and the working principles that bind them.
  Read it first if you are one of the team's agents, and reread it each night with the friction and lean-proofs skills.
---

# The team

AGENTS.md states the code's layout and rules. This skill states how the team that writes the code works.

## Machines

- **An agent VM only edits.** Its 16 GB are shared by every agent on it, and one out-of-memory kill takes them all down.
  Locally, run only git, reading and editing, messaging and the notes, one test file or `-k` selection under about a
  minute and 1 GB, `lake env lean` on one file of a built tree, a Lean package with no Mathlib or ArkLib, and the lock
  fast path. Everything else (every suite, `verity check`, circuit-check, any build of `Security`, cargo, nvcc, every
  e2e and GPU job) runs through `research run --on auto --cwd source [--wait] -- CMD` (the writing-runs skill): an
  `agent-test` job on the cluster queue (4 CPUs, 16 GiB, 60 min unless `--cpus`, `--mem-gb` or `--timeout` say
  otherwise), or a `research` pool when one starts sooner. The shell hook refuses those locally and names the remote
  command; `AGENT_GUARD_LOCAL_OK=1` in the command opts in.

  ~~~text
  research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py'  # --quick: no slow tests
  research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run --extra torch-cpu verity check'
  research status RUN  # check back on a run; --wait only for one under 20 min
  ~~~

- **No tool call blocks more than 20 minutes.** Every `--wait` stops there, cancels nothing and exits 79 naming the
  command to check back with, and you end your turn.
- **Code reaches the Nebius nodes** only through `research deploy install` of a pushed commit
  (`tools/research/src/research/pods/nebius/deploy.toml`). Delete data you don't own on a node only with
  `research retention gc` or `research retention rm --approved-by @OWNER`.
- **A guard never fails open**, and an agent VM can be suspended at any moment, so nothing that bounds spend or preserves
  results depends on one: pods bound themselves (`research pods create --max-hours`, under a line of the notes repo's
  `budgets.toml`) and runs preserve their results from the pod. The lane whose budget line pays for a machine stops it
  when its work there ends.
- **Secrets.** The shell hook (`.cursor/hooks.json`, `tools/agent-guard/shell_hook.py`) refuses a command that would
  print the environment or every process's command line; on a VM booted without it, run origin/main's
  `tools/agent-guard/boot.sh`. A secret is stored as one line (a key as base64, `NAME_B64`).

## Landing

PRs land on an eyeball review, and the full check runs after landing. ci's `research queue drive` lands them
(`research.drive`'s docstring has the rules; `tools/verity/check/drive.toml` the settings):

- The worker that made a branch opens its PR, `research queue open --head BRANCH --sha SHA --title T --body-file F --by
  LANE` with `Lane LANE` in the body, and marks it ready at its head, `research queue ready N --mark --by LANE`. A push
  makes a new head, which needs a new mark and a new review. `research queue status` says what each PR waits for. A
  `hold` label, a conflict on `next` or a Markdown file over its cap keeps a PR out.
- Before marking, run the Lean check of the packages you touched and the suites your change touches, `research run --on
  auto --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py
  --changed $(git rev-parse origin/main) --quick"`, plus a suite's slow tier when you changed its core. Prefer small PRs.
  You hear back only if yours is held, ejected or reverted.
- Drive stacks admitted PRs on `next`, non-Lean changes first. Reviewers ci staffs, never the owner, record a verdict at
  the head (`research queue review N --land|--hold REASON --by LANE`); no other per-head grant exists. Drive
  fast-forwards `main` to the longest reviewed prefix whose merges are clean and whose `.py` files compile; while no Lean
  check runs, it moves reviewed members below those awaiting a verdict, so a PR waiting for review holds only itself (a
  hold still keeps what is above it). A change to a
  `lean-audit.json` lock or to Lean that runs as code (a `.lean` outside `Security/`) also waits for a `lean-audit` that
  passed at or above it; Lean proofs alone land on the verdict.
- One run checks `main`'s tip after each landing: the whole check as shards, one per machine drive.toml's `on` lists,
  recorded as one check (`check.py --combine`) once every shard passed. A red suite's culprit is rung and reverted 60 min later unless a fix is
  marked. Citations and tables read only commits whose after-landing run passed.
- A pure move that `verity check move BASE HEAD` certifies lands alone, ahead of the queue, through the lander. Only
  drive's App pushes `main` and `next`.
- `check.py --record --on M --pool NAME`, from a checkout, records `check` as an Attempt of the clean commit in M's NAME
  pool.
- When a changed lock record lands on `main`, Daniel gets a DM showing it; that DM is the check, and nothing waits on it.
- Anyone may do whatever it takes to cut check time: a change to how `check` runs lands without its file owner's review,
  and the owner is told. infra owns check time and its spend, and ci owns landing: no agent acts between a ready mark
  and `main` moving unless the automation refuses.

## Outside the tree

- **Work and its status** are Linear issues in the owner's team, each in a project and milestone; the projects, grouped
  by Goal, are the campaign map. Slack carries rings and conversation, never the brief or status (the coordinating
  skill).
- **Notes** (lane reports, checkpoints, findings, drafts, friction notes) are Markdown under
  `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md` with YAML front-matter (`id`, `campaign`, `lane`,
  `kind`, `status`, `repo`, `origin`), cited as `note:<id>` and published with `research notes sync --path lanes/<lane>`.
  Notes are records, never messages: a handoff, an order or an ask goes on Slack, or to your own workers as a follow-up.
  A Cursor agent store is its agent's scratch: nothing another agent needs lives only there, and nothing cites it.
- **Writing for a person** is a Notion page, or cites only what they can open: PRs, repository files, run ids and Notion
  pages. The notes repo is private, so a `note:` id is for agents.
- **Evidence and generated output** go in the evidence store, never Git: `research run` publishes every run as an
  Attempt, and anything else goes in with `research data put --kind K --file|--tree P --preserve`, cited as `art:<id>`
  (`tools/research/src/research/store/README.md`). Tables and reports render on demand from code plus the store.
  Findings about a run are labels: `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>`.
- **Architecture decisions** and Daniel's decisions log are in Notion.

## Coordinators and lanes

A lane's rulebook is `~/.research/notes/kb/LANE-CONTRACT.md`. Each coordinator, a handle's holder (`research msg
handles`), is a plain Cursor cloud agent on the environment "Verity compute-verification" that owns one vertical slice
and runs its own workers. `@top` is the admin: handovers, the roster, the quiet-lane sweep, Daniel's decisions log,
relays to Daniel, announcing and settling, and consolidation (work, records and notation converge, not multiply); it
does no lane work. Launching, handing over and archiving coordinators: `infra/cursor/README.md`.

- **Keep chatter low.** Every message and background shell reaches the coordinator and the human: run long jobs
  detached, and message someone only to unblock them or change their plan.
- **Workers run where they can't take you down**: as cloud subagents (`Task` with `environment: "cloud"`, a standing
  request for every agent here, so nobody asks first) or as detached `research run`s; a local subagent only for a short
  read-only lookup. A cloud worker boots from the environment's snapshot, so its brief is complete (the task, the
  done-condition, the deadline, "reread `git show origin/main:AGENTS.md`"), tells it to create
  `cursor/<lane>-<topic>-<hex>` from `origin/main` and push only there, and names an inbox branch it reads hourly
  (`git show origin/B:inbox/<agent-id>.md`); larger changes go in when you resume it. It opens its own PR, or reports
  its branch and SHA for its launcher to.
- **Reach a running subagent** on your VM through its next shell command, since it never rereads a shared file mid-run:
  `python3 tools/agent-guard/inbox.py send AGENT_ID --text '…'`. Agents on other VMs: through their coordinator on Slack.
- **Stay reachable.** Push and write a one-line checkpoint note at least every 60 minutes; when you'd wait on a run, end
  your turn naming its id and ETA, and your launcher resumes you. The control pod's agent watch flags a stalled or
  stopped agent and an out-of-memory kill. A worker resumed twice, whose base branch is gone, or silent past the watch's
  flag is relaunched fresh from its last note. Briefs ask for a checkpoint before long work and word red-team work
  neutrally.
- **Check what's being tried.** Before work another agent could plausibly be doing, read the notes' `kb/onboarding.md`
  and `research notes approaches`, and claim yours (`research notes claim`). A lane's night is one or two outcomes, each
  with a done-condition anyone can check, which its lead proposed and Daniel approved.
- **Work in a checkout no other agent uses** (your VM, or your own `git worktree` and venv), push only to branches you
  created or were handed, and push after every commit (never `main`, never `--force`).
- **Slack** (computeverification.slack.com) is for handle holders and named humans; a worker asks its own coordinator.
  The coordination API is `research msg` (the using-slack skill). Work for another lead goes in as a Linear issue in its
  team; Slack is the conversation around it.

## Working principles

- **Research velocity comes first.** Be generous with what lands and cut ceremony: propose cutting any gate, review or
  step that slows landing without catching real errors, and fix what breaks later. Our machines are research
  infrastructure; don't mimic deployment settings.
- **Waiting is a bug.** A step you repeat that makes you wait more than about 10 minutes is friction even when it never
  fails: fix it at the source, or take it to its owner that day with its numbers (minutes a wait, waits a day). Push
  back the same way, through your coordinator, on a rule that costs velocity, Daniel's included. A slow loop's owner
  fixes it ahead of other work.
- **Make every decision yourself except claims and Daniel's own clicks.** What a guarantee says or rests on, and what
  only an account he holds can do (the GitHub organization's admin, billing, the Nebius and Cloudflare consoles), go to
  Daniel through your coordinator, with your recommendation. Keys and access between our own machines and accounts are
  the lanes' call, set up with the machine's owner and no ask: this is a simulation, so it needn't be perfect. Spend is
  its lead's call. A change that crosses owners goes
  to the other owner, and the coordinator settles a disagreement. Fix what's in your way at the source when the fix is
  yours, and record each ruling you get (the friction skill).
- **Each lead owns its areas' size.** top's hourly consolidation reading tracks `main`'s size. A PR that adds code says
  in its body what it deletes or unifies, or why nothing can go.
- **Parked work gets done.** A lead starts each Backlog or Todo issue worth doing on its own worker and closes the rest
  with a reason. Being busy is no reason to park one. Set more goals, not fewer.
- **A ruling rolls out in the change that records it**, as AGENTS.md says; when code must change first, the ruling's
  relay names the owner and the PR that changes the docs with it.
