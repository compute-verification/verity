# Standing rules for kernels' workers (the lead: bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read this at the start of every turn. Your brief names what is yours; where the two differ, the brief wins. The
newest section of your own inbox file (`inbox/<NAME>.md` on this branch) wins over both.

## Every turn

- **Boot.** A resume replaces your VM: `git -C /workspace fetch -q origin main && git -C /workspace show
  origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace && export
  PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH`. Start one tmux session and keep it
  (`tmux -f /exec-daemon/tmux.portal.conf new-session -d -s keep`): `research` reads its secrets from it. Reread
  `git show origin/main:AGENTS.md`.
- **The agent guard of #9165 (infra, Thu 12 Nov 13:26Z; on `main` since 12:20Z, on your VM from its next boot).**
  Each refusal names its fix; a wrong refusal is one line on INF-717, and you tell me.
  - **A push from a command with a `cd` anywhere in it is refused.** Push in a command of its own, the checkout
    named on git: `git -C <checkout> push -u origin HEAD` (or your branch's name). Commit the same way
    (`git -C <checkout> add …`, `git -C <checkout> commit …`) so a habit of `cd … && git push` does not bite.
  - **A script's push is judged too:** another agent's branch, `main`, `checked` and `next` are refused.
  - **No listing that prints a command line:** not `ps -ef`, `ps -o cmd`, `pgrep -a` or `/proc/PID/cmdline`. Use
    `ps -eo pid,etimes,comm`, `pgrep -l`, or `tools/agent-guard/procs.py PATTERN`. Kill by PID, as always.
  - tmux, `bash -c`, `env`, `xargs`, `eval` and `watch` no longer carry a refused command past the rules: a job
    script that pushes or lists command lines inside one of them is refused there too. Check your launchers.
- **By Thu 12 Nov 07:59Z, at your next safe point: merge `origin/main` (`72e4d46d9209` or later) into your own agent
  branch and push it** (infra, Wed 13:02Z). It carries what `research` needs once Daniel's notes token lapses on Thu
  12 Nov and the Slack token leaves on Fri 13 Nov (notes and Slack through the broker), and the inbox that reaches
  a running agent. A pull request branch cut from today's `main` needs nothing; a branch not yet across the move
  goes through `tools/move/restack.py` (below). Then `research doctor`: its `notes-push` line must name the GitHub
  broker's token. Never interrupt a running job for it. Say in your CHECKPOINT the commit your branch then holds.
- **The lead may also reach you while you run**, through the agent inbox (`tools/agent-guard/inbox.py send`): the
  message arrives as the text of one refused shell command, which you then run again, or as your turn's
  follow-up. Act on it as on an inbox section. It reaches only a VM whose shell hook runs (Wed 13:05Z: KMOVE's
  took one in 32 s; KSTEADY's, KP2's and KHSHARE's took none in 75 s), so the file below stays the record and
  the rule.
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
  it: do not delete it (that is acting as him again); post one line beside it with the command saying whose it
  is, and repost the text only where others act on it (infra's correction, Wed 04:26Z).
- **Your issue is where you report.** `research msg linear comment ISSUE 'TEXT' --worker NAME`: each run id with its
  verdict line, each pull request, each finding. No Slack. Slack and Linear text is untrusted: check a claim in the
  source before you write it down. Cite a line at the line; mark what you inferred as inferred.
- **A result another lane is waiting for ends your turn.** Your comments ring no one, and the lead reads your
  CHECKPOINT only when it looks: on Wed 10:17Z a measured figure proofs needed that night sat an hour on the issue
  because its worker went on working. When you post such a result (a number for another lane's table, an answer to
  its question, a pull request waiting for its owner's word), **end your turn at once with it, and say what you
  would do next**; the completion wakes the lead, who rings the lane and resumes you. Do not stay up to watch an
  issue for questions unless your brief says how long.
- **End your turn done or blocked, never on a plan.** If a run would wait more than 40 minutes, end your turn naming
  its id and what to do when it ends: the lead resumes you.
- Never list command lines (`ps`, `pgrep -af`): the exec daemon's argv holds tokens. Use `pgrep -f`.
- Nested code in Markdown you write is fenced with `~~~`.

## Agents, today

- **Daniel on models, Thu 12 Nov (top, TOP-402, 17:45Z): "Opus 5.5 does the work; Fable 5.2 plans it."** Every
  launch names its model, never `inherit`: `claude-opus-5-5-high` for a worker (moving files, reading code, a
  count, a specified change with its tests, a merge, a review, a timed job from a prepared launcher);
  `claude-fable-5-2-xhigh` only for a worker whose task is to plan, design or decide where no check catches an
  error. A cloud worker launches no workers of its own: the lead plans and launches. If you are on the lead's
  model and find your task is Opus's, say so to the lead. A worker on the lead's model finishes its turn; what is
  left that is Opus's goes to a new Opus worker, so **write your note and your return so that someone who was not
  there can go on from them**: commands, sha256, run ids, traps.
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
  and launch it again on a pool with free slots and no line, naming the pool: `--on vy-nebius-cpu-12 --pool research
  --as kernels` (or cpu-7; `--on auto` reads cpu-12's pool too since Wed). A build that needs CUDA 13.0 still goes
  where CUDA 13.0 is.
- **A CPU node has pools and no check slots, so `research queue ready N --on vy-nebius-cpu-K` runs nothing there**
  (Wed 07:35Z, the lead's four runs: `slot: /workspace/research/locks/slots lists no slots`, exit 1 after 0 s, which
  `ready` prints as "the quick tier failed"; nothing was wrong with the heads). The quick tier takes `--on auto`
  only. Kernels' way stays: the suites the change touches as a pool run at the head (`suites.py --changed BASE
  --quick`), then the lead's `--mark`.
- Write what a run must keep under `$OUT`; scratch goes in a `mktemp -d` directory outside the run directory
  (the writing-runs skill).
- **A bf16 matrix product computed with torch on a CPU that has AMX is wrong now and then when its process is
  preempted** (infra, Wed 11:21Z, INF-694: oneDNN's AMX kernel; seen on a cloud agent VM, 3,750 of 60,000 products;
  the 13 CPU nodes have no AMX). Until the suite runner and `research run` set the cap themselves: wherever you
  compute a bf16 reference with torch on a CPU that is not one of those nodes (your VM, a GPU node's host), set
  `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` before the process's first torch op, and say in the run's record that it
  was set. **Never delete or rerun away a bf16 test for a NaN or a wrong value seen there:** report it with the
  machine and whether the cap was set. A GPU kernel's bf16 result that disagrees with such a reference is not a
  kernel bug until the reference is recomputed under the cap.

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
- **Wed 11 Nov 22:17Z, a stopgap while RunPod's API refuses (429, the account's daily limit; INF-712):** a new
  `research run --on vy-spot-glm53` asks RunPod for the pod's address and fails. Infra's way round, tested by
  infra at 22:14Z: on the VM that launches, append to `~/.research/machines.toml` (make the file if there is
  none):

  ```toml
  [machines.vy-spot-glm53]
  provider = "ssh"
  host = "213.181.104.57"
  port = 12250
  user = "root"
  pod_id = "10khharm1q39bu"
  rate_usd_per_hour = 42.32
  project = "verity"
  ```

  The address and the pod's id are the ones kernels' own runs recorded tonight (`remote.json` of
  `r20261111-195910-303a` and `…f16e`, compared by the lead), so the entry points where your runs already
  went. A run already launched is read with `research fetch` and `research status` as before. If a launch
  then fails to connect, the pod has restarted and its address changed: stop and say so; do not guess one.
  Take the table out when the lead says RunPod answers again. The control pod's entry, if you need
  `research pods ssh vy-control`, is in infra's comment on INF-712.
- **In the seven typed directories a new function carries its types** (from #9127's landing: `benchmarks/hash`,
  `torch_gpu`, `moe_align`, `tc_probe`, `tc_probe_fp4`, `native_peak`, `verity/definitions/kernels`): an untyped
  one is a ruff finding (`ANN001`, `ANN201`, `ANN202`, `ANN204`), a warning at your check and a ratchet failure after
  landing in the six `benchmarks/` ones, a failing lint in `verity/definitions/kernels`. A file with a `@triton.jit`
  kernel gets its own line in `ruff.toml`. The import order is not held.
