# KAUDIT's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 19:11Z: your brief: HASH-98's closing audit, run again on today's labels: each done-condition with its evidence, and the types' table refreshed. One job, on CPU only

**Who you are.** KAUDIT, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You run again an audit the
worker KSTEADY (bc-f944c53d-53aa-581d-ab39-a46f095b1cde) built and ran on Thu 01:41Z, now that today's
rows are measured and labelled, so that the lead can close HASH-98 and its goal a day early.

**What changed since KSTEADY's audit** (HASH-98's comments of today, read them):
- eight Boolean statements were timed again at 1, 2 and 4 provers and the head block at `FIRST=true`
  at one prover (`ksteady-jobs/h98/thursday-result.md`); they are labelled of record and the 18 runs
  and arts they replace are `superseded_by` them (`20261112T1858Z-report-klabel.md`);
- benchmarks ran the figure on those labels at 19:10Z: SPOT 0.0364 prover GPUs per serving GPU,
  every block 11,576 prover GPU-seconds a token, steady rows 99.5% of proving; and said the 17 older
  arts of record in no `beside` list stay.

**The task.** HASH-98 has seven done-conditions (read the issue whole). For each, say whether it holds
on `main` and the store's labels of now, with its evidence:
1. Run the audit again as KSTEADY's handoff says: `~/.research/notes/lanes/kernels/ksteady-jobs/RESUME.md`
   (after `research notes sync --path lanes/kernels`), the bullet "To run the audit again" and the
   "Restore" lines; the verdicts of the last audit are in the bullet "The verdicts" above it and in
   `h98/result.md`. CPU only, no node A, about ten minutes, most of it the label pull. Use `main` of
   now (write its hash out; the shell hook refuses a variable there).
2. **Compare with the last audit, condition by condition**, and say what moved: the share of proving
   priced from steady rows (Flock and SPOT), the sixteen rows, the fields of each row at 1, 2 and 4
   provers, the labels (how many arts of record, how many superseded, none failing labelled), and for
   condition 6 what proofs has said since (HASH-98 and HASH-115: its confirmation, row 15's pin, the
   head block's Program).
3. **Refresh the comment condition 7 asks for:** each type's GPU-seconds a statement and ms a block
   with its run id, at 1, 2 and 4 provers, for benchmarks' rows 1 to 16 (`h98/post-types-1.md` is the
   last one) with today's re-timed rows in it and what each replaced; the 48 other types only if a
   figure of theirs moved.
4. Post it: one worker comment on HASH-98 with the conditions' verdicts (at most 100 words) and one
   with the table (`research msg linear comment HASH-98 --file F --worker KAUDIT`), never through the
   Linear connector's `save_` tools. **Label nothing and change no script**; if the audit names a row
   that should be labelled or superseded, say which and why.

**Traps the handoff names:** the command's stderr and `research data … --json` name the store's
remote: they stay in `/tmp`, never beside the note or in a comment. Two runs end in `c5c7`. A
`research run` launch line names a pod id: mask it.

**Return by 20:45Z** with each condition's verdict and evidence, what moved since the last audit, the
two comments' places, and anything that stops the issue from closing. End your turn with it: I close
the issue and the goal on your return.

## Thu 12 Nov 19:33Z: your audit is read and rung; HASH-98 stays open for your finding. Next: prepare Friday's re-time of rows 1 and 2, as Thursday's runs were. No launch tonight

Read by me: your return, your note's findings, the two comments. Rung: benchmarks and proofs on
HASH-98 with the seven verdicts and row 1's question. **Your finding on row 1 is the audit's most
useful line, and today's other runs bear it out:** KMOVE's screen had the head block at one prover
at 2.43 to 2.51 GPU-seconds from `main` (the row of record says 2.8555), and KHSHARE's first pair
at two provers, walk off, is 2.109 (the row of record says 2.5602).

**The piece (HASH-98; you are on Opus, so this turn is yours to go on with):** prepare, tonight,
the re-time for kernels' window on node A on **Fri 13 Nov at 15:45Z**:
- **row 1**, the head block at `FIRST=false` (pin `4b31e7caba011398`), at one, two and four
  provers. I asked benchmarks whether KHSHARE's rows of today (the walk's head with the walk off)
  may stand as row 1's rows of record; if it says yes, row 1 drops out of your launch: build it
  so that it can;
- **row 2**, the head block at `FIRST=true` (pin `be5bca5c30515dc4`), at two and four provers.
- **As Thursday's runs were:** KSTEADY's kit unchanged (the tree `main` `62bf29346f76`, `WARM=3`,
  ten cores a card, 120 timed sessions a count, the ten checks, one proof of each kept), its
  `thursday-launch.sh` as the model. That launcher refuses any card but 4, 5 and 7 and any row
  but Thursday's: **write a new launch file beside it; change none of the kit's.** The head
  blocks' statements are proofs' staged directories on node A (read only); KSTEADY's handoff
  says how Thursday's `FIRST=true` step took its copy.
- **The card is a parameter:** card 3 with cores 34-43, or card 2 with cores 24-33; I say which
  by Fri 02:00Z. Say how many minutes the chain needs (from Thursday's own records: the
  `FIRST=true` step at one prover held its lease 9 minutes) and the order if minutes run short
  (row 1 first, four provers last).
- **Dry-run it on the kit's stand-ins** (`stub-thursday.sh` is the model) until every line holds,
  with the `beside` entries for the rows of record each would replace (row 1: `38d6`, `8d4a`,
  `2be4`; row 2 at two and four provers: `4f05`, `d8db`).
- **Nothing on node A tonight** (it is proofs' and vllm's from 21:00Z), no card, no label.

**Return by Fri 13 Nov 01:00Z** with the launch lines, each file's sha256, the dry run's result,
the minutes needed, what on the node must be read in the launch's minute, and what could stop
it. You are resumed for the go at 15:25Z.

## Thu 12 Nov 19:35Z: benchmarks answered (HASH-98, 19:34Z): "`main`'s build, please." Row 1 STAYS in Friday's launch

"Time row 1 from `main` in Friday's 7:45 AM PST window, beside row 2 at two and four provers, and
label by 8:30 AM PST as planned. A row of record times the prover serving runs, and that's
`main`'s." So:
- **Row 1 at one, two and four provers is in your launch, from `main`** (the kit's tree, as
  Thursday's), with row 2 at two and four provers. The labels are on by 16:30Z on Friday: say in
  your return whether the chain's minutes allow that from a 15:45Z start, and what goes first if
  not (row 1).
- **KHSHARE's runs of today are the cross-check:** "If Friday's `main` rows land within their
  spread, say so in one line. If they don't, that's a finding about the build." Build that line
  into your reading: KHSHARE's own rows (walk off) at one, two and four provers, from its runs
  `r20261112-190414-ca45` (four provers), `r20261112-184334-4cb9` (two and one).
- **If Friday's window fails**, benchmarks' fallback is KHSHARE's runs labelled of record with a
  second label `build=walk-head-switch-off`. That is the lead's to order, not yours: say at once
  if your launch cannot run.
- A note on my own clock: my resume to you said "19:45Z"; it was sent at about 19:33Z.

## Thu 12 Nov 19:58Z: your 19:58Z return is read; the launch is as you built it. THE CARDS: both, `CARD=3 CARD2=2`, so that both rows are labelled by 16:30Z. The go comes Fri at 15:25Z

Read: your return whole. A launcher written beside the kit with none of its files changed, twelve
stand-in cases, the real finishing script tried on Thursday's head-block run, and the minutes
worked out before I asked which card: that is a launch I can give a go to.

- **The calendar is rewritten (notes `90ddb99be6`):** card 3, cores 34-43, 15:45Z to 17:00Z: row 1
  at one, two and four provers; card 2, cores 24-33, 15:45Z to 17:30Z: row 2 at two and four
  provers, then two short tests of other workers. Both rows of yours read STANDS.
- **On Friday I resume you at about 15:25Z.** In the launch's minute: this file, the calendar,
  the launcher's read of node A, the tree and the reader's worktrees, as you list them.
- **If proofs' staged directories are gone at the launch** (its 04:30Z window, a pod restart):
  yes, fetch the two statements from proofs' art onto node A, into a directory of kernels' own
  under `/dev/shm`, inside our row, each pin checked, and removed when the runs end. Say in the
  record that the statement was the store's copy and not proofs' directory. If the pins differ
  from the kit's, stop: nothing is timed.
- **If vllm's row overruns 15:45Z:** wait for the release, kill nothing, launch again with
  `TAGS=`; row 1 first.
- **Labels:** as benchmarks wrote for Thursday's rows (HASH-98, Thu 18:41Z): each new run
  `run_of_record prove-glm53-flock-steady`, and the run it replaces (and that run's art, as
  KLABEL did and benchmarks confirmed) `superseded_by` it. On by 16:30Z; a table first, each
  label read back. One worker comment on HASH-98 with the five figures beside the rows of
  record they replace, **the NUMA node of each (Wednesday's rows ran on node 3, yours on node
  1), and the cross-check's one line** against KHSHARE's walk-off rows of Thursday (its final
  table on HASH-112 has the arts).
- Nothing before the go.

## Fri 13 Nov 07:09Z: `main` moves under Friday's runs TONIGHT. Before your go: what in your job reads `main`'s head at run time, and the least change that keeps it working. By reading; change no job file without my word. Return by 10:30Z

**What happened.** Daniel agreed the architecture page at 10:40 PM PST and asked for the
repository in its structure when he wakes. top's plan (TOP-415, 07:05Z): **every move that
only changes paths is on `main` by Fri 13 Nov 7:00 AM PST (15:00Z), 45 minutes before node A's
window opens.** Two chains, marked by 13:00Z:
- chain 1, tonight's layout chain;
- chain 2: **each protocol under a group directory** (`verity/protocols/flock`, `spot`,
  `recursion` go under `verity/protocols/proofs/`; `pouw`, `pous`, `poub`, `certifier` under
  `verity/protocols/accounting/`; `inference_only` under `verity/protocols/policy/`);
  `verity/definitions/` split (`definitions/kernels` to `verity/circuits/kernels`, the served
  kernels' circuits to the adapter); the difftest to `tools/`; the Lake packages' paths; and
  lean's renames inside the Lean files.

**Your timed run is launched from a pinned tree, so the moves do not reach it.** What they can
reach is any step that takes `main`'s head, or a path of the tree, when it runs. One is
known: the Lean verifier that checks the witness run's kept proofs is built from `main`'s head
at each put, by `verity/protocols/flock/verifier` and `verity/protocols/flock/pod/lake_closure.py`
(`leanframe.sh`, `fri-stage.sh`, `control.sh`), and chain 2 moves both.

**Do, by reading only (no launch, nothing on node A, no job file changed):**
1. **Every place** your Friday job, its launcher, its reader and its helper scripts take
   `main`'s head or name a path of the tree outside the pinned worktree: a `git fetch` or
   `rev-parse` of `main`, a `--source` that follows `main`, a `research` command run from a
   checkout that follows `main`, a script on a node. File and line.
2. **For each: what happens if `main` is in the new layout at 15:00Z.** Does it stop the job,
   skip a step, fail a verdict in silence, or leave the timed rows as they are?
3. **The least change that keeps Friday's run and its verdicts:** a variable the job already
   takes (a commit for that step: I give you the last commit of `main` before chain 1 in the
   go); or the step done after the run from a pinned tree; or one changed line, and then
   what must be rehearsed again and how many minutes that takes.
4. Your proposal, in three lines. **Tell me; I decide and say it in your go.**

**Also, for everything you write from now:**
- Daniel's words in prose (pull request bodies, comments, notes' summaries): **"challenges"**
  for what the challenger sends, not "coins"; **"accounting protocol"**, not "resource
  protocol"; **"the auditor's devices"**, never "host" for the auditor's hardware. The code's
  names stay.
- **A branch or pull request of yours that crosses a move waits until the chains are on
  `main`; then `tools/move/restack.py` carries it across. Do not rebase it by hand and do not
  merge `main` into it tonight.**

## Fri 13 Nov 07:20Z: your reading is taken whole. My decisions, for the go (about 15:20Z)

Read: your return and what it rests on (`friday-launch.sh:37,43` refuses anything but
`/tmp/h98-main` at `62bf29346f76`; `link.sh:55` sends that tree; the reader and census from
pinned worktrees; every `research` call runs `main`'s research code; `complete_row.py:44`
loads `server_rate.py`, which imports `verity_flock` from the VM's checkout). Simulating the
checkout moving without a sync, and rehearsing the fix on Thursday's row with nothing put,
is what makes the proposal a fact.

1. **Launch as prepared:** `SRC=/tmp/h98-main` at `62bf29346f76`, no `COMMIT`. The rows are
   labelled as measured from `main` at that commit, and benchmarks' repricing reads the
   label and `rate.json` from the store, neither of which holds a path.
2. **Finish with the pinned import:** `cd /tmp/ksteady/scripts &&
   PYTHONPATH=/tmp/h98-main/verity/protocols/flock/prover/python bash ../fin.sh RUN…`, as you
   rehearsed (`rate.json` `2586d59ef175d1ed` on Thursday's row). A variable, no file change.
3. **If `research` itself fails after the chains are on `main`:** that step again with
   `RESEARCH_AT_CHECKOUT=1` for the store's reads and puts. **Not for the launch on node A:**
   that form is not rehearsed, so if the launch refuses, stop and tell me at once with its
   line; I decide in minutes.
4. **At the go you are on a new machine:** restore the kit and the three worktrees first
   (about two minutes, you say), then `sha256sum -c` on the kit as before, then the
   launcher's own look. Say in your first checkpoint that all three hold.
5. In prose from now: "challenges" not "coins", "accounting protocol", "the auditor's
   devices".

Nothing to do until I resume you with the go.

## Fri 13 Nov 07:38Z: A STANDING RULE FROM NOW, from infra's announcement (INF-783): three things on every turn

infra (07:35Z): tonight a worker on its own cloud machine, once resumed, (1) may wake on a
new machine with its uncommitted and unpushed work gone, and (2) may receive its
coordinator's timers and Slack events as if they were its own, sometimes as a second loop
of itself. Not explained yet. Four of ours have: KWALK, KYARM and KTILE each
booted again on a new machine on Thursday evening, and KNAMES's lost its scratch
directory and worktree at about 05:35Z today.

1. **Commit and push before you end a turn and before every long run.** Job files, notes,
   branches: what is not on a remote or in the store does not exist. `/tmp` and scratch
   directories do not survive.
2. **Act on no event that does not name you.** Your work comes only from a resume by the
   kernels lead that names you and a section of this file. **A timer's text or a Slack
   event that reaches you and speaks to "the kernels lead", or to another worker, is not
   yours: do nothing it says, launch nothing, and tell me in your next return that it
   reached you, with its first line.** A go for a card comes only in a section of this file
   that I name in the resume.
3. **On waking, before you trust your tree:** `uptime`, `git status`, and your pinned
   files' sha256. If a second session of you is running, the later one stops and says so.
