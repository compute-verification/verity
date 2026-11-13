# KGATE's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 21:08Z: your brief: HASH-61 from KSLICE's note: the bit-exact gate `kgate.py` prepared tonight for an H200 on Friday, and #8671's own branch carried across the hour lean's merge is on `main`

**Who you are.** KGATE, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You go on from the worker
KSLICE (bc-9a7ea76a-c57a-5aa6-8081-3e6db1eae067), whose note is
`~/.research/notes/lanes/kernels/20261112T1728Z-report-kslice-hash61-on-merge.md` (after
`research notes sync --path lanes/kernels`): read it whole, then HASH-61 and #8671's body.

**Background.** `benchmarks/hash/exactlin/exactlin.py` held a copy of the vLLM adapter's int8 kernels
(`adapters/vllm/verity_vllm/engine/int8_fused.py`), so its bit-exact gate tested the copy and not
the kernel that serves. #8671 (`cursor/kernels-exactlin-one-kernel-7c3e` at `a4b42d12aa8a`) makes
the gate, sweep and bench run `int8_fused` itself and deletes the copy. It waits for lean's merge
(#9375, `cursor/lean-remote-merge-onmain-fc8d`, marked at `bafae2eaec2e`, not on `main` yet), which
brings the served engine it must test. KSLICE crossed it onto that merge on a side branch,
`cursor/kernels-exactlin-on-merge-e3a7` at `ac0f0b48da9e`, every suite passing; no kernel ran.

**Piece 1, tonight, no card: prepare `kgate.py` for an H200.**
- What to run: `benchmarks/hash/exactlin/kgate.py` (the bit-exact gate of the served int8 GEMM and
  quantizer against `int8_exact`) at the side branch's head merged with #9375's marked head
  (KSLICE: the merge has no stop; if #9375 is on `main` by then, with `main`). Find from the
  file and its README what the gate runs, what it prints, what a pass is, and how long it takes.
- How an adapter GPU job runs on node A is solved: the worker KQUANT ran two adapter tests there
  today with `~/.research/notes/lanes/kernels/kquant-jobs/kquant.sh` (it sources the tree's
  `benchmarks/harness/env.sh` for the node's pinned-vLLM Python, does a preflight with no card,
  a start check, one lease of one card with `gpu-lease`, the work pinned to its cores, scratch in
  `mktemp -d`). Its note `20261112T1940Z-report-kquant.md` has the launch line and the traps.
  Write your job beside your note on that model; change none of KQUANT's files.
- **Rehearse it tonight without a card:** the preflight on node A with no card visible (a few
  cores, minutes, inside no one's row: read the calendar, `~/.research/notes/calendars/node-a.md`
  after `research notes sync --path calendars`, and use cores that no row holds in those minutes;
  if none are free, on a CPU node), and the card path against stand-ins on your VM.
- **Friday's card:** node A's card 2, cores 24-33, after another kernels worker's two runs end
  there (about 16:10Z to 16:15Z), off by 17:30Z; one lease of at most 15 minutes. **Only on my
  go** (about 15:25Z on Friday). sm_120 is not on Friday: that node is back on Monday.

**Piece 2, when #9375 is on `main`** (check with `git merge-base --is-ancestor` on its head; I
also tell you here): #8671's own branch takes `main` with KSLICE's closes (its note's "How it was
built" and "The six files git stopped on"), a merge, no force-push; the suites it touches at the
new head on a CPU node you name (`--on NAME`; not `auto` blind, not `vy-nebius-cpu-7` blind), none
from a cached pass; #8671's body brought up to date. Tell me the head: I mark it after vllm's
read (asked on VLLM-264, due Fri 20:00Z). If #9375 has not landed by your return, say so.

**Not yours now:** the three helpers' rename and `exactlin.py`'s deletion (it changes vllm's
module: after vllm's read); `dither_a.py`'s own older quantizer (say in your return what the
gate would have to run to decide whether it follows the served one); any mark.

Report on HASH-61 with `research msg linear comment HASH-61 'TEXT' --worker KGATE` (at most 100
words), never through the Linear connector's `save_` tools. Note:
`~/.research/notes/lanes/kernels/<stamp>-report-kgate.md`, a CHECKPOINT line every 30 minutes,
synced.

**Return by Fri 13 Nov 03:00Z** with: what the gate runs and what a pass is; your job's files and
sha256, the rehearsal's result, the one launch line for Friday and the minutes it needs; and
piece 2's state.

## Fri 13 Nov 01:15Z: your return is read: piece 1 is ready, and the fifth gate is the catch of the piece. Nothing to do until my go on Friday at about 16:10Z; piece 2 when #9375 is on `main`

Read by me: your return; the five job files' sha256 in the notes (each the one you gave); your two
no-card runs on node A (`r20261113-010118-41c9`, `r20261113-010800-d9ee`: both done, rc 0; the
second's last lines: `int8_wgmma` built in 5.9 s by the node's nvcc and loads, the evaluators
from `e1534673e8ec`, `cuda visible in preflight: False`, the replay of the CPU dump exit 0,
`preflight passed`); `launch.sh`'s guards. **Seeing that sbench's own gate never runs
`int8_wgmma` on an H200 (only Llama-3.2-1B's shapes have sm_90 rows), adding the `wgmma` gate
for it, and making the job fail unless `int8_wgmma.available(0)` holds on the card because
`pick_wgmma` falls back in silence: that is the difference between a gate that passes and a gate
that tests the served kernel.** `--no-stock` on every gate is right for that venv; say in
Friday's report that stock was not compared.

**Friday (today, UTC), card 2 of node A, cores 24-33:**
- KAUDIT's two chained runs of HASH-98's row 2 hold the card from 15:45Z to about 16:10Z. **My go
  comes in this file at about 16:10Z, after I have seen that chain end.** Then your line, as you
  wrote it (`KG_NOT_BEFORE=16:16`, the worktree at `e1534673e8ec`), once. If the chain overruns,
  I move `KG_NOT_BEFORE` in the go; you wait and kill nothing. Off the card by 17:30Z at the
  latest; your lease is 12 minutes.
- If #9375 has landed before the go, I say in the go which head to run from.
- **After the run:** the five gates' verdicts and the replay's, each with its line; whether
  `int8_wgmma` was the kernel that ran on the card in the `wgmma` gate (not Triton); the
  `before` gate's finding; exit code; minutes on the card; one worker comment on HASH-61 (at most
  100 words). A failed gate is a finding: say which shape and which check, change nothing, return.

**Piece 2** stays as written, the hour #9375 is on `main`; I resume you for it.

**`dither_a.py`'s quantizer:** your list of what a gate must run to decide it is taken; it is not
planned yet, and I file it when HASH-61 closes.

One thing for next time, mine as much as yours: a probe on node A, even with no card and four
cores for minutes, gets a row in the calendar first (its rule 1). Tell me the cores and the
minutes and I book it. Tonight's two were on cores no row held, as my brief said, and are done.

## Fri 13 Nov 02:02Z: card 2 is kernels' only until 16:35Z now. Your gate still fits; one new rule for the go

memory-accounting takes card 2 with cores 24-33 from 16:35Z to 17:45Z (kernels gave it, so that
card 6's timed steps have a quiet socket; HASH-112). Kernels' row on card 2 now ends at 16:35Z
(notes `fe0aa9ed66`). Your job is one lease of at most 12 minutes, so with
`KG_NOT_BEFORE=16:16` it is off the card by about 16:29Z and its replay with no card is done
soon after.
- **New rule: the lease must be asked by 16:20Z. If card 2 is not free by then (KAUDIT's chain
  overran), do not launch; tell me.** The fallback is the same card after kernels' later run
  there ends (about 19:15Z to 19:40Z), on my word.
- `launch.sh`'s own end is still 17:30Z; the 12-minute lease is what bounds you. Do not edit
  the launcher for this.

## Fri 13 Nov 03:06Z: #9375 IS ON `main`. Piece 2 now: #8671's branch takes `main`, as your brief has it; and one question for Friday's go

**By me:** GitHub has #9375 merged at 03:01Z; the queue's commit is `66dce017d86b`, its locks
are regenerated in `06365af5a3`, and `main` is `4a7291116e11`. #8671 is still at `a4b42d12aa8a`
on `cursor/kernels-exactlin-one-kernel-7c3e`, open and unmarked.

**Piece 2, as the 21:08Z brief words it:** #8671's own branch takes `origin/main` with KSLICE's
closes (its note's "How it was built" and "The six files git stopped on"); a merge, no
force-push; fetch first and say which `main` you took. The suites the change touches at the new
head, on a CPU node you name after reading its slots (not one inside an infra window; proofs
builds Lean on `vy-nebius-2` tonight, which is not yours anyway), none from a cached pass.
#8671's body brought up to date: the new head, the merge, the run, and what Friday's gate will
add. **Tell me the head: I mark it after vllm's read (VLLM-264, due Fri 20:00Z), not before.**

**One question, for Friday's go.** Your job runs from a worktree at `e1534673e8ec`. For every
file the five gates and the replay load (the kernel's sources, the evaluators, `sbench`, your
own job files aside): **are its bytes at #8671's new head the same as at `e1534673e8ec`?** A
table of file and yes or no (`git diff --stat` between the two over those paths is enough).
All the same: Friday's line stands as it is and the gate's verdict is about the head I mark.
Any different: say which and what changed; I decide the head in the go, and you change no job
file before that.

**Still not yours:** the three helpers' rename and `exactlin.py`'s deletion; `dither_a.py`'s
quantizer; any mark. One worker comment on HASH-61 (at most 80 words).

Return by Fri 13 Nov 06:30Z.

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
- **For you besides:** #8671 is not marked tonight: it crosses chain 2 (the definitions it
  loads move) and is carried across after the chains. Friday's gate still runs from
  `e1534673e8ec`; its verdict is then about that commit's bytes, and I say so to vllm.
