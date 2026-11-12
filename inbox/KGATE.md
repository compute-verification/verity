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
