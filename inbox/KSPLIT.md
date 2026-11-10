# For KSPLIT (HASH-101), from the kernels lead

## Tue 10 Nov 06:41Z: benchmarks' answer on where the table goes

benchmarks, the kit's owner, said yes (Slack thread 1794292569.687439, 06:40Z): `benchmarks/harness/verity_bench/nsys.py`
is the place for the per-kernel table.

- Its other kit calls land Thursday in `kit.py` (lease, host stamp, paired intervals, writer), which imports `nsys.py`
  and does not edit it, so your PR should not conflict.
- **Keep the one-sided SM-time cross check as it is.** Add your block; change nothing that exists in the file.

## Tue 10 Nov 07:31Z: two rules from infra, for every worker (INF-666, INF-671)

1. **Every resume replaces your VM, so unpushed work is gone.** Before you end a turn, push everything you would need
   again: commits to your branch, your note and any job scripts beside it in the notes (`research notes sync --path
   lanes/kernels`, last line `; pushed`), results in the store. Keep nothing only in `/tmp`, your home directory or an
   uncommitted file. Set no timer of your own: a worker with a timer pending gets doubled when it is resumed. When you
   wait, end your turn naming the run; the lead resumes you.
2. **Never list command lines, even filtered:** no `ps ... args`, `ps -ef`, `ps aux`, `pgrep -af` or `pgrep -a`. The
   exec daemon's arguments hold tokens. To find a process use `pgrep -f PATTERN` (pids only), and kill by pid.

## Tue 10 Nov 08:07Z: PARKED. Mark nothing, run nothing

Daniel pushed goal 4 back (top on TOP-394, 07:50Z): a fast check and the repository's new layout come first, and until
the moves land only landing fixes, the maps and cleanup are marked. #8573 stays open and unmarked at d2ffef178d. Both
of your suites runs ended rc 0. The lead resumes you after the moves land, to restack, recheck and mark. If you are
running before that for any reason: write one line in your note, launch nothing, and end your turn.

## Tue 10 Nov 17:50Z: #8573 is marked by the lead; no restack

The layout does not land today (top, TOP-394 17:39Z), and ci lands pull requests as usual again. The lead marked
#8573 at d2ffef178d at 17:43Z: its two suites runs pass at that head (…93b2: `verity-bench` 172; …30fc:
`verity-graph` 168, `repository` 280, `verity-bench` 172), `main` has not touched its two files since its base, and
it merges clean. Push nothing to it. Nothing is asked of you now.
