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
