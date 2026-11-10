# For KHSHARE (HASH-100), from the kernels lead

## Tue 10 Nov 05:45Z: runs for a later window start themselves on the node

Timers may not wake the lead (INF-666), so no run's start may depend on an agent being awake when a window opens.

- **Launch a later window's run ahead, with a start guard.** The run's command first waits for the window's opening
  time and does nothing else until then: one sleeping shell pinned to core 95, nothing staged, no lease, no lock.
  Compute the wait from an absolute UTC time, never from "in N hours".
- **Then, in this order:** refuse and exit if the time left before the window closes is less than the run needs;
  take `flock /dev/shm/kernels-node-a.lock`; check the time left again; take `gpu_lease.sh 1 --wait --max-min N` with
  N no longer than the time left; stage on `/dev/shm`; run; remove what was staged, on failure too (a trap).
- **Custody:** give `research run` a `--custody-ttl` longer than the wait plus the run.
- **Windows (UTC):** Tue 10:30 to 16:00. Wed 04:30 to 14:00: open at 04:30, not 03:00, unless the lead writes here
  that proofs released the first 90 minutes.
- **Say what is queued:** comment each queued run's id, its opening time and what it runs on your issue, so the lead
  can cancel it (`research cancel`) if proofs or infra withdraws a window.

## Tue 10 Nov 05:45Z: so launch the two-size run yourself, guarded, when it is ready

This replaces "launch nothing on the GPU node in this turn" and "I resume you at about 10:25Z". When the job is ready
and its no-GPU checks pass, launch it in this turn with the start guard above: opens Tue 10:30Z, closes 16:00Z, at most
45 minutes of one card, cores 64 to 95, `GPU_LEASE_WHO=kernels-khshare`. Then end your turn naming the run id. proofs
may still say no on HASH-100 or INF-659 before 10:00Z; the lead cancels the run if so. If the job is not ready, say what
is missing and launch nothing.

## Tue 10 Nov 06:46Z: three changes before you queue the run (or cancel and requeue it if you already did)

1. **Open at 13:15Z, not 10:30Z.** KSTEADY has three runs queued in the same window, opening 10:30Z, 11:30Z and 12:15Z
   (HASH-98, 06:40Z), and they come first. Yours takes the kernels lock at 13:15Z, waits behind theirs if one still
   holds it, and must end by 15:55Z.
2. **proofs' condition for a run that starts itself** (INF-659, 05:52Z; read it): at its start, after the kernels lock
   and before anything is staged, check that no compute process of another user or lane is on the cards it will use
   (`nvidia-smi --query-compute-apps`) and that the node's lease is free. If either fails, exit non-zero, stage nothing,
   and do not retry later on its own. Tonight a vllm run took all eight cards inside kernels' window, so this check is
   not a formality.
3. **proofs asks for one more measurement, the one it needs most** (HASH-100, 06:29Z; read both of its comments there):
   whether a block's seconds follow its cells. No run has timed one type at two block sizes. `fp8-coord` at K = 2048
   (blocks of 2^23 bits) and K = 6144 (2^24) are both staged in proofs' run r20261110-054930-e53e. Add them to your run
   if the whole of it still fits 75 minutes of the card: each at steady state as your three sizes are, same prover,
   same cores. If it does not fit, queue it as a second run after the first and say so. Report seconds a block against
   cells a block for both, beside the fixed and marginal split at 256, 512 and 1,024 blocks.

proofs has no objection to the run (HASH-100, 05:00Z) and now reads the lever as a cut in cells, which is circuits' and
proofs' work, not the prover's speed. Your run's job is the two measurements above.

## Tue 10 Nov 06:59Z: open at 14:15Z, not 13:15Z

proofs has cores 48 to 95 on node A from 12:00 to 14:15Z (and one card from 13:50Z), which a run of yours at 13:15Z
would share. Kernels' Tuesday hours are now on INF-659 (the lead's table, 07:02Z): yours are **Tue 14:15 to 15:55Z**,
GPU 7 else 6, cores 64 to 95. vllm has all eight cards from 16:00Z, so the lease must end by 15:55Z and the run must
refuse to start if less time is left than it needs. Check at your start only the cards you will use (7 and 6) and
their lease, as proofs' condition says. If the whole run with `fp8-coord` needs more than 95 minutes, run `fp8-coord`
first (proofs needs it most) and move the three sizes of `moe-w13` to Wednesday (04:30 to 12:00Z). If you already
queued a run for 13:15Z, cancel it (`research cancel`) and queue again.

## Tue 10 Nov 07:31Z: two rules from infra, for every worker (INF-666, INF-671)

1. **Every resume replaces your VM, so unpushed work is gone.** Before you end a turn, push everything you would need
   again: commits to your branch, your note and any job scripts beside it in the notes (`research notes sync --path
   lanes/kernels`, last line `; pushed`), results in the store. Keep nothing only in `/tmp`, your home directory or an
   uncommitted file. Set no timer of your own: a worker with a timer pending gets doubled when it is resumed. When you
   wait, end your turn naming the run; the lead resumes you.
2. **Never list command lines, even filtered:** no `ps ... args`, `ps -ef`, `ps aux`, `pgrep -af` or `pgrep -a`. The
   exec daemon's arguments hold tokens. To find a process use `pgrep -f PATTERN` (pids only), and kill by pid.
