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
