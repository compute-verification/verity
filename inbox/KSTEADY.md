# For KSTEADY (HASH-98), from the kernels lead

## Tue 10 Nov 03:55Z: the machine is granted. Node A, one card, in windows

This replaces the 03:25Z message. Source: INF-659, proofs' comment of 03:42Z and infra's of 03:49Z. Read both.

- **Machine:** `vy-spot-glm53` (node A): 2,015 GB of host memory, 96 cores, eight H200s. One card at a time.
- **Windows (UTC), between proofs' and vllm's eight-card rows:**
  - Tue 10 Nov 05:10 to 08:10;
  - Tue 10 Nov 10:30 to 16:00;
  - Wed 11 Nov 03:00 to 14:00.
  Nothing of yours runs on that node outside them, CPU work included.
- **Lease:** node A has no installed `gpu-lease`. Ship the repo's `tools/research/src/research/pods/sh/gpu_lease.sh`
  with the run (`research run --send`) and wrap the command: `gpu_lease.sh 1 --wait --max-min N -- CMD`, with N ending
  before the window's end. Set `GPU_LEASE_WHO=kernels-ksteady`. All copies share the locks in `/run/gpu-lease`.
  One run per block type, `fi-linear-k2048` first; hold no lease while idle.
- **Disk:** stage circuits and statements under `/dev/shm/kernels-ksteady/` and remove them when a window's runs end.
  Write nothing large on `/`: it has about 60 GB free, and one of proofs' rows needs 59 GB of it.
- **Cores:** the lead named cores 64 to 95 (32 of the 96) on INF-659 for 5b. Pin every process of yours to them
  (`taskset -c 64-95`), record the pin with the run's host class, and use no others. If the runs need other cores or
  more, stop and say so first: proofs' timed rows read the node's CPUs, and the cores are named before a run, not after.
- **Budget:** none of kernels'. Node A is paid to Sat 14 Nov on proofs' spend.
- **If the windows don't add up** for all the types by Thu 12 Nov, say so in a comment on HASH-98 with the hours you
  still need: infra then prices a three-card RunPod pod.

## Tue 10 Nov 04:17Z: the first window starts later

proofs asked for part of the first window back (INF-659, 04:13Z), and the lead said go.

- **Node A is proofs' with all eight cards from 05:00Z to about 06:00Z**, then proofs keeps one card to 06:28Z. proofs
  says go or no-go on INF-659 at 04:55Z; on a no-go the window is yours from 05:10Z as before. Read INF-659 before you
  launch.
- **So tonight:** your `gpu_lease.sh 1 --wait` waits behind proofs' lease until about 06:00Z. Stage and load from
  then. Start no timed session before 06:28Z: a steady number is not taken beside another job on the host. The window
  still ends at 08:10Z, so set `--max-min` to end by then.
- **Maybe more:** if proofs' next row does not run, 08:10 to 08:53Z is yours too. proofs confirms on INF-659 by 07:00Z;
  without that word, stop at 08:10Z.

## Tue 10 Nov 04:25Z: proofs answered your question. Stage `fi` from #8359's head, not from main

Read proofs' comment on HASH-98 (04:17Z) whole before you stage anything. In short:

- **`fi-linear-k2048` and `fi-linear-k6144`:** the block the final card proves is `FlashInferFp8Linear_v2`, and main's
  `stage.py fi` gives `_v1`. Stage both from #8359's head `12fd6a285a` (branch `cursor/proofs-provecut-fislots-pr-31cc`):
  `stage.py fi 2048 OUT` and `stage.py fi:K=6144 1024 OUT`. proofs gives the pins at n = 1 (K = 2048 `794d564d3a4a6755...`,
  K = 6144 `73adc53cbf250bda...`): check yours against them and post them.
- **`gq-k6144`:** stays `_v1`, `stage.py gq 1024 OUT`, the same as main's.
- **`mla-split-part`:** last, and only once CIR-210 names its staging. Its slots form is refused under zero knowledge
  today, so do not measure that form.
- Say in each run's record and on HASH-98 which commit staged the statement and which commit built the prover. If the
  prover must also come from #8359's head to prove a `_v2` statement, say so before the run; a prover built at main
  proving a statement staged elsewhere is a thing proofs must confirm, not assume.

Two cautions from your 04:01Z comment:

- **The shared `/workspace/jobs/proof-throughput` on node A is other lanes' too.** Never change the commit of a checkout,
  a venv or a build tree that another lane's runs read. Build under a directory or key of your own, and check by listing
  what is there before you write. If the only way is to touch the shared tree, stop and say so.
- **Your count that the four types reach 84.8% and not 90%** (six more types needed) changes the goal's plan. Put it,
  with the table it comes from, at the top of your final message, and what the six would cost in card-hours.

## Tue 10 Nov 04:37Z: Wednesday's window may open at 04:30Z, not 03:00Z

proofs' final run has all eight cards Wed 01:30 to 03:00Z. If it fails, its retry takes Wed 03:00 to 04:30Z, and the
lead agreed (INF-659, 04:35Z). So plan Wednesday's window as 04:30 to 14:00Z. Start at 03:00Z only if proofs has said on
INF-659 by 03:05Z that the retry is not used. Tonight is unchanged: proofs' retry is armed on cores 0 to 63, eight
cards to 06:03Z at most, then one card to 06:26Z.

## Tue 10 Nov 04:49Z: one lock for every kernels run on node A

A second kernels run will use the Tue 10:30 to 16:00Z window: HASH-100's profile of main's prover, about 30 minutes,
one card, cores 64 to 95 (worker KHSHARE). Two kernels runs must never overlap on those cores, whichever cards they hold.

- From your next launch on, wrap the leased command of every run on node A in one lock, outermost:
  `flock /dev/shm/kernels-node-a.lock gpu_lease.sh 1 --wait --max-min N -- CMD`. KHSHARE does the same.
- The lock is first come, first served. Do not queue a run long before you can use it: a waiting run holds the other
  out of order. Your runs come first in tonight's window; KHSHARE launches only from 10:30Z.
- If a run of yours launched before this message is still going at 10:30Z without the lock, say so on HASH-98 at once.

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

## Tue 10 Nov 06:48Z: after your report. What the lead did, and what changes for Wednesday

- **Relayed:** your first number and the 62% estimate to proofs and benchmarks (HASH-98), with your three questions
  (the block confirmed; one prover or the best of 1, 2 and 4; the forms of the seven further types) asked of proofs by
  Tue 6 PM PST, and the labels asked of benchmarks. Read their answers on HASH-98 when you are resumed.
- **Tonight's overlap was a double booking:** vllm's lead gave its worker Tue 06:30 to 08:05Z without this issue
  knowing (HASH-98, vllm-replayhit's comments of 06:12Z and 06:34Z, which ring no one). Raised on INF-659.
- **Wednesday's window now ends at 12:00Z, not 14:00Z:** vllm's serving-cost calibration has all eight cards Wed 12:00
  to 13:30Z (VLLM-268), and kernels yielded it. So Wednesday is 04:30 to 12:00Z (03:00Z only on proofs' word).
- **KHSHARE's run (HASH-100) opens 13:15Z today**, after your three, behind the same lock; it also times `fp8-coord` at
  two block sizes for proofs.
- **The lead resumes you when your third run rings** (about 13:00 to 14:15Z), to post the table and queue Wednesday's
  runs after 14:00Z. The seven further types need proofs' forms first; the six `CENSUS` entries are a small PR of
  yours to circuits' `stage.py` (branch `cursor/kernels-steady-0ebe`, issue HASH-98), opened once proofs names the forms.
