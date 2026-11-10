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
