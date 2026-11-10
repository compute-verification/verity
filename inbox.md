[2026-11-10T03:21Z] dbf658572d55
From the kernels lead: read inbox/KSTEADY.md on origin/cursor/verity-kernels-0ebe (the machine for 5b is probably node A, one card through gpu-lease 1 between proofs rows; plan one short run per block type; launch nothing there before a yes on INF-659).

[2026-11-10T03:51Z] bc7bfc595953
From the kernels lead: the machine for 5b is granted (INF-659, 03:49Z). Read inbox/KSTEADY.md on origin/cursor/verity-kernels-0ebe now: node A, one card, windows Tue 05:10-08:10Z, Tue 10:30-16:00Z, Wed 03:00-14:00Z; gpu_lease.sh 1 --wait --max-min N; stage on /dev/shm; pin to cores 64-95.

[2026-11-10T04:14Z] dfa4a6cb1112
From the kernels lead: the first window on node A starts later. proofs has all eight cards 05:00Z to about 06:00Z and one card to 06:28Z (INF-659, go or no-go there at 04:55Z). Your lease waits behind it; stage and load from about 06:00Z, no timed session before 06:28Z, stop by 08:10Z. Details: inbox/KSTEADY.md on origin/cursor/verity-kernels-0ebe.

[2026-11-10T04:18Z] 3da5283ed840
From the kernels lead, before your first run: proofs answered on HASH-98 (04:17Z). Stage fi (K=2048 and K=6144) from PR 8359 head 12fd6a285a, not main: the final card proves FlashInferFp8Linear_v2 and main stages _v1. gq as on main. mla-split-part last, only once CIR-210 names its staging. Also: never change a checkout, venv or build tree in the shared /workspace/jobs/proof-throughput that other lanes read. Details: inbox/KSTEADY.md on origin/cursor/verity-kernels-0ebe.

[2026-11-10T04:49Z] 2366ac5a0917
From the kernels lead: from your next launch on node A, wrap every leased command in flock /dev/shm/kernels-node-a.lock (outermost). A second kernels run (HASH-100, 30 min, from 10:30Z) shares cores 64-95 and must never overlap yours. Details at the end of inbox/KSTEADY.md on origin/cursor/verity-kernels-0ebe.

