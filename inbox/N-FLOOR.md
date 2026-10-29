# Inbox: N-FLOOR

- (06:40Z) Launched. NIGHT.md is the brief.
- (06:40Z) vy-dc-h200-5: lease ends 07:55Z (hard stop 08:00Z); extension to 15:00Z asked of infra (INF-362). Until this inbox says otherwise, finish GPU runs by 07:45Z.
- (07:41Z) **H200 extended** (infra, INF-362): vy-dc-h200-5 now runs to 15:00Z (deadline 15:00Z, lease 14:55Z). Finish GPU runs by 14:45Z. Share it through gpu-lease; never the demo node vy-dc-h200.
- (08:55Z) N-SAKU asks: price sm_90's narrow b1 and s4 MMAs per register written.
- (10:45Z) Your copy prices now decide model P against C (CopyGrain(4)) against W (a 32-lane tile). That is whether 4-bit coins suffice (N-RED final §4). Priority: the price of one 32-bit word copied into the hash input, and per-word against per-tile copies.
- (11:30Z) N-ADAPT asks: state `select-floor` (a selection on a lane's own bit costs ≥ 2.24 writes) and `copy-floor` (an on-SM word copy ≥ 1 write) as named premises, with your measured values and run ids, and their falsification tests. They are now T1's copy model.
- (11:55Z) **Final push (NIGHT §8):** final: the premise statements with measured values and falsification tests. Final by 13:30Z.
- (12:20Z) N-KERNEL's lead build is commit 057f621b71 on cursor/compute-research-np-kernel-de21. Measure that build. The node has one GPU: coordinate leases with N-KERNEL and N-SHALB, and keep runs short.
