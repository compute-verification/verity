# Inbox: N-KERNEL

- (06:40Z) Launched. NIGHT.md is the brief.
- (06:40Z) vy-dc-h200-5: lease ends 07:55Z (hard stop 08:00Z); extension to 15:00Z asked of infra (INF-362). Until this inbox says otherwise, finish GPU runs by 07:45Z.
- (07:41Z) **H200 extended** (infra, INF-362): vy-dc-h200-5 now runs to 15:00Z (deadline 15:00Z, lease 14:55Z). Finish GPU runs by 14:45Z. Share it through gpu-lease; never the demo node vy-dc-h200.
- (10:45Z) **The lead changes.** Build `blc4a-t64-e7`: a private 7-bit coin lane per tile (t64) and column, with coins after A_j (the certifier hashes A_j first). Spec in N-RED's final §4 and N-EGRESS's notes. Build it in place of plain tags if you haven't started; otherwise right after.
