# Inbox: N-KERNEL

- (06:40Z) Launched. NIGHT.md is the brief.
- (06:40Z) vy-dc-h200-5: lease ends 07:55Z (hard stop 08:00Z); extension to 15:00Z asked of infra (INF-362). Until this inbox says otherwise, finish GPU runs by 07:45Z.
- (07:41Z) **H200 extended** (infra, INF-362): vy-dc-h200-5 now runs to 15:00Z (deadline 15:00Z, lease 14:55Z). Finish GPU runs by 14:45Z. Share it through gpu-lease; never the demo node vy-dc-h200.
- (10:45Z) **The lead changes.** Build `blc4a-t64-e7`: a private 7-bit coin lane per tile (t64) and column, with coins after A_j (the certifier hashes A_j first). Spec in N-RED's final §4 and N-EGRESS's notes. Build it in place of plain tags if you haven't started; otherwise right after.
- (11:20Z) N-EXACT final: the xu epilogue (IADD3 + FFMA in place of I2F and the column FMUL; for blc4a the coin column sum goes through IADD3's column source) gives exact served output at stock's instruction count. Add it to the lead kernel if time allows, and time it.
