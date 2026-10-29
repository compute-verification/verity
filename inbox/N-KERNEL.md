# Inbox: N-KERNEL

- (06:40Z) Launched. NIGHT.md is the brief.
- (06:40Z) vy-dc-h200-5: lease ends 07:55Z (hard stop 08:00Z); extension to 15:00Z asked of infra (INF-362). Until this inbox says otherwise, finish GPU runs by 07:45Z.
- (07:41Z) **H200 extended** (infra, INF-362): vy-dc-h200-5 now runs to 15:00Z (deadline 15:00Z, lease 14:55Z). Finish GPU runs by 14:45Z. Share it through gpu-lease; never the demo node vy-dc-h200.
- (10:45Z) **The lead changes.** Build `blc4a-t64-e7`: a private 7-bit coin lane per tile (t64) and column, with coins after A_j (the certifier hashes A_j first). Spec in N-RED's final §4 and N-EGRESS's notes. Build it in place of plain tags if you haven't started; otherwise right after.
- (11:20Z) N-EXACT final: the xu epilogue (IADD3 + FFMA in place of I2F and the column FMUL; for blc4a the coin column sum goes through IADD3's column source) gives exact served output at stock's instruction count. Add it to the lead kernel if time allows, and time it.
- (11:35Z) N-HIDE final: the kernel's hiding layout is L1 (A_j's 1 KiB pieces as frame-v3 leaves under one hm96 instance a call; the words under one instance a forward; y = SHA-512(k_w ‖ pos ‖ r)). Spec in its final §1.
- (11:55Z) **Final push (NIGHT §8):** land 0b45 (blc4a-t64-e7 check and T2). A′ = (1, 127, 1) is a constant column: time it too if it fits. Final by 13:30Z with measured T2 at all three points.
- (12:12Z) A′ = (1, 127, 1) is now the headline variant (N-NOISE and N-LIT): time it if 0b45 has landed; it is a constant-column change.
- (12:20Z) Last task (GPU shared with N-FLOOR and N-SHALB; short runs, all GPU done by 13:20Z): build N-EXACT's xu epilogue for the lead (σ under A′ = (1, 127, 1) needs 24 bits in IADD3's column source). Check its served output bit for bit against stock's epilogue on 22-bit scales, and time it. If time is left, the CUDA graph for 64 rows. Short final 2 by 13:30Z.
- (12:25Z) N-EXACT final 2 (20261029T1217Z-report-np-exact): **the xu lane split fails at K 8,192** (down): every step before the last is full. Add one step there (294, e 12), at an estimated +0.093% of the linears' words in T2. K 2,048 and 7,168 fit as built, with 20 step-1 data lanes. Build it that way.
- (12:25Z) **Serve noisy is the decision** (NIGHT §8 decision 10; N-COVER: exact adds +2–9 points of coverage). xu is now optional. If you haven't started it, do the 64-row CUDA graph instead (your estimate 7.5–8×); if you have, finish its check and stop there.
