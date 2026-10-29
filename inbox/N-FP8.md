# Inbox: N-FP8

- (06:40Z) Launched. NIGHT.md is the brief.
- (06:40Z) vy-dc-h200-5: lease ends 07:55Z (hard stop 08:00Z); extension to 15:00Z asked of infra (INF-362). Until this inbox says otherwise, finish GPU runs by 07:45Z.
- (07:41Z) **H200 extended** (infra, INF-362): vy-dc-h200-5 now runs to 15:00Z (deadline 15:00Z, lease 14:55Z). Finish GPU runs by 14:45Z. Share it through gpu-lease; never the demo node vy-dc-h200.
- (11:55Z) **Final push (NIGHT §8):** final: BF16 design, int8-served FP8 and decode on the lead. Final by 13:30Z.
- (12:10Z) Last task: decode at m 1 fails T1 on G's forming (1.73% in W1). Find the cheapest G layout at decode (forming once a window, shared across decode calls with per-call coins, or G in the weights' layout) that keeps the tag closure and gets decode under 1%. Price it, and check it against N-RED's A1/A2c if you can. Short final 2 by 13:30Z.
