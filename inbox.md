[2026-11-10T15:56Z] 06bf730510c1
From the coordinator, Tue 10 Nov 7:56 AM PST. **Infra gave the two-host window (INF-664, its comment of 7:54 AM PST: read it whole). It is TODAY: first run 10:00 AM PST (18:00Z), second 10:55 AM PST (18:55Z). This now comes before your other three jobs.**

- **The pair: A = `vy-nebius-cpu-10` (serves: the certifier and the server), B = `vy-nebius-cpu-8` (observes).** Not cpu-2 and cpu-1.
- Ports 47811 and 47812 are open from cpu-8 to cpu-10's private address, tested by infra.
- A's address: read it on cpu-10 with `ip -o -4 addr show dev eth0 scope global` (a short run there prints it). Keep the address out of Linear, Slack and the notes; infra posts none. The serve side checks the tree's `peers` file, which names cpu-2: make the pair work for cpu-10 and say how.
- **Send both runs with `--on` and NO `--pool`** (`research run --on vy-nebius-cpu-10 --cwd source --max-min 60 --send window.sh -- …` and the same `--on vy-nebius-cpu-8`). From 9:00 to 11:35 AM PST cpu-10's pools take no new run; a run without a pool is not held, not niced and not pinned.
- The hold for the second run is on timers on cpu-10: `cpuidle_haltpoll` loads at 10:45:00 AM PST, the limit is 0 from 10:45:05 to 11:35:00. Your second run must end by 11:30. `host.txt` must show `haltpoll`, else that run is a second baseline: say which it was.
- cpu-10 has 64 cores and 251 GB. If the server needs more, tell me at once: A can become cpu-8.

**What I need from you, in order:**
1. **A rehearsal across the two hosts, at once:** the same `window.sh`, both sides, a few hundred buckets, no pool, to show the route, the address, the start and `pair_score.py` work across hosts. It has only ever run with both sides on one machine.
2. **By 9:15 AM PST, one line to my inbox: "window holds" or "move it", with why.** Infra moves the hour only if I say so by 9:30 AM PST; the pair is free tonight. No line from you by 9:15 and I move it to tonight.
3. Launch both sides from one clean pushed commit about 9:45 AM PST with START = 10:00:00 AM PST, and again about 10:40 AM PST with START = 10:55:00 AM PST. Fetch both, score, label each run with its numbers.
4. Comment the two results on NET-296 (the number with its bounds for each case, which run was under the hold, what compares with what) and send me the same.

Run none of your other runs on cpu-10 today. Your suites run goes to cpu-12 or cpu-4. The suites verdict moves from noon to 2 PM PST.

