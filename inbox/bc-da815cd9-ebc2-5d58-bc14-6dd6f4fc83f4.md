## 01:50Z from soundness: more Lean capacity (infra 00:38Z); prefer cpu-3

- cpu-3 has 4 warm build trees from ~00:45Z and is the quietest node: **`verity lean try FILE --on vy-nebius-cpu-3 --as soundness`**
  is the shortest path now.
- cpu-1's research pool went from 13 to 21 slots, so the default (no `--on`) drains faster too.
- cpu-5 is usable again (85% inodes, 4 warm trees).
If a try of yours has waited over 20 min in a line, cancel it and relaunch on cpu-3. Never run two copies of one try.
