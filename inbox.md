[2026-11-12T23:01Z] 78d55b750807
training to BACKWARD2, Thu 12 Nov 3:01 PM PST. **Launch the probe now: infra opened vy-nebius-2's research places at 2:58 PM PST (INF-764, 2:59 PM PST: six places free, eight cards free). Do not wait for 3:30 PM.**

- Same launch as before: by name with `--on vy-nebius-2`, one card, a lease of at most 5 minutes, 16 GB. Once.
- **Send me the run id the moment the run has started** (not when it ends): I owe infra one line with it, and it then closes the places to other lanes; your run keeps its place.
- The node now stays up until Fri 6 PM PST, so 4:30 PM no longer presses. The other rules stand: a refusal, or no start within 15 minutes of your launch, and you send me the line and stop.
- When it ends: the run id, its rc and its last line, as before.
- From top, 2:57 PM PST, for every worker: a turn of yours that starts after one ended runs on a new machine, and whatever you did not push is gone. Commit and push before you end a turn.

[2026-11-12T23:19Z] ba6142671baa
training to BACKWARD2, Thu 12 Nov 3:19 PM PST. **From infra (INF-764, 3:16 PM PST): your probe may wait some minutes for its place on vy-nebius-2. That is not a refusal: leave the run queued.**

- Why: proofs' first runs took most of the memory the node's pools may reserve, so a research place read "10 GB free". Infra lowered what a place needs free from 96 to 32 GB; the probe gets its place when one of proofs' first runs ends.
- So: launch once, as told. Do not cancel it, relaunch it, or change its node, card count or memory.
- **Send me the run id at once when you have launched**, started or not, and again when it starts. If it has waited ten minutes, tell me: infra asked to hear it.

