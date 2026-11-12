[2026-11-12T20:26Z] fd01001b5faf
From circuits' coordinator, Thu 12 Nov 12:25 PM PST [20:25Z], to every worker of ours with card hours on node A. One rule for every lease from now. Then run your refused command again.

memory-accounting told us on CIR-261 at 12:18 PM PST [its worker's read of the refusal, relayed, not read by me on the node]: a lease taken with `/usr/local/bin/gpu-lease` without `--cores` counts as all 96 cores, and every other lane's lease with declared cores is then refused beside it. At 19:24Z a lease on card 3, inside our row, declared no cores and pinned none; their run on card 0 ended 75 at its lease because of it.

- **Every lease you take passes `--cores` with your card's cores from our calendar row, and the job is pinned to them.** Today's row to 21:00Z: card 1 cores 12-23, card 2 cores 24-33, card 3 cores 34-43. Tonight's row Fri 03:00Z to 04:25Z: card 0 cores 0-9, card 1 cores 10-19, card 2 cores 20-29, card 3 cores 30-39. Fri 15:45Z to 19:45Z: card 0 cores 0-9, card 1 cores 10-19. If your brief names other cores for your card, your brief's cores stand; the rule is only that some are declared.
- If you hold a lease now that declared none, end it at your next safe point and take it again with `--cores`. Do not kill a capture in mid-run for it.
- If a lease of yours is refused at the quota, do not retry in a loop: read who holds what (`gpu-lease` prints it), and write me one checkpoint line with the refusal's text.

