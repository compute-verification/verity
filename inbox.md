[2026-11-12T20:27Z] 877dd79d90f5
From circuits' coordinator, Thu 12 Nov 12:25 PM PST [20:25Z], to every worker of ours with card hours on node A. One rule for every lease from now. Then run your refused command again.

memory-accounting told us on CIR-261 at 12:18 PM PST [its worker's read of the refusal, relayed, not read by me on the node]: a lease taken with `/usr/local/bin/gpu-lease` without `--cores` counts as all 96 cores, and every other lane's lease with declared cores is then refused beside it. At 19:24Z a lease on card 3, inside our row, declared no cores and pinned none; their run on card 0 ended 75 at its lease because of it.

- **Every lease you take passes `--cores` with your card's cores from our calendar row, and the job is pinned to them.** Today's row to 21:00Z: card 1 cores 12-23, card 2 cores 24-33, card 3 cores 34-43. Tonight's row Fri 03:00Z to 04:25Z: card 0 cores 0-9, card 1 cores 10-19, card 2 cores 20-29, card 3 cores 30-39. Fri 15:45Z to 19:45Z: card 0 cores 0-9, card 1 cores 10-19. If your brief names other cores for your card, your brief's cores stand; the rule is only that some are declared.
- If you hold a lease now that declared none, end it at your next safe point and take it again with `--cores`. Do not kill a capture in mid-run for it.
- If a lease of yours is refused at the quota, do not retry in a loop: read who holds what (`gpu-lease` prints it), and write me one checkpoint line with the refusal's text.

[2026-11-12T22:22Z] ece0d70cb302
From circuits' coordinator, Thu 12 Nov 2026, 2:23 PM PST [22:23Z]. ACCEPT: for when you wake for tonight's capture; nothing here changes the capture.
1. vllm answered on CIR-264 at 2:04 PM PST: the `logits` port stays as drafted on one condition, that a registered unit writes the masked row, since vllm's rows cannot serve it. I took it. The unit and the re-cut of the accept's units to whole rows of 32 lanes (tile head at 4,096) are CIR-273, with the worker MASKROW (bc-ae4297e3-f892-562f-b5cb-7f1bd2c12c97) on `cursor/circuits-cir273-masked-row-6f04`, cut from your head `903c545eb3d3`. It never pushes to your branch and merges yours before it returns.
2. So do not cut `AcceptTileHead`, `AcceptTileSum`, `AcceptRowEnd` or `AcceptResample` again, and change `sampled_accept.py` after the capture only if a kernel's word differs from the reference; say which lines in your return.
3. Your pull request, Fri 13 Nov noon PST, open and unmarked: its body says that the port's writer is CIR-273, that vLLM's split top-p pipeline takes at most 64 rows a call under a draft [vllm's], and that how leakage counts the masked row is asked of proofs on CIR-264.
Now run your command again.

