[2026-11-13T00:43Z] b01efe3ab68e
From the circuits lead, Thu 12 Nov 4:42 PM PST (00:42Z). Tonight's row on node A starts at 03:15Z, not 03:00Z. Cards 0 to 3 and cores 0-39 are vllm's from 03:00Z to 03:15Z for a serve on all eight cards (VLLM-339; my yes is on that thread and the row on calendars/node-a.md says 03:15Z). A lease of ours taken before 03:15Z would stop that serve.
- Card 0, cores 0-9, is yours from 03:15Z to 03:45Z. Launch with NOT_BEFORE=2026-11-13T03:15:00Z and take no lease on card 0 before then. Be off the card by 03:45Z: it is ACCEPT's from then, as before.
- That is 30 minutes, not 45. Run what fits in one lease, the first kernels first; Friday's row (15:45Z, card 0) holds the capture of record.
- If the start check finds the card busy at 03:15Z, vllm's serve overran: wait and try again, last start 03:25Z, and say in your return how many minutes it cost.
Then re-run the command this refused.

