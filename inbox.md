[2026-11-13T00:43Z] 990e6ce1b993
From the circuits lead, Thu 12 Nov 4:42 PM PST (00:42Z). Tonight's row on node A starts at 03:15Z, not 03:00Z. Cards 0 to 3 and cores 0-39 are vllm's from 03:00Z to 03:15Z for a serve on all eight cards (VLLM-339; my yes is on that thread and the row on calendars/node-a.md says 03:15Z). A lease of ours taken before 03:15Z would stop that serve.
- Card 2, cores 20-29, is yours from 03:15Z to 04:25Z. Launch the second try with NOT_BEFORE=2026-11-13T03:15:00Z and take no lease on card 2 before then. END_UTC=2026-11-13T04:25:00Z is unchanged.
- If the start check finds the card busy at 03:15Z, vllm's serve overran: wait and try again, last start 03:50Z.
Then re-run the command this refused.

