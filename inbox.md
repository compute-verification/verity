[2026-11-13T00:43Z] 36bcb8d700c1
From the circuits lead, Thu 12 Nov 4:42 PM PST (00:42Z), changing one line of my 4:38 PM PST text. Tonight's row on node A starts at 03:15Z, not 03:00Z. Cards 0 to 3 and cores 0-39 are vllm's from 03:00Z to 03:15Z for a serve on all eight cards (VLLM-339; my yes is on that thread and the row on calendars/node-a.md says 03:15Z). A lease of ours taken before 03:15Z would stop that serve.
- Card 3, cores 30-39: launch the capture with NOT_BEFORE=2026-11-13T03:15:00Z, and set your one-off timer for 03:16Z, not 03:01Z. If the 03:01Z timer is already set, replace it; keep one timer. Its prompt carries the new NOT_BEFORE.
- END_UTC=2026-11-13T04:25:00Z and the last start of 03:50Z are unchanged. If the start check finds the card busy at 03:16Z, vllm's serve overran: wait and try again.
Everything else in the 4:38 PM PST text stands. Then re-run the command this refused.

