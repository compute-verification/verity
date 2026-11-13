[2026-11-13T00:43Z] c88a4f14ee2e
From the circuits lead, Thu 12 Nov 4:42 PM PST (00:42Z). Tonight's row on node A starts at 03:15Z, not 03:00Z. Cards 0 to 3 and cores 0-39 are vllm's from 03:00Z to 03:15Z for a serve on all eight cards (VLLM-339; my yes is on that thread and the row on calendars/node-a.md says 03:15Z). A lease of ours taken before 03:15Z would stop that serve.
- If you run anything on card 1 (cores 10-19) tonight: NOT_BEFORE=2026-11-13T03:15:00Z, no lease on card 1 before then, off by 04:25Z.
- If you run nothing on a card tonight, nothing changes for you. Say which in your next return, in one line, so I can release card 1 on the calendar.
Then re-run the command this refused.

