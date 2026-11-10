[2026-11-10T20:46Z] faef832fb806
ci (20:50Z), two changes from top (CI-485, 20:28Z), after Daniel at 12:22 PM PST: "done today".
1. **Item 1 is due live at 4 PM PST (00:00Z), not 6 PM.** Put your PR up and marked by 23:00Z so I can review and deploy by 00:00Z.
2. **Item 8 is yours as a second PR after item 1, live by 6 PM PST (02:00Z).** An owner's withdrawn mark, new head or closed PR no longer drops a candidate in flight: it lands as built. Only a red (yours, item 1) or a reviewer's hold drops it.
   - Today a withdrawn mark dropped two stacked pairs, 29 PRs each (#8387 at 09:45Z, #8027 at 14:16Z).
   - Tell an owner's withdraw from ci's eject (`--by ci`, for a red) and from a reviewer's hold by the label's author.
   - The drop paths are in `Loop.snapshot` and `vqueue.plan`.

