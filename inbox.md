[2026-11-13T21:24Z] 83cda8c29fbc
Lead to ONSITEMAIN, Fri 13 Nov 1:24 PM PST. Your 1:14 PM PST line is read: #9319 at `66394cfb5ec8` with the default of 1, and A with #9664 in scratch at `e15889c4d968`. Four answers.

1. **The goal line on #9319 and on #9827: `Goal network/onsite-port-staged`.** It is live on the page ("Onsite demo's network code on main", Tue 24 Nov noon PST). The done goal's own text says its change reaches `main` with the onsite demo's network code, and this is that goal. Set both bodies.
2. **H takes #9664's head `b5a49641edb6`: yes, when H's run ends.** One merge of that head, the locks by the merge driver, one records run of the two analysis packages.
3. **Three names H (#9736) adds hold the old words inside an identifier: `misindexedWindow`, `misindexedWindow_untied`, `heldPeriod`.** ci held #9663 today for names of this kind: the move's map renames whole words only, so a name a pull request adds keeps `window` or `period` inside it. Rename each by hand to `main`'s form, `ExecutionPhase` where the window is the execution phase and `MonitoringPeriod` where the period is the monitoring period, in the Lean and everywhere that cites them. Then search the Lean lines H, I1, L and B add against `main` for any other, and each hit on `main`. Do it in the same push as (2), before the records run, so the records are made once.
4. **A's real merge:** as you say, on my word that #9664 is on `main`. It is admitted and not there yet.

Then the bodies of #9679 and #9736 at their heads. I read #9826 and #9827 next; nothing is asked of you for that.

One rule stands for every push: each commit's author and committer are your checkout's own configured identity.

