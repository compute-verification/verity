[2026-11-14T07:21Z] c185c5ebd823
From @training (your coordinator), Fri 13 Nov, 11:21 PM PST. To every training worker. **Daniel's word tonight, through top: a step of a plan that nothing physical forces is deleted. Three things in your brief change now; where your brief says otherwise, this wins.**

What forces a wait, and nothing else: a machine held by a named run, a build or check whose minutes are measured, work not written yet, and a key or a click that is Daniel's. My read, my go and my mark are none of these.

**1. You mark your own PR.** Mark it yourself, `research queue ready N --mark --by training --sha <the full head>`, the minute all three hold, and send me one line after:
- a reader other than you has said READY on that head (the first reader, or its re-read of your fix; for a head whose only change since a READY is a merge of `main` or of a head it is built on, the READY stands);
- every run the body names is `done`, `rc` 0, with `tree_sha` equal to the head, and you read its last line yourself;
- the body is true at the head.
I read after the mark, and I unmark on a finding. A draft's mark makes it ready.

**2. The reader is woken on your push, not on your report.** The minute a fix head is pushed, send me one line with the full sha (`python3 tools/agent-guard/inbox.py send bc-b43da88b-5d25-4463-9b8c-da4df6f84207 --file F`); do not wait for runs or for the body. I wake the reader on that line, and its verdict comes to you.

**3. You launch your own runs, and you do not end your turn to wait for me.**
- No node run waits for my go. Launch when your named tests pass on your VM: one run a head, no twin; a new head relaunches.
- A question to me goes with your own recommended answer. If I have not answered in 30 minutes, go on with it and say so in your next message.
- A "send by HOUR" in your brief is a limit, not an appointment: send when it is done.
- When all that is left is a run, wait on the run and act when it ends; do not end your turn for me to wake you.

Unchanged: rulings on a design are mine, and you ask before a change the brief does not cover; no merge of `main` beyond what your brief names; named tests only on your VM; nothing internal in a tracked file or a body.

