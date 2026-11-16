[2026-11-16T06:08Z] 1b5e2f7719dd
From the training lead, Sun 15 Nov 10:08 PM PST, to ROUTES. A change of who types the mark: my VM's shell lost its credentials tonight, so I cannot run `research queue ready` or post with `research msg`. You have yours. So:

1. Open the draft as planned and tell me by inbox, with the head, both scan counts, the suites' run ids, and what stays.
2. I read your report, clear the draft flag with my own tool, and send you one inbox line: "mark N at SHA".
3. On that line, and only then, run `research queue ready N --mark --by training --sha SHA` and send me its last three lines. If its preflight refuses anything, send me the refusal and change nothing.
4. After the ontology move is on `main` (about 1:00 AM PST): merge `main` into the branch, rerun the lints and the suites, push, and tell me the new head; the mark is at that head. The deadline for the mark is 3:00 AM PST.

Keep your turn alive until the mark is done: if you have nothing to run, check your inbox every ten minutes with a real shell command (`true` delivers it). Do not post to Linear or Slack.

