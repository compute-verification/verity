[2026-11-13T07:19Z] f786fcef980e
network-accounting (lead) to every worker of the lane, Thu 12 Nov 11:19 PM PST. **Daniel, 11:03 PM PST: the repository's structure comes first tonight. Two chains of path-only moves go onto `main` by Fri 13 Nov 7 AM PST, and every open pull request of ours crosses them. No answer is needed to this line.**

- **What moves** (top, on TOP-415, 11:05 PM PST): chain 1 tonight as rehearsed; chain 2, marked by 5 AM PST, moves `verity/protocols/certifier` under `verity/protocols/accounting/`, `inference_only` under `policy/`, the proof system's three under `proofs/`, the Lake packages, and nci and rho to `benchmarks/`. ci pauses the queue from about 2:30 AM until chain 2 is on `main`.
- **Your rule from now:** do not rebase, and do not merge `main` into a branch by hand, until chain 2 is on `main`. A pull request of yours that is not on `main` by then, marked or not, is carried across with `tools/move/restack.py`, by you, once. Then one check at the new head, then one line to me with the head and the run; I read and mark again.
- **Go on with the work you have.** Finish your runs, push your own changes, write your bodies. What you build tonight is carried Friday morning with the rest.
- **I mark no Lean change tonight:** ci stages none before the chains. A mark you were waiting on comes after the carry.
- **A run you launch after the chain lands uses the new paths.** A run from a pinned commit is not touched: tonight's served tries and Friday's rehearsal stand.
- **New files go where the protocol is today;** nobody moves a directory by hand.
- **In prose from now, Daniel's words:** "challenges" for what the challenger sends, not "coins"; "accounting protocol"; "the auditor's device", not "host". Code names stay.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

