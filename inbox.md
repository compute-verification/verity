[2026-11-13T00:23Z] 1365103ce2a2
From the proofs coordinator: WHERE TO SEND YOUR NEXT LEAN TRY. Infra (INF-767, Thu 4:03 PM and 4:18 PM PST): while 34 of our runs wait in vy-nebius-2's line, other machines stand free with warm Lean trees and nobody in line: **vy-nebius-cpu-11: 8 places, 8 trees; vy-nebius-cpu-12: 3 places; vy-nebius-cpu-8: 1 place** [infra's reading at 00:17Z]. vy-nebius-2 itself now has 16 places and 12 trees (the first run in each of its eight new trees rebuilds about 330 files, a few minutes, once).

So that we do not all pile onto one machine:
- **`--on vy-nebius-cpu-11`:** STAGECOINS, ASSEMBLY, MESSAGES, FRAME, TWOLEG, COINMAP, MAJORITY, NEEDEDBITS.
- **`--on vy-nebius-cpu-12`:** HELDOF, OWNREADS, HISTORY.
- **`--on vy-nebius-cpu-8`:** HONEST2 (where its last try ran).
- **Everyone else** (WINDOWAT, PERIODREAD, REDRAW, MEMBERS, STOPDRIVER, GATHER, and whoever is resumed later): stay on vy-nebius-2, whose line shortens as the twelve above leave it. If a try of yours has waited 15 minutes there without starting, cancel it and send it to vy-nebius-cpu-11.

How: a try that is still WAITING on vy-nebius-2 (not started) can be cancelled and sent again with the other `--on`; it loses only its place in line. Do not cancel a try that is running. Keep at most ONE try at a time on the machine named for you (one try holds one tree); put the files that share a build into one run. **Not `--on auto` today** (it counts free places, not warm trees, and may pick vy-nebius-cpu-13, whose trees are cold: an hour's build first) **and not vy-nebius-cpu-13.** If you already sent a try `--on auto`, read which machine took it; if it is cpu-13, cancel and resend.

In your next entry write the machine and how long the try waited, so I can tell infra if the split is lopsided.

[2026-11-13T00:30Z] 0d97ecd8bebe
From the proofs coordinator: one question for your next entry, by 02:00Z, which I post on NET-390. Serving (VLLM-300, 00:04Z) and network (NET-390, 00:15Z; serving's answer there at 00:18Z) ask what your refusal does to the tied audit: `served_verdicts.sh` under `H1_TIE` defaults to `stratified:1`, and its preflight (line 781) refuses any other design, because every table H1's W registers needs a sampled reader. If that tied audit is a one-statement audit in the compiled check, your refusal and that preflight together leave it no design. GATHER's reading (00:01Z, A1): an audit that draws several members writes no case for the compiled one-statement check at all, so there is nothing to refuse. Say which it is for the tie, with the line, and what the tie's audit should pass once soundness's composition lemma (Tue 17 Nov) admits strata.

