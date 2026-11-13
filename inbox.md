[2026-11-13T07:48Z] bf948e8166e0
From @training (your coordinator), Thu 12 Nov 11:55 PM PST. Two rules for tonight. Neither changes your task or its hour unless your own line at the end says so.

**1. Your machine (infra's announcement, INF-783, not yet explained).** A worker on its own cloud machine may wake on a new machine with uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own.
- Commit and push before you end a turn and before every long run.
- Act on no event that does not name you. A timer or a Slack line of mine that reaches you is not yours: ignore it and tell me it came.
- On waking, read `uptime` and `git status` before you trust your tree.

**2. Do not merge `main` into your branch from now until I tell you the two move chains are on `main`.** Daniel put the repository's structure first tonight. Two chains of path moves land on `main` between about 5 and 8 AM PST on Fri 13 Nov, and the queue builds nothing else from 2:30 AM PST until they are on.
- Chain 1 moves configuration, settings and benchmark instances. Chain 2 moves, among much else, `verity/protocols/inference_only/nci/` and `rho/` to `benchmarks/nci/` and `benchmarks/rho/`, regroups `verity/protocols/` under `proofs/`, `accounting/` and `policy/`, and moves Lean files. It rewrites every line that names a moved path, in `benchmarks/training_slowdown/` and `security/` too.
- A branch that has merged a `main` from after a move cannot be carried by the tool (`tools/move/restack.py` refuses a branch "based after" the move). So: keep working on the base you have. A merge of `main` you made before now is fine.
- When I tell you the chains are on `main`, you carry your branch across once with `tools/move/restack.py` (its docstring is the manual), run your suites on the result, and send me the head. Nobody rebases or resolves a move by hand. If `restack.py` refuses or reports anything but `clean`, stop and send me what it printed.
- If your work is done before the chains land, send me the head as planned. I say then whether it is marked before or after them.

**Yours:**
- **WALKER:** #9558's head is still due Fri 9 AM PST, on your present base. Send me one line now saying where it stands.
- **COUNTS:** #9519 is Lean, and chain 2 moves Lean files and rewrites the locks. Your head is still due Fri 10 AM PST on your present base; it is carried after the chain, and ci puts Lean changes in the queue only after the chain.
- **ONSITE:** network-accounting's cut that I told you of for Fri 7 PM is tonight's chain 2, so #8780 crosses it. Finish on your present base, send me the head by Fri noon PST as planned, and carry it once when I say.
- **DGRAD, GROWN:** nothing else changes. #9531 lands after the chains, not before.

