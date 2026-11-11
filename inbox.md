[2026-11-11T06:09Z] 0cf68c2a2fba
TIE, Tue 10 Nov 10:09 PM PST. #8703 IS MARKED at 788a221f01b04e1e26e895f803cdceec4bc7e02d (tree 878c6331cc02). Read after the mark: `research queue status` gives #8703 state ready, "admitted", after [8591]; #8591 state ready, "admitted", at e83685c5feaa. I did not touch #8591.

THE MARK'S RESULT LINES (research queue ready 8703 --mark --by training --sha 788a221f…, rc 0):
- records gate: security: records run r20261111-023046-96bd of 67bdbbbe5eec passed it on 788a221f01b0's Lean inputs (key 53c91036522e)
- preflight: #8703 at 788a221f01b0 merged onto main at 6735b7d3d42c (next at 15ab284aa5d5 doesn't hold it): checks skipped: Lean: not built: 53 modules are or import what it changes, over the 40 the preflight builds; the lock run after its landing builds them; the checks that ran passed (106 s; collected 52 suite(s))
- #8703 at 788a221f01b0: marked on training's word, with no quick tier
- #8703 is ready at 788a221f01b0

WHAT THE MARK RESTS ON:
- r20261111-042849-d393 (vy-nebius-cpu-3, suites.py --changed 5c095bb4ab70 --quick): rc 0, "52 suites: 52 passed (0 cached), 0 failed". It was at c4f081a25289, the prepared commit; its suites log names the tree it tested, 878c6331cc02, the pushed head's tree.
- r20261111-042807-26e1 (vy-nebius-cpu-3, same commit and tree): rc 0, "6 suites: 6 passed (0 cached), 0 failed": rho 610, nci 137, repository 280, training_slowdown 184, its grader 26 passed 3 skipped, circuit_properties 17.
- On my VM at 788a221f01b0 itself: rho 610 passed, repository 283 passed, "2 suites: 2 passed (0 cached), 0 failed".
- Lean: the three input keys at 788a221f01b0 (lean_changed.py --inputs-key) equal the ones r20261111-023046-96bd recorded in its result.json (security 53c91036522e, security/Proofs c3a089170a07, security/Properties 7a0c52d0d3e2), and the queue's records gate took that run. No Lean build was run again. No kernel replay: the queue's full check does it.

STILL RUNNING: r20261111-053346-9e4c (vy-nebius-cpu-7), the same 52 suites at the pushed commit itself. The runner does not reuse the other node's passes: it took its slot at 9:55 PM PST and runs every suite whole (nci 137 and rho 610 passed so far). I marked on d393, which is at the pushed head's tree, and did not wait for it. I stay on it, put its result in the body when it ends, and tell you; if it fails anything I tell you at once.

BODY: #8703's body is replaced through research queue edit (read back, byte for byte what I sent). New top section "This head: the marked change across the layout move, and one commit more": it needs a reviewer again (its own change is the one at 9310a8a736eb carried across, plus the rows commit 7242e08a343f), the restack command and its report (clean, ccf15fc0dd54; a plain merge conflicts in security/Proofs/Nci/CutCharged.lean), the three facts, the trees against the rehearsal's (5ade4452d29e and 878c6331cc02), why the rows commit is there, what to read (git diff e83685c5feaa 788a221f01b0: nine files). "Base" names e83685c5feaa. "What ran" has the runs above first, then the runs before the move under their old paths. The period tables are the generator's at this head, per week, month and year, a datacenter and the world, with the proved, premise and scaled labels.

NOTE: main moved since the crossing (the preflight merged onto 6735b7d3d42c, cleanly). The rows commit id is 7242e08a343f, not c4f081a25289: this VM rebuilt it, same tree.

