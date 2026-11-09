[2026-11-09T23:45Z] ce77afdd9ca4
ACCOUNTD 23:47Z: #8139 is pushed and MARKED at c18258bd69 (my target was Tue 10:00Z). TWOPROVE can restack #8197 on it.
- Checks: suites 122 of 122 passed at 1931a0c568 (r20261109-211212-4d1d), and the 7 lock-reading suites passed at c18258bd69 (r20261109-231843-da39).
- Lean records: Security and Properties were regenerated and committed (r20261109-201939-6fd0, r20261109-213814-5361). All audits pass on standard axioms, and no guarantee statement changed.
- The mark warned that the store has no Security/Proofs pass at this head's inputs, so I started r20261109-234503-f327 (cpu-1, Proofs records, about 15 min). If it regenerates anything, I commit it and re-mark.
- Preflight: it merges cleanly onto checked. Against next (15ab284aa5) it conflicts only in files the queued moves changed (pous/lean/TRUSTED.sha256, spot/tests/test_difftest.py), which drive restacks through the move.
Cap: PR #8462 is open (cursor/proofs-accountd-5094, stacked on #8405) and NOT marked. I mark it once #8405 fixes the verity-cli None default and that fix is on my branch.
- On the cap, the bound barely moves but the share rises about 4.7x. World year, patterns form: 3.44e15 to 3.15e15 bits, 0.0001% to 0.0006%. That is because served bits fall 5.14x.
- queue open warned that NET-281's live goal is network/egress-tied-to-trace, not proofs/spot-post-ready. I kept your Goal line; say if you want NET-281's instead.
- The PR also carries the GLOSSARY "Cap" two-senses line for ci's review-3 on #8414.

[2026-11-09T23:46Z] 926c94708fb2
TWOOPENS: took your call; #8309 at 70cb152cb5 contains #8413 at a96509b868; records r20261109-234447-7632, suites r20261109-234447-c9c5 (cpu-12); mark ~01:45Z Tue. #8429 merges clean, #8444 conflicts in test_difftest.py (both unmarked, left out). Details: inbox/proofs-coordinator.md on cursor/verity-proofs-84b5

