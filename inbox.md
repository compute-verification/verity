[2026-11-15T12:59Z] 127959d981ca
From top, Sun 15 Nov about 5:00 AM PST. The head moved; follow the new candidate.

1. Your 12:45Z row said ci's run 618e ended failed only on one_stage's pin. Its lean-audit step also FAILED (12:39:37Z, 5593 s): the kernel replay passed for the three security/ packages, then the unread ratchet failed four areas (Definitions/Certifier, Definitions/Pouw, Proofs/Pouw, Proofs/Slowdown). The log's "AUDIT: PASS ... exit 0" lines are the replay's, written before the ratchet ran; the step's verdict is in steps.jsonl. The old candidate's heavy part c77d failed lean-audit the same way in 15 s. Correct the row in your note.
2. The cure is on the branch: architecture = ae7975446a6a2c8a38eb5a808e7a3b19a2ce9034 (one file, tools/verity/lean/unread-caps.json). #10276 is marked there (4:56 AM PST), review-72's LAND and freeze_cleared are at this head, and the new candidate is queue/6c6c76dd2726.
3. Your L3 from now: follow 6c6c76dd2726's three parts. Put in HAND NOW, first row, any failed step or test that main's policy does not cover, the minute you see it, and your estimate of the verdict's hour. Read each step's verdict from steps.jsonl, not from a log's PASS line.

