[2026-11-12T19:51Z] 60359cfbc968
MAPS to training, Thu 12 Nov 19:55Z. **r20261112-190713-6890 (part A's suites at 4dfe6b139320, `--changed d8b4a44698a6 --fresh`) touches 53 suites, so it won't end by 1:30 PM PST.** It started at 19:32Z, after 25 min waiting for a slot. 2 of 53 have passed and none failed so far (RAN, its stdout). The same command over the larger change took 3 h 10 min at the old head, so expect near 22:45Z. The 53 are every suite whose declared inputs include the calculator's files or `tools/move/` (vllm, the certifier, circuit properties, network traces, graph, ...).

What is in now:
- Ratchet gate at part A's head, r20261112-190741-a08b: "ratchet gate d8b4a44698a6..4dfe6b139320: 1 change(s), 0 failed" (RAN).
- r20261112-174205-9914 (suites at d8b4a44698a6, 58 touched): 47 passed, none failed, the heavy 11 still running; ETA about 20:50Z (RAN, its stdout).
- A2's ratchet gate at `2e149dad14b1`, r20261112-191504-d41c: "ratchet gate d8b4a44698a6..2e149dad14b1: 1 change(s), 0 failed" (RAN).

Unless you say otherwise, I send the one message when 9914 ends: 9914's last line, 6890's count so far and its ETA, the four lines, and #9275's body posted, which names 6890 as running. Then a second short message, with 6890's last line and the body updated, when 6890 ends. If you would rather wait for 6890 before marking, say so; nothing else changes.

