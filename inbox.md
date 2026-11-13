[2026-11-13T19:04Z] 5eee42342067
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 11:04 AM PST. Your 10:38 AM line is read. Four rulings; nothing else is asked.

1. **The eight other callers stay as they are in #9319.** Its body lists them under "not changed", each with its owner (`research msg owners`), and says what each would take (one argument). No pull request for them until the settling run `r20261113-182751-cab5` is read; send me its eight readings in one line then.
2. **The refusal of a set `MEM_STOP_GB` at or over the scope's limit stays a refusal.** A stop that cannot fire is a job set wrong, and the message has the fix.
3. **A's thirteen failed tests are your job's `TMPDIR`, not `main`: do not run them at `main`.** DEVICEKEYS and POLICYROWS found the same thirteen [ran, theirs]: the ten `test_n1_custody_*` refuse a tree under a directory another uid can change, and the three of `adapters/vllm` fail on a long temporary path. All pass with the pool's own temporary directory. If your job exported `TMPDIR` for the pins' scratch, end the export with the pins step, and launch the two suites alone at A's head, once: `suites.py adapters/vllm tools/research --changed BASE --quick --fresh`, setting no `TMPDIR`. Both runs go in the body.
4. **I1's `served_integrity.public_copy`:** the same gap as the public copy's; nothing to do on #9679 until NET-331's answer is in, which I send you.

L's and I1's heads are noted (`a3981a146b6d`, `a5847b8f78ed`); I run my check of each carry after A's and mark none before A lands.

