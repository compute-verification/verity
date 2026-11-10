[2026-11-10T07:56Z] 843f1404e8bf
From @circuits (lead), Tue 10 Nov 08:00Z. Daniel ruled at 11:44 PM PST (top on TOP-394, 07:51Z): a fast CI and the repository layout come first, every other goal is pushed back, and each lead stops every worker on a pushed-back goal that is more than 2 hours from its result. Until the layout moves land (Tue night PST) only fixes to landing and the check, the layout maps and cleanup are marked ready. Your work is on a pushed-back goal. This replaces what I told you before. Re-run your command, then:
1. #8447 (head 60711e63, pushed 07:41Z): do NOT mark it. If its runs at that head end by 10:00Z, put their verdicts in the PR body; cancel the ones that cannot.
2. One comment on CIR-204 (--worker MNNVL-MATCH): the head, what the mask-port retyping changed, what is verified and what is left before a mark.
3. Return by 10:00Z.
Every head gets restacked across the layout moves, so suites rerun then anyway: do not launch new runs. Cancel with `research cancel RUN --reason "TOP-394: goal pushed back" --by circuits`. Push every commit. Write one checkpoint note under lanes/circuits (kind report: branches, heads, runs with verdicts, what is left in order) and sync it, so another worker can pick it up after the moves. Then return to me with the same in one short message. Start nothing else.

