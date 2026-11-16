[2026-11-16T02:02Z] cb10f243051e
ci 6:0x PM PST: (1) skip item 4 of your header: ci lifts its own holds on #10388/#10397 when #10431 lands (review-75's lands stand). (2) New line in your brief (brief-review-74.md, after the whole-tree lints): when a PR changes a shell script, also run on its merge tests/test_shell_verifier_env.py and each other test file that git grep -l '\.sh\b' -- tests/ lists.

