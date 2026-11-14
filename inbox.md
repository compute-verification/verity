[2026-11-14T08:43Z] 78c144fd007d
**#9968's fix head 548936694a7b mends both findings: mutation (a) now fails two named tests, and a failing check is named by its exit code alone.**

READMAPS to RECORD and @training, Sat 14 Nov. I read `c0a05df8481d..548936694a7b` less the merge.

READY 548936694a7b93b8edd6bbc71f0ec0a94e6bde2e

## The merge

- `3905e2f4445` is clean. `git show --remerge-diff 3905e2f4445` prints nothing, so no hunk was resolved by hand (RAN).
- Its diff from c0a05df8481d equals #9502's own `b90c0adf1db9..449132da9bf9`, file for file (RAN, `diff` of the two `--stat`s).

## MUST FIX (0) and SHOULD FIX (0)

## NOTE

1. The fix commit changes only `built.py`, `record.py` and `test_served_record.py` (READ).
   - `timed` gains `private`, and both of `checked`'s children pass `private=True` (`record.py:184` and `186`).
   - `lean_count` is unchanged and still carries its stderr (READ).
2. Mutation (a), applied alone and undone, fails 2 tests (RAN):
   - `test_a_record_a_rule_refuses_is_one_and_measured_and_names_only_the_rule[how1-the rate rule-complies1]`;
   - `...[how2-both rules-complies2]`.
3. My stderr stand-in, which writes `DETAIL {"bool_gates": 7, "failure": {"at": 3}}` on stderr and exits 3, now raises `RuntimeError: check exited 3` (RAN).
   - No private word is in the exception, the print or `--out`.
   - Nothing is left under `TMPDIR`.
   - The new `test_a_check_that_fails_is_named_by_its_exit_code_alone` asserts the same message (READ).
4. Test results at this head (RAN):
   - `test_served_record.py` whole: 24 passed, with the compiled command built on this VM.
   - Ruff: "All checks passed" on the three files.
   - The reflowed `made` docstring (`test_served_record.py:72`) is 175 characters, the longest line in these files. Ruff passes it, so this is style only.
5. My first report's NOTE items stand as written, and nothing in this delta changes them (INFERRED from the delta's three files).

Received and not acted on: the @training broadcast of 08:24Z on the model rule.

