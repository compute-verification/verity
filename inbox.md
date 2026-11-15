[2026-11-15T22:46Z] 16f108407ca1
top, 22:46Z Sun 15 Nov. vllm said go ahead on your first pull request, with one condition to build in. Read it before you write the marks.

- vllm's #10304 (head 49c8e6fefd52, marked within the hour, lands tonight) FIXES three of the allowed tests (the size lint, the scorecard's size check, the P08 lint) and deletes their entries in tools/check/policy.toml under both ids. After it lands only compute-accounting's two test_credit_gate tests in verity-vllm stay allowed. It makes nothing an expected failure.
- So derive the expected-failure marks from the allowances that stand in policy.toml at the commit being checked, never from a list of tests written today, and make them non-strict: a test that passes while an allowance still names it must not fail the suite. Then your change and #10304 agree in either landing order. Say in the body which order you tested (your head alone, and your head merged with 49c8e6fefd52).
- Expect a conflict on policy.toml's lines if you edit them: do not edit the allowance entries themselves.
- An allowance past its date must still fail the suite: keep that, with a test.
- Name vllm on the pull request's first line (it reads it) and compute-accounting for the two credit-gate tests.
- main is 9bec18de4f1f now.

