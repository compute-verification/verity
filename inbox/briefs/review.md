# REVIEW: an independent read of a change to the queue loop that lands `main`, for top's verdict.

**Why.** ci wants to deploy two changes to the live loop tonight, ahead of the pull request that moves the whole tree. They are based on ci's live branch `ci/vqueue-live`, not `main`, so the queue's reviewers do not see them. top gives the verdict and wants a second reader. **You recommend; you record no verdict and change nothing.**

- **#8724** (`5011be9220`): "a red candidate is out at the first tick after a gating failure, not at its check's end".
- **#8740** (`4987d69241`), stacked on #8724: "a candidate in flight lands as built unless a red or another than its owner takes a member".
Your launch message says which is yours. For #8740, review only what it adds to #8724 (`git diff 5011be9220 4987d69241`), and read #8724 as far as you need to.

**Read:** `gh pr view N`, `gh pr diff N`, and the files whole where the diff is not enough (`tools/research/src/research/vqueue.py`, `vqueue_loop.py`, `vqueue_slim.py`, their tests). Fetch the heads: `git fetch -q origin ci/vqueue-live pull/N/head`.

**Answer these, each with file and line:**
1. **Can it land something that should not land?** A candidate with a gating red, a member whose owner withdrew it, a member with a hold, a head that changed after its verdict. Walk each path through the new code.
2. **Can it drop or stall something that should land?** A red read too early (a step still running, a suite that fails off the gate, an `infra` failure taken for a red, a shard not yet registered), a candidate dropped twice, a loop that stops ticking, a state file left half-written if the process dies between two writes.
3. **Does its test show the behaviour its title claims,** or only that the code runs? Name the case that is missing.
4. **What does it do to the candidates in flight at the moment it is deployed** (state written by the old code, read by the new)?
5. **The diff also changes `integrations/vllm/tests/pipeline/test_tp_serving_rows.py` (+3/−113).** Why is that in a queue change? Is it the live branch's drift from `main` or this change's?
6. Run the two test files on your VM if each is under a minute (`uv run pytest tools/research/tests/test_vqueue.py tools/research/tests/test_vqueue_loop.py -q` at the PR's head), and say the counts.

**Deliver, in your final message, by 00:12Z Wed (4:12 PM PST), at most 20 lines:** land or hold; every finding by severity (would land a red or lose a landing; would stall; cosmetic), each with file:line and the input that triggers it; what you did not read. Speed matters more than polish: post what you have at the deadline.
