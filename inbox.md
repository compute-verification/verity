[2026-11-14T08:01Z] 37c9a5fbb996
WALKER (timer turn), Sat 14 Nov, 12:02 AM PST.

**r20261114-074757-b707 is cancelled, as you ordered at 11:56 PM PST; the one s-t20 reading left is r20261114-075207-9b8b at C's head 985ea1bd6bea.**

- RAN: `research cancel r20261114-074757-b707` at 08:01Z. Its status now reads state failed, rc 143, ended 08:01:12Z, "cancel_intent by manual ... delivered".
- RAN: it had waited 13 min for a check slot, so its workload never started.
- RAN: r20261114-075207-9b8b reads state waiting at 08:01Z, tree_sha 985ea1bd6bea3c1d5ddb78edb48a36bf3441fc66. I did not touch it.
- Your message came to my timer turn, not to either working session. I left its whole text and what I did in a file on the VM for both, and I edit no branch or body.

[2026-11-14T08:01Z] b44d62a6df1d
NOT READY 1 — read at 0d6dbc683514f2732a42365526f0ce5b1bb85afa (PR #9964, cursor/training-step-kinds-af01; head unchanged at 00:01 PST)

MUST FIX
1. benchmarks/training_slowdown/tests/synthetic.py:218-220. The PR's own rule (the body; the KINDS comment at :191-195) says blocks frozen below trained ones make a separate kind. All 16 library QAT configs train blocks [45], [345] or [2345] of six, with the blocks below them frozen. The three QAT kinds run on a one-block start with every block trained, and there is neither a case nor a reason for the frozen-below shape. Smallest cure: give the three QAT kinds `"blocks": 2` and targets `blocks.1.w*.weight`, or state why frozen-below doesn't matter for QAT. RAN: scratch probes of that shape pass (fp32 codes 6.5 s; rtn4 with low-width state 6.9 s). READ (config map built from attacks/*.toml with each route's DIALS merged in).

SHOULD FIX
1. Nothing makes a new kind fail until it gets a case. KINDS is hand-kept against attacks/ (list of record: attacks/*.toml; routes/*.py; subset.WHOLE/MATRICES; qat.GRAD_CODES). Cure: a test mapping each `library.entries()` config to a JOBS key, like test_entries.py:405 `test_every_entry_of_the_library_has_a_reference`. READ.
2. test_run.py:655-656. The testing skill says: "A test over about 5 s gets `@pytest.mark.slow`". So the rule asks for the slow mark, not a docstring. (AGENTS.md's docstring rule applies to a test adding more than about a minute; the docstring at :681-686 meets it anyway.) The run lists 15 unmarked tests over 5 s, six of them new. Cure: `pytest.param(job, 3, marks=pytest.mark.slow)` for the new cases over 5 s. READ/RAN.

NOTE
- Check 1 (READ): both sides are independent for all 23 kinds. The count comes from `flops.Counter` dispatching the real run (`job_counted` → `run.train`). The trace comes from `capture_steps` on fake tensors (`traced_step`). Neither is computed from the other. They do share `opnames` name tables and `gates` prices, so a wrong entry there moves both sides and this test can't see it. Shampoo's eigh is an n³ rule of thumb on both sides (`flops.py` LINALG and `capture/structure.py:232`), coded separately.
- Check 3 (RAN; one-line breaks to the counter in `flops.py` only, each reverted, worktree clean): Lion's `sign` counted ×2 failed only `[lion-3]`. Fused AdamW +1 per parameter failed `[grow-3]`, `[muon-whole-fused-3]`, `[fullmuon-3]`, `[muon-fused-3]`, `[fullmuon-fe-3]`, `[depthshare-3]`, `[depthshare-top-3]`, `[strideshare-3]`, `[ema-fullmuon-3]` and `[ema-fullmuon-fe-3]`. Linalg approx +1 failed `[shampoo-3]`, `[shampoo-decay-3]` and `[shampoo-whole-3]`.
- Check 4 (READ): no tolerance, rounding, skip, xfail or approximate comparison in the changed lines; `counted["gates"] == tr["circuit"]["gates"]` (:700).
- RAN: trace test 31/31 pass (140 s); record test 30/30 pass (32 s, every new case under 2.2 s). Probes: QAT rtn8 with low-width state (the stated 8-bit=4-bit reason) passes, and so does LoRA on every matrix (the library's targets).
- Body (RAN): run r20261114-062721-fad6 has state `done`, rc 0, tree_sha 0d6dbc683514f2732a42365526f0ce5b1bb85afa. stdout.log has "787 passed" and the last line "1 suites: 1 passed (0 cached), 0 failed; 13.7 min of test time" (plus a logs path). The 6.8-17.6 s figures and "six of ten" match its listing. The 23 kinds and 64 configs match (READ).
- Checks 7 and 8 (READ): only the two test files change; no ids, dates or names.
- No message addressed to @training arrived.

For the lead to read
- synthetic.py:191-220: KINDS and its claims about which dials branch a step.
- test_run.py:613-614: OPTIMIZER_PRODUCTS says QAT runs Muon by route name; a new QAT optimizer would slip past.
- test_run.py:707: approx_macs is checked against `opt == "shampoo"`; nothing applies it to QAT or grow.
- routes/subset.py:118-124: `embed`/`gain_lr` grouping, which the PR says doesn't branch a step.
- test_entries.py:405-407: the precedent for SHOULD FIX 1.

[2026-11-14T08:02Z] 3e3894a26e0d
NOT READY 1 — read at 0d6dbc683514f2732a42365526f0ce5b1bb85afa (#9964; replaces my earlier, longer message)

MUST FIX
1. tests/synthetic.py:218-220. The PR's own rule (body; comment :191-195) makes blocks frozen below trained ones a separate kind. All 16 library QAT configs train the top 2-4 of six blocks with the rest frozen below. The three QAT kinds use a one-block start, every block trained. There is no case or reason for frozen-below. Cure: `"blocks": 2` and targets `blocks.1.w*.weight` on them, or a stated reason. RAN: scratch probes of that shape pass (6.5 s, 6.9 s). READ: config map from attacks/*.toml with each route's DIALS.

SHOULD FIX
1. Nothing fails when a new kind lacks a case. KINDS is kept by hand. List of record: attacks/*.toml, routes/*.py, subset.WHOLE/MATRICES, qat.GRAD_CODES. Cure: a test mapping each `library.entries()` config to a JOBS key, as test_entries.py:405 does for REFERENCES. READ.
2. test_run.py:655-656. The testing skill says "A test over about 5 s gets `@pytest.mark.slow`", so the rule asks for a slow mark, not a docstring. AGENTS.md's docstring rule (a test adding more than about a minute) is met at :681-686. The run lists 15 unmarked tests over 5 s, six of them new. Cure: `pytest.param(job, 3, marks=pytest.mark.slow)` for those six. READ/RAN.

NOTE
- 1 (READ): the two sides are independent for all 23 kinds. One is `flops.Counter` dispatching the real run (`job_counted`). The other is `capture_steps` on fake tensors (`traced_step`). They share the `opnames` tables and the `gates` prices, so an error there moves both sides unseen. Shampoo's eigh is the n³ rule on both sides, coded separately.
- 3 (RAN; one-line breaks in `flops.py` only, reverted, worktree clean):
  - Lion's `sign` ×2 failed only `[lion-3]`.
  - Fused AdamW +1 failed `[grow-3]`, `[muon-whole-fused-3]`, `[fullmuon-3]`, `[muon-fused-3]`, `[fullmuon-fe-3]`, `[depthshare-3]`, `[depthshare-top-3]`, `[strideshare-3]`, `[ema-fullmuon-3]` and `[ema-fullmuon-fe-3]`.
  - Linalg approx +1 failed `[shampoo-3]`, `[shampoo-decay-3]` and `[shampoo-whole-3]`.
- 4 (READ): no tolerance, rounding, skip or xfail. :700 compares gates exactly.
- RAN:
  - The trace test passes 31/31 and the record test 30/30 (new cases under 2.2 s).
  - Two probes pass: QAT's rtn8 code with low-width state, which tests the body's 8-bit-equals-4-bit reason, and LoRA on every matrix.
- 6 (RAN): run r20261114-062721-fad6 has state `done`, rc 0 and tree_sha equal to the head. Its stdout.log says "787 passed", and its last line matches the body's (plus a logs path). The durations and "six of ten" match. The 23 kinds and 64 configs match (READ).
- 7 and 8 (READ): only the two test files change; no ids, dates or names.
- No message addressed to @training arrived.

For the lead to read
- synthetic.py:191-220: which dials the PR says branch a step.
- test_run.py:613-614: QAT counts as Muon by route name.
- test_run.py:707: approx_macs is keyed to `opt == "shampoo"` alone.
- routes/subset.py:118-124: `embed`/`gain_lr` grouping, which the PR says doesn't branch a step.
- test_entries.py:405-407: the precedent for SHOULD FIX 1.

