[2026-11-12T20:41Z] b97fea131047
network-accounting (lead), Thu 12 Nov 12:41 PM PST. **One thing to know, from SCANLEAN's work on NET-335 on the same branch of proofs' [argued from `audit_record` and `session_registered`; no round run].**

- The record check at proofs' R7 head is still the strict one (`opened != want`, `one_stage/audit.py`). A round in which a certified value's readers have no session is refused until proofs' relaxed check is in.
- At a group of one link-window every template's stratum is sampled, so both values are opened and the round passes.
- If a round of yours is refused there, it is that check and not your verifier. Do not work around it in one-stage. Put it in your 3 PM PST line with the round's shape, and I take it to proofs on NET-390.
- proofs' branch is `77414839aa90` (three test files of proofs' own past `c2aebff06095`). SCANLEAN's scratch for NET-335 is `cursor/na-scratch-net335-7c2e` at `a11f58adbfdc`: its `certified=` calls are the pattern to read, not to merge.
- Nothing in your order changes.

[2026-11-12T20:49Z] 95bef24cb909
network-accounting (lead), Thu 12 Nov 12:49 PM PST. **Daniel narrowed Friday at 12:04 PM PST: Friday is SPOT end to end on GLM-5.3, and the ties to the certifier follow next week. Your order is the same work; its bar moved. And at 12:37 PM PST he set the goal of every verdict as `proved`, so your Python verifier is a step and its Lean gets a sizing from you.**

- **The bar now:** vllm's small serve with both ties is Tue 17 Nov, rehearsed on CPU Mon 16 Nov. NET-386's pull request is open with its four refusal tests in time for that. 9 PM PST tonight and Fri 6 AM PST stay your targets, not bars: nothing is cut to make them.
- **Your 3 PM PST line, as before, plus two things:**
  1. The hour you hold for the pull request open with its tests.
  2. Where a rule of soundness's would go (NET-390, 12:40 PM PST): the auditor's side reads a head only at its fixed offset after the execution phase closes, and a head that comes later counts as no head, so its class is charged its whole schedule. One paragraph: which function of the tie's verifier or service would hold it, and what it needs to know the offset. Do not build it yet.
- **By Fri 13 Nov 9 AM PST, a sizing of the tie's verifier in Lean, ten lines at most:** each check your Python makes, the Lean on `main` or on lean's marked head `bafae2eaec2e` that already does it (the certificate's check in `NetworkCheck`, the head's digest, one-stage's record check), what is missing, and what the guarantee would say. No Lean is written for it yet.
- **The benchmark's record** (the verdict's file for benchmarks) is no longer due Fri 10 AM PST. It is due with the ties' serve; keep the helper.
- Runs for a tie go to the shared lines (`--on auto`), never to the node infra named for Friday's path.

