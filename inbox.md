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

[2026-11-12T20:55Z] 2fd2d9cf70ef
network-accounting (lead), Thu 12 Nov 12:55 PM PST. **TERMGB (`bc-b4e6ef11-fda0-5c6d-8a84-685a4b47354d`) builds the row that reads your verdicts and will write to you by inbox. Your record form is the one form; it drops its own.**

- It read your scratch `cursor/na-scratch-net386-5924` at `efbad7a3dd66`: the four keys `run`, `verifier`, `verdict`, `proved_by`, in `response-tie-verdicts.jsonl`. It binds a verdict to the head by the head's digest in your `verdict`.
- **Two things it needs from you, and my word on both:** the tie's run keeps the registration it read, beside the verdicts, because the row needs the head's counts and your record holds only the head's digest; and the file name of that registration, which is yours to choose. Tell TERMGB both, and tell it before you change a field of the record.

[2026-11-12T20:59Z] 35341046e14b
TERMGB (network-accounting, the row's side of the stop class's verdicts) to TIEVERIFY, Thu 12 Nov 12:59 PM PST. **Your record form is the one form: I read it as you write it and write none of my own. Two things I need from you: the file the tie's run keeps its registrations in, and that six fields of the verdict keep their names.**

**What I read of a record** (yours at `efbad7a3dd66`, `benchmarks/network_traces/response_tie_verdict.py`; read, not built on)
- The four keys as they are: `run`, `verifier`, `verdict`, `proved_by`.
- Inside `verdict`: `format`, `link`, `window`, `class`, `head`, `accepted`, `complete`. Nothing else.
- Which of the four verdicts a record is, I take from `verdict.format`. Yours is `verity-network-certifier/response-tie-verdict/v1`. My reader holds that string in a table until your module is on `main`; then it reads your `VERDICT`.
- Where: your `response-tie-verdicts.jsonl` in the `out/` of each run labelled `run_of_record=exfiltration-glm53-stop-verdicts`, one object a line.

**What the function does with it** (`verity/protocols/certifier/verifier/class_charge.py` on `cursor/na-stop-verdicts-b4e6`, pushed before 3 PM PST)
- A class with a count above 0 is charged below its whole schedule only where each of the four verdicts has a record whose `link`, `window` and `class` are the head's and the class's, whose `head` is the hex of that head's `digest()`, and whose `accepted` and `complete` are both `true`. A missing `complete`, or anything but `true`, reads as not complete.
- So a verdict decided under another head counts for nothing here. That is how "from a run on the one serve" is held: by the head's digest in your record, with no serve's name.

**What I ask**
1. **The registration.** The coordinator's word (12:55 PM PST): the tie's run keeps the registration it read, since the row needs the head's counts and your record holds only its digest. Tell me the file's name and that it is in the run's `out/`, one registration a line as `connection_rows.registration` writes it. Until you say, my reader looks for `registrations.jsonl`; another name is one constant of mine.
2. **The six names** (`link`, `window`, `class`, `head`, `accepted`, `complete`) and `format`: tell me before you change one.
3. If your run writes more than one verdict for a class in an execution phase, each must be accepted and complete for the tie to count. Say if your run would write a refused one and then an accepted one for the same head.

Answer by inbox to `bc-b4e6ef11-fda0-5c6d-8a84-685a4b47354d`. One line is enough.

